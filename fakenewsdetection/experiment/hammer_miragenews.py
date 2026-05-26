"""
HAMMER × MiRAGe-News — Binary (Real vs AI-Fake)
=================================================
  BATCH=16, EPOCHS=10, KFOLD=5
  EARLY_STOP_PATIENCE=4, EARLY_STOP_MIN_DELTA=1e-4
  early stop 기준: val macro-F1
  LR 스케줄러: 없음
  sampler: shuffle=True (--weighted_sampler 플래그로 전환 가능)
  weight_decay=1e-4

[HAMMER 고유 설정 — 아키텍처 특성상 유지]
  lr=2e-5 (BERT), lr_img=1e-4 (ViT), weight_decay=1e-4
  binary_only=True, use_bbox=False
  ALBEF 사전학습 가중치 자동 로드
  MAC loss warmup 비활성 
  AdamW (HAMMER 논문 설정)


[실행]
  python hammer_mirage.py
  python hammer_mirage.py --no_kfold
  python hammer_mirage.py --eval_only --checkpoint ./checkpoint/xxx.pt
"""

from __future__ import annotations

import argparse
import copy
import gc
import io
import math
import os
import random
import re
import sys
import warnings
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizerFast

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── 경로 ──────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_FND_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_VENDOR   = os.path.join(_FND_ROOT, "experiment", "DGM4", "multimodal_deepfake")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from models.HAMMER import HAMMER                       # noqa
from optim.optim_factory import create_optimizer       # noqa

LOG_TAG = "HAMMER-MIRAGE"

if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#1. 상수설정
SEED                 = 42
BATCH                = 16      
EPOCHS               = 10      
NUM_WORKERS          = 4       
EARLY_STOP_PATIENCE  = 4       
EARLY_STOP_MIN_DELTA = 1e-4    
KFOLD_SPLITS         = 5       
SAMPLER_ALPHA        = 0.5

# HAMMER 아키텍처 고유 LR (BERT/ViT 차등)
PAPER_LR       = 2e-5   # BERT
PAPER_LR_IMG   = 1e-4   # ViT
WEIGHT_DECAY   = 1e-4   

IMG_SIZE   = 256
MAX_WORDS  = 50

HF_DATASET_ID  = "anson-huang/mirage-news"
DEFAULT_ALBEF  = os.path.join(_FND_ROOT, "experiment", "DGM4", "ALBEF_4M.pth")
RESULT_DIR     = os.path.join(_HERE, "result")
CHECKPOINT_DIR = os.path.abspath(os.path.join(_HERE, "..", "checkpoint"))
LABEL_NAMES    = ["Real", "AI-Fake"]

os.makedirs(RESULT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# 2. 유틸
# ════════════════════════════════════════════════════════════════════════════════

def set_seed(s: int = SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed()


def pre_caption(caption: str, max_words: int) -> str:
    caption = (re.sub(r"([,.'!?\"()*#:;~])", "", caption.lower())
               .replace("-", " ").replace("/", " "))
    caption = re.sub(r"\s{2,}", " ", caption).strip()
    words = caption.split()
    return " ".join(words[:max_words]) if len(words) > max_words else caption


def resolve_test_splits(ds_dict) -> List[str]:
    """MiRAGe test split 우선순위로 정렬."""
    keys = list(ds_dict.keys())
    preferred = ["test_midjourneyv5","test_midjourney_v5","test_dalle3",
                 "test_dalle_3","test_sdxl","test_bbc","test_cnn"]
    ordered = [k for k in preferred if k in keys]
    for k in sorted(keys):
        if k.startswith("test") and k not in ordered:
            ordered.append(k)
    return ordered


def make_weighted_sampler(labels: List[int], alpha: float = 0.5) -> WeightedRandomSampler:
    from collections import Counter
    cnt = Counter(labels)
    w   = [(1.0 / cnt[l]) ** alpha for l in labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


# ════════════════════════════════════════════════════════════════════════════════
# 3. EarlyStopping — val macro-F1 기준
# ════════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE,
                 min_delta=EARLY_STOP_MIN_DELTA, mode="max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best: Optional[float] = None
        self.counter   = 0

    def step(self, v: float) -> bool:
        if self.best is None:
            self.best = v; return False
        imp = (v - self.best > self.min_delta if self.mode == "max"
               else self.best - v > self.min_delta)
        if imp:
            self.best = v; self.counter = 0; return False
        self.counter += 1
        return self.counter >= self.patience


# ════════════════════════════════════════════════════════════════════════════════
# 4. Dataset
# ════════════════════════════════════════════════════════════════════════════════

image_transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])

image_transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])


class MirageHammerDataset(Dataset):
    def __init__(self, hf_split, is_train: bool = False, max_words: int = MAX_WORDS):
        self.hf_split  = hf_split
        self.is_train  = is_train
        self.max_words = max_words
        self.transform = image_transform_train if is_train else image_transform_val

        cols = hf_split.column_names
        self.text_col  = next((c for c in ("caption","text","headline") if c in cols), None)
        self.image_col = next((c for c in ("image","img")               if c in cols), None)
        self.label_col = next((c for c in ("label","labels","fake")     if c in cols), None)
        if None in (self.text_col, self.image_col, self.label_col):
            raise ValueError(f"필수 컬럼 없음. 사용 가능: {cols}")

        self.labels = [int(x) for x in hf_split[self.label_col]]

    def __len__(self) -> int:
        return len(self.hf_split)

    def _load_pil(self, raw) -> Image.Image:
        if isinstance(raw, Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, dict) and "bytes" in raw:
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127))

    def __getitem__(self, idx: int):
        row   = self.hf_split[idx]
        text  = pre_caption(str(row[self.text_col]).strip(), self.max_words)
        label = int(row[self.label_col])
        img   = self._load_pil(row[self.image_col])

        # HAMMER string label
        hammer_label = "orig" if label == 0 else "fake"

        img_t          = self.transform(img)
        fake_image_box = torch.zeros(4,             dtype=torch.float32)
        fake_text_pos  = torch.zeros(self.max_words, dtype=torch.float32)

        return img_t, hammer_label, text, fake_image_box, fake_text_pos, label


def collate_mirage(batch):
    imgs          = torch.stack([b[0] for b in batch])
    hammer_labels = [b[1] for b in batch]
    texts         = [b[2] for b in batch]
    fake_boxes    = torch.stack([b[3] for b in batch])
    fake_pos      = torch.stack([b[4] for b in batch])
    labels        = torch.tensor([b[5] for b in batch], dtype=torch.long)
    return imgs, hammer_labels, texts, fake_boxes, fake_pos, labels


def build_loader(ds: MirageHammerDataset, batch_size: int, num_workers: int,
                 shuffle: bool, drop_last: bool = False,
                 weighted: bool = False, alpha: float = 0.5) -> DataLoader:
    kw: Dict[str, Any] = dict(
        batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_mirage,
        pin_memory=torch.cuda.is_available(), drop_last=drop_last)
    if weighted:
        kw["sampler"] = make_weighted_sampler(ds.labels, alpha)
        kw["shuffle"] = False
    else:
        kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


# ════════════════════════════════════════════════════════════════════════════════
# 5. Text 전처리
# ════════════════════════════════════════════════════════════════════════════════

def text_input_adjust(text_input, fake_word_pos, device):
    input_ids_rm = [x[:-1] for x in text_input.input_ids]
    maxlen = max(len(x) for x in text_input.input_ids) - 1
    text_input.input_ids = torch.LongTensor(
        [x + [0]*(maxlen-len(x)) for x in input_ids_rm]).to(device)

    attn_rm = [x[:-1] for x in text_input.attention_mask]
    text_input.attention_mask = torch.LongTensor(
        [x + [0]*(maxlen-len(x)) for x in attn_rm]).to(device)

    fake_token_pos_batch = []
    for bi in range(len(fake_word_pos)):
        ftp  = []
        fdec = np.where(fake_word_pos[bi].numpy() == 1)[0].tolist()
        sub  = np.array(text_input.word_ids(bi)[1:-1])
        for wi in fdec:
            ftp.extend(np.where(sub == wi)[0].tolist())
        fake_token_pos_batch.append(ftp)
    return text_input, fake_token_pos_batch


# ════════════════════════════════════════════════════════════════════════════════
# 6. 모델 빌드
# ════════════════════════════════════════════════════════════════════════════════

def interpolate_pos_embed(pos_embed: torch.Tensor,
                          new_num_tokens: int) -> torch.Tensor:
    cls_pos   = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]
    orig_size = int(patch_pos.shape[1] ** 0.5)
    new_size  = int((new_num_tokens - 1) ** 0.5)
    patch_pos = (patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2))
    patch_pos = F.interpolate(patch_pos, size=(new_size, new_size),
                              mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, pos_embed.shape[-1])
    return torch.cat([cls_pos, patch_pos], dim=1)


def build_model(tokenizer, image_res: int,
                albef_pretrained: str = DEFAULT_ALBEF) -> HAMMER:
    bert_cfg = os.path.join(_VENDOR, "configs", "config_bert.json")
    cfg = {
        "embed_dim": 256, "image_res": image_res, "vision_width": 768,
        "queue_size": 65536, "momentum": 0.995, "temp": 0.07,
        "label_smoothing": 0.0, "bert_config": bert_cfg,
        "use_bbox": False, "binary_only": True,
    }
    margs = SimpleNamespace(token_momentum=False)
    model = HAMMER(args=margs, config=cfg,
                   text_encoder="bert-base-uncased",
                   tokenizer=tokenizer, init_deit=True)

    if albef_pretrained and os.path.isfile(albef_pretrained):
        print(f"[{LOG_TAG}] ALBEF 로드: {albef_pretrained}")
        ckpt = torch.load(albef_pretrained, map_location="cpu")
        sd   = ckpt.get("model", ckpt)
        pe_key = "visual_encoder.pos_embed"
        new_np = (image_res // 16) ** 2 + 1
        if pe_key in sd and sd[pe_key].shape[1] != new_np:
            sd[pe_key] = interpolate_pos_embed(sd[pe_key], new_np)
            print(f"[{LOG_TAG}]   pos_embed 보간: {sd[pe_key].shape}")
        for k in list(sd.keys()):
            if k.startswith("visual_encoder."):
                sd[k.replace("visual_encoder.", "visual_encoder_m.")] = sd[k].clone()
            elif k.startswith("text_encoder."):
                sd[k.replace("text_encoder.", "text_encoder_m.")] = sd[k].clone()
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[{LOG_TAG}]   missing={len(miss)}  unexpected={len(unexp)}")
    else:
        print(f"[{LOG_TAG}] ⚠️  ALBEF 없음 → 랜덤 초기화")

    return model


# ════════════════════════════════════════════════════════════════════════════════
# 7. 학습 epoch — MAC warmup 비활성 유지 (HAMMER 고유)
#    ★ LR 스케줄러 없음
# ════════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model: HAMMER, loader: DataLoader,
                    tokenizer, device, optimizer,
                    epoch_idx: int, alpha: float,
                    warmup_epochs: int) -> float:
    model.train()
    mac_wgt = 0.0 if epoch_idx <= warmup_epochs else 0.1
    bic_wgt = 1.0

    tot = n = 0
    pbar = tqdm(loader, desc=f"[{LOG_TAG}] Epoch {epoch_idx}", file=sys.stderr)
    for i, (imgs, hammer_labels, texts, fake_boxes, fake_pos, _labels) in enumerate(pbar):
        imgs       = imgs.to(device, non_blocking=True)
        fake_boxes = fake_boxes.to(device, non_blocking=True)

        text_input = tokenizer(texts, max_length=128, truncation=True,
                               add_special_tokens=True, return_attention_mask=True,
                               return_token_type_ids=False)
        text_input, fake_token_pos = text_input_adjust(text_input, fake_pos, device)

        a = (alpha if epoch_idx > 1
             else alpha * min(1.0, (i + 1) / max(len(loader), 1)))

        loss_MAC, loss_BIC, _, _, _, _ = model(
            imgs, hammer_labels, text_input, fake_boxes, fake_token_pos, alpha=a)

        loss = mac_wgt * loss_MAC + bic_wgt * loss_BIC

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot += loss.item() * imgs.size(0)
        n   += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         mac="Y" if mac_wgt > 0 else "N")

    return tot / max(n, 1)


# ════════════════════════════════════════════════════════════════════════════════
# 8. 예측 수집 & 지표
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_predictions(model: HAMMER, loader: DataLoader,
                        tokenizer, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps, probs = [], [], []
    for imgs, hammer_labels, texts, fake_boxes, fake_pos, labels in tqdm(
            loader, desc=f"[{LOG_TAG}] Eval", file=sys.stderr):
        imgs       = imgs.to(device, non_blocking=True)
        fake_boxes = fake_boxes.to(device, non_blocking=True)

        text_input = tokenizer(texts, max_length=128, truncation=True,
                               add_special_tokens=True, return_attention_mask=True,
                               return_token_type_ids=False)
        text_input, fake_token_pos = text_input_adjust(text_input, fake_pos, device)

        logits_bin, _, _, _ = model(
            imgs, hammer_labels, text_input, fake_boxes, fake_token_pos,
            is_train=False)

        prob_fake = F.softmax(logits_bin, dim=-1)[:, 1]
        pred      = logits_bin.argmax(1)

        ys.extend(labels.tolist())
        ps.extend(pred.cpu().tolist())
        probs.extend(prob_fake.cpu().tolist())

    return np.array(ys), np.array(ps), np.array(probs)


def evaluate_binary(yt: np.ndarray, yp: np.ndarray,
                    scores: np.ndarray,
                    split_name: str) -> Tuple[str, Dict[str, float]]:
    """이진 분류 평가 리포트 생성."""
    report = classification_report(yt, yp, target_names=LABEL_NAMES,
                                   digits=4, zero_division=0)
    acc   = float(accuracy_score(yt, yp))
    f1    = float(f1_score(yt, yp, average="macro",  zero_division=0))
    f1_ai = float(f1_score(yt, yp, average="binary", pos_label=1, zero_division=0))
    try:
        auc = float(roc_auc_score(yt, scores))
    except ValueError:
        auc = float("nan")

    block = (
        f"\n=== {split_name} ===\n"
        f"{report}\n"
        f"[{split_name}]"
        f"  Acc={acc:.4f}"
        f"  Macro-F1={f1:.4f}"
        f"  AI-Fake-F1={f1_ai:.4f}"
        f"  AUC={auc:.4f}\n"
    )
    return block, {"acc": acc, "f1": f1, "f1_ai": f1_ai, "auc": auc}


# ════════════════════════════════════════════════════════════════════════════════
# 9. 학습 루프 — LR 스케줄러 없음, val macro-F1 기준
# ════════════════════════════════════════════════════════════════════════════════

def train_one_run(model: HAMMER,
                  train_ds: MirageHammerDataset,
                  val_ds:   MirageHammerDataset,
                  tokenizer, device, args) -> Tuple[HAMMER, float]:

    # shuffle=True (기본), --weighted_sampler 로 전환 가능
    train_loader = build_loader(
        train_ds, args.batch_size, args.num_workers,
        shuffle=(not args.weighted_sampler), drop_last=True,
        weighted=args.weighted_sampler, alpha=args.sampler_alpha)
    val_loader = build_loader(
        val_ds, args.batch_size, args.num_workers, shuffle=False)

    # AdamW (HAMMER 논문), weight_decay=1e-4
    opt_ns    = SimpleNamespace(opt="adamW", lr=args.lr, lr_img=args.lr_img,
                                weight_decay=args.weight_decay, momentum=0.9)
    optimizer = create_optimizer(opt_ns, model)

    # LR 스케줄러 없음
    early     = EarlyStopping(args.early_stop_patience,
                               args.early_stop_min_delta, mode="max")

    best_f1    = -1.0   # val macro-F1 기준
    best_state: Optional[Dict] = None

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, tokenizer, device, optimizer,
            ep, args.alpha, args.warmup_epochs)

        yt, yp, scores = collect_predictions(model, val_loader, tokenizer, device)
        va_f1  = float(f1_score(yt, yp, average="macro", zero_division=0))
        try:
            va_auc = float(roc_auc_score(yt, scores))
        except ValueError:
            va_auc = float("nan")
        va_acc = float(accuracy_score(yt, yp))

        # ★ LR 고정 (스케줄러 없음)
        cur_lr = optimizer.param_groups[-1]["lr"]
        mac_on = "Y" if ep > args.warmup_epochs else "N"
        print(f"[{LOG_TAG}] Ep {ep:02d}/{args.epochs}"
              f"  loss={tr_loss:.4f}"
              f"  va_f1={va_f1:.4f}"    # ★ early stop 기준
              f"  va_auc={va_auc:.4f}"
              f"  va_acc={va_acc:.4f}"
              f"  lr={cur_lr:.2e}"
              f"  MAC={mac_on}")

        # val macro-F1 기준 best 저장
        if va_f1 > best_f1:
            best_f1    = va_f1
            best_state = copy.deepcopy(model.state_dict())

        if early.step(va_f1):
            print(f"[{LOG_TAG}] Early stopping at Ep {ep}"
                  f" (best val_f1={best_f1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{LOG_TAG}] best val macro-F1={best_f1:.4f} 가중치 복원 완료")

    return model, best_f1


# ════════════════════════════════════════════════════════════════════════════════
# 10. 전체 test split 평가
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_all_test_splits(model: HAMMER, ds_dict,
                              test_splits: List[str],
                              tokenizer, device, args
                              ) -> Tuple[List[str], Dict[str, Dict]]:
    blocks:      List[str]           = []
    all_metrics: Dict[str, Dict]     = {}
    for split_name in test_splits:
        test_ds = MirageHammerDataset(ds_dict[split_name],
                                      is_train=False, max_words=args.max_words)
        loader  = build_loader(test_ds, args.batch_size, args.num_workers, shuffle=False)
        yt, yp, scores = collect_predictions(model, loader, tokenizer, device)
        block, metrics = evaluate_binary(yt, yp, scores, split_name)
        print(block)
        blocks.append(block)
        all_metrics[split_name] = metrics
    return blocks, all_metrics


# ════════════════════════════════════════════════════════════════════════════════
# 11. 요약
# ════════════════════════════════════════════════════════════════════════════════

def write_summary(rep: io.StringIO, all_metrics: Dict[str, Dict],
                  test_splits: List[str], tag: str = ""):
    rep.write(f"\n===== SUMMARY (avg over test splits) {tag}=====\n")
    for k in ("acc", "f1", "f1_ai", "auc"):
        vals = [all_metrics[s][k] for s in test_splits
                if not math.isnan(all_metrics[s].get(k, float("nan")))]
        if vals:
            line = (f"{k}: {float(np.mean(vals)):.4f}"
                    f"  (min={float(np.min(vals)):.4f},"
                    f" max={float(np.max(vals)):.4f})")
        else:
            line = f"{k}: nan"
        print(line); rep.write(line + "\n")


# ════════════════════════════════════════════════════════════════════════════════
# 12. 체크포인트 I/O
# ════════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path: str, model: HAMMER, args, metrics=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config":  vars(args),
        "metrics": metrics,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }, path)
    print(f"[{LOG_TAG}] checkpoint: {path}")


def load_checkpoint(model: HAMMER, path: str, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[{LOG_TAG}] load_checkpoint  missing={len(miss)}  unexpected={len(unexp)}")


def _save_report(rep: io.StringIO, timestamp: str, args):
    path = os.path.join(args.result_dir, f"hammer_mirage_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rep.getvalue())
    print(f"[{LOG_TAG}] 리포트: {path}")


# ════════════════════════════════════════════════════════════════════════════════
# 13. 메인 실험 함수
# ════════════════════════════════════════════════════════════════════════════════

def train_and_test(args):
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[{LOG_TAG}] 데이터셋 로드: {args.hf_dataset_id}")
    ds = load_dataset(args.hf_dataset_id)
    print(f"[{LOG_TAG}] splits: {list(ds.keys())}")
    test_splits = resolve_test_splits(ds)
    if not test_splits:
        raise RuntimeError("test split 없음.")
    print(f"[{LOG_TAG}] test_splits: {test_splits}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    rep       = io.StringIO()
    mode_name = "KFold" if not args.no_kfold else "Official Split"
    sampler_note = "WeightedSampler" if args.weighted_sampler else "shuffle=True"

    rep.write(f"###### {LOG_TAG} ({mode_name}) ######\n")
    rep.write(f"test_splits={test_splits}\n")
    rep.write(
        f"batch={args.batch_size}  epochs={args.epochs}"
        f"  lr={args.lr}  lr_img={args.lr_img}"
        f"  wd={args.weight_decay}"
        f"  patience={args.early_stop_patience}"
        f"  early_stop_monitor=val_macro_F1"
        f"  scheduler=None"
        f"  sampler={sampler_note}\n\n"
    )

    # ── 평가만 ───────────────────────────────────────────────────────────────
    if args.eval_only:
        model = build_model(tokenizer, args.image_res, args.albef_pretrained).to(DEVICE)
        if args.checkpoint:
            load_checkpoint(model, args.checkpoint, DEVICE)
        else:
            print(f"[{LOG_TAG}] ⚠️  --checkpoint 없음 → 랜덤 초기화")
        blocks, all_metrics = evaluate_all_test_splits(
            model, ds, test_splits, tokenizer, DEVICE, args)
        for b in blocks: rep.write(b)
        write_summary(rep, all_metrics, test_splits)
        _save_report(rep, timestamp, args)
        return

    # ── Official Split ────────────────────────────────────────────────────────
    if args.no_kfold:
        rep.write("===== TEST RESULTS (Official Split) =====\n")
        train_ds = MirageHammerDataset(ds["train"],      is_train=True,  max_words=args.max_words)
        val_ds   = MirageHammerDataset(ds["validation"], is_train=False, max_words=args.max_words)

        model = build_model(tokenizer, args.image_res, args.albef_pretrained).to(DEVICE)
        model, best_f1 = train_one_run(model, train_ds, val_ds, tokenizer, DEVICE, args)
        rep.write(f"best_val_macro_f1={best_f1:.4f}\n")

        blocks, all_metrics = evaluate_all_test_splits(
            model, ds, test_splits, tokenizer, DEVICE, args)
        for b in blocks: rep.write(b)
        write_summary(rep, all_metrics, test_splits)

        ckpt = os.path.join(args.checkpoint_dir, f"hammer_mirage_{timestamp}.pt")
        save_checkpoint(ckpt, model, args, all_metrics)

    # ── K-Fold ───────────────────────────────────────────────────────────────
    else:
        base_train = ds["train"]
        probe_ds   = MirageHammerDataset(base_train, is_train=False, max_words=args.max_words)
        labels     = np.array(probe_ds.labels)
        skf        = StratifiedKFold(n_splits=args.kfold_splits, shuffle=True,
                                     random_state=args.seed)
        rep.write(f"kfold_splits={args.kfold_splits}\n")
        rep.write("===== TEST RESULTS (KFold) =====\n")

        fold_results: List[Dict[str, Dict]] = []

        for fold_idx, (tr_idx, va_idx) in enumerate(
                skf.split(np.zeros(len(labels)), labels), start=1):
            print(f"\n[{LOG_TAG}] ===== Fold {fold_idx}/{args.kfold_splits} =====")
            rep.write(f"\n\n{'#'*70}\n### Fold {fold_idx}/{args.kfold_splits}\n{'#'*70}\n")

            train_ds = MirageHammerDataset(
                base_train.select(tr_idx.tolist()), is_train=True,  max_words=args.max_words)
            val_ds   = MirageHammerDataset(
                base_train.select(va_idx.tolist()), is_train=False, max_words=args.max_words)

            model = build_model(tokenizer, args.image_res, args.albef_pretrained).to(DEVICE)
            model, best_f1 = train_one_run(model, train_ds, val_ds, tokenizer, DEVICE, args)
            rep.write(f"best_val_macro_f1={best_f1:.4f}\n")

            blocks, fold_metrics = evaluate_all_test_splits(
                model, ds, test_splits, tokenizer, DEVICE, args)
            for b in blocks: rep.write(b)
            fold_results.append(fold_metrics)

            ckpt = os.path.join(args.checkpoint_dir,
                                f"hammer_mirage_kfold{args.kfold_splits}"
                                f"_{timestamp}_fold{fold_idx}.pt")
            save_checkpoint(ckpt, model, args, fold_metrics)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # KFold 글로벌 요약
        rep.write(f"\n\n===== KFOLD GLOBAL SUMMARY =====\n")
        global_metrics: Dict[str, Dict[str, float]] = {}
        for split_name in test_splits:
            global_metrics[split_name] = {}
            for mk in ("acc", "f1", "f1_ai", "auc"):
                vals = [fr[split_name][mk] for fr in fold_results
                        if not math.isnan(fr[split_name].get(mk, float("nan")))]
                global_metrics[split_name][mk] = float(np.mean(vals)) if vals else float("nan")

        for split_name in test_splits:
            m = global_metrics[split_name]
            line = (f"{split_name}:"
                    f"  Acc={m['acc']:.4f},"
                    f"  F1={m['f1']:.4f},"
                    f"  F1_AI={m['f1_ai']:.4f},"
                    f"  AUC={m['auc']:.4f}")
            print(line); rep.write(line + "\n")

        write_summary(rep, global_metrics, test_splits, tag="(mean over folds) ")

    _save_report(rep, timestamp, args)


# ════════════════════════════════════════════════════════════════════════════════
# 14. main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(
        description="HAMMER × MiRAGe-News (Binary: Real vs AI-Fake)")

    pa.add_argument("--hf_dataset_id",  default=HF_DATASET_ID)
    pa.add_argument("--result_dir",     default=RESULT_DIR)
    pa.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    pa.add_argument("--eval_only",      action="store_true")
    pa.add_argument("--checkpoint",     type=str, default="")
    pa.add_argument("--no_kfold",       action="store_true")
    pa.add_argument("--kfold_splits",   type=int,   default=KFOLD_SPLITS)
    pa.add_argument("--image_res",      type=int,   default=IMG_SIZE)
    pa.add_argument("--max_words",      type=int,   default=MAX_WORDS)
    pa.add_argument("--batch_size",     type=int,   default=BATCH)
    pa.add_argument("--num_workers",    type=int,   default=NUM_WORKERS)
    pa.add_argument("--epochs",         type=int,   default=EPOCHS)
    # LR: 아키텍처 고유 (BERT/ViT 차등)
    pa.add_argument("--lr",             type=float, default=PAPER_LR)
    pa.add_argument("--lr_img",         type=float, default=PAPER_LR_IMG)
    pa.add_argument("--weight_decay",   type=float, default=WEIGHT_DECAY)
    # warmup: MAC loss 초반 차단용 (HAMMER 고유)
    pa.add_argument("--warmup_epochs",  type=int,   default=2)
    pa.add_argument("--alpha",          type=float, default=0.4)
    pa.add_argument("--seed",           type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    # sampler: 기본 shuffle=True
    pa.add_argument("--weighted_sampler", action="store_true",
                    help="기본 shuffle=True → 이 플래그로 WeightedSampler 전환")
    pa.add_argument("--sampler_alpha",    type=float, default=SAMPLER_ALPHA)
    pa.add_argument("--albef_pretrained", type=str,   default=DEFAULT_ALBEF)

    args = pa.parse_args()
    set_seed(args.seed)

    os.makedirs(args.result_dir,     exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    mode         = "eval_only" if args.eval_only else ("Official" if args.no_kfold else f"KFold-{args.kfold_splits}")
    sampler_note = "WeightedSampler" if args.weighted_sampler else "shuffle=True"
    print(f"\n{LOG_TAG} | device={DEVICE} | mode={mode}")
    print(f"  [공통] batch={args.batch_size}  epochs={args.epochs}"
          f"  patience={args.early_stop_patience}"
          f"  early_stop=val_macro_F1  scheduler=None"
          f"  wd={args.weight_decay}  sampler={sampler_note}")
    print(f"  [HAMMER 고유] lr_bert={args.lr:.2e}  lr_vit={args.lr_img:.2e}"
          f"  warmup_MAC={args.warmup_epochs}ep")

    train_and_test(args)


if __name__ == "__main__":
    main()