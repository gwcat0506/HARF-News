"""
HAMMER × HARFM — 4-way Classification (HR / HF / AR / AF)
===========================================================
HAMMER의 복잡한 손실(MAC + BIC)을 완전히 활용하면서 4-way 분류 추가
[실행]
  python hammer.py               
  python hammer.py --no_kfold    # 60/20/20 단일
  python hammer.py --eval_only --checkpoint ./checkpoints/xxx.pt
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
from contextlib import redirect_stdout
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
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
from scheduler.scheduler_factory import create_scheduler  # noqa

LOG_TAG = "HAMMER-HARFM-4C"

if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════════════════════════════════
# 1. 상수
# ════════════════════════════════════════════════════════════════════════════════
SEED                 = 42
BATCH                = 16
EPOCHS               = 10
NUM_WORKERS          = 4
PAPER_LR             = 2e-5      # BERT
PAPER_LR_IMG         = 1e-4      # ViT
PAPER_LR_HEAD        = 2e-4      # 4-way head (새 모듈)
PAPER_WEIGHT_DECAY   = 0.02
PAPER_MIN_LR         = 1e-6
PAPER_WARMUP_LR      = 1e-6
PAPER_WARMUP_IN_50   = 10
PAPER_SCHED_TOTAL    = 50
EARLY_STOP_PATIENCE  = 4
EARLY_STOP_MIN_DELTA = 1e-4
SAMPLER_ALPHA        = 0.5
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2
KFOLD_SPLITS         = 5
KFOLD_RANDOM_STATE   = 42
KFOLD_TEST_SIZE      = 0.2

#loss 가중치
LOSS_MAC_WGT = 0.1   # warmup 동안은 0
LOSS_BIC_WGT = 1.0   # binary real/fake
LOSS_4C_WGT  = 1.0   # 4-way HR/HF/AR/AF

CLASS_NAMES = ["HR", "HF", "AR", "AF"]
CLASS2IDX   = {c: i for i, c in enumerate(CLASS_NAMES)}
# HAMMER binary label: HR+AR=real, HF+AF=fake
LABEL2HAMMER = {"HR": "orig", "AR": "orig",
                "HF": "gpt_image_edit", "AF": "gpt_image_edit"}

DEFAULT_CSV_PATH   = os.path.join(_FND_ROOT, "HARFM.csv")
DEFAULT_DATA_ROOT  = _FND_ROOT
DEFAULT_CKPT_DIR   = os.path.join(_FND_ROOT, "model", "checkpoints")
DEFAULT_RESULT_DIR = os.path.join(_HERE, "results")
DEFAULT_ALBEF      = os.path.join(_FND_ROOT, "experiment", "DGM4", "ALBEF_4M.pth")


# ════════════════════════════════════════════════════════════════════════════════
# 2. 유틸
# ════════════════════════════════════════════════════════════════════════════════

def set_seed(s: int = SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed()


def default_paper_warmup_epochs(train_epochs: int) -> int:
    if train_epochs <= 1:
        return 0
    w = int(round(train_epochs * (PAPER_WARMUP_IN_50 / PAPER_SCHED_TOTAL)))
    return max(1, min(train_epochs - 1, w))


def resolve_image_path(p: str, data_root: str) -> str:
    p = (p or "").strip()
    if not p:
        return ""
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    return os.path.join(os.path.abspath(data_root), p)


def pre_caption(caption: str, max_words: int) -> str:
    caption = (re.sub(r"([,.'!?\"()*#:;~])", "", caption.lower())
               .replace("-", " ").replace("/", " "))
    caption = re.sub(r"\s{2,}", " ", caption).strip()
    words = caption.split()
    return " ".join(words[:max_words]) if len(words) > max_words else caption


def make_weighted_sampler(labels: List[int], alpha: float = 0.5):
    s = pd.Series(labels)
    cnt = s.value_counts().to_dict()
    w = [(1.0 / cnt[l]) ** alpha for l in labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
        return len(data)
    def flush(self):
        for s in self.streams:
            if hasattr(s, "flush"): s.flush()


# ════════════════════════════════════════════════════════════════════════════════
# 3. 데이터 로드
# ════════════════════════════════════════════════════════════════════════════════

def load_harfm_df(csv_path: str, data_root: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path,
                     usecols=["final_headline", "image_path", "4_way_label"],
                     low_memory=False)
    df = df.dropna(subset=["final_headline", "4_way_label"])
    for c in ["final_headline", "image_path", "4_way_label"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df = df[df["4_way_label"].isin(CLASS_NAMES)].reset_index(drop=True)

    def _ok(p):
        full = resolve_image_path(p, data_root)
        return bool(full) and os.path.isfile(full)

    df = df[df["image_path"].map(_ok) &
            (df["final_headline"].str.len() > 0)].reset_index(drop=True)

    df["label"]        = df["4_way_label"].map(CLASS2IDX).astype(int)
    df["hammer_label"] = df["4_way_label"].map(LABEL2HAMMER)
    df["binary_label"] = df["4_way_label"].map(
        {"HR": 0, "AR": 0, "HF": 1, "AF": 1}).astype(int)

    lc = df["4_way_label"].value_counts().to_dict()
    print(f"[{LOG_TAG}] HARFM 로드: {len(df)}행 | {lc}")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# 4. EarlyStopping
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
# 5. Dataset
# ════════════════════════════════════════════════════════════════════════════════

class HARFMHammerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str,
                 transform, max_words: int, is_train: bool):
        super().__init__()
        self.data_root = os.path.abspath(data_root)
        self.transform = transform
        self.max_words = max_words
        self.image_res = int(transform.__dict__.get("_image_res", 256))
        self.is_train  = is_train
        self.df        = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row  = self.df.iloc[idx]
        full = resolve_image_path(str(row["image_path"]), self.data_root)
        try:
            img = Image.open(full).convert("RGB") if full and os.path.isfile(full) else None
        except Exception:
            img = None
        if img is None:
            img = Image.new("RGB", (self.image_res, self.image_res), (127, 127, 127))

        if self.is_train:
            if random.random() < 0.5:
                img = transforms.functional.hflip(img)
            img = transforms.functional.resize(
                img, [self.image_res, self.image_res], interpolation=Image.BICUBIC)

        img            = self.transform(img)
        hammer_label   = str(row["hammer_label"])      # "orig" or "gpt_image_edit"
        caption        = pre_caption(str(row["final_headline"]), self.max_words)
        label4         = int(row["label"])              # 0~3
        fake_image_box = torch.zeros(4, dtype=torch.float32)
        fake_text_pos  = torch.zeros(self.max_words, dtype=torch.float32)

        return img, hammer_label, caption, fake_image_box, fake_text_pos, label4


def collate_fn(batch):
    imgs          = torch.stack([b[0] for b in batch])
    hammer_labels = [b[1] for b in batch]
    captions      = [b[2] for b in batch]
    fake_boxes    = torch.stack([b[3] for b in batch])
    fake_pos      = torch.stack([b[4] for b in batch])
    labels4       = torch.tensor([b[5] for b in batch], dtype=torch.long)
    return imgs, hammer_labels, captions, fake_boxes, fake_pos, labels4


# ════════════════════════════════════════════════════════════════════════════════
# 6. Transforms
# ════════════════════════════════════════════════════════════════════════════════

def build_transforms(image_res: int, is_train: bool):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    if is_train:
        from dataset.randaugment import RandomAugment   # noqa
        tr = transforms.Compose([
            RandomAugment(2, 7, isPIL=True,
                          augs=["Identity","AutoContrast","Equalize",
                                "Brightness","Sharpness"]),
            transforms.ToTensor(), normalize,
        ])
    else:
        tr = transforms.Compose([
            transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
            transforms.ToTensor(), normalize,
        ])
    tr._image_res = image_res   # type: ignore[attr-defined]
    return tr


# ════════════════════════════════════════════════════════════════════════════════
# 7. Text 전처리 
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
        ftp = []
        fdec = np.where(fake_word_pos[bi].numpy() == 1)[0].tolist()
        sub  = np.array(text_input.word_ids(bi)[1:-1])
        for wi in fdec:
            ftp.extend(np.where(sub == wi)[0].tolist())
        fake_token_pos_batch.append(ftp)
    return text_input, fake_token_pos_batch


# ════════════════════════════════════════════════════════════════════════════════
# 8. HAMMER 4-way wrapper
#    핵심: itm_head에 hook을 걸어 멀티모달 CLS 피처 캡처 → 4-way 헤드
# ════════════════════════════════════════════════════════════════════════════════

class HammerFourClass(nn.Module):
    """
    HAMMER (binary_only=True) + 4-way 분류 헤드

    학습 손실:
      loss_MAC : HAMMER 모멘텀 대조 학습 (크로스모달 정렬)
      loss_BIC : HAMMER 이진 ITM (real vs fake)
      loss_4C  : 4-way CrossEntropy (HR / HF / AR / AF)

    itm_head forward hook:
      HAMMER가 itm_head 를 호출할 때 입력 피처(멀티모달 CLS, 768-dim)를
      self._mm_cls 에 저장 → 4-way 헤드에 전달.
    """
    def __init__(self, hammer: HAMMER, num_classes: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.hammer   = hammer
        self._mm_cls: Optional[torch.Tensor] = None

        #itm_head 에 hook 등록
        self._hook_handle = self.hammer.itm_head.register_forward_hook(
            self._capture_mm_cls)

        #4-way 헤드 (멀티모달 CLS 768-dim 입력)
        text_width = 768
        self.head4 = nn.Sequential(
            nn.Linear(text_width, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def _capture_mm_cls(self, module, inputs, output):
        """itm_head 입력(멀티모달 CLS 피처) 캡처."""
        if inputs and len(inputs) > 0:
            self._mm_cls = inputs[0]   # (B, 768)

    def _get_logits4(self) -> torch.Tensor:
        assert self._mm_cls is not None, "hook이 발동되지 않음 — HAMMER forward 순서 확인"
        return torch.nan_to_num(
            self.head4(self._mm_cls), nan=0., posinf=20., neginf=-20.)

    # ── 학습 forward ─────────────────────────────────────────────────────────
    def forward_train(self,
                      image: torch.Tensor,
                      hammer_label: List[str],
                      text_input,
                      fake_image_box: torch.Tensor,
                      fake_token_pos: List,
                      label4: torch.Tensor,
                      alpha: float = 0.4,
                      mac_wgt: float = LOSS_MAC_WGT,
                      bic_wgt: float = LOSS_BIC_WGT,
                      loss_4c_wgt: float = LOSS_4C_WGT,
                      ) -> Dict[str, torch.Tensor]:
        self._mm_cls = None

        # HAMMER 정상 학습 (MAC + BIC 손실 계산)
        loss_MAC, loss_BIC, _, _, _, _ = self.hammer(
            image, hammer_label, text_input,
            fake_image_box, fake_token_pos, alpha=alpha)

        # hook 에서 캡처된 멀티모달 CLS → 4-way
        logits4  = self._get_logits4()
        loss_4C  = F.cross_entropy(logits4, label4)

        total = mac_wgt * loss_MAC + bic_wgt * loss_BIC + loss_4c_wgt * loss_4C

        return {
            "total":    total,
            "loss_MAC": loss_MAC,
            "loss_BIC": loss_BIC,
            "loss_4C":  loss_4C,
            "logits4":  logits4,
        }

    # ── 평가 forward ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward_eval(self,
                     image: torch.Tensor,
                     hammer_label: List[str],
                     text_input,
                     fake_image_box: torch.Tensor,
                     fake_token_pos: List,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._mm_cls = None

        logits_binary, _, _, _ = self.hammer(
            image, hammer_label, text_input,
            fake_image_box, fake_token_pos, is_train=False)

        logits4 = self._get_logits4()
        return logits_binary, logits4

    def remove_hook(self):
        self._hook_handle.remove()


# ════════════════════════════════════════════════════════════════════════════════
# 9. 모델 빌드
# ════════════════════════════════════════════════════════════════════════════════

def interpolate_pos_embed(pos_embed: torch.Tensor,
                          new_num_tokens: int) -> torch.Tensor:
    cls_pos   = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]
    orig_size = int(patch_pos.shape[1] ** 0.5)
    new_size  = int((new_num_tokens - 1) ** 0.5)
    patch_pos = (patch_pos.reshape(1, orig_size, orig_size, -1)
                 .permute(0, 3, 1, 2))
    patch_pos = F.interpolate(patch_pos, size=(new_size, new_size),
                              mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, pos_embed.shape[-1])
    return torch.cat([cls_pos, patch_pos], dim=1)


def build_model(tokenizer, image_res: int,
                albef_pretrained: str = DEFAULT_ALBEF) -> HammerFourClass:
    bert_cfg = os.path.join(_VENDOR, "configs", "config_bert.json")
    cfg = {
        "embed_dim": 256, "image_res": image_res, "vision_width": 768,
        "queue_size": 65536, "momentum": 0.995, "temp": 0.07,
        "label_smoothing": 0.0, "bert_config": bert_cfg,
        "use_bbox": False, "binary_only": True,
    }
    margs  = SimpleNamespace(token_momentum=False)
    hammer = HAMMER(args=margs, config=cfg,
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
        miss, unexp = hammer.load_state_dict(sd, strict=False)
        print(f"[{LOG_TAG}]   missing={len(miss)}  unexpected={len(unexp)}")
    else:
        print(f"[{LOG_TAG}] ⚠️  ALBEF 없음 → 랜덤 초기화")

    return HammerFourClass(hammer, num_classes=4)


# ════════════════════════════════════════════════════════════════════════════════
# 10. 옵티마이저 & 스케줄러
# ════════════════════════════════════════════════════════════════════════════════

def build_optimizer(model: HammerFourClass, args) -> torch.optim.Optimizer:
    """
      visual_encoder → lr_img = 1e-4
      text_encoder   → lr     = 2e-5
      head4 + itm    → lr*10  = 2e-4
    """
    wd = args.weight_decay
    grp_img, grp_txt, grp_new = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "visual_encoder" in name or "visual_encoder_m" in name:
            grp_img.append(p)
        elif "text_encoder" in name or "text_encoder_m" in name:
            grp_txt.append(p)
        else:
            grp_new.append(p)

    return torch.optim.AdamW([
        {"params": grp_img,  "lr": args.lr_img,     "weight_decay": wd},
        {"params": grp_txt,  "lr": args.lr,          "weight_decay": wd},
        {"params": grp_new,  "lr": args.lr * 10,     "weight_decay": wd},
    ])


def build_scheduler(optimizer, args):
    warmup = args.warmup_epochs
    total  = args.epochs

    def lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(max(warmup, 1))
        progress = float(ep - warmup) / float(max(total - warmup, 1))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ════════════════════════════════════════════════════════════════════════════════
# 11. 학습 & 평가 epoch
# ════════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model: HammerFourClass, loader: DataLoader,
                    tokenizer, device, optimizer,
                    epoch_idx: int, alpha: float,
                    warmup_epochs: int) -> Dict[str, float]:
    model.train()
    # warmup 중 MAC 비활성 (큐가 랜덤 피처 상태)
    mac_wgt = 0.0 if epoch_idx <= warmup_epochs else LOSS_MAC_WGT

    tot_loss = tot_mac = tot_bic = tot_4c = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"[{LOG_TAG}] Epoch {epoch_idx}", file=sys.stderr)
    for imgs, hammer_labels, captions, fake_boxes, fake_pos, labels4 in pbar:
        imgs       = imgs.to(device, non_blocking=True)
        fake_boxes = fake_boxes.to(device, non_blocking=True)
        labels4    = labels4.to(device, non_blocking=True)

        text_input = tokenizer(
            captions, max_length=128, truncation=True,
            add_special_tokens=True, return_attention_mask=True,
            return_token_type_ids=False)
        text_input, fake_token_pos = text_input_adjust(text_input, fake_pos, device)

        a = (alpha if epoch_idx > 1
             else alpha * min(1.0, (pbar.n + 1) / max(len(loader), 1)))

        out = model.forward_train(
            imgs, hammer_labels, text_input, fake_boxes, fake_token_pos,
            labels4, alpha=a, mac_wgt=mac_wgt)

        optimizer.zero_grad(set_to_none=True)
        out["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        B = imgs.size(0)
        tot_loss += out["total"].item()  * B
        tot_mac  += out["loss_MAC"].item() * B
        tot_bic  += out["loss_BIC"].item() * B
        tot_4c   += out["loss_4C"].item()  * B
        n += B
        pbar.set_postfix(
            loss=f"{out['total'].item():.3f}",
            MAC=f"{out['loss_MAC'].item():.3f}",
            BIC=f"{out['loss_BIC'].item():.3f}",
            C4=f"{out['loss_4C'].item():.3f}",
            mac_on="Y" if mac_wgt > 0 else "N",
        )

    d = max(n, 1)
    return dict(loss=tot_loss/d, loss_MAC=tot_mac/d,
                loss_BIC=tot_bic/d, loss_4C=tot_4c/d)


def auc_scores_from_proba(yt: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """ROC-AUC from 4-class softmax: macro OVR, plus merged H/A and R/F scores."""
    out: Dict[str, float] = {}
    if len(yt) == 0 or proba.shape[0] != len(yt):
        for k in ("auc_4_ovr_macro", "auc_ha", "auc_rf"):
            out[k] = float("nan")
        return out
    try:
        out["auc_4_ovr_macro"] = float(
            roc_auc_score(
                yt,
                proba,
                multi_class="ovr",
                average="macro",
                labels=[0, 1, 2, 3],
            )
        )
    except ValueError:
        out["auc_4_ovr_macro"] = float("nan")
    yh = yt // 2
    p_ai = proba[:, 2:4].sum(axis=1)
    try:
        out["auc_ha"] = float(roc_auc_score(yh, p_ai))
    except ValueError:
        out["auc_ha"] = float("nan")
    yr = yt % 2
    p_fake = proba[:, [1, 3]].sum(axis=1)
    try:
        out["auc_rf"] = float(roc_auc_score(yr, p_fake))
    except ValueError:
        out["auc_rf"] = float("nan")
    return out


@torch.no_grad()
def eval_epoch(model: HammerFourClass, loader: DataLoader,
               tokenizer, device) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for imgs, hammer_labels, captions, fake_boxes, fake_pos, labels4 in loader:
        imgs       = imgs.to(device, non_blocking=True)
        fake_boxes = fake_boxes.to(device, non_blocking=True)

        text_input = tokenizer(
            captions, max_length=128, truncation=True,
            add_special_tokens=True, return_attention_mask=True,
            return_token_type_ids=False)
        text_input, fake_token_pos = text_input_adjust(text_input, fake_pos, device)

        _, logits4 = model.forward_eval(
            imgs, hammer_labels, text_input, fake_boxes, fake_token_pos)

        ys.extend(labels4.tolist())
        ps.extend(logits4.argmax(1).cpu().tolist())

    ys = np.array(ys); ps = np.array(ps)
    yh, ph = ys // 2, ps // 2
    yr, pr = ys % 2,  ps % 2
    return {
        "acc_4":  float(accuracy_score(ys, ps)),
        "f1_4":   float(f1_score(ys, ps, average="macro", zero_division=0)),
        "acc_ha": float(accuracy_score(yh, ph)),
        "f1_ha":  float(f1_score(yh, ph, average="macro", zero_division=0)),
        "acc_rf": float(accuracy_score(yr, pr)),
        "f1_rf":  float(f1_score(yr, pr, average="macro", zero_division=0)),
    }


# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_4class(model: HammerFourClass, loader: DataLoader,
                    tokenizer, device) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    prob_chunks: List[np.ndarray] = []
    for imgs, hammer_labels, captions, fake_boxes, fake_pos, labels4 in tqdm(
            loader, desc="Evaluation", file=sys.stderr):
        imgs       = imgs.to(device, non_blocking=True)
        fake_boxes = fake_boxes.to(device, non_blocking=True)

        text_input = tokenizer(
            captions, max_length=128, truncation=True,
            add_special_tokens=True, return_attention_mask=True,
            return_token_type_ids=False)
        text_input, fake_token_pos = text_input_adjust(text_input, fake_pos, device)

        _, logits4 = model.forward_eval(
            imgs, hammer_labels, text_input, fake_boxes, fake_token_pos)

        prob_chunks.append(
            F.softmax(logits4, dim=-1).detach().cpu().numpy()
        )
        ys.extend(labels4.tolist())
        ps.extend(logits4.argmax(1).cpu().tolist())

    ys = np.array(ys); ps = np.array(ps)
    yh, ph = ys // 2, ps // 2
    yr, pr = ys % 2,  ps % 2

    acc_4  = float(accuracy_score(ys, ps))
    acc_ha = float(accuracy_score(yh, ph))
    acc_rf = float(accuracy_score(yr, pr))
    f1_4   = float(f1_score(ys, ps, average="macro",  zero_division=0))
    f1_ha  = float(f1_score(yh, ph, average="macro",  zero_division=0))
    f1_rf  = float(f1_score(yr, pr, average="macro",  zero_division=0))

    print("\n=== HAMMER Human vs AI ===")
    print(classification_report(yh, ph, target_names=["Human", "AI"],
                                digits=4, zero_division=0))
    print("Confusion matrix (H/A):")
    print(confusion_matrix(yh, ph))

    print("\n=== HAMMER Real vs Fake ===")
    print(classification_report(yr, pr, target_names=["Real", "Fake"],
                                digits=4, zero_division=0))
    print("Confusion matrix (R/F):")
    print(confusion_matrix(yr, pr))

    print("\n=== HAMMER 4-class ===")
    print(classification_report(ys, ps, target_names=CLASS_NAMES,
                                digits=4, zero_division=0))
    print("Confusion matrix (4-way):")
    print(confusion_matrix(ys, ps))

    proba = np.vstack(prob_chunks) if prob_chunks else np.zeros((0, 4), dtype=np.float64)
    aucm = auc_scores_from_proba(ys, proba)
    print(
        f"\n[{LOG_TAG}] ROC-AUC: 4-class (macro-OVR)={aucm['auc_4_ovr_macro']:.4f} | "
        f"H/A={aucm['auc_ha']:.4f} | R/F={aucm['auc_rf']:.4f}"
    )
    print(f"\n[{LOG_TAG}] "
          f"H/A Acc={acc_ha:.4f}, F1={f1_ha:.4f} | "
          f"R/F Acc={acc_rf:.4f}, F1={f1_rf:.4f} | "
          f"4-way Acc={acc_4:.4f}, F1={f1_4:.4f} | "
          f"AUC(4c)={aucm['auc_4_ovr_macro']:.4f} AUC(H/A)={aucm['auc_ha']:.4f} "
          f"AUC(R/F)={aucm['auc_rf']:.4f}")

    return dict(
        acc_ha=acc_ha, f1_ha=f1_ha,
        acc_rf=acc_rf, f1_rf=f1_rf,
        acc_4=acc_4,  f1_4=f1_4,
        auc_4_ovr_macro=aucm["auc_4_ovr_macro"],
        auc_ha=aucm["auc_ha"],
        auc_rf=aucm["auc_rf"],
    )


# ════════════════════════════════════════════════════════════════════════════════
# 13. 학습 루프
# ════════════════════════════════════════════════════════════════════════════════

def train_one_run(model: HammerFourClass, tl, vl, device,
                  tokenizer, args, log=None) -> HammerFourClass:
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    early     = EarlyStopping(args.early_stop_patience,
                               args.early_stop_min_delta, mode="max")
    best_f1    = -1.
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, tl, tokenizer, device, optimizer,
                             ep, args.alpha, args.warmup_epochs)
        scheduler.step()
        va = eval_epoch(model, vl, tokenizer, device)

        cur_lr = optimizer.param_groups[-1]["lr"]
        mac_on = "Y" if ep > args.warmup_epochs else "N"
        line = (f"[{LOG_TAG}] Epoch {ep:02d}/{args.epochs}"
                f"  loss={tr['loss']:.4f}"
                f"  MAC={tr['loss_MAC']:.4f}({mac_on})"
                f"  BIC={tr['loss_BIC']:.4f}"
                f"  4C={tr['loss_4C']:.4f}"
                f"  val_f1={va['f1_4']:.4f}"
                f"  H/A={va['f1_ha']:.4f}"
                f"  R/F={va['f1_rf']:.4f}"
                f"  lr={cur_lr:.2e}")
        print(line)
        if log is not None:
            log.append(line)

        if va["f1_4"] > best_f1:
            best_f1    = va["f1_4"]
            best_state = copy.deepcopy(model.state_dict())

        if early.step(va["f1_4"]):
            msg = (f"[{LOG_TAG}] Early stopping at epoch {ep}"
                   f" (best val_f1={best_f1:.4f})")
            print(msg)
            if log: log.append(msg)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{LOG_TAG}] best val 4-way F1={best_f1:.4f} 가중치 복원 완료")
    return model


# ════════════════════════════════════════════════════════════════════════════════
# 14. 체크포인트 I/O
# ════════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path: str, model: HammerFourClass, args, metrics=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config":  vars(args),
        "metrics": metrics,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }, path)
    print(f"[{LOG_TAG}] checkpoint saved: {path}")


def load_checkpoint(model: HammerFourClass, path: str, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[{LOG_TAG}] load_checkpoint  missing={len(miss)}  unexpected={len(unexp)}")


# ════════════════════════════════════════════════════════════════════════════════
# 15. DataLoader 헬퍼
# ════════════════════════════════════════════════════════════════════════════════

def make_loader(df, data_root, transform, max_words, is_train,
                batch_size, num_workers, shuffle,
                weighted=False, sampler_alpha=0.5, drop_last=False):
    ds = HARFMHammerDataset(df, data_root, transform, max_words, is_train)
    kw: Dict[str, Any] = dict(
        batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), drop_last=drop_last)
    if weighted:
        kw["sampler"] = make_weighted_sampler(df["label"].tolist(), sampler_alpha)
        kw["shuffle"] = False
    else:
        kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


# ════════════════════════════════════════════════════════════════════════════════
# 16. K-Fold
# ════════════════════════════════════════════════════════════════════════════════

def run_kfold(df, device, tokenizer, args) -> None:
    y   = df["label"].values
    idx = np.arange(len(df))
    k   = args.kfold_splits
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    tva, te, _, _ = train_test_split(idx, y,
                                     test_size=args.kfold_test_size,
                                     stratify=y, random_state=args.seed)
    df_te = df.iloc[te].reset_index(drop=True)
    te_tf = build_transforms(args.image_res, is_train=False)
    el    = make_loader(df_te, args.data_root, te_tf, args.max_words, False,
                        args.batch_size, args.num_workers, shuffle=False)
    print(f"[{LOG_TAG}] KFold(k={k}) | TrainVal={len(tva)} Test={len(df_te)}")

    all_rep = io.StringIO()
    all_rep.write(f"###### {LOG_TAG} (k={k} KFold) ######\n")
    all_rep.write(f"손실 구성: λ_MAC={LOSS_MAC_WGT} + λ_BIC={LOSS_BIC_WGT}"
                  f" + λ_4C={LOSS_4C_WGT}\n\n")
    fold_m: List[Dict] = []

    skf = StratifiedKFold(n_splits=k, shuffle=True,
                          random_state=args.kfold_random_state)
    for fold, (str_, sva_) in enumerate(
            skf.split(np.zeros(len(tva)), y[tva]), 1):
        print(f"\n{'='*70}\n[Fold {fold}/{k}] {LOG_TAG}\n{'='*70}")

        df_tr_f = df.iloc[tva[str_]].reset_index(drop=True)
        df_va_f = df.iloc[tva[sva_]].reset_index(drop=True)
        tr_tf   = build_transforms(args.image_res, is_train=True)

        tl = make_loader(df_tr_f, args.data_root, tr_tf, args.max_words, True,
                         args.batch_size, args.num_workers, shuffle=True,
                         weighted=True,
                         sampler_alpha=args.sampler_alpha, drop_last=True)
        vl = make_loader(df_va_f, args.data_root, te_tf, args.max_words, False,
                         args.batch_size, args.num_workers, shuffle=False)

        model = build_model(tokenizer, args.image_res, args.albef_pretrained).to(device)

        buf = io.StringIO()
        tee = Tee(sys.stdout, buf)
        log: List[str] = []
        with redirect_stdout(tee):
            model = train_one_run(model, tl, vl, device, tokenizer, args, log)
            res   = evaluate_4class(model, el, tokenizer, device)
            print(f"\n{'='*70}")
            print(f"[Fold {fold}] SUMMARY: "
                  f"H/A Acc={res['acc_ha']:.4f}, F1={res['f1_ha']:.4f} | "
                  f"R/F Acc={res['acc_rf']:.4f}, F1={res['f1_rf']:.4f} | "
                  f"4-way Acc={res['acc_4']:.4f}, F1={res['f1_4']:.4f} | "
                  f"AUC(4c)={res['auc_4_ovr_macro']:.4f} AUC(H/A)={res['auc_ha']:.4f} "
                  f"AUC(R/F)={res['auc_rf']:.4f}")
            print(f"{'='*70}")

        all_rep.write(f"\n\n{'#'*80}\n### FOLD {fold}/{k}\n{'#'*80}\n")
        all_rep.write(buf.getvalue())

        ckpt = os.path.join(args.checkpoint_dir,
                            f"hammer_harfm4c_kfold_{ts}_fold{fold}.pt")
        save_checkpoint(ckpt, model, args, res)
        fold_m.append(res)
        model.remove_hook()

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _mean(k_): return float(np.mean([r[k_] for r in fold_m]))
    g = (f"H/A Acc={_mean('acc_ha'):.4f}, F1={_mean('f1_ha'):.4f} | "
         f"R/F Acc={_mean('acc_rf'):.4f}, F1={_mean('f1_rf'):.4f} | "
         f"4-way Acc={_mean('acc_4'):.4f}, F1={_mean('f1_4'):.4f} | "
         f"AUC(4c)={_mean('auc_4_ovr_macro'):.4f} AUC(H/A)={_mean('auc_ha'):.4f} "
         f"AUC(R/F)={_mean('auc_rf'):.4f}")
    all_rep.write(f"\n\n{'#'*80}\n### GLOBAL SUMMARY (mean over {k} folds)\n{'#'*80}\n")
    all_rep.write(f"{LOG_TAG}: {g}\n")
    all_rep.write("\n[TABLE_SUMMARY]\n")
    all_rep.write(
        "Variant\tHA_Acc\tHA_F1\tRF_Acc\tRF_F1\t4C_Acc\t4C_F1\tAUC_4c\tAUC_HA\tAUC_RF\n"
    )
    all_rep.write(f"HAMMER-4C\t"
                  f"{_mean('acc_ha'):.4f}\t{_mean('f1_ha'):.4f}\t"
                  f"{_mean('acc_rf'):.4f}\t{_mean('f1_rf'):.4f}\t"
                  f"{_mean('acc_4'):.4f}\t{_mean('f1_4'):.4f}\t"
                  f"{_mean('auc_4_ovr_macro'):.4f}\t{_mean('auc_ha'):.4f}\t"
                  f"{_mean('auc_rf'):.4f}\n")

    path = os.path.join(args.result_dir, f"hammer_harfm4c_kfold_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(all_rep.getvalue())
    print(f"\n[{LOG_TAG}] 리포트: {path}")
    print(f"[{LOG_TAG}] {g}")


# ════════════════════════════════════════════════════════════════════════════════
# 17. 단일 60/20/20
# ════════════════════════════════════════════════════════════════════════════════

def run_single(df, device, tokenizer, args) -> None:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    y   = df["label"].values
    idx = np.arange(len(df))
    tva, te, _, _ = train_test_split(idx, y, test_size=TEST_RATIO,
                                     stratify=y, random_state=args.seed)
    tr, va, _, _  = train_test_split(
        tva, y[tva],
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        stratify=y[tva], random_state=args.seed)

    df_tr = df.iloc[tr].reset_index(drop=True)
    df_va = df.iloc[va].reset_index(drop=True)
    df_te = df.iloc[te].reset_index(drop=True)
    print(f"[{LOG_TAG}] Train={len(df_tr)} Val={len(df_va)} Test={len(df_te)}")

    tr_tf = build_transforms(args.image_res, is_train=True)
    te_tf = build_transforms(args.image_res, is_train=False)

    tl = make_loader(df_tr, args.data_root, tr_tf, args.max_words, True,
                     args.batch_size, args.num_workers, shuffle=True,
                     weighted=True,
                     sampler_alpha=args.sampler_alpha, drop_last=True)
    vl = make_loader(df_va, args.data_root, te_tf, args.max_words, False,
                     args.batch_size, args.num_workers, shuffle=False)
    el = make_loader(df_te, args.data_root, te_tf, args.max_words, False,
                     args.batch_size, args.num_workers, shuffle=False)

    model = build_model(tokenizer, args.image_res, args.albef_pretrained).to(device)
    log: List[str] = []
    model = train_one_run(model, tl, vl, device, tokenizer, args, log)
    res   = evaluate_4class(model, el, tokenizer, device)

    ckpt_path = os.path.join(args.checkpoint_dir, f"hammer_harfm4c_single_{ts}.pt")
    save_checkpoint(ckpt_path, model, args, res)

    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (60/20/20) ######\n\n")
    for line in log: rep.write(line + "\n")
    rep.write(f"\nTest: {res}\n")
    path = os.path.join(args.result_dir, f"hammer_harfm4c_single_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rep.getvalue())
    print(f"[{LOG_TAG}] 리포트: {path}")


# ════════════════════════════════════════════════════════════════════════════════
# 18. eval_only
# ════════════════════════════════════════════════════════════════════════════════

def run_eval_only(df, device, tokenizer, args) -> None:
    te_tf  = build_transforms(args.image_res, is_train=False)
    loader = make_loader(df, args.data_root, te_tf, args.max_words, False,
                         args.batch_size, args.num_workers, shuffle=False)
    model = build_model(tokenizer, args.image_res, args.albef_pretrained).to(device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)
    else:
        print(f"[{LOG_TAG}] ⚠️  --checkpoint 없음 → 랜덤 초기화 (지표는 참고용)")
    evaluate_4class(model, loader, tokenizer, device)


# ════════════════════════════════════════════════════════════════════════════════
# 19. main
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    pa = argparse.ArgumentParser(
        description="HAMMER × HARFM — 4-way (HR/HF/AR/AF) "
                    "with MAC + BIC + 4C losses")
    pa.add_argument("--csv_path",     default=DEFAULT_CSV_PATH)
    pa.add_argument("--data_root",    default=DEFAULT_DATA_ROOT)
    pa.add_argument("--result_dir",   default=DEFAULT_RESULT_DIR)
    pa.add_argument("--checkpoint_dir", default=DEFAULT_CKPT_DIR)
    pa.add_argument("--eval_only",    action="store_true")
    pa.add_argument("--checkpoint",   type=str,   default="")
    pa.add_argument("--no_kfold",     action="store_true",
                    help="K-Fold 대신 60/20/20 단일 분할")
    pa.add_argument("--image_res",    type=int,   default=256)
    pa.add_argument("--max_words",    type=int,   default=50)
    pa.add_argument("--batch_size",   type=int,   default=BATCH)
    pa.add_argument("--num_workers",  type=int,   default=NUM_WORKERS)
    pa.add_argument("--epochs",       type=int,   default=EPOCHS)
    pa.add_argument("--lr",           type=float, default=PAPER_LR)
    pa.add_argument("--lr_img",       type=float, default=PAPER_LR_IMG)
    pa.add_argument("--weight_decay", type=float, default=PAPER_WEIGHT_DECAY)
    pa.add_argument("--warmup_epochs",type=int,   default=-1)
    pa.add_argument("--alpha",        type=float, default=0.4)
    pa.add_argument("--seed",         type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    pa.add_argument("--weighted_sampler", action="store_true")
    pa.add_argument("--sampler_alpha",    type=float, default=SAMPLER_ALPHA)
    pa.add_argument("--kfold_splits",     type=int,   default=KFOLD_SPLITS)
    pa.add_argument("--kfold_test_size",  type=float, default=KFOLD_TEST_SIZE)
    pa.add_argument("--kfold_random_state",type=int,  default=KFOLD_RANDOM_STATE)
    pa.add_argument("--albef_pretrained", type=str,   default=DEFAULT_ALBEF)

    args = pa.parse_args()
    set_seed(args.seed)

    if args.warmup_epochs < 0:
        args.warmup_epochs = default_paper_warmup_epochs(args.epochs)
    if args.epochs > 1 and args.warmup_epochs > args.epochs - 1:
        args.warmup_epochs = args.epochs - 1

    os.makedirs(args.result_dir,     exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    mode = ("eval_only" if args.eval_only
            else "60/20/20" if args.no_kfold
            else f"KFold-{args.kfold_splits}")
    print(f"\n{LOG_TAG} | device={DEVICE} | mode={mode}")
    print(f"  손실: λ_MAC={LOSS_MAC_WGT} + λ_BIC={LOSS_BIC_WGT} + λ_4C={LOSS_4C_WGT}")
    print(f"  lr={args.lr:.2e}  lr_img={args.lr_img:.2e}"
          f"  wd={args.weight_decay}  warmup={args.warmup_epochs}ep"
          f"  epochs={args.epochs}  batch={args.batch_size}"
          f"  sampler=WeightedRandomSampler(α={args.sampler_alpha})")  # ★ 항상 weighted

    df        = load_harfm_df(args.csv_path, args.data_root)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    if args.eval_only:
        run_eval_only(df, DEVICE, tokenizer, args)
    elif args.no_kfold:
        run_single(df, DEVICE, tokenizer, args)
    else:
        run_kfold(df, DEVICE, tokenizer, args)


if __name__ == "__main__":
    main()
