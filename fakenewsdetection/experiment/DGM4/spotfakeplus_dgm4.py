"""
SpotFake+ × DGM4 벤치마크 실험 코드
=====================================
SpotFake+ 모델(XLNet 텍스트 + VGG16 이미지 + MLP fusion)을
DGM4 이진 분류 벤치마크에 적용.

사용 방법:
  python spotfakeplus_DGM4.py --data_root ./datasets/DGM4
  python spotfakeplus_DGM4.py --data_root ... --eval_only --checkpoint ./checkpoints/xxx.pt
  python spotfakeplus_DGM4.py --data_root ... --kfold
"""

from __future__ import annotations

import argparse
import copy
import gc
import io
import math
import os
import random
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm
from transformers import XLNetModel, XLNetTokenizer

from dgm4_paths import DEFAULT_DATA_ROOT, FINE_LABELS, load_dgm4_splits, resolve_image_path

# ════════════════════════════════════════════════════════════════════════════════
# 1. 상수 
# ════════════════════════════════════════════════════════════════════════════════
BATCH                = 16
EPOCHS               = 10
LR                   = 2e-4
NUM_WORKERS          = 4
EARLY_STOP_PATIENCE  = 4
EARLY_STOP_MIN_DELTA = 1e-4
KFOLD_SPLITS         = 5
KFOLD_RANDOM_STATE   = 42
SEED                 = 42
SAMPLER_ALPHA        = 0.5

PAPER_WEIGHT_DECAY           = 0.02
PAPER_WARMUP_IN_50_EPOCHS    = 10
PAPER_SCHED_TOTAL_EPOCHS     = 50

# SpotFake+ 구조 고유 상수
TEXT_MODEL            = "xlnet-base-cased"
MAX_LEN               = 128
IMG_SIZE              = 224
GRAD_CLIP_NORM        = 1.0        
XLNET_UNFREEZE_LAYERS = (10, 11)  

# LR 그룹 배율 
LR_MULT_TEXT   = 0.05   # 2e-4 × 0.05 = 1e-5  (XLNet)
LR_MULT_VISUAL = 0.50   # 2e-4 × 0.50 = 1e-4  (VGG16 classifier)
LR_MULT_NEW    = 1.00   # 2e-4 × 1.00 = 2e-4  (Fusion MLP)

LOG_TAG = "SPOTFAKE-DGM4"

if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def default_paper_warmup_epochs(train_epochs: int) -> int:
    """50-epoch paper warmup 비율을 train_epochs에 맞게 스케일."""
    if train_epochs <= 1:
        return 0
    w = int(round(train_epochs * (PAPER_WARMUP_IN_50_EPOCHS / PAPER_SCHED_TOTAL_EPOCHS)))
    return max(1, min(train_epochs - 1, w))


def set_seed(s: int = SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


set_seed()


# ════════════════════════════════════════════════════════════════════════════════
# 2. 유틸
# ════════════════════════════════════════════════════════════════════════════════

def _tqdm(it, **kw):
    if os.environ.get("DGM4_NO_TQDM", "").lower() in ("1", "true", "yes"):
        return it
    kw.setdefault("file", sys.stderr)
    kw.setdefault("dynamic_ncols", True)
    try:
        sys.stdout.flush()
        return tqdm(it, **kw)
    except Exception:
        return it


def _ensure_dgm4_columns(df: pd.DataFrame) -> pd.DataFrame:
    """load_dgm4_splits 반환 DataFrame에 필요 컬럼이 없을 경우 자동 생성."""
    df = df.copy()
    if "has_image_flag" not in df.columns:
        df["has_image_flag"] = df["image"].apply(
            lambda p: isinstance(p, str) and len(p) > 0
        )
    if "binary_label" not in df.columns:
        df["binary_label"] = (df["fake_cls"] != "orig").astype(int)
    for lbl in FINE_LABELS:
        col = f"fine_{lbl}"
        if col not in df.columns:
            df[col] = (df["fake_cls"] == lbl).astype(float)
    return df


# ════════════════════════════════════════════════════════════════════════════════
# 3. EarlyStopping
# ════════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE,
                 min_delta=EARLY_STOP_MIN_DELTA, mode="max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = None
        self.counter   = 0

    def step(self, v: float) -> bool:
        if self.best is None:
            self.best = v
            return False
        imp = (v - self.best > self.min_delta if self.mode == "max"
               else self.best - v > self.min_delta)
        if imp:
            self.best = v
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ════════════════════════════════════════════════════════════════════════════════
# 4. Dataset
# ════════════════════════════════════════════════════════════════════════════════

class DGM4SpotFakeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str,
                 tokenizer, max_len: int = MAX_LEN):
        super().__init__()
        self.data_root = os.path.abspath(data_root)
        self.tokenizer = tokenizer
        self.max_len   = max_len

        df = _ensure_dgm4_columns(df)

        self.texts        = df["text"].tolist()
        self.image_rels   = df["image"].tolist()
        self.has_img_flag = df["has_image_flag"].tolist()
        self.binary_lbls  = df["binary_label"].tolist()
        self.fake_cls_str = df["fake_cls"].tolist()

    def __len__(self):
        return len(self.binary_lbls)

    def _load_pil(self, rel: str, flag: bool) -> Image.Image:
        if not flag or not rel:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127))
        full = resolve_image_path(rel, self.data_root)
        if not full or not os.path.isfile(full):
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127))
        try:
            return Image.open(full).convert("RGB")
        except Exception:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127))

    def __getitem__(self, idx):
        text = self.texts[idx]
        img  = self._load_pil(self.image_rels[idx], self.has_img_flag[idx])

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image":          image_transform(img),
            "binary_label":   torch.tensor(self.binary_lbls[idx], dtype=torch.long),
            "text":           text,
            "image_path":     self.image_rels[idx],
            "fake_cls":       self.fake_cls_str[idx],
        }


def collate_dgm4_spotfake(batch):
    tensor_keys = ("input_ids", "attention_mask", "image", "binary_label")
    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out["texts"]       = [b["text"]       for b in batch]
    out["image_paths"] = [b["image_path"] for b in batch]
    out["fake_cls"]    = [b["fake_cls"]   for b in batch]
    return out


def make_weighted_sampler(binary_labels: List[int], alpha: float = 0.5):
    s   = pd.Series(binary_labels)
    cnt = s.value_counts().to_dict()
    w   = [(1.0 / cnt[l]) ** alpha for l in binary_labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


# ════════════════════════════════════════════════════════════════════════════════
# 5. SpotFake+ 모델
# ════════════════════════════════════════════════════════════════════════════════

class SpotFakePlusDGM4(nn.Module):
    """XLNet(768) + VGG16(4096) → Fusion MLP → Binary (real/fake)."""
    def __init__(self, text_model_name: str = TEXT_MODEL):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained(text_model_name)

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg_feat = nn.Sequential(
            vgg.features,
            vgg.avgpool,
            nn.Flatten(),
            vgg.classifier[0],  # Linear(25088, 4096)
            vgg.classifier[1],  # ReLU
            vgg.classifier[2],  # Dropout
            vgg.classifier[3],  # Linear(4096, 4096)
            vgg.classifier[4],  # ReLU
        )  # → (B, 4096)

        fusion_in = 768 + 4096
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 2000), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(2000, 500),       nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(500, 100),        nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(100, 2),
        )

    def forward(self, input_ids, attention_mask, image):
        out       = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = out.last_hidden_state[:, -1, :]   # (B, 768) — XLNet: last token
        img_feat  = self.vgg_feat(image)               # (B, 4096)
        return self.fusion(torch.cat([text_feat, img_feat], dim=1))


# ════════════════════════════════════════════════════════════════════════════════
# 6. 백본 설정
# ════════════════════════════════════════════════════════════════════════════════

def configure_backbones(model: SpotFakePlusDGM4,
                        xlnet_unfreeze_layers=XLNET_UNFREEZE_LAYERS):
    """
    XLNet: 상위 2개 레이어만 학습, 나머지 freeze
    VGG16 features: 전체 freeze (features는 저수준 피처 추출)
    VGG16 classifier (vgg_feat 내 Linear 2개): 학습 가능
    Fusion MLP: 전체 학습
    """
    # XLNet
    for name, p in model.xlnet.named_parameters():
        p.requires_grad_(any(f"layer.{i}." in name for i in xlnet_unfreeze_layers))

    # VGG features freeze, classifier (vgg_feat 내) 학습
    for name, p in model.vgg_feat.named_parameters():
        # Sequential[0] = vgg.features → freeze
        # Sequential[3,6] = Linear → 학습
        is_linear = any(f"{i}." in name or name.startswith(f"{i}.") or name == str(i)
                        for i in [3, 6])
        p.requires_grad_(is_linear)

    # Fusion: 전체 학습
    for p in model.fusion.parameters():
        p.requires_grad_(True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{LOG_TAG}] 학습 파라미터: {n_train:,} / {n_total:,}"
          f" ({100*n_train/n_total:.1f}%)")


# ════════════════════════════════════════════════════════════════════════════════
# 7. 옵티마이저 & 스케줄러
# ════════════════════════════════════════════════════════════════════════════════

def make_optimizer(model: SpotFakePlusDGM4, args) -> torch.optim.Optimizer:
    """
    3-group 차등 LR:
      text   (XLNet 학습 레이어)  → lr × 0.05 = 1e-5
      visual (VGG classifier)     → lr × 0.50 = 1e-4
      new    (Fusion MLP)         → lr × 1.00 = 2e-4
    """
    grp_text, grp_visual, grp_new = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "xlnet" in name:
            grp_text.append(p)
        elif "vgg_feat" in name:
            grp_visual.append(p)
        else:
            grp_new.append(p)

    wd = getattr(args, "weight_decay", PAPER_WEIGHT_DECAY)
    groups = [
        {"params": grp_text,   "lr": args.lr * LR_MULT_TEXT,   "weight_decay": wd},
        {"params": grp_visual, "lr": args.lr * LR_MULT_VISUAL, "weight_decay": wd},
        {"params": grp_new,    "lr": args.lr * LR_MULT_NEW,    "weight_decay": wd},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]])


def make_scheduler(optimizer, args):
    """Linear Warmup + Cosine Annealing."""
    warmup = args.warmup_epochs
    total  = args.epochs

    def lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(max(warmup, 1))
        progress = float(ep - warmup) / float(max(total - warmup, 1))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ════════════════════════════════════════════════════════════════════════════════
# 8. 학습 & 평가 epoch
# ════════════════════════════════════════════════════════════════════════════════

def run_epoch(model: SpotFakePlusDGM4, loader, device,
              optimizer=None, epoch_idx=None, scheduler=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    ce   = nn.CrossEntropyLoss()
    tot, ys, ps, probs, n = 0., [], [], [], 0
    ctx  = torch.enable_grad() if train else torch.no_grad()
    desc = f"[{LOG_TAG}] Epoch {epoch_idx}" if (train and epoch_idx) else None
    it   = _tqdm(loader, desc=desc) if (train and desc) else loader

    with ctx:
        for batch in it:
            ids   = batch["input_ids"].to(device)
            attn  = batch["attention_mask"].to(device)
            img   = batch["image"].to(device)
            y_bin = batch["binary_label"].to(device)

            logits = model(ids, attn, img)
            loss   = ce(logits, y_bin)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            prob = torch.softmax(logits, dim=-1)[:, 1].detach()
            pred = logits.argmax(1).detach()
            tot  += loss.item() * y_bin.size(0)
            ys.extend(y_bin.cpu().tolist())
            ps.extend(pred.cpu().tolist())
            probs.extend(prob.cpu().tolist())
            n += y_bin.size(0)

    if train and scheduler is not None:
        scheduler.step()

    n         = max(n, 1)
    ys_arr    = np.array(ys)
    probs_arr = np.array(probs)
    ps_arr    = np.array(ps)

    try:
        auc = float(roc_auc_score(ys_arr, probs_arr))
    except Exception:
        auc = float("nan")

    # Youden's J 최적 임계값 F1
    try:
        fpr, tpr, thresholds = roc_curve(ys_arr, probs_arr)
        best_thresh = float(thresholds[np.argmax(tpr - fpr)])
        ps_opt      = (probs_arr >= best_thresh).astype(int)
        f1_opt      = float(f1_score(ys_arr, ps_opt, average="binary",
                                     pos_label=1, zero_division=0))
    except Exception:
        f1_opt = float("nan")

    return {
        "loss":       tot / n,
        "bin_acc":    float(accuracy_score(ys_arr, ps_arr)),
        "bin_f1":     float(f1_score(ys_arr, ps_arr, average="macro", zero_division=0)),
        "bin_f1_opt": f1_opt,
        "bin_auc":    auc,
    }


# ════════════════════════════════════════════════════════════════════════════════
# 9. 예측 수집 & 지표
# ════════════════════════════════════════════════════════════════════════════════

def collect_predictions(model: SpotFakePlusDGM4,
                        loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in _tqdm(loader, desc=f"[{LOG_TAG}] 예측 수집"):
            ids      = batch["input_ids"].to(device)
            attn     = batch["attention_mask"].to(device)
            img      = batch["image"].to(device)
            logits   = model(ids, attn, img)
            bin_prob = torch.softmax(logits, dim=-1)[:, 1]
            bin_pred = logits.argmax(1)
            bin_lbl  = batch["binary_label"]
            texts    = batch.get("texts",       [""] * bin_lbl.size(0))
            paths    = batch.get("image_paths", [""] * bin_lbl.size(0))
            fcs      = batch.get("fake_cls",    [""] * bin_lbl.size(0))
            for i in range(bin_lbl.size(0)):
                rows.append({
                    "text":           texts[i],
                    "image_path":     paths[i],
                    "fake_cls":       fcs[i],
                    "binary_label":   int(bin_lbl[i]),
                    "binary_pred":    int(bin_pred[i]),
                    "binary_correct": int(bin_lbl[i]) == int(bin_pred[i]),
                    "binary_prob":    round(float(bin_prob[i]), 4),
                })
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    yt    = df["binary_label"].values
    yp    = df["binary_pred"].values
    yprob = df["binary_prob"].values

    try:
        fpr, tpr, thresholds = roc_curve(yt, yprob)
        best_thresh = float(thresholds[np.argmax(tpr - fpr)])
        yp_opt      = (yprob >= best_thresh).astype(int)
        f1_opt      = float(f1_score(yt, yp_opt, average="binary",
                                     pos_label=1, zero_division=0))
    except Exception:
        f1_opt = float("nan")

    m = {
        "bin_acc":     float(accuracy_score(yt, yp)),
        "bin_f1":      float(f1_score(yt, yp, average="macro", zero_division=0)),
        "bin_f1_opt":  f1_opt,
        "bin_f1_real": float(f1_score(yt, yp, pos_label=0,
                                      average="binary", zero_division=0)),
        "bin_f1_fake": float(f1_score(yt, yp, pos_label=1,
                                      average="binary", zero_division=0)),
    }
    try:
        m["bin_auc"] = float(roc_auc_score(yt, yprob))
    except Exception:
        m["bin_auc"] = float("nan")
    return m


def eval_report_block(df: pd.DataFrame, label: str = LOG_TAG) -> Tuple[str, dict]:
    yt = df["binary_label"].values
    yp = df["binary_pred"].values
    buf = io.StringIO()
    buf.write(f"\n=== {label} — Binary Detection ===\n")
    buf.write(classification_report(yt, yp, target_names=["real", "fake"],
                                    digits=4, zero_division=0))
    m = compute_metrics(df)
    buf.write(
        f"\n[{label}]"
        f"  ACC={m['bin_acc']:.4f}"
        f"  F1={m['bin_f1']:.4f}"
        f"  F1_opt={m['bin_f1_opt']:.4f}"
        f"  AUC={m.get('bin_auc', float('nan')):.4f}\n"
    )
    return buf.getvalue(), m


# ════════════════════════════════════════════════════════════════════════════════
# 10. 체크포인트 I/O
# ════════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, model: SpotFakePlusDGM4, args, metrics=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config":           vars(args),
        "saved_at":         datetime.now().isoformat(timespec="seconds"),
    }
    if metrics:
        payload["metrics"] = metrics
    torch.save(payload, path)
    print(f"[{LOG_TAG}] 체크포인트 저장: {path}")


def load_checkpoint(model: SpotFakePlusDGM4, path: str, device) -> None:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict",
                ckpt.get("model",
                ckpt.get("state_dict", ckpt)))
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[{LOG_TAG}] load_checkpoint  missing={len(missing)}  unexpected={len(unexpected)}")


# ════════════════════════════════════════════════════════════════════════════════
# 11. 학습 루프
# ════════════════════════════════════════════════════════════════════════════════

def train_one_run(model: SpotFakePlusDGM4, tl, vl, device, args, log=None):
    if not args.no_configure_backbones:
        configure_backbones(model)

    opt   = make_optimizer(model, args)
    sched = make_scheduler(opt, args)
    early = EarlyStopping(args.early_stop_patience, args.early_stop_min_delta,
                          mode="max")
    best_auc   = -1.
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, tl, device, opt, epoch_idx=ep, scheduler=sched)
        va = run_epoch(model, vl, device)

        cur_lr = opt.param_groups[-1]["lr"]
        line = (
            f"[{LOG_TAG}] Ep {ep:02d}/{args.epochs}"
            f"  tr_loss={tr['loss']:.4f}"
            f"  va_f1_opt={va['bin_f1_opt']:.4f}"
            f"  va_auc={va['bin_auc']:.4f}"
            f"  va_acc={va['bin_acc']:.4f}"
            f"  lr={cur_lr:.2e}"
        )
        print(line)
        if log is not None:
            log.append(line)

        if not math.isnan(va["bin_auc"]) and va["bin_auc"] > best_auc:
            best_auc   = va["bin_auc"]
            best_state = copy.deepcopy(model.state_dict())

        monitor = va["bin_auc"] if not math.isnan(va["bin_auc"]) else va["bin_f1"]
        if early.step(monitor):
            msg = (f"[{LOG_TAG}] Early stopping at epoch {ep}"
                   f" (best_auc={best_auc:.4f})")
            print(msg)
            if log:
                log.append(msg)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{LOG_TAG}] best val AUC={best_auc:.4f} 가중치 복원 완료")
    return model


# ════════════════════════════════════════════════════════════════════════════════
# 12. DataLoader 헬퍼
# ════════════════════════════════════════════════════════════════════════════════

def _make_dl(df, data_root, tokenizer, args, shuffle, sampler=None):
    ds = DGM4SpotFakeDataset(df, data_root, tokenizer, args.max_len)
    kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_dgm4_spotfake,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )
    if sampler is not None:
        kw["sampler"] = sampler
        kw["shuffle"] = False
    else:
        kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


# ════════════════════════════════════════════════════════════════════════════════
# 13. eval_only 모드
# ════════════════════════════════════════════════════════════════════════════════

def run_eval_only(model: SpotFakePlusDGM4, loader, device,
                  split_name, result_dir, n_samples):
    print(f"\n[{LOG_TAG}] device={device} | split={split_name} | n={n_samples} | eval_only")
    full_df = collect_predictions(model, loader, device)
    blk, m  = eval_report_block(full_df, label=f"{LOG_TAG}_{split_name}")
    print(blk, end="")

    os.makedirs(result_dir, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rp  = os.path.join(result_dir, f"spotfake_dgm4_{split_name}_eval_only_{ts}.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(blk)
    print(f"[{LOG_TAG}] saved: {rp}")

    csv_p = rp.replace(".txt", "_predictions.csv")
    full_df.to_csv(csv_p, index=False)
    print(f"[{LOG_TAG}] predictions: {csv_p}")
    return m


# ════════════════════════════════════════════════════════════════════════════════
# 14. run_official
# ════════════════════════════════════════════════════════════════════════════════

def run_official(dfs, device, tokenizer, args, result_dir):
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (Official Split) ######\n\n")

    tr_ds = DGM4SpotFakeDataset(dfs["train"], args.data_root, tokenizer, args.max_len)

    # 기본 shuffle=True, --weighted_sampler 플래그로만 전환
    _sampler = (make_weighted_sampler(tr_ds.binary_lbls, args.sampler_alpha)
                if getattr(args, "weighted_sampler", False) else None)
    tl = _make_dl(dfs["train"], args.data_root, tokenizer, args,
                  shuffle=(_sampler is None), sampler=_sampler)
    vl = _make_dl(dfs["validation"], args.data_root, tokenizer, args, shuffle=False)
    el = _make_dl(dfs["test"],       args.data_root, tokenizer, args, shuffle=False)

    model = SpotFakePlusDGM4(args.text_model).to(device)

    log = []
    model = train_one_run(model, tl, vl, device, args, log)
    for line in log:
        rep.write(line + "\n")

    # validation 평가
    print(f"\n[{LOG_TAG}] validation 예측 수집...")
    val_df    = collect_predictions(model, vl, device)
    blk_v, mv = eval_report_block(val_df, label=f"{LOG_TAG}_val")
    print(blk_v, end="")
    rep.write(blk_v)

    # test 평가
    print(f"\n[{LOG_TAG}] test 예측 수집...")
    full_df = collect_predictions(model, el, device)
    blk, m  = eval_report_block(full_df, label=f"{LOG_TAG}_test")
    print(blk, end="")
    rep.write(blk)

    ckpt_path = os.path.join(args.checkpoint_dir,
                             f"spotfake_dgm4_official_{ts}.pt")
    save_checkpoint(ckpt_path, model, args, m)

    summ = (
        f"{LOG_TAG}: ACC={m['bin_acc']:.4f}"
        f"  F1={m['bin_f1']:.4f}"
        f"  F1_opt={m['bin_f1_opt']:.4f}"
        f"  AUC={m.get('bin_auc', float('nan')):.4f}"
    )
    print("\n" + "=" * 70 + "\n" + summ + "\n" + "=" * 70)

    path = os.path.join(result_dir, f"spotfake_dgm4_official_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rep.getvalue())

    full_df.to_csv(path.replace(".txt", "_test_predictions.csv"),
                   index=False, encoding="utf-8")
    print(f"[{LOG_TAG}] 리포트: {path}")


# ════════════════════════════════════════════════════════════════════════════════
# 15. run_kfold
# ════════════════════════════════════════════════════════════════════════════════

def run_kfold(dfs, device, tokenizer, args, result_dir):
    df_tv   = pd.concat([dfs["train"], dfs["validation"]], ignore_index=True)
    df_tv   = _ensure_dgm4_columns(df_tv)
    y_tv    = df_tv["binary_label"].values
    k       = args.kfold_splits
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rep = io.StringIO()
    all_rep.write(f"###### {LOG_TAG} (k={k} KFold) ######\n")
    el = _make_dl(dfs["test"], args.data_root, tokenizer, args, shuffle=False)

    skf    = StratifiedKFold(n_splits=k, shuffle=True,
                             random_state=args.kfold_random_state)
    fold_m = []

    for fold, (tr_idx, va_idx) in enumerate(
            skf.split(np.zeros(len(df_tv)), y_tv), 1):
        print(f"\n{'='*70}\n[Fold {fold}/{k}] {LOG_TAG}\n{'='*70}")
        df_tr_f = df_tv.iloc[tr_idx].reset_index(drop=True)
        df_va_f = df_tv.iloc[va_idx].reset_index(drop=True)

        tr_ds    = DGM4SpotFakeDataset(df_tr_f, args.data_root, tokenizer, args.max_len)
        _sampler = (make_weighted_sampler(tr_ds.binary_lbls, args.sampler_alpha)
                    if getattr(args, "weighted_sampler", False) else None)
        tl = _make_dl(df_tr_f, args.data_root, tokenizer, args,
                      shuffle=(_sampler is None), sampler=_sampler)
        vl = _make_dl(df_va_f, args.data_root, tokenizer, args, shuffle=False)

        model = SpotFakePlusDGM4(args.text_model).to(device)

        log = []
        model = train_one_run(model, tl, vl, device, args, log)

        full_df = collect_predictions(model, el, device)
        blk, m  = eval_report_block(full_df, label=f"Fold{fold}")
        print(blk, end="")

        ckpt = os.path.join(args.checkpoint_dir,
                            f"spotfake_dgm4_kfold_{ts}_fold{fold}.pt")
        save_checkpoint(ckpt, model, args, m)

        summ = (f"F1_opt={m['bin_f1_opt']:.4f}  AUC={m.get('bin_auc', float('nan')):.4f}")
        print(f"\n[Fold {fold}] {summ}")
        all_rep.write(f"\n{'#'*80}\n### FOLD {fold}\n{'#'*80}\n{blk}")
        fold_m.append(m)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _mean(k_): return float(np.mean([r[k_] for r in fold_m]))
    g = (f"{LOG_TAG}: F1_opt={_mean('bin_f1_opt'):.4f}"
         f"  AUC={_mean('bin_auc'):.4f}")
    all_rep.write(f"\n\n{'#'*80}\n### SUMMARY\n{g}\n")

    path = os.path.join(result_dir, f"spotfake_dgm4_kfold_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(all_rep.getvalue())
    print(f"\n[{LOG_TAG}] 리포트: {path}\n{g}")


# ════════════════════════════════════════════════════════════════════════════════
# 16. main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(
        description="SpotFake+ × DGM4 이진 분류 벤치마크"
    )

    # ── 데이터 & 경로 ─────────────────────────────────────────────────────────
    pa.add_argument("--data_root",      default=DEFAULT_DATA_ROOT)
    pa.add_argument("--checkpoint_dir", default="./checkpoints")
    pa.add_argument("--result_dir",     default="./results")

    # ── 평가 전용 ─────────────────────────────────────────────────────────────
    pa.add_argument("--eval_only",  action="store_true",
                    help="학습 없이 --checkpoint 로 평가만 수행")
    pa.add_argument("--checkpoint", type=str, default="",
                    help="평가 또는 학습 재개용 가중치 경로")
    pa.add_argument("--split",      type=str, default="test",
                    choices=("train", "validation", "test"),
                    help="--eval_only 시 평가할 split (기본 test)")

    # ── 학습 하이퍼 ───────────────────────────────────────────────────────────
    pa.add_argument("--batch_size",           type=int,   default=BATCH)
    pa.add_argument("--epochs",               type=int,   default=EPOCHS)
    pa.add_argument("--lr",                   type=float, default=LR)
    pa.add_argument("--weight_decay",         type=float, default=PAPER_WEIGHT_DECAY)
    pa.add_argument("--warmup_epochs",        type=int,   default=-1,
                    help="-1 이면 paper warmup 비율 자동 스케일")
    pa.add_argument("--num_workers",          type=int,   default=NUM_WORKERS)
    pa.add_argument("--seed",                 type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    pa.add_argument("--sampler_alpha",        type=float, default=SAMPLER_ALPHA)
    pa.add_argument("--limit",                type=int,   default=0,
                    help="각 split 앞 N개만 사용 (디버그)")

    # ── sampler 플래그 (기본 shuffle) ───────────────────────────────────────
    pa.add_argument("--weighted_sampler", action="store_true",
                    help="WeightedRandomSampler 사용 (기본 shuffle=True)")

    # ── 실험 모드 ─────────────────────────────────────────────────────────────
    pa.add_argument("--kfold",               action="store_true")
    pa.add_argument("--kfold_splits",        type=int, default=KFOLD_SPLITS)
    pa.add_argument("--kfold_random_state",  type=int, default=KFOLD_RANDOM_STATE)

    # ── 모델 ──────────────────────────────────────────────────────────────────
    pa.add_argument("--text_model", default=TEXT_MODEL,
                    help="XLNet 모델명 (기본 xlnet-base-cased)")
    pa.add_argument("--max_len",    type=int, default=MAX_LEN)

    # ── 기타 ──────────────────────────────────────────────────────────────────
    pa.add_argument("--no_configure_backbones", action="store_true",
                    help="백본 freeze/unfreeze 설정 건너뜀 (모든 파라미터 학습)")
    pa.add_argument("--no_progress",            action="store_true")

    args = pa.parse_args()

    if args.warmup_epochs < 0:
        args.warmup_epochs = default_paper_warmup_epochs(args.epochs)
    if args.epochs > 1 and args.warmup_epochs > args.epochs - 1:
        args.warmup_epochs = args.epochs - 1

    if args.no_progress:
        os.environ["DGM4_NO_TQDM"] = "1"

    set_seed(args.seed)
    os.makedirs(args.result_dir,     exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    mode         = "KFold" if args.kfold else "Official Split"
    sampler_note = "WeightedSampler" if getattr(args, "weighted_sampler", False) else "shuffle=True"
    print(f"\n{LOG_TAG} | device={DEVICE} | mode={mode}")
    print(
        f"  lr={args.lr:.2e}  wd={args.weight_decay}"
        f"  warmup={args.warmup_epochs}ep  epochs={args.epochs}"
        f"  batch={args.batch_size}  sampler={sampler_note}"
        f"  patience={args.early_stop_patience}"
        f"  grad_clip={GRAD_CLIP_NORM}"
    )

    dfs       = load_dgm4_splits(args.data_root, LOG_TAG)
    tokenizer = XLNetTokenizer.from_pretrained(args.text_model)

    # limit 적용
    if args.limit > 0:
        dfs = {k: v.head(args.limit).reset_index(drop=True) for k, v in dfs.items()}

    # ── 평가 전용 모드 ────────────────────────────────────────────────────────
    if args.eval_only:
        model = SpotFakePlusDGM4(args.text_model).to(DEVICE)
        if args.checkpoint:
            load_checkpoint(model, args.checkpoint, DEVICE)
        else:
            print(f"[{LOG_TAG}] ⚠️  --checkpoint 없음 → 랜덤 초기화 상태 (지표는 참고용)")

        ev_df = _ensure_dgm4_columns(dfs[args.split])
        ev_ds = DGM4SpotFakeDataset(ev_df, args.data_root, tokenizer, args.max_len)
        ev_loader = DataLoader(
            ev_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_dgm4_spotfake,
            pin_memory=torch.cuda.is_available(),
        )
        run_eval_only(model, ev_loader, DEVICE,
                      args.split, args.result_dir, len(ev_ds))
        return

    # ── 학습 모드 ─────────────────────────────────────────────────────────────
    if args.kfold:
        run_kfold(dfs, DEVICE, tokenizer, args, args.result_dir)
    else:
        run_official(dfs, DEVICE, tokenizer, args, args.result_dir)


if __name__ == "__main__":
    main()