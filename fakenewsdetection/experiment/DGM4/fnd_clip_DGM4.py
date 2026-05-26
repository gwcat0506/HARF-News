"""
FND-CLIP × DGM4 벤치마크 실험 코드
=====================================
FND-CLIP 모델 파이프라인을 DGM4 이진 분류 벤치마크에 적용.
모델: BERT(텍스트) + ResNet101(이미지) + CLIP(텍스트·이미지) → Binary (real/fake)

  AdamW weight_decay=0.02
  warmup: 50epoch 중 10warmup 비율을 --epochs 에 맞게 스케일
  EARLY_STOP_PATIENCE=4, EARLY_STOP_MIN_DELTA=1e-4, val AUC 기준 best 복원
  그래디언트 클립 없음
  LR 스케줄러: Linear Warmup + Cosine Annealing (LambdaLR)
  기본: shuffle=True (--weighted_sampler 플래그로 WeightedRandomSampler 전환)
  Youden's J 최적 임계값 F1(bin_f1_opt) 리포트
  --eval_only / --checkpoint / --split 지원

사용 방법:
  python fnd_clip_DGM4.py --data_root ./datasets/DGM4
  python fnd_clip_DGM4.py --data_root ... --eval_only --checkpoint ./checkpoints/xxx.pt
  python fnd_clip_DGM4.py --kfold
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
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPProcessor

from dgm4_paths import (
    DEFAULT_DATA_ROOT,
    FINE_LABELS,
    load_dgm4_splits,
    parse_fake_cls,
    resolve_image_path,
)

# ════════════════════════════════════════════════════════════════════════════════
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

LR_MULT_TEXT   = 0.05
LR_MULT_VISUAL = 0.50
LR_MULT_NEW    = 1.00

BERT_NAME    = "bert-base-uncased"
CLIP_NAME    = "openai/clip-vit-base-patch32"
MAX_LEN_BERT = 128
IMG_SIZE     = 224
MAX_LEN_CLIP = 77

BERT_UNFREEZE_LAYERS = (10, 11)
CLIP_UNFREEZE_KEYS   = (
    "vision_model.encoder.layers.11",
    "vision_model.post_layernorm",
    "visual_projection",
    "text_model.encoder.layers.11",
    "text_projection",
)

LOG_TAG = "FND-CLIP-DGM4"

if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def default_paper_warmup_epochs(train_epochs: int) -> int:
    """50epoch 중 10warmup 비율을 train_epochs 에 맞춤."""
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
    """
    load_dgm4_splits 반환 DataFrame 에 필요 컬럼이 없을 경우 자동 생성.
      has_image_flag : image 컬럼이 유효한 경로인지 여부
      binary_label   : orig=0, 나머지=1
      fine_{label}   : FINE_LABELS 각각에 대한 0/1
    """
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

class DGM4FNDCLIPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str,
                 bert_tokenizer, clip_processor,
                 max_len_bert: int = MAX_LEN_BERT,
                 require_image: bool = False):
        super().__init__()
        self.data_root      = os.path.abspath(data_root)
        self.bert_tokenizer = bert_tokenizer
        self.clip_processor = clip_processor
        self.max_len_bert   = max_len_bert

        df = _ensure_dgm4_columns(df)

        if require_image:
            df = df[df["has_image_flag"]].reset_index(drop=True)

        self.texts        = df["text"].tolist()
        self.image_rels   = df["image"].tolist()
        self.has_img_flag = df["has_image_flag"].tolist()
        self.binary_lbls  = df["binary_label"].tolist()
        self.fake_cls_str = df["fake_cls"].tolist()

        fine_cols      = [f"fine_{lbl}" for lbl in FINE_LABELS]
        self.fine_lbls = df[fine_cols].values.tolist()

    def __len__(self):
        return len(self.binary_lbls)

    def _load_pil(self, rel: str, flag: bool):
        if not flag or not rel:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127)), False
        full = resolve_image_path(rel, self.data_root)
        if not full or not os.path.isfile(full):
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127)), False
        try:
            return Image.open(full).convert("RGB"), True
        except Exception:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127)), False

    def __getitem__(self, idx):
        text    = self.texts[idx]
        img, ok = self._load_pil(self.image_rels[idx], self.has_img_flag[idx])

        bert_enc = self.bert_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len_bert,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        clip_enc = self.clip_processor(
            text=[text],
            images=[img],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN_CLIP,
            return_tensors="pt",
        )
        return {
            "bert_input_ids":      bert_enc["input_ids"].squeeze(0),
            "bert_attention_mask": bert_enc["attention_mask"].squeeze(0),
            "bert_token_type_ids": bert_enc["token_type_ids"].squeeze(0),
            "image_resnet":        resnet_transform(img),
            "clip_input_ids":      clip_enc["input_ids"].squeeze(0),
            "clip_attention_mask": clip_enc["attention_mask"].squeeze(0),
            "clip_pixel_values":   clip_enc["pixel_values"].squeeze(0),
            "has_image":           torch.tensor(1.0 if ok else 0.0),
            "binary_label":        torch.tensor(self.binary_lbls[idx], dtype=torch.long),
            "fine_labels":         torch.tensor(self.fine_lbls[idx], dtype=torch.float),
            "text":                text,
            "image_path":          self.image_rels[idx],
            "fake_cls":            self.fake_cls_str[idx],
        }


def collate_dgm4_fndclip(batch):
    tensor_keys = (
        "bert_input_ids", "bert_attention_mask", "bert_token_type_ids",
        "image_resnet", "clip_input_ids", "clip_attention_mask",
        "clip_pixel_values", "has_image", "binary_label", "fine_labels",
    )
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
# 5. FND-CLIP 모델
# ════════════════════════════════════════════════════════════════════════════════

class UnimodalDetection(nn.Module):
    def __init__(self, text_in=1280, image_in=1512, shared_dim=256, prime_dim=64):
        super().__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(text_in, shared_dim), nn.BatchNorm1d(shared_dim),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(shared_dim, prime_dim), nn.BatchNorm1d(prime_dim), nn.ReLU(),
        )
        self.image_uni = nn.Sequential(
            nn.Linear(image_in, shared_dim), nn.BatchNorm1d(shared_dim),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(shared_dim, prime_dim), nn.BatchNorm1d(prime_dim), nn.ReLU(),
        )

    def forward(self, text_encoding, image_encoding):
        return self.text_uni(text_encoding), self.image_uni(image_encoding)


class CrossModule(nn.Module):
    def __init__(self, corre_in=1024, corre_out_dim=64):
        super().__init__()
        self.c_specific = nn.Sequential(
            nn.Linear(corre_in, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, corre_out_dim), nn.BatchNorm1d(corre_out_dim), nn.ReLU(),
        )

    def forward(self, text, image):
        return self.c_specific(torch.cat((text, image), 1))


class FNDCLIPBinaryDGM4(nn.Module):
    """
    BERT(768) + CLIP-text(512) = 1280 차원 텍스트 인코딩
    ResNet101(1000) + CLIP-image(512) = 1512 차원 이미지 인코딩
    → Binary (real/fake) 분류
    """
    def __init__(self, feature_dim=192, h_dim=64):
        super().__init__()
        self.weights    = nn.Parameter(torch.randn(13, 1) * 0.01)
        self.senet      = nn.Sequential(nn.Linear(3, 3), nn.GELU(), nn.Linear(3, 3))
        self.sigmoid    = nn.Sigmoid()
        self.w          = nn.Parameter(torch.tensor(1.0))
        self.b          = nn.Parameter(torch.tensor(0.0))
        self.avepooling = nn.AvgPool1d(64, stride=1)
        self.maxpooling = nn.MaxPool1d(64, stride=1)

        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.resnet101 = resnet

        self.uni_repre    = UnimodalDetection(text_in=1280, image_in=1512)
        self.cross_module = CrossModule(corre_in=1024)
        self.classifier   = nn.Sequential(
            nn.Linear(feature_dim, h_dim), nn.BatchNorm1d(h_dim),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h_dim, h_dim), nn.BatchNorm1d(h_dim),
            nn.ReLU(), nn.Linear(h_dim, 1),
        )

    def forward(self, bert_hidden_states, image_resnet, text_clip, image_clip,
                has_image=None):
        B = image_resnet.shape[0]
        image_raw = self.resnet101(image_resnet)

        if has_image is not None:
            m = has_image.view(B, 1)
            image_raw  = image_raw  * m
            image_clip = image_clip * m

        ht_cls   = torch.stack(bert_hidden_states, dim=0)[:, :, 0, :]
        ht_cls   = ht_cls.view(13, B, 1, 768)
        atten    = torch.sum(ht_cls * self.weights.view(13, 1, 1, 1), dim=[1, 3])
        atten    = F.softmax(atten.view(-1), dim=0)
        text_raw = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2]).squeeze(1)

        text_enc  = torch.cat([text_raw, text_clip], 1)
        image_enc = torch.cat([image_raw, image_clip], 1)
        text_prime, image_prime = self.uni_repre(text_enc, image_enc)

        correlation = self.cross_module(text_clip, image_clip)

        if has_image is not None:
            image_prime = image_prime * m
            correlation = correlation * m

        sim = (text_clip * image_clip).sum(1) / (
            text_clip.norm(dim=1) * image_clip.norm(dim=1) + 1e-8)
        sim = sim * self.w + self.b
        if has_image is not None:
            sim = sim * has_image
        correlation = correlation * sim.unsqueeze(1)

        final_feature = torch.stack([text_prime, image_prime, correlation], 1)
        s1 = self.avepooling(final_feature).view(B, -1)
        s2 = self.maxpooling(final_feature).view(B, -1)
        s  = self.sigmoid(self.senet(s1) + self.senet(s2)).view(B, 3, 1)
        pooled = (s * final_feature).view(B, -1)
        return self.classifier(pooled).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════

def binary_focal_loss(logit, target, gamma=2.0, pos_weight=1.0):
    pw  = torch.tensor([pos_weight], device=logit.device)
    bce = F.binary_cross_entropy_with_logits(
        logit, target.float(), pos_weight=pw, reduction="none")
    prob = torch.sigmoid(logit)
    pt   = torch.where(target == 1, prob, 1 - prob)
    return ((1 - pt) ** gamma * bce).mean()


# ════════════════════════════════════════════════════════════════════════════════
# 7. 백본 설정
# ════════════════════════════════════════════════════════════════════════════════

def configure_backbones(bert_model, clip_model, fnd_module,
                        text_train_layers=BERT_UNFREEZE_LAYERS):
    for name, p in bert_model.named_parameters():
        p.requires_grad_(any(f"encoder.layer.{i}" in name
                             for i in text_train_layers))

    for name, p in clip_model.named_parameters():
        p.requires_grad_(any(k in name for k in CLIP_UNFREEZE_KEYS))

    for name, p in fnd_module.resnet101.named_parameters():
        p.requires_grad_("layer4" in name)

    for name, p in fnd_module.named_parameters():
        if "resnet101" not in name:
            p.requires_grad_(True)

    n_train = (sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
               + sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
               + sum(p.numel() for p in fnd_module.parameters() if p.requires_grad))
    n_total = (sum(p.numel() for p in bert_model.parameters())
               + sum(p.numel() for p in clip_model.parameters())
               + sum(p.numel() for p in fnd_module.parameters()))
    print(f"[{LOG_TAG}] 학습 파라미터: {n_train:,} / {n_total:,}"
          f" ({100*n_train/n_total:.1f}%)")


# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════

def make_optimizer(bert_model, clip_model, fnd_module, args):
    grp_text, grp_visual, grp_new = [], [], []

    for p in bert_model.parameters():
        if p.requires_grad:
            grp_text.append(p)
    for p in clip_model.parameters():
        if p.requires_grad:
            grp_visual.append(p)
    for name, p in fnd_module.named_parameters():
        if not p.requires_grad:
            continue
        (grp_visual if "resnet101" in name else grp_new).append(p)

    wd = getattr(args, "weight_decay", PAPER_WEIGHT_DECAY)
    groups = [
        {"params": grp_text,   "lr": args.lr * LR_MULT_TEXT,   "weight_decay": wd},
        {"params": grp_visual, "lr": args.lr * LR_MULT_VISUAL, "weight_decay": wd},
        {"params": grp_new,    "lr": args.lr * LR_MULT_NEW,    "weight_decay": wd},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]])


def make_scheduler(optimizer, args):
    warmup = args.warmup_epochs
    total  = args.epochs

    def lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(max(warmup, 1))
        progress = float(ep - warmup) / float(max(total - warmup, 1))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ════════════════════════════════════════════════════════════════════════════════
# 9. forward 헬퍼
# ════════════════════════════════════════════════════════════════════════════════

def _fwd(bert_model, clip_model, fnd_module, batch, device):
    bert_ids  = batch["bert_input_ids"].to(device)
    bert_attn = batch["bert_attention_mask"].to(device)
    bert_ttid = batch["bert_token_type_ids"].to(device)
    img_rsn   = batch["image_resnet"].to(device)
    clip_ids  = batch["clip_input_ids"].to(device)
    clip_attn = batch["clip_attention_mask"].to(device)
    clip_pix  = batch["clip_pixel_values"].to(device)
    has_img   = batch["has_image"].to(device)

    bert_out   = bert_model(input_ids=bert_ids, attention_mask=bert_attn,
                            token_type_ids=bert_ttid)
    text_clip  = clip_model.get_text_features(input_ids=clip_ids,
                                               attention_mask=clip_attn)
    image_clip = clip_model.get_image_features(pixel_values=clip_pix)

    logit = fnd_module(bert_out.hidden_states, img_rsn, text_clip, image_clip, has_img)
    return torch.nan_to_num(logit, nan=0., posinf=20., neginf=-20.)


# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════

def run_epoch(bert_model, clip_model, fnd_module, loader, device,
              optimizer, train, epoch_idx=None, args=None, scheduler=None):
    if train:
        bert_model.train(); clip_model.train(); fnd_module.train()
    else:
        bert_model.eval(); clip_model.eval(); fnd_module.eval()

    tot, ys, ps, probs, n = 0., [], [], [], 0
    ctx  = torch.enable_grad() if train else torch.no_grad()
    desc = f"[{LOG_TAG}] Epoch {epoch_idx}" if (train and epoch_idx) else None
    it   = _tqdm(loader, desc=desc) if (train and desc) else loader

    with ctx:
        for batch in it:
            y_bin = batch["binary_label"].to(device)
            logit = _fwd(bert_model, clip_model, fnd_module, batch, device)
            loss  = binary_focal_loss(logit, y_bin)

            if train and optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            prob = torch.sigmoid(logit).detach()
            pred = (prob > 0.5).long()
            tot  += loss.item() * y_bin.size(0)
            ys.extend(y_bin.cpu().tolist())
            ps.extend(pred.cpu().tolist())
            probs.extend(prob.cpu().tolist())
            n += y_bin.size(0)

    if train and scheduler is not None:
        scheduler.step()

    n = max(n, 1)
    ys_arr    = np.array(ys)
    probs_arr = np.array(probs)
    ps_arr    = np.array(ps)

    try:
        auc = float(roc_auc_score(ys_arr, probs_arr))
    except Exception:
        auc = float("nan")

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
        "bin_f1_opt": f1_opt,   # ★
        "bin_auc":    auc,
    }


# ════════════════════════════════════════════════════════════════════════════════
# 11. 예측 수집 & 지표  ★ bin_f1_opt 추가
# ════════════════════════════════════════════════════════════════════════════════

def collect_predictions(bert_model, clip_model, fnd_module,
                        loader, device) -> pd.DataFrame:
    bert_model.eval(); clip_model.eval(); fnd_module.eval()
    rows = []
    with torch.no_grad():
        for batch in _tqdm(loader, desc=f"[{LOG_TAG}] 예측 수집"):
            logit    = _fwd(bert_model, clip_model, fnd_module, batch, device)
            bin_prob = torch.sigmoid(logit)
            bin_pred = (bin_prob > 0.5).long()
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
        "bin_f1_opt":  f1_opt,   # ★
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
# ════════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, bert_model, clip_model, fnd_module, args, metrics=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "bert_state_dict": bert_model.state_dict(),
        "clip_state_dict": clip_model.state_dict(),
        "fnd_state_dict":  fnd_module.state_dict(),
        "config":          vars(args),
        "saved_at":        datetime.now().isoformat(timespec="seconds"),
    }
    if metrics:
        payload["metrics"] = metrics
    torch.save(payload, path)
    print(f"[{LOG_TAG}] 체크포인트 저장: {path}")


def load_checkpoint(bert_model, clip_model, fnd_module, path: str, device) -> None:
    """★ 세 모델을 한 번에 로드."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict):
        print(f"[{LOG_TAG}] ⚠️  체크포인트 형식 불명: {type(ckpt)}")
        return

    def _load(model, key, name):
        if key in ckpt:
            mis, unexp = model.load_state_dict(ckpt[key], strict=False)
            print(f"[{LOG_TAG}] {name}  missing={len(mis)}  unexpected={len(unexp)}")
        else:
            print(f"[{LOG_TAG}] ⚠️  '{key}' 키 없음 — {name} 로드 생략")

    _load(bert_model, "bert_state_dict", "BERT")
    _load(clip_model, "clip_state_dict", "CLIP")
    _load(fnd_module, "fnd_state_dict",  "FND")


# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════

def train_one_run(bert_model, clip_model, fnd_module, tl, vl, device, args, log=None):
    if not args.no_configure_backbones:
        configure_backbones(bert_model, clip_model, fnd_module)

    opt   = make_optimizer(bert_model, clip_model, fnd_module, args)
    sched = make_scheduler(opt, args)
    early = EarlyStopping(args.early_stop_patience, args.early_stop_min_delta,
                          mode="max")
    best_auc = -1.
    best_bert = best_clip = best_fnd = None

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(bert_model, clip_model, fnd_module, tl, device,
                       opt, True, epoch_idx=ep, args=args, scheduler=sched)
        va = run_epoch(bert_model, clip_model, fnd_module, vl, device,
                       None, False, args=args)

        cur_lr = opt.param_groups[-1]["lr"]
        line = (
            f"[{LOG_TAG}] Ep {ep:02d}/{args.epochs}"
            f"  tr_loss={tr['loss']:.4f}"
            f"  va_f1_opt={va['bin_f1_opt']:.4f}"   # ★
            f"  va_auc={va['bin_auc']:.4f}"
            f"  va_acc={va['bin_acc']:.4f}"
            f"  lr={cur_lr:.2e}"
        )
        print(line)
        if log is not None:
            log.append(line)

        if not math.isnan(va["bin_auc"]) and va["bin_auc"] > best_auc:
            best_auc  = va["bin_auc"]
            best_bert = copy.deepcopy(bert_model.state_dict())
            best_clip = copy.deepcopy(clip_model.state_dict())
            best_fnd  = copy.deepcopy(fnd_module.state_dict())

        monitor = va["bin_auc"] if not math.isnan(va["bin_auc"]) else va["bin_f1"]
        if early.step(monitor):
            msg = (f"[{LOG_TAG}] Early stopping at epoch {ep}"
                   f" (best_auc={best_auc:.4f})")
            print(msg)
            if log:
                log.append(msg)
            break

    if best_bert is not None:
        bert_model.load_state_dict(best_bert)
        clip_model.load_state_dict(best_clip)
        fnd_module.load_state_dict(best_fnd)
        print(f"[{LOG_TAG}] best val AUC={best_auc:.4f} 가중치 복원 완료")

    return bert_model, clip_model, fnd_module


# ════════════════════════════════════════════════════════════════════════════════
# 14. DataLoader 헬퍼
# ════════════════════════════════════════════════════════════════════════════════

def _make_dl(df, data_root, bert_tok, clip_proc, args, shuffle, sampler=None):
    ds = DGM4FNDCLIPDataset(df, data_root, bert_tok, clip_proc, args.max_len_bert)
    kw = dict(batch_size=args.batch_size,
              num_workers=args.num_workers,
              collate_fn=collate_dgm4_fndclip,
              pin_memory=torch.cuda.is_available(),
              drop_last=shuffle)
    if sampler is not None:
        kw["sampler"] = sampler
        kw["shuffle"] = False
    else:
        kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════

def run_eval_only(bert_model, clip_model, fnd_module,
                  loader, device, split_name, result_dir, n_samples):
    print(f"\n[{LOG_TAG}] device={device} | split={split_name} | n={n_samples} | eval_only")
    full_df = collect_predictions(bert_model, clip_model, fnd_module, loader, device)
    blk, m  = eval_report_block(full_df, label=f"{LOG_TAG}_{split_name}")
    print(blk, end="")

    os.makedirs(result_dir, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rp  = os.path.join(result_dir, f"fndclip_dgm4_{split_name}_eval_only_{ts}.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(blk)
    print(f"[{LOG_TAG}] saved: {rp}")

    csv_p = rp.replace(".txt", "_predictions.csv")
    full_df.to_csv(csv_p, index=False)
    print(f"[{LOG_TAG}] predictions: {csv_p}")
    return m


# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════

def run_official(dfs, device, bert_tok, clip_proc, args, result_dir):
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (Official Split) ######\n\n")

    tr_ds = DGM4FNDCLIPDataset(dfs["train"], args.data_root,
                                bert_tok, clip_proc, args.max_len_bert)

    _sampler = (make_weighted_sampler(tr_ds.binary_lbls, args.sampler_alpha)
                if getattr(args, "weighted_sampler", False) else None)
    tl = _make_dl(dfs["train"], args.data_root, bert_tok, clip_proc, args,
                  shuffle=(_sampler is None), sampler=_sampler)

    vl = _make_dl(dfs["validation"], args.data_root, bert_tok, clip_proc,
                  args, shuffle=False)
    el = _make_dl(dfs["test"], args.data_root, bert_tok, clip_proc,
                  args, shuffle=False)

    bert_model = BertModel.from_pretrained(
        args.bert_name, output_hidden_states=True).to(device)
    clip_model = CLIPModel.from_pretrained(args.clip_name).to(device)
    fnd_module = FNDCLIPBinaryDGM4().to(device)

    log = []
    bert_model, clip_model, fnd_module = train_one_run(
        bert_model, clip_model, fnd_module, tl, vl, device, args, log)
    for line in log:
        rep.write(line + "\n")

    print(f"\n[{LOG_TAG}] validation 예측 수집...")
    val_df   = collect_predictions(bert_model, clip_model, fnd_module, vl, device)
    blk_v, mv = eval_report_block(val_df, label=f"{LOG_TAG}_val")
    print(blk_v, end="")
    rep.write(blk_v)

    # test 평가
    print(f"\n[{LOG_TAG}] test 예측 수집...")
    full_df  = collect_predictions(bert_model, clip_model, fnd_module, el, device)
    blk, m   = eval_report_block(full_df, label=f"{LOG_TAG}_test")
    print(blk, end="")
    rep.write(blk)

    ckpt_path = os.path.join(args.checkpoint_dir,
                             f"fndclip_dgm4_official_{ts}.pt")
    save_checkpoint(ckpt_path, bert_model, clip_model, fnd_module, args, m)

    summ = (
        f"{LOG_TAG}: ACC={m['bin_acc']:.4f}"
        f"  F1={m['bin_f1']:.4f}"
        f"  F1_opt={m['bin_f1_opt']:.4f}"
        f"  AUC={m.get('bin_auc', float('nan')):.4f}"
    )
    print("\n" + "=" * 70 + "\n" + summ + "\n" + "=" * 70)

    path = os.path.join(result_dir, f"fndclip_dgm4_official_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rep.getvalue())

    full_df.to_csv(path.replace(".txt", "_test_predictions.csv"),
                   index=False, encoding="utf-8")
    print(f"[{LOG_TAG}] 리포트: {path}")


# ════════════════════════════════════════════════════════════════════════════════
# 17. run_kfold
# ════════════════════════════════════════════════════════════════════════════════

def run_kfold(dfs, device, bert_tok, clip_proc, args, result_dir):
    df_tv   = pd.concat([dfs["train"], dfs["validation"]], ignore_index=True)
    df_tv   = _ensure_dgm4_columns(df_tv)
    y_tv    = df_tv["binary_label"].values
    k       = args.kfold_splits
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rep = io.StringIO()
    all_rep.write(f"###### {LOG_TAG} (k={k} KFold) ######\n")
    el = _make_dl(dfs["test"], args.data_root, bert_tok, clip_proc, args, shuffle=False)

    skf    = StratifiedKFold(n_splits=k, shuffle=True,
                             random_state=args.kfold_random_state)
    fold_m = []

    for fold, (tr_idx, va_idx) in enumerate(
            skf.split(np.zeros(len(df_tv)), y_tv), 1):
        print(f"\n{'='*70}\n[Fold {fold}/{k}] {LOG_TAG}\n{'='*70}")
        df_tr_f = df_tv.iloc[tr_idx].reset_index(drop=True)
        df_va_f = df_tv.iloc[va_idx].reset_index(drop=True)

        tr_ds = DGM4FNDCLIPDataset(df_tr_f, args.data_root,
                                    bert_tok, clip_proc, args.max_len_bert)
        _sampler = (make_weighted_sampler(tr_ds.binary_lbls, args.sampler_alpha)
                    if getattr(args, "weighted_sampler", False) else None)
        tl = _make_dl(df_tr_f, args.data_root, bert_tok, clip_proc, args,
                      shuffle=(_sampler is None), sampler=_sampler)
        vl = _make_dl(df_va_f, args.data_root, bert_tok, clip_proc,
                      args, shuffle=False)

        bert_model = BertModel.from_pretrained(
            args.bert_name, output_hidden_states=True).to(device)
        clip_model = CLIPModel.from_pretrained(args.clip_name).to(device)
        fnd_module = FNDCLIPBinaryDGM4().to(device)

        log = []
        bert_model, clip_model, fnd_module = train_one_run(
            bert_model, clip_model, fnd_module, tl, vl, device, args, log)

        full_df = collect_predictions(bert_model, clip_model, fnd_module,
                                      el, device)
        blk, m  = eval_report_block(full_df, label=f"Fold{fold}")
        print(blk, end="")

        ckpt = os.path.join(args.checkpoint_dir,
                            f"fndclip_dgm4_kfold_{ts}_fold{fold}.pt")
        save_checkpoint(ckpt, bert_model, clip_model, fnd_module, args, m)

        summ = (f"F1_opt={m['bin_f1_opt']:.4f}  AUC={m.get('bin_auc', float('nan')):.4f}")
        print(f"\n[Fold {fold}] {summ}")
        all_rep.write(f"\n{'#'*80}\n### FOLD {fold}\n{'#'*80}\n{blk}")
        fold_m.append(m)

        del bert_model, clip_model, fnd_module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _mean(k_): return float(np.mean([r[k_] for r in fold_m]))
    g = (f"{LOG_TAG}: F1_opt={_mean('bin_f1_opt'):.4f}"
         f"  AUC={_mean('bin_auc'):.4f}")
    all_rep.write(f"\n\n{'#'*80}\n### SUMMARY\n{g}\n")

    path = os.path.join(result_dir, f"fndclip_dgm4_kfold_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(all_rep.getvalue())
    print(f"\n[{LOG_TAG}] 리포트: {path}\n{g}")


# ════════════════════════════════════════════════════════════════════════════════
# 18. main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(
        description="FND-CLIP × DGM4 benchmark"
    )

    # ── 데이터 & 경로 ─────────────────────────────────────────────────────────
    pa.add_argument("--data_root",      default=DEFAULT_DATA_ROOT)
    pa.add_argument("--checkpoint_dir", default="./checkpoints")
    pa.add_argument("--result_dir",     default="./results")

    pa.add_argument("--eval_only",  action="store_true",
                    help="학습 없이 --checkpoint 로 평가만 수행")
    pa.add_argument("--checkpoint", type=str, default="",
                    help="평가 또는 학습 재개용 가중치 경로")
    pa.add_argument("--split",      type=str, default="test",
                    choices=("train", "validation", "test"),
                    help="--eval_only 시 평가할 split (기본 test)")

    pa.add_argument("--batch_size",           type=int,   default=BATCH)
    pa.add_argument("--epochs",               type=int,   default=EPOCHS)
    pa.add_argument("--lr",                   type=float, default=LR)
    pa.add_argument("--weight_decay",         type=float, default=PAPER_WEIGHT_DECAY)
    pa.add_argument("--warmup_epochs",        type=int,   default=-1,
                    help="-1 이면")
    pa.add_argument("--num_workers",          type=int,   default=NUM_WORKERS)
    pa.add_argument("--seed",                 type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    pa.add_argument("--sampler_alpha",        type=float, default=SAMPLER_ALPHA)
    pa.add_argument("--limit",                type=int,   default=0,
                    help="각 split 앞 N개만 사용 (디버그)")

    pa.add_argument("--weighted_sampler", action="store_true",
                    help="WeightedRandomSampler 사용 (기본 shuffle=True,")

    # ── 실험 모드 ─────────────────────────────────────────────────────────────
    pa.add_argument("--kfold",               action="store_true")
    pa.add_argument("--kfold_splits",        type=int, default=KFOLD_SPLITS)
    pa.add_argument("--kfold_random_state",  type=int, default=KFOLD_RANDOM_STATE)

    # ── 모델 ──────────────────────────────────────────────────────────────────
    pa.add_argument("--bert_name",    default=BERT_NAME)
    pa.add_argument("--clip_name",    default=CLIP_NAME)
    pa.add_argument("--max_len_bert", type=int, default=MAX_LEN_BERT)

    # ── 기타 ──────────────────────────────────────────────────────────────────
    pa.add_argument("--no_configure_backbones", action="store_true")
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

    mode = "KFold" if args.kfold else "Official Split"
    sampler_note = "WeightedSampler" if getattr(args, "weighted_sampler", False) else "shuffle=True"
    print(f"\n{LOG_TAG} | device={DEVICE} | mode={mode}")
    print(
        f"  lr={args.lr:.2e}  wd={args.weight_decay}"
        f"  warmup={args.warmup_epochs}ep  epochs={args.epochs}"
        f"  batch={args.batch_size}  sampler={sampler_note}"
        f"  patience={args.early_stop_patience}"
    )

    dfs       = load_dgm4_splits(args.data_root, LOG_TAG)
    bert_tok  = BertTokenizer.from_pretrained(args.bert_name)
    clip_proc = CLIPProcessor.from_pretrained(args.clip_name)

    # limit 적용
    if args.limit > 0:
        dfs = {k: v.head(args.limit).reset_index(drop=True) for k, v in dfs.items()}

    if args.eval_only:
        bert_model = BertModel.from_pretrained(
            args.bert_name, output_hidden_states=True).to(DEVICE)
        clip_model = CLIPModel.from_pretrained(args.clip_name).to(DEVICE)
        fnd_module = FNDCLIPBinaryDGM4().to(DEVICE)

        if args.checkpoint:
            load_checkpoint(bert_model, clip_model, fnd_module,
                            args.checkpoint, DEVICE)
        else:
            print(f"[{LOG_TAG}] ⚠️  --checkpoint 없음 → 랜덤 초기화 상태 (지표는 참고용)")

        ev_df = _ensure_dgm4_columns(dfs[args.split])
        ev_ds = DGM4FNDCLIPDataset(ev_df, args.data_root,
                                    bert_tok, clip_proc, args.max_len_bert)
        ev_loader = DataLoader(
            ev_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_dgm4_fndclip,
            pin_memory=torch.cuda.is_available(),
        )
        run_eval_only(bert_model, clip_model, fnd_module,
                      ev_loader, DEVICE, args.split, args.result_dir, len(ev_ds))
        return

    # ── 학습 모드 ─────────────────────────────────────────────────────────────
    if args.kfold:
        run_kfold(dfs, DEVICE, bert_tok, clip_proc, args, args.result_dir)
    else:
        run_official(dfs, DEVICE, bert_tok, clip_proc, args, args.result_dir)


if __name__ == "__main__":
    main()