"""
MiRAGe × DGM4 벤치마크 실험 코드
=====================================

[의존 패키지]
  pip install open_clip_torch

[사용법]
  # Official split 학습
  python mirage_dgm4.py --data_root /path/to/DGM4

  # KFold
  python mirage_dgm4.py --data_root /path/to/DGM4 --kfold

  # 평가만
  python mirage_dgm4.py --data_root /path/to/DGM4 --eval_only --checkpoint ckpt.pt
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
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    print(
        "\n[오류] open_clip_torch 패키지가 없습니다.\n"
        "  pip install open_clip_torch\n"
        "명령어로 설치 후 재실행해주세요."
    )
    sys.exit(1)

from dgm4_paths import DEFAULT_DATA_ROOT, FINE_LABELS, load_dgm4_splits, resolve_image_path

# ════════════════════════════════════════════════════════════════════════════════
# 1. 상수
# ════════════════════════════════════════════════════════════════════════════════
BATCH                = 16
EPOCHS               = 10
LR                   = 1e-3
NUM_WORKERS          = 4
EARLY_STOP_PATIENCE  = 4
EARLY_STOP_MIN_DELTA = 1e-4
KFOLD_SPLITS         = 5
KFOLD_RANDOM_STATE   = 42
SEED                 = 42
SAMPLER_ALPHA        = 0.5
DROPOUT              = 0.3

LOG_TAG = "MiRAGe-DGM4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════════════════════════════════
# 2. Simplified-CBM 개념 프롬프트 (이미지 측)
#    여기서는 AI 생성/딥페이크 탐지에 적합한 25개 시각 개념을 사용.
# ════════════════════════════════════════════════════════════════════════════════
VISUAL_CONCEPTS = [
    "a real news photograph",
    "an AI-generated image",
    "a photorealistic synthetic image",
    "a person or human face",
    "a face that looks artificially generated",
    "a crowd of people",
    "a political event or rally",
    "a natural landscape or scenery",
    "an urban environment or city",
    "a building or architecture",
    "a vehicle or transportation",
    "an animal or wildlife",
    "a sports event",
    "a military scene or weapon",
    "text or signage visible in image",
    "professional journalism photo",
    "digitally manipulated or edited photo",
    "high resolution detailed image",
    "blurry or low quality image",
    "unusual lighting or unrealistic colors",
    "perfectly symmetrical or unnatural composition",
    "face swap or face manipulation",
    "deepfake video frame",
    "indoor scene",
    "outdoor scene",
]
NUM_CONCEPTS  = len(VISUAL_CONCEPTS)   # 25
NUM_TBM_FEATS = 15

# ════════════════════════════════════════════════════════════════════════════════
# 3. Simplified-TBM 언어 피처
# ════════════════════════════════════════════════════════════════════════════════
def compute_tbm_features(text: str) -> np.ndarray:
    """15개 언어통계 피처 반환 (Simplified TBM)."""
    text  = str(text)
    words = text.split()
    chars = list(text)
    word_count   = len(words)
    char_count   = len(text)
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    num_digits   = sum(c.isdigit() for c in chars)
    num_upper    = sum(c.isupper() for c in chars)
    num_punct    = sum(not c.isalnum() and not c.isspace() for c in chars)
    digit_ratio  = num_digits / max(char_count, 1)
    upper_ratio  = num_upper  / max(char_count, 1)
    punct_ratio  = num_punct  / max(char_count, 1)
    unique_words = len(set(w.lower() for w in words))
    ttr          = unique_words / max(word_count, 1)
    has_quote    = float('"' in text or "'" in text)
    has_number   = float(any(c.isdigit() for c in text))
    num_sents    = max(text.count('.') + text.count('!') + text.count('?'), 1)
    avg_sent_len = word_count / num_sents
    return np.array([
        word_count, char_count, avg_word_len,
        digit_ratio, upper_ratio, punct_ratio,
        ttr, has_quote, has_number,
        num_sents, avg_sent_len,
        num_digits, num_upper, num_punct, unique_words,
    ], dtype=np.float32)

# ════════════════════════════════════════════════════════════════════════════════
# 4. 유틸
# ════════════════════════════════════════════════════════════════════════════════
def set_seed(s: int = SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed()

def _tqdm(it, **kw):
    if os.environ.get("DGM4_NO_TQDM", "").lower() in ("1", "true", "yes"):
        return it
    kw.setdefault("file", sys.stderr); kw.setdefault("dynamic_ncols", True)
    try:
        sys.stdout.flush(); return tqdm(it, **kw)
    except Exception:
        return it

def _ensure_dgm4_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "has_image_flag" not in df.columns:
        df["has_image_flag"] = df["image"].apply(
            lambda p: isinstance(p, str) and len(p) > 0)
    if "binary_label" not in df.columns:
        df["binary_label"] = (df["fake_cls"] != "orig").astype(int)
    for lbl in FINE_LABELS:
        col = f"fine_{lbl}"
        if col not in df.columns:
            df[col] = (df["fake_cls"] == lbl).astype(float)
    return df

class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE,
                 min_delta=EARLY_STOP_MIN_DELTA, mode="max"):
        self.patience = patience; self.min_delta = min_delta
        self.mode = mode; self.best = None; self.counter = 0
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
# 5. CLIP 로드 (전역, 한 번만)
# ════════════════════════════════════════════════════════════════════════════════
print(f"[{LOG_TAG}] CLIP ViT-B-32 로드 중 ...")
_clip_model, _clip_preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai")
_clip_tokenizer    = open_clip.get_tokenizer("ViT-B-32")
_clip_model        = _clip_model.to(DEVICE).eval()

with torch.no_grad():
    _concept_tokens     = _clip_tokenizer(VISUAL_CONCEPTS).to(DEVICE)
    _concept_text_feats = _clip_model.encode_text(_concept_tokens)
    _concept_text_feats = F.normalize(_concept_text_feats, dim=-1)   # (25, 512)
print(f"[{LOG_TAG}] CLIP 준비 완료 | CBM concepts={NUM_CONCEPTS}, TBM feats={NUM_TBM_FEATS}")

# ════════════════════════════════════════════════════════════════════════════════
# 6. Dataset
# ════════════════════════════════════════════════════════════════════════════════
class MiRAGeDGM4Dataset(Dataset):
    """
    DGM4 샘플을 CLIP 전처리 이미지 + 원본 텍스트 + TBM 피처로 반환.
    CLIP 인코딩(512d)은 train/eval loop 에서 배치 단위로 수행 (on-the-fly).
    """
    def __init__(self, df: pd.DataFrame, data_root: str):
        super().__init__()
        self.data_root    = os.path.abspath(data_root)
        df                = _ensure_dgm4_columns(df)
        self.texts        = df["text"].tolist()
        self.image_rels   = df["image"].tolist()
        self.has_img_flag = df["has_image_flag"].tolist()
        self.binary_lbls  = df["binary_label"].tolist()
        self.fake_cls_str = df["fake_cls"].tolist()

        # TBM 피처 사전 계산 후 z-score 정규화
        print(f"  [{LOG_TAG}] TBM 피처 계산 중 ({len(df)}건) ...", end="", flush=True)
        feats = np.stack([compute_tbm_features(t) for t in self.texts]).astype(np.float32)
        m = feats.mean(axis=0); s = feats.std(axis=0) + 1e-8
        self.tbm_feats = (feats - m) / s
        print(" done.")

    def __len__(self):
        return len(self.binary_lbls)

    def _load_clip_image(self, rel: str, flag: bool) -> torch.Tensor:
        if not flag or not rel:
            return torch.zeros(3, 224, 224)
        full = resolve_image_path(rel, self.data_root)
        if not full or not os.path.isfile(full):
            return torch.zeros(3, 224, 224)
        try:
            return _clip_preprocess(Image.open(full).convert("RGB"))
        except Exception:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        return {
            "clip_image":   self._load_clip_image(
                                self.image_rels[idx], self.has_img_flag[idx]),
            "text":         self.texts[idx],
            "tbm_feats":    torch.tensor(self.tbm_feats[idx], dtype=torch.float32),
            "binary_label": torch.tensor(self.binary_lbls[idx], dtype=torch.long),
            "fake_cls":     self.fake_cls_str[idx],
            "image_path":   self.image_rels[idx],
        }

def collate_mirage_dgm4(batch):
    tensor_keys = ("clip_image", "tbm_feats", "binary_label")
    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out["texts"]       = [b["text"]       for b in batch]
    out["fake_cls"]    = [b["fake_cls"]   for b in batch]
    out["image_paths"] = [b["image_path"] for b in batch]
    return out

def make_weighted_sampler(binary_labels: List[int], alpha: float = SAMPLER_ALPHA):
    cnt = pd.Series(binary_labels).value_counts().to_dict()
    w   = [(1.0 / cnt[l]) ** alpha for l in binary_labels]
    return WeightedRandomSampler(w, len(w), replacement=True)

# ════════════════════════════════════════════════════════════════════════════════
# 7. 모델 (binary = 2-class)
# ════════════════════════════════════════════════════════════════════════════════
class MiRAGeImgDetector(nn.Module):
    """
    MiRAGe-Img (binary)
    Linear 브랜치(CLIP 이미지 임베딩 512d) + CBM 브랜치(개념 점수 25d)
    → 앙상블 퓨전 헤드 → 2-class logits
    """
    def __init__(self, img_dim=512, concept_dim=NUM_CONCEPTS,
                 num_classes=2, dropout=DROPOUT):
        super().__init__()
        self.linear_branch = nn.Sequential(
            nn.Linear(img_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))
        self.cbm_branch = nn.Sequential(
            nn.Linear(concept_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes))
        self.ensemble_head = nn.Sequential(
            nn.Linear(num_classes * 2, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, num_classes))

    def forward(self, img_feat, concept_scores):
        logits_lin = self.linear_branch(img_feat)
        logits_cbm = self.cbm_branch(concept_scores)
        logits_out = self.ensemble_head(torch.cat([logits_lin, logits_cbm], dim=1))
        return logits_out, logits_lin, logits_cbm


class MiRAGeTxtDetector(nn.Module):
    """
    MiRAGe-Txt (binary)
    Linear 브랜치(CLIP 텍스트 임베딩 512d) + TBM 브랜치(언어통계 15d)
    → 앙상블 퓨전 헤드 → 2-class logits
    """
    def __init__(self, txt_dim=512, tbm_dim=NUM_TBM_FEATS,
                 num_classes=2, dropout=DROPOUT):
        super().__init__()
        self.linear_branch = nn.Sequential(
            nn.Linear(txt_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))
        self.tbm_branch = nn.Sequential(
            nn.Linear(tbm_dim, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, num_classes))
        self.ensemble_head = nn.Sequential(
            nn.Linear(num_classes * 2, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, num_classes))

    def forward(self, txt_feat, tbm_feats):
        logits_lin = self.linear_branch(txt_feat)
        logits_tbm = self.tbm_branch(tbm_feats)
        logits_out = self.ensemble_head(torch.cat([logits_lin, logits_tbm], dim=1))
        return logits_out, logits_lin, logits_tbm

# ════════════════════════════════════════════════════════════════════════════════
# 8. CLIP 인코딩 헬퍼
# ════════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def _encode_batch(clip_images: torch.Tensor,
                  texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """한 배치에 대해 CLIP 인코딩 + CBM 개념 점수 계산."""
    img_feat = F.normalize(_clip_model.encode_image(clip_images), dim=-1)
    tokens   = _clip_tokenizer(texts).to(clip_images.device)
    txt_feat = F.normalize(_clip_model.encode_text(tokens), dim=-1)
    concept_scores = img_feat @ _concept_text_feats.T   # (B, 25)
    return img_feat, txt_feat, concept_scores

# ════════════════════════════════════════════════════════════════════════════════
# 9. 학습 epoch (Img + Txt 동시 1 pass, CLIP 인코딩 공유)
# ════════════════════════════════════════════════════════════════════════════════
def train_one_epoch(img_model, txt_model, loader,
                    opt_img, opt_txt, device, epoch_idx):
    img_model.train(); txt_model.train(); _clip_model.eval()
    loss_fn = nn.CrossEntropyLoss()
    ALPHA   = 0.3   # 보조 브랜치 손실 가중치

    tot_img = tot_txt = n_img = n_txt = 0.0
    it = _tqdm(loader, desc=f"[{LOG_TAG}] Epoch {epoch_idx}")

    for batch in it:
        clip_imgs = batch["clip_image"].to(device)
        tbm_f     = batch["tbm_feats"].to(device)
        labels    = batch["binary_label"].to(device)
        texts     = batch["texts"]
        B         = labels.size(0)

        # CLIP 인코딩 (공유, no_grad)
        img_feat, txt_feat, concept_scores = _encode_batch(clip_imgs, texts)

        # ── MiRAGe-Img 업데이트 ──────────────────────────────────────────────
        logits_img, logits_lin_img, logits_cbm = img_model(
            img_feat.detach(), concept_scores.detach())
        loss_img = (loss_fn(logits_img, labels)
                    + ALPHA * loss_fn(logits_lin_img, labels)
                    + ALPHA * loss_fn(logits_cbm, labels))
        opt_img.zero_grad(); loss_img.backward(); opt_img.step()
        tot_img += loss_img.item() * B; n_img += B

        # ── MiRAGe-Txt 업데이트 ──────────────────────────────────────────────
        logits_txt, logits_lin_txt, logits_tbm = txt_model(
            txt_feat.detach(), tbm_f)
        loss_txt = (loss_fn(logits_txt, labels)
                    + ALPHA * loss_fn(logits_lin_txt, labels)
                    + ALPHA * loss_fn(logits_tbm, labels))
        opt_txt.zero_grad(); loss_txt.backward(); opt_txt.step()
        tot_txt += loss_txt.item() * B; n_txt += B

    return tot_img / max(n_img, 1), tot_txt / max(n_txt, 1)

# ════════════════════════════════════════════════════════════════════════════════
# 10. 검증 & 메트릭
# ════════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_epoch(img_model, txt_model, loader, device, fusion="avg"):
    img_model.eval(); txt_model.eval(); _clip_model.eval()
    ys, probs_all = [], []

    for batch in loader:
        clip_imgs = batch["clip_image"].to(device)
        tbm_f     = batch["tbm_feats"].to(device)
        labels    = batch["binary_label"]
        img_feat, txt_feat, concept_scores = _encode_batch(clip_imgs, batch["texts"])
        logits_img, _, _ = img_model(img_feat, concept_scores)
        logits_txt, _, _ = txt_model(txt_feat, tbm_f)

        if fusion == "avg":
            prob = (F.softmax(logits_img, 1) + F.softmax(logits_txt, 1)) / 2.0
        elif fusion == "img":
            prob = F.softmax(logits_img, 1)
        else:
            prob = F.softmax(logits_txt, 1)

        ys.extend(labels.tolist())
        probs_all.extend(prob[:, 1].cpu().tolist())

    ys = np.array(ys); probs = np.array(probs_all)
    preds = (probs >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(ys, probs))
    except Exception:
        auc = float("nan")
    try:
        fpr, tpr, thr = roc_curve(ys, probs)
        p_opt  = (probs >= float(thr[np.argmax(tpr - fpr)])).astype(int)
        f1_opt = float(f1_score(ys, p_opt, average="binary",
                                pos_label=1, zero_division=0))
    except Exception:
        f1_opt = float("nan")
    return {
        "bin_acc":    float(accuracy_score(ys, preds)),
        "bin_f1":     float(f1_score(ys, preds, average="macro", zero_division=0)),
        "bin_f1_opt": f1_opt,
        "bin_auc":    auc,
    }

@torch.no_grad()
def collect_predictions(img_model, txt_model, loader, device, fusion="avg"):
    img_model.eval(); txt_model.eval(); _clip_model.eval()
    rows = []
    for batch in _tqdm(loader, desc=f"[{LOG_TAG}] 예측 수집 [{fusion}]"):
        clip_imgs = batch["clip_image"].to(device)
        tbm_f     = batch["tbm_feats"].to(device)
        labels    = batch["binary_label"]
        img_feat, txt_feat, concept_scores = _encode_batch(clip_imgs, batch["texts"])
        logits_img, _, _ = img_model(img_feat, concept_scores)
        logits_txt, _, _ = txt_model(txt_feat, tbm_f)

        if fusion == "avg":
            prob = (F.softmax(logits_img, 1) + F.softmax(logits_txt, 1)) / 2.0
        elif fusion == "img":
            prob = F.softmax(logits_img, 1)
        else:
            prob = F.softmax(logits_txt, 1)

        bin_prob = prob[:, 1]; bin_pred = prob.argmax(1)
        texts = batch["texts"]; paths = batch["image_paths"]; fcs = batch["fake_cls"]
        for i in range(len(labels)):
            rows.append({
                "text":           texts[i],
                "image_path":     paths[i],
                "fake_cls":       fcs[i],
                "binary_label":   int(labels[i]),
                "binary_pred":    int(bin_pred[i]),
                "binary_correct": int(labels[i]) == int(bin_pred[i]),
                "binary_prob":    round(float(bin_prob[i]), 4),
            })
    return pd.DataFrame(rows)

def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    yt = df["binary_label"].values; yp = df["binary_pred"].values
    yprob = df["binary_prob"].values
    try:
        fpr, tpr, thr = roc_curve(yt, yprob)
        yp_opt = (yprob >= float(thr[np.argmax(tpr - fpr)])).astype(int)
        f1_opt = float(f1_score(yt, yp_opt, average="binary",
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
    # per fake_cls 정확도
    if "fake_cls" in df.columns:
        for cls_name in df["fake_cls"].unique():
            sub = df[df["fake_cls"] == cls_name]
            if len(sub):
                m[f"acc_{cls_name}"] = float(accuracy_score(
                    sub["binary_label"].values, sub["binary_pred"].values))
    return m

def eval_report_block(df: pd.DataFrame, label: str = LOG_TAG):
    buf = io.StringIO()
    buf.write(f"\n=== {label} — Binary Detection ===\n")
    buf.write(classification_report(
        df["binary_label"].values, df["binary_pred"].values,
        target_names=["real", "fake"], digits=4, zero_division=0))
    m = compute_metrics(df)
    buf.write(
        f"\n[{label}]  ACC={m['bin_acc']:.4f}  F1={m['bin_f1']:.4f}"
        f"  F1_opt={m['bin_f1_opt']:.4f}"
        f"  AUC={m.get('bin_auc', float('nan')):.4f}\n")
    if "fake_cls" in df.columns:
        buf.write("\n--- Per-class accuracy (fake_cls) ---\n")
        for cls_name in sorted(df["fake_cls"].unique()):
            buf.write(f"  {cls_name:25s}: {m.get(f'acc_{cls_name}', float('nan')):.4f}\n")
    return buf.getvalue(), m

# ════════════════════════════════════════════════════════════════════════════════
# 11. 체크포인트 I/O
# ════════════════════════════════════════════════════════════════════════════════
def save_checkpoint(path, img_model, txt_model, args, metrics=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "img_model_state": img_model.state_dict(),
        "txt_model_state": txt_model.state_dict(),
        "config":  vars(args),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if metrics:
        payload["metrics"] = metrics
    torch.save(payload, path)
    print(f"[{LOG_TAG}] 체크포인트 저장: {path}")

def load_checkpoint(img_model, txt_model, path, device):
    try:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
    except Exception as e:
        print(f"[{LOG_TAG}] 체크포인트 로드 실패: {e}"); return False
    def _load(model, key, name):
        if key in ckpt:
            mis, unexp = model.load_state_dict(ckpt[key], strict=False)
            print(f"[{LOG_TAG}] {name}  missing={len(mis)} unexpected={len(unexp)}")
        else:
            print(f"[{LOG_TAG}] WARNING: '{key}' 없음 — {name} 로드 생략")
    _load(img_model, "img_model_state", "MiRAGeImgDetector")
    _load(txt_model, "txt_model_state", "MiRAGeTxtDetector")
    return True

# ════════════════════════════════════════════════════════════════════════════════
# 12. 모델 빌드 & 학습 루프
# ════════════════════════════════════════════════════════════════════════════════
def _build_models(device):
    img_model = MiRAGeImgDetector(num_classes=2).to(device)
    txt_model = MiRAGeTxtDetector(num_classes=2).to(device)
    n_img = sum(p.numel() for p in img_model.parameters() if p.requires_grad)
    n_txt = sum(p.numel() for p in txt_model.parameters() if p.requires_grad)
    print(f"[{LOG_TAG}] MiRAGe-Img 파라미터: {n_img:,}  |  MiRAGe-Txt 파라미터: {n_txt:,}  (CLIP 제외)")
    return img_model, txt_model

def train_one_run(img_model, txt_model, tl, vl, device, args, log=None):
    opt_img = torch.optim.AdamW(img_model.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)
    opt_txt = torch.optim.AdamW(txt_model.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)
    early    = EarlyStopping(args.early_stop_patience, args.early_stop_min_delta)
    best_f1 = -1.0; best_st = None

    for ep in range(1, args.epochs + 1):
        l_img, l_txt = train_one_epoch(
            img_model, txt_model, tl, opt_img, opt_txt, device, ep)

        va = eval_epoch(img_model, txt_model, vl, device, fusion="avg")
        cur_lr = opt_img.param_groups[0]["lr"]
        line = (f"[{LOG_TAG}] Ep {ep:02d}/{args.epochs}"
                f"  l_img={l_img:.4f}  l_txt={l_txt:.4f}"
                f"  va_f1={va['bin_f1']:.4f}"
                f"  va_auc={va['bin_auc']:.4f}"
                f"  va_acc={va['bin_acc']:.4f}"
                f"  lr={cur_lr:.2e}")
        print(line)
        if log is not None: log.append(line)

        monitor = va["bin_f1"]
        if not math.isnan(monitor) and monitor > best_f1:
            best_f1 = monitor
            best_st  = {"img": copy.deepcopy(img_model.state_dict()),
                        "txt": copy.deepcopy(txt_model.state_dict())}
        if early.step(monitor):
            msg = f"[{LOG_TAG}] Early stopping ep {ep} (best_f1={best_f1:.4f})"
            print(msg)
            if log: log.append(msg)
            break

    if best_st:
        img_model.load_state_dict(best_st["img"])
        txt_model.load_state_dict(best_st["txt"])
        print(f"[{LOG_TAG}] best val macro-F1={best_f1:.4f} 가중치 복원 완료")
    return img_model, txt_model

# ════════════════════════════════════════════════════════════════════════════════
# 13. DataLoader 헬퍼
# ════════════════════════════════════════════════════════════════════════════════
def _make_dl(df, data_root, args, shuffle, sampler=None):
    ds = MiRAGeDGM4Dataset(df, data_root)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              collate_fn=collate_mirage_dgm4,
              pin_memory=torch.cuda.is_available(), drop_last=shuffle)
    if sampler is not None:
        kw["sampler"] = sampler; kw["shuffle"] = False
    else:
        kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)

# ════════════════════════════════════════════════════════════════════════════════
# 14. run_eval_only / run_official / run_kfold
# ════════════════════════════════════════════════════════════════════════════════
def run_eval_only(img_model, txt_model, loader, device, split_name, result_dir):
    print(f"\n[{LOG_TAG}] device={device} | split={split_name} | eval_only")
    os.makedirs(result_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for fusion in ("img", "txt", "avg"):
        full_df = collect_predictions(img_model, txt_model, loader, device, fusion)
        blk, m  = eval_report_block(full_df, label=f"{LOG_TAG}_{split_name}_{fusion}")
        print(blk, end="")
        rp = os.path.join(result_dir, f"mirage_dgm4_{split_name}_{fusion}_{ts}.txt")
        with open(rp, "w", encoding="utf-8") as f: f.write(blk)
        full_df.to_csv(rp.replace(".txt", "_predictions.csv"), index=False)
        print(f"[{LOG_TAG}] 저장: {rp}")

def run_official(dfs, device, args, result_dir):
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (Official Split) ######\n\n")
    os.makedirs(result_dir, exist_ok=True)

    tr_df    = _ensure_dgm4_columns(dfs["train"])
    _sampler = (make_weighted_sampler(tr_df["binary_label"].tolist(), args.sampler_alpha)
                if getattr(args, "weighted_sampler", False) else None)
    tl = _make_dl(dfs["train"],      args.data_root, args, shuffle=(_sampler is None), sampler=_sampler)
    vl = _make_dl(dfs["validation"], args.data_root, args, shuffle=False)
    el = _make_dl(dfs["test"],       args.data_root, args, shuffle=False)

    img_model, txt_model = _build_models(device)
    log = []
    img_model, txt_model = train_one_run(img_model, txt_model, tl, vl, device, args, log)
    for line in log: rep.write(line + "\n")

    all_metrics = {}
    for fusion in ("img", "txt", "avg"):
        print(f"\n[{LOG_TAG}] test 평가 (fusion={fusion}) ...")
        full_df = collect_predictions(img_model, txt_model, el, device, fusion)
        blk, m  = eval_report_block(full_df, label=f"{LOG_TAG}_test_{fusion}")
        print(blk, end=""); rep.write(blk)
        full_df.to_csv(
            os.path.join(result_dir, f"mirage_dgm4_official_{fusion}_{ts}_predictions.csv"),
            index=False)
        all_metrics[fusion] = m

    print("\n" + "=" * 70)
    for fusion, m in all_metrics.items():
        summ = (f"{LOG_TAG} [{fusion:3s}]: ACC={m['bin_acc']:.4f}  F1={m['bin_f1']:.4f}"
                f"  F1_opt={m['bin_f1_opt']:.4f}  AUC={m.get('bin_auc', float('nan')):.4f}")
        print(summ); rep.write(summ + "\n")
    print("=" * 70)

    ckpt_path = os.path.join(getattr(args, "checkpoint_dir", result_dir),
                             f"mirage_dgm4_official_{ts}.pt")
    save_checkpoint(ckpt_path, img_model, txt_model, args, all_metrics.get("avg"))
    rp = os.path.join(result_dir, f"mirage_dgm4_official_{ts}.txt")
    with open(rp, "w", encoding="utf-8") as f: f.write(rep.getvalue())
    print(f"[{LOG_TAG}] 리포트: {rp}")

def run_kfold(dfs, device, args, result_dir):
    df_tv = _ensure_dgm4_columns(
        pd.concat([dfs["train"], dfs["validation"]], ignore_index=True))
    y_tv  = df_tv["binary_label"].values
    k     = args.kfold_splits
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(result_dir, exist_ok=True)
    all_rep = io.StringIO()
    all_rep.write(f"###### {LOG_TAG} (k={k} KFold) ######\n")
    el = _make_dl(dfs["test"], args.data_root, args, shuffle=False)
    skf    = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.kfold_random_state)
    fold_m = {"img": [], "txt": [], "avg": []}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df_tv)), y_tv), 1):
        print(f"\n{'='*70}\n[Fold {fold}/{k}] {LOG_TAG}\n{'='*70}")
        df_tr_f = df_tv.iloc[tr_idx].reset_index(drop=True)
        df_va_f = df_tv.iloc[va_idx].reset_index(drop=True)
        _sampler = (make_weighted_sampler(df_tr_f["binary_label"].tolist(), args.sampler_alpha)
                    if getattr(args, "weighted_sampler", False) else None)
        tl = _make_dl(df_tr_f, args.data_root, args, shuffle=(_sampler is None), sampler=_sampler)
        vl = _make_dl(df_va_f, args.data_root, args, shuffle=False)

        img_model, txt_model = _build_models(device)
        log = []
        img_model, txt_model = train_one_run(img_model, txt_model, tl, vl, device, args, log)

        fold_blk = io.StringIO()
        for fusion in ("img", "txt", "avg"):
            full_df = collect_predictions(img_model, txt_model, el, device, fusion)
            blk, m  = eval_report_block(full_df, label=f"Fold{fold}_{fusion}")
            print(blk, end=""); fold_blk.write(blk); fold_m[fusion].append(m)

        ckpt = os.path.join(getattr(args, "checkpoint_dir", result_dir),
                            f"mirage_dgm4_kfold_{ts}_fold{fold}.pt")
        save_checkpoint(ckpt, img_model, txt_model, args)
        all_rep.write(f"\n{'#'*80}\n### FOLD {fold}\n{'#'*80}\n"
                      + "\n".join(log) + "\n" + fold_blk.getvalue())
        del img_model, txt_model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _mean(lst, key):
        vals = [d[key] for d in lst if not math.isnan(d.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    all_rep.write(f"\n\n{'#'*80}\n### GLOBAL SUMMARY (mean over {k} folds)\n{'#'*80}\n")
    all_rep.write("Variant\tACC\tF1\tF1_opt\tAUC\n")
    print("\n" + "=" * 70)
    for fusion in ("img", "txt", "avg"):
        m_acc = _mean(fold_m[fusion], "bin_acc"); m_f1 = _mean(fold_m[fusion], "bin_f1")
        m_fo  = _mean(fold_m[fusion], "bin_f1_opt"); m_auc = _mean(fold_m[fusion], "bin_auc")
        summ  = (f"MiRAGe [{fusion:3s}]: ACC={m_acc:.4f}  F1={m_f1:.4f}"
                 f"  F1_opt={m_fo:.4f}  AUC={m_auc:.4f}")
        print(summ)
        all_rep.write(f"MiRAGe-{fusion}\t{m_acc:.4f}\t{m_f1:.4f}\t{m_fo:.4f}\t{m_auc:.4f}\n")
    print("=" * 70)

    rp = os.path.join(result_dir, f"mirage_dgm4_kfold_{ts}.txt")
    with open(rp, "w", encoding="utf-8") as f: f.write(all_rep.getvalue())
    print(f"[{LOG_TAG}] 리포트: {rp}")

# ════════════════════════════════════════════════════════════════════════════════
# 15. main
# ════════════════════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(description=f"{LOG_TAG} — MiRAGe × DGM4")
    pa.add_argument("--data_root",      default=DEFAULT_DATA_ROOT)
    pa.add_argument("--checkpoint_dir", default="./checkpoints")
    pa.add_argument("--result_dir",     default="./results")
    pa.add_argument("--eval_only",   action="store_true")
    pa.add_argument("--checkpoint",  type=str, default="")
    pa.add_argument("--split",       type=str, default="test",
                    choices=("train", "validation", "test"))
    pa.add_argument("--batch_size",           type=int,   default=BATCH)
    pa.add_argument("--epochs",               type=int,   default=EPOCHS)
    pa.add_argument("--lr",                   type=float, default=LR)
    pa.add_argument("--weight_decay",         type=float, default=1e-4)
    pa.add_argument("--num_workers",          type=int,   default=NUM_WORKERS)
    pa.add_argument("--seed",                 type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    pa.add_argument("--sampler_alpha",        type=float, default=SAMPLER_ALPHA)
    pa.add_argument("--dropout",              type=float, default=DROPOUT)
    pa.add_argument("--weighted_sampler",     action="store_true")
    pa.add_argument("--limit",                type=int,   default=0,
                    help="디버그용: 각 split 을 N 개로 제한 (0=전체)")
    pa.add_argument("--kfold",              action="store_true")
    pa.add_argument("--kfold_splits",       type=int, default=KFOLD_SPLITS)
    pa.add_argument("--kfold_random_state", type=int, default=KFOLD_RANDOM_STATE)
    pa.add_argument("--no_progress",        action="store_true")
    args = pa.parse_args()

    if args.no_progress:
        os.environ["DGM4_NO_TQDM"] = "1"
    set_seed(args.seed)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    mode = "KFold" if args.kfold else "Official Split"
    sampler_note = "WeightedSampler" if getattr(args, "weighted_sampler", False) else "shuffle=True"
    print(
        f"\n{LOG_TAG} | device={DEVICE} | mode={mode}\n"
        f"  CLIP=ViT-B-32(frozen)  CBM={NUM_CONCEPTS}개 개념  TBM={NUM_TBM_FEATS}개 언어통계\n"
        f"  lr={args.lr:.2e}  wd={args.weight_decay}"
        f"  epochs={args.epochs}  batch={args.batch_size}  sampler={sampler_note}"
    )

    dfs = load_dgm4_splits(args.data_root, LOG_TAG)
    if args.limit > 0:
        dfs = {k: v.head(args.limit).reset_index(drop=True) for k, v in dfs.items()}
        print(f"[{LOG_TAG}] --limit={args.limit} 적용")

    if args.eval_only:
        img_model, txt_model = _build_models(DEVICE)
        if args.checkpoint:
            load_checkpoint(img_model, txt_model, args.checkpoint, DEVICE)
        else:
            print(f"[{LOG_TAG}] WARNING: --checkpoint 없음 → 랜덤 초기화로 평가")
        ev_loader = _make_dl(_ensure_dgm4_columns(dfs[args.split]),
                             args.data_root, args, shuffle=False)
        run_eval_only(img_model, txt_model, ev_loader, DEVICE,
                      args.split, args.result_dir)
        return

    if args.kfold:
        run_kfold(dfs, DEVICE, args, args.result_dir)
    else:
        run_official(dfs, DEVICE, args, args.result_dir)


if __name__ == "__main__":
    main()