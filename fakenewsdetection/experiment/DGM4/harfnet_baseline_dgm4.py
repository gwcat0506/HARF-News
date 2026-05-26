"""
HARFNET-BASELINE-Concat × DGM4 벤치마크
=========================================
가장 일반적인 멀티모달 구조로 DGM4 이진 탐지 수행.

모델:
  RoBERTa → T_cls [B, d]   (CLS 토큰)
  CLIP    → V_cls [B, d]   (global image feature, frozen)
  concat([T_cls, V_cls])   [B, 2d]
  → MLP → Binary (real / fake)

AM / VM / OA / BilinearInteraction / Cross-Attention 전부 없음.
손실: Binary Focal Loss only

DGM4 데이터셋:
  로컬: {data_root}/metadata/{train,val,test}.json  + 이미지 폴더
  HuggingFace: rshaojimmy/DGM4  (로컬 파일 없을 때 자동 폴백)

레이블:
  binary_label 0 = real (orig)
  binary_label 1 = fake (face_swap / face_attribute / text_swap / text_attribute)

실행 예시:
  python baseline_dgm4.py --data_root /path/to/dgm4

  # Official Split (기본)
  python baseline_dgm4.py --data_root /path/to/dgm4

  # K-Fold (선택)
  python baseline_dgm4.py --data_root /path/to/dgm4 --kfold
"""

from __future__ import annotations

import argparse, copy, gc, io, math, os, random, sys, warnings
from datetime import datetime
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import clip as openai_clip
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizerFast
from dgm4_paths import DEFAULT_DATA_ROOT

# ─────────────────────────────────────────────────────────────
# 1. 상수
# ─────────────────────────────────────────────────────────────
BATCH               = 16
EPOCHS              = 10
LR                  = 1e-4
NUM_WORKERS         = 4
EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_DELTA= 1e-4
KFOLD_SPLITS        = 5
KFOLD_RANDOM_STATE  = 42
SEED                = 42

ROBERTA    = "roberta-base"
CLIP_RN101 = "RN101"
MAX_LENGTH = 128

# DGM4 레이블
FINE_LABELS  = ["face_swap", "face_attribute", "text_swap", "text_attribute"]
BINARY_NAMES = ["real", "fake"]

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR     = os.path.join(_SCRIPT_DIR, "results")
CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")

if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_TAG   = "HARFNET-BASELINE-Concat-DGM4"
LOG_BRAND = "HARFNET-BASELINE-Concat-DGM4"

os.makedirs(RESULT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def set_seed(s: int = SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed()


# ─────────────────────────────────────────────────────────────
# 2. DGM4 유틸
# ─────────────────────────────────────────────────────────────
def parse_fake_cls(cls_str: str) -> Tuple[int, List[int]]:
    """fake_cls 문자열 → (binary_label, fine_labels[4])"""
    cs = cls_str.strip().lower()
    if cs == "orig":
        return 0, [0, 0, 0, 0]
    parts = cs.split("&")
    fine  = [int(lbl in parts) for lbl in FINE_LABELS]
    return 1, fine


def resolve_image_path(rel: str, data_root: str) -> str:
    """상대 경로 → 절대 경로 (다양한 DGM4 폴더 구조 대응)"""
    rel = (rel or "").strip()
    if not rel:
        return ""
    if os.path.isabs(rel) and os.path.isfile(rel):
        return rel

    root     = os.path.abspath(data_root)
    rel_norm = rel.replace("\\", "/").lstrip("/")

    candidates = [os.path.join(root, rel_norm)]

    if rel_norm.startswith("DGM4/"):
        rel_wo = rel_norm[len("DGM4/"):]
        candidates.append(os.path.join(root, rel_wo))
    else:
        rel_wo = rel_norm

    parts = rel_wo.split("/")
    if len(parts) >= 3 and parts[0] in {"manipulation", "origin"}:
        nested = "/".join([parts[0], parts[1], parts[1], *parts[2:]])
        candidates.append(os.path.join(root, nested))

    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return candidates[-1] if candidates else ""


def _tqdm(it, **kw):
    if os.environ.get("DGM4_NO_TQDM", "").lower() in ("1","true","yes"):
        return it
    kw.setdefault("file", sys.stderr); kw.setdefault("dynamic_ncols", True)
    try: sys.stdout.flush(); return tqdm(it, **kw)
    except Exception: return it


# ─────────────────────────────────────────────────────────────
# 3. DGM4 데이터 로드
# ─────────────────────────────────────────────────────────────
def load_dgm4_splits(data_root: str) -> Dict[str, pd.DataFrame]:
    """
    로컬 JSON 우선 로드, 없으면 HuggingFace 폴백.
    반환: {"train": df, "validation": df, "test": df}
    """
    split_files = {
        "train":      os.path.join(data_root, "metadata", "train.json"),
        "validation": os.path.join(data_root, "metadata", "val.json"),
        "test":       os.path.join(data_root, "metadata", "test.json"),
    }
    hf_paths = {
        "train":      "metadata/train.json",
        "validation": "metadata/val.json",
        "test":       "metadata/test.json",
    }
    dfs = {}
    for split, path in split_files.items():
        if os.path.isfile(path):
            print(f"[{LOG_BRAND}] 로컬 로드: {path}")
            dfs[split] = pd.read_json(path)
        else:
            print(f"[{LOG_BRAND}] HuggingFace에서 로드: {split}")
            dfs[split] = pd.read_json(
                "hf://datasets/rshaojimmy/DGM4/" + hf_paths[split])

    for split, df in dfs.items():
        df["text"]     = df["text"].fillna("").astype(str).str.strip()
        df["image"]    = df["image"].fillna("").astype(str).str.strip()
        df["fake_cls"] = df["fake_cls"].fillna("orig").astype(str).str.strip()

        # 이미지 유효성 검사
        def _has(rel):
            full = resolve_image_path(rel, data_root)
            return bool(full) and os.path.isfile(full)
        df["has_image_flag"] = df["image"].map(_has)

        # 레이블 파싱
        parsed = df["fake_cls"].map(parse_fake_cls)
        df["binary_label"] = parsed.map(lambda x: x[0])
        for i, lbl in enumerate(FINE_LABELS):
            df[f"fine_{lbl}"] = parsed.map(lambda x, i=i: x[1][i])

        n_total = len(df)
        n_img   = int(df["has_image_flag"].sum())
        n_real  = int((df["binary_label"] == 0).sum())
        n_fake  = int((df["binary_label"] == 1).sum())
        print(f"[{LOG_BRAND}] {split}: {n_total}행  "
              f"이미지 유효={n_img}  real={n_real}  fake={n_fake}")
        dfs[split] = df.reset_index(drop=True)

    return dfs


# ─────────────────────────────────────────────────────────────
# 4. EarlyStopping
# ─────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE,
                 min_delta=EARLY_STOP_MIN_DELTA, mode="max"):
        self.patience=patience; self.min_delta=min_delta
        self.mode=mode; self.best=None; self.counter=0

    def step(self, v: float) -> bool:
        if self.best is None: self.best=v; return False
        imp=(v-self.best>self.min_delta if self.mode=="max"
             else self.best-v>self.min_delta)
        if imp: self.best=v; self.counter=0; return False
        self.counter+=1
        return self.counter>=self.patience


# ─────────────────────────────────────────────────────────────
# 5. DGM4Dataset
# ─────────────────────────────────────────────────────────────
class DGM4Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str,
                 tokenizer, clip_preprocess,
                 max_length: int = MAX_LENGTH):
        super().__init__()
        self.data_root       = os.path.abspath(data_root)
        self.tokenizer       = tokenizer
        self.clip_preprocess = clip_preprocess
        self.max_length      = max_length

        self.texts        = df["text"].tolist()
        self.image_rels   = df["image"].tolist()
        self.has_img_flag = df["has_image_flag"].tolist()
        self.binary_lbls  = df["binary_label"].tolist()
        self.fake_cls_str = df["fake_cls"].tolist()

    def __len__(self): return len(self.binary_lbls)

    def _load_pil(self, rel: str, flag: bool):
        if not flag or not rel:
            return Image.new("RGB", (224, 224), (127, 127, 127)), False
        full = resolve_image_path(rel, self.data_root)
        if not full or not os.path.isfile(full):
            return Image.new("RGB", (224, 224), (127, 127, 127)), False
        try:
            return Image.open(full).convert("RGB"), True
        except Exception:
            return Image.new("RGB", (224, 224), (127, 127, 127)), False

    def __getitem__(self, idx):
        text    = self.texts[idx]
        img, ok = self._load_pil(self.image_rels[idx],
                                  self.has_img_flag[idx])
        enc = self.tokenizer(
            text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids":        enc["input_ids"].squeeze(0),
            "attention_mask":   enc["attention_mask"].squeeze(0),
            "pixel_values":     self.clip_preprocess(img),
            "clip_text_tokens": openai_clip.tokenize([text], truncate=True)[0],
            "has_image":        torch.tensor(1.0 if ok else 0.0),
            "label":            torch.tensor(self.binary_lbls[idx], dtype=torch.long),
            "fake_cls":         self.fake_cls_str[idx],
        }


def collate_batch(batch):
    tensor_keys = ("input_ids", "attention_mask", "pixel_values",
                   "clip_text_tokens", "has_image", "label")
    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out["fake_cls"] = [b["fake_cls"] for b in batch]
    return out


def make_weighted_sampler(binary_labels: List[int], alpha: float = 0.5):
    cnt = pd.Series(binary_labels).value_counts().to_dict()
    w   = [(1.0 / cnt[l]) ** alpha for l in binary_labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


# ─────────────────────────────────────────────────────────────
# 6. 베이스라인 모델
#
#    RoBERTa → T_cls [B, d]  (CLS 토큰)
#    CLIP    → V_cls [B, d]  (global image feature, frozen)
#    concat([T_cls, V_cls])  [B, 2d]
#    → MLP → Binary (real / fake)
# ─────────────────────────────────────────────────────────────
class ConcatBaseline_DGM4(nn.Module):
    def __init__(self, roberta_name=ROBERTA, clip_name=CLIP_RN101,
                 dropout=0.1):
        super().__init__()
        d = 768

        # ── 텍스트 인코더 (RoBERTa) ──
        self.text_encoder = RobertaModel.from_pretrained(
            roberta_name, add_pooling_layer=False)

        # ── 이미지 인코더 (CLIP, 완전 frozen) ──
        _clip, _ = openai_clip.load(clip_name, device="cpu")
        self.clip_model = _clip.float()
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        self.visual   = self.clip_model.visual
        clip_edim     = self.visual.attnpool.c_proj.out_features
        self.cls_proj = nn.Linear(clip_edim, d)

        # ── Binary Classifier: concat(T_cls, V_cls) → [B, 2d] → 1 ──
        self.classifier = nn.Sequential(
            nn.Linear(d * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),   # binary logit (BCEWithLogitsLoss)
        )

    def _clip_visual_forward(self, pix, has_image):
        vis = self.visual; B = pix.size(0)
        x = vis.relu1(vis.bn1(vis.conv1(pix)))
        x = vis.relu2(vis.bn2(vis.conv2(x)))
        x = vis.relu3(vis.bn3(vis.conv3(x)))
        x = vis.avgpool(x) * has_image.view(B, 1, 1, 1)
        x = vis.layer1(x); x = vis.layer2(x)
        x = vis.layer3(x); x = vis.layer4(x)
        v_global = vis.attnpool(x)
        V_cls    = self.cls_proj(v_global) * has_image.view(B, 1)
        return V_cls

    def train(self, mode=True):
        super().train(mode); self.clip_model.eval(); return self

    def forward(self, input_ids, attention_mask, pixel_values,
                has_image, clip_text_tokens=None):
        device    = input_ids.device
        has_image = has_image.to(device)

        T_tok = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask).last_hidden_state
        T_cls = T_tok[:, 0]

        V_cls   = self._clip_visual_forward(pixel_values, has_image)
        F_fused = torch.cat([T_cls, V_cls], dim=-1)
        logit   = torch.nan_to_num(
            self.classifier(F_fused).squeeze(-1),
            nan=0., posinf=20., neginf=-20.)

        return {"binary_logit": logit}


# ─────────────────────────────────────────────────────────────
# 7. 손실 함수 — Binary Focal Loss only
# ─────────────────────────────────────────────────────────────
def binary_focal_loss(logit, target, gamma=2.0, pos_weight=1.0):
    """
    DGM4 train 셋이 real=fake 균형(각 104,092건)이므로
    pos_weight=1.0 사용 (과도한 fake 강조 방지).
    """
    pw  = torch.tensor([pos_weight], device=logit.device)
    bce = F.binary_cross_entropy_with_logits(
        logit, target.float(), pos_weight=pw, reduction="none")
    prob = torch.sigmoid(logit)
    pt   = torch.where(target == 1, prob, 1 - prob)
    return ((1 - pt) ** gamma * bce).mean()


def total_loss(out, y, has_image, args):
    return binary_focal_loss(out["binary_logit"], y)


# ─────────────────────────────────────────────────────────────
# 8. 학습 / 평가
# ─────────────────────────────────────────────────────────────
def _fwd(model, batch, device):
    return model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["pixel_values"].to(device),
        batch["has_image"].to(device),
    )


def run_epoch(model, loader, device, optimizer, train,
              epoch_idx=None, args=None):
    model.train() if train else model.eval()
    tot, ys, ps, probs, n = 0., [], [], [], 0
    ctx  = torch.enable_grad() if train else torch.no_grad()
    desc = f"[{LOG_TAG}] Epoch {epoch_idx}" if (train and epoch_idx) else None
    it   = _tqdm(loader, desc=desc) if (train and desc) else loader

    with ctx:
        for batch in it:
            y  = batch["label"].to(device)
            hi = batch["has_image"].to(device)
            out  = _fwd(model, batch, device)
            loss = total_loss(out, y, hi, args)
            if train and optimizer:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            prob = torch.sigmoid(out["binary_logit"]).detach()
            pred = (prob > 0.5).long()
            tot  += loss.item() * y.size(0)
            ys.extend(y.cpu().tolist())
            ps.extend(pred.cpu().tolist())
            probs.extend(prob.cpu().tolist())
            n += y.size(0)

    n = max(n, 1)
    try:    auc = float(roc_auc_score(ys, probs))
    except: auc = float("nan")
    return {
        "loss":     tot / n,
        "acc":      float(accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
        "f1_fake":  float(f1_score(ys, ps, pos_label=1, average="binary", zero_division=0)),
        "f1_real":  float(f1_score(ys, ps, pos_label=0, average="binary", zero_division=0)),
        "auc":      auc,
    }


def collect_predictions(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in _tqdm(loader, desc=f"Eval ({LOG_TAG})"):
            out      = _fwd(model, batch, device)
            bin_prob = torch.sigmoid(out["binary_logit"])
            bin_pred = (bin_prob > 0.5).long()
            bin_lbl  = batch["label"]
            fcs      = batch.get("fake_cls", [""] * bin_lbl.size(0))
            for i in range(bin_lbl.size(0)):
                rows.append({
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
    m = {
        "bin_acc":     float(accuracy_score(yt, yp)),
        "bin_f1":      float(f1_score(yt, yp, average="macro", zero_division=0)),
        "bin_f1_real": float(f1_score(yt, yp, pos_label=0,
                                      average="binary", zero_division=0)),
        "bin_f1_fake": float(f1_score(yt, yp, pos_label=1,
                                      average="binary", zero_division=0)),
    }
    try:    m["bin_auc"] = float(roc_auc_score(yt, yprob))
    except: m["bin_auc"] = float("nan")
    return m


def eval_report_block(df: pd.DataFrame, label: str = "") -> Tuple[str, dict]:
    label = label or LOG_TAG
    yt = df["binary_label"].values
    yp = df["binary_pred"].values
    buf = io.StringIO()
    buf.write(f"\n=== {label} — Binary Detection (Real vs Fake) ===\n")
    buf.write(classification_report(yt, yp, target_names=BINARY_NAMES,
                                    digits=4, zero_division=0))
    buf.write(f"Confusion Matrix:\n{confusion_matrix(yt, yp)}\n")
    m = compute_metrics(df)
    buf.write(f"\n[{label}] Acc={m['bin_acc']:.4f}  Macro-F1={m['bin_f1']:.4f}"
              f"  F1_real={m['bin_f1_real']:.4f}  F1_fake={m['bin_f1_fake']:.4f}"
              f"  AUC={m['bin_auc']:.4f}\n")
    # fake_cls별 정확도
    buf.write("\n[fake_cls별 정확도]\n")
    for cls in sorted(df["fake_cls"].unique()):
        sub = df[df["fake_cls"] == cls]
        acc = sub["binary_correct"].mean()
        buf.write(f"  {cls:30s}  n={len(sub):6d}  acc={acc:.4f}\n")
    return buf.getvalue(), m


# ─────────────────────────────────────────────────────────────
# 9. 학습 헬퍼
# ─────────────────────────────────────────────────────────────
def freeze_backbones(model: ConcatBaseline_DGM4):
    """RoBERTa 마지막 2개 레이어(10,11)만 학습. CLIP은 완전 frozen 유지."""
    for name, p in model.text_encoder.named_parameters():
        p.requires_grad_(any(f"encoder.layer.{i}" in name for i in (10, 11)))
    # CLIP은 이미 __init__에서 frozen
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{LOG_TAG}] 학습 파라미터: {n_train:,} / {n_total:,}"
          f" ({100 * n_train / n_total:.1f}%)")


def plot_epoch_curves(epoch_log: list, save_path: str, title: str = ""):
    epochs = [e["epoch"]      for e in epoch_log]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title or LOG_TAG, fontsize=13, fontweight="bold")
    ax1.plot(epochs, [e["train_loss"] for e in epoch_log],
             "o-", color="#2563EB", lw=2, ms=5, label="Train Loss")
    ax1.plot(epochs, [e["val_loss"]   for e in epoch_log],
             "s--", color="#DC2626", lw=2, ms=5, label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_xticks(epochs)
    vf = [e["val_f1"] for e in epoch_log]
    ax2.plot(epochs, vf, "^-", color="#16A34A", lw=2, ms=5, label="Val Macro F1")
    best_ep = epochs[int(np.argmax(vf))]
    ax2.axvline(best_ep, color="#16A34A", linestyle=":", alpha=0.6)
    ax2.annotate(f"Best:{max(vf):.4f}\n(Ep{best_ep})",
                 xy=(best_ep, max(vf)), xytext=(best_ep + 0.3, max(vf) - 0.01),
                 fontsize=9, color="#15803D",
                 arrowprops=dict(arrowstyle="->", color="#15803D"))
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1"); ax2.set_title("Val Macro F1")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"그래프 저장: {save_path}")


def save_checkpoint(path, model, args, metrics=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config":           vars(args),
        "saved_at":         datetime.now().isoformat(timespec="seconds"),
    }
    if metrics: payload["metrics"] = metrics
    torch.save(payload, path)
    print(f"체크포인트 저장: {path}")


def train_one_run(model, tl, vl, device, args, log=None):
    epoch_log = []
    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    early = EarlyStopping(args.early_stop_patience, args.early_stop_min_delta)
    best_f1, best_st = -1., None

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, tl, device, opt, True, ep, args)
        va = run_epoch(model, vl, device, None, False, args=args)
        line = (f"[{LOG_TAG}] Epoch {ep:02d} "
                f"train_loss={tr['loss']:.4f}  val_loss={va['loss']:.4f}  "
                f"val_f1={va['macro_f1']:.4f}  val_auc={va['auc']:.4f}  "
                f"val_acc={va['acc']:.4f}")
        print(line)
        epoch_log.append({
            "epoch":      ep,
            "train_loss": tr["loss"],
            "val_loss":   va["loss"],
            "val_f1":     va["macro_f1"],
        })
        if log: log.append(line)

        if best_st is None or (va["macro_f1"] - best_f1) > args.early_stop_min_delta:
            best_f1 = va["macro_f1"]
            best_st = copy.deepcopy(model.state_dict())
        if early.step(va["macro_f1"]):
            msg = f"[{LOG_TAG}] Early stopping at epoch {ep}"
            print(msg)
            if log: log.append(msg)
            break

    if best_st: model.load_state_dict(best_st)
    return model, epoch_log


def _make_dl(df, data_root, tok, prep, args, shuffle, sampler=None):
    ds = DGM4Dataset(df, data_root, tok, prep, args.max_length)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              collate_fn=collate_batch,
              pin_memory=torch.cuda.is_available(), drop_last=shuffle)
    if sampler: kw["sampler"] = sampler; kw["shuffle"] = False
    else:       kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


def _new_model(args, device):
    return ConcatBaseline_DGM4(
        roberta_name=args.roberta,
        clip_name=args.clip_model).to(device)


# ─────────────────────────────────────────────────────────────
# 10. Official Split 실행 (기본, 권장)
# ─────────────────────────────────────────────────────────────
def run_official(dfs: dict, device, tok, prep, args):
    """
    DGM4 공식 train/val/test split 사용.
    val/test: real:fake ≈ 1:2 → Macro-F1 기준 early stopping 적용.
    """
    set_seed(args.seed)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (Official Split) ######\n")
    rep.write(f"[구조] RoBERTa CLS + CLIP global → concat → MLP → binary\n")
    rep.write(f"[손실] Binary Focal Loss only\n\n")

    tr_ds = DGM4Dataset(dfs["train"], args.data_root, tok, prep, args.max_length)
    tl    = _make_dl(dfs["train"], args.data_root, tok, prep, args,
                     shuffle=True,
                     sampler=make_weighted_sampler(tr_ds.binary_lbls,
                                                    args.sampler_alpha))
    vl    = _make_dl(dfs["validation"], args.data_root, tok, prep,
                     args, shuffle=False)
    el    = _make_dl(dfs["test"],       args.data_root, tok, prep,
                     args, shuffle=False)

    model = _new_model(args, device)
    if not args.no_freeze_encoders:
        freeze_backbones(model)

    log = []
    model, epoch_log = train_one_run(model, tl, vl, device, args, log)
    for line in log: rep.write(line + "\n")

    plot_epoch_curves(epoch_log,
                      os.path.join(RESULT_DIR, f"baseline_dgm4_curve_{ts}.png"),
                      title=f"{LOG_TAG} (Official Split)")

    print(f"\n[{LOG_TAG}] 테스트셋 평가...")
    full_df = collect_predictions(model, el, device)
    blk, m  = eval_report_block(full_df, label="Test")
    print(blk); rep.write(blk)

    save_checkpoint(
        os.path.join(args.checkpoint_dir, f"baseline_dgm4_official_{ts}.pt"),
        model, args, m)

    summ = (f"{LOG_TAG}: Acc={m['bin_acc']:.4f}  Macro-F1={m['bin_f1']:.4f}"
            f"  F1_real={m['bin_f1_real']:.4f}  F1_fake={m['bin_f1_fake']:.4f}"
            f"  AUC={m['bin_auc']:.4f}")
    print("\n" + "="*70 + "\n" + summ + "\n" + "="*70)
    rep.write(f"\n{summ}\n")

    path = os.path.join(RESULT_DIR, f"baseline_dgm4_official_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f: f.write(rep.getvalue())
    print(f"리포트: {path}")


# ─────────────────────────────────────────────────────────────
# 11. K-Fold 실행 (선택)
# ─────────────────────────────────────────────────────────────
def run_kfold(dfs: dict, device, tok, prep, args):
    """
    [주의] DGM4 공식 split은 뉴스 소스 단위로 설계됨.
    소스 누수가 발생 가능. 성능 비교는 official split 권장.
    """
    set_seed(args.seed)
    df_tv = pd.concat([dfs["train"], dfs["validation"]], ignore_index=True)
    df_te = dfs["test"]
    y_tv  = df_tv["binary_label"].values
    k     = args.kfold_splits
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")

    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (KFold={k}) ######\n")
    rep.write(f"[구조] RoBERTa CLS + CLIP global → concat → MLP → binary\n")
    rep.write(f"[손실] Binary Focal Loss only\n\n")

    el     = _make_dl(df_te, args.data_root, tok, prep, args, shuffle=False)
    skf    = StratifiedKFold(n_splits=k, shuffle=True,
                              random_state=args.kfold_random_state)
    fold_m = []

    for fold, (tr_idx, va_idx) in enumerate(
            skf.split(np.zeros(len(df_tv)), y_tv), 1):
        print(f"\n{'='*70}\n[Fold {fold}/{k}] {LOG_BRAND}\n{'='*70}")
        rep.write(f"\n\n{'#'*70}\n### Fold {fold}/{k}\n{'#'*70}\n")

        df_tr_f = df_tv.iloc[tr_idx].reset_index(drop=True)
        df_va_f = df_tv.iloc[va_idx].reset_index(drop=True)

        tr_ds = DGM4Dataset(df_tr_f, args.data_root, tok, prep, args.max_length)
        tl    = _make_dl(df_tr_f, args.data_root, tok, prep, args,
                         shuffle=True,
                         sampler=make_weighted_sampler(tr_ds.binary_lbls,
                                                        args.sampler_alpha))
        vl    = _make_dl(df_va_f, args.data_root, tok, prep, args, shuffle=False)

        model = _new_model(args, device)
        if not args.no_freeze_encoders:
            freeze_backbones(model)

        log = []
        model, epoch_log = train_one_run(model, tl, vl, device, args, log)
        for line in log: rep.write(line + "\n")

        plot_epoch_curves(
            epoch_log,
            os.path.join(RESULT_DIR, f"baseline_dgm4_curve_fold{fold}_{ts}.png"),
            title=f"{LOG_TAG} · Fold {fold}/{k}")

        full_df = collect_predictions(model, el, device)
        blk, m  = eval_report_block(full_df, label=f"Fold{fold}")
        print(blk); rep.write(blk)

        save_checkpoint(
            os.path.join(args.checkpoint_dir,
                         f"baseline_dgm4_kfold_{ts}_fold{fold}.pt"),
            model, args, m)

        summ = (f"Fold {fold}: Macro-F1={m['bin_f1']:.4f}"
                f"  AUC={m['bin_auc']:.4f}"
                f"  F1_real={m['bin_f1_real']:.4f}"
                f"  F1_fake={m['bin_f1_fake']:.4f}")
        print(f"\n{summ}")
        rep.write(f"\n{summ}\n")
        fold_m.append(m)

        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 전체 요약
    rep.write(f"\n\n{'#'*70}\n### SUMMARY\n{'#'*70}\n")
    print(f"\n{'='*70}\n[{LOG_TAG}] K-Fold 최종 요약\n{'='*70}")
    for k_name in ("bin_acc", "bin_f1", "bin_f1_real", "bin_f1_fake", "bin_auc"):
        vals = [r[k_name] for r in fold_m]
        line = (f"  {k_name:20s}: {float(np.mean(vals)):.4f}"
                f"  ± {float(np.std(vals)):.4f}"
                f"  (min={min(vals):.4f}  max={max(vals):.4f})")
        print(line); rep.write(line + "\n")

    path = os.path.join(RESULT_DIR, f"baseline_dgm4_kfold{k}_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f: f.write(rep.getvalue())
    print(f"\n리포트: {path}")


# ─────────────────────────────────────────────────────────────
# 12. main
# ─────────────────────────────────────────────────────────────
def main():
    global RESULT_DIR, CHECKPOINT_DIR

    pa = argparse.ArgumentParser(
        description="HARFNET-BASELINE-Concat × DGM4 이진탐지")

    # 데이터
    pa.add_argument("--data_root",      default=DEFAULT_DATA_ROOT,
                    help="DGM4 루트 디렉터리 (metadata/ 폴더 포함)")
    pa.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    pa.add_argument("--result_dir",     default=RESULT_DIR)

    # 학습
    pa.add_argument("--batch_size",  type=int,   default=BATCH)
    pa.add_argument("--epochs",      type=int,   default=EPOCHS)
    pa.add_argument("--lr",          type=float, default=LR)
    pa.add_argument("--num_workers", type=int,   default=NUM_WORKERS)
    pa.add_argument("--seed",        type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)

    # 모델
    pa.add_argument("--roberta",    default=ROBERTA)
    pa.add_argument("--clip_model", default=CLIP_RN101)
    pa.add_argument("--max_length", type=int, default=MAX_LENGTH)
    pa.add_argument("--no_freeze_encoders", action="store_true",
                    help="RoBERTa 전체 파인튜닝 (기본: 마지막 2 레이어만)")

    # 샘플러
    pa.add_argument("--sampler_alpha", type=float, default=0.5)

    # 실험 모드
    pa.add_argument("--kfold",              action="store_true",
                    help="K-Fold 실행 (비권장: DGM4 공식 split 사용 권장)")
    pa.add_argument("--kfold_splits",       type=int, default=KFOLD_SPLITS)
    pa.add_argument("--kfold_random_state", type=int, default=KFOLD_RANDOM_STATE)

    pa.add_argument("--no_progress", action="store_true")
    args = pa.parse_args()

    if args.no_progress:
        os.environ["DGM4_NO_TQDM"] = "1"

    # result_dir / checkpoint_dir 갱신 (argparse 기본값 덮어쓰기 대응)
    RESULT_DIR     = args.result_dir
    CHECKPOINT_DIR = args.checkpoint_dir
    os.makedirs(RESULT_DIR,     exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    set_seed(args.seed)
    mode = "KFold (비권장)" if args.kfold else "Official Split (권장)"
    print(f"\n{LOG_BRAND} | device={DEVICE} | mode={mode}")
    print(f"[구조] RoBERTa CLS [B,d] + CLIP global [B,d] → concat [B,2d] → MLP → binary")
    print(f"[손실] Binary Focal Loss only")
    print(f"[CLIP] 완전 frozen (baseline)")
    print(f"[데이터] {args.data_root}\n")

    dfs     = load_dgm4_splits(args.data_root)
    tok     = RobertaTokenizerFast.from_pretrained(args.roberta)
    _, prep = openai_clip.load(args.clip_model, device="cpu")

    if args.kfold:
        run_kfold(dfs, DEVICE, tok, prep, args)
    else:
        run_official(dfs, DEVICE, tok, prep, args)


if __name__ == "__main__":
    main()
