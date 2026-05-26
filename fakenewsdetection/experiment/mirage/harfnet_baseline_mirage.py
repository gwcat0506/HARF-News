"""
HARFNET-BASELINE-Concat → MiRAGeNews 이진탐지
==============================================


레이블 매핑:
  0 = Real
  1 = AI-Fake
"""

from __future__ import annotations

import argparse, copy, gc, io, os, random, sys, warnings
from datetime import datetime
from typing import List, Optional

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import clip as openai_clip
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizerFast

# ─────────────────────────────────────────────────────────────
# 1. 상수
# ─────────────────────────────────────────────────────────────
BATCH               = 16
EPOCHS              = 10
LR                  = 1e-4
NUM_WORKERS         = 4
EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_DELTA= 1e-4
SEED                = 42

ROBERTA    = "roberta-base"
CLIP_RN101 = "RN101"
MAX_LENGTH = 128

HF_DATASET_ID = "anson-huang/mirage-news"

# 테스트 스플릿 후보 (우선순위 순, 실제 키와 교집합으로 결정)
TEST_SPLIT_CANDIDATES = [
    "test_midjourneyv5",
    "test_midjourney_v5",
    "test_midjourneyV5",
    "test_dalle3",
    "test_dalle_3",
    "test_sdxl",
    "test_bbc",
    "test_cnn",
]

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR     = os.path.join(_SCRIPT_DIR, "results")
CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")

if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BINARY_CLASS_NAMES = ["Real", "AI-Fake"]
LOG_TAG  = "HARFNET-BASELINE-Concat-MIRAGE"
LOG_BRAND= "HARFNET-BASELINE-Concat-MIRAGE"

os.makedirs(RESULT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def set_seed(s: int = SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed()


# ─────────────────────────────────────────────────────────────
# 2. 유틸
# ─────────────────────────────────────────────────────────────
def _tqdm(it, **kw):
    if os.environ.get("MIRAGE_NO_TQDM", "").lower() in ("1","true","yes"):
        return it
    kw.setdefault("file", sys.stderr)
    kw.setdefault("dynamic_ncols", True)
    try:
        sys.stdout.flush()
        return tqdm(it, **kw)
    except Exception:
        return it


# ─────────────────────────────────────────────────────────────
# 3. EarlyStopping
# ─────────────────────────────────────────────────────────────
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
            self.best = v; return False
        imp = (v - self.best > self.min_delta if self.mode == "max"
               else self.best - v > self.min_delta)
        if imp:
            self.best = v; self.counter = 0; return False
        self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────
# 4. MiRAGeNewsDataset
# ─────────────────────────────────────────────────────────────
class MiRAGeNewsDataset(Dataset):
    """
    HuggingFace load_dataset() 결과를 직접 래핑.
    컬럼 자동 탐지: caption/text/headline, image/img, label/labels/fake
    """
    def __init__(self,
                 hf_split,
                 tokenizer,
                 clip_preprocess,
                 max_length: int = MAX_LENGTH,
                 indices: Optional[List[int]] = None):
        super().__init__()
        self.tokenizer       = tokenizer
        self.clip_preprocess = clip_preprocess
        self.max_length      = max_length

        cols = hf_split.column_names
        for cand in ("caption", "text", "headline", "final_headline"):
            if cand in cols: self._text_col = cand; break
        else:
            raise ValueError(f"텍스트 컬럼 없음. 보유 컬럼: {cols}")
        for cand in ("image", "img"):
            if cand in cols: self._img_col = cand; break
        else:
            raise ValueError(f"이미지 컬럼 없음. 보유 컬럼: {cols}")
        for cand in ("label", "labels", "fake"):
            if cand in cols: self._lbl_col = cand; break
        else:
            raise ValueError(f"레이블 컬럼 없음. 보유 컬럼: {cols}")

        if indices is not None:
            hf_split = hf_split.select(indices)

        self.hf_split = hf_split
        self.labels   = [int(x) for x in hf_split[self._lbl_col]]

        lc = {0: self.labels.count(0), 1: self.labels.count(1)}
        print(f"[{LOG_TAG}] {self._text_col}/{self._img_col} "
              f"| {len(self.labels)}건 | Real={lc[0]}  AI-Fake={lc[1]}")

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        row  = self.hf_split[idx]
        text = str(row[self._text_col]).strip()

        img = row[self._img_col]
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            try:
                import io as _io
                raw = img.get("bytes") if isinstance(img, dict) else img
                img = Image.open(_io.BytesIO(raw)).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (127, 127, 127))

        enc = self.tokenizer(
            text, max_length=self.max_length,
            padding="max_length", truncation=True,
            return_tensors="pt")

        return {
            "input_ids":        enc["input_ids"].squeeze(0),
            "attention_mask":   enc["attention_mask"].squeeze(0),
            "pixel_values":     self.clip_preprocess(img),
            "clip_text_tokens": openai_clip.tokenize([text], truncate=True)[0],
            "has_image":        torch.tensor(1.0),   # MiRAGeNews는 항상 이미지 존재
            "label":            torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_batch(batch):
    keys = ("input_ids", "attention_mask", "pixel_values",
            "clip_text_tokens", "has_image", "label")
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def make_weighted_sampler(labels: List[int], alpha: float = 0.5):
    cnt = pd.Series(labels).value_counts().to_dict()
    w   = [(1.0 / cnt[l]) ** alpha for l in labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


# ─────────────────────────────────────────────────────────────
# 5. 베이스라인 모델 (이진 분류 버전)
#
#    RoBERTa → T_cls [B, d]  (CLS 토큰)
#    CLIP    → V_cls [B, d]  (global image feature)
#    concat([T_cls, V_cls])  [B, 2d]
#    → MLP → 2-class  ← 기존 4-class에서 2-class로만 변경
# ─────────────────────────────────────────────────────────────
class ConcatBaseline_Binary(nn.Module):
    def __init__(self, roberta_name=ROBERTA, clip_name=CLIP_RN101,
                 dropout=0.1):
        super().__init__()
        d = 768

        # ── 텍스트 인코더 ──
        self.text_encoder = RobertaModel.from_pretrained(
            roberta_name, add_pooling_layer=False)

        # ── 이미지 인코더 (CLIP frozen) ──
        _clip, _ = openai_clip.load(clip_name, device="cpu")
        self.clip_model = _clip.float()
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        self.visual   = self.clip_model.visual
        clip_edim     = self.visual.attnpool.c_proj.out_features
        self.cls_proj = nn.Linear(clip_edim, d)

        # ── 2-class Classifier (4-class → 2-class 변경) ──
        self.classifier = nn.Sequential(
            nn.Linear(d * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )

    def _clip_visual_forward(self, pix, has_image):
        vis = self.visual; B = pix.size(0)
        m4d = has_image.view(B, 1, 1, 1)
        x = vis.relu1(vis.bn1(vis.conv1(pix)))
        x = vis.relu2(vis.bn2(vis.conv2(x)))
        x = vis.relu3(vis.bn3(vis.conv3(x)))
        x = vis.avgpool(x) * m4d
        x = vis.layer1(x); x = vis.layer2(x)
        x = vis.layer3(x); x = vis.layer4(x)
        v_global = vis.attnpool(x)
        V_cls    = self.cls_proj(v_global)
        V_cls    = V_cls * has_image.view(B, 1)
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

        V_cls = self._clip_visual_forward(pixel_values, has_image)

        F_fused = torch.cat([T_cls, V_cls], dim=-1)
        logits2 = torch.nan_to_num(
            self.classifier(F_fused),
            nan=0., posinf=20., neginf=-20.)

        return {"logits2": logits2}

    @classmethod
    def from_harf_checkpoint(cls, ckpt_path, device,
                               roberta_name=ROBERTA, clip_name=CLIP_RN101):
        """
        HARFNET 4-class 베이스라인 체크포인트 → classifier 제외하고 인코더 재사용.
        """
        model  = cls(roberta_name, clip_name).to(device)
        ckpt   = torch.load(ckpt_path, map_location=device)
        state  = ckpt.get("model_state_dict", ckpt)
        # "classifier." 로 시작하는 head 가중치는 제외 (4-class → 2-class 불일치)
        filtered = {k: v for k, v in state.items()
                    if not k.startswith("classifier.")}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[{LOG_TAG}] HARFNET 체크포인트 로드: "
              f"missing={len(missing)}  unexpected={len(unexpected)}")
        if missing:
            print(f"  missing keys (샘플): {missing[:5]}")
        return model


# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
def focal_loss(logits, targets, gamma=2.0):
    n   = logits.size(0)
    n_cls = logits.size(1)
    cnt = torch.bincount(targets, minlength=n_cls).float().clamp(1)
    inv = (n / cnt)
    inv = (inv / inv.sum() * n_cls).clamp(max=20.)
    ce  = F.cross_entropy(logits, targets,
                          weight=inv.to(logits.device), reduction="none")
    pt  = torch.exp(-ce)
    return ((1. - pt) ** gamma * ce).mean()


def total_loss(out, y, has_image, args):
    return focal_loss(out["logits2"], y)


# ─────────────────────────────────────────────────────────────
# 7. 학습 / 평가
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
    tot, ys, ps, n = 0., [], [], 0
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
            pred = out["logits2"].argmax(-1)
            tot += loss.item() * y.size(0)
            ys.extend(y.cpu().tolist())
            ps.extend(pred.cpu().tolist())
            n += y.size(0)

    n = max(n, 1)
    return {
        "loss":     tot / n,
        "acc":      float(accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
        "f1_ai":    float(f1_score(ys, ps, pos_label=1, average="binary", zero_division=0)),
        "f1_real":  float(f1_score(ys, ps, pos_label=0, average="binary", zero_division=0)),
    }


def collect_predictions(model, loader, device):
    model.eval()
    ys, ps, scores = [], [], []
    with torch.no_grad():
        for batch in _tqdm(loader, desc=f"Eval ({LOG_TAG})"):
            out = _fwd(model, batch, device)
            ys.extend(batch["label"].tolist())
            ps.extend(out["logits2"].argmax(-1).cpu().tolist())
            scores.extend(
                torch.softmax(out["logits2"], -1)[:, 1].cpu().tolist())
    return np.array(ys), np.array(ps), np.array(scores)


def eval_report_binary(yt, yp, scores, label=""):
    buf = io.StringIO()
    buf.write(f"\n=== {label} | Real vs AI-Fake ===\n")
    buf.write(classification_report(
        yt, yp, target_names=BINARY_CLASS_NAMES, digits=4, zero_division=0))
    try:    auc = roc_auc_score(yt, scores)
    except: auc = float("nan")
    acc    = accuracy_score(yt, yp)
    f1     = f1_score(yt, yp, average="macro", zero_division=0)
    f1_ai  = f1_score(yt, yp, pos_label=1, average="binary", zero_division=0)
    buf.write(f"Confusion Matrix:\n{confusion_matrix(yt, yp)}\n")
    buf.write(f"[{label}] Acc={acc:.4f}  Macro-F1={f1:.4f}  "
              f"AI-Fake F1={f1_ai:.4f}  AUC={auc:.4f}\n")
    return buf.getvalue(), {"acc": acc, "f1": f1, "f1_ai": f1_ai, "auc": auc}


# ─────────────────────────────────────────────────────────────
# 8. 학습 헬퍼
# ─────────────────────────────────────────────────────────────
def freeze_backbones(model: ConcatBaseline_Binary):
    """RoBERTa 마지막 2개 레이어(10, 11)만 학습"""
    for name, p in model.text_encoder.named_parameters():
        p.requires_grad_(any(f"encoder.layer.{i}" in name for i in (10, 11)))


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
        line = (f"[{LOG_TAG}] Epoch {ep} "
                f"train_loss={tr['loss']:.4f}  val_loss={va['loss']:.4f}  "
                f"val_f1={va['macro_f1']:.4f}  val_acc={va['acc']:.4f}  "
                f"val_f1_ai={va['f1_ai']:.4f}")
        print(line)
        epoch_log.append({
            "epoch":      ep,
            "train_loss": tr["loss"],
            "val_loss":   va["loss"],
            "val_f1":     va["macro_f1"],
        })
        if log is not None:
            log.append(line)

        if best_st is None or (va["macro_f1"] - best_f1) > args.early_stop_min_delta:
            best_f1 = va["macro_f1"]
            best_st = copy.deepcopy(model.state_dict())
        if early.step(va["macro_f1"]):
            msg = f"[{LOG_TAG}] Early stopping at epoch {ep}"
            print(msg)
            if log: log.append(msg)
            break

    if best_st:
        model.load_state_dict(best_st)
    return model, epoch_log


def plot_epoch_curves(epoch_log: list, save_path: str, title: str = ""):
    epochs     = [e["epoch"]      for e in epoch_log]
    train_loss = [e["train_loss"] for e in epoch_log]
    val_loss   = [e["val_loss"]   for e in epoch_log]
    val_f1     = [e["val_f1"]     for e in epoch_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title or LOG_TAG, fontsize=13, fontweight="bold")

    ax1.plot(epochs, train_loss, "o-", color="#2563EB", linewidth=2,
             markersize=5, label="Train Loss")
    ax1.plot(epochs, val_loss, "s--", color="#DC2626", linewidth=2,
             markersize=5, label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Train / Val Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    ax2.plot(epochs, val_f1, "^-", color="#16A34A", linewidth=2,
             markersize=5, label="Val Macro F1")
    best_ep  = epochs[int(np.argmax(val_f1))]
    best_f1v = max(val_f1)
    ax2.axvline(best_ep, color="#16A34A", linestyle=":", alpha=0.6)
    ax2.annotate(f"Best: {best_f1v:.4f}\n(Epoch {best_ep})",
                 xy=(best_ep, best_f1v),
                 xytext=(best_ep + 0.3, best_f1v - 0.01),
                 fontsize=9, color="#15803D",
                 arrowprops=dict(arrowstyle="->", color="#15803D"))
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1")
    ax2.set_title("Val Macro F1")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)

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
    if metrics:
        payload["metrics"] = metrics
    torch.save(payload, path)
    print(f"체크포인트 저장: {path}")


def _make_dl(ds, batch_size, num_workers, shuffle,
             sampler=None, drop_last=False):
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              collate_fn=collate_batch,
              pin_memory=torch.cuda.is_available(),
              drop_last=drop_last)
    if sampler is not None:
        kw["sampler"] = sampler; kw["shuffle"] = False
    else:
        kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


def _new_model(args, device):
    return ConcatBaseline_Binary(
        roberta_name=args.roberta,
        clip_name=args.clip_model).to(device)


# ─────────────────────────────────────────────────────────────
# 9. HuggingFace 데이터셋 로드 + 스플릿 탐지
# ─────────────────────────────────────────────────────────────
def load_hf_splits(hf_id=HF_DATASET_ID):
    print(f"[{LOG_TAG}] HuggingFace 데이터셋 로드: {hf_id}")
    ds = load_dataset(hf_id)
    print(f"[{LOG_TAG}] 실제 스플릿 목록: {list(ds.keys())}")
    return ds


def resolve_test_splits(ds):
    """실제 HF 키와 후보 목록의 교집합으로 테스트 스플릿 결정 (동적 탐지)"""
    actual_keys = set(ds.keys())
    matched     = [k for k in TEST_SPLIT_CANDIDATES if k in actual_keys]
    matched_set = set(matched)
    extra       = sorted([k for k in actual_keys
                          if k.startswith("test") and k not in matched_set])
    result = matched + [e for e in extra if e not in matched_set]
    print(f"[{LOG_TAG}] 탐지된 테스트 스플릿: {result}")
    if not result:
        print(f"[{LOG_TAG}] ⚠️  test 스플릿 없음. 전체 키: {list(actual_keys)}")
    return result


# ─────────────────────────────────────────────────────────────
# 10. Official Split 실행
# ─────────────────────────────────────────────────────────────
def run_official_split(args):
    set_seed(args.seed)
    tok  = RobertaTokenizerFast.from_pretrained(args.roberta)
    _, prep = openai_clip.load(args.clip_model, device="cpu")

    hf_ds = load_hf_splits(args.hf_dataset_id)

    def _wrap(split_name):
        return MiRAGeNewsDataset(hf_ds[split_name], tok, prep, args.max_length)

    tr_ds = _wrap("train")
    sampler = (make_weighted_sampler(tr_ds.labels, args.sampler_alpha)
               if args.use_sampler else None)
    tl = _make_dl(tr_ds, args.batch_size, args.num_workers, True,
                  sampler=sampler, drop_last=True)
    vl = _make_dl(_wrap("validation"), args.batch_size, args.num_workers, False)

    if args.harf_ckpt:
        model = ConcatBaseline_Binary.from_harf_checkpoint(
            args.harf_ckpt, DEVICE, args.roberta, args.clip_model)
    else:
        model = _new_model(args, DEVICE)

    if not args.no_freeze_encoders:
        freeze_backbones(model)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (Official Split) ######\n")
    rep.write(f"모델: RoBERTa CLS + CLIP global → concat → MLP → 2-class\n")
    rep.write(f"손실: Focal Loss only\n\n")

    log = []
    model, epoch_log = train_one_run(model, tl, vl, DEVICE, args, log)
    for line in log: rep.write(line + "\n")

    # 에포크 커브 저장
    plot_path = os.path.join(RESULT_DIR, f"baseline_mirage_curve_{ts}.png")
    plot_epoch_curves(epoch_log, plot_path, title=f"{LOG_TAG} (Official Split)")

    # 테스트셋 평가
    test_splits = resolve_test_splits(hf_ds)
    print(f"\n[{LOG_TAG}] ===== 테스트셋 평가 ({len(test_splits)}개) =====")
    rep.write("\n\n===== TEST RESULTS =====\n")
    all_metrics = {}

    for sname in test_splits:
        el = _make_dl(_wrap(sname), args.batch_size, args.num_workers, False)
        yt, yp, sc = collect_predictions(model, el, DEVICE)
        blk, m = eval_report_binary(yt, yp, sc, label=sname)
        print(blk); rep.write(blk)
        all_metrics[sname] = m

    # 요약
    rep.write("\n===== SUMMARY (avg over test splits) =====\n")
    print(f"\n[{LOG_TAG}] ===== 평균 요약 =====")
    for k in ("acc", "f1", "f1_ai", "auc"):
        vals = [all_metrics[s][k] for s in all_metrics]
        line = (f"  {k}: {np.mean(vals):.4f}"
                f"  (min={min(vals):.4f}  max={max(vals):.4f})")
        rep.write(line + "\n"); print(line)

    save_checkpoint(
        os.path.join(CHECKPOINT_DIR, f"baseline_mirage_single_{ts}.pt"),
        model, args, all_metrics)

    path = os.path.join(RESULT_DIR, f"baseline_mirage_single_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rep.getvalue())
    print(f"\n리포트: {path}")


# ─────────────────────────────────────────────────────────────
# 11. K-Fold 실행
# ─────────────────────────────────────────────────────────────
def run_kfold_split(args):
    set_seed(args.seed)
    tok  = RobertaTokenizerFast.from_pretrained(args.roberta)
    _, prep = openai_clip.load(args.clip_model, device="cpu")

    hf_ds = load_hf_splits(args.hf_dataset_id)
    base_train   = hf_ds["train"]
    train_labels = np.array([int(x) for x in base_train["label"]])
    skf = StratifiedKFold(n_splits=args.kfold_splits,
                          shuffle=True, random_state=args.seed)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (KFold={args.kfold_splits}) ######\n")
    rep.write(f"모델: RoBERTa CLS + CLIP global → concat → MLP → 2-class\n")
    rep.write(f"손실: Focal Loss only\n\n")

    test_splits = resolve_test_splits(hf_ds)
    rep.write(f"test_splits={test_splits}\n\n")

    fold_results = []
    for fold_idx, (tr_idx, va_idx) in enumerate(
            skf.split(np.zeros(len(train_labels)), train_labels), start=1):
        print(f"\n{'='*70}")
        print(f"[Fold {fold_idx}/{args.kfold_splits}] {LOG_BRAND}")
        print(f"{'='*70}")
        rep.write(f"\n\n{'#'*70}\n"
                  f"### Fold {fold_idx}/{args.kfold_splits}\n"
                  f"{'#'*70}\n")

        tr_ds = MiRAGeNewsDataset(
            base_train, tok, prep, args.max_length, indices=tr_idx.tolist())
        va_ds = MiRAGeNewsDataset(
            base_train, tok, prep, args.max_length, indices=va_idx.tolist())

        sampler = (make_weighted_sampler(tr_ds.labels, args.sampler_alpha)
                   if args.use_sampler else None)
        tl = _make_dl(tr_ds, args.batch_size, args.num_workers, True,
                      sampler=sampler, drop_last=True)
        vl = _make_dl(va_ds, args.batch_size, args.num_workers, False)

        if args.harf_ckpt:
            model = ConcatBaseline_Binary.from_harf_checkpoint(
                args.harf_ckpt, DEVICE, args.roberta, args.clip_model)
        else:
            model = _new_model(args, DEVICE)

        if not args.no_freeze_encoders:
            freeze_backbones(model)

        log = []
        model, epoch_log = train_one_run(model, tl, vl, DEVICE, args, log)
        for line in log: rep.write(line + "\n")

        # 에포크 커브
        plot_path = os.path.join(
            RESULT_DIR, f"baseline_mirage_curve_fold{fold_idx}_{ts}.png")
        plot_epoch_curves(
            epoch_log, plot_path,
            title=f"{LOG_TAG} · Fold {fold_idx}/{args.kfold_splits}")

        fold_metrics = {}
        for sname in test_splits:
            test_ds = MiRAGeNewsDataset(
                hf_ds[sname], tok, prep, args.max_length)
            el = _make_dl(test_ds, args.batch_size, args.num_workers, False)
            yt, yp, sc = collect_predictions(model, el, DEVICE)
            blk, m = eval_report_binary(yt, yp, sc, label=sname)
            print(blk); rep.write(blk)
            fold_metrics[sname] = m

        fold_results.append(fold_metrics)
        summ = "  ".join(
            f"{s}: F1={fold_metrics[s]['f1']:.4f}"
            for s in test_splits)
        print(f"[Fold {fold_idx}] {summ}")

        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 전체 요약
    rep.write("\n\n===== KFOLD GLOBAL SUMMARY =====\n")
    global_metrics = {}
    for sname in test_splits:
        global_metrics[sname] = {}
        for k in ("acc", "f1", "f1_ai", "auc"):
            vals = [fr[sname][k] for fr in fold_results]
            global_metrics[sname][k] = float(np.mean(vals))

    for sname in test_splits:
        g = global_metrics[sname]
        line = (f"{sname}: "
                f"Acc={g['acc']:.4f}  Macro-F1={g['f1']:.4f}  "
                f"F1_AI={g['f1_ai']:.4f}  AUC={g['auc']:.4f}")
        print(line); rep.write(line + "\n")

    print("\n===== 전체 평균 (스플릿 간) =====")
    rep.write("\n===== 전체 평균 (스플릿 간) =====\n")
    for k in ("acc", "f1", "f1_ai", "auc"):
        vals = [global_metrics[s][k] for s in test_splits]
        line = f"  {k}: {float(np.mean(vals)):.4f}"
        print(line); rep.write(line + "\n")

    torch.save({
        "config":         vars(args),
        "fold_metrics":   fold_results,
        "global_metrics": global_metrics,
        "saved_at":       datetime.now().isoformat(timespec="seconds"),
    }, os.path.join(CHECKPOINT_DIR,
                    f"baseline_mirage_kfold{args.kfold_splits}_{ts}.pt"))

    path = os.path.join(RESULT_DIR,
                        f"baseline_mirage_kfold{args.kfold_splits}_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rep.getvalue())
    print(f"\n리포트: {path}")


# ─────────────────────────────────────────────────────────────
# 12. main
# ─────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(
        description="HARFNET-BASELINE-Concat → MiRAGeNews 이진탐지")

    # 데이터
    pa.add_argument("--hf_dataset_id",  default=HF_DATASET_ID,
                    help="HuggingFace 데이터셋 ID")
    pa.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    pa.add_argument("--harf_ckpt",     default=None,
                    help="HARFNET 4-class 체크포인트 경로 (인코더 재사용, 선택)")

    # 학습
    pa.add_argument("--batch_size",    type=int,   default=BATCH)
    pa.add_argument("--epochs",        type=int,   default=EPOCHS)
    pa.add_argument("--lr",            type=float, default=LR)
    pa.add_argument("--num_workers",   type=int,   default=NUM_WORKERS)
    pa.add_argument("--seed",          type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)

    # 모델
    pa.add_argument("--roberta",    default=ROBERTA)
    pa.add_argument("--clip_model", default=CLIP_RN101)
    pa.add_argument("--max_length", type=int, default=MAX_LENGTH)
    pa.add_argument("--no_freeze_encoders", action="store_true",
                    help="인코더 전체 파인튜닝 (기본: 마지막 2 레이어만)")

    # 샘플러
    pa.add_argument("--use_sampler",   action="store_true",
                    help="WeightedRandomSampler 사용 (불균형 대응)")
    pa.add_argument("--sampler_alpha", type=float, default=0.5)

    # K-Fold
    pa.add_argument("--no_kfold",     action="store_true",
                    help="Official Split 방식 사용 (기본: KFold)")
    pa.add_argument("--kfold_splits", type=int, default=5)

    pa.add_argument("--no_progress", action="store_true")
    args = pa.parse_args()

    if args.no_progress:
        os.environ["MIRAGE_NO_TQDM"] = "1"

    set_seed(args.seed)

    print(f"\n{LOG_BRAND} | KFold={not args.no_kfold} | device={DEVICE}")
    print(f"[구조] RoBERTa CLS [B,d] + CLIP global [B,d] → concat [B,2d] → MLP → 2-class")
    print(f"[손실] Focal Loss only")
    print(f"[데이터] {args.hf_dataset_id}")
    print(f"[저장] result={RESULT_DIR}")
    print(f"[저장] ckpt  ={CHECKPOINT_DIR}\n")

    if args.no_kfold:
        run_official_split(args)
    else:
        run_kfold_split(args)


if __name__ == "__main__":
    main()