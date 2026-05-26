"""
HARFNET-VER3 × DGM4  —  Ablation: w/o BI + OAM + VM
=======================================================
"""

from __future__ import annotations

import argparse, copy, gc, io, math, os, random, sys, warnings
from datetime import datetime
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import clip as openai_clip
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizerFast
from dgm4_paths import DEFAULT_DATA_ROOT


# ─────────────────────────────────────────────────────────────
# 1. 상수
# ─────────────────────────────────────────────────────────────
BATCH                = 16
EPOCHS               = 10
LR                   = 2e-4
NUM_WORKERS          = 4
EARLY_STOP_PATIENCE  = 4
EARLY_STOP_MIN_DELTA = 1e-4
WARMUP_EPOCHS        = 2
KFOLD_SPLITS         = 5
KFOLD_RANDOM_STATE   = 42
SEED                 = 42

LAMBDA_BINARY_BCE = 0.40
LAMBDA_FINE_BCE   = 0.00   # OAM 없음
LAMBDA_CON        = 0.10   # z_auth만 사용
LAMBDA_OA         = 0.00   # OAM 없음

LR_MULT_TEXT = 0.05   # RoBERTa 상위 레이어
LR_MULT_NEW  = 1.00   # AM + head (from scratch)
# LR_MULT_VISUAL 불필요 — CLIP visual 순전파 미사용

PROJ_DIM    = 128
TEMPERATURE = 0.07

ROBERTA    = "roberta-base"
CLIP_RN101 = "RN101"
MAX_LENGTH = 128

BINARY_NAMES = ["real", "fake"]
FINE_LABELS  = ["face_swap", "face_attribute", "text_swap", "text_attribute"]
FINE_N       = len(FINE_LABELS)


def parse_fake_cls(cls_str: str) -> Tuple[int, List[int]]:
    cs = cls_str.strip().lower()
    if cs == "orig":
        return 0, [0, 0, 0, 0]
    parts = cs.split("&")
    fine  = [int(lbl in parts) for lbl in FINE_LABELS]
    return 1, fine


LOG_TAG   = "HARFNET-DGM4-woBIOAMV"
LOG_BRAND = "HARFNET-DGM4-woBIOAMV"

if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(s: int = SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed()


# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
def resolve_image_path(rel: str, data_root: str) -> str:
    rel = (rel or "").strip()
    if not rel: return ""
    if os.path.isabs(rel) and os.path.isfile(rel): return rel
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
        if c and os.path.isfile(c): return c
    return candidates[-1] if candidates else ""


def _tqdm(it, **kw):
    if os.environ.get("DGM4_NO_TQDM", "").lower() in ("1", "true", "yes"):
        return it
    kw.setdefault("file", sys.stderr); kw.setdefault("dynamic_ncols", True)
    try: sys.stdout.flush(); return tqdm(it, **kw)
    except Exception: return it


def load_dgm4_splits(data_root: str) -> Dict[str, pd.DataFrame]:
    split_files = {
        "train":      os.path.join(data_root, "metadata", "train.json"),
        "validation": os.path.join(data_root, "metadata", "val.json"),
        "test":       os.path.join(data_root, "metadata", "test.json"),
    }
    dfs = {}
    for split, path in split_files.items():
        if os.path.isfile(path):
            print(f"[{LOG_BRAND}] 로컬 로드: {path}")
            dfs[split] = pd.read_json(path)
        else:
            print(f"[{LOG_BRAND}] HuggingFace에서 로드: {split}")
            hf = {"train": "metadata/train.json",
                  "validation": "metadata/val.json",
                  "test": "metadata/test.json"}
            dfs[split] = pd.read_json(
                "hf://datasets/rshaojimmy/DGM4/" + hf[split])
    for split, df in dfs.items():
        df["text"]     = df["text"].fillna("").astype(str).str.strip()
        df["image"]    = df["image"].fillna("").astype(str).str.strip()
        df["fake_cls"] = df["fake_cls"].fillna("orig").astype(str).str.strip()
        def _has(rel):
            full = resolve_image_path(rel, data_root)
            return bool(full) and os.path.isfile(full)
        df["has_image_flag"] = df["image"].map(_has)
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
# ─────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE,
                 min_delta=EARLY_STOP_MIN_DELTA, mode="max"):
        self.patience=patience; self.min_delta=min_delta
        self.mode=mode; self.best=None; self.counter=0

    def step(self, v: float) -> bool:
        if self.best is None: self.best=v; return False
        imp = (v-self.best>self.min_delta if self.mode=="max"
               else self.best-v>self.min_delta)
        if imp: self.best=v; self.counter=0; return False
        self.counter+=1
        return self.counter>=self.patience


# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
class DGM4Dataset(Dataset):
    def __init__(self, df, data_root, tokenizer, clip_preprocess,
                 max_length=MAX_LENGTH, require_image=False):
        super().__init__()
        self.data_root       = os.path.abspath(data_root)
        self.tokenizer       = tokenizer
        self.clip_preprocess = clip_preprocess
        self.max_length      = max_length
        if require_image:
            df = df[df["has_image_flag"]].reset_index(drop=True)
        self.texts        = df["text"].tolist()
        self.image_rels   = df["image"].tolist()
        self.has_img_flag = df["has_image_flag"].tolist()
        self.binary_lbls  = df["binary_label"].tolist()
        self.fake_cls_str = df["fake_cls"].tolist()
        fine_cols         = [f"fine_{lbl}" for lbl in FINE_LABELS]
        self.fine_lbls    = df[fine_cols].values.tolist()

    def __len__(self): return len(self.binary_lbls)

    def _load_pil(self, rel, flag):
        if not flag or not rel:
            return Image.new("RGB", (224, 224), (127, 127, 127)), False
        full = resolve_image_path(rel, self.data_root)
        if not full or not os.path.isfile(full):
            return Image.new("RGB", (224, 224), (127, 127, 127)), False
        try:    return Image.open(full).convert("RGB"), True
        except: return Image.new("RGB", (224, 224), (127, 127, 127)), False

    def __getitem__(self, idx):
        text    = self.texts[idx]
        img, ok = self._load_pil(self.image_rels[idx], self.has_img_flag[idx])
        enc = self.tokenizer(
            text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids":        enc["input_ids"].squeeze(0),
            "attention_mask":   enc["attention_mask"].squeeze(0),
            "pixel_values":     self.clip_preprocess(img),   # 배치 형상 통일용
            "clip_text_tokens": openai_clip.tokenize([text], truncate=True)[0],
            "has_image":        torch.tensor(1.0 if ok else 0.0),
            "binary_label":     torch.tensor(self.binary_lbls[idx], dtype=torch.long),
            "fine_labels":      torch.tensor(self.fine_lbls[idx], dtype=torch.float),
            "text":             text,
            "image_path":       self.image_rels[idx],
            "fake_cls":         self.fake_cls_str[idx],
        }


def collate_batch(batch):
    tensor_keys = ("input_ids", "attention_mask", "pixel_values",
                   "clip_text_tokens", "has_image",
                   "binary_label", "fine_labels")
    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out["texts"]       = [b["text"]       for b in batch]
    out["image_paths"] = [b["image_path"] for b in batch]
    out["fake_cls"]    = [b["fake_cls"]   for b in batch]
    return out


def make_weighted_sampler(binary_labels: List[int], alpha: float = 0.5):
    cnt = pd.Series(binary_labels).value_counts().to_dict()
    w   = [(1.0 / cnt[l]) ** alpha for l in binary_labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


# ─────────────────────────────────────────────────────────────
# 5. 서브모듈
#    ✗ VeracityModule   (VM) 제거
#    ✗ OverAlignModule  (OA) 제거
#    ✗ BilinearInteraction (BI) 제거
# ─────────────────────────────────────────────────────────────
class GatedPooling(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.score = nn.Linear(dim, 1)
    def forward(self, x, mask=None):
        s = self.score(x).squeeze(-1)
        if mask is not None: s = s.masked_fill(~mask, -1e4)
        return (torch.softmax(s, dim=-1).unsqueeze(-1) * x).sum(1)

class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, d_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_hid, d_out))
    def forward(self, x): return self.net(x)

class AuthorshipModule(nn.Module):
    """..."""
    def __init__(self, d, n_heads=8, dropout=0.1):
        super().__init__()
        self.style_self_attn = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True)
        self.style_var_proj  = nn.Sequential(
            nn.Linear(d, d//2), nn.GELU(), nn.Linear(d//2, d))
        self.style_pool      = GatedPooling(d)
        self.style_ffn       = FeedForward(d*2, d*2, d, dropout)
        self.style_ln        = nn.LayerNorm(d)
        self.auth_probe      = nn.Sequential(
            nn.Linear(d, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 1), nn.Sigmoid())
        self.auth_proj       = nn.Sequential(
            nn.Linear(d, d//2), nn.ReLU(), nn.Linear(d//2, PROJ_DIM))

    def forward(self, T_tok, txt_mask):
        T_sty, _ = self.style_self_attn(T_tok, T_tok, T_tok, key_padding_mask=~txt_mask)
        sr = T_tok - T_sty
        sv = self.style_var_proj(sr.var(dim=1).clamp(0.))
        sp = self.style_pool(sr, txt_mask)
        F_sty  = self.style_ln(self.style_ffn(torch.cat([sp, sv], dim=-1)))
        p_auth = self.auth_probe(F_sty).squeeze(-1)
        z_auth = F.normalize(self.auth_proj(F_sty), dim=-1)
        return F_sty, p_auth, z_auth


# ─────────────────────────────────────────────────────────────
# 6. 메인 모델 — w/o BI + OAM + VM  (AM Only)
#
#  ┌────────────────────────────────────────────────────────┐
#  │ 원본    F_sty + F_ver + F_oa + F_bi  →  d×4 → head    │
#  │ 본파일  F_sty + zeros + zeros + zeros →  d×4 → head   │
#  └────────────────────────────────────────────────────────┘
#
#  CLIP은 로드하되 visual forward 호출 안 함
#  (pixel_values는 DataLoader 통일용으로 수신만)
# ─────────────────────────────────────────────────────────────
class HARFNETdgm4_woBIOAMV(nn.Module):
    def __init__(self, roberta_name=ROBERTA,
                 clip_name=CLIP_RN101, n_heads=8, dropout=0.1):
        super().__init__()
        d = 768
        self._d = d

        # ── 텍스트 인코더 ──
        self.text_encoder = RobertaModel.from_pretrained(
            roberta_name, add_pooling_layer=False)

        # ── CLIP (완전 frozen, visual forward 미사용) ──
        _clip, _ = openai_clip.load(clip_name, device="cpu")
        self.clip_model = _clip.float()
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        # ── AM만 유지 (VM/OAM/BI 없음) ──
        self.am = AuthorshipModule(d, n_heads, dropout)

        # ── binary head: d×4 입력 유지 (공정 비교) ──
        self.binary_head = nn.Sequential(
            nn.Linear(d * 4, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        return self

    def forward(self, input_ids, attention_mask,
                pixel_values, has_image, clip_text_tokens=None):
        # pixel_values / has_image: 배치 통일용으로 받되 사용 안 함
        device    = input_ids.device
        B         = input_ids.size(0)

        # ── 텍스트 인코딩 ──
        T_tok    = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask).last_hidden_state
        txt_mask = attention_mask.bool()

        # ── AM: 문체 모듈 ──
        F_sty, p_auth, z_auth = self.am(T_tok, txt_mask)

        # ── VM 없음 → F_ver / p_fake / z_ver / gsim = zeros ──
        F_ver  = torch.zeros(B, self._d, device=device)
        p_fake = torch.zeros(B, device=device)
        z_ver  = torch.zeros(B, PROJ_DIM, device=device)
        gsim   = torch.zeros(B, device=device)

        # ── OAM 없음 → F_oa / p_fine = zeros ──
        F_oa   = torch.zeros(B, self._d, device=device)
        p_fine = torch.zeros(B, FINE_N, device=device)

        # ── BI 없음 → F_bi = zeros ──
        F_bi   = torch.zeros(B, self._d, device=device)

        # ── 분류 헤드: 차원 원본 유지 (d×4) ──
        feat         = torch.cat([F_sty, F_ver, F_oa, F_bi], dim=-1)
        binary_logit = torch.nan_to_num(
            self.binary_head(feat).squeeze(-1),
            nan=0., posinf=20., neginf=-20.)

        return {
            "binary_logit":  binary_logit,
            "p_fine":        p_fine,
            "p_auth":        p_auth,
            "p_fake":        p_fake,
            "z_auth":        z_auth,
            "z_ver":         z_ver,          # zeros — 손실에서 미사용
            "global_sim":    gsim,
            "oa_mean":       torch.zeros(B, device=device),
            "oa_std":        torch.zeros(B, device=device),
            "oa_uniformity": torch.zeros(B, device=device),
        }


# ─────────────────────────────────────────────────────────────
# 7. 손실 함수
#    ✗ fine_label_loss  (OAM 없음)
#    ✗ oa_reg_loss      (OAM 없음)
#    ✗ supcon(z_ver)    (VM 없음 → z_ver = zeros, 의미 없음)
#    ✓ binary_focal_loss
#    ✓ supcon(z_auth)   (AM 유지)
# ─────────────────────────────────────────────────────────────
def binary_focal_loss(logit, target, gamma=2.0, pos_weight=1.0):
    pw  = torch.tensor([pos_weight], device=logit.device)
    bce = F.binary_cross_entropy_with_logits(
        logit, target.float(), pos_weight=pw, reduction="none")
    prob = torch.sigmoid(logit)
    pt   = torch.where(target == 1, prob, 1 - prob)
    return ((1 - pt) ** gamma * bce).mean()


def supcon_loss(z, labels, temperature=TEMPERATURE):
    B = z.size(0); device = z.device
    if B < 2: return z.sum() * 0.
    same   = (labels.unsqueeze(0) == labels.unsqueeze(1))
    self_m = torch.eye(B, dtype=torch.bool, device=device)
    pos_m  = same & ~self_m
    if pos_m.sum() == 0: return z.sum() * 0.
    sim   = torch.matmul(z, z.T) / temperature
    sim_s = sim - sim.detach().max(1, keepdim=True).values
    exp_s = torch.exp(sim_s).masked_fill(self_m, 0.)
    log_p = sim_s - torch.log(exp_s.sum(1, keepdim=True).clamp(1e-8))
    n_pos = pos_m.sum(1).clamp(1).float()
    return (-(log_p * pos_m).sum(1) / n_pos).mean()


def total_loss(out, binary_label, fine_labels, has_image, args):
    """
    focal(binary) + λ_con·supcon(z_auth)
    — z_ver = zeros이므로 supcon(z_ver) 제거 (학습 노이즈 방지)
    """
    l_bin = binary_focal_loss(out["binary_logit"], binary_label)
    l_con = supcon_loss(out["z_auth"], binary_label)
    return LAMBDA_BINARY_BCE * l_bin + args.lambda_con * l_con


# ─────────────────────────────────────────────────────────────
# 8. forward 헬퍼
# ─────────────────────────────────────────────────────────────
def _fwd(model, batch, device):
    kw = {}
    if "clip_text_tokens" in batch:
        kw["clip_text_tokens"] = batch["clip_text_tokens"].to(device)
    return model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["pixel_values"].to(device),
        batch["has_image"].to(device), **kw)


# ─────────────────────────────────────────────────────────────
# 9. 옵티마이저 & 스케줄러
#    VM 없음 → grp_visual 제거 (CLIP visual 순전파 미사용)
#    2-group: backbone_text + new_modules
# ─────────────────────────────────────────────────────────────
def make_optimizer(model: HARFNETdgm4_woBIOAMV, args):
    """
    2-group 차등 LR
    ─ backbone_text : RoBERTa 학습 가능 레이어 → lr × LR_MULT_TEXT
    ─ new_modules   : AM + binary_head          → lr × LR_MULT_NEW
    """
    grp_text, grp_new = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if "text_encoder" in name:
            grp_text.append(p)
        else:
            grp_new.append(p)
    return torch.optim.AdamW([
        {"params": grp_text, "lr": args.lr * LR_MULT_TEXT, "weight_decay": 1e-4},
        {"params": grp_new,  "lr": args.lr * LR_MULT_NEW,  "weight_decay": 1e-4},
    ])


def make_scheduler(optimizer, args):
    warmup = args.warmup_epochs
    total  = args.epochs
    def lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(max(warmup, 1))
        progress = float(ep - warmup) / float(max(total - warmup, 1))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────
# 10. 백본 설정
#     RoBERTa layer 10, 11만 학습
# ─────────────────────────────────────────────────────────────
def configure_backbones(model: HARFNETdgm4_woBIOAMV,
                        text_train_layers: Tuple[int, ...] = (10, 11)):
    # RoBERTa: 상위 2개 레이어만
    for name, p in model.text_encoder.named_parameters():
        trainable = any(f"encoder.layer.{i}" in name
                        for i in text_train_layers)
        p.requires_grad_(trainable)
    # CLIP: VM 없음 → 완전 frozen 유지
    for p in model.clip_model.parameters():
        p.requires_grad_(False)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{LOG_TAG}] 학습 파라미터: {n_train:,} / {n_total:,}"
          f" ({100 * n_train / n_total:.1f}%)")


# ─────────────────────────────────────────────────────────────
# 11. 학습 epoch
# ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, device, optimizer, train,
              epoch_idx=None, args=None, scheduler=None):
    model.train() if train else model.eval()
    tot, ys_bin, ps_bin, probs_bin, n = 0., [], [], [], 0
    ctx  = torch.enable_grad() if train else torch.no_grad()
    desc = f"[{LOG_TAG}] Epoch {epoch_idx}" if (train and epoch_idx) else None
    it   = _tqdm(loader, desc=desc) if (train and desc) else loader

    with ctx:
        for batch in it:
            y_bin  = batch["binary_label"].to(device)
            y_fine = batch["fine_labels"].to(device)
            hi     = batch["has_image"].to(device)
            out    = _fwd(model, batch, device)
            loss   = total_loss(out, y_bin, y_fine, hi, args)
            if train and optimizer:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            prob     = torch.sigmoid(out["binary_logit"]).detach()
            pred_bin = (prob > 0.5).long()
            tot      += loss.item() * y_bin.size(0)
            ys_bin.extend(y_bin.cpu().tolist())
            ps_bin.extend(pred_bin.cpu().tolist())
            probs_bin.extend(prob.cpu().tolist())
            n += y_bin.size(0)

    if train and scheduler is not None:
        scheduler.step()

    n = max(n, 1)
    try:    auc = float(roc_auc_score(ys_bin, probs_bin))
    except: auc = float("nan")
    return {
        "loss":    tot / n,
        "bin_acc": float(accuracy_score(ys_bin, ps_bin)),
        "bin_f1":  float(f1_score(ys_bin, ps_bin, average="macro", zero_division=0)),
        "bin_auc": auc,
    }


# ─────────────────────────────────────────────────────────────
# 12. eval
# ─────────────────────────────────────────────────────────────
def collect_predictions(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in _tqdm(loader, desc=f"[{LOG_TAG}] 예측 수집"):
            out      = _fwd(model, batch, device)
            bin_prob = torch.sigmoid(out["binary_logit"])
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
                    "p_auth":         round(float(out["p_auth"][i]), 4),
                    "p_fake":         round(float(out["p_fake"][i]), 4),   # zeros
                    "oa_mean":        round(float(out["oa_mean"][i]), 4),  # zeros
                    "oa_std":         round(float(out["oa_std"][i]), 4),   # zeros
                    "oa_uniformity":  round(float(out["oa_uniformity"][i]), 4),  # zeros
                    "global_sim":     round(float(out["global_sim"][i]), 4),     # zeros
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


def extract_case_study(df: pd.DataFrame, n: int = 3) -> dict:
    results   = {}
    BASE_COLS = ["text", "image_path", "fake_cls",
                 "binary_label", "binary_pred", "binary_prob",
                 "p_auth", "p_fake", "oa_mean", "oa_uniformity", "global_sim"]
    success = {}
    for cls in ["orig"] + list(set(df["fake_cls"].unique()) - {"orig"}):
        mask = (df["fake_cls"] == cls) & df["binary_correct"]
        sub  = (df[mask].nlargest(n, "binary_prob") if cls != "orig"
                else df[mask].nsmallest(n, "binary_prob"))
        success[cls] = sub[BASE_COLS]
    results["success_per_class"] = success
    results["fp_cases"] = (
        df[(df["binary_label"] == 0) & (df["binary_pred"] == 1)]
        .nlargest(n * 2, "binary_prob")[BASE_COLS])
    results["fn_cases"] = (
        df[(df["binary_label"] == 1) & (df["binary_pred"] == 0)]
        .nsmallest(n * 2, "binary_prob")[BASE_COLS])
    sig_cols = ["binary_prob", "p_auth", "p_fake",
                "oa_mean", "oa_uniformity", "global_sim"]
    rows = []
    for cls in sorted(df["fake_cls"].unique()):
        sub = df[df["fake_cls"] == cls]
        row = {"fake_cls": cls, "n": len(sub),
               "bin_acc": round(sub["binary_correct"].mean(), 4)}
        for col in sig_cols:
            row[f"{col}_mean"] = round(sub[col].mean(), 4)
            row[f"{col}_std"]  = round(sub[col].std(),  4)
        rows.append(row)
    results["signal_summary"] = pd.DataFrame(rows)
    return results


def save_case_study_report(cases, full_df, metrics, tag, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(result_dir, f"{tag}_{ts}")
    full_df.to_csv(f"{prefix}_all_predictions.csv", index=False, encoding="utf-8")
    cases["signal_summary"].to_csv(
        f"{prefix}_signal_summary.csv", index=False, encoding="utf-8")
    buf = io.StringIO()
    buf.write(f"{'='*80}\n  {LOG_TAG} Case Study — {tag}\n{'='*80}\n\n")
    buf.write("── [지표 요약] ──\n")
    for k, v in metrics.items():
        buf.write(f"  {k:30s}: {v:.4f}\n")
    report_path = f"{prefix}_case_study_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"\n[CaseStudy] 저장 완료: {report_path}")
    return report_path


def eval_report_block(df, label=LOG_TAG):
    yt = df["binary_label"].values
    yp = df["binary_pred"].values
    buf = io.StringIO()
    buf.write(f"\n=== {label} — Binary Detection ===\n")
    buf.write(classification_report(yt, yp, target_names=["real", "fake"],
                                    digits=4, zero_division=0))
    m = compute_metrics(df)
    buf.write(f"\n[{label}] Binary F1={m['bin_f1']:.4f}"
              f"  AUC={m.get('bin_auc', float('nan')):.4f}\n")
    return buf.getvalue(), m


# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
def train_one_run(model, tl, vl, device, args, log=None):
    opt   = make_optimizer(model, args)
    sched = make_scheduler(opt, args)
    early = EarlyStopping(args.early_stop_patience,
                          args.early_stop_min_delta, mode="max")
    best_auc, best_st = -1., None

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, tl, device, opt, True,
                       epoch_idx=ep, args=args, scheduler=sched)
        va = run_epoch(model, vl, device, None, False, args=args)
        cur_lr = (sched.get_last_lr()[-1] if hasattr(sched, "get_last_lr")
                  else opt.param_groups[-1]["lr"])
        line = (f"[{LOG_TAG}] Ep {ep:02d}"
                f"  tr_loss={tr['loss']:.4f}"
                f"  va_f1={va['bin_f1']:.4f}"
                f"  va_auc={va['bin_auc']:.4f}"
                f"  va_acc={va['bin_acc']:.4f}"
                f"  lr={cur_lr:.2e}")
        print(line)
        if log: log.append(line)

        if not math.isnan(va["bin_auc"]) and va["bin_auc"] > best_auc:
            best_auc = va["bin_auc"]
            best_st  = copy.deepcopy(model.state_dict())
        if early.step(va["bin_auc"] if not math.isnan(va["bin_auc"])
                      else va["bin_f1"]):
            msg = (f"[{LOG_TAG}] Early stopping at epoch {ep}"
                   f" (best_auc={best_auc:.4f})")
            print(msg)
            if log: log.append(msg)
            break

    if best_st: model.load_state_dict(best_st)
    return model


# ─────────────────────────────────────────────────────────────
# 14. DataLoader 헬퍼
# ─────────────────────────────────────────────────────────────
def _make_dl(df, data_root, tok, prep, args, shuffle, sampler=None):
    ds = DGM4Dataset(df, data_root, tok, prep, args.max_length)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              collate_fn=collate_batch,
              pin_memory=torch.cuda.is_available(), drop_last=shuffle)
    if sampler: kw["sampler"] = sampler; kw["shuffle"] = False
    else:       kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


def save_checkpoint(path, model, args, metrics=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model_state_dict": model.state_dict(),
               "config":           vars(args),
               "saved_at":         datetime.now().isoformat(timespec="seconds")}
    if metrics: payload["metrics"] = metrics
    torch.save(payload, path)
    print(f"체크포인트 저장: {path}")


# ─────────────────────────────────────────────────────────────
# 15. run_official / run_kfold
# ─────────────────────────────────────────────────────────────
def run_official(dfs, device, tok, prep, args, configure, result_dir):
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (Official Split) ######\n")
    rep.write(f"[제거] BI + OAM + VM  [유지] AM\n")
    rep.write(f"[head] d×4 → 256 → 1  (F_sty + zeros×3, 차원 원본 유지)\n")
    rep.write(f"[손실] Focal + λ_con·supcon(z_auth)\n\n")

    tr_ds = DGM4Dataset(dfs["train"], args.data_root, tok, prep, args.max_length)
    tl    = _make_dl(dfs["train"], args.data_root, tok, prep, args,
                     shuffle=True,
                     sampler=make_weighted_sampler(tr_ds.binary_lbls, args.sampler_alpha))
    vl    = _make_dl(dfs["validation"], args.data_root, tok, prep, args, shuffle=False)
    el    = _make_dl(dfs["test"],       args.data_root, tok, prep, args, shuffle=False)

    model = HARFNETdgm4_woBIOAMV(args.roberta, args.clip_model).to(device)
    if configure:
        configure_backbones(model)

    log = []
    model = train_one_run(model, tl, vl, device, args, log)
    for line in log: rep.write(line + "\n")

    print(f"\n[{LOG_TAG}] 테스트 예측 수집...")
    full_df = collect_predictions(model, el, device)
    blk, m  = eval_report_block(full_df)
    print(blk, end=""); rep.write(blk)

    if args.case_study_n > 0:
        cases = extract_case_study(full_df, args.case_study_n)
        save_case_study_report(cases, full_df, m, "woBIOAMV_official", result_dir)

    save_checkpoint(
        os.path.join(args.checkpoint_dir, f"harfnet_dgm4_woBIOAMV_{ts}.pt"),
        model, args, m)
    summ = (f"{LOG_TAG}: BinF1={m['bin_f1']:.4f}"
            f"  AUC={m.get('bin_auc', float('nan')):.4f}")
    print("\n" + "="*70 + "\n" + summ + "\n" + "="*70)
    rep.write(f"\n{summ}\n")
    path = os.path.join(result_dir, f"harfnet_dgm4_woBIOAMV_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f: f.write(rep.getvalue())
    print(f"리포트: {path}")


def run_kfold(dfs, device, tok, prep, args, configure, result_dir):
    df_tv = pd.concat([dfs["train"], dfs["validation"]], ignore_index=True)
    df_te = dfs["test"]
    y_tv  = df_tv["binary_label"].values
    k     = args.kfold_splits
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rep = io.StringIO()
    all_rep.write(f"###### {LOG_TAG} (k={k} KFold) ######\n")
    el     = _make_dl(df_te, args.data_root, tok, prep, args, shuffle=False)
    skf    = StratifiedKFold(n_splits=k, shuffle=True,
                              random_state=args.kfold_random_state)
    fold_m = []
    for fold, (tr_idx, va_idx) in enumerate(
            skf.split(np.zeros(len(df_tv)), y_tv), 1):
        print(f"\n{'='*70}\n[Fold {fold}/{k}] {LOG_BRAND}\n{'='*70}")
        df_tr_f = df_tv.iloc[tr_idx].reset_index(drop=True)
        df_va_f = df_tv.iloc[va_idx].reset_index(drop=True)
        tr_ds   = DGM4Dataset(df_tr_f, args.data_root, tok, prep, args.max_length)
        tl      = _make_dl(df_tr_f, args.data_root, tok, prep, args,
                           shuffle=True,
                           sampler=make_weighted_sampler(tr_ds.binary_lbls,
                                                          args.sampler_alpha))
        vl      = _make_dl(df_va_f, args.data_root, tok, prep, args, shuffle=False)
        model   = HARFNETdgm4_woBIOAMV(args.roberta, args.clip_model).to(device)
        if configure: configure_backbones(model)
        log     = []
        model   = train_one_run(model, tl, vl, device, args, log)
        full_df = collect_predictions(model, el, device)
        blk, m  = eval_report_block(full_df, label=f"Fold{fold}")
        print(blk, end="")
        save_checkpoint(
            os.path.join(args.checkpoint_dir,
                         f"harfnet_dgm4_woBIOAMV_kfold_{ts}_fold{fold}.pt"),
            model, args, m)
        all_rep.write(f"\n{'#'*80}\n### FOLD {fold}\n{'#'*80}\n{blk}")
        fold_m.append(m)
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _mean(k_): return float(np.mean([r[k_] for r in fold_m]))
    g = (f"{LOG_TAG}: BinF1={_mean('bin_f1'):.4f}"
         f"  AUC={_mean('bin_auc'):.4f}")
    all_rep.write(f"\n\n{'#'*80}\n### SUMMARY\n{g}\n")
    path = os.path.join(result_dir, f"harfnet_dgm4_woBIOAMV_kfold_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f: f.write(all_rep.getvalue())
    print(f"\n리포트: {path}\n{g}")


# ─────────────────────────────────────────────────────────────
# 16. main
# ─────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(
        description="HARFNET-VER3 × DGM4 — Ablation: w/o BI+OAM+VM (AM Only)")
    pa.add_argument("--data_root",      default=DEFAULT_DATA_ROOT)
    pa.add_argument("--checkpoint_dir", default="./checkpoints")
    pa.add_argument("--result_dir",     default="./results")
    pa.add_argument("--batch_size",    type=int,   default=BATCH)
    pa.add_argument("--epochs",        type=int,   default=EPOCHS)
    pa.add_argument("--lr",            type=float, default=LR)
    pa.add_argument("--warmup_epochs", type=int,   default=WARMUP_EPOCHS)
    pa.add_argument("--num_workers",   type=int,   default=NUM_WORKERS)
    pa.add_argument("--seed",          type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    pa.add_argument("--kfold",              action="store_true")
    pa.add_argument("--kfold_splits",       type=int, default=KFOLD_SPLITS)
    pa.add_argument("--kfold_random_state", type=int, default=KFOLD_RANDOM_STATE)
    pa.add_argument("--roberta",    default=ROBERTA)
    pa.add_argument("--clip_model", default=CLIP_RN101)
    pa.add_argument("--max_length", type=int, default=MAX_LENGTH)
    pa.add_argument("--sampler_alpha",  type=float, default=0.5)
    pa.add_argument("--lambda_fine_bce", type=float, default=LAMBDA_FINE_BCE)
    pa.add_argument("--lambda_con",      type=float, default=LAMBDA_CON)
    pa.add_argument("--lambda_oa",       type=float, default=LAMBDA_OA)
    pa.add_argument("--no_configure_backbones", action="store_true")
    pa.add_argument("--no_progress",  action="store_true")
    pa.add_argument("--case_study_n", type=int, default=3)
    args = pa.parse_args()

    if args.no_progress:
        os.environ["DGM4_NO_TQDM"] = "1"

    configure = not args.no_configure_backbones
    set_seed(args.seed)
    os.makedirs(args.result_dir,     exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    mode = "KFold" if args.kfold else "Official Split"
    print(f"\n{LOG_BRAND} | device={DEVICE} | mode={mode}")
    print(f"[Ablation] w/o BI  w/o OAM  w/o VM  →  AM Only")
    print(f"[head]     d×4 → 256 → 1  (F_sty + zeros×3, 차원 원본 유지)")
    print(f"[손실]     Focal + {args.lambda_con}·supcon(z_auth)")
    print(f"           (supcon(z_ver) 제거 — z_ver=zeros 이므로)")
    print(f"[CLIP]     완전 frozen (VM 없어 visual 미사용)")
    print(f"  lr={args.lr:.2e}  warmup={args.warmup_epochs}ep"
          f"  epochs={args.epochs}\n")

    dfs     = load_dgm4_splits(args.data_root)
    tok     = RobertaTokenizerFast.from_pretrained(args.roberta)
    _, prep = openai_clip.load(args.clip_model, device="cpu")

    if args.kfold:
        run_kfold(dfs, DEVICE, tok, prep, args, configure, args.result_dir)
    else:
        run_official(dfs, DEVICE, tok, prep, args, configure, args.result_dir)


if __name__ == "__main__":
    main()
