"""
HARFNET-VER3-WO-V × DGM4 벤치마크
=====================================

사용 방법:

  # 깨끗한 로그 (터미널=tqdm, 파일=epoch 요약만):

  # K-Fold (비권장):
"""

from __future__ import annotations

import argparse, copy, gc, io, math, os, random, sys, warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import clip as openai_clip
import matplotlib
matplotlib.use("Agg")          # 헤드리스 환경에서도 PNG 저장 가능
import matplotlib.pyplot as plt
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
LAMBDA_FINE_BCE   = 0.30
LAMBDA_CON        = 0.10   # z_auth만 남음
LAMBDA_OA         = 0.20

LR_MULT_TEXT   = 0.05
LR_MULT_VISUAL = 0.02
LR_MULT_NEW    = 1.00

BILINEAR_RANK = 128
PROJ_DIM      = 128
TEMPERATURE   = 0.07

ROBERTA    = "roberta-base"
CLIP_RN101 = "RN101"
MAX_LENGTH = 128

BINARY_NAMES = ["real", "fake"]
FINE_LABELS  = ["face_swap", "face_attribute", "text_swap", "text_attribute"]
FINE_N       = len(FINE_LABELS)

CLIP_UNFREEZE_KEYS = ("layer4", "attnpool")

LOG_TAG   = "HARFNET-DGM4-WOV"
LOG_BRAND = "HARFNET-DGM4-WOV"


def parse_fake_cls(cls_str: str) -> Tuple[int, List[int]]:
    cs = cls_str.strip().lower()
    if cs == "orig":
        return 0, [0, 0, 0, 0]
    parts = cs.split("&")
    fine  = [int(lbl in parts) for lbl in FINE_LABELS]
    return 1, fine


if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(s: int = SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed()


# ─────────────────────────────────────────────────────────────
# 2. 유틸
# ─────────────────────────────────────────────────────────────
def resolve_image_path(rel: str, data_root: str) -> str:
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


def _tqdm_enabled() -> bool:
    """stderr가 TTY일 때만 progress bar (stdout 리다이렉션과 분리)."""
    if os.environ.get("DGM4_NO_TQDM", "").lower() in ("1", "true", "yes"):
        return False
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def _tqdm(it, **kw):
    kw.setdefault("file", sys.stderr)
    kw.setdefault("dynamic_ncols", True)
    kw.setdefault("disable", not _tqdm_enabled())
    try:
        sys.stdout.flush()
        return tqdm(it, **kw)
    except Exception:
        return it


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
# 4. Dataset & DataLoader
# ─────────────────────────────────────────────────────────────
class DGM4Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str,
                 tokenizer, clip_preprocess,
                 max_length: int = MAX_LENGTH,
                 require_image: bool = False):
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
    s   = pd.Series(binary_labels)
    cnt = s.value_counts().to_dict()
    w   = [(1.0 / cnt[l]) ** alpha for l in binary_labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


# ─────────────────────────────────────────────────────────────
# 5. 공통 서브모듈
# ─────────────────────────────────────────────────────────────
class GatedPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        s = self.score(x).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask, -1e4)
        return (torch.softmax(s, dim=-1).unsqueeze(-1) * x).sum(1)


class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, d_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_hid, d_out))

    def forward(self, x): return self.net(x)


class AuthorshipModule(nn.Module):
    def __init__(self, d, n_heads=8, dropout=0.1):
        super().__init__()
        self.style_self_attn = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True)
        self.style_var_proj  = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, d))
        self.style_pool      = GatedPooling(d)
        self.style_ffn       = FeedForward(d * 2, d * 2, d, dropout)
        self.style_ln        = nn.LayerNorm(d)
        self.auth_probe      = nn.Sequential(
            nn.Linear(d, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 1), nn.Sigmoid())
        self.auth_proj       = nn.Sequential(
            nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, PROJ_DIM))

    def forward(self, T_tok, txt_mask):
        T_sty, _ = self.style_self_attn(
            T_tok, T_tok, T_tok, key_padding_mask=~txt_mask)
        sr = T_tok - T_sty
        sv = self.style_var_proj(sr.var(dim=1).clamp(0.))
        sp = self.style_pool(sr, txt_mask)
        F_sty  = self.style_ln(self.style_ffn(torch.cat([sp, sv], dim=-1)))
        p_auth = self.auth_probe(F_sty).squeeze(-1)
        z_auth = F.normalize(self.auth_proj(F_sty), dim=-1)
        return F_sty, p_auth, z_auth


# ─────────────────────────────────────────────────────────────
# 6. WO-V 전용 서브모듈
# ─────────────────────────────────────────────────────────────
class OverAlignModuleWOV(nn.Module):
    """
    VeracityModule 없는 OAM.
    - gsim  : 제거 (VM이 계산하던 global cosine similarity)
    - p_fake: 제거 (VM이 계산하던 fake probe)
    - scalar 입력: 5개 (oa_mean, oa_std, oa_uni, oa_max, oa_min)
    - fine_head 입력: d + 2 (F_oa, p_auth, oa_mean)
    """
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.oa_scalar_proj = nn.Sequential(
            nn.Linear(5, 256), nn.GELU(),          # 6 → 5 (global_sim 제거)
            nn.Dropout(dropout), nn.Linear(256, d))
        self.oa_patch_pool  = nn.Sequential(
            nn.Linear(d, d // 2), nn.LayerNorm(d // 2),
            nn.GELU(), nn.Linear(d // 2, d))
        self.oa_ffn = nn.Sequential(
            nn.Linear(d * 2, d), nn.LayerNorm(d),
            nn.GELU(), nn.Dropout(dropout))
        self.fine_head = nn.Sequential(
            nn.Linear(d + 2, 256), nn.LayerNorm(256), nn.GELU(),  # d+3 → d+2
            nn.Dropout(dropout), nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, FINE_N), nn.Sigmoid())

    def forward(self, V_pat, T_cls, p_auth, has_image):
        """
        Args:
            V_pat    : (B, N_patch, d)  — CLIP patch tokens
            T_cls    : (B, d)           — RoBERTa [CLS]
            p_auth   : (B,)             — authorship probe (from AM)
            has_image: (B,)             — 0/1 마스크
        """
        B   = V_pat.size(0)
        m1  = has_image.unsqueeze(-1)
        eps = 1e-8

        t_n = F.normalize(T_cls, dim=-1, eps=eps).unsqueeze(1).expand_as(V_pat)
        v_n = F.normalize(V_pat, dim=-1, eps=eps)
        pts = (v_n * t_n).sum(-1) * has_image.unsqueeze(-1)
        pts = pts.clamp(min=0.)

        oa_mean = pts.mean(1)
        oa_std  = pts.std(1).clamp(0.)
        oa_max  = pts.max(1).values
        oa_min  = pts.min(1).values
        oa_uni  = oa_mean / ((oa_max - oa_min).clamp(min=eps))

        # 5개 스칼라 (global_sim 제거)
        F_oa_s = self.oa_scalar_proj(torch.stack(
            [oa_mean, oa_std, oa_uni, oa_max, oa_min], dim=-1))

        ow     = torch.softmax(pts * has_image.unsqueeze(-1), dim=1)
        F_oa_p = self.oa_patch_pool((ow.unsqueeze(-1) * V_pat).sum(1))
        F_oa   = self.oa_ffn(torch.cat([F_oa_s, F_oa_p], dim=-1)) * m1

        # fine_head: p_fake 제거 → p_auth + oa_mean 만 사용
        p_fine = self.fine_head(torch.cat([
            F_oa,
            p_auth.unsqueeze(-1),
            oa_mean.unsqueeze(-1)], dim=-1))
        p_fine = p_fine * has_image.unsqueeze(-1)

        return F_oa, p_fine, {
            "oa_mean":        oa_mean,
            "oa_std":         oa_std,
            "oa_uniformity":  oa_uni,
            "p_fine":         p_fine,
        }


class BilinearInteractionWOV(nn.Module):
    """
    VeracityModule 없는 Bilinear Interaction.
    - ver_low 브랜치 제거 → 2-way hadamard (auth × oa)
    - scalar: 3개 (p_auth·hi, p_fine_mean·hi, p_auth·p_fine_mean·hi)
    """
    def __init__(self, d, r=BILINEAR_RANK):
        super().__init__()
        self.auth_low = nn.Sequential(nn.Linear(d, r), nn.LayerNorm(r), nn.GELU())
        self.oa_low   = nn.Sequential(nn.Linear(d, r), nn.LayerNorm(r), nn.GELU())
        self.bi_proj  = nn.Sequential(nn.Linear(r, d), nn.LayerNorm(d), nn.GELU())
        self.scalar_proj = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, d))  # 6 → 3

    def forward(self, F_sty, F_oa, p_auth, p_fine_mean, has_image):
        """
        Args:
            F_sty      : (B, d)
            F_oa       : (B, d)
            p_auth     : (B,)
            p_fine_mean: (B,)
            has_image  : (B,)
        """
        m   = has_image.unsqueeze(-1)
        had = self.auth_low(F_sty) * self.oa_low(F_oa)   # 2-way

        scalars = torch.stack([
            p_auth     * has_image,
            p_fine_mean * has_image,
            p_auth * p_fine_mean * has_image,
        ], dim=-1)

        return self.bi_proj(had) + self.scalar_proj(scalars) * m


# ─────────────────────────────────────────────────────────────
# 7. 메인 모델 (WO-V)
# ─────────────────────────────────────────────────────────────
class HARFNETdgm4_WOV(nn.Module):
    """
    DGM4 전용 HARFNET-VER3 — VeracityModule 제거 버전.

    구조:
        RoBERTa → AM → F_sty, p_auth, z_auth
        CLIP    → V_pat, V_cls            (시각 패치·CLS)
                → OM(WOV) → F_oa, p_fine
                → BI(WOV) → F_bi
        binary_head( cat[F_sty, F_oa, F_bi] ) → logit   # d*3
    """
    def __init__(self, roberta_name=ROBERTA,
                 clip_name=CLIP_RN101, n_heads=8, dropout=0.1):
        super().__init__()
        d = 768

        # 텍스트 인코더
        self.text_encoder = RobertaModel.from_pretrained(
            roberta_name, add_pooling_layer=False)

        # 시각 인코더 (CLIP RN101) — layer4 + attnpool 부분 해동
        _clip, _ = openai_clip.load(clip_name, device="cpu")
        self.clip_model = _clip.float()
        self.clip_model.eval()
        for name, p in self.clip_model.named_parameters():
            trainable = any(k in name for k in CLIP_UNFREEZE_KEYS)
            p.requires_grad_(trainable)

        self.visual    = self.clip_model.visual
        clip_edim      = self.visual.attnpool.c_proj.out_features
        self.deep_proj = nn.Linear(2048, d)
        self.cls_proj  = nn.Linear(clip_edim, d)

        # 계층적 추론 모듈 (VM 없음)
        self.am = AuthorshipModule(d, n_heads, dropout)
        self.om = OverAlignModuleWOV(d, dropout)
        self.bi = BilinearInteractionWOV(d, BILINEAR_RANK)

        # Binary 분류 헤드: d*3 (F_sty + F_oa + F_bi)
        self.binary_head = nn.Sequential(
            nn.Linear(d * 3, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def _clip_visual_forward(self, pix, has_image):
        vis = self.visual
        B   = pix.size(0)
        x   = vis.relu1(vis.bn1(vis.conv1(pix)))
        x   = vis.relu2(vis.bn2(vis.conv2(x)))
        x   = vis.relu3(vis.bn3(vis.conv3(x)))
        x   = vis.avgpool(x) * has_image.view(B, 1, 1, 1)
        x   = vis.layer1(x); x = vis.layer2(x); x = vis.layer3(x)
        deep     = vis.layer4(x)
        v_global = vis.attnpool(deep)
        V_pat    = (self.deep_proj(deep.flatten(2).transpose(1, 2))
                    * has_image.view(B, 1, 1))
        V_cls    = self.cls_proj(v_global) * has_image.view(B, 1)
        return V_pat, V_cls

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()   # BatchNorm 안정성 유지
        return self

    def forward(self, input_ids, attention_mask,
                pixel_values, has_image, clip_text_tokens=None):
        device    = input_ids.device
        has_image = has_image.to(device)

        # 텍스트 인코딩
        T_tok    = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask).last_hidden_state
        txt_mask = attention_mask.bool()
        T_cls    = T_tok[:, 0]

        # 시각 인코딩
        V_pat, V_cls = self._clip_visual_forward(pixel_values, has_image)

        # AM: 문체 추론
        F_sty, p_auth, z_auth = self.am(T_tok, txt_mask)

        # OM(WOV): 정렬 이상 탐지 (gsim·p_fake 없음)
        F_oa, p_fine, oa_aux = self.om(V_pat, T_cls, p_auth, has_image)

        # BI(WOV): 2-way 교차 상호작용
        p_fine_mean = p_fine.mean(dim=-1)
        F_bi = self.bi(F_sty, F_oa, p_auth, p_fine_mean, has_image)

        # Binary 헤드: d*3
        feat         = torch.cat([F_sty, F_oa, F_bi], dim=-1)
        binary_logit = torch.nan_to_num(
            self.binary_head(feat).squeeze(-1),
            nan=0., posinf=20., neginf=-20.)

        return {
            "binary_logit": binary_logit,
            "p_fine":       p_fine,
            "p_auth":       p_auth,
            "z_auth":       z_auth,
            **oa_aux,
        }


# ─────────────────────────────────────────────────────────────
# 8. 손실 함수
# ─────────────────────────────────────────────────────────────
def binary_focal_loss(logit, target, gamma=2.0, pos_weight=1.0):
    pw  = torch.tensor([pos_weight], device=logit.device)
    bce = F.binary_cross_entropy_with_logits(
        logit, target.float(), pos_weight=pw, reduction="none")
    prob = torch.sigmoid(logit)
    pt   = torch.where(target == 1, prob, 1 - prob)
    return ((1 - pt) ** gamma * bce).mean()


def fine_label_loss(p_fine, fine_labels, has_image):
    loss     = F.binary_cross_entropy(
        p_fine.clamp(1e-6, 1 - 1e-6), fine_labels, reduction="none")
    mask     = has_image.unsqueeze(-1).expand_as(loss)
    mask_txt = torch.ones_like(mask)
    mask_txt[:, :2] = mask[:, :2]
    return (loss * mask_txt).mean()


def supcon_loss(z, labels, temperature=TEMPERATURE):
    B = z.size(0); device = z.device
    if B < 2: return z.sum() * 0.
    same  = (labels.unsqueeze(0) == labels.unsqueeze(1))
    self_m = torch.eye(B, dtype=torch.bool, device=device)
    pos_m  = same & ~self_m
    if pos_m.sum() == 0: return z.sum() * 0.
    sim   = torch.matmul(z, z.T) / temperature
    sim_s = sim - sim.detach().max(1, keepdim=True).values
    exp_s = torch.exp(sim_s).masked_fill(self_m, 0.)
    log_p = sim_s - torch.log(exp_s.sum(1, keepdim=True).clamp(1e-8))
    n_pos = pos_m.sum(1).clamp(1).float()
    return (-(log_p * pos_m).sum(1) / n_pos).mean()


def oa_reg_loss(out, binary_label, has_image):
    oa_mean = out["oa_mean"]
    loss    = oa_mean.sum() * 0.
    m_fake  = (binary_label == 1) & (has_image > 0.5)
    m_real  = (binary_label == 0) & (has_image > 0.5)
    if m_fake.any():
        loss = loss + ((1. - oa_mean[m_fake]) ** 2).mean()
    if m_real.any():
        loss = loss + (oa_mean[m_real] ** 2).mean()
    return loss


def total_loss_wov(out, binary_label, fine_labels, has_image, args):
    """
    WO-V 손실: z_ver SupCon 제거, z_auth SupCon 만 유지
    """
    l_bin  = binary_focal_loss(out["binary_logit"], binary_label)
    l_fine = fine_label_loss(out["p_fine"], fine_labels, has_image)
    l_con  = supcon_loss(out["z_auth"], binary_label)   # z_ver 없음
    l_oa   = oa_reg_loss(out, binary_label, has_image)

    return (LAMBDA_BINARY_BCE * l_bin
            + args.lambda_fine_bce * l_fine
            + args.lambda_con      * l_con
            + args.lambda_oa       * l_oa)


# ─────────────────────────────────────────────────────────────
# 9. Forward 헬퍼
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
# 10. 옵티마이저 & 스케줄러
# ─────────────────────────────────────────────────────────────
def make_optimizer(model: HARFNETdgm4_WOV, args):
    grp_text, grp_visual, grp_new = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "text_encoder" in name:
            grp_text.append(p)
        elif any(k in name for k in
                 ("visual", "clip_model", "deep_proj", "cls_proj")):
            grp_visual.append(p)
        else:
            grp_new.append(p)

    return torch.optim.AdamW([
        {"params": grp_text,   "lr": args.lr * LR_MULT_TEXT,   "weight_decay": 1e-4},
        {"params": grp_visual, "lr": args.lr * LR_MULT_VISUAL, "weight_decay": 1e-4},
        {"params": grp_new,    "lr": args.lr * LR_MULT_NEW,    "weight_decay": 1e-4},
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
# 11. 백본 설정
# ─────────────────────────────────────────────────────────────
def configure_backbones(model: HARFNETdgm4_WOV,
                        text_train_layers: Tuple[int, ...] = (10, 11),
                        unfreeze_clip_top: bool = True):
    for name, p in model.text_encoder.named_parameters():
        trainable = any(f"encoder.layer.{i}" in name
                        for i in text_train_layers)
        p.requires_grad_(trainable)

    for name, p in model.clip_model.named_parameters():
        trainable = (unfreeze_clip_top and
                     any(k in name for k in CLIP_UNFREEZE_KEYS))
        p.requires_grad_(trainable)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{LOG_TAG}] 학습 파라미터: {n_train:,} / {n_total:,}"
          f" ({100*n_train/n_total:.1f}%)")


# ─────────────────────────────────────────────────────────────
# 12. 학습 epoch
# ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, device, optimizer, train,
              epoch_idx=None, args=None, scheduler=None):
    model.train() if train else model.eval()
    tot, ys_bin, ps_bin, probs_bin, n = 0., [], [], [], 0

    # 검증 시 클래스별 신호 집계 (에폭 요약 로그용; train은 생략해 속도 유지)
    _sig: Dict[str, List[float]] = {
        "pa_real": [], "pa_fake": [],
        "pfine_real": [], "pfine_fake": [],
        "oam_real": [], "oam_fake": [],
    }

    ctx  = torch.enable_grad() if train else torch.no_grad()
    desc = (f"[{LOG_TAG}] Epoch {epoch_idx}"
            if (train and epoch_idx and _tqdm_enabled()) else None)
    it = _tqdm(loader, desc=desc) if desc else loader

    with ctx:
        for batch in it:
            y_bin  = batch["binary_label"].to(device)
            y_fine = batch["fine_labels"].to(device)
            hi     = batch["has_image"].to(device)

            out  = _fwd(model, batch, device)
            loss = total_loss_wov(out, y_bin, y_fine, hi, args)

            if train and optimizer:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            prob     = torch.sigmoid(out["binary_logit"]).detach()
            pred_bin = (prob > 0.5).long()
            tot      += loss.item() * y_bin.size(0)
            ys_bin.extend(y_bin.cpu().tolist())
            ps_bin.extend(pred_bin.cpu().tolist())
            probs_bin.extend(prob.cpu().tolist())
            n += y_bin.size(0)

            if not train:
                pa_d    = out["p_auth"].detach()
                pfine_d = out["p_fine"].mean(dim=-1).detach()
                oam_d   = out["oa_mean"].detach()
                m_real  = (y_bin == 0)
                m_fake  = (y_bin == 1)
                if m_real.any():
                    _sig["pa_real"].extend(pa_d[m_real].cpu().tolist())
                    _sig["pfine_real"].extend(pfine_d[m_real].cpu().tolist())
                    _sig["oam_real"].extend(oam_d[m_real].cpu().tolist())
                if m_fake.any():
                    _sig["pa_fake"].extend(pa_d[m_fake].cpu().tolist())
                    _sig["pfine_fake"].extend(pfine_d[m_fake].cpu().tolist())
                    _sig["oam_fake"].extend(oam_d[m_fake].cpu().tolist())

    if train and scheduler is not None:
        scheduler.step()

    n = max(n, 1)

    def _avg(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    try:
        auc = float(roc_auc_score(ys_bin, probs_bin))
    except Exception:
        auc = float("nan")

    return {
        "loss":     tot / n,
        "bin_acc":  float(accuracy_score(ys_bin, ps_bin)),
        "bin_f1":   float(f1_score(ys_bin, ps_bin,
                                   average="macro", zero_division=0)),
        "bin_auc":  auc,
        # 클래스별 평균 신호 (validation)
        "pa_real":    _avg(_sig["pa_real"]),
        "pa_fake":    _avg(_sig["pa_fake"]),
        "pfine_real": _avg(_sig["pfine_real"]),
        "pfine_fake": _avg(_sig["pfine_fake"]),
        "oam_real":   _avg(_sig["oam_real"]),
        "oam_fake":   _avg(_sig["oam_fake"]),
    }


def _format_epoch_line(ep: int, tr: dict, va: dict,
                       lr: Optional[float] = None) -> str:
    """MiRAGe 스타일 epoch 한 줄 (stdout → 로그 파일용)."""
    line = (
        f"[{LOG_TAG}] Epoch {ep}"
        f" train_loss={tr['loss']:.4f}"
        f" val_loss={va['loss']:.4f}"
        f" val_f1={va['bin_f1']:.4f}"
        f" val_acc={va['bin_acc']:.4f}"
        f" | Real:pauth={va['pa_real']:.2f},pfine={va['pfine_real']:.2f}"
        f",oam={va['oam_real']:.3f}"
        f" / Fake:pauth={va['pa_fake']:.2f},pfine={va['pfine_fake']:.2f}"
        f",oam={va['oam_fake']:.3f}"
    )
    if lr is not None:
        line += f"  lr={lr:.2e}"
    return line


# ─────────────────────────────────────────────────────────────
# 13. Matplotlib 학습 곡선 플롯
# ─────────────────────────────────────────────────────────────
def plot_training_curves(
    train_losses: List[float],
    val_losses:   List[float],
    val_f1s:      List[float],
    save_path:    str,
    title:        str = "Training Curves (WO-V)",
    quiet:        bool = False,
):
    """
    Epoch별 train/val loss 와 val macro-F1 을 하나의 figure에 그리고 저장.

    Args:
        train_losses : 에폭별 train loss 리스트
        val_losses   : 에폭별 val   loss 리스트
        val_f1s      : 에폭별 val   macro-F1 리스트
        save_path    : PNG 저장 경로
        title        : figure 제목
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── 왼쪽: Loss 곡선 ─────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, "o-", color="#185FA5",
             linewidth=2, markersize=5, label="Train Loss")
    ax1.plot(epochs, val_losses,   "s--", color="#993C1D",
             linewidth=2, markersize=5, label="Val Loss")

    best_val_ep = int(np.argmin(val_losses)) + 1
    ax1.axvline(best_val_ep, color="#993C1D", linestyle=":",
                linewidth=1, alpha=0.6, label=f"Best Val (ep {best_val_ep})")

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Train / Val Loss", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    ax1.tick_params(axis="both", labelsize=10)

    # ── 오른쪽: Val Macro-F1 곡선 ───────────────────────────
    ax2 = axes[1]
    ax2.plot(epochs, val_f1s, "D-", color="#0F6E56",
             linewidth=2, markersize=5, label="Val Macro-F1")

    best_f1_ep  = int(np.argmax(val_f1s)) + 1
    best_f1_val = max(val_f1s)
    ax2.axvline(best_f1_ep, color="#0F6E56", linestyle=":",
                linewidth=1, alpha=0.6, label=f"Best F1={best_f1_val:.4f} (ep {best_f1_ep})")

    # 최고점 강조 마커
    ax2.scatter([best_f1_ep], [best_f1_val], color="#0F6E56",
                s=80, zorder=5)
    ax2.annotate(f"{best_f1_val:.4f}",
                 xy=(best_f1_ep, best_f1_val),
                 xytext=(best_f1_ep + 0.2, best_f1_val - 0.005),
                 fontsize=9, color="#0F6E56")

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Macro-F1", fontsize=12)
    ax2.set_title("Val Macro-F1", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim(max(0, min(val_f1s) - 0.02), min(1.0, max(val_f1s) + 0.02))
    ax2.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if not quiet:
        print(f"그래프 저장: {save_path}", flush=True)


# ─────────────────────────────────────────────────────────────
# 14. 학습 루프
# ─────────────────────────────────────────────────────────────
def train_one_run(model, tl, vl, device, args,
                  log=None, plot_path: Optional[str] = None):
    """
    [수정]
    - 에폭마다 train_loss / val_loss / val_f1 기록
    - 학습 종료 후 matplotlib 그래프 저장 (plot_path 지정 시)
    - EarlyStopping: val AUC 기준
    - Best model: val AUC 기준
    """
    opt   = make_optimizer(model, args)
    sched = make_scheduler(opt, args)
    early = EarlyStopping(args.early_stop_patience,
                          args.early_stop_min_delta, mode="max")
    best_auc, best_st = -1., None

    # 에폭별 기록용
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, tl, device, opt, True,
                       epoch_idx=ep, args=args, scheduler=sched)
        va = run_epoch(model, vl, device, None, False, args=args)

        # 기록
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["val_f1"].append(va["bin_f1"])
        history["val_auc"].append(va["bin_auc"])

        cur_lr = sched.get_last_lr()[-1] if hasattr(sched, "get_last_lr")\
                 else opt.param_groups[-1]["lr"]

        line = _format_epoch_line(ep, tr, va, lr=cur_lr)
        print(line, flush=True)   # stdout only → 로그 파일에 깔끔히 기록
        if log is not None:
            log.append(line)

        if plot_path:
            plot_training_curves(
                train_losses=history["train_loss"],
                val_losses=history["val_loss"],
                val_f1s=history["val_f1"],
                save_path=plot_path,
                title="HARFNET-WOV DGM4 — Training Curves",
                quiet=True,
            )

        if not math.isnan(va["bin_auc"]) and va["bin_auc"] > best_auc:
            best_auc = va["bin_auc"]
            best_st  = copy.deepcopy(model.state_dict())

        if early.step(va["bin_auc"] if not math.isnan(va["bin_auc"])
                      else va["bin_f1"]):
            msg = f"[{LOG_TAG}] Early stopping at epoch {ep}"
            print(msg, flush=True)
            if log:
                log.append(msg)
            break

    if plot_path and history["train_loss"]:
        plot_training_curves(
            train_losses=history["train_loss"],
            val_losses=history["val_loss"],
            val_f1s=history["val_f1"],
            save_path=plot_path,
            title="HARFNET-WOV DGM4 — Training Curves",
        )

    if best_st:
        model.load_state_dict(best_st)
    return model, history


# ─────────────────────────────────────────────────────────────
# 15. 예측 수집 & 평가
# ─────────────────────────────────────────────────────────────
def collect_predictions(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in _tqdm(loader, desc=f"[{LOG_TAG}] 예측 수집",
                           leave=False):
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
                    # p_fake 없음: 0으로 채움 (호환성)
                    "p_fake":         0.0,
                    "oa_mean":        round(float(out["oa_mean"][i]), 4),
                    "oa_std":         round(float(out["oa_std"][i]), 4),
                    "oa_uniformity":  round(float(
                        out["oa_uniformity"][i].clamp(max=10.)), 4),
                    "global_sim":     0.0,  # VM 없음: 0으로 채움
                })
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    yt    = df["binary_label"].values
    yp    = df["binary_pred"].values
    yprob = df["binary_prob"].values
    m = {
        "bin_acc":     float(accuracy_score(yt, yp)),
        "bin_f1":      float(f1_score(yt, yp, average="macro",  zero_division=0)),
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


def eval_report_block(df: pd.DataFrame,
                      label: str = LOG_TAG) -> Tuple[str, dict]:
    yt = df["binary_label"].values
    yp = df["binary_pred"].values
    buf = io.StringIO()
    buf.write(f"\n=== {label} — Binary Detection ===\n")
    buf.write(classification_report(yt, yp,
                                    target_names=["real", "fake"],
                                    digits=4, zero_division=0))
    m = compute_metrics(df)
    buf.write(f"\n[{label}] Binary F1={m['bin_f1']:.4f}"
              f"  AUC={m.get('bin_auc', float('nan')):.4f}\n")
    return buf.getvalue(), m


# ─────────────────────────────────────────────────────────────
# 16. DataLoader 헬퍼
# ─────────────────────────────────────────────────────────────
def _make_dl(df, data_root, tok, prep, args, shuffle, sampler=None):
    ds = DGM4Dataset(df, data_root, tok, prep, args.max_length)
    kw = dict(batch_size=args.batch_size,
              num_workers=args.num_workers,
              collate_fn=collate_batch,
              pin_memory=torch.cuda.is_available(),
              drop_last=shuffle)
    if sampler is not None:
        kw["sampler"] = sampler; kw["shuffle"] = False
    else:
        kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


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


# ─────────────────────────────────────────────────────────────
# 17. run_official — DGM4 공식 split (권장)
# ─────────────────────────────────────────────────────────────
def run_official(dfs: dict, device, tok, prep, args, configure, result_dir):
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (Official Split) ######\n\n")

    tr_ds = DGM4Dataset(dfs["train"], args.data_root, tok, prep, args.max_length)
    tl    = _make_dl(dfs["train"], args.data_root, tok, prep, args,
                     shuffle=True,
                     sampler=make_weighted_sampler(tr_ds.binary_lbls,
                                                    args.sampler_alpha))
    vl    = _make_dl(dfs["validation"], args.data_root, tok, prep,
                     args, shuffle=False)
    el    = _make_dl(dfs["test"], args.data_root, tok, prep,
                     args, shuffle=False)

    model = HARFNETdgm4_WOV(args.roberta, args.clip_model).to(device)
    if configure:
        configure_backbones(model)

    log       = []
    plot_path = os.path.join(result_dir,
                             f"harfnet_wov_dgm4_official_curve_{ts}.png")

    model, history = train_one_run(model, tl, vl, device, args,
                                   log=log, plot_path=plot_path)
    for line in log:
        rep.write(line + "\n")

    print(f"\n[{LOG_TAG}] 테스트 예측 수집...")
    full_df = collect_predictions(model, el, device)
    blk, m  = eval_report_block(full_df)
    print(blk, end="")
    rep.write(blk)

    ckpt_path = os.path.join(args.checkpoint_dir,
                              f"harfnet_wov_dgm4_official_{ts}.pt")
    save_checkpoint(ckpt_path, model, args, m)

    summ = (f"{LOG_TAG}: BinF1={m['bin_f1']:.4f}"
            f"  AUC={m.get('bin_auc', float('nan')):.4f}")
    print("\n" + "=" * 70 + "\n" + summ + "\n" + "=" * 70)

    path = os.path.join(result_dir,
                         f"harfnet_wov_dgm4_official_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rep.getvalue())
    print(f"리포트: {path}")


# ─────────────────────────────────────────────────────────────
# 18. run_kfold — K-Fold (선택적)
# ─────────────────────────────────────────────────────────────
def run_kfold(dfs: dict, device, tok, prep, args, configure, result_dir):
    df_tv = pd.concat([dfs["train"], dfs["validation"]],
                       ignore_index=True)
    df_te = dfs["test"]
    y_tv  = df_tv["binary_label"].values
    k     = args.kfold_splits
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rep = io.StringIO()
    all_rep.write(f"###### {LOG_TAG} (k={k} KFold) ######\n")
    el = _make_dl(df_te, args.data_root, tok, prep, args, shuffle=False)

    skf    = StratifiedKFold(n_splits=k, shuffle=True,
                              random_state=args.kfold_random_state)
    fold_m = []

    for fold, (tr_idx, va_idx) in enumerate(
            skf.split(np.zeros(len(df_tv)), y_tv), 1):
        print(f"\n{'='*70}\n[Fold {fold}/{k}] {LOG_BRAND}\n{'='*70}")
        df_tr_f = df_tv.iloc[tr_idx].reset_index(drop=True)
        df_va_f = df_tv.iloc[va_idx].reset_index(drop=True)

        tr_ds = DGM4Dataset(df_tr_f, args.data_root, tok, prep, args.max_length)
        tl    = _make_dl(df_tr_f, args.data_root, tok, prep, args,
                         shuffle=True,
                         sampler=make_weighted_sampler(tr_ds.binary_lbls,
                                                        args.sampler_alpha))
        vl    = _make_dl(df_va_f, args.data_root, tok, prep, args,
                         shuffle=False)

        model = HARFNETdgm4_WOV(args.roberta, args.clip_model).to(device)
        if configure:
            configure_backbones(model)

        log       = []
        plot_path = os.path.join(
            result_dir,
            f"harfnet_wov_dgm4_curve_fold{fold}_{ts}.png")

        model, history = train_one_run(model, tl, vl, device, args,
                                       log=log, plot_path=plot_path)

        full_df = collect_predictions(model, el, device)
        blk, m  = eval_report_block(full_df, label=f"Fold{fold}")
        print(blk, end="")

        ckpt = os.path.join(
            args.checkpoint_dir,
            f"harfnet_wov_dgm4_kfold_{ts}_fold{fold}.pt")
        save_checkpoint(ckpt, model, args, m)

        summ = (f"BinF1={m['bin_f1']:.4f}"
                f"  AUC={m.get('bin_auc', float('nan')):.4f}")
        print(f"\n[Fold {fold}] {summ}")
        all_rep.write(
            f"\n{'#'*80}\n### FOLD {fold}\n{'#'*80}\n{blk}")
        fold_m.append(m)

        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _mean(k_): return float(np.mean([r[k_] for r in fold_m]))
    g = (f"{LOG_TAG}: BinF1={_mean('bin_f1'):.4f}"
         f"  AUC={_mean('bin_auc'):.4f}")
    all_rep.write(f"\n\n{'#'*80}\n### SUMMARY\n{g}\n")

    path = os.path.join(result_dir,
                         f"harfnet_wov_dgm4_kfold_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(all_rep.getvalue())
    print(f"\n리포트: {path}\n{g}")


# ─────────────────────────────────────────────────────────────
# 19. main
# ─────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(
        description="HARFNET-VER3-WO-V × DGM4 벤치마크")

    # 데이터
    pa.add_argument("--data_root",      default=DEFAULT_DATA_ROOT)
    pa.add_argument("--checkpoint_dir", default="./checkpoints")
    pa.add_argument("--result_dir",     default="./results")

    # 학습
    pa.add_argument("--batch_size",           type=int,   default=BATCH)
    pa.add_argument("--epochs",               type=int,   default=EPOCHS)
    pa.add_argument("--lr",                   type=float, default=LR)
    pa.add_argument("--warmup_epochs",        type=int,   default=WARMUP_EPOCHS)
    pa.add_argument("--num_workers",          type=int,   default=NUM_WORKERS)
    pa.add_argument("--seed",                 type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)

    # 실험 모드
    pa.add_argument("--kfold", action="store_true",
                    help="K-Fold 실행 (비권장: official split 기본)")
    pa.add_argument("--kfold_splits",       type=int, default=KFOLD_SPLITS)
    pa.add_argument("--kfold_random_state", type=int, default=KFOLD_RANDOM_STATE)

    # 모델
    pa.add_argument("--roberta",    default=ROBERTA)
    pa.add_argument("--clip_model", default=CLIP_RN101)
    pa.add_argument("--max_length", type=int, default=MAX_LENGTH)

    # 샘플러
    pa.add_argument("--sampler_alpha", type=float, default=0.5)

    # 손실 가중치
    pa.add_argument("--lambda_fine_bce", type=float, default=LAMBDA_FINE_BCE)
    pa.add_argument("--lambda_con",      type=float, default=LAMBDA_CON)
    pa.add_argument("--lambda_oa",       type=float, default=LAMBDA_OA)

    # 기타
    pa.add_argument("--no_configure_backbones", action="store_true")
    pa.add_argument("--no_progress",            action="store_true")

    args = pa.parse_args()

    if args.no_progress:
        os.environ["DGM4_NO_TQDM"] = "1"

    configure = not args.no_configure_backbones
    set_seed(args.seed)
    os.makedirs(args.result_dir,     exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    mode = "KFold" if args.kfold else "Official Split (권장)"
    print(f"\n{LOG_BRAND} | device={DEVICE} | mode={mode}")
    print(f"  [WO-V] VeracityModule 제거: F_ver·p_fake·z_ver 없음")
    print(f"  [WO-V] binary_head 입력: d*4 → d*3")
    print(f"  [WO-V] BI scalar: 6개 → 3개  |  OAM scalar: 6개 → 5개")
    print(f"  lr={args.lr:.2e}  warmup={args.warmup_epochs}ep"
          f"  epochs={args.epochs}"
          f"  λ_fine={args.lambda_fine_bce}"
          f"  λ_con={args.lambda_con}(z_auth only)"
          f"  λ_oa={args.lambda_oa}")

    dfs  = load_dgm4_splits(args.data_root)
    tok  = RobertaTokenizerFast.from_pretrained(args.roberta)
    _, prep = openai_clip.load(args.clip_model, device="cpu")

    if args.kfold:
        run_kfold(dfs, DEVICE, tok, prep, args, configure, args.result_dir)
    else:
        run_official(dfs, DEVICE, tok, prep, args, configure, args.result_dir)


if __name__ == "__main__":
    main()
