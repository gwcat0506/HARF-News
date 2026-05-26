"""
HARFNET-VER3 Balanced Test Evaluation Script
─────────────────────────────────────────────
체크포인트: harfnet_ver3_kfold_20260430_225114_fold5.pt
목적: 전체 20% test 데이터에 대해 클래스 imbalance를 해소한 뒤 평가
      → Appendix 삽입용 리포트 생성
"""

from __future__ import annotations

import argparse, io, os, random, sys, warnings
from datetime import datetime
from typing import Dict, List

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import clip as openai_clip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizerFast

# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
BATCH                = 16
NUM_WORKERS          = 4
KFOLD_TEST_SIZE      = 0.2
TRAIN_RATIO          = 0.6
VAL_RATIO            = 0.2
TEST_RATIO           = 0.2
SEED                 = 42

BILINEAR_RANK        = 128
PROJ_DIM             = 128
TEMPERATURE          = 0.07

ROBERTA              = "roberta-base"
CLIP_RN101           = "RN101"
MAX_LENGTH           = 128

_SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
_FND_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
DEFAULT_DATA_ROOT = _FND_ROOT
DEFAULT_CSV_PATH  = os.path.join(_FND_ROOT, "HARFM.csv")
DEFAULT_CKPT      = os.path.join(
    _SCRIPT_DIR,
    "checkpoints/harfnet_ver3_kfold_20260430_225114_fold5.pt",
)
RESULT_DIR           = os.path.join(_SCRIPT_DIR, "results")

CLASS_NAMES: List[str]      = ["HR", "HF", "AR", "AF"]
CLASS2IDX:   Dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULT_DIR, exist_ok=True)


def set_seed(s: int = SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed()


# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
def resolve_image_path(p: str, data_root: str) -> str:
    p = (p or "").strip()
    if not p: return p
    if os.path.isabs(p) and os.path.isfile(p): return p
    return os.path.join(os.path.abspath(data_root), p)


def harf_filter_multimodal_only(df: pd.DataFrame, data_root: str) -> pd.DataFrame:
    rel = df["image_path"].fillna("").astype(str).str.strip()
    def _ok(r): return bool(r) and os.path.isfile(resolve_image_path(r, data_root))
    return df[rel.map(_ok) & (df["final_headline"].astype(str).str.strip().str.len() > 0)
              ].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
class HARFDataset(Dataset):
    def __init__(self, csv_path, data_root, tokenizer, clip_preprocess,
                 max_length=MAX_LENGTH, indices=None):
        super().__init__()
        self.data_root       = os.path.abspath(data_root)
        self.tokenizer       = tokenizer
        self.clip_preprocess = clip_preprocess
        self.max_length      = max_length

        df = pd.read_csv(csv_path,
                         usecols=["final_headline", "image_path", "4_way_label"])
        df = df.dropna(subset=["final_headline", "4_way_label"])
        for c in ["final_headline", "image_path", "4_way_label"]:
            df[c] = df[c].fillna("").astype(str).str.strip()
        df["_y"] = df["4_way_label"].map(CLASS2IDX)
        df = df.dropna(subset=["_y"]).drop(columns=["_y"]).reset_index(drop=True)
        df = harf_filter_multimodal_only(df, self.data_root)
        if len(df) == 0:
            raise ValueError("유효 행 없음")

        self.texts  = df["final_headline"].tolist()
        self.paths  = df["image_path"].tolist()
        self.labels = [CLASS2IDX[x] for x in df["4_way_label"].tolist()]

        if indices is not None:
            self.texts  = [self.texts[i]  for i in indices]
            self.paths  = [self.paths[i]  for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self): return len(self.labels)

    def _load_pil(self, rel):
        if not rel:
            return Image.new("RGB", (224, 224), (127, 127, 127)), False
        full = resolve_image_path(rel, self.data_root)
        if not full or not os.path.isfile(full):
            return Image.new("RGB", (224, 224), (127, 127, 127)), False
        return Image.open(full).convert("RGB"), True

    def __getitem__(self, idx):
        text    = self.texts[idx]
        img, ok = self._load_pil(self.paths[idx])
        enc = self.tokenizer(text, max_length=self.max_length,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
        return {
            "input_ids":        enc["input_ids"].squeeze(0),
            "attention_mask":   enc["attention_mask"].squeeze(0),
            "pixel_values":     self.clip_preprocess(img),
            "clip_text_tokens": openai_clip.tokenize([text], truncate=True)[0],
            "has_image":        torch.tensor(1.0 if ok else 0.0),
            "label":            torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_batch(batch):
    keys = ("input_ids", "attention_mask", "pixel_values",
            "clip_text_tokens", "has_image", "label")
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
class GatedPooling(nn.Module):
    def __init__(self, dim: int):
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
    def __init__(self, d: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.style_self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.style_var_proj  = nn.Sequential(nn.Linear(d, d//2), nn.GELU(), nn.Linear(d//2, d))
        self.style_pool      = GatedPooling(d)
        self.style_ffn       = FeedForward(d*2, d*2, d, dropout)
        self.style_ln        = nn.LayerNorm(d)
        self.auth_probe      = nn.Sequential(nn.Linear(d, 128), nn.LayerNorm(128), nn.GELU(),
                                             nn.Linear(128, 1), nn.Sigmoid())
        self.auth_proj       = nn.Sequential(nn.Linear(d, d//2), nn.ReLU(), nn.Linear(d//2, PROJ_DIM))

    def forward(self, T_tok, txt_mask):
        T_sty, _      = self.style_self_attn(T_tok, T_tok, T_tok, key_padding_mask=~txt_mask)
        style_residual = T_tok - T_sty
        style_var      = self.style_var_proj(style_residual.var(dim=1).clamp(0.))
        style_pool     = self.style_pool(style_residual, txt_mask)
        F_sty  = self.style_ln(self.style_ffn(torch.cat([style_pool, style_var], dim=-1)))
        p_auth = self.auth_probe(F_sty).squeeze(-1)
        z_auth = F.normalize(self.auth_proj(F_sty), dim=-1)
        return F_sty, p_auth, z_auth


class VeracityModule(nn.Module):
    def __init__(self, d: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.txt2img_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.img2txt_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.gap_encoder  = nn.Sequential(nn.Linear(d*2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.sim_proj     = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, d))
        self.verac_ffn    = FeedForward(d*2, d*2, d, dropout)
        self.verac_ln     = nn.LayerNorm(d)
        self.fake_probe   = nn.Sequential(nn.Linear(d, 128), nn.LayerNorm(128), nn.GELU(),
                                          nn.Linear(128, 1), nn.Sigmoid())
        self.verac_proj   = nn.Sequential(nn.Linear(d, d//2), nn.ReLU(), nn.Linear(d//2, PROJ_DIM))

    def forward(self, T_tok, txt_mask, V_pat, T_cls, V_cls, has_image):
        B   = T_tok.size(0); m1 = has_image.unsqueeze(-1); m3 = has_image.view(B, 1, 1); eps = 1e-8
        F_t2v, _ = self.txt2img_attn(T_tok, V_pat, V_pat); F_t2v = F_t2v * m3
        F_v2t, _ = self.img2txt_attn(V_pat, T_tok, T_tok, key_padding_mask=~txt_mask); F_v2t = F_v2t * m3
        valid_L  = txt_mask.float().sum(1, keepdim=True).clamp(1)
        gap_t    = ((T_tok - F_t2v) * txt_mask.unsqueeze(-1).float()).sum(1) / valid_L
        gap_v    = (V_pat - F_v2t).mean(1) * m1
        F_gap    = self.gap_encoder(torch.cat([gap_t, gap_v], dim=-1))
        t_n      = F.normalize(T_cls, dim=-1, eps=eps); v_n = F.normalize(V_cls, dim=-1, eps=eps)
        s_sim    = (t_n * v_n).sum(-1, keepdim=True).clamp(-1., 1.) * m1
        F_sim    = self.sim_proj(s_sim)
        F_ver    = self.verac_ln(self.verac_ffn(torch.cat([F_gap, F_sim], dim=-1)))
        p_fake   = self.fake_probe(F_ver).squeeze(-1)
        z_ver    = F.normalize(self.verac_proj(F_ver), dim=-1)
        return F_ver, p_fake, z_ver, s_sim.squeeze(-1)


class OverAlignModule(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.oa_scalar_proj = nn.Sequential(nn.Linear(6, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, d))
        self.oa_patch_pool  = nn.Sequential(nn.Linear(d, d//2), nn.LayerNorm(d//2), nn.GELU(), nn.Linear(d//2, d))
        self.oa_ffn = nn.Sequential(nn.Linear(d*2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.af_head = nn.Sequential(nn.Linear(d+3, 256), nn.LayerNorm(256), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(256, 64), nn.GELU(),
                                     nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, V_pat, T_cls, global_sim, p_fake, p_auth, has_image):
        m1  = has_image.unsqueeze(-1); eps = 1e-8
        t_n = F.normalize(T_cls, dim=-1, eps=eps).unsqueeze(1).expand_as(V_pat)
        v_n = F.normalize(V_pat, dim=-1, eps=eps)
        patch_text_sim = (v_n * t_n).sum(-1) * has_image.unsqueeze(-1)
        patch_text_sim = patch_text_sim.clamp(min=0.)
        oa_mean        = patch_text_sim.mean(1)
        oa_std         = patch_text_sim.std(1).clamp(0.)
        oa_max         = patch_text_sim.max(1).values
        oa_min         = patch_text_sim.min(1).values
        oa_uniformity  = oa_mean / ((oa_max - oa_min).clamp(min=eps))
        oa_scalars     = torch.stack([oa_mean, oa_std, oa_uniformity, oa_max, oa_min, global_sim], dim=-1)
        F_oa_scalar    = self.oa_scalar_proj(oa_scalars)
        oa_weights     = torch.softmax(patch_text_sim * has_image.unsqueeze(-1), dim=1)
        F_oa_patch     = self.oa_patch_pool((oa_weights.unsqueeze(-1) * V_pat).sum(1))
        F_oa   = self.oa_ffn(torch.cat([F_oa_scalar, F_oa_patch], dim=-1)) * m1
        p_af   = self.af_head(torch.cat([
            F_oa, p_fake.unsqueeze(-1), oa_mean.unsqueeze(-1),
            oa_uniformity.clamp(max=50.).unsqueeze(-1),
        ], dim=-1)).squeeze(-1) * has_image
        return F_oa, p_af, {"oa_mean": oa_mean, "oa_std": oa_std,
                             "oa_uniformity": oa_uniformity, "p_af": p_af}


class BilinearInteraction(nn.Module):
    def __init__(self, d: int, r: int = BILINEAR_RANK):
        super().__init__()
        self.auth_low    = nn.Sequential(nn.Linear(d, r), nn.LayerNorm(r), nn.GELU())
        self.ver_low     = nn.Sequential(nn.Linear(d, r), nn.LayerNorm(r), nn.GELU())
        self.oa_low      = nn.Sequential(nn.Linear(d, r), nn.LayerNorm(r), nn.GELU())
        self.bi_proj     = nn.Sequential(nn.Linear(r, d), nn.LayerNorm(d), nn.GELU())
        self.scalar_proj = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, d))

    def forward(self, F_sty, F_ver, F_oa, p_auth, p_fake, p_af, has_image):
        m        = has_image.unsqueeze(-1)
        hadamard = self.auth_low(F_sty) * self.ver_low(F_ver) * self.oa_low(F_oa)
        F_had    = self.bi_proj(hadamard)
        scalars  = torch.stack([
            p_auth * has_image, p_fake * has_image, p_af * has_image,
            p_auth * (1. - p_af) * has_image, p_auth * p_af * has_image,
            p_fake * (1. - p_af) * has_image,
        ], dim=-1)
        return F_had + self.scalar_proj(scalars) * m


class HARFNETver3(nn.Module):
    def __init__(self, roberta_name=ROBERTA, clip_name=CLIP_RN101,
                 n_heads=8, dropout=0.1):
        super().__init__()
        d = 768
        self.text_encoder = RobertaModel.from_pretrained(roberta_name, add_pooling_layer=False)
        _clip, _ = openai_clip.load(clip_name, device="cpu")
        self.clip_model = _clip.float(); self.clip_model.eval()
        for p in self.clip_model.parameters(): p.requires_grad_(False)
        self.visual    = self.clip_model.visual
        clip_edim      = self.visual.attnpool.c_proj.out_features
        self.deep_proj = nn.Linear(2048, d)
        self.cls_proj  = nn.Linear(clip_edim, d)
        self.am = AuthorshipModule(d, n_heads, dropout)
        self.vm = VeracityModule(d, n_heads, dropout)
        self.om = OverAlignModule(d, dropout)
        self.bi = BilinearInteraction(d, BILINEAR_RANK)
        self.head4 = nn.Sequential(
            nn.Linear(d*4, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 4))

    def _clip_visual_forward(self, pix, has_image):
        vis = self.visual; B = pix.size(0)
        m4d = has_image.view(B, 1, 1, 1)
        x = vis.relu1(vis.bn1(vis.conv1(pix)))
        x = vis.relu2(vis.bn2(vis.conv2(x)))
        x = vis.relu3(vis.bn3(vis.conv3(x)))
        x = vis.avgpool(x) * m4d
        x = vis.layer1(x); x = vis.layer2(x); x = vis.layer3(x)
        deep = vis.layer4(x)
        v_global = vis.attnpool(deep)
        ms   = has_image.view(B, 1, 1)
        V_pat = self.deep_proj(deep.flatten(2).transpose(1, 2)) * ms
        V_cls = self.cls_proj(v_global) * has_image.view(B, 1)
        return V_pat, V_cls

    def train(self, mode=True):
        super().train(mode); self.clip_model.eval(); return self

    def forward(self, input_ids, attention_mask, pixel_values,
                has_image, clip_text_tokens=None):
        device    = input_ids.device
        has_image = has_image.to(device)
        T_tok     = self.text_encoder(input_ids=input_ids,
                                      attention_mask=attention_mask).last_hidden_state
        txt_mask  = attention_mask.bool()
        T_cls     = T_tok[:, 0]
        V_pat, V_cls = self._clip_visual_forward(pixel_values, has_image)
        F_sty, p_auth, z_auth = self.am(T_tok, txt_mask)
        F_ver, p_fake, z_ver, global_sim = self.vm(T_tok, txt_mask, V_pat, T_cls, V_cls, has_image)
        F_oa, p_af, oa_aux   = self.om(V_pat, T_cls, global_sim, p_fake, p_auth, has_image)
        F_bi  = self.bi(F_sty, F_ver, F_oa, p_auth, p_fake, p_af, has_image)
        logits4 = torch.nan_to_num(
            self.head4(torch.cat([F_sty, F_ver, F_oa, F_bi], dim=-1)),
            nan=0., posinf=20., neginf=-20.)
        return {"logits4": logits4, "p_auth": p_auth, "p_fake": p_fake,
                "p_af": p_af, "z_auth": z_auth, "z_ver": z_ver,
                "global_sim": global_sim, **oa_aux}


# ──────────────────────────────────────────────────────────────
# 4. 클래스 균형 서브샘플링
# ──────────────────────────────────────────────────────────────
def balance_indices_by_class(labels: List[int], seed: int = SEED) -> List[int]:
    """
    각 클래스에서 최소 클래스 수만큼 무작위 서브샘플링.
    HR/HF가 매우 많고 AR/AF가 적은 imbalance를 해소한다.
    """
    rng = np.random.default_rng(seed)
    per_class: Dict[int, List[int]] = {i: [] for i in range(4)}
    for idx, lbl in enumerate(labels):
        per_class[lbl].append(idx)

    min_cnt = min(len(v) for v in per_class.values())
    print("\n[BalancedTest] 클래스별 원본 샘플 수:")
    for ci, cn in enumerate(CLASS_NAMES):
        print(f"  {cn}: {len(per_class[ci])} → {min_cnt} (균형)")

    balanced = []
    for ci in range(4):
        chosen = rng.choice(per_class[ci], size=min_cnt, replace=False).tolist()
        balanced.extend(chosen)

    rng.shuffle(balanced)
    return balanced


# ──────────────────────────────────────────────────────────────
# 5. 추론 & 리포트
# ──────────────────────────────────────────────────────────────
def collect_predictions(model, loader, device):
    model.eval()
    ys, ps, prob_chunks = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval] Inference", dynamic_ncols=True):
            out    = model(batch["input_ids"].to(device),
                           batch["attention_mask"].to(device),
                           batch["pixel_values"].to(device),
                           batch["has_image"].to(device))
            logits = out["logits4"]
            pr     = F.softmax(logits, dim=-1)
            ys.extend(batch["label"].tolist())
            ps.extend(logits.argmax(-1).cpu().tolist())
            prob_chunks.append(pr.cpu().numpy())
    proba = np.vstack(prob_chunks) if prob_chunks else np.zeros((0, 4))
    return np.array(ys), np.array(ps), proba


def auc_scores(yt, proba):
    out = {}
    try:
        out["auc_4_ovr_macro"] = float(roc_auc_score(
            yt, proba, multi_class="ovr", average="macro", labels=[0, 1, 2, 3]))
    except ValueError:
        out["auc_4_ovr_macro"] = float("nan")
    yh   = yt // 2;  p_ai   = proba[:, 2:4].sum(1)
    yr   = yt %  2;  p_fake = proba[:, [1, 3]].sum(1)
    try: out["auc_ha"] = float(roc_auc_score(yh, p_ai))
    except ValueError: out["auc_ha"] = float("nan")
    try: out["auc_rf"] = float(roc_auc_score(yr, p_fake))
    except ValueError: out["auc_rf"] = float("nan")
    return out


def confusion_matrix_str(yt, yp):
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
    header = f"{'':>6}" + "".join(f"{cn:>8}" for cn in CLASS_NAMES) + "\n"
    rows   = ""
    for i, cn in enumerate(CLASS_NAMES):
        rows += f"{cn:>6}" + "".join(f"{cm[i, j]:>8d}" for j in range(4)) + "\n"
    return header + rows


def per_class_detail(yt, yp, proba):
    """클래스별 정밀도·재현율·F1·AUC(OvR) 상세 테이블"""
    lines = []
    lines.append(f"\n{'Class':<6} {'Precision':>10} {'Recall':>10} {'F1':>10} "
                 f"{'AUC-OvR':>10} {'Support':>10}")
    lines.append("-" * 62)
    for ci, cn in enumerate(CLASS_NAMES):
        mask    = (yt == ci)
        prec    = float(np.sum((yp == ci) & mask) /
                        max(np.sum(yp == ci), 1))
        rec     = float(np.sum((yp == ci) & mask) / max(mask.sum(), 1))
        f1      = (2 * prec * rec / max(prec + rec, 1e-9))
        try:
            auc_i = float(roc_auc_score((yt == ci).astype(int), proba[:, ci]))
        except ValueError:
            auc_i = float("nan")
        sup = int(mask.sum())
        lines.append(f"{cn:<6} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} "
                     f"{auc_i:>10.4f} {sup:>10d}")
    return "\n".join(lines)


def build_report(yt, yp, proba, mode_tag: str, n_original: int, n_balanced: int) -> str:
    buf = io.StringIO()
    yh, ph = yt // 2, yp // 2
    yr, pr = yt %  2, yp %  2
    aucm   = auc_scores(yt, proba)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    buf.write("=" * 72 + "\n")
    buf.write("  HARFNET-VER3 | APPENDIX: Balanced Test-Set Evaluation\n")
    buf.write("=" * 72 + "\n")
    buf.write(f"  Timestamp    : {ts}\n")
    buf.write(f"  Checkpoint   : harfnet_ver3_kfold_20260430_225114_fold5.pt\n")
    buf.write(f"  Eval mode    : {mode_tag}\n")
    buf.write(f"  Original N   : {n_original}  (full 20% test split)\n")
    buf.write(f"  Balanced N   : {n_balanced}  "
              f"({n_balanced // 4} per class × 4 classes)\n")
    buf.write("=" * 72 + "\n\n")

    # ── Human vs AI ──────────────────────────────────────────
    buf.write("━━━ [A] Human vs AI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    buf.write(classification_report(yh, ph, target_names=["Human", "AI"],
                                    digits=4, zero_division=0))

    # ── Real vs Fake ─────────────────────────────────────────
    buf.write("━━━ [B] Real vs Fake ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    buf.write(classification_report(yr, pr, target_names=["Real", "Fake"],
                                    digits=4, zero_division=0))

    # ── 4-class ──────────────────────────────────────────────
    buf.write("━━━ [C] 4-class (HR / HF / AR / AF) ━━━━━━━━━━━━━━━━━━━━━━━\n")
    buf.write(classification_report(yt, yp, target_names=CLASS_NAMES,
                                    digits=4, zero_division=0))

    # ── Per-class detail with AUC ────────────────────────────
    buf.write("━━━ [D] Per-class Detail (Precision / Recall / F1 / AUC-OvR)\n")
    buf.write(per_class_detail(yt, yp, proba) + "\n\n")

    # ── Confusion Matrix ─────────────────────────────────────
    buf.write("━━━ [E] Confusion Matrix (rows=True, cols=Pred) ━━━━━━━━━━━━\n")
    buf.write(confusion_matrix_str(yt, yp) + "\n")

    # ── Aggregate Metrics ────────────────────────────────────
    f1_4  = float(f1_score(yt, yp, average="macro", zero_division=0))
    f1_ha = float(f1_score(yh, ph, average="macro", zero_division=0))
    f1_rf = float(f1_score(yr, pr, average="macro", zero_division=0))
    acc4  = float(accuracy_score(yt, yp))

    buf.write("━━━ [F] Aggregate Metrics ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    buf.write(f"  4-way Accuracy        : {acc4:.4f}\n")
    buf.write(f"  4-way Macro F1        : {f1_4:.4f}\n")
    buf.write(f"  H/A   Macro F1        : {f1_ha:.4f}\n")
    buf.write(f"  R/F   Macro F1        : {f1_rf:.4f}\n")
    buf.write(f"  AUC  (4c macro-OVR)   : {aucm['auc_4_ovr_macro']:.4f}\n")
    buf.write(f"  AUC  (H/A)            : {aucm['auc_ha']:.4f}\n")
    buf.write(f"  AUC  (R/F)            : {aucm['auc_rf']:.4f}\n")
    buf.write("=" * 72 + "\n")

    return buf.getvalue()


# ──────────────────────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(description="HARFNET-VER3 Balanced Test Eval")
    pa.add_argument("--csv_path",    default=DEFAULT_CSV_PATH)
    pa.add_argument("--data_root",   default=DEFAULT_DATA_ROOT)
    pa.add_argument("--checkpoint",  default=DEFAULT_CKPT)
    pa.add_argument("--batch_size",  type=int,   default=BATCH)
    pa.add_argument("--num_workers", type=int,   default=NUM_WORKERS)
    pa.add_argument("--seed",        type=int,   default=SEED)
    pa.add_argument("--roberta",     default=ROBERTA)
    pa.add_argument("--clip_model",  default=CLIP_RN101)
    pa.add_argument("--max_length",  type=int,   default=MAX_LENGTH)
    pa.add_argument("--kfold_test_size", type=float, default=KFOLD_TEST_SIZE)
    pa.add_argument("--no_balance",  action="store_true",
                    help="균형화 없이 원본 test set 그대로 평가")
    args = pa.parse_args()
    set_seed(args.seed)

    # ── 토크나이저 / CLIP 전처리 로드 ──────────────────────
    print("[Setup] 토크나이저 & CLIP 전처리 로드 중 ...")
    tok  = RobertaTokenizerFast.from_pretrained(args.roberta)
    _, prep = openai_clip.load(args.clip_model, device="cpu")

    # ── 전체 데이터셋 로드 & test 인덱스 복원 ──────────────
    print("[Setup] 전체 데이터셋 로드 & 20% test 인덱스 복원 중 ...")
    full_ds = HARFDataset(args.csv_path, args.data_root, tok, prep, args.max_length)
    N       = len(full_ds)
    y_all   = np.array(full_ds.labels)
    idx_all = np.arange(N)

    _, te_idx, _, _ = train_test_split(
        idx_all, y_all,
        test_size=args.kfold_test_size,
        stratify=y_all,
        random_state=args.seed,
    )
    te_labels = [full_ds.labels[i] for i in te_idx]

    print(f"\n[TestSet] 전체 test 샘플 수: {len(te_idx)}")
    for ci, cn in enumerate(CLASS_NAMES):
        cnt = sum(1 for l in te_labels if l == ci)
        print(f"  {cn}: {cnt}")

    # ── 클래스 균형화 (서브샘플링) ─────────────────────────
    if not args.no_balance:
        bal_local_idx = balance_indices_by_class(te_labels, args.seed)
        # bal_local_idx는 te_idx 내 위치 → 원본 인덱스로 변환
        eval_idx = [te_idx[i] for i in bal_local_idx]
        mode_tag = "Balanced (under-sampling to min-class)"
    else:
        eval_idx = te_idx.tolist()
        mode_tag = "Original (no balancing)"

    n_original = len(te_idx)
    n_balanced = len(eval_idx)

    # ── DataLoader 구성 ─────────────────────────────────────
    eval_ds = HARFDataset(args.csv_path, args.data_root, tok, prep,
                           args.max_length, indices=eval_idx)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate_batch,
                         pin_memory=torch.cuda.is_available())

    # ── 모델 로드 ───────────────────────────────────────────
    print(f"\n[Model] 체크포인트 로드: {args.checkpoint}")
    model = HARFNETver3(roberta_name=args.roberta, clip_name=args.clip_model)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"[Model] 로드 완료 → {DEVICE}")

    # ── 추론 ───────────────────────────────────────────────
    print(f"\n[Eval] {mode_tag}")
    print(f"[Eval] 평가 샘플 수: {n_balanced}")
    yt, yp, proba = collect_predictions(model, eval_dl, DEVICE)

    # ── 리포트 생성 ─────────────────────────────────────────
    report = build_report(yt, yp, proba, mode_tag, n_original, n_balanced)
    print("\n" + report)

    # ── 저장 ───────────────────────────────────────────────
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "balanced_test" if not args.no_balance else "original_test"
    rpath = os.path.join(RESULT_DIR, f"harfnet_ver3_{stem}_eval_{ts}.txt")
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Save] 리포트 저장: {rpath}")

    # ── 예측 CSV 저장 ───────────────────────────────────────
    cpath = os.path.join(RESULT_DIR, f"harfnet_ver3_{stem}_predictions_{ts}.csv")
    pd.DataFrame({
        "true_label":  yt,
        "pred_label":  yp,
        "true_class":  [CLASS_NAMES[i] for i in yt],
        "pred_class":  [CLASS_NAMES[i] for i in yp],
        **{f"prob_{cn}": proba[:, ci] for ci, cn in enumerate(CLASS_NAMES)},
    }).to_csv(cpath, index=False)
    print(f"[Save] 예측 CSV 저장: {cpath}")


if __name__ == "__main__":
    main()