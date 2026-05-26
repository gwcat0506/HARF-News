"""
HARFNET-V3 — Ablation: w/o BilinearInteraction & w/o OverAlignModule & w/o AuthorshipModule
BilinearInteraction + OverAlignModule + AuthorshipModule 모두 제거한 버전.
head4 입력: [F_ver]  (d)
제거된 손실항: lambda_oa·OA_Loss, lambda_af_bce·AF_Cond_BCE, lambda_auth_bce·BCE(Auth)
"""

from __future__ import annotations

import argparse, copy, gc, io, os, random, sys, warnings
from datetime import datetime
from typing import Dict, List

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import clip as openai_clip
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizerFast

# ─────────────────────────────────────────────────────────────
# 1. 상수
# ─────────────────────────────────────────────────────────────
BATCH                = 16
EPOCHS               = 10
LR                   = 1e-4
NUM_WORKERS          = 4
EARLY_STOP_PATIENCE  = 4
EARLY_STOP_MIN_DELTA = 1e-4
KFOLD_SPLITS         = 5
KFOLD_TEST_SIZE      = 0.2
KFOLD_SHUFFLE        = True
KFOLD_RANDOM_STATE   = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2
SEED = 42

LAMBDA_FAKE_BCE  = 0.30
# LAMBDA_OA      미사용 (OverAlignModule 제거)
# LAMBDA_AUTH_BCE 미사용 (AuthorshipModule 제거)
LAMBDA_CON       = 0.05
# LAMBDA_AF_BCE  미사용 (OverAlignModule 제거)

PROJ_DIM      = 128
TEMPERATURE   = 0.07

ROBERTA    = "roberta-base"
CLIP_RN101 = "RN101"
MAX_LENGTH = 128

_SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR        = os.path.join(_SCRIPT_DIR, "results")
_FND_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
DEFAULT_DATA_ROOT = _FND_ROOT
DEFAULT_CSV_PATH  = os.path.join(_FND_ROOT, "HARFM.csv")
CHECKPOINT_DIR    = os.path.join(_SCRIPT_DIR, "checkpoints")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASS_NAMES: List[str]      = ["HR", "HF", "AR", "AF"]
CLASS2IDX:   Dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
LOG_TAG   = "HARFNET-VER3-woBilinear-woOA-woAM"
LOG_BRAND = "HARFNET-VER3-woBilinear-woOA-woAM"

os.makedirs(RESULT_DIR, exist_ok=True)


def set_seed(s: int = SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed()


# ─────────────────────────────────────────────────────────────
# 2. 유틸
# ─────────────────────────────────────────────────────────────
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


def _tqdm(it, **kw):
    if os.environ.get("VER3_NO_TQDM", "").lower() in ("1","true","yes"): return it
    kw.setdefault("file", sys.stderr); kw.setdefault("dynamic_ncols", True)
    try: sys.stdout.flush(); return tqdm(it, **kw)
    except Exception: return it


def loading_preamble(csv_path: str, data_root: str) -> str:
    print(f"[{LOG_BRAND}] Loading HARFM.csv ...")
    df = pd.read_csv(csv_path,
                     usecols=["final_headline","image_path","4_way_label"],
                     low_memory=False)
    df = df.dropna(subset=["final_headline","4_way_label"])
    for c in ["final_headline","image_path","4_way_label"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["_y"] = df["4_way_label"].map(CLASS2IDX)
    df = df.dropna(subset=["_y"]).drop(columns=["_y"]).reset_index(drop=True)
    n0 = len(df)
    df = harf_filter_multimodal_only(df, data_root)
    n1 = len(df)
    if n1 == 0: raise ValueError("유효 행 없음")
    y  = df["4_way_label"].map(CLASS2IDX).astype(int)
    lc = {i: int((y == i).sum()) for i in range(4)}
    print(f"[{LOG_BRAND}] {n0} → {n1} | label: {lc}")
    buf = io.StringIO(); buf.write(f"Dataset: {n0}→{n1}\nLabel: {lc}\n\n")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# 3. EarlyStopping
# ─────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE,
                 min_delta=EARLY_STOP_MIN_DELTA, mode="max"):
        self.patience = patience; self.min_delta = min_delta
        self.mode = mode; self.best = None; self.counter = 0

    def step(self, v: float) -> bool:
        if self.best is None: self.best = v; return False
        imp = (v - self.best > self.min_delta if self.mode == "max"
               else self.best - v > self.min_delta)
        if imp: self.best = v; self.counter = 0; return False
        self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────
# 4. Dataset
# ─────────────────────────────────────────────────────────────
class HARFDataset(Dataset):
    def __init__(self, csv_path, data_root, tokenizer, clip_preprocess,
                 max_length=MAX_LENGTH, indices=None):
        super().__init__()
        self.data_root       = os.path.abspath(data_root)
        self.tokenizer       = tokenizer
        self.clip_preprocess = clip_preprocess
        self.max_length      = max_length
        df = pd.read_csv(csv_path,
                         usecols=["final_headline","image_path","4_way_label"])
        df = df.dropna(subset=["final_headline","4_way_label"])
        for c in ["final_headline","image_path","4_way_label"]:
            df[c] = df[c].fillna("").astype(str).str.strip()
        df["_y"] = df["4_way_label"].map(CLASS2IDX)
        df = df.dropna(subset=["_y"]).drop(columns=["_y"]).reset_index(drop=True)
        df = harf_filter_multimodal_only(df, self.data_root)
        if len(df) == 0: raise ValueError("유효 행 없음")
        self.texts  = df["final_headline"].tolist()
        self.paths  = df["image_path"].tolist()
        self.labels = [CLASS2IDX[x] for x in df["4_way_label"].tolist()]
        if indices is not None:
            self.texts  = [self.texts[i]  for i in indices]
            self.paths  = [self.paths[i]  for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self): return len(self.labels)

    def _load_pil(self, rel):
        if not rel: return Image.new("RGB",(224,224),(127,127,127)), False
        full = resolve_image_path(rel, self.data_root)
        if not full or not os.path.isfile(full):
            return Image.new("RGB",(224,224),(127,127,127)), False
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
    keys = ("input_ids","attention_mask","pixel_values",
            "clip_text_tokens","has_image","label")
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def make_weighted_sampler(labels: List[int], alpha: float = 0.5):
    cnt = pd.Series(labels).value_counts().to_dict()
    w   = [(1.0 / cnt[l]) ** alpha for l in labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


# ─────────────────────────────────────────────────────────────
# 5. 공통 서브모듈
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# 6. AuthorshipModule → 제거됨
# 7. OverAlignModule  → 제거됨
# 8. BilinearInteraction → 제거됨
# head4 입력 차원: d (F_ver만 사용)
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# 9. VeracityModule
# ─────────────────────────────────────────────────────────────
class VeracityModule(nn.Module):
    def __init__(self, d: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.txt2img_attn = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True)
        self.img2txt_attn = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True)
        self.gap_encoder  = nn.Sequential(
            nn.Linear(d*2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.sim_proj     = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, d))
        self.verac_ffn    = FeedForward(d*2, d*2, d, dropout)
        self.verac_ln     = nn.LayerNorm(d)
        self.fake_probe   = nn.Sequential(
            nn.Linear(d, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 1), nn.Sigmoid())
        self.verac_proj   = nn.Sequential(
            nn.Linear(d, d//2), nn.ReLU(), nn.Linear(d//2, PROJ_DIM))

    def forward(self, T_tok, txt_mask, V_pat, T_cls, V_cls, has_image):
        B   = T_tok.size(0)
        m1  = has_image.unsqueeze(-1)
        m3  = has_image.view(B, 1, 1)
        eps = 1e-8

        F_t2v, _ = self.txt2img_attn(T_tok, V_pat, V_pat)
        F_t2v    = F_t2v * m3
        F_v2t, _ = self.img2txt_attn(
            V_pat, T_tok, T_tok, key_padding_mask=~txt_mask)
        F_v2t    = F_v2t * m3

        valid_L = txt_mask.float().sum(1, keepdim=True).clamp(1)
        gap_t   = ((T_tok - F_t2v) * txt_mask.unsqueeze(-1).float()).sum(1) / valid_L
        gap_v   = (V_pat - F_v2t).mean(1) * m1
        F_gap   = self.gap_encoder(torch.cat([gap_t, gap_v], dim=-1))

        t_n     = F.normalize(T_cls, dim=-1, eps=eps)
        v_n     = F.normalize(V_cls, dim=-1, eps=eps)
        s_sim   = (t_n * v_n).sum(-1, keepdim=True).clamp(-1., 1.) * m1
        F_sim   = self.sim_proj(s_sim)

        F_ver   = self.verac_ln(self.verac_ffn(torch.cat([F_gap, F_sim], dim=-1)))
        p_fake  = self.fake_probe(F_ver).squeeze(-1)
        z_ver   = F.normalize(self.verac_proj(F_ver), dim=-1)

        return F_ver, p_fake, z_ver, s_sim.squeeze(-1)


# ─────────────────────────────────────────────────────────────
# 10. 메인 모델
# ─────────────────────────────────────────────────────────────
class HARFNETver3_woBilinear_woOA_woAM(nn.Module):
    def __init__(self, roberta_name=ROBERTA, clip_name=CLIP_RN101,
                 n_heads=8, dropout=0.1):
        super().__init__()
        d = 768
        self.text_encoder = RobertaModel.from_pretrained(
            roberta_name, add_pooling_layer=False)
        _clip, _ = openai_clip.load(clip_name, device="cpu")
        self.clip_model = _clip.float(); self.clip_model.eval()
        for p in self.clip_model.parameters(): p.requires_grad_(False)
        self.visual    = self.clip_model.visual
        clip_edim      = self.visual.attnpool.c_proj.out_features
        self.deep_proj = nn.Linear(2048, d)
        self.cls_proj  = nn.Linear(clip_edim, d)

        # AuthorshipModule 없음
        self.vm = VeracityModule(d, n_heads, dropout)
        # OverAlignModule 없음
        # BilinearInteraction 없음

        # head4: d (F_ver만 사용)
        self.head4 = nn.Sequential(
            nn.Linear(d, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 4))

    def _clip_visual_forward(self, pix, has_image):
        vis = self.visual; B = pix.size(0)
        m4d = has_image.view(B, 1, 1, 1)
        x   = vis.relu1(vis.bn1(vis.conv1(pix)))
        x   = vis.relu2(vis.bn2(vis.conv2(x)))
        x   = vis.relu3(vis.bn3(vis.conv3(x)))
        x   = vis.avgpool(x) * m4d
        x   = vis.layer1(x); x = vis.layer2(x); x = vis.layer3(x)
        deep     = vis.layer4(x)
        v_global = vis.attnpool(deep)
        ms       = has_image.view(B, 1, 1)
        V_pat    = self.deep_proj(deep.flatten(2).transpose(1, 2)) * ms
        V_cls    = self.cls_proj(v_global) * has_image.view(B, 1)
        return V_pat, V_cls

    def train(self, mode=True):
        super().train(mode); self.clip_model.eval(); return self

    def forward(self, input_ids, attention_mask, pixel_values,
                has_image, clip_text_tokens=None):
        device    = input_ids.device
        has_image = has_image.to(device)

        T_tok    = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask).last_hidden_state
        txt_mask = attention_mask.bool()
        T_cls    = T_tok[:, 0]

        V_pat, V_cls = self._clip_visual_forward(pixel_values, has_image)

        # AuthorshipModule 호출 없음
        F_ver, p_fake, z_ver, global_sim = self.vm(
            T_tok, txt_mask, V_pat, T_cls, V_cls, has_image)
        # OverAlignModule 호출 없음

        # F_ver만으로 분류 (BilinearInteraction / F_sty 없음)
        logits4 = torch.nan_to_num(
            self.head4(F_ver),
            nan=0., posinf=20., neginf=-20.)

        return {
            "logits4":    logits4,
            "p_fake":     p_fake,
            "z_ver":      z_ver,
            "global_sim": global_sim,
            # p_auth / z_auth / p_af / oa_* 없음
        }


# ─────────────────────────────────────────────────────────────
# 11. 손실 함수
# ─────────────────────────────────────────────────────────────
def focal_loss(logits, targets, gamma=2.0):
    n   = logits.size(0)
    cnt = torch.bincount(targets, minlength=4).float().clamp(1)
    inv = (n / cnt); inv = (inv / inv.sum() * 4).clamp(max=30.)
    ce  = F.cross_entropy(logits, targets,
                          weight=inv.to(logits.device), reduction="none")
    pt  = torch.exp(-ce)
    return ((1. - pt) ** gamma * ce).mean()


def attribute_bce_loss(p, positive_mask, has_image):
    label = positive_mask.float()
    loss  = F.binary_cross_entropy(
        p.clamp(1e-6, 1-1e-6), label, reduction="none")
    return (loss * has_image).mean()


def axis_supcon_loss(z, axis_label, temperature=TEMPERATURE):
    B = z.size(0); device = z.device
    if B < 2: return z.sum() * 0.
    same   = (axis_label.unsqueeze(0) == axis_label.unsqueeze(1))
    self_m = torch.eye(B, dtype=torch.bool, device=device)
    pos_m  = same & ~self_m
    if pos_m.sum() == 0: return z.sum() * 0.
    sim    = torch.matmul(z, z.T) / temperature
    sim_s  = sim - sim.detach().max(1, keepdim=True).values
    exp_s  = torch.exp(sim_s).masked_fill(self_m, 0.)
    log_p  = sim_s - torch.log(exp_s.sum(1, keepdim=True).clamp(1e-8))
    n_pos  = pos_m.sum(1).clamp(1).float()
    return (-(log_p * pos_m).sum(1) / n_pos).mean()


def total_loss(out, y, has_image, args):
    """
    OA_Loss, AF_Cond_BCE, BCE(Auth) 모두 제거됨.
    활성 손실: Focal + BCE(Fake) + SupCon
    """
    loss = focal_loss(out["logits4"], y)

    fake_pos = (y % 2 == 1)
    loss = loss + args.lambda_fake_bce * attribute_bce_loss(
        out["p_fake"], fake_pos, has_image)

    # BCE(Auth) 제거 — AuthorshipModule 없음

    loss = loss + args.lambda_con * axis_supcon_loss(
        out["z_ver"], (y % 2).long())

    return loss


# ─────────────────────────────────────────────────────────────
# 12. 학습 / 평가
# ─────────────────────────────────────────────────────────────
def _fwd(model, batch, device):
    kw = {}
    if "clip_text_tokens" in batch:
        kw["clip_text_tokens"] = batch["clip_text_tokens"].to(device)
    return model(batch["input_ids"].to(device),
                 batch["attention_mask"].to(device),
                 batch["pixel_values"].to(device),
                 batch["has_image"].to(device), **kw)


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
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            pred = out["logits4"].argmax(-1)
            tot += loss.item() * y.size(0)
            ys.extend(y.cpu().tolist())
            ps.extend(pred.cpu().tolist())
            n += y.size(0)
    n = max(n, 1)
    return {"loss": tot/n,
            "acc4": float(np.mean(np.array(ys)==np.array(ps))),
            "macro_f1": f1_score(ys, ps, average="macro", zero_division=0)}


def collect_predictions(model, loader, device):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for batch in _tqdm(loader, desc=f"Eval ({LOG_TAG})"):
            out = _fwd(model, batch, device)
            ys.extend(batch["label"].tolist())
            ps.extend(out["logits4"].argmax(-1).cpu().tolist())
    return np.array(ys), np.array(ps)


def _monitor(model, loader, device) -> str:
    """AuthorshipModule 제거로 모니터링 항목을 p_fake/global_sim으로 축소."""
    model.eval()
    try:
        with torch.no_grad():
            bufs = {k: [] for k in ["lbl","hi","pf","gs"]}
            found = {i: False for i in range(4)}
            for b in loader:
                out = _fwd(model, b, device)
                lbl = b["label"]; hi = b["has_image"]
                bufs["lbl"].append(lbl); bufs["hi"].append(hi)
                bufs["pf"].append(out["p_fake"].cpu())
                bufs["gs"].append(out["global_sim"].cpu())
                for ci in range(4):
                    if ((lbl == ci) & (hi > 0.5)).any():
                        found[ci] = True
                if all(found.values()): break

            lbl = torch.cat(bufs["lbl"]); hi = torch.cat(bufs["hi"])
            pf  = torch.cat(bufs["pf"]); gs = torch.cat(bufs["gs"])

            parts = []
            for ci, cn in enumerate(CLASS_NAMES):
                mask = (lbl == ci) & (hi > 0.5)
                if mask.sum() > 0:
                    parts.append(
                        f"{cn}:"
                        f"pf={pf[mask].mean():.2f},"
                        f"gs={gs[mask].mean():.3f}")
            return " | " + " / ".join(parts) if parts else ""
    except Exception:
        return ""


def metrics_from_4way(yt, yp):
    yh, ph = yt//2, yp//2; yr, pr = yt%2, yp%2
    return {"acc_ha": float(accuracy_score(yh, ph)),
            "f1_ha":  float(f1_score(yh, ph, average="macro", zero_division=0)),
            "acc_rf": float(accuracy_score(yr, pr)),
            "f1_rf":  float(f1_score(yr, pr, average="macro", zero_division=0)),
            "acc_4":  float((yt==yp).mean()),
            "f1_4":   float(f1_score(yt, yp, average="macro", zero_division=0))}


def eval_report_block(yt, yp, label=None):
    label = label or LOG_TAG
    yh, ph = yt//2, yp//2; yr, pr = yt%2, yp%2
    buf = io.StringIO()
    buf.write(f"\n=== {label} Human vs AI ===\n")
    buf.write(classification_report(yh, ph, target_names=["Human","AI"],
                                    digits=4, zero_division=0))
    buf.write(f"\n=== {label} Real vs Fake ===\n")
    buf.write(classification_report(yr, pr, target_names=["Real","Fake"],
                                    digits=4, zero_division=0))
    buf.write(f"\n=== {label} 4-class ===\n")
    buf.write(classification_report(yt, yp, target_names=CLASS_NAMES,
                                    digits=4, zero_division=0))
    m = metrics_from_4way(yt, yp)
    buf.write(f"\n[{label}] H/A F1={m['f1_ha']:.4f} | "
              f"R/F F1={m['f1_rf']:.4f} | 4-way F1={m['f1_4']:.4f}\n")
    return buf.getvalue(), m


def save_checkpoint(path, model, args, metrics=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model_state_dict": model.state_dict(),
               "config": vars(args),
               "saved_at": datetime.now().isoformat(timespec="seconds")}
    if metrics: payload["metrics"] = metrics
    torch.save(payload, path)
    print(f"체크포인트 저장: {path}")


# ─────────────────────────────────────────────────────────────
# 13. 학습 루프
# ─────────────────────────────────────────────────────────────
def freeze_backbones(model: HARFNETver3_woBilinear_woOA_woAM):
    for name, p in model.text_encoder.named_parameters():
        p.requires_grad_(any(f"encoder.layer.{i}" in name for i in (10, 11)))


def train_one_run(model, tl, vl, device, args, log=None):
    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    early = EarlyStopping(args.early_stop_patience, args.early_stop_min_delta)
    best_f1, best_st = -1., None

    for ep in range(1, args.epochs+1):
        tr  = run_epoch(model, tl, device, opt, True, ep, args)
        va  = run_epoch(model, vl, device, None, False, args=args)
        mon = _monitor(model, vl, device)
        line = (f"[{LOG_TAG}] Epoch {ep} "
                f"loss={tr['loss']:.4f} val_f1={va['macro_f1']:.4f}{mon}")
        print(line)
        if log is not None: log.append(line)

        if best_st is None or (va["macro_f1"] - best_f1) > args.early_stop_min_delta:
            best_f1 = va["macro_f1"]; best_st = copy.deepcopy(model.state_dict())
        if early.step(va["macro_f1"]):
            msg = f"[{LOG_TAG}] Early stopping at epoch {ep}"
            print(msg)
            if log: log.append(msg)
            break

    if best_st: model.load_state_dict(best_st)
    return model


# ─────────────────────────────────────────────────────────────
# 14. 헬퍼 / 실행
# ─────────────────────────────────────────────────────────────
def _ds(csv, root, tok, prep, ml, idx):
    return HARFDataset(csv, root, tok, prep, ml, idx)

def _dl(ds, bs, shuf, nw, drop=False, sampler=None):
    kw = dict(batch_size=bs, num_workers=nw, collate_fn=collate_batch,
              pin_memory=torch.cuda.is_available(), drop_last=drop)
    if sampler is not None: kw["sampler"] = sampler; kw["shuffle"] = False
    else: kw["shuffle"] = shuf
    return DataLoader(ds, **kw)

def _new_model(args, device):
    return HARFNETver3_woBilinear_woOA_woAM(
        roberta_name=args.roberta, clip_name=args.clip_model).to(device)

def split_602020(labels, seed):
    idx = np.arange(len(labels)); y = np.array(labels)
    tva, te, _, _ = train_test_split(
        idx, y, test_size=TEST_RATIO, stratify=y, random_state=seed)
    tr,  va, _, _ = train_test_split(
        tva, y[tva],
        test_size=VAL_RATIO/(TRAIN_RATIO+VAL_RATIO),
        stratify=y[tva], random_state=seed)
    return tr.tolist(), va.tolist(), te.tolist()

def _run_body(model, tl, vl, device, args, log, rep):
    model = train_one_run(model, tl, vl, device, args, log)
    for line in log: rep.write(line+"\n")
    return model


def run_single(csv, root, device, tok, prep, args, freeze, preamble=""):
    labels = _ds(csv, root, tok, prep, args.max_length, None).labels
    tr, va, te = split_602020(labels, args.seed)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO(); rep.write(preamble)
    rep.write(f"###### {LOG_TAG} (60/20/20) ######\n\n")
    tr_ds = _ds(csv, root, tok, prep, args.max_length, tr)
    tl = _dl(tr_ds, args.batch_size, True, args.num_workers, True,
             sampler=make_weighted_sampler(tr_ds.labels, args.sampler_alpha))
    vl = _dl(_ds(csv, root, tok, prep, args.max_length, va),
             args.batch_size, False, args.num_workers)
    el = _dl(_ds(csv, root, tok, prep, args.max_length, te),
             args.batch_size, False, args.num_workers)
    model = _new_model(args, device)
    if freeze: freeze_backbones(model)
    log = []; model = _run_body(model, tl, vl, device, args, log, rep)
    yt, yp = collect_predictions(model, el, device)
    blk, m = eval_report_block(yt, yp); print(blk, end=""); rep.write(blk)
    ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_checkpoint(os.path.join(
        args.checkpoint_dir,
        f"harfnet_ver3_wobilinear_wooa_woam_single_{ts2}.pt"), model, args, m)
    summ = (f"{LOG_TAG}: 4-way F1={m['f1_4']:.4f} "
            f"H/A={m['f1_ha']:.4f} R/F={m['f1_rf']:.4f}")
    print("\n"+"="*70+"\n"+summ+"\n"+"="*70)
    path = os.path.join(RESULT_DIR, f"harfnet_ver3_wobilinear_wooa_woam_single_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f: f.write(rep.getvalue())
    print(f"리포트: {path}")


def run_kfold(csv, root, device, tok, prep, args, freeze, preamble=""):
    labels = _ds(csv, root, tok, prep, args.max_length, None).labels
    idx = np.arange(len(labels)); y = np.array(labels)
    tva, te, _, _ = train_test_split(
        idx, y, test_size=args.kfold_test_size,
        stratify=y, random_state=args.seed)
    k  = args.kfold_splits
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rep = io.StringIO(); all_rep.write(preamble)
    all_rep.write(f"###### {LOG_TAG} (k={k}) ######\n")
    el = _dl(_ds(csv, root, tok, prep, args.max_length, te.tolist()),
             args.batch_size, False, args.num_workers)
    skf = StratifiedKFold(n_splits=k, shuffle=KFOLD_SHUFFLE,
                          random_state=args.kfold_random_state)
    fold_m = []
    for fold, (str_, sva_) in enumerate(skf.split(np.zeros(len(tva)), y[tva]), 1):
        print(f"\n{'='*70}\n[Fold {fold}/{k}] {LOG_BRAND}\n{'='*70}")
        fd = _ds(csv, root, tok, prep, args.max_length, tva[str_].tolist())
        tl = _dl(fd, args.batch_size, True, args.num_workers, True,
                 sampler=make_weighted_sampler(fd.labels, args.sampler_alpha))
        vl = _dl(_ds(csv, root, tok, prep, args.max_length, tva[sva_].tolist()),
                 args.batch_size, False, args.num_workers)
        model = _new_model(args, device)
        if freeze: freeze_backbones(model)
        log = []; fold_buf = io.StringIO()
        model = _run_body(model, tl, vl, device, args, log, fold_buf)
        yt, yp = collect_predictions(model, el, device)
        blk, m = eval_report_block(yt, yp); print(blk, end=""); fold_buf.write(blk)
        ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_checkpoint(os.path.join(
            args.checkpoint_dir,
            f"harfnet_ver3_wobilinear_wooa_woam_kfold_{ts2}_fold{fold}.pt"),
            model, args, m)
        summ = (f"{LOG_TAG}: 4-way F1={m['f1_4']:.4f} "
                f"H/A={m['f1_ha']:.4f} R/F={m['f1_rf']:.4f}")
        print(f"\n[Fold {fold}] {summ}")
        fold_buf.write(f"\n[Fold {fold}] {summ}\n")
        all_rep.write(f"\n\n{'#'*80}\n### FOLD {fold}/{k}\n{'#'*80}\n")
        all_rep.write(fold_buf.getvalue())
        fold_m.append(m)
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _m(k_): return float(np.mean([r[k_] for r in fold_m]))
    g = (f"{LOG_TAG}: 4-way F1={_m('f1_4'):.4f} "
         f"H/A={_m('f1_ha'):.4f} R/F={_m('f1_rf'):.4f}")
    all_rep.write(f"\n\n{'#'*80}\n### SUMMARY\n{'#'*80}\n{g}\n")
    all_rep.write("Variant\tHA_F1\tRF_F1\t4C_F1\n")
    all_rep.write(
        f"VER3-woBilinear-woOA-woAM\t{_m('f1_ha'):.4f}\t{_m('f1_rf'):.4f}\t{_m('f1_4'):.4f}\n")
    path = os.path.join(RESULT_DIR, f"harfnet_ver3_wobilinear_wooa_woam_kfold_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f: f.write(all_rep.getvalue())
    print(f"\n리포트: {path}\n{g}")


# ─────────────────────────────────────────────────────────────
# 15. main
# ─────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(description="HARFNET-VER3-woBilinear-woOA-woAM")
    pa.add_argument("--csv_path",       default=DEFAULT_CSV_PATH)
    pa.add_argument("--data_root",      default=DEFAULT_DATA_ROOT)
    pa.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    pa.add_argument("--batch_size",     type=int,   default=BATCH)
    pa.add_argument("--epochs",         type=int,   default=EPOCHS)
    pa.add_argument("--lr",             type=float, default=LR)
    pa.add_argument("--num_workers",    type=int,   default=NUM_WORKERS)
    pa.add_argument("--seed",           type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    pa.add_argument("--no_kfold",             action="store_true")
    pa.add_argument("--kfold_splits",         type=int,   default=KFOLD_SPLITS)
    pa.add_argument("--kfold_test_size",      type=float, default=KFOLD_TEST_SIZE)
    pa.add_argument("--kfold_random_state",   type=int,   default=KFOLD_RANDOM_STATE)
    pa.add_argument("--roberta",        default=ROBERTA)
    pa.add_argument("--clip_model",     default=CLIP_RN101)
    pa.add_argument("--max_length",     type=int,   default=MAX_LENGTH)
    pa.add_argument("--sampler_alpha",  type=float, default=0.5)
    # 호환성 유지를 위해 파싱은 하되 모두 미사용
    pa.add_argument("--lambda_fake_bce", type=float, default=LAMBDA_FAKE_BCE)
    pa.add_argument("--lambda_oa",       type=float, default=0.0)   # 미사용
    pa.add_argument("--lambda_auth_bce", type=float, default=0.0)   # 미사용 (AM 제거)
    pa.add_argument("--lambda_con",      type=float, default=LAMBDA_CON)
    pa.add_argument("--lambda_af_bce",   type=float, default=0.0)   # 미사용
    pa.add_argument("--oa_target_ar",     type=float, default=0.6)  # 미사용
    pa.add_argument("--oa_target_hr_std", type=float, default=0.15) # 미사용
    pa.add_argument("--no_freeze_encoders", action="store_true")
    pa.add_argument("--no-progress",        action="store_true")
    args = pa.parse_args()

    if args.no_progress: os.environ["VER3_NO_TQDM"] = "1"
    freeze = not args.no_freeze_encoders
    set_seed(args.seed)

    preamble = loading_preamble(args.csv_path, args.data_root)
    print(f"\n{LOG_BRAND} | KFold={not args.no_kfold} | device={DEVICE}")
    print(f"[손실] Focal"
          f" + {args.lambda_fake_bce}·BCE(Fake)"
          f" + {args.lambda_con}·SupCon"
          f"  [OA_Loss/AF_BCE/BCE(Auth) 비활성]\n")

    tok = RobertaTokenizerFast.from_pretrained(args.roberta)
    _, prep = openai_clip.load(args.clip_model, device="cpu")

    fn = run_single if args.no_kfold else run_kfold
    fn(args.csv_path, args.data_root, DEVICE, tok, prep, args, freeze, preamble)


if __name__ == "__main__":
    main()