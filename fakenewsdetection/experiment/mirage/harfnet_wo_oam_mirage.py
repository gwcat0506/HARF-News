"""
HARFNET-VER3 → MiRAGeNews 이진탐지 (OAM 완전 제거판)
=======================================================
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, roc_auc_score, confusion_matrix,
)
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

# 이진 손실 가중치 (OAM 제거로 lambda_oa / lambda_af_bce 미사용)
LAMBDA_FAKE_BCE  = 0.30
LAMBDA_AUTH_BCE  = 0.15

BILINEAR_RANK = 128
PROJ_DIM      = 128

ROBERTA    = "roberta-base"
CLIP_RN101 = "RN101"
MAX_LENGTH = 128

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_FND_ROOT = os.path.abspath(os.path.join(_EXP_DIR, ".."))
RESULT_DIR = os.path.join(_EXP_DIR, "result")
CHECKPOINT_DIR = os.path.join(_FND_ROOT, "checkpoint")

HF_DATASET_ID = "anson-huang/mirage-news"

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_TAG = "HARFNET-VER3-NO-OAM-MIRAGE"
BINARY_CLASS_NAMES = ["Real", "AI-Fake"]

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
    if os.environ.get("MIRAGE_NO_TQDM", "").lower() in ("1","true","yes"): return it
    kw.setdefault("file", sys.stderr); kw.setdefault("dynamic_ncols", True)
    try: sys.stdout.flush(); return tqdm(it, **kw)
    except Exception: return it


# ─────────────────────────────────────────────────────────────
# 3. EarlyStopping
# ─────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE,
                 min_delta=EARLY_STOP_MIN_DELTA, mode="max"):
        self.patience=patience; self.min_delta=min_delta
        self.mode=mode; self.best=None; self.counter=0

    def step(self, v: float) -> bool:
        if self.best is None: self.best=v; return False
        imp = (v-self.best > self.min_delta if self.mode=="max"
               else self.best-v > self.min_delta)
        if imp: self.best=v; self.counter=0; return False
        self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────
# 4. MiRAGeNewsDataset
# ─────────────────────────────────────────────────────────────
class MiRAGeNewsDataset(Dataset):
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
            "has_image":        torch.tensor(1.0),
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
# 5. 공용 서브모듈
# ─────────────────────────────────────────────────────────────
class GatedPooling(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.score = nn.Linear(dim, 1)
    def forward(self, x, mask=None):
        s = self.score(x).squeeze(-1)
        if mask is not None: s = s.masked_fill(~mask, -1e4)
        return (torch.softmax(s,-1).unsqueeze(-1)*x).sum(1)

class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, d_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,d_hid), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_hid,d_out))
    def forward(self, x): return self.net(x)


# ─────────────────────────────────────────────────────────────
# 6. AuthorshipModule  (원본 그대로)
# ─────────────────────────────────────────────────────────────
class AuthorshipModule(nn.Module):
    def __init__(self, d, n_heads=8, dropout=0.1):
        super().__init__()
        self.style_self_attn = nn.MultiheadAttention(d,n_heads,dropout=dropout,batch_first=True)
        self.style_var_proj  = nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,d))
        self.style_pool      = GatedPooling(d)
        self.style_ffn       = FeedForward(d*2,d*2,d,dropout)
        self.style_ln        = nn.LayerNorm(d)
        self.auth_probe      = nn.Sequential(
            nn.Linear(d,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,1),nn.Sigmoid())
        self.auth_proj       = nn.Sequential(
            nn.Linear(d,d//2),nn.ReLU(),nn.Linear(d//2,PROJ_DIM))

    def forward(self, T_tok, txt_mask):
        T_sty,_      = self.style_self_attn(T_tok,T_tok,T_tok,key_padding_mask=~txt_mask)
        sr           = T_tok - T_sty
        style_var    = self.style_var_proj(sr.var(dim=1).clamp(0.))
        style_pool   = self.style_pool(sr, txt_mask)
        F_sty = self.style_ln(self.style_ffn(torch.cat([style_pool,style_var],dim=-1)))
        p_auth= self.auth_probe(F_sty).squeeze(-1)
        z_auth= F.normalize(self.auth_proj(F_sty),dim=-1)
        return F_sty, p_auth, z_auth


# ─────────────────────────────────────────────────────────────
# 7. VeracityModule  (원본 그대로)
# ─────────────────────────────────────────────────────────────
class VeracityModule(nn.Module):
    def __init__(self, d, n_heads=8, dropout=0.1):
        super().__init__()
        self.txt2img_attn = nn.MultiheadAttention(d,n_heads,dropout=dropout,batch_first=True)
        self.img2txt_attn = nn.MultiheadAttention(d,n_heads,dropout=dropout,batch_first=True)
        self.gap_encoder  = nn.Sequential(nn.Linear(d*2,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(dropout))
        self.sim_proj     = nn.Sequential(nn.Linear(1,64),nn.ReLU(),nn.Linear(64,d))
        self.verac_ffn    = FeedForward(d*2,d*2,d,dropout)
        self.verac_ln     = nn.LayerNorm(d)
        self.fake_probe   = nn.Sequential(
            nn.Linear(d,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,1),nn.Sigmoid())
        self.verac_proj   = nn.Sequential(nn.Linear(d,d//2),nn.ReLU(),nn.Linear(d//2,PROJ_DIM))

    def forward(self, T_tok, txt_mask, V_pat, T_cls, V_cls, has_image):
        B=T_tok.size(0); m1=has_image.unsqueeze(-1); m3=has_image.view(B,1,1); eps=1e-8
        F_t2v,_= self.txt2img_attn(T_tok,V_pat,V_pat); F_t2v=F_t2v*m3
        F_v2t,_= self.img2txt_attn(V_pat,T_tok,T_tok,key_padding_mask=~txt_mask); F_v2t=F_v2t*m3
        valid_L= txt_mask.float().sum(1,keepdim=True).clamp(1)
        gap_t  = ((T_tok-F_t2v)*txt_mask.unsqueeze(-1).float()).sum(1)/valid_L
        gap_v  = (V_pat-F_v2t).mean(1)*m1
        F_gap  = self.gap_encoder(torch.cat([gap_t,gap_v],dim=-1))
        t_n=F.normalize(T_cls,dim=-1,eps=eps); v_n=F.normalize(V_cls,dim=-1,eps=eps)
        s_sim=(t_n*v_n).sum(-1,keepdim=True).clamp(-1.,1.)*m1
        F_sim=self.sim_proj(s_sim)
        F_ver=self.verac_ln(self.verac_ffn(torch.cat([F_gap,F_sim],dim=-1)))
        p_fake=self.fake_probe(F_ver).squeeze(-1)
        z_ver=F.normalize(self.verac_proj(F_ver),dim=-1)
        return F_ver, p_fake, z_ver, s_sim.squeeze(-1)


# ─────────────────────────────────────────────────────────────
# 8. BilinearInteraction  (OAM 제거 — p_af 관련 스칼라 3개 삭제)
#
#  원본 6개 스칼라:
#    p_auth, p_fake, p_af,
#    p_auth*(1-p_af),  p_auth*p_af,  p_fake*(1-p_af)
#
#  제거 후 3개 스칼라:
#    p_auth, p_fake,
#    p_auth*(1-p_fake)   ← AR 암묵 신호(p_af 대체)
# ─────────────────────────────────────────────────────────────
class BilinearInteraction(nn.Module):
    def __init__(self, d, r=BILINEAR_RANK):
        super().__init__()
        self.auth_low    = nn.Sequential(nn.Linear(d,r),nn.LayerNorm(r),nn.GELU())
        self.ver_low     = nn.Sequential(nn.Linear(d,r),nn.LayerNorm(r),nn.GELU())
        # OAM 제거: oa_low 삭제
        self.bi_proj     = nn.Sequential(nn.Linear(r,d),nn.LayerNorm(d),nn.GELU())
        # 스칼라 3개 (p_af 관련 3개 제거)
        self.scalar_proj = nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,d))

    def forward(self, F_sty, F_ver, p_auth, p_fake, has_image):
        # OAM 제거: F_oa, p_af 인자 없음
        had = self.auth_low(F_sty) * self.ver_low(F_ver)
        scalars = torch.stack([
            p_auth * has_image,
            p_fake * has_image,
            p_auth * (1. - p_fake) * has_image,   # AR 암묵 신호
        ], dim=-1)
        return self.bi_proj(had) + self.scalar_proj(scalars)


# ─────────────────────────────────────────────────────────────
# 9. HARFNETver3_Binary  (OAM 완전 제거)
#
#  변경점:
#    - self.om 삭제
#    - forward: F_oa, p_af, oa_aux 제거
#    - BilinearInteraction 호출: F_oa, p_af 인자 제거
#    - head2 입력 차원: d*4 → d*3  (F_sty + F_ver + F_bi)
# ─────────────────────────────────────────────────────────────
class HARFNETver3_NoOAM(nn.Module):
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

        self.am = AuthorshipModule(d, n_heads, dropout)
        self.vm = VeracityModule(d, n_heads, dropout)
        # OAM 삭제: self.om 없음
        self.bi = BilinearInteraction(d, BILINEAR_RANK)

        # 입력 차원: d*3 (F_sty + F_ver + F_bi)
        self.head2 = nn.Sequential(
            nn.Linear(d*3, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 2))

    def _clip_visual_forward(self, pix, has_image):
        vis=self.visual; B=pix.size(0)
        x=vis.relu1(vis.bn1(vis.conv1(pix))); x=vis.relu2(vis.bn2(vis.conv2(x)))
        x=vis.relu3(vis.bn3(vis.conv3(x)))
        x=vis.avgpool(x)*has_image.view(B,1,1,1)
        x=vis.layer1(x); x=vis.layer2(x); x=vis.layer3(x)
        deep=vis.layer4(x); v_global=vis.attnpool(deep)
        V_pat=self.deep_proj(deep.flatten(2).transpose(1,2))*has_image.view(B,1,1)
        V_cls=self.cls_proj(v_global)*has_image.view(B,1)
        return V_pat, V_cls

    def train(self, mode=True):
        super().train(mode); self.clip_model.eval(); return self

    def forward(self, input_ids, attention_mask, pixel_values,
                has_image, clip_text_tokens=None):
        device=input_ids.device; has_image=has_image.to(device)
        T_tok=self.text_encoder(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state
        txt_mask=attention_mask.bool(); T_cls=T_tok[:,0]
        V_pat,V_cls=self._clip_visual_forward(pixel_values,has_image)

        F_sty, p_auth, z_auth = self.am(T_tok, txt_mask)
        F_ver, p_fake, z_ver, gsim = self.vm(T_tok, txt_mask, V_pat, T_cls, V_cls, has_image)

        # OAM 호출 없음
        F_bi = self.bi(F_sty, F_ver, p_auth, p_fake, has_image)

        # 헤드 입력: F_sty + F_ver + F_bi  (F_oa 제거)
        logits2 = torch.nan_to_num(
            self.head2(torch.cat([F_sty, F_ver, F_bi], dim=-1)),
            nan=0., posinf=20., neginf=-20.)

        return {
            "logits2":    logits2,
            "p_auth":     p_auth,
            "p_fake":     p_fake,
            # p_af / oa_mean / oa_std / oa_uniformity 없음
            "z_auth":     z_auth,
            "z_ver":      z_ver,
            "global_sim": gsim,
        }

    @classmethod
    def from_harf_checkpoint(cls, ckpt_path, device,
                               roberta_name=ROBERTA, clip_name=CLIP_RN101):
        """
        기존 HARFNET 체크포인트에서 인코더 가중치만 재사용.
        head4 / om(OAM) 관련 키는 모두 무시.
        """
        model = cls(roberta_name, clip_name).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        # head4, om(OverAlignModule) 키 모두 제거
        filtered = {
            k: v for k, v in state.items()
            if not k.startswith("head4.") and not k.startswith("om.")
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[{LOG_TAG}] HARFNET 체크포인트 로드: "
              f"missing={len(missing)}  unexpected={len(unexpected)}")
        if missing:
            print(f"  missing keys (샘플): {missing[:5]}")
        return model


# ─────────────────────────────────────────────────────────────
# 10. 이진 손실 함수  (OAM 관련 손실 제거)
# ─────────────────────────────────────────────────────────────
def binary_focal_loss(logits2, targets, gamma=2.0):
    n=logits2.size(0)
    cnt=torch.bincount(targets,minlength=2).float().clamp(1)
    inv=(n/cnt); inv=(inv/inv.sum()*2).clamp(max=20.)
    ce=F.cross_entropy(logits2,targets,weight=inv.to(logits2.device),reduction="none")
    return ((1.-torch.exp(-ce))**gamma*ce).mean()

def attribute_bce_loss(p, pos_mask, has_image=None):
    label = pos_mask.float()
    loss  = F.binary_cross_entropy(
        p.clamp(1e-6, 1-1e-6), label, reduction="none")
    if has_image is not None:
        return (loss * has_image).mean()
    return loss.mean()

# overalign_loss_binary  ← OAM 제거로 삭제
# af_bce_loss_binary     ← OAM 제거로 삭제

def binary_total_loss(out, y, has_image, args):
    """
    OAM 제거 후 손실:
      Focal  +  lambda_fake_bce·BCE(p_fake)  +  lambda_auth_bce·BCE(p_auth)
    (lambda_oa·OA_binary, lambda_af_bce·BCE(p_af) 제거)
    """
    loss  = binary_focal_loss(out["logits2"], y)
    loss  = loss + args.lambda_fake_bce * attribute_bce_loss(
                out["p_fake"], y == 1, has_image)
    loss  = loss + args.lambda_auth_bce * attribute_bce_loss(
                out["p_auth"], y == 1, has_image)
    return loss


# ─────────────────────────────────────────────────────────────
# 11. 학습 / 평가 루프
# ─────────────────────────────────────────────────────────────
def _fwd(model, batch, device):
    return model(batch["input_ids"].to(device),
                 batch["attention_mask"].to(device),
                 batch["pixel_values"].to(device),
                 batch["has_image"].to(device))

def run_epoch(model, loader, device, optimizer, train, epoch_idx=None, args=None):
    model.train() if train else model.eval()
    tot,ys,ps,n=0.,[],[],0
    ctx=torch.enable_grad() if train else torch.no_grad()
    it=_tqdm(loader,desc=f"[{LOG_TAG}] Epoch {epoch_idx}") if (train and epoch_idx) else loader
    with ctx:
        for batch in it:
            y  = batch["label"].to(device)
            hi = batch["has_image"].to(device)
            out = _fwd(model, batch, device)
            loss = binary_total_loss(out, y, hi, args)
            if train and optimizer:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
            pred=out["logits2"].argmax(-1)
            tot+=loss.item()*y.size(0)
            ys.extend(y.cpu().tolist()); ps.extend(pred.cpu().tolist())
            n+=y.size(0)
    n=max(n,1)
    return {
        "loss":     tot/n,
        "acc":      float(accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
        "f1_ai":    float(f1_score(ys, ps, pos_label=1, average="binary", zero_division=0)),
        "f1_real":  float(f1_score(ys, ps, pos_label=0, average="binary", zero_division=0)),
    }

def collect_predictions(model, loader, device):
    model.eval(); ys,ps,scores=[],[],[]
    with torch.no_grad():
        for batch in _tqdm(loader, desc=f"Eval ({LOG_TAG})"):
            out=_fwd(model, batch, device)
            ys.extend(batch["label"].tolist())
            ps.extend(out["logits2"].argmax(-1).cpu().tolist())
            scores.extend(torch.softmax(out["logits2"],-1)[:,1].cpu().tolist())
    return np.array(ys), np.array(ps), np.array(scores)

def eval_report_binary(yt, yp, scores, label=""):
    buf=io.StringIO()
    buf.write(f"\n=== {label} | Real vs AI-Fake ===\n")
    buf.write(classification_report(yt, yp, target_names=BINARY_CLASS_NAMES,
                                    digits=4, zero_division=0))
    try:    auc=roc_auc_score(yt, scores)
    except: auc=float("nan")
    acc=accuracy_score(yt, yp)
    f1=f1_score(yt, yp, average="macro", zero_division=0)
    f1_ai=f1_score(yt, yp, pos_label=1, average="binary", zero_division=0)
    buf.write(f"Confusion Matrix:\n{confusion_matrix(yt,yp)}\n")
    buf.write(f"[{label}] Acc={acc:.4f}  Macro-F1={f1:.4f}  "
              f"AI-Fake F1={f1_ai:.4f}  AUC={auc:.4f}\n")
    return buf.getvalue(), {"acc":acc,"f1":f1,"f1_ai":f1_ai,"auc":auc}

def _monitor_binary(model, loader, device):
    """OAM 제거: p_af / oa_mean 출력 없음"""
    model.eval()
    try:
        with torch.no_grad():
            bufs = {k: [] for k in ["lbl","pf","pauth"]}
            found = {0: False, 1: False}
            for b in loader:
                out = _fwd(model, b, device); lbl = b["label"]
                for k, v in [("lbl", lbl),
                              ("pf",    out["p_fake"].cpu()),
                              ("pauth", out["p_auth"].cpu())]:
                    bufs[k].append(v)
                for ci in range(2):
                    if (lbl==ci).any(): found[ci]=True
                if all(found.values()): break
            lbl   = torch.cat(bufs["lbl"])
            pf    = torch.cat(bufs["pf"])
            pauth = torch.cat(bufs["pauth"])
            parts = []
            for ci, cn in enumerate(BINARY_CLASS_NAMES):
                mask = (lbl == ci)
                if mask.sum() > 0:
                    parts.append(
                        f"{cn}:pauth={pauth[mask].mean():.2f},"
                        f"pf={pf[mask].mean():.2f}")
            return " | " + " / ".join(parts) if parts else ""
    except Exception: return ""


# ─────────────────────────────────────────────────────────────
# 12. 학습 헬퍼
# ─────────────────────────────────────────────────────────────
def freeze_backbones(model):
    for name, p in model.text_encoder.named_parameters():
        p.requires_grad_(any(f"encoder.layer.{i}" in name for i in (10, 11)))


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


def _write_epoch_log(rep, epoch_log: list, plot_path: str, tag: str = ""):
    rep.write("\n[Epoch Log]\n")
    rep.write("Epoch\tTrain_Loss\tVal_Loss\tVal_F1\n")
    for e in epoch_log:
        rep.write(f"{e['epoch']}\t{e['train_loss']:.4f}\t"
                  f"{e['val_loss']:.4f}\t{e['val_f1']:.4f}\n")
    title = f"{LOG_TAG}{(' · ' + tag) if tag else ''}"
    plot_epoch_curves(epoch_log, plot_path, title=title)


def train_one_run(model, tl, vl, device, args, log=None):
    epoch_log = []
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    early = EarlyStopping(args.early_stop_patience, args.early_stop_min_delta)
    best_f1, best_st = -1., None
    for ep in range(1, args.epochs+1):
        tr = run_epoch(model, tl, device, opt, True,  ep, args)
        va = run_epoch(model, vl, device, None, False, args=args)
        mon = _monitor_binary(model, vl, device)
        line = (f"[{LOG_TAG}] Epoch {ep} "
                f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
                f"val_f1={va['macro_f1']:.4f} val_acc={va['acc']:.4f}{mon}")
        print(line)
        epoch_log.append({
            "epoch":      ep,
            "train_loss": tr["loss"],
            "val_loss":   va["loss"],
            "val_f1":     va["macro_f1"],
        })
        if log is not None: log.append(line)
        if best_st is None or (va["macro_f1"]-best_f1) > args.early_stop_min_delta:
            best_f1 = va["macro_f1"]; best_st = copy.deepcopy(model.state_dict())
        if early.step(va["macro_f1"]):
            msg = f"[{LOG_TAG}] Early stopping at epoch {ep}"
            print(msg)
            if log: log.append(msg)
            break
    if best_st: model.load_state_dict(best_st)
    return model, epoch_log

def save_checkpoint(path, model, args, metrics=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model_state_dict": model.state_dict(),
               "config": vars(args),
               "saved_at": datetime.now().isoformat(timespec="seconds")}
    if metrics: payload["metrics"] = metrics
    torch.save(payload, path)
    print(f"체크포인트 저장: {path}")

def _make_dl(ds, batch_size, num_workers, shuffle, sampler=None, drop_last=False):
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              collate_fn=collate_batch,
              pin_memory=torch.cuda.is_available(),
              drop_last=drop_last)
    if sampler: kw["sampler"]=sampler; kw["shuffle"]=False
    else:       kw["shuffle"]=shuffle
    return DataLoader(ds, **kw)


# ─────────────────────────────────────────────────────────────
# 13. HuggingFace 데이터셋 로드 + 스플릿 탐지
# ─────────────────────────────────────────────────────────────
def load_hf_splits(hf_id=HF_DATASET_ID):
    print(f"[{LOG_TAG}] HuggingFace 데이터셋 로드: {hf_id}")
    ds = load_dataset(hf_id)
    actual_keys = list(ds.keys())
    print(f"[{LOG_TAG}] 실제 스플릿 목록: {actual_keys}")
    return ds

def resolve_test_splits(ds):
    actual_keys = set(ds.keys())
    matched     = [k for k in TEST_SPLIT_CANDIDATES if k in actual_keys]
    matched_set = set(matched)
    extra       = sorted([k for k in actual_keys
                          if k.startswith("test") and k not in matched_set])
    result      = matched + [e for e in extra if e not in matched_set]
    print(f"[{LOG_TAG}] 탐지된 테스트 스플릿: {result}")
    if not result:
        print(f"[{LOG_TAG}] ⚠️  test 스플릿 없음. 전체 키: {list(actual_keys)}")
    return result


# ─────────────────────────────────────────────────────────────
# 14. Official Split 실행
# ─────────────────────────────────────────────────────────────
def run_official_split(args):
    set_seed(args.seed)
    tok  = RobertaTokenizerFast.from_pretrained(args.roberta)
    _, prep = openai_clip.load(args.clip_model, device="cpu")

    hf_ds = load_hf_splits(args.hf_dataset_id)

    def _wrap(split_name):
        return MiRAGeNewsDataset(hf_ds[split_name], tok, prep, args.max_length)

    tr_ds = _wrap("train")
    sampler = make_weighted_sampler(tr_ds.labels, args.sampler_alpha)
    tl = _make_dl(tr_ds, args.batch_size, args.num_workers, True,
                  sampler=sampler, drop_last=True)
    vl = _make_dl(_wrap("validation"), args.batch_size, args.num_workers, False)

    if args.harf_ckpt:
        model = HARFNETver3_NoOAM.from_harf_checkpoint(
            args.harf_ckpt, DEVICE, args.roberta, args.clip_model)
    else:
        model = HARFNETver3_NoOAM(args.roberta, args.clip_model).to(DEVICE)
    if not args.no_freeze_encoders:
        freeze_backbones(model)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (Official Split) ######\n\n")

    log = []
    model, epoch_log = train_one_run(model, tl, vl, DEVICE, args, log)
    for line in log: rep.write(line + "\n")
    plot_path = os.path.join(RESULT_DIR, f"harfnet_no_oam_mirage_curve_{ts}.png")
    _write_epoch_log(rep, epoch_log, plot_path)

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

    rep.write("\n===== SUMMARY (avg over test splits) =====\n")
    print(f"\n[{LOG_TAG}] ===== 평균 요약 =====")
    for k in ("acc","f1","f1_ai","auc"):
        vals = [all_metrics[s][k] for s in all_metrics]
        line = (f"  {k}: {np.mean(vals):.4f}"
                f"  (min={min(vals):.4f}  max={max(vals):.4f})")
        rep.write(line + "\n"); print(line)

    save_checkpoint(
        os.path.join(CHECKPOINT_DIR, f"harfnet_no_oam_mirage_{ts}.pt"),
        model, args, all_metrics)

    path = os.path.join(RESULT_DIR, f"harfnet_no_oam_mirage_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f: f.write(rep.getvalue())
    print(f"\n리포트: {path}")


# ─────────────────────────────────────────────────────────────
# 15. KFold Split 실행
# ─────────────────────────────────────────────────────────────
def run_kfold_split(args):
    set_seed(args.seed)
    tok  = RobertaTokenizerFast.from_pretrained(args.roberta)
    _, prep = openai_clip.load(args.clip_model, device="cpu")

    hf_ds       = load_hf_splits(args.hf_dataset_id)
    base_train  = hf_ds["train"]
    train_labels = np.array([int(x) for x in base_train["label"]])
    skf = StratifiedKFold(n_splits=args.kfold_splits, shuffle=True, random_state=args.seed)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (KFold={args.kfold_splits}) ######\n\n")
    test_splits = resolve_test_splits(hf_ds)
    rep.write(f"test_splits={test_splits}\n")

    fold_results = []
    for fold_idx, (tr_idx, va_idx) in enumerate(
        skf.split(np.zeros(len(train_labels)), train_labels), start=1
    ):
        print(f"\n[{LOG_TAG}] ===== Fold {fold_idx}/{args.kfold_splits} =====")
        rep.write(f"\n\n{'#'*70}\n### Fold {fold_idx}/{args.kfold_splits}\n{'#'*70}\n")

        tr_ds = MiRAGeNewsDataset(base_train, tok, prep, args.max_length, tr_idx.tolist())
        va_ds = MiRAGeNewsDataset(base_train, tok, prep, args.max_length, va_idx.tolist())
        tl = _make_dl(tr_ds, args.batch_size, args.num_workers, True,  drop_last=True)
        vl = _make_dl(va_ds, args.batch_size, args.num_workers, False)

        if args.harf_ckpt:
            model = HARFNETver3_NoOAM.from_harf_checkpoint(
                args.harf_ckpt, DEVICE, args.roberta, args.clip_model)
        else:
            model = HARFNETver3_NoOAM(args.roberta, args.clip_model).to(DEVICE)
        if not args.no_freeze_encoders:
            freeze_backbones(model)

        log = []
        model, epoch_log = train_one_run(model, tl, vl, DEVICE, args, log)
        for line in log: rep.write(line + "\n")
        plot_path = os.path.join(
            RESULT_DIR, f"harfnet_no_oam_mirage_curve_fold{fold_idx}_{ts}.png")
        _write_epoch_log(rep, epoch_log, plot_path, tag=f"fold{fold_idx}")

        fold_metrics = {}
        for sname in test_splits:
            test_ds = MiRAGeNewsDataset(hf_ds[sname], tok, prep, args.max_length)
            el = _make_dl(test_ds, args.batch_size, args.num_workers, False)
            yt, yp, sc = collect_predictions(model, el, DEVICE)
            blk, m = eval_report_binary(yt, yp, sc, label=sname)
            print(blk); rep.write(blk)
            fold_metrics[sname] = m
        fold_results.append(fold_metrics)

        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    rep.write("\n\n===== KFOLD GLOBAL SUMMARY =====\n")
    global_metrics = {}
    for split_name in test_splits:
        global_metrics[split_name] = {}
        for metric_name in ("acc","f1","f1_ai","auc"):
            vals = [fr[split_name][metric_name] for fr in fold_results]
            global_metrics[split_name][metric_name] = float(np.mean(vals))

    for split_name in test_splits:
        line = (f"{split_name}: "
                f"Acc={global_metrics[split_name]['acc']:.4f}, "
                f"F1={global_metrics[split_name]['f1']:.4f}, "
                f"F1_AI={global_metrics[split_name]['f1_ai']:.4f}, "
                f"AUC={global_metrics[split_name]['auc']:.4f}")
        print(line); rep.write(line + "\n")

    rep.write("\n===== SUMMARY (avg over test splits, then folds) =====\n")
    for k in ("acc","f1","f1_ai","auc"):
        vals = [global_metrics[s][k] for s in test_splits]
        line = f"{k}: {float(np.mean(vals)):.4f}"
        print(line); rep.write(line + "\n")

    torch.save(
        {"config": vars(args),
         "fold_metrics": fold_results,
         "global_metrics": global_metrics,
         "saved_at": datetime.now().isoformat(timespec="seconds")},
        os.path.join(CHECKPOINT_DIR,
                     f"harfnet_no_oam_mirage_kfold{args.kfold_splits}_{ts}.pt"))

    path = os.path.join(RESULT_DIR,
                        f"harfnet_no_oam_mirage_kfold{args.kfold_splits}_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f: f.write(rep.getvalue())
    print(f"\n리포트: {path}")


# ─────────────────────────────────────────────────────────────
# 16. main
# ─────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(
        description="HARFNET-VER3 (OAM 완전 제거) → MiRAGeNews 이진탐지")

    pa.add_argument("--hf_dataset_id",  default=HF_DATASET_ID)
    pa.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    pa.add_argument("--harf_ckpt",     default=None,
                    help="기존 HARFNET 체크포인트 경로 (인코더 재활용, 선택)")

    pa.add_argument("--batch_size",   type=int,   default=BATCH)
    pa.add_argument("--epochs",       type=int,   default=EPOCHS)
    pa.add_argument("--lr",           type=float, default=LR)
    pa.add_argument("--num_workers",  type=int,   default=NUM_WORKERS)
    pa.add_argument("--seed",         type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)

    pa.add_argument("--sampler_alpha",   type=float, default=0.5)

    pa.add_argument("--roberta",    default=ROBERTA)
    pa.add_argument("--clip_model", default=CLIP_RN101)
    pa.add_argument("--max_length", type=int, default=MAX_LENGTH)
    pa.add_argument("--no_freeze_encoders", action="store_true")

    # OAM 제거로 lambda_oa / lambda_af_bce 인수 삭제
    pa.add_argument("--lambda_fake_bce", type=float, default=LAMBDA_FAKE_BCE)
    pa.add_argument("--lambda_auth_bce", type=float, default=LAMBDA_AUTH_BCE)

    pa.add_argument("--no_kfold",     action="store_true")
    pa.add_argument("--kfold_splits", type=int, default=5)
    pa.add_argument("--no_progress",  action="store_true")

    args = pa.parse_args()
    if args.no_progress: os.environ["MIRAGE_NO_TQDM"] = "1"

    print(f"\n{LOG_TAG} | KFold={not args.no_kfold} | device={DEVICE}")
    print(f"[OAM 제거] OverAlignModule 완전 삭제")
    print(f"  · self.om 삭제")
    print(f"  · BilinearInteraction 스칼라: 6개 → 3개 (p_af 관련 제거)")
    print(f"  · head2 입력 차원: d*4 → d*3")
    print(f"  · 손실: Focal + {args.lambda_fake_bce}·BCE(Fake)"
          f" + {args.lambda_auth_bce}·BCE(Auth)")
    print(f"  · from_harf_checkpoint: 'om.' 키도 추가 필터링")
    print(f"[저장] result={RESULT_DIR}")
    print(f"[저장] ckpt  ={CHECKPOINT_DIR}\n")

    if args.no_kfold:
        run_official_split(args)
    else:
        run_kfold_split(args)


if __name__ == "__main__":
    main()