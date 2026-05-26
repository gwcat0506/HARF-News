"""
HARFNET → MiRAGeNews 이진탐지용
레이블 매핑:
  0 = Real  (실제 이미지 + 실제 캡션)
  1 = AI-Fake (AI 이미지 + AI 캡션)
  신호 기대값:
    label=1 : p_auth↑  p_fake↑  p_af↑  oa_mean↑
    label=0 : p_auth↓  p_fake↓  p_af↓  oa_mean↓
"""

from __future__ import annotations

import argparse, copy, gc, io, os, random, sys, warnings
from datetime import datetime
from typing import List, Optional

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

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

# 이진 손실 가중치
LAMBDA_FAKE_BCE  = 0.30   
LAMBDA_OA        = 0.30   
LAMBDA_AUTH_BCE  = 0.15   # p_auth: label=1 → 1 (AI저작)
LAMBDA_AF_BCE    = 0.20  

BILINEAR_RANK = 128
PROJ_DIM      = 128

ROBERTA    = "roberta-base"
CLIP_RN101 = "RN101"
MAX_LENGTH = 128

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FND_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
RESULT_DIR = os.path.join(_SCRIPT_DIR, "mirage", "result")
CHECKPOINT_DIR = os.path.join(_FND_ROOT, "checkpoint")

HF_DATASET_ID = "anson-huang/mirage-news"

TEST_SPLIT_CANDIDATES = [
    "test_midjourneyv5",     # 원 코드 표기
    "test_midjourney_v5",    # 언더스코어 변형
    "test_midjourneyV5",     # 대문자 변형
    "test_dalle3",
    "test_dalle_3",
    "test_sdxl",
    "test_bbc",
    "test_cnn",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_TAG = "HARFNET-MIRAGE"
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
    """
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
            # MiRAGeNews는 항상 이미지 존재 → has_image 항상 1.0
            "has_image":        torch.tensor(1.0),
            "label":            torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_batch(batch):
    keys = ("input_ids","attention_mask","pixel_values",
            "clip_text_tokens","has_image","label")
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def make_weighted_sampler(labels: List[int], alpha: float = 0.5):
    """클래스 불균형 대응용 WeightedRandomSampler."""
    cnt = pd.Series(labels).value_counts().to_dict()
    w   = [(1.0 / cnt[l]) ** alpha for l in labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


# ─────────────────────────────────────────────────────────────
# 5–9. 서브모듈
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

class OverAlignModule(nn.Module):
    """af_head 입력 d+3, patch_text_sim은 clamp(min=0.)로 음수 방지."""
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.oa_scalar_proj = nn.Sequential(
            nn.Linear(6,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,d))
        self.oa_patch_pool  = nn.Sequential(
            nn.Linear(d,d//2),nn.LayerNorm(d//2),nn.GELU(),nn.Linear(d//2,d))
        self.oa_ffn = nn.Sequential(
            nn.Linear(d*2,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(dropout))
        # af_head 입력: [F_oa(d), p_fake(1), oa_mean(1), oa_uni(1)] => d+3
        self.af_head = nn.Sequential(
            nn.Linear(d+3, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, V_pat, T_cls, global_sim, p_fake, p_auth, has_image):
        B=V_pat.size(0); m1=has_image.unsqueeze(-1); eps=1e-8
        t_n=F.normalize(T_cls,dim=-1,eps=eps).unsqueeze(1).expand_as(V_pat)
        v_n=F.normalize(V_pat,dim=-1,eps=eps)
        patch_text_sim=(v_n*t_n).sum(-1)*has_image.unsqueeze(-1)
        patch_text_sim=patch_text_sim.clamp(min=0.)
        oa_mean=patch_text_sim.mean(1); oa_std=patch_text_sim.std(1).clamp(0.)
        oa_max=patch_text_sim.max(1).values; oa_min=patch_text_sim.min(1).values
        oa_uni=oa_mean/((oa_max-oa_min).clamp(min=eps))
        F_oa_s=self.oa_scalar_proj(torch.stack([oa_mean,oa_std,oa_uni,oa_max,oa_min,global_sim],dim=-1))
        ow=torch.softmax(patch_text_sim*has_image.unsqueeze(-1),dim=1)
        F_oa_p=self.oa_patch_pool((ow.unsqueeze(-1)*V_pat).sum(1))
        F_oa=self.oa_ffn(torch.cat([F_oa_s,F_oa_p],dim=-1))*m1
        p_af=self.af_head(torch.cat([
            F_oa,
            p_fake.unsqueeze(-1),
            oa_mean.unsqueeze(-1),
            oa_uni.clamp(max=50.).unsqueeze(-1),
        ], dim=-1)).squeeze(-1)*has_image
        return F_oa, p_af, {"oa_mean":oa_mean,"oa_std":oa_std,
                             "oa_uniformity":oa_uni,"p_af":p_af}

class BilinearInteraction(nn.Module):
    """p_ar 없이 p_auth*(1-p_af)로 AR 신호 암묵 표현."""
    def __init__(self, d, r=BILINEAR_RANK):
        super().__init__()
        self.auth_low    = nn.Sequential(nn.Linear(d,r),nn.LayerNorm(r),nn.GELU())
        self.ver_low     = nn.Sequential(nn.Linear(d,r),nn.LayerNorm(r),nn.GELU())
        self.oa_low      = nn.Sequential(nn.Linear(d,r),nn.LayerNorm(r),nn.GELU())
        self.bi_proj     = nn.Sequential(nn.Linear(r,d),nn.LayerNorm(d),nn.GELU())
        self.scalar_proj = nn.Sequential(nn.Linear(6,128),nn.ReLU(),nn.Linear(128,d))
    def forward(self, F_sty, F_ver, F_oa, p_auth, p_fake, p_af, has_image):
        m=has_image.unsqueeze(-1)
        had=self.auth_low(F_sty)*self.ver_low(F_ver)*self.oa_low(F_oa)
        scalars=torch.stack([
            p_auth*has_image,
            p_fake*has_image,
            p_af*has_image,
            p_auth*(1.-p_af)*has_image,   # AR 암묵 신호 (p_ar 대체)
            p_auth*p_af*has_image,         # AF 교차
            p_fake*(1.-p_af)*has_image,    # HF 교차
        ],dim=-1)
        return self.bi_proj(had)+self.scalar_proj(scalars)*m


# ─────────────────────────────────────────────────────────────
# 10. 메인 모델
# ─────────────────────────────────────────────────────────────
class HARFNETver3_Binary(nn.Module):
    """이진 분류용 HARFNET (head4 → head2)."""
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
        self.om = OverAlignModule(d, dropout)
        self.bi = BilinearInteraction(d, BILINEAR_RANK)
        self.head2 = nn.Sequential(
            nn.Linear(d*4, 512), nn.LayerNorm(512),
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
        F_sty,p_auth,z_auth=self.am(T_tok,txt_mask)
        F_ver,p_fake,z_ver,gsim=self.vm(T_tok,txt_mask,V_pat,T_cls,V_cls,has_image)
        F_oa,p_af,oa_aux=self.om(V_pat,T_cls,gsim,p_fake,p_auth,has_image)
        F_bi=self.bi(F_sty,F_ver,F_oa,p_auth,p_fake,p_af,has_image)
        logits2=torch.nan_to_num(
            self.head2(torch.cat([F_sty,F_ver,F_oa,F_bi],dim=-1)),
            nan=0.,posinf=20.,neginf=-20.)
        return {"logits2":logits2,"p_auth":p_auth,"p_fake":p_fake,
                "p_af":p_af,"z_auth":z_auth,"z_ver":z_ver,
                "global_sim":gsim,**oa_aux}

    @classmethod
    def from_harf_checkpoint(cls, ckpt_path, device,
                               roberta_name=ROBERTA, clip_name=CLIP_RN101):
        """
        HARFM 4-class 체크포인트 → head4 제외하고 인코더 가중치만 재사용.
        ★ 수정: "head4" → "head4." (dot 포함) — 안전한 prefix 필터링
        """
        model = cls(roberta_name, clip_name).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        # ★ "head4." 으로 수정: "head4_extra" 같은 다른 키 오탐 방지
        filtered = {k: v for k, v in state.items() if not k.startswith("head4.")}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[{LOG_TAG}] HARFM 체크포인트 로드: "
              f"missing={len(missing)}  unexpected={len(unexpected)}")
        if missing:
            print(f"  missing keys (샘플): {missing[:5]}")
        return model


# ─────────────────────────────────────────────────────────────
# 11. 이진 손실 함수
# ─────────────────────────────────────────────────────────────
def binary_focal_loss(logits2, targets, gamma=2.0):
    n=logits2.size(0)
    cnt=torch.bincount(targets,minlength=2).float().clamp(1)
    inv=(n/cnt); inv=(inv/inv.sum()*2).clamp(max=20.)
    ce=F.cross_entropy(logits2,targets,weight=inv.to(logits2.device),reduction="none")
    return ((1.-torch.exp(-ce))**gamma*ce).mean()

def attribute_bce_loss(p, pos_mask, has_image=None):
    """속성 BCE. has_image 마스크 선택 적용."""
    label = pos_mask.float()
    loss  = F.binary_cross_entropy(
        p.clamp(1e-6, 1-1e-6), label, reduction="none")
    if has_image is not None:
        return (loss * has_image).mean()
    return loss.mean()

def overalign_loss_binary(out, y, has_image=None):
    """
    label=1(AI-Fake) → oa_mean↑, oa_uniformity↑
    label=0(Real)    → oa_mean↓

    ★ 수정: oa_uni 정규화 시 음수 방지를 위해 clamp(0., 1.) 추가
    기존 코드: (1. - oa_uni[m1]/10.)**2  → oa_uni > 10이면 음수 → 역방향 기울기
    수정 코드: (1. - (oa_uni[m1]/10.).clamp(0., 1.))**2  → 항상 [0, 1]
    """
    oa_mean = out["oa_mean"]
    oa_uni  = out["oa_uniformity"].clamp(max=10.)
    loss    = oa_mean.sum() * 0.

    m1 = (y == 1)
    if has_image is not None:
        m1 = m1 & (has_image > 0.5)
    if m1.any():
        loss = loss + ((1. - oa_mean[m1]) ** 2).mean()
        # ★ 수정: clamp(0., 1.)으로 음수 방지
        loss = loss + ((1. - (oa_uni[m1] / 10.).clamp(0., 1.)) ** 2).mean()

    m0 = (y == 0)
    if has_image is not None:
        m0 = m0 & (has_image > 0.5)
    if m0.any():
        loss = loss + (oa_mean[m0] ** 2).mean()

    return loss

def af_bce_loss_binary(p_af, y, has_image=None):
    loss = F.binary_cross_entropy(
        p_af.clamp(1e-6, 1-1e-6), y.float(), reduction="none")
    if has_image is not None:
        return (loss * has_image).mean()
    return loss.mean()

def binary_total_loss(out, y, has_image, args):
    """이진 focal + 속성/OA/AF 보조 손실 합산."""
    loss  = binary_focal_loss(out["logits2"], y)
    loss  = loss + args.lambda_fake_bce * attribute_bce_loss(
                out["p_fake"], y==1, has_image)
    loss  = loss + args.lambda_oa * overalign_loss_binary(out, y, has_image)
    loss  = loss + args.lambda_auth_bce * attribute_bce_loss(
                out["p_auth"], y==1, has_image)
    loss  = loss + args.lambda_af_bce * af_bce_loss_binary(
                out["p_af"], y, has_image)
    return loss


# ─────────────────────────────────────────────────────────────
# 12. 학습 / 평가 루프
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
            hi = batch["has_image"].to(device)   # ★ 수정: has_image 명시적 추출
            out = _fwd(model, batch, device)
            loss = binary_total_loss(out, y, hi, args)   # ★ 수정: hi 전달
            if train and optimizer:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
            pred=out["logits2"].argmax(-1)
            tot+=loss.item()*y.size(0)
            ys.extend(y.cpu().tolist()); ps.extend(pred.cpu().tolist())
            n+=y.size(0)
    n=max(n,1)
    # run_epoch return에 추가
    return {
        "loss": tot/n,
        "acc":  float(accuracy_score(ys, ps)),
        "macro_f1":   float(f1_score(ys, ps, average="macro", zero_division=0)),
        # 이진분류에서 실질적으로 의미있는 지표들
        "f1_ai":  float(f1_score(ys, ps, pos_label=1, average="binary", zero_division=0)),
        "f1_real":float(f1_score(ys, ps, pos_label=0, average="binary", zero_division=0)),
    }

def collect_predictions(model, loader, device):
    model.eval(); ys,ps,scores=[],[],[]
    with torch.no_grad():
        for batch in _tqdm(loader,desc=f"Eval ({LOG_TAG})"):
            out=_fwd(model,batch,device)
            ys.extend(batch["label"].tolist())
            ps.extend(out["logits2"].argmax(-1).cpu().tolist())
            scores.extend(torch.softmax(out["logits2"],-1)[:,1].cpu().tolist())
    return np.array(ys), np.array(ps), np.array(scores)

def eval_report_binary(yt, yp, scores, label=""):
    buf=io.StringIO()
    buf.write(f"\n=== {label} | Real vs AI-Fake ===\n")
    buf.write(classification_report(yt,yp,target_names=BINARY_CLASS_NAMES,
                                    digits=4,zero_division=0))
    try:    auc=roc_auc_score(yt,scores)
    except: auc=float("nan")
    acc=accuracy_score(yt,yp)
    f1=f1_score(yt,yp,average="macro",zero_division=0)
    f1_ai=f1_score(yt,yp,pos_label=1,average="binary",zero_division=0)
    buf.write(f"Confusion Matrix:\n{confusion_matrix(yt,yp)}\n")
    buf.write(f"[{label}] Acc={acc:.4f}  Macro-F1={f1:.4f}  "
              f"AI-Fake F1={f1_ai:.4f}  AUC={auc:.4f}\n")
    return buf.getvalue(), {"acc":acc,"f1":f1,"f1_ai":f1_ai,"auc":auc}

def _monitor_binary(model, loader, device):
    model.eval()
    try:
        with torch.no_grad():
            bufs={k:[] for k in ["lbl","pf","paf","pauth","oam"]}
            found={0:False,1:False}
            for b in loader:
                out=_fwd(model,b,device); lbl=b["label"]
                for k,v in [("lbl",lbl),("pf",out["p_fake"].cpu()),
                             ("paf",out["p_af"].cpu()),
                             ("pauth",out["p_auth"].cpu()),
                             ("oam",out["oa_mean"].cpu())]:
                    bufs[k].append(v)
                for ci in range(2):
                    if (lbl==ci).any(): found[ci]=True
                if all(found.values()): break
            lbl=torch.cat(bufs["lbl"]); pf=torch.cat(bufs["pf"])
            paf=torch.cat(bufs["paf"]); pauth=torch.cat(bufs["pauth"])
            oam=torch.cat(bufs["oam"])
            parts=[]
            for ci,cn in enumerate(BINARY_CLASS_NAMES):
                mask=(lbl==ci)
                if mask.sum()>0:
                    parts.append(f"{cn}:pauth={pauth[mask].mean():.2f},"
                                 f"pf={pf[mask].mean():.2f},"
                                 f"paf={paf[mask].mean():.2f},"
                                 f"oam={oam[mask].mean():.3f}")
            return " | "+" / ".join(parts) if parts else ""
    except Exception: return ""


# ─────────────────────────────────────────────────────────────
# 13. 학습 헬퍼
# ─────────────────────────────────────────────────────────────
def freeze_backbones(model):
    for name,p in model.text_encoder.named_parameters():
        p.requires_grad_(any(f"encoder.layer.{i}" in name for i in (10,11)))

def train_one_run(model, tl, vl, device, args, log=None):
    opt=torch.optim.AdamW(
        filter(lambda p:p.requires_grad,model.parameters()),
        lr=args.lr,weight_decay=1e-4)
    early=EarlyStopping(args.early_stop_patience,args.early_stop_min_delta)
    best_f1,best_st=-1.,None
    for ep in range(1,args.epochs+1):
        tr=run_epoch(model,tl,device,opt,True,ep,args)
        va=run_epoch(model,vl,device,None,False,args=args)
        mon=_monitor_binary(model,vl,device)
        line=(f"[{LOG_TAG}] Epoch {ep} "
              f"loss={tr['loss']:.4f} val_f1={va['macro_f1']:.4f} "
              f"val_acc={va['acc']:.4f}{mon}")
        print(line)
        if log is not None: log.append(line)
        if best_st is None or (va["macro_f1"]-best_f1)>args.early_stop_min_delta:
            best_f1=va["macro_f1"]; best_st=copy.deepcopy(model.state_dict())
        if early.step(va["macro_f1"]):
            msg=f"[{LOG_TAG}] Early stopping at epoch {ep}"
            print(msg)
            if log: log.append(msg)
            break
    if best_st: model.load_state_dict(best_st)
    return model

def save_checkpoint(path, model, args, metrics=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload={"model_state_dict":model.state_dict(),
             "config":vars(args),
             "saved_at":datetime.now().isoformat(timespec="seconds")}
    if metrics: payload["metrics"]=metrics
    torch.save(payload,path)
    print(f"체크포인트 저장: {path}")

def _make_dl(ds, batch_size, num_workers, shuffle, sampler=None, drop_last=False):
    kw=dict(batch_size=batch_size,num_workers=num_workers,
            collate_fn=collate_batch,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last)
    if sampler: kw["sampler"]=sampler; kw["shuffle"]=False
    else:       kw["shuffle"]=shuffle
    return DataLoader(ds,**kw)


# ─────────────────────────────────────────────────────────────
# 14. HuggingFace 데이터셋 로드 + 스플릿 탐지
# ─────────────────────────────────────────────────────────────
def load_hf_splits(hf_id=HF_DATASET_ID):
    print(f"[{LOG_TAG}] HuggingFace 데이터셋 로드: {hf_id}")
    ds = load_dataset(hf_id)
    actual_keys = list(ds.keys())
    print(f"[{LOG_TAG}] 실제 스플릿 목록: {actual_keys}")
    return ds

def resolve_test_splits(ds):
    """
    ★ 수정: 하드코딩 대신 실제 HF 키와 후보 목록의 교집합으로 결정.
    후보에 없는 test_* 스플릿도 자동 포함 (완전 안전).
    """
    actual_keys = set(ds.keys())

    # 1) 후보 목록과 교집합 (순서 유지)
    matched = [k for k in TEST_SPLIT_CANDIDATES if k in actual_keys]
    matched_set = set(matched)

    # 2) 후보에 없는 test_* 스플릿 추가 (완전 자동 탐지)
    extra = sorted([k for k in actual_keys
                    if k.startswith("test") and k not in matched_set])
    result = matched + [e for e in extra if e not in matched_set]

    print(f"[{LOG_TAG}] 탐지된 테스트 스플릿: {result}")
    if not result:
        print(f"[{LOG_TAG}] ⚠️  test 스플릿 없음. 전체 키: {list(actual_keys)}")
    return result


# ─────────────────────────────────────────────────────────────
# 15. 메인 실행
# ─────────────────────────────────────────────────────────────
def run_official_split(args):
    set_seed(args.seed)
    tok=RobertaTokenizerFast.from_pretrained(args.roberta)
    _,prep=openai_clip.load(args.clip_model,device="cpu")

    hf_ds=load_hf_splits(args.hf_dataset_id)

    def _wrap(split_name):
        return MiRAGeNewsDataset(hf_ds[split_name],tok,prep,args.max_length)

    tr_ds=_wrap("train")

    sampler = make_weighted_sampler(tr_ds.labels, args.sampler_alpha)
    tl = _make_dl(tr_ds, args.batch_size, args.num_workers, True,
                  sampler=sampler, drop_last=True)

    vl=_make_dl(_wrap("validation"),args.batch_size,args.num_workers,False)

    if args.harf_ckpt:
        model=HARFNETver3_Binary.from_harf_checkpoint(
            args.harf_ckpt,DEVICE,args.roberta,args.clip_model)
    else:
        model=HARFNETver3_Binary(args.roberta,args.clip_model).to(DEVICE)
    if not args.no_freeze_encoders:
        freeze_backbones(model)

    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    rep=io.StringIO()
    rep.write(f"###### {LOG_TAG} (Official Split) ######\n\n")

    log=[]
    model=train_one_run(model,tl,vl,DEVICE,args,log)
    for line in log: rep.write(line+"\n")

    # ★ 수정: resolve_test_splits()로 실제 HF 키 기반 동적 탐지
    test_splits = resolve_test_splits(hf_ds)
    print(f"\n[{LOG_TAG}] ===== 테스트셋 평가 ({len(test_splits)}개) =====")
    rep.write("\n\n===== TEST RESULTS =====\n")
    all_metrics={}

    for sname in test_splits:
        el=_make_dl(_wrap(sname),args.batch_size,args.num_workers,False)
        yt,yp,sc=collect_predictions(model,el,DEVICE)
        blk,m=eval_report_binary(yt,yp,sc,label=sname)
        print(blk); rep.write(blk)
        all_metrics[sname]=m

    rep.write("\n===== SUMMARY (avg over test splits) =====\n")
    print(f"\n[{LOG_TAG}] ===== 평균 요약 =====")
    for k in ("acc","f1","f1_ai","auc"):
        vals=[all_metrics[s][k] for s in all_metrics]
        line=f"  {k}: {np.mean(vals):.4f}  (min={min(vals):.4f}  max={max(vals):.4f})"
        rep.write(line+"\n"); print(line)

    save_checkpoint(
        os.path.join(CHECKPOINT_DIR, f"harfnet_mirage_{ts}.pt"),
        model, args, all_metrics)

    path=os.path.join(RESULT_DIR,f"harfnet_mirage_{ts}.txt")
    with open(path,"w",encoding="utf-8") as f: f.write(rep.getvalue())
    print(f"\n리포트: {path}")


def run_kfold_split(args):
    set_seed(args.seed)
    tok = RobertaTokenizerFast.from_pretrained(args.roberta)
    _, prep = openai_clip.load(args.clip_model, device="cpu")

    hf_ds = load_hf_splits(args.hf_dataset_id)
    base_train = hf_ds["train"]
    train_labels = np.array([int(x) for x in base_train["label"]])
    skf = StratifiedKFold(n_splits=args.kfold_splits, shuffle=True, random_state=args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (KFold={args.kfold_splits}) ######\n\n")
    test_splits = resolve_test_splits(hf_ds)
    rep.write(f"test_splits={test_splits}\n")

    fold_results = []
    for fold_idx, (tr_idx, va_idx) in enumerate(
        skf.split(np.zeros(len(train_labels)), train_labels), start=1
    ):
        print(f"\n[{LOG_TAG}] ===== Fold {fold_idx}/{args.kfold_splits} =====")
        rep.write(f"\n\n{'#' * 70}\n### Fold {fold_idx}/{args.kfold_splits}\n{'#' * 70}\n")

        tr_ds = MiRAGeNewsDataset(
            base_train, tok, prep, args.max_length, indices=tr_idx.tolist()
        )
        va_ds = MiRAGeNewsDataset(
            base_train, tok, prep, args.max_length, indices=va_idx.tolist()
        )
        tl = _make_dl(tr_ds, args.batch_size, args.num_workers, True, drop_last=True)
        vl = _make_dl(va_ds, args.batch_size, args.num_workers, False)

        if args.harf_ckpt:
            model = HARFNETver3_Binary.from_harf_checkpoint(
                args.harf_ckpt, DEVICE, args.roberta, args.clip_model
            )
        else:
            model = HARFNETver3_Binary(args.roberta, args.clip_model).to(DEVICE)
        if not args.no_freeze_encoders:
            freeze_backbones(model)

        log = []
        model = train_one_run(model, tl, vl, DEVICE, args, log)
        for line in log:
            rep.write(line + "\n")

        fold_metrics = {}
        for sname in test_splits:
            test_ds = MiRAGeNewsDataset(hf_ds[sname], tok, prep, args.max_length)
            el = _make_dl(test_ds, args.batch_size, args.num_workers, False)
            yt, yp, sc = collect_predictions(model, el, DEVICE)
            blk, m = eval_report_binary(yt, yp, sc, label=sname)
            print(blk)
            rep.write(blk)
            fold_metrics[sname] = m
        fold_results.append(fold_metrics)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    rep.write("\n\n===== KFOLD GLOBAL SUMMARY =====\n")
    global_metrics = {}
    for split_name in test_splits:
        global_metrics[split_name] = {}
        for metric_name in ("acc", "f1", "f1_ai", "auc"):
            vals = [fr[split_name][metric_name] for fr in fold_results]
            global_metrics[split_name][metric_name] = float(np.mean(vals))

    for split_name in test_splits:
        line = (
            f"{split_name}: "
            f"Acc={global_metrics[split_name]['acc']:.4f}, "
            f"F1={global_metrics[split_name]['f1']:.4f}, "
            f"F1_AI={global_metrics[split_name]['f1_ai']:.4f}, "
            f"AUC={global_metrics[split_name]['auc']:.4f}"
        )
        print(line)
        rep.write(line + "\n")

    rep.write("\n===== SUMMARY (avg over test splits, then folds) =====\n")
    for k in ("acc", "f1", "f1_ai", "auc"):
        vals = [global_metrics[s][k] for s in test_splits]
        line = f"{k}: {float(np.mean(vals)):.4f}"
        print(line)
        rep.write(line + "\n")

    kfold_ckpt_path = os.path.join(
        CHECKPOINT_DIR, f"harfnet_mirage_kfold{args.kfold_splits}_{ts}.pt"
    )
    torch.save(
        {
            "config": vars(args),
            "fold_metrics": fold_results,
            "global_metrics": global_metrics,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
        kfold_ckpt_path,
    )
    print(f"체크포인트 저장: {kfold_ckpt_path}")

    path = os.path.join(RESULT_DIR, f"harfnet_mirage_kfold{args.kfold_splits}_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rep.getvalue())
    print(f"\n리포트: {path}")


# ─────────────────────────────────────────────────────────────
# 16. main
# ─────────────────────────────────────────────────────────────
def main():
    pa=argparse.ArgumentParser(description="HARFNET → MiRAGeNews 이진탐지")

    pa.add_argument("--hf_dataset_id",  default=HF_DATASET_ID)
    pa.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    pa.add_argument("--harf_ckpt",     default=None,
                    help="HARFM 4-class 체크포인트 (인코더 재활용, 선택)")

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

    pa.add_argument("--lambda_fake_bce", type=float, default=LAMBDA_FAKE_BCE)
    pa.add_argument("--lambda_oa",       type=float, default=LAMBDA_OA)
    pa.add_argument("--lambda_auth_bce", type=float, default=LAMBDA_AUTH_BCE)
    pa.add_argument("--lambda_af_bce",   type=float, default=LAMBDA_AF_BCE)
    pa.add_argument("--no_kfold", action="store_true")
    pa.add_argument("--kfold_splits", type=int, default=5)

    pa.add_argument("--no_progress", action="store_true")
    args=pa.parse_args()

    if args.no_progress: os.environ["MIRAGE_NO_TQDM"]="1"

    print(f"\n{LOG_TAG} | KFold={not args.no_kfold} | device={DEVICE}")
    print(f"[설정] af_head d+3, patch_text_sim clamp(min=0.)")
    print(f"       overalign_loss_binary oa_uni clamp(0,1)")
    print(f"       from_harf_checkpoint 필터 'head4.'")
    print(f"       TEST_SPLITS 동적 탐지, has_image 손실 전달")
    print(f"[저장] result={RESULT_DIR}")
    print(f"[저장] ckpt  ={CHECKPOINT_DIR}")
    print(f"[손실] Focal"
          f" + {args.lambda_fake_bce}·BCE(Fake)"
          f" + {args.lambda_oa}·OA_binary"
          f" + {args.lambda_auth_bce}·BCE(Auth)"
          f" + {args.lambda_af_bce}·BCE(p_af)\n")

    if args.no_kfold:
        run_official_split(args)
    else:
        run_kfold_split(args)


if __name__ == "__main__":
    main()