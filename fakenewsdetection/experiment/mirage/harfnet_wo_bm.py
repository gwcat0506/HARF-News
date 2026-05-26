"""
HARFNET-VER3 → MiRAGeNews 이진탐지  [Ablation: w/o BM]
=========================================================
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

LAMBDA_FAKE_BCE = 0.30
LAMBDA_OA       = 0.30
LAMBDA_AUTH_BCE = 0.15
LAMBDA_AF_BCE   = 0.20

PROJ_DIM   = 128
ROBERTA    = "roberta-base"
CLIP_RN101 = "RN101"
MAX_LENGTH = 128

HF_DATASET_ID = "anson-huang/mirage-news"

TEST_SPLIT_CANDIDATES = [
    "test_midjourneyv5", "test_midjourney_v5", "test_midjourneyV5",
    "test_dalle3", "test_dalle_3", "test_sdxl", "test_bbc", "test_cnn",
]

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_FND_ROOT      = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
DEFAULT_CKPT   = os.path.join(
    _FND_ROOT, "model", "checkpoints",
    "harfnet_ver3_kfold_20260507_170535_fold5.pt",
)
RESULT_DIR     = os.path.join(_SCRIPT_DIR, "results")
CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")

if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BINARY_CLASS_NAMES = ["Real", "AI-Fake"]
LOG_TAG   = "HARFNET-noBM-MIRAGE"
LOG_BRAND = "HARFNET-noBM-MIRAGE"

os.makedirs(RESULT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def set_seed(s: int = SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed()


# ─────────────────────────────────────────────────────────────
# 2. 유틸 / EarlyStopping
# ─────────────────────────────────────────────────────────────
def _tqdm(it, **kw):
    if os.environ.get("MIRAGE_NO_TQDM", "").lower() in ("1","true","yes"):
        return it
    kw.setdefault("file", sys.stderr); kw.setdefault("dynamic_ncols", True)
    try: sys.stdout.flush(); return tqdm(it, **kw)
    except Exception: return it


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
# ─────────────────────────────────────────────────────────────
class MiRAGeNewsDataset(Dataset):
    def __init__(self, hf_split, tokenizer, clip_preprocess,
                 max_length=MAX_LENGTH, indices=None):
        super().__init__()
        self.tokenizer=tokenizer; self.clip_preprocess=clip_preprocess
        self.max_length=max_length
        cols=hf_split.column_names
        for c in ("caption","text","headline","final_headline"):
            if c in cols: self._text_col=c; break
        else: raise ValueError(f"텍스트 컬럼 없음: {cols}")
        for c in ("image","img"):
            if c in cols: self._img_col=c; break
        else: raise ValueError(f"이미지 컬럼 없음: {cols}")
        for c in ("label","labels","fake"):
            if c in cols: self._lbl_col=c; break
        else: raise ValueError(f"레이블 컬럼 없음: {cols}")
        if indices is not None: hf_split=hf_split.select(indices)
        self.hf_split=hf_split
        self.labels=[int(x) for x in hf_split[self._lbl_col]]
        lc={0:self.labels.count(0),1:self.labels.count(1)}
        print(f"[{LOG_TAG}] {len(self.labels)}건 | Real={lc[0]} AI-Fake={lc[1]}")

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        row=self.hf_split[idx]; text=str(row[self._text_col]).strip()
        img=row[self._img_col]
        if isinstance(img, Image.Image): img=img.convert("RGB")
        else:
            try:
                import io as _io
                raw=img.get("bytes") if isinstance(img,dict) else img
                img=Image.open(_io.BytesIO(raw)).convert("RGB")
            except Exception: img=Image.new("RGB",(224,224),(127,127,127))
        enc=self.tokenizer(text,max_length=self.max_length,
                           padding="max_length",truncation=True,return_tensors="pt")
        return {"input_ids":enc["input_ids"].squeeze(0),
                "attention_mask":enc["attention_mask"].squeeze(0),
                "pixel_values":self.clip_preprocess(img),
                "clip_text_tokens":openai_clip.tokenize([text],truncate=True)[0],
                "has_image":torch.tensor(1.0),
                "label":torch.tensor(self.labels[idx],dtype=torch.long)}


def collate_batch(batch):
    keys=("input_ids","attention_mask","pixel_values","clip_text_tokens","has_image","label")
    return {k:torch.stack([b[k] for b in batch]) for k in keys}


def make_weighted_sampler(labels, alpha=0.5):
    cnt=pd.Series(labels).value_counts().to_dict()
    w=[(1./cnt[l])**alpha for l in labels]
    return WeightedRandomSampler(w,len(w),replacement=True)


# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
class GatedPooling(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.score=nn.Linear(dim,1)
    def forward(self, x, mask=None):
        s=self.score(x).squeeze(-1)
        if mask is not None: s=s.masked_fill(~mask,-1e4)
        return (torch.softmax(s,-1).unsqueeze(-1)*x).sum(1)

class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, d_out, dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_in,d_hid),nn.GELU(),
                               nn.Dropout(dropout),nn.Linear(d_hid,d_out))
    def forward(self, x): return self.net(x)

class AuthorshipModule(nn.Module):
    def __init__(self, d, n_heads=8, dropout=0.1):
        super().__init__()
        self.style_self_attn=nn.MultiheadAttention(d,n_heads,dropout=dropout,batch_first=True)
        self.style_var_proj=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,d))
        self.style_pool=GatedPooling(d)
        self.style_ffn=FeedForward(d*2,d*2,d,dropout)
        self.style_ln=nn.LayerNorm(d)
        self.auth_probe=nn.Sequential(nn.Linear(d,128),nn.LayerNorm(128),nn.GELU(),
                                      nn.Linear(128,1),nn.Sigmoid())
        self.auth_proj=nn.Sequential(nn.Linear(d,d//2),nn.ReLU(),nn.Linear(d//2,PROJ_DIM))
    def forward(self, T_tok, txt_mask):
        T_sty,_=self.style_self_attn(T_tok,T_tok,T_tok,key_padding_mask=~txt_mask)
        sr=T_tok-T_sty
        style_var=self.style_var_proj(sr.var(dim=1).clamp(0.))
        style_pool=self.style_pool(sr,txt_mask)
        F_sty=self.style_ln(self.style_ffn(torch.cat([style_pool,style_var],dim=-1)))
        p_auth=self.auth_probe(F_sty).squeeze(-1)
        z_auth=F.normalize(self.auth_proj(F_sty),dim=-1)
        return F_sty,p_auth,z_auth

class VeracityModule(nn.Module):
    def __init__(self, d, n_heads=8, dropout=0.1):
        super().__init__()
        self.txt2img_attn=nn.MultiheadAttention(d,n_heads,dropout=dropout,batch_first=True)
        self.img2txt_attn=nn.MultiheadAttention(d,n_heads,dropout=dropout,batch_first=True)
        self.gap_encoder=nn.Sequential(nn.Linear(d*2,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(dropout))
        self.sim_proj=nn.Sequential(nn.Linear(1,64),nn.ReLU(),nn.Linear(64,d))
        self.verac_ffn=FeedForward(d*2,d*2,d,dropout)
        self.verac_ln=nn.LayerNorm(d)
        self.fake_probe=nn.Sequential(nn.Linear(d,128),nn.LayerNorm(128),nn.GELU(),
                                      nn.Linear(128,1),nn.Sigmoid())
        self.verac_proj=nn.Sequential(nn.Linear(d,d//2),nn.ReLU(),nn.Linear(d//2,PROJ_DIM))
    def forward(self, T_tok, txt_mask, V_pat, T_cls, V_cls, has_image):
        B=T_tok.size(0); m1=has_image.unsqueeze(-1); m3=has_image.view(B,1,1); eps=1e-8
        F_t2v,_=self.txt2img_attn(T_tok,V_pat,V_pat); F_t2v=F_t2v*m3
        F_v2t,_=self.img2txt_attn(V_pat,T_tok,T_tok,key_padding_mask=~txt_mask); F_v2t=F_v2t*m3
        valid_L=txt_mask.float().sum(1,keepdim=True).clamp(1)
        gap_t=((T_tok-F_t2v)*txt_mask.unsqueeze(-1).float()).sum(1)/valid_L
        gap_v=(V_pat-F_v2t).mean(1)*m1
        F_gap=self.gap_encoder(torch.cat([gap_t,gap_v],dim=-1))
        t_n=F.normalize(T_cls,dim=-1,eps=eps); v_n=F.normalize(V_cls,dim=-1,eps=eps)
        s_sim=(t_n*v_n).sum(-1,keepdim=True).clamp(-1.,1.)*m1
        F_sim=self.sim_proj(s_sim)
        F_ver=self.verac_ln(self.verac_ffn(torch.cat([F_gap,F_sim],dim=-1)))
        p_fake=self.fake_probe(F_ver).squeeze(-1)
        z_ver=F.normalize(self.verac_proj(F_ver),dim=-1)
        return F_ver,p_fake,z_ver,s_sim.squeeze(-1)

class OverAlignModule(nn.Module):
    """af_head 입력 d+3, patch_text_sim clamp(min=0.)"""
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.oa_scalar_proj=nn.Sequential(
            nn.Linear(6,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,d))
        self.oa_patch_pool=nn.Sequential(
            nn.Linear(d,d//2),nn.LayerNorm(d//2),nn.GELU(),nn.Linear(d//2,d))
        self.oa_ffn=nn.Sequential(
            nn.Linear(d*2,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(dropout))
        self.af_head=nn.Sequential(
            nn.Linear(d+3,256),nn.LayerNorm(256),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(256,64),nn.GELU(),
            nn.Linear(64,1),nn.Sigmoid())
    def forward(self, V_pat, T_cls, global_sim, p_fake, p_auth, has_image):
        B=V_pat.size(0); m1=has_image.unsqueeze(-1); eps=1e-8
        t_n=F.normalize(T_cls,dim=-1,eps=eps).unsqueeze(1).expand_as(V_pat)
        v_n=F.normalize(V_pat,dim=-1,eps=eps)
        patch_text_sim=(v_n*t_n).sum(-1)*has_image.unsqueeze(-1)
        patch_text_sim=patch_text_sim.clamp(min=0.)
        oa_mean=patch_text_sim.mean(1); oa_std=patch_text_sim.std(1).clamp(0.)
        oa_max=patch_text_sim.max(1).values; oa_min=patch_text_sim.min(1).values
        oa_uni=oa_mean/((oa_max-oa_min).clamp(min=eps))
        F_oa_s=self.oa_scalar_proj(
            torch.stack([oa_mean,oa_std,oa_uni,oa_max,oa_min,global_sim],dim=-1))
        ow=torch.softmax(patch_text_sim*has_image.unsqueeze(-1),dim=1)
        F_oa_p=self.oa_patch_pool((ow.unsqueeze(-1)*V_pat).sum(1))
        F_oa=self.oa_ffn(torch.cat([F_oa_s,F_oa_p],dim=-1))*m1
        p_af=self.af_head(torch.cat([
            F_oa,
            p_fake.unsqueeze(-1),
            oa_mean.unsqueeze(-1),
            oa_uni.clamp(max=50.).unsqueeze(-1),
        ],dim=-1)).squeeze(-1)*has_image
        return F_oa,p_af,{"oa_mean":oa_mean,"oa_std":oa_std,
                           "oa_uniformity":oa_uni,"p_af":p_af}


# ─────────────────────────────────────────────────────────────
# 5. 메인 모델 — BM 제거
#
#    AM → F_sty   VM → F_ver   OA → F_oa   BM → F_bi
#    head2([F_sty, F_ver, F_oa, F_bi])   d×4 → 2
#
#  [본 파일]
#    AM → F_sty   VM → F_ver   OA → F_oa
#    head2([F_sty, F_ver, F_oa])         d×3 → 2   ← F_bi 제거
# ─────────────────────────────────────────────────────────────
class HARFNETver3_noBM(nn.Module):
    def __init__(self, roberta_name=ROBERTA, clip_name=CLIP_RN101,
                 n_heads=8, dropout=0.1):
        super().__init__()
        d=768
        self.text_encoder=RobertaModel.from_pretrained(roberta_name,add_pooling_layer=False)
        _clip,_=openai_clip.load(clip_name,device="cpu")
        self.clip_model=_clip.float(); self.clip_model.eval()
        for p in self.clip_model.parameters(): p.requires_grad_(False)
        self.visual=self.clip_model.visual
        clip_edim=self.visual.attnpool.c_proj.out_features
        self.deep_proj=nn.Linear(2048,d)
        self.cls_proj=nn.Linear(clip_edim,d)
        # AM / VM / OA 유지, BM(self.bi) 제거
        self.am=AuthorshipModule(d,n_heads,dropout)
        self.vm=VeracityModule(d,n_heads,dropout)
        self.om=OverAlignModule(d,dropout)
        # head2: d×3 (F_sty + F_ver + F_oa)
        self.head2=nn.Sequential(
            nn.Linear(d*3,512),nn.LayerNorm(512),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(512,2))

    def _clip_visual_forward(self, pix, has_image):
        vis=self.visual; B=pix.size(0)
        x=vis.relu1(vis.bn1(vis.conv1(pix))); x=vis.relu2(vis.bn2(vis.conv2(x)))
        x=vis.relu3(vis.bn3(vis.conv3(x)))
        x=vis.avgpool(x)*has_image.view(B,1,1,1)
        x=vis.layer1(x); x=vis.layer2(x); x=vis.layer3(x); deep=vis.layer4(x)
        v_global=vis.attnpool(deep)
        V_pat=self.deep_proj(deep.flatten(2).transpose(1,2))*has_image.view(B,1,1)
        V_cls=self.cls_proj(v_global)*has_image.view(B,1)
        return V_pat,V_cls

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
        # BM 없이 세 특징을 직접 concat
        fused=torch.cat([F_sty,F_ver,F_oa],dim=-1)   # [B, d×3]
        logits2=torch.nan_to_num(self.head2(fused),nan=0.,posinf=20.,neginf=-20.)
        return {"logits2":logits2,"p_auth":p_auth,"p_fake":p_fake,
                "p_af":p_af,"z_auth":z_auth,"z_ver":z_ver,
                "global_sim":gsim,**oa_aux}

    @classmethod
    def from_harf_checkpoint(cls, ckpt_path, device,
                               roberta_name=ROBERTA, clip_name=CLIP_RN101):
        """
        VER3 체크포인트 → bi(BM) 및 head 제외, 나머지 전부 재사용.
        """
        model=cls(roberta_name,clip_name).to(device)
        ckpt=torch.load(ckpt_path,map_location=device)
        state=ckpt.get("model_state_dict",ckpt)
        # BM(bi.)과 head만 제외 — am/vm/om/인코더는 전부 재사용
        exclude=("head4.","head2.","bi.")
        filtered={k:v for k,v in state.items()
                  if not any(k.startswith(p) for p in exclude)}
        missing,unexpected=model.load_state_dict(filtered,strict=False)
        print(f"[{LOG_TAG}] 체크포인트 로드: missing={len(missing)} unexpected={len(unexpected)}")
        if missing: print(f"  missing (샘플): {missing[:5]}")
        return model


# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
def binary_focal_loss(logits2, targets, gamma=2.0):
    n=logits2.size(0)
    cnt=torch.bincount(targets,minlength=2).float().clamp(1)
    inv=(n/cnt); inv=(inv/inv.sum()*2).clamp(max=20.)
    ce=F.cross_entropy(logits2,targets,weight=inv.to(logits2.device),reduction="none")
    return ((1.-torch.exp(-ce))**gamma*ce).mean()

def attribute_bce_loss(p, pos_mask, has_image=None):
    loss=F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),pos_mask.float(),reduction="none")
    return (loss*has_image).mean() if has_image is not None else loss.mean()

def overalign_loss_binary(out, y, has_image=None):
    oa_mean=out["oa_mean"]; oa_uni=out["oa_uniformity"].clamp(max=10.)
    loss=oa_mean.sum()*0.
    m1=(y==1)
    if has_image is not None: m1=m1&(has_image>0.5)
    if m1.any():
        loss=loss+((1.-oa_mean[m1])**2).mean()
        loss=loss+((1.-(oa_uni[m1]/10.).clamp(0.,1.))**2).mean()
    m0=(y==0)
    if has_image is not None: m0=m0&(has_image>0.5)
    if m0.any():
        loss=loss+(oa_mean[m0]**2).mean()
    return loss

def af_bce_loss_binary(p_af, y, has_image=None):
    loss=F.binary_cross_entropy(p_af.clamp(1e-6,1-1e-6),y.float(),reduction="none")
    return (loss*has_image).mean() if has_image is not None else loss.mean()

def total_loss(out, y, has_image, args):
    loss =binary_focal_loss(out["logits2"],y)
    loss+=args.lambda_fake_bce*attribute_bce_loss(out["p_fake"],y==1,has_image)
    loss+=args.lambda_oa       *overalign_loss_binary(out,y,has_image)
    loss+=args.lambda_auth_bce *attribute_bce_loss(out["p_auth"],y==1,has_image)
    loss+=args.lambda_af_bce   *af_bce_loss_binary(out["p_af"],y,has_image)
    return loss


# ─────────────────────────────────────────────────────────────
# 7. 학습 / 평가
# ─────────────────────────────────────────────────────────────
def _fwd(model, batch, device):
    return model(batch["input_ids"].to(device),batch["attention_mask"].to(device),
                 batch["pixel_values"].to(device),batch["has_image"].to(device))

def _monitor(model, loader, device):
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
                    parts.append(f"{cn}:pauth={pauth[mask].mean():.2f}"
                                 f" pf={pf[mask].mean():.2f}"
                                 f" paf={paf[mask].mean():.2f}"
                                 f" oam={oam[mask].mean():.3f}")
            return " | "+" / ".join(parts) if parts else ""
    except Exception: return ""

def run_epoch(model, loader, device, optimizer, train, epoch_idx=None, args=None):
    model.train() if train else model.eval()
    tot,ys,ps,n=0.,[],[],0
    ctx=torch.enable_grad() if train else torch.no_grad()
    it=_tqdm(loader,desc=f"[{LOG_TAG}] Epoch {epoch_idx}") if (train and epoch_idx) else loader
    with ctx:
        for batch in it:
            y=batch["label"].to(device); hi=batch["has_image"].to(device)
            out=_fwd(model,batch,device); loss=total_loss(out,y,hi,args)
            if train and optimizer:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
            pred=out["logits2"].argmax(-1)
            tot+=loss.item()*y.size(0)
            ys.extend(y.cpu().tolist()); ps.extend(pred.cpu().tolist()); n+=y.size(0)
    n=max(n,1)
    return {"loss":tot/n,"acc":float(accuracy_score(ys,ps)),
            "macro_f1":float(f1_score(ys,ps,average="macro",zero_division=0)),
            "f1_ai":float(f1_score(ys,ps,pos_label=1,average="binary",zero_division=0))}

def collect_predictions(model, loader, device):
    model.eval(); ys,ps,scores=[],[],[]
    with torch.no_grad():
        for batch in _tqdm(loader,desc=f"Eval ({LOG_TAG})"):
            out=_fwd(model,batch,device)
            ys.extend(batch["label"].tolist())
            ps.extend(out["logits2"].argmax(-1).cpu().tolist())
            scores.extend(torch.softmax(out["logits2"],-1)[:,1].cpu().tolist())
    return np.array(ys),np.array(ps),np.array(scores)

def eval_report_binary(yt, yp, scores, label=""):
    buf=io.StringIO()
    buf.write(f"\n=== {label} | Real vs AI-Fake ===\n")
    buf.write(classification_report(yt,yp,target_names=BINARY_CLASS_NAMES,digits=4,zero_division=0))
    try: auc=roc_auc_score(yt,scores)
    except: auc=float("nan")
    acc=accuracy_score(yt,yp); f1=f1_score(yt,yp,average="macro",zero_division=0)
    f1_ai=f1_score(yt,yp,pos_label=1,average="binary",zero_division=0)
    buf.write(f"Confusion Matrix:\n{confusion_matrix(yt,yp)}\n")
    buf.write(f"[{label}] Acc={acc:.4f}  Macro-F1={f1:.4f}  AI-Fake F1={f1_ai:.4f}  AUC={auc:.4f}\n")
    return buf.getvalue(),{"acc":acc,"f1":f1,"f1_ai":f1_ai,"auc":auc}


# ─────────────────────────────────────────────────────────────
# 8. 학습 헬퍼
# ─────────────────────────────────────────────────────────────
def freeze_backbones(model):
    for name,p in model.text_encoder.named_parameters():
        p.requires_grad_(any(f"encoder.layer.{i}" in name for i in (10,11)))

def train_one_run(model, tl, vl, device, args, log=None):
    epoch_log=[]
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),
                          lr=args.lr,weight_decay=1e-4)
    early=EarlyStopping(args.early_stop_patience,args.early_stop_min_delta)
    best_f1,best_st=-1.,None
    for ep in range(1,args.epochs+1):
        tr=run_epoch(model,tl,device,opt,True,ep,args)
        va=run_epoch(model,vl,device,None,False,args=args)
        mon=_monitor(model,vl,device)
        line=(f"[{LOG_TAG}] Epoch {ep} "
              f"train_loss={tr['loss']:.4f}  val_loss={va['loss']:.4f}  "
              f"val_f1={va['macro_f1']:.4f}  val_acc={va['acc']:.4f}{mon}")
        print(line)
        epoch_log.append({"epoch":ep,"train_loss":tr["loss"],
                          "val_loss":va["loss"],"val_f1":va["macro_f1"]})
        if log: log.append(line)
        if best_st is None or (va["macro_f1"]-best_f1)>args.early_stop_min_delta:
            best_f1=va["macro_f1"]; best_st=copy.deepcopy(model.state_dict())
        if early.step(va["macro_f1"]):
            msg=f"[{LOG_TAG}] Early stopping at epoch {ep}"
            print(msg)
            if log: log.append(msg)
            break
    if best_st: model.load_state_dict(best_st)
    return model,epoch_log

def plot_epoch_curves(epoch_log, save_path, title=""):
    epochs=[e["epoch"] for e in epoch_log]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
    fig.suptitle(title or LOG_TAG,fontsize=13,fontweight="bold")
    ax1.plot(epochs,[e["train_loss"] for e in epoch_log],"o-",color="#2563EB",lw=2,ms=5,label="Train")
    ax1.plot(epochs,[e["val_loss"] for e in epoch_log],"s--",color="#DC2626",lw=2,ms=5,label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(True,alpha=0.3); ax1.set_xticks(epochs)
    vf=[e["val_f1"] for e in epoch_log]
    ax2.plot(epochs,vf,"^-",color="#16A34A",lw=2,ms=5,label="Val Macro F1")
    best_ep=epochs[int(np.argmax(vf))]
    ax2.axvline(best_ep,color="#16A34A",linestyle=":",alpha=0.6)
    ax2.annotate(f"Best:{max(vf):.4f}\n(Ep{best_ep})",xy=(best_ep,max(vf)),
                 xytext=(best_ep+0.3,max(vf)-0.01),fontsize=9,color="#15803D",
                 arrowprops=dict(arrowstyle="->",color="#15803D"))
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1"); ax2.set_title("Val Macro F1")
    ax2.legend(); ax2.grid(True,alpha=0.3); ax2.set_xticks(epochs)
    plt.tight_layout(); plt.savefig(save_path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"그래프 저장: {save_path}")

def save_checkpoint(path, model, args, metrics=None):
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
    payload={"model_state_dict":model.state_dict(),"config":vars(args),
             "saved_at":datetime.now().isoformat(timespec="seconds")}
    if metrics: payload["metrics"]=metrics
    torch.save(payload,path); print(f"체크포인트 저장: {path}")

def _make_dl(ds, batch_size, num_workers, shuffle, sampler=None, drop_last=False):
    kw=dict(batch_size=batch_size,num_workers=num_workers,collate_fn=collate_batch,
            pin_memory=torch.cuda.is_available(),drop_last=drop_last)
    if sampler: kw["sampler"]=sampler; kw["shuffle"]=False
    else: kw["shuffle"]=shuffle
    return DataLoader(ds,**kw)

def _new_model(args, device):
    return HARFNETver3_noBM(roberta_name=args.roberta,clip_name=args.clip_model).to(device)


# ─────────────────────────────────────────────────────────────
# 9. HF 데이터셋 로드 + 스플릿 탐지
# ─────────────────────────────────────────────────────────────
def load_hf_splits(hf_id=HF_DATASET_ID):
    print(f"[{LOG_TAG}] HuggingFace 데이터셋 로드: {hf_id}")
    ds=load_dataset(hf_id); print(f"[{LOG_TAG}] 스플릿: {list(ds.keys())}"); return ds

def resolve_test_splits(ds):
    actual=set(ds.keys())
    matched=[k for k in TEST_SPLIT_CANDIDATES if k in actual]; ms=set(matched)
    extra=sorted([k for k in actual if k.startswith("test") and k not in ms])
    result=matched+[e for e in extra if e not in ms]
    print(f"[{LOG_TAG}] 테스트 스플릿: {result}"); return result


# ─────────────────────────────────────────────────────────────
# 10. Official / KFold 실행
# ─────────────────────────────────────────────────────────────
def run_official_split(args):
    set_seed(args.seed)
    tok=RobertaTokenizerFast.from_pretrained(args.roberta)
    _,prep=openai_clip.load(args.clip_model,device="cpu")
    hf_ds=load_hf_splits(args.hf_dataset_id)
    def _wrap(s): return MiRAGeNewsDataset(hf_ds[s],tok,prep,args.max_length)
    tr_ds=_wrap("train")
    sampler=make_weighted_sampler(tr_ds.labels,args.sampler_alpha) if args.use_sampler else None
    tl=_make_dl(tr_ds,args.batch_size,args.num_workers,True,sampler=sampler,drop_last=True)
    vl=_make_dl(_wrap("validation"),args.batch_size,args.num_workers,False)
    model=(HARFNETver3_noBM.from_harf_checkpoint(
               args.harf_ckpt,DEVICE,args.roberta,args.clip_model)
           if args.harf_ckpt else _new_model(args,DEVICE))
    if not args.no_freeze_encoders: freeze_backbones(model)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    rep=io.StringIO()
    rep.write(f"###### {LOG_TAG} (Official Split) ######\n")
    rep.write(f"[제거] BM  [유지] AM + VM + OA\n")
    rep.write(f"[head] d×3 → 512 → 2\n")
    rep.write(f"[손실] Focal + {args.lambda_fake_bce}·BCE(fake)"
              f" + {args.lambda_oa}·OA"
              f" + {args.lambda_auth_bce}·BCE(auth)"
              f" + {args.lambda_af_bce}·BCE(p_af)\n\n")
    log=[]
    model,epoch_log=train_one_run(model,tl,vl,DEVICE,args,log)
    for line in log: rep.write(line+"\n")
    plot_epoch_curves(epoch_log,os.path.join(RESULT_DIR,f"nobm_mirage_curve_{ts}.png"),
                      title=f"{LOG_TAG} (Official Split)")
    test_splits=resolve_test_splits(hf_ds)
    rep.write("\n\n===== TEST RESULTS =====\n")
    all_metrics={}
    for sname in test_splits:
        el=_make_dl(_wrap(sname),args.batch_size,args.num_workers,False)
        yt,yp,sc=collect_predictions(model,el,DEVICE)
        blk,m=eval_report_binary(yt,yp,sc,label=sname)
        print(blk); rep.write(blk); all_metrics[sname]=m
    rep.write("\n===== SUMMARY =====\n")
    for k in ("acc","f1","f1_ai","auc"):
        vals=[all_metrics[s][k] for s in all_metrics]
        line=f"  {k}: {np.mean(vals):.4f}  (min={min(vals):.4f} max={max(vals):.4f})"
        rep.write(line+"\n"); print(line)
    save_checkpoint(os.path.join(CHECKPOINT_DIR,f"nobm_mirage_single_{ts}.pt"),
                    model,args,all_metrics)
    path=os.path.join(RESULT_DIR,f"nobm_mirage_single_{ts}.txt")
    with open(path,"w",encoding="utf-8") as f: f.write(rep.getvalue())
    print(f"\n리포트: {path}")

def run_kfold_split(args):
    set_seed(args.seed)
    tok=RobertaTokenizerFast.from_pretrained(args.roberta)
    _,prep=openai_clip.load(args.clip_model,device="cpu")
    hf_ds=load_hf_splits(args.hf_dataset_id)
    base_train=hf_ds["train"]
    train_labels=np.array([int(x) for x in base_train["label"]])
    skf=StratifiedKFold(n_splits=args.kfold_splits,shuffle=True,random_state=args.seed)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    rep=io.StringIO()
    rep.write(f"###### {LOG_TAG} (KFold={args.kfold_splits}) ######\n")
    rep.write(f"[제거] BM  [유지] AM + VM + OA\n")
    rep.write(f"[손실] Focal + {args.lambda_fake_bce}·BCE(fake)"
              f" + {args.lambda_oa}·OA"
              f" + {args.lambda_auth_bce}·BCE(auth)"
              f" + {args.lambda_af_bce}·BCE(p_af)\n\n")
    test_splits=resolve_test_splits(hf_ds)
    rep.write(f"test_splits={test_splits}\n\n")
    fold_results=[]
    for fold_idx,(tr_idx,va_idx) in enumerate(
            skf.split(np.zeros(len(train_labels)),train_labels),start=1):
        print(f"\n{'='*70}\n[Fold {fold_idx}/{args.kfold_splits}] {LOG_BRAND}\n{'='*70}")
        rep.write(f"\n\n{'#'*70}\n### Fold {fold_idx}/{args.kfold_splits}\n{'#'*70}\n")
        tr_ds=MiRAGeNewsDataset(base_train,tok,prep,args.max_length,indices=tr_idx.tolist())
        va_ds=MiRAGeNewsDataset(base_train,tok,prep,args.max_length,indices=va_idx.tolist())
        sampler=make_weighted_sampler(tr_ds.labels,args.sampler_alpha) if args.use_sampler else None
        tl=_make_dl(tr_ds,args.batch_size,args.num_workers,True,sampler=sampler,drop_last=True)
        vl=_make_dl(va_ds,args.batch_size,args.num_workers,False)
        model=(HARFNETver3_noBM.from_harf_checkpoint(
                   args.harf_ckpt,DEVICE,args.roberta,args.clip_model)
               if args.harf_ckpt else _new_model(args,DEVICE))
        if not args.no_freeze_encoders: freeze_backbones(model)
        log=[]
        model,epoch_log=train_one_run(model,tl,vl,DEVICE,args,log)
        for line in log: rep.write(line+"\n")
        plot_epoch_curves(epoch_log,
            os.path.join(RESULT_DIR,f"nobm_mirage_curve_fold{fold_idx}_{ts}.png"),
            title=f"{LOG_TAG} · Fold {fold_idx}/{args.kfold_splits}")
        fold_metrics={}
        for sname in test_splits:
            test_ds=MiRAGeNewsDataset(hf_ds[sname],tok,prep,args.max_length)
            el=_make_dl(test_ds,args.batch_size,args.num_workers,False)
            yt,yp,sc=collect_predictions(model,el,DEVICE)
            blk,m=eval_report_binary(yt,yp,sc,label=sname)
            print(blk); rep.write(blk); fold_metrics[sname]=m
        fold_results.append(fold_metrics)
        print(f"[Fold {fold_idx}] "
              +" ".join(f"{s}:F1={fold_metrics[s]['f1']:.4f}" for s in test_splits))
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    rep.write("\n\n===== KFOLD GLOBAL SUMMARY =====\n")
    global_metrics={}
    for sname in test_splits:
        global_metrics[sname]={k:float(np.mean([fr[sname][k] for fr in fold_results]))
                                for k in ("acc","f1","f1_ai","auc")}
    for sname in test_splits:
        g=global_metrics[sname]
        line=(f"{sname}: Acc={g['acc']:.4f}  Macro-F1={g['f1']:.4f}"
              f"  F1_AI={g['f1_ai']:.4f}  AUC={g['auc']:.4f}")
        print(line); rep.write(line+"\n")
    rep.write("\n===== 전체 평균 =====\n")
    for k in ("acc","f1","f1_ai","auc"):
        vals=[global_metrics[s][k] for s in test_splits]
        line=f"  {k}: {float(np.mean(vals)):.4f}"
        print(line); rep.write(line+"\n")
    torch.save({"config":vars(args),"fold_metrics":fold_results,
                "global_metrics":global_metrics,
                "saved_at":datetime.now().isoformat(timespec="seconds")},
               os.path.join(CHECKPOINT_DIR,
                            f"nobm_mirage_kfold{args.kfold_splits}_{ts}.pt"))
    path=os.path.join(RESULT_DIR,f"nobm_mirage_kfold{args.kfold_splits}_{ts}.txt")
    with open(path,"w",encoding="utf-8") as f: f.write(rep.getvalue())
    print(f"\n리포트: {path}")


# ─────────────────────────────────────────────────────────────
# 11. main
# ─────────────────────────────────────────────────────────────
def main():
    pa=argparse.ArgumentParser(description="HARFNET w/o BM → MiRAGeNews 이진탐지")
    pa.add_argument("--hf_dataset_id",  default=HF_DATASET_ID)
    pa.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    pa.add_argument("--harf_ckpt",     default=DEFAULT_CKPT)
    pa.add_argument("--batch_size",   type=int,   default=BATCH)
    pa.add_argument("--epochs",       type=int,   default=EPOCHS)
    pa.add_argument("--lr",           type=float, default=LR)
    pa.add_argument("--num_workers",  type=int,   default=NUM_WORKERS)
    pa.add_argument("--seed",         type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    pa.add_argument("--roberta",    default=ROBERTA)
    pa.add_argument("--clip_model", default=CLIP_RN101)
    pa.add_argument("--max_length", type=int,   default=MAX_LENGTH)
    pa.add_argument("--no_freeze_encoders", action="store_true")
    pa.add_argument("--lambda_fake_bce", type=float, default=LAMBDA_FAKE_BCE)
    pa.add_argument("--lambda_oa",       type=float, default=LAMBDA_OA)
    pa.add_argument("--lambda_auth_bce", type=float, default=LAMBDA_AUTH_BCE)
    pa.add_argument("--lambda_af_bce",   type=float, default=LAMBDA_AF_BCE)
    pa.add_argument("--use_sampler",     action="store_true")
    pa.add_argument("--sampler_alpha",   type=float, default=0.5)
    pa.add_argument("--no_kfold",     action="store_true")
    pa.add_argument("--kfold_splits", type=int, default=5)
    pa.add_argument("--no_progress",  action="store_true")
    args=pa.parse_args()
    if args.no_progress: os.environ["MIRAGE_NO_TQDM"]="1"
    set_seed(args.seed)
    print(f"\n{LOG_BRAND} | KFold={not args.no_kfold} | device={DEVICE}")
    print(f"[Ablation] w/o BM  →  AM + VM + OA 유지")
    print(f"[head]     d×3 → 512 → 2  (원본 d×4 → 512 → 2)")
    print(f"[손실]     Focal + {args.lambda_fake_bce}·BCE(fake)"
          f" + {args.lambda_oa}·OA"
          f" + {args.lambda_auth_bce}·BCE(auth)"
          f" + {args.lambda_af_bce}·BCE(p_af)")
    print(f"[ckpt]     {args.harf_ckpt}\n")
    run_kfold_split(args) if not args.no_kfold else run_official_split(args)

if __name__=="__main__":
    main()