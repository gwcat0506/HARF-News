"""
CAFE × DGM4 벤치마크 실험 코드
=====================================
cyxanna/CAFE (WWW2022) 모델을 DGM4 이진 분류 벤치마크에 적용.

[수정 사항]
  _SafeBN: BatchNorm1d를 batch=1에서도 안전하게 동작하도록 래핑
  prepare_similarity_data: fake 샘플 < 2이면 sim loss 건너뜀 (BN 에러 방지)
"""

from __future__ import annotations

import argparse
import copy
import gc
import io
import math
import os
import random
import re
import sys
import warnings
from collections import Counter
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
from torch.distributions import Independent, Normal
from torch.nn.functional import softplus
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

from dgm4_paths import DEFAULT_DATA_ROOT, FINE_LABELS, load_dgm4_splits, resolve_image_path

# ════════════════════════════════════════════════════════════════════════════════
# 1. 상수
# ════════════════════════════════════════════════════════════════════════════════
BATCH                = 16
EPOCHS               = 10
LR                   = 2e-4
NUM_WORKERS          = 4
EARLY_STOP_PATIENCE  = 4
EARLY_STOP_MIN_DELTA = 1e-4
KFOLD_SPLITS         = 5
KFOLD_RANDOM_STATE   = 42
SEED                 = 42
SAMPLER_ALPHA        = 0.5

PAPER_WEIGHT_DECAY        = 0.02
PAPER_WARMUP_IN_50_EPOCHS = 10
PAPER_SCHED_TOTAL_EPOCHS  = 50

LR_MULT_VISUAL = 0.50
LR_MULT_NEW    = 1.00

TEXT_DIM       = 200
SEQ_LEN        = 64
IMG_SIZE       = 224
VOCAB_MIN_FREQ = 2

LOG_TAG = "CAFE-DGM4"

if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ════════════════════════════════════════════════════════════════════════════════
# ★ BatchNorm 안전 래퍼 — batch=1 학습 시 에러 방지
# ════════════════════════════════════════════════════════════════════════════════

class _SafeBN(nn.Module):
    """
    nn.BatchNorm1d 래퍼.
    학습 중 batch_size == 1 이면 BN을 건너뜀 (항등 함수로 동작).
    → Similarity task가 fake 샘플만 사용할 때 batch=1이 되는 상황 방지.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) == 1:
            return x
        return self.bn(x)


def default_paper_warmup_epochs(train_epochs: int) -> int:
    if train_epochs <= 1:
        return 0
    w = int(round(train_epochs * (PAPER_WARMUP_IN_50_EPOCHS / PAPER_SCHED_TOTAL_EPOCHS)))
    return max(1, min(train_epochs - 1, w))


def set_seed(s: int = SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


set_seed()


# ════════════════════════════════════════════════════════════════════════════════
# 2. 유틸
# ════════════════════════════════════════════════════════════════════════════════

def _tqdm(it, **kw):
    if os.environ.get("DGM4_NO_TQDM", "").lower() in ("1", "true", "yes"):
        return it
    kw.setdefault("file", sys.stderr)
    kw.setdefault("dynamic_ncols", True)
    try:
        sys.stdout.flush()
        return tqdm(it, **kw)
    except Exception:
        return it


def _ensure_dgm4_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "has_image_flag" not in df.columns:
        df["has_image_flag"] = df["image"].apply(
            lambda p: isinstance(p, str) and len(p) > 0
        )
    if "binary_label" not in df.columns:
        df["binary_label"] = (df["fake_cls"] != "orig").astype(int)
    for lbl in FINE_LABELS:
        col = f"fine_{lbl}"
        if col not in df.columns:
            df[col] = (df["fake_cls"] == lbl).astype(float)
    return df


# ════════════════════════════════════════════════════════════════════════════════
# 3. Vocabulary
# ════════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> List[str]:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    return text.split()


def build_vocab(texts: List[str], min_freq: int = VOCAB_MIN_FREQ) -> Dict[str, int]:
    counter: Counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in counter.most_common():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def text_to_ids(text: str, vocab: Dict[str, int], max_len: int = SEQ_LEN) -> List[int]:
    pad_idx = vocab["<pad>"]
    unk_idx = vocab["<unk>"]
    tokens  = tokenize(text)
    ids     = [vocab.get(w, unk_idx) for w in tokens[:max_len]]
    ids     = ids + [pad_idx] * (max_len - len(ids))
    return ids


# ════════════════════════════════════════════════════════════════════════════════
# 4. EarlyStopping
# ════════════════════════════════════════════════════════════════════════════════

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
            self.best = v
            return False
        imp = (v - self.best > self.min_delta if self.mode == "max"
               else self.best - v > self.min_delta)
        if imp:
            self.best = v
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ════════════════════════════════════════════════════════════════════════════════
# 5. Dataset
# ════════════════════════════════════════════════════════════════════════════════

class DGM4CAFEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str,
                 vocab: Dict[str, int], max_len: int = SEQ_LEN):
        super().__init__()
        self.data_root = os.path.abspath(data_root)
        self.vocab     = vocab
        self.max_len   = max_len
        df = _ensure_dgm4_columns(df)
        self.texts        = df["text"].tolist()
        self.image_rels   = df["image"].tolist()
        self.has_img_flag = df["has_image_flag"].tolist()
        self.binary_lbls  = df["binary_label"].tolist()
        self.fake_cls_str = df["fake_cls"].tolist()

    def __len__(self):
        return len(self.binary_lbls)

    def _load_pil(self, rel: str, flag: bool) -> Image.Image:
        if not flag or not rel:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127))
        full = resolve_image_path(rel, self.data_root)
        if not full or not os.path.isfile(full):
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127))
        try:
            return Image.open(full).convert("RGB")
        except Exception:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127))

    def __getitem__(self, idx):
        text = self.texts[idx]
        img  = self._load_pil(self.image_rels[idx], self.has_img_flag[idx])
        ids  = text_to_ids(text, self.vocab, self.max_len)
        return {
            "text_ids":     torch.tensor(ids, dtype=torch.long),
            "image":        image_transform(img),
            "binary_label": torch.tensor(self.binary_lbls[idx], dtype=torch.long),
            "text":         text,
            "image_path":   self.image_rels[idx],
            "fake_cls":     self.fake_cls_str[idx],
        }


def collate_dgm4_cafe(batch):
    tensor_keys = ("text_ids", "image", "binary_label")
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


# ════════════════════════════════════════════════════════════════════════════════
# 6. 텍스트/이미지 인코더
# ════════════════════════════════════════════════════════════════════════════════

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = TEXT_DIM, pad_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

    def forward(self, ids):
        return self.embed(ids).float()


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.features(x).flatten(1)


# ════════════════════════════════════════════════════════════════════════════════
# 7. CAFE 원본 model.py (cyxanna/CAFE)
#    수정 1: squeeze() → squeeze(-1)       [batch=1 안전]
#    수정 2: torch.sigmoid                  [deprecated 수정]
#    수정 3: nn.BatchNorm1d → _SafeBN       [★ batch=1 에러 완전 방지]
# ════════════════════════════════════════════════════════════════════════════════

class FastCNN(nn.Module):
    def __init__(self, channel=32, kernel_size=(1, 2, 4, 8)):
        super(FastCNN, self).__init__()
        self.fast_cnn = nn.ModuleList()
        for kernel in kernel_size:
            self.fast_cnn.append(nn.Sequential(
                nn.Conv1d(200, channel, kernel_size=kernel),
                _SafeBN(channel),          # ★
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            ))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_out = [module(x).squeeze(-1) for module in self.fast_cnn]
        return torch.cat(x_out, 1)


class EncodingPart(nn.Module):
    def __init__(self, cnn_channel=32, cnn_kernel_size=(1, 2, 4, 8),
                 shared_image_dim=128, shared_text_dim=128):
        super(EncodingPart, self).__init__()
        self.shared_text_encoding = FastCNN(channel=cnn_channel, kernel_size=cnn_kernel_size)
        self.shared_text_linear = nn.Sequential(
            nn.Linear(128, 64), _SafeBN(64), nn.ReLU(), nn.Dropout(),  # ★
            nn.Linear(64, shared_text_dim), _SafeBN(shared_text_dim), nn.ReLU(),  # ★
        )
        self.shared_image = nn.Sequential(
            nn.Linear(512, 256), _SafeBN(256), nn.ReLU(), nn.Dropout(),  # ★
            nn.Linear(256, shared_image_dim), _SafeBN(shared_image_dim), nn.ReLU(),  # ★
        )

    def forward(self, text, image):
        text_shared  = self.shared_text_linear(self.shared_text_encoding(text))
        image_shared = self.shared_image(image)
        return text_shared, image_shared


class SimilarityModule(nn.Module):
    def __init__(self, shared_dim=128, sim_dim=64):
        super(SimilarityModule, self).__init__()
        self.encoding      = EncodingPart()
        self.text_aligner  = nn.Sequential(
            nn.Linear(shared_dim, shared_dim), _SafeBN(shared_dim), nn.ReLU(),  # ★
            nn.Linear(shared_dim, sim_dim),    _SafeBN(sim_dim),    nn.ReLU(),  # ★
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim), _SafeBN(shared_dim), nn.ReLU(),  # ★
            nn.Linear(shared_dim, sim_dim),    _SafeBN(sim_dim),    nn.ReLU(),  # ★
        )
        self.sim_classifier = nn.Sequential(
            _SafeBN(sim_dim * 2),  # ★
            nn.Linear(sim_dim * 2, 64), _SafeBN(64), nn.ReLU(),  # ★
            nn.Linear(64, 2),
        )

    def forward(self, text, image):
        text_enc, image_enc = self.encoding(text, image)
        text_aligned        = self.text_aligner(text_enc)
        image_aligned       = self.image_aligner(image_enc)
        pred_similarity     = self.sim_classifier(
            torch.cat([text_aligned, image_aligned], 1))
        return text_aligned, image_aligned, pred_similarity


class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(True), nn.Linear(64, z_dim * 2),
        )

    def forward(self, x):
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


class AmbiguityLearning(nn.Module):
    def __init__(self):
        super(AmbiguityLearning, self).__init__()
        self.encoding      = EncodingPart()
        self.encoder_text  = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_encoding, image_encoding):
        p_z1 = self.encoder_text(text_encoding)
        p_z2 = self.encoder_image(image_encoding)
        z1, z2 = p_z1.rsample(), p_z2.rsample()
        kl_1_2 = p_z1.log_prob(z1) - p_z2.log_prob(z1)
        kl_2_1 = p_z2.log_prob(z2) - p_z1.log_prob(z2)
        skl = (kl_1_2 + kl_2_1) / 2.
        return torch.sigmoid(skl)


class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=128, prime_dim=16):
        super(UnimodalDetection, self).__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim), _SafeBN(shared_dim), nn.ReLU(),  # ★
            nn.Linear(shared_dim, prime_dim),  _SafeBN(prime_dim),  nn.ReLU(),  # ★
        )
        self.image_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim), _SafeBN(shared_dim), nn.ReLU(),  # ★
            nn.Linear(shared_dim, prime_dim),  _SafeBN(prime_dim),  nn.ReLU(),  # ★
        )

    def forward(self, text_encoding, image_encoding):
        return self.text_uni(text_encoding), self.image_uni(image_encoding)


class CrossModule4Batch(nn.Module):
    def __init__(self, text_in_dim=64, image_in_dim=64, corre_out_dim=64):
        super(CrossModule4Batch, self).__init__()
        self.softmax   = nn.Softmax(-1)
        self.corre_dim = 64
        self.pooling   = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            _SafeBN(corre_out_dim), nn.ReLU(),  # ★
        )

    def forward(self, text, image):
        similarity    = torch.matmul(
            text.unsqueeze(2), image.unsqueeze(1)) / math.sqrt(text.shape[1])
        correlation   = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze(-1)
        return self.c_specific_2(correlation_p)


class DetectionModule(nn.Module):
    def __init__(self, feature_dim=64 + 16 + 16, h_dim=64):
        super(DetectionModule, self).__init__()
        self.encoding         = EncodingPart()
        self.ambiguity_module = AmbiguityLearning()
        self.uni_repre        = UnimodalDetection()
        self.cross_module     = CrossModule4Batch()
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim), _SafeBN(h_dim), nn.ReLU(),  # ★
            nn.Linear(h_dim, h_dim),       _SafeBN(h_dim), nn.ReLU(),  # ★
            nn.Linear(h_dim, 2),
        )

    def forward(self, text_raw, image_raw, text, image):
        skl                     = self.ambiguity_module(text, image)
        text_prime, image_prime = self.encoding(text_raw, image_raw)
        text_prime, image_prime = self.uni_repre(text_prime, image_prime)
        correlation             = self.cross_module(text, image)
        weight_uni              = (1 - skl).unsqueeze(1)
        weight_corre            = skl.unsqueeze(1)
        final_corre = torch.cat([weight_uni * text_prime,
                                  weight_uni * image_prime,
                                  weight_corre * correlation], 1)
        return self.classifier_corre(final_corre)


# ════════════════════════════════════════════════════════════════════════════════
# 8. 백본 설정
# ════════════════════════════════════════════════════════════════════════════════

def configure_backbones(img_encoder: ImageEncoder):
    for name, p in img_encoder.named_parameters():
        p.requires_grad_("layer4" in name)
    n_train = sum(p.numel() for p in img_encoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in img_encoder.parameters())
    print(f"[{LOG_TAG}] ImageEncoder 학습 파라미터: {n_train:,} / {n_total:,}"
          f" ({100*n_train/n_total:.1f}%)")


# ════════════════════════════════════════════════════════════════════════════════
# 9. 옵티마이저 & 스케줄러
# ════════════════════════════════════════════════════════════════════════════════

def _make_optim(img_encoder, modules, args):
    wd = getattr(args, "weight_decay", PAPER_WEIGHT_DECAY)
    grp_visual, grp_new = [], []
    for p in img_encoder.parameters():
        if p.requires_grad:
            grp_visual.append(p)
    for m in modules:
        for p in m.parameters():
            if p.requires_grad:
                grp_new.append(p)
    groups = [
        {"params": grp_visual, "lr": args.lr * LR_MULT_VISUAL, "weight_decay": wd},
        {"params": grp_new,    "lr": args.lr * LR_MULT_NEW,    "weight_decay": wd},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]])


def make_optimizer_sim(img_encoder, text_embed, sim_module, args):
    return _make_optim(img_encoder, [text_embed, sim_module], args)


def make_optimizer_det(img_encoder, text_embed, det_module, args):
    return _make_optim(img_encoder, [text_embed, det_module], args)


def make_scheduler(optimizer, args):
    warmup = args.warmup_epochs
    total  = args.epochs

    def lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(max(warmup, 1))
        progress = float(ep - warmup) / float(max(total - warmup, 1))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ════════════════════════════════════════════════════════════════════════════════
# 10. prepare_similarity_data
#     ★ fake < 2 이면 None 반환 → sim loss 건너뜀 (BatchNorm batch=1 방지)
# ════════════════════════════════════════════════════════════════════════════════

def prepare_similarity_data(text_feat, img_feat, labels):
    fake_idx = (labels == 1).nonzero(as_tuple=True)[0]
    if len(fake_idx) < 2:   # ★ 1개 이하이면 BN 에러 → 건너뜀
        return None, None, None
    fixed_text      = text_feat[fake_idx]
    matched_image   = img_feat[fake_idx]
    shift           = min(3, len(fake_idx) - 1)
    unmatched_image = matched_image.roll(shifts=shift, dims=0)
    return fixed_text, matched_image, unmatched_image


# ════════════════════════════════════════════════════════════════════════════════
# 11. 학습 epoch
# ════════════════════════════════════════════════════════════════════════════════

def train_one_epoch(img_encoder, text_embed, sim_module, det_module,
                    loader, device, opt_sim, opt_det,
                    sched_sim, sched_det, epoch_idx: int):
    img_encoder.train(); text_embed.train()
    sim_module.train();  det_module.train()

    loss_fn_sim = nn.CosineEmbeddingLoss()
    loss_fn_det = nn.CrossEntropyLoss()
    tot_sim = tot_det = 0.0
    n_sim   = n_det   = 0

    it = _tqdm(loader, desc=f"[{LOG_TAG}] Epoch {epoch_idx}")
    for batch in it:
        text_ids = batch["text_ids"].to(device)
        images   = batch["image"].to(device)
        y_bin    = batch["binary_label"].to(device)
        B        = text_ids.size(0)

        # ── 1) Similarity 학습 (fake >= 2 인 경우만) ─────────────────────────
        img_feat_sim  = img_encoder(images)
        text_feat_sim = text_embed(text_ids)
        fixed_t, mat_i, unmat_i = prepare_similarity_data(
            text_feat_sim, img_feat_sim, y_bin)

        if fixed_t is not None:
            n_fake    = fixed_t.shape[0]
            ta_m, ia_m, _ = sim_module(fixed_t, mat_i)
            ta_u, ia_u, _ = sim_module(fixed_t, unmat_i)
            ta_cat    = torch.cat([ta_m, ta_u], dim=0)
            ia_cat    = torch.cat([ia_m, ia_u], dim=0)
            sim_label = torch.cat([
                torch.ones(n_fake, device=device),
                -torch.ones(n_fake, device=device),
            ])
            loss_sim  = loss_fn_sim(ta_cat, ia_cat, sim_label)
            opt_sim.zero_grad()
            loss_sim.backward()
            opt_sim.step()
            tot_sim += loss_sim.item() * n_fake * 2
            n_sim   += n_fake * 2

        # ── 2) Detection 학습 (별도 forward) ─────────────────────────────────
        img_feat_det  = img_encoder(images)
        text_feat_det = text_embed(text_ids)
        ta, ia, _     = sim_module(text_feat_det, img_feat_det)
        logits        = det_module(text_feat_det, img_feat_det, ta, ia)
        loss_det      = loss_fn_det(logits, y_bin)
        opt_det.zero_grad()
        loss_det.backward()
        opt_det.step()
        tot_det += loss_det.item() * B
        n_det   += B

    sched_sim.step()
    sched_det.step()
    return tot_sim / max(n_sim, 1), tot_det / max(n_det, 1)


# ════════════════════════════════════════════════════════════════════════════════
# 12. 평가 epoch
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_epoch(img_encoder, text_embed, sim_module, det_module,
               loader, device) -> Dict[str, float]:
    img_encoder.eval(); text_embed.eval()
    sim_module.eval();  det_module.eval()
    ys, ps, probs = [], [], []

    for batch in loader:
        text_ids  = batch["text_ids"].to(device)
        images    = batch["image"].to(device)
        y_bin     = batch["binary_label"].to(device)
        img_feat  = img_encoder(images)
        text_feat = text_embed(text_ids)
        ta, ia, _ = sim_module(text_feat, img_feat)
        logits    = det_module(text_feat, img_feat, ta, ia)
        prob      = torch.softmax(logits, dim=-1)[:, 1]
        pred      = logits.argmax(dim=1)
        ys.extend(y_bin.cpu().tolist())
        ps.extend(pred.cpu().tolist())
        probs.extend(prob.cpu().tolist())

    ys_arr    = np.array(ys)
    probs_arr = np.array(probs)
    ps_arr    = np.array(ps)

    try:
        auc = float(roc_auc_score(ys_arr, probs_arr))
    except Exception:
        auc = float("nan")
    try:
        fpr, tpr, thresholds = roc_curve(ys_arr, probs_arr)
        ps_opt  = (probs_arr >= float(thresholds[np.argmax(tpr - fpr)])).astype(int)
        f1_opt  = float(f1_score(ys_arr, ps_opt, average="binary",
                                  pos_label=1, zero_division=0))
    except Exception:
        f1_opt = float("nan")

    return {
        "bin_acc":    float(accuracy_score(ys_arr, ps_arr)),
        "bin_f1":     float(f1_score(ys_arr, ps_arr, average="macro", zero_division=0)),
        "bin_f1_opt": f1_opt,
        "bin_auc":    auc,
    }


# ════════════════════════════════════════════════════════════════════════════════
# 13. 예측 수집 & 지표
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_predictions(img_encoder, text_embed, sim_module, det_module,
                        loader, device) -> pd.DataFrame:
    img_encoder.eval(); text_embed.eval()
    sim_module.eval();  det_module.eval()
    rows = []
    for batch in _tqdm(loader, desc=f"[{LOG_TAG}] 예측 수집"):
        text_ids  = batch["text_ids"].to(device)
        images    = batch["image"].to(device)
        y_bin     = batch["binary_label"]
        img_feat  = img_encoder(images)
        text_feat = text_embed(text_ids)
        ta, ia, _ = sim_module(text_feat, img_feat)
        logits    = det_module(text_feat, img_feat, ta, ia)
        bin_prob  = torch.softmax(logits, dim=-1)[:, 1]
        bin_pred  = logits.argmax(dim=1)
        texts     = batch.get("texts",       [""] * y_bin.size(0))
        paths     = batch.get("image_paths", [""] * y_bin.size(0))
        fcs       = batch.get("fake_cls",    [""] * y_bin.size(0))
        for i in range(y_bin.size(0)):
            rows.append({
                "text":           texts[i],
                "image_path":     paths[i],
                "fake_cls":       fcs[i],
                "binary_label":   int(y_bin[i]),
                "binary_pred":    int(bin_pred[i]),
                "binary_correct": int(y_bin[i]) == int(bin_pred[i]),
                "binary_prob":    round(float(bin_prob[i]), 4),
            })
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    yt    = df["binary_label"].values
    yp    = df["binary_pred"].values
    yprob = df["binary_prob"].values
    try:
        fpr, tpr, thresholds = roc_curve(yt, yprob)
        yp_opt = (yprob >= float(thresholds[np.argmax(tpr - fpr)])).astype(int)
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
    return m


def eval_report_block(df: pd.DataFrame, label: str = LOG_TAG) -> Tuple[str, dict]:
    buf = io.StringIO()
    buf.write(f"\n=== {label} — Binary Detection ===\n")
    buf.write(classification_report(df["binary_label"].values, df["binary_pred"].values,
                                    target_names=["real", "fake"],
                                    digits=4, zero_division=0))
    m = compute_metrics(df)
    buf.write(f"\n[{label}]  ACC={m['bin_acc']:.4f}  F1={m['bin_f1']:.4f}"
              f"  F1_opt={m['bin_f1_opt']:.4f}"
              f"  AUC={m.get('bin_auc', float('nan')):.4f}\n")
    return buf.getvalue(), m


# ════════════════════════════════════════════════════════════════════════════════
# 14. 체크포인트 I/O
# ════════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, img_encoder, text_embed, sim_module, det_module,
                    vocab, args, metrics=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "img_encoder_state": img_encoder.state_dict(),
        "text_embed_state":  text_embed.state_dict(),
        "sim_module_state":  sim_module.state_dict(),
        "det_module_state":  det_module.state_dict(),
        "vocab":             vocab,
        "config":            vars(args),
        "saved_at":          datetime.now().isoformat(timespec="seconds"),
    }
    if metrics:
        payload["metrics"] = metrics
    torch.save(payload, path)
    print(f"[{LOG_TAG}] 체크포인트 저장: {path}")


def load_checkpoint(img_encoder, text_embed, sim_module, det_module,
                    path: str, device) -> Optional[Dict]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        print(f"[{LOG_TAG}] WARNING: 체크포인트 형식 불명")
        return None

    def _load(model, key, name):
        if key in ckpt:
            mis, unexp = model.load_state_dict(ckpt[key], strict=False)
            print(f"[{LOG_TAG}] {name}  missing={len(mis)}  unexpected={len(unexp)}")
        else:
            print(f"[{LOG_TAG}] WARNING: '{key}' 없음 — {name} 로드 생략")

    _load(img_encoder, "img_encoder_state", "ImageEncoder")
    _load(text_embed,  "text_embed_state",  "TextEmbedding")
    _load(sim_module,  "sim_module_state",  "SimilarityModule")
    _load(det_module,  "det_module_state",  "DetectionModule")
    return ckpt.get("vocab", None)


# ════════════════════════════════════════════════════════════════════════════════
# 15. 학습 루프
# ════════════════════════════════════════════════════════════════════════════════

def _build_models(vocab_size: int, device):
    img_encoder = ImageEncoder().to(device)
    text_embed  = TextEmbedding(vocab_size, TEXT_DIM).to(device)
    sim_module  = SimilarityModule().to(device)
    det_module  = DetectionModule().to(device)
    return img_encoder, text_embed, sim_module, det_module


def train_one_run(img_encoder, text_embed, sim_module, det_module,
                  tl, vl, device, args, log=None):
    if not args.no_configure_backbones:
        configure_backbones(img_encoder)

    opt_sim   = make_optimizer_sim(img_encoder, text_embed, sim_module, args)
    opt_det   = make_optimizer_det(img_encoder, text_embed, det_module, args)
    sched_sim = make_scheduler(opt_sim, args)
    sched_det = make_scheduler(opt_det, args)
    early     = EarlyStopping(args.early_stop_patience,
                               args.early_stop_min_delta, mode="max")
    best_auc  = -1.
    best_st   = None

    for ep in range(1, args.epochs + 1):
        ls, ld = train_one_epoch(
            img_encoder, text_embed, sim_module, det_module,
            tl, device, opt_sim, opt_det, sched_sim, sched_det, ep)
        va = eval_epoch(img_encoder, text_embed, sim_module, det_module,
                        vl, device)

        cur_lr = opt_det.param_groups[-1]["lr"]
        line = (f"[{LOG_TAG}] Ep {ep:02d}/{args.epochs}"
                f"  ls={ls:.4f}  ld={ld:.4f}"
                f"  va_f1_opt={va['bin_f1_opt']:.4f}"
                f"  va_auc={va['bin_auc']:.4f}"
                f"  va_acc={va['bin_acc']:.4f}"
                f"  lr={cur_lr:.2e}")
        print(line)
        if log is not None:
            log.append(line)

        if not math.isnan(va["bin_auc"]) and va["bin_auc"] > best_auc:
            best_auc = va["bin_auc"]
            best_st  = {
                "img": copy.deepcopy(img_encoder.state_dict()),
                "emb": copy.deepcopy(text_embed.state_dict()),
                "sim": copy.deepcopy(sim_module.state_dict()),
                "det": copy.deepcopy(det_module.state_dict()),
            }

        monitor = va["bin_auc"] if not math.isnan(va["bin_auc"]) else va["bin_f1"]
        if early.step(monitor):
            msg = (f"[{LOG_TAG}] Early stopping at epoch {ep}"
                   f" (best_auc={best_auc:.4f})")
            print(msg)
            if log:
                log.append(msg)
            break

    if best_st is not None:
        img_encoder.load_state_dict(best_st["img"])
        text_embed.load_state_dict(best_st["emb"])
        sim_module.load_state_dict(best_st["sim"])
        det_module.load_state_dict(best_st["det"])
        print(f"[{LOG_TAG}] best val AUC={best_auc:.4f} 가중치 복원 완료")

    return img_encoder, text_embed, sim_module, det_module


# ════════════════════════════════════════════════════════════════════════════════
# 16. DataLoader 헬퍼
# ════════════════════════════════════════════════════════════════════════════════

def _make_dl(df, data_root, vocab, args, shuffle, sampler=None):
    ds = DGM4CAFEDataset(df, data_root, vocab, args.max_len)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              collate_fn=collate_dgm4_cafe,
              pin_memory=torch.cuda.is_available(), drop_last=shuffle)
    if sampler is not None:
        kw["sampler"] = sampler
        kw["shuffle"] = False
    else:
        kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


# ════════════════════════════════════════════════════════════════════════════════
# 17. eval_only / run_official / run_kfold
# ════════════════════════════════════════════════════════════════════════════════

def run_eval_only(img_encoder, text_embed, sim_module, det_module,
                  loader, device, split_name, result_dir, n_samples):
    print(f"\n[{LOG_TAG}] device={device} | split={split_name}"
          f" | n={n_samples} | eval_only")
    full_df = collect_predictions(img_encoder, text_embed, sim_module,
                                   det_module, loader, device)
    blk, m  = eval_report_block(full_df, label=f"{LOG_TAG}_{split_name}")
    print(blk, end="")
    os.makedirs(result_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rp = os.path.join(result_dir,
                      f"cafe_dgm4_{split_name}_eval_only_{ts}.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(blk)
    full_df.to_csv(rp.replace(".txt", "_predictions.csv"), index=False)
    print(f"[{LOG_TAG}] saved: {rp}")
    return m


def run_official(dfs, device, vocab, args, result_dir):
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO()
    rep.write(f"###### {LOG_TAG} (Official Split) ######\n\n")

    tr_ds    = DGM4CAFEDataset(dfs["train"], args.data_root, vocab, args.max_len)
    _sampler = (make_weighted_sampler(tr_ds.binary_lbls, args.sampler_alpha)
                if getattr(args, "weighted_sampler", False) else None)
    tl = _make_dl(dfs["train"],      args.data_root, vocab, args,
                  shuffle=(_sampler is None), sampler=_sampler)
    vl = _make_dl(dfs["validation"], args.data_root, vocab, args, shuffle=False)
    el = _make_dl(dfs["test"],       args.data_root, vocab, args, shuffle=False)

    img_encoder, text_embed, sim_module, det_module = _build_models(
        len(vocab), device)
    log = []
    img_encoder, text_embed, sim_module, det_module = train_one_run(
        img_encoder, text_embed, sim_module, det_module,
        tl, vl, device, args, log)
    for line in log:
        rep.write(line + "\n")

    print(f"\n[{LOG_TAG}] validation 예측 수집...")
    val_df    = collect_predictions(img_encoder, text_embed, sim_module,
                                     det_module, vl, device)
    blk_v, _ = eval_report_block(val_df, label=f"{LOG_TAG}_val")
    print(blk_v, end="")
    rep.write(blk_v)

    print(f"\n[{LOG_TAG}] test 예측 수집...")
    full_df = collect_predictions(img_encoder, text_embed, sim_module,
                                   det_module, el, device)
    blk, m  = eval_report_block(full_df, label=f"{LOG_TAG}_test")
    print(blk, end="")
    rep.write(blk)

    ckpt_path = os.path.join(args.checkpoint_dir,
                             f"cafe_dgm4_official_{ts}.pt")
    save_checkpoint(ckpt_path, img_encoder, text_embed, sim_module,
                    det_module, vocab, args, m)

    summ = (f"{LOG_TAG}: ACC={m['bin_acc']:.4f}  F1={m['bin_f1']:.4f}"
            f"  F1_opt={m['bin_f1_opt']:.4f}"
            f"  AUC={m.get('bin_auc', float('nan')):.4f}")
    print("\n" + "=" * 70 + "\n" + summ + "\n" + "=" * 70)

    path = os.path.join(result_dir, f"cafe_dgm4_official_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rep.getvalue())
    full_df.to_csv(path.replace(".txt", "_test_predictions.csv"), index=False)
    print(f"[{LOG_TAG}] 리포트: {path}")


def run_kfold(dfs, device, vocab, args, result_dir):
    df_tv   = _ensure_dgm4_columns(
        pd.concat([dfs["train"], dfs["validation"]], ignore_index=True))
    y_tv    = df_tv["binary_label"].values
    k       = args.kfold_splits
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rep = io.StringIO()
    all_rep.write(f"###### {LOG_TAG} (k={k} KFold) ######\n")
    el      = _make_dl(dfs["test"], args.data_root, vocab, args, shuffle=False)
    skf     = StratifiedKFold(n_splits=k, shuffle=True,
                              random_state=args.kfold_random_state)
    fold_m  = []

    for fold, (tr_idx, va_idx) in enumerate(
            skf.split(np.zeros(len(df_tv)), y_tv), 1):
        print(f"\n{'='*70}\n[Fold {fold}/{k}] {LOG_TAG}\n{'='*70}")
        df_tr_f  = df_tv.iloc[tr_idx].reset_index(drop=True)
        df_va_f  = df_tv.iloc[va_idx].reset_index(drop=True)
        tr_ds    = DGM4CAFEDataset(df_tr_f, args.data_root, vocab, args.max_len)
        _sampler = (make_weighted_sampler(tr_ds.binary_lbls, args.sampler_alpha)
                    if getattr(args, "weighted_sampler", False) else None)
        tl = _make_dl(df_tr_f, args.data_root, vocab, args,
                      shuffle=(_sampler is None), sampler=_sampler)
        vl = _make_dl(df_va_f, args.data_root, vocab, args, shuffle=False)

        img_encoder, text_embed, sim_module, det_module = _build_models(
            len(vocab), device)
        log = []
        img_encoder, text_embed, sim_module, det_module = train_one_run(
            img_encoder, text_embed, sim_module, det_module,
            tl, vl, device, args, log)

        full_df = collect_predictions(img_encoder, text_embed, sim_module,
                                       det_module, el, device)
        blk, m  = eval_report_block(full_df, label=f"Fold{fold}")
        print(blk, end="")
        ckpt = os.path.join(args.checkpoint_dir,
                            f"cafe_dgm4_kfold_{ts}_fold{fold}.pt")
        save_checkpoint(ckpt, img_encoder, text_embed, sim_module,
                        det_module, vocab, args, m)
        print(f"\n[Fold {fold}] F1_opt={m['bin_f1_opt']:.4f}"
              f"  AUC={m.get('bin_auc', float('nan')):.4f}")
        all_rep.write(f"\n{'#'*80}\n### FOLD {fold}\n{'#'*80}\n{blk}")
        fold_m.append(m)

        del img_encoder, text_embed, sim_module, det_module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _mean(k_): return float(np.mean([r[k_] for r in fold_m]))
    g = (f"{LOG_TAG}: F1_opt={_mean('bin_f1_opt'):.4f}"
         f"  AUC={_mean('bin_auc'):.4f}")
    all_rep.write(f"\n\n{'#'*80}\n### SUMMARY\n{g}\n")
    path = os.path.join(result_dir, f"cafe_dgm4_kfold_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(all_rep.getvalue())
    print(f"\n[{LOG_TAG}] 리포트: {path}\n{g}")


# ════════════════════════════════════════════════════════════════════════════════
# 18. main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(description="CAFE (cyxanna/CAFE) x DGM4")
    pa.add_argument("--data_root",      default=DEFAULT_DATA_ROOT)
    pa.add_argument("--checkpoint_dir", default="./checkpoints")
    pa.add_argument("--result_dir",     default="./results")
    pa.add_argument("--eval_only",      action="store_true")
    pa.add_argument("--checkpoint",     type=str, default="")
    pa.add_argument("--split",          type=str, default="test",
                    choices=("train", "validation", "test"))
    pa.add_argument("--batch_size",           type=int,   default=BATCH)
    pa.add_argument("--epochs",               type=int,   default=EPOCHS)
    pa.add_argument("--lr",                   type=float, default=LR)
    pa.add_argument("--weight_decay",         type=float, default=PAPER_WEIGHT_DECAY)
    pa.add_argument("--warmup_epochs",        type=int,   default=-1)
    pa.add_argument("--num_workers",          type=int,   default=NUM_WORKERS)
    pa.add_argument("--seed",                 type=int,   default=SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    pa.add_argument("--sampler_alpha",        type=float, default=SAMPLER_ALPHA)
    pa.add_argument("--limit",                type=int,   default=0)
    pa.add_argument("--weighted_sampler",     action="store_true")
    pa.add_argument("--kfold",                action="store_true")
    pa.add_argument("--kfold_splits",         type=int,   default=KFOLD_SPLITS)
    pa.add_argument("--kfold_random_state",   type=int,   default=KFOLD_RANDOM_STATE)
    pa.add_argument("--max_len",              type=int,   default=SEQ_LEN)
    pa.add_argument("--vocab_min_freq",       type=int,   default=VOCAB_MIN_FREQ)
    pa.add_argument("--no_configure_backbones", action="store_true")
    pa.add_argument("--no_progress",            action="store_true")

    args = pa.parse_args()

    if args.warmup_epochs < 0:
        args.warmup_epochs = default_paper_warmup_epochs(args.epochs)
    if args.epochs > 1 and args.warmup_epochs > args.epochs - 1:
        args.warmup_epochs = args.epochs - 1
    if args.no_progress:
        os.environ["DGM4_NO_TQDM"] = "1"

    set_seed(args.seed)
    os.makedirs(args.result_dir,     exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    sampler_note = ("WeightedSampler"
                    if getattr(args, "weighted_sampler", False) else "shuffle=True")
    mode = "KFold" if args.kfold else "Official Split"
    print(f"\n{LOG_TAG} | device={DEVICE} | mode={mode}")
    print(f"  lr={args.lr:.2e}  wd={args.weight_decay}"
          f"  warmup={args.warmup_epochs}ep"
          f"  epochs={args.epochs}  batch={args.batch_size}"
          f"  sampler={sampler_note}")

    dfs = load_dgm4_splits(args.data_root, LOG_TAG)
    if args.limit > 0:
        dfs = {k: v.head(args.limit).reset_index(drop=True)
               for k, v in dfs.items()}

    train_texts = _ensure_dgm4_columns(dfs["train"])["text"].tolist()
    print(f"[{LOG_TAG}] vocab 구축 중 (train {len(train_texts)}개,"
          f" min_freq={args.vocab_min_freq})...")
    vocab = build_vocab(train_texts, min_freq=args.vocab_min_freq)
    print(f"[{LOG_TAG}] vocab 크기: {len(vocab)}")

    if args.eval_only:
        img_encoder, text_embed, sim_module, det_module = _build_models(
            len(vocab), DEVICE)
        if args.checkpoint:
            ckpt_vocab = load_checkpoint(
                img_encoder, text_embed, sim_module, det_module,
                args.checkpoint, DEVICE)
            if ckpt_vocab is not None:
                vocab = ckpt_vocab
                print(f"[{LOG_TAG}] 체크포인트 vocab 사용 (크기={len(vocab)})")
        else:
            print(f"[{LOG_TAG}] WARNING: --checkpoint 없음 → 랜덤 초기화")
        ev_df = _ensure_dgm4_columns(dfs[args.split])
        ev_ds = DGM4CAFEDataset(ev_df, args.data_root, vocab, args.max_len)
        ev_loader = DataLoader(
            ev_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_dgm4_cafe,
            pin_memory=torch.cuda.is_available())
        run_eval_only(img_encoder, text_embed, sim_module, det_module,
                      ev_loader, DEVICE, args.split, args.result_dir,
                      len(ev_ds))
        return

    if args.kfold:
        run_kfold(dfs, DEVICE, vocab, args, args.result_dir)
    else:
        run_official(dfs, DEVICE, vocab, args, args.result_dir)


if __name__ == "__main__":
    main()