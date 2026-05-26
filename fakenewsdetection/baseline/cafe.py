"""
CAFE Baseline for HARFM (4-class: HR, HF, AR, AF)
- 데이터: HARFM.csv (text=final_headline, image=image_path, label=4_way_label)
- Text: FastCNN (learned embedding 200-dim), Image: ResNet18 (512-dim)
- 태스크: Similarity (auxiliary) + Detection (4-class) 
"""
import os
import io
import sys
import re
import math
import copy
import gc
import random
from contextlib import redirect_stdout
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from torchvision import models, transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# -------------------------
# 0. 재현성 세팅
# -------------------------
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)
# -------------------------
# 1. 기본 설정
# -------------------------
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR="results"
os.makedirs(RESULT_DIR, exist_ok=True)
_SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
_FND_ROOT=os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
CSV_PATH=os.path.join(_FND_ROOT, "HARFM.csv")
#모델에 필요한 컬럼 3가지만 사용
#text: final_headline, image: image_path, label: 4_way_label(언니가 말한대로!)

TEXT_DIM = 200#텍스트 임베딩할때 200차원으로 임베딩 아래도 마찬가지로 cafe에서 쓰는 일반적인 기준?을 그대로 지키는게 좋아보여서...
SEQ_LEN = 64
IMG_SIZE = 224
VOCAB_MIN_FREQ = 2
BATCH = 16
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_DELTA = 1e-4
SAMPLER_ALPHA = 0.5
USE_KFOLD = True
KFOLD_SPLITS = 5
KFOLD_TEST_SIZE = 0.2
KFOLD_SHUFFLE = True
KFOLD_RANDOM_STATE = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2
LABEL_MAP = {"HR": 0, "HF": 1, "AR": 2, "AF": 3}
REQUIRED_COLS = {"final_headline", "image_path", "4_way_label"}
# -------------------------
# 2. 데이터 로드 (HARFM.csv — final_headline, image_path, 4_way_label)
# -------------------------
def _as_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()

def _resolve_image_path(p, base_dir):
    p = _as_str(p)
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))

def _strip_quotes(val):
    if not isinstance(val, str):
        return val
    val = val.strip()
    while len(val) >= 2 and (
        (val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")
    ):
        val = val[1:-1].strip()
    return val

print("cafe모델을 학습하기 위해 HARFM.csv를 로드시작~ (text=final_headline, image=image_path, label=4_way_label)")
raw=pd.read_csv(CSV_PATH, low_memory=False)

missing=REQUIRED_COLS - set(raw.columns)
if missing:
    raise ValueError(f"HARFM CSV에 다음 컬럼이 없습니다: {missing}")

raw = raw[list(REQUIRED_COLS)].copy()
raw["text"] = raw["final_headline"].fillna("").astype(str).apply(_strip_quotes)
raw["image_path"] = raw["image_path"].apply(lambda p: _resolve_image_path(p, _FND_ROOT))
raw["label"] = raw["4_way_label"].map(LABEL_MAP)
raw = raw[raw["label"].notna()].astype({"label": int})
data = raw[["text", "image_path", "label"]].copy()
print(f"Dataset loaded: {len(data)} rows (4-way)")
print("Label counts:", data["label"].value_counts().sort_index().to_dict())
n_before = len(data)
data = data[
    data["text"].astype(str).str.strip().ne("") &
    data["image_path"].apply(
        lambda p: isinstance(p, str) and p != "" and os.path.isfile(p)
    )
].reset_index(drop=True)
n_removed = n_before - len(data)
print(f"\n[전처리] 멀티모달 유효 샘플(text+image) 필터: {n_before} → {len(data)} ({n_removed}개 제거)")
print("[전처리] 제거 후 label 분포:", data["label"].value_counts().sort_index().to_dict())
# -------------------------
# 3. Vocabulary & Embedding (text = final_headline 하나로만 구축)
# -------------------------
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    return text.split()

def build_vocab(texts, min_freq=VOCAB_MIN_FREQ):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in counter.most_common():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab

VOCAB = build_vocab(data["text"].tolist())
PAD_IDX = VOCAB["<pad>"]
UNK_IDX = VOCAB["<unk>"]

def text_to_ids(text, max_len=SEQ_LEN):#단어 토큰화하고 vocab id로 바꾸고 학습가능한 임베딩200차원으로 넣어서 텐서화
    tokens = tokenize(text)
    ids = [VOCAB.get(w, UNK_IDX) for w in tokens[:max_len]]
    ids = ids + [PAD_IDX] * (max_len - len(ids))
    return ids

class FastCNN(nn.Module):
    def __init__(self, channel=32, kernel_size=(1, 2, 4, 8)):
        super().__init__()
        self.fast_cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(TEXT_DIM, channel, kernel_size=k),
                nn.BatchNorm1d(channel),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
            for k in kernel_size
        ])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_out = [m(x).squeeze(-1) for m in self.fast_cnn]
        return torch.cat(x_out, 1)


class EncodingPart(nn.Module):#같은 차원으로 공유 하는 표현 만드는 인코딩 부분 
    def __init__(self, shared_image_dim=128, shared_text_dim=128):
        super().__init__()#텍스트에 대해서는 (8,64,200)
        self.shared_text_encoding = FastCNN(channel=32, kernel_size=(1, 2, 4, 8))#1,2,4,8크기커널로 1d convolution
        self.shared_text_linear = nn.Sequential(
            nn.Linear(128, 64),#(B,128)
            nn.BatchNorm1d(64),#(B,64)
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),#
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        self.shared_image = nn.Sequential(
            nn.Linear(512, 256),#(B,512)
            nn.BatchNorm1d(256),#(B,256)
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_image_dim),#(B,128)
            nn.BatchNorm1d(shared_image_dim),#(B,128)
            nn.ReLU()
        )

    def forward(self, text, image):
        text_encoding = self.shared_text_encoding(text)
        text_shared = self.shared_text_linear(text_encoding)#(B,64) -> (B,128)
        image_shared = self.shared_image(image)#(B,128)
        return text_shared, image_shared


class SimilarityModule(nn.Module):#텍스트,이미지를 같은 공간에 정렬시키고 정렬되 ㄴ표현으로 유사도 예측하는 모듈부분
    def __init__(self, shared_dim=128, sim_dim=64):#
        super().__init__()
        self.encoding = EncodingPart()#text shared(B,128) 랑 image_shared(B,128) 만들어주는 부분
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),#(B,128) -> (B,128)
            nn.BatchNorm1d(shared_dim),#(B,128)
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),#(B,128) -> (B,64)
            nn.BatchNorm1d(sim_dim),#(B,64)
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),#(B,128) -> (B,128)
            nn.BatchNorm1d(shared_dim),#(B,128)
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),#(B,128) -> (B,64)
            nn.BatchNorm1d(sim_dim),#(B,64)
            nn.ReLU()
        )
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(sim_dim * 2),
            nn.Linear(sim_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, text, image):
        text_enc, image_enc = self.encoding(text, image)
        text_aligned = self.text_aligner(text_enc)#텍스트헤드라인을 정렬공간으로 보낸 벡터(B,64)
        image_aligned = self.image_aligner(image_enc)#이미지를 정렬공간으로 보낸 벡터(B,64)
        sim_feat = torch.cat([text_aligned, image_aligned], 1)
        pred_sim = self.sim_classifier(sim_feat)
        return text_aligned, image_aligned, pred_sim

class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
        )

    def forward(self, x):
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)

class AmbiguityLearning(nn.Module):#64차원 텍스트랑 이미지 aligned벡터를 VAE스타일 인코더에 넣어서 분포 얻고 KL divergence로 텍스트랑 이미지가 얼마나 모호한지(서로 다른거 담는지) 스칼라화
    def __init__(self):
        super().__init__()
        self.encoding = EncodingPart()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_encoding, image_encoding):
        p_z1 = self.encoder_text(text_encoding)
        p_z2 = self.encoder_image(image_encoding)
        z1, z2 = p_z1.rsample(), p_z2.rsample()
        kl_1_2 = p_z1.log_prob(z1) - p_z2.log_prob(z1)
        kl_2_1 = p_z2.log_prob(z2) - p_z1.log_prob(z2)
        skl = (kl_1_2 + kl_2_1) / 2.0#skl값이 높을수록 텍스트랑 이미지 잘 안 맞아!작으면 맞는 쌍임
        return torch.sigmoid(skl)

class UnimodalDetection(nn.Module):#16차원으로 줄여서 텍스트~이미지only 표현 얻기 
    def __init__(self, shared_dim=128, prime_dim=16):
        super().__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )
        self.image_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )

    def forward(self, text_enc, image_enc):
        return self.text_uni(text_enc), self.image_uni(image_enc)


class CrossModule4Batch(nn.Module):#pooling+linear로 correlation 한 벡터로 압축
    def __init__(self, corre_out_dim=64):
        super().__init__()
        self.corre_dim = 64
        self.softmax = nn.Softmax(-1)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_in = text.unsqueeze(2)
        image_in = image.unsqueeze(1)
        sim = torch.matmul(text_in, image_in) / math.sqrt(text.shape[1])
        corr = self.softmax(sim)
        corr_p = self.pooling(corr).squeeze(-1)
        return self.c_specific_2(corr_p)


class DetectionModule(nn.Module):#textraw, image raw, text x image정렬 벡터
    #정렬된 표현과 원본 텍스트/이미지 표현을 써서 모호성 학습하고 uni+cross모달 다 합쳐서 4 way로 분류
    def __init__(self, num_classes=4, feature_dim=64+16+16, h_dim=64):
        super().__init__()
        self.encoding = EncodingPart()
        self.ambiguity_module = AmbiguityLearning()
        self.uni_repre = UnimodalDetection()
        self.cross_module = CrossModule4Batch()
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_classes)
        )

    def forward(self, text_raw, image_raw, text, image):
        skl = self.ambiguity_module(text, image)
        text_prime, image_prime = self.encoding(text_raw, image_raw)
        text_prime, image_prime = self.uni_repre(text_prime, image_prime)
        correlation = self.cross_module(text, image)
        weight_uni = (1 - skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        text_final = weight_uni * text_prime
        img_final = weight_uni * image_prime
        corre_final = weight_corre * correlation
        final = torch.cat([text_final, img_final, corre_final], 1)
        return self.classifier_corre(final)

#이미지인코더:resnet18을 나열해서 마지막 FClayer는 제거하고 sequential로 묶고 flatten하고 512차원으로 압축
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.features(x).flatten(1)

image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#텍스트임베딩:embedding layer로 텍스트 토큰 id를 200차원 임베딩으로 변환
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=200):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

    def forward(self, ids):
        return self.embed(ids).float()

#cafe베이스라인에 쓰일 데이터셋 준비
class CAFEDataset(Dataset):
    """text=final_headline, image=image_path, label=4_way_label"""
    def __init__(self, df, vocab, image_transform_fn, max_len=SEQ_LEN, is_train=True):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.image_transform = image_transform_fn
        self.max_len = max_len
        self.is_train = is_train
    def __len__(self):
        return len(self.df)
    def _get_text(self, row):
        t = row.get("text", "")
        return t if isinstance(t, str) else ""
    def _get_image(self, row):
        path = row.get("image_path", "")
        img = Image.open(path).convert("RGB")
        return img
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self._get_text(row)
        ids = text_to_ids(text, self.max_len)
        img = self._get_image(row)
        img_t = self.image_transform(img)
        label4 = int(row["label"])
        return {
            "text_ids": torch.tensor(ids, dtype=torch.long),
            "image": img_t,
            "label4": torch.tensor(label4, dtype=torch.long),
        }

def collate_cafe(batch):
    text_ids = torch.stack([b["text_ids"] for b in batch])
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label4"] for b in batch])
    return text_ids, images, labels


def make_weighted_sampler(labels, alpha: float = SAMPLER_ALPHA):
    cnt = pd.Series(labels).value_counts().to_dict()
    w = [(1.0 / cnt[l]) ** alpha for l in labels]
    return WeightedRandomSampler(w, len(w), replacement=True)

#similarity 학습을 위해 matched/unmatched 준비
def prepare_similarity_data(text_emb, image_feat, n):
    matched_t = text_emb
    matched_i = image_feat
    shift = min(3, n - 1) if n > 1 else 0
    unmatched_i = image_feat.roll(shifts=shift, dims=0)
    return matched_t, matched_i, unmatched_i

class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, metric_value: float) -> bool:
        if self.best is None:
            self.best = metric_value
            return False
        if self.mode == "max":
            improved = (metric_value - self.best) > self.min_delta
        else:
            improved = (self.best - metric_value) > self.min_delta
        if improved:
            self.best = metric_value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
    #학습 시 (matched_t, matched_i)에는 레이블 +1, (matched_t, unmatched_i)에는 -1 줘서
    #CosineEmbeddingLoss로 "맞는 쌍은 가깝게, 안 맞는 쌍은 멀게"학습


def auc_scores_from_proba(yt: np.ndarray, proba: np.ndarray) -> dict:
    """ROC-AUC from 4-class softmax: macro OVR, plus merged H/A and R/F scores."""
    out = {}
    if len(yt) == 0 or proba.shape[0] != len(yt):
        for k in ("auc_4_ovr_macro", "auc_ha", "auc_rf"):
            out[k] = float("nan")
        return out
    try:
        out["auc_4_ovr_macro"] = float(
            roc_auc_score(
                yt,
                proba,
                multi_class="ovr",
                average="macro",
                labels=[0, 1, 2, 3],
            )
        )
    except ValueError:
        out["auc_4_ovr_macro"] = float("nan")
    yh = yt // 2
    p_ai = proba[:, 2:4].sum(axis=1)
    try:
        out["auc_ha"] = float(roc_auc_score(yh, p_ai))
    except ValueError:
        out["auc_ha"] = float("nan")
    yr = yt % 2
    p_fake = proba[:, [1, 3]].sum(axis=1)
    try:
        out["auc_rf"] = float(roc_auc_score(yr, p_fake))
    except ValueError:
        out["auc_rf"] = float("nan")
    return out


def evaluate_cafe(sim_module, det_module, text_embed_module, img_encoder, loader):
    sim_module.eval()
    det_module.eval()
    all_true_4, all_pred_4 = [], []
    prob_chunks = []

    with torch.no_grad():
        for text_ids, images, labels in tqdm(loader, desc="Evaluation (CAFE 4C)"):
            text_ids = text_ids.to(DEVICE)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            text_emb = text_embed_module(text_ids)
            img_feat = img_encoder(images)
            ta, ia, _ = sim_module(text_emb, img_feat)
            pred = det_module(text_emb, img_feat, ta, ia)
            prob_chunks.append(
                torch.softmax(pred, dim=-1).detach().cpu().numpy()
            )
            p4 = pred.argmax(1).cpu().numpy()
            y4 = labels.cpu().numpy()
            all_pred_4.extend(p4.tolist())
            all_true_4.extend(y4.tolist())

    all_true_4 = np.array(all_true_4)
    all_pred_4 = np.array(all_pred_4)
    y_ha, p_ha = all_true_4 // 2, all_pred_4 // 2
    y_rf, p_rf = all_true_4 % 2, all_pred_4 % 2

    acc_4 = accuracy_score(all_true_4, all_pred_4)
    acc_ha = accuracy_score(y_ha, p_ha)
    acc_rf = accuracy_score(y_rf, p_rf)
    f1_4 = f1_score(all_true_4, all_pred_4, average="macro")
    f1_ha = f1_score(y_ha, p_ha, average="macro")
    f1_rf = f1_score(y_rf, p_rf, average="macro")

    target4 = ["Human Real", "Human Fake", "AI Real", "AI Fake"]
    print("\n=== CAFE Human vs AI ===")
    print(classification_report(y_ha, p_ha, target_names=["Human", "AI"], digits=4))
    print("=== CAFE Real vs Fake ===")
    print(classification_report(y_rf, p_rf, target_names=["Real", "Fake"], digits=4))
    print("=== CAFE 4-class ===")
    print(classification_report(all_true_4, all_pred_4, target_names=target4, digits=4))
    proba = np.vstack(prob_chunks) if prob_chunks else np.zeros((0, 4), dtype=np.float64)
    aucm = auc_scores_from_proba(all_true_4, proba)
    print(
        f"\n[CAFE] ROC-AUC: 4-class (macro-OVR)={aucm['auc_4_ovr_macro']:.4f} | "
        f"H/A={aucm['auc_ha']:.4f} | R/F={aucm['auc_rf']:.4f}"
    )
    print(
        f"\n[CAFE] "
        f"H/A Acc={acc_ha:.4f} F1={f1_ha:.4f} | "
        f"R/F Acc={acc_rf:.4f} F1={f1_rf:.4f} | "
        f"4-way Acc={acc_4:.4f} F1={f1_4:.4f} | "
        f"AUC(4c)={aucm['auc_4_ovr_macro']:.4f} AUC(H/A)={aucm['auc_ha']:.4f} "
        f"AUC(R/F)={aucm['auc_rf']:.4f}"
    )

    return {
        "acc_ha": acc_ha,
        "acc_rf": acc_rf,
        "acc_4": acc_4,
        "f1_ha": f1_ha,
        "f1_rf": f1_rf,
        "f1_4": f1_4,
        "auc_4_ovr_macro": aucm["auc_4_ovr_macro"],
        "auc_ha": aucm["auc_ha"],
        "auc_rf": aucm["auc_rf"],
    }


def _validate_cafe(sim_module, det_module, text_embed_module, img_encoder, val_loader):
    sim_module.eval()
    det_module.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for text_ids, images, labels in val_loader:
            text_ids = text_ids.to(DEVICE)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            text_emb = text_embed_module(text_ids)
            img_feat = img_encoder(images)
            ta, ia, _ = sim_module(text_emb, img_feat)
            pred = det_module(text_emb, img_feat, ta, ia)
            all_pred.extend(pred.argmax(1).cpu().numpy().tolist())
            all_true.extend(labels.cpu().numpy().tolist())
    return f1_score(all_true, all_pred, average="macro")

#학습: similarity + detection 학습
def train_epoch_cafe(sim_module, det_module, text_embed_module, img_encoder,
                     train_loader, opt_sim, opt_det):
    sim_module.train()
    det_module.train()
    loss_sim_tot, loss_det_tot = 0.0, 0.0
    cnt_sim, cnt_det = 0, 0
    loss_fn_sim = nn.CosineEmbeddingLoss()
    loss_fn_det = nn.CrossEntropyLoss()
    for text_ids, images, labels in tqdm(train_loader, desc="Train"):
        text_ids = text_ids.to(DEVICE)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        # img_encoder는 requires_grad=False로 얼려져 있으므로 그래프를 만들 필요가 없음
        with torch.no_grad():
            img_feat = img_encoder(images)

        # 1) Similarity 학습용 forward (여기서 backward로 그래프가 소멸됨)
        text_emb = text_embed_module(text_ids)
        n = text_emb.shape[0]
        mat_t, mat_i, unmat_i = prepare_similarity_data(text_emb, img_feat, n)
        if mat_t.shape[0] == 0:
            continue
        ta_m, ia_m, pred_m = sim_module(mat_t, mat_i)
        ta_u, ia_u, pred_u = sim_module(mat_t, unmat_i)
        ta_cat = torch.cat([ta_m, ta_u], 0)
        ia_cat = torch.cat([ia_m, ia_u], 0)
        sim_label = torch.cat([
            torch.ones(pred_m.shape[0], device=DEVICE),
            -torch.ones(pred_u.shape[0], device=DEVICE)
        ])
        loss_sim = loss_fn_sim(ta_cat, ia_cat, sim_label)
        opt_sim.zero_grad()
        loss_sim.backward()
        opt_sim.step()
        loss_sim_tot += loss_sim.item() * n * 2
        cnt_sim += n * 2

        # 2) Detection 학습은 새 그래프로 다시 forward해야 함 (위 backward로 그래프가 해제됨)
        text_emb_det = text_embed_module(text_ids)
        ta, ia, _ = sim_module(text_emb_det, img_feat)
        pred_det = det_module(text_emb_det, img_feat, ta, ia)
        loss_det = loss_fn_det(pred_det, labels)
        opt_det.zero_grad()
        loss_det.backward()
        opt_det.step()
        loss_det_tot += loss_det.item() * n
        cnt_det += n
    return loss_sim_tot / max(1, cnt_sim), loss_det_tot / max(1, cnt_det)


def train_cafe_4c(sim_module, det_module, text_embed_module, img_encoder,
                  train_loader, val_loader, epochs=EPOCHS):
    img_encoder.to(DEVICE)
    sim_module.to(DEVICE)
    det_module.to(DEVICE)
    text_embed_module.to(DEVICE)
    for p in img_encoder.parameters():
        p.requires_grad = False
    opt_sim = torch.optim.AdamW(
        list(sim_module.parameters()) + list(text_embed_module.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )
    opt_det = torch.optim.AdamW(
        list(det_module.parameters()) + list(text_embed_module.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )
    early = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, mode="max")
    best_val = -1.0
    best_state = None
    for epoch in range(epochs):
        ls, ld = train_epoch_cafe(
            sim_module, det_module, text_embed_module, img_encoder,
            train_loader, opt_sim, opt_det
        )
        val_f1 = _validate_cafe(sim_module, det_module, text_embed_module, img_encoder, val_loader)
        print(f"[CAFE 4C] Epoch {epoch+1} loss_sim={ls:.4f} loss_det={ld:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val:
            best_val = val_f1
            best_state = {
                "sim": copy.deepcopy(sim_module.state_dict()),
                "det": copy.deepcopy(det_module.state_dict()),
                "emb": copy.deepcopy(text_embed_module.state_dict()),
            }
        if early.step(val_f1):
            print(f"[CAFE 4C] Early stopping at epoch {epoch+1}")
            break
    if best_state is not None:
        sim_module.load_state_dict(best_state["sim"])
        det_module.load_state_dict(best_state["det"])
        text_embed_module.load_state_dict(best_state["emb"])
    return sim_module, det_module, text_embed_module, img_encoder

class Tee(io.TextIOBase):#그냥 이거는 콘솔창에 출력하는거 뿐만 아니라 파일에도 출력하는 용도
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)
    def flush(self):
        for s in self.streams:
            if hasattr(s, "flush"):
                s.flush()

def kfold_cafe_4c_with_reports(
    full_df,
    k=KFOLD_SPLITS,
    test_size=KFOLD_TEST_SIZE,
    shuffle=KFOLD_SHUFFLE,
    random_state=KFOLD_RANDOM_STATE,
):
    trainval_df, test_df = train_test_split(
        full_df,
        test_size=test_size,
        stratify=full_df["label"],
        random_state=random_state,
    )
    print(f"[KFold-CAFE 4C] Split: TrainVal={len(trainval_df)}, Test={len(test_df)}")

    test_ds = CAFEDataset(test_df, VOCAB, image_transform, max_len=SEQ_LEN, is_train=False)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_cafe,
    )

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    all_report = io.StringIO()
    all_report.write(f"###### CAFE 4C (k={k}) ######\n")
    metrics_4c = []
    y = trainval_df["label"].values

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(trainval_df)), y), start=1):
        print(f"\n========== [Fold {fold}/{k} - CAFE 4C] ==========")
        tr_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
        va_df = trainval_df.iloc[va_idx].reset_index(drop=True)

        train_ds = CAFEDataset(tr_df, VOCAB, image_transform, max_len=SEQ_LEN, is_train=True)
        val_ds = CAFEDataset(va_df, VOCAB, image_transform, max_len=SEQ_LEN, is_train=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH,
            shuffle=False,
            sampler=make_weighted_sampler(tr_df["label"].tolist()),
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
            collate_fn=collate_cafe, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
            collate_fn=collate_cafe,
        )

        text_embed_module = TextEmbedding(len(VOCAB), TEXT_DIM).to(DEVICE)
        img_encoder = ImageEncoder().to(DEVICE)
        sim_module = SimilarityModule().to(DEVICE)
        det_module = DetectionModule(num_classes=4).to(DEVICE)

        buffer = io.StringIO()
        tee = Tee(sys.stdout, buffer)
        with redirect_stdout(tee):
            print("\n" + "=" * 70)
            print(f"[Fold {fold}] CAFE 4-class")
            print("=" * 70)
            sim_module, det_module, text_embed_module, img_encoder = train_cafe_4c(
                sim_module, det_module, text_embed_module, img_encoder,
                train_loader, val_loader,
            )
            res = evaluate_cafe(sim_module, det_module, text_embed_module, img_encoder, test_loader)
            print("\n" + "=" * 70)
            print(f"[Fold {fold}] SUMMARY (CAFE, Acc/F1 on Test)")
            print("=" * 70)
            print(
                f"CAFE 4C: "
                f"H/A Acc={res['acc_ha']:.4f}, F1={res['f1_ha']:.4f} | "
                f"R/F Acc={res['acc_rf']:.4f}, F1={res['f1_rf']:.4f} | "
                f"4-way Acc={res['acc_4']:.4f}, F1={res['f1_4']:.4f} | "
                f"AUC(4c)={res['auc_4_ovr_macro']:.4f} AUC(H/A)={res['auc_ha']:.4f} "
                f"AUC(R/F)={res['auc_rf']:.4f}"
            )
            print("=" * 70)

        all_report.write(f"\n\n{'#'*90}\n### FOLD {fold}/{k} REPORT (CAFE 4C)\n{'#'*90}\n")
        all_report.write(buffer.getvalue())
        all_report.flush()
        metrics_4c.append(res)
        gc.collect()

    def _mean_metric(dict_list, key):
        vals = [d[key] for d in dict_list if d.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    mean_acc_ha = _mean_metric(metrics_4c, "acc_ha")
    mean_acc_rf = _mean_metric(metrics_4c, "acc_rf")
    mean_acc_4 = _mean_metric(metrics_4c, "acc_4")
    mean_f1_ha = _mean_metric(metrics_4c, "f1_ha")
    mean_f1_rf = _mean_metric(metrics_4c, "f1_rf")
    mean_f1_4 = _mean_metric(metrics_4c, "f1_4")
    mean_auc_4 = _mean_metric(metrics_4c, "auc_4_ovr_macro")
    mean_auc_ha = _mean_metric(metrics_4c, "auc_ha")
    mean_auc_rf = _mean_metric(metrics_4c, "auc_rf")

    all_report.write("\n\n" + "#" * 90 + "\n")
    all_report.write("### GLOBAL SUMMARY (CAFE Mean over {} folds)\n".format(k))
    all_report.write("#" * 90 + "\n")
    all_report.write(
        f"CAFE 4C: "
        f"mean H/A Acc={mean_acc_ha:.4f}, F1={mean_f1_ha:.4f} | "
        f"mean R/F Acc={mean_acc_rf:.4f}, F1={mean_f1_rf:.4f} | "
        f"mean 4-way Acc={mean_acc_4:.4f}, F1={mean_f1_4:.4f} | "
        f"mean AUC(4c)={mean_auc_4:.4f} AUC(H/A)={mean_auc_ha:.4f} AUC(R/F)={mean_auc_rf:.4f}\n"
    )
    all_report.write("\n[TABLE_SUMMARY]\n")
    all_report.write(
        "Variant\tHA_Acc\tHA_F1\tRF_Acc\tRF_F1\t4C_Acc\t4C_F1\tAUC_4c\tAUC_HA\tAUC_RF\n"
    )
    all_report.write(
        f"CAFE 4C\t"
        f"{mean_acc_ha:.4f}\t{mean_f1_ha:.4f}\t"
        f"{mean_acc_rf:.4f}\t{mean_f1_rf:.4f}\t"
        f"{mean_acc_4:.4f}\t{mean_f1_4:.4f}\t"
        f"{mean_auc_4:.4f}\t{mean_auc_ha:.4f}\t{mean_auc_rf:.4f}\n"
    )
    output_path = os.path.join(RESULT_DIR, f"kfold_cafe_4c_{timestamp}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_report.getvalue())
    print(f"\n리포트 저장: {output_path}\n")

def get_single_split_loaders(full_df):
    idx = np.arange(len(full_df))
    y = full_df["label"].values
    tva, te, _, _ = train_test_split(
        idx, y, test_size=TEST_RATIO, stratify=y, random_state=SEED,
    )
    tr, va, _, _ = train_test_split(
        tva, y[tva],
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        stratify=y[tva], random_state=SEED,
    )
    train_data = full_df.iloc[tr].reset_index(drop=True)
    val_data = full_df.iloc[va].reset_index(drop=True)
    test_data = full_df.iloc[te].reset_index(drop=True)
    print(f"Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    train_ds = CAFEDataset(train_data, VOCAB, image_transform, max_len=SEQ_LEN, is_train=True)
    val_ds = CAFEDataset(val_data, VOCAB, image_transform, max_len=SEQ_LEN, is_train=False)
    test_ds = CAFEDataset(test_data, VOCAB, image_transform, max_len=SEQ_LEN, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=False,
        sampler=make_weighted_sampler(train_data["label"].tolist()),
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_cafe, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_cafe,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_cafe,
    )
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print(
        f"CAFE Baseline | "
        f"KFold={USE_KFOLD}, BATCH={BATCH}, LR={LR}, EPOCHS={EPOCHS}, "
        f"EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE}, "
        f"WeightedRandomSampler(alpha={SAMPLER_ALPHA}), SEED={SEED}, "
        f"data=HARFM.csv (final_headline, image_path, 4_way_label)"
    )
    if USE_KFOLD:
        kfold_cafe_4c_with_reports(
            data,
            k=KFOLD_SPLITS,
            test_size=KFOLD_TEST_SIZE,
            shuffle=KFOLD_SHUFFLE,
            random_state=KFOLD_RANDOM_STATE,
        )
    else:
        train_loader, val_loader, test_loader = get_single_split_loaders(data)
        text_embed_module = TextEmbedding(len(VOCAB), TEXT_DIM).to(DEVICE)
        img_encoder = ImageEncoder().to(DEVICE)
        sim_module = SimilarityModule().to(DEVICE)
        det_module = DetectionModule(num_classes=4).to(DEVICE)
        sim_module, det_module, text_embed_module, img_encoder = train_cafe_4c(
            sim_module, det_module, text_embed_module, img_encoder,
            train_loader, val_loader,
        )
        _ = evaluate_cafe(sim_module, det_module, text_embed_module, img_encoder, test_loader)
