"""
MiRAGe Baseline for HARFM (4-class: HR, HF, AR, AF)
- 데이터: HARFM.csv (text=final_headline, image=image_path, label=4_way_label)

  MiRAGe-Img : CLIP 이미지 임베딩(512d) + Simplified-CBM(CLIP zero-shot 개념 점수, N개)
               → Linear 브랜치 + CBM 브랜치 앙상블 → 4-class
  MiRAGe-Txt : CLIP 텍스트 임베딩(512d) + Simplified-TBM(언어통계 피처, M개)
               → Linear 브랜치 + TBM 브랜치 앙상블 → 4-class
  MiRAGe     : MiRAGe-Img & MiRAGe-Txt 각각 독립 학습 후
               테스트 시 softmax 평균(레이트 퓨전, 추가 학습 없음)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. 임포트 & CLIP 가용성 확인
# ─────────────────────────────────────────────────────────────────────────────
import os, io, sys, re, copy, gc, random
from contextlib import redirect_stdout
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score,
    classification_report,
    accuracy_score,
    roc_auc_score,
)

try:
    import open_clip
except ImportError:
    print(
        "\n[오류] open_clip_torch 패키지가 없습니다.\n"
        "  pip install open_clip_torch\n"
        "명령어로 설치 후 재실행해주세요."
    )
    sys.exit(1)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ─────────────────────────────────────────────────────────────────────────────
# 1. 재현성 & 기본 설정
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR   = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_FND_ROOT    = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
CSV_PATH     = os.path.join(_FND_ROOT, "HARFM.csv")

# 학습 하이퍼파라미터
BATCH                = 16
EPOCHS               = 10
LR                   = 1e-3
NUM_WORKERS          = 4
EARLY_STOP_PATIENCE  = 4
EARLY_STOP_MIN_DELTA = 1e-4
SAMPLER_ALPHA        = 0.5
USE_KFOLD            = True
KFOLD_SPLITS         = 5
KFOLD_TEST_SIZE      = 0.2
KFOLD_SHUFFLE        = True
KFOLD_RANDOM_STATE   = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2

LABEL_MAP     = {"HR": 0, "HF": 1, "AR": 2, "AF": 3}
REQUIRED_COLS = {"final_headline", "image_path", "4_way_label"}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Simplified-CBM 개념 프롬프트 (이미지 측)
# ─────────────────────────────────────────────────────────────────────────────
VISUAL_CONCEPTS = [
    "a real news photograph",
    "an AI-generated image",
    "a photorealistic synthetic image",
    "a person or human face",
    "a crowd of people",
    "a political event or rally",
    "a natural landscape or scenery",
    "an urban environment or city",
    "a building or architecture",
    "a vehicle or transportation",
    "an animal or wildlife",
    "a sports event",
    "a military or weapon",
    "a medical scene",
    "text or signage visible in image",
    "professional journalism photo",
    "social media screenshot",
    "digitally manipulated or edited photo",
    "high resolution detailed image",
    "blurry or low quality image",
    "unusual lighting or unrealistic colors",
    "perfectly symmetrical or unnatural composition",
    "news headline or article",
    "indoor scene",
    "outdoor scene",
]
NUM_CONCEPTS = len(VISUAL_CONCEPTS)  # 25

# ─────────────────────────────────────────────────────────────────────────────
# 3. Simplified-TBM 언어 피처 (텍스트 측)
# ─────────────────────────────────────────────────────────────────────────────
def compute_tbm_features(text: str) -> np.ndarray:
    """
    15개 언어통계 피처 (Simplified TBM).
    원본 TBM 의 "언어 개념 점수" 대체 역할.
    """
    text = str(text)
    words = text.split()
    chars = list(text)

    word_count       = len(words)
    char_count       = len(text)
    avg_word_len     = np.mean([len(w) for w in words]) if words else 0.0
    num_digits       = sum(c.isdigit() for c in chars)
    num_upper        = sum(c.isupper() for c in chars)
    num_punct        = sum(not c.isalnum() and not c.isspace() for c in chars)
    digit_ratio      = num_digits / max(char_count, 1)
    upper_ratio      = num_upper / max(char_count, 1)
    punct_ratio      = num_punct / max(char_count, 1)
    unique_words     = len(set(w.lower() for w in words))
    ttr              = unique_words / max(word_count, 1)       # type-token ratio
    has_quote        = float('"' in text or "'" in text)
    has_number       = float(any(c.isdigit() for c in text))
    num_sentences    = max(text.count('.') + text.count('!') + text.count('?'), 1)
    avg_sent_len     = word_count / num_sentences

    feats = np.array([
        word_count, char_count, avg_word_len,
        digit_ratio, upper_ratio, punct_ratio,
        ttr, has_quote, has_number,
        num_sentences, avg_sent_len,
        num_digits, num_upper, num_punct, unique_words,
    ], dtype=np.float32)

    return feats

NUM_TBM_FEATS = 15  # compute_tbm_features 반환 길이

# ─────────────────────────────────────────────────────────────────────────────
# 4. 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────
def _as_str(x):
    if x is None: return ""
    if isinstance(x, float) and np.isnan(x): return ""
    return str(x).strip()

def _resolve_image_path(p, base_dir):
    p = _as_str(p)
    if not p: return ""
    if os.path.isabs(p): return p
    return os.path.normpath(os.path.join(base_dir, p))

def _strip_quotes(val):
    if not isinstance(val, str): return val
    val = val.strip()
    while len(val) >= 2 and (
        (val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")
    ):
        val = val[1:-1].strip()
    return val

print("MiRAGe 모델 학습을 위해 HARFM.csv 로드 시작 (text=final_headline, image=image_path, label=4_way_label)")
raw = pd.read_csv(CSV_PATH, low_memory=False)

missing = REQUIRED_COLS - set(raw.columns)
if missing:
    raise ValueError(f"HARFM CSV에 다음 컬럼이 없습니다: {missing}")

raw = raw[list(REQUIRED_COLS)].copy()
raw["text"]       = raw["final_headline"].fillna("").astype(str).apply(_strip_quotes)
raw["image_path"] = raw["image_path"].apply(lambda p: _resolve_image_path(p, _FND_ROOT))
raw["label"]      = raw["4_way_label"].map(LABEL_MAP)
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

# ─────────────────────────────────────────────────────────────────────────────
# 5. CLIP 모델 로드 & 전체 데이터 피처 사전 인코딩
# ─────────────────────────────────────────────────────────────────────────────
print("\n[CLIP] ViT-B-32 모델 로드 중 ...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model = clip_model.to(DEVICE).eval()

# 개념 텍스트 토큰화 (CBM용)
concept_tokens = clip_tokenizer(VISUAL_CONCEPTS).to(DEVICE)
with torch.no_grad():
    concept_text_feats = clip_model.encode_text(concept_tokens)
    concept_text_feats = F.normalize(concept_text_feats, dim=-1)  # (25, 512)

@torch.no_grad()
def encode_all(df: pd.DataFrame, batch_size: int = 128):
    """
    전체 데이터프레임을 CLIP으로 인코딩.
    반환:
        img_feats    : (N, 512)  CLIP 이미지 임베딩
        txt_feats    : (N, 512)  CLIP 텍스트 임베딩
        concept_scores: (N, NUM_CONCEPTS)  CBM 개념 점수 (코사인 유사도)
        tbm_feats    : (N, NUM_TBM_FEATS) 언어통계 피처
    """
    N = len(df)
    img_feats_all      = np.zeros((N, 512),          dtype=np.float32)
    txt_feats_all      = np.zeros((N, 512),          dtype=np.float32)
    concept_scores_all = np.zeros((N, NUM_CONCEPTS), dtype=np.float32)
    tbm_feats_all      = np.zeros((N, NUM_TBM_FEATS),dtype=np.float32)

    for start in tqdm(range(0, N, batch_size), desc="CLIP 인코딩"):
        end  = min(start + batch_size, N)
        rows = df.iloc[start:end]

        # ── 이미지 인코딩 ──
        imgs = []
        for _, row in rows.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                imgs.append(clip_preprocess(img))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        img_tensor = torch.stack(imgs).to(DEVICE)
        img_feat   = clip_model.encode_image(img_tensor)
        img_feat   = F.normalize(img_feat, dim=-1)
        img_feats_all[start:end] = img_feat.cpu().numpy()

        # CBM 개념 점수: 이미지 피처 @ 개념 텍스트 피처^T → (B, 25)
        cscore = img_feat @ concept_text_feats.T  # (B, 25)
        concept_scores_all[start:end] = cscore.cpu().numpy()

        # ── 텍스트 인코딩 ──
        texts = rows["text"].tolist()
        tok   = clip_tokenizer(texts).to(DEVICE)
        txt_feat = clip_model.encode_text(tok)
        txt_feat = F.normalize(txt_feat, dim=-1)
        txt_feats_all[start:end] = txt_feat.cpu().numpy()

        # TBM 언어통계 피처
        for i, (_, row) in enumerate(rows.iterrows()):
            tbm_feats_all[start + i] = compute_tbm_features(row["text"])

    return img_feats_all, txt_feats_all, concept_scores_all, tbm_feats_all

print("\n[피처 사전 인코딩] 전체 데이터 CLIP 인코딩 중 (최초 1회만 수행) ...")
ALL_IMG_FEATS, ALL_TXT_FEATS, ALL_CONCEPT_SCORES, ALL_TBM_FEATS = encode_all(data)

# TBM 피처 정규화 (학습 안정성)
tbm_mean = ALL_TBM_FEATS.mean(axis=0)
tbm_std  = ALL_TBM_FEATS.std(axis=0) + 1e-8
ALL_TBM_FEATS = (ALL_TBM_FEATS - tbm_mean) / tbm_std
print(f"[피처 인코딩 완료] img={ALL_IMG_FEATS.shape}, txt={ALL_TXT_FEATS.shape}, "
      f"concept={ALL_CONCEPT_SCORES.shape}, tbm={ALL_TBM_FEATS.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Dataset (사전 인코딩된 피처 기반)
# ─────────────────────────────────────────────────────────────────────────────
class MiRAGeDataset(Dataset):
    """
    사전 인코딩된 CLIP 피처를 반환하는 Dataset.
    """
    def __init__(self, indices, img_feats, txt_feats, concept_scores, tbm_feats, labels):
        self.indices        = indices
        self.img_feats      = img_feats
        self.txt_feats      = txt_feats
        self.concept_scores = concept_scores
        self.tbm_feats      = tbm_feats
        self.labels         = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return (
            torch.tensor(self.img_feats[idx],      dtype=torch.float32),
            torch.tensor(self.txt_feats[idx],      dtype=torch.float32),
            torch.tensor(self.concept_scores[idx], dtype=torch.float32),
            torch.tensor(self.tbm_feats[idx],      dtype=torch.float32),
            torch.tensor(self.labels[idx],         dtype=torch.long),
        )

def collate_mirage(batch):
    img_f, txt_f, cscore, tbm_f, lbl = zip(*batch)
    return (
        torch.stack(img_f),
        torch.stack(txt_f),
        torch.stack(cscore),
        torch.stack(tbm_f),
        torch.stack(lbl),
    )

def make_weighted_sampler(labels, alpha: float = SAMPLER_ALPHA):
    cnt = pd.Series(labels).value_counts().to_dict()
    w   = [(1.0 / cnt[l]) ** alpha for l in labels]
    return WeightedRandomSampler(w, len(w), replacement=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. 모델 정의
# ─────────────────────────────────────────────────────────────────────────────
class MiRAGeImgDetector(nn.Module):
    """
    MiRAGe-Img (4-class 버전)
    ─ Linear 브랜치: CLIP 이미지 임베딩(512d) → FC → 4-class logits
    ─ CBM 브랜치  : 개념 점수(NUM_CONCEPTS d) → FC → 4-class logits
    ─ 앙상블 헤드 : cat(linear_logits, cbm_logits) → FC → 4-class logits 최종

    원본: linear_model 예측 스칼라 + CBM 개념벡터(300d) → MLP → real/fake
          → HARFM 적응: 4-class logits로 확장, 앙상블 헤드로 퓨전
    """
    def __init__(self, img_dim=512, concept_dim=NUM_CONCEPTS, num_classes=4, dropout=0.3):
        super().__init__()
        # 브랜치 1: Linear (직접 CLIP 임베딩 → 4-class)
        self.linear_branch = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        # 브랜치 2: Simplified CBM (개념 점수 → 4-class)
        self.cbm_branch = nn.Sequential(
            nn.Linear(concept_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        # 앙상블 퓨전 헤드 (원본: [linear_pred, cbm_vec] → MLP)
        self.ensemble_head = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, img_feat, concept_scores):
        logits_lin = self.linear_branch(img_feat)       # (B, 4)
        logits_cbm = self.cbm_branch(concept_scores)    # (B, 4)
        fused      = torch.cat([logits_lin, logits_cbm], dim=1)  # (B, 8)
        logits_out = self.ensemble_head(fused)          # (B, 4)
        return logits_out, logits_lin, logits_cbm


class MiRAGeTxtDetector(nn.Module):
    """
    MiRAGe-Txt (4-class 버전)
    ─ Linear 브랜치: CLIP 텍스트 임베딩(512d) → FC → 4-class logits
    ─ TBM 브랜치  : 언어통계 피처(NUM_TBM_FEATS d) → FC → 4-class logits
    ─ 앙상블 헤드 : cat(linear_logits, tbm_logits) → FC → 4-class logits 최종

    원본: linear_model 예측 스칼라 + TBM 개념벡터(18d) → MLP → real/fake
          → HARFM 적응: 4-class logits로 확장
    """
    def __init__(self, txt_dim=512, tbm_dim=NUM_TBM_FEATS, num_classes=4, dropout=0.3):
        super().__init__()
        # 브랜치 1: Linear
        self.linear_branch = nn.Sequential(
            nn.Linear(txt_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        # 브랜치 2: Simplified TBM
        self.tbm_branch = nn.Sequential(
            nn.Linear(tbm_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
        # 앙상블 퓨전 헤드
        self.ensemble_head = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, txt_feat, tbm_feats):
        logits_lin = self.linear_branch(txt_feat)       # (B, 4)
        logits_tbm = self.tbm_branch(tbm_feats)         # (B, 4)
        fused      = torch.cat([logits_lin, logits_tbm], dim=1)  # (B, 8)
        logits_out = self.ensemble_head(fused)          # (B, 4)
        return logits_out, logits_lin, logits_tbm

# ─────────────────────────────────────────────────────────────────────────────
# 8. 유틸리티
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, mode="max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = None
        self.counter   = 0

    def step(self, val: float) -> bool:
        if self.best is None:
            self.best = val; return False
        improved = (val - self.best) > self.min_delta if self.mode == "max"\
                   else (self.best - val) > self.min_delta
        if improved:
            self.best = val; self.counter = 0; return False
        self.counter += 1
        return self.counter >= self.patience


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
        return len(data)
    def flush(self):
        for s in self.streams:
            if hasattr(s, "flush"): s.flush()


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


# ─────────────────────────────────────────────────────────────────────────────
# 9. 학습 함수
# ─────────────────────────────────────────────────────────────────────────────
def _train_one_epoch(model, loader, optimizer, mode: str, epoch_1based: int):
    """
    mode: "img" or "txt"
    보조 손실(linear 브랜치 + bottleneck 브랜치)과 앙상블 헤드 손실을 합산.
    """
    model.train()
    loss_fn  = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0

    for batch in tqdm(
        loader,
        desc=f"[MiRAGe-{mode.upper()}] Epoch {epoch_1based}",
    ):
        img_f, txt_f, cscore, tbm_f, labels = [b.to(DEVICE) for b in batch]

        if mode == "img":
            logits, logits_lin, logits_aux = model(img_f, cscore)
        else:
            logits, logits_lin, logits_aux = model(txt_f, tbm_f)

        # 앙상블 헤드 + 각 브랜치 보조 손실 (멀티태스크 학습)
        loss = (loss_fn(logits, labels)
                + 0.3 * loss_fn(logits_lin, labels)
                + 0.3 * loss_fn(logits_aux, labels))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n          += len(labels)

    return total_loss / max(n, 1)


@torch.no_grad()
def _validate(model, loader, mode: str) -> float:
    model.eval()
    all_true, all_pred = [], []
    for batch in loader:
        img_f, txt_f, cscore, tbm_f, labels = [b.to(DEVICE) for b in batch]
        if mode == "img":
            logits, _, _ = model(img_f, cscore)
        else:
            logits, _, _ = model(txt_f, tbm_f)
        all_pred.extend(logits.argmax(1).cpu().tolist())
        all_true.extend(labels.cpu().tolist())
    return f1_score(all_true, all_pred, average="macro")


def train_mirage_unimodal(model, train_loader, val_loader, mode: str, epochs=EPOCHS):
    """MiRAGe-Img 또는 MiRAGe-Txt 단일 모달 학습."""
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    early     = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode="max")
    best_val  = -1.0
    best_state = None

    for epoch in range(epochs):
        loss = _train_one_epoch(
            model, train_loader, optimizer, mode, epoch + 1
        )
        val_f1 = _validate(model, val_loader, mode)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  [MiRAGe-{mode.upper()}] Epoch {epoch+1:02d} | loss={loss:.4f} | val_macro_f1={val_f1:.4f} | lr={cur_lr:.2e}")

        if val_f1 > best_val:
            best_val   = val_f1
            best_state = copy.deepcopy(model.state_dict())

        if early.step(val_f1):
            print(f"  [MiRAGe-{mode.upper()}] Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 10. 평가 함수
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_mirage(img_model, txt_model, loader, fusion="avg"):
    """
    MiRAGe 멀티모달 평가.
    fusion="img"  : 이미지 단독 (MiRAGe-Img)
    fusion="txt"  : 텍스트 단독 (MiRAGe-Txt)
    """
    img_model.eval()
    txt_model.eval()
    all_true, all_pred = [], []
    prob_chunks = []

    for batch in tqdm(
        loader,
        desc=f"[MiRAGe-{fusion}] Evaluation (4C)",
    ):
        img_f, txt_f, cscore, tbm_f, labels = [b.to(DEVICE) for b in batch]

        logits_img, _, _ = img_model(img_f, cscore)
        logits_txt, _, _ = txt_model(txt_f, tbm_f)

        if fusion == "avg":
            prob = (F.softmax(logits_img, dim=1) + F.softmax(logits_txt, dim=1)) / 2.0
        elif fusion == "img":
            prob = F.softmax(logits_img, dim=1)
        else:
            prob = F.softmax(logits_txt, dim=1)

        prob_chunks.append(prob.detach().cpu().numpy())
        all_pred.extend(prob.argmax(1).cpu().tolist())
        all_true.extend(labels.cpu().tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    y_ha, p_ha = all_true // 2, all_pred // 2   # Human(0,1) vs AI(2,3)
    y_rf, p_rf = all_true % 2,  all_pred % 2    # Real(0,2) vs Fake(1,3)

    acc_4  = accuracy_score(all_true, all_pred)
    acc_ha = accuracy_score(y_ha, p_ha)
    acc_rf = accuracy_score(y_rf, p_rf)
    f1_4   = f1_score(all_true, all_pred, average="macro")
    f1_ha  = f1_score(y_ha, p_ha, average="macro")
    f1_rf  = f1_score(y_rf, p_rf, average="macro")

    target4 = ["Human Real", "Human Fake", "AI Real", "AI Fake"]
    print(f"\n=== MiRAGe [{fusion}] Human vs AI ===")
    print(classification_report(y_ha, p_ha, target_names=["Human", "AI"], digits=4))
    print(f"=== MiRAGe [{fusion}] Real vs Fake ===")
    print(classification_report(y_rf, p_rf, target_names=["Real", "Fake"], digits=4))
    print(f"=== MiRAGe [{fusion}] 4-class ===")
    print(classification_report(all_true, all_pred, target_names=target4, digits=4))
    proba = np.vstack(prob_chunks) if prob_chunks else np.zeros((0, 4), dtype=np.float64)
    aucm = auc_scores_from_proba(all_true, proba)
    print(
        f"\n[MiRAGe-{fusion}] ROC-AUC: 4-class (macro-OVR)={aucm['auc_4_ovr_macro']:.4f} | "
        f"H/A={aucm['auc_ha']:.4f} | R/F={aucm['auc_rf']:.4f}"
    )
    print(
        f"\n[MiRAGe-{fusion}] "
        f"H/A Acc={acc_ha:.4f} F1={f1_ha:.4f} | "
        f"R/F Acc={acc_rf:.4f} F1={f1_rf:.4f} | "
        f"4-way Acc={acc_4:.4f} F1={f1_4:.4f} | "
        f"AUC(4c)={aucm['auc_4_ovr_macro']:.4f} AUC(H/A)={aucm['auc_ha']:.4f} "
        f"AUC(R/F)={aucm['auc_rf']:.4f}"
    )

    return dict(
        acc_ha=acc_ha,
        acc_rf=acc_rf,
        acc_4=acc_4,
        f1_ha=f1_ha,
        f1_rf=f1_rf,
        f1_4=f1_4,
        auc_4_ovr_macro=aucm["auc_4_ovr_macro"],
        auc_ha=aucm["auc_ha"],
        auc_rf=aucm["auc_rf"],
    )

# ─────────────────────────────────────────────────────────────────────────────
# 11. KFold 전체 실험
# ─────────────────────────────────────────────────────────────────────────────
def kfold_mirage_4c(
    full_df,
    all_img_feats, all_txt_feats, all_concept_scores, all_tbm_feats,
    k=KFOLD_SPLITS,
    test_size=KFOLD_TEST_SIZE,
    shuffle=KFOLD_SHUFFLE,
    random_state=KFOLD_RANDOM_STATE,
):
    labels_arr = full_df["label"].values

    trainval_idx, test_idx = train_test_split(
        np.arange(len(full_df)),
        test_size=test_size,
        stratify=labels_arr,
        random_state=random_state,
    )
    print(f"[KFold-MiRAGe] Split: TrainVal={len(trainval_idx)}, Test={len(test_idx)}")

    test_ds     = MiRAGeDataset(test_idx,
                                all_img_feats, all_txt_feats,
                                all_concept_scores, all_tbm_feats,
                                labels_arr)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                             num_workers=NUM_WORKERS,
                             pin_memory=torch.cuda.is_available(),
                             collate_fn=collate_mirage)

    skf      = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    all_report = io.StringIO()
    all_report.write(f"###### MiRAGe 4C (k={k}) ######\n")

    # fusion 모드별 누적 메트릭
    metrics_per_fusion = defaultdict(list)
    y_trainval = labels_arr[trainval_idx]

    for fold, (tr_rel, va_rel) in enumerate(
        skf.split(np.zeros(len(trainval_idx)), y_trainval), start=1
    ):
        print(f"\n========== [Fold {fold}/{k} - MiRAGe 4C] ==========")
        tr_idx = trainval_idx[tr_rel]
        va_idx = trainval_idx[va_rel]

        tr_labels = labels_arr[tr_idx].tolist()
        va_labels = labels_arr[va_idx].tolist()

        train_ds = MiRAGeDataset(tr_idx,
                                 all_img_feats, all_txt_feats,
                                 all_concept_scores, all_tbm_feats,
                                 labels_arr)
        val_ds   = MiRAGeDataset(va_idx,
                                 all_img_feats, all_txt_feats,
                                 all_concept_scores, all_tbm_feats,
                                 labels_arr)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH, shuffle=False,
            sampler=make_weighted_sampler(tr_labels),
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_mirage, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH, shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_mirage,
        )

        img_model = MiRAGeImgDetector(num_classes=4).to(DEVICE)
        txt_model = MiRAGeTxtDetector(num_classes=4).to(DEVICE)

        buffer = io.StringIO()
        tee    = Tee(sys.stdout, buffer)

        with redirect_stdout(tee):
            print("\n" + "=" * 70)
            print(f"[Fold {fold}] MiRAGe-Img 학습")
            print("=" * 70)
            img_model = train_mirage_unimodal(img_model, train_loader, val_loader, mode="img")

            print("\n" + "=" * 70)
            print(f"[Fold {fold}] MiRAGe-Txt 학습")
            print("=" * 70)
            txt_model = train_mirage_unimodal(txt_model, train_loader, val_loader, mode="txt")

            print("\n" + "=" * 70)
            print(f"[Fold {fold}] MiRAGe 평가 (레이트 퓨전, 추가 학습 없음)")
            print("=" * 70)

            fold_metrics = {}
            for fusion_mode in ("img", "txt", "avg"):
                print(f"\n--- fusion={fusion_mode} ---")
                res = evaluate_mirage(img_model, txt_model, test_loader, fusion=fusion_mode)
                fold_metrics[fusion_mode] = res
                metrics_per_fusion[fusion_mode].append(res)

            print("\n" + "=" * 70)
            print(f"[Fold {fold}] SUMMARY (MiRAGe 4C, Acc/F1 on Test)")
            print("=" * 70)
            for fm, res in fold_metrics.items():
                print(
                    f"  MiRAGe [{fm:3s}]: "
                    f"H/A Acc={res['acc_ha']:.4f} F1={res['f1_ha']:.4f} | "
                    f"R/F Acc={res['acc_rf']:.4f} F1={res['f1_rf']:.4f} | "
                    f"4-way Acc={res['acc_4']:.4f} F1={res['f1_4']:.4f} | "
                    f"AUC(4c)={res['auc_4_ovr_macro']:.4f} AUC(H/A)={res['auc_ha']:.4f} "
                    f"AUC(R/F)={res['auc_rf']:.4f}"
                )
            print("=" * 70)

        all_report.write(f"\n\n{'#'*90}\n### FOLD {fold}/{k} REPORT (MiRAGe 4C)\n{'#'*90}\n")
        all_report.write(buffer.getvalue())
        all_report.flush()
        gc.collect()

    # ── 전체 평균 요약 ──
    def _mean(dict_list, key):
        vals = [d[key] for d in dict_list if d.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    all_report.write("\n\n" + "#" * 90 + "\n")
    all_report.write(f"### GLOBAL SUMMARY (MiRAGe Mean over {k} folds)\n")
    all_report.write("#" * 90 + "\n")
    all_report.write("\n[TABLE_SUMMARY]\n")
    all_report.write(
        "Variant\tHA_Acc\tHA_F1\tRF_Acc\tRF_F1\t4C_Acc\t4C_F1\tAUC_4c\tAUC_HA\tAUC_RF\n"
    )

    for fm, mlist in metrics_per_fusion.items():
        m_ha_acc = _mean(mlist, "acc_ha")
        m_ha_f1  = _mean(mlist, "f1_ha")
        m_rf_acc = _mean(mlist, "acc_rf")
        m_rf_f1  = _mean(mlist, "f1_rf")
        m_4_acc  = _mean(mlist, "acc_4")
        m_4_f1   = _mean(mlist, "f1_4")
        m_auc_4  = _mean(mlist, "auc_4_ovr_macro")
        m_auc_ha = _mean(mlist, "auc_ha")
        m_auc_rf = _mean(mlist, "auc_rf")
        all_report.write(
            f"MiRAGe-{fm}\t"
            f"{m_ha_acc:.4f}\t{m_ha_f1:.4f}\t"
            f"{m_rf_acc:.4f}\t{m_rf_f1:.4f}\t"
            f"{m_4_acc:.4f}\t{m_4_f1:.4f}\t"
            f"{m_auc_4:.4f}\t{m_auc_ha:.4f}\t{m_auc_rf:.4f}\n"
        )
        print(
            f"[MEAN Fold={k}] MiRAGe-{fm}: "
            f"H/A Acc={m_ha_acc:.4f} F1={m_ha_f1:.4f} | "
            f"R/F Acc={m_rf_acc:.4f} F1={m_rf_f1:.4f} | "
            f"4-way Acc={m_4_acc:.4f} F1={m_4_f1:.4f} | "
            f"AUC(4c)={m_auc_4:.4f} AUC(H/A)={m_auc_ha:.4f} AUC(R/F)={m_auc_rf:.4f}"
        )

    output_path = os.path.join(RESULT_DIR, f"kfold_mirage_4c_{timestamp}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_report.getvalue())
    print(f"\n리포트 저장: {output_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 12. 단일 스플릿 학습 (USE_KFOLD=False 일 때)
# ─────────────────────────────────────────────────────────────────────────────
def single_split_mirage(full_df, all_img_feats, all_txt_feats,
                        all_concept_scores, all_tbm_feats):
    labels_arr = full_df["label"].values
    idx        = np.arange(len(full_df))

    tva_idx, te_idx = train_test_split(
        idx, test_size=TEST_RATIO, stratify=labels_arr, random_state=SEED
    )
    tr_idx, va_idx = train_test_split(
        tva_idx,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        stratify=labels_arr[tva_idx], random_state=SEED,
    )
    print(f"Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")

    mk_ds = lambda idx_: MiRAGeDataset(
        idx_, all_img_feats, all_txt_feats, all_concept_scores, all_tbm_feats, labels_arr
    )
    train_loader = DataLoader(
        mk_ds(tr_idx), batch_size=BATCH, shuffle=False,
        sampler=make_weighted_sampler(labels_arr[tr_idx].tolist()),
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_mirage, drop_last=True,
    )
    val_loader  = DataLoader(mk_ds(va_idx), batch_size=BATCH, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_mirage)
    test_loader = DataLoader(mk_ds(te_idx), batch_size=BATCH, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_mirage)

    img_model = MiRAGeImgDetector(num_classes=4).to(DEVICE)
    txt_model = MiRAGeTxtDetector(num_classes=4).to(DEVICE)

    print("\n[MiRAGe-Img] 학습 시작")
    img_model = train_mirage_unimodal(img_model, train_loader, val_loader, mode="img")

    print("\n[MiRAGe-Txt] 학습 시작")
    txt_model = train_mirage_unimodal(txt_model, train_loader, val_loader, mode="txt")

    print("\n[MiRAGe] 멀티모달 평가 (레이트 퓨전)")
    for fm in ("img", "txt", "avg"):
        evaluate_mirage(img_model, txt_model, test_loader, fusion=fm)


# ─────────────────────────────────────────────────────────────────────────────
# 13. 메인
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(
        f"\nMiRAGe Baseline (HARFM 4-class) | "
        f"KFold={USE_KFOLD}, BATCH={BATCH}, LR={LR}, EPOCHS={EPOCHS}, "
        f"EARLY_STOP={EARLY_STOP_PATIENCE}, "
        f"CBM_concepts={NUM_CONCEPTS}, TBM_feats={NUM_TBM_FEATS}, "
        f"WeightedSampler(alpha={SAMPLER_ALPHA}), SEED={SEED}, "
        f"data=HARFM.csv (final_headline, image_path, 4_way_label)\n"
    )

    if USE_KFOLD:
        kfold_mirage_4c(
            data,
            ALL_IMG_FEATS, ALL_TXT_FEATS, ALL_CONCEPT_SCORES, ALL_TBM_FEATS,
            k=KFOLD_SPLITS,
            test_size=KFOLD_TEST_SIZE,
            shuffle=KFOLD_SHUFFLE,
            random_state=KFOLD_RANDOM_STATE,
        )
    else:
        single_split_mirage(
            data,
            ALL_IMG_FEATS, ALL_TXT_FEATS, ALL_CONCEPT_SCORES, ALL_TBM_FEATS,
        )
