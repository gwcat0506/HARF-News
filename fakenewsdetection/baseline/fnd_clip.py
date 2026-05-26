"""
FND-CLIP Baseline for HARFM (4-class: HR, HF, AR, AF)
- FND-CLIP-Fake-News-Detection 참고: BERT(텍스트) + ResNet101(이미지) + CLIP(텍스트·이미지) → MultiModal → 4-way
- 데이터: HARFM.csv (text=final_headline, image=image_path, label=4_way_label)
"""
import os
import io
import sys
import copy
import gc
from contextlib import redirect_stdout
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from transformers import BertModel, BertTokenizer, CLIPModel, CLIPProcessor

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
SEED = 42


def set_seed(seed=SEED):
    import random
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
BERT_NAME="bert-base-uncased"
CLIP_NAME="openai/clip-vit-base-patch32"
MAX_LEN_BERT=128
IMG_SIZE=224
MAX_LEN_CLIP=77
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
LABEL_MAP={"HR": 0, "HF": 1, "AR": 2, "AF": 3}
REQUIRED_COLS={"final_headline", "image_path", "4_way_label"}

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
    val=val.strip()
    while len(val) >= 2 and (
        (val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")
    ):
        val=val[1:-1].strip()
    return val
print("FND CLIP베이스라인에 넣은 HARFM데이터셋 로딩시작~(text=final_headline, image=image_path, label=4_way_label)...")
raw=pd.read_csv(CSV_PATH, low_memory=False)
missing=REQUIRED_COLS - set(raw.columns)
if missing:
    raise ValueError(f"HARFM CSV에 다음 컬럼이 없습니다: {missing}")
raw = raw[list(REQUIRED_COLS)].copy()
raw["text"]=raw["final_headline"].fillna("").astype(str).apply(_strip_quotes)
raw["image_path"]=raw["image_path"].apply(lambda p: _resolve_image_path(p, _FND_ROOT))
raw["label"]=raw["4_way_label"].map(LABEL_MAP)
raw=raw[raw["label"].notna()].astype({"label": int})
data=raw[["text", "image_path", "label"]].copy()
print(f"Dataset loaded: {len(data)} rows (4-way)")
print("Label counts:", data["label"].value_counts().sort_index().to_dict())

n_before=len(data)
data=data[
    data["image_path"].apply(
        lambda p: isinstance(p, str) and p != "" and os.path.exists(p)
    )
].reset_index(drop=True)
print(f"[전처리] 이미지 결측 제거: {n_before} → {len(data)}")
print("[전처리] 제거 후 label 분포:", data["label"].value_counts().sort_index().to_dict())

bert_tokenizer=BertTokenizer.from_pretrained(BERT_NAME)
clip_processor=CLIPProcessor.from_pretrained(CLIP_NAME)

resnet_transform=transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



class FNDCLIPDataset(Dataset):
    def __init__(self, df, bert_tokenizer, clip_processor, max_len_bert=MAX_LEN_BERT, is_train=True):
        self.df=df.reset_index(drop=True)
        self.bert_tokenizer=bert_tokenizer
        self.clip_processor=clip_processor
        self.max_len_bert=max_len_bert
        self.is_train=is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        text=row.get("text", "") or ""
        path=row.get("image_path", "")
        img=Image.open(path).convert("RGB")
        label4=int(row["label"])

        bert_enc=self.bert_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len_bert,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        image_resnet=resnet_transform(img)
        clip_enc=self.clip_processor(
            text=[text],
            images=[img],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN_CLIP,
            return_tensors="pt",
        )
        return {
            "bert_input_ids": bert_enc["input_ids"].squeeze(0),
            "bert_attention_mask": bert_enc["attention_mask"].squeeze(0),
            "bert_token_type_ids": bert_enc["token_type_ids"].squeeze(0),
            "image_resnet": image_resnet,
            "clip_input_ids": clip_enc["input_ids"].squeeze(0),
            "clip_attention_mask": clip_enc["attention_mask"].squeeze(0),
            "clip_pixel_values": clip_enc["pixel_values"].squeeze(0),
            "label4": torch.tensor(label4, dtype=torch.long),
        }


def collate_fnd_clip(batch):
    return {
        "bert_input_ids": torch.stack([b["bert_input_ids"] for b in batch]),
        "bert_attention_mask": torch.stack([b["bert_attention_mask"] for b in batch]),
        "bert_token_type_ids": torch.stack([b["bert_token_type_ids"] for b in batch]),
        "image_resnet": torch.stack([b["image_resnet"] for b in batch]),
        "clip_input_ids": torch.stack([b["clip_input_ids"] for b in batch]),
        "clip_attention_mask": torch.stack([b["clip_attention_mask"] for b in batch]),
        "clip_pixel_values": torch.stack([b["clip_pixel_values"] for b in batch]),
        "label4": torch.stack([b["label4"] for b in batch]),
    }


def make_weighted_sampler(labels, alpha: float = SAMPLER_ALPHA) -> WeightedRandomSampler:
    """역빈도 ** alpha 가중 WeightedRandomSampler."""
    cnt = pd.Series(labels).value_counts().to_dict()
    w = [(1.0 / cnt[l]) ** alpha for l in labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


#FND-CLIP네트워크(4-class)
class UnimodalDetection(nn.Module):
    def __init__(self, text_in=1280, image_in=1512, shared_dim=256, prime_dim=64):
        super().__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(text_in, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU(),
        )
        self.image_uni = nn.Sequential(
            nn.Linear(image_in, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU(),
        )

    def forward(self, text_encoding, image_encoding):
        return self.text_uni(text_encoding), self.image_uni(image_encoding)


class CrossModule(nn.Module):
    def __init__(self, corre_in=1024, corre_out_dim=64):
        super().__init__()
        self.c_specific = nn.Sequential(
            nn.Linear(corre_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU(),
        )

    def forward(self, text, image):
        correlation = torch.cat((text, image), 1)
        return self.c_specific(correlation)


class FNDCLIPMultiModal4C(nn.Module):
    """BERT(768) + CLIP text(512)=1280, ResNet101(1000) + CLIP image(512)=1512, Cross 1024 → 4-class"""
    def __init__(self, num_classes=4, feature_dim=64 * 3, h_dim=64):
        super().__init__()
        self.weights=nn.Parameter(torch.randn(13, 1) * 0.01)
        self.senet=nn.Sequential(
            nn.Linear(3, 3),
            nn.GELU(),
            nn.Linear(3, 3),
        )
        self.sigmoid=nn.Sigmoid()
        self.w=nn.Parameter(torch.tensor(1.0))
        self.b=nn.Parameter(torch.tensor(0.0))
        self.avepooling=nn.AvgPool1d(64, stride=1)
        self.maxpooling=nn.MaxPool1d(64, stride=1)
        resnet=models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.resnet101=resnet
        self.uni_repre=UnimodalDetection(text_in=1280, image_in=1512)#unimodal detection(텍스트 1280, 1512차원->각 64차원)
        self.cross_module=CrossModule(corre_in=512 + 512)#cross module(512+512차원->256차원)
        self.classifier_corre=nn.Sequential(#3x62=192차원 4클래스로 분류함 
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_classes),
        )

    def forward(self, bert_hidden_states, image_resnet, text_clip, image_clip):
        B=image_resnet.shape[0]
        image_raw=self.resnet101(image_resnet)
        ht_cls=torch.stack(bert_hidden_states, dim=0)[:, :, 0, :]
        ht_cls=ht_cls.view(13, B, 1, 768)
        atten=torch.sum(ht_cls * self.weights.view(13, 1, 1, 1), dim=[1, 3])
        atten=F.softmax(atten.view(-1), dim=0)
        text_raw=torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2]).squeeze(1)#bert 13개 레이러의 cls hidden state를 가중치 합산(768차원)

        #unimodal 입력 unimodal detection 
        text_enc=torch.cat([text_raw, text_clip], 1)#1280차원
        image_enc=torch.cat([image_raw, image_clip], 1)#1512차원
        text_prime, image_prime=self.uni_repre(text_enc, image_enc)#(B,64),(B,64)
        correlation=self.cross_module(text_clip, image_clip)#(B,1024)->(B,256)->(B,64)
        sim=(text_clip * image_clip).sum(1) / (
            text_clip.norm(dim=1) * image_clip.norm(dim=1) + 1e-8
        )
        sim=sim * self.w + self.b
        mweight=sim.unsqueeze(1)
        correlation=correlation * mweight# clip유사도 가중치 적용

        final_feature=torch.stack([text_prime, image_prime, correlation], 1)#(B,3,64)
        s1=self.avepooling(final_feature).view(B, -1)#stride=1넣어서 각 채널에서 64->1로 축소되고 (B,3,1)이 됨
        s2=self.maxpooling(final_feature).view(B, -1)
        s1=self.senet(s1)
        s2=self.senet(s2)
        s=self.sigmoid(s1 + s2).view(B, 3, 1)#(B,3,1)
        final_feature=s * final_feature#(B,3,64)X(B,3,1)->(B,3,64)
        pooled=final_feature.view(B, -1)#(B,3,64)->(B,192) classifier 입력 차원에 맞춤
        return self.classifier_corre(pooled)


#earlystopping설정
class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, mode="max"):
        self.patience=patience
        self.min_delta=min_delta
        self.mode=mode
        self.best=None
        self.counter=0

    def step(self, metric_value: float) -> bool:
        if self.best is None:
            self.best=metric_value
            return False
        improved=(metric_value - self.best) > self.min_delta if self.mode == "max" else (self.best - metric_value) > self.min_delta
        if improved:
            self.best=metric_value
            self.counter=0
            return False
        self.counter += 1
        return self.counter >= self.patience


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


#평가함수
def evaluate_fnd_clip(bert_model, clip_model, fnd_module, loader):
    bert_model.eval()
    clip_model.eval()
    fnd_module.eval()
    all_true_4, all_pred_4 = [], []
    prob_chunks = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation (FND-CLIP 4C)"):
            bert_ids=batch["bert_input_ids"].to(DEVICE)
            bert_attn=batch["bert_attention_mask"].to(DEVICE)
            bert_ttid=batch["bert_token_type_ids"].to(DEVICE)
            img_resnet=batch["image_resnet"].to(DEVICE)
            clip_ids=batch["clip_input_ids"].to(DEVICE)
            clip_attn=batch["clip_attention_mask"].to(DEVICE)
            clip_pix=batch["clip_pixel_values"].to(DEVICE)
            lab4=batch["label4"].to(DEVICE)

            bert_out=bert_model(
                input_ids=bert_ids,
                attention_mask=bert_attn,
                token_type_ids=bert_ttid,
            )
            text_clip=clip_model.get_text_features(
                input_ids=clip_ids,
                attention_mask=clip_attn,
            )
            image_clip=clip_model.get_image_features(pixel_values=clip_pix)

            logits=fnd_module(
                bert_out.hidden_states,
                img_resnet,
                text_clip,
                image_clip,
            )
            prob_chunks.append(F.softmax(logits, dim=-1).detach().cpu().numpy())
            p4=logits.argmax(1).cpu().numpy()
            y4=lab4.cpu().numpy()
            all_pred_4.extend(p4.tolist())
            all_true_4.extend(y4.tolist())

    all_true_4=np.array(all_true_4)
    all_pred_4=np.array(all_pred_4)
    y_ha, p_ha=all_true_4 // 2, all_pred_4 // 2
    y_rf, p_rf=all_true_4 % 2, all_pred_4 % 2
    acc_4=accuracy_score(all_true_4, all_pred_4)
    acc_ha=accuracy_score(y_ha, p_ha)
    acc_rf=accuracy_score(y_rf, p_rf)
    f1_4=f1_score(all_true_4, all_pred_4, average="macro")
    f1_ha=f1_score(y_ha, p_ha, average="macro")
    f1_rf=f1_score(y_rf, p_rf, average="macro")

    target4=["Human Real", "Human Fake", "AI Real", "AI Fake"]
    print("\n=== FND-CLIP Human vs AI ===")
    print(classification_report(y_ha, p_ha, target_names=["Human", "AI"], digits=4))
    print("=== FND-CLIP Real vs Fake ===")
    print(classification_report(y_rf, p_rf, target_names=["Real", "Fake"], digits=4))
    print("=== FND-CLIP 4-class ===")
    print(classification_report(all_true_4, all_pred_4, target_names=target4, digits=4))
    proba = np.vstack(prob_chunks) if prob_chunks else np.zeros((0, 4), dtype=np.float64)
    aucm = auc_scores_from_proba(all_true_4, proba)
    print(
        f"\n[FND-CLIP] ROC-AUC: 4-class (macro-OVR)={aucm['auc_4_ovr_macro']:.4f} | "
        f"H/A={aucm['auc_ha']:.4f} | R/F={aucm['auc_rf']:.4f}"
    )
    print(
        f"\n[FND-CLIP] H/A Acc={acc_ha:.4f} F1={f1_ha:.4f} | R/F Acc={acc_rf:.4f} F1={f1_rf:.4f} | "
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


def _validate_fnd_clip(bert_model, clip_model, fnd_module, val_loader):
    bert_model.eval()
    clip_model.eval()
    fnd_module.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            bert_ids=batch["bert_input_ids"].to(DEVICE)
            bert_attn=batch["bert_attention_mask"].to(DEVICE)
            bert_ttid=batch["bert_token_type_ids"].to(DEVICE)
            img_resnet=batch["image_resnet"].to(DEVICE)
            clip_ids=batch["clip_input_ids"].to(DEVICE)
            clip_attn=batch["clip_attention_mask"].to(DEVICE)
            clip_pix=batch["clip_pixel_values"].to(DEVICE)
            lab4=batch["label4"]

            bert_out=bert_model(input_ids=bert_ids, attention_mask=bert_attn, token_type_ids=bert_ttid)
            text_clip=clip_model.get_text_features(input_ids=clip_ids, attention_mask=clip_attn)
            image_clip=clip_model.get_image_features(pixel_values=clip_pix)
            logits=fnd_module(bert_out.hidden_states, img_resnet, text_clip, image_clip)
            all_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            all_true.extend(lab4.numpy().tolist())
    return f1_score(all_true, all_pred, average="macro")


# -------------------------
# 8. 학습
# -------------------------
def train_fnd_clip_4c(bert_model, clip_model, fnd_module, train_loader, val_loader, epochs=EPOCHS):
    bert_model.to(DEVICE)
    clip_model.to(DEVICE)
    fnd_module.to(DEVICE)
    for p in bert_model.parameters():
        p.requires_grad = False
    for p in clip_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, fnd_module.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )
    ce=nn.CrossEntropyLoss()#4클래스 분류 손실함수
    early=EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, mode="max")
    best_val, best_state=-1.0, None

    for epoch in range(epochs):
        fnd_module.train()
        total_loss=0.0
        for batch in tqdm(train_loader, desc=f"[FND-CLIP 4C] Epoch {epoch+1}"):
            bert_ids = batch["bert_input_ids"].to(DEVICE)
            bert_attn = batch["bert_attention_mask"].to(DEVICE)
            bert_ttid = batch["bert_token_type_ids"].to(DEVICE)
            img_resnet = batch["image_resnet"].to(DEVICE)
            clip_ids = batch["clip_input_ids"].to(DEVICE)
            clip_attn = batch["clip_attention_mask"].to(DEVICE)
            clip_pix = batch["clip_pixel_values"].to(DEVICE)
            lab4 = batch["label4"].to(DEVICE)

            bert_out = bert_model(input_ids=bert_ids, attention_mask=bert_attn, token_type_ids=bert_ttid)
            text_clip = clip_model.get_text_features(input_ids=clip_ids, attention_mask=clip_attn)
            image_clip = clip_model.get_image_features(pixel_values=clip_pix)

            logits = fnd_module(bert_out.hidden_states, img_resnet, text_clip, image_clip)
            loss = ce(logits, lab4)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(fnd_module.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        val_f1 = _validate_fnd_clip(bert_model, clip_model, fnd_module, val_loader)
        print(f"[FND-CLIP 4C] Epoch {epoch+1} loss={avg_loss:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(fnd_module.state_dict())
        if early.step(val_f1):
            print(f"[FND-CLIP 4C] Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        fnd_module.load_state_dict(best_state)
    return bert_model, clip_model, fnd_module


# -------------------------
# 9. Tee
# -------------------------
class Tee(io.TextIOBase):
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


# -------------------------
# 10. K-Fold
# -------------------------
def kfold_fnd_clip_4c_with_reports(
    full_df,
    k=KFOLD_SPLITS,
    test_size=KFOLD_TEST_SIZE,
    shuffle=KFOLD_SHUFFLE,
    random_state=KFOLD_RANDOM_STATE,
):
    trainval_df, test_df = train_test_split(
        full_df, test_size=test_size, stratify=full_df["label"], random_state=random_state
    )
    print(f"[KFold-FND-CLIP 4C] Split: TrainVal={len(trainval_df)}, Test={len(test_df)}")
    print(
        f"Train loader: WeightedRandomSampler (hier2-style, alpha={SAMPLER_ALPHA})\n"
    )

    test_ds = FNDCLIPDataset(test_df, bert_tokenizer, clip_processor, max_len_bert=MAX_LEN_BERT, is_train=False)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fnd_clip,
    )

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    all_report = io.StringIO()
    metrics_4c = []
    y = trainval_df["label"].values

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(trainval_df)), y), start=1):
        print(f"\n========== [Fold {fold}/{k} - FND-CLIP 4C] ==========")
        tr_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
        va_df = trainval_df.iloc[va_idx].reset_index(drop=True)
        train_ds = FNDCLIPDataset(tr_df, bert_tokenizer, clip_processor, max_len_bert=MAX_LEN_BERT, is_train=True)
        val_ds = FNDCLIPDataset(va_df, bert_tokenizer, clip_processor, max_len_bert=MAX_LEN_BERT, is_train=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH,
            shuffle=False,
            sampler=make_weighted_sampler(tr_df["label"].tolist()),
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fnd_clip,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fnd_clip,
        )

        bert_model = BertModel.from_pretrained(BERT_NAME, output_hidden_states=True)
        clip_model = CLIPModel.from_pretrained(CLIP_NAME)
        fnd_module = FNDCLIPMultiModal4C(num_classes=4).to(DEVICE)

        buffer = io.StringIO()
        with redirect_stdout(Tee(sys.stdout, buffer)):
            print("\n" + "=" * 70)
            print(f"[Fold {fold}] FND-CLIP 4-class")
            print("=" * 70)
            bert_model, clip_model, fnd_module = train_fnd_clip_4c(
                bert_model, clip_model, fnd_module, train_loader, val_loader
            )
            res = evaluate_fnd_clip(bert_model, clip_model, fnd_module, test_loader)
            print("\n" + "=" * 70)
            print(f"[Fold {fold}] SUMMARY (FND-CLIP, Acc/F1 on Test)")
            print("=" * 70)
            print(
                f"FND-CLIP 4C: H/A Acc={res['acc_ha']:.4f}, F1={res['f1_ha']:.4f} | "
                f"R/F Acc={res['acc_rf']:.4f}, F1={res['f1_rf']:.4f} | 4-way Acc={res['acc_4']:.4f}, F1={res['f1_4']:.4f} | "
                f"AUC(4c)={res['auc_4_ovr_macro']:.4f} AUC(H/A)={res['auc_ha']:.4f} AUC(R/F)={res['auc_rf']:.4f}"
            )
            print("=" * 70)

        all_report.write(f"\n\n{'#'*90}\n### FOLD {fold}/{k} REPORT (FND-CLIP 4C)\n{'#'*90}\n")
        all_report.write(buffer.getvalue())
        metrics_4c.append(res)
        gc.collect()

    def _mean_metric(dict_list, key):
        vals = [d[key] for d in dict_list if d.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    all_report.write("\n\n" + "#" * 90 + "\n")
    all_report.write("### GLOBAL SUMMARY (FND-CLIP Mean over {} folds)\n".format(k))
    all_report.write("#" * 90 + "\n")
    all_report.write(
        f"FND-CLIP 4C: mean H/A Acc={_mean_metric(metrics_4c,'acc_ha'):.4f}, F1={_mean_metric(metrics_4c,'f1_ha'):.4f} | "
        f"mean R/F Acc={_mean_metric(metrics_4c,'acc_rf'):.4f}, F1={_mean_metric(metrics_4c,'f1_rf'):.4f} | "
        f"mean 4-way Acc={_mean_metric(metrics_4c,'acc_4'):.4f}, F1={_mean_metric(metrics_4c,'f1_4'):.4f} | "
        f"mean AUC(4c)={_mean_metric(metrics_4c,'auc_4_ovr_macro'):.4f} "
        f"AUC(H/A)={_mean_metric(metrics_4c,'auc_ha'):.4f} AUC(R/F)={_mean_metric(metrics_4c,'auc_rf'):.4f}\n"
    )
    all_report.write("\n[TABLE_SUMMARY]\n")
    all_report.write(
        "Variant\tHA_Acc\tHA_F1\tRF_Acc\tRF_F1\t4C_Acc\t4C_F1\tAUC_4c\tAUC_HA\tAUC_RF\n"
    )
    all_report.write(
        f"FND-CLIP 4C\t{_mean_metric(metrics_4c,'acc_ha'):.4f}\t{_mean_metric(metrics_4c,'f1_ha'):.4f}\t"
        f"{_mean_metric(metrics_4c,'acc_rf'):.4f}\t{_mean_metric(metrics_4c,'f1_rf'):.4f}\t"
        f"{_mean_metric(metrics_4c,'acc_4'):.4f}\t{_mean_metric(metrics_4c,'f1_4'):.4f}\t"
        f"{_mean_metric(metrics_4c,'auc_4_ovr_macro'):.4f}\t{_mean_metric(metrics_4c,'auc_ha'):.4f}\t"
        f"{_mean_metric(metrics_4c,'auc_rf'):.4f}\n"
    )
    output_path = os.path.join(RESULT_DIR, f"kfold_fnd_clip_4c_{timestamp}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_report.getvalue())
    print(f"\n리포트 저장: {output_path}\n")


# -------------------------
# 11. 단일 split
# -------------------------
def get_single_split_loaders(full_df):
    # hier2 split_602020: 60% / 20% / 20%
    idx = np.arange(len(full_df))
    y = full_df["label"].values
    tva, te, _, _ = train_test_split(
        idx, y, test_size=TEST_RATIO, stratify=y, random_state=SEED
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
    print(
        f"Train loader: WeightedRandomSampler (hier2-style, alpha={SAMPLER_ALPHA})\n"
    )
    train_ds = FNDCLIPDataset(train_data, bert_tokenizer, clip_processor, max_len_bert=MAX_LEN_BERT, is_train=True)
    val_ds = FNDCLIPDataset(val_data, bert_tokenizer, clip_processor, max_len_bert=MAX_LEN_BERT, is_train=False)
    test_ds = FNDCLIPDataset(test_data, bert_tokenizer, clip_processor, max_len_bert=MAX_LEN_BERT, is_train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=False,
        sampler=make_weighted_sampler(train_data["label"].tolist()),
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fnd_clip,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fnd_clip,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fnd_clip,
    )
    return train_loader, val_loader, test_loader


# -------------------------
# 12. Main
# -------------------------
if __name__ == "__main__":
    print(
        f"FND-CLIP Baseline (hier2-aligned) | KFold={USE_KFOLD}, BATCH={BATCH}, "
        f"LR={LR}, EPOCHS={EPOCHS}, EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE}, "
        f"sampler_alpha={SAMPLER_ALPHA}, SEED={SEED} | "
        f"data=HARFM.csv (final_headline, image_path, 4_way_label)"
    )
    if USE_KFOLD:
        kfold_fnd_clip_4c_with_reports(
            data, k=KFOLD_SPLITS, test_size=KFOLD_TEST_SIZE,
            shuffle=KFOLD_SHUFFLE, random_state=KFOLD_RANDOM_STATE,
        )
    else:
        train_loader, val_loader, test_loader = get_single_split_loaders(data)
        bert_model = BertModel.from_pretrained(BERT_NAME, output_hidden_states=True)
        clip_model = CLIPModel.from_pretrained(CLIP_NAME)
        fnd_module = FNDCLIPMultiModal4C(num_classes=4).to(DEVICE)
        bert_model, clip_model, fnd_module = train_fnd_clip_4c(
            bert_model, clip_model, fnd_module, train_loader, val_loader
        )
        _ = evaluate_fnd_clip(bert_model, clip_model, fnd_module, test_loader)
