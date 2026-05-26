"""
SpotFake+ Baseline for HARFM (4-class: HR, HF, AR, AF)
- 데이터: HARFM.csv (text=final_headline, image=image_path, label=4_way_label)
- Text: XLNet (768-dim), Image: VGG16 fc2 (4096-dim), Fusion → 4-way
"""

import os
import io
import sys
import warnings
import copy
import gc
from contextlib import redirect_stdout
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from transformers import XLNetTokenizer, XLNetModel
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
warnings.filterwarnings("ignore")

SEED = 42


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FND_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
CSV_PATH = os.path.join(_FND_ROOT, "HARFM.csv")

# 모델에 필요한 컬럼 3가지만 사용
# text: final_headline (hr/hd/ar/af 모두 담긴 텍스트)
# image: image_path
# label: 4_way_label

TEXT_MODEL = "xlnet-base-cased"
MAX_LEN = 64
IMG_SIZE = 224

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


#데이터 로드 (HARFM.csv — final_headline, image_path, 4_way_label)

LABEL_MAP = {"HR": 0, "HF": 1, "AR": 2, "AF": 3}
REQUIRED_COLS = {"final_headline", "image_path", "4_way_label"}


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


print("Loading HARFM.csv for SpotFake+ baseline (text=final_headline, image=image_path, label=4_way_label)...")
raw = pd.read_csv(CSV_PATH, low_memory=False)

missing = REQUIRED_COLS - set(raw.columns)
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

# 이미지 경로 유효한 샘플만 유지
n_before = len(data)
data = data[
    data["image_path"].apply(
        lambda p: isinstance(p, str) and p != "" and os.path.exists(p)
    )
].reset_index(drop=True)
n_removed = n_before - len(data)
print(f"\n[전처리] 이미지 결측 제거: {n_before} → {len(data)} ({n_removed}개 제거)")
print("[전처리] 제거 후 label 분포:", data["label"].value_counts().sort_index().to_dict())


# -------------------------
# 3. 전처리기
# -------------------------
tokenizer = XLNetTokenizer.from_pretrained(TEXT_MODEL)

image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# -------------------------
# 4. Dataset
# -------------------------
class SpotFakePlusDataset(Dataset):
    """
    - image: image_path
    - label: 0=HR, 1=HF, 2=AR, 3=AF
    """

    def __init__(self, df, tokenizer, max_len=MAX_LEN, is_train=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
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
        img = self._get_image(row)

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        img_tensor = image_transform(img)
        label4 = int(row["label"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": img_tensor,
            "label4": torch.tensor(label4, dtype=torch.long),
        }


def make_weighted_sampler(labels, alpha: float = SAMPLER_ALPHA):
    cnt = pd.Series(labels).value_counts().to_dict()
    weights = [(1.0 / cnt[lbl]) ** alpha for lbl in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# -------------------------
# 5. SpotFake+ Multimodal 모델 (XLNet 768 + VGG16 4096 → Fusion → 4-class)
# -------------------------
class SpotFakePlus4C(nn.Module):
    """Text(XLNet 768) + Image(VGG16 4096) → Fusion → 4-class"""

    def __init__(self, text_model_name=TEXT_MODEL):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained(text_model_name)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg_feat = nn.Sequential(
            vgg.features,
            vgg.avgpool,
            nn.Flatten(),
            vgg.classifier[0],
            vgg.classifier[1],
            vgg.classifier[2],
            vgg.classifier[3],
            vgg.classifier[4],
        )
        fusion_in = 768 + 4096
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 2000),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(100, 4),
        )

    def forward(self, input_ids, attention_mask, image):
        out = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = out.last_hidden_state[:, -1, :]  # (B, 768)
        img_feat = self.vgg_feat(image)  # (B, 4096)
        h = torch.cat([text_feat, img_feat], dim=1)
        return self.fusion(h)


# -------------------------
# 6. EarlyStopping
# -------------------------
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


def evaluate_spotfake(model, loader):
    model.eval()
    all_true_4, all_pred_4 = [], []
    prob_chunks = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation (SpotFake+ 4C)"):
            ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            img = batch["image"].to(DEVICE)
            lab4 = batch["label4"].to(DEVICE)

            logits4 = model(ids, attn, img)
            prob_chunks.append(
                torch.softmax(logits4, dim=-1).detach().cpu().numpy()
            )
            p4 = torch.argmax(logits4, dim=1).cpu().numpy()
            y4 = lab4.cpu().numpy()

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
    print("\n=== SpotFake+ Human vs AI ===")
    print(classification_report(y_ha, p_ha, target_names=["Human", "AI"], digits=4))
    print("=== SpotFake+ Real vs Fake ===")
    print(classification_report(y_rf, p_rf, target_names=["Real", "Fake"], digits=4))
    print("=== SpotFake+ 4-class ===")
    print(classification_report(all_true_4, all_pred_4, target_names=target4, digits=4))
    proba = np.vstack(prob_chunks) if prob_chunks else np.zeros((0, 4), dtype=np.float64)
    aucm = auc_scores_from_proba(all_true_4, proba)
    print(
        f"\n[SpotFake+] ROC-AUC: 4-class (macro-OVR)={aucm['auc_4_ovr_macro']:.4f} | "
        f"H/A={aucm['auc_ha']:.4f} | R/F={aucm['auc_rf']:.4f}"
    )
    print(
        f"\n[SpotFake+] "
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


def _validate(model, val_loader):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            img = batch["image"].to(DEVICE)
            lab4 = batch["label4"]
            logits4 = model(ids, attn, img)
            p4 = torch.argmax(logits4, dim=1).cpu().numpy()
            all_pred.extend(p4.tolist())
            all_true.extend(lab4.numpy().tolist())
    return f1_score(all_true, all_pred, average="macro")

def train_spotfake_4c(model, train_loader, val_loader, epochs=EPOCHS):
    model.to(DEVICE)
    for p in model.xlnet.parameters():
        p.requires_grad = False
    for p in model.vgg_feat.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    ce = nn.CrossEntropyLoss()
    early = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, mode="max")
    best_val, best_state = -1.0, None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[SpotFake+ 4C] Epoch {epoch+1}"):
            ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            img = batch["image"].to(DEVICE)
            lab4 = batch["label4"].to(DEVICE)

            optimizer.zero_grad()
            logits4 = model(ids, attn, img)
            loss = ce(logits4, lab4)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        val_f1 = _validate(model, val_loader)
        print(f"[SpotFake+ 4C] Epoch {epoch+1} loss={avg_loss:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(model.state_dict())

        if early.step(val_f1):
            print(f"[SpotFake+ 4C] Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

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

def kfold_spotfake_4c_with_reports(
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
    print(f"[KFold-SpotFake+ 4C] Split: TrainVal={len(trainval_df)}, Test={len(test_df)}")

    test_ds = SpotFakePlusDataset(test_df, tokenizer=tokenizer, max_len=MAX_LEN, is_train=False)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    all_report = io.StringIO()
    all_report.write(f"###### SpotFake+ 4C (k={k}) ######\n")
    metrics_4c = []
    y = trainval_df["label"].values

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(trainval_df)), y), start=1):
        print(f"\n========== [Fold {fold}/{k} - SpotFake+ 4C] ==========")
        tr_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
        va_df = trainval_df.iloc[va_idx].reset_index(drop=True)

        train_ds = SpotFakePlusDataset(tr_df, tokenizer=tokenizer, max_len=MAX_LEN, is_train=True)
        val_ds = SpotFakePlusDataset(va_df, tokenizer=tokenizer, max_len=MAX_LEN, is_train=False)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH, shuffle=False,
            sampler=make_weighted_sampler(tr_df["label"].tolist()),
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        )

        buffer = io.StringIO()
        tee = Tee(sys.stdout, buffer)
        with redirect_stdout(tee):
            print("\n" + "=" * 70)
            print(f"[Fold {fold}] SpotFake+ 4-class")
            print("=" * 70)
            model = SpotFakePlus4C().to(DEVICE)
            model = train_spotfake_4c(model, train_loader, val_loader)
            res = evaluate_spotfake(model, test_loader)
            print("\n" + "=" * 70)
            print(f"[Fold {fold}] SUMMARY (SpotFake+, Acc/F1 on Test)")
            print("=" * 70)
            print(
                f"SpotFake+ 4C: "
                f"H/A Acc={res['acc_ha']:.4f}, F1={res['f1_ha']:.4f} | "
                f"R/F Acc={res['acc_rf']:.4f}, F1={res['f1_rf']:.4f} | "
                f"4-way Acc={res['acc_4']:.4f}, F1={res['f1_4']:.4f} | "
                f"AUC(4c)={res['auc_4_ovr_macro']:.4f} AUC(H/A)={res['auc_ha']:.4f} "
                f"AUC(R/F)={res['auc_rf']:.4f}"
            )
            print("=" * 70)

        all_report.write(f"\n\n{'#'*90}\n### FOLD {fold}/{k} REPORT (SpotFake+ 4C)\n{'#'*90}\n")
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
    all_report.write("### GLOBAL SUMMARY (SpotFake+ Mean over {} folds)\n".format(k))
    all_report.write("#" * 90 + "\n")
    all_report.write(
        f"SpotFake+ 4C: "
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
        f"SpotFake+ 4C\t"
        f"{mean_acc_ha:.4f}\t{mean_f1_ha:.4f}\t"
        f"{mean_acc_rf:.4f}\t{mean_f1_rf:.4f}\t"
        f"{mean_acc_4:.4f}\t{mean_f1_4:.4f}\t"
        f"{mean_auc_4:.4f}\t{mean_auc_ha:.4f}\t{mean_auc_rf:.4f}\n"
    )

    output_path = os.path.join(RESULT_DIR, f"kfold_spotfake_4c_{timestamp}.txt")
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

    train_ds = SpotFakePlusDataset(train_data, tokenizer=tokenizer, max_len=MAX_LEN, is_train=True)
    val_ds = SpotFakePlusDataset(val_data, tokenizer=tokenizer, max_len=MAX_LEN, is_train=False)
    test_ds = SpotFakePlusDataset(test_data, tokenizer=tokenizer, max_len=MAX_LEN, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=False,
        sampler=make_weighted_sampler(train_data["label"].tolist()),
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


# -------------------------
# 12. Main
# -------------------------
if __name__ == "__main__":
    print(
        f"SpotFake+ Baseline | "
        f"KFold={USE_KFOLD}, BATCH={BATCH}, LR={LR}, EPOCHS={EPOCHS}, "
        f"EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE}, "
        f"WeightedRandomSampler(alpha={SAMPLER_ALPHA}), SEED={SEED}, "
        f"data=HARFM.csv (final_headline, image_path, 4_way_label)"
    )
    if USE_KFOLD:
        kfold_spotfake_4c_with_reports(
            data,
            k=KFOLD_SPLITS,
            test_size=KFOLD_TEST_SIZE,
            shuffle=KFOLD_SHUFFLE,
            random_state=KFOLD_RANDOM_STATE,
        )
    else:
        train_loader, val_loader, test_loader = get_single_split_loaders(data)
        model = SpotFakePlus4C().to(DEVICE)
        model = train_spotfake_4c(model, train_loader, val_loader)
        _ = evaluate_spotfake(model, test_loader)
