#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import io
import os
import sys
import warnings
from contextlib import nullcontext, redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import ViT_B_16_Weights, vit_b_16

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


SEED = 42


def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# -------------------------
# 1. 기본 설정
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FND_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
CSV_PATH = os.environ.get("HARFM_CSV_PATH", os.path.join(_FND_ROOT, "HARFM.csv"))

BATCH = 16
ACCUM_STEPS = 1
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4
USE_AMP = False
IMG_SIZE = 224

EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_DELTA = 1e-4
SAMPLER_ALPHA = 0.5

USE_KFOLD = True
KFOLD_SPLITS = 5
KFOLD_TEST_SIZE = 0.2
KFOLD_SHUFFLE = True
KFOLD_RANDOM_STATE = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2


# -------------------------
# 2. 데이터 로드
# -------------------------
print("Loading CSV for ViT-B/16 image-only baseline...")
data = pd.read_csv(CSV_PATH, low_memory=False)

required_cols = {"final_headline", "image_path", "4_way_label"}
missing = required_cols - set(data.columns)
if missing:
    raise ValueError(f"CSV에 다음 컬럼이 없습니다: {missing}")

print(f"Dataset loaded: {len(data)} rows")


def _normalize_label(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, np.integer)):
        if 0 <= int(v) <= 3:
            return int(v)
        return None
    if isinstance(v, float):
        iv = int(v)
        if abs(v - iv) < 1e-9 and 0 <= iv <= 3:
            return iv
        return None
    if isinstance(v, str):
        s = v.strip().strip('"').strip("'").lower()
        mapping = {
            "0": 0, "1": 1, "2": 2, "3": 3,
            "hr": 0, "human_real": 0, "humanreal": 0,
            "hf": 1, "human_fake": 1, "humanfake": 1, "hd": 1,
            "ar": 2, "ai_real": 2, "aireal": 2,
            "af": 3, "ai_fake": 3, "aifake": 3,
        }
        return mapping.get(s, None)
    return None


def _resolve_image_path(path):
    if not isinstance(path, str) or not path.strip():
        return ""
    p = path.strip().strip('"').strip("'")
    if not os.path.isabs(p):
        p = os.path.join(_FND_ROOT, p)
    return p


data["image_path"] = data["image_path"].apply(_resolve_image_path)
data["label4"] = data["4_way_label"].apply(_normalize_label)
data["text"] = data["final_headline"].fillna("").astype(str).str.strip()

n_before = len(data)
data = data[
    data["text"].ne("") &
    data["image_path"].apply(lambda p: isinstance(p, str) and p.strip() != "" and os.path.isfile(p))
].copy()
data = data[data["label4"].notna()].copy()
data["label4"] = data["label4"].astype(int)
data = data.reset_index(drop=True)
n_removed = n_before - len(data)

print(f"[전처리] 멀티모달 유효 샘플(text+image) 필터: {n_before} -> {len(data)} ({n_removed}개 제거)")
print("Label counts:", data["label4"].value_counts().sort_index().to_dict())


# -------------------------
# 3. Image Transform
# -------------------------
vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
vit_transform = vit_weights.transforms()

train_transform = vit_transform
val_transform = vit_transform


# -------------------------
# 4. Dataset (Image-only 4-class)
# -------------------------
class ViTImageOnlyDataset(Dataset):
    def __init__(self, df, img_transform=None):
        self.df = df.reset_index(drop=True)
        self.img_transform = img_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.img_transform is not None:
            img = self.img_transform(img)
        label4 = int(row["label4"])

        return {
            "image": img,
            "label4": torch.tensor(label4, dtype=torch.long),
        }


def make_weighted_sampler(labels, alpha: float = SAMPLER_ALPHA):
    cnt = pd.Series(labels).value_counts().to_dict()
    w = [(1.0 / cnt[l]) ** alpha for l in labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


# -------------------------
# 5. ViT-B/16 모델 (Image-only)
# -------------------------
class ViTImageClassifier4C(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vit_b_16(weights=vit_weights)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, 4)

    def forward(self, images):
        return self.backbone(images)


# -------------------------
# 6. EarlyStopping
# -------------------------
class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, mode="max"):
        assert mode in ["max", "min"]
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


# -------------------------
# 7. 평가 함수 (H/A, R/F, 4-way)
# -------------------------
def evaluate_vit(model, loader):
    model.eval()
    all_true_4, all_pred_4 = [], []
    prob_chunks = []
    all_true_ha, all_pred_ha = [], []
    all_true_rf, all_pred_rf = [], []
    target4 = ["Human Real", "Human Fake", "AI Real", "AI Fake"]

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation (ViT image-only 4C)"):
            imgs = batch["image"].to(DEVICE, non_blocking=True)
            lab4 = batch["label4"].to(DEVICE, non_blocking=True)

            with nullcontext():
                logits4 = model(imgs)

            prob_chunks.append(
                torch.softmax(logits4, dim=-1).detach().cpu().numpy()
            )
            p4 = torch.argmax(logits4, dim=1)
            p4_np = p4.cpu().numpy()
            y4_np = lab4.cpu().numpy()

            all_pred_4.extend(p4_np.tolist())
            all_true_4.extend(y4_np.tolist())

            y_ha = y4_np // 2
            p_ha = p4_np // 2
            y_rf = y4_np % 2
            p_rf = p4_np % 2

            all_true_ha.extend(y_ha.tolist())
            all_pred_ha.extend(p_ha.tolist())
            all_true_rf.extend(y_rf.tolist())
            all_pred_rf.extend(p_rf.tolist())

    all_true_4_arr = np.array(all_true_4)
    all_pred_4_arr = np.array(all_pred_4)
    all_true_ha_arr = np.array(all_true_ha)
    all_pred_ha_arr = np.array(all_pred_ha)
    all_true_rf_arr = np.array(all_true_rf)
    all_pred_rf_arr = np.array(all_pred_rf)

    acc_4 = accuracy_score(all_true_4_arr, all_pred_4_arr)
    acc_ha = accuracy_score(all_true_ha_arr, all_pred_ha_arr)
    acc_rf = accuracy_score(all_true_rf_arr, all_pred_rf_arr)

    f1_4 = f1_score(all_true_4_arr, all_pred_4_arr, average="macro")
    f1_ha = f1_score(all_true_ha_arr, all_pred_ha_arr, average="macro")
    f1_rf = f1_score(all_true_rf_arr, all_pred_rf_arr, average="macro")

    print("\n=== ViT image-only Human vs AI report ===")
    print(classification_report(all_true_ha_arr, all_pred_ha_arr, target_names=["Human", "AI"], digits=4))
    print("Confusion matrix (H/A):")
    print(confusion_matrix(all_true_ha_arr, all_pred_ha_arr))

    print("\n=== ViT image-only Real vs Fake report ===")
    print(classification_report(all_true_rf_arr, all_pred_rf_arr, target_names=["Real", "Fake"], digits=4))
    print("Confusion matrix (R/F):")
    print(confusion_matrix(all_true_rf_arr, all_pred_rf_arr))

    print("\n=== ViT image-only 4-class report ===")
    print(classification_report(all_true_4_arr, all_pred_4_arr, target_names=target4, digits=4))
    print("Confusion matrix (4-way):")
    print(confusion_matrix(all_true_4_arr, all_pred_4_arr))

    proba = np.vstack(prob_chunks) if prob_chunks else np.zeros((0, 4), dtype=np.float64)
    aucm = auc_scores_from_proba(all_true_4_arr, proba)
    print(
        f"\n[ViT Image-only] ROC-AUC: 4-class (macro-OVR)={aucm['auc_4_ovr_macro']:.4f} | "
        f"H/A={aucm['auc_ha']:.4f} | R/F={aucm['auc_rf']:.4f}"
    )
    print(
        f"\n[ViT Image-only] "
        f"H/A -> Acc={acc_ha:.4f}, F1={f1_ha:.4f} | "
        f"R/F -> Acc={acc_rf:.4f}, F1={f1_rf:.4f} | "
        f"4-way -> Acc={acc_4:.4f}, F1={f1_4:.4f} | "
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


# -------------------------
# 8. Validation용 (4-way macro F1)
# -------------------------
def _validate_vit_four_class(model, val_loader):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(DEVICE, non_blocking=True)
            lab4 = batch["label4"].to(DEVICE, non_blocking=True)

            with nullcontext():
                logits4 = model(imgs)

            p4 = torch.argmax(logits4, dim=1).cpu().numpy()
            all_pred.extend(p4.tolist())
            all_true.extend(lab4.cpu().numpy().tolist())

    return f1_score(all_true, all_pred, average="macro")


# -------------------------
# 9. ViT 학습 함수
# -------------------------
def train_vit_4c(model, train_loader, val_loader, epochs=EPOCHS):
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    early = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, mode="max")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"[ViT-4C] Epoch {epoch + 1}"):
            imgs = batch["image"].to(DEVICE, non_blocking=True)
            lab4 = batch["label4"].to(DEVICE, non_blocking=True)

            with nullcontext():
                logits4 = model(imgs)
                loss = ce(logits4, lab4)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())

        avg_train_loss = total_loss / max(1, len(train_loader))
        print(f"[ViT-4C] avg_train_loss={avg_train_loss:.4f}")

        val_f1 = _validate_vit_four_class(model, val_loader)
        print(f"[ViT-4C] val_macro_f1(4-way)={val_f1:.4f}")

        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(model.state_dict())
            print("[ViT-4C] Saved best model state (in-memory).")

        if early.step(val_f1):
            print(
                f"[ViT-4C] [EarlyStop] No improvement for {EARLY_STOP_PATIENCE} epochs "
                f"(min_delta={EARLY_STOP_MIN_DELTA}). Stopping."
            )
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


# -------------------------
# 10. K-Fold Runner
# -------------------------
def kfold_vit_4c_with_reports(
    full_df,
    k=KFOLD_SPLITS,
    test_size=KFOLD_TEST_SIZE,
    shuffle=KFOLD_SHUFFLE,
    random_state=KFOLD_RANDOM_STATE,
):
    trainval_df, test_df = train_test_split(
        full_df,
        test_size=test_size,
        stratify=full_df["label4"],
        random_state=random_state,
    )
    print(f"[KFold-ViT-4C] Split: TrainVal={len(trainval_df)}, Test={len(test_df)}")

    test_ds = ViTImageOnlyDataset(test_df, img_transform=val_transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    all_report = io.StringIO()
    all_report.write(f"###### ViT-B/16 Image-only 4C (k={k}) ######\n")

    metrics_4c = []
    y = trainval_df["label4"].values

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(trainval_df)), y), start=1):
        print(f"\n========== [Fold {fold}/{k} - ViT-4C] ==========")
        tr_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
        va_df = trainval_df.iloc[va_idx].reset_index(drop=True)

        train_ds = ViTImageOnlyDataset(tr_df, img_transform=train_transform)
        val_ds = ViTImageOnlyDataset(va_df, img_transform=val_transform)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH,
            shuffle=False,
            sampler=make_weighted_sampler(tr_df["label4"].tolist()),
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        buffer = io.StringIO()
        tee = Tee(sys.stdout, buffer)
        with redirect_stdout(tee):
            print("\n" + "=" * 70)
            print(f"[Fold {fold}] ViT-B/16 Image-only 4-class")
            print("=" * 70)
            model = ViTImageClassifier4C().to(DEVICE)
            model = train_vit_4c(model, train_loader, val_loader)
            res = evaluate_vit(model, test_loader)

            print("\n" + "=" * 70)
            print(f"[Fold {fold}] SUMMARY (ViT-B/16 Image-only, Acc/F1 on Test)")
            print("=" * 70)
            print(
                f"ViT-B16-4C: "
                f"H/A Acc={res['acc_ha']:.4f}, F1={res['f1_ha']:.4f} | "
                f"R/F Acc={res['acc_rf']:.4f}, F1={res['f1_rf']:.4f} | "
                f"4-way Acc={res['acc_4']:.4f}, F1={res['f1_4']:.4f} | "
                f"AUC(4c)={res['auc_4_ovr_macro']:.4f} AUC(H/A)={res['auc_ha']:.4f} "
                f"AUC(R/F)={res['auc_rf']:.4f}"
            )
            print("=" * 70)

        fold_text = buffer.getvalue()
        all_report.write(f"\n\n{'#' * 90}\n### FOLD {fold}/{k} REPORT (ViT-4C)\n{'#' * 90}\n")
        all_report.write(fold_text)
        all_report.flush()
        metrics_4c.append(res)

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
    all_report.write(f"### GLOBAL SUMMARY (ViT-B/16 image-only, Mean over {k} folds)\n")
    all_report.write("#" * 90 + "\n")
    all_report.write(
        f"ViT-B16-4C (image-only): "
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
        "ViT-B16-4C(image-only)\t"
        f"{mean_acc_ha:.4f}\t{mean_f1_ha:.4f}\t"
        f"{mean_acc_rf:.4f}\t{mean_f1_rf:.4f}\t"
        f"{mean_acc_4:.4f}\t{mean_f1_4:.4f}\t"
        f"{mean_auc_4:.4f}\t{mean_auc_ha:.4f}\t{mean_auc_rf:.4f}\n"
    )

    output_path = os.path.join(RESULT_DIR, f"kfold_vitb16_4c_{timestamp}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_report.getvalue())
    print(f"\n모든 fold 리포트 + TABLE_SUMMARY 저장 완료: {output_path}\n")


def get_single_split_loaders_vit(full_df):
    idx = np.arange(len(full_df))
    y = full_df["label4"].values
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

    train_ds = ViTImageOnlyDataset(train_data, img_transform=train_transform)
    val_ds = ViTImageOnlyDataset(val_data, img_transform=val_transform)
    test_ds = ViTImageOnlyDataset(test_data, img_transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=False,
        sampler=make_weighted_sampler(train_data["label4"].tolist()),
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def print_memory_usage():
    if torch.cuda.is_available():
        print(
            f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / "
            f"{torch.cuda.memory_reserved()/1024**3:.2f}GB"
        )


if __name__ == "__main__":
    print("Initial memory usage:")
    print_memory_usage()
    print(
        f"ViT-B/16 image-only Baseline | "
        f"KFold={USE_KFOLD}, BATCH={BATCH}, LR={LR}, EPOCHS={EPOCHS}, "
        f"EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE}, "
        f"WeightedRandomSampler(alpha={SAMPLER_ALPHA}), SEED={SEED}, IMG_SIZE={IMG_SIZE}"
    )

    if USE_KFOLD:
        kfold_vit_4c_with_reports(
            data,
            k=KFOLD_SPLITS,
            test_size=KFOLD_TEST_SIZE,
            shuffle=KFOLD_SHUFFLE,
            random_state=KFOLD_RANDOM_STATE,
        )
    else:
        train_loader, val_loader, test_loader = get_single_split_loaders_vit(data)
        model = ViTImageClassifier4C().to(DEVICE)
        model = train_vit_4c(model, train_loader, val_loader)
        _ = evaluate_vit(model, test_loader)
