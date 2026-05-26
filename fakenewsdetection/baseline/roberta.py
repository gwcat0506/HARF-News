"""
RoBERTa text-only baseline (4-class).
- 데이터: HARFM.csv (text=final_headline, label=4_way_label)
- final_headline: hr, hd, ar, af 모두 포함된 텍스트
"""

import os, io, sys, warnings, copy, gc
from contextlib import nullcontext, redirect_stdout
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
from datetime import datetime
from torch.amp import autocast, GradScaler

timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
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
CSV_PATH = os.environ.get("HARFM_CSV_PATH", os.path.join(_FND_ROOT, "HARFM.csv"))
BATCH = 16
ACCUM_STEPS = 1
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4
USE_AMP = False
MAX_LEN = 128
EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_DELTA = 1e-4
SAMPLER_ALPHA = 0.5
USE_KFOLD = True
KFOLD_SPLITS = 5
KFOLD_TEST_SIZE = 0.2
KFOLD_SHUFFLE = True
KFOLD_RANDOM_STATE = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2

print("Loading HARFM.csv for RoBERTa text-only baseline...")
data = pd.read_csv(CSV_PATH, low_memory=False)
#HARFM.csv 필수 컬럼: final_headline (text), image_path (멀티모달 필터), 4_way_label (label)
required_cols = {"final_headline", "image_path", "4_way_label"}
missing=required_cols - set(data.columns)
if missing:
    raise ValueError(f"CSV에 다음 컬럼이 없습니다: {missing}")
#4_way_label (HR/HF/AR/AF) -> numeric label (0=HR, 1=HF, 2=AR, 3=AF)
label_map = {"HR": 0, "HF": 1, "AR": 2, "AF": 3}
data["label"] = data["4_way_label"].map(label_map)
data=data[data["label"].notna()].astype({"label": int}).reset_index(drop=True)
#final_headline 결측/빈 값 제거
data=data[
    data["final_headline"].notna() & (data["final_headline"].astype(str).str.strip() != "")
].reset_index(drop=True)
data["image_path"] = data["image_path"].fillna("").astype(str).str.strip()
data["image_path"] = data["image_path"].apply(
    lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(_FND_ROOT, p))
)
before_mm = len(data)
data = data[
    data["image_path"].apply(lambda p: isinstance(p, str) and p != "" and os.path.isfile(p))
].reset_index(drop=True)

print(f"Dataset loaded: {len(data)} rows")
print(f"[전처리] 멀티모달 필터(이미지 파일 유효): {before_mm} -> {len(data)}")
print("Label counts (0=HR, 1=HF, 2=AR, 3=AF):", data["label"].value_counts().sort_index().to_dict())


# -------------------------
# 2-1. 전처리: 텍스트 따옴표 제거
# -------------------------
def _strip_quotes(val):
    if not isinstance(val, str):
        return val
    val = val.strip()
    while (len(val) >= 2
           and ((val[0] == '"' and val[-1] == '"')
                or (val[0] == "'" and val[-1] == "'"))):
        val = val[1:-1].strip()
    return val


data["final_headline"] = data["final_headline"].apply(_strip_quotes)
print("[전처리] 텍스트 따옴표 제거 완료 (final_headline)\n")


# -------------------------
# 3. Tokenizer
# -------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


# -------------------------
# 4. Dataset (Text-only 4-class)
# -------------------------
class TextOnlyFourClassDataset(Dataset):
    """
    - text: final_headline (hr, hd, ar, af 모두 포함)
    - label: 4_way_label -> 0/1/2/3
    - 출력: input_ids, attention_mask, label4
    """

    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _get_text(self, row):
        t = row.get("final_headline", "")
        return t if isinstance(t, str) else ""

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self._get_text(row)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        label4 = int(row["label"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label4": torch.tensor(label4, dtype=torch.long),
        }


def make_weighted_sampler(labels, alpha: float = SAMPLER_ALPHA):
    cnt = pd.Series(labels).value_counts().to_dict()
    w = [(1.0 / cnt[l]) ** alpha for l in labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


# -------------------------
# 5. RoBERTa Text-only 4-class 모델
# -------------------------
class RobertaTextOnly4C(nn.Module):
    """
    RoBERTa-base + 4-class head
    """

    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        hidden = self.roberta.config.hidden_size  # 768
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, 4)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0, :]  # [CLS]
        x = self.dropout(cls)
        logits = self.classifier(x)           # (B, 4)
        return logits


# -------------------------
# 6. EarlyStopping
# -------------------------
class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE,
                 min_delta=EARLY_STOP_MIN_DELTA,
                 mode='max'):
        assert mode in ['max', 'min']
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, metric_value: float) -> bool:
        if self.best is None:
            self.best = metric_value
            return False
        if self.mode == 'max':
            improved = (metric_value - self.best) > self.min_delta
        else:
            improved = (self.best - metric_value) > self.min_delta

        if improved:
            self.best = metric_value
            self.counter = 0
            return False
        else:
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
# 7. 평가 함수 (H/A, R/F, 4-way Acc & F1 모두)
# -------------------------
def evaluate_4class_roberta(model, loader):
    model.eval()
    all_true_4, all_pred_4 = [], []
    prob_chunks = []
    all_true_ha, all_pred_ha = [], []
    all_true_rf, all_pred_rf = [], []

    target4 = ['Human Real', 'Human Fake', 'AI Real', 'AI Fake']

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation (RoBERTa text-only 4C)"):
            ids  = batch["input_ids"].to(DEVICE, non_blocking=True)
            attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
            lab4 = batch["label4"].to(DEVICE, non_blocking=True)

            ctx = autocast(device_type="cuda", enabled=USE_AMP) if USE_AMP else nullcontext()
            with ctx:
                logits4 = model(ids, attn)

            prob_chunks.append(
                torch.softmax(logits4.float(), dim=-1).detach().cpu().numpy()
            )
            p4    = torch.argmax(logits4, dim=1)
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

            del ids, attn, lab4, logits4, p4

    acc_4  = accuracy_score(all_true_4,  all_pred_4)
    acc_ha = accuracy_score(all_true_ha, all_pred_ha)
    acc_rf = accuracy_score(all_true_rf, all_pred_rf)

    f1_4  = f1_score(all_true_4,  all_pred_4,  average="macro")
    f1_ha = f1_score(all_true_ha, all_pred_ha, average="macro")
    f1_rf = f1_score(all_true_rf, all_pred_rf, average="macro")

    print("\n=== RoBERTa Text-only Human vs AI report ===")
    print(classification_report(
        all_true_ha, all_pred_ha,
        target_names=['Human', 'AI'], digits=4
    ))
    print("Confusion matrix (H/A):")
    print(confusion_matrix(all_true_ha, all_pred_ha))

    print("\n=== RoBERTa Text-only Real vs Fake report ===")
    print(classification_report(
        all_true_rf, all_pred_rf,
        target_names=['Real', 'Fake'], digits=4
    ))
    print("Confusion matrix (R/F):")
    print(confusion_matrix(all_true_rf, all_pred_rf))

    print("\n=== RoBERTa Text-only 4-class report ===")
    print(classification_report(
        all_true_4, all_pred_4,
        target_names=target4, digits=4
    ))
    print("Confusion matrix (4-way):")
    print(confusion_matrix(all_true_4, all_pred_4))

    yt = np.array(all_true_4)
    proba = np.vstack(prob_chunks) if prob_chunks else np.zeros((0, 4), dtype=np.float64)
    aucm = auc_scores_from_proba(yt, proba)
    print(
        f"\n[RoBERTa Text-only] ROC-AUC: 4-class (macro-OVR)={aucm['auc_4_ovr_macro']:.4f} | "
        f"H/A={aucm['auc_ha']:.4f} | R/F={aucm['auc_rf']:.4f}"
    )
    print(
        f"\n[RoBERTa Text-only] "
        f"H/A → Acc={acc_ha:.4f}, F1={f1_ha:.4f} | "
        f"R/F → Acc={acc_rf:.4f}, F1={f1_rf:.4f} | "
        f"4-way → Acc={acc_4:.4f}, F1={f1_4:.4f} | "
        f"AUC(4c)={aucm['auc_4_ovr_macro']:.4f} AUC(H/A)={aucm['auc_ha']:.4f} "
        f"AUC(R/F)={aucm['auc_rf']:.4f}"
    )

    return {
        "acc_ha": acc_ha,
        "f1_ha":  f1_ha,
        "acc_rf": acc_rf,
        "f1_rf":  f1_rf,
        "acc_4":  acc_4,
        "f1_4":   f1_4,
        "auc_4_ovr_macro": aucm["auc_4_ovr_macro"],
        "auc_ha": aucm["auc_ha"],
        "auc_rf": aucm["auc_rf"],
    }


# -------------------------
# 8. Validation용 (4-way macro F1 기준)
# -------------------------
def _validate_roberta_four_class(model, val_loader):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids  = batch["input_ids"].to(DEVICE, non_blocking=True)
            attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
            lab4 = batch["label4"].to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=USE_AMP):
                logits4 = model(ids, attn)

            p4 = torch.argmax(logits4, dim=1).cpu().numpy()
            all_pred.extend(p4.tolist())
            all_true.extend(lab4.cpu().numpy().tolist())

            del ids, attn, lab4, logits4

    return f1_score(all_true, all_pred, average="macro")


# -------------------------
# 9. RoBERTa 4-class 학습 함수
# -------------------------
def train_roberta_4c(model, train_loader, val_loader, epochs=EPOCHS):
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scaler = GradScaler("cuda", enabled=USE_AMP) if USE_AMP else None
    ce = nn.CrossEntropyLoss()

    best_val  = -1.0
    best_state = None
    early = EarlyStopping(
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        mode='max'
    )

    step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch in tqdm(train_loader, desc=f"[RoBERTa-4C] Epoch {epoch+1}"):
            ids  = batch["input_ids"].to(DEVICE, non_blocking=True)
            attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
            lab4 = batch["label4"].to(DEVICE, non_blocking=True)

            ctx = autocast(device_type="cuda", enabled=USE_AMP) if USE_AMP else nullcontext()
            with ctx:
                logits4 = model(ids, attn)
                loss = ce(logits4, lab4)
                loss_scaled = loss / ACCUM_STEPS

            if USE_AMP:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            step += 1

            if (step % ACCUM_STEPS) == 0:
                if USE_AMP:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            del ids, attn, lab4, logits4

        avg_train_loss = total_loss / max(1, len(train_loader))
        print(f"[RoBERTa-4C] avg_train_loss={avg_train_loss:.4f}")

        # Validation
        val_f1 = _validate_roberta_four_class(model, val_loader)
        print(f"[RoBERTa-4C] val_macro_f1(4-way)={val_f1:.4f}")

        if val_f1 > best_val:
            best_val   = val_f1
            best_state = copy.deepcopy(model.state_dict())
            print("[RoBERTa-4C] Saved best model state (in-memory).")

        if early.step(val_f1):
            print(
                f"[RoBERTa-4C] [EarlyStop] No improvement for {EARLY_STOP_PATIENCE} epochs "
                f"(min_delta={EARLY_STOP_MIN_DELTA}). Stopping."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -------------------------
# 10. Tee: 콘솔 + 파일 동시 저장용
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
            if hasattr(s, 'flush'):
                s.flush()


# -------------------------
# 11. K-Fold Runner
# -------------------------
def kfold_roberta_4c_with_reports(
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
    print(f"[KFold-RoBERTa-4C] Split: TrainVal={len(trainval_df)}, Test={len(test_df)}")

    test_ds = TextOnlyFourClassDataset(
        test_df, tokenizer=tokenizer, max_len=MAX_LEN
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)

    all_report = io.StringIO()
    all_report.write(f"###### RoBERTa Text-only 4C (k={k}) ######\n")

    metrics_list = []
    y = trainval_df["label"].values

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(np.zeros(len(trainval_df)), y), start=1
    ):
        print(f"\n========== [Fold {fold}/{k} - RoBERTa-4C] ==========")
        tr_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
        va_df = trainval_df.iloc[va_idx].reset_index(drop=True)

        train_ds = TextOnlyFourClassDataset(
            tr_df, tokenizer=tokenizer, max_len=MAX_LEN
        )
        val_ds = TextOnlyFourClassDataset(
            va_df, tokenizer=tokenizer, max_len=MAX_LEN
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH,
            shuffle=False,
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
            print(f"[Fold {fold}] RoBERTa Text-only 4-class")
            print("=" * 70)
            model = RobertaTextOnly4C().to(DEVICE)
            model = train_roberta_4c(model, train_loader, val_loader)
            res = evaluate_4class_roberta(model, test_loader)

            print("\n" + "=" * 70)
            print(f"[Fold {fold}] SUMMARY (RoBERTa Text-only, Acc/F1 on Test)")
            print("=" * 70)
            print(
                f"RoBERTa-4C: "
                f"H/A Acc={res['acc_ha']:.4f}, F1={res['f1_ha']:.4f} | "
                f"R/F Acc={res['acc_rf']:.4f}, F1={res['f1_rf']:.4f} | "
                f"4-way Acc={res['acc_4']:.4f}, F1={res['f1_4']:.4f} | "
                f"AUC(4c)={res['auc_4_ovr_macro']:.4f} AUC(H/A)={res['auc_ha']:.4f} "
                f"AUC(R/F)={res['auc_rf']:.4f}"
            )
            print("=" * 70)

        fold_text = buffer.getvalue()
        all_report.write(
            f"\n\n{'#'*90}\n### FOLD {fold}/{k} REPORT (RoBERTa-4C)\n{'#'*90}\n"
        )
        all_report.write(fold_text)
        all_report.flush()

        metrics_list.append(res)
        gc.collect()

    def _mean_metric(dict_list, key):
        vals = [d[key] for d in dict_list if d.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    mean_acc_ha = _mean_metric(metrics_list, "acc_ha")
    mean_f1_ha  = _mean_metric(metrics_list, "f1_ha")
    mean_acc_rf = _mean_metric(metrics_list, "acc_rf")
    mean_f1_rf  = _mean_metric(metrics_list, "f1_rf")
    mean_acc_4  = _mean_metric(metrics_list, "acc_4")
    mean_f1_4   = _mean_metric(metrics_list, "f1_4")
    mean_auc_4  = _mean_metric(metrics_list, "auc_4_ovr_macro")
    mean_auc_ha = _mean_metric(metrics_list, "auc_ha")
    mean_auc_rf = _mean_metric(metrics_list, "auc_rf")

    all_report.write("\n\n" + "#"*90 + "\n")
    all_report.write(
        "### GLOBAL SUMMARY (RoBERTa text-only, Mean over {} folds)\n".format(k)
    )
    all_report.write("#"*90 + "\n")
    all_report.write(
        f"RoBERTa-4C (text-only): "
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
        f"RoBERTa-4C(text-only)\t"
        f"{mean_acc_ha:.4f}\t{mean_f1_ha:.4f}\t"
        f"{mean_acc_rf:.4f}\t{mean_f1_rf:.4f}\t"
        f"{mean_acc_4:.4f}\t{mean_f1_4:.4f}\t"
        f"{mean_auc_4:.4f}\t{mean_auc_ha:.4f}\t{mean_auc_rf:.4f}\n"
    )

    output_path = os.path.join(
        RESULT_DIR, f"kfold_roberta_textonly_4c_{timestamp}.txt"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_report.getvalue())
    print(f"\n✅ 모든 fold 리포트 및 요약을 한 파일에 저장 완료: {output_path}\n")


# -------------------------
# 12. 단일 split
# -------------------------
def get_single_split_loaders_roberta(full_df):
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

    train_ds = TextOnlyFourClassDataset(train_data, tokenizer=tokenizer, max_len=MAX_LEN)
    val_ds   = TextOnlyFourClassDataset(val_data,   tokenizer=tokenizer, max_len=MAX_LEN)
    test_ds  = TextOnlyFourClassDataset(test_data,  tokenizer=tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=False,
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


def print_memory_usage():
    if torch.cuda.is_available():
        print(
            f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / "
            f"{torch.cuda.memory_reserved()/1024**3:.2f}GB"
        )


# -------------------------
# 13. Main
# -------------------------
if __name__ == "__main__":
    print("Initial memory usage:")
    print_memory_usage()
    print(
        f"RoBERTa text-only Baseline | "
        f"KFold={USE_KFOLD}, BATCH={BATCH}, LR={LR}, "
        f"EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE}, "
        f"WeightedRandomSampler(alpha={SAMPLER_ALPHA}), "
        f"ACCUM_STEPS={ACCUM_STEPS}, USE_AMP={USE_AMP}, EPOCHS={EPOCHS}, MAX_LEN={MAX_LEN}, SEED={SEED}"
    )
    print(f"Dataset: HARFM.csv (final_headline, 4_way_label)\n")

    if USE_KFOLD:
        kfold_roberta_4c_with_reports(
            data,
            k=KFOLD_SPLITS,
            test_size=KFOLD_TEST_SIZE,
            shuffle=KFOLD_SHUFFLE,
            random_state=KFOLD_RANDOM_STATE,
        )
    else:
        train_loader, val_loader, test_loader = get_single_split_loaders_roberta(data)
        model = RobertaTextOnly4C().to(DEVICE)
        model = train_roberta_4c(model, train_loader, val_loader)
        _ = evaluate_4class_roberta(model, test_loader)
