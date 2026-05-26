"""
BERT text-only baseline for HARFM (4-class: HR, HF, AR, AF)
- 데이터: HARFM.csv (text=final_headline, image=image_path, label=4_way_label)
"""

import os, io, sys, warnings, copy, gc
from contextlib import redirect_stdout
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from datetime import datetime

from torch.cuda.amp import autocast, GradScaler

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
warnings.filterwarnings("ignore")


# -------------------------
# 0. 재현성 세팅
# -------------------------
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FND_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

# USER 요청: 데이터셋 경로는 아래를 기본값으로 사용
CSV_PATH = os.environ.get(
    "HARFM_CSV_PATH",
    os.path.join(_FND_ROOT, "HARFM.csv"),
)

MODEL_NAME = os.environ.get("BERT_MODEL_NAME", "bert-base-uncased")

# 학습 관련
BATCH = 16
ACCUM_STEPS = 1
EPOCHS = 10
LR = 2e-5
NUM_WORKERS = 4
USE_AMP = False      

MAX_LEN = 128        # 토크나이저 max_length

# Early Stopping
EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_DELTA = 1e-4
SAMPLER_ALPHA = 0.5

# K-Fold 설정
USE_KFOLD = True
KFOLD_SPLITS = 5
KFOLD_TEST_SIZE = 0.2
KFOLD_SHUFFLE = True
KFOLD_RANDOM_STATE = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2


# -------------------------
# 2. 데이터 로드
# -------------------------
print("Loading CSV for BERT text-only baseline...")
data = pd.read_csv(CSV_PATH, low_memory=False)

required_cols = {"final_headline", "image_path", "4_way_label"}
missing = required_cols - set(data.columns)
if missing:
    raise ValueError(f"HARFM CSV에 다음 컬럼이 없습니다: {missing}")

label_map = {"HR": 0, "HF": 1, "AR": 2, "AF": 3}

def _strip_quotes(val):
    if not isinstance(val, str):
        return val
    val = val.strip()
    while len(val) >= 2 and (
        (val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")
    ):
        val = val[1:-1].strip()
    return val

# 원본 컬럼명은 유지하면서, 내부 학습용 숫자 라벨만 별도 생성
data = data[list(required_cols)].copy()
data["final_headline"] = data["final_headline"].fillna("").astype(str).apply(_strip_quotes)
data["image_path"] = data["image_path"].fillna("").astype(str).str.strip()
data["image_path"] = data["image_path"].apply(
    lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(_FND_ROOT, p))
)
data["label4"] = data["4_way_label"].map(label_map)
data = data[data["label4"].notna()].astype({"label4": int}).reset_index(drop=True)
before_mm = len(data)
data = data[
    data["image_path"].apply(lambda p: isinstance(p, str) and p != "" and os.path.isfile(p))
].reset_index(drop=True)

print(f"Dataset loaded: {len(data)} rows")
print(f"[전처리] 멀티모달 필터(이미지 파일 유효): {before_mm} -> {len(data)}")
print("Label counts (4_way_label):", data["4_way_label"].value_counts().sort_index().to_dict())
print("Label counts (label4: 0=HR,1=HF,2=AR,3=AF):", data["label4"].value_counts().sort_index().to_dict())


# -------------------------
# 3. Tokenizer
# -------------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


# -------------------------
# 4. Dataset (Text-only 4-class)
# -------------------------
class TextOnlyFourClassDataset(Dataset):
    """
    - text: final_headline
    - label: 4_way_label -> 0=HR, 1=HF, 2=AR, 3=AF
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

        label4 = int(row["label4"])

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
# 5. BERT Text-only 4-class 모델
# -------------------------
class BertTextOnly4C(nn.Module):
    """
    BERT-base + 4-class head
    """

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        hidden = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, 4)

    def forward(self, input_ids, attention_mask):
        out = self.bert(
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


# -------------------------
# 7. 평가 함수 (H/A, R/F, 4-way Acc & F1 모두)
# -------------------------
def evaluate_4class_bert(model, loader):
    model.eval()
    all_true_4, all_pred_4 = [], []
    all_true_ha, all_pred_ha = [], []
    all_true_rf, all_pred_rf = [], []

    target4 = ['Human Real', 'Human Fake', 'AI Real', 'AI Fake']

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation (BERT text-only 4C)"):
            ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
            lab4 = batch["label4"].to(DEVICE, non_blocking=True)

            with autocast(enabled=USE_AMP):
                logits4 = model(ids, attn)

            # 4-way prediction
            p4 = torch.argmax(logits4, dim=1)       # (B,)
            p4_np = p4.cpu().numpy()
            y4_np = lab4.cpu().numpy()

            # 4-way 저장
            all_pred_4.extend(p4_np.tolist())
            all_true_4.extend(y4_np.tolist())

            # ---- 축으로 매핑 ----
            # Human / AI (0=Human, 1=AI)
            y_ha = y4_np // 2
            p_ha = p4_np // 2

            # Real / Fake (0=Real, 1=Fake)
            y_rf = y4_np % 2
            p_rf = p4_np % 2

            all_true_ha.extend(y_ha.tolist())
            all_pred_ha.extend(p_ha.tolist())
            all_true_rf.extend(y_rf.tolist())
            all_pred_rf.extend(p_rf.tolist())

            del ids, attn, lab4, logits4, p4

    # ----- Accuracy / F1 계산 -----
    acc_4 = accuracy_score(all_true_4, all_pred_4)
    acc_ha = accuracy_score(all_true_ha, all_pred_ha)
    acc_rf = accuracy_score(all_true_rf, all_pred_rf)

    f1_4 = f1_score(all_true_4, all_pred_4, average="macro")
    f1_ha = f1_score(all_true_ha, all_pred_ha, average="macro")
    f1_rf = f1_score(all_true_rf, all_pred_rf, average="macro")

    # ----- 리포트 출력 -----
    print("\n=== BERT Text-only Human vs AI report ===")
    print(classification_report(
        all_true_ha, all_pred_ha,
        target_names=['Human', 'AI'], digits=4
    ))
    print("Confusion matrix (H/A):")
    print(confusion_matrix(all_true_ha, all_pred_ha))

    print("\n=== BERT Text-only Real vs Fake report ===")
    print(classification_report(
        all_true_rf, all_pred_rf,
        target_names=['Real', 'Fake'], digits=4
    ))
    print("Confusion matrix (R/F):")
    print(confusion_matrix(all_true_rf, all_pred_rf))

    print("\n=== BERT Text-only 4-class report ===")
    print(classification_report(
        all_true_4, all_pred_4,
        target_names=target4, digits=4
    ))
    print("Confusion matrix (4-way):")
    print(confusion_matrix(all_true_4, all_pred_4))

    print(
        f"\n[BERT Text-only] "
        f"H/A → Acc={acc_ha:.4f}, F1={f1_ha:.4f} | "
        f"R/F → Acc={acc_rf:.4f}, F1={f1_rf:.4f} | "
        f"4-way → Acc={acc_4:.4f}, F1={f1_4:.4f}"
    )

    return {
        "acc_ha": acc_ha,
        "f1_ha": f1_ha,
        "acc_rf": acc_rf,
        "f1_rf": f1_rf,
        "acc_4": acc_4,
        "f1_4": f1_4,
    }


# -------------------------
# 8. Validation용 (4-way F1 기준)
# -------------------------
def _validate_bert_four_class(model, val_loader):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
            lab4 = batch["label4"].to(DEVICE, non_blocking=True)

            with autocast(enabled=USE_AMP):
                logits4 = model(ids, attn)

            p4 = torch.argmax(logits4, dim=1).cpu().numpy()
            all_pred.extend(p4.tolist())
            all_true.extend(lab4.cpu().numpy().tolist())

            del ids, attn, lab4, logits4

    return f1_score(all_true, all_pred, average="macro")


# -------------------------
# 9. BERT 4-class 학습 함수
# -------------------------
def train_bert_4c(model, train_loader, val_loader, epochs=EPOCHS):
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scaler = GradScaler(enabled=USE_AMP)
    ce = nn.CrossEntropyLoss()

    best_val = -1.0
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

        for batch in tqdm(train_loader, desc=f"[BERT-4C] Epoch {epoch+1}"):
            ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attn = batch["attention_mask"].to(DEVICE, non_blocking=True)
            lab4 = batch["label4"].to(DEVICE, non_blocking=True)

            with autocast(enabled=USE_AMP):
                logits4 = model(ids, attn)
                loss = ce(logits4, lab4)
                loss_scaled = loss / ACCUM_STEPS

            scaler.scale(loss_scaled).backward()
            step += 1

            if (step % ACCUM_STEPS) == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())

            del ids, attn, lab4, logits4

        avg_train_loss = total_loss / max(1, len(train_loader))
        print(f"[BERT-4C] avg_train_loss={avg_train_loss:.4f}")

        # Validation
        val_f1 = _validate_bert_four_class(model, val_loader)
        print(f"[BERT-4C] val_macro_f1(4-way)={val_f1:.4f}")

        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(model.state_dict())
            print("[BERT-4C] Saved best model state (in-memory).")

        if early.step(val_f1):
            print(
                f"[BERT-4C] [EarlyStop] No improvement for {EARLY_STOP_PATIENCE} epochs "
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
# 11. K-Fold Runner (BERT text-only 4C)
# -------------------------
def kfold_bert_4c_with_reports(
    full_df,
    k=KFOLD_SPLITS,
    test_size=KFOLD_TEST_SIZE,
    shuffle=KFOLD_SHUFFLE,
    random_state=KFOLD_RANDOM_STATE,
):
    # Train/Val vs Test 분할
    trainval_df, test_df = train_test_split(
        full_df,
        test_size=test_size,
        stratify=full_df["label4"],
        random_state=random_state,
    )
    print(f"[KFold-BERT-4C] Split: TrainVal={len(trainval_df)}, Test={len(test_df)}")

    # Test loader
    test_ds = TextOnlyFourClassDataset(
        test_df,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    skf = StratifiedKFold(
        n_splits=k,
        shuffle=shuffle,
        random_state=random_state
    )

    all_report = io.StringIO()
    all_report.write(f"###### BERT Text-only 4C (k={k}) ######\n")

    metrics_list = []

    y = trainval_df["label4"].values
    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(np.zeros(len(trainval_df)), y),
        start=1
    ):
        print(f"\n========== [Fold {fold}/{k} - BERT-4C] ==========")
        tr_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
        va_df = trainval_df.iloc[va_idx].reset_index(drop=True)

        train_ds = TextOnlyFourClassDataset(
            tr_df,
            tokenizer=tokenizer,
            max_len=MAX_LEN
        )
        val_ds = TextOnlyFourClassDataset(
            va_df,
            tokenizer=tokenizer,
            max_len=MAX_LEN
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH,
            shuffle=False,
            sampler=make_weighted_sampler(tr_df["label4"].tolist()),
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        buffer = io.StringIO()
        tee = Tee(sys.stdout, buffer)
        with redirect_stdout(tee):
            print("\n" + "=" * 70)
            print(f"[Fold {fold}] BERT Text-only 4-class")
            print("=" * 70)
            model = BertTextOnly4C().to(DEVICE)
            model = train_bert_4c(model, train_loader, val_loader)
            res = evaluate_4class_bert(model, test_loader)

            print("\n" + "=" * 70)
            print(f"[Fold {fold}] 📊 SUMMARY (BERT Text-only, Acc/F1 on Test)")
            print("=" * 70)
            print(
                f"BERT-4C: "
                f"H/A Acc={res['acc_ha']:.4f}, F1={res['f1_ha']:.4f} | "
                f"R/F Acc={res['acc_rf']:.4f}, F1={res['f1_rf']:.4f} | "
                f"4-way Acc={res['acc_4']:.4f}, F1={res['f1_4']:.4f}"
            )
            print("=" * 70)

        fold_text = buffer.getvalue()
        all_report.write(
            f"\n\n{'#' * 90}\n### FOLD {fold}/{k} REPORT (BERT-4C)\n{'#' * 90}\n"
        )
        all_report.write(fold_text)
        all_report.flush()

        metrics_list.append(res)
        gc.collect()

    def _mean_metric(dict_list, key):
        vals = [d[key] for d in dict_list if d.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    mean_acc_ha = _mean_metric(metrics_list, "acc_ha")
    mean_f1_ha = _mean_metric(metrics_list, "f1_ha")
    mean_acc_rf = _mean_metric(metrics_list, "acc_rf")
    mean_f1_rf = _mean_metric(metrics_list, "f1_rf")
    mean_acc_4 = _mean_metric(metrics_list, "acc_4")
    mean_f1_4 = _mean_metric(metrics_list, "f1_4")

    all_report.write("\n\n" + "#" * 90 + "\n")
    all_report.write(
        "### GLOBAL SUMMARY (BERT text-only, Mean over {} folds)\n".format(k)
    )
    all_report.write("#" * 90 + "\n")
    all_report.write(
        f"BERT-4C (text-only): "
        f"mean H/A Acc={mean_acc_ha:.4f}, F1={mean_f1_ha:.4f} | "
        f"mean R/F Acc={mean_acc_rf:.4f}, F1={mean_f1_rf:.4f} | "
        f"mean 4-way Acc={mean_acc_4:.4f}, F1={mean_f1_4:.4f}\n"
    )

    # 논문 테이블용으로 복사해서 쓰기 쉽게 한 줄 요약도 같이 저장
    all_report.write("\n[TABLE_SUMMARY]\n")
    all_report.write(
        "Variant\tHA_Acc\tHA_F1\tRF_Acc\tRF_F1\t4C_Acc\t4C_F1\n"
    )
    all_report.write(
        f"BERT-4C(text-only)\t"
        f"{mean_acc_ha:.4f}\t{mean_f1_ha:.4f}\t"
        f"{mean_acc_rf:.4f}\t{mean_f1_rf:.4f}\t"
        f"{mean_acc_4:.4f}\t{mean_f1_4:.4f}\n"
    )

    output_path = os.path.join(
        RESULT_DIR, f"kfold_bert_textonly_4c_{timestamp}.txt"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_report.getvalue())
    print(f"\n✅ 모든 fold 리포트 및 요약을 한 파일에 저장 완료: {output_path}\n")


# -------------------------
# 12. 단일 split (KFold 사용 안 할 때)
# -------------------------
def get_single_split_loaders_bert(full_df):
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

    train_ds = TextOnlyFourClassDataset(
        train_data,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
    )
    val_ds = TextOnlyFourClassDataset(
        val_data,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
    )
    test_ds = TextOnlyFourClassDataset(
        test_data,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=False,
        sampler=make_weighted_sampler(train_data["label4"].tolist()),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
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
        f"BERT text-only Baseline | "
        f"KFold={USE_KFOLD}, BATCH={BATCH}, LR={LR}, "
        f"EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE}, "
        f"WeightedRandomSampler(alpha={SAMPLER_ALPHA}), "
        f"ACCUM_STEPS={ACCUM_STEPS}, USE_AMP={USE_AMP}, EPOCHS={EPOCHS}, MAX_LEN={MAX_LEN}, SEED={SEED}"
    )

    if USE_KFOLD:
        kfold_bert_4c_with_reports(
            data,
            k=KFOLD_SPLITS,
            test_size=KFOLD_TEST_SIZE,
            shuffle=KFOLD_SHUFFLE,
            random_state=KFOLD_RANDOM_STATE,
        )
    else:
        train_loader, val_loader, test_loader = get_single_split_loaders_bert(data)
        model = BertTextOnly4C().to(DEVICE)
        model = train_bert_4c(model, train_loader, val_loader)
        res = evaluate_4class_bert(model, test_loader)
