"""
SpotFake+ for MiRAGe-News (Binary: Real vs AI-Fake)
  (XLNet text encoder + VGG16 image encoder + MLP fusion)
- Benchmark protocol:
  - default: 5-fold on official train split, evaluate all test* splits
  - --no_kfold: train on official train, validate on official validation,
                evaluate all test* splits
"""

from __future__ import annotations

import argparse
import copy
import gc
import io
import os
import random
import warnings
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from transformers import XLNetModel, XLNetTokenizer

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_DATASET_ID = "anson-huang/mirage-news"

TEXT_MODEL = "xlnet-base-cased"
MAX_LEN = 128
IMG_SIZE = 224

BATCH = 16
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_DELTA = 1e-4

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
CHECKPOINT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoint")
)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LABEL_NAMES = ["Real", "AI-Fake"]

tokenizer = XLNetTokenizer.from_pretrained(TEXT_MODEL)
image_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MirageSpotFakeDataset(Dataset):
    def __init__(self, hf_split, max_len=MAX_LEN):
        self.hf_split = hf_split
        self.max_len = max_len
        cols = hf_split.column_names

        self.text_col = next((c for c in ("caption", "text", "headline") if c in cols), None)
        self.image_col = next((c for c in ("image", "img") if c in cols), None)
        self.label_col = next((c for c in ("label", "labels", "fake") if c in cols), None)
        if self.text_col is None or self.image_col is None or self.label_col is None:
            raise ValueError(f"Required columns not found. columns={cols}")

        self.labels = [int(x) for x in hf_split[self.label_col]]

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        row = self.hf_split[idx]
        text = str(row[self.text_col]).strip()
        label = int(row[self.label_col])

        img = row[self.image_col]
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        elif isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        else:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127))

        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": image_transform(img),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_spotfake(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


class SpotFakePlusBinary(nn.Module):
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
        )  # -> 4096

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
            nn.Linear(100, 2),
        )

    def forward(self, input_ids, attention_mask, image):
        out = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = out.last_hidden_state[:, -1, :]  # (B, 768)
        img_feat = self.vgg_feat(image)  # (B, 4096)
        h = torch.cat([text_feat, img_feat], dim=1)
        return self.fusion(h)


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
        improved = (
            (metric_value - self.best) > self.min_delta
            if self.mode == "max"
            else (self.best - metric_value) > self.min_delta
        )
        if improved:
            self.best = metric_value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def resolve_test_splits(ds_dict) -> List[str]:
    keys = list(ds_dict.keys())
    preferred = [
        "test_midjourneyv5",
        "test_midjourney_v5",
        "test_dalle3",
        "test_dalle_3",
        "test_sdxl",
        "test_bbc",
        "test_cnn",
    ]
    ordered = [k for k in preferred if k in keys]
    for k in sorted(keys):
        if k.startswith("test") and k not in ordered:
            ordered.append(k)
    return ordered


def build_loader(ds, batch_size, num_workers, shuffle, drop_last=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_spotfake,
        drop_last=drop_last,
    )


def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    ys, ps = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="Train" if train else "Eval"):
            ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            img = batch["image"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            logits = model(ids, attn, img)
            loss = ce(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            ys.extend(y.detach().cpu().numpy().tolist())
            ps.extend(logits.argmax(1).detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(ys))
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="macro", zero_division=0)
    return {"loss": avg_loss, "acc": acc, "f1": f1}


def train_one_run(train_ds, val_ds, args):
    train_loader = build_loader(
        train_ds, args.batch_size, args.num_workers, shuffle=True, drop_last=True
    )
    val_loader = build_loader(
        val_ds, args.batch_size, args.num_workers, shuffle=False, drop_last=False
    )

    model = SpotFakePlusBinary().to(DEVICE)
    for p in model.xlnet.parameters():
        p.requires_grad = False
    for p in model.vgg_feat.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    early = EarlyStopping(args.early_stop_patience, args.early_stop_min_delta, mode="max")

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer=optimizer)
        va = run_epoch(model, val_loader, optimizer=None)
        print(
            f"[Epoch {epoch}] train_loss={tr['loss']:.4f} train_f1={tr['f1']:.4f} | "
            f"val_loss={va['loss']:.4f} val_f1={va['f1']:.4f}"
        )
        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            best_state = copy.deepcopy(model.state_dict())
        if early.step(va["f1"]):
            print(f"[SPOTFAKE-MIRAGE] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


def collect_predictions(model, loader):
    model.eval()
    ys, ps, scores = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collect"):
            ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            img = batch["image"].to(DEVICE)
            y = batch["label"]

            logits = model(ids, attn, img)
            prob_fake = torch.softmax(logits, dim=-1)[:, 1]
            pred = logits.argmax(1)

            ys.extend(y.numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())
            scores.extend(prob_fake.cpu().numpy().tolist())
    return np.array(ys), np.array(ps), np.array(scores)


def evaluate_binary(yt, yp, scores, split_name):
    report = classification_report(yt, yp, target_names=LABEL_NAMES, digits=4, zero_division=0)
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, average="macro", zero_division=0)
    f1_ai = f1_score(yt, yp, average="binary", pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(yt, scores)
    except ValueError:
        auc = float("nan")

    block = (
        f"\n=== {split_name} ===\n"
        f"{report}\n"
        f"[{split_name}] Acc={acc:.4f} Macro-F1={f1:.4f} AI-Fake-F1={f1_ai:.4f} AUC={auc:.4f}\n"
    )
    return block, {"acc": acc, "f1": f1, "f1_ai": f1_ai, "auc": auc}


def evaluate_all_test_splits(model, ds, test_splits, args):
    blocks = []
    all_metrics = {}
    for split_name in test_splits:
        test_ds = MirageSpotFakeDataset(ds[split_name], max_len=MAX_LEN)
        test_loader = build_loader(test_ds, args.batch_size, args.num_workers, shuffle=False)
        yt, yp, scores = collect_predictions(model, test_loader)
        block, metrics = evaluate_binary(yt, yp, scores, split_name)
        blocks.append(block)
        all_metrics[split_name] = metrics
    return blocks, all_metrics


def train_and_test(args):
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[SPOTFAKE-MIRAGE] loading dataset: {args.hf_dataset_id}")
    ds = load_dataset(args.hf_dataset_id)
    print(f"[SPOTFAKE-MIRAGE] splits: {list(ds.keys())}")
    test_splits = resolve_test_splits(ds)
    if not test_splits:
        raise RuntimeError("No test split found in dataset.")

    report_io = io.StringIO()
    mode_name = "KFold" if not args.no_kfold else "Official Split"
    report_io.write(f"###### SPOTFAKE-MIRAGE ({mode_name}) ######\n")
    report_io.write(f"test_splits={test_splits}\n")
    report_io.write("\n===== TEST RESULTS =====\n")

    if args.no_kfold:
        train_ds = MirageSpotFakeDataset(ds["train"], max_len=MAX_LEN)
        val_ds = MirageSpotFakeDataset(ds["validation"], max_len=MAX_LEN)
        model, best_f1 = train_one_run(train_ds, val_ds, args)
        report_io.write(f"best_val_macro_f1={best_f1:.4f}\n")

        blocks, all_metrics = evaluate_all_test_splits(model, ds, test_splits, args)
        for b in blocks:
            print(b)
            report_io.write(b)

        report_io.write("\n===== SUMMARY (avg over test splits) =====\n")
        for k in ("acc", "f1", "f1_ai", "auc"):
            vals = [all_metrics[s][k] for s in all_metrics]
            line = f"{k}: {float(np.mean(vals)):.4f} (min={float(np.min(vals)):.4f}, max={float(np.max(vals)):.4f})"
            print(line)
            report_io.write(line + "\n")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"spotfake_mirage_{timestamp}.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "metrics": all_metrics,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
            ckpt_path,
        )
    else:
        base_train = ds["train"]
        probe_ds = MirageSpotFakeDataset(base_train, max_len=MAX_LEN)
        labels = np.array(probe_ds.labels)
        skf = StratifiedKFold(n_splits=args.kfold_splits, shuffle=True, random_state=args.seed)

        fold_results = []
        report_io.write(f"kfold_splits={args.kfold_splits}\n")
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
            print(f"\n[SPOTFAKE-MIRAGE] ===== Fold {fold_idx}/{args.kfold_splits} =====")
            report_io.write(f"\n\n{'#' * 70}\n### Fold {fold_idx}/{args.kfold_splits}\n{'#' * 70}\n")

            train_ds = MirageSpotFakeDataset(base_train.select(tr_idx.tolist()), max_len=MAX_LEN)
            val_ds = MirageSpotFakeDataset(base_train.select(va_idx.tolist()), max_len=MAX_LEN)
            model, best_f1 = train_one_run(train_ds, val_ds, args)
            report_io.write(f"best_val_macro_f1={best_f1:.4f}\n")

            blocks, fold_metrics = evaluate_all_test_splits(model, ds, test_splits, args)
            for b in blocks:
                print(b)
                report_io.write(b)
            fold_results.append(fold_metrics)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        report_io.write("\n\n===== KFOLD GLOBAL SUMMARY =====\n")
        global_metrics: Dict[str, Dict[str, float]] = {}
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
            report_io.write(line + "\n")

        report_io.write("\n===== SUMMARY (avg over test splits, then folds) =====\n")
        for k in ("acc", "f1", "f1_ai", "auc"):
            vals = [global_metrics[s][k] for s in test_splits]
            line = f"{k}: {float(np.mean(vals)):.4f}"
            print(line)
            report_io.write(line + "\n")

        ckpt_path = os.path.join(
            CHECKPOINT_DIR, f"spotfake_mirage_kfold{args.kfold_splits}_{timestamp}.pt"
        )
        torch.save(
            {
                "config": vars(args),
                "fold_metrics": fold_results,
                "global_metrics": global_metrics,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
            ckpt_path,
        )

    report_path = os.path.join(RESULT_DIR, f"spotfake_mirage_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_io.getvalue())

    print(f"[SPOTFAKE-MIRAGE] checkpoint: {ckpt_path}")
    print(f"[SPOTFAKE-MIRAGE] report: {report_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="SpotFake+ for MiRAGe-News (binary)")
    ap.add_argument("--hf_dataset_id", default=HF_DATASET_ID)
    ap.add_argument("--batch_size", type=int, default=BATCH)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE)
    ap.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    ap.add_argument("--no_kfold", action="store_true")
    ap.add_argument("--kfold_splits", type=int, default=5)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"SPOTFAKE-MIRAGE | KFold={not args.no_kfold} | device={DEVICE}")
    train_and_test(args)
