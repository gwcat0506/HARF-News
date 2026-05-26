"""
FND-CLIP for MiRAGe-News (Binary: Real vs AI-Fake)
- Backbone: BERT(text) + ResNet101(image) + CLIP(text,image)
"""

from __future__ import annotations

import argparse
import copy
import io
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPProcessor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_DATASET_ID = "anson-huang/mirage-news"

BERT_NAME = "bert-base-uncased"
CLIP_NAME = "openai/clip-vit-base-patch32"
MAX_LEN_BERT = 128
MAX_LEN_CLIP = 77
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

LABEL_NAMES = ["Real", "AI-Fake"]  # 0, 1


bert_tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME)
resnet_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MirageFNDCLIPDataset(Dataset):
    def __init__(self, hf_split, max_len_bert: int = MAX_LEN_BERT):
        self.hf_split = hf_split
        self.max_len_bert = max_len_bert
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

        bert_enc = bert_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len_bert,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        image_resnet = resnet_transform(img)
        clip_enc = clip_processor(
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
            "label": torch.tensor(label, dtype=torch.long),
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
        "label": torch.stack([b["label"] for b in batch]),
    }


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
        return self.c_specific(torch.cat((text, image), 1))


class FNDCLIPMultiModalBinary(nn.Module):
    def __init__(self, num_classes=2, feature_dim=64 * 3, h_dim=64):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(13, 1) * 0.01)
        self.senet = nn.Sequential(nn.Linear(3, 3), nn.GELU(), nn.Linear(3, 3))
        self.sigmoid = nn.Sigmoid()
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.avepooling = nn.AvgPool1d(64, stride=1)
        self.maxpooling = nn.MaxPool1d(64, stride=1)

        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.resnet101 = resnet

        self.uni_repre = UnimodalDetection(text_in=1280, image_in=1512)
        self.cross_module = CrossModule(corre_in=512 + 512)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_classes),
        )

    def forward(self, bert_hidden_states, image_resnet, text_clip, image_clip):
        batch_size = image_resnet.shape[0]
        image_raw = self.resnet101(image_resnet)

        ht_cls = torch.stack(bert_hidden_states, dim=0)[:, :, 0, :]
        ht_cls = ht_cls.view(13, batch_size, 1, 768)

        attn = torch.sum(ht_cls * self.weights.view(13, 1, 1, 1), dim=[1, 3])
        attn = F.softmax(attn.view(-1), dim=0)
        text_raw = torch.sum(ht_cls * attn.view(13, 1, 1, 1), dim=[0, 2]).squeeze(1)

        text_enc = torch.cat([text_raw, text_clip], 1)
        image_enc = torch.cat([image_raw, image_clip], 1)
        text_prime, image_prime = self.uni_repre(text_enc, image_enc)

        correlation = self.cross_module(text_clip, image_clip)
        sim = (text_clip * image_clip).sum(1) / (
            text_clip.norm(dim=1) * image_clip.norm(dim=1) + 1e-8
        )
        sim = sim * self.w + self.b
        correlation = correlation * sim.unsqueeze(1)

        final_feature = torch.stack([text_prime, image_prime, correlation], 1)
        s1 = self.senet(self.avepooling(final_feature).view(batch_size, -1))
        s2 = self.senet(self.maxpooling(final_feature).view(batch_size, -1))
        s = self.sigmoid(s1 + s2).view(batch_size, 3, 1)
        pooled = (s * final_feature).view(batch_size, -1)
        return self.classifier(pooled)


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


def _forward_all(bert_model, clip_model, fnd_model, batch):
    bert_ids = batch["bert_input_ids"].to(DEVICE)
    bert_attn = batch["bert_attention_mask"].to(DEVICE)
    bert_ttid = batch["bert_token_type_ids"].to(DEVICE)
    img_resnet = batch["image_resnet"].to(DEVICE)
    clip_ids = batch["clip_input_ids"].to(DEVICE)
    clip_attn = batch["clip_attention_mask"].to(DEVICE)
    clip_pix = batch["clip_pixel_values"].to(DEVICE)

    bert_out = bert_model(
        input_ids=bert_ids,
        attention_mask=bert_attn,
        token_type_ids=bert_ttid,
    )
    text_clip = clip_model.get_text_features(input_ids=clip_ids, attention_mask=clip_attn)
    image_clip = clip_model.get_image_features(pixel_values=clip_pix)
    logits = fnd_model(bert_out.hidden_states, img_resnet, text_clip, image_clip)
    return logits


def run_epoch(bert_model, clip_model, fnd_model, loader, optimizer=None):
    is_train = optimizer is not None
    bert_model.eval()
    clip_model.eval()
    fnd_model.train() if is_train else fnd_model.eval()

    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_true, all_pred = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="Train" if is_train else "Eval"):
            labels = batch["label"].to(DEVICE)
            logits = _forward_all(bert_model, clip_model, fnd_model, batch)
            loss = ce(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(fnd_model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            all_true.extend(labels.detach().cpu().numpy().tolist())
            all_pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(all_true))
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return {"loss": avg_loss, "acc": acc, "f1": f1}


def collect_predictions(bert_model, clip_model, fnd_model, loader):
    bert_model.eval()
    clip_model.eval()
    fnd_model.eval()
    ys, ps, scores = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collect"):
            logits = _forward_all(bert_model, clip_model, fnd_model, batch)
            prob_fake = torch.softmax(logits, dim=-1)[:, 1]
            pred = logits.argmax(1)
            ys.extend(batch["label"].numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())
            scores.extend(prob_fake.cpu().numpy().tolist())
    return np.array(ys), np.array(ps), np.array(scores)


def evaluate_binary(yt, yp, scores, split_name: str):
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


def build_loader(dataset, batch_size, num_workers, shuffle, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fnd_clip,
        drop_last=drop_last,
    )


def train_one_run(train_ds, val_ds, args):
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fnd_clip,
        drop_last=True,
    )
    val_loader = build_loader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    bert_model = BertModel.from_pretrained(BERT_NAME, output_hidden_states=True).to(DEVICE)
    clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE)
    fnd_model = FNDCLIPMultiModalBinary(num_classes=2).to(DEVICE)

    for p in bert_model.parameters():
        p.requires_grad = False
    for p in clip_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(fnd_model.parameters(), lr=args.lr, weight_decay=1e-4)
    early = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        mode="max",
    )

    best_f1 = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(bert_model, clip_model, fnd_model, train_loader, optimizer=optimizer)
        va = run_epoch(bert_model, clip_model, fnd_model, val_loader, optimizer=None)
        print(
            f"[Epoch {epoch}] train_loss={tr['loss']:.4f} train_f1={tr['f1']:.4f} | "
            f"val_loss={va['loss']:.4f} val_f1={va['f1']:.4f}"
        )
        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            best_state = copy.deepcopy(fnd_model.state_dict())
        if early.step(va["f1"]):
            print(f"[FND-CLIP-MIRAGE] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        fnd_model.load_state_dict(best_state)
    return bert_model, clip_model, fnd_model, best_f1


def evaluate_all_test_splits(bert_model, clip_model, fnd_model, ds, test_splits, args):
    all_metrics: Dict[str, Dict[str, float]] = {}
    block_texts = []
    for split_name in test_splits:
        test_ds = MirageFNDCLIPDataset(ds[split_name])
        test_loader = build_loader(
            test_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
        )
        yt, yp, scores = collect_predictions(bert_model, clip_model, fnd_model, test_loader)
        block, metrics = evaluate_binary(yt, yp, scores, split_name)
        block_texts.append(block)
        all_metrics[split_name] = metrics
    return block_texts, all_metrics


def _mean_stats(metric_dict: Dict[str, Dict[str, float]]):
    out = {}
    for k in ("acc", "f1", "f1_ai", "auc"):
        vals = [metric_dict[s][k] for s in metric_dict]
        out[k] = (float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)))
    return out


def train_and_test(args):
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[FND-CLIP-MIRAGE] loading dataset: {args.hf_dataset_id}")
    ds = load_dataset(args.hf_dataset_id)
    print(f"[FND-CLIP-MIRAGE] splits: {list(ds.keys())}")
    test_splits = resolve_test_splits(ds)
    if len(test_splits) == 0:
        raise RuntimeError("No test split found in dataset.")

    report_io = io.StringIO()
    mode_name = "KFold" if not args.no_kfold else "Official Split"
    report_io.write(f"###### FND-CLIP-MIRAGE ({mode_name}) ######\n")
    report_io.write(f"test_splits={test_splits}\n")
    report_io.write("\n===== TEST RESULTS =====\n")

    if args.no_kfold:
        train_ds = MirageFNDCLIPDataset(ds["train"])
        val_ds = MirageFNDCLIPDataset(ds["validation"])
        bert_model, clip_model, fnd_model, best_f1 = train_one_run(train_ds, val_ds, args)

        report_io.write(f"best_val_macro_f1={best_f1:.4f}\n")
        blocks, all_metrics = evaluate_all_test_splits(
            bert_model, clip_model, fnd_model, ds, test_splits, args
        )
        for b in blocks:
            print(b)
            report_io.write(b)

        report_io.write("\n===== SUMMARY (avg over test splits) =====\n")
        mean_stats = _mean_stats(all_metrics)
        for k in ("acc", "f1", "f1_ai", "auc"):
            mean_v, min_v, max_v = mean_stats[k]
            line = f"{k}: {mean_v:.4f} (min={min_v:.4f}, max={max_v:.4f})"
            print(line)
            report_io.write(line + "\n")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"fnd_clip_mirage_{timestamp}.pt")
        torch.save(
            {
                "model_state_dict": fnd_model.state_dict(),
                "config": vars(args),
                "metrics": all_metrics,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
            ckpt_path,
        )
    else:
        base_train = ds["train"]
        probe_ds = MirageFNDCLIPDataset(base_train)
        labels = np.array(probe_ds.labels)
        skf = StratifiedKFold(
            n_splits=args.kfold_splits, shuffle=True, random_state=args.seed
        )
        fold_results: List[Dict[str, Dict[str, float]]] = []
        report_io.write(f"kfold_splits={args.kfold_splits}\n")

        for fold_idx, (tr_idx, va_idx) in enumerate(
            skf.split(np.zeros(len(labels)), labels), start=1
        ):
            print(f"\n[FND-CLIP-MIRAGE] ===== Fold {fold_idx}/{args.kfold_splits} =====")
            report_io.write(
                f"\n\n{'#' * 70}\n### Fold {fold_idx}/{args.kfold_splits}\n{'#' * 70}\n"
            )
            train_ds = MirageFNDCLIPDataset(base_train.select(tr_idx.tolist()))
            val_ds = MirageFNDCLIPDataset(base_train.select(va_idx.tolist()))
            bert_model, clip_model, fnd_model, best_f1 = train_one_run(train_ds, val_ds, args)
            report_io.write(f"best_val_macro_f1={best_f1:.4f}\n")

            blocks, fold_metrics = evaluate_all_test_splits(
                bert_model, clip_model, fnd_model, ds, test_splits, args
            )
            for b in blocks:
                print(b)
                report_io.write(b)
            fold_results.append(fold_metrics)

            del bert_model, clip_model, fnd_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        overall_summary = {}
        for k in ("acc", "f1", "f1_ai", "auc"):
            vals = [global_metrics[s][k] for s in test_splits]
            overall_summary[k] = float(np.mean(vals))
            line = f"{k}: {overall_summary[k]:.4f}"
            print(line)
            report_io.write(line + "\n")

        ckpt_path = os.path.join(
            CHECKPOINT_DIR, f"fnd_clip_mirage_kfold{args.kfold_splits}_{timestamp}.pt"
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

    report_path = os.path.join(RESULT_DIR, f"fnd_clip_mirage_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_io.getvalue())

    print(f"[FND-CLIP-MIRAGE] checkpoint: {ckpt_path}")
    print(f"[FND-CLIP-MIRAGE] report: {report_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="FND-CLIP for MiRAGe-News (binary)")
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
    print(f"FND-CLIP-MIRAGE | device={DEVICE}")
    train_and_test(args)
