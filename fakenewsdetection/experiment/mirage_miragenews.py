"""
MiRAGe for MiRAGe-News (Binary: Real vs AI-Fake)
"""

from __future__ import annotations

import argparse
import io
import math
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    print(
        "\n[오류] open_clip_torch 패키지가 없습니다.\n"
        "  pip install open_clip_torch\n"
        "명령어로 설치 후 재실행해주세요."
    )
    raise SystemExit(1)


LOG_TAG = "MiRAGe-MIRAGE"
HF_DATASET_ID = "anson-huang/mirage-news"
LABEL_NAMES = ["Real", "AI-Fake"]

SEED = 42
BATCH = 16
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_DELTA = 1e-4
SAMPLER_ALPHA = 0.5
KFOLD_SPLITS = 5
DROPOUT = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(_HERE, "result")
CHECKPOINT_DIR = os.path.abspath(os.path.join(_HERE, "..", "checkpoint"))
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
NUM_CONCEPTS = len(VISUAL_CONCEPTS)
NUM_TBM_FEATS = 15


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_tbm_features(text: str) -> np.ndarray:
    text = str(text)
    words = text.split()
    chars = list(text)
    word_count = len(words)
    char_count = len(text)
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    num_digits = sum(c.isdigit() for c in chars)
    num_upper = sum(c.isupper() for c in chars)
    num_punct = sum(not c.isalnum() and not c.isspace() for c in chars)
    digit_ratio = num_digits / max(char_count, 1)
    upper_ratio = num_upper / max(char_count, 1)
    punct_ratio = num_punct / max(char_count, 1)
    unique_words = len(set(w.lower() for w in words))
    ttr = unique_words / max(word_count, 1)
    has_quote = float('"' in text or "'" in text)
    has_number = float(any(c.isdigit() for c in text))
    num_sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
    avg_sent_len = word_count / num_sentences
    return np.array(
        [
            word_count, char_count, avg_word_len,
            digit_ratio, upper_ratio, punct_ratio,
            ttr, has_quote, has_number,
            num_sentences, avg_sent_len,
            num_digits, num_upper, num_punct, unique_words,
        ],
        dtype=np.float32,
    )


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


class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, val: float) -> bool:
        if self.best is None:
            self.best = val
            return False
        improved = ((val - self.best) > self.min_delta) if self.mode == "max" else ((self.best - val) > self.min_delta)
        if improved:
            self.best = val
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class MiRAGeFeatureDataset(Dataset):
    def __init__(self, img_feats, txt_feats, concept_scores, tbm_feats, labels):
        self.img_feats = img_feats
        self.txt_feats = txt_feats
        self.concept_scores = concept_scores
        self.tbm_feats = tbm_feats
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (
            torch.tensor(self.img_feats[i], dtype=torch.float32),
            torch.tensor(self.txt_feats[i], dtype=torch.float32),
            torch.tensor(self.concept_scores[i], dtype=torch.float32),
            torch.tensor(self.tbm_feats[i], dtype=torch.float32),
            torch.tensor(self.labels[i], dtype=torch.long),
        )


def collate_mirage(batch):
    img_f, txt_f, cscore, tbm_f, lbl = zip(*batch)
    return torch.stack(img_f), torch.stack(txt_f), torch.stack(cscore), torch.stack(tbm_f), torch.stack(lbl)


def make_weighted_sampler(labels, alpha: float = SAMPLER_ALPHA):
    cnt = {}
    for l in labels:
        cnt[l] = cnt.get(l, 0) + 1
    w = [(1.0 / cnt[l]) ** alpha for l in labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


class MiRAGeImgDetector(nn.Module):
    def __init__(self, img_dim=512, concept_dim=NUM_CONCEPTS, num_classes=2, dropout=DROPOUT):
        super().__init__()
        self.linear_branch = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.cbm_branch = nn.Sequential(
            nn.Linear(concept_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.ensemble_head = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, img_feat, concept_scores):
        logits_lin = self.linear_branch(img_feat)
        logits_cbm = self.cbm_branch(concept_scores)
        logits_out = self.ensemble_head(torch.cat([logits_lin, logits_cbm], dim=1))
        return logits_out, logits_lin, logits_cbm


class MiRAGeTxtDetector(nn.Module):
    def __init__(self, txt_dim=512, tbm_dim=NUM_TBM_FEATS, num_classes=2, dropout=DROPOUT):
        super().__init__()
        self.linear_branch = nn.Sequential(
            nn.Linear(txt_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.tbm_branch = nn.Sequential(
            nn.Linear(tbm_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
        self.ensemble_head = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, txt_feat, tbm_feats):
        logits_lin = self.linear_branch(txt_feat)
        logits_tbm = self.tbm_branch(tbm_feats)
        logits_out = self.ensemble_head(torch.cat([logits_lin, logits_tbm], dim=1))
        return logits_out, logits_lin, logits_tbm


def _hf_cols(hf_split):
    cols = hf_split.column_names
    text_col = next((c for c in ("caption", "text", "headline") if c in cols), None)
    image_col = next((c for c in ("image", "img") if c in cols), None)
    label_col = next((c for c in ("label", "labels", "fake") if c in cols), None)
    if None in (text_col, image_col, label_col):
        raise ValueError(f"필수 컬럼 없음. columns={cols}")
    return text_col, image_col, label_col


def _load_pil(raw):
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict) and "bytes" in raw:
        return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
    return Image.new("RGB", (224, 224), (127, 127, 127))


@torch.no_grad()
def encode_split(hf_split, clip_model, clip_preprocess, clip_tokenizer, concept_text_feats, batch_size=128):
    text_col, image_col, label_col = _hf_cols(hf_split)
    labels = np.array([int(x) for x in hf_split[label_col]], dtype=np.int64)
    texts = [str(x).strip() for x in hf_split[text_col]]
    n = len(labels)
    img_feats_all = np.zeros((n, 512), dtype=np.float32)
    txt_feats_all = np.zeros((n, 512), dtype=np.float32)
    concept_scores_all = np.zeros((n, NUM_CONCEPTS), dtype=np.float32)
    tbm_feats_all = np.zeros((n, NUM_TBM_FEATS), dtype=np.float32)

    for start in tqdm(range(0, n, batch_size), desc=f"[{LOG_TAG}] CLIP encode"):
        end = min(start + batch_size, n)
        imgs = [_load_pil(hf_split[i][image_col]) for i in range(start, end)]
        img_tensor = torch.stack([clip_preprocess(im) for im in imgs]).to(DEVICE)
        img_feat = F.normalize(clip_model.encode_image(img_tensor), dim=-1)
        img_feats_all[start:end] = img_feat.cpu().numpy()
        concept_scores_all[start:end] = (img_feat @ concept_text_feats.T).cpu().numpy()

        text_batch = texts[start:end]
        tok = clip_tokenizer(text_batch).to(DEVICE)
        txt_feat = F.normalize(clip_model.encode_text(tok), dim=-1)
        txt_feats_all[start:end] = txt_feat.cpu().numpy()

        for i, t in enumerate(text_batch):
            tbm_feats_all[start + i] = compute_tbm_features(t)

    return img_feats_all, txt_feats_all, concept_scores_all, tbm_feats_all, labels


def _build_loader(img_feats, txt_feats, c_scores, tbm_feats, labels, batch_size, num_workers, shuffle=False, weighted=False):
    ds = MiRAGeFeatureDataset(img_feats, txt_feats, c_scores, tbm_feats, labels)
    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_mirage,
        drop_last=shuffle,
    )
    if weighted:
        kw["sampler"] = make_weighted_sampler(labels.tolist())
        kw["shuffle"] = False
    else:
        kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


def _train_one_epoch(model, loader, optimizer, mode: str):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    for batch in tqdm(loader, desc=f"[{LOG_TAG}] Train[{mode}]", leave=False):
        img_f, txt_f, cscore, tbm_f, labels = [b.to(DEVICE) for b in batch]
        if mode == "img":
            logits, logits_lin, logits_aux = model(img_f, cscore)
        else:
            logits, logits_lin, logits_aux = model(txt_f, tbm_f)
        loss = (
            loss_fn(logits, labels)
            + 0.3 * loss_fn(logits_lin, labels)
            + 0.3 * loss_fn(logits_aux, labels)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / max(n, 1)


@torch.no_grad()
def _validate(model, loader, mode: str):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        img_f, txt_f, cscore, tbm_f, labels = [b.to(DEVICE) for b in batch]
        if mode == "img":
            logits, _, _ = model(img_f, cscore)
        else:
            logits, _, _ = model(txt_f, tbm_f)
        ys.extend(labels.cpu().tolist())
        ps.extend(logits.argmax(1).cpu().tolist())
    return float(f1_score(ys, ps, average="macro", zero_division=0))


def train_mirage_unimodal(model, train_loader, val_loader, mode: str, epochs=EPOCHS):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    early = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode="max")
    best_val = -1.0
    best_state = None
    for epoch in range(epochs):
        loss = _train_one_epoch(model, train_loader, optimizer, mode)
        val_f1 = _validate(model, val_loader, mode)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  [MiRAGe-{mode.upper()}] Epoch {epoch+1:02d} | loss={loss:.4f} | val_macro_f1={val_f1:.4f} | lr={cur_lr:.2e}")
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if early.step(val_f1):
            print(f"  [MiRAGe-{mode.upper()}] Early stopping at epoch {epoch+1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def evaluate_mirage(img_model, txt_model, loader, fusion="avg"):
    img_model.eval()
    txt_model.eval()
    ys, ps, scores = [], [], []
    for batch in tqdm(loader, desc=f"[{LOG_TAG}] Eval[{fusion}]"):
        img_f, txt_f, cscore, tbm_f, labels = [b.to(DEVICE) for b in batch]
        logits_img, _, _ = img_model(img_f, cscore)
        logits_txt, _, _ = txt_model(txt_f, tbm_f)
        if fusion == "avg":
            prob = (F.softmax(logits_img, 1) + F.softmax(logits_txt, 1)) / 2.0
        elif fusion == "img":
            prob = F.softmax(logits_img, 1)
        else:
            prob = F.softmax(logits_txt, 1)
        pred = prob.argmax(1)
        ys.extend(labels.cpu().tolist())
        ps.extend(pred.cpu().tolist())
        scores.extend(prob[:, 1].cpu().tolist())
    yt = np.array(ys)
    yp = np.array(ps)
    sc = np.array(scores)
    acc = float(accuracy_score(yt, yp))
    f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
    f1_ai = float(f1_score(yt, yp, average="binary", pos_label=1, zero_division=0))
    try:
        auc = float(roc_auc_score(yt, sc))
    except Exception:
        auc = float("nan")
    block = (
        classification_report(yt, yp, target_names=LABEL_NAMES, digits=4, zero_division=0)
        + f"\n[{LOG_TAG}-{fusion}] Acc={acc:.4f} Macro-F1={f1:.4f} AI-Fake-F1={f1_ai:.4f} AUC={auc:.4f}\n"
    )
    return block, {"acc": acc, "f1": f1, "f1_ai": f1_ai, "auc": auc}


def train_and_test(args):
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{LOG_TAG}] loading dataset: {args.hf_dataset_id}")
    ds = load_dataset(args.hf_dataset_id)
    test_splits = resolve_test_splits(ds)
    if not test_splits:
        raise RuntimeError("No test split found.")

    print(f"[{LOG_TAG}] loading CLIP ViT-B-32 ...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(DEVICE).eval()
    with torch.no_grad():
        concept_tokens = clip_tokenizer(VISUAL_CONCEPTS).to(DEVICE)
        concept_text_feats = F.normalize(clip_model.encode_text(concept_tokens), dim=-1)

    report_io = io.StringIO()
    mode_name = "Official Split" if args.no_kfold else "KFold"
    report_io.write(f"###### {LOG_TAG} ({mode_name}) ######\n")
    report_io.write(f"test_splits={test_splits}\n")

    tr_img, tr_txt, tr_cs, tr_tbm, tr_y = encode_split(ds["train"], clip_model, clip_preprocess, clip_tokenizer, concept_text_feats)
    va_img, va_txt, va_cs, va_tbm, va_y = encode_split(ds["validation"], clip_model, clip_preprocess, clip_tokenizer, concept_text_feats)
    tbm_mean = tr_tbm.mean(axis=0)
    tbm_std = tr_tbm.std(axis=0) + 1e-8
    tr_tbm = (tr_tbm - tbm_mean) / tbm_std
    va_tbm = (va_tbm - tbm_mean) / tbm_std

    if args.no_kfold:
        train_loader = _build_loader(tr_img, tr_txt, tr_cs, tr_tbm, tr_y, args.batch_size, args.num_workers, shuffle=False, weighted=True)
        val_loader = _build_loader(va_img, va_txt, va_cs, va_tbm, va_y, args.batch_size, args.num_workers, shuffle=False)

        img_model = MiRAGeImgDetector(num_classes=2).to(DEVICE)
        txt_model = MiRAGeTxtDetector(num_classes=2).to(DEVICE)
        img_model, best_img = train_mirage_unimodal(img_model, train_loader, val_loader, mode="img", epochs=args.epochs)
        txt_model, best_txt = train_mirage_unimodal(txt_model, train_loader, val_loader, mode="txt", epochs=args.epochs)
        report_io.write(f"best_val_f1_img={best_img:.4f} best_val_f1_txt={best_txt:.4f}\n")

        all_metrics: Dict[str, Dict[str, float]] = {}
        for split_name in test_splits:
            te_img, te_txt, te_cs, te_tbm, te_y = encode_split(ds[split_name], clip_model, clip_preprocess, clip_tokenizer, concept_text_feats)
            te_tbm = (te_tbm - tbm_mean) / tbm_std
            te_loader = _build_loader(te_img, te_txt, te_cs, te_tbm, te_y, args.batch_size, args.num_workers, shuffle=False)
            report_io.write(f"\n=== {split_name} ===\n")
            for fm in ("img", "txt", "avg"):
                block, metrics = evaluate_mirage(img_model, txt_model, te_loader, fusion=fm)
                print(f"\n[{split_name}] fusion={fm}\n{block}")
                report_io.write(f"\n--- fusion={fm} ---\n{block}")
                all_metrics[f"{split_name}:{fm}"] = metrics

        ckpt_path = os.path.join(args.checkpoint_dir, f"mirage_miragenews_{timestamp}.pt")
        torch.save(
            {
                "img_model_state": img_model.state_dict(),
                "txt_model_state": txt_model.state_dict(),
                "tbm_mean": tbm_mean,
                "tbm_std": tbm_std,
                "config": vars(args),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "metrics": all_metrics,
            },
            ckpt_path,
        )
    else:
        labels = tr_y
        skf = StratifiedKFold(n_splits=args.kfold_splits, shuffle=True, random_state=args.seed)
        fold_results = []
        report_io.write(f"kfold_splits={args.kfold_splits}\n")
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
            print(f"\n[{LOG_TAG}] ===== Fold {fold_idx}/{args.kfold_splits} =====")
            ftr_img, ftr_txt, ftr_cs, ftr_tbm, ftr_y = tr_img[tr_idx], tr_txt[tr_idx], tr_cs[tr_idx], tr_tbm[tr_idx], tr_y[tr_idx]
            fva_img, fva_txt, fva_cs, fva_tbm, fva_y = tr_img[va_idx], tr_txt[va_idx], tr_cs[va_idx], tr_tbm[va_idx], tr_y[va_idx]

            train_loader = _build_loader(ftr_img, ftr_txt, ftr_cs, ftr_tbm, ftr_y, args.batch_size, args.num_workers, shuffle=False, weighted=True)
            val_loader = _build_loader(fva_img, fva_txt, fva_cs, fva_tbm, fva_y, args.batch_size, args.num_workers, shuffle=False)

            img_model = MiRAGeImgDetector(num_classes=2).to(DEVICE)
            txt_model = MiRAGeTxtDetector(num_classes=2).to(DEVICE)
            img_model, _ = train_mirage_unimodal(img_model, train_loader, val_loader, mode="img", epochs=args.epochs)
            txt_model, _ = train_mirage_unimodal(txt_model, train_loader, val_loader, mode="txt", epochs=args.epochs)

            fold_metrics = {}
            for split_name in test_splits:
                te_img, te_txt, te_cs, te_tbm, te_y = encode_split(ds[split_name], clip_model, clip_preprocess, clip_tokenizer, concept_text_feats)
                te_tbm = (te_tbm - tbm_mean) / tbm_std
                te_loader = _build_loader(te_img, te_txt, te_cs, te_tbm, te_y, args.batch_size, args.num_workers, shuffle=False)
                for fm in ("img", "txt", "avg"):
                    _, metrics = evaluate_mirage(img_model, txt_model, te_loader, fusion=fm)
                    fold_metrics[f"{split_name}:{fm}"] = metrics
            fold_results.append(fold_metrics)
            del img_model, txt_model
            gc_collect()

        report_io.write("\n===== KFOLD GLOBAL SUMMARY =====\n")
        keys = fold_results[0].keys()
        for k in keys:
            acc = float(np.mean([fr[k]["acc"] for fr in fold_results]))
            f1m = float(np.mean([fr[k]["f1"] for fr in fold_results]))
            line = f"{k}: acc={acc:.4f}, f1={f1m:.4f}"
            print(line)
            report_io.write(line + "\n")
        ckpt_path = os.path.join(args.checkpoint_dir, f"mirage_miragenews_kfold{args.kfold_splits}_{timestamp}.pt")
        torch.save({"fold_metrics": fold_results, "config": vars(args)}, ckpt_path)

    report_path = os.path.join(args.result_dir, f"mirage_miragenews_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_io.getvalue())
    print(f"[{LOG_TAG}] checkpoint: {ckpt_path}")
    print(f"[{LOG_TAG}] report: {report_path}")


def gc_collect():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    ap = argparse.ArgumentParser(description="MiRAGe for MiRAGe-News (binary)")
    ap.add_argument("--hf_dataset_id", default=HF_DATASET_ID)
    ap.add_argument("--batch_size", type=int, default=BATCH)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--result_dir", default=RESULT_DIR)
    ap.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    ap.add_argument("--no_kfold", action="store_true", help="official split(train/val + all test*) 사용")
    ap.add_argument("--kfold_splits", type=int, default=KFOLD_SPLITS)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    print(f"{LOG_TAG} | device={DEVICE} | batch={args.batch_size} epochs={args.epochs}")
    train_and_test(args)
