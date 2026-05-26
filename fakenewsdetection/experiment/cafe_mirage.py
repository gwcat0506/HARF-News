"""
CAFE for MiRAGe-News (Binary: Real vs AI-Fake)
- Train on official train/validation splits
"""

from __future__ import annotations

import argparse
import io
import math
import os
import random
import re
import copy
import gc
from collections import Counter
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_DATASET_ID = "anson-huang/mirage-news"

TEXT_DIM = 200
SEQ_LEN = 64
IMG_SIZE = 224
VOCAB_MIN_FREQ = 2

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


def tokenize(text: str):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    return text.split()


def build_vocab(texts: List[str], min_freq=VOCAB_MIN_FREQ):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in counter.most_common():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def text_to_ids(text: str, vocab: Dict[str, int], max_len=SEQ_LEN):
    pad_idx = vocab["<pad>"]
    unk_idx = vocab["<unk>"]
    tokens = tokenize(text)
    ids = [vocab.get(w, unk_idx) for w in tokens[:max_len]]
    ids = ids + [pad_idx] * (max_len - len(ids))
    return ids


image_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MirageCAFEDataset(Dataset):
    def __init__(self, hf_split, vocab, max_len=SEQ_LEN):
        self.hf_split = hf_split
        self.vocab = vocab
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
        ids = text_to_ids(text, self.vocab, self.max_len)

        img = row[self.image_col]
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        elif isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        else:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127))

        return {
            "text_ids": torch.tensor(ids, dtype=torch.long),
            "image": image_transform(img),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_cafe(batch):
    text_ids = torch.stack([b["text_ids"] for b in batch])
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return text_ids, images, labels


class FastCNN(nn.Module):
    def __init__(self, channel=32, kernel_size=(1, 2, 4, 8)):
        super().__init__()
        self.fast_cnn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(TEXT_DIM, channel, kernel_size=k),
                    nn.BatchNorm1d(channel),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                for k in kernel_size
            ]
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_out = [m(x).squeeze(-1) for m in self.fast_cnn]
        return torch.cat(x_out, 1)


class EncodingPart(nn.Module):
    def __init__(self, shared_image_dim=128, shared_text_dim=128):
        super().__init__()
        self.shared_text_encoding = FastCNN(channel=32, kernel_size=(1, 2, 4, 8))
        self.shared_text_linear = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU(),
        )
        self.shared_image = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_image_dim),
            nn.BatchNorm1d(shared_image_dim),
            nn.ReLU(),
        )

    def forward(self, text, image):
        text_encoding = self.shared_text_encoding(text)
        text_shared = self.shared_text_linear(text_encoding)
        image_shared = self.shared_image(image)
        return text_shared, image_shared


class SimilarityModule(nn.Module):
    def __init__(self, shared_dim=128, sim_dim=64):
        super().__init__()
        self.encoding = EncodingPart()
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
        )
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(sim_dim * 2),
            nn.Linear(sim_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, text, image):
        text_enc, image_enc = self.encoding(text, image)
        text_aligned = self.text_aligner(text_enc)
        image_aligned = self.image_aligner(image_enc)
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
        mu, sigma = params[:, : self.z_dim], params[:, self.z_dim :]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


class AmbiguityLearning(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_encoding, image_encoding):
        p_z1 = self.encoder_text(text_encoding)
        p_z2 = self.encoder_image(image_encoding)
        z1, z2 = p_z1.rsample(), p_z2.rsample()
        kl_1_2 = p_z1.log_prob(z1) - p_z2.log_prob(z1)
        kl_2_1 = p_z2.log_prob(z2) - p_z1.log_prob(z2)
        skl = (kl_1_2 + kl_2_1) / 2.0
        return torch.sigmoid(skl)


class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=128, prime_dim=16):
        super().__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU(),
        )
        self.image_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU(),
        )

    def forward(self, text_enc, image_enc):
        return self.text_uni(text_enc), self.image_uni(image_enc)


class CrossModule4Batch(nn.Module):
    def __init__(self, corre_out_dim=64):
        super().__init__()
        self.corre_dim = 64
        self.softmax = nn.Softmax(-1)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU(),
        )

    def forward(self, text, image):
        text_in = text.unsqueeze(2)
        image_in = image.unsqueeze(1)
        sim = torch.matmul(text_in, image_in) / math.sqrt(text.shape[1])
        corr = self.softmax(sim)
        corr_p = self.pooling(corr).squeeze(-1)
        return self.c_specific_2(corr_p)


class DetectionModule(nn.Module):
    def __init__(self, num_classes=2, feature_dim=64 + 16 + 16, h_dim=64):
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
            nn.Linear(h_dim, num_classes),
        )

    def forward(self, text_raw, image_raw, text, image):
        skl = self.ambiguity_module(text, image)
        text_prime, image_prime = self.encoding(text_raw, image_raw)
        text_prime, image_prime = self.uni_repre(text_prime, image_prime)
        correlation = self.cross_module(text, image)
        weight_uni = (1 - skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        final = torch.cat(
            [weight_uni * text_prime, weight_uni * image_prime, weight_corre * correlation], 1
        )
        return self.classifier_corre(final)


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.features(x).flatten(1)


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=TEXT_DIM, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

    def forward(self, ids):
        return self.embed(ids).float()


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


def run_train_epoch(sim_module, det_module, text_embed_module, img_encoder, train_loader, opt_sim, opt_det):
    sim_module.train()
    det_module.train()
    text_embed_module.train()
    img_encoder.eval()

    loss_fn_sim = nn.CosineEmbeddingLoss()
    loss_fn_det = nn.CrossEntropyLoss()
    loss_sim_tot, loss_det_tot = 0.0, 0.0
    cnt_sim, cnt_det = 0, 0
    all_y, all_p = [], []

    for text_ids, images, labels in tqdm(train_loader, desc="Train"):
        text_ids = text_ids.to(DEVICE)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            img_feat = img_encoder(images)

        text_emb = text_embed_module(text_ids)
        n = text_emb.shape[0]
        mat_t, mat_i, unmat_i = prepare_similarity_data(text_emb, img_feat, n)

        ta_m, ia_m, _ = sim_module(mat_t, mat_i)
        ta_u, ia_u, _ = sim_module(mat_t, unmat_i)
        ta_cat = torch.cat([ta_m, ta_u], 0)
        ia_cat = torch.cat([ia_m, ia_u], 0)
        sim_label = torch.cat(
            [
                torch.ones(ta_m.shape[0], device=DEVICE),
                -torch.ones(ta_u.shape[0], device=DEVICE),
            ]
        )
        loss_sim = loss_fn_sim(ta_cat, ia_cat, sim_label)
        opt_sim.zero_grad()
        loss_sim.backward()
        opt_sim.step()

        text_emb_det = text_embed_module(text_ids)
        ta, ia, _ = sim_module(text_emb_det, img_feat)
        pred_det = det_module(text_emb_det, img_feat, ta, ia)
        loss_det = loss_fn_det(pred_det, labels)
        opt_det.zero_grad()
        loss_det.backward()
        opt_det.step()

        loss_sim_tot += loss_sim.item() * n * 2
        cnt_sim += n * 2
        loss_det_tot += loss_det.item() * n
        cnt_det += n

        all_y.extend(labels.detach().cpu().numpy().tolist())
        all_p.extend(pred_det.argmax(1).detach().cpu().numpy().tolist())

    tr_acc = accuracy_score(all_y, all_p)
    tr_f1 = f1_score(all_y, all_p, average="macro", zero_division=0)
    return {
        "loss_sim": loss_sim_tot / max(1, cnt_sim),
        "loss_det": loss_det_tot / max(1, cnt_det),
        "acc": tr_acc,
        "f1": tr_f1,
    }


def run_eval_epoch(sim_module, det_module, text_embed_module, img_encoder, loader):
    sim_module.eval()
    det_module.eval()
    text_embed_module.eval()
    img_encoder.eval()

    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    ys, ps, scores = [], [], []
    with torch.no_grad():
        for text_ids, images, labels in tqdm(loader, desc="Eval"):
            text_ids = text_ids.to(DEVICE)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            text_emb = text_embed_module(text_ids)
            img_feat = img_encoder(images)
            ta, ia, _ = sim_module(text_emb, img_feat)
            logits = det_module(text_emb, img_feat, ta, ia)
            loss = ce(logits, labels)
            total_loss += loss.item() * labels.size(0)

            prob_fake = torch.softmax(logits, dim=-1)[:, 1]
            ys.extend(labels.cpu().numpy().tolist())
            ps.extend(logits.argmax(1).cpu().numpy().tolist())
            scores.extend(prob_fake.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(ys))
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="macro", zero_division=0)
    return {"loss": avg_loss, "acc": acc, "f1": f1}, np.array(ys), np.array(ps), np.array(scores)


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
        collate_fn=collate_cafe,
        drop_last=drop_last,
    )


def train_one_run(train_ds, val_ds, vocab, args):
    train_loader = build_loader(train_ds, args.batch_size, args.num_workers, shuffle=True, drop_last=True)
    val_loader = build_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)

    pad_idx = vocab["<pad>"]
    text_embed_module = TextEmbedding(len(vocab), TEXT_DIM, pad_idx=pad_idx).to(DEVICE)
    img_encoder = ImageEncoder().to(DEVICE)
    sim_module = SimilarityModule().to(DEVICE)
    det_module = DetectionModule(num_classes=2).to(DEVICE)

    for p in img_encoder.parameters():
        p.requires_grad = False

    opt_sim = torch.optim.Adam(
        list(sim_module.parameters()) + list(text_embed_module.parameters()),
        lr=args.lr,
    )
    opt_det = torch.optim.Adam(
        list(det_module.parameters()) + list(text_embed_module.parameters()),
        lr=args.lr,
    )
    early = EarlyStopping(args.early_stop_patience, args.early_stop_min_delta, mode="max")

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        tr = run_train_epoch(
            sim_module, det_module, text_embed_module, img_encoder, train_loader, opt_sim, opt_det
        )
        va, _, _, _ = run_eval_epoch(sim_module, det_module, text_embed_module, img_encoder, val_loader)
        print(
            f"[Epoch {epoch}] "
            f"sim_loss={tr['loss_sim']:.4f} det_loss={tr['loss_det']:.4f} train_f1={tr['f1']:.4f} | "
            f"val_loss={va['loss']:.4f} val_f1={va['f1']:.4f}"
        )
        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            best_state = {
                "sim": copy.deepcopy(sim_module.state_dict()),
                "det": copy.deepcopy(det_module.state_dict()),
                "emb": copy.deepcopy(text_embed_module.state_dict()),
            }
        if early.step(va["f1"]):
            print(f"[CAFE-MIRAGE] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        sim_module.load_state_dict(best_state["sim"])
        det_module.load_state_dict(best_state["det"])
        text_embed_module.load_state_dict(best_state["emb"])

    return sim_module, det_module, text_embed_module, img_encoder, best_f1


def evaluate_all_test_splits(sim_module, det_module, text_embed_module, img_encoder, ds, test_splits, vocab, args):
    all_metrics = {}
    blocks = []
    for split_name in test_splits:
        test_ds = MirageCAFEDataset(ds[split_name], vocab=vocab, max_len=SEQ_LEN)
        test_loader = build_loader(test_ds, args.batch_size, args.num_workers, shuffle=False)
        _, yt, yp, scores = run_eval_epoch(sim_module, det_module, text_embed_module, img_encoder, test_loader)
        block, metrics = evaluate_binary(yt, yp, scores, split_name)
        blocks.append(block)
        all_metrics[split_name] = metrics
    return blocks, all_metrics


def train_and_test(args):
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[CAFE-MIRAGE] loading dataset: {args.hf_dataset_id}")
    ds = load_dataset(args.hf_dataset_id)
    print(f"[CAFE-MIRAGE] splits: {list(ds.keys())}")
    test_splits = resolve_test_splits(ds)
    if not test_splits:
        raise RuntimeError("No test split found in dataset.")

    # CAFE vocabulary is built from training text only.
    train_text_col = next((c for c in ("caption", "text", "headline") if c in ds["train"].column_names), None)
    if train_text_col is None:
        raise RuntimeError(f"Train text column not found. columns={ds['train'].column_names}")
    train_texts = [str(x).strip() for x in ds["train"][train_text_col]]
    vocab = build_vocab(train_texts, min_freq=VOCAB_MIN_FREQ)
    print(f"[CAFE-MIRAGE] vocab size: {len(vocab)}")

    report_io = io.StringIO()
    mode_name = "KFold" if not args.no_kfold else "Official Split"
    report_io.write(f"###### CAFE-MIRAGE ({mode_name}) ######\n")
    report_io.write(f"test_splits={test_splits}\n")
    report_io.write("\n===== TEST RESULTS =====\n")

    if args.no_kfold:
        train_ds = MirageCAFEDataset(ds["train"], vocab=vocab, max_len=SEQ_LEN)
        val_ds = MirageCAFEDataset(ds["validation"], vocab=vocab, max_len=SEQ_LEN)
        sim_module, det_module, text_embed_module, img_encoder, best_f1 = train_one_run(
            train_ds, val_ds, vocab, args
        )
        report_io.write(f"best_val_macro_f1={best_f1:.4f}\n")
        blocks, all_metrics = evaluate_all_test_splits(
            sim_module, det_module, text_embed_module, img_encoder, ds, test_splits, vocab, args
        )
        for b in blocks:
            print(b)
            report_io.write(b)

        report_io.write("\n===== SUMMARY (avg over test splits) =====\n")
        for k in ("acc", "f1", "f1_ai", "auc"):
            vals = [all_metrics[s][k] for s in all_metrics]
            line = f"{k}: {float(np.mean(vals)):.4f} (min={float(np.min(vals)):.4f}, max={float(np.max(vals)):.4f})"
            print(line)
            report_io.write(line + "\n")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"cafe_mirage_{timestamp}.pt")
        torch.save(
            {
                "sim_state_dict": sim_module.state_dict(),
                "det_state_dict": det_module.state_dict(),
                "text_emb_state_dict": text_embed_module.state_dict(),
                "vocab": vocab,
                "config": vars(args),
                "metrics": all_metrics,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
            ckpt_path,
        )
    else:
        base_train = ds["train"]
        label_col = next((c for c in ("label", "labels", "fake") if c in base_train.column_names), None)
        if label_col is None:
            raise RuntimeError(f"Label column not found. columns={base_train.column_names}")
        labels = np.array([int(x) for x in base_train[label_col]])
        skf = StratifiedKFold(n_splits=args.kfold_splits, shuffle=True, random_state=args.seed)

        fold_results = []
        report_io.write(f"kfold_splits={args.kfold_splits}\n")
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
            print(f"\n[CAFE-MIRAGE] ===== Fold {fold_idx}/{args.kfold_splits} =====")
            report_io.write(f"\n\n{'#' * 70}\n### Fold {fold_idx}/{args.kfold_splits}\n{'#' * 70}\n")

            train_ds = MirageCAFEDataset(base_train.select(tr_idx.tolist()), vocab=vocab, max_len=SEQ_LEN)
            val_ds = MirageCAFEDataset(base_train.select(va_idx.tolist()), vocab=vocab, max_len=SEQ_LEN)
            sim_module, det_module, text_embed_module, img_encoder, best_f1 = train_one_run(
                train_ds, val_ds, vocab, args
            )
            report_io.write(f"best_val_macro_f1={best_f1:.4f}\n")

            blocks, fold_metrics = evaluate_all_test_splits(
                sim_module, det_module, text_embed_module, img_encoder, ds, test_splits, vocab, args
            )
            for b in blocks:
                print(b)
                report_io.write(b)
            fold_results.append(fold_metrics)

            del sim_module, det_module, text_embed_module, img_encoder
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

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"cafe_mirage_kfold{args.kfold_splits}_{timestamp}.pt")
        torch.save(
            {
                "config": vars(args),
                "vocab": vocab,
                "fold_metrics": fold_results,
                "global_metrics": global_metrics,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
            ckpt_path,
        )

    report_path = os.path.join(RESULT_DIR, f"cafe_mirage_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_io.getvalue())

    print(f"[CAFE-MIRAGE] checkpoint: {ckpt_path}")
    print(f"[CAFE-MIRAGE] report: {report_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="CAFE for MiRAGe-News (binary)")
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
    print(f"CAFE-MIRAGE | device={DEVICE}")
    train_and_test(args)
