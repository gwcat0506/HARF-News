"""
HAMMER × DGM4 — 이진(real/fake) 탐지 기본 + 논문 train.yaml 하이퍼
================================================================
**옵션**  
- `--weighted_sampler` : class-balanced 샘플러 (공식 train 은 shuffle)  
- `--albef_pretrained` : ALBEF_4M.pth 경로 (기본값 자동 지정)

[rshaojimmy/MultiModal-DeepFake](https://github.com/rshaojimmy/MultiModal-DeepFake)
"""

from __future__ import annotations

import argparse
import copy
import io
import math
import os
import random
import re
import sys
import warnings
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizerFast

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── 공식 코드 경로 (models.HAMMER, dataset.utils.pre_caption 등)
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "multimodal_deepfake")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from models.HAMMER import HAMMER  # noqa: E402
from optim.optim_factory import create_optimizer  # noqa: E402
from scheduler.scheduler_factory import create_scheduler  # noqa: E402
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label  # noqa: E402

from dgm4_paths import (  # noqa: E402
    DEFAULT_ALBEF_PATH,
    DEFAULT_DATA_ROOT,
    FINE_LABELS,
    load_dgm4_splits,
    parse_fake_cls,
    resolve_image_path,
)

LOG_TAG = "HAMMER-DGM4"

if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── `multimodal_deepfake/configs/train.yaml` + 사용자 지정 에폭/배치 ───────────
SEED = 42
BATCH = 16
EPOCHS = 10
NUM_WORKERS = 4
# 논문 train.yaml
PAPER_LR = 2e-5
PAPER_LR_IMG = 1e-4
PAPER_WEIGHT_DECAY = 0.02
PAPER_MIN_LR = 1e-6
PAPER_WARMUP_LR = 1e-6
PAPER_WARMUP_IN_50_EPOCHS = 10  # yaml schedular.warmup_epochs (total_epochs=50)
PAPER_SCHED_TOTAL_EPOCHS = 50   # yaml schedular.epochs
EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_DELTA = 1e-4
SAMPLER_ALPHA = 0.5

# ════════════════════════════════════════════════════════════════════════════════
# 유틸
# ════════════════════════════════════════════════════════════════════════════════

def default_paper_warmup_epochs(train_epochs: int) -> int:
    """50 epoch 학습에서 warmup 10 비율을 train_epochs 에 맞춤."""
    if train_epochs <= 1:
        return 0
    w = int(round(train_epochs * (PAPER_WARMUP_IN_50_EPOCHS / PAPER_SCHED_TOTAL_EPOCHS)))
    return max(1, min(train_epochs - 1, w))


def set_seed(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed()


def pre_caption(caption: str, max_words: int) -> str:
    """Official `dataset/utils.pre_caption` (avoids importing `dataset` package → cv2)."""
    caption = (
        re.sub(r"([,.'!?\"()*#:;~])", "", caption.lower())
        .replace("-", " ")
        .replace("/", " ")
        .replace("<person>", "person")
    )
    caption = re.sub(r"\s{2,}", " ", caption).rstrip("\n").strip(" ")
    words = caption.split(" ")
    if len(words) > max_words:
        caption = " ".join(words[:max_words])
    return caption


def make_weighted_sampler(
    binary_labels: List[int], alpha: float = 0.5
) -> WeightedRandomSampler:
    s = pd.Series(binary_labels)
    cnt = s.value_counts().to_dict()
    w = [(1.0 / cnt[l]) ** alpha for l in binary_labels]
    return WeightedRandomSampler(w, len(w), replacement=True)


# ════════════════════════════════════════════════════════════════════════════════
# Early Stopping
# ════════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(
        self,
        patience: int = EARLY_STOP_PATIENCE,
        min_delta: float = EARLY_STOP_MIN_DELTA,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0

    def step(self, v: float) -> bool:
        if self.best is None:
            self.best = v
            return False
        imp = (
            v - self.best > self.min_delta
            if self.mode == "max"
            else self.best - v > self.min_delta
        )
        if imp:
            self.best = v
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ════════════════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════════════════

class DGM4HammerDataset(Dataset):
    """..."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        transform,
        max_words: int,
        is_train: bool,
        limit: int = 0,
    ):
        super().__init__()
        self.data_root = os.path.abspath(data_root)
        self.transform = transform
        self.max_words = max_words
        self.image_res = int(transform.__dict__.get("_image_res", 256))
        self.is_train = is_train

        if limit and limit > 0:
            df = df.head(limit).reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def _get_bbox_xywh(self, row) -> Optional[Tuple[int, int, int, int]]:
        if "fake_image_box" not in row.index:
            return None
        box = row["fake_image_box"]
        if box is None or (isinstance(box, float) and (math.isnan(box) or np.isnan(box))):
            return None
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            return None
        xmin, ymin, xmax, ymax = [float(x) for x in box[:4]]
        w, h = xmax - xmin, ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        rel = row["image"]
        full = resolve_image_path(rel, self.data_root)

        try:
            image = Image.open(full).convert("RGB") if full and os.path.isfile(full) else None
        except Exception:
            image = None
        if image is None:
            image = Image.new("RGB", (self.image_res, self.image_res), (127, 127, 127))

        W, H = image.size
        xywh = self._get_bbox_xywh(row)
        has_bbox = xywh is not None

        fake_image_box: torch.Tensor
        do_hflip = False
        if self.is_train:
            if random.random() < 0.5:
                image = transforms.functional.hflip(image)
                do_hflip = True
            image = transforms.functional.resize(
                image, [self.image_res, self.image_res], interpolation=Image.BICUBIC
            )

        if has_bbox and xywh is not None:
            x, y, w, h = xywh
            if do_hflip:
                x = (W - x) - w
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h
            cx = x + 0.5 * w
            cy = y + 0.5 * h
            fake_image_box = torch.tensor(
                [
                    cx / self.image_res,
                    cy / self.image_res,
                    w / self.image_res,
                    h / self.image_res,
                ],
                dtype=torch.float32,
            )
        else:
            fake_image_box = torch.zeros(4, dtype=torch.float32)

        image = self.transform(image)

        label = str(row["fake_cls"])
        caption = pre_caption(row["text"], self.max_words)

        ftp = row.get("fake_text_pos", [])
        if ftp is None or (isinstance(ftp, float) and np.isnan(ftp)):
            ftp = []
        if not isinstance(ftp, (list, tuple)):
            ftp = []

        fake_text_pos_list = torch.zeros(self.max_words)
        for i in ftp:
            if isinstance(i, (int, float)) and int(i) < self.max_words:
                fake_text_pos_list[int(i)] = 1.0

        return image, label, caption, fake_image_box, fake_text_pos_list, W, H


# ════════════════════════════════════════════════════════════════════════════════
# Transforms
# ════════════════════════════════════════════════════════════════════════════════

def build_transforms(image_res: int, is_train: bool):
    """공식 `configs/train.yaml` / `dataset`"""
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    if is_train:
        from dataset.randaugment import RandomAugment  # noqa: WPS433

        tr = transforms.Compose(
            [
                RandomAugment(
                    2,
                    7,
                    isPIL=True,
                    augs=["Identity", "AutoContrast", "Equalize", "Brightness", "Sharpness"],
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        tr = transforms.Compose(
            [
                transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ]
        )
    tr._image_res = image_res  # type: ignore[attr-defined]
    return tr


# ════════════════════════════════════════════════════════════════════════════════
# Text 전처리
# ════════════════════════════════════════════════════════════════════════════════

def text_input_adjust(text_input, fake_word_pos, device):
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max(len(x) for x in text_input.input_ids) - 1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP]
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device)

    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    fake_token_pos_batch = []
    for bi in range(len(fake_word_pos)):
        fake_token_pos = []
        fake_word_pos_decimal = np.where(fake_word_pos[bi].numpy() == 1)[0].tolist()
        subword_idx = text_input.word_ids(bi)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP)
        for wi in fake_word_pos_decimal:
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == wi)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch


# ════════════════════════════════════════════════════════════════════════════════
# Model 빌드  ★ 개선: ALBEF 사전학습 가중치 로드
# ════════════════════════════════════════════════════════════════════════════════

def hammer_config_dict(image_res: int, use_bbox: bool, binary_only: bool) -> Dict[str, Any]:
    bert_cfg = os.path.join(_VENDOR, "configs", "config_bert.json")
    return {
        "embed_dim": 256,
        "image_res": image_res,
        "vision_width": 768,
        "queue_size": 65536,
        "momentum": 0.995,
        "temp": 0.07,
        "label_smoothing": 0.0,
        "bert_config": bert_cfg,
        "use_bbox": use_bbox,
        "binary_only": binary_only,
    }


def interpolate_pos_embed(pos_embed: torch.Tensor, new_num_tokens: int) -> torch.Tensor:
    """
    ALBEF visual_encoder.pos_embed (1, 197, 768) →  (1, new_num_tokens, 768)
    image_res=256 → 패치 수 (256/16)^2 + 1 = 257
    """
    cls_pos   = pos_embed[:, :1, :]                       # (1, 1, 768)
    patch_pos = pos_embed[:, 1:, :]                        # (1, 196, 768)
    orig_size = int(patch_pos.shape[1] ** 0.5)             # 14
    new_size  = int((new_num_tokens - 1) ** 0.5)           # 16

    patch_pos = (
        patch_pos
        .reshape(1, orig_size, orig_size, -1)
        .permute(0, 3, 1, 2)                               # (1, 768, 14, 14)
    )
    patch_pos = F.interpolate(
        patch_pos, size=(new_size, new_size),
        mode="bicubic", align_corners=False,
    )                                                       # (1, 768, 16, 16)
    patch_pos = (
        patch_pos
        .permute(0, 2, 3, 1)
        .reshape(1, -1, pos_embed.shape[-1])               # (1, 256, 768)
    )
    return torch.cat([cls_pos, patch_pos], dim=1)           # (1, 257, 768)


def build_model(
    tokenizer,
    image_res: int,
    use_bbox: bool,
    binary_only: bool,
    albef_pretrained: str = DEFAULT_ALBEF_PATH,
) -> HAMMER:
    """
    HAMMER 모델 생성 후 ALBEF 사전학습 가중치를 로드한다.

    정상 로드 시 로그 예시:
        [HAMMER-DGM4] ALBEF pretrained 로드: .../ALBEF_4M.pth
        [HAMMER-DGM4]   pos_embed 보간: torch.Size([1, 257, 768])
        [HAMMER-DGM4]   missing=8  unexpected=3
    missing 이 5~15 개이면 정상 (HAMMER 전용 head·proj 들).
    30 개 이상이면 key 불일치 → 알려주세요.
    """
    cfg   = hammer_config_dict(image_res, use_bbox, binary_only)
    margs = SimpleNamespace(token_momentum=False)
    model = HAMMER(
        args=margs,
        config=cfg,
        text_encoder="bert-base-uncased",
        tokenizer=tokenizer,
        init_deit=True,
    )

    # ── ALBEF 사전학습 가중치 로드 ────────────────────────────────────────────
    if albef_pretrained and os.path.isfile(albef_pretrained):
        print(f"[{LOG_TAG}] ALBEF pretrained 로드: {albef_pretrained}")
        ckpt       = torch.load(albef_pretrained, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)

        # ① visual_encoder.pos_embed 크기 보정 (196 → image_res 기반 패치 수)
        new_num_patches = (image_res // 16) ** 2 + 1  # 256 → 257
        pe_key = "visual_encoder.pos_embed"
        if pe_key in state_dict and state_dict[pe_key].shape[1] != new_num_patches:
            pe = interpolate_pos_embed(state_dict[pe_key], new_num_patches)
            state_dict[pe_key] = pe
            print(f"[{LOG_TAG}]   pos_embed 보간: {pe.shape}")

        for k in list(state_dict.keys()):
            if k.startswith("visual_encoder."):
                mk = k.replace("visual_encoder.", "visual_encoder_m.")
                state_dict[mk] = state_dict[k].clone()
            elif k.startswith("text_encoder."):
                mk = k.replace("text_encoder.", "text_encoder_m.")
                state_dict[mk] = state_dict[k].clone()

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[{LOG_TAG}]   missing={len(missing)}  unexpected={len(unexpected)}")

        if len(missing) > 30:
            print(f"[{LOG_TAG}] ⚠️  missing 이 너무 많음 ({len(missing)}개) — key 불일치 확인 필요")
            for mk in missing[:10]:
                print(f"       missing: {mk}")
    else:
        print(
            f"[{LOG_TAG}] ⚠️  ALBEF pretrained 파일 없음: {albef_pretrained}\n"
            f"           → cross-modal 인코더가 랜덤 초기화 상태입니다 (성능 심각하게 저하)."
        )

    return model


# ════════════════════════════════════════════════════════════════════════════════
# 평가  ★ 개선: Youden's J 최적 임계값 F1 추가
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_split(
    model: HAMMER,
    loader: DataLoader,
    tokenizer: BertTokenizerFast,
    device: torch.device,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    y_true: List[float] = []
    y_pred: List[float] = []
    y_pred_cls: List[int] = []
    IOU_pred: List[float] = []
    IOU_50: List[int] = []
    IOU_75: List[int] = []
    IOU_95: List[int] = []

    cls_nums_all = 0
    cls_acc_all  = 0
    TP_all = TN_all = FP_all = FN_all = 0

    TP_all_multicls = np.zeros(4, dtype=int)
    TN_all_multicls = np.zeros(4, dtype=int)
    FP_all_multicls = np.zeros(4, dtype=int)
    FN_all_multicls = np.zeros(4, dtype=int)

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    bin_only = getattr(model, "binary_only", False)

    for bi, (image, label, text, fake_image_box, fake_word_pos, _W, _H) in enumerate(
        tqdm(loader, desc="eval", file=sys.stderr)
    ):
        if max_batches and bi >= max_batches:
            break

        image         = image.to(device, non_blocking=True)
        fake_image_box = fake_image_box.to(device, non_blocking=True)

        text_input = tokenizer(
            text,
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

        logits_real_fake, logits_multicls, output_coord, logits_tok = model(
            image, label, text_input, fake_image_box, fake_token_pos, is_train=False
        )

        cls_label = torch.ones(len(label), dtype=torch.long, device=device)
        real_label_pos = np.where(np.array(label) == "orig")[0].tolist()
        cls_label[real_label_pos] = 0

        y_pred.extend(F.softmax(logits_real_fake, dim=1)[:, 1].cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())

        pred_acc = logits_real_fake.argmax(1)
        cls_nums_all += cls_label.shape[0]
        cls_acc_all  += torch.sum(pred_acc == cls_label).item()
        y_pred_cls.extend(pred_acc.cpu().flatten().tolist())

        if not bin_only and logits_multicls is not None:
            target, _ = get_multi_label(label, image)
            multi_label_meter.add(logits_multicls, target)

            for cls_idx in range(logits_multicls.shape[1]):
                cls_pred = logits_multicls[:, cls_idx].clone()
                cls_pred[cls_pred >= 0] = 1
                cls_pred[cls_pred < 0]  = 0
                TP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 1)).item()
                TN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 0)).item()
                FP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 1)).item()
                FN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 0)).item()

        if output_coord is not None:
            from models import box_ops  # noqa: WPS433

            boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
            boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)
            IOU, _ = box_ops.box_iou(boxes1, boxes2, test=True)

            IOU_pred.extend(IOU.cpu().tolist())
            IOU_50.extend((IOU > 0.50).long().cpu().tolist())
            IOU_75.extend((IOU > 0.75).long().cpu().tolist())
            IOU_95.extend((IOU > 0.95).long().cpu().tolist())

        if not bin_only and logits_tok is not None:
            token_label = text_input.attention_mask[:, 1:].clone()
            token_label[token_label == 0] = -100
            token_label[token_label == 1] = 0
            for batch_idx in range(len(fake_token_pos)):
                for pos in fake_token_pos[batch_idx]:
                    token_label[batch_idx, pos] = 1

            logits_tok_reshape = logits_tok.view(-1, 2)
            logits_tok_pred    = logits_tok_reshape.argmax(1)
            token_label_reshape = token_label.view(-1)
            valid = token_label_reshape != -100
            TP_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1) * valid).item()
            TN_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0) * valid).item()
            FP_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1) * valid).item()
            FN_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0) * valid).item()

    # ── 이진 분류 지표 ─────────────────────────────────────────────────────────
    y_true_a = np.array(y_true)
    y_pred_a = np.array(y_pred)
    y_pc     = np.array(y_pred_cls, dtype=np.int64)

    try:
        AUC_cls = float(roc_auc_score(y_true_a, y_pred_a))
    except ValueError:
        AUC_cls = float("nan")

    ACC_cls = cls_acc_all / max(cls_nums_all, 1)

    # 고정 threshold=0.5 F1
    try:
        F1_cls = float(f1_score(y_true_a, y_pc, average="binary", pos_label=1, zero_division=0))
    except ValueError:
        F1_cls = float("nan")

    # ★ Youden's J (TPR-FPR 최대화) 최적 임계값 F1
    F1_cls_opt  = float("nan")
    best_thresh = float("nan")
    try:
        fpr, tpr, thresholds = roc_curve(y_true_a, y_pred_a)
        j_scores   = tpr - fpr
        best_idx   = int(np.argmax(j_scores))
        best_thresh = float(thresholds[best_idx])
        y_pc_opt   = (y_pred_a >= best_thresh).astype(int)
        F1_cls_opt = float(f1_score(y_true_a, y_pc_opt, average="binary", pos_label=1, zero_division=0))
    except Exception:
        pass

    # ── 멀티태스크 지표 ────────────────────────────────────────────────────────
    MAP = float("nan")
    if not bin_only:
        try:
            MAP = multi_label_meter.value().mean().item()
        except ZeroDivisionError:
            MAP = float("nan")

    denom_tok    = TP_all + TN_all + FP_all + FN_all
    ACC_tok      = (TP_all + TN_all) / max(denom_tok, 1)
    Precision_tok = TP_all / max(TP_all + FP_all, 1)
    Recall_tok   = TP_all / max(TP_all + FN_all, 1)
    F1_tok       = 2 * Precision_tok * Recall_tok / max(Precision_tok + Recall_tok, 1e-8)

    # ── 결과 dict ─────────────────────────────────────────────────────────────
    out: Dict[str, float] = {
        "AUC_cls":    AUC_cls,
        "ACC_cls":    ACC_cls,
        "F1_cls":     F1_cls,       # threshold=0.5
        "F1_cls_opt": F1_cls_opt,   # ★ Youden's J 최적 임계값
        "best_thresh": best_thresh, # ★ 최적 임계값 값
    }
    if not bin_only:
        out["MAP"]     = MAP
        out["ACC_tok"] = float(ACC_tok)
        out["F1_tok"]  = float(F1_tok)
    if IOU_pred:
        out["IOU_mean"] = float(sum(IOU_pred) / len(IOU_pred))
        out["IOU@50"]   = float(sum(IOU_50)   / len(IOU_50))
        out["IOU@75"]   = float(sum(IOU_75)   / len(IOU_75))
        out["IOU@95"]   = float(sum(IOU_95)   / len(IOU_95))
    return out


# ════════════════════════════════════════════════════════════════════════════════
# 스케줄러
# ════════════════════════════════════════════════════════════════════════════════

def build_paper_cosine_scheduler(
    optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int
):
    """`train.py` + `configs/train.yaml` 의 CosineLRScheduler."""
    sche_args = SimpleNamespace(
        sched="cosine",
        epochs=total_epochs,
        min_lr=PAPER_MIN_LR,
        decay_rate=1,
        warmup_lr=PAPER_WARMUP_LR,
        warmup_epochs=warmup_epochs,
        cooldown_epochs=0,
    )
    lr_scheduler, _ = create_scheduler(sche_args, optimizer)
    return lr_scheduler


# ════════════════════════════════════════════════════════════════════════════════
# 학습  ★ 개선: warmup 기간 loss_MAC 비활성
# ════════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: HAMMER,
    train_loader: DataLoader,
    tokenizer: BertTokenizerFast,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    yaml_cfg: Dict[str, Any],
    epoch_idx: int,
    alpha: float,
    warmup_epochs: int = 2,   # ★ warmup 동안 MAC 비활성에 사용
) -> float:
    model.train()
    loss_weights = {
        "MAC":  yaml_cfg["loss_MAC_wgt"],
        "BIC":  yaml_cfg["loss_BIC_wgt"],
        "bbox": yaml_cfg["loss_bbox_wgt"],
        "giou": yaml_cfg["loss_giou_wgt"],
        "TMG":  yaml_cfg["loss_TMG_wgt"],
        "MLC":  yaml_cfg["loss_MLC_wgt"],
    }

    # ★ warmup 기간: 모멘텀 큐가 랜덤 피처로 가득 → MAC loss 노이즈 차단
    mac_wgt = 0.0 if epoch_idx <= warmup_epochs else loss_weights["MAC"]

    tot = 0.0
    n   = 0
    pbar = tqdm(train_loader, desc=f"train ep{epoch_idx}", file=sys.stderr)
    for i, (image, label, text, fake_image_box, fake_word_pos, _w, _h) in enumerate(pbar):
        image          = image.to(device, non_blocking=True)
        fake_image_box = fake_image_box.to(device, non_blocking=True)

        text_input = tokenizer(
            text,
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

        # alpha 워밍업 (1 epoch 안에서)
        a = alpha if epoch_idx > 1 else alpha * min(1.0, (i + 1) / max(len(train_loader), 1))

        loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC = model(
            image, label, text_input, fake_image_box, fake_token_pos, alpha=a
        )

        loss = (
            mac_wgt                  * loss_MAC
            + loss_weights["BIC"]  * loss_BIC
            + loss_weights["bbox"] * loss_bbox
            + loss_weights["giou"] * loss_giou
            + loss_weights["TMG"]  * loss_TMG
            + loss_weights["MLC"]  * loss_MLC
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs   = image.size(0)
        tot += loss.item() * bs
        n   += bs
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            mac_wgt=f"{mac_wgt:.2f}",
        )

    return tot / max(n, 1)


def train_hammer_run(
    model: HAMMER,
    train_loader: DataLoader,
    tokenizer: BertTokenizerFast,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    yaml_cfg: Dict[str, Any],
    args: Any,
    val_loader: Optional[DataLoader] = None,
    max_eval_batches: int = 0,
) -> None:
    """
    공식 `train.py`: CosineLRScheduler + val AUC best 복원 + early stopping.
    `--no_epoch_val` 시 val·early stop 생략.
    """
    lr_sched: Optional[Any] = None
    if args.epochs > 0:
        lr_sched = build_paper_cosine_scheduler(optimizer, args.epochs, args.warmup_epochs)

    early = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        mode="max",
    )
    best_auc   = -1.0
    best_state: Optional[Dict[str, Any]] = None

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            tokenizer,
            device,
            optimizer,
            yaml_cfg,
            ep,
            args.alpha,
            warmup_epochs=args.warmup_epochs,   # ★ 전달
        )

        auc_v = f1_v = acc_v = float("nan")
        if val_loader is not None:
            model.eval()
            vm = evaluate_split(
                model, val_loader, tokenizer, device, max_batches=max_eval_batches
            )
            model.train()
            auc_v = vm.get("AUC_cls",    float("nan"))
            acc_v = vm.get("ACC_cls",    float("nan"))
            f1_v  = vm.get("F1_cls_opt", float("nan"))  # ★ 최적 임계값 F1 로 모니터링

            if not math.isnan(auc_v) and auc_v > best_auc:
                best_auc   = auc_v
                best_state = copy.deepcopy(model.state_dict())

            monitor = auc_v if not math.isnan(auc_v) else f1_v
            if not math.isnan(monitor) and early.step(monitor):
                if lr_sched is not None:
                    lr_sched.step((ep - 1) + args.warmup_epochs + 1)
                print(f"[{LOG_TAG}] Early stopping at epoch {ep} (best_auc={best_auc:.4f})")
                break

        if lr_sched is not None:
            lr_sched.step((ep - 1) + args.warmup_epochs + 1)
        cur_lr = optimizer.param_groups[-1]["lr"]

        if val_loader is not None:
            print(
                f"[{LOG_TAG}] Ep {ep:02d}/{args.epochs}"
                f"  tr_loss={tr_loss:.4f}"
                f"  va_f1_opt={f1_v:.4f}"
                f"  va_auc={auc_v:.4f}"
                f"  va_acc={acc_v:.4f}"
                f"  lr={cur_lr:.2e}"
                f"  mac_on={'No' if ep <= args.warmup_epochs else 'Yes'}"
            )
        else:
            print(
                f"[{LOG_TAG}] Ep {ep:02d}/{args.epochs}"
                f"  tr_loss={tr_loss:.4f}"
                f"  lr={cur_lr:.2e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{LOG_TAG}] best val AUC={best_auc:.4f} 가중치 복원 완료")


# ════════════════════════════════════════════════════════════════════════════════
# 체크포인트 I/O
# ════════════════════════════════════════════════════════════════════════════════

def load_checkpoint(model: HAMMER, path: str, device: torch.device) -> None:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[{LOG_TAG}] load_checkpoint strict=False  "
        f"missing={len(missing)}  unexpected={len(unexpected)}"
    )


def save_checkpoint(model: HAMMER, path: str, args_for_meta: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = vars(args_for_meta) if hasattr(args_for_meta, "__dict__") else args_for_meta
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": meta,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
        path,
    )
    print(f"[{LOG_TAG}] checkpoint saved: {path}")


# ════════════════════════════════════════════════════════════════════════════════
# DataLoader 헬퍼
# ════════════════════════════════════════════════════════════════════════════════

def make_eval_loader(
    dfs: Dict[str, pd.DataFrame],
    split_key: str,
    data_root: str,
    te_transform,
    max_words: int,
    limit: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, int]:
    df = dfs[split_key]
    ds = DGM4HammerDataset(
        df, data_root, te_transform,
        max_words=max_words, is_train=False, limit=limit,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, len(ds)


def run_eval_report(
    model: HAMMER,
    ev_loader: DataLoader,
    n_samples: int,
    split_name: str,
    tag: str,
    tokenizer: BertTokenizerFast,
    device: torch.device,
    use_bbox: bool,
    result_dir: str,
    max_eval_batches: int,
) -> Dict[str, float]:
    bbox_note  = "with_bbox" if use_bbox else "no_bbox"
    mode_note  = []
    if getattr(model, "binary_only", False):
        mode_note.append("binary_ITM")
    if getattr(model, "use_bbox", False):
        mode_note.append("bbox_on")
    else:
        mode_note.append("no_bbox")
    extra = " ".join(mode_note)
    print(
        f"\n{LOG_TAG} | device={device} | split={split_name} | "
        f"n={n_samples} | {tag} | {bbox_note} | {extra}"
    )
    m = evaluate_split(model, ev_loader, tokenizer, device, max_batches=max_eval_batches)
    buf = io.StringIO()
    for k, v in m.items():
        line = f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}"
        print(line)
        buf.write(line + "\n")
    os.makedirs(result_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = tag.replace(" ", "_")
    rp = os.path.join(result_dir, f"hammer_dgm4_{split_name}_{safe_tag}_{ts}.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"[{LOG_TAG}] saved: {rp}")
    return m


# ════════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    pa = argparse.ArgumentParser(
        description=(
            "HAMMER × DGM4 — 기본 이진(real/fake) 탐지, "
            "논문 train.yaml LR/WD/스케줄/증강 (ep=10 bs=16)"
        )
    )
    pa.add_argument(
        "--full_multitask", action="store_true",
        help="논문 전체: MAC·BIC·MLC·TMG·bbox(+GIoU). 지정하지 않으면 이진 ITM(BIC)만.",
    )
    pa.add_argument(
        "--no_bbox", action="store_true",
        help="--full_multitask 일 때만 의미 있음: bbox 모듈 끔",
    )
    pa.add_argument(
        "--weighted_sampler", action="store_true",
        help="train 에 WeightedRandomSampler 사용 (공식 train 은 shuffle)",
    )
    pa.add_argument("--data_root",      type=str, default=DEFAULT_DATA_ROOT)
    pa.add_argument("--image_res",      type=int, default=256)
    pa.add_argument("--max_words",      type=int, default=50)
    pa.add_argument("--batch_size",     type=int, default=BATCH)
    pa.add_argument("--num_workers",    type=int, default=NUM_WORKERS)
    pa.add_argument("--epochs",         type=int, default=EPOCHS)
    pa.add_argument("--eval_only",      action="store_true")
    pa.add_argument("--checkpoint",     type=str, default="")
    pa.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    pa.add_argument("--limit",          type=int, default=0)
    pa.add_argument("--max_eval_batches", type=int, default=0)
    pa.add_argument(
        "--split", type=str, default="test",
        choices=("train", "validation", "test"),
    )
    pa.add_argument("--skip_post_val",       action="store_true")
    pa.add_argument("--skip_post_test",      action="store_true")
    pa.add_argument("--no_epoch_val",        action="store_true")
    pa.add_argument("--eval_before_train",   action="store_true")
    pa.add_argument(
        "--eval_before_split", type=str, default="validation",
        choices=("train", "validation", "test"),
    )
    pa.add_argument("--seed",           type=int,   default=SEED)
    pa.add_argument("--lr",             type=float, default=PAPER_LR)
    pa.add_argument("--lr_img",         type=float, default=PAPER_LR_IMG)
    pa.add_argument("--weight_decay",   type=float, default=PAPER_WEIGHT_DECAY)
    pa.add_argument("--warmup_epochs",  type=int,   default=-1)
    pa.add_argument("--early_stop_patience",  type=int,   default=EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    pa.add_argument("--sampler_alpha",  type=float, default=SAMPLER_ALPHA)
    pa.add_argument("--alpha",          type=float, default=0.4)
    pa.add_argument("--loss_MAC_wgt",   type=float, default=0.1)
    pa.add_argument("--loss_BIC_wgt",   type=float, default=1.0)
    pa.add_argument("--loss_bbox_wgt",  type=float, default=0.1)
    pa.add_argument("--loss_giou_wgt",  type=float, default=0.1)
    pa.add_argument("--loss_TMG_wgt",   type=float, default=1.0)
    pa.add_argument("--loss_MLC_wgt",   type=float, default=1.0)
    pa.add_argument("--result_dir",     type=str,   default="./results")
    pa.add_argument(
        "--albef_pretrained", type=str, default=DEFAULT_ALBEF_PATH,
        help="ALBEF_4M.pth 경로 (기본값: DEFAULT_ALBEF_PATH)",
    )

    args = pa.parse_args()
    set_seed(args.seed)

    if args.warmup_epochs < 0:
        args.warmup_epochs = default_paper_warmup_epochs(args.epochs)
    if args.epochs > 1 and args.warmup_epochs > args.epochs - 1:
        args.warmup_epochs = args.epochs - 1

    # HAMMER_bbox.py: 항상 binary + bbox 고정
    binary_only = True
    use_bbox    = True

    # binary이므로 MAC/TMG/MLC 끔, bbox/giou는 유지
    args.loss_MAC_wgt = 0.0
    args.loss_MLC_wgt = 0.0
    args.loss_TMG_wgt = 0.0

    yaml_cfg = {
        "loss_MAC_wgt":  args.loss_MAC_wgt,
        "loss_BIC_wgt":  args.loss_BIC_wgt,
        "loss_bbox_wgt": args.loss_bbox_wgt,
        "loss_giou_wgt": args.loss_giou_wgt,
        "loss_TMG_wgt":  args.loss_TMG_wgt,
        "loss_MLC_wgt":  args.loss_MLC_wgt,
    }

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = build_model(
        tokenizer, args.image_res,
        use_bbox=use_bbox, binary_only=binary_only,
        albef_pretrained=args.albef_pretrained,     # ★
    ).to(DEVICE)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, DEVICE)

    dfs          = load_dgm4_splits(args.data_root, LOG_TAG)
    te_transform = build_transforms(args.image_res, is_train=False)

    # ── 평가만 ───────────────────────────────────────────────────────────────
    if args.eval_only:
        if not args.checkpoint:
            print(
                f"[{LOG_TAG}] 경고: --checkpoint 없음 → BERT/헤드 초기화 상태 (지표는 참고용)"
            )
        ev_loader, n_ev = make_eval_loader(
            dfs, args.split, args.data_root, te_transform,
            args.max_words, args.limit, args.batch_size, args.num_workers,
        )
        run_eval_report(
            model, ev_loader, n_ev, args.split, "eval_only",
            tokenizer, DEVICE, use_bbox, args.result_dir, args.max_eval_batches,
        )
        return

    # ── 학습 전 평가 (기준선) ─────────────────────────────────────────────────
    if args.eval_before_train:
        be_loader, n_be = make_eval_loader(
            dfs, args.eval_before_split, args.data_root, te_transform,
            args.max_words, args.limit, args.batch_size, args.num_workers,
        )
        run_eval_report(
            model, be_loader, n_be, args.eval_before_split, "before_train",
            tokenizer, DEVICE, use_bbox, args.result_dir, args.max_eval_batches,
        )

    # ── 학습 ─────────────────────────────────────────────────────────────────
    train_df = (
        dfs["train"] if not args.limit
        else dfs["train"].head(args.limit).reset_index(drop=True)
    )
    tr_transform = build_transforms(args.image_res, is_train=True)
    tr_ds = DGM4HammerDataset(
        train_df, args.data_root, tr_transform,
        max_words=args.max_words, is_train=True, limit=0,
    )
    tr_kw: Dict[str, Any] = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if args.weighted_sampler:
        tr_sampler = make_weighted_sampler(train_df["binary_label"].tolist(), args.sampler_alpha)
        tr_kw["sampler"] = tr_sampler
        tr_kw["shuffle"] = False
    else:
        tr_kw["shuffle"] = True
    tr_loader = DataLoader(tr_ds, **tr_kw)

    val_loader_ep: Optional[DataLoader] = None
    if not args.no_epoch_val:
        val_loader_ep, _ = make_eval_loader(
            dfs, "validation", args.data_root, te_transform,
            args.max_words, args.limit, args.batch_size, args.num_workers,
        )

    opt_ns = SimpleNamespace(
        opt="adamW", lr=args.lr, lr_img=args.lr_img,
        weight_decay=args.weight_decay, momentum=0.9,
    )
    optimizer = create_optimizer(opt_ns, model)
    sm = (
        f" weighted_sampler α={args.sampler_alpha}"
        if args.weighted_sampler else " shuffle=True"
    )
    print(
        f"[{LOG_TAG}] train | multitask={not binary_only} bbox={use_bbox} | "
        f"lr={args.lr:.2e} lr_img={args.lr_img:.2e} wd={args.weight_decay} "
        f"warmup_ep={args.warmup_epochs} sched_cosine_t={args.epochs}{sm}"
    )
    train_hammer_run(
        model, tr_loader, tokenizer, DEVICE, optimizer,
        yaml_cfg, args,
        val_loader=val_loader_ep, max_eval_batches=args.max_eval_batches,
    )

    # ── 체크포인트 저장 ───────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(args.checkpoint_dir, f"hammer_dgm4_{ts}.pt")
    save_checkpoint(model, ckpt_path, args)

    # ── 학습 후 평가 ─────────────────────────────────────────────────────────
    if not args.skip_post_val:
        val_loader, n_val = make_eval_loader(
            dfs, "validation", args.data_root, te_transform,
            args.max_words, args.limit, args.batch_size, args.num_workers,
        )
        run_eval_report(
            model, val_loader, n_val, "validation", "after_train",
            tokenizer, DEVICE, use_bbox, args.result_dir, args.max_eval_batches,
        )

    if not args.skip_post_test:
        tst_loader, n_tst = make_eval_loader(
            dfs, "test", args.data_root, te_transform,
            args.max_words, args.limit, args.batch_size, args.num_workers,
        )
        run_eval_report(
            model, tst_loader, n_tst, "test", "after_train",
            tokenizer, DEVICE, use_bbox, args.result_dir, args.max_eval_batches,
        )


if __name__ == "__main__":
    main()