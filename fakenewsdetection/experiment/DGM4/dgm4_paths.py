"""
DGM4 메타데이터·이미지 경로 공통 로직.

데이터 루트 우선순위:
  1. 환경변수 DGM4_DATA_ROOT
  2. experiment/DGM4/datasets/DGM4 (README 권장 로컬 배치)
  3. 리포지토리 상위 datasets/DGM4 (metadata/train.json 존재 시)
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd

_DGM4_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_DGM4_DIR, "multimodal_deepfake")

FINE_LABELS = ["face_swap", "face_attribute", "text_swap", "text_attribute"]


def get_default_data_root() -> str:
    """DGM4 데이터셋 루트 (manipulation/, origin/, metadata/ 가 바로 아래)."""
    env = os.environ.get("DGM4_DATA_ROOT", "").strip()
    if env:
        return os.path.abspath(env)

    candidates = [
        os.path.join(_DGM4_DIR, "datasets", "DGM4"),
        os.path.abspath(
            os.path.join(_DGM4_DIR, "..", "..", "..", "..", "datasets", "DGM4")
        ),
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "metadata", "train.json")):
            return path
    return candidates[0]


def get_default_albef_path() -> str:
    """공식 HAMMER 사전학습 가중치 ALBEF_4M.pth."""
    env = os.environ.get("DGM4_ALBEF_PATH", "").strip()
    if env:
        return os.path.abspath(env)
    return os.path.join(_VENDOR_DIR, "ALBEF_4M.pth")


# argparse default 용 (import 시점 기준)
DEFAULT_DATA_ROOT = get_default_data_root()
DEFAULT_ALBEF_PATH = get_default_albef_path()


def parse_fake_cls(cls_str: str) -> Tuple[int, List[int]]:
    """DGM4 fake_cls → 이진 라벨 + fine 멀티라벨."""
    cs = cls_str.strip().lower()
    if cs == "orig":
        return 0, [0, 0, 0, 0]
    parts = cs.split("&")
    fine = [int(lbl in parts) for lbl in FINE_LABELS]
    return 1, fine


def resolve_image_path(rel: str, data_root: str) -> str:
    """
    metadata 의 상대 경로를 디스크 절대 경로로 해석.
    - data_root = manipulation·origin 이 바로 아래에 있는 DGM4 루트
    - 메타에 `DGM4/manipulation/...` 형태가 오면 `DGM4/` 접두사 제거 후 결합
    - 일부 레이아웃: manipulation/foo/bar.jpg → manipulation/foo/foo/bar.jpg 후보
    """
    rel = (rel or "").strip()
    if not rel:
        return ""
    if os.path.isabs(rel) and os.path.isfile(rel):
        return rel

    root = os.path.abspath(data_root)
    rel_norm = rel.replace("\\", "/").lstrip("/")

    candidates = [os.path.join(root, rel_norm)]

    if rel_norm.startswith("DGM4/"):
        rel_wo = rel_norm[len("DGM4/") :]
        candidates.append(os.path.join(root, rel_wo))
    else:
        rel_wo = rel_norm

    parts = rel_wo.split("/")
    if len(parts) >= 3 and parts[0] in {"manipulation", "origin"}:
        nested = "/".join([parts[0], parts[1], parts[1], *parts[2:]])
        candidates.append(os.path.join(root, nested))

    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return candidates[-1] if candidates else ""


def load_dgm4_splits(data_root: str, log_tag: str) -> Dict[str, pd.DataFrame]:
    """train/val/test JSON 로드, 이진·fine·has_image_flag 부여."""
    split_files = {
        "train": os.path.join(data_root, "metadata", "train.json"),
        "validation": os.path.join(data_root, "metadata", "val.json"),
        "test": os.path.join(data_root, "metadata", "test.json"),
    }
    dfs: Dict[str, pd.DataFrame] = {}
    for split, path in split_files.items():
        if os.path.isfile(path):
            print(f"[{log_tag}] 로컬 로드: {path}")
            dfs[split] = pd.read_json(path)
        else:
            print(f"[{log_tag}] HuggingFace에서 로드: {split}")
            hf = {
                "train": "metadata/train.json",
                "validation": "metadata/val.json",
                "test": "metadata/test.json",
            }
            dfs[split] = pd.read_json("hf://datasets/rshaojimmy/DGM4/" + hf[split])

        df = dfs[split]
        df["text"] = df["text"].fillna("").astype(str).str.strip()
        df["image"] = df["image"].fillna("").astype(str).str.strip()
        df["fake_cls"] = df["fake_cls"].fillna("orig").astype(str).str.strip()

        def _has(rel: str) -> bool:
            full = resolve_image_path(rel, data_root)
            return bool(full) and os.path.isfile(full)

        df["has_image_flag"] = df["image"].map(_has)
        parsed = df["fake_cls"].map(parse_fake_cls)
        df["binary_label"] = parsed.map(lambda x: x[0])
        for i, lbl in enumerate(FINE_LABELS):
            df[f"fine_{lbl}"] = parsed.map(lambda x, i=i: x[1][i])

        n_total = len(df)
        n_img = int(df["has_image_flag"].sum())
        n_real = int((df["binary_label"] == 0).sum())
        n_fake = int((df["binary_label"] == 1).sum())
        print(
            f"[{log_tag}] {split}: {n_total}행  "
            f"이미지 유효={n_img}  real={n_real}  fake={n_fake}"
        )
        if n_total > 0 and n_img == 0:
            s0 = str(df["image"].iloc[0]).strip()
            tried = resolve_image_path(s0, data_root)
            print(
                f"[{log_tag}] 경고: 메타의 이미지 경로를 디스크에서 찾지 못했습니다.\n"
                f"  --data_root 는 manipulation·origin 폴더가 바로 아래에 있는 DGM4 루트여야 합니다.\n"
                f"  현재 data_root={os.path.abspath(data_root)}\n"
                f"  첫 행 image 필드(예시)={s0[:120]!r}\n"
                f"  resolve 결과={tried!r}  존재={bool(tried and os.path.isfile(tried))}\n"
                f"  로컬 메타를 쓰려면 {os.path.join(data_root, 'metadata', 'train.json')} 등이 실제 파일이어야 "
                f"(로그에 「로컬 로드」가 뜨면 HF 원격 메타가 아닙니다)."
            )
        dfs[split] = df.reset_index(drop=True)

    return dfs
