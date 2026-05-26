# DGM4 실험 (HARF-News)

[DGM⁴ / HAMMER 공식 코드](https://github.com/rshaojimmy/MultiModal-DeepFake)를 기반으로 한 크로스데이터셋 실험입니다.  
데이터 준비·폴더 구조·사전학습 가중치는 **공식 README를 그대로 따르세요.**

## 1. 데이터셋 & ALBEF 가중치

1. [MultiModal-DeepFake README — Dataset Preparation](https://github.com/rshaojimmy/MultiModal-DeepFake#dataset-preparation) 에서 **DGM4** 다운로드  
2. 압축 해제 후 아래 구조가 되도록 배치 (`metadata/train.json` 등 포함)

```
experiment/DGM4/datasets/DGM4/
├── manipulation/
├── origin/
└── metadata/
    ├── train.json
    ├── val.json
    └── test.json
```

3. [ALBEF_4M.pth](https://github.com/rshaojimmy/MultiModal-DeepFake#dataset-preparation) 를 받아  
   `multimodal_deepfake/ALBEF_4M.pth` 에 둡니다 (이미 있으면 생략).

다른 경로에 두었다면 실행 시 지정:

```bash
export DGM4_DATA_ROOT=/path/to/DGM4
export DGM4_ALBEF_PATH=/path/to/ALBEF_4M.pth   # HAMMER_bbox.py 등
```

## 2. 환경

공식 HAMMER 학습만 할 때는 `multimodal_deepfake/requirements.txt` + Python 3.8 / PyTorch 1.10 (공식 안내).  
HARFNet·baseline 스크립트는 상위 [`fakenews/requirements.txt`](../../../requirements.txt) 를 사용합니다.

```bash
cd ../../../   # fakenews/
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

## 3. 실행 예시

`experiment/DGM4/` 에서:

```bash
# HARFNet on DGM4 (기본 data_root: datasets/DGM4 또는 DGM4_DATA_ROOT)
python harfnet_DGM4.py

# Ablation / baseline
python harfnet_dgm4_wo_bi.py --kfold
python harfnet_baseline_dgm4.py --data_root ./datasets/DGM4

# HAMMER (공식 구현 래퍼)
python HAMMER_bbox.py --data_root ./datasets/DGM4

# 기타
python fnd_clip_DGM4.py
python cafe_DGM4.py
python spotfakeplus_dgm4.py
```

`--data_root` 를 생략하면 `dgm4_paths.get_default_data_root()` 가  
`DGM4_DATA_ROOT` → `./datasets/DGM4` → 상위 `datasets/DGM4` 순으로 찾습니다.

## 4. 디렉터리 요약

| 경로 | 설명 |
|------|------|
| `multimodal_deepfake/` | [공식 HAMMER 저장소](https://github.com/rshaojimmy/MultiModal-DeepFake) 벤더 코드 |
| `dgm4_paths.py` | 메타 JSON·이미지 경로 해석, 기본 `data_root` |
| `harfnet_*.py`, `*_DGM4.py` | HARF-News 모델·baseline 실험 스크립트 |
| `datasets/DGM4/` | **직접 받아 둘** 데이터 (git 미포함) |
| `results/`, `logs/` | 실행 결과 |

## 5. 인용

```bibtex
@inproceedings{shao2023dgm4,
  title={Detecting and Grounding Multi-Modal Media Manipulation},
  author={Shao, Rui and Wu, Tianxing and Liu, Ziwei},
  booktitle={CVPR},
  year={2023}
}
```
