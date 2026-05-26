# HARF-News: A Multimodal Benchmark for Disentangling Authorship and Veracity in AI-Era News

> **Paper:** [Coming Soon]  
> **Dataset:** [🤗 harf-news/HARF-News on Hugging Face](https://huggingface.co/datasets/harf-news/HARF-News)

---

## Overview

**HARF-News** is a large-scale multimodal fake news benchmark designed to disentangle *authorship* (Human vs. AI) and *veracity* (Real vs. Fake) in the age of generative AI. Unlike prior datasets that conflate these two dimensions, HARF-News defines a four-class label space that enables fine-grained analysis of AI-generated misinformation.

| Label | Authorship | Veracity | Description |
|-------|-----------|----------|-------------|
| **HR** | Human | Real | Original human-written headlines from real news posts |
| **HF** | Human | Fake | Original human-written headlines from fake news posts |
| **AR** | AI | Real | Human-real headlines rewritten by GPT-4o-mini (facts preserved) |
| **AF** | AI | Fake | AI-generated fake headlines via six manipulation strategies |

### Dataset Statistics

| Label | Count | Ratio |
|-------|-------|-------|
| HR    | 84,954 | 35.2% |
| AR    | 60,000 | 24.8% |
| HF    | 56,636 | 23.4% |
| AF    | 40,000 | 16.6% |
| **Total** | **241,590** | **100%** |

---

## Dataset Access

The dataset is publicly available on Hugging Face:

**[🤗 https://huggingface.co/datasets/harf-news/HARF-News](https://huggingface.co/datasets/harf-news/HARF-News)**

```python
from datasets import load_dataset

dataset = load_dataset("harf-news/HARF-News")
```

The dataset includes:
- `text`: News headline
- `label`: One of `HR`, `HF`, `AR`, `AF`
- `image_url`: URL to the associated image (sourced from Fakeddit)
- `split`: `train` / `val` / `test`

> **Note:** Images are not bundled in the HuggingFace repository due to size constraints. Image files can be downloaded from the original [Fakeddit](https://github.com/entitize/Fakeddit) dataset using the provided `image_url` field, or obtained by contacting the authors.

---

## Repository Structure

```
HARF-News/
├── annotation/
│   ├── annotation_image.csv     # Human annotation results — image modality
│   └── annotation_text.csv      # Human annotation results — text modality
├── dataset/
│   ├── 01_preprocessing.ipynb   # Fakeddit filtering & sampling pipeline
│   ├── 02_generation.ipynb      # AR & AF generation via OpenAI Batch API
│   ├── 03_eda.ipynb             # Exploratory data analysis
│   └── README.md
├── fakenewsdetection/
│   ├── model/                   # HARFNet (proposed) and ablation variants
│   ├── baseline/                # Baseline models (BERT, RoBERTa, ViT, CLIP, CAFE, HAMMER, MIRAGE, SpotFake+)
│   └── experiment/              # Cross-dataset experiments (MIRAGE-News, DGM4)
├── scripts/
│   └── clean_cross_ref_comments_safe.py
├── requirements.txt
└── README.md
```

---

## HARFNet

We propose **HARFNet** (Human-AI Real-Fake Network), a multimodal fake news detection model that jointly captures:
- **Cross-modal bilinear fusion** of text (RoBERTa) and image (CLIP) features
- **Outlier-Aware Module (OAM)** to detect distributional shifts in AI-generated content
- **Auxiliary authorship classification** to explicitly disentangle authorship from veracity

### Model Architecture

| Component | Backbone |
|-----------|----------|
| Text encoder | `roberta-base` |
| Image encoder | CLIP `RN101` |
| Fusion | Bilinear pooling (rank=128) |
| Output | 4-class classification (HR / HF / AR / AF) |

### Ablation Variants

| File | Description |
|------|-------------|
| `harfnet.py` | Full model (proposed) |
| `harfnet_wo_a.py` | w/o auxiliary authorship loss |
| `harfnet_wo_bi.py` | w/o bilinear fusion |
| `harfnet_wo_oa.py` | w/o Outlier-Aware Module |
| `harfnet_wo_v.py` | w/o visual stream |
| `harfnet_wo_both.py` | w/o bilinear fusion & OAM |
| `harfnet_baseline.py` | Simple concat baseline |

---

## Baselines

| Model | Modality | File |
|-------|----------|------|
| BERT | Text only | `baseline/Bert.py` |
| RoBERTa | Text only | `baseline/roberta.py` |
| DeBERTa | Text only | `baseline/deberta.py` |
| CLIP | Image only | `baseline/Clip.py` |
| ViT-B/16 | Image only | `baseline/vit.py` |
| ResNet-50 | Image only | `baseline/resnet.py` |
| CAFE | Multimodal | `baseline/cafe.py` |
| HAMMER | Multimodal | `baseline/hammer.py` |
| MIRAGE | Multimodal | `baseline/mirage.py` |
| SpotFake+ | Multimodal | `baseline/spotfake_plus.py` |
| FND-CLIP | Multimodal | `baseline/fnd_clip.py` |

---

## Cross-Dataset Experiments

Generalizability experiments on two additional benchmarks:

| Benchmark | Description | Code |
|-----------|-------------|------|
| [MIRAGE-News](https://github.com/MIRAGE-News) | AI-generated news detection | `experiment/mirage/` |
| [DGM4](https://github.com/rshaojimmy/DGM4) | Multimodal disinformation grounding | `experiment/DGM4/` |

---

## Dataset Construction

The HARF-News dataset is constructed from the [Fakeddit](https://github.com/entitize/Fakeddit) corpus through a three-stage pipeline:

1. **Preprocessing** (`01_preprocessing.ipynb`) — Image validation, headline length filtering, subreddit exclusion
2. **AI Generation** (`02_generation.ipynb`) — AR headlines via GPT-4o-mini paraphrase; AF headlines via six manipulation strategies using OpenAI Batch API
3. **EDA** (`03_eda.ipynb`) — Label distribution, text/image statistics, inter-annotator agreement

### AF Generation Strategies

| Strategy | Seed | Description |
|----------|------|-------------|
| FS-1 | HR | Flip main outcome / stance |
| FS-2 | HR | Change multiple key facts |
| FS-3 | HR | Change one or two facts |
| FB-1 | HF | Paraphrase (same claim, different wording) |
| FB-2 | HF | Expand with contextual phrase |
| FB-3 | HF | Disguise tone/style while preserving claim |

---

## Setup

```bash
# 1. Install PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git
```

### Running HARFNet

```bash
cd fakenewsdetection/model
python harfnet.py
```

The script expects the following files under `fakenewsdetection/`:
- `HARFM.csv` — Label metadata (available via HuggingFace dataset)
- `images/` — Image files (download from Fakeddit using `image_url`)

---

## Human Annotation

Inter-annotator agreement experiments are provided in `annotation/`:
- `annotation_text.csv` — Crowd-sourced headline-level veracity and authorship labels
- `annotation_image.csv` — Crowd-sourced image-level veracity labels

---

## Citation

If you use HARF-News in your research, please cite:

```bibtex
@article{harfnews2026,
  title   = {HARF-News: A Multimodal Benchmark for Disentangling Authorship and Veracity in AI-Era News},
  author  = {},
  year    = {2026}
}
```

---

## License

This repository is released under the [MIT License](LICENSE).  
The dataset follows the license terms of the original [Fakeddit](https://github.com/entitize/Fakeddit) dataset.
