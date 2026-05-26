# HARFM Dataset — Construction Pipeline

This repository contains the full pipeline for constructing the **HARFM** (Human/AI Real/Fake Multimodal) dataset from the [Fakeddit](https://github.com/entitize/Fakeddit) corpus.

---

## Dataset Overview

HARFM is a four-class multimodal fake news dataset built on top of Fakeddit.

| Label | Full Name | Description |
|---|---|---|
| **HR** | Human-Real | Original human-written headlines from real Reddit posts |
| **HF** | Human-Fake | Original human-written headlines from fake Reddit posts |
| **AR** | AI-Real | Human-real headlines rewritten by GPT-4o-mini (facts preserved) |
| **AF** | AI-Fake | AI-generated fake headlines using six manipulation strategies |

### Final Label Counts

| Label | Count | Ratio |
|---|---|---|
| HR | 84,954 | 35.2% |
| AR | 60,000 | 24.8% |
| HF | 56,636 | 23.4% |
| AF | 40,000 | 16.6% |
| **Total** | **241,590** | **100%** |

---

## Preprocessing Summary

| Stage | Real | Fake | Total |
|---|---|---|---|
| Original Fakeddit (3 splits) | 268,908 | 413,753 | 682,661 |
| After image filtering | 267,601 | 413,197 | 680,798 |
| → Removed | 1,307 | 556 | 1,863 |
| After length filtering | 164,954 | 110,156 | 275,110 |
| → Removed | 102,647 | 303,041 | 405,688 |
| Final HR / HF (human) | 84,954 | 56,636 | 141,590 |
| AR / AF (AI-generated) | 60,000 | 40,000 | 100,000 |
| **HARFM total** | **144,954** | **96,636** | **241,590** |

### Filtering Steps

1. **Image filtering** — Remove samples with no valid `image_url` or `hasImage=False` (deleted posts / broken URLs). Removes 1,863 samples (1,307 real, 556 fake).

2. **Headline length filtering** — Three sub-steps:
   - IQR-based outlier removal (Q1=19, Q3=57, IQR=38; bounds: [−38, 114])
   - Below-mean removal (mean = 38.7 chars after IQR filter)
   - `propagandaposters` subreddit exclusion (14,642 samples, all fake; image–text veracity labels are systematically inconsistent)
   - Total removed: 405,688 samples (102,647 real, 303,041 fake)

---

## AF Generation Strategies

| Strategy | Source | Description |
|---|---|---|
| FS-1 | HR seed | Flip the main outcome / stance |
| FS-2 | HR seed | Change multiple key facts (large shift) |
| FS-3 | HR seed | Change one or two facts (small shift) |
| FB-1 | HF seed | Paraphrase — same claim, different wording |
| FB-2 | HF seed | Expand — add a short contextual phrase |
| FB-3 | HF seed | Disguise — change tone/style while keeping claim |

Strategies are assigned in round-robin order within each source group (HR / HF).

---

## Repository Structure

```
dataset/
├── 01_preprocessing.ipynb   # Fakeddit loading, filtering, sampling → HARFM_Step_1.csv
├── 02_generation.ipynb      # AR & AF generation via OpenAI Batch API → HARFM.csv
├── 03_eda.ipynb             # Exploratory data analysis of the final dataset
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy openai tqdm matplotlib seaborn
```

### 2. Download Fakeddit

Place the following files in the `dataset/` directory:
- `multimodal_train.tsv`
- `multimodal_validate.tsv`
- `multimodal_test_public.tsv`

Download from the official [Fakeddit repository](https://github.com/entitize/Fakeddit).

### 3. Run notebooks in order

```
01_preprocessing.ipynb  →  HARFM_Step_1.csv
02_generation.ipynb     →  HARFM_Step_2_AR.csv  →  HARFM.csv
03_eda.ipynb            →  figures & statistics
```

> **Note:** `02_generation.ipynb` requires an OpenAI API key with Batch API access.  
> Set `API_KEY = "YOUR_OPENAI_API_KEY"` in the first cell.

---

## Citation

If you use HARFM in your research, please cite:

```bibtex
@article{harfm2026,
  title   = {HARFM: A Human/AI Real/Fake Multimodal Fake News Dataset},
  author  = {...},
  year    = {2026}
}
```
