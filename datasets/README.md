# Datasets for LLM Offense Detection Research

This directory contains datasets used for probing LLM internal representations
of "offense" — whether models represent taking-offense internally, and whether
such probes can be trusted.

Large dataset files (Arrow/Parquet) are excluded from git via `.gitignore`.
Each subdirectory contains a `sample.json` with 5-10 representative examples.

To re-download all datasets, run:

```bash
source .venv/bin/activate
python datasets/download_all.py
```

---

## Datasets

### 1. Civil Comments (`civil_comments/`)
**Source:** `google/civil_comments` on HuggingFace
**Size downloaded:** 10,000 examples (streaming subset of full ~1.8M)
**Task:** Toxicity classification (multi-label)
**Fields:** `text`, `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`, `sexual_explicit`
**Use in project:** Primary training set for offense/toxicity probes. The `toxicity` field (continuous 0–1) can be binarized.

```python
from datasets import load_dataset
ds = load_dataset("google/civil_comments", split="train", streaming=True)
```

---

### 2. ToxiGen (`toxigen/`)
**Source:** `skg/toxigen-data` on HuggingFace (train split)
**Size downloaded:** 250,951 examples (full dataset)
**Task:** Machine-generated hate speech vs. benign text classification
**Fields:** `prompt`, `generation`, `generation_method`, `group`, `prompt_label`, `roberta_prediction`
**Use in project:** Targeted toxicity (by demographic group); useful for studying identity-based offense. `roberta_prediction` gives a toxicity score.

```python
from datasets import load_dataset
ds = load_dataset("skg/toxigen-data", name="train", split="train")
```

---

### 3. RealToxicityPrompts (`real_toxicity_prompts/`)
**Source:** `allenai/real-toxicity-prompts` on HuggingFace
**Size downloaded:** 10,000 examples (streaming subset of full ~100K)
**Task:** Prompted generation toxicity measurement
**Fields:** `filename`, `begin`, `end`, `challenging`, `prompt` (dict with text + toxicity score), `continuation` (dict with text + toxicity score)
**Use in project:** Prompts with known toxicity scores — useful for studying how LLMs respond to offensive inputs vs. non-offensive inputs.

```python
from datasets import load_dataset
ds = load_dataset("allenai/real-toxicity-prompts", split="train", streaming=True)
```

---

### 4. Emotion (`emotion/`)
**Source:** `dair-ai/emotion` on HuggingFace
**Size downloaded:** Full dataset (16,000 train / 2,000 val / 2,000 test)
**Task:** 6-class emotion classification
**Labels:** 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
**Fields:** `text`, `label`
**Use in project:** Baseline for emotion representation probing. Establishes whether LLMs encode "anger" and "fear" (related to offense) in activation space.

```python
from datasets import load_dataset
ds = load_dataset("dair-ai/emotion")
```

---

### 5. GoEmotions (`go_emotions/`)
**Source:** `google-research-datasets/go_emotions` (simplified config) on HuggingFace
**Size downloaded:** Full dataset (43,410 train / 5,426 val / 5,427 test)
**Task:** Fine-grained 28-class emotion classification from Reddit comments
**Fields:** `text`, `labels` (multi-label list), `id`
**Use in project:** More granular emotion representation; includes "annoyance", "disgust", "disapproval" — directly relevant to offense. Multi-label format.

```python
from datasets import load_dataset
ds = load_dataset("google-research-datasets/go_emotions", "simplified")
```

---

### 6. XSTest (`xstest/`)
**Source:** `natolambert/xstest-v2-copy` on HuggingFace (public copy of paul-rottger/xstest)
**Size downloaded:** Full dataset (450 prompts × 6 model splits = 2,700 examples)
**Task:** Exaggerated safety testing — safe prompts that models incorrectly refuse
**Splits:** `prompts`, `gpt4`, `llama2new`, `llama2orig`, `mistralguard`, `mistralinstruct`
**Fields:** `id`, `type`, `prompt`, `completion`, `annotation_1`, `annotation_2`, `agreement`, `final_label`
**Prompt types:** `safe_*` (should be answered) and `unsafe_*` (should be refused)
**Use in project:** Key dataset for studying over-refusal. Tests whether offense probes fire on safe prompts (false positives). 250 safe / 200 unsafe prompts.

```python
from datasets import load_dataset
ds = load_dataset("natolambert/xstest-v2-copy")
# Use ds["prompts"] for the raw prompts without model completions
```

Original paper: Rottger et al. (2023) "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models"

---

### 7. TruthfulQA (`truthful_qa/`)
**Source:** `truthfulqa/truthful_qa` (multiple_choice config) on HuggingFace
**Size downloaded:** Full dataset (817 examples, validation split only)
**Task:** Truthfulness measurement via multiple-choice QA
**Fields:** `question`, `mc1_targets` (single correct answer), `mc2_targets` (multiple correct answers)
**Use in project:** Baseline for representation engineering probes. Used in the original RepE paper (Zou et al. 2023) to probe truthfulness vs. deception.

```python
from datasets import load_dataset
ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
```

---

## Directory Structure

```
datasets/
├── README.md                        # This file
├── .gitignore                       # Excludes large Arrow/Parquet files
├── civil_comments/
│   ├── data/                        # Arrow files (gitignored)
│   └── sample.json                  # 10 representative examples
├── toxigen/
│   ├── data/                        # Arrow files (gitignored)
│   └── sample.json
├── real_toxicity_prompts/
│   ├── data/                        # Arrow files (gitignored)
│   └── sample.json
├── emotion/
│   ├── data/                        # Arrow files (gitignored)
│   └── sample.json
├── go_emotions/
│   ├── data/                        # Arrow files (gitignored)
│   └── sample.json
├── xstest/
│   ├── data/                        # Arrow files (gitignored)
│   └── sample.json
└── truthful_qa/
    ├── data/                        # Arrow files (gitignored)
    └── sample.json
```

## Loading Saved Datasets

All datasets were saved in HuggingFace Arrow format and can be loaded with:

```python
from datasets import load_from_disk

ds = load_from_disk("datasets/civil_comments/data")
ds = load_from_disk("datasets/emotion/data")
# etc.
```

## Download Script

To reproduce the full download:

```python
# datasets/download_all.py
from datasets import load_dataset, Dataset
import json

# Civil Comments (10K subset)
ds = load_dataset("google/civil_comments", split="train", streaming=True)
subset = list(itertools.islice(ds, 10000))
Dataset.from_list(subset).save_to_disk("datasets/civil_comments/data")

# ToxiGen (full)
load_dataset("skg/toxigen-data", name="train", split="train").save_to_disk("datasets/toxigen/data")

# RealToxicityPrompts (10K subset)
ds = load_dataset("allenai/real-toxicity-prompts", split="train", streaming=True)
subset = list(itertools.islice(ds, 10000))
Dataset.from_list(subset).save_to_disk("datasets/real_toxicity_prompts/data")

# Emotion (full)
load_dataset("dair-ai/emotion").save_to_disk("datasets/emotion/data")

# GoEmotions (full)
load_dataset("google-research-datasets/go_emotions", "simplified").save_to_disk("datasets/go_emotions/data")

# XSTest (full)
load_dataset("natolambert/xstest-v2-copy").save_to_disk("datasets/xstest/data")

# TruthfulQA (full)
load_dataset("truthfulqa/truthful_qa", "multiple_choice").save_to_disk("datasets/truthful_qa/data")
```
