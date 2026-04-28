# Attend to Fragments: How Key Information Affects Large Language Models for Factual Inconsistency Detection

This repository contains the code, benchmark, and experimental results for our SIGIR 2026 short paper:

> **Attend to Fragments: How Key Information Affects Large Language Models for Factual Inconsistency Detection**
> Xindi Guo, Zhen Xie, Patrick H. Chen
> *SIGIR 2026 (Short Papers Track)*

## Overview

We investigate whether NLI-based factual inconsistency detectors actually leverage the relevant evidence in source documents, or whether their decisions are driven by surface-level fragments. We make three contributions:

1. **KIFI Benchmark** — 1,032 instances from TRUE and ScreenEval with human-annotated minimal sufficient evidence.
2. **Empirical Findings** — LLMs frequently fail to retrieve the correct minimal evidence; the "Lost in the Middle" effect does *not* hold for factual inconsistency detection; instead, models exhibit a behavior we term **Attend to Fragments**.
3. **Uncertainty Filtering** — A word-level permutation probe that identifies untrustworthy predictions, improving overall correlation by 1.3% on the standard TRUE benchmark.

## Repository Structure

```
.
├── data/
│   └── Perm2.csv                  # KIFI benchmark (1,032 instances with key info annotations)
├── scripts/
│   ├── untouched_oss20.py         # Baseline: original document → entailment score
│   ├── non_key_oss.py             # NON-KIFI: remove key information, score remaining
│   ├── word_scrammble_key_intact_oss.py   # SHUFFLE: shuffle non-key context, key info intact
│   ├── keyinfo_permuted_rest_intact_oss.py # PERMUTE-G: permute key info into G chunks
├── evaluates/
│   └── ...                        # Analysis scripts (basic_result_oss20.py, etc.)
├── results/
│   ├── untouched_perm_oss20b.pkl  # Per-instance scores (10 generations each)
│   ├── non_key_oss20b.pkl
│   ├── word_scrammble_key_intact_oss20.pkl
│   └── key_shuffeled_15-20b.pkl
└── README.md
```

## KIFI Benchmark

The `data/Perm2.csv` file contains the KIFI benchmark with the following columns:

| Column     | Description |
|------------|-------------|
| `Dataset`  | Source dataset (TRUE subsets or ScreenEval) |
| `index`    | Original index in the source dataset |
| `convo`    | Source document |
| `inf_summ` | Generated claim to verify |
| `label`    | 1 = consistent, 0 = inconsistent |
| `Support`  | Human-annotated minimal sufficient evidence (sentences separated by ` \| `) |

**Statistics:** 1,032 instances · 698 consistent / 334 inconsistent · 9 source datasets · mean document length 510 words · mean key information length 57 words · mean 2.26 evidence sentences per instance.

| Source     | n   | Cons / Inc | Doc len (words) | KI len |
|------------|-----|------------|-----------------|--------|
| SummEval   | 364 | 308 / 56   | 368             | 68     |
| FRANK      | 197 | 131 / 66   | 569             | 67     |
| QAGS-CNNDM | 141 | 89 / 52    | 320             | 70     |
| QAGS-XSUM  | 134 | 78 / 56    | 347             | 32     |
| FEVER      | 89  | 37 / 52    | 135             | 32     |
| MNBM       | 54  | 15 / 39    | 567             | 33     |
| ScreenEval | 40  | 35 / 5     | 3629            | 60     |
| DialFact   | 8   | 1 / 7      | 71              | 27     |
| VITC       | 5   | 4 / 1      | 57              | 20     |

## Setup

### Requirements

```bash
python >= 3.8
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas
numpy
nltk
scikit-learn
tqdm
torch
transformers
ollama        # for GPT-OSS:20B inference
openai        # for GPT-4 inference (optional)
```

### Models

The experiments support three model scales:

| Model           | Backend     | Notes |
|-----------------|-------------|-------|
| Flan-T5-xl      | HuggingFace `transformers` | logit-based scoring |
| GPT-OSS:20B     | Ollama      | sampling-based (10 generations) |
| GPT-4           | OpenAI API  | sampling-based (10 generations) |

For GPT-OSS:20B, install [Ollama](https://ollama.com) and pull the model:
```bash
ollama pull gpt-oss:20b
```

## Running Experiments

All scripts read `data/Perm2.csv` and write per-instance results to `results/*.pkl`.

### 1. Baseline (untouched document)

```bash
python scripts/untouched_oss20.py
# → results/untouched_perm_oss20b.pkl
```

### 2. NON-KIFI (remove key information)

```bash
python scripts/non_key_oss.py
# → results/non_key_oss20b.pkl
```

### 3. SHUFFLE-HEAD/MIDDLE/TAIL (shuffle non-key context, key info intact at different positions)

```bash
python scripts/word_scrammble_key_intact_oss.py
# → results/word_scrammble_key_intact_oss20.pkl  (3 lists: head, middle, tail)
```

### 4. PERMUTE-G (permute key information into G chunks)

The `block` variable at the top of the script controls G:

```bash
# Edit `block = 15` (or 3, 5, 10, 20, etc.) inside the script
python scripts/keyinfo_permuted_rest_intact_oss.py
# → results/key_shuffeled_{G}-20b.pkl
```

Each permutation is repeated 10 times per instance (controlled by `np.random.shuffle`).

## Analyzing Results

Result `.pkl` files store entailment generations as `[[response_1, ..., response_10], ...]`. Convert to per-instance entailment scores by computing the fraction of "Yes" responses:

```python
import pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

with open("results/untouched_perm_oss20b.pkl", "rb") as f:
    r = pickle.load(f)

df = pd.read_csv("data/Perm2.csv")
labels = df["label"].values
probs = np.array([np.mean([1 if "Yes" in g or "yes" in g else 0
                           for g in inst]) for inst in r])

print(f"AUC: {roc_auc_score(labels, probs):.4f}")
```

For 95% bootstrap confidence intervals over instances:

```python
def bootstrap_auc(labels, probs, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    return np.mean(aucs), np.percentile(aucs, [2.5, 97.5])
```

## Reproducing Paper Results

| Paper Section | Script | Result file |
|---------------|--------|-------------|
| Table 1 (key info retrieval) | (uses GPT-4 API; see paper) | — |
| Figure 3a (existence + location) | `untouched_oss20.py`, `non_key_oss.py`, `word_scrammble_key_intact_oss.py` | corresponding pkls |
| Figure 3b (PERMUTE-G) | `keyinfo_permuted_rest_intact_oss.py` (run with G ∈ {3, 5, 10, 15, 20, \|K\|}) | `key_shuffeled_{G}-20b.pkl` |
| Table 2 (Attend to Fragments on full TRUE) | (additional scripts forthcoming) | — |

The same pipeline applies to Flan-T5-xl and GPT-4 with minor changes to the inference call.

## Citation

If you use KIFI or any code from this repository, please cite:

```bibtex
@inproceedings{guo2026attend,
  title     = {Attend to Fragments: How Key Information Affects Large Language Models for Factual Inconsistency Detection},
  author    = {Guo, Xindi and Xie, Zhen and Chen, Patrick H.},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '26)},
  year      = {2026}
}
```

## Acknowledgements

KIFI is built on top of two prior benchmarks:

- **TRUE** [Honovich et al. 2022] — *True: Re-evaluating factual consistency evaluation.*
- **ScreenEval** [Lattimer et al. 2023] — *Fast and accurate factual inconsistency detection over long documents.*

We thank our annotators for their substantial labeling effort (300+ senior reviewer hours, 100+ first-stage annotator hours).

## License

Code: GPL-3.0 license (see `LICENSE`).
KIFI annotations: released under CC BY 4.0, consistent with the source datasets' licensing terms.

## Contact

For questions about the benchmark or code, please open a GitHub issue or contact the authors via the SIGIR 2026 author profiles.
