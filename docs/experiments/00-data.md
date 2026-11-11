---
title: 00 Data
parent: Experiments
nav_order: 0
---

# 00 Data: splits and cleaning

[Kaggle notebook](https://www.kaggle.com/code/tensorpanda231/00-data-math-reasoning)

## Goal
Download GSM8K and SVAMP once, clean them, and fix the splits that every later notebook reads.

## Setup
- **GSM8K** (`openai/gsm8k`, `main`): the official train (7,473) and test (1,319) splits.
- **Validation:** the train set is shuffled with seed 42 and the first 500 problems are held out. The remaining 6,973 are used for training.
- **SVAMP** (`ChilleD/SVAMP`): train and test splits merged, 1,000 problems, used for evaluation only. The question is `Body + Question`.
- **Cleaning:**
  - The `<<48/2=24>>` calculator annotations are removed from the solutions (the model has no calculator).
  - The gold answer is the text after `####`, with commas removed.

## Results

| Split | Problems | Used for |
|---|---|---|
| `gsm8k_train` | 6,973 | SFT / RFT / RL training |
| `gsm8k_val` | 500 | Checkpoint selection |
| `gsm8k_test` | 1,319 | Final evaluation only |
| `svamp_test` | 1,000 | Generalization (evaluation only) |

**EDA, to fill in:** the executed notebook's outputs are not saved in the Kaggle version, so these numbers still need adding:
- Median question length (words) per split.
- Median solution steps.
- Counts of negative, decimal and ≥1000 answers.
- Gold-solution token lengths: median / p90 / p99.

## Takeaways
- All splits are written once, as JSONL, and read by every later notebook, so all experiments use exactly the same problems.
- The val split comes from GSM8K train. Later results show that val is consistently 11–17 pts easier than test (see [05]({{ site.baseurl }}{% link experiments/05-sft-rft.md %})).

## Cost
CPU only; negligible.
