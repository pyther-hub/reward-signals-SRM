---
title: 04 RFT data
parent: Experiments
nav_order: 4
---

# 04 Rejection-sampling (RFT) data

[Kaggle notebook](https://www.kaggle.com/code/tensorpanda231/04-rft-data)

## Goal
Build an SFT dataset from the model's **own** correct answers, so fine-tuning teaches correctness without forcing a foreign style (see [03]({{ site.baseurl }}{% link experiments/03-sft-vs-zeroshot.md %})).

## Setup
- **Sampling:** the base Qwen2.5-0.5B-Instruct samples 8 answers per train and val problem. T = 0.7, top-k / top-p / repetition penalty turned off, max 512 tokens.
- **Answer checking:** strict extraction. The number after `####`, else the number in the last `\boxed{}`; no last-number fallback.
- **Keep** correct, non-truncated answers. If the answer used `\boxed{}`, a final `#### <gold>` line is appended so all training targets share the format.
- **Dedupe** per problem by the sequence of equations, and keep at most 4 answers per problem.

## Results

| | Train | Val |
|---|---|---|
| Problems | 6,973 | 500 |
| Sample accuracy (pass@1, T = 0.7) | 50.0% | 51.9% |
| Coverage (pass@8: ≥ 1 correct of 8) | **86.0%** | 87.2% |
| Truncated samples | 5.0% | 4.3% |
| Answers kept | **16,515** (5,982 problems) | 1,209 (436 problems) |
| Kept per problem | 2.37 | 2.42 |
| Median tokens: kept answer vs gold | 267 vs 82 | 259 vs 78 |

How the samples ended (train):

| | `\boxed{}` | `####` | Neither |
|---|---|---|---|
| All samples | 54.7% | 33.5% | 11.8% |
| Correct samples | 67.0% | 33.0% | — |

## Takeaways
- The RFT data is ~3× longer than gold and in the model's own style. That is exactly the property gold SFT lacked.
- **Blind spot:** the 991 train problems with 0 of 8 correct (14%) are absent. RFT can only reinforce what the model can already sometimes do, so it is biased towards easier problems. RL should target the hard ones.
- **To check:** the 11.8% of samples with neither format may include correct answers that strict extraction rejects (fractions, "the answer is X").

## Cost
6.1 GPU-hours for train (182 min on 2× T4) + 0.44 for val.
