---
title: 02 SFT on gold
parent: Experiments
nav_order: 2
---

# 02 SFT on gold solutions

[Kaggle notebook](https://www.kaggle.com/code/tensorpanda231/02-sft-0-5b)

## Goal
Full fine-tuning of the 0.5B model on the human-written GSM8K solutions (calculator notes removed). This was meant to be the start point for RL.

## Setup
- TRL `SFTTrainer`, DDP on 2× T4 (`torchrun`).
- fp16, gradient checkpointing.
- Loss on answer tokens only.
- lr 1e-5, cosine schedule, 3% warmup.
- Effective batch 64 (2 GPUs × 8 × 4 accumulation steps), max length 640, up to 3 epochs.
- Early stopping on **val loss** (patience 3, 4 evaluations per epoch).
- The best checkpoint was step 108 of 327; training stopped early at step 189 (1.7 epochs).

## Results

| | Val | GSM8K test | SVAMP |
|---|---|---|---|
| Accuracy | 43.4% | **32.7%** (zero-shot 43.8) | **35.5%** (zero-shot 53.7) |
| Uses `####` format | 99.8% | 99.6% | 100% |
| Mean answer length (tokens) | 88 | 92 | 48 |

![Training and val loss]({{ '/assets/plots/02_loss.png' | relative_url }})

![Learning rate and gradient norm]({{ '/assets/plots/02_lr_gradnorm.png' | relative_url }})

## Takeaways
- **SFT on gold hurt badly:** −11 pts on GSM8K and −18 on SVAMP.
- **The format was learned perfectly, but the style changed:** answers shrank from ~300 to ~90 tokens, copying the terse gold solutions.
- **Train loss drops sharply at the epoch boundary** (step ~110) while val loss is flat. That is memorization, not learning.
- [03]({{ site.baseurl }}{% link experiments/03-sft-vs-zeroshot.md %}) investigates why.

## Cost
0.36 GPU-hours of training (11 min) + 0.21 of evaluation.
