---
title: 01 Zero-shot
parent: Experiments
nav_order: 1
---

# 01 Zero-shot baseline

[Kaggle notebook](https://www.kaggle.com/code/tensorpanda231/01-zeroshot-qwen2-5-0-5b)

## Goal
Measure the base Qwen2.5-0.5B-Instruct model before any training. This is the number every later method is compared to.

## Setup
- **Generation:** greedy, max 512 new tokens, batch 128, one model copy per T4.
- **Prompt:** the shared prompt asking for a final `#### <number>` line.
- **Answer extraction:** the number after the last `####`. If the model skipped `####`, the last number in the text is used instead (the fallback).

## Results

| | GSM8K test | SVAMP |
|---|---|---|
| Accuracy | **43.8%** | **53.7%** |
| Uses `####` format | 16.5% | 36.4% |
| Mean answer length (tokens) | 305 | 220 |
| Truncated at 512 tokens | 5.6% | 1.4% |

Val accuracy for the same model is 57.8%, measured at step 0 of [05]({{ site.baseurl }}{% link experiments/05-sft-rft.md %}).

## Takeaways
- The model mostly ignores the `####` instruction: it ends with `\boxed{}` or prose. Its accuracy depends on the last-number fallback.
- Its answers are long (~300 tokens) and step-by-step. This "native style" turns out to matter (see [03]({{ site.baseurl }}{% link experiments/03-sft-vs-zeroshot.md %})).
- 5.6% of GSM8K answers run out of tokens and are counted wrong. The limit is kept fixed so later runs stay comparable.

## Cost
0.24 GPU-hours (GSM8K 4.2 min, SVAMP 2.9 min on 2× T4).
