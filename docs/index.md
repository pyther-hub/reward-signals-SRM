---
title: Overview
nav_order: 1
---

# Rewards and math reasoning in a small LLM

Qwen2.5-0.5B-Instruct on GSM8K: what does each post-training step add? Same model, prompt, data splits and greedy evaluation throughout (max 512 new tokens, exact match on the number after `####`). Hardware: Kaggle 2× T4, fp16.

## Results

Greedy accuracy (%). GSM8K train 6,973 · val 500 · test 1,319; SVAMP 1,000 (generalization only).

| Model | GSM8K test | SVAMP | Val | Answer length (tokens) | GPU-hours |
|---|---|---|---|---|---|
| [Zero-shot]({{ site.baseurl }}{% link experiments/01-zeroshot.md %}) · [Kaggle](https://www.kaggle.com/code/tensorpanda231/01-zeroshot-qwen2-5-0-5b) | 43.8 | 53.7 | 57.8 | 305 | 0.24 |
| [SFT on gold solutions]({{ site.baseurl }}{% link experiments/02-sft-gold.md %}) · [Kaggle](https://www.kaggle.com/code/tensorpanda231/02-sft-0-5b) | 32.7 | 35.5 | 43.4 | 92 | 0.57 |
| [SFT on own correct answers (RFT)]({{ site.baseurl }}{% link experiments/05-sft-rft.md %}) · [Kaggle](https://www.kaggle.com/code/tensorpanda231/05-sft-with-rft-qwen-05b) | 49.4 | 61.9 | 66.6 | 309 | 5.1 (+6.5 data) |
| [RFT-SFT + GRPO]({{ site.baseurl }}{% link experiments/06-grpo.md %}) · [Kaggle](https://www.kaggle.com/code/tensorpanda231/06-grpo-0-5b) | **54.4** | **63.2** | **69.8** | ~300 | 19.2 |

![Accuracy of zero-shot, gold SFT and RFT SFT]({{ '/assets/plots/05_accuracy_compare.png' | relative_url }})

## Findings

1. **SFT on gold solutions hurt:** −11.1 pts on GSM8K, −18.2 on SVAMP. The terse gold style (~90 tokens) replaced the model's own step-by-step reasoning. ([02]({{ site.baseurl }}{% link experiments/02-sft-gold.md %}), [03]({{ site.baseurl }}{% link experiments/03-sft-vs-zeroshot.md %}))
2. **SFT on the model's own correct answers (RFT) helped:** +5.6 on GSM8K, +8.2 on SVAMP, same answer style, and the `####` format learned. ([04]({{ site.baseurl }}{% link experiments/04-rft-data.md %}), [05]({{ site.baseurl }}{% link experiments/05-sft-rft.md %}))
3. **GRPO with a correct/incorrect reward added +5.0 on GSM8K** (SVAMP only +1.3), with small KL and no length drift. ([06]({{ site.baseurl }}{% link experiments/06-grpo.md %}))
4. **Pick checkpoints by val accuracy, not val loss**, and read val as easier than test (by 11–17 pts).

Setup notes: [00 Data]({{ site.baseurl }}{% link experiments/00-data.md %}).
