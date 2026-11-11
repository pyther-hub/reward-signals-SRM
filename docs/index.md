---
title: Overview
nav_order: 1
---

# How different rewards shape math reasoning in a small LLM

**Question:** starting from the same small model, how do different post-training signals (gold-solution SFT, rejection-sampling fine-tuning, and next, RL with outcome, LLM-judge and teacher rewards) change math reasoning, and why?

## Fixed setup

Kept identical across every experiment so numbers compare directly.

| | |
|---|---|
| Policy model | Qwen2.5-0.5B-Instruct (full fine-tuning) |
| Data | GSM8K train 6,973 · val 500 (held out from train, seed 42) · test 1,319; SVAMP 1,000 (generalization only) |
| Prompt | Question + *"Solve the problem step by step. End your answer with a final line of the form '#### &lt;number&gt;'."* |
| Eval | Greedy decoding, max 512 new tokens, exact match on the number after `####` (fallback: last number) |
| Hardware | Kaggle 2× NVIDIA T4 (16 GB), fp16 (no bf16), one experiment per session |

## Results so far

Greedy exact-match accuracy (%).

| Model | Val | GSM8K test | SVAMP | Uses `####` (GSM8K) | Answer length (tokens) | GPU-hours |
|---|---|---|---|---|---|---|
| [Zero-shot]({{ site.baseurl }}{% link experiments/01-zeroshot.md %}) | 57.8 | 43.8 | 53.7 | 16.5% | 305 | 0.24 |
| [SFT on gold solutions]({{ site.baseurl }}{% link experiments/02-sft-gold.md %}) | 43.4 | 32.7 | 35.5 | 99.6% | 92 | 0.57 |
| [SFT on own correct answers (RFT)]({{ site.baseurl }}{% link experiments/05-sft-rft.md %}) | **66.6** | **49.4** | **61.9** | 94.4% | 309 | 5.1 (+6.5 data) |

Zero-shot val is the step-0 value measured in [05]({{ site.baseurl }}{% link experiments/05-sft-rft.md %}).

![Accuracy of the three models]({{ '/assets/plots/05_accuracy_compare.png' | relative_url }})

## Key findings

1. **SFT on human gold solutions made the model worse:** −11.1 pts on GSM8K and −18.2 on SVAMP, both significant. It broke 49% of the problems the base model already solved. ([02]({{ site.baseurl }}{% link experiments/02-sft-gold.md %}), [03]({{ site.baseurl }}{% link experiments/03-sft-vs-zeroshot.md %}))
2. **The gold text is "foreign" to the model.** Per-token NLL is 0.91 on gold vs 0.14 on its own answers, and SFT copied the terse gold style (~90 tokens instead of ~300), losing the step-by-step reasoning that made it work. ([03]({{ site.baseurl }}{% link experiments/03-sft-vs-zeroshot.md %}))
3. **Training on the model's own correct answers fixed it.** Rejection-sampling fine-tuning (RFT) gave +5.6 pts on GSM8K and +8.2 on SVAMP over zero-shot, kept the native answer style and length, and taught the `####` format. ([04]({{ site.baseurl }}{% link experiments/04-rft-data.md %}), [05]({{ site.baseurl }}{% link experiments/05-sft-rft.md %}))
4. **Val loss is the wrong checkpoint metric.** In 05 val loss rose after step 512 while greedy val accuracy kept improving (best at step 704).
5. **Val is easier than test for every model**, by 11–17 pts. The val split comes from GSM8K train, which the model may have seen in pretraining. Val is used only to compare checkpoints; test numbers are what count.

## Roadmap

| Notebook | Experiment | Status |
|---|---|---|
| [00 Data]({{ site.baseurl }}{% link experiments/00-data.md %}) | Splits, cleaning | Done |
| [01 Zero-shot]({{ site.baseurl }}{% link experiments/01-zeroshot.md %}) | Exp 1: 0.5B baseline | Done |
| [02 SFT gold]({{ site.baseurl }}{% link experiments/02-sft-gold.md %}) | Exp 2: SFT on gold solutions | Done (hurt) |
| [03 SFT vs zero-shot]({{ site.baseurl }}{% link experiments/03-sft-vs-zeroshot.md %}) | Why SFT hurt | Done |
| [04 RFT data]({{ site.baseurl }}{% link experiments/04-rft-data.md %}) | Rejection sampling | Done |
| [05 SFT on RFT]({{ site.baseurl }}{% link experiments/05-sft-rft.md %}) | Exp 2b: RFT-SFT, the start point for RL | Done |
| — | Exp 1b: 7B (4-bit) zero-shot baseline | Planned |
| — | Exp 3: 7B QLoRA SFT (upper bound / teacher) | Planned |
| — | Exp 4: GRPO, outcome reward (control) | Planned |
| — | Dr. GRPO, same config as Exp 4 | Planned |
| — | Exp 5: GRPO + LLM-judge process reward | Planned |
| — | Exp 6: GRPO + on-policy distillation from 7B | Planned |
| — | Exp 7: λ ablation · Exp 8: reward reliability | Planned |
