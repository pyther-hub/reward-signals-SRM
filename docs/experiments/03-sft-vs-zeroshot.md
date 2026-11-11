---
title: 03 Why SFT hurt
parent: Experiments
nav_order: 3
---

# 03 SFT vs zero-shot: why did SFT make the model worse?

[Kaggle notebook](https://www.kaggle.com/code/tensorpanda231/03-sft-vs-zeroshot)

## Goal
Find out whether the SFT drop is real, and what caused it. The answer decides how to build the start point for RL.

## Setup
- Reuses the greedy predictions from [01]({{ site.baseurl }}{% link experiments/01-zeroshot.md %}) and [02]({{ site.baseurl }}{% link experiments/02-sft-gold.md %}).
- Every comparison is **paired**: same problem, two models.
- Bootstrap: 10,000 resamples.
- pass@k: 8 samples per GSM8K test problem at T = 0.7, with top-k / top-p / repetition penalty turned off.

## Results

### 1. Is the drop real? Yes.

| | Zero-shot | SFT | Difference (95% CI) |
|---|---|---|---|
| GSM8K | 43.8 [41.0, 46.5] | 32.7 [30.2, 35.2] | **−11.1** [−14.2, −8.0] |
| SVAMP | 53.7 [50.6, 56.8] | 35.5 [32.5, 38.5] | **−18.2** [−21.6, −14.8] |

![Accuracy with confidence intervals]({{ '/assets/plots/03_accuracy_ci.png' | relative_url }})

### 2. SFT broke problems the base model could already solve

| | Right → right | Right → wrong | Wrong → right | Wrong → wrong | Share of solved problems broken | McNemar p |
|---|---|---|---|---|---|---|
| GSM8K | 294 | **283** | 137 | 605 | 49% | 9e-13 |
| SVAMP | 280 | **257** | 75 | 388 | 48% | 2e-24 |

### 3. Answers got much shorter

Median answer tokens: zero-shot 292, SFT 81, gold solutions 86 (GSM8K). SFT copied the gold length.

![Answer lengths]({{ '/assets/plots/03_lengths.png' | relative_url }})

The drop happens at every difficulty level (number of steps in the gold solution):

| Steps | Problems | Zero-shot | SFT | Drop |
|---|---|---|---|---|
| 1–2 | 326 | 62.3 | 49.7 | −12.6 |
| 3 | 371 | 48.8 | 35.8 | −12.9 |
| 4 | 297 | 37.0 | 27.3 | −9.8 |
| 5 | 175 | 27.4 | 23.4 | −4.0 |
| 6+ | 150 | 23.3 | 9.3 | −14.0 |

![Accuracy by number of steps]({{ '/assets/plots/03_acc_by_steps.png' | relative_url }})

### 4. Gold text is unfamiliar to the model

Measured on the 577 GSM8K test problems the zero-shot model solved, scored by the zero-shot model itself:

| Text | Mean NLL per token | Perplexity |
|---|---|---|
| Gold solution | 0.906 | 2.48 |
| Model's own answer | 0.139 | 1.15 |

Gold is less likely than the model's own answer on **100%** of these problems. Fine-tuning on high-loss, foreign-style text forces large changes (cf. Ren et al. 2024, *I Learn Better If You Speak My Language*).

![Familiarity]({{ '/assets/plots/03_familiarity.png' | relative_url }})

### 5. Exploration: pass@k and group diversity

| k | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Zero-shot | 41.7 | 55.0 | 66.2 | 75.4 |
| SFT | 27.1 | 39.1 | 51.7 | 63.9 |

![pass@k]({{ '/assets/plots/03_pass_at_k.png' | relative_url }})

GRPO learns only from groups of 8 samples where some answers are right and some wrong ("mixed"):

| | All wrong | Mixed | All right | Distinct answers per problem |
|---|---|---|---|---|
| Zero-shot | 24.6% | **65.6%** | 9.8% | 4.5 |
| SFT | 36.1% | 60.3% | 3.6% | 5.6 |

## Takeaways
- The SFT drop is large, significant, and not just "failed to improve": SFT **broke** half of the already-solved problems.
- **Likely cause:** imitating short, unfamiliar gold text replaced the model's own step-by-step reasoning.
- **Fix to try:** train on the model's *own* correct answers instead, which is rejection sampling ([04]({{ site.baseurl }}{% link experiments/04-rft-data.md %})). Zero-shot pass@8 of 75% shows there are plenty of correct answers to harvest.
- The base model is a better RL start point than gold-SFT: it has more mixed groups and a higher pass@k.

## Cost
About 2.5 GPU-hours, mostly pass@8 sampling (74 min on 2× T4).
