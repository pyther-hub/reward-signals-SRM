---
title: 06 GRPO (outcome reward)
parent: Experiments
nav_order: 6
---

# 06 GRPO with outcome reward (Exp 4, control run)

[Kaggle notebook](https://www.kaggle.com/code/tensorpanda231/06-grpo-0-5b)

## Goal
Reinforcement learning from the [RFT-SFT model]({{ site.baseurl }}{% link experiments/05-sft-rft.md %}), with the simplest possible reward: 1 if the final `#### <number>` is correct, else 0. This is the control run. Later reward signals (LLM judge, teacher distillation) and Dr. GRPO reuse exactly this config.

## Setup
- **Start and reference model:** RFT-SFT from 05, full fine-tuning (fp32 master weights, fp16 mixed precision).
- **Framework:** TRL `GRPOTrainer` 1.17.1. Generation runs on a **vLLM 0.32 server on GPU 0**, and training runs on GPU 1. After every step the trainer pushes its new weights to vLLM over NCCL, with direct GPU-to-GPU (P2P) copy disabled because it hangs on Kaggle T4s.
- **Reward:** 1.0 if the answer has a `####` line *and* the number after it is correct, else 0.0. The last-number fallback used in eval earns no reward.
- **Prompts:** GSM8K train minus the problems the base model solved 8/8 in [04]({{ site.baseurl }}{% link experiments/04-rft-data.md %}), leaving 6,279 prompts.
- **"Normal" GRPO:**
  - `loss_type="grpo"` (sequence mean); `scale_rewards="group"` (advantage = (r − group mean) / group std).
  - `beta=0.04` (KL to the reference model); one update per rollout batch.
  - 8 answers per prompt at T = 1.0, max 512 tokens.
  - 32 prompts × 8 = 256 answers per step.
  - lr 1e-6, constant schedule after 10 warmup steps; gradient clipping 1.0.
- **Checkpoint choice:** greedy val accuracy (500 problems) at step 0 and every 50 steps. The run stopped on the 9.5 h time budget at step 325 of 400.

## Results

| | Val | GSM8K test | SVAMP |
|---|---|---|---|
| RFT-SFT (start) | 66.6% | 49.4% | 61.9% |
| **GRPO (best = step 325)** | **69.8%** | **54.4%** | **63.2%** |
| Change | +3.2 | **+5.0** | +1.3 |
| Uses `####` format | 96.6% | 94.4% | 99.0% |

### Val accuracy during training

| Step | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 325 |
|---|---|---|---|---|---|---|---|---|
| Val acc | 66.8 | 67.6 | 67.4 | 67.6 | 68.8 | 69.4 | 69.2 | **69.8** |

![GRPO training curves]({{ '/assets/plots/06_curves.png' | relative_url }})

### Training health

| Signal | What happened | Reading |
|---|---|---|
| Reward (T = 1 samples) | ~0.60 → ~0.67, flat after ~100 steps | learning, then plateau |
| KL to reference | 0 → 0.012, slow and smooth | stays close to RFT-SFT |
| Entropy | 0.125 → 0.10 | mild sharpening, no collapse |
| Answer length | ~300 tokens throughout | no length drift |
| Truncated (512 tokens) | 4–5% throughout | stable |
| Groups with zero reward std | ~0.40–0.47 | ~45% of prompts gave no gradient |

## Takeaways
- **GRPO adds +5.0 pts on GSM8K on top of RFT.** It is now +10.6 over zero-shot, the best model so far. Val was still rising when the time budget stopped training.
- **SVAMP barely moved (+1.3, within noise).** RL on GSM8K mostly sharpened GSM8K-style solving, while most of the generalization gain came from RFT. A paired significance test is still to do.
- **The run is healthy:** small KL, no entropy collapse, no length blow-up, format kept. With a binary, format-gated reward there was no sign of reward hacking.
- **Wasted compute:** ~45% of groups were all-correct (the mean reward is 0.67), so they gave zero advantage. The prompt filter used the *base* model's 8/8 stats; filtering with the RFT-SFT model, or dynamic sampling, would recover most of this.
- **The config is conservative:** KL is only 0.012 and the KL term is negligible, so a higher learning rate would likely move faster. The config is still kept frozen so that Dr. GRPO and Exp 5–6 compare against the same control.

## Cost
9.6 h of training on 2 GPUs = **19.2 GPU-hours**. The vLLM GPU is mostly idle while the training GPU works. Final eval on one GPU: ~0.6 GPU-h.
