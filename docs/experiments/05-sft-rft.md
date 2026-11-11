---
title: 05 SFT on RFT data
parent: Experiments
nav_order: 5
---

# 05 SFT on RFT data

[Kaggle notebook](https://www.kaggle.com/code/tensorpanda231/05-sft-with-rft-qwen-05b)

## Goal
Fine-tune on the model's own correct answers from [04]({{ site.baseurl }}{% link experiments/04-rft-data.md %}) instead of gold solutions. If it beats zero-shot, it becomes the start point for RL.

## Setup
- **Same as [02]({{ site.baseurl }}{% link experiments/02-sft-gold.md %}):** full fine-tuning, TRL `SFTTrainer`, DDP on 2× T4, fp16, gradient checkpointing, lr 1e-5 with a cosine schedule, effective batch 64.
- **Different from 02:**
  - Data: 16,515 RFT answers for 5,982 problems. Median 376 tokens with the prompt, max 724.
  - 3 epochs, 4 × 8 per device, max length 768.
  - **Checkpoint picked by greedy val accuracy** (500 problems) instead of val loss, evaluated every 64 steps, patience 3.
- **Outcome:** the best step was 704 of 777. Training did not stop early.

## Results

| | Val | GSM8K test | SVAMP |
|---|---|---|---|
| Accuracy | **66.6%** (step 0: 57.8) | **49.4%** (zero-shot 43.8) | **61.9%** (zero-shot 53.7) |
| Uses `####` format | 97.4% | 94.4% | 99.3% |
| Mean answer length (tokens) | 292 | 309 | 224 |
| Truncated at 512 tokens | 2.8% | 5.6% | 0.6% |

![Accuracy: zero-shot vs gold SFT vs RFT SFT]({{ '/assets/plots/05_accuracy_compare.png' | relative_url }})

### Val accuracy vs val loss during training

| Step | 0 | 64 | 128 | 192 | 256 | 320 | 384 | 448 | 512 | 576 | 640 | **704** | 768 | 777 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Val acc | 57.8 | 58.2 | 55.0 | 59.8 | 58.6 | 59.6 | 64.4 | 60.0 | 62.0 | 65.8 | 65.0 | **66.4** | 65.4 | 64.6 |
| Val loss | .174 | .161 | .161 | .162 | .162 | .162 | .161 | .160 | .157 | .168 | .167 | .167 | .167 | .167 |

![Val accuracy vs val loss]({{ '/assets/plots/05_val_accuracy.png' | relative_url }})

![Training loss]({{ '/assets/plots/05_loss.png' | relative_url }})

![Learning rate and gradient norm]({{ '/assets/plots/05_lr_gradnorm.png' | relative_url }})

![GPU usage]({{ '/assets/plots/05_gpu_usage.png' | relative_url }})

## Takeaways
- **RFT fixed the SFT drop:** +5.6 pts on GSM8K and +8.2 on SVAMP over zero-shot. The SVAMP gain shows it is not just fitting GSM8K. (A paired significance test vs zero-shot is still to do.)
- **The native style was kept:** answers stay ~300 tokens like zero-shot (gold SFT: ~90), while the `####` format is now used 94% of the time (zero-shot: 17%).
- **Val loss would have picked the wrong checkpoint:** it bottoms out at step 512 and then rises, while val accuracy keeps improving until step 704.
- **Train loss drops in steps at each epoch boundary** (steps ~259, ~518), which is memorization of repeated answers. Val accuracy still rose, so it didn't hurt here.
- **Val is ~17 pts above test**, and the base model shows the same gap (57.8 vs 43.8). Hypothesis: GSM8K train (and therefore val) was seen in pretraining. Use val only for relative choices.
- **Caveats:**
  - 66.6 is the best of 14 noisy val evaluations (±2 pts each), so it is slightly optimistic. Test is unaffected.
  - 5.6% of GSM8K answers still hit the 512-token limit.

## Cost
4.47 GPU-hours of training (134 min on 2× T4) + 0.61 of evaluation = 5.1. Including the RFT data from [04]({{ site.baseurl }}{% link experiments/04-rft-data.md %}), the full method costs about 11.6 GPU-hours.
