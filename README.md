# reward-signals-SRM

### 📄 Report: **[pyther-hub.github.io/reward-signals-SRM](https://pyther-hub.github.io/reward-signals-SRM/)**

**Self-Generated versus Human-Written Supervision for Small-Model Math Reasoning: a paired study with Qwen2.5-0.5B.**

This repo studies how different post-training signals change math reasoning in a small LLM. Qwen2.5-0.5B-Instruct is trained on GSM8K and evaluated on GSM8K test and on SVAMP, which is held out from all training. It compares four recipes:

1. **Zero-shot:** the base model.
2. **SFT on gold:** full fine-tuning on the human-written GSM8K solutions.
3. **SFT on RFT:** rejection-sampling fine-tuning. The model samples 8 answers per problem, and it is fine-tuned on its own correct answers.
4. **GRPO:** RL from the RFT checkpoint with a binary reward: 1 if the final `#### <number>` is correct, else 0.

Every run uses the same prompt, the same data splits and greedy decoding, and the same answer extraction for evaluation and rewards. Models are compared problem by problem with paired bootstrap CIs and McNemar tests. Everything runs on 2× Kaggle T4 GPUs (fp16).

## Results

| Model | GSM8K test | SVAMP |
|---|---|---|
| Zero-shot | 43.7 | 53.7 |
| SFT on gold | 32.7 | 35.5 |
| SFT on RFT | 49.4 | 61.9 |
| GRPO | **54.4** | **63.2** |

Greedy accuracy (%).
- **SFT on gold hurts:** it breaks about half of the problems the base model already solved.
- **RFT and GRPO help:** both improve accuracy, but the GRPO gain over RFT is significant only on GSM8K.

The [report](https://pyther-hub.github.io/reward-signals-SRM/) has the full tables, CIs, analysis and limitations.

## Kaggle experiments

Each experiment is one Kaggle session that runs end-to-end and saves its config, metrics and logs. All notebooks are on [kaggle.com/tensorpanda231](https://www.kaggle.com/tensorpanda231).

| # | Experiment | Notebook |
|---|---|---|
| 00 | Data splits and cleaning | [Kaggle](https://www.kaggle.com/code/tensorpanda231/00-data-math-reasoning) |
| 01 | Zero-shot evaluation | [Kaggle](https://www.kaggle.com/code/tensorpanda231/01-zeroshot-qwen2-5-0-5b) |
| 02 | SFT on gold solutions | [Kaggle](https://www.kaggle.com/code/tensorpanda231/02-sft-0-5b) |
| 03 | Why gold SFT hurts: paired tests, lengths, pass@k | [Kaggle](https://www.kaggle.com/code/tensorpanda231/03-sft-vs-zeroshot) |
| 04 | RFT data: sample 8 answers per problem, keep the correct ones | [Kaggle](https://www.kaggle.com/code/tensorpanda231/04-rft-data) |
| 05 | SFT on RFT data | [Kaggle](https://www.kaggle.com/code/tensorpanda231/05-sft-with-rft-qwen-05b) |
| 06 | GRPO with outcome reward (TRL + vLLM) | [Kaggle](https://www.kaggle.com/code/tensorpanda231/06-grpo-0-5b) |
| 12 | Paired comparison of the four models | [Kaggle](https://www.kaggle.com/code/tensorpanda231/12-compare-0-5b) |

## Code structure

```
├── docs/                 GitHub Pages site (the report: docs/index.html)
├── kaggle_experiments/   executed notebooks downloaded back from Kaggle
├── code/                 shared helpers that notebooks import after cloning the repo
│   ├── common.py         prompt template, answer extraction, is_correct (shared by eval and rewards)
│   ├── rft.py            filter, dedupe and summarize RFT samples
│   └── ...               model loading, sampling, evaluation, SFT helpers
├── utils/logger.py       console logger + W&B init (logs config and git hash)
├── dataset.py            GSM8K loading
└── requirements.txt
```

## Setup

- **Model:** Qwen2.5-0.5B-Instruct, full fine-tuning.
- **Data:** GSM8K train 6,973 · val 500 (held out from train) · test 1,319; SVAMP 1,000 (evaluation only).
- **Evaluation:** greedy decoding, max 512 new tokens, exact match on the number after `####`.
- **Libraries:** Transformers, TRL (`SFTTrainer`, `GRPOTrainer`), vLLM for GRPO generation.
- **Theory reference:** *RLHF* by Nathan Lambert.
