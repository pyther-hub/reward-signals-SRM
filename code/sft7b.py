"""QLoRA SFT of the 7B model with Unsloth + TRL SFTTrainer (one GPU). Adapters saved every `save_every` steps."""
from unsloth import FastLanguageModel  # noqa: F401  (import before trl so Unsloth can patch it)

import math, time
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from common import fmt_time, prompt_msgs


def make_dataset(df):
    """Prompt-completion (conversational) format -> loss only on the answer."""
    return Dataset.from_list([{"prompt": prompt_msgs(q), "completion": [{"role": "assistant", "content": s}]}
                              for q, s in zip(df.question, df.solution)])


class LogAndTimeGuard(TrainerCallback):
    """One line per logging step; saves + stops after `max_hours`."""

    def __init__(self, log, max_hours):
        self.log, self.max_sec, self.t0 = log, max_hours * 3600, time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            el = time.time() - self.t0
            self.log.info(f"step {state.global_step}/{state.max_steps} | ep {state.epoch:.2f} | loss {logs['loss']:.4f} | "
                          f"lr {logs.get('learning_rate', 0):.2e} | gnorm {logs.get('grad_norm', float('nan')):.2f} | "
                          f"elapsed {fmt_time(el)} | ETA {fmt_time(el / max(state.global_step, 1) * (state.max_steps - state.global_step))}")

    def on_step_end(self, args, state, control, **kw):
        if time.time() - self.t0 > self.max_sec:
            self.log.info(f"time budget reached at step {state.global_step} -> save + stop")
            control.should_save, control.should_training_stop = True, True


def train_lora(model, tok, train_df, cfg, out_dir, log):
    """cfg keys: per_device_bs, grad_accum, epochs, lr, warmup_ratio, weight_decay, max_grad_norm, max_len,
    saves_per_epoch, max_hours, seed, max_steps (None = full). Returns (sorted adapter dirs, log_history df)."""
    ds = make_dataset(train_df)
    steps_per_epoch = math.ceil(len(ds) / (cfg["per_device_bs"] * cfg["grad_accum"]))
    save_every = max(1, steps_per_epoch // cfg["saves_per_epoch"])
    args = SFTConfig(
        output_dir=str(Path(out_dir) / "checkpoints"), max_length=cfg["max_len"], completion_only_loss=True, packing=False,
        per_device_train_batch_size=cfg["per_device_bs"], gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"], max_steps=cfg.get("max_steps") or -1,
        learning_rate=cfg["lr"], lr_scheduler_type="cosine", warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"], max_grad_norm=cfg["max_grad_norm"], optim="adamw_8bit",
        fp16=True, bf16=False, logging_steps=5, save_strategy="steps", save_steps=save_every, save_only_model=True,
        report_to="none", disable_tqdm=True, log_level="error", seed=cfg["seed"],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok,
                         callbacks=[LogAndTimeGuard(log, cfg["max_hours"])])

    ex = trainer.train_dataset[0]                     # loss-mask check: only the answer should be trained on
    mask = ex["completion_mask"] if "completion_mask" in ex else [int(l != -100) for l in ex["labels"]]
    log.info(f"example 0: {len(ex['input_ids'])} tokens, loss on {sum(mask)}: "
             f"{tok.decode([t for t, m in zip(ex['input_ids'], mask) if m])[:300]!r}")
    log.info(f"{len(ds)} examples | {steps_per_epoch} steps/epoch | adapter saved every {save_every} steps")

    trainer.train()
    ckpts = sorted(Path(out_dir, "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return ckpts, pd.DataFrame(trainer.state.log_history)
