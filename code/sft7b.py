"""QLoRA SFT of the 7B model with Unsloth + TRL SFTTrainer (one GPU). Adapters saved every `save_every` steps."""
from unsloth import FastLanguageModel  # noqa: F401  (import before trl so Unsloth can patch it)

import json, math, os, subprocess, sys, time
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from common import fmt_time, prompt_msgs, summarize


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


class ValEarlyStopping(TrainerCallback):
    """Early stopping on greedy val accuracy without slowing training.

    Every saved adapter is evaluated (eval7b.py, val split) on the spare GPU `gpu` while training continues on GPU 0;
    one eval at a time, later checkpoints wait in a queue. After `patience` evals in a row without a new best
    val accuracy, training stops. Results: self.results = {checkpoint name: val metrics}.
    """

    def __init__(self, data_dir, out_dir, log, patience=2, gpu=1, limit=None, batch_size=32):
        self.data_dir, self.out_dir, self.log, self.patience, self.gpu = data_dir, Path(out_dir), log, patience, gpu
        self.limit, self.batch_size = limit, batch_size
        self.queue, self.running, self.results, self.best, self.bad, self.stop = [], None, {}, -1.0, 0, False

    def _start(self, ckpt):
        shard = self.out_dir / "val_evals" / ckpt.name
        shard.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, Path(__file__).with_name("eval7b.py"), self.data_dir, "gsm8k_val", shard, 0, 1, ckpt,
               self.limit or 0, self.batch_size]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu), "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(list(map(str, cmd)), env=env, stdout=open(shard / "eval.log", "w"), stderr=subprocess.STDOUT)
        self.running = (ckpt, shard, proc)
        self.log.info(f"VAL eval of {ckpt.name} started on GPU {self.gpu}")

    def _poll(self, wait=False):
        """Collect a finished eval (or wait for all with wait=True) and start the next queued one."""
        while True:
            if self.running is None:
                if not self.queue:
                    return
                self._start(self.queue.pop(0))
            ckpt, shard, proc = self.running
            if wait:
                proc.wait()
            if proc.poll() is None:
                return
            assert proc.returncode == 0, f"val eval of {ckpt.name} failed, see {shard / 'eval.log'}"
            rows = pd.DataFrame([json.loads(line) for line in (shard / "gsm8k_val_rank0.jsonl").read_text().splitlines()])
            m = summarize(rows, json.loads((shard / "gsm8k_val_rank0.time").read_text()), n_gpus=1)
            self.results[ckpt.name] = m
            if m["accuracy"] > self.best:
                self.best, self.bad = m["accuracy"], 0
            else:
                self.bad += 1
            self.stop = self.stop or self.bad >= self.patience
            self.log.info(f"VAL {ckpt.name}: acc {m['accuracy']:.4f} | best {self.best:.4f} | "
                          f"no improvement {self.bad}/{self.patience}")
            self.running = None

    def on_save(self, args, state, control, **kw):
        self.queue.append(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        self._poll()

    def on_step_end(self, args, state, control, **kw):
        self._poll()
        if self.stop and not control.should_training_stop:
            self.log.info(f"early stopping at step {state.global_step}: {self.patience} val evals without improvement")
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kw):
        self._poll(wait=True)                    # finish the evals of the remaining saved adapters


def train_lora(model, tok, train_df, cfg, out_dir, log, data_dir):
    """cfg keys: per_device_bs, grad_accum, epochs, lr, warmup_ratio, weight_decay, max_grad_norm, max_len,
    saves_per_epoch, max_hours, seed, max_steps (None = full), patience, eval_batch, limit.
    Returns (sorted adapter dirs, log_history df, {adapter name: val metrics})."""
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
    val_cb = ValEarlyStopping(data_dir, out_dir, log, patience=cfg["patience"], gpu=1, limit=cfg["limit"],
                              batch_size=cfg["eval_batch"])
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok,
                         callbacks=[LogAndTimeGuard(log, cfg["max_hours"]), val_cb])

    ex = trainer.train_dataset[0]                     # loss-mask check: only the answer should be trained on
    mask = ex["completion_mask"] if "completion_mask" in ex else [int(l != -100) for l in ex["labels"]]
    log.info(f"example 0: {len(ex['input_ids'])} tokens, loss on {sum(mask)}: "
             f"{tok.decode([t for t, m in zip(ex['input_ids'], mask) if m])[:300]!r}")
    log.info(f"{len(ds)} examples | {steps_per_epoch} steps/epoch | adapter saved every {save_every} steps")

    trainer.train()
    ckpts = sorted(Path(out_dir, "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return ckpts, pd.DataFrame(trainer.state.log_history), val_cb.results
