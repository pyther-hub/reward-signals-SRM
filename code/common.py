"""Shared helpers for the 7B notebooks: prompt, answer checking, data, logging, multi-GPU launcher."""
import json, logging, os, re, subprocess, sys, threading, time
from pathlib import Path

import pandas as pd

INSTRUCTION = "Solve the problem step by step. End your answer with a final line of the form '#### <number>'."
SPLIT_KEYS = {"gsm8k_val": "val", "gsm8k_test": "gsm8k", "svamp_test": "svamp"}   # metrics.json keys used since 01

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


# ---- prompt + answer checking (identical to the 0.5B notebooks)
def prompt_msgs(question):
    return [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]


def build_prompt(tok, question):
    return tok.apply_chat_template(prompt_msgs(question), tokenize=False, add_generation_prompt=True)


def extract_pred(text):
    """Eval: number after the last '####' -> (pred, True); else the last number -> (pred, False)."""
    if "####" in text:
        nums = NUM_RE.findall(text.split("####")[-1])
        if nums:
            return nums[0].replace(",", ""), True
    nums = NUM_RE.findall(text)
    return (nums[-1].replace(",", ""), False) if nums else (None, False)


def extract_pred_strict(text):
    """RFT filter: number after the last '####' -> (pred, '####'); else in the last \\boxed{} -> (pred, 'boxed'); else (None, None)."""
    if "####" in text:
        nums = NUM_RE.findall(text.split("####")[-1])
        if nums:
            return nums[0].replace(",", ""), "####"
    boxed = BOXED_RE.findall(text)
    if boxed:
        nums = NUM_RE.findall(boxed[-1])
        if nums:
            return nums[0].replace(",", ""), "boxed"
    return None, None


def is_correct(pred, gold):
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


# ---- data / io
def load_split(data_dir, name, limit=None):
    """name: gsm8k_train | gsm8k_val | gsm8k_test | svamp_test (00_data output)."""
    df = pd.read_json(Path(data_dir) / f"{name}.jsonl", lines=True, dtype={"gold": str})
    return df.head(limit) if limit else df


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


# ---- logging
def get_logger(out_dir=None, name="main"):
    handlers = [logging.StreamHandler(sys.stdout)]
    if out_dir:
        handlers.append(logging.FileHandler(Path(out_dir) / "run.log"))
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(name)-5s | %(message)s", datefmt="%H:%M:%S",
                        handlers=handlers, force=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    return log


def fmt_time(sec):
    return f"{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m" if sec >= 3600 else f"{int(sec // 60)}m{int(sec % 60):02d}s"


def summarize(df, wall_sec, n_gpus, max_new_tokens=512):
    """Eval metrics, same keys as 01 / 05 metrics.json."""
    return dict(n=len(df), accuracy=round(float(df.correct.mean()), 4), format_rate=round(float(df.has_format.mean()), 4),
                mean_tokens=round(float(df.n_tokens.mean()), 1), truncated_rate=round(float((df.n_tokens >= max_new_tokens).mean()), 4),
                wall_min=round(wall_sec / 60, 2), gpu_hours=round(wall_sec * n_gpus / 3600, 3))


# ---- run one script per GPU (each process sees only its GPU), stream its output with a [rank] prefix
def run_on_gpus(script, args_per_rank, log=None):
    def pump(rank, proc):
        for line in proc.stdout:
            print(f"[gpu{rank}] {line}", end="", flush=True)

    procs, threads = [], []
    for rank, args in enumerate(args_per_rank):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(rank), "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
        proc = subprocess.Popen([sys.executable, str(script), *map(str, args)], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append(proc)
        threads.append(threading.Thread(target=pump, args=(rank, proc), daemon=True))
        threads[-1].start()
    t0 = time.time()
    codes = [p.wait() for p in procs]
    for t in threads:
        t.join()
    if log:
        log.info(f"{Path(script).name} on {len(procs)} GPU(s) done in {fmt_time(time.time() - t0)}")
    assert all(c == 0 for c in codes), f"{Path(script).name} failed on some GPU: exit codes {codes}"
    return time.time() - t0
