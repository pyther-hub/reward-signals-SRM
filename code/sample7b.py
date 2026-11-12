"""RFT sampling with vLLM on 2 GPUs: one vLLM engine per GPU (same bnb-4bit weights LoRA trains on), every 2nd problem each.

Notebook:  samples_df = run_sharded_sampling(data_dir, "gsm8k_train", out_dir, n=8)
Per GPU:   python sample7b.py <data_dir> <split> <out_dir> <rank> <world> <n> <temperature> <limit|0> <seed>
"""
import json, sys, time
from pathlib import Path

import pandas as pd

from common import extract_pred_strict, is_correct, load_split, run_on_gpus

MODEL_ID = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_TOKENS = 512


def run_sharded_sampling(data_dir, split, out_dir, n=8, temperature=0.7, limit=None, seed=42, world=2, log=None):
    """Returns (samples df sorted by problem_id/sample_id, wall seconds); writes samples_<split>.jsonl."""
    out_dir, shard_dir = Path(out_dir), Path(out_dir) / "sample_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    args = [[data_dir, split, shard_dir, rank, world, n, temperature, limit or 0, seed] for rank in range(world)]
    wall = run_on_gpus(Path(__file__), args, log)
    rows = [json.loads(line) for r in range(world) for line in (shard_dir / f"{split}_rank{r}.jsonl").read_text().splitlines()]
    samples = pd.DataFrame(rows).sort_values(["problem_id", "sample_id"]).reset_index(drop=True)
    expected = len(load_split(data_dir, split, limit)) * n
    assert len(samples) == expected, f"{split}: {len(samples)} samples, expected {expected}"
    samples.to_json(out_dir / f"samples_{split}.jsonl", orient="records", lines=True)
    if log:
        log.info(f"{split}: {len(samples)} samples | sample acc {samples.correct.mean():.3f} | {wall / 60:.1f} min")
    return samples, wall


def main(data_dir, split, out_dir, rank, world, n, temperature, limit, seed, chunk=512):
    from vllm import LLM, SamplingParams
    from common import build_prompt, get_logger
    rank, world, n, limit = int(rank), int(world), int(n), int(limit) or None
    log = get_logger(name=f"gpu{rank}")
    df = load_split(data_dir, split, limit)
    ids = list(range(rank, len(df), world))

    t0 = time.time()
    llm = LLM(model=MODEL_ID, quantization="bitsandbytes", dtype="half", max_model_len=1024,
              gpu_memory_utilization=0.9, seed=int(seed) + rank)
    tok = llm.get_tokenizer()
    log.info(f"vLLM loaded in {time.time() - t0:.0f}s | {split}: {len(ids)} problems x {n} samples")
    params = SamplingParams(n=n, temperature=float(temperature), top_p=1.0, top_k=0, repetition_penalty=1.0,
                            max_tokens=MAX_TOKENS, seed=int(seed) + rank)   # explicit: ignore Qwen's generation_config

    out_path = Path(out_dir, f"{split}_rank{rank}.jsonl")
    out_path.write_text("")
    t0, n_correct, n_done = time.time(), 0, 0
    for start in range(0, len(ids), chunk):                  # chunks only for progress logging / incremental writes
        batch = ids[start:start + chunk]
        outputs = llm.generate([build_prompt(tok, df.question[i]) for i in batch], params, use_tqdm=False)
        rows = []
        for pid, out in zip(batch, outputs):
            for sid, o in enumerate(out.outputs):
                pred, source = extract_pred_strict(o.text)
                ok = is_correct(pred, df.gold[pid])
                rows.append(dict(problem_id=pid, sample_id=sid, question=df.question[pid], gold=df.gold[pid], response=o.text,
                                 pred=pred, source=source, correct=ok, n_tokens=len(o.token_ids), truncated=o.finish_reason == "length"))
                n_correct += ok
        with open(out_path, "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        n_done += len(rows)
        el = time.time() - t0
        log.info(f"{split}: {start + len(batch)}/{len(ids)} problems | sample acc {n_correct / n_done:.3f} | "
                 f"{n_done / el:.1f} samples/s | ETA {(len(ids) - start - len(batch)) * n / max(n_done / el, 1e-9) / 60:.0f} min")


if __name__ == "__main__":
    main(*sys.argv[1:])
