"""Greedy eval of the 7B model (optionally + LoRA adapter) on 2 GPUs: one process per GPU, each takes every 2nd problem.

Notebook:  results, metrics = run_sharded_eval(data_dir, ["gsm8k_val", "gsm8k_test", "svamp_test"], out_dir, adapter=...)
Per GPU:   python eval7b.py <data_dir> <splits,comma> <out_dir> <rank> <world> <adapter|none> <limit|0> <batch_size>
"""
import json, sys, time
from pathlib import Path

import pandas as pd

from common import SPLIT_KEYS, extract_pred, is_correct, load_split, run_on_gpus, save_json, summarize


def run_sharded_eval(data_dir, splits, out_dir, adapter=None, limit=None, batch_size=32, world=2, log=None):
    """Returns ({split: predictions df}, {metrics key: metrics dict}); writes predictions_<split>.jsonl."""
    out_dir, shard_dir = Path(out_dir), Path(out_dir) / "eval_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    args = [[data_dir, ",".join(splits), shard_dir, rank, world, adapter or "none", limit or 0, batch_size] for rank in range(world)]
    run_on_gpus(Path(__file__), args, log)

    results, metrics = {}, {}
    for split in splits:
        df = load_split(data_dir, split, limit)
        rows = [json.loads(line) for r in range(world) for line in (shard_dir / f"{split}_rank{r}.jsonl").read_text().splitlines()]
        res = pd.DataFrame(rows).set_index("idx").sort_index()
        df = df.join(res)
        wall = max(json.loads((shard_dir / f"{split}_rank{r}.time").read_text()) for r in range(world))
        df.to_json(out_dir / f"predictions_{split}.jsonl", orient="records", lines=True)
        results[split], metrics[SPLIT_KEYS.get(split, split)] = df, summarize(df, wall, world)
        if log:
            log.info(f"{split}: {metrics[SPLIT_KEYS.get(split, split)]}")
    return results, metrics


def main(data_dir, splits, out_dir, rank, world, adapter, limit, batch_size):
    from common import get_logger
    from model7b import generate_greedy, load_model
    rank, world, limit, batch_size = int(rank), int(world), int(limit) or None, int(batch_size)
    log = get_logger(name=f"gpu{rank}")
    t0 = time.time()
    model, tok = load_model(None if adapter == "none" else adapter)
    log.info(f"model loaded ({'base' if adapter == 'none' else adapter}) in {time.time() - t0:.0f}s")
    for split in splits.split(","):
        df = load_split(data_dir, split, limit)
        order = df.question.str.len().sort_values(ascending=False).index[rank::world]   # long first, less padding
        t0 = time.time()
        texts, n_tokens = generate_greedy(model, tok, list(df.question[order]), batch_size)
        rows = []
        for idx, text, n in zip(order, texts, n_tokens):
            pred, has_format = extract_pred(text)
            rows.append(dict(idx=int(idx), response=text, pred=pred, has_format=has_format,
                             correct=is_correct(pred, df.gold[idx]), n_tokens=n))
        Path(out_dir, f"{split}_rank{rank}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        save_json(time.time() - t0, Path(out_dir, f"{split}_rank{rank}.time"))
        log.info(f"{split}: {len(rows)} problems | acc {sum(r['correct'] for r in rows) / max(len(rows), 1):.3f} | "
                 f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
