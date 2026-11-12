"""Rejection-sampling filter (same recipe as 04_rft_data): keep correct, non-truncated answers, dedupe by equations."""
import re
from collections import Counter

import pandas as pd

NUMBER = r"\(?\$?\d[\d,]*(?:\.\d+)?%?\)?"
EQ_RE = re.compile(rf"({NUMBER}(?:\s*[-+*/×÷x]\s*{NUMBER})+)\s*=\s*\$?(-?\d[\d,]*(?:\.\d+)?)")
OP_MAP = str.maketrans({"×": "*", "x": "*", "÷": "/"})


def equation_key(response):
    """Sequence of equations in an answer (spaces / $ / commas removed, ×x÷ normalized); falls back to the whitespace-normalized text."""
    equations = tuple(re.sub(r"[\s$,]", "", lhs).translate(OP_MAP) + "=" + rhs.replace(",", "").rstrip(".")
                      for lhs, rhs in EQ_RE.findall(response))
    return equations or " ".join(response.split())


def filter_and_dedupe(samples, max_kept=4):
    """samples: one row per sampled answer (problem_id, question, gold, response, correct, truncated).
    Returns (rft_df[question, solution, gold], problem_stats[n_correct, n_kept] indexed by problem_id)."""
    kept_rows, n_kept = [], Counter()
    for problem_id, group in samples[samples.correct & ~samples.truncated].groupby("problem_id"):
        seen = set()
        for r in group.itertuples():
            key = equation_key(r.response)
            if key in seen or n_kept[problem_id] >= max_kept:
                continue
            seen.add(key)
            n_kept[problem_id] += 1
            last_line = r.response.rstrip().split("\n")[-1].strip()
            solution = r.response if last_line.startswith("####") else r.response.rstrip() + f"\n#### {r.gold}"
            kept_rows.append(dict(question=r.question, solution=solution, gold=r.gold))
    rft = pd.DataFrame(kept_rows, columns=["question", "solution", "gold"])
    stats = (samples.groupby("problem_id").correct.sum().rename("n_correct").to_frame()
             .assign(n_kept=lambda s: [n_kept[pid] for pid in s.index]))
    return rft, stats


def rft_metrics(samples, stats, rft, gold_solutions, tok, n, wall_sec, n_gpus):
    """Same keys as 04's metrics.json."""
    kept_tokens = rft.solution.map(lambda s: len(tok(s).input_ids))
    gold_tokens = gold_solutions.map(lambda s: len(tok(s).input_ids))
    coverage = round(float((stats.n_correct > 0).mean()), 4)
    return {"n_problems": len(stats), "coverage": coverage, "pass_at_1": round(float(samples.correct.mean()), 4),
            f"pass_at_{n}": coverage, "truncated_rate": round(float(samples.truncated.mean()), 4),
            "n_kept": len(rft), "mean_kept_per_problem": round(float(stats.n_kept.mean()), 3),
            "median_tokens_kept": float(kept_tokens.median()) if len(rft) else None,
            "median_tokens_gold": float(gold_tokens.median()),
            "source_all": samples.source.fillna("none").value_counts(normalize=True).round(4).to_dict(),
            "source_correct": samples[samples.correct].source.value_counts(normalize=True).round(4).to_dict(),
            "wall_min": round(wall_sec / 60, 2), "gpu_hours": round(wall_sec * n_gpus / 3600, 3)}
