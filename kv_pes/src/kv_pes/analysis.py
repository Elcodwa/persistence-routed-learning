"""Aggregation + paired statistical tests over results.jsonl (Phase 4).

Produces, for every comparison of interest, means, standard deviations, and
exact sign-test p-values -- including the comparisons where PES LOSES
(none are filtered out).
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import List, Optional

try:
    from .stats import paired_compare, summarize
except (ImportError, ValueError):   # loaded outside the package (report script)
    from stats import paired_compare, summarize

BASELINES = ["full", "random", "window", "streaming", "h2o"]


def load_records(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _cell(records, **flt):
    """Mean score per (model, task, difficulty, budget, policy, seed, case_id)
    for records matching the filter."""
    sel = [r for r in records if all(r.get(k) == v for k, v in flt.items())]
    return sel


def cell_summary(records, **flt) -> dict:
    return summarize([r["score"] for r in _cell(records, **flt)])


def paired_vs_baselines(records, policy_a: str = "pes", model: Optional[str] = None,
                        task: Optional[str] = None, budget: Optional[int] = None,
                        difficulty: Optional[int] = None) -> List[dict]:
    """Pair PES against every baseline case-by-case (identical prompts and
    seeds), reporting mean/std/p for each comparison."""
    flt = {"policy": policy_a}
    if model is not None:
        flt["model"] = model
    if task is not None:
        flt["task"] = task
    if budget is not None:
        flt["budget"] = budget
    if difficulty is not None:
        flt["difficulty"] = difficulty
    a_recs = _cell(records, **flt)
    a_by = {(r["model"], r["task"], r["difficulty"], r["budget"], r["seed"], r["case_id"]): r
            for r in a_recs}
    out = []
    for base in BASELINES:
        bflt = dict(flt, policy=base)
        b_recs = _cell(records, **bflt)
        common = []
        for r in b_recs:
            k = (r["model"], r["task"], r["difficulty"], r["budget"], r["seed"], r["case_id"])
            if k in a_by:
                common.append((a_by[k]["score"], r["score"]))
        if not common:
            continue
        pa = paired_compare([x for x, _ in common], [y for _, y in common])
        pa.update({"policy_a": policy_a, "policy_b": base,
                   "model": model, "task": task, "budget": budget,
                   "difficulty": difficulty,
                   "a_wins": sum(1 for x, y in common if x > y),
                   "b_wins": sum(1 for x, y in common if y > x),
                   "ties": sum(1 for x, y in common if x == y)})
        out.append(pa)
    return out


def dose_response(records, model: str, task: str, difficulty: int = 0,
                  seeds=(0, 1, 2)) -> List[dict]:
    """Mean accuracy per (policy, budget) for the dose-response curve
    (mirrors the paper's Figure 4 style)."""
    rows = []
    budgets = sorted({r["budget"] for r in records
                      if r["model"] == model and r["task"] == task
                      and r["difficulty"] == difficulty and r["budget"] >= 0})
    for budget in budgets:
        for pol in sorted({r["policy"] for r in records
                           if r["model"] == model and r["budget"] == budget
                           and r["difficulty"] == difficulty}):
            recs = _cell(records, model=model, task=task, difficulty=difficulty,
                         budget=budget, policy=pol)
            if recs:
                s = summarize([r["score"] for r in recs])
                s.update({"model": model, "task": task, "difficulty": difficulty,
                          "budget": budget, "policy": pol})
                rows.append(s)
    return rows


def needle_position_curve(records, model: str, budget: int, seeds=(0, 1, 2)) -> List[dict]:
    """Mean accuracy per (policy, needle position task) at a fixed budget."""
    rows = []
    for task in ["needle_early", "needle_mid", "needle_late"]:
        for pol in sorted({r["policy"] for r in records
                           if r["model"] == model and r["budget"] == budget}):
            recs = _cell(records, model=model, task=task, budget=budget, policy=pol)
            if recs:
                s = summarize([r["score"] for r in recs])
                s.update({"model": model, "budget": budget, "policy": pol, "task": task})
                rows.append(s)
    return rows
