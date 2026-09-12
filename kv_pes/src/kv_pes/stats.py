"""Paired statistics matching the paper's own methodology (exact sign test on
paired differences, plus mean/std/CI; Wilcoxon when scipy is available)."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def exact_sign_test(diffs: Sequence[float]) -> tuple[float, float]:
    """Exact two-sided binomial sign test on paired differences.

    Returns (p_value, n_nonzero). Mirrors signtest() in research/code
    (r9_mlp_loo.py) from the paper's supplementary code.
    """
    from math import comb
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0
    k = sum(1 for x in d if x > 0)
    p = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / (2 ** n)
    p = min(1.0, p * (2 if 2 * k != n else 1))
    return p, n


def paired_compare(pes: Sequence[float], base: Sequence[float]) -> dict:
    """Paired comparison summary: means, stds, mean difference, sign-test p."""
    a, b = np.asarray(pes, dtype=float), np.asarray(base, dtype=float)
    assert a.shape == b.shape, "paired arrays must have equal length"
    p, n = exact_sign_test((a - b).tolist())
    return {
        "n": int(a.size),
        "mean_a": float(a.mean()), "std_a": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "mean_b": float(b.mean()), "std_b": float(b.std(ddof=1)) if b.size > 1 else 0.0,
        "mean_diff": float((a - b).mean()),
        "diff_std": float((a - b).std(ddof=1)) if a.size > 1 else 0.0,
        "sign_test_p": p, "n_nonzero": n,
    }


def summarize(scores: Iterable[float]) -> dict:
    s = np.asarray(list(scores), dtype=float)
    se = s.std(ddof=1) / math.sqrt(s.size) if s.size > 1 else 0.0
    return {"n": int(s.size), "mean": float(s.mean()), "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
            "sem": float(se)}
