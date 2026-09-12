"""Plots (Phase 6). matplotlib only, no seaborn."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICY_ORDER = ["full", "h2o", "pes", "streaming", "window", "random"]
COLORS = {"full": "k", "h2o": "tab:blue", "pes": "tab:red",
          "streaming": "tab:green", "window": "tab:orange", "random": "tab:gray"}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def plot_dose_response(rows: List[dict], model: str, task: str, out_dir: str):
    """Accuracy vs cache budget per policy (paper Fig. 4 style dose-response)."""
    by_pol = defaultdict(lambda: ([], []))
    for r in rows:
        if r["model"] == model and r["task"] == task:
            xs, ys = by_pol[r["policy"]]
            xs.append(r["budget"])
            ys.append(r["mean"])
    if not by_pol:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for pol in POLICY_ORDER:
        if pol in by_pol:
            xs, ys = zip(*sorted(zip(*by_pol[pol])))
            ax.plot(xs, ys, marker="o", label=pol, color=COLORS.get(pol))
    ax.set_xlabel("cache budget (tokens)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{model} -- {task}")
    ax.legend()
    _save(fig, os.path.join(out_dir, f"dose_response_{model}_{task}.png"))


def plot_needle_position(rows: List[dict], model: str, budget: int, out_dir: str):
    """Accuracy vs needle position per policy at a fixed budget."""
    by_pol = defaultdict(lambda: ([], []))
    for r in rows:
        if r.get("model") == model and r.get("budget") == budget and r.get("task", "").startswith("needle"):
            xs, ys = by_pol[r["policy"]]
            xs.append(r["task"])
            ys.append(r["mean"])
    if not by_pol:
        return
    order = ["needle_early", "needle_mid", "needle_late"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for pol in POLICY_ORDER:
        if pol in by_pol:
            xs, ys = by_pol[pol]
            xi = [order.index(x) if x in order else -1 for x in xs]
            pts = sorted(zip(xi, ys))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s", label=pol,
                    color=COLORS.get(pol))
    ax.set_xticks(range(3), ["early", "mid", "late"])
    ax.set_xlabel("needle position")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{model} -- budget {budget}")
    ax.legend()
    _save(fig, os.path.join(out_dir, f"needle_position_{model}_b{budget}.png"))


def plot_boundary(boundary_rows: List[dict], out_dir: str):
    """Phase 5: PES advantage (pes score - best baseline score) vs difficulty /
    tightness -- locating where the margin rule stops working."""
    if not boundary_rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, pts in boundary_rows.items():
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("difficulty knob")
    ax.set_ylabel("PES minus best baseline (accuracy)")
    ax.set_title("Boundary mapping: where PES stops winning")
    ax.legend()
    _save(fig, os.path.join(out_dir, "boundary_map.png"))
