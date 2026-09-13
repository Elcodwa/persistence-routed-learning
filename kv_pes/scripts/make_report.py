"""Generate results/results.md from the raw JSONL records (Phase 6).

Fills a reporting template with per-cell summaries and ALL paired PES-vs-
baseline comparisons (wins and losses alike), plus explicit pointers to where
the Phase 5 boundary evidence lives. Nothing is filtered or cherry-picked.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "kv_pes"))

import analysis as A
from analysis import load_records, paired_vs_baselines, dose_response
from stats import summarize

TEMPLATE = """# Results: PES vs published KV-eviction baselines

*Auto-generated from raw records; every comparison is reported, favorable or not.*

## Headline paired comparisons (PES minus baseline)

{paired_table}

| model | task | budget | vs | n | PES mean±std | baseline mean±std | diff | p(sign) | PES wins/losses/ties |
|---|---|---|---|---|---|---|---|---|---|
{paired_rows}

## Dose-response (accuracy vs cache budget)

{dose_notes}

## Boundary mapping (Phase 5)

Where PES stops winning, per the paper's Sec. 7.4 prediction of degradation
under nonlinear credit assignment:

{boundary_notes}

## Honest-summary checklist

- [ ] PES beats H2O/StreamingLLM/window/random somewhere: (fill from tables)
- [ ] PES loses somewhere: (fill from tables)
- [ ] Failure mode matches Sec. 7.4 prediction (EMA value decay under delayed
      relevance / distractors): (fill from boundary.jsonl)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="kv_pes/results/results.jsonl")
    ap.add_argument("--boundary", default="kv_pes/results/boundary.jsonl")
    ap.add_argument("--out", default="kv_pes/results/results.md")
    args = ap.parse_args()

    recs = load_records(args.results)
    rows, table = [], []
    models = sorted({r["model"] for r in recs})
    for m in models:
        budgets = sorted({r["budget"] for r in recs if r["model"] == m and r["budget"] >= 0})
        for b in budgets:
            for t in sorted({r["task"] for r in recs if r["model"] == m and r["budget"] == b}):
                for c in paired_vs_baselines(recs, model=m, task=t, budget=b):
                    c["budget"] = b
                    rows.append(c)
    for c in rows:
        table.append(
            f"| {c['model']} | {c['task']} | {c['budget']} | {c['policy_b']} | {c['n']} "
            f"| {c['mean_a']:.3f}±{c['diff_std']:.3f} | {c['mean_b']:.3f}±{c['std_b']:.3f} "
            f"| {c['mean_diff']:+.3f} | {c['sign_test_p']:.3f} "
            f"| {c['a_wins']}/{c['b_wins']}/{c['ties']} |")
    if not table:
        table.append("| (no records yet — run the Colab notebook) | | | | | | | | | |")

    dose_lines = []
    for m in models:
        for t in sorted({r["task"] for r in recs if r["model"] == m}):
            dr = dose_response(recs, m, t)
            if dr:
                pts = ", ".join(f"{r['budget']}:{r['mean']:.2f}({r['policy']})" for r in dr)
                dose_lines.append(f"- `{m}` `{t}`: {pts}")

    bnd = []
    if args.boundary:
        try:
            brecs = load_records(args.boundary)
        except FileNotFoundError:
            brecs = []
        by_d = defaultdict(dict)
        for r in brecs:
            by_d[r["difficulty"]].setdefault(r["policy"], []).append(r["score"])
        for d in sorted(by_d):
            parts = []
            for pol, scores in sorted(by_d[d].items()):
                parts.append(f"{pol}={summarize(scores)['mean']:.3f}")
            bnd.append(f"- difficulty={d}: " + ", ".join(parts))

    text = TEMPLATE.format(paired_table=f"{len(rows)} comparisons",
                           paired_rows="\n".join(table) or "_(none)_",
                           dose_notes="\n".join(dose_lines) or "_none_",
                           boundary_notes="\n".join(bnd) or "_run boundary.jsonl sweeps_")
    with open(args.out, "w") as f:
        f.write(text)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
