#!/usr/bin/env python
"""Main sweep entry point (Phases 3-5). Runs on Colab T4 (or anywhere).

Examples
--------
# Phase 3 grid for GPT-2 (full policies x budgets x tasks x seeds):
python scripts/run_sweep.py --model gpt2 --seeds 0 1 2 \
    --budgets 64 128 256 512 --results kv_pes/results/results.jsonl

# Phase 5 boundary mapping (tight budget + distractor difficulty + PES knobs):
python scripts/run_sweep.py --model gpt2 --policies pes h2o streaming window \
    --tasks delayed_needle --budgets 128 --difficulty 8 \
    --pes-mu 0.0 0.001 0.01 0.05 --pes-alpha 0.05 0.5 \
    --results kv_pes/results/boundary.jsonl
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kv_pes.policies import POLICIES
from kv_pes.runner import MODELS, ResultStore, sweep


def load_model(key: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODELS[key]
    dtype = torch.float16 if cfg["dtype"] == "float16" and device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], torch_dtype=dtype, attn_implementation="eager")
    model.to(device).eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=list(MODELS))
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--budgets", nargs="+", type=int, default=[64, 128, 256, 512])
    ap.add_argument("--tasks", nargs="+",
                    default=["needle_early", "needle_mid", "needle_late", "multifact"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-cases", type=int, default=12)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--pes-mu", nargs="+", type=float, default=None,
                    help="hold_cost_mu values for PES boundary sweep")
    ap.add_argument("--pes-alpha", nargs="+", type=float, default=None,
                    help="ema_alpha values for PES boundary sweep")
    ap.add_argument("--results", default="kv_pes/results/results.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prefill-chunk", type=int, default=64)
    args = ap.parse_args()

    device = args.device
    model, tok = load_model(args.model, device)
    store = ResultStore(args.results)

    pes_grid = [{}]
    if args.pes_mu or args.pes_alpha:
        mus = args.pes_mu or [0.001]
        alphas = args.pes_alpha or [0.1]
        pes_grid = [{"hold_cost_mu": mu, "ema_alpha": a} for mu in mus for a in alphas]

    sweep(args.model, model, tok, args.policies, args.budgets, args.tasks,
          args.seeds, n_cases_per_cell=args.n_cases, difficulty=args.difficulty,
          store=store, device=device, prefill_chunk=args.prefill_chunk,
          pes_grid=pes_grid)
    store.close()
    print(f"done -> {args.results} ({len(store.done)} records total)")


if __name__ == "__main__":
    main()
