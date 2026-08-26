"""Main experimental suite for PES (Persistence-Economics Selection).

Phases:
  1  Toy sanity (handled by unit tests)
  2-3 Regime comparison: stationary / drifting / volatile / alternating(A-B-A)
     methods: sgd_dense, pes, random_routing, clock(value), clock_mag, single_decay
  4  Ablations: no switching cost, no hysteresis, no fast decay
  5  Capacity & tariff sweeps

Usage: python run_experiments.py [--quick]
Outputs: ../results/main_results.json (+ prints summary tables)
"""
import json, sys, time, argparse, zlib
import numpy as np
from pes_core import (DriftingSparseEnv, make_learner, run_stream,
                      rare_recall_probe, exact_sign_test)

METHODS = ["sgd_dense", "sgd_l2", "pes", "random_routing", "clock", "clock_mag", "single_decay"]
REGIMES = {
    "stationary": dict(p_change=0.0),
    "drifting":   dict(p_change=0.002),
    "volatile":   dict(p_change=0.01),
    "alternating":dict(p_change=0.0, aba=True, aba_period=600),
}
ETA = 0.15
D, S0 = 200, 25
CAPS = (45, 15)


def run_one(method, regime, seed, T=3000, caps=CAPS, **overrides):
    env_kw = dict(D=D, s0=S0, noise=0.05,
                  seed=1000 * seed + zlib.crc32(regime.encode()) % 997,
                  frac_common=0.8, rare_factor=30)
    env_kw.update(REGIMES[regime])
    if regime == "alternating":
        env_kw["aba_period"] = max(600, T // 4)   # ~4 blocks per run regardless of T
    if overrides.pop("env", None):
        env_kw.update(overrides["env"])
    env = DriftingSparseEnv(**env_kw)
    L = make_learner(method, D=D, eta=ETA, seed=seed, caps=caps)
    for k, v in overrides.items():
        setattr(L, k, v)
    out = run_stream(L, env, T=T)
    tail = slice(int(0.8 * T), T)
    res = dict(
        method=method, regime=regime, seed=seed,
        loss_tail=float(out["losses"][tail].mean()),
        loss_cum=float(out["losses"].mean()),
        n_promo=len(getattr(L, "promo_events", [])),
        n_evict=len(getattr(L, "evict_events", [])),
    )
    # rare-coordinate recall probe (isolated recall of low-frequency support coords)
    pa = env.p_appear[env.support]
    rare_coords = list(env.support[pa <= np.median(pa)])
    res["recall_rare"] = rare_recall_probe(L, env, rare_coords, n_probe=30,
                                           rng=np.random.default_rng(seed))
    # P2 data: residence durations per trace
    if hasattr(L, "store") and out["residence"]:
        from collections import defaultdict
        seen = {}
        durs = defaultdict(list)
        for (t, i, s) in out["residence"]:
            if i in seen and seen[i][1] != s:
                durs[seen[i][1]].append(t - seen[i][0])
            seen[i] = (t, s)
        res["residence_durs_fast"] = durs.get(0, [])
        res["residence_durs_slow"] = durs.get(1, [])
    # P3 data: promo events with preceding error magnitudes
    if hasattr(L, "promo_events"):
        res["promo_ts"] = list(L.promo_ts) if hasattr(L, "promo_ts") else [e[0] for e in L.promo_events]
    res["err"] = [float(x) for x in out["err"][:: max(1, T // 1500)]]  # decimated series
    # P4: alternating regime - time-to-reacquire per block
    if regime == "alternating":
        errs = out["err"]
        blocks = []
        period = REGIMES["alternating"]["aba_period"]
        nb = T // period
        thr = np.quantile(errs, 0.3)
        for b in range(nb):
            seg = errs[b * period:(b + 1) * period]
            below = np.nonzero(seg < thr)[0]
            blocks.append(int(below[0]) if len(below) else period)
        res["reacquire_steps_per_block"] = blocks
    return res


def paired_stats(pes_vals, base_vals):
    diffs = [a - b for a, b in zip(pes_vals, base_vals)]
    p = exact_sign_test(diffs)
    md = float(np.mean(diffs)); sd = float(np.std(diffs)) or 1e-12
    return dict(mean_diff=md, sd_diff=sd, d=md / sd, sign_p=p, n=len(diffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    T = 800 if args.quick else 3000
    SEEDS = range(3) if args.quick else range(10)
    results = []

    t0 = time.time()
    for regime in REGIMES:
        for method in METHODS:
            for seed in SEEDS:
                r = run_one(method, regime, seed, T=T)
                results.append(r)
        print(f"[{time.time()-t0:7.1f}s] regime done: {regime}", flush=True)

    # ---------------- Phase 4: ablations (volatile regime, hardest) -------------
    abl_configs = {
        "pes_no_switch_cost": dict(kappa_move=0.0),
        "pes_no_hysteresis":  dict(hysteresis_w=1),
        "pes_no_fast_decay":  dict(mu_fast=0.0),
        "pes_no_slow_tariff": dict(lam_slow=0.0),
    }
    for name, kw in abl_configs.items():
        for seed in SEEDS:
            r = run_one("pes", "volatile", seed, T=T, **kw)
            r["method"] = name
            results.append(r)
    print(f"[{time.time()-t0:7.1f}s] ablations done", flush=True)

    # ---------------- Phase 5: capacity sweep -----------------------------------
    for cf, cs in [(25, 15), (45, 15), (70, 15), (45, 0), (45, 30)]:
        for method in ["pes", "single_decay", "clock"]:
            for seed in SEEDS:
                r = run_one(method, "drifting", seed, T=T, caps=(cf, cs))
                r["caps"] = f"{cf}/{cs}"
                results.append(r)
    print(f"[{time.time()-t0:7.1f}s] capacity sweep done", flush=True)

    # ---------------- Phase 5b: tariff law (P5) ---------------------------------
    tariff_rows = []
    for lam in [1e-5, 1e-4, 1e-3, 1e-2]:
        for mu in [0.005, 0.02, 0.08]:
            promos = []
            for seed in range(5):
                r = run_one("pes", "drifting", seed, T=T,
                            lam_slow=lam, mu_fast=mu)
                promos.append(r["n_promo"])
            tariff_rows.append(dict(lam_slow=lam, mu_fast=mu,
                                    promo_rate=float(np.mean(promos)) / T))
    print(f"[{time.time()-t0:7.1f}s] tariff law done", flush=True)

    import os
    os.makedirs("../results", exist_ok=True)
    with open("../results/main_results.json", "w") as f:
        json.dump(dict(results=results, tariff_rows=tariff_rows,
                       meta=dict(T=T, D=D, S0=S0, ETA=ETA, CAPS=list(CAPS),
                                 seeds=len(list(SEEDS)))), f, indent=1)

    # ---------------- summary ----------------------------------------------------
    def agg(meth, reg):
        vals = [r["loss_tail"] for r in results
                if r["method"] == meth and r["regime"] == reg and "caps" not in r]
        rec = [r.get("recall_rare", np.nan) for r in results
               if r["method"] == meth and r["regime"] == reg and "caps" not in r]
        return (float(np.mean(vals)), float(np.std(vals)),
                float(np.nanmean(rec)) if rec else float("nan"))

    print("\n=== Loss tail (mean±sd) by regime ===")
    hdr = "method".ljust(16) + "".join(reg.ljust(18) for reg in REGIMES)
    print(hdr)
    for m in METHODS:
        row = m.ljust(16)
        for reg in REGIMES:
            mv, sv, _ = agg(m, reg)
            row += f"{mv:.4f}±{sv:.4f}    "
        print(row)
    print("\n=== Rare recall MSE (lower better) ===")
    print(hdr)
    for m in METHODS:
        row = m.ljust(16)
        for reg in REGIMES:
            _, _, rc = agg(m, reg)
            row += f"{rc:>14.4f}    "
        print(row)
    print("\n=== Paired stats: PES vs baseline (loss_tail, per regime) ===")
    for reg in REGIMES:
        pv = [r["loss_tail"] for r in results
              if r["method"] == "pes" and r["regime"] == reg and "caps" not in r]
        for m in METHODS:
            if m == "pes": continue
            bv = [r["loss_tail"] for r in results
                  if r["method"] == m and r["regime"] == reg and "caps" not in r]
            st = paired_stats(pv, bv)
            print(f"{reg:11s} pes vs {m:13s} d={st['d']:+.2f} p(sign)={st['sign_p']:.3f}"
                  f" diff={st['mean_diff']:+.5f}")
    print(f"\ntotal wall: {time.time()-t0:.1f}s | rows: {len(results)}")


if __name__ == "__main__":
    main()
