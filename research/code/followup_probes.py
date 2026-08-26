"""Registered follow-up probes R2 + R4 (see results/RESULTS_NARRATIVE.md).
R2: P3 spike-locking re-emergence prediction at small hysteresis windows.
R4: LR sweep best-vs-best to close competing hypothesis C-c.
Writes ../results/followup_probes.json"""
import json
import numpy as np
from pes_core import DriftingSparseEnv, make_learner, run_stream

out = {"R2": [], "R4": []}

# ---------- R2 ----------
for hw in [1, 5, 10]:
    rows_ = []
    for seed in range(3):
        env = DriftingSparseEnv(D=200, s0=25, noise=0.05,
                                seed=1000*seed+31, frac_common=0.8, rare_factor=30,
                                p_change=0.01)
        L = make_learner("pes", D=200, eta=0.15, seed=seed)
        L.hysteresis_w = hw
        o = run_stream(L, env, T=2000)
        ts = [e[0] for e in L.promo_events]; err = o["err"]
        if len(ts) < 30 or len(err) < 100:
            continue
        T_err = 2000 * max(1, 2000 // 1500)
        idxs = [int(t / T_err * len(err)) for t in ts]
        idxs = [i for i in idxs if 31 < i < len(err)]
        base = float(np.mean(err))
        r5 = float(np.mean([err[i-5] for i in idxs]) / base)
        r30 = float(np.mean([err[i-30] for i in idxs]) / base)
        rows_.append(dict(seed=seed, n_promo=len(ts), lag5=r5, lag30=r30))
    if rows_:
        out["R2"].append(dict(hysteresis=hw,
                              mean_lag5=float(np.mean([r['lag5'] for r in rows_])),
                              mean_lag30=float(np.mean([r['lag30'] for r in rows_])),
                              contrast=float(np.mean([r['lag5']/r['lag30'] for r in rows_])),
                              runs=rows_))
    print("R2 hw=", hw, "done", flush=True)

# ---------- R4 ----------
REGS = {"drifting": dict(p_change=0.002), "volatile": dict(p_change=0.01)}
for reg, kw in REGS.items():
    for lr in [0.05, 0.15, 0.45]:
        res = {}
        for m in ["pes", "clock", "single_decay"]:
            vals = []
            for seed in range(3):
                env = DriftingSparseEnv(D=200, s0=25, noise=0.05, seed=1000*seed+77,
                                        frac_common=0.8, rare_factor=30, **kw)
                L = make_learner(m, D=200, eta=lr, seed=seed)
                o = run_stream(L, env, T=2000)
                vals.append(float(o["losses"][1600:].mean()))
            res[m] = float(np.mean(vals))
        best = min(res, key=res.get)
        out["R4"].append(dict(regime=reg, lr=lr, means=res, best=best))
        print("R4", reg, lr, "done ->", best, flush=True)

with open("../results/followup_probes.json", "w") as f:
    json.dump(out, f, indent=1)
print("WROTE followup_probes.json")
