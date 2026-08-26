"""R2': P3 spike-locking test IN THE SELECTIVE REGIME.
Registered prediction (RESULTS_NARRATIVE.md): coupling can only manifest when the switch budget is
NOT saturated. Raise kappa_move until promo rate << budget, then measure lag-coupling.
Outcome A: coupling appears -> P3 confirmed in correct regime.
Outcome B: stays flat -> migration timing is price-driven, NOT surprise-driven (stronger finding).
Writes ../results/r2prime.json"""
import json
import numpy as np
from pes_core import DriftingSparseEnv, make_learner, run_stream

HW = 5
BUDGET_PER_STEP = 1.0 / HW   # openings per step
out = []
for km in [0.5, 1.0, 2.0]:
    rows_ = []
    for seed in range(3):
        env = DriftingSparseEnv(D=200, s0=25, noise=0.05, seed=1000*seed+31,
                                frac_common=0.8, rare_factor=30, p_change=0.01)
        L = make_learner("pes", D=200, eta=0.15, seed=seed)
        L.kappa_move = km; L.hysteresis_w = HW
        o = run_stream(L, env, T=3000)
        ts = [e[0] for e in L.promo_events]
        err = o["err"]
        n = len(ts)
        rate = n / 3000
        res = dict(seed=seed, n_promo=n, promo_rate=rate,
                   saturated=bool(rate > 0.5 * BUDGET_PER_STEP))
        if n >= 15 and len(err) >= 100:
            T_err = 3000 * max(1, 3000 // 1500)
            idxs = [int(t / T_err * len(err)) for t in ts]
            idxs = [i for i in idxs if 31 < i < len(err)]
            base = float(np.mean(err))
            if idxs:
                r5 = float(np.mean([err[i-5] for i in idxs]) / base)
                r30 = float(np.mean([err[i-30] for i in idxs]) / base)
                res.update(lag5=r5, lag30=r30, contrast=r5/r30)
        rows_.append(res)
        print("km", km, "seed", seed, "->", res, flush=True)
    ok = [r for r in rows_ if "contrast" in r and not r["saturated"]]
    out.append(dict(kappa=km,
                    mean_rate=float(np.mean([r["promo_rate"] for r in rows_])),
                    any_unsaturated=bool(ok),
                    mean_contrast=float(np.mean([r["contrast"] for r in ok])) if ok else None,
                    runs=rows_))

with open("../results/r2prime.json", "w") as f:
    json.dump(out, f, indent=1)
print("WROTE r2prime.json")
