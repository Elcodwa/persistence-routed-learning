"""R7/R8: the two decisive experiments.
R7  - LOO-priced PES vs EMA-graded PES vs clock (does pricing quality close the gap?)
      Pre-registration: LOO must (a) not lose to clock in stationary, (b) keep drift parity,
      (c) keep random-routing strictly worst. If it fails all three, R7 is negative.
R8  - dense SGD + weight decay sweep: does single-store regularization already
      achieve what routing achieves? (the 'you do not need two stores' kill test)
Writes ../results/r7r8_results.json
"""
import json, time
import numpy as np
from math import comb
from pes_core import DriftingSparseEnv, make_learner, run_stream

T = 3000; SEEDS = range(10)
REGS = {"stationary": dict(p_change=0.0), "drifting": dict(p_change=0.002)}
METHODS_R7 = ["pes_loo", "pes", "clock", "random_routing"]
LAMBDAS = [1e-4, 5e-4, 2e-3]

def signtest(a, b):
    d = [x - y for x, y in zip(a, b) if x != y]; n = len(d)
    if n == 0: return 1.0, 0.0
    k = sum(1 for x in d if x > 0)
    p = sum(comb(n, i) for i in range(min(k, n-k)+1)) / (2**n) * (2 if 2*k != n else 1)
    return min(p, 1.0), float(np.mean([x-y for x, y in zip(a, b)]))

t0 = time.time(); out = {"r7": [], "r8": [], "notes": "preregistered before run"}

# ---------------- R7: LOO pricing -------------------------------------------
for reg, kw in REGS.items():
    tails = {}
    for m in METHODS_R7:
        vals = []
        for seed in SEEDS:
            env = DriftingSparseEnv(D=200, s0=25, noise=0.05, seed=1000*seed+91,
                                    frac_common=0.8, rare_factor=30, **kw)
            L = make_learner(m, D=200, eta=0.15, seed=seed+400)
            o = run_stream(L, env, T=T)
            vals.append(float(o["losses"][2400:].mean()))
        tails[m] = vals
        print(f"[{time.time()-t0:6.0f}s] R7 {reg}/{m}: {np.mean(vals):.4f}±{np.std(vals):.4f}", flush=True)
    rec = dict(regime=reg,
               means={m: float(np.mean(v)) for m, v in tails.items()},
               sds={m: float(np.std(v)) for m, v in tails.items()},
               runs={m: [round(x, 5) for x in v] for m, v in tails.items()})
    p, d = signtest(tails["pes_loo"], tails["clock"])
    rec["loo_vs_clock"] = dict(p=p, diff=d); print(f"   loo vs clock: {d:+.4f} p={p:.3f}", flush=True)
    p, d = signtest(tails["pes_loo"], tails["pes"])
    rec["loo_vs_ema"] = dict(p=p, diff=d); print(f"   loo vs ema:   {d:+.4f} p={p:.3f}", flush=True)
    p, d = signtest(tails["random_routing"], tails["pes_loo"])
    rec["random_vs_loo"] = dict(p=p, diff=d); print(f"   random vs loo:{d:+.4f} p={p:.3f}", flush=True)
    out["r7"].append(rec)

# ---------------- R8: weight-decay sweep -------------------------------------
for lam in LAMBDAS:
    for reg, kw in REGS.items():
        vals = []
        for seed in SEEDS:
            env = DriftingSparseEnv(D=200, s0=25, noise=0.05, seed=1000*seed+91,
                                    frac_common=0.8, rare_factor=30, **kw)
            L = make_learner("sgd_wd", D=200, eta=0.15, seed=seed+400)
            L.l2 = lam
            o = run_stream(L, env, T=T)
            vals.append(float(o["losses"][2400:].mean()))
        out["r8"].append(dict(lambda_=lam, regime=reg,
                              mean=float(np.mean(vals)), sd=float(np.std(vals)),
                              runs=[round(x, 5) for x in vals]))
        print(f"[{time.time()-t0:6.0f}s] R8 wd={lam} {reg}: {np.mean(vals):.4f}±{np.std(vals):.4f}", flush=True)

with open("../results/r7r8_results.json", "w") as f:
    json.dump(out, f, indent=1)
print("WROTE r7r8_results.json")
