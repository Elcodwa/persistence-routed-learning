"""R9: LOO-surrogate pricing on the MLP substrate at the R6 inversion point.
Pre-registration (commit BEFORE running):
At caps=(20,10), regime=stationary, T=8000, n=5 paired seeds:
  (a) PES-LOO must not significantly lose to the fixed clock;
  (b) PES-LOO must beat tag-priced PES numerically on mean tail;
  (c) sanity: both valued variants remain far below noise-routing's typical tail.
If (a) fails, R9 is NEGATIVE and the paper keeps its current honest boundary.
Writes ../results/r9_mlp_loo.json
"""
import json, time
import numpy as np
from math import comb
import pes_torch
from pes_torch import TorchStoreLearner
from pes_core import DriftingSparseEnv

def signtest(a, b):
    d = [x - y for x, y in zip(a, b) if x != y]; n = len(d)
    if n == 0: return 1.0
    k = sum(1 for x in d if x > 0)
    return min(1.0, sum(comb(n, i) for i in range(min(k, n-k)+1)) / (2**n) * (2 if 2*k != n else 1))

def run(policy, price_mode, caps, seed, kw, T=8000):
    pes_torch.PRICE_MODE = price_mode
    env = DriftingSparseEnv(D=120, s0=10, noise=0.05, seed=1000*seed+29,
                            frac_common=0.8, rare_factor=30, **kw)
    env.wstar[env.support] *= 0.6
    L = TorchStoreLearner(policy, 120, seed, caps=caps)
    losses = []
    for t in range(1, T+1):
        x, y = env.step()
        losses.append(L.step(x, y, t))
    return float(np.mean(losses[int(0.8*T):]))

if __name__ == "__main__":
    t0 = time.time(); CAPS = (20, 10); KW = dict(p_change=0.0); N = 5
    tails = {"pes_loo": [], "pes_tag": [], "clock": []}
    for seed in range(N):
        for key, (pol, pm) in {"pes_loo": ("pes", "loo_surrogate"),
                               "pes_tag": ("pes", "tag"),
                               "clock": ("clock", "tag")}.items():
            v = run(pol, pm, CAPS, seed, KW)
            tails[key].append(v)
            print(f"[{time.time()-t0:7.1f}s] seed{seed} {key}: {v:.4f}", flush=True)
    out = dict(caps=list(CAPS), regime="stationary", n=N,
               means={k: float(np.mean(v)) for k, v in tails.items()},
               sds={k: float(np.std(v)) for k, v in tails.items()},
               runs=tails,
               p_loo_vs_clock=signtest(tails["pes_loo"], tails["clock"]),
               p_loo_vs_tag=signtest(tails["pes_loo"], tails["pes_tag"]))
    with open("../results/r9_mlp_loo.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
    print("WROTE r9_mlp_loo.json")
