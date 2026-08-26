"""R6: capacity-pressure sweep on the MLP substrate.
Registered prediction (RESULTS_NARRATIVE.md Phase-6 section): PES-vs-clock advantage re-opens as
capacity shrinks toward support size.
Paired design: each policy gets its OWN env instance seeded identically => identical streams.
Writes ../results/r6_capacity.json"""
import json, time
import numpy as np
from pes_torch import TorchStoreLearner, DriftingSparseEnv

t0 = time.time()
out = []
CONFIGS = [(14, 6), (20, 10), (30, 15)]   # totals 20/30/45 vs support 10
REGS = {"stationary": dict(p_change=0.0), "drifting": dict(p_change=0.002)}
T = 8000

def run_stream(policy, caps, seed, kw):
    env = DriftingSparseEnv(D=120, s0=10, noise=0.05, seed=1000*seed+29,
                            frac_common=0.8, rare_factor=30, **kw)
    env.wstar[env.support] *= 0.6
    L = TorchStoreLearner(policy, 120, seed, caps=caps)
    losses = []
    for t in range(1, T + 1):
        x, y = env.step()
        losses.append(L.step(x, y, t))
    return float(np.mean(losses[int(0.8*T):]))

for cf, cs in CONFIGS:
    for reg, kw in REGS.items():
        tails = {"pes": [], "clock": []}
        for seed in range(3):
            for pol in ["pes", "clock"]:
                tails[pol].append(run_stream(pol, (cf, cs), seed, kw))
            print(f"[{time.time()-t0:7.1f}s] {cf}/{cs} {reg} seed{seed} "
                  f"pes={tails['pes'][-1]:.4f} clock={tails['clock'][-1]:.4f}", flush=True)
        out.append(dict(caps=f"{cf}+{cs}", regime=reg,
                        pes=float(np.mean(tails["pes"])), pes_sd=float(np.std(tails["pes"])),
                        clock=float(np.mean(tails["clock"])), clock_sd=float(np.std(tails["clock"])),
                        pes_runs=tails["pes"], clock_runs=tails["clock"]))

with open("../results/r6_capacity.json", "w") as f:
    json.dump(out, f, indent=1)
print("WROTE r6_capacity.json", f"{time.time()-t0:.0f}s")
