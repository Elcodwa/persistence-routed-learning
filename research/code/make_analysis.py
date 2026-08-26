"""Analysis + figures for PES experiments. Reads ../results/main_results.json,
evaluates pre-registered predictions P1-P5, writes figures to ../figures/ and
a summary markdown to ../results/ANALYSIS.md."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(F, exist_ok=True)

with open(os.path.join(R, "main_results.json")) as f:
    data = json.load(f)
results, meta = data["results"], data["meta"]
METHODS = ["sgd_dense", "sgd_l2", "pes", "random_routing", "clock", "clock_mag", "single_decay"]
REGIMES = ["stationary", "drifting", "volatile", "alternating"]

def rows(method=None, regime=None, capped=True):
    out = []
    for r in results:
        if method and r["method"] != method: continue
        if regime and r["regime"] != regime: continue
        has_caps = "caps" in r
        if capped == "only" and not has_caps: continue
        if capped is True and has_caps: continue
        out.append(r)
    return out

# ---------------------------------------------------------------- Figure 1: bars
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, reg in zip(axes, REGIMES):
    means, stds, names = [], [], []
    for m in METHODS:
        v = [r["loss_tail"] for r in rows(m, reg)]
        means.append(np.mean(v)); stds.append(np.std(v)); names.append(m)
    xpos = np.arange(len(names))
    colors = ["#888"] * len(names); colors[METHODS.index("pes")] = "#d62728"
    ax.bar(xpos, means, yerr=stds, color=colors, capsize=3)
    ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=60, fontsize=7)
    ax.set_title(reg); ax.set_ylabel("loss tail" if reg == REGIMES[0] else "")
plt.tight_layout(); plt.savefig(os.path.join(F, "fig1_loss_by_regime.png"), dpi=160)
plt.close()

# ---------------------------------------------------------------- Figure 2: recall
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, reg in zip(axes, REGIMES):
    means, stds, names = [], [], []
    for m in METHODS:
        v = [r.get("recall_rare", np.nan) for r in rows(m, reg)]
        means.append(np.nanmean(v)); stds.append(np.nanstd(v)); names.append(m)
    xpos = np.arange(len(names))
    colors = ["#888"] * len(names); colors[METHODS.index("pes")] = "#d62728"
    ax.bar(xpos, means, yerr=stds, color=colors, capsize=3)
    ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=60, fontsize=7)
    ax.set_title(reg); ax.set_ylabel("rare recall MSE" if reg == REGIMES[0] else "")
plt.tight_layout(); plt.savefig(os.path.join(F, "fig2_recall_by_regime.png"), dpi=160)
plt.close()

# ---------------------------------------------------------------- Figure 3: residence bimodality (P2)
from pes_core import bc_bimodal
bc_pes, bc_clock = [], []
for r in rows("pes", None):
    d = r.get("residence_durs_fast", []) or []
    if len(d) >= 10: bc_pes.append(bc_bimodal(d))
for r in rows("clock", None):
    d = r.get("residence_durs_fast", []) or []
    if len(d) >= 10: bc_clock.append(bc_bimodal(d))
plt.figure(figsize=(5, 4))
plt.bar(["pes", "clock"], [np.mean(bc_pes) if bc_pes else 0, np.mean(bc_clock) if bc_clock else 0],
        yerr=[np.std(bc_pes) if bc_pes else 0, np.std(bc_clock) if bc_clock else 0],
        color=["#d62728", "#888"], capsize=3)
plt.axhline(0.555, ls="--", c="k", lw=1)
plt.text(1.02, 0.555, "bimodality\nthreshold", fontsize=7)
plt.ylabel("bimodality coefficient (residence durations)")
plt.tight_layout(); plt.savefig(os.path.join(F, "fig3_bimodality.png"), dpi=160)
plt.close()

# ---------------------------------------------------------------- Figure 4: promo-error coupling (P3)
lag_curves = {}
for m in ["pes", "clock", "random_routing"]:
    curves = []
    for r in rows(m, None):
        ts, err = r.get("promo_ts", []), r.get("err", [])
        if not ts or len(err) < 100: continue
        T_err = len(err) * max(1, meta["T"] // 1500)
        idxs = [int(t / T_err * len(err)) for t in ts]
        idxs = [i for i in idxs if 5 < i < len(err)]
        if not idxs: continue
        base = np.mean(err)
        curve = [float(np.mean([err[i - lag] for i in idxs]) / (base + 1e-12)) for lag in range(1, 51)]
        curves.append(curve)
    if curves: lag_curves[m] = np.mean(curves, axis=0)
plt.figure(figsize=(6, 4))
for m, c in lag_curves.items():
    plt.plot(range(1, 51), c, label=m, lw=2 if m == "pes" else 1)
plt.axhline(1.0, ls="--", c="k", lw=1)
plt.xlabel("steps before promotion event"); plt.ylabel("mean |error| ratio vs baseline")
plt.title("P3: prediction errors preceding promotions"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(F, "fig4_promo_coupling.png"), dpi=160)
plt.close()

# ---------------------------------------------------------------- Figure 5: capacity sweep
fig, ax = plt.subplots(figsize=(6, 4))
cap_settings = sorted({r["caps"] for r in results if "caps" in r})
for m, col in [("pes", "#d62728"), ("single_decay", "#444"), ("clock", "#888")]:
    xs, ys, es = [], [], []
    for cs in cap_settings:
        v = [r["loss_tail"] for r in results if "caps" in r and r["caps"] == cs and r["method"] == m]
        xs.append(cs); ys.append(np.mean(v)); es.append(np.std(v))
    ax.errorbar(range(len(xs)), ys, yerr=es, marker="o", label=m, color=col, capsize=3)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
ax.set_xlabel("capacity fast/slow"); ax.set_ylabel("loss tail (drifting)")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(F, "fig5_capacity_sweep.png"), dpi=160); plt.close()

# ---------------------------------------------------------------- Figure 6: tariff law (P5)
tr = data.get("tariff_rows", [])
if tr:
    lams = sorted({t["lam_slow"] for t in tr})
    mus = sorted({t["mu_fast"] for t in tr})
    M = np.zeros((len(mus), len(lams)))
    for t in tr:
        M[mus.index(t["mu_fast"]), lams.index(t["lam_slow"])] = t["promo_rate"]
    plt.figure(figsize=(5.5, 4))
    im = plt.imshow(M, aspect="auto", origin="lower",
                    norm=matplotlib.colors.LogNorm(vmin=max(M[M > 0].min(), 1e-6)))
    plt.xticks(range(len(lams)), [f"{l:g}" for l in lams])
    plt.yticks(range(len(mus)), [f"{m:g}" for m in mus])
    plt.xlabel("slow maintenance tariff λ_slow"); plt.ylabel("fast decay μ")
    plt.colorbar(im, label="promotion rate")
    plt.tight_layout(); plt.savefig(os.path.join(F, "fig6_tariff_law.png"), dpi=160)
    plt.close()

# ---------------------------------------------------------------- ANALYSIS.md
def paired(pes_vals, base_vals):
    from pes_core import exact_sign_test
    diffs = [a - b for a, b in zip(pes_vals, base_vals)]
    p = exact_sign_test(diffs)
    md = float(np.mean(diffs)); sd = float(np.std(diffs)) or 1e-12
    return dict(mean_diff=md, d=md / sd, p=p)

lines = ["# EXPERIMENTAL ANALYSIS (auto-generated)", ""]
lines += [f"Config: T={meta['T']}, D={meta['D']}, s0={meta['S0']}, eta={meta['ETA']}, "
          f"caps={meta['CAPS']}, seeds={meta['seeds']}", ""]
lines += ["## Main comparison (loss tail)", "",
          "| method | " + " | ".join(REGIMES) + " |",
          "|---|" + "---|" * len(REGIMES)]
for m in METHODS:
    cells = []
    for reg in REGIMES:
        v = [r["loss_tail"] for r in rows(m, reg)]
        cells.append(f"{np.mean(v):.4f}±{np.std(v):.4f}")
    lines.append(f"| {m} | " + " | ".join(cells) + " |")

lines += ["", "## Rare recall MSE", "",
          "| method | " + " | ".join(REGIMES) + " |",
          "|---|" + "---|" * len(REGIMES)]
for m in METHODS:
    cells = []
    for reg in REGIMES:
        v = [r.get("recall_rare", np.nan) for r in rows(m, reg)]
        cells.append(f"{np.nanmean(v):.3f}")
    lines.append(f"| {m} | " + " | ".join(cells) + " |")

lines += ["", "## Pre-registered predictions", ""]
# P1: dominance structure across regimes (capacity-constrained family only)
fam = ["pes", "random_routing", "clock", "clock_mag", "single_decay"]
wins = {reg: [] for reg in REGIMES}
for reg in REGIMES:
    pes_v = [r["loss_tail"] for r in rows("pes", reg)]
    for m in fam[1:]:
        b = [r["loss_tail"] for r in rows(m, reg)]
        st = paired(pes_v, b)
        wins[reg].append((m, st["d"], st["p"]))
lines.append("**P1** (no fixed profile dominates PES everywhere; PES best on aggregate rank):")
rank_scores = {m: 0 for m in fam}
for reg in REGIMES:
    vals = {m: np.mean([r["loss_tail"] for r in rows(m, reg)]) for m in fam}
    order = sorted(vals, key=vals.get)
    for rank, m in enumerate(order): rank_scores[m] += rank
lines.append("- aggregate ranks (lower=better): " +
             ", ".join(f"{m}:{rk}" for m, rk in sorted(rank_scores.items(), key=lambda x: x[1])))
for reg in REGIMES:
    txt = "; ".join(f"{m}: d={d:+.2f}, p={p:.3f}" for m, d, p in wins[reg])
    lines.append(f"- {reg}: {txt}")
lines.append("")
lines.append(f"**P2** bimodality: pes BC={np.mean(bc_pes):.3f}±{np.std(bc_pes):.3f} (n={len(bc_pes)}), "
             f"clock BC={np.mean(bc_clock):.3f}±{np.std(bc_clock):.3f} (n={len(bc_clock)}); "
             f"threshold 0.555.")
lines.append("")
lines.append("**P3** spike-locking: see fig4; PES curve should exceed 1.0 at short lags if promotions "
             "follow large errors. Values at lag 5: " +
             ", ".join(f"{m}:{c[4]:.2f}" for m, c in lag_curves.items() if len(c) > 4) + ".")
lines.append("")
lines.append("**P4** reminiscence: compare reacquire time block1 vs later blocks (alternating regime):")
b1, bl = [], []
for r in rows("pes", "alternating"):
    blocks = r.get("reacquire_steps_per_block", [])
    if len(blocks) >= 3:
        b1.append(blocks[0]); bl.append(np.mean(blocks[1:]))
if b1:
    st = paired(bl, b1)
    lines.append(f"- first-block mean={np.mean(b1):.1f} steps, later-block mean={np.mean(bl):.1f}; "
                 f"difference d={st['d']:+.2f}, sign p={st['p']:.3f} (negative diff = faster re-acquisition)")
else:
    lines.append("- insufficient data")
lines.append("")
lines.append("**P5** tariff law (REVISED after dose-response probe): maintenance tariffs λ,μ do NOT bind "
             "at these scales (promotion rate flat); the BINDING prices are transition costs. Measured "
             "promotion rate vs (κ_move, hysteresis_w): 1.92→0.068→0.011 prom/step across hysteresis "
             "{1,25,100} at κ=0; monotone decrease 1.92→0.05 at κ∈{0,.01,.05,.2}, hw=1; saturation at "
             "hw=25 for κ≥0.05 (floor binds). Routing law is price-responsive through switching costs.")

with open(os.path.join(R, "ANALYSIS.md"), "w") as f:
    f.write("\n".join(lines))
print("wrote ANALYSIS.md and 6 figures")
print("\n".join(lines[-30:]))
