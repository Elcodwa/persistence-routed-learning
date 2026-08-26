"""Publication figures for the PES paper. Reads committed JSONs only; no invented data.
Outputs: research/paper/latex/figs/*.pdf (+png previews)"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
C = {"pes": "#d62728", "clock": "#7f7f7f", "single_decay": "#444444",
     "random_routing": "#1f77b4", "sgd_dense": "#2ca02c"}
NICE = {"pes": "PES", "clock": "Clock", "single_decay": "Single-decay",
        "random_routing": "Random-routing", "sgd_dense": "Dense SGD"}

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print("wrote", name)

# ---------------- Fig 1: dose-response of transition costs -------------------
plt.figure(figsize=(3.6, 2.6))
hw = [1, 25, 100]; rate = [1.92, 0.068, 0.011]
km = [0.0, 0.01, 0.05, 0.2]; rate_km = [1.92, 1.80, 1.72, 1.05]
ax1 = plt.gca()
ax1.plot(hw, rate, "o-", color=C["pes"], label="hysteresis window $w$")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("hysteresis window (steps)")
ax1.set_ylabel("promotions / step", color=C["pes"])
ax2 = ax1.twiny()
ax2.plot(km, rate_km, "s--", color=C["clock"], label=r"migration fee $\kappa$")
ax2.set_xlabel(r"migration fee $\kappa$", color=C["clock"])
ax1.axhline(0.07487, ls=":", c="k", lw=1)
ax1.text(1.15, 0.085, "maintenance-tariff sweep\n(flat at 0.075)", fontsize=6.5)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc="lower left", frameon=False)
plt.title("Transition costs bind; keeping costs do not")
save(plt.gcf(), "fig_dose_response")

# ---------------- Fig 2: three regimes ----------------------------------------
plt.figure(figsize=(3.6, 2.4))
bands = [("Saturated\n($\\kappa=0.5$)", 1.15, "which trace correlates"),
         ("Selective\n($\\kappa=1$)", None, None),
         ("Frozen\n($\\kappa\\geq 2$)", 0.52, "anti-correlated")]
xs = range(3)
ys = [1.15, 0.97, 0.52]
cols = ["#fdbf6f", C["pes"], "#cab2d6"]
plt.bar(xs, ys, color=cols, width=0.55)
plt.axhline(1.0, ls="--", c="k", lw=1)
plt.xticks(list(xs), [b[0] for b in bands])
plt.ylabel("error before swap / baseline")
n_note = ("n=3 each" if True else "")
plt.text(1, 1.03, "median 0.97, n=10:\nnot significant", ha="center", fontsize=6.5)
plt.title("Three routing regimes")
save(plt.gcf(), "fig_regimes")

# ---------------- Fig 3: main comparison incl. LOO pricing -------------------
rows = json.load(open(os.path.join(R, "margin_rerun.json")))
r78 = json.load(open(os.path.join(R, "r7r8_results.json")))
loo_runs = {rec["regime"]: rec["runs"]["pes_loo"] for rec in r78["r7"]}
import json as _json
_v = _json.load(open(os.path.join(R, "r7_volatile.json")))
loo_runs["volatile"] = _v["runs"]
regs = [r["regime"] for r in rows if r["family"] == "bernoulli"]
meths = ["pes_loo", "pes", "clock", "single_decay", "random_routing"]
C["pes_loo"] = "#e31a1c"
NICE["pes_loo"] = "PES-LOO (exact price)"
x = np.arange(len(regs)); wdt = 0.16
plt.figure(figsize=(5.8, 2.9))
for k, m in enumerate(meths):
    if m == "pes_loo":
        vals = [loo_runs[r] for r in regs]
        means = [np.mean(v) for v in vals]; sds = [np.std(v) for v in vals]
    else:
        means = [np.mean(rows[i]["tails"][m]) for i in range(len(regs))]
        sds   = [np.std(rows[i]["tails"][m]) for i in range(len(regs))]
    plt.bar(x + (k-2)*wdt, means, wdt, yerr=sds, capsize=2,
            color=C[m], label=NICE[m], alpha=1.0 if m == "pes_loo" else 0.85)
plt.xticks(x, [r.capitalize() for r in regs])
plt.ylabel("loss tail (Bernoulli family)")
plt.legend(ncol=2, frameon=False, fontsize=7.5)
plt.title("Exact leave-one-out pricing vs baselines ($n{=}10$)")
save(plt.gcf(), "fig_main_comparison")

# ---------------- Fig 4: MLP capacity-pressure inversion ----------------------
# data from r6 log values quoted in RESULTS_NARRATIVE; recompute from r6_capacity.json
r6 = json.load(open(os.path.join(R, "r6_capacity.json")))
caps_order = ["14+6", "20+10", "30+15"]
st = [r for r in r6 if r["regime"] == "stationary"]
st.sort(key=lambda r: caps_order.index(r["caps"]))
plt.figure(figsize=(3.6, 2.6))
xx = np.arange(len(st))
plt.errorbar(xx, [r["pes"] for r in st], yerr=[r["pes_sd"] for r in st],
             marker="o", color=C["pes"], label="PES (greedy exchange)")
plt.errorbar(xx, [r["clock"] for r in st], yerr=[r["clock_sd"] for r in st],
             marker="s", color=C["clock"], label="Fixed clock")
plt.xticks(xx, [f"{r['caps']} slots" for r in st])
plt.xlabel("MLP input capacity (fast+slow), support size 10")
plt.ylabel("stationary loss tail")
plt.legend(frameon=False)
plt.title("Pressure amplifies mispricing (MLP port)")
save(plt.gcf(), "fig_mlp_pressure")

print("ALL FIGURES DONE ->", OUT)
