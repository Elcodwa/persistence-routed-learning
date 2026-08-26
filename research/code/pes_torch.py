"""Phase 6: PES routing with a nonlinear (MLP) predictor — torch port.

Scientific question: do the linear-model findings survive when a learned
nonlinearity sits between selected inputs and predictions?

Design:
- Same environment family as Phases 1-5 (DriftingSparseEnv).
- Learners select which input coordinates are resident (identical routing
  policies as before: pes / random_routing / clock / single_decay), but
  prediction is now a 2-layer MLP over the RESIDENT inputs only.
- All methods share identical MLP shape, LR, training steps per sample;
  differ only in the routing policy (same fairness contract).

Writes ../results/torch_phase6.json
"""
import json, time
import numpy as np

import torch
import torch.nn as nn

from pes_core import DriftingSparseEnv

DEVICE = "cpu"
ETA_SGD = 0.003         # inner LR of the MLP (batch-1 online stability; see mlp_diag6)
GRAD_CLIP = 1.0         # stabilize single-sample updates
PRICE_MODE = "tag"      # R9: 'tag' (|grad*x| EMA, original) or 'loo_surrogate' (deletion cost)
REGIME_KW = {
    "stationary": dict(p_change=0.0),
    "drifting":   dict(p_change=0.002),
    "volatile":   dict(p_change=0.01),
}
CAPS = (45, 15)
D, S0 = 200, 25


class ResidualMLP(nn.Module):
    """Predicts from the resident coordinates' raw values (fixed-size input via mask+pad)."""
    def __init__(self, cap_total, hidden=64):
        super().__init__()
        self.cap = cap_total
        self.net = nn.Sequential(
            nn.Linear(cap_total, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # small output-layer init: early predictions start near zero
        nn.init.normal_(self.net[2].weight, std=0.01)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, packed):          # packed: [B, cap]
        return self.net(packed).squeeze(-1)


class TorchStoreLearner:
    """Routing policies reused conceptually from pes_core, operating on coords.
    The MLP consumes exactly cap_fast+cap_slow resident values each step."""

    FAST, SLOW = 0, 1

    def __init__(self, policy, D, seed, caps=CAPS, rho=0.98,
                 mu_fast=0.02, lam_slow=1e-4, kappa_move=0.02, hysteresis_w=25):
        self.policy = policy            # 'pes' | 'random' | 'clock' | 'single'
        self.rng = np.random.default_rng(seed)
        self.D = D
        self.cap_fast, self.cap_slow = caps if policy != "single" else (caps[0] + caps[1], 0)
        self.rho, self.mu_fast, self.lam_slow = rho, mu_fast, lam_slow
        self.kappa_move, self.hysteresis_w = kappa_move, hysteresis_w
        self.w = {}; self.v = {}; self.age = {}; self.store = {}
        self.entry_t = {}; self.birth_t = {}; self.last_touch = {}
        self.last_route_t = -10**9
        # Stable slot binding: each resident coordinate owns ONE input slot of the
        # MLP until evicted (like a cache line). Without this, admissions shift
        # everyone's input position and scramble the network's learned weights.
        cap_total = self.cap_fast + max(self.cap_slow, 0)
        self.slot_of = {}
        self.free_slots = list(range(cap_total))
        self.maturity = 50
        self._pending_price = None   # R9: per-step deletion-cost prices when active
        self.torch_rng = torch.Generator().manual_seed(seed + 5)
        self.model = ResidualMLP(self.cap_fast + max(self.cap_slow, 0)).to(DEVICE)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=ETA_SGD)

    def _clip(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)

    # ---------- bookkeeping (same rules as numpy core) ----------
    def _pi(self, i, s, t):
        mu_s = 0.0 if s == self.SLOW else self.mu_fast
        ref = max(self.entry_t.get(i, t), self.last_touch.get(i, t))
        u = self.v[i] + 1e-9
        return u - (self.lam_slow if s == self.SLOW else 0.0) - mu_s * (t - ref)

    def _drop(self, i):
        s = self.slot_of.pop(i, None)
        if s is not None:
            self.free_slots.append(s)
        for d in (self.w, self.v, self.age, self.store, self.entry_t):
            d.pop(i, None)

    def _admit(self, i, t):
        members = [j for j in self.store if self.store[j] == self.FAST and j != i]
        if len(members) >= self.cap_fast:
            j = min(members, key=lambda k: self.v[k])
            self._drop(j)
        # reuse the most recently freed slot (locality), else any free slot
        self.slot_of[i] = self.free_slots.pop() if self.free_slots else None
        self.w[i] = 0.0; self.v[i] = 1e-3; self.age[i] = 0
        self.store[i] = self.FAST; self.entry_t[i] = t; self.birth_t.setdefault(i, t)

    def route(self, t):
        if self.policy == "single":
            return
        if self.policy == "clock":
            if t % 200 != 0:
                return
            fast = [i for i in self.store if self.store[i] == self.FAST]
            slow = [i for i in self.store if self.store[i] == self.SLOW]
            if not fast:
                return
            i_star = max(fast, key=lambda i: self.v[i])
            if len(slow) < self.cap_slow:
                self.store[i_star] = self.SLOW; self.entry_t[i_star] = t
            elif slow:
                j = min(slow, key=lambda i: self.v[i])
                self.store[j] = self.FAST; self.store[i_star] = self.SLOW
                self.entry_t[i_star] = t
            return
        if self.policy == "random":
            for i in list(self.v):
                self.v[i] = float(self.rng.random())
        # PES exchange (hysteresis-gated, maturity-gated)
        if t - self.last_route_t < self.hysteresis_w:
            return
        fast = [i for i in self.store if self.store[i] == self.FAST]
        slow = [i for i in self.store if self.store[i] == self.SLOW]
        if not slow:
            cands = [i for i in fast if t - self.birth_t.get(i, t) >= self.maturity]
            if cands:
                for i in sorted(cands, key=lambda k: -self._pi(k, self.SLOW, t))[:min(self.cap_slow, 5)]:
                    self.store[i] = self.SLOW; self.entry_t[i] = t
                self.last_route_t = t
            return
        elig = [i for i in fast if t - self.birth_t.get(i, t) >= self.maturity]
        if not elig:
            return
        up = max(elig, key=lambda i: self._pi(i, self.SLOW, t))
        dn = min(slow, key=lambda i: self._pi(i, self.FAST, t))
        if self._pi(up, self.SLOW, t) - self._pi(dn, self.FAST, t) > 2 * self.kappa_move:
            self.store[up] = self.SLOW; self.entry_t[up] = t
            self.v[up] = max(0.0, self.v[up] - self.kappa_move)
            if len([i for i in self.store if self.store[i] == self.SLOW]) > self.cap_slow:
                self.store[dn] = self.FAST; self.entry_t[dn] = t
            self.last_route_t = t

    # ---------- one stream step ----------
    def step(self, x, y, t):
        idx = np.array(sorted(self.w.keys()), dtype=int)
        packed = np.zeros(self.cap_fast + max(self.cap_slow, 0), dtype=np.float32)
        pos = {}
        for c in idx:
            k = self.slot_of.get(c)
            if k is not None:
                packed[k] = x[c]; pos[c] = k
        xb = torch.tensor(packed[None, :], device=DEVICE)
        xb.requires_grad_(True)                 # need d(output)/d(input) for valuation
        yb = torch.tensor([y], dtype=torch.float32, device=DEVICE)

        self.opt.zero_grad()
        yhat = self.model(xb)
        # value signal FIRST (graph still alive): price = deletion-cost estimate
        gi = torch.autograd.grad(yhat, xb,
                                 grad_outputs=torch.ones_like(yhat),
                                 retain_graph=True)[0][0].detach().numpy()
        if PRICE_MODE == "loo_surrogate":
            # R9: first-order Taylor of the deletion cost on the OUTPUT side.
            # Zeroing resident coord i's slot changes yhat by ~ -gi[slot]*x[slot];
            # the loss change is r*deltayhat + 0.5*deltayhat^2 (clamped at 0).
            # Sign check: if removing i moves prediction TOWARD y (reduces |r|),
            # dy has the same sign as r and the linear term prices it as a gain
            # (resp 0); if it moves prediction AWAY, resp grows. The quadratic
            # term keeps large deletions expensive either way.
            r_now = float(y - float(yhat.item()))
            self._pending_price = {}
            for i in list(self.w.keys()):
                k = self.slot_of.get(i)
                resp = 0.0
                if k is not None and i in pos:
                    dy = float(gi[k]) * float(packed[k])
                    lin = r_now * (-dy)              # loss change from removal, linear term
                    quad = 0.5 * dy * dy             # curvature safeguard
                    resp = max(0.0, lin + quad)
                self._pending_price[i] = resp
        else:
            self._pending_price = None
        loss = 0.5 * (yhat - yb) ** 2
        loss.backward()
        self._clip()
        self.opt.step()

        r = float(y - float(yhat.item()))
        touched_abs = np.abs(x)
        thr = np.quantile(touched_abs, 1 - 0.05)
        touched = set(int(i) for i in np.nonzero(np.abs(x) > thr)[0])
        for i in list(self.w.keys()):
            if self._pending_price is not None and i in self._pending_price:
                resp = self._pending_price[i]
            else:
                resp = abs(gi[pos[i]] * x[i]) if i in pos else 0.0
            self.v[i] = self.rho * self.v[i] + (1 - self.rho) * resp
            if i in touched:
                self.age[i] = 0; self.last_touch[i] = t
            else:
                self.age[i] += 1
        for i in sorted(touched):
            if i not in self.w:
                self._admit(i, t)
        self.route(t)
        return r * r


def run(policy, regime, seed, T=1500):
    env_kw = dict(D=D, s0=S0, noise=0.05, seed=1000 * seed + 13,
                  frac_common=0.8, rare_factor=30)
    env_kw.update(REGIME_KW[regime])
    env = DriftingSparseEnv(**env_kw)
    L = TorchStoreLearner(policy, D, seed)
    losses = []
    for t in range(1, T + 1):
        x, y = env.step()
        losses.append(L.step(x, y, t))
    tail = float(np.mean(losses[int(0.8 * T):]))
    cum = float(np.mean(losses))
    return dict(policy=policy, regime=regime, seed=seed, loss_tail=tail, loss_cum=cum)


if __name__ == "__main__":
    assert torch.__version__ >= "2", "torch>=2 required"
    t0 = time.time()
    rows = []
    # Calibrated Phase-6 config (see results/mlp_diag*.txt for the derivation):
    # easier task (s0=10, amplitude 0.6) so the ONLINE MLP converges; caps 55/15.
    PD, PS0 = 120, 10
    CAPS6 = (55, 15)
    REGIMES6 = {
        "stationary": dict(p_change=0.0),
        "drifting":   dict(p_change=0.002),
        "volatile":   dict(p_change=0.01),
    }
    T6 = 8000
    SEEDS6 = range(3)
    for regime, kw in REGIMES6.items():
        for policy in ["pes", "random", "clock", "single"]:
            for seed in SEEDS6:
                env = DriftingSparseEnv(D=PD, s0=PS0, noise=0.05,
                                        seed=1000 * seed + 13,
                                        frac_common=0.8, rare_factor=30, **kw)
                env.wstar[env.support] *= 0.6
                L = TorchStoreLearner(policy, PD, seed, caps=CAPS6)
                losses = []
                for t in range(1, T6 + 1):
                    x, y = env.step()
                    losses.append(L.step(x, y, t))
                tail = float(np.mean(losses[int(0.8 * T6):]))
                rows.append(dict(policy=policy, regime=regime, seed=seed,
                                 loss_tail=tail, loss_cum=float(np.mean(losses))))
            print(f"[{time.time()-t0:7.1f}s] {regime}/{policy} done", flush=True)
    import os
    os.makedirs("../results", exist_ok=True)
    with open("../results/torch_phase6.json", "w") as f:
        json.dump(dict(rows=rows, meta=dict(T=T6, D=PD, S0=PS0, seeds=len(list(SEEDS6)),
                                            caps=list(CAPS6), device=DEVICE,
                                            torch=torch.__version__,
                                            eta_sgd=ETA_SGD, grad_clip=GRAD_CLIP)), f, indent=1)
    print("\n=== Phase 6 (MLP) loss tails ===")
    regs = list(REGIMES6)
    print("policy".ljust(10) + "".join(r.ljust(18) for r in regs))
    for p in ["pes", "random", "clock", "single"]:
        line = p.ljust(10)
        for rg in regs:
            v = [r["loss_tail"] for r in rows if r["policy"] == p and r["regime"] == rg]
            line += f"{np.mean(v):.4f}±{np.std(v):.4f}   "
        print(line)
