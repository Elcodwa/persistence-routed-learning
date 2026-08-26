"""PES — Persistence-Economics Selection: minimal experimental core (numpy-only).

Research prototype, intentionally small. All learners share the same predictor
(linear over resident traces) and the same SGD update; they differ ONLY in the
store/routing policy. This makes policy effects attributable.

Environment: sparse linear regression with drifting sparse targets and
heterogeneous feature recurrence (common vs rare coordinates), Poisson
change-points that re-randomize parts of the true support.
"""
import numpy as np

# ----------------------------------------------------------------------------- environment
class DriftingSparseEnv:
    """y_t = sum_{i in active} w*_i * x_{t,i} + noise.
    Coordinates have heterogeneous appearance probabilities (common/rare).
    At Poisson times, a random subset of the true support is re-randomized."""

    def __init__(self, D=200, s0=25, noise=0.05, p_change=0.002,
                 frac_common=0.8, rare_factor=30, seed=0,
                 aba=False, aba_period=None):
        self.rng = np.random.default_rng(seed)
        self.D, self.s0, self.noise, self.p_change = D, s0, noise, p_change
        self.aba, self.aba_period = aba, aba_period
        common = self.rng.choice(D, int(frac_common * D), replace=False)
        self.p_appear = np.full(D, 1.0 / (rare_factor * D))
        self.p_appear[common] = 5.0 / D  # commons appear much more often
        self.wstar = np.zeros(D)
        self.support = self.rng.choice(D, s0, replace=False)
        self.wstar[self.support] = self.rng.normal(0, 1, s0)
        self.t = 0
        self._saved_rules = {}   # for A/B/A: snapshot of (support,wstar)
        self.change_events = []  # (t, changed_coords)

    def _apply_change(self):
        k = max(1, int(self.s0 * 0.4))
        drop = self.rng.choice(self.support, k, replace=False)
        rest = np.setdiff1d(np.arange(self.D), self.support, assume_unique=True)
        add = self.rng.choice(rest, k, replace=False)
        self.support = np.setdiff1d(self.support, drop); self.support = np.append(self.support, add)
        self.wstar[:] = 0.0
        self.wstar[self.support] = self.rng.normal(0, 1, len(self.support))

    def reset_to_snapshot(self, key):
        sup, w = self._saved_rules[key]
        self.support, self.wstar = sup.copy(), w.copy()

    def step(self):
        self.t += 1
        if self.aba and self.aba_period and self.t % self.aba_period == 0:
            # cycle between two frozen rules instead of fresh randomness
            key = f"r{(self.t // self.aba_period) % 2}"
            if key not in self._saved_rules:
                self._saved_rules[key] = (self.support.copy(), self.wstar.copy())
            else:
                self.reset_to_snapshot(key)
        elif self.rng.random() < self.p_change:
            before = self.support.copy()
            self._apply_change()
            self.change_events.append((self.t, np.setdiff1d(before, self.support)))
        x = (self.rng.random(self.D) < self.p_appear) * self.rng.normal(0, 1, self.D)
        y = float(self.wstar @ x) + self.noise * self.rng.normal()
        return x, y


# ----------------------------------------------------------------------------- learners
class BaseLearner:
    """Linear predictor over resident coordinates + shared SGD."""
    def __init__(self, D, eta=0.1, seed=0):
        self.D, self.eta = D, eta
        self.rng = np.random.default_rng(seed + 999)

    def predict(self, x):
        idx = self.resident()
        w = self.get_w(idx)
        return float(w @ x[idx]), idx

    def update(self, x, y, yhat, idx):
        r = y - yhat
        grad = r * x[idx]
        self.apply_grad(idx, grad)
        return r, grad

    # --- interface ---
    def resident(self): raise NotImplementedError
    def get_w(self, idx): raise NotImplementedError
    def apply_grad(self, idx, g): raise NotImplementedError
    def observe_post(self, x, r, grad, idx): pass  # policy hook after SGD


class SGDDense(BaseLearner):
    """Unconstrained dense SGD (+optional L2). Upper-capacity baseline."""
    def __init__(self, D, eta=0.1, l2=0.0, seed=0):
        super().__init__(D, eta, seed)
        self.w = np.zeros(D); self.l2 = l2
    def resident(self): return np.arange(self.D)
    def get_w(self, idx): return self.w[idx]
    def apply_grad(self, idx, g):
        self.w[idx] += self.eta * g
        if self.l2 > 0: self.w *= (1 - self.l2)


class SparseStore(BaseLearner):
    """Base class for capacity-limited learners with bookkeeping.
    Each resident coordinate i carries: w_i, value v_i (EMA of |grad|),
    age since last stimulation, store membership (0 fast / 1 slow)."""
    FAST, SLOW = 0, 1

    def __init__(self, D, eta=0.1, cap_fast=45, cap_slow=15, rho=0.98,
                 mu_fast=0.02, lam_slow=1e-4, kappa=0.05, t_recover=100, seed=0,
                 kappa_move=0.02, hysteresis_w=25, price_mode="ema_grad"):
        super().__init__(D, eta, seed)
        self.cap_fast, self.cap_slow = cap_fast, cap_slow
        self.rho = rho                      # value EMA memory
        self.mu_fast = mu_fast              # unstimulated fast decay rate
        self.lam_slow = lam_slow            # slow maintenance tariff per step
        self.kappa = kappa                  # legacy scalar (kept for config compat)
        self.kappa_move = kappa_move        # one-time value charge per migration
        self.hysteresis_w = hysteresis_w    # min steps between routing swaps
        self.t_recover = t_recover          # (unused since F-07 redesign; kept)
        self.maturity = 50                  # min lifetime before promo eligibility
        self.birth_t = {}                   # i -> first admission time
        self.w = {}                         # i -> weight
        self.v = {}                         # i -> value estimate
        self.age = {}                       # i -> steps since last stimulation
        self.store = {}                     # i -> FAST/SLOW
        self.entry_t = {}                   # i -> time entered current store
        self.last_touch = {}                # i -> last time the trace was used
        self.last_route_t = -10**9          # hysteresis: no swaps within window
        self.promo_events = []              # (t, i, direction)
        self.evict_events = []
        self.stim_q = 0.05                  # stimulation quantile on |x|
        # R7: price_mode 'ema_grad' (original EMA|grad| tag) or 'loo'
        # (exact leave-one-out loss increase if trace were deleted now).
        self.price_mode = price_mode
        self.last_r = 0.0                   # cached prediction residual for LOO

    def resident(self):
        return np.array(sorted(self.w.keys()), dtype=int)

    def get_w(self, idx): return np.array([self.w[i] for i in idx])

    def apply_grad(self, idx, g):
        for i, gi in zip(idx, g):
            self.w[i] += self.eta * gi

    def _capacity(self, s):
        return self.cap_slow if s == self.SLOW else self.cap_fast

    def _pi(self, i, s, t):
        """Persistence score of placing trace i in store s (see formalization).
        staleness = time since the trace last mattered (entered store OR was used)."""
        mu_s = 0.0 if s == self.SLOW else self.mu_fast
        ref = max(self.entry_t.get(i, t), self.last_touch.get(i, 0))
        staleness = t - ref
        lam_s = self.lam_slow if s == self.SLOW else 0.0
        # NOTE (F-06): utility-DENSITY v/(eps+|w|) perversely rewards zero-weight
        # newcomers (denominator -> 0) and evicts trained traces. Raw accumulated
        # responsibility is the correct valuation here; cost side is carried by
        # the maintenance tariffs (lambda, mu*staleness) and the SWITCH cost kappa
        # (applied as a hysteresis gate in route(), not inside the score).
        u = self.v[i] + 1e-9
        return u - lam_s - mu_s * staleness

    def _admit(self, i, w0, s, t):
        tgt = self._evict_target(s, exclude=i)
        self.store[i] = s; self.w[i] = w0; self.v[i] = 1e-3
        self.age[i] = 0; self.entry_t[i] = t; self.last_touch[i] = t
        self.birth_t.setdefault(i, t)
        if tgt is not None: self._drop(tgt, t, reason="evict")

    def _evict_target(self, s, exclude=-1):
        members = [j for j in self.store if self.store[j] == s and j != exclude]
        if len(members) >= self._capacity(s):
            pis = [(self._pi(j, s, self._now()), j) for j in members]
            return min(pis)[1]
        return None

    def _now(self): return self._t

    def _drop(self, i, t, reason="forget"):
        self.w.pop(i, None); self.v.pop(i, None); self.age.pop(i, None)
        st = self.store.pop(i, None); self.entry_t.pop(i, None)
        self.evict_events.append((t, i, reason))

    # ---- main policy hook ----
    def observe_post(self, x, r, grad, idx):
        self._t = getattr(self, "_t", 0) + 1
        t = self._t
        self.last_r = float(r)              # cached residual drives LOO pricing
        stim = np.quantile(np.abs(x), 1 - self.stim_q)
        # write rule scans ALL stimulated coordinates (not just current residents,
        # otherwise unseen coordinates could never enter the store)
        touched = set(int(i) for i in np.nonzero(np.abs(x) > stim)[0])
        # 1) value + age update for residents
        for i in list(self.w.keys()):
            if self.price_mode == "loo":
                # R7 exact leave-one-out price: loss increase from deleting trace i
                # NOW, holding the residual fixed (linear predictor):
                #   L(w - w_i e_i) - L(w) = r*(w_i x_i) + 0.5*(w_i x_i)^2
                wx = float(self.w.get(i, 0.0)) * float(x[i])
                resp = max(0.0, self.last_r * wx + 0.5 * wx * wx)
            else:
                resp = abs(grad[idx == i][0]) if i in idx else 0.0
            self.v[i] = self.rho * self.v[i] + (1 - self.rho) * resp
            if i in touched:
                self.age[i] = 0; self.last_touch[i] = t
                if self.store[i] == self.FAST and self.mu_fast > 0:
                    self.w[i] *= (1 - self.mu_fast * 0.5)   # partial decay on use
            elif i in idx:
                # RESIDENT but not among top-stimulated inputs: still refresh the
                # touch reference so slow-trace staleness keeps moving (Prop 1's
                # eviction axis needs live staleness for ALL residents).
                self.age[i] += 1; self.last_touch[i] = t
            else:
                self.age[i] += 1
                if self.store[i] == self.FAST:
                    self.w[i] *= (1 - self.mu_fast)
        # 2) admit unseen stimulated coords into fast (write rule)
        for i in sorted(touched):
            if i not in self.w:
                self._admit(int(i), 0.0, self.FAST, t)
        # 3) routing: exchanges between stores
        self.route(t)
        # 4) cleanup: fully-decayed traces carry no information -> remove.
        #    (Deletion is NOT driven by low Pi alone: idle fast traces cost ~0,
        #     so they are kept until capacity pressure evicts them. This matches
        #     the formalization: maintenance cost of fast is paid via decay.)
        for i in list(self.w.keys()):
            if abs(self.w[i]) < 1e-5 and self.v[i] < 1e-8 and i not in touched:
                self._drop(i, t, reason="decayed")

    def route(self, t):
        # switching-cost hysteresis: no swaps within hysteresis_w steps
        if t - self.last_route_t < self.hysteresis_w:
            return
        fast = [i for i in self.store if self.store[i] == self.FAST]
        slow = [i for i in self.store if self.store[i] == self.SLOW]
        if not fast or not slow:
            if fast and len(slow) == 0:
                # cold-start fill: only MATURE traces (survived >= maturity steps)
                cands = [i for i in fast if t - self.birth_t.get(i, t) >= self.maturity]
                # MARGIN rule (matches Propositions): rank by Pi(i,slow)-Pi(i,fast)
                cands.sort(key=lambda i: self._pi(i, self.SLOW, t)
                                      - self._pi(i, self.FAST, t), reverse=True)
                for i_star in cands[:min(self.cap_slow, 5)]:
                    self._move(i_star, self.SLOW, t)
                if cands: self.last_route_t = t
            return
        # main exchange: MARGIN rule per corrected Propositions 1--2
        # donor maximizes Delta_i = Pi(i,slow)-Pi(i,fast); evictee minimizes Delta_j
        elig = [i for i in fast
                if t - self.birth_t.get(i, t) >= self.maturity and i in self.last_touch]
        if not elig:
            return
        deltas = {i: self._pi(i, self.SLOW, t) - self._pi(i, self.FAST, t) for i in elig}
        pi_slow_cand = max(elig, key=lambda i: deltas[i])
        pi_fast_cand = min(slow, key=lambda j: self._pi(j, self.FAST, t)
                                         - self._pi(j, self.SLOW, t))
        gain_up = deltas[pi_slow_cand] - (self._pi(pi_fast_cand, self.FAST, t)
                                          - self._pi(pi_fast_cand, self.SLOW, t))
        # migrate only if gain clears the one-time switching charge 2*kappa_move
        if gain_up > 2 * self.kappa_move:
            self._move(pi_slow_cand, self.SLOW, t)
            slow2 = [i for i in self.store if self.store[i] == self.SLOW]
            if len(slow2) > self.cap_slow:
                j = min(slow2, key=lambda k: self._pi(k, self.FAST, t)
                                            - self._pi(k, self.SLOW, t))
                self._move(j, self.FAST, t)
            self.last_route_t = t

    def _move(self, i, s, t):
        if self.store.get(i) == s: return
        old = self.store.get(i)
        # one-time migration charge: moving disrupts the trace, discount its value
        self.v[i] = max(0.0, self.v[i] - self.kappa_move)
        self.store[i] = s; self.entry_t[i] = t
        self.promo_events.append((t, i, "up" if s == self.SLOW else "down"))


class PES(SparseStore):
    """The proposed mechanism exactly as formalized (greedy exchange on Pi)."""
    pass


class RandomRouting(SparseStore):
    """Control C-b: identical machinery, valuation replaced by noise."""
    def route(self, t):
        self.v = {i: float(self.rng.random()) for i in self.w}
        super().route(t)


class ClockConsolidator(SparseStore):
    """Fixed-profile baselines: same stores, consolidation purely periodic.
    mode='value': promote/demote by EMA-responsibility rank (STRONG baseline:
                  identical valuation signal to PES, fixed schedule, no economics).
    mode='mag':   promote/demote by |w| rank (classic magnitude heuristic)."""
    def __init__(self, period=200, mode="value", **kw):
        super().__init__(**kw); self.period = period; self.mode = mode
    def _rank_key(self, i):
        return (self.v[i] if self.mode == "value" else abs(self.w[i]))
    def route(self, t):
        if t % self.period != 0: return
        fast = [i for i in self.store if self.store[i] == self.FAST]
        slow = [i for i in self.store if self.store[i] == self.SLOW]
        if not fast: return
        i_star = max(fast, key=self._rank_key)
        if len(slow) < self.cap_slow:
            self._move(i_star, self.SLOW, t)
        elif slow:
            j = min(slow, key=self._rank_key)
            self._move(j, self.FAST, t); self._move(i_star, self.SLOW, t)


class SingleStoreDecay(SparseStore):
    """Delta-rule-like: one big fast store, no slow, tuned decay."""
    def __init__(self, cap_total=60, **kw):
        kw["cap_slow"] = 0; super().__init__(cap_fast=cap_total, **kw)
    def route(self, t): pass


# ----------------------------------------------------------------------------- runner
def make_learner(name, D, eta, seed, caps=(45, 15)):
    cf, cs = caps
    if name == "sgd_dense":      return SGDDense(D, eta, l2=0.0, seed=seed)
    if name == "sgd_l2":         return SGDDense(D, eta, l2=1e-3, seed=seed)
    if name == "pes_loo":        # R7: PES with exact leave-one-out prices
        return PES(D, eta, cap_fast=cf, cap_slow=cs, seed=seed, price_mode="loo")
    if name == "sgd_wd":         # R8: dense SGD + weight decay (single-store kill test)
        return SGDDense(D, eta, l2=2e-3, seed=seed)
    if name == "pes":            return PES(D, eta, cap_fast=cf, cap_slow=cs, seed=seed)
    if name == "random_routing": return RandomRouting(D, eta, cap_fast=cf, cap_slow=cs, seed=seed)
    if name == "clock":          return ClockConsolidator(period=200, mode="value", D=D, eta=eta,
                                                          cap_fast=cf, cap_slow=cs, seed=seed)
    if name == "clock_mag":      return ClockConsolidator(period=200, mode="mag", D=D, eta=eta,
                                                          cap_fast=cf, cap_slow=cs, seed=seed)
    if name == "single_decay":   return SingleStoreDecay(cap_total=cf + cs, D=D, eta=eta, seed=seed)
    raise ValueError(name)


def run_stream(learner, env, T, record_residence=True):
    losses = np.empty(T); resid = np.empty(T)
    residence = []   # (t, i, store) snapshots sampled every 50 steps
    promo_ts, promo_mag = [], []
    err_series = np.empty(T)
    for t in range(T):
        x, y = env.step()
        yhat, idx = learner.predict(x)
        losses[t] = 0.5 * (y - yhat) ** 2
        r, g = learner.update(x, y, yhat, idx)
        resid[t] = abs(r); err_series[t] = abs(r)
        before = len(getattr(learner, "promo_events", []) or [])
        learner.observe_post(x, r, g, idx)
        ne = len(getattr(learner, "promo_events", []) or [])
        if ne > before:
            for ev in learner.promo_events[before:]:
                promo_ts.append(ev[0]); promo_mag.append(abs(r))
        if record_residence and t % 50 == 0 and hasattr(learner, "store"):
            residence.extend((t, i, s) for i, s in learner.store.items())
    return dict(losses=losses, resid=resid, residence=residence,
                promo_ts=promo_ts, promo_mag=promo_mag, err=err_series)


def rare_recall_probe(learner, env, coords, n_probe=40, rng=None):
    """MSE on given coordinates with others zeroed (isolated recall test)."""
    rng = rng or np.random.default_rng(7)
    errs = []
    for _ in range(n_probe):
        x = np.zeros(env.D)
        sel = rng.choice(coords, size=min(len(coords), 3), replace=False)
        x[sel] = rng.normal(0, 1, len(sel))
        y = float(env.wstar @ x)
        yhat, _ = learner.predict(x)
        errs.append((y - yhat) ** 2)
    return float(np.mean(errs))


def exact_sign_test(diffs):
    """Exact paired sign test (two-sided) via full enumeration; n<=20."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0: return 1.0
    k = sum(1 for x in d if x > 0)
    from math import comb
    p = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / (2 ** n) * (2 if k * 2 != n else 1)
    return float(min(1.0, p))


def skew_kurt(x):
    x = np.asarray(x, dtype=float); m = x.mean(); s = x.std()
    if s == 0: return 0.0, 0.0
    sk = (((x - m) ** 3).mean()) / s ** 3
    ku = (((x - m) ** 4).mean()) / s ** 4 - 3.0
    return sk, ku


def bc_bimodal(x):
    sk, ku = skew_kurt(x)
    denom = max(ku + 3.0, 0.1)          # guard only against near-zero denominator
    return (sk ** 2 + 1) / denom


def xcorr_lag(a, b, max_lag=50):
    """Normalized cross-correlation of series a against impulse train b up to max_lag."""
    a = np.asarray(a) - np.mean(a); b = np.asarray(b) - np.mean(b)
    denom = (np.std(a) * np.std(b) * len(a))
    out = {}
    for lag in range(0, max_lag + 1):
        if lag == 0: num = float(np.mean(a * b))
        else: num = float(np.mean(a[lag:] * b[:-lag]))
        out[lag] = num / (denom / len(a) * len(a)) if denom != 0 else 0.0
    return out
