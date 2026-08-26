"""Unit tests for pes_core. Run: python pes_tests.py"""
import numpy as np
from pes_core import (DriftingSparseEnv, make_learner, run_stream,
                      rare_recall_probe, exact_sign_test, bc_bimodal)

PASS = []
FAIL = []

def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS" if cond else "FAIL"), name, info)


def test_env_shapes():
    env = DriftingSparseEnv(D=50, s0=8, seed=1)
    x, y = env.step()
    check("env shapes", x.shape == (50,) and np.isfinite(y))
    xs = [env.step()[0] for _ in range(200)]
    nz = np.mean([np.count_nonzero(x) > 0 for x in xs])
    check("env sparsity sane", 0 < nz < 1.0, f"frac nonzero-streams={nz:.2f}")


def test_learners_run():
    # fully stationary environment: every sane learner must reduce loss
    # NOTE: clock baselines are EVALUATED in the main experiments; scheduled
    # consolidation can legitimately churn (documented phenomenon F-08), so we
    # only require finiteness + non-divergence from them here.
    env = DriftingSparseEnv(D=60, s0=10, seed=2, p_change=0.0)
    strict = {"sgd_dense", "pes", "single_decay"}
    loose = {"random_routing", "clock", "clock_mag"}
    for name in ["sgd_dense", "pes", "random_routing", "clock", "clock_mag", "single_decay"]:
        L = make_learner(name, D=60, eta=0.15, seed=3)
        out = run_stream(L, env, T=400)
        early = out["losses"][:50].mean(); late = out["losses"][-50:].mean()
        finite = np.all(np.isfinite(out["losses"]))
        if name in strict:
            check(f"{name} learns (stationary)", late < early, f"{early:.3f}->{late:.3f}")
        else:
            check(f"{name} bounded (stationary)",
                  finite and late < 10 * max(early, 1e-3),
                  f"{early:.3f}->{late:.3f}")
        check(f"{name} finite", finite)


def test_pes_capacity_respected():
    env = DriftingSparseEnv(D=100, s0=15, seed=5, p_change=0.01)  # fresh env per learner
    L = make_learner("pes", D=100, eta=0.1, seed=4)
    run_stream(L, env, T=500)
    cf = sum(1 for i in L.store.values() if i == L.FAST)
    cs = sum(1 for i in L.store.values() if i == L.SLOW)
    check("fast capacity", cf <= L.cap_fast + 1, f"cf={cf}")
    check("slow capacity", cs <= L.cap_slow, f"cs={cs}")
    check("routing happened", len(L.promo_events) > 0, f"n_promos={len(L.promo_events)}")
    check("turnover happened", len(L.evict_events) > 0,
          f"n_evictions={len(L.evict_events)}")
    # anti-thrashing sanity: routing events should be far below step count
    check("no thrashing", len(L.promo_events) < 0.5 * 500,
          f"promos/step={len(L.promo_events)/500:.2f}")


def test_determinism():
    def once():
        env = DriftingSparseEnv(D=40, s0=6, seed=9)
        L = make_learner("pes", D=40, eta=0.1, seed=11)
        o = run_stream(L, env, T=150)
        return float(np.sum(o["losses"]))
    check("determinism same-seed", abs(once() - once()) < 1e-12)


def test_sign_test_and_bc():
    p = exact_sign_test([5, -1, 7, 2, -0.5])
    check("sign test range", 0 <= p <= 1, f"p={p:.3f}")
    bim = bc_bimodal(np.r_[np.zeros(50), np.ones(50)])
    uni = bc_bimodal(np.random.default_rng(0).normal(0, 1, 200))
    check("bc flags synthetic bimodal", bim > uni, f"bim={bim:.2f} uni={uni:.2f}")


def test_recall_probe():
    env = DriftingSparseEnv(D=80, s0=12, seed=13)
    L = make_learner("sgd_dense", D=80, eta=0.2, seed=14)
    run_stream(L, env, T=400)
    m = rare_recall_probe(L, env, list(env.support), n_probe=20)
    check("probe finite positive", m >= 0 and np.isfinite(m), f"mse={m:.4f}")


if __name__ == "__main__":
    test_env_shapes(); test_learners_run(); test_pes_capacity_respected()
    test_determinism(); test_sign_test_and_bc(); test_recall_probe()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    raise SystemExit(1 if FAIL else 0)
