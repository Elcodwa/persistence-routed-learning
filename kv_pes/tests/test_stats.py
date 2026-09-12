"""Unit tests for paired statistics (Phase 4 methodology gate)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kv_pes.stats import exact_sign_test, paired_compare


def test_sign_test_all_same_direction():
    p, n = exact_sign_test([1, 2, 3, 4, 5])
    assert n == 5
    assert abs(p - 2 / 32) < 1e-9


def test_sign_test_mixed():
    p, n = exact_sign_test([1, -1, 0, 2, -3])
    assert n == 4  # zeros excluded


def test_sign_test_empty():
    p, n = exact_sign_test([0, 0, 0])
    assert p == 1.0 and n == 0


def test_paired_compare_fields():
    r = paired_compare([0.9, 0.8, 1.0], [0.5, 0.6, 0.4])
    assert r["n"] == 3
    assert r["mean_diff"] > 0
    assert r["sign_test_p"] == 0.25
