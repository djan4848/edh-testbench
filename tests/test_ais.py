"""Sanity for the discrete MI / AIS estimator: independent -> ~0 (after bias
subtraction), copy -> ~H, and the degeneracy guard fires on tiny support."""
from __future__ import annotations

import numpy as np

from edh.ais import discretize, mutual_info_discrete


def test_independent_is_zero_after_bias():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 5000)
    y = rng.integers(0, 4, 5000)
    r = mutual_info_discrete(x, y, min_occupied=4, rng=rng)
    assert r.ok
    assert abs(r.ais) < 0.02          # bias-corrected MI of independent ~ 0


def test_copy_is_entropy():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 4, 5000)
    r = mutual_info_discrete(x, x.copy(), min_occupied=4, rng=rng)
    assert r.ok
    assert abs(r.ais - r.h_target) < 0.05    # MI(x;x) = H(x)
    assert r.h_target > 1.5


def test_degenerate_support_guard():
    x = np.zeros(100, dtype=int)
    y = np.zeros(100, dtype=int)
    r = mutual_info_discrete(x, y, min_occupied=8)
    assert not r.ok and "degenerate" in r.reason


def test_discretize_edges():
    d = np.array([0.0, 0.24, 0.26, 0.5, 0.99])
    s = discretize(d, q=4)
    assert s.min() >= 0 and s.max() <= 3
    assert s[0] == 0 and s[-1] == 3
