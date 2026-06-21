"""
Phase 0 acceptance: same seed -> bit-identical results (§3 rule 7).

Covers two RNG surfaces the bench depends on:
  * edh.seed_everything's returned Generator.
  * The reference stochastic ensemble simulator (Track B engine).
"""
from __future__ import annotations

import numpy as np

from edh import seed_everything


def test_generator_is_deterministic():
    g1 = seed_everything(12345)
    a = g1.random(1000)
    g2 = seed_everything(12345)
    b = g2.random(1000)
    assert np.array_equal(a, b)


def test_different_seed_differs():
    a = seed_everything(1).random(1000)
    b = seed_everything(2).random(1000)
    assert not np.array_equal(a, b)


def test_ensemble_simulation_is_deterministic():
    from dynamics_stochastic import simulate_ensemble

    kw = dict(N=5, R=32, L=8, T_isolated=20, T_coupled=30, seed=777)
    s1, tc1 = simulate_ensemble(**kw)
    s2, tc2 = simulate_ensemble(**kw)
    assert tc1 == tc2
    assert np.array_equal(s1, s2)


def test_ensemble_seed_changes_result():
    from dynamics_stochastic import simulate_ensemble

    base = dict(N=5, R=32, L=8, T_isolated=20, T_coupled=30)
    s1, _ = simulate_ensemble(seed=1, **base)
    s2, _ = simulate_ensemble(seed=2, **base)
    assert not np.array_equal(s1, s2)
