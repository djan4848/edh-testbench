"""
Phase 0 acceptance: the BROJA PID wrapper reproduces canonical logic gates.

This pins the (target, s1, s2) ordering. If any of these fail, the ordering in
the reference solver wiring is wrong -- fix it there, not by tweaking tolerances.

  XOR  -> pure synergy   (Syn~1, U1~0, U2~0, Red~0)
  COPY -> pure unique-1  (U1~1,  rest ~0)
  DUP  -> pure redundancy (Red~1, rest ~0)
"""
from __future__ import annotations

import numpy as np
import pytest

from edh.pid import pid

TOL = 0.05
N = 20000
SEED = 0


@pytest.fixture(scope="module")
def sources():
    rng = np.random.default_rng(SEED)
    a = rng.integers(0, 2, N)
    b = rng.integers(0, 2, N)
    return a, b


def test_xor_is_pure_synergy(sources):
    a, b = sources
    r = pid(a, b, a ^ b, min_occupied=2)
    assert r.ok
    assert abs(r.syn - 1.0) < TOL, f"Syn={r.syn}"
    assert abs(r.unq1) < TOL and abs(r.unq2) < TOL, f"U1={r.unq1} U2={r.unq2}"
    assert abs(r.red) < TOL, f"Red={r.red}"


def test_copy_is_pure_unique1(sources):
    a, b = sources
    r = pid(a, b, a, min_occupied=2)
    assert r.ok
    assert abs(r.unq1 - 1.0) < TOL, f"U1={r.unq1}"
    assert abs(r.syn) < TOL and abs(r.unq2) < TOL and abs(r.red) < TOL, (
        f"Syn={r.syn} U2={r.unq2} Red={r.red}"
    )


def test_dup_is_pure_redundancy(sources):
    a, _ = sources
    r = pid(a, a, a, min_occupied=2)
    assert r.ok
    assert abs(r.red - 1.0) < TOL, f"Red={r.red}"
    assert abs(r.syn) < TOL and abs(r.unq1) < TOL and abs(r.unq2) < TOL, (
        f"Syn={r.syn} U1={r.unq1} U2={r.unq2}"
    )
