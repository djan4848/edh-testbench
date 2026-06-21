"""
edh — Entropic Drive Hypothesis test bench.

A reproducible bench that simulates and tries to FALSIFY the falsifiable
predictions (P1, P2, P3) of the EDH working paper. Success is honest machinery
that *could* refute the paper, not confirmation. See README.md and §3 of the
spec for the anti-cheat rules (no exogenous truncation, no hardcoded cost, N*
is measured, ensemble-at-fixed-t PID, local coupling, pre-registered stats).
"""
from __future__ import annotations

import numpy as np

# Project-wide physical constant. We work in units where k_B = 1, so all
# energies/temperatures are dimensionless multiples of k_B. Landauer's bound is
# then ln2 * T per erased bit. Kept as a named constant so it never becomes a
# magic number scattered across modules.
K_B: float = 1.0
LN2: float = float(np.log(2.0))

__all__ = ["K_B", "LN2", "seed_everything"]


def seed_everything(seed: int) -> np.random.Generator:
    """Seed the global legacy numpy RNG and return a fresh Generator.

    The Generator is what callers should thread through simulations; the legacy
    global seed is set too so any stray ``np.random.*`` call is also
    deterministic. Same seed in -> same arrays out (verified in
    tests/test_determinism.py).
    """
    np.random.seed(seed)
    return np.random.default_rng(seed)
