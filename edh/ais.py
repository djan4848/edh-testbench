"""
edh.ais — Active Information Storage (plain discrete MI, not PID).

AIS(N) = I(x_t ; x_{t-1}) over the stochastic ENSEMBLE at fixed (stationary)
time -- NEVER over a single deterministic trajectory (same limit-cycle
degeneracy that breaks PID). Here the agent state x is the discretized digitality
(code choice) of Track A; the ensemble is replicas x agents over a short
stationary tail window.

The estimator mirrors pid_fast's guards: it rejects degenerate support
(too few occupied joint cells), reports adequacy = n / occupied, and offers
permutation bias subtraction (the finite-sample MI floor estimated by shuffling
one variable). It also returns the marginal entropy H(x_t) and the normalized
AIS = MI / H, so we can see whether AIS tracks the cross-sectional spread or
genuine temporal self-predictability.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

__all__ = ["AISResult", "mutual_info_discrete", "discretize", "ais_from_traces"]


@dataclass
class AISResult:
    ais: float            # I(x_t ; history) in bits (bias-corrected if requested)
    ais_raw: float        # plug-in MI before bias subtraction
    bias: float           # permutation MI floor
    h_target: float       # H(x_t) marginal entropy (bits)
    normalized: float     # ais / h_target (fraction of state entropy stored)
    n: int
    occupied: int
    adequacy: float       # n / occupied joint cells
    ok: bool
    reason: str = ""


def _mi_plugin(src: np.ndarray, tgt: np.ndarray) -> tuple[float, int]:
    """Plug-in mutual information (bits) and number of occupied joint cells."""
    n = len(src)
    pxy = Counter(zip(src.tolist(), tgt.tolist()))
    px = Counter(src.tolist())
    py = Counter(tgt.tolist())
    mi = 0.0
    for (a, b), c in pxy.items():
        pab = c / n
        mi += pab * np.log2(pab / ((px[a] / n) * (py[b] / n)))
    return float(mi), len(pxy)


def _entropy(x: np.ndarray) -> float:
    n = len(x)
    c = Counter(x.tolist())
    p = np.array(list(c.values()), dtype=float) / n
    return float(-np.sum(p * np.log2(p)))


def mutual_info_discrete(src, tgt, min_occupied=8, min_adequacy=4.0,
                         bias_correct=True, n_shuffle=20, rng=None) -> AISResult:
    """Discrete MI I(src; tgt) with support/adequacy guards and optional
    permutation bias subtraction. src/tgt are equal-length integer arrays."""
    rng = np.random.default_rng(0) if rng is None else rng
    src = np.asarray(src).ravel()
    tgt = np.asarray(tgt).ravel()
    n = len(src)
    if n == 0:
        return AISResult(np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0.0,
                         False, "empty")
    mi_raw, occupied = _mi_plugin(src, tgt)
    adequacy = n / occupied if occupied else 0.0
    if occupied < min_occupied:
        return AISResult(np.nan, mi_raw, np.nan, np.nan, np.nan, n, occupied,
                         adequacy, False,
                         f"degenerate support ({occupied} cells)")
    bias = 0.0
    if bias_correct:
        floors = [_mi_plugin(src, rng.permutation(tgt))[0] for _ in range(n_shuffle)]
        bias = float(np.mean(floors))
    h_tgt = _entropy(tgt)
    ais = mi_raw - bias
    reason = "" if adequacy >= min_adequacy else "undersampled"
    return AISResult(ais=ais, ais_raw=mi_raw, bias=bias, h_target=h_tgt,
                     normalized=(ais / h_tgt if h_tgt > 1e-12 else 0.0),
                     n=n, occupied=occupied, adequacy=adequacy, ok=True,
                     reason=reason)


def discretize(d: np.ndarray, q: int = 4) -> np.ndarray:
    """Bin digitality in [0,1] into q integer states with FIXED edges (so states
    are comparable across N and E)."""
    edges = np.linspace(0.0, 1.0, q + 1)[1:-1]
    return np.digitize(np.asarray(d), edges).astype(int)


def ais_from_traces(d_stack: np.ndarray, q: int = 4, tail: int = 40, lag: int = 1,
                    min_occupied=8, min_adequacy=4.0, bias_correct=True,
                    n_shuffle=20, seed=0) -> AISResult:
    """AIS = I(x_t ; x_{t-1..t-lag}) from stacked digitality traces.

    d_stack: (S, T, n_agents) per-agent digitality over S replicas. We take the
    last ``tail`` steps (stationary), discretize to q states, and over the
    ensemble of (replica, agent, t) form target x_t and a base-q encoded history
    of the previous ``lag`` states. MI is estimated with guards.
    """
    rng = np.random.default_rng(seed)
    S, T, nag = d_stack.shape
    t0 = max(lag, T - tail)
    x = discretize(d_stack, q)              # (S, T, nag)
    tgt, hist = [], []
    for t in range(t0, T):
        tgt.append(x[:, t, :].ravel())
        code = np.zeros(S * nag, dtype=int)
        for j in range(1, lag + 1):
            code = code * q + x[:, t - j, :].ravel()
        hist.append(code)
    tgt = np.concatenate(tgt)
    hist = np.concatenate(hist)
    return mutual_info_discrete(hist, tgt, min_occupied=min_occupied,
                                min_adequacy=min_adequacy,
                                bias_correct=bias_correct, n_shuffle=n_shuffle,
                                rng=rng)
