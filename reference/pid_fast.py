"""
pid_fast.py
===========
Drop-in replacement for the dit-based BROJA call in `information_dynamics.py`.

Why this exists
---------------
The original pipeline was slow/freezing for two coupled reasons:
  1. `dit`'s PID_BROJA wraps a generic convex optimizer that stalls on
     near-degenerate distributions (almost all target outcomes have zero
     probability). With the deterministic CA collapsing to period-2 cycles,
     every window fed BROJA a distribution with ~2 occupied cells out of 4096
     -> ill-conditioned cone program -> hang.
  2. `dit` drags in `pycddlib` (compiles cddlib/GMP), which is fragile to
     install on a fresh machine.

This module:
  * uses the dedicated, fast, robust BROJA_2PID solver (Makkeh, Theis,
    Vicente 2018) -- numpy + scipy + ecos only, no pycddlib;
  * MEMOIZES on the canonical empirical count table (huge win once the
    system settles -- many windows are identical). This is legitimate caching:
    identical input distribution -> identical output, by definition.
  * GUARDS against degenerate support: if the distribution is too sparse to
    estimate, it returns NaN + a reason flag instead of feeding garbage to the
    solver or fabricating a number;
  * reports a sampling-adequacy ratio so you can SEE under-sampling instead of
    silently trusting a biased estimate;
  * offers optional shuffle bias-subtraction (estimate the synergy floor from a
    label-permuted target and subtract it).

Install the solver locally:
    pip install ecos
    pip install BROJA_2PID            # if on PyPI for you, else:
    # git clone https://github.com/Abzinger/BROJA_2PID && add to PYTHONPATH

RUN THE SELF-TEST FIRST (verifies the source/target ordering against canonical
gates -- XOR must be pure synergy, COPY pure unique):
    python pid_fast.py
"""

from __future__ import annotations
import numpy as np
from collections import Counter
from functools import lru_cache

# ---------------------------------------------------------------------------
# Solver backend
# ---------------------------------------------------------------------------
try:
    from BROJA_2PID import pid as _broja_pid          # reference implementation
    _HAVE_BROJA = True
except Exception:
    _HAVE_BROJA = False


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------
class PIDResult:
    __slots__ = ("syn", "unq1", "unq2", "red", "mi", "n", "occupied",
                 "adequacy", "ok", "reason")

    def __init__(self, syn, unq1, unq2, red, mi, n, occupied, adequacy,
                 ok=True, reason=""):
        self.syn, self.unq1, self.unq2, self.red = syn, unq1, unq2, red
        self.mi = mi
        self.n = n
        self.occupied = occupied
        self.adequacy = adequacy      # n / occupied  (>= ~10 is comfortable)
        self.ok = ok
        self.reason = reason

    @property
    def unq(self):
        return np.nan if not self.ok else (self.unq1 + self.unq2) / 2.0


def _nan_result(n, occupied, reason):
    return PIDResult(np.nan, np.nan, np.nan, np.nan, np.nan,
                     n, occupied, (n / occupied if occupied else 0.0),
                     ok=False, reason=reason)


# ---------------------------------------------------------------------------
# Memoized core. Key = canonical sorted count table.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=200_000)
def _pid_from_counts(count_key):
    """count_key: tuple of ((s1, s2, target), count) sorted. Returns tuple."""
    counts = dict(count_key)
    total = sum(counts.values())

    # BROJA_2PID convention: pdf keyed by (T, S1, S2) -> probability.
    # We store keys as (s1, s2, target); remap to (target, s1, s2) here.
    pdf = {(t, a, b): c / total for (a, b, t), c in counts.items()}

    out = _broja_pid(pdf, cone_solver="ECOS", output=0)
    # BROJA_2PID returns: 'SI' shared/redundant, 'UIY','UIZ' unique,
    # 'CI' complementary/synergy.
    syn = max(out["CI"], 0.0)
    red = max(out["SI"], 0.0)
    unq1 = max(out["UIY"], 0.0)
    unq2 = max(out["UIZ"], 0.0)
    mi = syn + red + unq1 + unq2
    return syn, unq1, unq2, red, mi


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def pid(s1, s2, target, min_occupied=8, min_adequacy=4.0,
        coarse_grain=None):
    """
    Compute BROJA PID for two integer sources about an integer target.

    Parameters
    ----------
    s1, s2, target : 1D int arrays of equal length (the ensemble / window).
    min_occupied   : minimum number of distinct (s1,s2,target) cells required
                     to attempt estimation; below this the support is degenerate
                     (e.g. a short limit cycle) -> return NaN, ok=False.
    min_adequacy   : minimum samples-per-occupied-cell to trust the estimate.
                     Below this the result is returned but flagged (ok=True,
                     reason='undersampled') so you can filter/correct downstream.
    coarse_grain   : optional callable target->target to shrink the target
                     alphabet (recommended: reduce 6-bit joint target to <=3 bits
                     when ensemble size is modest). Applied before counting.

    Returns
    -------
    PIDResult
    """
    if not _HAVE_BROJA:
        raise ImportError(
            "BROJA_2PID not importable. Install with `pip install ecos` and add "
            "BROJA_2PID to PYTHONPATH (https://github.com/Abzinger/BROJA_2PID)."
        )

    s1 = np.asarray(s1).ravel()
    s2 = np.asarray(s2).ravel()
    target = np.asarray(target).ravel()
    if coarse_grain is not None:
        target = np.asarray(coarse_grain(target)).ravel()

    n = len(s1)
    if n == 0:
        return _nan_result(0, 0, "empty")

    triples = list(zip(s1.tolist(), s2.tolist(), target.tolist()))
    counts = Counter(triples)
    occupied = len(counts)

    if occupied < min_occupied:
        return _nan_result(n, occupied,
                           f"degenerate support ({occupied} cells) "
                           f"-- likely a deterministic short cycle; "
                           f"estimate over a stochastic ensemble instead")

    count_key = tuple(sorted(counts.items()))
    syn, unq1, unq2, red, mi = _pid_from_counts(count_key)

    adequacy = n / occupied
    reason = "" if adequacy >= min_adequacy else "undersampled"
    return PIDResult(syn, unq1, unq2, red, mi, n, occupied, adequacy,
                     ok=True, reason=reason)


def pid_bias_corrected(s1, s2, target, n_shuffle=20, rng=None, **kw):
    """
    Synergy is the component most inflated by under-sampling. Estimate the bias
    floor by permuting the target (destroys real structure, keeps marginals &
    sample size) and subtract it. Returns (PIDResult, syn_bias, syn_corrected).
    """
    rng = np.random.default_rng() if rng is None else rng
    base = pid(s1, s2, target, **kw)
    if not base.ok:
        return base, np.nan, np.nan
    tgt = np.asarray(target).ravel()
    syns = []
    for _ in range(n_shuffle):
        r = pid(s1, s2, rng.permutation(tgt), **kw)
        if r.ok:
            syns.append(r.syn)
    syn_bias = float(np.mean(syns)) if syns else np.nan
    return base, syn_bias, base.syn - syn_bias


# ---------------------------------------------------------------------------
# Self-test: canonical gates. RUN THIS LOCALLY before trusting any result.
# ---------------------------------------------------------------------------
def _self_test():
    if not _HAVE_BROJA:
        print("BROJA_2PID not installed -- cannot self-test. See install note.")
        return
    rng = np.random.default_rng(0)
    N = 20000
    a = rng.integers(0, 2, N)
    b = rng.integers(0, 2, N)

    print(f"{'gate':6s} {'Syn':>6s} {'U1':>6s} {'U2':>6s} {'Red':>6s}   expected")
    # XOR -> pure synergy (1 bit)
    r = pid(a, b, a ^ b, min_occupied=2)
    print(f"{'XOR':6s} {r.syn:6.3f} {r.unq1:6.3f} {r.unq2:6.3f} {r.red:6.3f}"
          f"   Syn~1, rest~0")
    # COPY of source1 -> pure unique-1 (1 bit)
    r = pid(a, b, a, min_occupied=2)
    print(f"{'COPY1':6s} {r.syn:6.3f} {r.unq1:6.3f} {r.unq2:6.3f} {r.red:6.3f}"
          f"   U1~1, rest~0")
    # AND -> mix (canonical: Red~0.31, U~0.04, Syn~0.5)
    r = pid(a, b, a & b, min_occupied=2)
    print(f"{'AND':6s} {r.syn:6.3f} {r.unq1:6.3f} {r.unq2:6.3f} {r.red:6.3f}"
          f"   Red~.31 Syn~.50")
    # Duplicated source -> pure redundancy
    r = pid(a, a, a, min_occupied=2)
    print(f"{'DUP':6s} {r.syn:6.3f} {r.unq1:6.3f} {r.unq2:6.3f} {r.red:6.3f}"
          f"   Red~1, rest~0")
    print("\nIf XOR is not ~pure-synergy and COPY1 not ~pure-unique-1, the "
          "(target, s1, s2) ordering in _pid_from_counts is wrong -- fix it.")


if __name__ == "__main__":
    _self_test()
