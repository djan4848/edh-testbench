"""
edh.stats — statistics helpers (bootstrap, log-linearity, Monte-Carlo slope).

Track-A P1 q-sweep uses: per-point N* bootstrap over seeds, Monte-Carlo
propagation of per-point N* uncertainty into the slope q, and a log-linearity
diagnostic (R^2 + quadratic curvature) to flag when a power-law fit is only
LOCAL (the curve is bending). Track-B onset/surrogate helpers will be added in
Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "bootstrap_nstar",
    "ols_slope_intercept",
    "loglinearity",
    "mc_slope_distribution",
    "ci95",
    "phi_half_crossing",
    "two_regime_fit",
    "onset",
    "sign_test_one_sided",
]


def ci95(samples: np.ndarray) -> tuple[float, float]:
    """2.5/97.5 percentile interval."""
    s = np.asarray(samples, dtype=float)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def bootstrap_nstar(
    n_grid: np.ndarray,
    phi_matrix: np.ndarray,
    locate,
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap N* over seeds.

    phi_matrix is (n_N, n_seed). ``locate`` maps (n_grid, phi_matrix) -> N*
    (float). We resample seed columns with replacement and re-locate N* each time.
    Returns the bootstrap N* sample (n_boot,).
    """
    phi = np.asarray(phi_matrix, dtype=float)
    n_seed = phi.shape[1]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        cols = rng.integers(0, n_seed, n_seed)
        out[b] = locate(n_grid, phi[:, cols])
    return out


def ols_slope_intercept(logx: np.ndarray, logy: np.ndarray) -> tuple[float, float]:
    s, i = np.polyfit(np.asarray(logx, float), np.asarray(logy, float), 1)
    return float(s), float(i)


@dataclass
class LogLinearity:
    slope: float
    r2: float
    curvature: float          # quadratic coeff of a 2nd-order fit (0 => straight)
    concave: bool             # curvature significantly < 0 over the window
    note: str


def loglinearity(logx: np.ndarray, logy: np.ndarray,
                 curv_tol: float = 0.05) -> LogLinearity:
    """Assess whether logy is linear in logx.

    Returns the OLS slope, R^2, and the quadratic curvature coefficient of a
    degree-2 fit (normalized by the x-range). |curvature| <= curv_tol => treat as
    straight; curvature < -curv_tol => concave (power-law exponent still bending,
    so any q ~ 1/p relation is LOCAL, not global).
    """
    lx = np.asarray(logx, float)
    ly = np.asarray(logy, float)
    slope, intercept = ols_slope_intercept(lx, ly)
    resid = ly - (slope * lx + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    # quadratic fit; scale curvature by the squared x-span for a dimensionless number
    c2, c1, c0 = np.polyfit(lx, ly, 2)
    span = lx.max() - lx.min()
    curvature = float(c2 * span * span)
    concave = curvature < -curv_tol
    if abs(curvature) <= curv_tol:
        note = "log-linear (straight)"
    elif concave:
        note = "concave: exponent bending; treat power law as LOCAL"
    else:
        note = "convex: exponent steepening; treat power law as LOCAL"
    return LogLinearity(slope=slope, r2=r2, curvature=curvature,
                        concave=concave, note=note)


def onset(time: np.ndarray, x: np.ndarray, t_c: float, k: float = 3.0,
          require_consecutive: int = 1, min_abs_rise: float = 0.0) -> float:
    """Onset time of series x: first t > t_c at which x crosses
    baseline + k*sigma (baseline/sigma = mean/std over the ISOLATED phase t<t_c)
    AND stays above it for ``require_consecutive`` windows, AND exceeds the
    baseline by at least ``min_abs_rise`` in absolute terms.

    The sustained-crossing requirement removes the high false-positive rate of a
    single noise sample crossing a tiny-sigma baseline (measured ~35% on the
    decoupled placebo with require_consecutive=1). The isolated baseline absorbs
    the finite-sample estimator floor. NaN samples are skipped; returns NaN if no
    qualifying onset."""
    time = np.asarray(time, float)
    x = np.asarray(x, float)
    base_mask = (time < t_c) & np.isfinite(x)
    if base_mask.sum() < 2:
        return float("nan")
    base = float(np.mean(x[base_mask]))
    sd = float(np.std(x[base_mask]))
    thr = max(base + k * sd, base + min_abs_rise)
    post = np.where(time >= t_c)[0]
    for pos, i in enumerate(post):
        window = post[pos:pos + require_consecutive]
        if len(window) < require_consecutive:
            break
        vals = x[window]
        if np.all(np.isfinite(vals)) and np.all(vals > thr):
            return float(time[i])
    return float("nan")


def sign_test_one_sided(deltas: np.ndarray) -> tuple[float, float, int]:
    """One-sided sign test of median(delta) > 0 (binomial on positives vs negatives,
    zeros/NaN dropped). Returns (median, p_value, n_used)."""
    from scipy.stats import binomtest

    d = np.asarray(deltas, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), 0
    pos = int(np.sum(d > 0))
    p = binomtest(pos, n, 0.5, alternative="greater").pvalue
    med = float(np.median(np.asarray(deltas)[np.isfinite(deltas)]))
    return med, float(p), n


def phi_half_crossing(n_grid: np.ndarray, phi_mean: np.ndarray,
                      level: float = 0.5) -> float:
    """Sub-integer N where the order parameter phi(N) first crosses ``level``
    (linear interpolation). A smoother N* estimator than the chi-peak when the
    susceptibility is modest. Returns NaN if phi never crosses the level."""
    n = np.asarray(n_grid, float)
    p = np.asarray(phi_mean, float)
    for k in range(len(p) - 1):
        if (p[k] < level <= p[k + 1]) or (p[k + 1] < level <= p[k]):
            if p[k + 1] == p[k]:
                return float(n[k])
            f = (level - p[k]) / (p[k + 1] - p[k])
            return float(n[k] + f * (n[k + 1] - n[k]))
    return float("nan")


def _ols(mask, x, y):
    return float(np.polyfit(x[mask], y[mask], 1)[0]) if mask.sum() >= 2 else np.nan


def two_regime_fit(E, nstar, boot=None, sd=None, tol=1.0, n_ceil=24.0,
                   n_boot=4000, seed=0) -> dict:
    """Robust plateau+rise decomposition of N*(E) in log-log.

    A plateau is the leading run of points whose N* stays within ``tol`` of the
    floor (median of the 3 lowest N*); the rest is the rise. Slopes are fit by
    unweighted OLS (per-point bootstrap CIs are quantization-tight and would
    over-weight snapped points); regime-slope CIs come from a point-resampling
    bootstrap within each regime, drawing each point's N* from its saved
    bootstrap sample (``boot``) or a normal approx (``sd``).
    """
    E = np.asarray(E, float)
    nstar = np.asarray(nstar, float)
    x, y = np.log(E), np.log(nstar)
    order = np.argsort(E)
    floor = float(np.median(np.sort(nstar)[:3]))
    plateau = np.zeros(len(nstar), dtype=bool)
    for i in order:
        if nstar[i] <= floor + tol:
            plateau[i] = True
        else:
            break
    rise = ~plateau
    knee_lo = float(E[plateau].max()) if plateau.any() else float(E.min())
    knee_hi = float(E[rise].min()) if rise.any() else float(E.max())
    E_knee = float(np.sqrt(knee_lo * knee_hi))
    q_low, q_high = _ols(plateau, x, y), _ols(rise, x, y)

    rng = np.random.default_rng(seed)

    def boot_slope(mask):
        idxs = np.where(mask)[0]
        if len(idxs) < 2:
            return np.array([np.nan])
        out = []
        for _ in range(n_boot):
            pick = rng.choice(idxs, len(idxs), replace=True)
            if boot is not None:
                draw = np.array([boot[i][rng.integers(0, len(boot[i]))] for i in pick])
            elif sd is not None:
                draw = rng.normal(nstar[pick], sd[pick])
            else:
                draw = nstar[pick]
            xb = x[pick]
            if np.ptp(xb) < 1e-9:
                continue
            out.append(np.polyfit(xb, np.log(np.clip(draw, 2.0, n_ceil)), 1)[0])
        return np.array(out) if out else np.array([np.nan])

    return {
        "plateau_mask": plateau.tolist(),
        "rise_mask": rise.tolist(),
        "E_knee": E_knee,
        "E_knee_bracket": [knee_lo, knee_hi],
        "n_low": int(plateau.sum()),
        "n_high": int(rise.sum()),
        "q_low": q_low, "q_low_ci95": list(ci95(boot_slope(plateau))),
        "q_high": q_high, "q_high_ci95": list(ci95(boot_slope(rise))),
    }


def mc_slope_distribution(
    logx: np.ndarray,
    nstar_boot: list[np.ndarray],
    n_mc: int = 5000,
    seed: int = 0,
) -> np.ndarray:
    """Monte-Carlo distribution of the slope q = d log(N*) / d log(E/kT).

    For each MC draw, sample one N* per point from that point's bootstrap sample,
    take log, and fit the slope vs logx. Returns the q sample (n_mc,).
    Points with non-positive sampled N* are dropped from that fit.
    """
    lx = np.asarray(logx, float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_mc)
    for k in range(n_mc):
        ns = np.array([b[rng.integers(0, len(b))] for b in nstar_boot])
        good = ns > 0
        if good.sum() < 2:
            out[k] = np.nan
            continue
        out[k] = np.polyfit(lx[good], np.log(ns[good]), 1)[0]
    return out[~np.isnan(out)]
