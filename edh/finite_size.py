"""
edh.finite_size — order parameter, susceptibility, and N* location.

N* is a MEASURED quantity (§3.3): the peak of the susceptibility
chi(N) = Var_seeds[phi(N)], located at SUB-INTEGER resolution by parabolic
interpolation of chi around its argmax -- never the integer argmax, never the
"largest jump" of the mean (§6 of PROTOCOL.md).

This module is pure analysis over a phi[N, seed] matrix; it does not run the
dynamics (that is track_a_signaling) so it can be unit-tested on synthetic chi.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "susceptibility",
    "parabolic_peak",
    "fwhm",
    "unimodality_ratio",
    "PeakReport",
    "peak_report_from_chi",
    "locate_n_star",
]


def susceptibility(phi_matrix: np.ndarray) -> np.ndarray:
    """chi(N) = variance across seeds of phi at each N. phi_matrix is (n_N, n_seed)."""
    return np.var(np.asarray(phi_matrix, dtype=float), axis=1)


def parabolic_peak(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """Sub-sample peak abscissa via parabolic fit to (x[i-1..i+1], y[i-1..i+1]).

    Falls back to x[i] at the array boundary or for a degenerate parabola.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if i <= 0 or i >= len(x) - 1:
        return float(x[i])
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-15:
        return float(x1)
    # vertex of parabola through the three points (uniform spacing assumed)
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(x1 + delta * (x1 - x0))


def fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Full width at half maximum of curve y(x), by linear interpolation of the
    half-max crossings around the global peak. Returns the span in x-units."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ymax = y.max()
    if ymax <= 0:
        return float(x[-1] - x[0])
    half = 0.5 * ymax
    ipk = int(np.argmax(y))

    def cross(lo, hi):  # interpolate where y crosses `half` between idx lo,hi
        if y[hi] == y[lo]:
            return x[lo]
        f = (half - y[lo]) / (y[hi] - y[lo])
        return x[lo] + f * (x[hi] - x[lo])

    # left crossing
    left = x[0]
    for k in range(ipk, 0, -1):
        if y[k - 1] <= half <= y[k] or y[k] <= half <= y[k - 1]:
            left = cross(k - 1, k)
            break
    # right crossing
    right = x[-1]
    for k in range(ipk, len(x) - 1):
        if y[k] >= half >= y[k + 1] or y[k + 1] >= half >= y[k]:
            right = cross(k, k + 1)
            break
    return float(right - left)


def unimodality_ratio(y: np.ndarray) -> float:
    """global max / second-highest local maximum. Large => clean single peak.

    If there is no second local maximum, returns +inf (perfectly unimodal).
    """
    y = np.asarray(y, dtype=float)
    ipk = int(np.argmax(y))
    local_max = []
    for k in range(1, len(y) - 1):
        if y[k] >= y[k - 1] and y[k] >= y[k + 1] and k != ipk:
            local_max.append(y[k])
    # also consider boundary points that are not the peak as candidate competitors
    for k in (0, len(y) - 1):
        if k != ipk:
            local_max.append(y[k])
    second = max(local_max) if local_max else 0.0
    if second <= 0:
        return float("inf")
    return float(y[ipk] / second)


@dataclass
class PeakReport:
    n_star: float
    n_peak_int: int
    chi_max: float
    fwhm: float
    unimodal_ratio: float
    interior: bool
    sharp: bool
    centered: bool


def peak_report_from_chi(
    n_grid: np.ndarray,
    chi: np.ndarray,
    accept_band: tuple[float, float] = (3.0, 18.0),
    target_band: tuple[float, float] = (8.0, 12.0),
    fwhm_max: float = 8.0,
    smooth: bool = True,
) -> PeakReport:
    """Locate N* as the (sub-integer) peak of a given chi(N) curve and report
    peak quality. Pure analysis -- unit-testable on synthetic chi.

    ``smooth`` applies a light 3-point smoothing to chi *only for locating the
    peak* (chi_max is still read from the raw chi at the integer argmax).
    """
    n_grid = np.asarray(n_grid, dtype=float)
    chi = np.asarray(chi, dtype=float)
    chi_loc = chi.copy()
    if smooth and len(chi) >= 3:
        k = np.array([0.25, 0.5, 0.25])
        chi_loc = np.convolve(chi, k, mode="same")

    i = int(np.argmax(chi_loc))
    n_star = parabolic_peak(n_grid, chi_loc, i)
    width = fwhm(n_grid, chi_loc)
    uni = unimodality_ratio(chi_loc)
    interior = bool(accept_band[0] <= n_star <= accept_band[1])
    sharp = bool(width <= fwhm_max)
    centered = bool(target_band[0] <= n_star <= target_band[1])
    return PeakReport(
        n_star=float(n_star),
        n_peak_int=int(n_grid[i]),
        chi_max=float(chi[i]),
        fwhm=float(width),
        unimodal_ratio=float(uni),
        interior=interior,
        sharp=sharp,
        centered=centered,
    )


def locate_n_star(
    n_grid: np.ndarray,
    phi_matrix: np.ndarray,
    accept_band: tuple[float, float] = (3.0, 18.0),
    target_band: tuple[float, float] = (8.0, 12.0),
    fwhm_max: float = 8.0,
    smooth: bool = True,
) -> PeakReport:
    """Compute chi = Var_seeds[phi] then locate N* (see peak_report_from_chi)."""
    chi = susceptibility(phi_matrix)
    return peak_report_from_chi(
        n_grid, chi, accept_band=accept_band, target_band=target_band,
        fwhm_max=fwhm_max, smooth=smooth,
    )
