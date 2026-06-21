"""
experiments/exp_p1_decompose_cost.py
====================================
DIAGNOSTIC (not a hypothesis test). Analog-only, no adaptation dynamics, no
lambda. The cost function still never receives N.

It decomposes the measured analog cost exponent p into two physically distinct
factors, read off the entropy bookkeeping:

    a = d log(M)            / d log(N)   (signal-DEMAND exponent)
    b = d log(cost_per_sig) / d log(M)   (per-signal COST exponent vs M)

With the bookkeeping structure  D_analog = M * cost_per_signal  and
cost_per_signal ~ M^b, M ~ N^a, we have  D_analog ~ N^{a(1+b)}  =>  p = a(1+b).
We verify p_reconstructed ~= p_measured(=2.64).

Paper (Nowak pairwise demand + per-event ~M^2 cost):
    a_paper = 2  (M ~ N^2),   b_paper = 2   =>  a(1+b) = 6.

The script states EXPLICITLY which factor (a or b) is responsible for the gap
between our conservative model and the paper's 6.

§4 variant: a 'pairwise' demand task (M ~ N^2) with the SAME per-signal cost
geometry, to show whether forcing quadratic demand pushes p toward ~6 -- i.e.
that the paper's N^6 hinges entirely on the demand assumption, not the cost
physics. This is a comparison, not the primary model (flagged in LIMITATIONS.md).

Outputs: results/diagnostics/cost_decomposition.{png,csv,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "reference"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from edh.entropy import analog_dissipation  # noqa: E402
from edh.track_a_signaling import TrackAParams  # noqa: E402

OUTDIR = _ROOT / "results" / "diagnostics"
N_GRID = np.arange(2, 21)
N_SEEDS = 30
ROUNDS_FACTOR = 12        # comm. events per meaning -> near-full coverage
N_BOOT = 2000


# ---------------------------------------------------------------------------
# Demand models (task definition; may depend on N -- the cost never does)
# ---------------------------------------------------------------------------
def demand_linear(n: int) -> int:
    """Primary conservative task: one meaning per agent. M = N (a -> 1)."""
    return max(int(n), 1)


def demand_pairwise(n: int) -> int:
    """Nowak-style pairwise demand: a distinct signal per pair. M = N(N-1)/2
    (~N^2, a -> 2). Same per-signal cost geometry as the primary model."""
    return max(int(n) * (int(n) - 1) // 2, 1)


# ---------------------------------------------------------------------------
# Realized usage -> M_used and analog cost (stochastic over seeds via coverage)
# ---------------------------------------------------------------------------
def realized_usage(
    n: int, demand: Callable[[int], int], params: TrackAParams, seed: int
) -> tuple[int, float]:
    """Run analog-only usage for one seed; return (M_used, cost_total).

    Speakers draw meanings uniformly; M_used = number of DISTINCT meanings
    actually emitted (stochastic, ~M for enough rounds). The realized analog
    register holds those M_used signals: cost = separation energy over the
    realized amplitudes + Landauer to refresh log2(M_used) bits. The cost
    function receives the realized signal count, never N.
    """
    rng = np.random.default_rng(seed * 100_003 + n)
    m = demand(n)
    n_rounds = ROUNDS_FACTOR * m
    draws = rng.integers(0, m, size=n_rounds)
    m_used = int(np.unique(draws).size)
    cost = analog_dissipation(
        m_used, params.sep, params.sigma, params.temperature, params.stiffness
    ).total
    return m_used, cost


def _ols_slope(logx: np.ndarray, logy: np.ndarray) -> float:
    return float(np.polyfit(logx, logy, 1)[0])


def decompose(demand: Callable[[int], int], params: TrackAParams) -> dict:
    """Measure a, b, p_measured, p_reconstructed with seed-bootstrap CIs."""
    # per-N, per-seed realized usage
    m_used = np.zeros((len(N_GRID), N_SEEDS))
    cost = np.zeros((len(N_GRID), N_SEEDS))
    for ni, n in enumerate(N_GRID):
        for s in range(N_SEEDS):
            m_used[ni, s], cost[ni, s] = realized_usage(int(n), demand, params, s)

    m_bar_full = m_used.mean(axis=1)
    c_bar_full = cost.mean(axis=1)
    # Keep only well-defined points: at least 2 distinct signals and positive
    # cost (pairwise demand degenerates to M=1 -> cost 0 at the smallest N).
    valid = (m_bar_full >= 2) & (c_bar_full > 0)
    n_v = N_GRID[valid].astype(float)
    m_v = m_bar_full[valid]
    c_v = c_bar_full[valid]
    logN, logM = np.log(n_v), np.log(m_v)
    cps = c_v / m_v                                  # cost per signal
    logCPS, logC = np.log(cps), np.log(c_v)

    def fit(idx):
        a = _ols_slope(logN[idx], logM[idx])
        b = _ols_slope(logM[idx], logCPS[idx])
        p_meas = _ols_slope(logN[idx], logC[idx])
        return a, b, p_meas, a * (1.0 + b)

    full = np.arange(len(n_v))
    point = fit(full)
    # The analog cost law is DETERMINISTIC given the micro-model (full coverage
    # -> zero seed variance), so a seed bootstrap is degenerate. The meaningful
    # uncertainty is finite-N curvature: bootstrap by resampling N-grid points.
    rng = np.random.default_rng(0)
    boot = np.array([fit(rng.integers(0, len(n_v), len(n_v)))
                     for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    keys = ["a", "b", "p_measured", "p_reconstructed"]
    return {
        "point": dict(zip(keys, [float(x) for x in point])),
        "ci95_Nbootstrap": {k: [float(lo[i]), float(hi[i])]
                            for i, k in enumerate(keys)},
        "ci_note": "seed-bootstrap degenerate (deterministic cost law); "
                   "CI is N-grid bootstrap (finite-N curvature)",
        "N_used": n_v.tolist(),
        "M_used_mean": m_v.tolist(),
        "cost_mean": c_v.tolist(),
    }


def write_outputs(linear: dict, pairwise: dict) -> None:
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTDIR.mkdir(parents=True, exist_ok=True)
    pl, pp = linear["point"], pairwise["point"]

    # JSON
    payload = {
        "p_measured_reference": 2.64,
        "paper": {"a": 2, "b": 2, "p": 6},
        "linear_demand": linear,
        "pairwise_demand": pairwise,
        "gap_attribution": gap_attribution(linear),
    }
    with open(OUTDIR / "cost_decomposition.json", "w") as f:
        json.dump(payload, f, indent=2)

    # CSV
    with open(OUTDIR / "cost_decomposition.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "a", "b", "p_measured", "p_reconstructed"])
        for name, d in (("linear", linear), ("pairwise", pairwise)):
            pt = d["point"]
            w.writerow([name, f"{pt['a']:.3f}", f"{pt['b']:.3f}",
                        f"{pt['p_measured']:.3f}", f"{pt['p_reconstructed']:.3f}"])
        w.writerow(["paper", 2, 2, 6, 6])

    # Figure: cost-total vs N (p) for both demand models
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name, d in ((axes[0], "linear (M=N)", linear),
                        (axes[1], "pairwise (M~N^2)", pairwise)):
        nn = np.array(d["N_used"])
        c = np.array(d["cost_mean"])
        ax.loglog(nn, c, "o-", label="analog cost")
        pt = d["point"]
        ax.set_title(f"{name}\n a={pt['a']:.2f}  b={pt['b']:.2f}  "
                     f"p={pt['p_measured']:.2f} (recon {pt['p_reconstructed']:.2f})")
        ax.set_xlabel("N"); ax.set_ylabel("analog dissipation")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("P1 cost decomposition: p = a(1+b).  paper: a=2,b=2,p=6")
    fig.tight_layout()
    fig.savefig(OUTDIR / "cost_decomposition.png", dpi=130)
    plt.close(fig)


def gap_attribution(linear: dict) -> str:
    a, b = linear["point"]["a"], linear["point"]["b"]
    da = abs(2 - a)       # gap contribution from demand
    db = abs(2 - b)       # gap contribution from per-signal cost
    if da > db:
        return (f"DEMAND factor a dominates the gap: a={a:.2f} vs a_paper=2 "
                f"(b={b:.2f} ~ b_paper=2). Our conservative model is cheaper "
                f"because it does NOT assume quadratic pairwise demand.")
    return (f"COST-per-signal factor b dominates the gap: b={b:.2f} vs "
            f"b_paper=2 (a={a:.2f}).")


def main() -> int:
    params = TrackAParams()
    print("=== P1 cost decomposition (diagnostic) ===")
    linear = decompose(demand_linear, params)
    pairwise = decompose(demand_pairwise, params)
    write_outputs(linear, pairwise)

    def show(name, d):
        pt, ci = d["point"], d["ci95_Nbootstrap"]
        print(f"\n[{name}]  ({d['ci_note']})")
        for k in ("a", "b", "p_measured", "p_reconstructed"):
            print(f"  {k:16s} = {pt[k]:6.3f}   95% CI [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")

    show("linear demand (PRIMARY, M=N)", linear)
    show("pairwise demand (VARIANT, M~N^2)", pairwise)
    print(f"\nPaper: a=2, b=2 -> p=a(1+b)=6")
    print(f"Gap attribution: {gap_attribution(linear)}")
    print(f"\nWrote {OUTDIR}/cost_decomposition.(png|csv|json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
