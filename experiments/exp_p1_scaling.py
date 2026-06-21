"""
experiments/exp_p1_scaling.py
=============================
P1 q-sweep: measure N*(E_protocol) and the threshold exponent
q = d log(N*) / d log(E/kT), then test the EDH mechanism q ~= 1/p (dominant
balance k_B*T*N*^p ~= E_protocol).

Reusable for both demand models. The PRIMARY run is linear demand (M=N). The
queued pairwise run (M~N^2) reuses the same machinery with its own recalibrated
lambda and its own derived E-range.

Procedure per E point:
  * adaptive dynamics over N=2..20, n_seeds seeds, at (E, T=1, lambda).
  * N*(E) = sub-integer peak of chi(N)=Var_seeds[phi]; bootstrap CI over seeds.
q:
  * slope of log(N*) vs log(E/kT); CI by Monte-Carlo propagation of per-point N*.
p_sweep (range-matched, KEY): refit the analog cost exponent over [3,18] (the
range N* actually visits), analog-only -- NOT the global [2,20] p. Plus
log-linearity checks of (i) log D_analog vs log N and (ii) log N* vs log E; if
either is concave, q~=1/p is reported as LOCAL.

Verdict (mechanism, conservative model):
  guards red                                   -> INCONCLUSO
  IC(q) excludes 0 and overlaps 1/p_sweep      -> VERIFICADA (cost/entropy-driven)
  IC(q) excludes 0, entirely below 1/p_sweep   -> FALSEADA of the cost-driven
                                                  mechanism (coordination failure
                                                  contributes to the threshold)
  IC(q) includes 0                             -> INCONCLUSO
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "reference"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from edh.finite_size import peak_report_from_chi, susceptibility  # noqa: E402
from edh.stats import (  # noqa: E402
    bootstrap_nstar,
    ci95,
    loglinearity,
    mc_slope_distribution,
)
from edh.track_a_signaling import (  # noqa: E402
    TrackAParams,
    analog_cost_law,
    meanings_for_N,
    run_population,
)

ACCEPT_BAND = (3.0, 18.0)


def _locate_nstar(n_grid, phi_matrix) -> float:
    chi = susceptibility(phi_matrix)
    return peak_report_from_chi(n_grid, chi, accept_band=ACCEPT_BAND).n_star


def derive_E_range(e_center: float, n_peak: float, p: float,
                   n_points: int = 12) -> np.ndarray:
    """E grid keeping N* inside ACCEPT_BAND: E = e_center*(N_target/n_peak)^p."""
    e_lo = e_center * (ACCEPT_BAND[0] / n_peak) ** p
    e_hi = e_center * (ACCEPT_BAND[1] / n_peak) ** p
    return np.logspace(np.log10(e_lo), np.log10(e_hi), n_points)


def run_q_sweep(
    demand: Callable[[int], int],
    e_center: float,
    n_peak: float,
    p_global: float,
    lam: float,
    outdir: Path,
    n_seeds: int = 30,
    n_grid=tuple(range(2, 21)),
    n_points: int = 12,
    params: TrackAParams | None = None,
    n_boot: int = 1000,
    label: str = "linear",
    E_vals=None,
) -> dict:
    from tqdm import tqdm

    n_grid = np.array(n_grid, dtype=float)
    params = params or TrackAParams(e_center=e_center)
    E_vals = (derive_E_range(e_center, n_peak, p_global, n_points)
              if E_vals is None else np.asarray(E_vals, dtype=float))

    nstar_pt, nstar_boot, nstar_ci, interior = [], [], [], []
    for E in tqdm(E_vals, desc=f"q-sweep ({label})"):
        phi = np.zeros((len(n_grid), n_seeds))
        for ni, n in enumerate(n_grid):
            for s in range(n_seeds):
                phi[ni, s] = run_population(int(n), float(E), lam, params,
                                            seed=s, demand=demand).phi
        chi = susceptibility(phi)
        rep = peak_report_from_chi(n_grid, chi, accept_band=ACCEPT_BAND)
        nstar_pt.append(rep.n_star)
        interior.append(rep.interior)
        boot = bootstrap_nstar(n_grid, phi, _locate_nstar, n_boot=n_boot, seed=0)
        nstar_boot.append(boot)
        nstar_ci.append(ci95(boot))

    nstar_pt = np.array(nstar_pt)
    interior = np.array(interior, dtype=bool)
    logE = np.log(E_vals)                       # T=1 -> log(E/kT)=log E

    # --- q on the interior (transition-present) points ---
    use = interior & (nstar_pt > 0)
    q_point = float(np.polyfit(logE[use], np.log(nstar_pt[use]), 1)[0]) \
        if use.sum() >= 2 else float("nan")
    q_mc = mc_slope_distribution(
        logE[use], [nstar_boot[i] for i in np.where(use)[0]], n_mc=5000, seed=0)
    q_lo, q_hi = ci95(q_mc) if len(q_mc) else (float("nan"), float("nan"))

    # --- range-matched p_sweep over [3,18], analog-only ---
    n_rng = np.arange(3, 19)
    cost_rng = np.array([analog_cost_law(int(n), params, demand=demand)
                         for n in n_rng])
    ll_cost = loglinearity(np.log(n_rng), np.log(cost_rng))
    p_sweep = ll_cost.slope
    inv_p = 1.0 / p_sweep if p_sweep else float("nan")

    # --- log-linearity of N* vs E ---
    ll_nstar = loglinearity(logE[use], np.log(nstar_pt[use])) \
        if use.sum() >= 3 else None

    # --- guards ---
    logn = np.log10(nstar_pt[use]) if use.sum() else np.array([0.0])
    logn_range = float(logn.max() - logn.min()) if use.sum() else 0.0
    transition_frac = float(interior.mean())
    q_halfwidth_rel = (abs(q_hi - q_lo) / 2.0) / abs(q_point) \
        if q_point not in (0.0, float("nan")) and not np.isnan(q_point) else float("inf")
    guards = {
        "min_logN_star_range": {"value": logn_range, "threshold": 0.4,
                                "ok": logn_range >= 0.4},
        "require_transition_fraction": {"value": transition_frac, "threshold": 0.7,
                                        "ok": transition_frac >= 0.7},
        "max_q_CI_halfwidth_rel": {"value": q_halfwidth_rel, "threshold": 0.5,
                                   "ok": q_halfwidth_rel < 0.5},
    }
    guards_ok = all(g["ok"] for g in guards.values())

    # --- mechanism test + verdict ---
    excludes_zero = (q_lo > 0) or (q_hi < 0)
    overlaps_invp = (q_lo <= inv_p <= q_hi)
    below_invp = q_hi < inv_p
    if not guards_ok:
        verdict = "INCONCLUSO"
        reason = "power guards failed"
    elif not excludes_zero:
        verdict = "INCONCLUSO"
        reason = "IC(q) includes 0 (no threshold scaling resolved)"
    elif overlaps_invp:
        verdict = "VERIFICADA"
        reason = "IC(q) excludes 0 and overlaps 1/p_sweep: cost/entropy-driven"
    elif below_invp:
        verdict = "FALSEADA"
        reason = ("IC(q) excludes 0 but lies below 1/p_sweep: cost-driven "
                  "mechanism falsified; coordination failure contributes")
    else:
        verdict = "FALSEADA"
        reason = "IC(q) excludes 0 and lies above 1/p_sweep"

    # persist per-point bootstrap N* samples for downstream regime re-analysis
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "nstar_boot.npz",
             E_vals=E_vals, nstar_point=nstar_pt,
             interior=interior, nstar_boot=np.array(nstar_boot))

    result = {
        "label": label,
        "lambda": lam,
        "e_center": e_center,
        "n_peak_calib": n_peak,
        "p_global_for_range": p_global,
        "E_vals": E_vals.tolist(),
        "nstar_point": nstar_pt.tolist(),
        "nstar_ci95": [list(c) for c in nstar_ci],
        "interior": interior.tolist(),
        "q_point": q_point,
        "q_ci95": [q_lo, q_hi],
        "p_sweep": p_sweep,
        "one_over_p_sweep": inv_p,
        "loglin_cost_[3,18]": {"slope": ll_cost.slope, "r2": ll_cost.r2,
                               "curvature": ll_cost.curvature,
                               "concave": ll_cost.concave, "note": ll_cost.note},
        "loglin_nstar_vs_E": (None if ll_nstar is None else
                              {"slope": ll_nstar.slope, "r2": ll_nstar.r2,
                               "curvature": ll_nstar.curvature,
                               "concave": ll_nstar.concave, "note": ll_nstar.note}),
        "guards": guards,
        "guards_ok": guards_ok,
        "verdict": verdict,
        "reason": reason,
    }
    write_outputs(result, outdir)
    return result


def write_outputs(res: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    E = np.array(res["E_vals"])
    nstar = np.array(res["nstar_point"])
    ci = np.array(res["nstar_ci95"])
    interior = np.array(res["interior"], dtype=bool)

    # CSV
    with open(outdir / "sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["E", "N_star", "N_star_lo", "N_star_hi", "interior"])
        for i in range(len(E)):
            w.writerow([f"{E[i]:.4f}", f"{nstar[i]:.3f}",
                        f"{ci[i, 0]:.3f}", f"{ci[i, 1]:.3f}", interior[i]])

    # Nstar_vs_E.png
    fig, ax = plt.subplots(figsize=(7, 5))
    yerr = np.abs(ci.T - nstar)
    ax.errorbar(E[interior], nstar[interior], yerr=yerr[:, interior],
                fmt="o", capsize=3, label="interior (used)")
    if (~interior).any():
        ax.errorbar(E[~interior], nstar[~interior], yerr=yerr[:, ~interior],
                    fmt="x", color="gray", alpha=0.6, label="edge (excluded)")
    ax.axhspan(ACCEPT_BAND[0], ACCEPT_BAND[1], color="green", alpha=0.06)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("E_protocol / (k_B T)"); ax.set_ylabel("N*")
    ax.set_title(f"P1 {res['label']}: N*(E)   q={res['q_point']:.3f} "
                 f"[{res['q_ci95'][0]:.3f}, {res['q_ci95'][1]:.3f}]")
    ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "Nstar_vs_E.png", dpi=130)
    plt.close(fig)

    # q_fit.png : log-log with fitted slope + 1/p_sweep reference
    fig, ax = plt.subplots(figsize=(7, 5))
    use = interior & (nstar > 0)
    lx = np.log(E[use]); ly = np.log(nstar[use])
    ax.plot(lx, ly, "o", label="N* (interior)")
    xs = np.linspace(lx.min(), lx.max(), 50)
    q, b = np.polyfit(lx, ly, 1)
    ax.plot(xs, q * xs + b, "-", label=f"fit q={q:.3f}")
    # reference line of slope 1/p_sweep anchored at the midpoint
    invp = res["one_over_p_sweep"]
    mid_x, mid_y = lx.mean(), ly.mean()
    ax.plot(xs, invp * (xs - mid_x) + mid_y, "--",
            label=f"slope 1/p_sweep={invp:.3f}")
    ax.set_xlabel("log E/kT"); ax.set_ylabel("log N*")
    ax.set_title(f"{res['label']}: q vs 1/p_sweep  ->  {res['verdict']}")
    ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "q_fit.png", dpi=130)
    plt.close(fig)

    with open(outdir / "verdict.json", "w") as f:
        json.dump(res, f, indent=2)


def _load_calibration() -> tuple[float, float]:
    """Return (e_center, n_peak) from the frozen calibration outputs."""
    j = json.loads((_ROOT / "results" / "calibration" / "chosen_lambda.json")
                   .read_text())
    return float(j["e_protocol_center"]), float(j["n_peak_at_chosen"])


def _load_p_linear() -> float:
    j = json.loads((_ROOT / "results" / "diagnostics" / "cost_decomposition.json")
                   .read_text())
    return float(j["linear_demand"]["point"]["p_measured"])


def main(n_seeds: int = 30) -> int:
    e_center, n_peak = _load_calibration()
    p_global = _load_p_linear()
    outdir = _ROOT / "results" / "q_sweep_linear"
    res = run_q_sweep(
        demand=meanings_for_N, e_center=e_center, n_peak=n_peak,
        p_global=p_global, lam=1.0, outdir=outdir, n_seeds=n_seeds,
        label="linear",
    )
    print("\n=== P1 q-sweep (PRIMARY, linear demand) ===")
    print(f"E range [{res['E_vals'][0]:.2f}, {res['E_vals'][-1]:.2f}]  "
          f"({len(res['E_vals'])} pts), n_seeds={n_seeds}")
    print(f"q = {res['q_point']:.4f}  CI95 [{res['q_ci95'][0]:.4f}, "
          f"{res['q_ci95'][1]:.4f}]")
    print(f"p_sweep[3,18] = {res['p_sweep']:.4f}  ->  1/p = "
          f"{res['one_over_p_sweep']:.4f}")
    print(f"log-lin cost[3,18]: {res['loglin_cost_[3,18]']['note']} "
          f"(R^2={res['loglin_cost_[3,18]']['r2']:.4f}, "
          f"curv={res['loglin_cost_[3,18]']['curvature']:.3f})")
    if res["loglin_nstar_vs_E"]:
        print(f"log-lin N*vsE:      {res['loglin_nstar_vs_E']['note']} "
              f"(R^2={res['loglin_nstar_vs_E']['r2']:.4f}, "
              f"curv={res['loglin_nstar_vs_E']['curvature']:.3f})")
    for k, g in res["guards"].items():
        print(f"guard {k:30s} value={g['value']:.3f} thr={g['threshold']} "
              f"{'OK' if g['ok'] else 'FAIL'}")
    print(f"\nVERDICT: {res['verdict']}  -- {res['reason']}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=30)
    args = ap.parse_args()
    raise SystemExit(main(n_seeds=args.n_seeds))
