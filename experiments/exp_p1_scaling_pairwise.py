"""
experiments/exp_p1_scaling_pairwise.py
======================================
B1 -- P1 q-sweep for the PAIRWISE (M~N^2) model, the paper's literal task.
Regime-resolved from the start, with two improvements over the noisy linear run:

  Mejora 1 -- N* by TWO estimators, cross-checked per point:
      (i)  chi(N)=Var_seeds[phi] peak (parabolic, sub-integer);
      (ii) phi(N)=0.5 crossing of the order parameter (linear interp).
    We report both and pick the smoother (lower point-to-point roughness) as
    primary, the other as control.

  Mejora 2 -- E_protocol dominance check R(E) = E_protocol / digital_Landauer(N*).
      R >> 1  : the fixed toolkit dominates -> if q_high < 1/p_high it is a
                COORDINATION contribution.
      R ~ O(1): the digital code's own per-message Landauer term is comparable to
                E_protocol -> q < 1/p is expected from the digital cost itself,
                NOT coordination. Distinguished explicitly in the verdict.

Objective: adjudicate the paper's internal inconsistency. With N^6 (p~6) the
dominant balance predicts q ~= 1/6 ~= 0.167, NOT eq.(15)'s 1/4 = 0.25. We report
q_high vs {0.167, 0.25} and set implied_sixth_supported / paper_eq15_quarter_supported.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "reference"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from edh import LN2  # noqa: E402
from edh.entropy import digital_message_length  # noqa: E402
from edh.finite_size import peak_report_from_chi, susceptibility  # noqa: E402
from edh.stats import (  # noqa: E402
    bootstrap_nstar,
    ci95,
    loglinearity,
    phi_half_crossing,
    two_regime_fit,
)
from edh.track_a_signaling import (  # noqa: E402
    TrackAParams,
    analog_cost_law,
    meanings_pairwise,
    run_population,
)

OUTDIR = _ROOT / "results" / "q_sweep_pairwise"
ACCEPT_BAND = (3.0, 18.0)        # for chi peak_report interior flag (diagnostic)
N_GRID = np.arange(2, 27)        # widened to 26 so N*~18-20 is not edge-censored
# Usable band for the q fit: grid-relative (N* resolved away from the grid edges),
# not the strict 18 -- the E-range targets N* up to ~19 and the wider grid
# resolves it cleanly.
USABLE_BAND = (3.0, float(N_GRID.max()) - 3.0)   # (3, 23)
E_HI_TARGET = 19.0               # derive E_hi so N* reaches ~19 (margin over 18)
LAM = 1.0
ALPHABET = TrackAParams().alphabet
T = 1.0


def _locate_chi(n_grid, phi):
    return peak_report_from_chi(n_grid, susceptibility(phi),
                                accept_band=ACCEPT_BAND).n_star


def _locate_cross(n_grid, phi):
    return phi_half_crossing(n_grid, phi.mean(axis=1), 0.5)


def roughness(logn: np.ndarray) -> float:
    good = np.isfinite(logn)
    if good.sum() < 3:
        return np.inf
    return float(np.mean(np.abs(np.diff(np.diff(logn[good])))))


def digital_landauer(n_star: float) -> float:
    n = int(round(n_star))
    m = meanings_pairwise(n)
    L = digital_message_length(m, ALPHABET)
    return L * T * LN2


def main(n_seeds: int = 50, n_points: int = 13) -> int:
    from tqdm import tqdm

    calib = json.loads((_ROOT / "results" / "calibration_pairwise"
                        / "chosen_lambda.json").read_text())
    e_center = float(calib["e_protocol_center"])
    n_peak = float(calib["n_peak_at_chosen"])
    decomp = json.loads((_ROOT / "results" / "diagnostics"
                         / "cost_decomposition.json").read_text())
    p_pw = float(decomp["pairwise_demand"]["point"]["p_measured"])

    e_lo = e_center * (ACCEPT_BAND[0] / n_peak) ** p_pw
    e_hi = e_center * (E_HI_TARGET / n_peak) ** p_pw
    E_vals = np.logspace(np.log10(e_lo), np.log10(e_hi), n_points)
    params = TrackAParams(temperature=T, e_center=e_center)

    nstar_chi, nstar_cross, interior, R_vals = [], [], [], []
    phi_store = []
    for E in tqdm(E_vals, desc="pairwise q-sweep"):
        phi = np.zeros((len(N_GRID), n_seeds))
        for ni, n in enumerate(N_GRID):
            for s in range(n_seeds):
                phi[ni, s] = run_population(int(n), float(E), LAM, params,
                                            seed=s, demand=meanings_pairwise).phi
        phi_store.append(phi)
        rep = peak_report_from_chi(N_GRID.astype(float), susceptibility(phi),
                                   accept_band=ACCEPT_BAND)
        nstar_chi.append(rep.n_star)
        nstar_cross.append(_locate_cross(N_GRID.astype(float), phi))
        interior.append(rep.interior)

    nstar_chi = np.array(nstar_chi)
    nstar_cross = np.array(nstar_cross)
    interior = np.array(interior, dtype=bool)   # chi peak_report flag (diagnostic)

    def usable(arr):
        return np.isfinite(arr) & (arr >= USABLE_BAND[0]) & (arr <= USABLE_BAND[1])

    usable_chi, usable_cross = usable(nstar_chi), usable(nstar_cross)

    # --- Mejora 1: pick the smoother estimator as primary ---
    rough_chi = roughness(np.log(np.where(usable_chi, nstar_chi, np.nan)))
    rough_cross = roughness(np.log(np.where(usable_cross, nstar_cross, np.nan)))
    use_cross = rough_cross < rough_chi
    primary_name = "phi=0.5 crossing" if use_cross else "chi-peak"
    nstar = nstar_cross if use_cross else nstar_chi
    use_mask = usable_cross if use_cross else usable_chi

    # per-point bootstrap of the primary estimator
    locate = _locate_cross if use_cross else _locate_chi
    boot = [bootstrap_nstar(N_GRID.astype(float), phi_store[i], locate,
                            n_boot=1000, seed=0)
            for i in range(len(E_vals))]

    # R(E) dominance using the primary N*
    R_vals = np.array([float(E_vals[i] / digital_landauer(nstar[i]))
                       if use_mask[i] else np.nan for i in range(len(E_vals))])

    # --- regime fit on the primary estimator, interior points only ---
    idx = np.where(use_mask)[0]
    reg = two_regime_fit(E_vals[idx], nstar[idx],
                         boot=[boot[i] for i in idx], n_ceil=float(N_GRID.max()))
    plateau = np.array(reg["plateau_mask"], bool)
    rise = np.array(reg["rise_mask"], bool)
    E_used = E_vals[idx]
    nstar_used = nstar[idx]

    # --- p_high_pw over the N range the rise visits (pairwise demand) ---
    rise_N = nstar_used[rise]
    n_lo = int(max(2, np.floor(rise_N.min()))); n_hi = int(np.ceil(rise_N.max()))
    n_rng = np.arange(n_lo, n_hi + 1)
    cost = np.array([analog_cost_law(int(n), params, demand=meanings_pairwise)
                     for n in n_rng])
    ll_cost = loglinearity(np.log(n_rng), np.log(cost))
    p_high = ll_cost.slope
    inv_p_high = 1.0 / p_high

    q_high = reg["q_high"]; qh_ci = reg["q_high_ci95"]
    q_low = reg["q_low"]; ql_ci = reg["q_low_ci95"]

    # --- guards ---
    logn = np.log10(nstar_used[rise]) if rise.any() else np.array([0.0])
    logn_range = float(logn.max() - logn.min())
    transition_frac = float(use_mask.mean())
    qh_hw_rel = (abs(qh_ci[1] - qh_ci[0]) / 2.0) / abs(q_high) \
        if q_high not in (0.0,) and np.isfinite(q_high) else np.inf
    guards = {
        "min_logN_star_range": {"value": logn_range, "thr": 0.4, "ok": logn_range >= 0.4},
        "require_transition_fraction": {"value": transition_frac, "thr": 0.7,
                                        "ok": transition_frac >= 0.7},
        "max_q_CI_halfwidth_rel": {"value": qh_hw_rel, "thr": 0.5, "ok": qh_hw_rel < 0.5},
    }
    guards_ok = all(g["ok"] for g in guards.values())

    # --- R dominance interpretation ---
    R_rise = R_vals[idx][rise]
    R_rise = R_rise[np.isfinite(R_rise)]
    R_med = float(np.median(R_rise)) if len(R_rise) else np.nan
    e_dominates = R_med > 10.0

    # --- mechanism test + adjudication ---
    qh_excl0 = (qh_ci[0] > 0) or (qh_ci[1] < 0)
    qh_overlaps_invp = qh_ci[0] <= inv_p_high <= qh_ci[1]
    qh_below_invp = qh_ci[1] < inv_p_high
    implied_sixth_supported = bool(qh_ci[0] <= 0.1667 <= qh_ci[1])
    quarter_supported = bool(qh_ci[0] <= 0.25 <= qh_ci[1])

    if not guards_ok:
        mech = "INCONCLUSO"; reason = "power guards failed"
    elif not qh_excl0:
        mech = "INCONCLUSO"; reason = "IC(q_high) includes 0"
    elif qh_overlaps_invp:
        mech = "VERIFICADO"; reason = "IC(q_high) overlaps 1/p_high and excludes 0"
    elif qh_below_invp and e_dominates:
        mech = "PARCIAL"; reason = ("q_high<1/p_high with R>>1 (E dominates): "
                                    "coordination contributes even at high E")
    elif qh_below_invp:
        mech = "AMBIGUO"; reason = ("q_high<1/p_high but R~O(1): digital's own "
                                    "Landauer cost can explain q<1/p (not coordination)")
    else:
        mech = "AMBIGUO"; reason = "q_high above 1/p_high"

    plateau_confirmed = (ql_ci[0] <= 0 <= ql_ci[1]) and (ql_ci[1] < inv_p_high)

    res = {
        "model": "pairwise (M~N^2)",
        "lambda_pairwise": LAM,
        "e_center": e_center, "n_peak_calib": n_peak, "p_pairwise_for_range": p_pw,
        "E_range": [float(e_lo), float(e_hi)], "E_decades": float(np.log10(e_hi/e_lo)),
        "primary_estimator": primary_name,
        "roughness_chi": rough_chi, "roughness_cross": rough_cross,
        "E_vals": E_vals.tolist(),
        "nstar_chi": nstar_chi.tolist(), "nstar_cross": nstar_cross.tolist(),
        "interior": interior.tolist(), "R_of_E": R_vals.tolist(),
        "E_knee": reg["E_knee"], "E_knee_bracket": reg["E_knee_bracket"],
        "n_low": reg["n_low"], "n_high": reg["n_high"],
        "q_low": q_low, "q_low_ci95": ql_ci,
        "q_high": q_high, "q_high_ci95": qh_ci,
        "p_high_pw": p_high, "one_over_p_high": inv_p_high,
        "p_high_N_range": [n_lo, n_hi],
        "loglin_cost_high": {"slope": ll_cost.slope, "r2": ll_cost.r2,
                             "curvature": ll_cost.curvature, "note": ll_cost.note},
        "R_median_rise": R_med, "E_protocol_dominates": bool(e_dominates),
        "guards": guards, "guards_ok": guards_ok,
        "plateau_confirmed": bool(plateau_confirmed),
        "mechanism_high_regime": mech, "mechanism_reason": reason,
        "implied_sixth_supported": implied_sixth_supported,
        "paper_eq15_quarter_supported": quarter_supported,
        "verdicts": {
            "H3_strong_single_driver": "FALSEADA" if plateau_confirmed else "INCONCLUSO",
            "cost_driven_high_regime": mech,
            "coordination_driven_low_regime": "CONFIRMADO" if plateau_confirmed else "INCONCLUSO",
        },
    }
    write_outputs(res, E_vals, nstar_chi, nstar_cross, interior, idx, plateau, rise,
                  nstar_used, inv_p_high, R_vals)

    print("\n=== P1 q-sweep PAIRWISE (M~N^2) ===")
    print(f"E range [{e_lo:.2f}, {e_hi:.1f}] ({res['E_decades']:.2f} decades), "
          f"{n_points} pts, n_seeds={n_seeds}")
    print(f"primary estimator: {primary_name} (roughness chi={rough_chi:.4f} "
          f"cross={rough_cross:.4f})")
    print(f"E_knee ~ {reg['E_knee']:.1f}  bracket {reg['E_knee_bracket']}")
    print(f"  q_low  = {q_low:+.3f}  CI95 [{ql_ci[0]:+.3f}, {ql_ci[1]:+.3f}]  (~0)")
    print(f"  q_high = {q_high:+.3f}  CI95 [{qh_ci[0]:+.3f}, {qh_ci[1]:+.3f}]")
    print(f"p_high_pw over N[{n_lo},{n_hi}] = {p_high:.3f} -> 1/p_high = {inv_p_high:.3f}")
    print(f"R(E) median (rise) = {R_med:.1f}  E_protocol_dominates={e_dominates}")
    for k, g in guards.items():
        print(f"guard {k:28s} {g['value']:.3f} thr={g['thr']} "
              f"{'OK' if g['ok'] else 'FAIL'}")
    print(f"\nMechanism (high regime): {mech} -- {reason}")
    print(f"Adjudication: q_high vs 1/6={0.1667} -> supported={implied_sixth_supported}; "
          f"vs 1/4={0.25} -> supported={quarter_supported}")
    print(f"H3-strong single-driver: {res['verdicts']['H3_strong_single_driver']}")
    return 0


def write_outputs(res, E, nchi, ncross, interior, idx, plateau, rise,
                  nstar_used, inv_p_high, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / "sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["E", "nstar_chi", "nstar_cross", "interior", "R_of_E"])
        for i in range(len(E)):
            w.writerow([f"{E[i]:.4f}", f"{nchi[i]:.3f}", f"{ncross[i]:.3f}",
                        interior[i], f"{R[i]:.3f}"])
    with open(OUTDIR / "verdict.json", "w") as f:
        json.dump(res, f, indent=2)

    # Dual-estimator N*(E)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(E, nchi, "o", label="N* chi-peak", alpha=0.7)
    ax.plot(E, ncross, "x", label="N* phi=0.5 crossing", alpha=0.7)
    ax.axhspan(ACCEPT_BAND[0], ACCEPT_BAND[1], color="green", alpha=0.06)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("E_protocol / kT"); ax.set_ylabel("N*")
    ax.set_title(f"P1 pairwise: dual N* estimators (primary: {res['primary_estimator']})")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "Nstar_vs_E.png", dpi=130); plt.close(fig)

    # Regime fit on primary estimator
    Eu = E[idx]; x = np.log(Eu); y = np.log(nstar_used)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x[plateau], y[plateau], "o", color="C0", label="plateau")
    ax.plot(x[rise], y[rise], "s", color="C3", label="rise")

    def seg(mask, color, lbl):
        if mask.sum() < 2:
            return
        s, b = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 30)
        ax.plot(xs, s * xs + b, "-", color=color, label=lbl)
        return s, b
    seg(plateau, "C0", f"q_low={res['q_low']:+.3f}")
    seg(rise, "C3", f"q_high={res['q_high']:+.3f}")
    if rise.sum() >= 2:
        xs = np.linspace(x[rise].min(), x[rise].max(), 30)
        s, b = np.polyfit(x[rise], y[rise], 1)
        mx = xs.mean(); my = s*mx+b
        ax.plot(xs, inv_p_high*(xs-mx)+my, "--", color="green",
                label=f"1/p_high={inv_p_high:.3f}")
        # reference 1/6 and 1/4 slopes
        for val, c in ((0.1667, "purple"), (0.25, "orange")):
            ax.plot(xs, val*(xs-mx)+my, ":", color=c, alpha=0.7,
                    label=f"slope {val}")
    ax.set_xlabel("log E/kT"); ax.set_ylabel("log N*")
    ax.set_title(f"P1 pairwise two-regime (cost-driven: {res['mechanism_high_regime']})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "regimes.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--n-points", type=int, default=13)
    args = ap.parse_args()
    raise SystemExit(main(n_seeds=args.n_seeds, n_points=args.n_points))
