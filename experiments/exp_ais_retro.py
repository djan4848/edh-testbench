"""
experiments/exp_ais_retro.py
============================
Retrospective Active Information Storage (AIS) analysis on the q-sweep dynamics.
The sweeps saved only phi/N*; AIS needs states, so we cheaply RE-GENERATE the
stochastic ensembles at a subset of (E, N) points per regime and model, recording
the per-agent digitality trajectory, and estimate AIS(N) = I(x_t; x_{t-1..k}).

Hypothesis to TEST (not assume): the analog->digital transition is a dip-recovery
of AIS, and AIS is a MORE universal order parameter than digitality -- same
signature across regimes (coordination vs cost) and across models (linear vs
pairwise). "AIS moves" proves nothing; the coincidence of the signature across
regimes and models is the test.

We report THREE metrics per profile because the raw MI conflates marginal entropy
with temporal structure:
  * ais        = I(x_t; history)  [bits]
  * h_target   = H(x_t)           [bits]  (cross-sectional spread)
  * normalized = ais / h_target           (self-predictability fraction)
The "loss of self-predictability" reading maps best onto the NORMALIZED metric.

Outputs: results/ais/{ais_vs_N_linear.png, ais_vs_N_pairwise.png, ais_profile.csv}
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

from edh.ais import ais_from_traces  # noqa: E402
from edh.track_a_signaling import (  # noqa: E402
    TrackAParams,
    meanings_for_N,
    meanings_pairwise,
    run_population,
)

OUTDIR = _ROOT / "results" / "ais"
N_GRID = np.arange(2, 25)
S_SEEDS = 40
Q = 4
TAIL = 40
LAGS = (1, 2, 3)


def _nearest_nstar(E_target, E_arr, nstar_arr):
    i = int(np.argmin(np.abs(np.log(E_arr) - np.log(E_target))))
    return float(nstar_arr[i])


def select_points(model: str):
    """Return [(E, regime, N_star)] -- 2 plateau + 2 rise per model -- and the
    (demand, params) for re-generation, matching each sweep's settings."""
    if model == "linear":
        rows = list(csv.DictReader(
            open(_ROOT / "results" / "q_sweep_linear" / "sweep.csv")))
        E = np.array([float(r["E"]) for r in rows])
        nstar = np.array([float(r["N_star"]) for r in rows])
        knee = json.loads((_ROOT / "results" / "q_sweep_linear"
                           / "regime_verdict.json").read_text())["E_knee"]
        demand, params = meanings_for_N, TrackAParams(temperature=1.0, e_center=40.0)
    else:
        v = json.loads((_ROOT / "results" / "q_sweep_pairwise"
                        / "verdict.json").read_text())
        E = np.array(v["E_vals"])
        nstar = np.array(v["nstar_cross"])
        knee = v["E_knee"]
        demand = meanings_pairwise
        params = TrackAParams(temperature=1.0, e_center=float(v["e_center"]))

    plateau = E < knee
    rise = E >= knee
    pe, re = E[plateau], E[rise]
    pn, rn = nstar[plateau], nstar[rise]
    pick = []
    for arr_e, arr_n, reg in ((pe, pn, "plateau"), (re, rn, "rise")):
        if len(arr_e) == 0:
            continue
        idxs = [0, len(arr_e) // 2] if len(arr_e) >= 2 else [0]
        # for the rise, prefer a mid and a high point
        if reg == "rise" and len(arr_e) >= 3:
            idxs = [len(arr_e) // 3, (2 * len(arr_e)) // 3]
        for i in sorted(set(idxs)):
            pick.append((float(arr_e[i]), reg, float(arr_n[i])))
    return pick, demand, params


def ais_profile(E, demand, params):
    """AIS(N) over N_GRID at fixed E. Returns dict of arrays per lag + h + norm."""
    out = {f"ais{lag}": [] for lag in LAGS}
    out.update({"h": [], "norm1": [], "adeq": [], "ok": []})
    for n in N_GRID:
        stack = np.stack([
            run_population(int(n), float(E), 1.0, params, seed=s, demand=demand,
                           record_trace=True).d_traj
            for s in range(S_SEEDS)
        ])  # (S, n_steps, n)
        for lag in LAGS:
            r = ais_from_traces(stack, q=Q, tail=TAIL, lag=lag, seed=0)
            out[f"ais{lag}"].append(r.ais if r.ok else np.nan)
            if lag == 1:
                out["h"].append(r.h_target if r.ok else np.nan)
                out["norm1"].append(r.normalized if r.ok else np.nan)
                out["adeq"].append(r.adequacy)
                out["ok"].append(r.ok)
    return {k: np.array(v) for k, v in out.items()}


def dip_recovery(nstar, ais):
    """Characterise dip-recovery: location of AIS min, distance to N*, recovery."""
    good = np.isfinite(ais)
    if good.sum() < 4:
        return {"n_dip": np.nan, "dist_to_nstar": np.nan, "recovery": np.nan,
                "dip_near_nstar": False}
    ng = N_GRID[good]; ag = ais[good]
    i = int(np.argmin(ag))
    n_dip = float(ng[i])
    recovery = float(ag[-1] - ag[i])   # rise from the dip to the digital end
    dist = abs(n_dip - nstar)
    return {"n_dip": n_dip, "dist_to_nstar": float(dist), "recovery": recovery,
            "dip_near_nstar": bool(dist <= 3.0 and recovery > 0.02)}


def main() -> int:
    from tqdm import tqdm
    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = {}
    for model in ("linear", "pairwise"):
        picks, demand, params = select_points(model)
        profiles = []
        for E, reg, nstar in tqdm(picks, desc=f"AIS {model}"):
            prof = ais_profile(E, demand, params)
            profiles.append((E, reg, nstar, prof))
            dr = dip_recovery(nstar, prof["ais1"])
            drn = dip_recovery(nstar, prof["norm1"])
            summary[f"{model}|E={E:.3g}|{reg}"] = {
                "N_star": nstar, "raw_ais": dr, "normalized": drn}
            for k, n in enumerate(N_GRID):
                all_rows.append([model, f"{E:.4g}", reg, nstar, int(n),
                                 prof["ais1"][k], prof["ais2"][k], prof["ais3"][k],
                                 prof["h"][k], prof["norm1"][k], prof["adeq"][k]])
        plot_model(model, profiles)

    with open(OUTDIR / "ais_profile.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "E", "regime", "N_star", "N", "ais_lag1", "ais_lag2",
                    "ais_lag3", "h_target", "normalized", "adequacy"])
        w.writerows(all_rows)

    verdict = universality_verdict(summary)
    with open(OUTDIR / "ais_universality.json", "w") as f:
        json.dump({"profiles": summary, "verdict": verdict}, f, indent=2)

    print("\n=== AIS retrospective ===")
    for k, v in summary.items():
        dr, drn = v["raw_ais"], v["normalized"]
        print(f"{k:38s} N*={v['N_star']:5.2f}  raw: n_dip={dr['n_dip']!s:>5} "
              f"near={dr['dip_near_nstar']!s:5}  norm: n_dip={drn['n_dip']!s:>5} "
              f"near={drn['dip_near_nstar']!s:5}")
    print("\nUNIVERSALITY VERDICT:")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    return 0


def universality_verdict(summary: dict) -> dict:
    def frac(metric):
        vals = [v[metric]["dip_near_nstar"] for v in summary.values()]
        return sum(vals) / len(vals) if vals else 0.0
    raw_frac = frac("raw_ais"); norm_frac = frac("normalized")
    # within/between model consistency on the normalized metric
    by_model = {}
    for k, v in summary.items():
        m = k.split("|")[0]
        by_model.setdefault(m, []).append(v["normalized"]["dip_near_nstar"])
    within = {m: (sum(b) / len(b)) for m, b in by_model.items()}
    return {
        "raw_ais_dip_fraction": raw_frac,
        "normalized_dip_fraction": norm_frac,
        "within_model_consistency_normalized": within,
        "universal_normalized": bool(norm_frac >= 0.75
                                     and all(w >= 0.5 for w in within.values())),
        "note": ("universal = dip-near-N* holds in >=75% of profiles AND in both "
                 "models; raw AIS conflates spread, so the normalized metric is "
                 "the primary read."),
    }


def plot_model(model, profiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for E, reg, nstar, prof in profiles:
        c = "C0" if reg == "plateau" else "C3"
        ls = "-" if reg == "plateau" else "--"
        lbl = f"E={E:.3g} ({reg}) N*={nstar:.1f}"
        axes[0].plot(N_GRID, prof["ais1"], ls, color=c, marker="o", ms=3, label=lbl)
        axes[0].axvline(nstar, color=c, alpha=0.25, lw=1)
        axes[1].plot(N_GRID, prof["norm1"], ls, color=c, marker="o", ms=3, label=lbl)
        axes[1].axvline(nstar, color=c, alpha=0.25, lw=1)
    axes[0].set_title(f"{model}: raw AIS = I(x_t;x_t-1) [bits]")
    axes[1].set_title(f"{model}: normalized AIS = MI/H (self-predictability)")
    for ax in axes:
        ax.set_xlabel("N"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("bits")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"ais_vs_N_{model}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
