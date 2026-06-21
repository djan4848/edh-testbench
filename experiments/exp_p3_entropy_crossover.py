"""
experiments/exp_p3_entropy_crossover.py
=======================================
P3 -- entropy crossover (close-out of Track A). NO new sweeps: uses the
deterministic dissipation bookkeeping (edh.entropy) already exercised by the
q-sweeps, plus the measured N*(E) from those sweeps.

For each model (linear, pairwise) and each swept E:
  D_analog(N)  = analog dissipation rate (E-independent)           [bookkeeping]
  D_digital(N) = digital dissipation rate = E_protocol + L*T*ln2   [bookkeeping]
  N_cross(E)   = first N where D_analog(N) > D_digital(N)  (sub-integer interp)
Compare N_cross(E) to N*(E) (q-sweep dual estimators). Report N_cross - N* per
point, resolved by regime.

Pre-registered P3 rule, resolved by regime:
  * COST regime (high E): N_cross ~= N* (+-1) -> the entropy crossover IS the
    transition -> P3 VERIFICADA there.
  * COORDINATION regime (low E): N_cross != N* -> P3 (literal) regime-limited.
We report the actual SIGN of N_cross - N* (do not force the expected direction).
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

from edh.entropy import analog_dissipation, digital_dissipation  # noqa: E402
from edh.track_a_signaling import TrackAParams, meanings_for_N, meanings_pairwise  # noqa: E402

OUTDIR = _ROOT / "results" / "p3_entropy_crossover"
N_GRID = np.arange(2, 25)


def n_cross(d_analog: np.ndarray, d_digital: np.ndarray, ngrid: np.ndarray) -> float:
    """First N where D_analog > D_digital, sub-integer by linear interpolation of
    the gap g=D_analog-D_digital crossing zero. NaN if it never crosses in-range;
    returns the low edge if analog already exceeds digital at the smallest N."""
    g = d_analog - d_digital
    if g[0] > 0:
        return float(ngrid[0])      # analog already more expensive at N=2
    for i in range(len(g) - 1):
        if g[i] <= 0 < g[i + 1]:
            f = -g[i] / (g[i + 1] - g[i])
            return float(ngrid[i] + f * (ngrid[i + 1] - ngrid[i]))
    return float("nan")


def load_model(model: str):
    """Return (E_vals, nstar, interior_mask, knee, demand, params, nstar_label)."""
    params = TrackAParams()
    if model == "linear":
        rows = list(csv.DictReader(
            open(_ROOT / "results" / "q_sweep_linear" / "sweep.csv")))
        E = np.array([float(r["E"]) for r in rows])
        nstar = np.array([float(r["N_star"]) for r in rows])
        interior = np.array([r["interior"].strip().lower() == "true" for r in rows])
        knee = json.loads((_ROOT / "results" / "q_sweep_linear"
                           / "regime_verdict.json").read_text())["E_knee"]
        return E, nstar, interior, knee, meanings_for_N, params, "chi-peak"
    v = json.loads((_ROOT / "results" / "q_sweep_pairwise" / "verdict.json").read_text())
    E = np.array(v["E_vals"])
    nstar = np.array(v["nstar_cross"])
    interior = np.array(v["interior"], dtype=bool)
    return E, nstar, interior, v["E_knee"], meanings_pairwise, params, "phi=0.5 crossing"


def analyze(model: str) -> dict:
    E, nstar, interior, knee, demand, params, label = load_model(model)
    Ms = np.array([demand(int(n)) for n in N_GRID])
    d_analog = np.array([
        analog_dissipation(int(m), params.sep, params.sigma, params.temperature,
                           params.stiffness).total for m in Ms])

    rows = []
    for k, e in enumerate(E):
        d_digital = np.array([
            digital_dissipation(int(m), params.alphabet, float(e),
                                params.temperature).total for m in Ms])
        nc = n_cross(d_analog, d_digital, N_GRID.astype(float))
        regime = "rise" if e >= knee else "plateau"
        diff = (nc - nstar[k]) if (np.isfinite(nc) and interior[k]) else np.nan
        rows.append({"E": float(e), "regime": regime, "n_cross": nc,
                     "n_star": float(nstar[k]), "interior": bool(interior[k]),
                     "diff": diff, "coincide_pm1": bool(np.isfinite(diff)
                                                        and abs(diff) <= 1.0)})

    def regime_stats(reg):
        d = [r["diff"] for r in rows if r["regime"] == reg and np.isfinite(r["diff"])]
        if not d:
            return {"n": 0}
        d = np.array(d)
        return {"n": len(d), "median_diff": float(np.median(d)),
                "frac_coincide_pm1": float(np.mean(np.abs(d) <= 1.0)),
                "sign": ("N_cross>N* (crossover after transition)" if np.median(d) > 0
                         else "N_cross<N* (crossover precedes transition)")}

    cost = regime_stats("rise")
    coord = regime_stats("plateau")
    p3_cost = ("VERIFICADA" if cost.get("frac_coincide_pm1", 0) >= 0.7
               else "FALSEADA/INCONCLUSO")
    p3_coord = ("VERIFICADA(inesperado)" if coord.get("frac_coincide_pm1", 0) >= 0.7
                else "FALSEADA/regime-limited")
    return {"model": model, "nstar_label": label, "E_knee": knee,
            "rows": rows, "cost_regime": cost, "coordination_regime": coord,
            "P3_cost_regime": p3_cost, "P3_coordination_regime": p3_coord,
            "d_analog": d_analog.tolist()}


def plot(model: str, res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["rows"]
    E = np.array([r["E"] for r in rows])
    nc = np.array([r["n_cross"] for r in rows])
    ns = np.array([r["n_star"] for r in rows])
    interior = np.array([r["interior"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(E, nc, "o-", color="C2", label="N_cross (entropy crossover)")
    ax.plot(E[interior], ns[interior], "s--", color="C0", label="N* (transition)")
    ax.axvline(res["E_knee"], color="gray", ls=":", label=f"E_knee~{res['E_knee']:.0f}")
    ax.set_xscale("log")
    ax.set_xlabel("E_protocol / kT"); ax.set_ylabel("N")
    ax.set_title(f"P3 {model}: N_cross vs N*  "
                 f"(cost {res['P3_cost_regime']})")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / f"Ncross_vs_E_{model}.png", dpi=130)
    plt.close(fig)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = {}
    all_rows = []
    for model in ("linear", "pairwise"):
        res = analyze(model)
        plot(model, res)
        results[model] = {k: v for k, v in res.items() if k != "rows"}
        for r in res["rows"]:
            all_rows.append([model, f"{r['E']:.4g}", r["regime"], f"{r['n_cross']:.3f}",
                             f"{r['n_star']:.3f}", r["interior"],
                             (f"{r['diff']:.3f}" if np.isfinite(r['diff']) else "nan"),
                             r["coincide_pm1"]])

    with open(OUTDIR / "crossover.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "E", "regime", "N_cross", "N_star", "interior",
                    "N_cross_minus_N_star", "coincide_pm1"])
        w.writerows(all_rows)
    with open(OUTDIR / "verdict.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== P3 entropy crossover ===")
    for model, res in results.items():
        print(f"\n[{model}]  (N* = {res['nstar_label']}, E_knee~{res['E_knee']:.1f})")
        c, co = res["cost_regime"], res["coordination_regime"]
        if c.get("n"):
            print(f"  COST regime (rise):   n={c['n']} median(N_cross-N*)="
                  f"{c['median_diff']:+.2f} coincide(+-1)={c['frac_coincide_pm1']:.2f}"
                  f"  [{c['sign']}]")
        if co.get("n"):
            print(f"  COORD regime (plateau): n={co['n']} median(N_cross-N*)="
                  f"{co['median_diff']:+.2f} coincide(+-1)={co['frac_coincide_pm1']:.2f}"
                  f"  [{co['sign']}]")
        print(f"  P3 cost regime: {res['P3_cost_regime']}  |  "
              f"P3 coordination regime: {res['P3_coordination_regime']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
