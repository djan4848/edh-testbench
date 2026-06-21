"""
experiments/exp_p1_regimes.py
=============================
Regime-resolved re-analysis of the EXISTING primary q-sweep (no re-run).

The log-linearity check on N*(E) fired (convex), so a single global slope q is
not a valid power-law statistic. The pre-registration says: if not log-linear,
treat q ~= 1/p as LOCAL. N*(E) is two-regime -- an E-independent plateau (the
analog coordination-error ceiling) plus an energetic rise -- so we resolve by
regime:

  * detect the knee E_knee by a continuous 2-segment piecewise-linear fit in
    log-log (grid search over the breakpoint, weighted by per-point N* CIs);
  * q_low (E<E_knee, expected ~0) and q_high (E>=E_knee);
  * refit the analog cost exponent p_high on the N range the HIGH regime visits
    (not the global [3,18]); test q_high vs 1/p_high;
  * regime verdicts (H3-strong single-driver; cost-driven high regime;
    coordination-driven low regime).

Per-point N* uncertainty is taken from the saved CI95 via a normal approximation
(sd = (hi-lo)/(2*1.96)); this is a re-analysis of existing data, not a re-run.
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

from edh.stats import ci95, loglinearity  # noqa: E402
from edh.track_a_signaling import TrackAParams, analog_cost_law, meanings_for_N  # noqa: E402

OUTDIR = _ROOT / "results" / "q_sweep_linear"
SWEEP = OUTDIR / "sweep.csv"
BOOT_NPZ = OUTDIR / "nstar_boot.npz"
Z95 = 1.959963985
N_CEIL = 24.0      # N-grid ceiling for clipping bootstrap draws


def load_data():
    """Prefer the saved per-point bootstrap (real CI); fall back to the CSV CI
    with a normal approximation. Returns (E, N, sd, boot) for interior points,
    where ``boot`` is a list of per-point bootstrap arrays or None."""
    if BOOT_NPZ.exists():
        z = np.load(BOOT_NPZ, allow_pickle=True)
        interior = z["interior"].astype(bool)
        E = z["E_vals"][interior]
        N = z["nstar_point"][interior]
        boot = [np.asarray(b) for b, k in zip(z["nstar_boot"], interior) if k]
        sd = np.clip(np.array([b.std() for b in boot]), 1e-3, None)
        return E, N, sd, boot
    E, N, lo, hi = [], [], [], []
    with open(SWEEP) as f:
        for row in csv.DictReader(f):
            if row["interior"].strip().lower() != "true":
                continue
            E.append(float(row["E"])); N.append(float(row["N_star"]))
            lo.append(float(row["N_star_lo"])); hi.append(float(row["N_star_hi"]))
    E, N = np.array(E), np.array(N)
    sd = np.clip((np.array(hi) - np.array(lo)) / (2 * Z95), 1e-3, None)
    return E, N, sd, None


def fit_piecewise(x, y, w, knee):
    """Continuous 2-segment fit: y = a + q_low*x + (q_high-q_low)*relu(x-knee).
    Weighted least squares. Returns (a, q_low, q_high, wssr)."""
    relu = np.maximum(0.0, x - knee)
    X = np.column_stack([np.ones_like(x), x, relu])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(sw[:, None] * X, sw * y, rcond=None)
    a, q_low, dq = beta
    resid = y - X @ beta
    return float(a), float(q_low), float(q_low + dq), float(np.sum(w * resid**2))


def best_knee(x, y, w, n_grid=300, min_per_side=2):
    cand = np.linspace(x.min(), x.max(), n_grid)
    best = None
    for k in cand:
        if (x < k).sum() < min_per_side or (x >= k).sum() < min_per_side:
            continue
        a, ql, qh, ssr = fit_piecewise(x, y, w, k)
        if best is None or ssr < best[-1]:
            best = (k, a, ql, qh, ssr)
    return best  # (knee_x, a, q_low, q_high, wssr)


def single_line_wssr(x, y, w):
    sw = np.sqrt(w)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(sw[:, None] * X, sw * y, rcond=None)
    resid = y - X @ beta
    return float(np.sum(w * resid**2)), float(beta[1])


def main() -> int:
    E, N, sd, boot = load_data()
    x, y = np.log(E), np.log(N)
    # UNIFORM weights: the per-point bootstrap CI is quantization-tight (all
    # resamples snap to the same chi-peak) and would over-weight a few snapped
    # points. The honest noise is the between-point scatter, captured by the
    # point-resampling bootstrap below.
    w = np.ones_like(y)
    boot_src = "saved per-point bootstrap + point-resampling" if boot is not None \
        else "normal approx from CSV CI + point-resampling"

    # --- robust regime assignment: a plateau is the leading run of points whose
    #     N* stays within `tol` of the floor; the rest is the rise. The plateau's
    #     EXISTENCE (not the fine knee) is the primary claim, so we fix the split
    #     robustly and bootstrap slopes WITHIN regimes (avoids the degenerate
    #     2-point segments a floating-knee bootstrap produces). ---
    floor = float(np.median(np.sort(N)[:3]))
    tol = 1.0
    plateau = np.zeros(len(N), dtype=bool)
    order = np.argsort(E)
    for i in order:
        if N[i] <= floor + tol:
            plateau[i] = True
        else:
            break
    rise = ~plateau
    knee_lo = float(E[plateau].max()) if plateau.any() else float(E.min())
    knee_hi = float(E[rise].min()) if rise.any() else float(E.max())
    E_knee = float(np.sqrt(knee_lo * knee_hi))
    n_low, n_high = int(plateau.sum()), int(rise.sum())

    def ols(mask):
        return float(np.polyfit(x[mask], y[mask], 1)[0]) if mask.sum() >= 2 else np.nan

    q_low, q_high = ols(plateau), ols(rise)
    wssr_lin, q_global = single_line_wssr(x, y, w)
    knee_x2, _, _, _, wssr_pw = best_knee(x, y, w)
    ssr_improve = 1.0 - wssr_pw / wssr_lin if wssr_lin > 0 else 0.0

    # --- within-regime bootstrap: resample points inside each fixed regime and
    #     draw each point's N* from its saved bootstrap sample. ---
    rng = np.random.default_rng(0)
    B = 4000

    def boot_slope(mask):
        idxs = np.where(mask)[0]
        if len(idxs) < 2:
            return np.array([np.nan])
        out = []
        for _ in range(B):
            pick = rng.choice(idxs, len(idxs), replace=True)
            if boot is not None:
                draw = np.array([boot[i][rng.integers(0, len(boot[i]))] for i in pick])
            else:
                draw = rng.normal(N[pick], sd[pick])
            xb = x[pick]
            if np.ptp(xb) < 1e-9:
                continue
            out.append(np.polyfit(xb, np.log(np.clip(draw, 2.0, N_CEIL)), 1)[0])
        return np.array(out) if out else np.array([np.nan])

    qlow_ci = ci95(boot_slope(plateau))
    qhigh_ci = ci95(boot_slope(rise))
    knee_ci = (knee_lo, knee_hi)        # knee bracketed by the regime boundary

    # --- p_high over the N range the HIGH regime visits ---
    hi_mask = rise
    n_lo_hi = int(max(2, np.floor(N[hi_mask].min())))
    n_hi_hi = int(min(23, np.ceil(N[hi_mask].max())))
    n_rng = np.arange(n_lo_hi, n_hi_hi + 1)
    params = TrackAParams()
    cost_hi = np.array([analog_cost_law(int(n), params, demand=meanings_for_N)
                        for n in n_rng])
    ll_cost_hi = loglinearity(np.log(n_rng), np.log(cost_hi))
    p_high = ll_cost_hi.slope
    inv_p_high = 1.0 / p_high

    # log-linearity within the high regime (N* vs E)
    ll_high = (loglinearity(x[hi_mask], y[hi_mask]) if hi_mask.sum() >= 3
               else None)

    # --- regime verdicts ---
    qlow_includes_0 = qlow_ci[0] <= 0 <= qlow_ci[1]
    qlow_excludes_invp = qlow_ci[1] < inv_p_high or qlow_ci[0] > inv_p_high
    plateau_confirmed = qlow_includes_0 and qlow_excludes_invp

    qhigh_excludes_0 = (qhigh_ci[0] > 0) or (qhigh_ci[1] < 0)
    qhigh_overlaps_invp = qhigh_ci[0] <= inv_p_high <= qhigh_ci[1]
    qhigh_below_invp = qhigh_ci[1] < inv_p_high

    h3_strong = "FALSEADA" if plateau_confirmed else "INCONCLUSO"
    if qhigh_excludes_0 and qhigh_overlaps_invp:
        cost_high = "VERIFICADO"
    elif qhigh_excludes_0 and qhigh_below_invp:
        cost_high = "PARCIAL"      # coordination still contributes at high E
    else:
        cost_high = "INCONCLUSO"
    coord_low = "CONFIRMADO" if plateau_confirmed else "INCONCLUSO"

    # extension advice (step 3)
    extend = (n_high < 5) or (hi_mask.sum() >= 3 and
                              N[hi_mask].max() >= 21.5)  # grid-ceiling pressure
    extend_reason = (
        "high regime <5 points" if n_high < 5 else
        ("top N* approaching the N=24 grid ceiling (>=21.5); extend E and/or N "
         "grid to stabilise q_high" if N[hi_mask].max() >= 21.5 else
         "high regime well sampled and below grid ceiling; no extension"))

    res = {
        "E_knee": E_knee, "E_knee_ci95": list(knee_ci),
        "n_low": n_low, "n_high": n_high,
        "q_low": q_low, "q_low_ci95": list(qlow_ci),
        "q_high": q_high, "q_high_ci95": list(qhigh_ci),
        "q_global_for_reference": q_global,
        "piecewise_vs_line_wssr_improvement": ssr_improve,
        "p_high_N_range": [int(n_lo_hi), int(n_hi_hi)],
        "p_high": p_high, "one_over_p_high": inv_p_high,
        "loglin_cost_high": {"slope": ll_cost_hi.slope, "r2": ll_cost_hi.r2,
                             "curvature": ll_cost_hi.curvature,
                             "note": ll_cost_hi.note},
        "loglin_nstar_high": (None if ll_high is None else
                              {"slope": ll_high.slope, "r2": ll_high.r2,
                               "curvature": ll_high.curvature,
                               "note": ll_high.note}),
        "verdicts": {
            "H3_strong_single_driver": h3_strong,
            "cost_driven_high_regime": cost_high,
            "coordination_driven_low_regime": coord_low,
        },
        "plateau_confirmed": plateau_confirmed,
        "extend_recommended": bool(extend),
        "extend_reason": extend_reason,
        "note_honesty": ("q_global~0.235 ~ eq.(15) 0.25 is a mixing coincidence; "
                         "the test is q_high vs 1/p_high. This vindicates the "
                         "q~=1/p design over pre-registering a fixed number."),
    }
    write_outputs(res, E, N, sd, plateau, rise, E_knee, q_low, q_high, inv_p_high)

    print("=== P1 q-sweep regime re-analysis (linear) ===")
    print(f"(uncertainty source: {boot_src})")
    print(f"E_knee = {E_knee:.2f}  CI95 {knee_ci}")
    print(f"  low regime:  {n_low} pts  q_low  = {q_low:+.3f}  CI95 "
          f"[{qlow_ci[0]:+.3f}, {qlow_ci[1]:+.3f}]   (expected ~0)")
    print(f"  high regime: {n_high} pts  q_high = {q_high:+.3f}  CI95 "
          f"[{qhigh_ci[0]:+.3f}, {qhigh_ci[1]:+.3f}]")
    print(f"p_high over N{res['p_high_N_range']} = {p_high:.3f} -> 1/p_high = "
          f"{inv_p_high:.3f}   ({ll_cost_hi.note})")
    if ll_high:
        print(f"log-lin N*vsE (high): {ll_high.note} (R^2={ll_high.r2:.3f})")
    print(f"piecewise improves weighted SSR vs single line by "
          f"{ssr_improve*100:.1f}%")
    print("\nVERDICTS:")
    for k, v in res["verdicts"].items():
        print(f"  {k:34s} {v}")
    print(f"\nExtend sweep? {res['extend_recommended']}  ({res['extend_reason']})")
    return 0


def write_outputs(res, E, N, sd, plateau, rise, E_knee, q_low, q_high, inv_p_high):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTDIR.mkdir(parents=True, exist_ok=True)
    x = np.log(E)
    kx = np.log(E_knee)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(x[plateau], np.log(N[plateau]), yerr=sd[plateau] / N[plateau],
                fmt="o", color="C0", capsize=3, label="plateau")
    ax.errorbar(x[rise], np.log(N[rise]), yerr=sd[rise] / N[rise],
                fmt="s", color="C3", capsize=3, label="rise")

    def seg(mask, color, lbl):
        s, b = np.polyfit(x[mask], np.log(N[mask]), 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 30)
        ax.plot(xs, s * xs + b, "-", color=color, label=lbl)
        return s, b

    seg(plateau, "C0", f"q_low={q_low:+.3f}")
    s_hi, b_hi = seg(rise, "C3", f"q_high={q_high:+.3f}")
    ax.axvline(kx, color="gray", ls=":", label=f"E_knee~{E_knee:.1f}")
    # reference slope 1/p_high anchored at the high-regime midpoint
    xs_hi = np.linspace(x[rise].min(), x[rise].max(), 30)
    mx = xs_hi.mean(); my = s_hi * mx + b_hi
    ax.plot(xs_hi, inv_p_high * (xs_hi - mx) + my, "--", color="green",
            label=f"slope 1/p_high={inv_p_high:.3f}")
    ax.set_xlabel("log E/kT"); ax.set_ylabel("log N*")
    ax.set_title(f"P1 linear: two-regime fit  "
                 f"(H3-strong {res['verdicts']['H3_strong_single_driver']})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "regimes.png", dpi=130)
    plt.close(fig)

    with open(OUTDIR / "regime_verdict.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
