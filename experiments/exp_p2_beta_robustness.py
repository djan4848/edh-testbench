"""
experiments/exp_p2_beta_robustness.py  (Phase-4 step 1 -- the GATE)
==================================================================
The SECTION_6_REHABILITATED verdict hangs on beta=2.5. beta moves the PID
materially, so we must confirm the sign flip of dAIS_unit (fixed ~ -0.65 vs
plastic ~ +0.11) is NOT beta-specific before it goes in the report.

For each beta in {1.0,1.5,2.0,2.5,3.0}: size-controlled dAIS_unit (coupled -
uncoupled) for the FIXED PICARD model and the PLASTIC model, with bootstrap CIs,
plus the plastic specialization precondition.

Re-verdict:
  * fixed<0 AND plastic>0 (CIs exclude 0) across ALL beta -> REHABILITATED (firm).
  * plastic>0 only on a sub-range -> REHABILITATED_REGIME_DEPENDENT.
  * no robust flip -> INCONCLUSO (the beta=2.5 result was knife-edge).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "reference"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from edh.stats import ci95  # noqa: E402
from edh.track_b_proto_dialogue import (  # noqa: E402
    agent_core2,
    plastic_core2,
    simulate_picard,
    simulate_plastic,
    unit_ais,
)

OUTDIR = _ROOT / "results" / "p2_rg_plastic"
BETAS = [1.0, 1.5, 2.0, 2.5, 3.0]
N_SEEDS = 20
N_PRE = 8
BASE = dict(N=6, R=1536, L=8, T_isolated=120, T_coupled=240)


def _boot_median(x, n=3000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(0)
    return ci95([np.median(rng.choice(x, len(x))) for _ in range(n)])


def dais(simulate, core_fn, beta, coupling, n_seeds, **extra):
    d = []
    for s in range(n_seeds):
        stc, t_c = simulate(seed=s, beta=beta, coupling=coupling, **BASE, **extra)
        stu, _ = simulate(seed=s, beta=beta, coupling=0.0, **BASE, **extra)
        lo, hi = t_c + 40, stc.shape[0]
        ac = unit_ais(stc, lo, hi, (0, 1), lag=1, core_fn=core_fn)
        au = unit_ais(stu, lo, hi, (0, 1), lag=1, core_fn=core_fn)
        if ac.ok and au.ok:
            d.append(ac.ais - au.ais)
    d = np.array(d)
    return float(np.median(d)), list(_boot_median(d))


def specialization(beta, n_seeds=N_PRE):
    def sel(filt_win):
        p = np.bincount(filt_win.ravel(), minlength=4) / filt_win.size
        return float(p.max() - 0.25)
    diffs = []
    for s in range(n_seeds):
        st, t_c, fr = simulate_plastic(seed=s, beta=beta, coupling=1.0,
                                       return_filters=True, noise_partner=False, **BASE)
        _, _, fn = simulate_plastic(seed=s, beta=beta, coupling=1.0,
                                    return_filters=True, noise_partner=True, **BASE)
        lo, hi = t_c + 40, st.shape[0]
        diffs.append(sel(fr[lo:hi, :, 0]) - sel(fn[lo:hi, :, 0]))
    lo, hi = _boot_median(diffs)
    return float(np.median(diffs)), bool(lo > 0)


def main():
    from tqdm import tqdm
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in tqdm(BETAS, desc="beta gate"):
        fm, fci = dais(simulate_picard, agent_core2, b, 0.6, N_SEEDS)
        pm, pci = dais(simulate_plastic, plastic_core2, b, 1.0, N_SEEDS)
        smed, sok = specialization(b)
        rows.append({"beta": b, "fixed_dAIS": fm, "fixed_ci": fci,
                     "plastic_dAIS": pm, "plastic_ci": pci,
                     "spec_delta": smed, "specialized": sok})

    fixed_neg = all(r["fixed_ci"][1] < 0 for r in rows)
    plastic_pos = [r["plastic_ci"][0] > 0 and r["specialized"] for r in rows]
    if fixed_neg and all(plastic_pos):
        verdict = "SECTION_6_REHABILITATED"
    elif any(plastic_pos):
        good = [r["beta"] for r, ok in zip(rows, plastic_pos) if ok]
        verdict = f"REHABILITATED_REGIME_DEPENDENT (plastic flip at beta in {good})"
    else:
        verdict = "INCONCLUSO (no robust beta flip)"

    summary = {"betas": BETAS, "n_seeds": N_SEEDS, "rows": rows,
               "section_6_verdict_beta_gated": verdict}
    with open(OUTDIR / "beta_robustness.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot(rows, verdict)

    print("=== Phase-4 step 1: beta-robustness GATE for the section-6 sign flip ===")
    print(f"{'beta':>5} {'fixed dAIS [CI]':>26} {'plastic dAIS [CI]':>26} "
          f"{'spec?':>6}")
    for r in rows:
        print(f"{r['beta']:>5} {r['fixed_dAIS']:>8.3f} "
              f"[{r['fixed_ci'][0]:.3f},{r['fixed_ci'][1]:.3f}]   "
              f"{r['plastic_dAIS']:>8.3f} [{r['plastic_ci'][0]:.3f},{r['plastic_ci'][1]:.3f}]  "
              f"{str(r['specialized']):>6}")
    print(f"\nSECTION-6 VERDICT (beta-gated): {verdict}")
    return 0


def plot(rows, verdict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = [r["beta"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, color, lab in (("fixed", "C3", "fixed boundary"),
                            ("plastic", "C2", "plastic boundary (§6.5)")):
        m = np.array([r[f"{key}_dAIS"] for r in rows])
        lo = np.array([r[f"{key}_ci"][0] for r in rows])
        hi = np.array([r[f"{key}_ci"][1] for r in rows])
        ax.plot(b, m, "o-", color=color, label=lab)
        ax.fill_between(b, lo, hi, color=color, alpha=0.2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("beta"); ax.set_ylabel("dAIS_unit (coupled - uncoupled) [bits]")
    ax.set_title(f"Section-6 sign-flip vs beta\n{verdict}")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "beta_robustness.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
