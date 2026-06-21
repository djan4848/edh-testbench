"""
experiments/exp_robustness_trackb.py  (Phase-4 step 2, Track B)
==============================================================
Sensitivity of the §6 finding. Priority item: the NON-HEBBIAN realization of
eq.20 -- does boundary plasticity flip the sign in general, or only Hebbian?

Each check reports the fixed-vs-plastic ΔAIS_unit contrast (sign is what decides;
plastic magnitudes are small and their CIs deterministically tight, so we lead
with sign + separation, not magnitude) and the specialization precondition.
  * beta extension {4,5}        -- characterise the non-monotone plastic curve.
  * NON-HEBBIAN (consensus)     -- DECISIVE: Hebbian-specific or general?
  * coupling strength           -- does the contrast survive?
Outputs: results/robustness/track_b_sensitivity.{csv,json}
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

from edh.stats import ci95  # noqa: E402
from edh.track_b_proto_dialogue import (  # noqa: E402
    agent_core2,
    plastic_core2,
    simulate_picard,
    simulate_plastic,
    unit_ais,
)

OUTDIR = _ROOT / "results" / "robustness"
BASE = dict(N=6, R=1536, L=8, T_isolated=120, T_coupled=240)
N_SEEDS = 20


def _boot(x, n=3000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(0)
    return ci95([np.median(rng.choice(x, len(x))) for _ in range(n)]) if len(x) else (np.nan, np.nan)


def dais(simulate, core_fn, beta, coupling, n_seeds=N_SEEDS, **extra):
    d = []
    for s in range(n_seeds):
        stc, t_c = simulate(seed=s, beta=beta, coupling=coupling, **BASE, **extra)
        stu, _ = simulate(seed=s, beta=beta, coupling=0.0, **BASE, **extra)
        lo, hi = t_c + 40, stc.shape[0]
        ac = unit_ais(stc, lo, hi, (0, 1), lag=1, core_fn=core_fn)
        au = unit_ais(stu, lo, hi, (0, 1), lag=1, core_fn=core_fn)
        if ac.ok and au.ok:
            d.append(ac.ais - au.ais)
    return float(np.median(d)), list(_boot(d))


def specialized(beta, plasticity, n_seeds=8):
    def sel(fw):
        p = np.bincount(fw.ravel(), minlength=4) / fw.size
        return float(p.max() - 0.25)
    diffs = []
    for s in range(n_seeds):
        st, t_c, fr = simulate_plastic(seed=s, beta=beta, coupling=1.0,
                                       return_filters=True, noise_partner=False,
                                       plasticity=plasticity, **BASE)
        _, _, fn = simulate_plastic(seed=s, beta=beta, coupling=1.0,
                                    return_filters=True, noise_partner=True,
                                    plasticity=plasticity, **BASE)
        lo, hi = t_c + 40, st.shape[0]
        diffs.append(sel(fr[lo:hi, :, 0]) - sel(fn[lo:hi, :, 0]))
    return float(np.median(diffs)), bool(_boot(diffs)[0] > 0)


def contrast_row(label, fixed, plastic, spec_med, spec_ok):
    fm, fci = fixed
    pm, pci = plastic
    flip = bool(fci[1] < 0 and pci[0] > 0 and spec_ok)
    return {"check": label, "fixed_dAIS": round(fm, 4), "fixed_ci": [round(x, 4) for x in fci],
            "plastic_dAIS": round(pm, 4), "plastic_ci": [round(x, 4) for x in pci],
            "spec_delta": round(spec_med, 4), "specialized": spec_ok,
            "sign_flip": flip}


def main():
    from tqdm import tqdm
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # 1) beta extension {4,5} (Hebbian)
    for b in tqdm([4.0, 5.0], desc="beta-ext"):
        fixed = dais(simulate_picard, agent_core2, b, 0.6)
        plastic = dais(simulate_plastic, plastic_core2, b, 1.0, plasticity="hebbian")
        sm, sok = specialized(b, "hebbian")
        rows.append(contrast_row(f"beta={b} (hebbian)", fixed, plastic, sm, sok))

    # 2) NON-HEBBIAN (consensus) at beta=2.5 -- DECISIVE
    fixed = dais(simulate_picard, agent_core2, 2.5, 0.6)
    plastic = dais(simulate_plastic, plastic_core2, 2.5, 1.0, plasticity="consensus")
    sm, sok = specialized(2.5, "consensus")
    rows.append(contrast_row("beta=2.5 NON-HEBBIAN (consensus)", fixed, plastic, sm, sok))

    # 3) coupling strength (Hebbian, beta=2.5)
    for cstr in [0.5, 1.5]:
        fixed = dais(simulate_picard, agent_core2, 2.5, min(cstr, 0.9))
        plastic = dais(simulate_plastic, plastic_core2, 2.5, cstr, plasticity="hebbian")
        sm, sok = specialized(2.5, "hebbian")
        rows.append(contrast_row(f"coupling={cstr} (hebbian)", fixed, plastic, sm, sok))

    with open(OUTDIR / "track_b_sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "fixed_dAIS", "fixed_ci", "plastic_dAIS", "plastic_ci",
                    "spec_delta", "specialized", "sign_flip"])
        for r in rows:
            w.writerow([r["check"], r["fixed_dAIS"], r["fixed_ci"], r["plastic_dAIS"],
                        r["plastic_ci"], r["spec_delta"], r["specialized"], r["sign_flip"]])
    nonheb = next(r for r in rows if "NON-HEBBIAN" in r["check"])
    summary = {"rows": rows,
               "nonhebbian_flips": nonheb["sign_flip"],
               "section6_generality": ("boundary plasticity GENERALLY flips (non-Hebbian also)"
                                       if nonheb["sign_flip"] else
                                       "flip is Hebbian-specific (existence proof only)")}
    with open(OUTDIR / "track_b_sensitivity.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Track-B robustness (sign is what decides) ===")
    print(f"{'check':38s} {'fixed':>9} {'plastic':>9} {'spec?':>6} {'FLIP?':>6}")
    for r in rows:
        print(f"{r['check']:38s} {r['fixed_dAIS']:>9.3f} {r['plastic_dAIS']:>9.3f} "
              f"{str(r['specialized']):>6} {str(r['sign_flip']):>6}")
    print(f"\nNON-HEBBIAN flips sign: {nonheb['sign_flip']} -> {summary['section6_generality']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
