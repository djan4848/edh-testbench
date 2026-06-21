"""
experiments/exp_p2_rg_plastic.py  (P2_RG_plastic -- the faithful section-6 test)
===============================================================================
The LAST model elaboration: the autopoietic PLASTIC-BOUNDARY model (section 6.5 /
eq.20), where each agent's filter bits (which neighbour bits it attends to) are
state, read AND written by the same rule. This is the paper's OWN unit-formation
mechanism, omitted by the fixed-boundary test (which dissolved).

PRECONDITION (checked BEFORE the verdict): the filters must specialise toward the
partner (selectivity > a noise-partner control, CI excludes 0). If not, the test
is not faithful -> INVALID_TEST.

unit_formation: dAIS_unit = AIS(coupled plastic pair) - AIS(uncoupled), size-
controlled, plus macro coherence/self-similarity AND dip WITH recovery.

Lock -> SECTION_6_REHABILITATED / SECTION_6_FALSIFIED / INVALID_TEST.
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
    micro_ais,
    plastic_core2,
    simulate_plastic,
    unit_ais,
)

OUTDIR = _ROOT / "results" / "p2_rg_plastic"
SIM = dict(N=6, R=2048, L=8, T_isolated=120, T_coupled=240, beta=2.5, coupling=1.0)
PAIR = (0, 1)


def window(t_c, T):
    return t_c + 40, T


def _selectivity(filt_win):           # (W,R) addresses 0..3 -> excess over uniform
    p = np.bincount(filt_win.ravel(), minlength=4) / filt_win.size
    return float(p.max() - 0.25)


def precondition(n_seeds=15):
    d, real, noise, iso = [], [], [], []
    for s in range(n_seeds):
        st, t_c, filt = simulate_plastic(seed=s, return_filters=True,
                                         noise_partner=False, **SIM)
        _, _, filtn = simulate_plastic(seed=s, return_filters=True,
                                       noise_partner=True, **SIM)
        lo, hi = window(t_c, st.shape[0])
        sr = _selectivity(filt[lo:hi, :, 0]); sn = _selectivity(filtn[lo:hi, :, 0])
        si = _selectivity(filt[40:t_c - 2, :, 0])
        real.append(sr); noise.append(sn); iso.append(si); d.append(sr - sn)
    rng = np.random.default_rng(0)
    d = np.array(d)
    boot = ci95([np.median(rng.choice(d, len(d))) for _ in range(4000)])
    return {"sel_real": float(np.mean(real)), "sel_noise": float(np.mean(noise)),
            "sel_isolated": float(np.mean(iso)),
            "delta_selectivity_median": float(np.median(d)),
            "delta_selectivity_ci95": list(boot),
            "specialized": bool(boot[0] > 0 and np.mean(real) > np.mean(iso) + 0.05)}


def unit_formation(n_seeds=20):
    d1, mc1 = [], []
    for s in range(n_seeds):
        stc, t_c = simulate_plastic(seed=s, noise_partner=False, **SIM)
        stu, _ = simulate_plastic(seed=s, **{**SIM, "coupling": 0.0})
        lo, hi = window(t_c, stc.shape[0])
        ac = unit_ais(stc, lo, hi, PAIR, lag=1, core_fn=plastic_core2)
        au = unit_ais(stu, lo, hi, PAIR, lag=1, core_fn=plastic_core2)
        if ac.ok and au.ok:
            d1.append(ac.ais - au.ais); mc1.append(ac.ais)
    rng = np.random.default_rng(0); d1 = np.array(d1)
    boot = ci95([np.median(rng.choice(d1, len(d1))) for _ in range(4000)])
    macro1 = float(np.mean(mc1))
    return {"dAIS_unit_lag1_median": float(np.median(d1)),
            "dAIS_unit_lag1_ci95": list(boot),
            "macro_AIS_coupled_lag1": macro1,
            "unit_more_predictable": bool(boot[0] > 0),
            "macro_coherent": bool(macro1 > 0.02)}, d1


def dip_recovery(seed=0, win_w=24, step=8):
    stc, t_c = simulate_plastic(seed=seed, noise_partner=False, **SIM)
    T = stc.shape[0]
    times, ais = [], []
    for c in range(win_w, T - 1, step):
        r = unit_ais(stc, c - win_w, c, PAIR, lag=1, core_fn=plastic_core2)
        times.append(c); ais.append(r.ais if r.ok else np.nan)
    times, ais = np.array(times), np.array(ais)
    post = ais[times >= t_c]
    post = post[np.isfinite(post)]
    if len(post) >= 4:
        dip = float(np.min(post[:len(post) // 2])); end = float(np.mean(post[-3:]))
        recovers = bool(end > dip + 0.05)
    else:
        dip = end = np.nan; recovers = False
    return times, ais, t_c, {"dip": dip, "end": end, "recovers": recovers}


def decide(pre, uf, dipinfo):
    if not pre["specialized"]:
        return "INVALID_TEST"
    unit_formed = (uf["unit_more_predictable"] and uf["macro_coherent"]
                   and dipinfo["recovers"])
    return "SECTION_6_REHABILITATED" if unit_formed else "SECTION_6_FALSIFIED"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pre = precondition()
    uf, d1 = unit_formation()
    times, ais, t_c, dipinfo = dip_recovery()
    verdict = decide(pre, uf, dipinfo)
    summary = {"precondition": pre, "unit_formation": uf, "dip": dipinfo,
               "verdict_P2_RG_plastic": verdict}
    with open(OUTDIR / "verdict.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot(pre, uf, d1, times, ais, t_c, dipinfo, verdict)

    print("=== P2_RG_plastic (autopoietic plastic boundary, section 6.5) ===")
    print(f"PRECONDITION: sel_real={pre['sel_real']:.3f} sel_noise={pre['sel_noise']:.3f} "
          f"sel_iso={pre['sel_isolated']:.3f}  dSel={pre['delta_selectivity_median']:+.3f} "
          f"CI{np.round(pre['delta_selectivity_ci95'],3)} -> specialized={pre['specialized']}")
    print(f"UNIT FORMATION: dAIS_unit lag1={uf['dAIS_unit_lag1_median']:+.4f} "
          f"CI{np.round(uf['dAIS_unit_lag1_ci95'],4)}  macroAIS={uf['macro_AIS_coupled_lag1']:.3f} "
          f"more_predictable={uf['unit_more_predictable']}")
    print(f"DIP-RECOVERY: dip={dipinfo['dip']:.3f} end={dipinfo['end']:.3f} "
          f"recovers={dipinfo['recovers']}")
    print(f"\nP2_RG_plastic VERDICT: {verdict}")
    return 0


def plot(pre, uf, d1, times, ais, t_c, dipinfo, verdict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(["real", "noise", "isolated"],
                [pre["sel_real"], pre["sel_noise"], pre["sel_isolated"]],
                color=["C2", "C1", "C7"])
    axes[0].set_title(f"precondition: filter selectivity\nspecialized={pre['specialized']}")
    axes[0].set_ylabel("selectivity (excess over uniform)")
    axes[1].hist(d1, bins=12, color="C0", alpha=0.8)
    axes[1].axvline(0, color="k"); axes[1].axvline(uf["dAIS_unit_lag1_median"], color="C3")
    axes[1].set_title(f"dAIS_unit lag1 (med={uf['dAIS_unit_lag1_median']:+.3f})")
    axes[1].set_xlabel("AIS_coupled - AIS_uncoupled [bits]")
    axes[2].plot(times, ais, "o-"); axes[2].axvline(t_c, color="gray", ls=":")
    axes[2].set_title(f"unit AIS(t): dip recovers={dipinfo['recovers']}")
    axes[2].set_xlabel("t"); axes[2].set_ylabel("unit AIS [bits]")
    fig.suptitle(f"P2_RG_plastic -> {verdict}")
    fig.tight_layout(); fig.savefig(OUTDIR / "rg_plastic.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
