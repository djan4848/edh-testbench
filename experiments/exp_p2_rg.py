"""
experiments/exp_p2_rg.py  (C2-RG + C3)
======================================
Does the C1 de-individuation reflect RG COARSE-GRAINING (a coherent macro-unit /
I-operator forms, eq.19) or DISSOLUTION (structure destroyed)? Measured at the
UNIT level on the EMERGENT PICARD model (never the XOR). H2 stays FALSIFIED.

C2-RG:
  (1) Size-controlled unit AIS: dAIS_unit = AIS(coupled pair as one unit) -
      AIS(uncoupled pair), same 4-bit macro, same beta/seeds, lag 1,2. Bootstrap CI.
      dAIS_unit>0 (CI excludes 0) => the unit is MORE self-predictable coupled than
      two independent agents => coarse-graining. <=0 => dissolution.
  (2) Macro coherence / self-similarity: AIS_macro_lag1>0 (predictable) and
      near-Markovian (lag-2 adds little), compared to the micro level.
  (3) Dip-recovery at the unit level: sliding-window unit AIS around t_c.

C3 bridge (Still 2012): sweep coupling; I_nopred = I(input_t;x_{t+1}) -
I(input_{t+1};x_t) vs measured Landauer dissipation. Test lower-bound + correlation.

Falsification lock P2_RG -> RG_COARSE_GRAINING / DISSOLUTION_DEAD / PARTIAL.
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
    i_nopred,
    micro_ais,
    simulate_picard,
    unit_ais,
    unit_macro,
)

OUTDIR = _ROOT / "results" / "p2_rg"
SIM = dict(N=6, R=2048, L=8, T_isolated=120, T_coupled=240, beta=3.0)
COUPLING = 0.6
PAIR = (0, 1)


def coupled_window(t_c, T):
    return t_c + 40, T            # stationary coupled phase


# ---------------------------------------------------------------------------
# C2-RG (1)+(2): size-controlled unit AIS + self-similarity
# ---------------------------------------------------------------------------
def unit_formation(n_seeds=20):
    d1, d2 = [], []                 # dAIS_unit at lag 1, 2
    macro1_c, macro2_c, micro1_c, micro2_c = [], [], [], []
    for s in range(n_seeds):
        stc, t_c = simulate_picard(seed=s, coupling=COUPLING, **SIM)
        stu, _ = simulate_picard(seed=s, coupling=0.0, **SIM)
        lo, hi = coupled_window(t_c, stc.shape[0])
        ac1 = unit_ais(stc, lo, hi, PAIR, lag=1); au1 = unit_ais(stu, lo, hi, PAIR, lag=1)
        ac2 = unit_ais(stc, lo, hi, PAIR, lag=2); au2 = unit_ais(stu, lo, hi, PAIR, lag=2)
        if ac1.ok and au1.ok:
            d1.append(ac1.ais - au1.ais); macro1_c.append(ac1.ais)
        if ac2.ok and au2.ok:
            d2.append(ac2.ais - au2.ais); macro2_c.append(ac2.ais)
        mi1 = micro_ais(stc, lo, hi, agent=0, lag=1)
        mi2 = micro_ais(stc, lo, hi, agent=0, lag=2)
        if mi1.ok:
            micro1_c.append(mi1.ais)
        if mi2.ok:
            micro2_c.append(mi2.ais)

    def boot_median(x, n=4000):
        x = np.asarray(x, float); rng = np.random.default_rng(0)
        return ci95([np.median(rng.choice(x, len(x))) for _ in range(n)])

    d1 = np.array(d1); d2 = np.array(d2)
    macro1 = float(np.mean(macro1_c)); macro2 = float(np.mean(macro2_c))
    micro1 = float(np.mean(micro1_c)); micro2 = float(np.mean(micro2_c))
    res = {
        "dAIS_unit_lag1_median": float(np.median(d1)),
        "dAIS_unit_lag1_ci95": list(boot_median(d1)),
        "dAIS_unit_lag2_median": float(np.median(d2)),
        "dAIS_unit_lag2_ci95": list(boot_median(d2)),
        "macro_AIS_coupled_lag1": macro1, "macro_AIS_coupled_lag2": macro2,
        "micro_AIS_coupled_lag1": micro1, "micro_AIS_coupled_lag2": micro2,
        "macro_markov_excess": float((macro2 - macro1) / macro1) if macro1 > 0 else np.nan,
        "micro_markov_excess": float((micro2 - micro1) / micro1) if micro1 > 0 else np.nan,
    }
    lo_ci, hi_ci = res["dAIS_unit_lag1_ci95"]
    res["unit_more_predictable"] = bool(lo_ci > 0)
    res["macro_coherent"] = bool(macro1 > 0.02)
    res["unit_formation"] = bool(res["unit_more_predictable"] and res["macro_coherent"])
    return res, d1


# ---------------------------------------------------------------------------
# C2-RG (3): dip-recovery at the unit level (sliding window)
# ---------------------------------------------------------------------------
def dip_recovery(seed=0, win=24, step=8):
    stc, t_c = simulate_picard(seed=seed, coupling=COUPLING, **SIM)
    T = stc.shape[0]
    times, ais = [], []
    for c in range(win, T - 1, step):
        r = unit_ais(stc, c - win, c, PAIR, lag=1)
        times.append(c); ais.append(r.ais if r.ok else np.nan)
    return np.array(times), np.array(ais), t_c


# ---------------------------------------------------------------------------
# C3: Still thermodynamic bridge -- I_nopred vs measured dissipation
# ---------------------------------------------------------------------------
def bridge(couplings=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), n_seeds=8):
    rows = []
    for c in couplings:
        inp, dpd, dis = [], [], []
        for s in range(n_seeds):
            st, t_c, diss = simulate_picard(seed=s, coupling=c, return_dissipation=True,
                                            **SIM)
            lo, hi = coupled_window(t_c, st.shape[0])
            m, p, inopred_ij = i_nopred(st, lo, hi, driven=0, driver=1)
            _, _, inopred_ji = i_nopred(st, lo, hi, driven=1, driver=0)
            if np.isfinite(inopred_ij) and np.isfinite(inopred_ji):
                inp.append(0.5 * (inopred_ij + inopred_ji))
            dis.append(float(np.mean(diss[lo:hi])))
        rows.append({"coupling": c, "I_nopred": float(np.mean(inp)) if inp else np.nan,
                     "dissipation": float(np.mean(dis))})
    inp = np.array([r["I_nopred"] for r in rows])
    dss = np.array([r["dissipation"] for r in rows])
    good = np.isfinite(inp) & np.isfinite(dss)
    corr = float(np.corrcoef(inp[good], dss[good])[0, 1]) if good.sum() > 2 else np.nan
    bound_holds = bool(np.all(inp[good] <= dss[good] + 1e-9))
    return {"rows": rows, "correlation": corr, "bound_holds": bound_holds,
            "bridge": bool(bound_holds and np.isfinite(corr) and corr > 0.5)}


def decide(uf, br):
    if uf["unit_formation"] and br["bridge"]:
        return "RG_COARSE_GRAINING"
    if (not uf["unit_formation"]) and (not br["bridge"]):
        return "DISSOLUTION_DEAD"
    return "PARTIAL"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    uf, d1 = unit_formation()
    times, ais, t_c = dip_recovery()
    br = bridge()
    verdict = decide(uf, br)
    summary = {"unit_formation": uf, "bridge": br, "verdict_P2_RG": verdict}
    with open(OUTDIR / "verdict.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot(uf, d1, times, ais, t_c, br, verdict)

    print("=== C2-RG: unit formation vs dissolution ===")
    print(f"dAIS_unit lag1 = {uf['dAIS_unit_lag1_median']:+.4f} "
          f"CI95 {np.round(uf['dAIS_unit_lag1_ci95'],4)}  (lag2 "
          f"{uf['dAIS_unit_lag2_median']:+.4f})")
    print(f"macro AIS coupled lag1 = {uf['macro_AIS_coupled_lag1']:.4f} "
          f"(markov excess macro={uf['macro_markov_excess']:.2f} "
          f"micro={uf['micro_markov_excess']:.2f})")
    print(f"unit_more_predictable={uf['unit_more_predictable']} "
          f"macro_coherent={uf['macro_coherent']} -> unit_formation={uf['unit_formation']}")
    print("\n=== C3: Still bridge ===")
    for r in br["rows"]:
        print(f"  c={r['coupling']:.1f}  I_nopred={r['I_nopred']:.4f}  "
              f"dissipation={r['dissipation']:.4f}")
    print(f"  bound_holds={br['bound_holds']} corr={br['correlation']:.2f} "
          f"-> bridge={br['bridge']}")
    print(f"\nP2_RG VERDICT: {verdict}")
    return 0


def plot(uf, d1, times, ais, t_c, br, verdict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(d1, bins=12, color="C0", alpha=0.8)
    axes[0].axvline(0, color="k"); axes[0].axvline(uf["dAIS_unit_lag1_median"], color="C3")
    axes[0].set_title(f"dAIS_unit lag1 (med={uf['dAIS_unit_lag1_median']:+.3f})\n"
                      f"unit_formation={uf['unit_formation']}")
    axes[0].set_xlabel("AIS_coupled - AIS_uncoupled [bits]")
    axes[1].plot(times, ais, "o-"); axes[1].axvline(t_c, color="gray", ls=":")
    axes[1].set_title("unit AIS(t) -- dip-recovery?"); axes[1].set_xlabel("t")
    axes[1].set_ylabel("unit AIS [bits]")
    inp = [r["I_nopred"] for r in br["rows"]]; dss = [r["dissipation"] for r in br["rows"]]
    axes[2].plot([r["coupling"] for r in br["rows"]], inp, "o-", label="I_nopred")
    axes[2].plot([r["coupling"] for r in br["rows"]], dss, "s-", label="dissipation")
    axes[2].set_title(f"Still bridge (corr={br['correlation']:.2f}, "
                      f"bound={br['bound_holds']})")
    axes[2].set_xlabel("coupling"); axes[2].legend()
    fig.suptitle(f"C2-RG + C3  ->  {verdict}")
    fig.tight_layout(); fig.savefig(OUTDIR / "rg_bridge.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
