"""
experiments/exp_p2_synergy_onset.py  (C1, re-done with EMERGENT coupling)
========================================================================
H2 / P2 onset test on the PICARD self-referential model where the neighbour
perturbs the INPUT to the (state-read) rule -- so the Syn/Unq/Red split EMERGES
rather than being injected (the previous XOR coupling pre-loaded pure synergy).

Statistic Delta = onset(Unq) - onset(Syn) per seed; H2 predicts Delta > 0.
onset = sustained crossing (3 windows) of baseline+3sigma with a 0.02-bit floor.

Conditions: SYMMETRIC and ASYMMETRIC (distinct rule tables = "two distinct
interlocutors"). Controls per condition: placebo (decoupled), temporal surrogate,
target-permuted surrogate. We also report the coupled-vs-isolated LEVEL changes
(dSyn, dUnq, dRed) -- the substantive characterization when nothing onsets.
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

from edh.stats import ci95, onset, sign_test_one_sided  # noqa: E402
from edh.track_b_proto_dialogue import picard_pid_series, simulate_picard  # noqa: E402

OUTDIR = _ROOT / "results" / "p2_synergy_onset"
ALPHA = 0.01
SIM = dict(N=6, R=2048, L=8, T_isolated=120, T_coupled=240, beta=3.0, coupling=0.6)
STRIDE = 4
K_SIGMA, REQUIRE_CONSEC, MIN_ABS_RISE = 3.0, 3, 0.02


def _onset(series, key, t_c):
    return onset(series["time"], series[key], t_c, K_SIGMA,
                 require_consecutive=REQUIRE_CONSEC, min_abs_rise=MIN_ABS_RISE)


def _levels(series, t_c):
    t = np.asarray(series["time"])
    iso, cou = t < t_c, t >= t_c
    out = {}
    for kkey in ("syn", "unq", "red"):
        out["d_" + kkey] = float(np.nanmean(series[kkey][cou]) - np.nanmean(series[kkey][iso]))
        out["iso_" + kkey] = float(np.nanmean(series[kkey][iso]))
    return out


def temporal_surrogate(series, rng):
    out = dict(series)
    perm = rng.permutation(len(series["time"]))
    out["syn"] = series["syn"][perm]; out["unq"] = series["unq"][perm]
    return out


def run_condition(asymmetric, n_seeds, master_seed):
    rng = np.random.default_rng(master_seed + (1 if asymmetric else 0))
    kinds = ("main", "placebo", "temporal", "target")
    acc = {k: {"syn_on": [], "unq_on": [], "delta": [], "lev": []} for k in kinds}
    ex_main = ex_plac = None
    for s in range(n_seeds):
        st, t_c = simulate_picard(seed=s, asymmetric=asymmetric, **SIM)
        main = picard_pid_series(st, stride=STRIDE)
        tgt = picard_pid_series(st, stride=STRIDE, permute_target=True, seed=s)
        st0, _ = simulate_picard(seed=1000 + s, asymmetric=asymmetric,
                                 **{**SIM, "coupling": 0.0})
        plac = picard_pid_series(st0, stride=STRIDE)
        temp = temporal_surrogate(main, rng)
        if s == 0:
            ex_main, ex_plac = main, plac
        for k, ser in (("main", main), ("placebo", plac), ("temporal", temp),
                       ("target", tgt)):
            so, uo = _onset(ser, "syn", t_c), _onset(ser, "unq", t_c)
            acc[k]["syn_on"].append(so); acc[k]["unq_on"].append(uo)
            acc[k]["delta"].append(uo - so); acc[k]["lev"].append(_levels(ser, t_c))

    out = {}
    for k, d in acc.items():
        deltas = np.array(d["delta"], float)
        med, p, n = sign_test_one_sided(deltas)
        syn_on = np.array(d["syn_on"], float); unq_on = np.array(d["unq_on"], float)
        lev = {kk: float(np.mean([x[kk] for x in d["lev"]]))
               for kk in d["lev"][0]}
        out[k] = {"median_delta": med, "p_one_sided": p, "n_used": n,
                  "syn_onset_rate": float(np.mean(np.isfinite(syn_on))),
                  "unq_onset_rate": float(np.mean(np.isfinite(unq_on))),
                  "deltas": deltas.tolist(), **lev}

    main_sig = out["main"]["p_one_sided"] < ALPHA and out["main"]["median_delta"] > 0
    placebo_sig = out["placebo"]["p_one_sided"] < ALPHA and out["placebo"]["median_delta"] > 0
    if placebo_sig:
        verdict, reason = "PIPELINE_FAILURE", "placebo Delta>0 significant"
    elif main_sig and not placebo_sig:
        verdict, reason = "VERIFICADA", "median(Delta)>0 p<alpha and placebo n.s."
    elif out["main"]["syn_onset_rate"] < 0.3:
        verdict, reason = "FALSEADA/no-onset", (
            "synergy does NOT onset under emergent coupling (dSyn~0); H2 ordering "
            "not realized")
    else:
        verdict, reason = "INCONCLUSO", "no significant onset ordering"
    return {"controls": out, "verdict": verdict, "reason": reason,
            "example": {"main": ex_main, "placebo": ex_plac, "t_c": SIM["T_isolated"]}}


def main(n_seeds=30, master_seed=20250621):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for asym in (False, True):
        results["asymmetric" if asym else "symmetric"] = run_condition(
            asym, n_seeds, master_seed)
    write_outputs(results)
    print("=== C1 (emergent PICARD coupling): H2 synergy-onset test ===")
    print(f"sim={SIM}, stride={STRIDE}, alpha={ALPHA}\n")
    for cond, res in results.items():
        print(f"[{cond}]  VERDICT: {res['verdict']} -- {res['reason']}")
        for k, d in res["controls"].items():
            print(f"   {k:9s} Syn-onset={d['syn_onset_rate']:.2f} "
                  f"Unq-onset={d['unq_onset_rate']:.2f} medDelta={d['median_delta']:+.1f} "
                  f"p={d['p_one_sided']:.3f} | dSyn={d['d_syn']:+.3f} "
                  f"dUnq={d['d_unq']:+.3f} dRed={d['d_red']:+.3f}")
        print()
    return 0


def write_outputs(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    slim = {c: {"verdict": r["verdict"], "reason": r["reason"],
                "controls": {k: {kk: vv for kk, vv in v.items() if kk != "deltas"}
                             for k, v in r["controls"].items()}}
            for c, r in results.items()}
    with open(OUTDIR / "verdict.json", "w") as f:
        json.dump(slim, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, cond in zip(axes, ("symmetric", "asymmetric")):
        ser = results[cond]["example"]["main"]; t_c = results[cond]["example"]["t_c"]
        ax.plot(ser["time"], ser["syn"], label="Syn", color="C3")
        ax.plot(ser["time"], ser["unq"], label="Unq", color="C0")
        ax.plot(ser["time"], ser["red"], label="Red", color="C2", alpha=0.6)
        ax.axvline(t_c, color="gray", ls=":", label="t_c")
        ax.set_title(f"{cond}: {results[cond]['verdict']}")
        ax.set_xlabel("t"); ax.legend(fontsize=8)
    axes[0].set_ylabel("bits")
    fig.suptitle("C1 emergent coupling: Syn/Unq/Red (main, seed 0)")
    fig.tight_layout(); fig.savefig(OUTDIR / "syn_unq_series.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
