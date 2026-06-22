"""
experiments/make_paper_figures.py — publication re-render into paper_figures/.

PURELY COSMETIC. No analysis is recomputed in a way that changes any verdict:
results are loaded from results/ where the plotted arrays were cached; where only
a PNG was ever saved (example trajectories, per-seed ΔAIS histograms, χ curves),
the EXISTING deterministic analysis functions are called with the SAME seeds/params
(identical numbers — verified against the cached verdicts in check #4). Verdicts,
preregistration.yaml and results/ are untouched.

Output: paper_figures/ (main) and paper_figures/supp/ (supplementary), each figure
as .pdf (vector) + .png (300 dpi). No titles/suptitles, no internal verdict
codenames; panel letters (a)(b)(c) added; axis labels, reference lines (E_knee,
t_c), shaded bands, error bars and the quantitative legends are KEPT.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "experiments", _ROOT / "reference"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

RES = _ROOT / "results"
PAPER = _ROOT / "paper_figures"
SUPP = PAPER / "supp"
MANIFEST = []          # (filename, section, source)
CHECKS = []


def _save(fig, stem, section, source, supp=False):
    d = SUPP if supp else PAPER
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{stem}.{ext}")
    plt.close(fig)
    rel = f"supp/{stem}" if supp else stem
    MANIFEST.append((f"{rel}.pdf / .png", section, source))
    print(f"  wrote {rel}.pdf + .png")


def _panel(ax, letter):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes, va="top", ha="left",
            fontweight="bold", fontsize=12)


def _load(rel):
    return json.loads((RES / rel).read_text())


# ---------------------------------------------------------------------------
# f1 — cost decomposition (cached json)
# ---------------------------------------------------------------------------
def fig_f1():
    d = _load("diagnostics/cost_decomposition.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, key, ltr in ((axes[0], "linear_demand", "(a)"),
                         (axes[1], "pairwise_demand", "(b)")):
        dd = d[key]; pt = dd["point"]
        ax.loglog(dd["N_used"], dd["cost_mean"], "o-", color="C0")
        ax.text(0.05, 0.92, f"a = {pt['a']:.2f}\nb = {pt['b']:.2f}\n"
                f"p = {pt['p_measured']:.2f}", transform=ax.transAxes,
                va="top", fontsize=9)
        ax.set_xlabel("N"); ax.set_ylabel("analog dissipation")
        ax.grid(True, which="both", alpha=0.3); _panel(ax, ltr)
    fig.tight_layout()
    _save(fig, "f1_cost_decomposition", "Main / P1 axis A",
          "exp_p1_decompose_cost (cached)")


# ---------------------------------------------------------------------------
# f2 — two-regime N*(E), linear (npz + regime_verdict; masks via two_regime_fit)
# ---------------------------------------------------------------------------
def fig_f2():
    from edh.stats import two_regime_fit
    z = np.load(RES / "q_sweep_linear/nstar_boot.npz", allow_pickle=True)
    rv = _load("q_sweep_linear/regime_verdict.json")
    interior = z["interior"].astype(bool)
    E = z["E_vals"][interior]; nstar = z["nstar_point"][interior]
    boot = [np.asarray(b) for b, k in zip(z["nstar_boot"], interior) if k]
    reg = two_regime_fit(E, nstar, boot=boot)
    plateau = np.array(reg["plateau_mask"]); rise = np.array(reg["rise_mask"])
    inv_p = rv["one_over_p_high"]; E_knee = reg["E_knee"]
    x, y = np.log(E), np.log(nstar)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(x[plateau], y[plateau], "o", color="C0", label="plateau")
    ax.plot(x[rise], y[rise], "s", color="C3", label="rise")

    def seg(mask, color, lbl):
        s, b = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 30)
        ax.plot(xs, s * xs + b, "-", color=color, label=lbl)
    seg(plateau, "C0", f"$q_{{low}}$ = {reg['q_low']:+.3f}")
    s_hi, b_hi = np.polyfit(x[rise], y[rise], 1)
    xs = np.linspace(x[rise].min(), x[rise].max(), 30)
    ax.plot(xs, s_hi * xs + b_hi, "-", color="C3",
            label=f"$q_{{high}}$ = {reg['q_high']:+.3f}")
    mx = xs.mean(); my = s_hi * mx + b_hi
    ax.plot(xs, inv_p * (xs - mx) + my, "--", color="green",
            label=f"slope $1/p_{{high}}$ = {inv_p:.3f}")
    ax.axvline(np.log(E_knee), color="gray", ls=":", label=f"$E_{{knee}}$ ≈ {E_knee:.0f}")
    ax.set_xlabel(r"$\log\, E_{\rm protocol}/k_BT$"); ax.set_ylabel(r"$\log\, N^*$")
    ax.legend()
    fig.tight_layout()
    _save(fig, "f2_two_regime_Nstar", "Main / P1 axis B",
          "exp_p1_regimes (npz + regime_verdict cached)")


# ---------------------------------------------------------------------------
# f3 — P3 entropy crossover, combined pairwise (a) + linear (b)  (crossover.csv)
# ---------------------------------------------------------------------------
def fig_f3():
    rows = list(csv.DictReader(open(RES / "p3_entropy_crossover/crossover.csv")))
    v = _load("p3_entropy_crossover/verdict.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, model, ltr in ((axes[0], "pairwise", "(a)"), (axes[1], "linear", "(b)")):
        rr = [r for r in rows if r["model"] == model]
        E = np.array([float(r["E"]) for r in rr])
        nc = np.array([float(r["N_cross"]) for r in rr])
        ns = np.array([float(r["N_star"]) for r in rr])
        inter = np.array([r["interior"].strip().lower() == "true" for r in rr])
        knee = v[model]["E_knee"]
        ax.plot(E, nc, "o-", color="C2", label=r"$N_{\rm cross}$")
        ax.plot(E[inter], ns[inter], "s--", color="C0", label=r"$N^*$")
        ax.axvline(knee, color="gray", ls=":", label=r"$E_{\rm knee}$")
        ax.set_xscale("log"); ax.set_xlabel(r"$E_{\rm protocol}/k_BT$")
        ax.set_ylabel("N"); ax.legend(); _panel(ax, ltr)
    fig.tight_layout()
    _save(fig, "f3_P3_entropy_crossover", "Main / P3",
          "exp_p3_entropy_crossover (cached, panels combined)")


# ---------------------------------------------------------------------------
# f4 — H2 synergy/unique series (re-sim example, seed 0, sym + asym)
# ---------------------------------------------------------------------------
def fig_f4():
    import exp_p2_synergy_onset as e
    from edh.track_b_proto_dialogue import picard_pid_series, simulate_picard
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, asym, name, ltr in ((axes[0], False, "symmetric", "(a)"),
                                (axes[1], True, "asymmetric", "(b)")):
        st, t_c = simulate_picard(seed=0, asymmetric=asym, **e.SIM)
        s = picard_pid_series(st, stride=e.STRIDE)
        ax.plot(s["time"], s["syn"], color="C3", label="Syn")
        ax.plot(s["time"], s["unq"], color="C0", label="Unq")
        ax.plot(s["time"], s["red"], color="C2", alpha=0.7, label="Red")
        ax.axvline(t_c, color="gray", ls=":", label=r"$t_c$")
        ax.set_xlabel("t"); ax.text(0.5, 1.01, name, transform=ax.transAxes,
                                    ha="center", va="bottom", fontsize=10)
        ax.legend(); _panel(ax, ltr)
    axes[0].set_ylabel("information [bits]")
    fig.tight_layout()
    _save(fig, "f4_H2_synergy_unique", "Main / P2-H2",
          "exp_p2_synergy_onset (example re-sim, seed 0)")


# ---------------------------------------------------------------------------
# f5 — fixed-boundary dissolution (ΔAIS hist + dip + bridge)
# ---------------------------------------------------------------------------
def fig_f5():
    import exp_p2_rg as e
    uf, d1 = e.unit_formation(n_seeds=20)
    times, ais, t_c = e.dip_recovery()
    br = _load("p2_rg/verdict.json")["bridge"]["rows"]
    cached_med = _load("p2_rg/verdict.json")["unit_formation"]["dAIS_unit_lag1_median"]
    CHECKS.append(("f5 ΔAIS median re-derived vs cached",
                   f"{np.median(d1):+.3f} vs {cached_med:+.3f}",
                   abs(np.median(d1) - cached_med) < 1e-6))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    axes[0].hist(d1, bins=12, color="C0", alpha=0.85)
    axes[0].axvline(0, color="k"); axes[0].axvline(np.median(d1), color="C3")
    axes[0].text(0.05, 0.92, f"med = {np.median(d1):+.3f}",
                 transform=axes[0].transAxes, va="top", fontsize=9)
    axes[0].set_xlabel(r"$\Delta$AIS$_{\rm unit}$ (coupled − uncoupled) [bits]")
    axes[0].set_ylabel("count"); _panel(axes[0], "(a)")
    axes[1].plot(times, ais, "o-", color="C0"); axes[1].axvline(t_c, color="gray", ls=":")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("unit AIS [bits]"); _panel(axes[1], "(b)")
    cs = [r["coupling"] for r in br]
    axes[2].plot(cs, [r["I_nopred"] for r in br], "o-", label=r"$I_{\rm nopred}$")
    axes[2].plot(cs, [r["dissipation"] for r in br], "s-", label="dissipation")
    axes[2].set_xlabel("coupling"); axes[2].set_ylabel("[bits]")
    axes[2].legend(); _panel(axes[2], "(c)")
    fig.tight_layout()
    _save(fig, "f5_fixed_boundary_dissolution", "Main / P2-RG fixed",
          "exp_p2_rg (ΔAIS+dip re-derived; bridge cached)")


# ---------------------------------------------------------------------------
# f6 — plastic-boundary formation (precondition + ΔAIS hist + dip)
# ---------------------------------------------------------------------------
def fig_f6():
    import exp_p2_rg_plastic as e
    v = _load("p2_rg_plastic/verdict.json")
    pre = v["precondition"]
    uf, d1 = e.unit_formation(n_seeds=20)
    times, ais, t_c, _ = e.dip_recovery()
    cached_med = v["unit_formation"]["dAIS_unit_lag1_median"]
    CHECKS.append(("f6 ΔAIS median re-derived vs cached",
                   f"{np.median(d1):+.3f} vs {cached_med:+.3f}",
                   abs(np.median(d1) - cached_med) < 1e-6))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    axes[0].bar(["real", "noise", "isolated"],
                [pre["sel_real"], pre["sel_noise"], pre["sel_isolated"]],
                color=["C2", "C1", "C7"])
    axes[0].set_ylabel("filter selectivity"); _panel(axes[0], "(a)")
    axes[1].hist(d1, bins=12, color="C0", alpha=0.85)
    axes[1].axvline(0, color="k"); axes[1].axvline(np.median(d1), color="C3")
    axes[1].text(0.05, 0.92, f"med = {np.median(d1):+.3f}",
                 transform=axes[1].transAxes, va="top", fontsize=9)
    axes[1].set_xlabel(r"$\Delta$AIS$_{\rm unit}$ (coupled − uncoupled) [bits]")
    axes[1].set_ylabel("count"); _panel(axes[1], "(b)")
    axes[2].plot(times, ais, "o-", color="C0"); axes[2].axvline(t_c, color="gray", ls=":")
    axes[2].set_xlabel("t"); axes[2].set_ylabel("unit AIS [bits]"); _panel(axes[2], "(c)")
    fig.tight_layout()
    _save(fig, "f6_plastic_boundary_formation", "Main / P2-RG plastic (§6.5)",
          "exp_p2_rg_plastic (precondition cached; ΔAIS+dip re-derived)")


# ---------------------------------------------------------------------------
# f7 — β sign-flip (beta_robustness.json)
# ---------------------------------------------------------------------------
def fig_f7():
    d = _load("p2_rg_plastic/beta_robustness.json")
    b = [r["beta"] for r in d["rows"]]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for key, color, lab in (("fixed", "C3", "fixed boundary"),
                            ("plastic", "C2", "plastic boundary (§6.5)")):
        m = np.array([r[f"{key}_dAIS"] for r in d["rows"]])
        lo = np.array([r[f"{key}_ci"][0] for r in d["rows"]])
        hi = np.array([r[f"{key}_ci"][1] for r in d["rows"]])
        ax.plot(b, m, "o-", color=color, label=lab)
        ax.fill_between(b, lo, hi, color=color, alpha=0.2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Delta$AIS$_{\rm unit}$ (coupled − uncoupled) [bits]")
    ax.legend()
    fig.tight_layout()
    _save(fig, "f7_beta_sign_flip", "Main / §6 β-gate",
          "exp_p2_beta_robustness (cached)")


# ---------------------------------------------------------------------------
# s1 — susceptibility calibration χ(N) (re-run calibration for χ arrays)
# ---------------------------------------------------------------------------
def fig_s1():
    import exp_p1_calibrate_lambda as e
    cfg = e.CalibConfig(); cfg.resolve_e_center()
    from edh.track_a_signaling import TrackAParams
    params = TrackAParams(temperature=cfg.T, e_center=cfg.e_protocol_center,
                          tail_frac=cfg.tail_frac)
    n_grid = np.array(cfg.n_grid, dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for lam in cfg.lambdas:
        row = e.run_one_lambda(lam, cfg, params)
        line, = ax.plot(n_grid, row.chi, "o-", ms=3,
                        label=f"$\\lambda$={lam:g} ($N^*$={row.n_peak:.1f})")
        ax.axvline(row.n_peak, color=line.get_color(), alpha=0.25, lw=1)
    ax.set_xlabel("N (agents / meanings)")
    ax.set_ylabel(r"$\chi(N)=\mathrm{Var}_{\rm seeds}[\varphi]$")
    ax.legend()
    fig.tight_layout()
    _save(fig, "s1_susceptibility_calibration", "Supp / calibration",
          "exp_p1_calibrate_lambda (χ re-derived, same seeds)", supp=True)


# ---------------------------------------------------------------------------
# s2 — N*(E) with q CI (npz + qlin verdict)
# ---------------------------------------------------------------------------
def fig_s2():
    z = np.load(RES / "q_sweep_linear/nstar_boot.npz", allow_pickle=True)
    v = _load("q_sweep_linear/verdict.json")
    E = np.array(v["E_vals"]); nstar = np.array(v["nstar_point"])
    ci = np.array(v["nstar_ci95"]); interior = np.array(v["interior"], dtype=bool)
    q, qlo, qhi = v["q_point"], v["q_ci95"][0], v["q_ci95"][1]
    yerr = np.abs(ci.T - nstar)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.errorbar(E[interior], nstar[interior], yerr=yerr[:, interior], fmt="o",
                capsize=3, color="C0", label="interior")
    if (~interior).any():
        ax.errorbar(E[~interior], nstar[~interior], yerr=yerr[:, ~interior], fmt="x",
                    color="gray", alpha=0.6, label="edge")
    ax.axhspan(3.0, 18.0, color="green", alpha=0.06)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$E_{\rm protocol}/k_BT$"); ax.set_ylabel(r"$N^*$")
    ax.plot([], [], " ", label=f"q = {q:.3f} [{qlo:.3f}, {qhi:.3f}]")
    ax.legend()
    fig.tight_layout()
    _save(fig, "s2_Nstar_vs_E", "Supp / P1 q-sweep",
          "exp_p1_scaling (cached)", supp=True)


# ---------------------------------------------------------------------------
# s3 / s4 — AIS(N) profiles (ais_profile.csv)
# ---------------------------------------------------------------------------
def _fig_ais(model, stem):
    rows = [r for r in csv.DictReader(open(RES / "ais/ais_profile.csv"))
            if r["model"] == model]
    Evals = sorted(set(r["E"] for r in rows), key=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for E in Evals:
        rr = [r for r in rows if r["E"] == E]
        rr.sort(key=lambda r: int(r["N"]))
        N = [int(r["N"]) for r in rr]
        regime = rr[0]["regime"]; nstar = float(rr[0]["N_star"])
        color = "C0" if regime == "plateau" else "C3"
        ls = "-" if regime == "plateau" else "--"
        lbl = f"E={float(E):.3g} ({regime}) $N^*$={nstar:.1f}"
        axes[0].plot(N, [float(r["ais_lag1"]) for r in rr], ls, color=color,
                     marker="o", ms=3, label=lbl)
        axes[1].plot(N, [float(r["normalized"]) for r in rr], ls, color=color,
                     marker="o", ms=3, label=lbl)
        for ax in axes:
            ax.axvline(nstar, color=color, alpha=0.2, lw=1)
    axes[0].set_ylabel("AIS = I($x_t;x_{t-1}$) [bits]")
    axes[1].set_ylabel("normalized AIS = MI/H")
    for ax, ltr in zip(axes, ("(a)", "(b)")):
        ax.set_xlabel("N"); ax.grid(alpha=0.3); ax.legend(fontsize=7); _panel(ax, ltr)
    fig.tight_layout()
    _save(fig, stem, "Supp / AIS retrospective", "exp_ais_retro (cached)", supp=True)


# ---------------------------------------------------------------------------
def main():
    print("=== Regenerating publication figures ===")
    figs = [("f1", fig_f1), ("f2", fig_f2), ("f3", fig_f3), ("f4", fig_f4),
            ("f5", fig_f5), ("f6", fig_f6), ("f7", fig_f7), ("s1", fig_s1),
            ("s2", fig_s2), ("s3", lambda: _fig_ais("linear", "s3_ais_vs_N_linear")),
            ("s4", lambda: _fig_ais("pairwise", "s4_ais_vs_N_pairwise"))]
    for tag, fn in figs:
        try:
            print(f"[{tag}]")
            fn()
        except Exception as ex:
            print(f"  FAILED {tag}: {type(ex).__name__}: {ex}")

    # MANIFEST
    PAPER.mkdir(parents=True, exist_ok=True)
    with open(PAPER / "MANIFEST.txt", "w") as f:
        f.write("file -> paper section -> source\n")
        for name, sec, src in MANIFEST:
            f.write(f"{name}  ->  {sec}  ->  {src}\n")
    print("\n=== files written ===")
    for name, sec, src in MANIFEST:
        print(f"  {name:42s} | {sec}")

    # check #4: re-derived numbers match cached verdicts
    print("\n=== number-match checks ===")
    for label, val, ok in CHECKS:
        print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: {val}")

    # SELFCHECK_SENTINEL  (everything below is the self-check; excluded from scan)
    # check #2: scan ONLY the figure-plotting code (above the sentinel), so the
    # self-check's own pattern strings are not matched.
    import re
    src = Path(__file__).read_text()
    fig_code = src.split("# SELFCHECK_SENTINEL")[0]
    title_calls = re.findall(r"\.(?:set_title|suptitle)\(", fig_code)
    codenames = ["FALSEADA", "INCONCLUSO", "DISSOLUTION", "REHABILIT", "VERIFICADA",
                 "specialized=", "recovers=", "bound="]
    cn_hits = [t for t in codenames if t in fig_code]
    print("\n=== title / codename check (figure code only) ===")
    print(f"  real .set_title()/.suptitle() calls: {len(title_calls)} "
          f"(expected 0 => no figure carries a title)")
    print(f"  internal verdict codenames in plotted text: {cn_hits if cn_hits else 'NONE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
