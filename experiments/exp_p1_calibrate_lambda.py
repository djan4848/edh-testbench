"""
experiments/exp_p1_calibrate_lambda.py
======================================
INSTRUMENT CALIBRATION, not a hypothesis test. Its only job is to pick
``lambda_tradeoff`` so the transition is "on screen" with a clean susceptibility
peak. It does NOT measure or report q. By construction it cannot bias q: lambda
shifts the prefactor and the sharpness of N*, not the slope of log(N*) vs
log(E/kT). The selection depends ONLY on the *shape* of chi(N) (interior,
unimodal, sharp), never on any hypothesis outcome.

Outputs (results/calibration/):
  chi_curves.png         chi(N) per lambda with interpolated N* + verdict.
  lambda_calibration.csv  one row per lambda with all machine-checked metrics.
  chosen_lambda.json     chosen lambda + provenance, or {chosen: null, reason}.

Freezing is MANUAL: this script prints the chosen value; the user pastes it into
preregistration.yaml (lambda_tradeoff). It never writes the pre-registration.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "reference"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from edh.finite_size import PeakReport, peak_report_from_chi, susceptibility  # noqa: E402
from edh.track_a_signaling import (  # noqa: E402
    TrackAParams,
    analog_cost_law,
    meanings_for_N,
    meanings_pairwise,
    run_population,
)

DEMANDS = {"linear": meanings_for_N, "pairwise": meanings_pairwise}


# ---------------------------------------------------------------------------
# Configuration (calibration spec §1)
# ---------------------------------------------------------------------------
@dataclass
class CalibConfig:
    demand_name: str = "linear"
    # E_protocol_center: if None, auto-set to the analog cost at the target-band
    # midpoint so the digital toolkit and the analog cost balance near N~10 (the
    # linear model lands ~40; the pairwise M~N^2 model is ~100x larger).
    e_protocol_center: float | None = None
    T: float = 1.0
    lambdas: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    n_grid: tuple[int, ...] = tuple(range(2, 21))
    n_seeds: int = 20
    tail_frac: float = 0.2
    target_band: tuple[float, float] = (8.0, 12.0)
    accept_band: tuple[float, float] = (3.0, 18.0)
    fwhm_max: float = 8.0
    unimodal_ratio: float = 2.0

    @property
    def demand(self):
        return DEMANDS[self.demand_name]

    @property
    def outdir(self) -> Path:
        sub = "calibration" if self.demand_name == "linear" \
            else f"calibration_{self.demand_name}"
        return _ROOT / "results" / sub

    def resolve_e_center(self) -> float:
        if self.e_protocol_center is not None:
            return self.e_protocol_center
        n_mid = int(round(sum(self.target_band) / 2))
        self.e_protocol_center = float(
            analog_cost_law(n_mid, TrackAParams(temperature=self.T),
                            demand=self.demand))
        return self.e_protocol_center


@dataclass
class LambdaRow:
    lam: float
    n_peak: float
    n_peak_int: int
    chi_max: float
    fwhm: float
    unimodal_ratio: float
    interior: bool
    sharp: bool
    centered: bool
    unimodal: bool
    converged: bool
    valid: bool
    chi: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


# ---------------------------------------------------------------------------
# Core: run one lambda, build phi[N, seed], chi(N), and the peak metrics
# ---------------------------------------------------------------------------
def run_one_lambda(lam: float, cfg: CalibConfig, params: TrackAParams) -> LambdaRow:
    n_grid = np.array(cfg.n_grid, dtype=float)
    phi = np.zeros((len(cfg.n_grid), cfg.n_seeds))
    conv = np.zeros((len(cfg.n_grid), cfg.n_seeds), dtype=bool)
    for ni, n in enumerate(cfg.n_grid):
        for s in range(cfg.n_seeds):
            r = run_population(int(n), cfg.e_protocol_center, lam, params,
                               seed=s, demand=cfg.demand)
            phi[ni, s] = r.phi
            conv[ni, s] = r.converged
    chi = susceptibility(phi)
    peak = peak_report_from_chi(
        n_grid, chi,
        accept_band=cfg.accept_band, target_band=cfg.target_band,
        fwhm_max=cfg.fwhm_max,
    )
    return classify_lambda(lam, n_grid, chi, peak, conv.mean(), cfg)


def classify_lambda(
    lam: float,
    n_grid: np.ndarray,
    chi: np.ndarray,
    peak: PeakReport,
    converged_frac: float,
    cfg: CalibConfig,
) -> LambdaRow:
    """Apply the machine-checkable validity criteria (calibration §3)."""
    unimodal = peak.unimodal_ratio >= cfg.unimodal_ratio
    converged = converged_frac >= 0.7
    valid = bool(peak.interior and unimodal and peak.sharp and converged)
    return LambdaRow(
        lam=float(lam),
        n_peak=peak.n_star,
        n_peak_int=peak.n_peak_int,
        chi_max=peak.chi_max,
        fwhm=peak.fwhm,
        unimodal_ratio=peak.unimodal_ratio,
        interior=peak.interior,
        sharp=peak.sharp,
        centered=peak.centered,
        unimodal=unimodal,
        converged=converged,
        valid=valid,
        chi=np.asarray(chi, dtype=float),
    )


# ---------------------------------------------------------------------------
# Selection (calibration §4) + failure diagnosis (§5) -- pure, unit-tested
# ---------------------------------------------------------------------------
def select_lambda(rows: list[LambdaRow]) -> tuple[float | None, dict]:
    """Among valid rows pick the SMALLEST lambda (least adversarial). Returns
    (chosen_lambda or None, provenance/diagnosis dict)."""
    valid = [r for r in rows if r.valid]
    if valid:
        chosen = min(valid, key=lambda r: r.lam)
        centered = [r.lam for r in valid if r.centered]
        return chosen.lam, {
            "chosen_lambda": chosen.lam,
            "rule": "smallest lambda among valid (interior & unimodal & sharp & converged)",
            "valid_lambdas": [r.lam for r in valid],
            "centered_lambdas": centered,
            "n_peak_at_chosen": chosen.n_peak,
        }
    return None, {"chosen_lambda": None, "reason": diagnose_failure(rows)}


def diagnose_failure(rows: list[LambdaRow]) -> str:
    """Diagnose WHY no lambda was valid (do not patch -- §5)."""
    if not rows:
        return "no lambdas evaluated"
    peaks = np.array([r.n_peak for r in rows])
    fwhms = np.array([r.fwhm for r in rows])
    interiors = [r.interior for r in rows]
    if all(p >= 18.0 for p in peaks):
        return ("all chi peaks pinned to the HIGH edge (N*~20): analog too cheap "
                "-> lower e_protocol_center or revisit the separation geometry "
                "(do NOT just raise lambda)")
    if all(p <= 3.0 for p in peaks):
        return ("all chi peaks pinned to the LOW edge (N*~2): dissipation already "
                "dominates at N=2 -> lower the lambda floor (try 0.03, 0.01)")
    if all(f > 8.0 for f in fwhms):
        return ("all peaks too wide (fwhm>fwhm_max): selection too soft -> raise "
                "lambda or increase n_seeds/horizon; if even max lambda won't "
                "sharpen, phi may be a poor order parameter")
    if not any(interiors):
        return "no interior peak found; transition not on-screen in N=2..20"
    return ("peaks present but failing unimodality/convergence; inspect "
            "chi_curves.png and consider more seeds")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_outputs(rows: list[LambdaRow], chosen, provenance, cfg: CalibConfig,
                  params: TrackAParams) -> None:
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    od = cfg.outdir
    od.mkdir(parents=True, exist_ok=True)
    n_grid = np.array(cfg.n_grid, dtype=float)

    # CSV
    with open(od / "lambda_calibration.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "N_peak", "fwhm", "unimodal_ratio", "interior",
                    "sharp", "centered", "unimodal", "converged", "valid"])
        for r in rows:
            w.writerow([r.lam, f"{r.n_peak:.3f}", f"{r.fwhm:.3f}",
                        f"{r.unimodal_ratio:.3f}", r.interior, r.sharp,
                        r.centered, r.unimodal, r.converged, r.valid])

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        style = "-" if r.valid else "--"
        lbl = (f"λ={r.lam:g}  N*={r.n_peak:.1f}  "
               f"{'VALID' if r.valid else 'rej'}"
               f"{' (centered)' if r.centered else ''}")
        line, = ax.plot(n_grid, r.chi, style, marker="o", ms=3, label=lbl)
        ax.axvline(r.n_peak, color=line.get_color(), alpha=0.25, lw=1)
    ax.set_xlabel("N (agents / meanings)")
    ax.set_ylabel(r"$\chi(N)=\mathrm{Var}_{\mathrm{seeds}}[\varphi]$")
    title = "P1 λ-calibration: susceptibility curves"
    if chosen is not None:
        title += f"   chosen λ={chosen:g}"
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(od / "chi_curves.png", dpi=130)
    plt.close(fig)

    # JSON
    payload = {
        "demand": cfg.demand_name,
        "e_protocol_center": cfg.e_protocol_center,
        "T": cfg.T,
        "n_seeds": cfg.n_seeds,
        "criteria": {
            "accept_band": cfg.accept_band, "target_band": cfg.target_band,
            "fwhm_max": cfg.fwhm_max, "unimodal_ratio": cfg.unimodal_ratio,
        },
        "params": asdict(params),
        "provenance": provenance,
    }
    payload.update(provenance)
    with open(od / "chosen_lambda.json", "w") as f:
        json.dump(payload, f, indent=2)


def usable_E_range(n_peak: float, p: float, cfg: CalibConfig) -> tuple[float, float]:
    """If p (cost exponent) is known, the E range keeping N* in accept_band:
    N* ~ (E/kT)^{1/p} => E_usable ~ E_center * (accept_band / N_peak)^p."""
    lo = cfg.e_protocol_center * (cfg.accept_band[0] / n_peak) ** p
    hi = cfg.e_protocol_center * (cfg.accept_band[1] / n_peak) ** p
    return lo, hi


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(n_seeds: int | None = None, demand_name: str = "linear") -> int:
    from tqdm import tqdm

    cfg = CalibConfig(demand_name=demand_name)
    if n_seeds is not None:
        cfg.n_seeds = n_seeds
    cfg.resolve_e_center()
    print(f"[demand={demand_name}] auto E_protocol_center = "
          f"{cfg.e_protocol_center:.3f}")
    params = TrackAParams(
        temperature=cfg.T, e_center=cfg.e_protocol_center,
        tail_frac=cfg.tail_frac,
    )
    rows: list[LambdaRow] = []
    for lam in tqdm(cfg.lambdas, desc="lambda calibration"):
        rows.append(run_one_lambda(lam, cfg, params))

    chosen, provenance = select_lambda(rows)
    write_outputs(rows, chosen, provenance, cfg, params)

    print("\n=== P1 lambda calibration ===")
    print(f"E_protocol_center = {cfg.e_protocol_center}, T = {cfg.T}, "
          f"n_seeds = {cfg.n_seeds}")
    for r in rows:
        print(f"  λ={r.lam:5g}  N*={r.n_peak:5.2f}  fwhm={r.fwhm:4.2f}  "
              f"uni={r.unimodal_ratio:5.2f}  interior={r.interior!s:5}  "
              f"sharp={r.sharp!s:5}  centered={r.centered!s:5}  "
              f"valid={r.valid!s:5}")
    key = ("lambda_tradeoff" if cfg.demand_name == "linear"
           else f"lambda_tradeoff_{cfg.demand_name}")
    if chosen is not None:
        print(f"\nCHOSEN lambda = {chosen}  (provenance: {provenance['rule']})")
        print(f"  N* at chosen = {provenance['n_peak_at_chosen']:.2f}")
        print(f"  -> freeze into preregistration.yaml P1.fixed_parameters.{key}")
        return 0
    print(f"\nNO VALID lambda. Diagnosis: {provenance['reason']}")
    return 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demand", choices=list(DEMANDS), default="linear")
    ap.add_argument("--n-seeds", type=int, default=None)
    args = ap.parse_args()
    raise SystemExit(main(n_seeds=args.n_seeds, demand_name=args.demand))
