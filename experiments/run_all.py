"""
experiments/run_all.py — reproduce the whole bench from scratch and write REPORT.md.

Runs every experiment in dependency order, then regenerates REPORT.md via
edh.report. WARNING: a full run is heavy (hours): the q-sweeps and the Track-B
PID/AIS ensembles dominate. Use --skip-heavy to only (re)generate the report from
existing results/.

Each step is a module with a main(); failures are reported but do not abort the
rest (so a partial reproduction still yields a partial report).
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "experiments", _ROOT / "reference"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# (module, note) in dependency order. lambda calibration is MANUAL-freeze; we run
# it for reproducibility but the frozen lambda in preregistration.yaml is used.
STEPS = [
    ("exp_p1_calibrate_lambda", "P1 lambda calibration (linear)"),
    ("exp_p1_decompose_cost", "P1 cost decomposition (axis A)"),
    ("exp_p1_scaling", "P1 q-sweep linear (axis B)"),
    ("exp_p1_regimes", "P1 linear regime re-analysis"),
    ("exp_p3_entropy_crossover", "P3 entropy crossover"),
    ("exp_ais_retro", "AIS retrospective (Track A, negative)"),
    ("exp_p1_scaling_pairwise", "P1 q-sweep pairwise + 1/6-vs-1/4"),
    ("exp_p2_synergy_onset", "P2/H2 onset (emergent coupling)"),
    ("exp_p2_rg", "C2-RG/C3 fixed-boundary (dissolution)"),
    ("exp_p2_rg_plastic", "P2_RG_plastic (section 6.5)"),
    ("exp_p2_beta_robustness", "section-6 beta gate"),
    ("exp_robustness_trackb", "Track-B robustness battery"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-heavy", action="store_true",
                    help="only regenerate REPORT.md from existing results/")
    args = ap.parse_args()

    if not args.skip_heavy:
        for mod, note in STEPS:
            print(f"\n=== {mod}: {note} ===", flush=True)
            t0 = time.time()
            try:
                m = importlib.import_module(mod)
                m.main()
                print(f"  done in {time.time()-t0:.0f}s")
            except Exception as e:  # keep going; partial report is still useful
                print(f"  FAILED: {type(e).__name__}: {e}")

    from edh import report
    report.main()
    print("\nREPORT.md regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
