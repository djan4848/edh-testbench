# EDH test bench — falsifying the Entropic Drive Hypothesis

A reproducible bench that **simulates and tries to refute** the three falsifiable
predictions of the EDH working paper. Success is honest machinery that *could*
refute the paper; a well-measured negative is a success. See the project spec
and `preregistration.yaml` (frozen) for the decision rules.

| Pred | Claim (one line) | Track |
|------|------------------|-------|
| **P1** | Coordination collapse at a critical `N*`, then combinatorial code, with `N* ≈ (E_protocol/k_B T)^{1/4}` | A |
| **P3** | At the transition, analog entropy production `dS/dt` first overtakes digital | A |
| **P2** | When two self-referential systems couple, synergy (PID) rises *before* unique info | B |

## Anti-cheat rules (§3 of the spec)
No exogenous truncation; cost derived from a micro-model (never `N**6`); `N*`
measured as the susceptibility peak; ensemble-at-fixed-t PID; **local** coupling;
pre-registered statistics; everything seeded; negatives reported.

## Setup

The project lives on an **exFAT SSD**, which has no symlinks — so `python -m venv`
cannot create its venv here. The venv therefore lives on the home filesystem and
the code stays on the SSD.

```bash
# 1. Create the venv off the SSD (one time)
python3 -m venv /home/neuraldyn/.venvs/edh
. /home/neuraldyn/.venvs/edh/bin/activate
export PIP_USER=0                      # a global PIP_USER=1 breaks venv installs
pip install -r requirements.txt

# 2. BROJA_2PID is NOT on PyPI — install from source (one time)
git clone https://github.com/Abzinger/BROJA_2PID /home/neuraldyn/.venvs/BROJA_2PID
pip install -e /home/neuraldyn/.venvs/BROJA_2PID   # installs as package `broja2pid`

# 3. Thereafter, just:
. activate.sh
```

`edh/pid.py` registers a `BROJA_2PID` module alias pointing at the installed
`broja2pid` package, so the validated reference wrapper imports unchanged.

> Note: environment is Python **3.10.12**, not the 3.11+ the spec requested
> (declared deviation — no 3.11-only syntax is used; recorded in LIMITATIONS.md).

## Verify the install (Phase 0)

```bash
python reference/pid_fast.py          # canonical-gate self-test (XOR=synergy, …)
pytest tests/test_pid_gates.py tests/test_determinism.py -v
```

## Layout
```
edh/        pid (BROJA wrapper), entropy, track_a/_b, finite_size, stats, report
experiments/ exp_p1_threshold, exp_p1_scaling, exp_p3_entropy_crossover,
             exp_p2_synergy_onset, run_all
reference/  validated modules (pid_fast.py, dynamics_stochastic.py) — do not reinvent
tests/      pytest acceptance per phase
results/    CSVs, .npz, figures (gitignored except summaries)
```

## Status
- **Phase 0 — scaffolding + solver verification: DONE.** BROJA gates correct
  (XOR→Syn≈1, COPY→U1≈1, DUP→Red≈1); determinism verified.
- **Phase 1 — Track A machinery: DONE.** 23/23 tests green. Emergent analog→digital
  transition; cost exponent **p=2.64** (separation-dominated, clean power law).
  λ calibrated & frozen at **1.0** (N*≈10.05). Cost decomposition: p=a(1+b) with
  a=1 (linear demand), b≈2 (per-signal); pairwise demand M~N² recovers p≈6.3 — the
  paper's N⁶ hinges entirely on the quadratic-demand assumption (see LIMITATIONS.md).
- **Phase 2 — P1 q-sweep (primary, linear demand): DONE, resolved by regime.**
  N*(E) is two-regime (global fit convex → local). Plateau (E≲9): q_low≈0 →
  Nowak coordination ceiling, E-independent. Rise (E≳9): q_high=0.258
  [0.199,0.380] overlaps 1/p_high=0.343 → Landauer cost-driven. **H3 single-driver
  FALSEADA; two-ceiling picture SUPPORTED; cost-driven VERIFICADO in the energetic
  regime.** See results/q_sweep_linear/ and REPORT.md.
- **Phase 2b — P1 q-sweep (pairwise, paper's literal M~N²): DONE.** Recalibrated
  λ (auto E_center=3799). Two-regime: plateau (N*≈4.7) + rise q_high=0.165
  [0.158,0.177] = 1/p_high=0.159 → **cost-driven cleanly VERIFICADO** (R≈4125≫1).
  **Adjudication: q≈1/6 supported, eq.(15)'s 1/4 rejected** → 1/4 is an erratum
  given N⁶. H3 single-driver FALSEADA. See results/q_sweep_pairwise/.
- **Phase 3 — P3 entropy crossover (close-out Track A): DONE, regime-limited.**
  N_cross=N* exactly in the pairwise cost regime (VERIFICADA); not verified
  elsewhere, with N_cross<N* (crossover precedes transition — system overpays for
  coordination). P3 holds only where cost drives the transition. results/p3_entropy_crossover/.
- **AIS retrospective: DONE — NEGATIVE.** AIS=I(x_t;x_{t-1}) on the digitality
  state tracks χ (peaks at N*), shows no universal dip-recovery (0/8 raw), and is
  inconsistent across regimes. Descriptive correlate, not a fundamental order
  parameter; pure phases degenerate (state-definition limit). Bridge test not
  opened. See results/ais/ and LIMITATIONS.md.
- **Phase C0–C1 — Track B (P2/H2) synergy onset: DONE — H2 FALSIFIED (emergent).**
  Machinery: stochastic self-referential PICARD automata where the neighbour
  perturbs the INPUT to the state-read rule (EMERGENT coupling; an XOR gate was
  rejected as circular — XOR is the pure-synergy gate). Non-circularity verified
  (coupled PID is a param-dependent mix, not a vertex). Result (sym + asym, 30
  seeds, 3 controls): synergy does NOT rise on coupling (dSyn≈0, no onset); unique
  is SUPPRESSED (dUnq≈−0.06). De-individuation, not synergy. results/p2_synergy_onset/.
- **Phase C2-RG/C3 — coarse-graining vs dissolution (fixed boundary): DONE —
  DISSOLUTION_DEAD.** ΔAIS_unit=−0.65 (CI excludes 0); unit AIS dips at t_c, no
  recovery. DEAD for the AIS/I-operator idea on THIS model. **§6 status PENDING**
  (the fixed-boundary model omits §6.5 plasticity / eq.20). results/p2_rg/.
- **Phase P2_RG_plastic + Phase-4 robustness — DONE — §6 REHABILITATED-BUT-REGIME-
  DEPENDENT.** Plastic boundary flips ΔAIS_unit positive (dip WITH recovery) at
  β=2.5, BUT the robustness battery shows: flip only in a β-window ~[1,3] (negative
  at β=4,5), needs strong coupling (fails at c=0.5), and is Hebbian-specific so far
  (the non-Hebbian consensus variant failed its specialization precondition).
  **Necessity of boundary plasticity is robust; sufficiency is narrow/regime-bound.**
  The Still energy bridge is OPEN (untested by construction). results/p2_rg_plastic/,
  results/robustness/.
- **report.py:** `python -m edh.report` regenerates REPORT.md from results/.
  `python experiments/run_all.py [--skip-heavy]` reproduces the bench + report.

## Commands per experiment
```bash
pytest -q                                       # all acceptance tests
python experiments/exp_p1_calibrate_lambda.py   # λ calibration (chi_curves.png)
python experiments/exp_p1_decompose_cost.py     # cost decomposition diagnostic
python experiments/exp_p1_scaling.py --n-seeds 30   # P1 q-sweep (primary)
```
