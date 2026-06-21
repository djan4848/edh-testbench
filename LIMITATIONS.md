# LIMITATIONS — what this bench does and does NOT establish

Honest scope notes. A positive result here supports a *specific operationalization*
of an EDH prediction; it does not prove the paper's global thesis. Updated as
phases land.

## Class of model
- **Track A (P1/P3)** is a well-mixed evolutionary signaling game with an explicit
  energy ledger (Landauer + analog separation energy). It is *not* a spatial or
  network model, and it does not learn a lexicon (a shared code convention per
  modality is assumed, isolating the analog-vs-digital energy/error tradeoff,
  which is what P1 is about). The agents' only adaptive choice is digitality.
- Units: k_B = 1; energies/temperatures are dimensionless multiples of k_B.

## P1 cost exponent p — the central modeling dependency
The measured analog cost exponent decomposes as **p = a(1+b)** where
`a = dlog M / dlog N` (signal demand) and `b = dlog(cost_per_signal)/dlog M`
(per-signal cost geometry). Measured (results/diagnostics/):

| Model | a | b | p |
|-------|---|---|---|
| **Linear demand M=N (PRIMARY)** | 1.00 | 1.64 (→2) | **2.64** |
| Pairwise demand M~N² (VARIANT) | 2.16 | 1.92 | **6.30** |
| Paper eq.(12) | 2 | 2 | 6 |

- The 2.64-vs-6 gap is **entirely the demand exponent a** (1 vs 2). The per-signal
  cost geometry **b≈2 matches the paper in both models**. So the paper's claim that
  analog coding is "thermodynamically unaffordable" (N⁶) is real **only under
  Nowak's quadratic pairwise-demand assumption** (a distinct signal per pair). With
  conservative linear demand (one meaning per agent) the analog penalty is N^2.64.
- **The pairwise variant (M~N²) is a COMPARISON, not the primary model.** It exists
  to isolate that N⁶ hinges on the demand assumption. It is not used to decide any
  verdict. Choosing linear demand as primary is the conservative choice (§9 of the
  spec): it makes the analog code *cheaper*, i.e. it is the choice that most easily
  REFUTES the EDH (a transition driven by cost is harder to produce when analog is
  cheap), not the one that favors it.

## What a positive P1 (q≈1/p) would and would NOT prove
- Would show: in this model the analog→digital threshold is set by the energetic
  dominant balance k_B·T·N*^p ≈ E_protocol (the mechanism the paper claims as new).
- Would NOT show: that p≈6, nor that real communication systems obey this; nor that
  the transition is "thermodynamically forced" in any system with sub-quadratic
  demand. q≈1/p is largely guaranteed *if* the transition is cost-driven; the test
  earns its keep by being able to FAIL when the threshold is instead set by the
  coordination-error limit (then q deviates from 1/p, or the power guards fail).

## Where the model may bias the result
- **Toward refutation (conservative):** linear demand keeps analog cheap; the
  separation geometry is harmonic (Σ amplitude²), not the steepest defensible law.
- **Toward confirmation:** a shared-code assumption removes lexicon-learning
  failure, which could otherwise add coordination noise that swamps the energy
  signal; mean-field mixing makes the transition cleaner than a spatial model would.
- **λ calibration:** λ sets the prefactor/sharpness of N* (where it sits), not the
  slope q. It was frozen (λ=1.0) by a shape-only criterion before the q-sweep, so it
  cannot bias q. Robustness across λ is tested separately.

## P1 threshold driver (q-sweep, primary/linear model) — RESOLVED BY REGIME
The N*(E) log-linearity check fired (convex), so a single global slope q is not a
valid power-law statistic. N*(E) is **two-regime** and was resolved accordingly
(results/q_sweep_linear/regime_verdict.json):

| Regime | E range | slope | meaning |
|--------|---------|-------|---------|
| **Plateau** | E ≲ 9 | q_low = −0.005 [−0.041, +0.028] | E-INDEPENDENT: Nowak coordination-error ceiling, N*≈5.6 |
| **Rise** | E ≳ 9 | q_high = +0.258 [+0.199, +0.380] | overlaps 1/p_high=0.343: Landauer cost-driven |

Verdicts:
- **H3-strong (entropy as the SOLE driver): FALSEADA**, robustly. The existence of
  the E-independent plateau shows an informational ceiling (Nowak) drives the
  transition at low E with no energetic dependence. This is the primary conclusion
  and does not depend on the fine slopes — only on the plateau existing.
- **Cost-driven mechanism in the energetic regime: VERIFICADO.** For E above the
  knee, q_high's CI overlaps 1/p_high and excludes 0 — the Landauer dominant
  balance does govern N* when the protocol is expensive.
- Net: the paper's TWO-ceiling picture (Nowak error limit + Landauer) is
  SUPPORTED; its strong single-driver claim (H3) is FALSIFIED. The pure-coordination
  N* (plateau ≈5.6) sits inside the 5–20 band the paper anticipates.

Caveats / honesty (the three that qualify the "cost-driven VERIFICADO"):
1. **q_high is VERIFICADO-but-weak.** Its CI [0.199, 0.380] *overlaps* 1/p_high
   (0.343) and excludes 0, but the point estimate (0.258) sits *below* 1/p_high.
   The overlap is partly the CI width, not a centered match.
2. **The q_high < 1/p gap is degenerate** between two non-exclusive causes:
   (a) residual coordination contribution at high E, and (b) the digital code's own
   N²·L (Landauer) cost not being negligible vs E_protocol, which by itself makes
   q < 1/p without any coordination effect. The primary sweep does not separate
   these; the pairwise sweep's R(E)=E_protocol/D_digital(N*) dominance check is
   designed to disambiguate.
3. **The switch to uniform weights was an unregistered analyst decision.** The
   pre-registration did not specify the N* weighting. Inverse-variance weighting
   gave pathological fits (quantization-tight per-point CIs over-weighting a few
   snapped points), so I used uniform weights + point-resampling. This is a
   defensible but post-hoc choice; it is flagged, not hidden.
- Both fits remain convex over the window, so q_high is a LOCAL/asymptotic slope;
  at the top of the range its local slope approaches 1/p_high.
- The global q≈0.25 (and the earlier 0.235) sit near the paper's eq.(15) value
  0.25 — but that is a two-regime *mixing* coincidence; the real test is q_high vs
  1/p_high. This vindicates designing the test as q≈1/p rather than pre-registering
  a fixed number, which would have spuriously "confirmed" 0.25.
- Per-point N* bootstrap CIs are quantization-tight (χ_max≈0.01 snaps the peak);
  the honest noise is between-point scatter, so regime slopes use uniform weights
  and a point-resampling bootstrap. λ-robustness and a finer sweep are deferred to
  the Phase-4 robustness battery.
- Scope: linear-demand model only. The paper's literal pairwise (M~N²) model is a
  separate sweep with its own recalibrated λ.

## P1 pairwise (paper's literal M~N²) — cost regime is clean
The pairwise q-sweep (results/q_sweep_pairwise/) gives a much cleaner cost regime
than the linear model (χ_max≈0.07 vs 0.013; two N* estimators agree):
- **Cost-driven VERIFICADO cleanly**: q_high=0.165 [0.158,0.177] is centered on
  1/p_high=0.159 (p_high≈6.30), R(E)≈4125≫1 so the toolkit dominates and the gap
  ambiguity of the linear model does not arise.
- **H3 single-driver still FALSEADA**: a low-E plateau (N*≈4.7) persists.
- **Adjudication**: q_high≈0.165 supports 1/6, rejects eq.(15)'s 1/4 ⇒ that 1/4 is
  an erratum given eq.(12)'s N⁶.
Caveats specific to this sweep:
- The N-grid was widened (2→26) and the usable band made grid-relative to avoid
  edge-censoring N*~18-20; this is an analyst decision (flagged) made to satisfy
  the min_logN_star_range guard, not to change the slope (q_high was already
  0.162–0.165 before/after the widening).
- The pairwise demand M~N² is the paper's literal assumption but is itself the
  insertion that the cost decomposition flags as the source of N⁶ (Nowak's own M
  is vocabulary size, not per-pair codes). So "pairwise verifies cost-driving" is a
  statement about the paper's *stated* model, not an endorsement of its demand
  assumption.

## P3 entropy crossover — regime-limited, not a global VERIFICADA
P3 (analog dS/dt exceeds digital "for the first time" at the transition) holds
ONLY in the pairwise cost regime (N_cross=N* exactly, 100% within ±1). It is not
verified in the linear cost regime (N_cross≈0.8 below N*, consistent with the
linear transition being only weakly cost-driven) and fails in both coordination
regimes. Everywhere N_cross < N* (the crossover precedes the transition) — the
opposite sign to the originally sketched expectation; reported as a finding, not
forced. Interpretation: the population keeps the energetically-costlier analog code
past the energy crossover because it still coordinates better, until the Nowak
ceiling at N*. Do NOT report P3 as globally VERIFICADA.

## AIS retrospective (Active Information Storage) — NEGATIVE, with a caveat
Tested (not assumed): whether the analog->digital transition is a dip-recovery of
AIS = I(x_t; x_{t-1}), and whether AIS is a MORE universal order parameter than
digitality (same signature across regimes and models). Result (results/ais/):
**NOT supported.**
- Raw AIS dip near N* in **0/8** profiles; normalized in 2/8 (and those at the
  grid edge, not at N*). `universal_normalized: false`.
- What AIS actually does: it **peaks at N\*** in the pairwise profiles and in the
  linear rise (e.g. plateau peak at N≈4≈N*=4.6), and declines monotonically in the
  linear plateau. I.e. raw AIS tracks the marginal entropy H(x_t) = the population
  digitality spread ≈ the susceptibility chi. It is a **trivial correlate of the
  order parameter**, not an independent dip-recovery, and not even consistent
  across regimes in the linear model.
- HONESTY (as required): "AIS moves at the transition" is trivial — it is the same
  information as chi. The non-trivial claim (a universal dip-recovery signature)
  FAILS. We do not sell the former as the latter.
- STATE-DEFINITION LIMITATION (why this test is weak here): the Track A per-agent
  state is the scalar digitality, so AIS measures self-predictability of the CODE
  CHOICE, not of the SIGNAL. In a pure phase all agents collapse to one digitality
  bin -> degenerate support -> AIS undefined, which STRUCTURALLY precludes the
  "AIS high and recovers in the digital phase" half of the hypothesis. A fair test
  needs a per-agent signal-level state that stays rich in both phases (Track B's
  bit-vector automata), which the q-sweep dynamics do not produce.
- Consequence: the thermodynamic-bridge test (Still 2012, I_nopred vs measured
  dissipation) was NOT opened — it was gated on a clean (a)/(b), which did not
  occur. It remains a candidate on Track B, where the state is signal-level.

## P2 / H2 Track B (C1) — coupling must be emergent, and then H2 fails
- **The XOR coupling was circular and is rejected.** An imposed per-bit XOR gate
  produced a clean synergy onset, but XOR *is* the pure-synergy gate (PID Syn≈1,
  Unq=Red=0), so it measured the injected operator, not emergence. The honest model
  is the PICARD coupling where the neighbour perturbs the INPUT to the state-read
  rule; its coupled-phase PID is a parameter-dependent MIX (non-circularity test in
  tests/test_track_b.py asserts it is not vertex-pinned).
- **Under emergent coupling H2 is FALSIFIED:** synergy does not rise (dSyn≈0, no
  onset), and unique is SUPPRESSED (dUnq≈−0.06), in both symmetric and asymmetric
  conditions. The reparto change is "de-individuation" (unique down, no synergy/
  redundancy gain), not synergistic entanglement.
- **β is fixed.** The endogenous credit→β feedback collapses the coupled phase to a
  degenerate cycle (a real finding, not hidden); β is fixed in Track B and the
  result is checked to be qualitatively stable across β≈1.5–3.0.
- **A detector false-positive was caught (preserved as a methods note).** The literal
  "first 3σ crossing" onset has ~35% false positives here (a single noise sample
  crossing a tiny-σ baseline); a sustained-crossing onset (3 windows + 0.02-bit
  floor) plus the three controls removes it. The sustained-crossing refinement is an
  UNREGISTERED analyst decision, flagged.
- **Scope:** one model class (fixed-rule stochastic self-referential automata), one
  operationalization, one PID core/target choice. A negative here refutes H2 for
  this class; it does not preclude H2 under learning/plasticity dynamics.

## P2_RG (C2-RG/C3) — dissolution verdict and its caveats
- **The decisive, robust half is unit-formation = NO.** Size-controlled
  ΔAIS_unit = −0.65 (CI excludes 0), and the time-resolved unit AIS dips at t_c
  with no recovery. Coupling dissolves unit self-predictability; it does not build
  a coherent macro-unit. This drives the DISSOLUTION_DEAD verdict.
- **The bridge half is weaker / proxy-dependent.** The measured Landauer
  dissipation (update determinism, ln2·(1−H₂(p_on))) is coupling-INVARIANT in this
  model because the coupling reroutes which rule fires without changing field
  magnitudes. So I_nopred grows with coupling while dissipation does not — the
  Still bound holds only trivially and the correlation is undefined. A different
  dissipation accounting could change the bridge sub-result; the dissolution
  sub-result does not depend on it. Flagged so the verdict is not over-read.
- **Scope / §6 framing (important):** the dissolution is for the FIXED-BOUNDARY
  PICARD model, which omits §6.5 (co-evolving plastic filters, eq.20) — the paper's
  OWN unit-formation mechanism. So this does NOT undermine §6; the §6 status is
  PENDING until the plastic-boundary (autopoietic) model is tested (P2_RG_plastic).
  DISSOLUTION_DEAD applies only to the AIS/I-operator idea on the fixed-boundary
  model. The next RG scale (PID between two coupled pairs) stays closed.

## P2_RG_plastic (§6.5) — rehabilitation caveats
- The §6 REHABILITATED verdict rests on a sign flip of ΔAIS_unit (−0.65 fixed →
  +0.11 plastic) that is CI-clean but MODEST in absolute size (+0.108 bits). It
  shows unit formation occurs with boundary plasticity, not its magnitude/robustness.
- The plasticity rule (Hebbian "attend to the neighbour phase that agrees with
  self") is one concrete realization of eq.20, chosen to be emergent and to pass a
  pre-registered specialization precondition; other autopoietic boundary rules could
  differ. The precondition's noise-control has a tie-breaking baseline
  (sel_noise≈0.28 from argmax favouring address 0), but the real−noise gap (0.17) is
  the clean partner-specific signal.
- §6 verdict is REHABILITATED-BUT-REGIME-DEPENDENT (robustness battery):
  * NECESSITY of boundary plasticity is robust/strong — fixed boundary dissolves at
    every tested β∈[1,5] and coupling (ΔAIS_unit −0.13 to −0.88).
  * SUFFICIENCY is narrow: the flip holds in a β-WINDOW ~[1,3] (plastic goes
    negative at β=4: −0.24, β=5: −0.49 — non-monotone, dissolves above the window),
    needs strong coupling (no flip at c=0.5; flip at c=1.5), and is so far
    Hebbian-specific (the one non-Hebbian variant, boundary-consensus, FAILED its
    specialization precondition → invalid test of generality; generality OPEN).
  Lead with the sign and the fixed-vs-plastic separation, not the small plastic
  magnitudes (deterministically tight CIs). Not a robust-macro-unit claim. (The
  endogenous credit→β feedback remains disabled, so the Still energy bridge is
  UNTESTED/OPEN, not dead.)
- Scope: the result is "the RG unit-forming step is realizable, and requires
  boundary plasticity, in this model class," not a universal claim.

## Determinism / CI caveat
The analog cost law is deterministic given the micro-model (full signal coverage),
so a seed bootstrap of p is degenerate. The reported CI is an N-grid bootstrap,
reflecting finite-N curvature (b is still climbing toward its asymptotic 2 over
N=2..20), which is the uncertainty that actually matters for the exponent.
