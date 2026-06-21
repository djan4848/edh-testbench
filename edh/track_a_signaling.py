"""
edh.track_a_signaling — evolutionary signaling game for P1 / P3 (Track A).

N well-mixed agents must coordinate over M(N) meanings (coordination demand grows
with N). Each agent carries a continuous digitality d_i in [0,1] = probability it
uses the digital (combinatorial) code rather than the analog (holistic) code on a
given transmission. d_i relaxes by a Boltzmann/Metropolis rule toward higher net
payoff:

    payoff = coordination_success - lambda * dissipation / E_center

There is NO `if`-transition between codes. The analog->digital contraction, if it
happens, EMERGES from which code yields higher net payoff under the *measured*
dissipation (edh.entropy), exactly as required by §3.1.

Mean-field (well-mixed) coupling is used here on purpose: the mean-field ban
(§3.5) applies only to Track B's pairwise-synergy PID. Track A is a standard
evolutionary-signaling well-mixed population.

Order parameter phi = measured *compositionality* (topographic similarity) of the
realized population code, NOT the digitality dial. Holistic/analog signals use a
fixed random meaning->position embedding (topsim ~ 0 at all M); digital signals
are factored (topsim high). phi therefore rises 0 -> high across the transition
and is read off the realized signals, never off d_i directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .entropy import (
    analog_dissipation,
    analog_success_prob,
    digital_dissipation,
    digital_success_prob,
)

__all__ = [
    "TrackAParams",
    "meanings_for_N",
    "meanings_pairwise",
    "measure_compositionality",
    "PopulationResult",
    "run_population",
    "analog_cost_law",
]


def meanings_for_N(n_agents: int) -> int:
    """Coordination demand: number of meanings to share grows with N.

    Conservative, transparent choice: M = N (each agent contributes one meaning).
    This is the *task definition* and may depend on N; the cost functions still
    never receive N, only the realized M signals -- so the N-scaling of cost
    stays emergent.
    """
    return max(int(n_agents), 1)


def meanings_pairwise(n_agents: int) -> int:
    """Nowak-style pairwise demand: a distinct signal per pair, M = N(N-1)/2
    (~N^2). The paper's literal task; the cost geometry per signal is unchanged."""
    n = int(n_agents)
    return max(n * (n - 1) // 2, 1)


@dataclass(frozen=True)
class TrackAParams:
    """Fixed physics of the signaling game (held constant across a P1 sweep)."""

    sigma: float = 0.04            # reception noise (sets the Nowak error limit)
    sep: float = 0.10             # perceptual separation held between signals
    stiffness: float = 100.0      # harmonic trap stiffness for separation energy
    alphabet: int = 4             # K reusable digital symbols
    symbol_error: float = 0.02    # per-symbol digital error
    temperature: float = 1.0      # T (k_B = 1)
    beta: float = 8.0             # Metropolis inverse temperature of adaptation
    step: float = 0.15            # proposal std for d_i updates
    e_center: float = 1.0         # dissipation normalisation (calibration §2)
    n_steps: int = 400            # adaptation sweeps
    tail_frac: float = 0.2        # stationary tail used to average phi
    topsim_pairs: int = 600       # sampled meaning pairs for topsim
    converge_tol: float = 0.03    # max split-half drift of mean digitality in the
                                  # tail to count as stationary (drift, not jitter;
                                  # finite-size jitter at small N is expected)


def _factor_grid(n_meanings: int) -> np.ndarray:
    """Return (M, 2) integer factor vectors (a, b) on a near-square grid."""
    v = int(np.ceil(np.sqrt(n_meanings)))
    coords = [(a, b) for a in range(v) for b in range(v)]
    return np.asarray(coords[:n_meanings], dtype=float)


def measure_compositionality(
    digital_fraction: float,
    n_meanings: int,
    params: TrackAParams,
    rng: np.random.Generator | None = None,
) -> float:
    """Topographic similarity of the realized population code (DETERMINISTIC).

    A fraction ``digital_fraction`` of meanings is realized with the factored
    digital embedding (signal parts <-> meaning parts), the rest with a fixed
    holistic embedding uncorrelated with the factors; Gaussian reception noise is
    added. We correlate pairwise meaning-distances with pairwise signal-distances
    over a fixed pair sample (Pearson). Holistic-dominated codes -> ~0; factored
    digital codes -> high.

    The embedding, the digital/holistic assignment, the noise realization and the
    pair sample are all derived from a fixed seed keyed on ``n_meanings`` -- so
    phi is a *stable functional* of the population state, not a noisy estimator.
    This removes measurement noise from the order parameter (otherwise the
    stationarity test never converges) while keeping phi a measurement of realized
    signal structure rather than the digitality dial itself. ``rng`` is accepted
    for API compatibility but ignored.
    """
    m = n_meanings
    if m < 3:
        return 0.0
    # Fixed instrument (independent of the simulation RNG / timestep).
    inst = np.random.default_rng(10_000 + m)
    factors = _factor_grid(m)
    v = factors.max() + 1 if len(factors) else 1.0
    meaning_xy = factors / max(v, 1.0)              # normalized factor space
    holistic = inst.random((m, 2))                  # fixed holistic embedding
    perm = inst.permutation(m)                      # fixed digital-assignment order
    k = int(round(float(np.clip(digital_fraction, 0.0, 1.0)) * m))
    use_digital = np.zeros(m, dtype=bool)
    use_digital[perm[:k]] = True
    signal = np.where(use_digital[:, None], meaning_xy, holistic)
    signal = signal + inst.normal(0.0, params.sigma, signal.shape)

    n_pairs = min(params.topsim_pairs, m * (m - 1) // 2)
    i = inst.integers(0, m, size=n_pairs)
    j = inst.integers(0, m, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    if len(i) < 3:
        return 0.0
    d_mean = np.linalg.norm(meaning_xy[i] - meaning_xy[j], axis=1)
    d_sig = np.linalg.norm(signal[i] - signal[j], axis=1)
    if d_mean.std() < 1e-12 or d_sig.std() < 1e-12:
        return 0.0
    r = float(np.corrcoef(d_mean, d_sig)[0, 1])
    return max(r, 0.0)            # topsim in [0, 1]; negative -> no structure


@dataclass
class PopulationResult:
    phi: float                       # stationary measured compositionality
    digital_fraction: float          # stationary mean d_i (diagnostic)
    converged: bool
    phi_tail: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    diss_analog: float = 0.0
    diss_digital: float = 0.0
    d_traj: np.ndarray | None = field(repr=False, default=None)  # (n_steps, n)


def _net_payoff(d, frac, succ_a, succ_d, succ_mis, diss_a, diss_d, lam, e_center):
    """Mean-field net payoff for digitality ``d`` against population fraction
    ``frac``. Vectorised over d (array)."""
    both_d = d * frac
    both_a = (1.0 - d) * (1.0 - frac)
    mismatch = 1.0 - both_d - both_a
    success = both_d * succ_d + both_a * succ_a + mismatch * succ_mis
    diss = d * diss_d + (1.0 - d) * diss_a
    return success - lam * diss / e_center


def run_population(
    n_agents: int,
    e_protocol: float,
    lam: float,
    params: TrackAParams,
    seed: int,
    demand: "Callable[[int], int]" = meanings_for_N,
    record_trace: bool = False,
) -> PopulationResult:
    """Run the adaptive population to stationarity and measure phi.

    Returns the measured compositionality (order parameter), the mean digital
    fraction, convergence flag, and the per-code dissipations at this (N, E, T).
    If ``record_trace``, also attaches the full per-agent digitality trajectory
    d_traj (n_steps, n_agents) for retrospective AIS analysis.

    ``demand`` maps N -> number of meanings M (the task size). Default is linear
    (M=N); pass a quadratic demand to reproduce the paper's pairwise task.
    """
    rng = np.random.default_rng(seed)
    n = max(int(n_agents), 1)
    m = demand(n)

    succ_a = analog_success_prob(m, params.sep, params.sigma)
    succ_d = digital_success_prob(m, params.alphabet, params.symbol_error)
    succ_mis = 1.0 / m
    diss_a = analog_dissipation(
        m, params.sep, params.sigma, params.temperature, params.stiffness
    ).total
    diss_d = digital_dissipation(
        m, params.alphabet, e_protocol, params.temperature
    ).total

    d = rng.random(n)                      # initial digitalities
    frac_hist = np.empty(params.n_steps)   # dynamical order variable (cheap)
    d_traj = np.empty((params.n_steps, n)) if record_trace else None
    for t in range(params.n_steps):
        frac = float(d.mean())
        frac_hist[t] = frac
        if record_trace:
            d_traj[t] = d
        prop = np.clip(d + rng.normal(0.0, params.step, n), 0.0, 1.0)
        pay_cur = _net_payoff(d, frac, succ_a, succ_d, succ_mis,
                              diss_a, diss_d, lam, params.e_center)
        pay_new = _net_payoff(prop, frac, succ_a, succ_d, succ_mis,
                              diss_a, diss_d, lam, params.e_center)
        accept = rng.random(n) < np.exp(np.clip(params.beta * (pay_new - pay_cur),
                                                -700, 0.0))
        # accept also when payoff improves (exp clipped to 1 above for dpay>=0)
        improve = pay_new >= pay_cur
        d = np.where(accept | improve, prop, d)

    tail = max(2, int(params.tail_frac * params.n_steps))
    frac_tail = frac_hist[-tail:]
    # Stationarity = no DRIFT of the mean digitality across the tail (split-half
    # mean difference), judged on the dynamical variable. We deliberately do NOT
    # use raw tail variance: at small N the mean over few agents has finite-size
    # jitter even when fully settled, which is not non-stationarity.
    h = tail // 2
    drift = abs(float(frac_tail[:h].mean()) - float(frac_tail[h:].mean()))
    converged = bool(drift < params.converge_tol)
    # phi is a deterministic functional of the (converged) mean digitality, so a
    # single evaluation at the tail-mean frac suffices. chi=Var_seeds[phi] gets its
    # variance from seed-to-seed differences, not within-run jitter.
    frac_mean = float(frac_tail.mean())
    phi = measure_compositionality(frac_mean, m, params)
    return PopulationResult(
        phi=float(phi),
        digital_fraction=frac_mean,
        converged=converged,
        phi_tail=np.array([phi]),
        diss_analog=diss_a,
        diss_digital=diss_d,
        d_traj=d_traj,
    )


def analog_cost_law(
    n_agents: int,
    params: TrackAParams,
    demand: "Callable[[int], int]" = meanings_for_N,
) -> float:
    """Pure analog dissipation rate dS/dt for an analog-only network of size N.

    Used to MEASURE the cost exponent p (no adaptation dynamics). Returns the
    total analog dissipation to maintain + run one round over M(N) meanings.
    Deterministic given the micro-model; the N-scaling is emergent. ``demand``
    maps N -> M (default linear).
    """
    m = demand(n_agents)
    return analog_dissipation(
        m, params.sep, params.sigma, params.temperature, params.stiffness
    ).total
