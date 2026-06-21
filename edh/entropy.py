"""
edh.entropy — REAL dissipation bookkeeping for Track A (P1, P3).

Hard rule (§3.2, anti-tautology): every cost function receives ONLY the realized
microstate (emitted signal amplitudes, separations, bits erased, sigma, T). NONE
of them receives N (the agent count) or M (could be argued as a proxy) as a knob
to raise to a literal power. The scaling of cost with N is therefore EMERGENT
from how many signals get used and how crowded they are, and is *measured*
(edh.finite_size / experiments), never written as `N**6`.

Units: k_B = 1 (see edh.K_B / edh.LN2), so Landauer's bound is T*ln2 per erased
bit and temperature/energy are dimensionless multiples of k_B.

Two codes:

* Analog (Nowak error limit, emergent). M holistic signals live on a 1-D
  amplitude axis, held at a perceptual separation `sep`. Adjacent signals are
  confused with a probability set by the Gaussian overlap of their reception
  noise (depends on sep/sigma -- the error limit is *measured*, not imposed).
  Keeping them distinguishable costs harmonic "separation energy" proportional to
  the squared amplitude each signal is displaced to (work to hold a register away
  from the thermal bath). As M grows, signals must span a wider amplitude range
  to stay sep apart -> sum of squared amplitudes grows -> cost rises with M with
  an exponent we MEASURE.

* Digital (combinatorial escape). K reusable symbols, message length
  L = ceil(log_K M); a fixed toolkit cost E_protocol plus L*T*ln2 Landauer per
  message. Per-symbol distinguishability is fixed, so cost rises only ~log M.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from . import LN2

__all__ = [
    "landauer_bits_cost",
    "analog_amplitudes",
    "analog_separation_energy",
    "adjacent_confusion_prob",
    "analog_success_prob",
    "AnalogCost",
    "analog_dissipation",
    "digital_message_length",
    "digital_success_prob",
    "DigitalCost",
    "digital_dissipation",
]


# ---------------------------------------------------------------------------
# Landauer: the only fundamental floor. Cost to irreversibly erase `n_bits`.
# ---------------------------------------------------------------------------
def landauer_bits_cost(n_bits: float, temperature: float) -> float:
    """Landauer cost to erase ``n_bits`` at ``temperature`` (k_B = 1)."""
    return float(n_bits) * temperature * LN2


# ---------------------------------------------------------------------------
# Analog code micro-model
# ---------------------------------------------------------------------------
def analog_amplitudes(n_signals: int, sep: float) -> np.ndarray:
    """Realized 1-D placement: ``n_signals`` equally spaced by ``sep``, centered.

    This is the *microstate* the cost functions consume. Note the argument is the
    realized number of emitted signals, not the agent count N.
    """
    idx = np.arange(n_signals, dtype=float) - (n_signals - 1) / 2.0
    return idx * sep


def analog_separation_energy(amplitudes: np.ndarray, stiffness: float) -> float:
    """Harmonic work to HOLD each signal at its amplitude vs the thermal bath.

    Sum of (1/2) k x^2 over the realized amplitudes. Receives only the realized
    amplitude array and the trap stiffness -- never N.
    """
    a = np.asarray(amplitudes, dtype=float)
    return float(0.5 * stiffness * np.sum(a * a))


def adjacent_confusion_prob(sep: float, sigma: float) -> float:
    """Prob. that Gaussian reception noise pushes a signal past the midpoint to
    its neighbour. Emergent Nowak error limit: depends only on sep/sigma."""
    if sigma <= 0:
        return 0.0
    # P(|noise| > sep/2) for one-sided crossing toward a neighbour.
    return float(norm.sf(sep / (2.0 * sigma)))


def analog_success_prob(n_signals: int, sep: float, sigma: float) -> float:
    """Coordination success for an analog transmission among ``n_signals``.

    A signal is correctly decoded if it is not confused with either neighbour.
    With per-side confusion p_c, interior signals have ~2 neighbours and edge
    signals 1; success ~= (1 - p_c)^(mean #neighbours). As n_signals grows, the
    same physical sep yields more confusable neighbours -> success drops (the
    error limit). Returned in [0, 1].
    """
    if n_signals <= 1:
        return 1.0
    p_c = adjacent_confusion_prob(sep, sigma)
    mean_neighbours = 2.0 * (n_signals - 1) / n_signals  # ->2 as n grows
    return float(np.clip((1.0 - p_c) ** mean_neighbours, 0.0, 1.0))


@dataclass(frozen=True)
class AnalogCost:
    """Decomposed analog dissipation for one round of M-message communication."""

    separation: float
    landauer: float

    @property
    def total(self) -> float:
        return self.separation + self.landauer


def analog_dissipation(
    n_signals: int,
    sep: float,
    sigma: float,
    temperature: float,
    stiffness: float,
) -> AnalogCost:
    """Total analog dissipation to run a round addressing ``n_signals`` meanings.

    = separation energy (hold all signals apart) + Landauer (erase/refresh the
    receiver's state estimate, log2(n_signals) bits per resolved message).
    Receives the realized signal count, sep, sigma, T, stiffness -- never N.
    """
    amps = analog_amplitudes(n_signals, sep)
    sep_e = analog_separation_energy(amps, stiffness)
    bits = np.log2(max(n_signals, 1))
    land = landauer_bits_cost(bits, temperature)
    return AnalogCost(separation=sep_e, landauer=land)


# ---------------------------------------------------------------------------
# Digital code micro-model
# ---------------------------------------------------------------------------
def digital_message_length(n_messages: int, alphabet: int) -> int:
    """L = ceil(log_K n_messages); minimum symbols to index n_messages."""
    if n_messages <= 1:
        return 1
    return int(np.ceil(np.log(n_messages) / np.log(alphabet)))


def digital_success_prob(n_messages: int, alphabet: int, symbol_error: float) -> float:
    """Whole-message success = (1 - symbol_error)^L. Per-symbol distinguishability
    is fixed (reused alphabet), so success degrades only ~log in n_messages."""
    L = digital_message_length(n_messages, alphabet)
    return float(np.clip((1.0 - symbol_error) ** L, 0.0, 1.0))


@dataclass(frozen=True)
class DigitalCost:
    """Decomposed digital dissipation for one round."""

    toolkit: float
    landauer: float

    @property
    def total(self) -> float:
        return self.toolkit + self.landauer


def digital_dissipation(
    n_messages: int,
    alphabet: int,
    e_protocol: float,
    temperature: float,
) -> DigitalCost:
    """Total digital dissipation: fixed toolkit E_protocol + L*T*ln2 Landauer.

    Receives the realized message count, alphabet, E_protocol, T -- never N. The
    toolkit term is independent of n_messages (combinatorial reuse), so this cost
    rises only logarithmically in the number of meanings.
    """
    L = digital_message_length(n_messages, alphabet)
    land = landauer_bits_cost(L, temperature)
    return DigitalCost(toolkit=float(e_protocol), landauer=land)
