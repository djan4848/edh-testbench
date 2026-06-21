"""
Phase 1 acceptance: the dissipation micro-model behaves like real physics.

Checks the Landauer floor, the analog separation/error-limit monotonicities, and
that the digital toolkit cost is independent of the number of meanings (the
combinatorial reuse that makes digital escape the analog cost).
"""
from __future__ import annotations

import numpy as np

from edh import LN2
from edh.entropy import (
    adjacent_confusion_prob,
    analog_dissipation,
    analog_separation_energy,
    analog_amplitudes,
    analog_success_prob,
    digital_dissipation,
    digital_message_length,
    landauer_bits_cost,
)


def test_landauer_floor():
    assert landauer_bits_cost(1.0, 1.0) == LN2
    assert landauer_bits_cost(3.0, 2.0) == 3.0 * 2.0 * LN2
    # monotone in bits and temperature
    assert landauer_bits_cost(2, 1) > landauer_bits_cost(1, 1)
    assert landauer_bits_cost(1, 2) > landauer_bits_cost(1, 1)


def test_separation_energy_monotone():
    e_small = analog_separation_energy(analog_amplitudes(10, 0.1), stiffness=1.0)
    e_wide = analog_separation_energy(analog_amplitudes(10, 0.2), stiffness=1.0)
    e_more = analog_separation_energy(analog_amplitudes(20, 0.1), stiffness=1.0)
    assert e_wide > e_small          # wider separation costs more
    assert e_more > e_small          # more signals span a wider range -> more
    assert analog_separation_energy(analog_amplitudes(1, 0.1), 1.0) == 0.0


def test_confusion_emergent_error_limit():
    p_close = adjacent_confusion_prob(sep=0.05, sigma=0.05)
    p_far = adjacent_confusion_prob(sep=0.5, sigma=0.05)
    assert 0.0 <= p_far < p_close <= 1.0      # closer/noisier -> more confusion
    # success drops as more meanings crowd the same physical separation
    s_few = analog_success_prob(2, sep=0.1, sigma=0.05)
    s_many = analog_success_prob(20, sep=0.1, sigma=0.05)
    assert 0.0 <= s_many <= s_few <= 1.0


def test_digital_message_length():
    assert digital_message_length(4, 4) == 1
    assert digital_message_length(16, 4) == 2
    assert digital_message_length(17, 4) == 3
    assert digital_message_length(1, 4) == 1


def test_digital_toolkit_is_meaning_count_independent():
    c_few = digital_dissipation(4, alphabet=4, e_protocol=50.0, temperature=1.0)
    c_many = digital_dissipation(4000, alphabet=4, e_protocol=50.0, temperature=1.0)
    assert c_few.toolkit == c_many.toolkit == 50.0
    # total grows only ~log (a few Landauer bits), never explodes
    assert c_many.total - c_few.total < 10.0 * LN2


def test_analog_overtakes_digital_with_scale():
    """Sanity: at small M analog is cheap, at large M it overtakes the fixed
    digital toolkit -- the qualitative crossover P1/P3 are about."""
    kw = dict(sep=0.1, sigma=0.04, temperature=1.0, stiffness=1.0)
    small = analog_dissipation(3, **kw).total
    large = analog_dissipation(60, **kw).total
    digital = digital_dissipation(60, alphabet=4, e_protocol=50.0,
                                  temperature=1.0).total
    assert small < digital < large
