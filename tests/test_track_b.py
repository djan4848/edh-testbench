"""
C0 acceptance (Track B, emergent PICARD coupling):
  * non-degenerate ensemble support in BOTH phases (PID well-posed);
  * adequacy >= guard;
  * NON-CIRCULARITY: the coupled-phase PID is NOT pinned to a vertex (not pure
    synergy like an injected XOR gate) -- unique carries a non-negligible share,
    i.e. the Syn/Unq/Red split emerges from the dynamics.
Determinism is covered by test_determinism; the PID gates by test_pid_gates.
"""
from __future__ import annotations

import numpy as np

from edh.track_b_proto_dialogue import (
    picard_occupied_at_t,
    picard_pid_series,
    simulate_picard,
)

MIN_OCCUPIED = 8


def test_support_nondegenerate_both_phases():
    states, t_c = simulate_picard(N=6, R=2048, T_isolated=120, T_coupled=240,
                                  coupling=0.6, beta=3.0, seed=1)
    assert picard_occupied_at_t(states, t_c // 2) >= MIN_OCCUPIED
    assert picard_occupied_at_t(states, t_c + 200) >= MIN_OCCUPIED


def test_picard_pid_series_runs_and_is_mostly_ok():
    states, t_c = simulate_picard(N=6, R=2048, T_isolated=80, T_coupled=160,
                                  coupling=0.6, beta=3.0, seed=2)
    s = picard_pid_series(states, pair=(0, 1), stride=6)
    assert len(s["time"]) > 5
    assert s["ok"].mean() > 0.8
    assert s["adequacy"][s["ok"]].min() >= 4.0


def test_emergent_coupling_is_not_a_vertex():
    """The coupled-phase PID must be a MIX, not pure-synergy (the XOR fingerprint).
    Unique must carry a non-negligible share of the joint information."""
    states, t_c = simulate_picard(N=6, R=2048, T_isolated=80, T_coupled=160,
                                  coupling=0.6, beta=3.0, seed=3)
    s = picard_pid_series(states, pair=(0, 1), stride=6)
    t = np.asarray(s["time"]); cou = t >= t_c
    syn = np.nanmean(s["syn"][cou]); unq = np.nanmean(s["unq"][cou])
    red = np.nanmean(s["red"][cou])
    tot = syn + unq + red
    assert tot > 0
    assert unq / tot > 0.1, f"unique share {unq/tot:.2f} -> looks vertex-pinned"
    assert syn / tot < 0.95, f"synergy share {syn/tot:.2f} -> looks XOR-imposed"
