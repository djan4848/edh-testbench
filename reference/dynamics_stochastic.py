"""
dynamics_stochastic.py
=======================
The SCIENTIFIC fix, not just a speed fix.

Problem with the original h2_simulation.py
------------------------------------------
The CA is deterministic and collapses to period-2/4 limit cycles within ~25
steps of coupling. The analysis then pooled W=300 consecutive timesteps of ONE
trajectory as if they were i.i.d. draws from a stationary ensemble. They are not:
each window is a 2-point cycle. PID on a 2-point cycle is meaningless, and it is
also what made BROJA hang.

Two independent design errors were entangled:
  (E1) Deterministic dynamics  -> no genuine ensemble / no real "information".
  (E2) Time-pooling within one trajectory -> the support is the cycle, not the
       reachable state distribution.

This module fixes both:
  (F1) STOCHASTIC dynamics. Bits follow a Boltzmann/Glauber update. A continuous
       "credits" variable governs the inverse temperature beta (no `if`
       truncation -- the analog->digital contraction, if it happens, must emerge
       from the physics of attractors under stress, exactly as you wanted).
  (F2) ENSEMBLE-AT-FIXED-t estimation. Run R independent replicas in parallel;
       at each measurement time t, the distribution over the R replicas is a
       proper estimate of the stochastic map's law -> rich support -> BROJA is
       well-posed and the PID means what it claims.

  Coupling is LOCAL (a ring), not mean-field. Mean-field all-to-all coupling
  makes every pair see the same common drive -> pairwise PID is redundancy-
  dominated by construction and cannot test a *pairwise* synergy claim (H2).

This is a TEMPLATE. Replace `field()` and the credits/Landauer law with your
own physics; the estimation architecture (parallel replicas, fixed-t ensemble)
is what makes H2 testable.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Continuous Landauer-style cost: cheaper when behaviour is ordered (low entropy)
# ---------------------------------------------------------------------------
def behavioural_entropy(history_bits):
    """Shannon entropy (bits) of recent per-agent activity. Continuous, no `if`.
    history_bits: (window, R, N, L) -> returns (R, N) mean-bit entropy proxy."""
    p = history_bits.mean(axis=0).clip(1e-9, 1 - 1e-9)   # (R,N,L) P(bit=1)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))     # per-bit binary entropy
    return h.mean(axis=2)                                  # (R,N)


def simulate_ensemble(N, R=512, L=10,
                      T_isolated=200, T_coupled=600,
                      beta0=0.8, beta_gain=2.5, credit_relax=0.05,
                      coupling=0.9, hist_w=20, seed=0):
    """
    Returns
    -------
    states : (T_total, R, N, L) int8     full ensemble trajectory
    t_couple : int                       index at which coupling switches on
    Dynamics (per replica r, agent i, bit k):
        field = self_bias + (coupled ? local_coupling : 0)
        beta  = beta0 + beta_gain * credits        (credits in [0,1], continuous)
        P(bit=1) = sigmoid(beta * field)
        credits relaxes toward (1 - recent behavioural entropy):
            ordered/digital behaviour -> low entropy -> high credits -> high beta
            -> sharper (more digital) updates.  Pure positive feedback, no `if`.
    """
    rng = np.random.default_rng(seed)
    T = T_isolated + T_coupled
    states = np.zeros((T, R, N, L), dtype=np.int8)
    states[0] = rng.integers(0, 2, (R, N, L), dtype=np.int8)
    credits = np.full((R, N), 0.3)                      # continuous reservoir

    def field(X, coupled):
        # self bias toward current state (memory) -- continuous in [-1,1]
        f = (X * 2 - 1) * 0.5
        if coupled:
            # LOCAL ring coupling: agent i driven by neighbours i-1, i+1 means
            m = X.mean(axis=2)                          # (R,N) activity
            nbr = 0.5 * (np.roll(m, 1, axis=1) + np.roll(m, -1, axis=1))
            f = f + (nbr[:, :, None] * 2 - 1) * coupling
        return f

    for t in range(T - 1):
        coupled = t >= T_isolated
        X = states[t]
        f = field(X, coupled)
        beta = (beta0 + beta_gain * credits)[:, :, None]   # (R,N,1)
        p_on = 1.0 / (1.0 + np.exp(-beta * f))
        states[t + 1] = (rng.random(X.shape) < p_on).astype(np.int8)

        # continuous credit update from recent behavioural entropy
        if t >= hist_w:
            ent = behavioural_entropy(states[t - hist_w:t])   # (R,N) in [0,1]
            target_credit = (1.0 - ent).clip(0, 1)
            credits += credit_relax * (target_credit - credits)

    return states, T_isolated


# ---------------------------------------------------------------------------
# Fixed-t ensemble PID time series (uses pid_fast)
# ---------------------------------------------------------------------------
def pid_timeseries(states, pair=(0, 1), core_bits=(3, 4, 5),
                   coarse_grain=None, stride=2):
    """
    For each time t, build the ensemble distribution over R replicas and compute
    BROJA PID for sources = agent i,j cores at t ; target = joint core at t+1.

    Returns dict of np.arrays: time, syn, unq, red, mi, adequacy, ok(bool).
    """
    from pid_fast import pid                      # local import keeps deps lazy
    i, j = pair
    T = states.shape[0]
    b0, b1, b2 = core_bits

    def to_int(arr):   # arr (R,) of 3 bits -> int 0..7
        return (arr[:, b0] << 2) | (arr[:, b1] << 1) | arr[:, b2]

    times, syn, unq, red, mi, adeq, okv = [], [], [], [], [], [], []
    for t in range(0, T - 1, stride):
        Xt, Xn = states[t], states[t + 1]         # (R,N,L)
        s1 = to_int(Xt[:, i, :]); s2 = to_int(Xt[:, j, :])
        tgt = (to_int(Xn[:, i, :]) << 3) | to_int(Xn[:, j, :])   # 0..63
        r = pid(s1, s2, tgt, coarse_grain=coarse_grain)
        times.append(t)
        syn.append(r.syn); unq.append(r.unq); red.append(r.red)
        mi.append(r.mi); adeq.append(r.adequacy); okv.append(r.ok)
    return {k: np.array(v) for k, v in dict(
        time=times, syn=syn, unq=unq, red=red, mi=mi,
        adequacy=adeq, ok=okv).items()}


if __name__ == "__main__":
    # quick sanity demo (does not need BROJA to show support is now rich)
    st, tc = simulate_ensemble(N=6, R=256, T_isolated=100, T_coupled=200, seed=1)
    from collections import Counter
    t = tc + 50
    core = lambda X: (X[:, :, 3] << 2) | (X[:, :, 4] << 1) | X[:, :, 5]
    c0, c1 = core(st[t])[:, 0], core(st[t])[:, 1]
    cn = core(st[t + 1])
    tgt = (cn[:, 0] << 3) | cn[:, 1]
    occ = len(set(zip(c0.tolist(), c1.tolist(), tgt.tolist())))
    print(f"occupied cells at fixed t over R replicas: {occ} (degenerate was 2)")
