"""
edh.track_b_proto_dialogue — Track B (P2 / H2) orchestration.

Reuses the VALIDATED reference machinery (reference/dynamics_stochastic.py for the
stochastic, locally-coupled self-referential automata, and reference/pid_fast.py
via edh.pid for the BROJA PID). The agent state is the per-agent bit-vector (rich
in BOTH the isolated and coupled phases -- this fixes Track A, where the state was
the scalar digitality and the digital phase was degenerate).

H2 (P2): when two self-referential systems start perturbing each other's
boundaries (coupling on at t_c), the synergy Syn(t) of their joint information
rises BEFORE the unique information Unq(t). Statistic Delta = onset(Unq)-onset(Syn);
H2 predicts Delta > 0.

The joint 6-bit next-state target is coarse-grained to 4 bits (2 MSBs per agent)
to keep sampling adequacy = R/occupied comfortably above the guard while retaining
the JOINT structure PID needs.
"""
from __future__ import annotations

import numpy as np

import edh.pid  # noqa: F401  -- registers the BROJA_2PID alias + reference path
from edh.pid import pid as _pid
from edh.stats import onset

# reference modules are on sys.path via edh.pid
from dynamics_stochastic import simulate_ensemble  # noqa: E402

__all__ = [
    "simulate_ensemble",
    "simulate_dialogue",
    "simulate_picard",
    "simulate_plastic",
    "plastic_core2",
    "core_int",
    "joint_target",
    "occupied_at_t",
    "pid_series",
    "h2_onsets",
]


def simulate_picard(N=6, R=1024, L=8, T_isolated=120, T_coupled=240,
                    beta=2.0, coupling=0.4, self_strength=1.1, rule_strength=1.1,
                    asymmetric=False, seed=0, return_dissipation=False):
    """Stochastic self-referential (PICARD) automata with EMERGENT coupling.

    The neighbour perturbs through the PICARD mechanism, NOT an imposed gate:
      * State X_i (L bits) = rule-selector bits [0,1] + data bits [2:L]. The rule
        R_i = f(X_i) is READ from the state (the selector chooses which of 4
        shift/sign maps drives the data bits). The selector itself updates from the
        data (majority) -- the self-referential loop X(t+1)=R[X(t)](t).
      * Coupling (t>=t_c): with prob ``coupling`` (per replica/agent) the neighbour
        j's selector bits replace i's selector bits IN THE INPUT used to compute i's
        update. So j changes WHICH rule i applies to its own data -- a structural,
        state-dependent perturbation of the input, never an XOR on i's output. The
        Syn/Unq/Red split is therefore an emergent property of the dynamics, not of
        the coupling operator.
      * ``asymmetric``: agents alternate between two DIFFERENT rule tables ("two
        distinct interlocutors"); the symmetric case suppresses unique by symmetry.

    beta is FIXED (the endogenous credit->beta feedback collapses the coupled phase
    to a degenerate cycle -- a finding noted in LIMITATIONS, not hidden; robustness
    to beta is tested separately). Returns (states (T,R,N,L) int8, t_isolated).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    T = T_isolated + T_coupled
    ndata = L - 2
    states = np.zeros((T, R, N, L), dtype=np.int8)
    states[0] = rng.integers(0, 2, (R, N, L), dtype=np.int8)

    # two rule tables (shift + sign per rule); agents use table A, or alternate A/B
    shifts_A = np.array([1, 2, 3, 5]); signs_A = np.array([1.0, 1.0, -1.0, -1.0])
    shifts_B = np.array([2, 4, 1, 3]); signs_B = np.array([-1.0, 1.0, 1.0, -1.0])
    sh_tab = np.stack([shifts_A if (not asymmetric or a % 2 == 0) else shifts_B
                       for a in range(N)])           # (N,4)
    sg_tab = np.stack([signs_A if (not asymmetric or a % 2 == 0) else signs_B
                       for a in range(N)])           # (N,4)
    agents = np.arange(N)
    LN2 = float(np.log(2.0))
    diss = np.zeros(T)                                 # per-step Landauer (ln2 units)

    for t in range(T - 1):
        X = states[t]                                  # (R,N,L)
        sel = X[:, :, 0] * 2 + X[:, :, 1]              # (R,N) rule selector 0..3
        if t >= T_isolated:
            # mutual coupling: borrow the selector from a RANDOM neighbour (+-1) so
            # the measured pair (0,1) perturbs each other (not one-directional).
            direction = np.where(rng.random((R, N)) < 0.5, 1, -1)
            nbr_idx = (agents[None, :] + direction) % N
            nbr_sel = np.take_along_axis(X[:, :, 0] * 2 + X[:, :, 1], nbr_idx, axis=1)
            swap = rng.random((R, N)) < coupling
            sel = np.where(swap, nbr_sel, sel)         # j perturbs i's input rule
        sh = sh_tab[agents[None, :].repeat(R, 0), sel]  # (R,N)
        sg = sg_tab[agents[None, :].repeat(R, 0), sel]  # (R,N)
        data = X[:, :, 2:].astype(np.float64)          # (R,N,ndata)

        field = np.zeros((R, N, L))
        for k in range(ndata):
            src = (k + sh) % ndata                      # (R,N)
            val = np.take_along_axis(data, src[:, :, None], axis=2)[:, :, 0]
            field[:, :, 2 + k] = self_strength * sg * (2.0 * val - 1.0)
        # selector bits update from data majority (self-reference closes the loop)
        maj0 = data[:, :, :ndata // 2].mean(axis=2) > 0.5
        maj1 = data[:, :, ndata // 2:].mean(axis=2) > 0.5
        field[:, :, 0] = rule_strength * (2.0 * maj0 - 1.0)
        field[:, :, 1] = rule_strength * (2.0 * maj1 - 1.0)

        p_on = 1.0 / (1.0 + np.exp(-beta * field))
        states[t + 1] = (rng.random((R, N, L)) < p_on).astype(np.int8)
        if return_dissipation:
            pc = np.clip(p_on, 1e-12, 1 - 1e-12)
            h2 = -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))
            diss[t] = float(np.mean(1.0 - h2)) * LN2   # ~bits erased * ln2, per bit
    if return_dissipation:
        return states, T_isolated, diss
    return states, T_isolated


def simulate_dialogue(N=6, R=1024, L=8, T_isolated=120, T_coupled=240,
                      beta=2.5, self_strength=0.7, couple_strength=1.3,
                      seed=0):
    """Stochastic self-referential automata with SYNERGISTIC local coupling.

    Why not the reference field(): its coupling is an additive neighbour-mean
    bias -> a common drive -> pairwise PID is redundancy by construction (the very
    failure mode PROTOCOL.md warns about), so it CANNOT test a pairwise-synergy
    claim. We keep the validated estimation architecture (stochastic Boltzmann
    update, LOCAL ring coupling, ensemble-at-fixed-t) and inject physics that can
    actually generate synergy:

      * Self-reference (always on): each agent's field is a noisy cyclic-shift rule
        read from its own state -> internal self-predictability (the UNIQUE info
        baseline). PICARD-style X(t+1) read from X(t).
      * Coupling (t >= t_c): a per-bit XOR of the agent with its ring-+1 neighbour
        is added to the field. XOR is the canonical synergy generator: the joint
        (self, neighbour) determines the drive in a way neither does alone.

    beta is FIXED (no credit->beta runaway, which collapsed the coupled phase to a
    degenerate cycle). Whether synergy rises BEFORE unique is NOT imposed -- it is
    an emergent property of the transient and is what C1 tests.

    Returns (states (T,R,N,L) int8, t_isolated).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    T = T_isolated + T_coupled
    states = np.zeros((T, R, N, L), dtype=np.int8)
    states[0] = rng.integers(0, 2, (R, N, L), dtype=np.int8)
    shift_idx = (np.arange(L) + 1) % L
    for t in range(T - 1):
        X = states[t].astype(np.float64)
        field = self_strength * (2.0 * states[t][:, :, shift_idx] - 1.0)
        if t >= T_isolated:
            nbr = np.roll(states[t], -1, axis=1)           # ring +1 neighbour
            xor = (states[t] ^ nbr).astype(np.float64)
            field = field + couple_strength * (2.0 * xor - 1.0)
        p_on = 1.0 / (1.0 + np.exp(-beta * field))
        states[t + 1] = (rng.random(X.shape) < p_on).astype(np.int8)
    return states, T_isolated

SRC_BITS = (4, 5)      # 2-bit source core per agent (0..3)
TGT_BIT = 5            # one next-state bit per agent -> 2-bit joint target (0..3)


def core_int(bits: np.ndarray) -> np.ndarray:
    """(R, L) bit array -> 2-bit integer source core from SRC_BITS."""
    b0, b1 = SRC_BITS
    return (bits[:, b0] << 1) | bits[:, b1]


def joint_target(Xn: np.ndarray, i: int, j: int) -> np.ndarray:
    """Joint next-state target: one next-bit per agent -> 2-bit (0..3). Keeps the
    JOINT structure PID needs while keeping the joint (s1,s2,target) support small
    enough for adequacy = R/occupied >= guard."""
    return (Xn[:, i, TGT_BIT].astype(int) << 1) | Xn[:, j, TGT_BIT].astype(int)


def occupied_at_t(states: np.ndarray, t: int, pair=(0, 1)) -> int:
    """Number of occupied (s1, s2, target) cells over the R-replica ensemble at a
    fixed time t (C0 non-degeneracy check)."""
    i, j = pair
    Xt, Xn = states[t], states[t + 1]
    s1 = core_int(Xt[:, i, :]); s2 = core_int(Xt[:, j, :])
    tgt = joint_target(Xn, i, j)
    return len(set(zip(s1.tolist(), s2.tolist(), tgt.tolist())))


def pid_series(states: np.ndarray, pair=(0, 1), stride: int = 3,
               permute_target: bool = False, seed: int = 0) -> dict:
    """Fixed-t ensemble PID time series over R replicas. sources = 2-bit cores of
    i,j at t; target = 2-bit joint next-bit at t+1. ``permute_target`` shuffles the
    target across replicas within each window (target-surrogate control -> destroys
    real structure, keeps marginals/sample size)."""
    rng = np.random.default_rng(seed)
    i, j = pair
    T = states.shape[0]
    times, syn, unq, red, mi, adeq, ok = [], [], [], [], [], [], []
    for t in range(0, T - 1, stride):
        Xt, Xn = states[t], states[t + 1]
        s1 = core_int(Xt[:, i, :]); s2 = core_int(Xt[:, j, :])
        tgt = joint_target(Xn, i, j)
        if permute_target:
            tgt = rng.permutation(tgt)
        r = _pid(s1, s2, tgt, min_occupied=4)
        times.append(t); syn.append(r.syn); unq.append(r.unq); red.append(r.red)
        mi.append(r.mi); adeq.append(r.adequacy); ok.append(r.ok)
    return {k: np.array(v) for k, v in dict(
        time=times, syn=syn, unq=unq, red=red, mi=mi, adequacy=adeq, ok=ok).items()}


def _core_bits(bits2d: np.ndarray, idx) -> np.ndarray:
    """(R, L) bit array -> integer over the bit indices in ``idx`` (MSB first)."""
    v = np.zeros(bits2d.shape[0], dtype=int)
    for b in idx:
        v = (v << 1) | bits2d[:, b]
    return v


def picard_pid_series(states, pair=(0, 1), stride=3, src_bits=(0, 1),
                      tgt_bits=(2, 3), permute_target=False, seed=0) -> dict:
    """Fixed-t ensemble PID for the PICARD model. sources = ``src_bits`` core of
    i,j at t (the rule-selector interface); target = JOINT next-state over
    ``tgt_bits`` of i and j at t+1. permute_target -> target-surrogate control."""
    import numpy as np
    rng = np.random.default_rng(seed)
    i, j = pair
    T = states.shape[0]
    nb = len(tgt_bits)
    times, syn, unq, red, mi, adeq, ok = [], [], [], [], [], [], []
    for t in range(0, T - 1, stride):
        Xt, Xn = states[t], states[t + 1]
        s1 = _core_bits(Xt[:, i, :], src_bits)
        s2 = _core_bits(Xt[:, j, :], src_bits)
        tgt = (_core_bits(Xn[:, i, :], tgt_bits) << nb) | _core_bits(Xn[:, j, :], tgt_bits)
        if permute_target:
            tgt = rng.permutation(tgt)
        r = _pid(s1, s2, tgt, min_occupied=4)
        times.append(t); syn.append(r.syn); unq.append(r.unq); red.append(r.red)
        mi.append(r.mi); adeq.append(r.adequacy); ok.append(r.ok)
    return {k: np.array(v) for k, v in dict(
        time=times, syn=syn, unq=unq, red=red, mi=mi, adequacy=adeq, ok=ok).items()}


def picard_occupied_at_t(states, t, pair=(0, 1), src_bits=(0, 1), tgt_bits=(2, 3)):
    i, j = pair
    Xt, Xn = states[t], states[t + 1]
    s1 = _core_bits(Xt[:, i, :], src_bits); s2 = _core_bits(Xt[:, j, :], src_bits)
    nb = len(tgt_bits)
    tgt = (_core_bits(Xn[:, i, :], tgt_bits) << nb) | _core_bits(Xn[:, j, :], tgt_bits)
    return len(set(zip(s1.tolist(), s2.tolist(), tgt.tolist())))


def h2_onsets(series: dict, t_c: float, k: float = 3.0) -> dict:
    """onset(Syn), onset(Unq), Delta = onset(Unq)-onset(Syn) for one series."""
    s_on = onset(series["time"], series["syn"], t_c, k)
    u_on = onset(series["time"], series["unq"], t_c, k)
    return {"onset_syn": s_on, "onset_unq": u_on, "delta": u_on - s_on}


def simulate_plastic(N=6, R=2048, L=8, T_isolated=120, T_coupled=240,
                     beta=2.5, coupling=1.0, self_strength=0.9, plast_strength=1.6,
                     noise_partner=False, seed=0, return_filters=False,
                     return_dissipation=False, plasticity="hebbian"):
    """Autopoietic PLASTIC-BOUNDARY model (section 6.5 / eq.20).

    State X_i (L=8) = CONTENT bits [0:6] + FILTER bits [6:8]. The filter F_i is a
    2-bit address = which phase of the neighbour's content i attends to. The SAME
    rule reads and writes both: content updates from a self-shift plus the
    neighbour content read through F_i; F_i updates (Boltzmann) toward the address
    whose attended neighbour content best AGREES with i's own content (Hebbian
    co-activation). So the boundary is state, co-evolving with content -- a single
    self-producing loop. Plasticity is active only while coupled.

    ``noise_partner`` replaces the neighbour content with i.i.d. noise (the
    precondition control: filters should specialise toward a STRUCTURED partner,
    not toward noise). beta fixed. Returns (states, t_isolated[, filters][, diss]).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    T = T_isolated + T_coupled
    nc = 6                                          # content bits
    states = np.zeros((T, R, N, L), dtype=np.int8)
    states[0] = rng.integers(0, 2, (R, N, L), dtype=np.int8)
    agents = np.arange(N)
    LN2 = float(np.log(2.0))
    filt = np.zeros((T, R, N), dtype=np.int8) if return_filters else None
    diss = np.zeros(T)

    for t in range(T - 1):
        X = states[t]
        C = X[:, :, :nc].astype(np.float64)         # content (R,N,nc)
        Faddr = (X[:, :, nc] * 2 + X[:, :, nc + 1]).astype(int)   # (R,N) 0..3
        if return_filters:
            filt[t] = Faddr
        field = np.zeros((R, N, L))
        for k in range(nc):                          # self-recurrence (shift)
            field[:, :, k] = self_strength * (2.0 * C[:, :, (k + 1) % nc] - 1.0)

        if t >= T_isolated:
            direction = np.where(rng.random((R, N)) < 0.5, 1, -1)
            nbr = (agents[None, :] + direction) % N
            Cj = np.take_along_axis(C, nbr[:, :, None].repeat(nc, axis=2), axis=1)
            if noise_partner:
                Cj = rng.integers(0, 2, (R, N, nc)).astype(np.float64)
            # content coupling: read neighbour content at phase = filter address
            for k in range(nc):
                src = (k + Faddr) % nc
                cjk = np.take_along_axis(Cj, src[:, :, None], axis=2)[:, :, 0]
                field[:, :, k] += coupling * (2.0 * cjk - 1.0)
            # filter plasticity (two realizations of eq.20):
            if plasticity == "hebbian":
                # attend to the neighbour phase that AGREES with own content
                agree = np.stack([
                    np.sum(C == np.roll(Cj, -a, axis=2), axis=2) for a in range(4)
                ], axis=2)                           # (R,N,4)
                best = np.argmax(agree, axis=2)      # (R,N)
            elif plasticity == "consensus":
                # NON-Hebbian: boundary consensus -- attend where the neighbour
                # attends (mutual addressing); no content matching.
                best = np.take_along_axis(Faddr, nbr, axis=1)
                if noise_partner:                    # noise partner -> random target
                    best = rng.integers(0, 4, (R, N))
            else:
                raise ValueError(f"unknown plasticity {plasticity!r}")
            field[:, :, nc] = plast_strength * (2.0 * (best >> 1) - 1.0)
            field[:, :, nc + 1] = plast_strength * (2.0 * (best & 1) - 1.0)

        p_on = 1.0 / (1.0 + np.exp(-beta * field))
        states[t + 1] = (rng.random((R, N, L)) < p_on).astype(np.int8)
        if return_dissipation:
            pc = np.clip(p_on, 1e-12, 1 - 1e-12)
            h2 = -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))
            diss[t] = float(np.mean(1.0 - h2)) * LN2

    out = [states, T_isolated]
    if return_filters:
        out.append(filt)
    if return_dissipation:
        out.append(diss)
    return tuple(out) if len(out) > 2 else (states, T_isolated)


# ---------------------------------------------------------------------------
# C2-RG: unit-level AIS (coarse-grained pair) + Still thermodynamic bridge
# ---------------------------------------------------------------------------
def plastic_core2(bits2d: np.ndarray) -> np.ndarray:
    """(R,L) -> 2-bit content summary for the plastic model: (content bit 0,
    block-spin of the 6 content bits)."""
    c0 = bits2d[:, 0].astype(int)
    block = (bits2d[:, :6].mean(axis=1) > 0.5).astype(int)
    return (c0 << 1) | block


def agent_core2(bits2d: np.ndarray) -> np.ndarray:
    """(R,L) -> 2-bit agent summary: (selector bit 0, block-spin of data bits)."""
    sel0 = bits2d[:, 0].astype(int)
    block = (bits2d[:, 2:].mean(axis=1) > 0.5).astype(int)
    return (sel0 << 1) | block


def unit_macro(states: np.ndarray, t: int, pair=(0, 1), core_fn=agent_core2) -> np.ndarray:
    """4-bit joint macro-state of the pair as ONE unit at time t (R,)."""
    i, j = pair
    return (core_fn(states[t, :, i, :]) << 2) | core_fn(states[t, :, j, :])


def _ais_from_macro(macros: np.ndarray, base: int, lag: int,
                    bias_correct=True, min_occupied=8):
    """macros: (W, R) integer macro-states. AIS = I(u_t ; u_{t-1..t-lag})."""
    from edh.ais import mutual_info_discrete
    tgt, hist = [], []
    for w in range(lag, macros.shape[0]):
        tgt.append(macros[w])
        code = np.zeros(macros.shape[1], dtype=int)
        for d in range(1, lag + 1):
            code = code * base + macros[w - d]
        hist.append(code)
    if not tgt:
        from edh.ais import AISResult
        return AISResult(np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0, False, "empty")
    return mutual_info_discrete(np.concatenate(hist), np.concatenate(tgt),
                                min_occupied=min_occupied, bias_correct=bias_correct)


def unit_ais(states, t_lo, t_hi, pair=(0, 1), lag=1, bias_correct=True,
             core_fn=agent_core2):
    """AIS of the pair-as-unit over [t_lo, t_hi). base-16 macro (4-bit)."""
    macros = np.array([unit_macro(states, t, pair, core_fn=core_fn)
                       for t in range(t_lo, t_hi)])
    return _ais_from_macro(macros, base=16, lag=lag, bias_correct=bias_correct)


def micro_ais(states, t_lo, t_hi, agent=0, lag=1, bias_correct=True,
              core_fn=agent_core2):
    """AIS of a SINGLE agent core (2-bit) over [t_lo, t_hi) -- for self-similarity."""
    macros = np.array([core_fn(states[t, :, agent, :])
                       for t in range(t_lo, t_hi)])
    return _ais_from_macro(macros, base=4, lag=lag, bias_correct=bias_correct)


def i_nopred(states, t_lo, t_hi, driven=0, driver=1, bias_correct=True):
    """Still 2012 non-predictive information for the driven agent.
    input = driver's selector (2-bit); x = driven agent's core (2-bit).
    I_nopred = I(input_t; x_{t+1}) - I(input_{t+1}; x_t).  (bits)"""
    from edh.ais import mutual_info_discrete
    sel = lambda t: (states[t, :, driver, 0].astype(int) << 1) | states[t, :, driver, 1]
    x = lambda t: agent_core2(states[t, :, driven, :])
    inp_t, x_next, inp_next, x_t = [], [], [], []
    for t in range(t_lo, t_hi - 1):
        inp_t.append(sel(t)); x_next.append(x(t + 1))
        inp_next.append(sel(t + 1)); x_t.append(x(t))
    mem = mutual_info_discrete(np.concatenate(inp_t), np.concatenate(x_next),
                               min_occupied=4, bias_correct=bias_correct)
    pred = mutual_info_discrete(np.concatenate(inp_next), np.concatenate(x_t),
                                min_occupied=4, bias_correct=bias_correct)
    if not (mem.ok and pred.ok):
        return np.nan, np.nan, np.nan
    return mem.ais, pred.ais, mem.ais - pred.ais
