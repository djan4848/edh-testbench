"""
Phase 1 acceptance (anti-tautology, §3.2 / cost_exponent_p):

PRIMARY guard: no cost function in edh.entropy receives N (the agent count). The
cost depends only on the realized microstate, so the N-scaling is emergent.

SECONDARY guard: an AST scan asserts no literal power of an agent-count variable
(`N**6`, `n_agents**2`, ...) is used anywhere in edh.entropy.

It also MEASURES the analog cost exponent p = d log(cost)/d log(N) and reports it
against the paper's eq.(12) value of 6. p != 6 is a FINDING, not a failure, so
the test only asserts p is a clean, positive power law.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np

import edh.entropy as entropy
from edh.track_a_signaling import TrackAParams, analog_cost_law

# Names that would mean "the agent count N leaked into the cost".
FORBIDDEN_PARAMS = {"n", "N", "n_agents", "num_agents", "nagents", "agents"}
FORBIDDEN_POW_BASES = {"n", "N", "n_agents", "num_agents", "nagents"}

COST_FUNCS = [
    entropy.landauer_bits_cost,
    entropy.analog_separation_energy,
    entropy.adjacent_confusion_prob,
    entropy.analog_success_prob,
    entropy.analog_dissipation,
    entropy.digital_message_length,
    entropy.digital_success_prob,
    entropy.digital_dissipation,
]


def test_cost_functions_do_not_receive_N():
    for fn in COST_FUNCS:
        params = set(inspect.signature(fn).parameters)
        leaked = params & FORBIDDEN_PARAMS
        assert not leaked, f"{fn.__name__} receives agent-count param(s): {leaked}"


def test_no_literal_power_of_agent_count():
    src = Path(entropy.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base = node.left
            if isinstance(base, ast.Name) and base.id in FORBIDDEN_POW_BASES:
                raise AssertionError(
                    f"hardcoded power of agent-count `{base.id}` at line {node.lineno}"
                )
    # Also forbid the literal in executable code (ignoring strings/comments) by
    # tokenizing: an agent-count NAME immediately followed by `**` and an int.
    import io
    import tokenize

    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for a, b, c in zip(toks, toks[1:], toks[2:]):
        if (a.type == tokenize.NAME and a.string in FORBIDDEN_POW_BASES
                and b.type == tokenize.OP and b.string == "**"
                and c.type == tokenize.NUMBER):
            raise AssertionError(
                f"hardcoded power `{a.string}**{c.string}` at line {a.start[0]}"
            )


def test_analog_cost_exponent_is_clean_power_law(capsys):
    params = TrackAParams()
    ns = np.arange(2, 21)
    cost = np.array([analog_cost_law(int(n), params) for n in ns])
    logN, logC = np.log(ns), np.log(cost)
    p, intercept = np.polyfit(logN, logC, 1)
    resid = logC - (p * logN + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((logC - logC.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    with capsys.disabled():
        print(f"\n[cost_exponent_p] measured p = {p:.3f}  (paper eq.12: 6)  "
              f"R^2 = {r2:.4f}")
    # p != 6 is a finding; we only require a real, clean, positive power law.
    assert p > 0.5, f"analog cost does not grow with N (p={p:.3f})"
    assert r2 > 0.95, f"analog cost is not a clean power law (R^2={r2:.3f})"
