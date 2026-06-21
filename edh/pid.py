"""
edh.pid — thin wrapper over the validated reference BROJA PID solver.

We do NOT reinvent the solver (§4 of the spec). This module:
  1. Registers the module alias ``BROJA_2PID`` -> ``broja2pid.BROJA_2PID`` so the
     reference wrapper's ``from BROJA_2PID import pid`` resolves against the
     pip-installed ``broja2pid`` package (the upstream repo ships the package
     under the lowercase name).
  2. Puts ``reference/`` on sys.path and re-exports the validated symbols from
     ``reference/pid_fast.py``.

Re-exported: ``pid``, ``pid_bias_corrected``, ``PIDResult``.
The canonical-gate self-test lives in tests/test_pid_gates.py.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# --- 1. Make `from BROJA_2PID import pid` resolve to the installed package. ---
if "BROJA_2PID" not in sys.modules:
    try:
        sys.modules["BROJA_2PID"] = importlib.import_module("broja2pid.BROJA_2PID")
    except Exception:  # pragma: no cover - import errors surface at pid() call
        # Leave it unregistered; pid_fast handles the missing-solver case with a
        # clear ImportError when pid() is actually invoked.
        pass

# --- 2. Import the validated reference wrapper. ---
_REF = Path(__file__).resolve().parent.parent / "reference"
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))

from pid_fast import (  # noqa: E402  (path manipulation must precede import)
    PIDResult,
    pid,
    pid_bias_corrected,
)

__all__ = ["pid", "pid_bias_corrected", "PIDResult"]
