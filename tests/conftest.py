"""Shared test fixtures / path setup.

Puts the project root (so ``import edh``) and ``reference/`` (so the reference
modules import directly) on sys.path, regardless of the pytest invocation dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "reference", _ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
