"""Import bridge for the hyphenated S0 deployment artifact directory."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_probe() -> ModuleType:
    path = Path(__file__).with_name("s0-solver-join") / "solver_join_probe.py"
    spec = importlib.util.spec_from_file_location("tycho_s0_solver_join_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load solver join probe: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
