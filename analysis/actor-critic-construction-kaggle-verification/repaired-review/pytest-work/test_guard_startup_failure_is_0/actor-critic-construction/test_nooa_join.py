from __future__ import annotations
import importlib.util, sys
from pathlib import Path
from types import ModuleType
from typing import Any
from arcengine import FrameData, GameAction, GameState
CANDIDATE = Path(__file__).resolve().parent / "nooa-join"
def load_create_solver() -> ModuleType:
    spec = importlib.util.spec_from_file_location('s0_nooa_join_local_solver', CANDIDATE / 'local_solver.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
