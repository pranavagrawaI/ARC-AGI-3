"""create_solver for the NOOA join: same LocalSolver, q3c worker command."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from tycho_nooa.protocol import ObservationPayload

from runtime.local_solver import (
    LOCAL_MODEL_COST_MAP_ENV,
    WORKER_TIMEOUT_ENV,
    LocalSolver,
    _positive_timeout,
)

JOIN_ROOT_ENV = "NOOA_JOIN_ROOT"


def _join_root() -> Path:
    raw = os.environ.get(JOIN_ROOT_ENV)
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent


def create_solver(
    *,
    game_id: str,
    workspace_root: Path,
    initial_observation: ObservationPayload,
) -> LocalSolver:
    worker_python = os.environ.get("TYCHO_NOOA_WORKER_PYTHON", sys.executable)
    timeout = _positive_timeout(os.environ.get(WORKER_TIMEOUT_ENV))
    environment = dict(os.environ)
    environment.setdefault(LOCAL_MODEL_COST_MAP_ENV, "True")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault(JOIN_ROOT_ENV, str(_join_root()))
    worker = _join_root() / "nooa_worker.py"
    return LocalSolver(
        game_id=game_id,
        workspace_root=workspace_root,
        initial_observation=initial_observation,
        command=[worker_python, str(worker)],
        timeout_seconds=timeout,
        environment=environment,
    )
