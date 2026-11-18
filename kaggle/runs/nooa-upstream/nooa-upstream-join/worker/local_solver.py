"""create_solver for the NOOA-upstream join: shared LocalSolver, NOOA worker command."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from tycho_nooa.protocol import ObservationPayload

from runtime.local_solver import (
    GAME_GID_ENV,
    GAME_UID_ENV,
    HARDENED_HOST_ENV,
    LOCAL_MODEL_COST_MAP_ENV,
    WORKER_TIMEOUT_ENV,
    LocalSolver,
    _identity_for_game,
    _positive_timeout,
    _require_hardened_host_supervisor,
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
    if environment.get(HARDENED_HOST_ENV) == "1":
        _require_hardened_host_supervisor(environment)
        identity = _identity_for_game(game_id)
        environment[GAME_UID_ENV] = str(identity)
        environment[GAME_GID_ENV] = str(identity)
    worker = _join_root() / "worker" / "nooa_worker.py"
    return LocalSolver(
        game_id=game_id,
        workspace_root=workspace_root,
        initial_observation=initial_observation,
        command=[worker_python, str(worker)],
        timeout_seconds=timeout,
        environment=environment,
    )
