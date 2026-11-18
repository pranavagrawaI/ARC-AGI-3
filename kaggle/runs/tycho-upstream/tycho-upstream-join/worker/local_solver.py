"""create_solver for the Tycho-upstream join: shared LocalSolver, Tycho worker command.

The shared adapter forwards only an explicit allowlist of environment variables to
the worker, so an unlisted variable is silently dropped rather than misapplied.
Upstream Tycho is configured entirely through ``TYCHO_*`` and ``LLM_*``, so this
arm widens that allowlist before any solver is built.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from tycho_nooa.protocol import ObservationPayload

from runtime import local_solver as shared

# Upstream Tycho reads its whole configuration from the environment
# (``TychoSettings.from_env``, ``LLMConfig.from_env``, ``resolve_vision``,
# ``resolve_context_config``, ``resolve_mode``). Every name below is an upstream
# setting; the two ``TYCHO_SYNTHETIC_*`` names belong to this join's worker.
TYCHO_UPSTREAM_ENVIRONMENT_KEYS = frozenset(
    {
        "LLM_API_KEY",
        "LLM_BACKEND",
        "LLM_BASE_URL",
        "LLM_EFFORT",
        "LLM_LOG",
        "LLM_MODEL",
        "LLM_PROMPT_CACHE",
        "MPLBACKEND",
        "OPENAI_BASE_URL",
        "TYCHO_AGENT_GRAPH",
        "TYCHO_ANIMATION_SUMMARY",
        "TYCHO_CACHE_ROLLING_EVERY",
        "TYCHO_CONTEXT_EMERGENCY_SOFT_TOKENS",
        "TYCHO_EDIT_FUNC",
        "TYCHO_EFFORT",
        "TYCHO_GRID_KEEP",
        "TYCHO_HISTORY",
        "TYCHO_IMAGE_CAP",
        "TYCHO_IMAGE_EVICT_MULT",
        "TYCHO_IMAGE_KEEP",
        "TYCHO_MAX_INFERENCE_COST_PER_GAME",
        "TYCHO_MAX_INFERENCE_COST_PER_LEVEL",
        "TYCHO_MAX_LLM_CALLS",
        "TYCHO_MAX_TOKENS",
        "TYCHO_MAX_TOKENS_RETRY",
        "TYCHO_MAX_TOOL_STEPS",
        "TYCHO_META_REFLECT",
        "TYCHO_PROMPT_CACHING",
        "TYCHO_REASONING_KEEP",
        "TYCHO_RENDER_SCALE",
        "TYCHO_SYNTHETIC_BASELINE",
        "TYCHO_SYNTHETIC_LEVELS",
        "TYCHO_TEXT_GRID",
        "TYCHO_TEXT_KEEP",
        "TYCHO_TOOL_FLAVOR",
        "TYCHO_UPSTREAM_JOIN_ROOT",
        "TYCHO_VISION",
    }
)

# `worker_environment` reads this module global at call time, so rebinding it here
# widens the forwarded set for every solver this process creates.
shared.WORKER_ENVIRONMENT_KEYS = frozenset(
    shared.WORKER_ENVIRONMENT_KEYS | TYCHO_UPSTREAM_ENVIRONMENT_KEYS
)

JOIN_ROOT_ENV = "TYCHO_UPSTREAM_JOIN_ROOT"


def _join_root() -> Path:
    raw = os.environ.get(JOIN_ROOT_ENV)
    return Path(raw) if raw else Path(__file__).resolve().parent


def create_solver(
    *,
    game_id: str,
    workspace_root: Path,
    initial_observation: ObservationPayload,
) -> shared.LocalSolver:
    worker_python = os.environ.get("TYCHO_NOOA_WORKER_PYTHON", sys.executable)
    timeout = shared._positive_timeout(os.environ.get(shared.WORKER_TIMEOUT_ENV))
    environment = dict(os.environ)
    environment.setdefault(shared.LOCAL_MODEL_COST_MAP_ENV, "True")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault(JOIN_ROOT_ENV, str(_join_root()))
    if environment.get(shared.HARDENED_HOST_ENV) == "1":
        shared._require_hardened_host_supervisor(environment)
        identity = shared._identity_for_game(game_id)
        environment[shared.GAME_UID_ENV] = str(identity)
        environment[shared.GAME_GID_ENV] = str(identity)
    worker = _join_root() / "worker" / "tycho_worker.py"
    return shared.LocalSolver(
        game_id=game_id,
        workspace_root=workspace_root,
        initial_observation=initial_observation,
        command=[worker_python, str(worker)],
        timeout_seconds=timeout,
        environment=environment,
    )
