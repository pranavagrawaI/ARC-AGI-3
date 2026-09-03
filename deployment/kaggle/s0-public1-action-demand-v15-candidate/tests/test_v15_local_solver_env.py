from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parents[1]
LOCAL_SOLVER = HERE / "local_solver.py"

ACTION_DEMAND_KEYS = (
    "ARC_LLM_ACTION_DEMAND_MAX_TOKENS",
    "ARC_LLM_ACTION_DEMAND_DISABLE_THINKING",
)


def load_local_solver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "v15_candidate_local_solver", LOCAL_SOLVER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_environment_forwards_the_action_demand_commitment_knobs() -> None:
    # This is the exact V14 (Kaggle kernel 12) regression: the notebook set both
    # vars, but the worker-env allowlist dropped them, so the forced
    # action-demand request went out at 16000 tokens with thinking enabled.
    module = load_local_solver()

    for key in ACTION_DEMAND_KEYS:
        assert key in module.WORKER_ENVIRONMENT_KEYS

    forwarded = module.worker_environment(
        {
            "ARC_LLM_ACTION_DEMAND_MAX_TOKENS": "2048",
            "ARC_LLM_ACTION_DEMAND_DISABLE_THINKING": "1",
            "ARC_LLM_MAX_TOKENS": "16000",
            "PATH": "/usr/bin",
            "UNRELATED_SECRET": "do-not-forward",
        }
    )

    assert forwarded["ARC_LLM_ACTION_DEMAND_MAX_TOKENS"] == "2048"
    assert forwarded["ARC_LLM_ACTION_DEMAND_DISABLE_THINKING"] == "1"
    assert "UNRELATED_SECRET" not in forwarded


def test_v15_local_solver_only_adds_the_two_action_demand_keys() -> None:
    module = load_local_solver()
    runtime_solver = HERE.parent / "runtime/local_solver.py"
    runtime_spec = importlib.util.spec_from_file_location(
        "shared_runtime_local_solver", runtime_solver
    )
    assert runtime_spec is not None and runtime_spec.loader is not None
    shared = importlib.util.module_from_spec(runtime_spec)
    runtime_spec.loader.exec_module(shared)

    added = module.WORKER_ENVIRONMENT_KEYS - shared.WORKER_ENVIRONMENT_KEYS
    removed = shared.WORKER_ENVIRONMENT_KEYS - module.WORKER_ENVIRONMENT_KEYS
    assert added == set(ACTION_DEMAND_KEYS)
    assert removed == set()
