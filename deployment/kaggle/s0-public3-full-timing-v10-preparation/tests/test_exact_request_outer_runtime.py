from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "exact_request_outer_runtime.py"


def load_runtime() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exact_request_outer_runtime", RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launcher_result() -> dict:
    return {
        "status": "passed",
        "solver_sessions": {
            "schema": "tycho-nooa-public-solver-sessions-v1",
            "game_count": 3,
            "total_actor_calls": 8,
            "games": {
                "ls20": {
                    "model_evidence": {
                        "actor_call_count": 2,
                        "actor_calls_by_sequence": {"0": 1, "1": 1},
                    }
                },
                "s5i5": {
                    "model_evidence": {
                        "actor_call_count": 4,
                        "actor_calls_by_sequence": {"0": 3, "1": 1},
                    }
                },
                "sb26": {
                    "model_evidence": {
                        "actor_call_count": 2,
                        "actor_calls_by_sequence": {"0": 1, "1": 1},
                    }
                },
            },
        },
    }


def warmup() -> dict:
    return {
        "chat_completion_accesses": 1,
        "successful_chat_completion_accesses": 1,
        "successful_freeze_gc_accesses": 1,
        "ready_marker_count": 1,
    }


def post_requests(*, total: int = 9, successful: int = 9) -> dict:
    return {
        "chat_completion_accesses": total,
        "successful_chat_completion_accesses": successful,
        "successful_freeze_gc_accesses": 1,
        "ready_marker_count": 1,
    }


def test_exact_actor_calls_equal_exact_provider_requests() -> None:
    runtime = load_runtime()

    facts = runtime.reconcile_launcher_request_count(
        launcher_result(), warmup(), post_requests()
    )

    assert facts["expected_launcher_solver_requests"] == 8
    assert facts["observed_launcher_solver_requests"] == 8
    assert facts["expected_total_chat_completion_accesses"] == 9
    assert facts["whole_model_request_count_verified"] is True


@pytest.mark.parametrize(
    ("total", "successful"),
    [
        (8, 8),
        (10, 10),
        (9, 8),
        (10, 9),
    ],
)
def test_missing_extra_or_failed_provider_request_is_rejected(
    total: int, successful: int
) -> None:
    runtime = load_runtime()

    with pytest.raises(RuntimeError, match="request-count contract did not close"):
        runtime.reconcile_launcher_request_count(
            launcher_result(),
            warmup(),
            post_requests(total=total, successful=successful),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("solver_sessions"),
        lambda value: value["solver_sessions"].update(schema="wrong"),
        lambda value: value["solver_sessions"].update(total_actor_calls=7),
        lambda value: value["solver_sessions"]["games"]["s5i5"][
            "model_evidence"
        ].update(actor_call_count=3),
        lambda value: value["solver_sessions"]["games"]["s5i5"][
            "model_evidence"
        ].update(actor_calls_by_sequence={"0": 3, "1": True}),
    ],
)
def test_missing_or_nonreconciling_solver_evidence_is_rejected(mutation) -> None:
    runtime = load_runtime()
    result = launcher_result()
    mutation(result)

    with pytest.raises((RuntimeError, TypeError), match="official launcher"):
        runtime.expected_launcher_solver_requests(result)


def test_v10_preparation_does_not_mutate_running_v9_candidate() -> None:
    source = RUNTIME.read_text()
    running_runtime = RUNTIME.parents[1] / "s0-official-launcher-kernel" / (
        "official_launcher_runtime.py"
    )

    assert RUNTIME != running_runtime
    assert "minimum_launcher_solver_requests" not in source
    assert "whole_model_request_count_verified" in source
    assert "minimum_launcher_solver_requests" in running_runtime.read_text()
