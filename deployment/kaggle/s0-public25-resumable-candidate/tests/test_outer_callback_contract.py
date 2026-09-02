from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

HERE = Path(__file__).resolve().parents[1]
HARNESS_PATH = HERE / "resumable_public25.py"
CONTRACT_PATH = HERE / "outer_callback_contract.py"


def load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def request_facts(total: int) -> dict[str, int]:
    return {
        "chat_completion_accesses": total,
        "successful_chat_completion_accesses": total,
        "successful_freeze_gc_accesses": 1,
        "ready_marker_count": 1,
    }


def partial_failure_fixture(tmp_path: Path) -> tuple[Path, Path, ModuleType]:
    harness = load(HARNESS_PATH, "outer_callback_harness_failure")
    contract_sha256 = "1" * 64
    checkpoint_root = tmp_path / "checkpoint"
    (checkpoint_root / "games").mkdir(parents=True)
    checkpoint = harness.empty_checkpoint(contract_sha256)
    game = harness.RUN_ORDER[0]
    failure_root = checkpoint_root / "failures" / game
    failure_root.mkdir(parents=True)
    (failure_root / "orchestrator-command.json").write_text(
        json.dumps({"returncode": 70, "stdout": "", "stderr": "failed\n"})
        + "\n"
    )
    checkpoint["failures"].append(
        {
            "game": game,
            "returncode": 70,
            "result_status": None,
            "files": harness.regular_file_manifest(failure_root),
        }
    )
    harness.write_json_atomic(checkpoint_root / "checkpoint.json", checkpoint)
    callback_path = tmp_path / "callback-result.json"
    harness.write_json_atomic(
        callback_path,
        {
            "schema": harness.RESULT_SCHEMA,
            "status": "passed",
            "qualification_status": "partial_failure",
            "all25_complete": False,
            "contract_sha256": contract_sha256,
            "checkpoint_manifest_sha256": harness.sha256_file(
                checkpoint_root / "checkpoint.json"
            ),
            "games_attempted_this_run": [game],
            "games_completed_this_run": [],
            "failure": {
                "game": game,
                "returncode": 70,
                "result_status": None,
            },
            "metrics": harness.aggregate(checkpoint),
        },
    )
    return callback_path, checkpoint_root, harness


def publish_fake_game(
    harness: ModuleType,
    checkpoint_root: Path,
    checkpoint: dict[str, object],
    scratch_root: Path,
    game: str,
) -> None:
    scratch = scratch_root / game
    session = scratch / "solver-workspaces" / f"{game}-session"
    session.mkdir(parents=True)
    cap = harness.FIRST_LEVEL_ACTION_CAPS[game]
    (session / "llm-trace.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "reasoning_tokens": 3,
                }
            )
            + "\n"
            for _ in range(cap)
        )
    )
    (scratch / "recordings").mkdir()
    (scratch / "recordings" / f"{game}.recording.jsonl").write_text("{}\n")
    (scratch / "public25-launcher.stdout.log").write_text("passed\n")
    (scratch / "public25-launcher.stderr.log").write_text("")
    (scratch / "orchestrator-command.json").write_text("{}\n")
    official = scratch / "ARC-AGI-3-Agents"
    official.mkdir()
    (official / "logs.log").write_text("scorecard\n")
    result = {
        "status": "passed",
        "public_games": [game],
        "public_progress": {
            "game_count": 1,
            "games": {
                game: {
                    "final_state": "NOT_FINISHED",
                    "final_levels_completed": 1,
                    "max_levels_completed": 1,
                    "recorded_action_results": cap,
                }
            },
        },
        "solver_sessions": {
            "game_count": 1,
            "total_solver_actions": cap,
            "total_actor_calls": cap,
            "games": {game: {"session": str(session)}},
        },
    }
    (scratch / "game-result.json").write_text(json.dumps(result) + "\n")
    harness.publish_completed_game(
        scratch,
        checkpoint_root,
        checkpoint,
        game=game,
        result=result,
    )


def callback_fixture(
    tmp_path: Path,
    *,
    completed_games: list[str],
    attempted_this_run: list[str],
    completed_this_run: list[str],
    qualification_status: str,
) -> tuple[Path, Path, ModuleType]:
    harness = load(HARNESS_PATH, f"outer_callback_harness_{qualification_status}")
    contract_sha256 = "3" * 64
    checkpoint_root = tmp_path / "checkpoint"
    (checkpoint_root / "games").mkdir(parents=True)
    checkpoint = harness.empty_checkpoint(contract_sha256)
    for game in completed_games:
        publish_fake_game(
            harness, checkpoint_root, checkpoint, tmp_path / "scratch", game
        )
    harness.write_json_atomic(checkpoint_root / "checkpoint.json", checkpoint)
    callback_path = tmp_path / "callback-result.json"
    harness.write_json_atomic(
        callback_path,
        {
            "schema": harness.RESULT_SCHEMA,
            "status": "passed",
            "qualification_status": qualification_status,
            "all25_complete": qualification_status == "complete",
            "contract_sha256": contract_sha256,
            "checkpoint_manifest_sha256": harness.sha256_file(
                checkpoint_root / "checkpoint.json"
            ),
            "games_attempted_this_run": attempted_this_run,
            "games_completed_this_run": completed_this_run,
            "failure": None,
            "metrics": harness.aggregate(checkpoint),
        },
    )
    return callback_path, checkpoint_root, harness


def test_manifest_valid_pre_request_partial_failure_accepts_zero_solver_requests(
    tmp_path: Path,
) -> None:
    callback_path, checkpoint_root, _ = partial_failure_fixture(tmp_path)
    contract = load(CONTRACT_PATH, "outer_callback_contract_failure")

    result = contract.validate_outer_callback(
        callback_path,
        checkpoint_root,
        warmup_facts=request_facts(1),
        post_callback_facts=request_facts(1),
    )

    assert result == {
        "schema": "tycho-nooa-public25-outer-callback-contract-v1",
        "status": "passed",
        "qualification_status": "partial_failure",
        "minimum_solver_requests": 0,
        "observed_solver_requests": 0,
        "zero_request_disposition": "manifest_valid_pre_request_partial_failure",
        "checkpoint_verified": True,
        "lifecycle_requirement": "accepted_r8_l3b_term_only_unchanged",
        "notebook_subprocess_requirement": "check_true",
    }


def test_callback_rejects_checkpoint_tampering(tmp_path: Path) -> None:
    callback_path, checkpoint_root, _ = partial_failure_fixture(tmp_path)
    contract = load(CONTRACT_PATH, "outer_callback_contract_tamper")
    with (checkpoint_root / "checkpoint.json").open("a") as handle:
        handle.write(" ")

    with pytest.raises(RuntimeError, match="checkpoint manifest hash differs"):
        contract.validate_outer_callback(
            callback_path,
            checkpoint_root,
            warmup_facts=request_facts(1),
            post_callback_facts=request_facts(1),
        )


def test_complete_callback_requires_at_least_one_solver_request(
    tmp_path: Path,
) -> None:
    harness = load(HARNESS_PATH, "outer_callback_complete_roster")
    # Build a genuinely complete checkpoint rather than weakening callback
    # validation to test request counts in isolation.
    callback_path, checkpoint_root, harness = callback_fixture(
        tmp_path,
        completed_games=list(harness.RUN_ORDER),
        attempted_this_run=list(harness.RUN_ORDER),
        completed_this_run=list(harness.RUN_ORDER),
        qualification_status="complete",
    )
    contract = load(CONTRACT_PATH, "outer_callback_contract_complete")

    with pytest.raises(RuntimeError, match="fewer Solver requests"):
        contract.validate_outer_callback(
            callback_path,
            checkpoint_root,
            warmup_facts=request_facts(1),
            post_callback_facts=request_facts(1),
        )

    result = contract.validate_outer_callback(
        callback_path,
        checkpoint_root,
        warmup_facts=request_facts(1),
        post_callback_facts=request_facts(2),
    )
    assert result["minimum_solver_requests"] == 1
    assert result["observed_solver_requests"] == 1
    assert result["zero_request_disposition"] is None


def test_no_game_partial_budget_accepts_zero_but_current_game_success_does_not(
    tmp_path: Path,
) -> None:
    roster = load(HARNESS_PATH, "outer_callback_budget_roster")
    predecessor_game = roster.RUN_ORDER[0]
    no_game_path, no_game_checkpoint, harness = callback_fixture(
        tmp_path / "no-game",
        completed_games=[predecessor_game],
        attempted_this_run=[],
        completed_this_run=[],
        qualification_status="partial_budget",
    )
    contract = load(CONTRACT_PATH, "outer_callback_contract_budget")
    no_game = contract.validate_outer_callback(
        no_game_path,
        no_game_checkpoint,
        warmup_facts=request_facts(1),
        post_callback_facts=request_facts(1),
    )
    assert no_game["zero_request_disposition"] == (
        "manifest_valid_no_game_partial_budget"
    )

    game = harness.RUN_ORDER[0]
    current_path, current_checkpoint, _ = callback_fixture(
        tmp_path / "current-game",
        completed_games=[game],
        attempted_this_run=[game],
        completed_this_run=[game],
        qualification_status="partial_budget",
    )
    with pytest.raises(RuntimeError, match="fewer Solver requests"):
        contract.validate_outer_callback(
            current_path,
            current_checkpoint,
            warmup_facts=request_facts(1),
            post_callback_facts=request_facts(1),
        )
