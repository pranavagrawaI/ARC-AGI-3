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


NEXT_SCRATCH_BOUND_BYTES = 8_000_000
STORAGE_PUBLICATION_MARGIN_BYTES = 1_000_000


def expected_storage_projection() -> dict[str, int]:
    return {
        "next_scratch_bound_bytes": NEXT_SCRATCH_BOUND_BYTES,
        "storage_publication_margin_bytes": STORAGE_PUBLICATION_MARGIN_BYTES,
    }


def publish_storage_ledger(
    harness: ModuleType,
    root: Path,
    *,
    attempted: list[str],
    completed: list[str],
    qualification_status: str,
) -> dict[str, object]:
    ledger_path = root / "storage-ledger.jsonl"
    storage_root = root.resolve()
    component_names = {
        "storage_root",
        "predecessor",
        "checkpoint",
        "workspace",
        "scratch",
        "published_bundle",
        "failure_evidence",
    }
    event_games: list[tuple[str, str | None]] = [
        ("after_predecessor_transport_import", None),
        ("after_predecessor_copy", None),
    ]
    for game in attempted:
        event_games.extend(
            [
                ("pre_game_admission", game),
                ("post_game_prepublication", game),
            ]
        )
        if game in completed:
            event_games.extend(
                [
                    ("post_publication_pre_cleanup", game),
                    ("post_publication_cleanup", game),
                ]
            )
        elif qualification_status == "partial_failure":
            event_games.extend(
                [("pre_failure_copy", game), ("post_failure_copy", game)]
            )
    rows = []
    required = 2 * NEXT_SCRATCH_BOUND_BYTES + STORAGE_PUBLICATION_MARGIN_BYTES
    for sequence, (event, game) in enumerate(event_games):
        scratch_exists = event in {
            "post_game_prepublication",
            "post_publication_pre_cleanup",
            "pre_failure_copy",
            "post_failure_copy",
        }
        failure_exists = event == "post_failure_copy"
        components = {
            name: {
                "path": str(storage_root / name),
                "exists": (
                    scratch_exists
                    if name == "scratch"
                    else failure_exists
                    if name == "failure_evidence"
                    else True
                ),
                "regular_file_bytes": 1,
            }
            for name in component_names
        }
        admission = None
        if event == "pre_game_admission":
            admission = {
                "next_scratch_bound_bytes": NEXT_SCRATCH_BOUND_BYTES,
                "storage_publication_margin_bytes": (
                    STORAGE_PUBLICATION_MARGIN_BYTES
                ),
                "required_available_bytes": required,
                "available_bytes": required + 1,
                "passed": True,
            }
        rows.append(
            {
                "schema": harness.STORAGE_LEDGER_SCHEMA,
                "sequence": sequence,
                "captured_at": f"2026-09-01T00:00:{sequence:02d}+00:00",
                "event": event,
                "game": game,
                "storage_root": str(storage_root),
                "filesystem": {
                    "fragment_bytes": 4096,
                    "total_bytes": required * 2,
                    "free_bytes": required + 1,
                    "available_bytes": required + 1,
                },
                "components": components,
                "admission": admission,
            }
        )
    ledger_path.write_bytes(b"".join(harness.canonical_json(row) for row in rows))
    return {
        "ledger_schema": harness.STORAGE_LEDGER_SCHEMA,
        "ledger_path": str(ledger_path),
        "ledger_sha256": harness.sha256_file(ledger_path),
        "ledger_record_count": len(rows),
        "next_scratch_bound_bytes": NEXT_SCRATCH_BOUND_BYTES,
        "storage_publication_margin_bytes": STORAGE_PUBLICATION_MARGIN_BYTES,
    }


def partial_failure_fixture(
    tmp_path: Path,
    *,
    trace_records: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, ModuleType]:
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
    if trace_records is not None:
        session = failure_root / "solver-workspaces" / f"{game}-session"
        session.mkdir(parents=True)
        (session / "llm-trace.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in trace_records)
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
    storage_safety = publish_storage_ledger(
        harness,
        tmp_path,
        attempted=[game],
        completed=[],
        qualification_status="partial_failure",
    )
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
            "storage_safety": storage_safety,
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
    storage_safety = publish_storage_ledger(
        harness,
        tmp_path,
        attempted=attempted_this_run,
        completed=completed_this_run,
        qualification_status=qualification_status,
    )
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
            "storage_safety": storage_safety,
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
        expected_storage_projection=expected_storage_projection(),
    )

    assert result == {
        "schema": "tycho-nooa-public25-outer-callback-contract-v2",
        "status": "passed",
        "qualification_status": "partial_failure",
        "expected_solver_requests": 0,
        "observed_solver_requests": 0,
        "whole_model_request_count_verified": True,
        "zero_request_disposition": "manifest_valid_pre_request_partial_failure",
        "checkpoint_verified": True,
        "storage_ledger_verified": True,
        "storage_ledger_sha256": result["storage_ledger_sha256"],
        "storage_ledger_record_count": 6,
        "lifecycle_requirement": "accepted_r8_l3b_term_only_unchanged",
        "notebook_subprocess_requirement": "check_true",
    }


def test_callback_rejects_absent_storage_ledger_binding(tmp_path: Path) -> None:
    callback_path, checkpoint_root, harness = partial_failure_fixture(tmp_path)
    callback = json.loads(callback_path.read_text())
    callback.pop("storage_safety")
    harness.write_json_atomic(callback_path, callback)
    contract = load(CONTRACT_PATH, "outer_callback_contract_missing_storage")

    with pytest.raises(RuntimeError, match="storage"):
        contract.validate_outer_callback(
            callback_path,
            checkpoint_root,
            warmup_facts=request_facts(1),
            post_callback_facts=request_facts(1),
            expected_storage_projection=expected_storage_projection(),
        )


def test_callback_rejects_storage_ledger_hash_count_or_bound_drift(
    tmp_path: Path,
) -> None:
    contract = load(CONTRACT_PATH, "outer_callback_contract_storage_binding")
    for mutation, match in (
        ("hash", "ledger hash"),
        ("count", "record count"),
        ("bound", "frozen contract"),
    ):
        callback_path, checkpoint_root, harness = partial_failure_fixture(
            tmp_path / mutation
        )
        callback = json.loads(callback_path.read_text())
        if mutation == "hash":
            callback["storage_safety"]["ledger_sha256"] = "0" * 64
        elif mutation == "count":
            callback["storage_safety"]["ledger_record_count"] += 1
        else:
            callback["storage_safety"]["next_scratch_bound_bytes"] += 1
            ledger_path = Path(callback["storage_safety"]["ledger_path"])
            rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            changed_bound = callback["storage_safety"]["next_scratch_bound_bytes"]
            required = 2 * changed_bound + STORAGE_PUBLICATION_MARGIN_BYTES
            for row in rows:
                if row["event"] == "pre_game_admission":
                    row["admission"]["next_scratch_bound_bytes"] = changed_bound
                    row["admission"]["required_available_bytes"] = required
                    row["admission"]["available_bytes"] = required + 1
            ledger_path.write_bytes(
                b"".join(harness.canonical_json(row) for row in rows)
            )
            callback["storage_safety"]["ledger_sha256"] = harness.sha256_file(
                ledger_path
            )
        harness.write_json_atomic(callback_path, callback)

        with pytest.raises(RuntimeError, match=match):
            contract.validate_outer_callback(
                callback_path,
                checkpoint_root,
                warmup_facts=request_facts(1),
                post_callback_facts=request_facts(1),
                expected_storage_projection=expected_storage_projection(),
            )


def test_callback_rejects_storage_event_or_cleanup_drift(tmp_path: Path) -> None:
    contract = load(CONTRACT_PATH, "outer_callback_contract_storage_events")
    for mutation, match in (
        ("event", "event sequence"),
        ("cleanup", "cleanup evidence"),
    ):
        roster = load(HARNESS_PATH, f"outer_callback_storage_{mutation}")
        game = roster.RUN_ORDER[0]
        callback_path, checkpoint_root, harness = callback_fixture(
            tmp_path / mutation,
            completed_games=[game],
            attempted_this_run=[game],
            completed_this_run=[game],
            qualification_status="partial_budget",
        )
        callback = json.loads(callback_path.read_text())
        ledger_path = Path(callback["storage_safety"]["ledger_path"])
        rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        if mutation == "event":
            rows[-1]["event"] = "invented_cleanup"
        else:
            rows[-1]["components"]["scratch"]["exists"] = True
        ledger_path.write_bytes(
            b"".join(harness.canonical_json(row) for row in rows)
        )
        callback["storage_safety"]["ledger_sha256"] = harness.sha256_file(
            ledger_path
        )
        harness.write_json_atomic(callback_path, callback)

        with pytest.raises(RuntimeError, match=match):
            contract.validate_outer_callback(
                callback_path,
                checkpoint_root,
                warmup_facts=request_facts(1),
                post_callback_facts=request_facts(
                    harness.FIRST_LEVEL_ACTION_CAPS[game] + 1
                ),
                expected_storage_projection=expected_storage_projection(),
            )


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
            expected_storage_projection=expected_storage_projection(),
        )


def test_partial_failure_reconciles_exact_manifested_actor_trace_rows(
    tmp_path: Path,
) -> None:
    callback_path, checkpoint_root, _ = partial_failure_fixture(
        tmp_path,
        trace_records=[
            {"source": "actor", "phase": "decision", "sequence": 0},
            {"source": "actor", "phase": "decision", "sequence": 1},
        ],
    )
    contract = load(CONTRACT_PATH, "outer_callback_contract_traced_failure")

    with pytest.raises(RuntimeError, match="differs from manifested Actor calls"):
        contract.validate_outer_callback(
            callback_path,
            checkpoint_root,
            warmup_facts=request_facts(1),
            post_callback_facts=request_facts(2),
            expected_storage_projection=expected_storage_projection(),
        )

    result = contract.validate_outer_callback(
        callback_path,
        checkpoint_root,
        warmup_facts=request_facts(1),
        post_callback_facts=request_facts(3),
        expected_storage_projection=expected_storage_projection(),
    )
    assert result["expected_solver_requests"] == 2
    assert result["observed_solver_requests"] == 2
    assert result["zero_request_disposition"] is None


def test_partial_failure_rejects_non_actor_trace_lane(tmp_path: Path) -> None:
    callback_path, checkpoint_root, _ = partial_failure_fixture(
        tmp_path,
        trace_records=[
            {"source": "scribe", "phase": "post_action", "sequence": 0}
        ],
    )
    contract = load(CONTRACT_PATH, "outer_callback_contract_hidden_lane")

    with pytest.raises(RuntimeError, match="hidden lane"):
        contract.validate_outer_callback(
            callback_path,
            checkpoint_root,
            warmup_facts=request_facts(1),
            post_callback_facts=request_facts(2),
            expected_storage_projection=expected_storage_projection(),
        )


def test_complete_callback_requires_exact_manifested_solver_requests(
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

    expected = sum(harness.FIRST_LEVEL_ACTION_CAPS.values())
    with pytest.raises(RuntimeError, match="differs from manifested Actor calls"):
        contract.validate_outer_callback(
            callback_path,
            checkpoint_root,
            warmup_facts=request_facts(1),
            post_callback_facts=request_facts(1),
            expected_storage_projection=expected_storage_projection(),
        )

    result = contract.validate_outer_callback(
        callback_path,
        checkpoint_root,
        warmup_facts=request_facts(1),
        post_callback_facts=request_facts(expected + 1),
        expected_storage_projection=expected_storage_projection(),
    )
    assert result["expected_solver_requests"] == expected
    assert result["observed_solver_requests"] == expected
    assert result["whole_model_request_count_verified"] is True
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
        expected_storage_projection=expected_storage_projection(),
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
    with pytest.raises(RuntimeError, match="differs from manifested Actor calls"):
        contract.validate_outer_callback(
            current_path,
            current_checkpoint,
            warmup_facts=request_facts(1),
            post_callback_facts=request_facts(1),
            expected_storage_projection=expected_storage_projection(),
        )

    exact = contract.validate_outer_callback(
        current_path,
        current_checkpoint,
        warmup_facts=request_facts(1),
        post_callback_facts=request_facts(
            harness.FIRST_LEVEL_ACTION_CAPS[game] + 1
        ),
        expected_storage_projection=expected_storage_projection(),
    )
    assert exact["expected_solver_requests"] == harness.FIRST_LEVEL_ACTION_CAPS[game]
