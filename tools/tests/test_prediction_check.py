from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.prediction_check_core import (
    check_corpus,
    check_run,
    detect_declared_tool_activity,
    detect_goal_lifecycle_activity,
    detect_superseded_model_unreachability,
    load_corpus_manifest,
)


def write_checkpoint(
    game: Path,
    index: int,
    *,
    source: str,
    model: bytes | None = None,
    tools: list[str] = (),
    history_invoked: list[str] = (),
    response_invoked: list[str] = (),
    extra_files: dict[str, bytes] | None = None,
) -> None:
    call = game / "decision-checkpoints" / "calls" / f"{index:06d}"
    call.mkdir(parents=True)
    files = dict(extra_files or {})
    if model is not None:
        files["world_model.py"] = model
    manifest = []
    for path, content in files.items():
        digest = hashlib.sha256(content).hexdigest()
        blob = game / "decision-checkpoints" / "blobs" / "sha256" / digest
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(content)
        manifest.append(
            {"path": path, "sha256": digest, "size": len(content), "mode": 0o644}
        )
    (call / "workspace.json").write_text(
        json.dumps({"directories": [], "files": manifest})
    )
    (call / "request.json").write_text(
        json.dumps(
            {
                "call_index": index,
                "source": source,
                "request_body": {
                    "model": "test",
                    "tools": [{"name": name, "parameters": {}} for name in tools],
                    "messages": [_assistant_message(history_invoked)],
                },
            }
        )
    )
    (call / "response.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "tool_calls": [
                    {"id": f"r{i}", "name": name, "arguments": "{}"}
                    for i, name in enumerate(response_invoked)
                ],
                "assistant_message": _assistant_message(response_invoked),
            }
        )
    )


def _assistant_message(names: list[str]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"c{i}",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for i, name in enumerate(names)
        ],
    }


def checkpoints(game: Path) -> list[dict]:
    from tools.prediction_check_core import _checkpoints

    return list(_checkpoints(game))


def write_final_file(game: Path, path: str, content: bytes) -> None:
    target = game / "workspace" / "ar25" / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def write_tool_trace(game: Path, records: list[dict]) -> None:
    game.mkdir(parents=True, exist_ok=True)
    (game / "tool-trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def goals_file(game: Path, goals: list[dict]) -> None:
    path = game / "workspace" / "ar25" / "epistemic" / "candidate-goals.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(goals))


def test_rollback_reports_superseded_model_unreachability(tmp_path: Path) -> None:
    game = tmp_path / "game"
    write_checkpoint(game, 1, source="builder", model=b"v1")
    write_checkpoint(game, 2, source="builder", model=b"v2-candidate")
    write_checkpoint(game, 3, source="builder", model=b"v1")
    write_final_file(game, "world_model.py", b"v1")

    result = detect_superseded_model_unreachability(game, checkpoints(game))

    assert result["status"] == "observed"
    assert result["unique_versions"] == 2
    assert result["version_changes"] == 2
    assert result["superseded_versions"] == 1
    assert [row["call"] for row in result["reverts"]] == ["000003"]
    assert result["analyst_recoverable"] is True
    assert result["agent_recoverable_paths"] == []
    assert result["value_assessed"] is False


def test_candidate_added_after_last_checkpoint_is_reachable(tmp_path: Path) -> None:
    game = tmp_path / "game"
    write_checkpoint(game, 1, source="builder", model=b"v1")
    write_checkpoint(game, 2, source="builder", model=b"v2-candidate")
    write_checkpoint(game, 3, source="builder", model=b"v1")
    write_final_file(game, "world_model.py", b"v1")
    write_final_file(game, "candidates/world_model_002.py", b"v2-candidate")

    result = detect_superseded_model_unreachability(game, checkpoints(game))

    assert result["status"] == "not_observed"
    assert result["agent_recoverable_paths"] == [
        "ar25/candidates/world_model_002.py"
    ]


def test_archive_named_file_with_wrong_content_does_not_clear_observation(
    tmp_path: Path,
) -> None:
    game = tmp_path / "game"
    write_checkpoint(game, 1, source="builder", model=b"v1")
    write_checkpoint(game, 2, source="builder", model=b"v2-candidate")
    write_checkpoint(game, 3, source="builder", model=b"v1")
    write_final_file(game, "world_model.py", b"v1")
    write_final_file(game, "candidate_archive.py", b"unrelated")

    result = detect_superseded_model_unreachability(game, checkpoints(game))

    assert result["status"] == "observed"
    assert result["agent_recoverable_paths"] == []


def test_single_model_run_has_no_superseded_model(tmp_path: Path) -> None:
    game = tmp_path / "game"
    write_checkpoint(game, 1, source="actor", model=b"v1")
    write_checkpoint(game, 2, source="actor", model=b"v1")
    write_final_file(game, "world_model.py", b"v1")

    result = detect_superseded_model_unreachability(game, checkpoints(game))

    assert result["status"] == "not_observed"


def test_declared_tool_activity_separates_attempt_and_completion(tmp_path: Path) -> None:
    game = tmp_path / "game"
    write_checkpoint(
        game,
        1,
        source="actor",
        tools=["take_action", "write_file"],
        response_invoked=["take_action"],
    )
    write_tool_trace(
        game,
        [{"name": "take_action", "status": "complete"}],
    )

    result = detect_declared_tool_activity(game, checkpoints(game))

    actor = result["by_source"]["actor"]
    assert result["status"] == "observed"
    assert actor["attempted"] == ["take_action"]
    assert actor["completed"] == ["take_action"]
    assert actor["offered_but_never_attempted"] == ["write_file"]


def test_final_response_tool_attempt_is_not_right_censored(tmp_path: Path) -> None:
    game = tmp_path / "game"
    write_checkpoint(
        game,
        1,
        source="actor",
        tools=["take_action"],
        response_invoked=["take_action"],
    )

    result = detect_declared_tool_activity(game, checkpoints(game))

    assert result["status"] == "not_observed"
    assert result["by_source"]["actor"]["attempted"] == ["take_action"]
    assert result["by_source"]["actor"]["attempted_but_not_completed"] == [
        "take_action"
    ]


def test_activity_across_checkpoints_clears_unattempted_affordance(
    tmp_path: Path,
) -> None:
    game = tmp_path / "game"
    write_checkpoint(
        game,
        1,
        source="actor",
        tools=["take_action", "write_file"],
        response_invoked=["take_action"],
    )
    write_checkpoint(
        game,
        2,
        source="actor",
        tools=["take_action", "write_file"],
        response_invoked=["write_file"],
    )

    result = detect_declared_tool_activity(game, checkpoints(game))

    assert result["status"] == "not_observed"


def test_frozen_goal_uses_tool_events_not_raw_identifier_mentions(
    tmp_path: Path,
) -> None:
    game = tmp_path / "game"
    write_checkpoint(
        game,
        1,
        source="actor",
        tools=["propose_goal", "update_goal", "take_action"],
    )
    request = game / "decision-checkpoints" / "calls" / "000001" / "request.json"
    request.write_text(request.read_text().replace('"model": "test"', '"model": "goal-0001"'))
    goals_file(game, [{"id": "goal-0001", "revision": 1, "status": "proposed"}])
    write_tool_trace(game, [{"name": "propose_goal", "status": "complete"}])

    result = detect_goal_lifecycle_activity(game, checkpoints(game))

    assert result["status"] == "frozen_after_proposal"
    assert result["activity"]["propose_goal"]["completed"] == 1
    assert result["activity"]["update_goal"]["attempted"] == 0
    assert "checkpoint_references" not in result


def test_goal_update_event_and_revision_report_advanced(tmp_path: Path) -> None:
    game = tmp_path / "game"
    write_checkpoint(
        game,
        1,
        source="actor",
        tools=["propose_goal", "update_goal"],
    )
    goals_file(game, [{"id": "goal-0001", "revision": 3, "status": "rejected"}])
    write_tool_trace(
        game,
        [
            {"name": "propose_goal", "status": "complete"},
            {"name": "update_goal", "status": "complete"},
        ],
    )

    result = detect_goal_lifecycle_activity(game, checkpoints(game))

    assert result["status"] == "advanced"
    assert result["max_revision"] == 3
    assert result["activity"]["update_goal"]["completed"] == 1


def test_missing_goals_artifact_reports_exposure_without_inventing_epoch_claim(
    tmp_path: Path,
) -> None:
    game = tmp_path / "game"
    write_checkpoint(game, 1, source="actor", tools=["propose_goal"])

    result = detect_goal_lifecycle_activity(game, checkpoints(game))

    assert result["status"] == "no_goal_created"


def test_run_without_checkpoints_is_unsupported(tmp_path: Path) -> None:
    empty = tmp_path / "empty-run"
    empty.mkdir()

    result = check_run(empty)

    assert result["inspectable"] is False


def test_corpus_tally_counts_only_reviewed_eligible_runs(tmp_path: Path) -> None:
    admitted = tmp_path / "admitted"
    excluded = tmp_path / "excluded"
    write_checkpoint(admitted / "game", 1, source="actor", tools=["unused"])
    write_checkpoint(excluded / "game", 1, source="actor", tools=["other"])
    eligibility = {
        admitted.resolve(): {"reviewed": True, "eligible": True, "reason": "valid"},
        excluded.resolve(): {
            "reviewed": True,
            "eligible": False,
            "reason": "invalid prefix",
        },
    }

    result = check_corpus([admitted, excluded], eligibility=eligibility)

    assert result["status"] == "pass"
    assert result["corpus"] == {
        "observed_runs": 2,
        "inspectable_runs": 2,
        "eligible_runs": 1,
        "excluded_runs": 1,
        "unreviewed_runs": 0,
    }
    assert result["tallies"]["declared_tool_activity"] == {"observed": 1}


def test_unreviewed_discovered_run_requires_review(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_checkpoint(run / "game", 1, source="actor", tools=["unused"])
    eligibility = {
        run.resolve(): {
            "reviewed": False,
            "eligible": False,
            "reason": "not listed",
        }
    }

    result = check_corpus([run], eligibility=eligibility)

    assert result["status"] == "review_required"
    assert result["tallies"]["declared_tool_activity"] == {}


def test_manifest_loads_relative_paths_and_rejects_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {"run": "runs/a", "eligible": True, "reason": "selected"}
                ]
            }
        )
    )

    paths, eligibility = load_corpus_manifest(manifest, root)

    assert paths == [(root / "runs/a").resolve()]
    assert eligibility[paths[0]]["eligible"] is True
    assert eligibility[paths[0]]["reviewed"] is True

    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {"run": "runs/a", "eligible": True},
                    {"run": "runs/a", "eligible": False},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus_manifest(manifest, root)
