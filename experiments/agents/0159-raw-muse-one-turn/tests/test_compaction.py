from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EXPERIMENT = Path(__file__).resolve().parents[1]
MODULE_PATH = EXPERIMENT / "compaction.py"
SPEC = importlib.util.spec_from_file_location("compaction", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

COMPARE_PATH = EXPERIMENT / "compare_compaction.py"
COMPARE_SPEC = importlib.util.spec_from_file_location(
    "compare_compaction", COMPARE_PATH
)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(COMPARE_SPEC)
sys.modules[COMPARE_SPEC.name] = COMPARE
COMPARE_SPEC.loader.exec_module(COMPARE)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _synthetic_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "transcript.jsonl",
        [
            {
                "type": "observation",
                "turn": 0,
                "state": "NOT_FINISHED",
                "levels_completed": 0,
                "full_reset": False,
                "legal_actions": ["UP", "DOWN"],
                "trajectory_length": 0,
                "trajectory_sha256": "before",
                "image_path": "/run/turn-000.png",
            },
            {"type": "decision", "sequence": 0, "action": "UP", "explanation": "probe"},
            {
                "type": "observation",
                "turn": 1,
                "state": "NOT_FINISHED",
                "levels_completed": 0,
                "full_reset": False,
                "legal_actions": ["UP", "DOWN"],
                "trajectory_length": 1,
                "trajectory_sha256": "after",
                "image_path": "/run/turn-001.png",
            },
        ],
    )
    _write_jsonl(
        run_dir / "trajectory.jsonl",
        [{"turn": 0, "action": "UP", "before": [[0]], "after": [[1]]}],
    )
    _write_jsonl(
        run_dir / "provider-events.jsonl",
        [
            {
                "event": "provider_call_failed",
                "sequence": 0,
                "message": "visit /keys/not-for-the-model and use api_key",
                "api_key": "secret",
            }
        ],
    )
    knowledge = run_dir / "knowledge"
    knowledge.mkdir()
    (knowledge / "hypotheses.md").write_text("UP moved the object.", encoding="utf-8")
    return run_dir


def _candidate() -> dict:
    return {
        "schema": MODULE.SCHEMA,
        "resume_for": "Continue at turn 1.",
        "episode_summary": "UP was tested once and changed the grid.",
        "episode_index": [
            {
                "topic": "UP probe",
                "gist": "The first action changed one cell.",
                "event_ids": ["transcript:000001", "trajectory:000000"],
            }
        ],
        "beliefs": [
            {
                "claim": "UP changes the object.",
                "status": "supported",
                "evidence_event_ids": ["trajectory:000000"],
            }
        ],
        "contradictions": [],
        "open_questions": [],
        "current_plan": ["Test DOWN."],
        "artifact_manifest": [
            {
                "artifact": "knowledge/hypotheses.md",
                "role": "durable beliefs",
                "status": "current",
                "event_ids": ["knowledge:hypotheses.md"],
            }
        ],
        "review": [
            {
                "observation": "Only one direction has been tested.",
                "possible_diagnosis": "The mapping is incomplete.",
                "alternative_explanation": "Other evidence may exist.",
                "recommended_check": "Test DOWN.",
                "evidence_event_ids": ["trajectory:000000"],
            }
        ],
        "next_moves": ["Test DOWN."],
        "suggested_capabilities": [
            {
                "capability": "history.turns",
                "use_when": "Comparing the probe with its result.",
            }
        ],
    }


def test_archive_supports_exact_turn_and_lexical_retrieval_with_redaction(tmp_path):
    archive = MODULE.HistoryArchive(_synthetic_run(tmp_path))

    assert archive.get(["trajectory:000000"])[0]["record"]["action"] == "UP"
    assert [event["event_id"] for event in archive.turns(0, 0)] == [
        "transcript:000000",
        "transcript:000001",
        "trajectory:000000",
        "provider:000000",
    ]
    assert archive.find("action UP", kinds={"decision"})[0]["event_id"] == (
        "transcript:000001"
    )
    provider = archive.get(["provider:000000"])[0]["record"]
    assert provider["api_key"] == "[REDACTED]"
    assert "/keys/[REDACTED]" in provider["message"]
    assert "not-for-the-model" not in provider["message"]


def test_checkpoint_validation_rejects_untraceable_evidence(tmp_path):
    archive = MODULE.HistoryArchive(_synthetic_run(tmp_path))
    candidate = _candidate()
    candidate["beliefs"][0]["evidence_event_ids"] = ["trajectory:999999"]

    with pytest.raises(ValueError, match="unknown archive events"):
        MODULE.validate_checkpoint(candidate, archive)


def test_bundle_is_an_explicit_handoff_with_recent_complete_turns(tmp_path):
    run_dir = _synthetic_run(tmp_path)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(_candidate()), encoding="utf-8")
    output_dir = tmp_path / "compactions" / "0001"

    result = MODULE.write_bundle(
        run_dir,
        candidate_path,
        output_dir,
        resume_turn=1,
        recent_turns=1,
    )

    packet = (output_dir / "recovery-packet.md").read_text(encoding="utf-8")
    assert "SAME GAME, SAME AGENT" in packet
    assert "not a game reset" in packet
    assert '"turn": 1' in packet
    assert "transcript:000001" in packet
    assert "history.find(query, filters)" in packet
    assert result["resume_turn"] == 1
    assert (output_dir / "source-manifest.json").is_file()
    assert (output_dir / "episode-index.json").is_file()
    assert (output_dir / "review.json").is_file()


def test_frozen_v14_reference_checkpoint_covers_the_real_boundary(tmp_path):
    run_dir = (
        EXPERIMENT
        / "artifacts"
        / ("20260815T-decision-transport-v14-codeact-notes-ls20-retry2")
    )
    if not run_dir.is_dir():
        pytest.skip("frozen v14 artifact is not present in this checkout")
    candidate = EXPERIMENT / "examples" / "v14-reference-checkpoint.json"
    output_dir = tmp_path / "v14-compaction"

    result = MODULE.write_bundle(
        run_dir,
        candidate,
        output_dir,
        resume_turn=16,
        recent_turns=6,
    )
    archive = MODULE.HistoryArchive(run_dir)

    assert result["event_count"] > 700
    context_error = archive.find(
        "maximum context length 131072", streams={"provider"}, start_turn=15
    )
    assert context_error[0]["event_id"] == "provider:000563"
    turn_three = archive.turns(3, 3, streams={"transcript", "trajectory"})
    assert any(event["event_id"] == "transcript:000014" for event in turn_three)
    no_op = archive.get(["trajectory:000015"])[0]["record"]
    changed = [
        (row, column)
        for row, (before_row, after_row) in enumerate(
            zip(no_op["before"], no_op["after"], strict=True)
        )
        for column, (before, after) in enumerate(
            zip(before_row, after_row, strict=True)
        )
        if before != after
    ]
    assert changed == [(61, 28), (62, 28)]
    packet = (output_dir / "recovery-packet.md").read_text(encoding="utf-8")
    assert "turn-15 DOWN attempt did not move" in packet
    assert "turns 10–15" in packet
    assert "provider:000563" in packet

    turn_view = archive.turn_view(3)
    transition = turn_view["mechanical_transitions"][0]
    assert transition["event_id"] == "trajectory:000003"
    assert transition["value_bbox_motions"]["12"]["bbox_delta_row_col"] == [5, 0]
    assert turn_view["omitted_streams"] == ["provider"]

    recovery = archive.provider_episode(15, "context_recovery")
    assert recovery["initial_request"]["event_id"] == "provider:000562"
    assert recovery["initial_request"]["message_count"] == 72
    assert recovery["recovered_request"]["event_id"] == "provider:000567"
    assert recovery["recovered_request"]["message_count"] == 27

    commitment = archive.provider_episode(15, "commitment")
    submissions = [
        event
        for event in commitment["events"]
        if event["event"] == "action_commitment_response"
    ]
    assert [event["tool_calls"][0]["name"] for event in submissions] == [
        "execute_python",
        "return_result",
    ]

    note = archive.note("hypotheses.md")
    assert "Current object position after turn 3" in note["record"]["content"]

    around = archive.around("provider:000563", before=1, after=4)
    assert [item["event_id"] for item in around["items"]] == [
        "provider:000562",
        "provider:000563",
        "provider:000564",
        "provider:000565",
        "provider:000566",
        "provider:000567",
    ]


def test_comparison_score_is_exact_and_experiment_card_is_bounded(tmp_path):
    archive = MODULE.HistoryArchive(_synthetic_run(tmp_path))
    result = {
        **COMPARE.EXPECTED,
        "next_check": "Contrast the changed cells in turns 14 and 15.",
        "evidence_event_ids": ["transcript:000001", "trajectory:000000"],
        "uncertainties": [],
    }

    score = COMPARE.score_probe(result, archive)
    assert score["score"] == score["max_score"]

    card = tmp_path / "EXPERIMENT-CARD.md"
    COMPARE.write_card(card, repeats=3)
    assert card.stat().st_size <= 1_024
