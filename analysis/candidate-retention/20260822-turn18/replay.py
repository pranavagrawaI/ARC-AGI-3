#!/usr/bin/env python3
"""Replay the preserved turn-18 rollback through the candidate-retention path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nooa.unifiedllm import FakeLLMClient
from tycho_nooa import GameTurn, TychoNooaAgent
from tycho_nooa.builder import FocusedBuilder
from tycho_nooa.trigger_builder import (
    _rollback_with_candidate_retention,
    _snapshot_workspace,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "runs/tycho-nooa-trigger-replacement-gemma4-31b-ar25-20260821T154601Z"
    / "game"
)
OUTPUT = Path(__file__).resolve().parent
WORKSPACE_ROOT = OUTPUT / "workspace"
WORKSPACE = WORKSPACE_ROOT / "ar25"
BEFORE_CALL = "000491"
CANDIDATE_CALL = "000492"
EXPECTED_BEFORE_SHA = "060648964ad3943f0083650a8b5fa382986a18a7c59e440db17ba1d959788905"
EXPECTED_CANDIDATE_SHA = (
    "4f78ef3b371d3e8f4d106710d19d03f94de6c60cfa399776818714fa1c5a86df"
)


def main() -> None:
    if WORKSPACE_ROOT.exists():
        raise FileExistsError(
            f"refusing to overwrite preserved replay output: {WORKSPACE_ROOT}"
        )
    before_manifest = _read_json(_call_path(BEFORE_CALL) / "workspace.json")
    candidate_manifest = _read_json(_call_path(CANDIDATE_CALL) / "workspace.json")
    before_sha = _workspace_sha(before_manifest, "world_model.py")
    candidate_sha = _workspace_sha(candidate_manifest, "world_model.py")
    if before_sha != EXPECTED_BEFORE_SHA or candidate_sha != EXPECTED_CANDIDATE_SHA:
        raise RuntimeError("preserved turn-18 model commitments drifted")

    _materialize(before_manifest, WORKSPACE)
    before = _snapshot_workspace(WORKSPACE)
    candidate_body = _blob(candidate_sha).read_bytes()
    (WORKSPACE / "world_model.py").write_bytes(candidate_body)
    partial_calls = _turn18_builder_calls()
    deadline = _turn18_deadline()
    if deadline.get("partial_tool_calls") != len(partial_calls):
        raise RuntimeError("deadline trace and Builder tool trace disagree")

    changed, retained, retention_error = _rollback_with_candidate_retention(
        WORKSPACE,
        before,
        partial_calls,
        turn=GameTurn(
            grid=[[0]],
            available_actions=[
                "RESET",
                "UP",
                "DOWN",
                "LEFT",
                "RIGHT",
                "SPACEBAR",
                "CLICK",
                "UNDO",
            ],
            level=0,
            turn_in_level=18,
            state="NOT_FINISHED",
        ),
        reason="preserved turn-18 dynamics discrepancy",
        focus=None,
    )
    if retained is None or retention_error is not None:
        raise RuntimeError(f"candidate retention failed: {retention_error}")

    canonical_after_sha = _sha256(WORKSPACE / "world_model.py")
    record = _read_json(WORKSPACE / retained.record_path)
    index = _read_json(WORKSPACE / "candidates/world_model/index.json")
    builder_access = _fresh_builder_access(retained.model_path, candidate_body)
    score = retained.contemporaneous_score
    acceptance = {
        "source_deadline_observed": deadline.get("event")
        == "builder_deadline_exceeded",
        "canonical_restored_byte_exact": canonical_after_sha == before_sha,
        "candidate_hash_preserved": retained.model_sha256 == candidate_sha,
        "contemporaneous_score_preserved": (
            score.get("matched_transitions") == 8
            and score.get("evaluated_transitions") == 16
            and score.get("observed_transitions") == 18
        ),
        "fresh_builder_can_list_and_read": all(builder_access.values()),
        "publication_denied": record.get("publication_allowed") is False,
        "planner_authority_denied": record.get("planner_eligible") is False,
    }
    result = {
        "schema": "tycho-nooa-candidate-retention-replay-v1",
        "source_run": str(SOURCE.parent.relative_to(ROOT)),
        "source_boundary": {
            "level": 0,
            "turn_in_level": 18,
            "before_call": BEFORE_CALL,
            "candidate_call": CANDIDATE_CALL,
            "deadline_trace": deadline,
            "completed_builder_tool_calls": len(partial_calls),
        },
        "canonical_before_sha256": before_sha,
        "canonical_after_sha256": canonical_after_sha,
        "rolled_back_paths": changed,
        "retained_candidate": {
            "record_id": retained.record_id,
            "model_sha256": retained.model_sha256,
            "model_path": retained.model_path,
            "record_path": retained.record_path,
            "contemporaneous_score": score,
            "evidence_frontier_entries": len(record["evidence_frontier"]),
        },
        "fresh_builder_access": builder_access,
        "index_records": len(index["records"]),
        "acceptance": acceptance,
        "status": "pass" if all(acceptance.values()) else "fail",
        "execution": {"model_calls": 0, "arc_environment_calls": 0},
    }
    _write_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


def _fresh_builder_access(model_path: str, expected: bytes) -> dict[str, bool]:
    agent = TychoNooaAgent(
        llm=FakeLLMClient(),
        game_id="ar25",
        workspace_root=WORKSPACE_ROOT,
        available_actions=["UP", "DOWN"],
        resume_workspace=True,
    )
    builder = FocusedBuilder(
        llm=FakeLLMClient(),
        executor=agent._tool_executor,
        trace_path=None,
    )
    listing = builder._tool_surface.ls("candidates/world_model")
    body = builder._tool_surface.read_file(model_path)
    return {
        "index_listed": "index.json" in listing,
        "candidate_read_exact": body.encode() == expected,
    }


def _materialize(manifest: dict, destination: Path) -> None:
    destination.mkdir(parents=True)
    for directory in manifest["directories"]:
        path = destination / directory["path"]
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(directory["mode"])
    for entry in manifest["files"]:
        path = destination / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_blob(entry["sha256"]).read_bytes())
        path.chmod(entry["mode"])
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"blob commitment mismatch: {entry['path']}")


def _turn18_builder_calls() -> list[dict]:
    return [
        row
        for row in _read_jsonl(SOURCE / "builder-tool-trace.jsonl")
        if row.get("level") == 0 and row.get("turn_in_level") == 18
    ]


def _turn18_deadline() -> dict:
    rows = [
        row
        for row in _read_jsonl(SOURCE / "trigger-builder-trace.jsonl")
        if row.get("event") == "builder_deadline_exceeded"
        and row.get("level") == 0
        and row.get("turn_in_level") == 18
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one preserved turn-18 deadline, found {len(rows)}")
    return rows[0]


def _call_path(call: str) -> Path:
    return SOURCE / "decision-checkpoints/calls" / call


def _blob(sha256: str) -> Path:
    path = SOURCE / "decision-checkpoints/blobs/sha256" / sha256
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _workspace_sha(manifest: dict, relative: str) -> str:
    matches = [row["sha256"] for row in manifest["files"] if row["path"] == relative]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {relative} commitment")
    return matches[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
