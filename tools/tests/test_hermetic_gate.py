from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.hermetic_gate_core import (
    audit_bundle,
    audit_checkpoints,
    audit_run,
    rename_tool_call_ids,
    run_fault_campaign,
    synthetic_bundle,
    violation_codes,
)


def commitment(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def write_valid_checkpoint(root: Path) -> tuple[Path, str]:
    content = b"MODEL = 'candidate'\n"
    digest = hashlib.sha256(content).hexdigest()
    blob = root / "blobs" / "sha256" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content)

    workspace = {
        "schema_version": 1,
        "directories": [],
        "files": [
            {
                "path": "world_model.py",
                "sha256": digest,
                "size": len(content),
                "mode": 0o644,
            }
        ],
    }
    body = {"model": "fake", "messages": [], "tools": [], "parameters": {}}
    request = {
        "call_index": 1,
        "source": "actor",
        "request_payload_commitment": commitment(body),
        "workspace_commitment": commitment(workspace),
        "request_body": body,
    }
    call = root / "calls" / "000001"
    call.mkdir(parents=True)
    (call / "workspace.json").write_text(json.dumps(workspace))
    (call / "request.json").write_text(json.dumps(request))
    (call / "response.json").write_text(
        json.dumps({"schema_version": 1, "status": "complete"})
    )
    return call, digest


def test_synthetic_baseline_satisfies_every_gate() -> None:
    assert audit_bundle(synthetic_bundle()) == []


def test_generated_campaign_detects_and_reduces_every_declared_fault() -> None:
    result = run_fault_campaign()

    assert result["status"] == "pass"
    assert result["generated_cases"] == 12
    assert all(case["detected"] for case in result["cases"])
    assert all(case["reduced_rows"] > 0 for case in result["cases"])
    assert all(result["metamorphic_relations"].values())


def test_bijective_tool_id_rename_is_invariant_but_one_sided_rename_is_not() -> None:
    baseline = synthetic_bundle()

    assert audit_bundle(rename_tool_call_ids(baseline, prefix="renamed")) == []

    baseline.llm[0]["tool_call_ids"] = ["unjoined-id"]
    assert {
        "missing_tool_result",
        "orphan_tool_result",
    } <= violation_codes(audit_bundle(baseline))


def test_failed_take_action_cannot_masquerade_as_host_execution() -> None:
    bundle = synthetic_bundle()
    bundle.actor_tools[0]["status"] = "error"

    assert "action_not_traceable" in violation_codes(audit_bundle(bundle))


def test_actor_requests_and_builder_deadlines_are_complete_lifecycles() -> None:
    bundle = synthetic_bundle()
    bundle.triggers[:0] = [
        {"event": "actor_request", "level": 0, "turn_in_level": 0},
        {"event": "builder_result", "level": 0, "turn_in_level": 0},
        {
            "event": "mechanical_trigger",
            "level": 0,
            "turn_in_level": 1,
            "fired": True,
        },
        {"event": "builder_deadline_exceeded", "level": 0, "turn_in_level": 1},
    ]

    assert not (
        {
            "builder_without_trigger",
            "trigger_without_builder_terminal",
        }
        & violation_codes(audit_bundle(bundle))
    )


def test_checkpoint_audit_verifies_commitments_and_content_addressed_blobs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "decision-checkpoints"
    call, digest = write_valid_checkpoint(root)
    assert audit_checkpoints(root) == []

    request = json.loads((call / "request.json").read_text())
    request["request_body"]["model"] = "silently-mutated"
    (call / "request.json").write_text(json.dumps(request))
    (root / "blobs" / "sha256" / digest).write_bytes(b"corrupt")

    assert {
        "request_commitment_mismatch",
        "checkpoint_blob_corrupt",
    } <= violation_codes(audit_checkpoints(root))


def test_checkpoint_audit_rejects_a_missing_required_file(tmp_path: Path) -> None:
    root = tmp_path / "decision-checkpoints"
    call, _ = write_valid_checkpoint(root)
    (call / "response.json").unlink()

    assert "checkpoint_file_missing" in violation_codes(audit_checkpoints(root))


def test_checkpoint_audit_rejects_unsafe_workspace_paths(tmp_path: Path) -> None:
    root = tmp_path / "decision-checkpoints"
    call, _ = write_valid_checkpoint(root)
    workspace = json.loads((call / "workspace.json").read_text())
    workspace["files"][0]["path"] = "../escaped.py"
    (call / "workspace.json").write_text(json.dumps(workspace))
    request = json.loads((call / "request.json").read_text())
    request["workspace_commitment"] = commitment(workspace)
    (call / "request.json").write_text(json.dumps(request))

    assert "unsafe_checkpoint_path" in violation_codes(audit_checkpoints(root))


def test_malformed_checkpoint_shapes_become_violations_not_crashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "decision-checkpoints"
    call, _ = write_valid_checkpoint(root)
    (call / "response.json").write_text("[]")
    (call / "workspace.json").write_text(json.dumps({"files": "not-a-list"}))

    codes = violation_codes(audit_checkpoints(root))
    assert "malformed_checkpoint" in codes
    assert "workspace_commitment_mismatch" in codes


def test_nonexistent_run_cannot_pass_the_gate(tmp_path: Path) -> None:
    result = audit_run(tmp_path / "absent")

    assert result["status"] == "fail"
    assert result["promotion_allowed"] is False
    assert result["violations"][0]["code"] == "missing_transcript"
