from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from tools.hermetic_gate_core import (
    _audit_checkpoint_trace_joins,
    _captured_schema_accepts,
    _validated_arguments_derive_from_parsed,
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
        "schema_version": 2,
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
        "schema_version": 2,
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
    response = {"schema_version": 2, "status": "complete"}
    response["response_payload_commitment"] = commitment(response)
    (call / "response.json").write_text(json.dumps(response))
    return call, digest


def test_synthetic_baseline_satisfies_every_gate() -> None:
    assert audit_bundle(synthetic_bundle()) == []


def test_audit_run_rejects_legacy_typed_goal_evidence_authority(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace" / "ar25"
    evidence = workspace / "level_0" / "turn_000.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("host frame")
    index = workspace / "epistemic" / "authoritative-evidence.json"
    index.parent.mkdir()
    index.write_text(json.dumps(["level_0/turn_000.txt"]))

    result = audit_run(tmp_path)

    assert "legacy_authoritative_evidence_index" in {
        violation["code"] for violation in result["violations"]
    }
    assert result["promotion_allowed"] is False


def test_audit_run_recomputes_typed_goal_evidence_commitments(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace" / "ar25"
    evidence = workspace / "level_0" / "turn_000.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("host frame")
    payload = evidence.read_bytes()
    index = workspace / "epistemic" / "authoritative-evidence.json"
    index.parent.mkdir()
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    "level_0/turn_000.txt": {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                },
            }
        )
    )

    result = audit_run(tmp_path)
    assert not {
        violation["code"]
        for violation in result["violations"]
        if "authoritative_evidence" in violation["code"]
    }

    evidence.write_text("changed frame")
    result = audit_run(tmp_path)
    assert "authoritative_evidence_commitment_mismatch" in {
        violation["code"] for violation in result["violations"]
    }


def test_audit_run_rejects_invalid_typed_goal_authority_shapes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"

    malformed = workspace / "malformed" / "epistemic"
    malformed.mkdir(parents=True)
    (malformed / "authoritative-evidence.json").write_text("{")

    unsafe = workspace / "unsafe" / "epistemic"
    unsafe.mkdir(parents=True)
    (unsafe / "authoritative-evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    "../../outside.txt": {"sha256": "0" * 64, "size_bytes": 0}
                },
            }
        )
    )

    missing = workspace / "missing" / "epistemic"
    missing.mkdir(parents=True)
    (missing / "authoritative-evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    "absent.txt": {"sha256": "0" * 64, "size_bytes": 0}
                },
            }
        )
    )

    invalid = workspace / "invalid" / "epistemic"
    invalid.mkdir(parents=True)
    evidence = workspace / "invalid" / "evidence.txt"
    evidence.write_text("host frame")
    (invalid / "authoritative-evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    "evidence.txt": {"sha256": "not-a-digest", "size_bytes": True}
                },
            }
        )
    )

    result = audit_run(tmp_path)
    codes = {violation["code"] for violation in result["violations"]}

    assert "invalid_authoritative_evidence_index" in codes
    assert "authoritative_evidence_reference_invalid" in codes
    assert "invalid_authoritative_evidence_commitment" in codes
    assert result["promotion_allowed"] is False


def test_actor_only_mode_does_not_require_a_builder_outcome_checkpoint() -> None:
    bundle = synthetic_bundle()
    bundle.summary["mode"] = "no_world_model"
    bundle.triggers.clear()

    assert audit_bundle(bundle) == []


def test_schema_two_checkpoint_rejects_a_tampered_response(tmp_path: Path) -> None:
    root = tmp_path / "decision-checkpoints"
    call, _ = write_valid_checkpoint(root)
    request_path = call / "request.json"
    request = json.loads(request_path.read_text())
    request["schema_version"] = 2
    request_path.write_text(json.dumps(request))
    response = {"schema_version": 2, "status": "complete", "content": "before"}
    response["response_payload_commitment"] = commitment(response)
    response_path = call / "response.json"
    response_path.write_text(json.dumps(response))
    assert audit_checkpoints(root) == []

    request.pop("schema_version")
    request_path.write_text(json.dumps(request))
    assert "unsupported_checkpoint_schema" in violation_codes(
        audit_checkpoints(root)
    )
    request["schema_version"] = 2
    request_path.write_text(json.dumps(request))

    response["content"] = "after"
    response_path.write_text(json.dumps(response))
    assert "response_commitment_mismatch" in violation_codes(
        audit_checkpoints(root)
    )

    response.pop("schema_version")
    response.pop("response_payload_commitment")
    response_path.write_text(json.dumps(response))
    assert "response_commitment_mismatch" in violation_codes(
        audit_checkpoints(root)
    )


def test_schema_two_tool_contract_rejects_semantic_mutations() -> None:
    bundle = synthetic_bundle()
    for llm_row, tool_row in zip(bundle.llm, bundle.actor_tools, strict=True):
        llm_row.update({"schema_version": 2, "phase": "decision"})
        tool_row.update(
            {
                "schema_version": 2,
                "source": "actor",
                "phase": "decision",
                "raw_arguments": json.dumps(tool_row["arguments"]),
                "parse_status": "complete",
                "validation_status": "complete",
                "validated_arguments": tool_row["arguments"],
                "execution_status": "complete",
                "error_stage": None,
            }
        )
    assert audit_bundle(bundle) == []

    mutations = {
        "schema_version": None,
        "phase": "post_action",
        "raw_arguments": json.dumps({"action": "LEFT"}),
        "execution_status": "not_attempted",
    }
    for field, value in mutations.items():
        candidate = copy.deepcopy(bundle)
        candidate.actor_tools[0][field] = value
        assert audit_bundle(candidate), field


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


def test_new_tool_sequences_disambiguate_reused_level_turn_coordinates() -> None:
    bundle = synthetic_bundle()
    for row in bundle.transcript:
        if row.get("sequence") == 1 and row.get("type") in {
            "turn",
            "action_result",
        }:
            row["turn_in_level"] = 0
    for sequence, row in enumerate(bundle.actor_tools):
        row["sequence"] = sequence
        row["turn_in_level"] = 0
    for row in bundle.triggers:
        row["turn_in_level"] = 0

    assert audit_bundle(bundle) == []


def test_schema_two_actor_evidence_requires_direct_sequences() -> None:
    bundle = synthetic_bundle()
    bundle.llm[0]["sequence"] = True
    bundle.actor_tools[0]["sequence"] = True

    assert {
        "missing_llm_sequence",
        "missing_tool_sequence",
    } <= violation_codes(audit_bundle(bundle))

    builder_bundle = synthetic_bundle()
    builder_bundle.llm[0]["source"] = "builder"
    builder_bundle.llm[0].pop("sequence")
    builder_bundle.actor_tools[0]["source"] = "builder"
    builder_bundle.actor_tools[0]["sequence"] = "bogus"
    assert {
        "invalid_builder_llm_sequence",
        "invalid_builder_tool_sequence",
    } <= violation_codes(audit_bundle(builder_bundle))


def test_successful_generic_tool_requires_a_result_field() -> None:
    bundle = synthetic_bundle()
    bundle.llm.append(
        {
            "schema_version": 2,
            "call_index": 3,
            "source": "builder",
            "phase": "decision",
            "sequence": None,
            "tool_calls": ["read_file"],
            "tool_call_ids": ["builder-read"],
        }
    )
    row = dict(bundle.actor_tools[0])
    row.update(
        {
            "call_index": 3,
            "tool_call_id": "builder-read",
            "source": "builder",
            "sequence": None,
            "name": "read_file",
            "raw_arguments": "{}",
            "arguments": {},
            "validated_arguments": {},
        }
    )
    row.pop("result")
    bundle.builder_tools.append(row)

    assert "missing_tool_result_value" in violation_codes(audit_bundle(bundle))


def test_captured_schema_enforces_constraints_beyond_basic_types() -> None:
    schema = {"type": "integer", "minimum": 1, "maximum": 300}

    assert _captured_schema_accepts(300, schema)
    assert not _captured_schema_accepts(301, schema)
    assert not _captured_schema_accepts(1, {"type": "not-a-json-schema-type"})
    assert not _captured_schema_accepts(
        1,
        {"$schema": "https://example.invalid/unknown-dialect", "type": "integer"},
    )
    assert not _captured_schema_accepts(1, {"$ref": "#/$defs/missing"})
    assert not _captured_schema_accepts(1, {"$ref": "#"})


def test_validated_arguments_cannot_rewrite_parsed_values() -> None:
    schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "mode": {"type": "string", "default": "safe"},
        },
        "required": ["content"],
        "additionalProperties": False,
    }

    assert _validated_arguments_derive_from_parsed(
        {"content": "a"},
        {"content": "a", "mode": "safe"},
        schema,
    )
    assert not _validated_arguments_derive_from_parsed(
        {"content": "a"},
        {"content": "DIFFERENT", "mode": "safe"},
        schema,
    )
    assert not _validated_arguments_derive_from_parsed(
        {"content": None},
        {},
        schema,
    )

    referenced_schema = {
        "$defs": {
            "arguments": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "mode": {"type": "string"},
                },
                "required": ["content"],
                "additionalProperties": False,
            }
        },
        "allOf": [{"$ref": "#/$defs/arguments"}],
    }
    assert _captured_schema_accepts(
        {"content": "a", "mode": "safe"},
        referenced_schema,
    )
    assert _validated_arguments_derive_from_parsed(
        {"content": "a"},
        {"content": "a", "mode": "safe"},
        referenced_schema,
    )

    dependent_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "dependentSchemas": {
            "content": {
                "properties": {"mode": {"type": "string"}},
            }
        },
        "unevaluatedProperties": False,
    }
    assert _captured_schema_accepts(
        {"content": "a", "mode": "safe"},
        dependent_schema,
    )
    assert _validated_arguments_derive_from_parsed(
        {"content": "a"},
        {"content": "a", "mode": "safe"},
        dependent_schema,
    )

    draft_seven_ignored_keyword = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "dependentSchemas": {
            "content": {"properties": {"mode": {"type": "string"}}},
        },
    }
    assert not _validated_arguments_derive_from_parsed(
        {"content": "a"},
        {"content": "a", "mode": "safe"},
        draft_seven_ignored_keyword,
    )

    draft_seven_dependency = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "dependencies": {
            "content": {"properties": {"mode": {"type": "string"}}},
        },
    }
    assert _validated_arguments_derive_from_parsed(
        {"content": "a"},
        {"content": "a", "mode": "safe"},
        draft_seven_dependency,
    )

    exclusive_schema = {
        "type": "object",
        "oneOf": [
            {
                "required": ["a"],
                "properties": {"a": {}, "x": {}},
            },
            {"required": ["b"], "properties": {"b": {}}},
        ],
    }
    assert not _validated_arguments_derive_from_parsed(
        {"b": 1},
        {"b": 1, "x": 2},
        exclusive_schema,
    )
    assert _validated_arguments_derive_from_parsed(
        {"a": 1},
        {"a": 1, "x": 2},
        exclusive_schema,
    )

    conditional_schema = {
        "type": "object",
        "if": {"required": ["a"]},
        "then": {"properties": {"x": {}}},
        "else": {"properties": {"b": {}}},
    }
    assert not _validated_arguments_derive_from_parsed(
        {"b": 1},
        {"b": 1, "x": 2},
        conditional_schema,
    )
    assert _validated_arguments_derive_from_parsed(
        {"a": 1},
        {"a": 1, "x": 2},
        conditional_schema,
    )


def test_scribe_requests_join_the_shared_actor_tool_stream() -> None:
    bundle = synthetic_bundle()
    bundle.llm.append(
        {
            "schema_version": 2,
            "call_index": 3,
            "source": "scribe",
            "phase": "post_action",
            "sequence": 1,
            "tool_calls": ["write_file"],
            "tool_call_ids": ["scribe-note"],
        }
    )
    bundle.actor_tools.append(
        {
            "schema_version": 2,
            "call_index": 3,
            "tool_call_id": "scribe-note",
            "source": "scribe",
            "phase": "post_action",
            "sequence": 1,
            "level": 0,
            "turn_in_level": 1,
            "name": "write_file",
            "raw_arguments": json.dumps(
                {"path": "notes/level_0_insights.md", "content": "done"}
            ),
            "arguments": {"path": "notes/level_0_insights.md", "content": "done"},
            "parse_status": "complete",
            "validation_status": "complete",
            "validated_arguments": {
                "path": "notes/level_0_insights.md",
                "content": "done",
            },
            "execution_status": "complete",
            "error_stage": None,
            "status": "complete",
            "result": "wrote note",
        }
    )

    assert audit_bundle(bundle) == []


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


def test_checkpoint_llm_completion_ordinal_skips_failed_calls(tmp_path: Path) -> None:
    bundle = synthetic_bundle()
    bundle.llm = [
        {
            **bundle.llm[1],
            "call_index": 1,
            "content": "",
            "reasoning": None,
        }
    ]
    root = tmp_path / "decision-checkpoints"
    failed = root / "calls/000001"
    complete = root / "calls/000002"
    failed.mkdir(parents=True)
    complete.mkdir(parents=True)
    (failed / "request.json").write_text(
        json.dumps(
            {
                "call_index": 1,
                "source": "actor",
                "decision_point": {"phase": "decision", "sequence": 0},
            }
        )
    )
    (failed / "response.json").write_text(json.dumps({"status": "error"}))
    (complete / "request.json").write_text(
        json.dumps(
            {
                "call_index": 2,
                "source": "actor",
                "decision_point": {"phase": "decision", "sequence": 1},
                "request_body": {
                    "tools": [
                        {
                            "name": "take_action",
                            "parameters": {
                                "type": "object",
                                "properties": {"action": {"type": "string"}},
                                "required": ["action"],
                            },
                        }
                    ]
                },
            }
        )
    )
    (complete / "response.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "content": "",
                "reasoning": None,
                "tool_calls": [
                    {
                        "id": "act-1",
                        "name": "take_action",
                        "arguments": "{\"action\":\"DOWN\"}",
                    }
                ],
            }
        )
    )

    assert _audit_checkpoint_trace_joins(root, bundle) == []


def test_audit_run_accepts_a_faithful_validation_rejection(
    tmp_path: Path,
) -> None:
    bundle = synthetic_bundle()
    initial_observation = bundle.transcript[0]["initial_observation"]
    bundle.transcript = [
        bundle.transcript[0],
        bundle.transcript[1],
        {
            "type": "finish",
            "actions": 0,
            "levels_completed": 0,
            "final_observation": initial_observation,
        },
    ]
    bundle.llm = [
        {
            "schema_version": 2,
            "call_index": 1,
            "source": "builder",
            "phase": "decision",
            "sequence": None,
            "tool_calls": ["read_file"],
            "tool_call_ids": ["builder-read"],
        }
    ]
    bundle.actor_tools = []
    bundle.builder_tools = [
        {
            "schema_version": 2,
            "call_index": 1,
            "tool_call_id": "builder-read",
            "source": "builder",
            "phase": "decision",
            "sequence": None,
            "level": 0,
            "turn_in_level": 0,
            "name": "read_file",
            "raw_arguments": "{}",
            "arguments": {},
            "parse_status": "complete",
            "validation_status": "error",
            "validated_arguments": None,
            "execution_status": "not_attempted",
            "error_stage": "validation",
            "status": "error",
            "result": "ValidationError: path is required",
        }
    ]
    bundle.triggers = []
    bundle.summary = {
        "actions": 0,
        "levels_completed": 0,
        "mode": "no_world_model",
    }

    run_dir = tmp_path / "run"
    checkpoint_root = run_dir / "decision-checkpoints"
    call, _ = write_valid_checkpoint(checkpoint_root)
    request_path = call / "request.json"
    request = json.loads(request_path.read_text())
    request_body = {
        "model": "fake",
        "messages": [],
        "tools": [
            {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        "parameters": {},
    }
    request.update(
        {
            "source": "builder",
            "decision_point": {"phase": "decision", "sequence": None},
            "request_body": request_body,
            "request_payload_commitment": commitment(request_body),
        }
    )
    request_path.write_text(json.dumps(request))
    response = {
        "schema_version": 2,
        "status": "complete",
        "content": "",
        "reasoning": None,
        "tool_calls": [
            {
                "id": "builder-read",
                "name": "read_file",
                "arguments": "{}",
            }
        ],
    }
    response["response_payload_commitment"] = commitment(response)
    (call / "response.json").write_text(json.dumps(response))

    stream_files = {
        "transcript": "transcript.jsonl",
        "llm": "llm-trace.jsonl",
        "actor_tools": "tool-trace.jsonl",
        "builder_tools": "builder-tool-trace.jsonl",
        "triggers": "trigger-builder-trace.jsonl",
        "planner": "planner-trace.jsonl",
        "builder_planner": "builder-planner-trace.jsonl",
    }
    for stream, filename in stream_files.items():
        rows = getattr(bundle, stream)
        (run_dir / filename).write_text(
            "".join(f"{json.dumps(row)}\n" for row in rows)
        )
    (run_dir / "summary.json").write_text(json.dumps(bundle.summary))

    result = audit_run(run_dir)
    assert result["status"] == "pass"
    assert result["promotion_allowed"] is True

    bundle.builder_tools[0].update(
        {
            "validation_status": "complete",
            "validated_arguments": {},
            "execution_status": "complete",
            "error_stage": None,
            "status": "complete",
            "result": "unexpected success",
        }
    )
    (run_dir / "builder-tool-trace.jsonl").write_text(
        f"{json.dumps(bundle.builder_tools[0])}\n"
    )

    assert "validated_tool_arguments_mismatch" in {
        violation["code"] for violation in audit_run(run_dir)["violations"]
    }


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
