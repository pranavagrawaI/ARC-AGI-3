from __future__ import annotations

import json
from pathlib import Path

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.compatibility_experiment import execute_compatibility_gate
from tycho_nooa.parity_compatibility import (
    CompatibilityEvidencePolicy,
    CompatibilityExpected,
    CompatibilityGate,
    CompatibilityTask,
)


def response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=name, name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 2, "completion_tokens": 1},
    )


def answer(evidence_ref: str = "L0:T1") -> dict:
    return {"evidence_ref": evidence_ref}


def termination_gate() -> CompatibilityGate:
    return CompatibilityGate(
        task=CompatibilityTask(
            gate_id="g0-termination",
            gate_kind="termination",
            instruction="Return READY.",
            inputs={"supplied_answer": True},
        ),
        evidence=CompatibilityEvidencePolicy(enabled=False, max_queries=0),
        expected=CompatibilityExpected(ok=True),
    )


def immutable_identifier_gate() -> CompatibilityGate:
    return CompatibilityGate(
        task=CompatibilityTask(
            gate_id="g2a-identifier-binding",
            gate_kind="identifier_binding",
            instruction="Return the record ref.",
            inputs={
                "record": {"ref": "L0:T1"},
                "frontier": "L1:T2",
                "next_after": "L0:T1",
            },
        ),
        evidence=CompatibilityEvidencePolicy(enabled=False, max_queries=0),
        expected=CompatibilityExpected(
            evidence_ref="L0:T1",
            frontier_substitution="L1:T2",
            turn_number_leakage="L0:T2",
        ),
    )


async def test_return_only_gate_passes_both_arms_without_evidence(
    tmp_path: Path,
) -> None:
    native_llm = FakeLLMClient(
        scripted_responses=[response("return_result", {"ok": True})]
    )
    codeact_llm = FakeLLMClient(
        scripted_responses=[response("return_result", {"ok": True})]
    )
    report = await execute_compatibility_gate(
        termination_gate(),
        repository_root=tmp_path,
        destination=tmp_path / "result",
        native_llm=native_llm,
        codeact_llm=codeact_llm,
    )

    assert report.passed is True
    assert report.native.evaluation is not None
    assert report.native.evaluation.capability_use_correct is True
    assert report.codeact.evaluation is not None
    assert report.codeact.evaluation.capability_use_correct is True
    assert report.native.metrics.evidence_queries == 0
    assert report.codeact.metrics.evidence_queries == 0
    assert report.native.metrics.provider_calls == 1
    assert report.codeact.metrics.provider_calls == 1
    assert report.native.metrics.return_call_index == 1
    assert report.codeact.metrics.return_call_index == 1
    assert (tmp_path / "result/report.json").is_file()
    assert (tmp_path / "result/codeact/provenance.jsonl").is_file()
    for llm in (native_llm, codeact_llm):
        tool = next(
            tool for tool in llm.last_tools or [] if tool.name == "return_result"
        )
        schema = tool.parameters_model.model_json_schema()
        assert set(schema["properties"]) == {"ok"}
        assert set(schema["required"]) == {"ok"}
        assert "result" not in schema["properties"]


async def test_flat_return_rejects_union_wrapper_then_accepts_exact_shape(
    tmp_path: Path,
) -> None:
    nested = {"result": {"ok": True}}
    native_llm = FakeLLMClient(
        scripted_responses=[
            response("return_result", nested),
            response("return_result", {"ok": True}),
        ]
    )
    codeact_llm = FakeLLMClient(
        scripted_responses=[
            response("return_result", nested),
            response("return_result", {"ok": True}),
        ]
    )

    report = await execute_compatibility_gate(
        termination_gate(),
        repository_root=tmp_path,
        destination=tmp_path / "result",
        native_llm=native_llm,
        codeact_llm=codeact_llm,
    )

    assert report.passed is True
    assert report.native.metrics.invalid_interactions == 1
    assert report.codeact.metrics.invalid_interactions == 1


async def test_immutable_binding_can_return_without_query_or_python(
    tmp_path: Path,
) -> None:
    direct = response("return_result", {"evidence_ref": "L0:T1"})

    report = await execute_compatibility_gate(
        immutable_identifier_gate(),
        repository_root=tmp_path,
        destination=tmp_path / "result",
        native_llm=FakeLLMClient(scripted_responses=[direct]),
        codeact_llm=FakeLLMClient(scripted_responses=[direct]),
    )

    assert report.passed is True
    assert report.native.evaluation is not None
    assert report.native.evaluation.identifier_classification == (
        "correct_record_binding"
    )
    for arm in (report.native, report.codeact):
        assert arm.metrics.evidence_queries == 0
        assert arm.metrics.query_attempts == 0
        assert arm.metrics.python_cells == 0
        assert arm.metrics.provider_calls == 1
        assert arm.metrics.return_call_index == 1


async def test_valid_answer_without_required_query_fails_eligibility(
    tmp_path: Path,
) -> None:
    gate = CompatibilityGate(
        checkpoint_root="checkpoint",
        checkpoint_sha256="sha256:" + "0" * 64,
        task=CompatibilityTask(
            gate_id="g1-metadata",
            gate_kind="capability_retrieval",
            instruction="Query once, then return LEFT.",
        ),
        evidence=CompatibilityEvidencePolicy(
            enabled=True,
            kind="transitions",
            level=0,
            limit=4,
            projection="metadata",
            grid_encoding="raw",
            max_queries=1,
        ),
        expected=CompatibilityExpected(evidence_ref="L0:T1"),
    )
    (tmp_path / "checkpoint").mkdir()
    direct = response("return_result", answer())

    report = await execute_compatibility_gate(
        gate,
        repository_root=tmp_path,
        destination=tmp_path / "result",
        native_llm=FakeLLMClient(scripted_responses=[direct]),
        codeact_llm=FakeLLMClient(scripted_responses=[direct]),
    )

    assert report.passed is False
    assert report.native.evaluation is not None
    assert report.native.evaluation.capability_use_correct is False
    assert report.codeact.evaluation is not None
    assert report.codeact.evaluation.capability_use_correct is False


async def test_call_budget_exhaustion_is_a_scored_gate_failure(
    tmp_path: Path,
) -> None:
    empty = LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 2, "completion_tokens": 1},
    )

    report = await execute_compatibility_gate(
        termination_gate(),
        repository_root=tmp_path,
        destination=tmp_path / "result",
        native_llm=FakeLLMClient(scripted_responses=[empty] * 3),
        codeact_llm=FakeLLMClient(scripted_responses=[empty] * 3),
        max_llm_calls=3,
    )

    assert report.passed is False
    assert report.native.failure == "native parity exhausted its LLM-call budget"
    assert report.codeact.failure == "sterile CodeAct exhausted its LLM-call budget"
    assert report.native.metrics.llm_calls == 3
    assert report.codeact.metrics.llm_calls == 3


async def test_both_arms_share_normalization_and_codeact_can_repair(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    workspace = GameWorkspace(
        "compatibility",
        root=str(checkpoint),
        render=False,
        available_actions=["LEFT"],
    )
    workspace.record([[1]], level=0, turn_in_level=0)
    workspace.record(
        [[2]],
        level=0,
        turn_in_level=1,
        action="LEFT",
        state="NOT_FINISHED",
        available=["LEFT"],
    )
    gate = CompatibilityGate(
        checkpoint_root="checkpoint",
        checkpoint_sha256="sha256:" + "0" * 64,
        task=CompatibilityTask(
            gate_id="g1-metadata",
            gate_kind="capability_retrieval",
            instruction="Query metadata and return LEFT.",
        ),
        evidence=CompatibilityEvidencePolicy(
            enabled=True,
            kind="transitions",
            level=0,
            limit=4,
            projection="metadata",
            grid_encoding="raw",
            max_queries=1,
            max_query_attempts=2,
        ),
        expected=CompatibilityExpected(evidence_ref="L0:T1"),
    )
    exact = {
        "kind": "transitions",
        "level": 0,
        "limit": 4,
        "projection": "metadata",
        "grid_encoding": "raw",
    }

    report = await execute_compatibility_gate(
        gate,
        repository_root=tmp_path,
        destination=tmp_path / "result",
        native_llm=FakeLLMClient(
            scripted_responses=[
                response("evidence_query", exact),
                response("return_result", answer()),
            ]
        ),
        codeact_llm=FakeLLMClient(
            scripted_responses=[
                response(
                    "execute_python",
                    {
                        "code": (
                            "evidence.query(level=0, limit=4, "
                            "projection='metadata')"
                        )
                    },
                ),
                response(
                    "execute_python",
                    {"code": f"evidence.query(**{exact!r})"},
                ),
                response("return_result", answer()),
            ]
        ),
        max_llm_calls=3,
    )

    assert report.passed is True
    assert report.native.metrics.query_attempts == 1
    assert report.native.metrics.evidence_queries == 1
    assert report.codeact.metrics.query_attempts == 2
    assert report.codeact.metrics.evidence_queries == 1
    provenance = (
        tmp_path / "result/codeact/provenance.jsonl"
    ).read_text()
    assert provenance.count('"event_type":"CapabilityNormalized"') == 1
    assert provenance.count('"event_type":"CapabilityDisclosed"') == 1
    assert '"grid_encoding":"raw"' in provenance
    assert '"grid_encoding":null' in provenance


async def test_codeact_contract_help_leads_to_the_admitted_metadata_query(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    workspace = GameWorkspace(
        "compatibility-help",
        root=str(checkpoint),
        render=False,
        available_actions=["LEFT"],
    )
    workspace.record([[1]], level=0, turn_in_level=0)
    workspace.record(
        [[2]],
        level=0,
        turn_in_level=1,
        action="LEFT",
        state="NOT_FINISHED",
        available=["LEFT"],
    )
    gate = CompatibilityGate(
        checkpoint_root="checkpoint",
        checkpoint_sha256="sha256:" + "0" * 64,
        task=CompatibilityTask(
            gate_id="g1-metadata-help",
            gate_kind="capability_retrieval",
            instruction="Inspect the contract, query metadata, and return LEFT.",
        ),
        evidence=CompatibilityEvidencePolicy(
            enabled=True,
            kind="transitions",
            level=0,
            limit=4,
            projection="metadata",
            grid_encoding="raw",
            max_queries=1,
            max_query_attempts=2,
        ),
        expected=CompatibilityExpected(evidence_ref="L0:T1"),
    )
    exact = {
        "kind": "transitions",
        "level": 0,
        "limit": 4,
        "projection": "metadata",
    }

    report = await execute_compatibility_gate(
        gate,
        repository_root=tmp_path,
        destination=tmp_path / "result",
        native_llm=FakeLLMClient(
            scripted_responses=[
                response("evidence_query", exact),
                response("return_result", answer()),
            ]
        ),
        codeact_llm=FakeLLMClient(
            scripted_responses=[
                response(
                    "execute_python",
                    {"code": "print(evidence)"},
                ),
                response(
                    "execute_python",
                    {"code": f"page = evidence.query(**{exact!r})"},
                ),
                response("return_result", answer()),
            ]
        ),
        max_llm_calls=3,
    )

    assert report.passed is True
    assert report.codeact.metrics.python_cells == 2
    assert report.codeact.metrics.query_attempts == 1
    assert report.codeact.metrics.evidence_queries == 1
    provenance_path = tmp_path / "result/codeact/provenance.jsonl"
    provenance = provenance_path.read_text()
    assert "kind='transitions'" in provenance
    events = [json.loads(line) for line in provenance.splitlines()]
    execution_manifest = events[0]["payload"]["manifest"]
    assert execution_manifest["capability_contracts"]["evidence.query"] == (
        execution_manifest["input_values"]["task"]["evidence_contract"]
    )


async def test_codeact_can_query_directly_without_discovery_step(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    workspace = GameWorkspace(
        "compatibility-direct",
        root=str(checkpoint),
        render=False,
        available_actions=["LEFT"],
    )
    workspace.record([[1]], level=0, turn_in_level=0)
    workspace.record(
        [[2]],
        level=0,
        turn_in_level=1,
        action="LEFT",
        state="NOT_FINISHED",
        available=["LEFT"],
    )
    gate = CompatibilityGate(
        checkpoint_root="checkpoint",
        checkpoint_sha256="sha256:" + "0" * 64,
        task=CompatibilityTask(
            gate_id="g1-metadata-direct",
            gate_kind="capability_retrieval",
            instruction="Query metadata and return LEFT.",
        ),
        evidence=CompatibilityEvidencePolicy(
            enabled=True,
            kind="transitions",
            level=0,
            limit=4,
            projection="metadata",
            grid_encoding="raw",
            max_queries=1,
            max_query_attempts=2,
        ),
        expected=CompatibilityExpected(evidence_ref="L0:T1"),
    )
    exact = {
        "kind": "transitions",
        "level": 0,
        "limit": 4,
        "projection": "metadata",
    }

    report = await execute_compatibility_gate(
        gate,
        repository_root=tmp_path,
        destination=tmp_path / "result",
        native_llm=FakeLLMClient(
            scripted_responses=[
                response("evidence_query", exact),
                response("return_result", answer()),
            ]
        ),
        codeact_llm=FakeLLMClient(
            scripted_responses=[
                response(
                    "execute_python",
                    {"code": f"page = evidence.query(**{exact!r})"},
                ),
                response("return_result", answer()),
            ]
        ),
        max_llm_calls=3,
    )

    assert report.passed is True
    assert report.codeact.metrics.llm_calls == 2
    assert report.codeact.metrics.query_attempts == 1
    assert report.codeact.metrics.evidence_queries == 1
