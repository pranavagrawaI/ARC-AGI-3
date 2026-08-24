from __future__ import annotations

import json
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.cognitive_parity import (
    CognitiveParityPacket,
    CognitiveTask,
    ExpectedAnswer,
)
from tycho_nooa.evidence import EvidenceRef, WorkspaceEvidenceCapability
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.matched_actor import (
    MatchedCodeActController,
    MatchedNativeActorController,
    MatchedRetrievalReturn,
    cognitive_answer,
)
from tycho_nooa.matched_actor_audit import audit_matched_actor_packet
from tycho_nooa.matched_actor_corpus import load_matched_actor_corpus
from tycho_nooa.matched_actor_experiment import execute_matched_actor_packet
from tycho_nooa.matched_actor_manifest import load_matched_actor_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORPUS = REPOSITORY_ROOT / "analysis/matched-actor-v1/corpus.json"
EXPERIMENT = REPOSITORY_ROOT / "analysis/matched-actor-v1/experiment.json"


def response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(id=name, name=name, arguments=json.dumps(arguments))
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 2, "completion_tokens": 1, "cost": 0.01},
    )


def packet() -> CognitiveParityPacket:
    return CognitiveParityPacket(
        checkpoint_root="unused",
        task=CognitiveTask(
            packet_id="matched-control",
            checkpoint_sha256="sha256:" + "0" * 64,
            evidence_frontier=EvidenceRef(level=0, turn=1),
            task_kind="retrieve",
            selection_stratum="remote_retrieval",
            historical_outcome="success",
            instruction="Return the only transition.",
        ),
        expected=ExpectedAnswer(
            rationale="Only L0:T1 exists.",
            required_evidence_refs=(EvidenceRef(level=0, turn=1),),
        ),
    )


def capability(tmp_path: Path) -> WorkspaceEvidenceCapability:
    workspace = GameWorkspace(
        "matched-actor",
        root=str(tmp_path),
        render=False,
        available_actions=["UP"],
    )
    workspace.record([[11]], level=0, turn_in_level=0)
    workspace.record([[5]], action="UP", level=0, turn_in_level=1)
    return WorkspaceEvidenceCapability(workspace)


async def test_native_actor_keeps_evidence_out_of_python(tmp_path: Path) -> None:
    final = {
        "conclusion": "The only transition is L0:T1.",
        "evidence_refs": [{"level": 0, "turn": 1}],
    }
    llm = FakeLLMClient(
        scripted_responses=[
            response("evidence_query", {"kind": "transitions", "grid_encoding": "rle"}),
            response(
                "run_python",
                {
                    "code": (
                        "print(total, 'evidence' in globals(), "
                        "list(rle.decode_row(row)))"
                    ),
                    "values": {"total": 1, "row": "1:b"},
                },
            ),
            response("return_result", final),
        ]
    )
    subject = MatchedNativeActorController(
        llm=llm,
        packet=packet(),
        evidence=capability(tmp_path),
        max_llm_calls=4,
        max_output_tokens=256,
        cell_timeout_seconds=2,
    )

    result = await subject.run()

    assert result == MatchedRetrievalReturn.model_validate(final)
    assert subject.python_cells == 1
    assert subject.computation_records[0].stdout == "1 False [11]\n"
    assert subject.terminal_return_call == 3
    assert [action.name for action in subject.model_actions] == [
        "evidence_query",
        "run_python",
        "return_result",
    ]
    assert cognitive_answer(packet(), result).task_kind == "retrieve"
    assert {tool.name for tool in llm.last_tools or ()} == {
        "evidence_query",
        "run_python",
        "rle_decode_row",
        "return_result",
    }


async def test_codeact_actor_composes_query_decode_and_typed_return(
    tmp_path: Path,
) -> None:
    final = {
        "conclusion": "The only transition is L0:T1.",
        "evidence_refs": [{"level": 0, "turn": 1}],
    }
    llm = FakeLLMClient(
        scripted_responses=[
            response(
                "execute_python",
                {
                    "code": (
                        "page = evidence.query(kind='transitions', level=0, limit=1, "
                        "projection='full', grid_encoding='rle')\n"
                        "record = page.records[0]\n"
                        "print(record.ref, rle.decode_row(record.next_rows[0])[0])"
                    )
                },
            ),
            response("return_result", final),
        ]
    )
    log = ExecutionProvenanceLog(
        tmp_path / "provenance.jsonl", actor_session_id="matched"
    )
    provenance = InvocationProvenance(log, invocation_id="matched-codeact")
    subject = MatchedCodeActController(
        llm=llm,
        packet=packet(),
        evidence=capability(tmp_path / "evidence"),
        provenance=provenance,
        max_llm_calls=3,
        max_output_tokens=256,
        cell_timeout_seconds=2,
    )

    result = await subject.run()

    assert result == MatchedRetrievalReturn.model_validate(final)
    assert subject.terminal_return_call == 2
    assert [action.name for action in subject.model_actions] == [
        "execute_python",
        "return_result",
    ]
    completed = [event for event in log.events if event.event_type == "CellCompleted"]
    assert completed[0].payload["stdout"] == "L0:T1 5\n"


async def test_every_frozen_packet_has_matched_scripted_reachability(
    tmp_path: Path,
) -> None:
    manifest, packets = load_matched_actor_corpus(
        CORPUS, repository_root=REPOSITORY_ROOT
    )

    audits = [
        await audit_matched_actor_packet(
            packet,
            repository_root=REPOSITORY_ROOT,
            destination=tmp_path / packet.task.packet_id,
        )
        for packet in packets
    ]

    assert manifest.packet_count == 12
    assert all(audit.evidence_disclosures_identical for audit in audits)
    assert all(audit.computation_stdout_identical for audit in audits)
    assert all(audit.native_answer_correct for audit in audits)
    assert all(audit.codeact_answer_correct for audit in audits)
    assert all(not audit.native_python_has_evidence_authority for audit in audits)
    assert all(not audit.canonical_write_authority for audit in audits)
    assert all(not audit.arc_action_authority for audit in audits)


def test_disabled_package_refuses_paid_execution() -> None:
    manifest = load_matched_actor_manifest(
        EXPERIMENT, repository_root=REPOSITORY_ROOT
    )

    with pytest.raises(RuntimeError, match="execution is disabled"):
        manifest.require_execution_authorized()

    assert manifest.execution_authorized is False


async def test_matched_runner_reports_correctness_cost_and_milestones(
    tmp_path: Path,
) -> None:
    _, packets = load_matched_actor_corpus(CORPUS, repository_root=REPOSITORY_ROOT)
    packet_value = next(
        item for item in packets if item.task.packet_id == "remote-success-old-up"
    )
    final = {
        "conclusion": "The only and earliest UP is L0:T1.",
        "evidence_refs": [{"level": 0, "turn": 1}],
    }
    native_llm = FakeLLMClient(
        scripted_responses=[
            response(
                "evidence_query",
                {
                    "kind": "transitions",
                    "level": 0,
                    "limit": 32,
                    "projection": "metadata",
                    "grid_encoding": "rle",
                },
            ),
            response("return_result", final),
        ]
    )
    codeact_llm = FakeLLMClient(
        scripted_responses=[
            response(
                "execute_python",
                {
                    "code": (
                        "page = evidence.query(kind='transitions', level=0, limit=32, "
                        "projection='metadata', grid_encoding='rle')\n"
                        "print(page.records[0].ref)"
                    )
                },
            ),
            response("return_result", final),
        ]
    )

    report = await execute_matched_actor_packet(
        packet_value,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "matched-report",
        native_llm=native_llm,
        codeact_llm=codeact_llm,
        max_provider_calls=3,
        max_output_tokens=256,
        session_timeout_seconds=10,
        cell_timeout_seconds=2,
    )

    assert report.native.evaluation.evidence_recall == 1
    assert report.codeact.evaluation.evidence_recall == 1
    assert report.native.resources.provider_calls == 2
    assert report.codeact.resources.provider_calls == 2
    assert report.native.resources.dollar_cost == pytest.approx(0.02)
    assert report.codeact.resources.dollar_cost == pytest.approx(0.02)
    assert report.native.milestones.target_record_disclosed == 1
    assert report.codeact.milestones.target_record_disclosed == 1
    assert report.native.milestones.candidate_answer_formed == 2
    assert report.codeact.milestones.terminal_result_accepted == 2
