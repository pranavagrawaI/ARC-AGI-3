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
from tycho_nooa.sterile_codeact import SterileCodeActController
from tycho_nooa.sterile_execution import SterileCellTimeoutError


def response(name: str, arguments: dict, *, content: str = "") -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[ToolCall(id="call", name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": content},
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


def task_packet() -> CognitiveParityPacket:
    return CognitiveParityPacket(
        checkpoint_root="unused",
        task=CognitiveTask(
            packet_id="retrieve-old-transition",
            checkpoint_sha256="sha256:" + "0" * 64,
            evidence_frontier=EvidenceRef(level=0, turn=0),
            task_kind="retrieve",
            selection_stratum="remote_retrieval",
            historical_outcome="success",
            instruction="Retrieve the only recorded frame.",
        ),
        expected=ExpectedAnswer(
            rationale="The sole frame is L0:T0.",
            required_evidence_refs=(EvidenceRef(level=0, turn=0),),
        ),
    )


def workspace(tmp_path: Path) -> WorkspaceEvidenceCapability:
    recorded = GameWorkspace(
        "shadow-game",
        root=str(tmp_path),
        render=False,
        available_actions=["UP"],
    )
    recorded.record([[1]], level=0, turn_in_level=0)
    return WorkspaceEvidenceCapability(recorded)


def controller(
    tmp_path: Path,
    responses: list[LLMResponse],
    *,
    max_calls: int = 8,
    cell_timeout_seconds: float = 5.0,
) -> SterileCodeActController:
    log = ExecutionProvenanceLog(
        tmp_path / "provenance.jsonl", actor_session_id="actor-shadow"
    )
    provenance = InvocationProvenance(log, invocation_id="shadow-1")
    return SterileCodeActController(
        llm=FakeLLMClient(scripted_responses=responses),
        packet=task_packet(),
        evidence=workspace(tmp_path),
        provenance=provenance,
        max_llm_calls=max_calls,
        cell_timeout_seconds=cell_timeout_seconds,
    )


def retrieval_result() -> dict:
    return {
        "task_kind": "retrieve",
        "evidence_refs": [{"level": 0, "turn": 0}],
        "conclusion": "The frame exists.",
    }


async def test_evidence_cell_next_turn_and_context_reconstruct_from_raw_events(
    tmp_path: Path,
) -> None:
    subject = controller(
        tmp_path,
        [
            response(
                "execute_python",
                {
                    "code": (
                        "page = evidence.query(kind='frames'); "
                        "print(page['records'][0]['level'])"
                    )
                },
                content="I will inspect the frozen evidence.",
            ),
            response("return_result", {"result": retrieval_result()}),
        ],
    )

    answer = await subject.run()

    assert answer.conclusion == "The frame exists."
    assert len(subject.provider_turns) == 2
    for turn in subject.provider_turns:
        assert list(turn.messages) == subject.reconstructed_messages(
            event_seq=turn.event_seq
        )
    second = "\n".join(
        str(message["content"]) for message in subject.provider_turns[1].messages
    )
    assert "I will inspect the frozen evidence" in second
    assert "&quot;stdout&quot;" in second
    assert "CellCompleted" in second
    assert subject.provider_turns[0].usage == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
    }
    assert subject.provider_turns[0].tool_names == ("execute_python",)
    returned = next(
        event
        for event in subject._provenance.log.events
        if event.event_type == "CapabilityReturned"
    )
    assert returned.payload["result"]["grid_encoding"] == "rle"
    names = {tool.name for tool in subject._llm.last_tools or []}
    assert names == {"execute_python", "return_result"}


async def test_syntax_error_is_visible_before_repaired_cell(tmp_path: Path) -> None:
    subject = controller(
        tmp_path,
        [
            response("execute_python", {"code": "broken ="}),
            response("execute_python", {"code": "repaired = True"}),
            response("return_result", {"result": retrieval_result()}),
        ],
    )

    await subject.run()

    assert len(subject.provider_turns) == 3
    assert "SyntaxError" in json.dumps(subject.provider_turns[1].messages)


async def test_capability_rejection_keeps_repl_alive(tmp_path: Path) -> None:
    subject = controller(
        tmp_path,
        [
            response(
                "execute_python",
                {"code": "marker = 7\nevidence.query(kind='terminal')"},
            ),
            response("execute_python", {"code": "print(marker)"}),
            response("return_result", {"result": retrieval_result()}),
        ],
    )

    await subject.run()

    events = subject._provenance.log.events
    assert any(event.event_type == "CapabilityRejected" for event in events)
    visible = "\n".join(
        str(message["content"]) for message in subject.provider_turns[2].messages
    )
    assert "&quot;stdout&quot;" in visible
    assert "7" in visible


async def test_invalid_typed_return_is_recorded_then_retried(tmp_path: Path) -> None:
    invalid = {**retrieval_result(), "evidence_refs": "not-a-list"}
    subject = controller(
        tmp_path,
        [
            response("return_result", {"result": invalid}),
            response("return_result", {"result": retrieval_result()}),
        ],
    )

    await subject.run()

    types = [event.event_type for event in subject._provenance.log.events]
    assert types.count("ReturnRejected") == 1
    assert types.count("ReturnAccepted") == 1
    assert types.count("InvocationCompleted") == 1


async def test_timeout_terminates_without_another_provider_turn(tmp_path: Path) -> None:
    subject = controller(
        tmp_path,
        [response("execute_python", {"code": "while True: pass"})],
        max_calls=2,
        cell_timeout_seconds=0.1,
    )
    with pytest.raises(SterileCellTimeoutError):
        await subject.run()

    assert len(subject.provider_turns) == 1
    assert subject._provenance.log.events[-1].event_type == "InvocationFailed"


async def test_zero_tool_provider_turn_is_rejected_and_retried(
    tmp_path: Path,
) -> None:
    malformed = LLMResponse(
        raw_response=None,
        content="",
        reasoning="malformed provider reasoning",
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )
    subject = controller(
        tmp_path,
        [
            malformed,
            response("return_result", {"result": retrieval_result()}),
        ],
    )

    answer = await subject.run()

    assert answer.conclusion == "The frame exists."
    rejected = next(
        event
        for event in subject._provenance.log.events
        if event.event_type == "ProviderTurnRejected"
    )
    assert rejected.payload["reason"] == (
        "sterile CodeAct requires exactly one tool call per turn"
    )
    visible = json.dumps(subject.provider_turns[1].messages)
    assert "ProviderTurnRejected" in visible
    assert "exactly one execute_python or return_result" in visible
