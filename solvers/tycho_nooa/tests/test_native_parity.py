from __future__ import annotations

import json
from pathlib import Path

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.cognitive_parity import (
    CognitiveParityPacket,
    CognitiveTask,
    ExpectedAnswer,
)
from tycho_nooa.evidence import EvidenceRef, WorkspaceEvidenceCapability
from tycho_nooa.native_parity import NativeParityController


def response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=name, name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 2, "completion_tokens": 1},
    )


def packet() -> CognitiveParityPacket:
    return CognitiveParityPacket(
        checkpoint_root="unused",
        task=CognitiveTask(
            packet_id="native-control",
            checkpoint_sha256="sha256:" + "0" * 64,
            evidence_frontier=EvidenceRef(level=0, turn=0),
            task_kind="retrieve",
            selection_stratum="remote_retrieval",
            historical_outcome="success",
            instruction="Return the only frame.",
        ),
        expected=ExpectedAnswer(
            rationale="Only L0:T0 exists.",
            required_evidence_refs=(EvidenceRef(level=0, turn=0),),
        ),
    )


def capability(tmp_path: Path) -> WorkspaceEvidenceCapability:
    workspace = GameWorkspace(
        "native-parity",
        root=str(tmp_path),
        render=False,
        available_actions=["UP"],
    )
    workspace.record([[1]], level=0, turn_in_level=0)
    return WorkspaceEvidenceCapability(workspace)


def answer() -> dict:
    return {
        "task_kind": "retrieve",
        "evidence_refs": [{"level": 0, "turn": 0}],
        "conclusion": "Found it.",
    }


async def test_native_control_exposes_only_query_and_typed_return(tmp_path: Path) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            response("evidence_query", {"kind": "frames"}),
            response("return_result", {"result": answer()}),
        ]
    )
    subject = NativeParityController(
        llm=llm,
        packet=packet(),
        evidence=capability(tmp_path),
    )

    result = await subject.run()

    assert result.conclusion == "Found it."
    assert subject.evidence_queries == 1
    assert subject.invalid_returns == 0
    assert [turn.tool_name for turn in subject.provider_turns] == [
        "evidence_query",
        "return_result",
    ]
    assert {tool.name for tool in llm.last_tools or []} == {
        "evidence_query",
        "return_result",
    }
    assert "expected" not in json.dumps(llm.last_messages)


async def test_invalid_return_is_feedback_not_terminal(tmp_path: Path) -> None:
    bad = {**answer(), "evidence_refs": "wrong"}
    llm = FakeLLMClient(
        scripted_responses=[
            response("return_result", {"result": bad}),
            response("return_result", {"result": answer()}),
        ]
    )
    subject = NativeParityController(
        llm=llm,
        packet=packet(),
        evidence=capability(tmp_path),
    )

    await subject.run()

    assert subject.invalid_returns == 1
    assert "RETURN_REJECTED" in json.dumps(llm.last_messages)
