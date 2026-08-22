from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa import GameTurn, TychoNooaAgent
from tycho_nooa.decision_checkpoints import (
    DecisionCheckpointRecorder,
    restore_workspace,
)


def response(tool: str, arguments: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name=tool,
                arguments=json.dumps(arguments, separators=(",", ":")),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def prose_response(content: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


class MessageRecordingFakeLLMClient(FakeLLMClient):
    def __init__(self, scripted_responses: list[LLMResponse]) -> None:
        super().__init__(scripted_responses=scripted_responses)
        self.messages_per_call: list[list[dict]] = []

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.messages_per_call.append(messages)
        return await super().acall(
            messages, tools=tools, output_model=output_model, **kwargs
        )


def turn() -> GameTurn:
    return GameTurn(
        grid=[[0, 1], [2, 3]],
        available_actions=["UP", "DOWN"],
        sequence=7,
        level=0,
        turn_in_level=0,
    )


@pytest.mark.asyncio
async def test_each_model_call_captures_a_committed_request_and_restorable_workspace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "run" / "tool-trace.jsonl"
    llm = MessageRecordingFakeLLMClient(
        scripted_responses=[
            response(
                "write_file",
                {"path": "notes/probe.md", "content": "first hypothesis"},
                "write-1",
            ),
            response("take_action", {"action": "UP"}, "act-1"),
        ]
    )
    llm.config.update(
        api_key="must-not-enter-evidence",
        api_base="https://example.invalid/v1",
        reasoning={"enabled": True},
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path / "workspace",
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )

    await agent.act(turn())

    root = trace_path.parent / "decision-checkpoints"
    calls = sorted((root / "calls").iterdir())
    assert [path.name for path in calls] == ["000001", "000002"]

    request_bytes = (calls[0] / "request.json").read_bytes()
    request = json.loads(request_bytes)
    assert request["schema_version"] == 1
    assert request["call_index"] == 1
    assert request["source"] == "actor"
    assert request["request_boundary"] == "nooa.llm_call"
    assert request["decision_point"] == {
        "sequence": 7,
        "level": 0,
        "turn_in_level": 0,
        "phase": "decision",
    }
    assert request["request_body"]["model"] == "fake-model"
    assert request["replay_context"]["client_type"].endswith("FakeLLMClient")
    assert request["replay_context"]["client_config"] == {
        "api_base": "https://example.invalid/v1",
        "reasoning": {"enabled": True},
    }
    assert request["replay_context"]["omitted_credential_keys"] == ["api_key"]
    assert request["request_body"]["messages"] == llm.messages_per_call[0]
    assert {tool["name"] for tool in request["request_body"]["tools"]} == {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "run_python",
        "set_verbosity",
        "take_action",
    }
    committed = request["request_payload_commitment"]
    body_bytes = json.dumps(
        request["request_body"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert committed == f"sha256:{hashlib.sha256(body_bytes).hexdigest()}"
    assert "must-not-enter-evidence" not in request_bytes.decode()

    before_write = tmp_path / "restored-before-write"
    restore_workspace(root, call_index=1, destination=before_write)
    assert not (before_write / "notes/probe.md").exists()
    assert (before_write / "world_model.py").is_file()

    after_write = tmp_path / "restored-after-write"
    restore_workspace(root, call_index=2, destination=after_write)
    assert (after_write / "notes/probe.md").read_text() == "first hypothesis"

    response_record = json.loads((calls[0] / "response.json").read_text())
    assert response_record["status"] == "complete"
    assert response_record["tool_calls"] == [
        {
            "arguments": '{"path":"notes/probe.md","content":"first hypothesis"}',
            "id": "write-1",
            "name": "write_file",
        }
    ]

    blob_files = [path for path in (root / "blobs" / "sha256").iterdir()]
    manifest_entries = sum(
        len(json.loads((call / "workspace.json").read_text())["files"])
        for call in calls
    )
    assert len(blob_files) < manifest_entries, "unchanged files must be deduplicated"


def test_restore_refuses_to_overwrite_an_existing_destination(tmp_path: Path) -> None:
    root = tmp_path / "decision-checkpoints"
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        restore_workspace(root, call_index=1, destination=destination)


def test_a_new_recorder_appends_to_an_existing_run_directory(tmp_path: Path) -> None:
    root = tmp_path / "decision-checkpoints"
    (root / "calls" / "000001").mkdir(parents=True)
    recorder = DecisionCheckpointRecorder(
        root,
        workspace_dir=tmp_path / "workspace",
        llm=FakeLLMClient(),
    )

    call_index, call_dir = recorder._claim_call_directory()

    assert call_index == 2
    assert call_dir == root / "calls" / "000002"
