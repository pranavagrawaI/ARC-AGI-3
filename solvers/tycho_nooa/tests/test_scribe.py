from __future__ import annotations

import json
from pathlib import Path

import pytest
from nooa.context_blocks import ToolCallEvent
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa import GameTurn, TychoNooaAgent
from tycho_nooa import scribe as scribe_module


def tool_response(tool: str, arguments: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=tool, arguments=json.dumps(arguments))],
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


class RecordingFakeLLMClient(FakeLLMClient):
    """A FakeLLMClient that also records the tools offered on every call."""

    def __init__(self, scripted_responses: list[LLMResponse]) -> None:
        super().__init__(scripted_responses=scripted_responses)
        self.tools_per_call: list[list[str]] = []

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.tools_per_call.append([tool.name for tool in tools] if tools else [])
        return await super().acall(
            messages, tools=tools, output_model=output_model, **kwargs
        )


def turn(number: int, previous_action: str = "") -> GameTurn:
    return GameTurn(
        grid=[[0, number], [2, 3]],
        available_actions=["UP", "DOWN"],
        level=0,
        turn_in_level=number,
        previous_action=previous_action,
    )


def make_agent(
    llm: FakeLLMClient, tmp_path: Path
) -> TychoNooaAgent:
    return TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
        trigger_builder=False,
    )


@pytest.mark.asyncio
async def test_scribe_blocks_world_model_edit(tmp_path: Path) -> None:
    """world_model.py is builder-owned in this mode (tycho_nooa's mode always
    has actor_writes_wm=False) — a scribe attempt to edit it must be replaced
    with the fixed block message, mirroring pinned agent.py's
    _is_builder_owned_world_model_edit path."""
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            tool_response(
                "write_file",
                {"path": "world_model.py", "content": "sneaky"},
                "scribe-edit",
            ),
            prose_response("done"),
        ]
    )
    agent = make_agent(llm, tmp_path)

    world_model_path = agent.workspace_dir / "world_model.py"
    seeded = world_model_path.read_text() if world_model_path.exists() else None

    await agent.act(turn(0))
    await agent._summarize_level(0)

    assert (world_model_path.read_text() if world_model_path.exists() else None) == seeded
    tool_events = [
        event
        for event in agent.event_manager.values()
        if isinstance(event, ToolCallEvent) and event.name == "write_file"
    ]
    blocked = next(
        event for event in tool_events if event.arguments.get("path") == "world_model.py"
    )
    assert blocked.result is not None
    assert blocked.result.content == scribe_module._WORLD_MODEL_EDIT_BLOCK_MESSAGE


@pytest.mark.asyncio
async def test_scribe_withholds_take_action_from_offered_tools(tmp_path: Path) -> None:
    """The scribe consolidates a just-finished level; it must never be able to
    commit an environment action. Tycho offers take_action then intercepts its
    execution — tycho_nooa withholds it structurally instead."""
    llm = RecordingFakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("done"),
        ]
    )
    agent = make_agent(llm, tmp_path)

    await agent.act(turn(0))
    await agent._summarize_level(0)

    scribe_call_tools = llm.tools_per_call[1]
    assert scribe_call_tools, "scribe call must still offer non-action tools"
    assert "take_action" not in scribe_call_tools
    assert "submit_action" not in scribe_call_tools
    assert "write_file" in scribe_call_tools


@pytest.mark.asyncio
async def test_scribe_fallback_writes_deterministic_note_when_model_writes_nothing(
    tmp_path: Path,
) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("done"),
        ]
    )
    agent = make_agent(llm, tmp_path)

    await agent.act(turn(0))
    await agent._summarize_level(0)

    insights = (agent.workspace_dir / "notes/level_0_insights.md").read_text()
    assert insights.startswith("# Level 0 Insights")
    assert scribe_module._FALLBACK_REASON in insights
    assert "solved.json:" in insights
    assert "terminal.json:" in insights


@pytest.mark.asyncio
async def test_scribe_genuine_note_is_not_overwritten_by_fallback(tmp_path: Path) -> None:
    """Unlike the forced-report fallback in trigger_builder, the scribe's
    fallback only applies when the target note was never written at all — a
    genuine model-authored note must survive untouched."""
    genuine = "Level 0: UP wins by moving to the corner."
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            tool_response(
                "write_file",
                {"path": "notes/level_0_insights.md", "content": genuine},
                "scribe-write",
            ),
            prose_response("done"),
        ]
    )
    agent = make_agent(llm, tmp_path)

    await agent.act(turn(0))
    await agent._summarize_level(0)

    assert (agent.workspace_dir / "notes/level_0_insights.md").read_text() == genuine
