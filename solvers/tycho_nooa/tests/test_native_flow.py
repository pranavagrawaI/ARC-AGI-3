from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from nooa.context_blocks import ToolCallEvent
from nooa.errors import GenerationError
from nooa.events import Error, Summary, Task
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tycho.agent.colors import COLORMAP
from tycho.prompts.render import render_prompt

from tycho_nooa import ActionDecision, GameTurn, TychoNooaAgent
from tycho_nooa.llm_trace import CAPTURE_RESPONSE_TEXT_ENV
from tycho_nooa.models import (
    TYCHO_WORKSPACE_TURN_SCHEMA,
    tycho_workspace_available_actions,
)
from tycho_nooa.strategy import NativeToolConfig


def action_arguments(tool: str, arguments: dict) -> dict:
    if tool == "take_actions" and "actions" not in arguments:
        return {
            "actions": [arguments],
            "rationale": "test one-action commitment",
        }
    return arguments


def response(tool: str, arguments: dict, call_id: str) -> LLMResponse:
    arguments = action_arguments(tool, arguments)
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=tool, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def malformed_response(tool: str, arguments: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=tool, arguments=arguments)],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def batched_response(calls: list[tuple[str, dict, str]]) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name=tool,
                arguments=json.dumps(action_arguments(tool, arguments)),
            )
            for tool, arguments, call_id in calls
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def reasoned_response(
    tool: str, arguments: dict, call_id: str, *, content: str, reasoning: str
) -> LLMResponse:
    """A tool call the model narrated, in both channels a provider can fill."""
    arguments = action_arguments(tool, arguments)
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[ToolCall(id=call_id, name=tool, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": content},
        reasoning=reasoning,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def prose_response(content: str) -> LLMResponse:
    """A tool-free reply, used to model refusal or prose-only output."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


class ToolsRecordingFakeLLMClient(FakeLLMClient):
    """A FakeLLMClient that also records the tools offered on every call."""

    def __init__(self, scripted_responses: list[LLMResponse]) -> None:
        super().__init__(scripted_responses=scripted_responses)
        self.tools_per_call: list[list[str]] = []
        self.tool_choices: list[object] = []
        self.max_tokens_per_call: list[int | None] = []

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.tools_per_call.append([tool.name for tool in tools] if tools else [])
        self.tool_choices.append(kwargs.get("tool_choice"))
        self.max_tokens_per_call.append(kwargs.get("max_tokens"))
        return await super().acall(
            messages, tools=tools, output_model=output_model, **kwargs
        )


class MessageRecordingFakeLLMClient(FakeLLMClient):
    """A FakeLLMClient that keeps every call's messages, not just the last."""

    def __init__(self, scripted_responses: list[LLMResponse]) -> None:
        super().__init__(scripted_responses=scripted_responses)
        self.messages_per_call: list[list[dict]] = []

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.messages_per_call.append(messages)
        return await super().acall(
            messages, tools=tools, output_model=output_model, **kwargs
        )


class RequestRecordingFakeLLMClient(FakeLLMClient):
    """Record the complete model-facing request contract for each call."""

    def __init__(self, scripted_responses: list[LLMResponse]) -> None:
        super().__init__(scripted_responses=scripted_responses)
        self.requests: list[dict[str, object]] = []

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.requests.append(
            {
                "messages": messages,
                "tools": [tool.name for tool in tools] if tools else [],
                "tool_descriptions": {
                    tool.name: tool.description for tool in tools or []
                },
                "tool_choice": kwargs.get("tool_choice"),
                "max_tokens": kwargs.get("max_tokens"),
            }
        )
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


def test_tycho_workspace_turn_v1_projects_available_actions_once(
    tmp_path: Path,
) -> None:
    agent = TychoNooaAgent(
        llm=FakeLLMClient(),
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    agent._begin_decision(turn(0), None)

    metadata = json.loads(
        (agent.workspace_dir / "level_0/turn_000.json").read_text()
    )
    assert TYCHO_WORKSPACE_TURN_SCHEMA == "tycho-workspace-turn-v1"
    assert metadata["available"] == ["UP", "DOWN"]
    assert "available_actions" not in metadata
    assert tycho_workspace_available_actions(metadata) == ["UP", "DOWN"]


@pytest.mark.asyncio
async def test_accepted_action_does_not_wait_for_post_action_llm_call(
    tmp_path: Path,
) -> None:
    """Regression for the live 600-second timeout at decision sequence 2.

    Once take_action is accepted, the worker must have no provider-dependent
    work left before it can return the decision to the host.
    """
    llm = ToolsRecordingFakeLLMClient(
        scripted_responses=[response("take_actions", {"action": "UP"}, "act-1")]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")
    assert len(llm.tools_per_call) == 1


@pytest.mark.asyncio
async def test_files_and_conversation_survive_the_action_boundary(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "tool-trace.jsonl"
    llm = FakeLLMClient(
        scripted_responses=[
            response(
                "write_file",
                {
                    "path": "notes/world_model.md",
                    "content": "UP may move the blue cell.",
                },
                "write-1",
            ),
            response("take_actions", {"action": "UP"}, "act-1"),
            response("read_file", {"path": "notes/world_model.md"}, "read-2"),
            response("take_actions", {"action": "DOWN"}, "act-2"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )

    first = await agent.act(turn(0))
    identity = agent.agent_id
    second = await agent.act(turn(1, previous_action="UP"))

    assert first == ActionDecision(action="UP")
    assert second == ActionDecision(action="DOWN")
    assert agent.agent_id == identity
    assert (agent.workspace_dir / "notes/world_model.md").read_text() == (
        "UP may move the blue cell."
    )

    calls = [
        event
        for event in agent.event_manager.values()
        if isinstance(event, ToolCallEvent)
    ]
    assert [event.name for event in calls] == [
        "write_file",
        "take_actions",
        "read_file",
        "take_actions",
    ]
    assert calls[2].result is not None
    assert calls[2].result.content == "UP may move the blue cell."
    rendered_history = json.dumps(llm.last_messages)
    assert "UP may move the blue cell." in rendered_history
    assert "turn_in_level': 1" in rendered_history
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [record["name"] for record in trace] == [
        "write_file",
        "take_actions",
        "read_file",
        "take_actions",
    ]
    assert [record["turn_in_level"] for record in trace] == [0, 0, 1, 1]
    assert trace[0]["arguments"]["content"] == "UP may move the blue cell."
    assert trace[2]["result"] == "UP may move the blue cell."


@pytest.mark.asyncio
async def test_no_world_model_actor_can_dispatch_every_offered_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The complete control surface reaches the real Actor dispatch path."""
    monkeypatch.setenv("TYCHO_MODE", "no_world_model")
    monkeypatch.setenv("TYCHO_NOOA_TREATMENT", "control")
    trace_path = tmp_path / "tool-trace.jsonl"
    offered = [
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "run_python",
        "set_verbosity",
        "take_actions",
    ]
    llm = ToolsRecordingFakeLLMClient(
        scripted_responses=[
            response(
                "write_file",
                {"path": "notes/access.txt", "content": "before"},
                "write",
            ),
            response(
                "edit_file",
                {"path": "notes/access.txt", "old": "before", "new": "after"},
                "edit",
            ),
            response("ls", {"path": "notes"}, "list"),
            response("read_file", {"path": "notes/access.txt"}, "read"),
            response(
                "run_python",
                {"code": "print('actor-tool-access')"},
                "python",
            ),
            response(
                "set_verbosity",
                {"grid": "off", "diff": "summary"},
                "verbosity",
            ),
            response("take_actions", {"action": "UP"}, "action"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")
    assert llm.tools_per_call == [offered] * 7
    assert (agent.workspace_dir / "notes/access.txt").read_text() == "after"
    assert agent._tool_surface.grid_verbosity == "off"
    assert agent._tool_surface.diff_verbosity == "summary"

    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [record["name"] for record in trace] == [
        "write_file",
        "edit_file",
        "ls",
        "read_file",
        "run_python",
        "set_verbosity",
        "take_actions",
    ]
    assert all(
        record["parse_status"] == "complete"
        and record["validation_status"] == "complete"
        and record["execution_status"] == "complete"
        and record["status"] == "complete"
        for record in trace
    )
    assert "access.txt" in trace[2]["result"]
    assert trace[1]["validated_arguments"] == {
        "path": "notes/access.txt",
        "old": "before",
        "new": "after",
    }
    assert trace[3]["result"] == "after"
    assert "actor-tool-access" in trace[4]["result"]


@pytest.mark.asyncio
async def test_actor_beliefs_checkpoint_is_forced_when_turn_leaves_it_stale(
    tmp_path: Path,
) -> None:
    """A stale hand-off gets a factual local breadcrumb after the action."""
    llm = ToolsRecordingFakeLLMClient(
        scripted_responses=[response("take_actions", {"action": "UP"}, "act-1")]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    notes_path = agent.workspace_dir / "notes/actor_beliefs.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("turn 0: UP moved the marker\n")

    assert await agent.act(turn(0)) == ActionDecision(action="UP")

    updated = notes_path.read_text()
    assert updated.startswith("turn 0: UP moved the marker")
    assert (
        "[checkpoint: model committed UP with no model-authored beliefs update]"
        in updated
    )

    tasks = [event for event in agent.event_manager.values() if isinstance(event, Task)]
    assert not any("checkpoint your current hypotheses" in task.prompt for task in tasks)
    assert len(llm.tools_per_call) == 1


@pytest.mark.asyncio
async def test_actor_beliefs_written_before_action_is_preserved_without_fallback(
    tmp_path: Path,
) -> None:
    """The normal tool loop remains the model's chance to author hypotheses."""
    llm = FakeLLMClient(
        scripted_responses=[
            response(
                "write_file",
                {
                    "path": "notes/actor_beliefs.md",
                    "content": "genuine hypothesis: UP moves the marker up",
                },
                "beliefs-write",
            ),
            response("take_actions", {"action": "UP"}, "act-1"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")

    content = (agent.workspace_dir / "notes/actor_beliefs.md").read_text()
    assert content == "genuine hypothesis: UP moves the marker up"
    assert "[checkpoint: model committed" not in content


@pytest.mark.asyncio
async def test_illegal_action_is_feedback_not_a_host_action(tmp_path: Path) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            response("take_actions", {"action": "LEFT"}, "bad-action"),
            response("take_actions", {"action": "UP"}, "good-action"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    decision = await agent.act(turn(0))

    assert decision == ActionDecision(action="UP")
    calls = [
        event
        for event in agent.event_manager.values()
        if isinstance(event, ToolCallEvent)
    ]
    assert calls[0].result is not None
    assert calls[0].result.result_status == "error"


@pytest.mark.asyncio
async def test_malformed_native_arguments_are_feedback_not_a_host_action(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "tool-trace.jsonl"
    llm = FakeLLMClient(
        scripted_responses=[
            malformed_response("take_actions", "{broken", "bad-json"),
            response("take_actions", {"action": "UP"}, "good-action"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert records[0]["name"] == "take_actions"
    assert records[0]["status"] == "error"
    assert records[0]["raw_arguments"] == "{broken"
    assert records[0]["parse_status"] == "error"
    assert records[0]["validation_status"] == "not_attempted"
    assert records[0]["execution_status"] == "not_attempted"
    assert records[0]["error_stage"] == "parse"
    assert "Invalid native tool call" in records[0]["result"]
    assert records[1]["parse_status"] == "complete"
    assert records[1]["validation_status"] == "complete"
    assert records[1]["execution_status"] == "complete"
    assert records[1]["validated_arguments"] == {
        "actions": [{"action": "UP"}],
        "rationale": "test one-action commitment",
        "stop_on_no_change": True,
    }
    parse_error = next(
        event for event in agent.event_manager.values() if isinstance(event, Error)
    )
    assert json.loads(parse_error.content) == {
        "available_actions": ["UP", "DOWN"],
        "code": "invalid_json",
        "message": "Tool arguments are not valid JSON.",
        "stage": "parse",
        "turn_consumed": False,
    }


@pytest.mark.asyncio
async def test_native_schema_validation_is_distinct_from_execution(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "tool-trace.jsonl"
    llm = FakeLLMClient(
        scripted_responses=[
            response(
                "take_actions",
                {"action": "DOWN", "row": 10, "col": 10},
                "bad-schema",
            ),
            response("take_actions", {"action": "UP"}, "good-action"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")
    request = json.loads(
        (tmp_path / "decision-checkpoints/calls/000001/request.json").read_text()
    )
    take_actions = next(
        tool
        for tool in request["request_body"]["tools"]
        if tool["name"] == "take_actions"
    )
    assert [
        tool["name"]
        for tool in request["request_body"]["tools"]
        if tool["name"].startswith("take_action")
    ] == ["take_actions"]
    parameters = take_actions["parameters"]
    assert parameters["properties"]["actions"]["minItems"] == 1
    assert parameters["properties"]["actions"]["maxItems"] == 4
    item = parameters["properties"]["actions"]["items"]
    assert item["properties"]["action"] == {
        "enum": ["UP", "DOWN"],
        "type": "string",
    }
    assert parameters["additionalProperties"] is False
    rejected = json.loads(trace_path.read_text().splitlines()[0])
    assert rejected["parse_status"] == "complete"
    assert rejected["validation_status"] == "error"
    assert rejected["execution_status"] == "not_attempted"
    assert rejected["validated_arguments"] is None
    assert rejected["error_stage"] == "validation"
    rejected_event = next(
        event
        for event in agent.event_manager.values()
        if isinstance(event, ToolCallEvent) and event.tool_call_id == "bad-schema"
    )
    assert rejected_event.result is not None
    assert json.loads(rejected_event.result.content) == {
        "available_actions": ["UP", "DOWN"],
        "code": "invalid_arguments",
        "message": "Tool arguments failed validation.",
        "stage": "validation",
        "turn_consumed": False,
    }


@pytest.mark.asyncio
async def test_take_actions_request_schema_uses_current_available_actions(
    tmp_path: Path,
) -> None:
    llm = FakeLLMClient(
        scripted_responses=[response("take_actions", {"action": "UP"}, "act")]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "CLICK"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    current = GameTurn(
        grid=[[0, 0], [2, 3]],
        available_actions=["UP", "CLICK"],
        level=0,
        turn_in_level=0,
    )
    assert await agent.act(current) == ActionDecision(action="UP")

    request = json.loads(
        (tmp_path / "decision-checkpoints/calls/000001/request.json").read_text()
    )
    take_actions = next(
        tool
        for tool in request["request_body"]["tools"]
        if tool["name"] == "take_actions"
    )
    item = take_actions["parameters"]["properties"]["actions"]["items"]
    click, non_click = item["oneOf"]
    assert click["properties"]["action"] == {"const": "CLICK"}
    assert click["required"] == ["action", "row", "col"]
    assert non_click["properties"]["action"] == {"enum": ["UP"]}
    assert non_click["required"] == ["action"]
    assert all(
        "expected_before_grid_sha256" in branch["properties"]
        for branch in item["oneOf"]
    )
    assert "ACTION1" not in take_actions["description"]
    assert "ACTION6" not in take_actions["description"]


@pytest.mark.asyncio
async def test_native_execution_failure_is_distinct_from_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = tmp_path / "tool-trace.jsonl"
    llm = FakeLLMClient(
        scripted_responses=[
            response("run_python", {"code": "raise"}, "execution-error"),
            response("take_actions", {"action": "UP"}, "good-action"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )

    def fail_execution(code: str, timeout_s: int = 15) -> str:
        del code, timeout_s
        raise RuntimeError("execution failed")

    monkeypatch.setattr(agent._tool_surface, "run_python", fail_execution)

    assert await agent.act(turn(0)) == ActionDecision(action="UP")
    rejected = json.loads(trace_path.read_text().splitlines()[0])
    assert rejected["parse_status"] == "complete"
    assert rejected["validation_status"] == "complete"
    assert rejected["execution_status"] == "error"
    assert rejected["error_stage"] == "execution"
    assert rejected["result"] == "RuntimeError: execution failed"
    rejected_event = next(
        event
        for event in agent.event_manager.values()
        if isinstance(event, ToolCallEvent) and event.tool_call_id == "execution-error"
    )
    assert rejected_event.result is not None
    assert json.loads(rejected_event.result.content) == {
        "available_actions": ["UP", "DOWN"],
        "code": "execution_failed",
        "message": "execution failed",
        "stage": "execution",
        "turn_consumed": False,
    }


@pytest.mark.asyncio
async def test_first_model_context_omits_internal_self_and_protocol_docs(
    tmp_path: Path,
) -> None:
    llm = MessageRecordingFakeLLMClient(
        scripted_responses=[response("take_actions", {"action": "UP"}, "act-1")]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    await agent.act(turn(0))

    first_call_messages = llm.messages_per_call[0]
    assert [message["role"] for message in first_call_messages] == ["system", "user"]
    rendered = json.dumps(first_call_messages)
    assert "<strategy_prompt>" in rendered
    assert "grid (row labels at left" in rendered
    assert "<self " not in rendered
    assert "<state " not in rendered
    assert "ActionResultMessage" not in rendered
    assert "ObservationPayload" not in rendered
    # Independent oracle: render Tycho's own sources directly rather than reusing
    # trigger_actor_world_model_paragraph, so this test can catch a broken
    # extraction or a duplicated insertion rather than just echoing it back.
    expected_wm_trigger = render_prompt("wm_trigger").replace(
        "The builder fires when the model is detectably wrong on dynamics or "
        "outcome, at level starts, and after GAME_OVER resets.",
        "The builder fires when the model is detectably wrong on dynamics or "
        "outcome, and at level starts.",
    )
    expected_system = render_prompt(
        "actor.system",
        colormap=COLORMAP,
        wm_variant="trigger",
        vision=False,
        show_grid=True,
        show_diff=True,
        lossless_cells=False,
        meta_reflect=False,
    )
    expected_beliefs = expected_system[
        expected_system.index("Keep your evolving hypotheses") :
    ].strip()
    # Compare against the raw system-message content, not the JSON-dumped
    # `rendered` string: json.dumps escapes newlines, so a multi-paragraph
    # substring match against `rendered` would spuriously fail even when the
    # text is present verbatim in the actual prompt.
    system_content = next(
        message["content"]
        for message in first_call_messages
        if message["role"] == "system"
    )

    assert expected_wm_trigger in system_content
    assert expected_beliefs in system_content
    assert "For CLICK games" in rendered
    assert "Planner mechanics" not in rendered
    assert "ACTION6" not in expected_wm_trigger
    assert "ACTION6" not in expected_beliefs

    # The ownership/beliefs/advisory instruction must appear exactly once —
    # from Tycho's paragraph alone, not echoed by the NOOA-specific framing too.
    assert rendered.count("notes/actor_beliefs.md") == 1
    assert rendered.count("report is advisory") == 1
    assert rendered.count("world_model.py") == 2
    assert rendered.count("ACTIVE SOLVER MODE: trigger") == 1


@pytest.mark.asyncio
async def test_validated_plan_hint_is_injected_and_traced(tmp_path: Path) -> None:
    trace_path = tmp_path / "tool-trace.jsonl"
    llm = FakeLLMClient(
        scripted_responses=[
            response("take_actions", {"action": "UP"}, "act-1"),
            response("take_actions", {"action": "DOWN"}, "act-2"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )
    current = turn(0)
    grid_bytes = np.asarray(current.grid, dtype=np.int16).tobytes()
    artifact = {
        "schema_version": 1,
        "status": "validated",
        "method": "astar",
        "start": {
            "level": 0,
            "turn": 0,
            "grid_sha256": hashlib.sha256(grid_bytes).hexdigest(),
        },
        "world_model_sha256": hashlib.sha256(
            (agent.workspace_dir / "world_model.py").read_bytes()
        ).hexdigest(),
        "plan_length": 1,
        "actions": [{"action": "UP", "row": None, "col": None}],
        "expected_grid_sha256": [hashlib.sha256(grid_bytes).hexdigest()],
        "validation": "canonical world_model replay reached level_complete",
        "elapsed_s": 0.01,
    }
    artifact_path = agent.workspace_dir / "notes" / "validated_plan.json"
    artifact_path.parent.mkdir(exist_ok=True)
    artifact_path.write_text(json.dumps(artifact))

    assert await agent.act(current) == ActionDecision(action="UP")
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    tasks = [event for event in agent.event_manager.values() if isinstance(event, Task)]
    assert "validated planner feedback:" in tasks[0].prompt
    assert "next action is UP" in tasks[0].prompt
    assert "observed frame diverged" in tasks[1].prompt
    planner_trace = [
        json.loads(line)
        for line in (tmp_path / "planner-trace.jsonl").read_text().splitlines()
    ]
    assert [record["status"] for record in planner_trace] == [
        "continuation",
        "invalidated_observation_diverged",
    ]
    assert planner_trace[0]["event_index"] == 1
    assert planner_trace[0]["level"] == 0
    assert planner_trace[0]["turn_in_level"] == 0
    assert "next action is UP" in planner_trace[0]["hint"]


@pytest.mark.asyncio
async def test_click_plan_hint_preserves_validated_coordinates(tmp_path: Path) -> None:
    trace_path = tmp_path / "tool-trace.jsonl"
    llm = FakeLLMClient(
        scripted_responses=[
            response(
                "take_actions",
                {"action": "CLICK", "row": 1, "col": 0},
                "click-1",
            )
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["CLICK"],
        tool_trace_path=trace_path,
    )
    current = GameTurn(
        grid=[[0, 0], [0, 0]],
        available_actions=["CLICK"],
        level=0,
        turn_in_level=0,
    )
    grid_hash = hashlib.sha256(
        np.asarray(current.grid, dtype=np.int16).tobytes()
    ).hexdigest()
    artifact = {
        "status": "validated",
        "method": "astar",
        "start": {"level": 0, "turn": 0, "grid_sha256": grid_hash},
        "world_model_sha256": hashlib.sha256(
            (agent.workspace_dir / "world_model.py").read_bytes()
        ).hexdigest(),
        "plan_length": 1,
        "actions": [{"action": "CLICK", "row": 1, "col": 0}],
        "expected_grid_sha256": [grid_hash],
    }
    artifact_path = agent.workspace_dir / "notes" / "validated_plan.json"
    artifact_path.parent.mkdir(exist_ok=True)
    artifact_path.write_text(json.dumps(artifact))

    assert await agent.act(current) == ActionDecision(action="CLICK", row=1, col=0)

    tasks = [event for event in agent.event_manager.values() if isinstance(event, Task)]
    assert "next action is CLICK(row=1,col=0)" in tasks[0].prompt


@pytest.mark.asyncio
async def test_planner_result_must_be_seen_before_action_and_artifact_is_traced(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "tool-trace.jsonl"
    llm = FakeLLMClient(
        scripted_responses=[
            batched_response(
                [
                    ("run_python", {"code": "python plan.py auto"}, "plan-1"),
                    ("take_actions", {"action": "UP"}, "blocked-act"),
                ]
            ),
            response("take_actions", {"action": "UP"}, "act-2"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )
    def execute_with_plan(command: str, timeout_s: int) -> str:
        assert command == "python plan.py auto"
        assert timeout_s == 15
        artifact_path = agent.workspace_dir / "notes" / "validated_plan.json"
        artifact_path.parent.mkdir(exist_ok=True)
        artifact_path.write_text(
            json.dumps(
                {
                    "status": "validated",
                    "method": "astar",
                    "start": {"level": 0, "turn": 0, "grid_sha256": "grid"},
                    "world_model_sha256": "model",
                    "plan_length": 1,
                    "actions": [{"action": "UP"}],
                    "expected_grid_sha256": ["next"],
                }
            )
        )
        return (
            "PLANNER_STATUS: plan_found (elapsed_s=0.01, length=1)\n"
            "PLANNER_ARTIFACT: notes/validated_plan.json"
        )

    agent._tool_surface._run_disposable_planner = execute_with_plan

    assert await agent.act(turn(0)) == ActionDecision(action="UP")

    calls = [
        event
        for event in agent.event_manager.values()
        if isinstance(event, ToolCallEvent)
    ]
    blocked = next(event for event in calls if event.tool_call_id == "blocked-act")
    assert blocked.result is not None
    assert blocked.result.result_status == "error"
    assert "cannot share one tool batch" in blocked.result.content
    assert "PLANNER_STATUS: plan_found" in json.dumps(llm.last_messages)
    attempted = [
        json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()
    ]
    blocked_trace = next(
        record for record in attempted if record["tool_call_id"] == "blocked-act"
    )
    assert blocked_trace["status"] == "error"
    assert "cannot share one tool batch" in blocked_trace["result"]

    planner_trace = [
        json.loads(line)
        for line in (tmp_path / "planner-trace.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in planner_trace] == [
        "planner_invocation",
        "planner_artifact",
    ]
    assert planner_trace[0]["status"] == "plan_found"
    assert planner_trace[1]["plan_length"] == 1


@pytest.mark.asyncio
async def test_subprocess_planner_call_is_traced_and_blocks_same_batch_action(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "tool-trace.jsonl"
    observed_code = (
        "import subprocess, textwrap, sys\n"
        "result = subprocess.run([sys.executable, 'plan.py', 'astar'], "
        "capture_output=True, text=True)\n"
        "print(result.stdout)\n"
        "print(result.stderr, file=sys.stderr)\n"
    )
    llm = FakeLLMClient(
        scripted_responses=[
            batched_response(
                [
                    ("run_python", {"code": observed_code}, "plan-subprocess"),
                    ("take_actions", {"action": "UP"}, "blocked-act"),
                ]
            ),
            response("take_actions", {"action": "UP"}, "act-2"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )
    original_execute = agent._tool_executor.execute

    def execute_with_no_plan(name: str, arguments: dict) -> str:
        if name == "run_python":
            return (
                "PLANNER_COMMAND: python plan.py astar (max_nodes=20000)\n"
                "PLANNER_STATUS: no_plan_exhausted (elapsed_s=0.02)"
            )
        return original_execute(name, arguments)

    agent._tool_executor.execute = execute_with_no_plan

    assert await agent.act(turn(0)) == ActionDecision(action="UP")

    calls = [
        event
        for event in agent.event_manager.values()
        if isinstance(event, ToolCallEvent)
    ]
    blocked = next(event for event in calls if event.tool_call_id == "blocked-act")
    assert blocked.result is not None
    assert blocked.result.result_status == "error"
    assert "cannot share one tool batch" in blocked.result.content
    planner_trace = [
        json.loads(line)
        for line in (tmp_path / "planner-trace.jsonl").read_text().splitlines()
    ]
    assert planner_trace[0]["event"] == "planner_invocation"
    assert planner_trace[0]["command"] == "python plan.py astar"
    assert planner_trace[0]["status"] == "no_plan_exhausted"


def test_planner_detector_ignores_incidental_plan_filename(tmp_path: Path) -> None:
    agent = TychoNooaAgent(
        llm=FakeLLMClient(),
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP"],
    )
    surface = agent._tool_surface

    assert surface.is_planner_call("run_python", {"code": "print('plan.py')"}) is False


@pytest.mark.parametrize(
    "path",
    [
        "plan.py",
        "verify.py",
        "notes/validated_plan.json",
        "notes/validated_plan_expected.npz",
        "notes/.validated_plan.forged.tmp",
    ],
)
def test_model_file_tools_cannot_write_host_owned_planner_paths(
    tmp_path: Path,
    path: str,
) -> None:
    agent = TychoNooaAgent(
        llm=FakeLLMClient(),
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP"],
    )
    agent._begin_decision(turn(0), None)

    for mutation in (
        lambda: agent._tool_surface.write_file(path, "FORGED"),
        lambda: agent._tool_surface.edit_file(path, "old", "FORGED"),
        lambda: agent._tool_surface.edit_function(path, "forged", "def forged(): pass"),
    ):
        with pytest.raises(PermissionError, match="host-owned planner path"):
            mutation()


def test_nested_plan_script_cannot_enter_t1_publication_path(tmp_path: Path) -> None:
    agent = TychoNooaAgent(
        llm=FakeLLMClient(),
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP"],
    )
    agent._begin_decision(turn(0), None)
    forged = """from pathlib import Path
print('PLANNER_COMMAND: python plan.py auto')
Path('notes/validated_plan.json').write_text('{}')
print('PLANNER_STATUS: plan_found (FORGED)')
"""

    assert agent._tool_surface.write_file("notes/plan.py", forged).startswith("wrote")
    assert not agent._tool_surface.is_planner_call(
        "run_python", {"code": "python notes/plan.py auto"}
    )
    result = agent._tool_surface.run_python("python notes/plan.py auto")

    assert "Read-only file system" in result
    assert "PLANNER_PUBLICATION: accepted" not in result
    assert not (agent.workspace_dir / "notes" / "validated_plan.json").exists()


def test_seeded_planner_publishes_only_validated_artifact_from_disposable_workspace(
    tmp_path: Path,
) -> None:
    agent = TychoNooaAgent(
        llm=FakeLLMClient(),
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP"],
    )
    current = turn(0)
    agent._begin_decision(current, None)
    seeded_plan = (agent.workspace_dir / "plan.py").read_bytes()
    (agent.workspace_dir / "world_model.py").write_text(
        """from dataclasses import dataclass
from pathlib import Path
import numpy as np
import wmlib

Path("level_999").mkdir(exist_ok=True)
Path("level_999/turn_000.json").write_text("fabricated")
Path("plan.py").write_text("forged after seed execution began")

@dataclass
class State:
    grid: np.ndarray
    done: bool = False

def init_state(grid, level):
    return State(grid=grid.copy())

def transition(state, action):
    result = wmlib.clone(state)
    result.done = action.get("action") == "UP"
    return result

def render(state):
    return state.grid

def outcome(state):
    return "level_complete" if state.done else "ongoing"

def actions(state):
    return [{"action": "UP", "row": None, "col": None}]
""",
        encoding="utf-8",
    )

    result = agent._tool_surface.run_python("python plan.py auto")

    artifact_path = agent.workspace_dir / "notes" / "validated_plan.json"
    assert artifact_path.is_file(), result
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "PLANNER_STATUS: plan_found" in result
    assert "Read-only file system" not in result
    assert artifact["status"] == "validated"
    assert artifact["actions"] == [{"action": "UP", "row": None, "col": None}]
    assert artifact["world_model_sha256"] == hashlib.sha256(
        (agent.workspace_dir / "world_model.py").read_bytes()
    ).hexdigest()
    assert not (agent.workspace_dir / "level_999").exists()
    assert (agent.workspace_dir / "plan.py").read_bytes() == seeded_plan


def test_t1_rejects_pinned_planner_output_with_host_illegal_action(
    tmp_path: Path,
) -> None:
    agent = TychoNooaAgent(
        llm=FakeLLMClient(),
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP"],
    )
    agent._begin_decision(turn(0), None)
    (agent.workspace_dir / "world_model.py").write_text(
        """from dataclasses import dataclass
import numpy as np
import wmlib

@dataclass
class State:
    grid: np.ndarray
    done: bool = False

def init_state(grid, level):
    return State(grid=grid.copy())

def transition(state, action):
    result = wmlib.clone(state)
    result.done = True
    return result

def render(state):
    return state.grid

def outcome(state):
    return "level_complete" if state.done else "ongoing"

def actions(state):
    return [{"action": "NOT_AVAILABLE", "row": -999, "col": 999}]
""",
        encoding="utf-8",
    )

    result = agent._tool_surface.run_python("python plan.py auto")

    assert "PLANNER_STATUS: plan_found" in result
    assert "PLANNER_PUBLICATION: rejected" in result
    assert "invalid action" in result
    assert not (agent.workspace_dir / "notes" / "validated_plan.json").exists()


@pytest.mark.asyncio
async def test_planner_output_fallback_traces_indirect_unrecognized_execution(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "tool-trace.jsonl"
    llm = FakeLLMClient(
        scripted_responses=[
            response(
                "run_python",
                {"code": "custom_runner()"},
                "opaque-plan",
            ),
            response("take_actions", {"action": "UP"}, "act-1"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP"],
        tool_trace_path=trace_path,
    )
    original_execute = agent._tool_executor.execute

    def execute_with_planner_marker(name: str, arguments: dict) -> str:
        if name == "run_python":
            return (
                "PLANNER_COMMAND: python plan.py bfs (max_nodes=20000)\n"
                "PLANNER_STATUS: no_plan_exhausted (elapsed_s=0.02)"
            )
        return original_execute(name, arguments)

    agent._tool_executor.execute = execute_with_planner_marker

    assert await agent.act(turn(0)) == ActionDecision(action="UP")
    planner_trace = [
        json.loads(line)
        for line in (tmp_path / "planner-trace.jsonl").read_text().splitlines()
    ]
    assert planner_trace[0]["command"] == "python plan.py bfs"
    assert planner_trace[0]["status"] == "no_plan_exhausted"


@pytest.mark.asyncio
async def test_verbosity_changes_the_next_observation_payload(tmp_path: Path) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            response(
                "set_verbosity",
                {"grid": "off", "diff": "summary"},
                "verbosity-1",
            ),
            response("take_actions", {"action": "UP"}, "act-1"),
            response("take_actions", {"action": "DOWN"}, "act-2"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    await agent.act(turn(0))
    await agent.act(turn(1, previous_action="UP"))

    tasks = [event for event in agent.event_manager.values() if isinstance(event, Task)]
    assert len(tasks) == 2
    assert "grid (row labels at left" in tasks[0].prompt
    assert "grid (row labels at left" not in tasks[1].prompt
    assert "1 cells changed; region [rows 0-0, cols 1-1]" in tasks[1].prompt
    assert "  0->1" not in tasks[1].prompt


@pytest.mark.asyncio
async def test_only_the_last_three_observations_keep_their_inlined_grid(
    tmp_path: Path,
) -> None:
    """Keep three exact turns and archive older events behind structural facts.

    Exact evidence remains on disk and NOOA retains every archived event by its
    stable tag. The provider view receives one bounded, non-semantic summary plus
    the three recent observations with their full grids.
    """
    llm = FakeLLMClient(
        scripted_responses=[
            response("take_actions", {"action": "UP"}, "act-1"),
            response("take_actions", {"action": "DOWN"}, "act-2"),
            response("take_actions", {"action": "UP"}, "act-3"),
            response("take_actions", {"action": "DOWN"}, "act-4"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    assert NativeToolConfig().grid_keep == 3
    assert NativeToolConfig().history_keep_turns == 3

    for number in range(4):
        await agent.act(turn(number, previous_action="UP" if number else ""))

    observations = [
        event for event in agent.event_manager.values() if isinstance(event, Task)
    ]
    assert len(observations) == 3
    assert all("grid (row labels at left" in item.prompt for item in observations)
    summaries = [
        event for event in agent.event_manager.values() if isinstance(event, Summary)
    ]
    assert len(summaries) == 1
    summary = json.loads(summaries[0].summary_text)
    assert summary["schema"] == "production_actor_history_collapse.v1"
    assert summary["collapsed_turns"] == 1
    assert "semantic" not in summaries[0].summary_text

    archived_tasks = [
        agent.event_manager[tag]
        for tag in summaries[0].children_tags
        if isinstance(agent.event_manager[tag], Task)
    ]
    assert "frame on disk = level_0/turn_000.txt" in archived_tasks[0].prompt
    assert (agent.workspace_dir / "level_0/turn_000.txt").exists()

    # What the model actually saw on the last call, not just what is stored.
    rendered_history = json.dumps(llm.last_messages)
    assert rendered_history.count("grid (row labels at left") == 3
    assert "production_actor_history_collapse.v1" in rendered_history
    assert "grid and transition elided" not in rendered_history


@pytest.mark.asyncio
async def test_the_actor_sees_its_own_prose_and_reasoning_on_the_next_turn(
    tmp_path: Path,
) -> None:
    """What the model said about a turn has to survive into the next one.

    NOOA's `ToolCallEvent` renders with `"content": None` hardcoded, so the
    `LLMOutput` event is the only carrier for the assistant's prose. Dropping it
    left the actor arguing with a history in which it had never spoken. Upstream
    Tycho keeps text and reasoning on the same assistant message it appends after
    every reply (`agent/agent.py`).
    """
    llm = MessageRecordingFakeLLMClient(
        scripted_responses=[
            reasoned_response(
                "take_actions",
                {"action": "UP"},
                "act-1",
                content="The blue cell moved with me, so it is the player.",
                reasoning="cell (0,1) tracks every UP",
            ),
            response("take_actions", {"action": "DOWN"}, "act-2"),
            response("take_actions", {"action": "UP"}, "act-3"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    await agent.act(turn(0))
    await agent.act(turn(1, previous_action="UP"))
    await agent.act(turn(2, previous_action="DOWN"))

    narrated = [
        message
        for message in llm.messages_per_call[1]
        if message.get("role") == "assistant"
        and "it is the player" in str(message.get("content") or "")
    ]
    assert len(narrated) == 1, "the actor's own prose must reach the next turn"
    assert narrated[0]["reasoning"] == "cell (0,1) tracks every UP"

    # A tool call the model did not narrate must not add an empty assistant turn.
    silent = [
        message
        for message in llm.messages_per_call[2]
        if message.get("role") == "assistant"
        and not message.get("content")
        and not message.get("tool_calls")
    ]
    assert silent == []


@pytest.mark.asyncio
async def test_the_llm_trace_records_the_counters_for_every_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every LLM round-trip leaves a durable line, text or no text.

    The other traces are written by our own code about our own actions, so a
    finished run could not answer "did the model reason at all". `LLMComplete`
    can, and it is emitted to handlers even though `record=False` keeps it out
    of the stored history (`runtime/event_manager.py::add`).
    """
    # Absence now means capture is on; a research run must explicitly fail rather
    # than silently discard the only provider-visible account of its reasoning.
    monkeypatch.delenv(CAPTURE_RESPONSE_TEXT_ENV, raising=False)
    trace_path = tmp_path / "run" / "tool-trace.jsonl"
    llm = ToolsRecordingFakeLLMClient(
        scripted_responses=[
            reasoned_response(
                "take_actions",
                {"action": "UP"},
                "act-1",
                content="The blue cell moved with me.",
                reasoning="cell (0,1) tracks every UP",
            ),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )

    await agent.act(turn(0))

    lines = (trace_path.parent / "llm-trace.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [record["call_index"] for record in records] == [1]
    assert {record["source"] for record in records} == {"actor"}
    assert {record["phase"] for record in records} == {"decision"}
    assert records[0]["tool_calls"] == ["take_actions"]
    assert records[0]["tool_call_ids"] == ["act-1"]
    tool_record = json.loads(trace_path.read_text().splitlines()[0])
    assert tool_record["tool_call_id"] == "act-1"
    assert records[0]["prompt_tokens"] == 1
    assert "reasoning_tokens" in records[0]
    assert records[0]["reasoning"] == "cell (0,1) tracks every UP"
    assert records[0]["content"] == "The blue cell moved with me."


@pytest.mark.asyncio
async def test_the_llm_trace_records_the_raw_response_when_explicitly_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit enabled value preserves both response channels.

    `LLMComplete` carries the reasoning but not `response.content`; the prose is
    on the `LLMOutput` NOOA adds immediately afterwards in the same return path.
    Both have to land on one record for the retention decision to be judgeable.
    """
    monkeypatch.setenv(CAPTURE_RESPONSE_TEXT_ENV, "1")
    trace_path = tmp_path / "run" / "tool-trace.jsonl"
    llm = ToolsRecordingFakeLLMClient(
        scripted_responses=[
            reasoned_response(
                "take_actions",
                {"action": "UP"},
                "act-1",
                content="The blue cell moved with me.",
                reasoning="cell (0,1) tracks every UP",
            ),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )

    await agent.act(turn(0))

    first = json.loads(
        (trace_path.parent / "llm-trace.jsonl").read_text().splitlines()[0]
    )
    assert first["reasoning"] == "cell (0,1) tracks every UP"
    assert first["content"] == "The blue cell moved with me."


def test_native_surface_matches_tycho_and_not_codeact(tmp_path: Path) -> None:
    agent = TychoNooaAgent(
        llm=FakeLLMClient(),
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    names = [tool.name for tool in agent._tool_surface.native_tools()]

    assert names == [
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "run_python",
        "set_verbosity",
        "take_actions",
    ]
    assert "execute_python" not in names
    assert "return_result" not in names
    assert (agent.workspace_dir / "world_model.py").is_file()
    assert (agent.workspace_dir / "verify.py").is_file()
    assert (agent.workspace_dir / "plan.py").is_file()


@pytest.mark.asyncio
async def test_provider_projection_bounds_history_but_keeps_exact_traces(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "run" / "tool-trace.jsonl"
    llm = MessageRecordingFakeLLMClient(
        scripted_responses=[
            reasoned_response(
                "run_python",
                {"code": 'print("x" * 10000)'},
                "compute-1",
                content="",
                reasoning="r" * 10_000,
            ),
            response("take_actions", {"action": "UP"}, "act-1"),
            reasoned_response(
                "run_python",
                {"code": 'print("y" * 10000)'},
                "compute-2",
                content="",
                reasoning="s" * 10_000,
            ),
            response("take_actions", {"action": "DOWN"}, "act-2"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
        strategy_config=NativeToolConfig(
            max_tool_steps=1,
            grid_keep=1,
            history_keep_turns=1,
            max_context_value_bytes=512,
        ),
    )

    await agent.act(turn(0))
    await agent.act(turn(1, previous_action="UP"))

    provider_view = json.dumps(llm.messages_per_call[-1])
    assert "production_actor_history_collapse.v1" in provider_view
    assert "assistant reasoning externalized" in provider_view
    assert "run_python result externalized" in provider_view
    assert "x" * 1_000 not in provider_view
    assert "y" * 1_000 not in provider_view

    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    python_results = [row["result"] for row in trace if row["name"] == "run_python"]
    assert "x" * 10_000 in python_results[0]
    assert "y" * 10_000 in python_results[1]
    llm_trace = [
        json.loads(line)
        for line in trace_path.with_name("llm-trace.jsonl").read_text().splitlines()
    ]
    assert llm_trace[0]["reasoning"] == "r" * 10_000
    assert llm_trace[2]["reasoning"] == "s" * 10_000


@pytest.mark.asyncio
async def test_provider_projection_bounds_large_tool_arguments(tmp_path: Path) -> None:
    trace_path = tmp_path / "run" / "tool-trace.jsonl"
    large_code = "print(1)\n#" + "x" * 16_000
    llm = MessageRecordingFakeLLMClient(
        scripted_responses=[
            response("run_python", {"code": large_code}, f"compute-{index}")
            for index in range(4)
        ]
        + [response("take_actions", {"action": "UP"}, "act-1")]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
        strategy_config=NativeToolConfig(
            max_tool_steps=4,
            max_context_value_bytes=2_048,
        ),
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")

    request_sizes = [
        len(json.dumps(messages, separators=(",", ":")).encode())
        for messages in llm.messages_per_call
    ]
    assert max(request_sizes) < 65_536
    assert "externalized_tool_arguments.v1" in json.dumps(
        llm.messages_per_call[-1]
    )
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert trace[0]["arguments"]["code"] == large_code


@pytest.mark.asyncio
async def test_provider_projection_bounds_prose_only_response_before_retry(
    tmp_path: Path,
) -> None:
    large_prose = "analysis " * 2_000
    llm = MessageRecordingFakeLLMClient(
        scripted_responses=[
            prose_response(large_prose),
            response("take_actions", {"action": "UP"}, "act-1"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        strategy_config=NativeToolConfig(max_context_value_bytes=512),
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")

    retry = json.dumps(llm.messages_per_call[1])
    assert "assistant prose externalized" in retry
    assert large_prose not in retry


@pytest.mark.asyncio
async def test_decision_discipline_keeps_one_replaceable_budget_cue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYCHO_MODE", "trigger")
    monkeypatch.setenv("TYCHO_NOOA_TREATMENT", "decision_discipline")
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")
    llm = MessageRecordingFakeLLMClient(
        scripted_responses=[
            response("ls", {"path": path}, f"look-{index}")
            for index, path in enumerate((".", "notes", "level_0", "."))
        ]
        + [response("take_actions", {"action": "UP"}, "act-1")]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        strategy_config=NativeToolConfig(max_tool_steps=4),
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")

    markers = (
        "DECISION CLOCK",
        "METACOGNITIVE CHECK",
        "COMMITMENT PRESSURE",
        "EXPLORATION CLOSED",
    )
    requests = [json.dumps(messages) for messages in llm.messages_per_call]
    assert len(requests) == 5
    assert "DECISION CLOCK" in requests[0]
    assert "METACOGNITIVE CHECK" in requests[2]
    assert "COMMITMENT PRESSURE" in requests[3]
    assert "EXPLORATION CLOSED" in requests[4]
    assert "Model calls used: 4 of 5" in requests[4]
    assert "If this response lacks one valid take_actions tool call" in requests[4]
    assert all(sum(request.count(marker) for marker in markers) == 1 for request in requests)


@pytest.mark.asyncio
async def test_decision_discipline_allows_immediate_action_without_reflection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYCHO_MODE", "trigger")
    monkeypatch.setenv("TYCHO_NOOA_TREATMENT", "decision_discipline")
    llm = MessageRecordingFakeLLMClient(
        scripted_responses=[response("take_actions", {"action": "UP"}, "act-1")]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")
    assert len(llm.messages_per_call) == 1
    assert "DECISION CLOCK" in json.dumps(llm.messages_per_call[0])


@pytest.mark.asyncio
async def test_a_spent_tool_budget_demands_the_action_instead_of_ending_the_episode(
    tmp_path: Path,
) -> None:
    """The live failure this guard exists for: exploration eats every step.

    `runs/tycho-nooa-goal-gate-ar25-20260819T120904Z` spent all 25 steps reading
        the grid, never committed an action, and lost the episode's other 19 actions
    to a ProtocolError at action 0.
    """
    llm = ToolsRecordingFakeLLMClient(
        scripted_responses=[
            response("ls", {"path": "."}, "look-1"),
            response("ls", {"path": "notes"}, "look-2"),
            response("take_actions", {"action": "UP"}, "act-1"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    agent._native_tool_config = NativeToolConfig(max_tool_steps=2)

    assert await agent.act(turn(0)) == ActionDecision(action="UP")

    demand_call_tools = llm.tools_per_call[2]
    assert demand_call_tools == ["take_actions"]
    assert len(llm.tools_per_call) == 3


@pytest.mark.asyncio
async def test_final_action_demand_can_force_the_only_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")
    llm = ToolsRecordingFakeLLMClient(
        scripted_responses=[
            response("ls", {"path": "."}, "look-1"),
            response("take_actions", {"action": "UP"}, "act-1"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        strategy_config=NativeToolConfig(
            max_tool_steps=1,
            force_action_tool=True,
        ),
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")
    assert llm.tool_choices == ["auto", "required"]


@pytest.mark.asyncio
async def test_zero_exploration_budget_is_one_required_action_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direct-action lane must be one request, not one plus a final request."""
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")
    llm = ToolsRecordingFakeLLMClient(
        scripted_responses=[
            response("take_actions", {"action": "UP"}, "act-1"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        strategy_config=NativeToolConfig(
            max_tool_steps=0,
            force_action_tool=True,
        ),
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")
    assert llm.tools_per_call == [["take_actions"]]
    assert llm.tool_choices == ["required"]


@pytest.mark.asyncio
async def test_successor_turn_uses_its_smaller_required_action_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initial cognition is ample; every later turn has one terminal request."""
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")
    llm = ToolsRecordingFakeLLMClient(
        scripted_responses=[
            response("ls", {"path": "."}, "look-1"),
            response("ls", {"path": "notes"}, "look-2"),
            response("take_actions", {"action": "UP"}, "act-1"),
            response("take_actions", {"action": "DOWN"}, "act-2"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        strategy_config=NativeToolConfig(
            max_tool_steps=2,
            max_tool_steps_after_initial=0,
            max_tokens=16_000,
            max_tokens_after_initial=4_096,
            force_action_tool=True,
        ),
    )

    initial = turn(0).model_copy(update={"sequence": 0})
    successor = turn(1, previous_action="UP").model_copy(update={"sequence": 1})
    assert await agent.act(initial) == ActionDecision(action="UP")
    assert await agent.act(successor) == ActionDecision(action="DOWN")
    assert llm.tool_choices == ["auto", "auto", "required", "required"]
    assert llm.max_tokens_per_call == [16_000, 16_000, 16_000, 4_096]
    assert llm.tools_per_call[-1] == ["take_actions"]


@pytest.mark.parametrize(
    ("action_limit", "actions_used", "actions_remaining", "final"),
    (
        (32, 30, 2, False),
        (32, 31, 1, True),
        (80, 78, 2, False),
        (80, 79, 1, True),
    ),
)
@pytest.mark.asyncio
async def test_successor_request_exposes_exact_environment_action_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action_limit: int,
    actions_used: int,
    actions_remaining: int,
    final: bool,
) -> None:
    monkeypatch.setenv("TYCHO_MODE", "single")
    monkeypatch.setenv("TYCHO_NOOA_TREATMENT", "single_decision_discipline")
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")
    llm = RequestRecordingFakeLLMClient(
        scripted_responses=[response("take_actions", {"action": "UP"}, "act")]
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        strategy_config=NativeToolConfig(
            max_tool_steps=5,
            max_tool_steps_after_initial=0,
            max_tokens=16_000,
            max_tokens_after_initial=16_000,
            force_action_tool=True,
        ),
    )
    current = GameTurn(
        grid=[[0]],
        available_actions=["UP", "DOWN"],
        sequence=actions_used,
        level=0,
        turn_in_level=actions_used,
        previous_action="UP",
        action_limit=action_limit,
        actions_used=actions_used,
        actions_remaining=actions_remaining,
        final_permitted_action=final,
    )

    assert await agent.act(current) == ActionDecision(action="UP")
    assert len(llm.requests) == 1
    request = llm.requests[0]
    assert request["tool_choice"] == "required"
    assert request["max_tokens"] == 16_000
    assert request["tools"] == ["take_actions"]
    model_view = json.dumps(request, sort_keys=True)
    assert (
        f"Action horizon: {actions_used} of {action_limit} permitted environment "
        f"actions have been used; {actions_remaining} remain including this decision."
        in model_view
    )
    terminal_notice = (
        "This is the final permitted environment action. Its result will be recorded, "
        "but no next model decision is guaranteed."
    )
    assert (terminal_notice in model_view) is final
    assert ("the successor arrives on the next call" in model_view.lower()) is False


@pytest.mark.asyncio
async def test_a_model_that_declines_the_demand_still_ends_the_episode(
    tmp_path: Path,
) -> None:
    """A spent budget is degradable; a model that will not act is not."""
    llm = FakeLLMClient(
        scripted_responses=[
            response("ls", {"path": "."}, "look-1"),
            prose_response("I would rather keep looking."),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    agent._native_tool_config = NativeToolConfig(max_tool_steps=1)

    with pytest.raises(GenerationError, match="take_actions was the only tool left"):
        await agent.act(turn(0))
