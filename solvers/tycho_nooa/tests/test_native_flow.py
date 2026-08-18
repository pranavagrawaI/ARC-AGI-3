from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from nooa.context_blocks import ToolCallEvent
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tycho.agent.colors import COLORMAP
from tycho.prompts.render import render_prompt

from tycho_nooa import ActionDecision, GameTurn, TychoNooaAgent


def response(tool: str, arguments: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=tool, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def batched_response(calls: list[tuple[str, dict, str]]) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(id=call_id, name=tool, arguments=json.dumps(arguments))
            for tool, arguments, call_id in calls
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def turn(number: int, previous_action: str = "") -> GameTurn:
    return GameTurn(
        grid=[[0, number], [2, 3]],
        available_actions=["UP", "DOWN"],
        level=0,
        turn_in_level=number,
        previous_action=previous_action,
    )


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
            response("take_action", {"action": "UP"}, "act-1"),
            response("read_file", {"path": "notes/world_model.md"}, "read-2"),
            response("take_action", {"action": "DOWN"}, "act-2"),
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
        "take_action",
        "read_file",
        "take_action",
    ]
    assert calls[2].result is not None
    assert calls[2].result.content == "UP may move the blue cell."
    rendered_history = json.dumps(llm.last_messages)
    assert "UP may move the blue cell." in rendered_history
    assert "turn_in_level': 1" in rendered_history
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [record["name"] for record in trace] == [
        "write_file",
        "take_action",
        "read_file",
        "take_action",
    ]
    assert [record["turn_in_level"] for record in trace] == [0, 0, 1, 1]
    assert trace[0]["arguments"]["content"] == "UP may move the blue cell."
    assert trace[2]["result"] == "UP may move the blue cell."


@pytest.mark.asyncio
async def test_illegal_action_is_feedback_not_a_host_action(tmp_path: Path) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            response("take_action", {"action": "LEFT"}, "bad-action"),
            response("take_action", {"action": "UP"}, "good-action"),
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
async def test_first_model_context_omits_internal_self_and_protocol_docs(
    tmp_path: Path,
) -> None:
    llm = FakeLLMClient(
        scripted_responses=[response("take_action", {"action": "UP"}, "act-1")]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )

    await agent.act(turn(0))

    assert [message["role"] for message in llm.last_messages] == ["system", "user"]
    rendered = json.dumps(llm.last_messages)
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
        "The builder fires when the model is detectably wrong on dynamics or outcome.",
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
        for message in llm.last_messages
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
    assert rendered.count("world_model.py") == 1


@pytest.mark.asyncio
async def test_validated_plan_hint_is_injected_and_traced(tmp_path: Path) -> None:
    trace_path = tmp_path / "tool-trace.jsonl"
    llm = FakeLLMClient(
        scripted_responses=[
            response("take_action", {"action": "UP"}, "act-1"),
            response("take_action", {"action": "DOWN"}, "act-2"),
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
                "take_action",
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
                    ("take_action", {"action": "UP"}, "blocked-act"),
                ]
            ),
            response("take_action", {"action": "UP"}, "act-2"),
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

    def execute_with_plan(name: str, arguments: dict) -> str:
        if name != "run_python":
            return original_execute(name, arguments)
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

    agent._tool_executor.execute = execute_with_plan

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
                    ("take_action", {"action": "UP"}, "blocked-act"),
                ]
            ),
            response("take_action", {"action": "UP"}, "act-2"),
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
            response("take_action", {"action": "UP"}, "act-1"),
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
            response("take_action", {"action": "UP"}, "act-1"),
            response("take_action", {"action": "DOWN"}, "act-2"),
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
        "take_action",
    ]
    assert "execute_python" not in names
    assert "return_result" not in names
    assert (agent.workspace_dir / "world_model.py").is_file()
    assert (agent.workspace_dir / "verify.py").is_file()
    assert (agent.workspace_dir / "plan.py").is_file()
