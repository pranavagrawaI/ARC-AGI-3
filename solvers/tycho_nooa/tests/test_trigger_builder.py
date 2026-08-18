from __future__ import annotations

import json
from pathlib import Path

import pytest
from nooa.context_blocks import ToolCallEvent
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tycho.agent.builder import REPORT_FILE
from tycho.agent.colors import COLORMAP
from tycho.prompts.render import render_prompt

from tycho_nooa import ActionDecision, GameTurn, TychoNooaAgent
from tycho_nooa.builder import FALLBACK_REPORT
from tycho_nooa.trigger_builder import TriggerDecision


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


@pytest.mark.asyncio
async def test_trigger_builds_fresh_report_before_actor_and_traces_uptake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = (
        "confidence: medium\n"
        "model: UP changes the marker\n"
        "dynamics: one observed transition\n"
        "outcome: unknown\n"
        "outcome_verified: no-terminal-yet\n"
        "plan: start=current L0 turn 1; validated=no; length=none; assumptions=unknown\n"
        "recommended_action: UP   subgoal: test the encoded transition\n"
        "note: read this report, then test UP once"
    )
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("actor checkpoint after turn 0"),
            tool_response(
                "write_file",
                {
                    "path": "notes/world_model.md",
                    "content": "Observed: UP changes the marker.",
                },
                "builder-notes",
            ),
            tool_response(
                "write_file",
                {"path": "notes/world_model_report.md", "content": report},
                "builder-report",
            ),
            prose_response("done"),
            tool_response(
                "read_file",
                {"path": "notes/world_model_report.md"},
                "actor-read-report",
            ),
            tool_response("take_action", {"action": "UP"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True,
            cause="dynamics",
            reason="exact trigger reason",
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    trace_path = tmp_path / "tool-trace.jsonl"
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(action="UP")

    assert (agent.workspace_dir / "notes/world_model.md").read_text() == (
        "Observed: UP changes the marker."
    )
    assert (agent.workspace_dir / "notes/world_model_report.md").read_text() == report

    builder_tools = agent._trigger_builder.builder._tool_surface.native_tools()
    assert "take_action" not in {tool.name for tool in builder_tools}
    assert "edit_function" in {tool.name for tool in builder_tools}
    assert agent._trigger_builder.builder.agent_id != agent.agent_id
    assert agent._trigger_builder.builder._resolve_system_prompt() == render_prompt(
        "builder.system",
        colormap=COLORMAP,
        report_file=REPORT_FILE,
        meta_reflect=False,
    )
    builder_events = agent._trigger_builder.builder.event_manager.values()
    builder_tasks = [event for event in builder_events if isinstance(event, Task)]
    assert builder_tasks[0].prompt == render_prompt(
        "builder.user",
        level=0,
        hint="exact trigger reason",
        available_actions=["UP", "DOWN"],
        report_file=REPORT_FILE,
    )
    assert builder_tasks[1].prompt == "[verify state] test divergence"
    assert [
        event.name for event in builder_events if isinstance(event, ToolCallEvent)
    ] == ["write_file", "write_file"]
    assert [
        event.name
        for event in agent.event_manager.values()
        if isinstance(event, ToolCallEvent)
    ] == ["take_action", "read_file", "take_action"]
    actor_tasks = [
        event for event in agent.event_manager.values() if isinstance(event, Task)
    ]
    # The last task is turn 1's own actor_beliefs.md checkpoint prompt, inserted
    # after the decision; the turn's observation (carrying the report) is -2.
    assert report in actor_tasks[-2].prompt

    builder_trace = [
        json.loads(line)
        for line in (tmp_path / "builder-tool-trace.jsonl").read_text().splitlines()
    ]
    assert [entry["name"] for entry in builder_trace] == [
        "write_file",
        "write_file",
    ]
    trigger_trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert [entry["event"] for entry in trigger_trace] == [
        "mechanical_trigger",
        "builder_result",
        "actor_uptake",
    ]
    assert trigger_trace[0]["fired"] is True
    assert trigger_trace[1]["report_fresh"] is True
    assert trigger_trace[1]["artifact_hashes"]["notes/world_model.md"]["changed"]
    assert trigger_trace[1]["artifact_hashes"]["notes/world_model_report.md"]["changed"]
    assert trigger_trace[2]["followed_recommended_action"] is True
    assert trigger_trace[2]["tested_report"] is True


@pytest.mark.asyncio
async def test_stale_or_missing_report_is_replaced_by_tool_free_final_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = (
        "confidence: low\nmodel: none — insufficient evidence\n"
        "dynamics: n/a\noutcome: unknown\noutcome_verified: no-terminal-yet\n"
        "plan: start=current; validated=no; length=none; assumptions=unknown\n"
        "recommended_action: none   subgoal: gather one more transition\n"
        "note: test DOWN"
    )
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("actor checkpoint after turn 0"),
            prose_response("done"),
            prose_response("nothing new yet"),
            prose_response(fallback),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True,
            cause="dynamics",
            reason="exact trigger reason",
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    stale_path = agent.workspace_dir / "notes/world_model_report.md"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("confidence: high\nnote: stale prior report")

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    assert (agent.workspace_dir / "notes/world_model_report.md").read_text() == fallback
    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace[1]["report_fresh"] is True
    assert trace[1]["report_forced"] is True
    assert trace[1]["world_model_checkpoint_forced"] is True
    assert trace[1]["artifact_hashes"]["notes/world_model.md"]["changed"] is True


@pytest.mark.asyncio
async def test_forced_report_identical_to_prior_report_is_not_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for DIAGNOSIS.md: turns 11/19 forced-reported `report_fresh=true`
    while the report bytes were byte-identical to the prior pass. Forcing a final
    report is a claim about which code path ran, not about whether anything new
    was learned; only an actual content change should count as fresh."""
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("actor checkpoint after turn 0"),
            prose_response("done"),
            prose_response("nothing new yet"),
            prose_response("done"),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True,
            cause="dynamics",
            reason="exact trigger reason",
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    report_path = agent.workspace_dir / "notes/world_model_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("done")

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    assert report_path.read_text() == "done"
    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace[1]["report_forced"] is True
    assert trace[1]["report_fresh"] is False
    assert trace[1]["artifact_hashes"]["notes/world_model_report.md"]["changed"] is False


@pytest.mark.asyncio
async def test_forced_report_call_offers_tools_but_ignores_attempted_tool_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matches Tycho's builder.py forced-report fallback: the final generate
    call still offers tools (it is not hard-blocked via tools=[]/tool_choice=
    "none"), but Tycho's handling of that reply only reads reply["text"] and
    never inspects reply["tool_calls"] — a tool-call attempt there is silently
    unexecuted, not structurally disallowed. If the model tries to call a tool
    anyway instead of replying in prose, the forced pass must fall back to the
    fixed report template rather than executing that tool call."""
    llm = RecordingFakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("actor checkpoint after turn 0"),
            prose_response("still exploring"),
            prose_response("nothing new yet"),
            tool_response(
                "write_file",
                {"path": "notes/world_model_report.md", "content": "sneaky"},
                "forced-attempt",
            ),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True,
            cause="dynamics",
            reason="exact trigger reason",
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    # Call order: actor turn 0, actor's own actor_beliefs.md checkpoint (filler),
    # builder main loop (breaks on the prose reply), builder world-model
    # checkpoint enforcement, builder forced final call, actor turn 1.
    checkpoint_call_tools = llm.tools_per_call[3]
    assert checkpoint_call_tools, "checkpoint call must offer tools too"
    assert "take_action" not in checkpoint_call_tools

    forced_call_tools = llm.tools_per_call[4]
    assert forced_call_tools, "forced call must still offer tools, like Tycho does"
    assert "take_action" not in forced_call_tools

    report_path = agent.workspace_dir / "notes/world_model_report.md"
    assert report_path.read_text() == FALLBACK_REPORT

    # The checkpoint step offered no tool calls, so the fallback checkpoint text
    # (not the "sneaky" write) is what advances notes/world_model.md. The forced
    # final call's tool-call attempt is never executed regardless.
    world_model_notes = agent.workspace_dir / "notes/world_model.md"
    assert world_model_notes.read_text() != "sneaky"

    builder_trace_path = tmp_path / "builder-tool-trace.jsonl"
    if builder_trace_path.exists():
        assert builder_trace_path.read_text().strip() == ""

    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace[1]["report_forced"] is True
    assert trace[1]["world_model_checkpoint_forced"] is True


@pytest.mark.asyncio
async def test_world_model_checkpoint_is_forced_when_pass_leaves_it_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for DIAGNOSIS.md: notes/world_model.md was read on all 12
    post-turn-8 builder firings in the first live run but never updated, so every
    later pass repeated the same stale turn-1 scratchpad. A pass that leaves this
    file untouched must not be allowed to end quietly."""
    fallback_report = (
        "confidence: low\nmodel: none — insufficient evidence\n"
        "dynamics: n/a\noutcome: unknown\noutcome_verified: no-terminal-yet\n"
        "plan: start=current; validated=no; length=none; assumptions=unknown\n"
        "recommended_action: none   subgoal: gather one more transition\n"
        "note: test DOWN"
    )
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("actor checkpoint after turn 0"),
            prose_response("done"),
            prose_response("nothing new yet"),
            prose_response(fallback_report),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True, cause="dynamics", reason="exact trigger reason"
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    notes_path = agent.workspace_dir / "notes/world_model.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("turn 1: UP moved the marker\n")

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    updated = notes_path.read_text()
    assert updated.startswith("turn 1: UP moved the marker")
    assert "[checkpoint: pass ended with no model-authored update]" in updated

    builder_tasks = [
        event
        for event in agent._trigger_builder.builder.event_manager.values()
        if isinstance(event, Task)
    ]
    assert any(
        "Before this pass ends, checkpoint your current understanding" in task.prompt
        for task in builder_tasks
    )

    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace[1]["world_model_checkpoint_forced"] is True
    assert trace[1]["artifact_hashes"]["notes/world_model.md"]["changed"] is True


@pytest.mark.asyncio
async def test_world_model_checkpoint_executes_a_genuine_model_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlike the forced-report call, the checkpoint call's tool calls are real:
    the model gets one genuine chance to write before the fallback breadcrumb
    would apply, and that write must actually land."""
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("actor checkpoint after turn 0"),
            prose_response("done"),
            tool_response(
                "write_file",
                {
                    "path": "notes/world_model.md",
                    "content": "genuine checkpoint: UP moves the marker up",
                },
                "checkpoint-write",
            ),
            prose_response("done"),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True, cause="dynamics", reason="exact trigger reason"
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    notes_path = agent.workspace_dir / "notes/world_model.md"
    content = notes_path.read_text()
    assert content == "genuine checkpoint: UP moves the marker up"
    assert "[checkpoint: pass ended" not in content

    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace[1]["world_model_checkpoint_forced"] is True


@pytest.mark.asyncio
async def test_level_start_fires_only_on_genuine_level_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the same-level `full_reset` ambiguity: arc-host resets
    turn_in_level to 0 WITHOUT changing level (runner.py), so turn_in_level ==
    0 alone cannot distinguish a genuine level advance from a same-level
    restart. Only a level that actually differs from the immediately
    preceding turn's level (`_last_level`) may fire the unconditional
    LEVEL_START cause."""
    calls: list[str] = []
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate_level_start",
        lambda self, level: calls.append(f"level_start:{level}")
        or TriggerDecision(fired=False, cause="level_start", reason="entered level"),
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: calls.append(f"evaluate:{level}")
        or TriggerDecision(fired=False, cause=None, reason=None),
    )
    llm = FakeLLMClient(scripted_responses=[])
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    loop = agent._trigger_builder
    assert loop is not None

    def level_turn(level: int, turn_in_level: int) -> GameTurn:
        return GameTurn(
            grid=[[0, 1], [2, 3]],
            available_actions=["UP", "DOWN"],
            level=level,
            turn_in_level=turn_in_level,
            previous_action="UP" if turn_in_level > 0 else "",
        )

    # Level 0's own turn_in_level == 0 is guarded out unconditionally (level > 0
    # is required), so it neither fires nor consumes a scripted response.
    assert await loop.before_actor(level_turn(0, 0)) is None
    # A genuine advance into level 1 fires level_start exactly once.
    assert await loop.before_actor(level_turn(1, 0)) is None
    # arc-host's full_reset recurs turn_in_level == 0 at the SAME level — this
    # must not be mistaken for a second level-start.
    assert await loop.before_actor(level_turn(1, 0)) is None

    assert calls == ["level_start:1"]

    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert [entry["mechanism"] for entry in trace] == [
        "tycho TriggerDispatcher.level_start_reason (unconditional)"
    ]
    assert trace[0]["level"] == 1
    assert trace[0]["turn_in_level"] == 0


@pytest.mark.asyncio
async def test_unavailable_sandbox_degrades_the_trigger_instead_of_ending_the_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The divergence probe runs verify.py in Tycho's sandbox. When no container
    runtime exists it raises on the actor's path, where the same fault is only a
    tool result — that killed a whole live run at action 1. An unanswerable probe
    must be traced as absence of signal and let the actor's turn proceed."""

    def unavailable(self, level, executor):
        raise RuntimeError("no usable container runtime found")

    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate", unavailable
    )
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("actor checkpoint after turn 0"),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
            prose_response("actor checkpoint after turn 1"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert [entry["event"] for entry in trace] == ["trigger_error"]
    assert trace[0]["error_type"] == "RuntimeError"
    assert "no usable container runtime" in trace[0]["message"]
    assert trace[0]["turn_in_level"] == 1
    assert not (agent.workspace_dir / "notes/world_model_report.md").exists()


@pytest.mark.asyncio
async def test_clean_signal_does_not_dispatch_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("actor checkpoint after turn 0"),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=False,
            cause=None,
            reason=None,
        ),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    assert not (agent.workspace_dir / "notes/world_model_report.md").exists()
    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace == [
        {
            "event": "mechanical_trigger",
            "mechanism": "tycho TriggerDispatcher.should_fire -> wm_signal",
            "level": 0,
            "turn_in_level": 1,
            "fired": False,
            "cause": None,
            "reason": None,
        }
    ]
