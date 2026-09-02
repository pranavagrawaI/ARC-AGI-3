from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from tycho_nooa.llm import SCRIPTED_TOOL_CALLS_ENV
from tycho_nooa.models import ActionDecision
from tycho_nooa.protocol import (
    HOST_PREFIX,
    PROTOCOL_VERSION,
    WORKER_PREFIX,
    ActionResultMessage,
    FinishMessage,
    ObservationPayload,
    StartGameMessage,
    TurnMessage,
    parse_host_message,
)
from tycho_nooa.worker import WorkerSession, serve


def observation(
    value: int,
    *,
    actions: list[str] | None = None,
    levels: int = 0,
    full_reset: bool = False,
) -> ObservationPayload:
    return ObservationPayload(
        decision_frame=[[value, value + 1], [value + 2, value + 3]],
        terminal_frame=None,
        event_frames=[],
        provenance={
            "raw_frame_count": 1,
            "decision_frame_index": 0,
            "terminal_frame_index": None,
            "event_frame_indices": [],
            "terminal_reason": None,
        },
        available_actions=actions if actions is not None else ["UP", "DOWN"],
        levels_completed=levels,
        state="NOT_FINISHED",
        full_reset=full_reset,
    )


def test_protocol_rejects_unknown_fields() -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "close",
        "unearned_authority": True,
    }

    with pytest.raises(ValidationError, match="unearned_authority"):
        parse_host_message(json.dumps(payload))


def test_protocol_preserves_the_complete_frame_evidence_envelope() -> None:
    evidence = ObservationPayload(
        decision_frame=[[3]],
        terminal_frame=[[1]],
        event_frames=[{"kind": "animation", "source_index": 1, "frame": [[2]]}],
        provenance={
            "raw_frame_count": 3,
            "decision_frame_index": 2,
            "terminal_frame_index": 0,
            "event_frame_indices": [1],
            "terminal_reason": "level_advanced",
        },
        available_actions=["UP"],
        levels_completed=1,
        state="NOT_FINISHED",
        full_reset=False,
    )

    assert evidence.grid == [[3]]
    assert evidence.terminal_frame == [[1]]
    assert evidence.event_frames[0].frame == [[2]]
    assert evidence.provenance.terminal_reason == "level_advanced"


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"actions_remaining": None}, "supplied together"),
        ({"actions_used": 30}, "equal the turn sequence"),
        ({"actions_remaining": 2}, "action_limit - actions_used"),
        ({"final_permitted_action": False}, "actions_remaining == 1"),
        ({"action_limit": True}, "valid integer"),
    ),
)
def test_protocol_rejects_an_inexact_action_horizon(
    change: dict[str, object],
    message: str,
) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "turn",
        "sequence": 31,
        "action_limit": 32,
        "actions_used": 31,
        "actions_remaining": 1,
        "final_permitted_action": True,
        "level": 0,
        "turn_in_level": 31,
        "decision_timeout_seconds": 600,
        "observation": observation(0),
        "previous_decision": None,
        **change,
    }

    with pytest.raises(ValidationError, match=message):
        TurnMessage(**payload)  # type: ignore[arg-type]


def test_horizonless_turn_keeps_the_existing_protocol_v3_wire_shape() -> None:
    message = TurnMessage(
        protocol_version=PROTOCOL_VERSION,
        type="turn",
        sequence=0,
        level=0,
        turn_in_level=0,
        decision_timeout_seconds=600,
        observation=observation(0),
        previous_decision=None,
    )

    payload = message.model_dump(mode="json")
    assert "action_limit" not in payload
    assert "actions_used" not in payload
    assert "actions_remaining" not in payload
    assert "final_permitted_action" not in payload


@pytest.mark.asyncio
async def test_worker_passes_the_host_decision_timeout_to_the_agent() -> None:
    act = AsyncMock(return_value=ActionDecision(action="UP"))
    session = WorkerSession()
    session.agent = SimpleNamespace(act=act)  # type: ignore[assignment]
    first = observation(0)
    session.initial_observation = first

    await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision_timeout_seconds=600,
            observation=first,
            previous_decision=None,
        )
    )

    assert act.await_args.kwargs == {"decision_timeout_seconds": 600.0}


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
async def test_worker_passes_the_exact_action_horizon_to_the_agent(
    action_limit: int,
    actions_used: int,
    actions_remaining: int,
    final: bool,
) -> None:
    act = AsyncMock(return_value=ActionDecision(action="UP"))
    session = WorkerSession()
    session.agent = SimpleNamespace(act=act)  # type: ignore[assignment]
    first = observation(0)
    session.initial_observation = first
    session.next_sequence = actions_used

    await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=actions_used,
            action_limit=action_limit,
            actions_used=actions_used,
            actions_remaining=actions_remaining,
            final_permitted_action=final,
            level=0,
            turn_in_level=actions_used,
            decision_timeout_seconds=600,
            observation=first,
            previous_decision=None,
        )
    )

    turn = act.await_args.args[0]
    assert turn.action_limit == action_limit
    assert turn.actions_used == actions_used
    assert turn.actions_remaining == actions_remaining
    assert turn.final_permitted_action is final


@pytest.mark.asyncio
async def test_worker_rejects_a_first_turn_that_substitutes_the_initial_observation() -> None:
    act = AsyncMock(return_value=ActionDecision(action="UP"))
    session = WorkerSession()
    session.agent = SimpleNamespace(act=act)  # type: ignore[assignment]
    session.initial_observation = observation(0)

    with pytest.raises(RuntimeError, match="initial observation"):
        await session.decide(
            TurnMessage(
                protocol_version=PROTOCOL_VERSION,
                type="turn",
                sequence=0,
                level=0,
                turn_in_level=0,
                decision_timeout_seconds=600,
                observation=observation(8),
                previous_decision=None,
            )
        )

    act.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_rejects_action_result_from_a_different_turn() -> None:
    agent = SimpleNamespace(
        act=AsyncMock(return_value=ActionDecision(action="UP")),
        observe=AsyncMock(return_value=None),
    )
    session = WorkerSession()
    session.agent = agent  # type: ignore[assignment]
    before = observation(0)
    session.initial_observation = before

    await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision_timeout_seconds=600,
            observation=before,
            previous_decision=None,
        )
    )

    with pytest.raises(RuntimeError, match="pending turn"):
        await session.observe(
            ActionResultMessage(
                protocol_version=PROTOCOL_VERSION,
                type="action_result",
                sequence=0,
                level=0,
                turn_in_level=0,
                decision=ActionDecision(action="UP"),
                before=observation(8),
                after=observation(12),
            )
        )

    agent.observe.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_rejects_a_next_turn_that_substitutes_the_successor() -> None:
    decision = ActionDecision(action="UP")
    agent = SimpleNamespace(
        act=AsyncMock(return_value=decision),
        observe=AsyncMock(return_value=None),
    )
    session = WorkerSession()
    session.agent = agent  # type: ignore[assignment]
    before = observation(0)
    after = observation(4)
    session.initial_observation = before

    await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision_timeout_seconds=600,
            observation=before,
            previous_decision=None,
        )
    )
    await session.observe(
        ActionResultMessage(
            protocol_version=PROTOCOL_VERSION,
            type="action_result",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision=decision,
            before=before,
            after=after,
        )
    )

    with pytest.raises(RuntimeError, match="previous action result"):
        await session.decide(
            TurnMessage(
                protocol_version=PROTOCOL_VERSION,
                type="turn",
                sequence=1,
                level=0,
                turn_in_level=1,
                decision_timeout_seconds=600,
                observation=observation(8),
                previous_decision=decision,
            )
        )

    assert agent.act.await_count == 1


@pytest.mark.parametrize(
    ("levels", "full_reset"),
    [(1, False), (0, True)],
)
@pytest.mark.asyncio
async def test_worker_accepts_a_cleared_previous_decision_at_a_boundary(
    levels: int,
    full_reset: bool,
) -> None:
    decision = ActionDecision(action="UP")
    agent = SimpleNamespace(
        act=AsyncMock(return_value=decision),
        observe=AsyncMock(return_value=None),
    )
    session = WorkerSession()
    session.agent = agent  # type: ignore[assignment]
    before = observation(0)
    after = observation(4, levels=levels, full_reset=full_reset)
    session.initial_observation = before

    await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision_timeout_seconds=600,
            observation=before,
            previous_decision=None,
        )
    )
    await session.observe(
        ActionResultMessage(
            protocol_version=PROTOCOL_VERSION,
            type="action_result",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision=decision,
            before=before,
            after=after,
        )
    )

    await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=1,
            level=levels,
            turn_in_level=0,
            decision_timeout_seconds=600,
            observation=after,
            previous_decision=None,
        )
    )

    assert agent.act.await_count == 2


@pytest.mark.asyncio
async def test_single_decision_discipline_crosses_a_level_without_scribe_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Kaggle treatment keeps boundary evidence without a hidden model lane."""
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")
    monkeypatch.setenv("TYCHO_MODE", "single")
    monkeypatch.setenv("TYCHO_NOOA_TREATMENT", "single_decision_discipline")
    monkeypatch.setenv("ARC_LLM_MAX_ACTOR_CALLS", "1")
    monkeypatch.setenv("ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL", "1")
    monkeypatch.setenv("ARC_LLM_MAX_TOKENS", "16000")
    monkeypatch.setenv("ARC_LLM_MAX_TOKENS_AFTER_INITIAL", "4096")
    monkeypatch.setenv("ARC_LLM_FORCE_ACTION_TOOL", "1")
    monkeypatch.setenv(
        SCRIPTED_TOOL_CALLS_ENV,
        json.dumps(
            [
                {
                    "tool": "take_actions",
                    "arguments": {
                        "actions": [{"action": "UP"}],
                        "rationale": "test one-action commitment",
                    },
                },
                {
                    "tool": "take_actions",
                    "arguments": {
                        "actions": [{"action": "DOWN"}],
                        "rationale": "test one-action commitment",
                    },
                },
            ]
        ),
    )
    workspace_root = tmp_path / "workspaces"
    before = observation(0)
    after = ObservationPayload(
        decision_frame=[[4, 5], [6, 7]],
        terminal_frame=[[9, 9], [9, 9]],
        event_frames=[],
        provenance={
            "raw_frame_count": 2,
            "decision_frame_index": 1,
            "terminal_frame_index": 0,
            "event_frame_indices": [],
            "terminal_reason": "level_advanced",
        },
        available_actions=["UP", "DOWN"],
        levels_completed=1,
        state="NOT_FINISHED",
        full_reset=False,
    )
    session = WorkerSession()
    ready = session.start_game(
        StartGameMessage(
            protocol_version=PROTOCOL_VERSION,
            type="start_game",
            game_id="fake-game",
            seed=0,
            workspace_root=str(workspace_root),
            initial_observation=before,
        )
    )

    first = await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision_timeout_seconds=600,
            observation=before,
            previous_decision=None,
        )
    )
    first_decision = ActionDecision.model_validate(first["decision"])
    await session.observe(
        ActionResultMessage(
            protocol_version=PROTOCOL_VERSION,
            type="action_result",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision=first_decision,
            before=before,
            after=after,
        )
    )
    second = await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=1,
            level=1,
            turn_in_level=0,
            decision_timeout_seconds=600,
            observation=after,
            previous_decision=None,
        )
    )

    assert first_decision == ActionDecision(action="UP")
    assert ready["metadata"]["level_boundary_policy"] == (
        "deterministic_level_checkpoint"
    )
    assert ActionDecision.model_validate(second["decision"]) == ActionDecision(
        action="DOWN"
    )
    agent = session.agent
    assert agent is not None
    assert json.loads(
        (agent.workspace_dir / "level_0/level_complete.json").read_text()
    )["outcome"] == "level_complete"
    assert json.loads(
        (agent.workspace_dir / "level_0/solved.json").read_text()
    ) == {"action": "UP", "row": None, "col": None}
    assert json.loads(
        (agent.workspace_dir / "level_0/terminal.json").read_text()
    )["terminal_grid"] == [[9, 9], [9, 9]]
    insights = (agent.workspace_dir / "notes/level_0_insights.md").read_text()
    assert "single_decision_discipline" in insights
    assert "model-backed Scribe was not invoked" in insights
    assert json.loads(
        (agent.workspace_dir / "level_0/level_boundary_checkpoint.json").read_text()
    ) == {
        "schema_version": 1,
        "source": "tycho_nooa host treatment",
        "treatment": "single_decision_discipline",
        "policy": "deterministic_level_checkpoint",
        "model_scribe_invoked": False,
        "level": 0,
        "carry_forward_note": "notes/level_0_insights.md",
    }

    llm_trace = [
        json.loads(line)
        for line in (tmp_path / "llm-trace.jsonl").read_text().splitlines()
    ]
    assert [row["source"] for row in llm_trace] == ["actor", "actor"]
    assert [row["phase"] for row in llm_trace] == ["decision", "decision"]
    assert [row["sequence"] for row in llm_trace] == [0, 1]
    checkpoints = sorted((tmp_path / "decision-checkpoints/calls").iterdir())
    assert [path.name for path in checkpoints] == ["000001", "000002"]
    requests = [
        json.loads((path / "request.json").read_text()) for path in checkpoints
    ]
    assert [request["source"] for request in requests] == ["actor", "actor"]
    assert [request["decision_point"] for request in requests] == [
        {"sequence": 0, "level": 0, "turn_in_level": 0, "phase": "decision"},
        {"sequence": 1, "level": 1, "turn_in_level": 0, "phase": "decision"},
    ]
    assert [
        request["request_body"]["parameters"]["max_tokens"]
        for request in requests
    ] == [16_000, 4_096]
    assert [
        request["request_body"]["parameters"]["tool_choice"]
        for request in requests
    ] == ["required", "required"]
    assert [
        [tool["name"] for tool in request["request_body"]["tools"]]
        for request in requests
    ] == [["take_actions"], ["take_actions"]]


@pytest.mark.asyncio
async def test_worker_requires_the_previous_decision_without_a_boundary() -> None:
    decision = ActionDecision(action="UP")
    agent = SimpleNamespace(
        act=AsyncMock(return_value=decision),
        observe=AsyncMock(return_value=None),
    )
    session = WorkerSession()
    session.agent = agent  # type: ignore[assignment]
    before = observation(0)
    after = observation(4)
    session.initial_observation = before

    await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision_timeout_seconds=600,
            observation=before,
            previous_decision=None,
        )
    )
    await session.observe(
        ActionResultMessage(
            protocol_version=PROTOCOL_VERSION,
            type="action_result",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision=decision,
            before=before,
            after=after,
        )
    )

    with pytest.raises(RuntimeError, match="previous action result"):
        await session.decide(
            TurnMessage(
                protocol_version=PROTOCOL_VERSION,
                type="turn",
                sequence=1,
                level=0,
                turn_in_level=1,
                decision_timeout_seconds=600,
                observation=after,
                previous_decision=None,
            )
        )

    assert agent.act.await_count == 1


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        ([], "non-empty"),
        ([[0]] * 65, "64x64"),
        ([[0, 1], [2]], "rectangular"),
        ([[True]], "integers in"),
    ],
)
def test_observation_rejects_invalid_grids(grid: list[list[int]], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ObservationPayload(
            decision_frame=grid,
            terminal_frame=None,
            event_frames=[],
            provenance={
                "raw_frame_count": 1,
                "decision_frame_index": 0,
                "terminal_frame_index": None,
                "event_frame_indices": [],
                "terminal_reason": None,
            },
            available_actions=[],
            levels_completed=0,
            state="NOT_FINISHED",
            full_reset=False,
        )


@pytest.mark.asyncio
async def test_worker_keeps_one_agent_and_observes_level_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")
    monkeypatch.setenv(
        SCRIPTED_TOOL_CALLS_ENV,
        json.dumps(
            [
                {
                    "tool": "write_file",
                    "arguments": {
                        "path": "notes/first-contact.md",
                        "content": "UP changed the public grid.",
                    },
                },
                {
                    "tool": "take_actions",
                    "arguments": {
                        "actions": [{"action": "UP"}],
                        "rationale": "test one-action commitment",
                    },
                },
            ]
        ),
    )
    monkeypatch.setenv("ARC_LLM_MAX_TOKENS", "8000")
    monkeypatch.setenv("ARC_LLM_MAX_TOKENS_AFTER_INITIAL", "4096")
    monkeypatch.setenv("ARC_LLM_MAX_ACTOR_CALLS", "4")
    monkeypatch.setenv("ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL", "1")
    monkeypatch.setenv("ARC_LLM_FORCE_ACTION_TOOL", "1")
    monkeypatch.setenv("TYCHO_NOOA_TREATMENT", "decision_discipline")
    session = WorkerSession()
    ready = session.start_game(
        StartGameMessage(
            protocol_version=PROTOCOL_VERSION,
            type="start_game",
            game_id="fake-game",
            seed=0,
            workspace_root=str(tmp_path),
            initial_observation=observation(0),
        )
    )
    agent = session.agent
    assert agent is not None
    assert agent._trigger_builder is not None
    completion_checkpoint = AsyncMock(return_value=None)
    agent._trigger_builder.after_level_complete = completion_checkpoint
    after = ObservationPayload(
        decision_frame=[[4, 5], [6, 7]],
        terminal_frame=[[9, 9], [9, 9]],
        event_frames=[
            {
                "kind": "animation",
                "source_index": 1,
                "frame": [[8, 8], [8, 8]],
            }
        ],
        provenance={
            "raw_frame_count": 3,
            "decision_frame_index": 2,
            "terminal_frame_index": 0,
            "event_frame_indices": [1],
            "terminal_reason": "level_advanced",
        },
        available_actions=[],
        levels_completed=1,
        state="NOT_FINISHED",
        full_reset=False,
    )

    response = await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision_timeout_seconds=600,
            observation=observation(0),
            previous_decision=None,
        )
    )
    decision = response["decision"]
    await session.observe(
        ActionResultMessage(
            protocol_version=PROTOCOL_VERSION,
            type="action_result",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision=decision,
            before=observation(0),
            after=after,
        )
    )
    completion_checkpoint.assert_awaited_once()
    checkpoint_turn, checkpoint_evidence = completion_checkpoint.await_args.args
    assert checkpoint_turn.level == 0
    assert checkpoint_turn.turn_in_level == 1
    assert checkpoint_turn.levels_completed == 1
    assert '"outcome":"level_complete"' in checkpoint_evidence
    finish = FinishMessage(
        protocol_version=PROTOCOL_VERSION,
        type="finish",
        stop_reason="level_limit",
        actions=1,
        levels_completed=1,
        final_observation=after,
    )
    with pytest.raises(RuntimeError, match="last accepted successor"):
        session.finish(
            finish.model_copy(
                update={"final_observation": observation(12, actions=[], levels=1)}
            )
        )
    with pytest.raises(RuntimeError, match="level count"):
        session.finish(finish.model_copy(update={"levels_completed": 0}))
    session.finish(finish)
    with pytest.raises(RuntimeError, match="already finished"):
        session.finish(finish)
    with pytest.raises(RuntimeError, match="already finished"):
        await session.decide(
            TurnMessage(
                protocol_version=PROTOCOL_VERSION,
                type="turn",
                sequence=1,
                level=1,
                turn_in_level=0,
                decision_timeout_seconds=600,
                observation=after,
                previous_decision=decision,
            )
        )

    assert session.agent is agent
    assert agent._native_tool_config.max_tokens == 8_000
    assert agent._native_tool_config.max_tokens_after_initial == 4_096
    assert agent._native_tool_config.action_demand_max_tokens is None
    assert agent._native_tool_config.action_demand_disable_thinking is False
    # NativeToolStrategy reserves its last request for the action-only demand.
    assert agent._native_tool_config.max_tool_steps == 3
    assert agent._native_tool_config.max_tool_steps_after_initial == 0
    assert agent._scribe_config.max_tokens == 8_000
    assert agent._trigger_builder is not None
    assert agent._trigger_builder._builder_config is not None
    assert agent._trigger_builder._builder_config.max_tokens == 8_000
    assert ready["metadata"]["client_kind"] == "scripted"
    assert ready["metadata"]["decision_envelope"] == {
        "force_action_tool": True,
        "max_actor_calls": 4,
        "max_actor_calls_after_initial": 1,
        "max_output_tokens_per_call": 8_000,
        "max_output_tokens_per_call_after_initial": 4_096,
        "action_demand_max_output_tokens": None,
        "action_demand_disable_thinking": False,
        "model_retries": None,
        "model_timeout_seconds": 180.0,
        "max_actions_per_batch": 4,
    }
    assert ready["metadata"]["mode"] == "trigger"
    assert ready["metadata"]["treatment"] == "decision_discipline"
    assert ready["metadata"]["level_boundary_policy"] == "model_scribe"
    assert ready["metadata"]["sandbox_runtime"] == "host"
    assert (agent.workspace_dir / "notes/first-contact.md").read_text() == (
        "UP changed the public grid."
    )
    trace = [
        json.loads(line)
        for line in (tmp_path.parent / "tool-trace.jsonl").read_text().splitlines()
    ]
    assert [record["name"] for record in trace] == ["write_file", "take_actions"]
    assert [record["sequence"] for record in trace] == [0, 0]
    assert json.loads((agent.workspace_dir / "level_0/solved.json").read_text()) == {
        "action": "UP",
        "row": None,
        "col": None,
    }
    terminal = json.loads(
        (agent.workspace_dir / "level_0/terminal.json").read_text()
    )
    assert terminal["terminal_grid"] == [[9, 9], [9, 9]]
    animation = next((agent.workspace_dir / "level_0").glob("animation_*/frame_000.txt"))
    assert animation.read_text().strip().splitlines()[-2:] == [
        "y00: 8 8",
        "y01: 8 8",
    ]
    assert json.loads(
        (agent.workspace_dir / "level_0/level_complete.json").read_text()
    ) == {
        "source": "arc-host action_result",
        "outcome": "level_complete",
        "action": "UP",
        "row": None,
        "col": None,
        "levels_completed_before": 0,
        "levels_completed_after": 1,
        "after_observation": "next_level_start",
    }
    insights = (agent.workspace_dir / "notes/level_0_insights.md").read_text()
    assert insights.startswith("# Level 0 Insights")
    assert "summary pass did not create the requested note" in insights
    assert "level_complete.json:" in insights
    assert '"outcome": "level_complete"' in insights


@pytest.mark.asyncio
async def test_worker_protocol_routes_a_complete_game_and_reports_bad_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        SCRIPTED_TOOL_CALLS_ENV,
        json.dumps(
            [
                {
                    "tool": "write_file",
                    "arguments": {
                        "path": "notes/protocol.md",
                        "content": "served through the process boundary",
                    },
                },
                {
                    "tool": "take_actions",
                    "arguments": {
                        "actions": [{"action": "UP"}],
                        "rationale": "test one-action commitment",
                    },
                },
            ]
        ),
    )
    before = observation(0)
    after = observation(4, actions=[], levels=1)
    messages = [
        "diagnostic noise\n",
        HOST_PREFIX + "{not-json}\n",
        HOST_PREFIX
        + StartGameMessage(
            protocol_version=PROTOCOL_VERSION,
            type="start_game",
            game_id="served-game",
            seed=0,
            workspace_root=str(tmp_path),
            initial_observation=before,
        ).model_dump_json()
        + "\n",
        HOST_PREFIX
        + TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision_timeout_seconds=600,
            observation=before,
            previous_decision=None,
        ).model_dump_json()
        + "\n",
        HOST_PREFIX
        + ActionResultMessage(
            protocol_version=PROTOCOL_VERSION,
            type="action_result",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision=ActionDecision(action="UP"),
            before=before,
            after=after,
        ).model_dump_json()
        + "\n",
        HOST_PREFIX
        + FinishMessage(
            protocol_version=PROTOCOL_VERSION,
            type="finish",
            stop_reason="level_limit",
            actions=1,
            levels_completed=1,
            final_observation=after,
        ).model_dump_json()
        + "\n",
        HOST_PREFIX
        + json.dumps({"protocol_version": PROTOCOL_VERSION, "type": "close"})
        + "\n",
    ]
    stdout = io.StringIO()

    await serve(io.StringIO("".join(messages)), stdout)

    replies = [
        json.loads(line.removeprefix(WORKER_PREFIX))
        for line in stdout.getvalue().splitlines()
    ]
    assert [reply["type"] for reply in replies] == [
        "error",
        "ready",
        "decision",
        "observed",
        "finished",
    ]
    assert replies[0]["error_type"] == "ValidationError"
    assert replies[2]["decision"] == {"action": "UP", "row": None, "col": None}
    workspace = Path(replies[1]["metadata"]["workspace"])
    assert (workspace / "notes/protocol.md").read_text() == (
        "served through the process boundary"
    )


def test_action_decision_contract_matches_host() -> None:
    from tycho_nooa.models import ActionDecision

    with pytest.raises(ValidationError, match="CLICK requires"):
        ActionDecision(action="CLICK")
    with pytest.raises(ValidationError, match="does not accept"):
        ActionDecision(action="UP", row=1, col=2)
