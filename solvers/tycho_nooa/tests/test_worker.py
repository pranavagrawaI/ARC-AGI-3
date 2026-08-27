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
                {"tool": "take_action", "arguments": {"action": "UP"}},
            ]
        ),
    )
    monkeypatch.setenv("ARC_LLM_MAX_TOKENS", "8000")
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
    assert agent._scribe_config.max_tokens == 8_000
    assert agent._trigger_builder is not None
    assert agent._trigger_builder._builder_config is not None
    assert agent._trigger_builder._builder_config.max_tokens == 8_000
    assert ready["metadata"]["client_kind"] == "scripted"
    assert ready["metadata"]["mode"] == "trigger"
    assert ready["metadata"]["treatment"] == "control"
    assert ready["metadata"]["sandbox_runtime"] == "host"
    assert (agent.workspace_dir / "notes/first-contact.md").read_text() == (
        "UP changed the public grid."
    )
    trace = [
        json.loads(line)
        for line in (tmp_path.parent / "tool-trace.jsonl").read_text().splitlines()
    ]
    assert [record["name"] for record in trace] == ["write_file", "take_action"]
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
                {"tool": "take_action", "arguments": {"action": "UP"}},
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
