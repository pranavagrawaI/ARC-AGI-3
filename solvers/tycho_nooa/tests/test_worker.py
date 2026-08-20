from __future__ import annotations

import io
import json
from pathlib import Path
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
) -> ObservationPayload:
    return ObservationPayload(
        grid=[[value, value + 1], [value + 2, value + 3]],
        available_actions=actions if actions is not None else ["UP", "DOWN"],
        levels_completed=levels,
        state="NOT_FINISHED",
        full_reset=False,
    )


def test_protocol_rejects_unknown_fields() -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "close",
        "unearned_authority": True,
    }

    with pytest.raises(ValidationError, match="unearned_authority"):
        parse_host_message(json.dumps(payload))


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
            grid=grid,
            available_actions=[],
            levels_completed=0,
            state="NOT_FINISHED",
            full_reset=False,
        )


@pytest.mark.asyncio
async def test_worker_keeps_one_agent_and_observes_level_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    response = await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=0,
            level=0,
            turn_in_level=0,
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
            after=observation(4, actions=[], levels=1),
        )
    )
    completion_checkpoint.assert_awaited_once()
    checkpoint_turn, checkpoint_evidence = completion_checkpoint.await_args.args
    assert checkpoint_turn.level == 0
    assert checkpoint_turn.turn_in_level == 1
    assert checkpoint_turn.levels_completed == 1
    assert '"outcome":"level_complete"' in checkpoint_evidence
    session.finish(
        FinishMessage(
            protocol_version=PROTOCOL_VERSION,
            type="finish",
            stop_reason="level_limit",
            actions=1,
            levels_completed=1,
            final_observation=observation(4, actions=[], levels=1),
        )
    )

    assert session.agent is agent
    assert agent._native_tool_config.max_tokens == 8_000
    assert agent._scribe_config.max_tokens == 8_000
    assert agent._trigger_builder is not None
    assert agent._trigger_builder._builder_config is not None
    assert agent._trigger_builder._builder_config.max_tokens == 8_000
    assert ready["metadata"]["client_kind"] == "scripted"
    assert (agent.workspace_dir / "notes/first-contact.md").read_text() == (
        "UP changed the public grid."
    )
    trace = [
        json.loads(line)
        for line in (tmp_path.parent / "tool-trace.jsonl").read_text().splitlines()
    ]
    assert [record["name"] for record in trace] == ["write_file", "take_action"]
    assert json.loads((agent.workspace_dir / "level_0/solved.json").read_text()) == {
        "action": "UP",
        "row": None,
        "col": None,
    }
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
