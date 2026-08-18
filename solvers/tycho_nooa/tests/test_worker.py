from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tycho_nooa.llm import SCRIPTED_TOOL_CALLS_ENV
from tycho_nooa.protocol import (
    PROTOCOL_VERSION,
    ActionResultMessage,
    FinishMessage,
    ObservationPayload,
    StartGameMessage,
    TurnMessage,
    parse_host_message,
)
from tycho_nooa.worker import WorkerSession


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
    insights = (agent.workspace_dir / "notes/level_0_insights.md").read_text()
    assert insights.startswith("# Level 0 Insights")
    assert "summary pass did not create the requested note" in insights


def test_action_decision_contract_matches_host() -> None:
    from tycho_nooa.models import ActionDecision

    with pytest.raises(ValidationError, match="CLICK requires"):
        ActionDecision(action="CLICK")
    with pytest.raises(ValidationError, match="does not accept"):
        ActionDecision(action="UP", row=1, col=2)
