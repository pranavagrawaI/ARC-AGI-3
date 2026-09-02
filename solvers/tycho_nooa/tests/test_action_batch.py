from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa import ActionDecision, GameTurn, TychoNooaAgent
from tycho_nooa.action_batch import ActionBatchDecision, PendingActionBatch
from tycho_nooa.protocol import (
    PROTOCOL_VERSION,
    ActionResultMessage,
    ObservationPayload,
    TurnMessage,
)
from tycho_nooa.strategy import NativeToolConfig
from tycho_nooa.worker import WorkerSession


def response(tool: str, arguments: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=tool, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


class RecordingFakeLLM(FakeLLMClient):
    def __init__(self, scripted_responses: list[LLMResponse]) -> None:
        super().__init__(scripted_responses=scripted_responses)
        self.requests: list[dict[str, object]] = []

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.requests.append(
            {
                "tools": [tool.name for tool in tools or []],
                "schemas": {
                    tool.name: tool.get_parameter_schema() for tool in tools or []
                },
                "kwargs": kwargs,
            }
        )
        return await super().acall(
            messages, tools=tools, output_model=output_model, **kwargs
        )


def observation(
    value: int,
    *,
    actions: list[str] | None = None,
    state: str = "NOT_FINISHED",
    levels_completed: int = 0,
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
        available_actions=actions or ["UP", "DOWN", "LEFT", "CLICK"],
        levels_completed=levels_completed,
        state=state,
        full_reset=full_reset,
    )


def game_turn(value: int, sequence: int, *, previous: ActionDecision | None = None) -> GameTurn:
    return GameTurn(
        grid=observation(value).grid,
        available_actions=["UP", "DOWN", "LEFT", "CLICK"],
        sequence=sequence,
        action_limit=8,
        actions_used=sequence,
        actions_remaining=8 - sequence,
        final_permitted_action=sequence == 7,
        level=0,
        turn_in_level=sequence,
        previous_action=previous.action if previous else "",
        previous_row=previous.row if previous else None,
        previous_col=previous.col if previous else None,
    )


def action_result(
    sequence: int,
    decision: ActionDecision,
    before: ObservationPayload,
    after: ObservationPayload,
) -> ActionResultMessage:
    return ActionResultMessage(
        protocol_version=PROTOCOL_VERSION,
        type="action_result",
        sequence=sequence,
        level=0,
        turn_in_level=sequence,
        decision=decision,
        before=before,
        after=after,
    )


def batch_arguments(*actions: dict, **extra: object) -> dict:
    return {
        "actions": list(actions),
        "rationale": "repeat the established motion",
        **extra,
    }


def pending_batch(
    *,
    next_action: str = "DOWN",
    expected_grid: str | None = None,
    stop_on_no_change: bool = True,
) -> PendingActionBatch:
    return PendingActionBatch(
        batch=ActionBatchDecision.model_validate(
            batch_arguments(
                {"action": "UP"},
                {
                    "action": next_action,
                    "expected_before_grid_sha256": expected_grid,
                },
                stop_on_no_change=stop_on_no_change,
            )
        ),
        anchor_sequence=0,
        anchor_level=0,
        anchor_turn_in_level=0,
    )


def release_failure(
    pending: PendingActionBatch,
    turn: GameTurn,
    previous: ActionResultMessage | None,
) -> str | None:
    return pending.release(turn, previous)[1]


@pytest.mark.asyncio
async def test_actor_has_one_action_primitive_and_one_action_is_a_batch(
    tmp_path: Path,
) -> None:
    llm = RecordingFakeLLM(
        [
            response(
                "take_actions",
                batch_arguments({"action": "UP"}),
                "single-action-batch",
            )
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="batch-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    assert await agent.act(game_turn(0, 0)) == ActionDecision(action="UP")

    assert "take_actions" in llm.requests[0]["tools"]
    assert "take_action" not in llm.requests[0]["tools"]
    schema = llm.requests[0]["schemas"]["take_actions"]  # type: ignore[index]
    assert schema["properties"]["actions"]["minItems"] == 1  # type: ignore[index]
    records = [
        json.loads(line)
        for line in (tmp_path / "action-batch-trace.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "submitted",
        "released",
        "exhausted",
    ]
    assert records[0]["batch_length"] == 1


@pytest.mark.asyncio
async def test_removed_take_action_name_is_rejected_not_aliased(tmp_path: Path) -> None:
    trace_path = tmp_path / "tool-trace.jsonl"
    llm = RecordingFakeLLM(
        [
            response("take_action", {"action": "DOWN"}, "removed-tool"),
            response(
                "take_actions",
                batch_arguments({"action": "UP"}),
                "single-action-batch",
            ),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="batch-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )

    assert await agent.act(game_turn(0, 0)) == ActionDecision(action="UP")

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert records[0]["name"] == "take_action"
    assert records[0]["status"] == "error"
    assert records[0]["error_stage"] == "availability"
    assert records[0]["execution_status"] == "not_attempted"
    assert records[1]["name"] == "take_actions"
    assert records[1]["status"] == "complete"
    assert all(
        "take_action" not in request["tools"] for request in llm.requests
    )


@pytest.mark.asyncio
async def test_final_action_request_is_bounded_and_disables_thinking(
    tmp_path: Path,
) -> None:
    llm = RecordingFakeLLM(
        [
            response(
                "take_actions",
                batch_arguments({"action": "UP"}),
                "bounded-commitment",
            )
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="batch-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
        strategy_config=NativeToolConfig(
            max_tool_steps=0,
            force_action_tool=True,
            action_demand_max_tokens=2_048,
            action_demand_disable_thinking=True,
        ),
    )

    assert await agent.act(game_turn(0, 0)) == ActionDecision(action="UP")

    assert len(llm.requests) == 1
    assert llm.requests[0]["tools"] == ["take_actions"]
    request_options = llm.requests[0]["kwargs"]  # type: ignore[assignment]
    assert request_options["max_tokens"] == 2_048  # type: ignore[index]
    assert request_options["temperature"] is None  # type: ignore[index]
    assert request_options["tool_choice"] == "required"  # type: ignore[index]
    assert request_options["extra_body"] == {  # type: ignore[index]
        "chat_template_kwargs": {"enable_thinking": False}
    }


@pytest.mark.asyncio
async def test_one_model_call_releases_a_three_action_batch_across_observations(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "tool-trace.jsonl"
    llm = RecordingFakeLLM(
        [
            response(
                "take_actions",
                batch_arguments(
                    {"action": "UP"},
                    {"action": "UP"},
                    {"action": "DOWN"},
                ),
                "batch-1",
            )
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="batch-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN", "LEFT", "CLICK"],
        tool_trace_path=trace_path,
    )

    before = observation(0)
    first = await agent.act(game_turn(0, 0))
    assert first == ActionDecision(action="UP")
    await agent.observe(action_result(0, first, before, observation(5)))

    second = await agent.act(game_turn(5, 1, previous=first))
    assert second == ActionDecision(action="UP")
    await agent.observe(action_result(1, second, observation(5), observation(10)))

    third = await agent.act(game_turn(10, 2, previous=second))
    assert third == ActionDecision(action="DOWN")
    assert len(llm.requests) == 1

    records = [
        json.loads(line)
        for line in (tmp_path / "action-batch-trace.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "submitted",
        "released",
        "released",
        "released",
        "exhausted",
    ]
    assert [
        record["offset"] for record in records if record["event"] == "released"
    ] == [0, 1, 2]
    assert records[0]["batch_length"] == 3
    assert records[-1]["model_calls_avoided"] == 2
    batch_observations = [
        event.prompt
        for event in agent.event_manager.values()
        if isinstance(event, Task) and "BATCH CONTINUATION RECEIPT" in event.prompt
    ]
    assert len(batch_observations) == 2
    assert "'turn_in_level': 1" in batch_observations[0]
    assert "'turn_in_level': 2" in batch_observations[1]


@pytest.mark.asyncio
async def test_guarded_batch_continuation_skips_builder_dispatch(tmp_path: Path) -> None:
    llm = RecordingFakeLLM(
        [
            response(
                "take_actions",
                batch_arguments({"action": "UP"}, {"action": "DOWN"}),
                "batch-1",
            )
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="batch-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN", "LEFT", "CLICK"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    first = await agent.act(game_turn(0, 0))
    await agent.observe(action_result(0, first, observation(0), observation(5)))

    with patch.object(
        TychoNooaAgent,
        "_automatic_builder_exposure",
        new_callable=AsyncMock,
    ) as builder:
        assert await agent.act(game_turn(5, 1, previous=first)) == ActionDecision(
            action="DOWN"
        )

    builder.assert_not_awaited()


def test_every_public_batch_guard_has_a_stable_failure_reason() -> None:
    up = ActionDecision(action="UP")
    left = ActionDecision(action="LEFT")
    before = observation(0)
    after = observation(1)
    current = game_turn(1, 1, previous=up)
    previous = action_result(0, up, before, after)

    assert release_failure(pending_batch(), current, None) == (
        "previous_action_result_missing"
    )
    assert release_failure(
        pending_batch(), current, action_result(0, left, before, after)
    ) == "previous_action_result_mismatch"
    assert release_failure(
        pending_batch(), game_turn(2, 1, previous=up), previous
    ) == "successor_observation_mismatch"
    assert release_failure(
        pending_batch(), current, action_result(0, up, before, observation(1, full_reset=True))
    ) == "public_full_reset"
    assert release_failure(
        pending_batch(),
        current,
        action_result(0, up, before, observation(1, levels_completed=1)),
    ) == "level_advanced"
    assert release_failure(
        pending_batch(),
        current,
        action_result(0, up, before, observation(1, state="GAME_OVER")),
    ) == "terminal_state_game_over"
    assert release_failure(
        pending_batch(), current.model_copy(update={"level": 1}), previous
    ) == "level_changed"
    assert release_failure(
        pending_batch(), current.model_copy(update={"turn_in_level": 3}), previous
    ) == "turn_offset_mismatch"
    assert release_failure(
        pending_batch(), current.model_copy(update={"sequence": 3}), previous
    ) == "sequence_offset_mismatch"
    assert release_failure(
        pending_batch(), current.model_copy(update={"actions_remaining": 0}), previous
    ) == "action_horizon_exhausted"
    assert release_failure(
        pending_batch(), current.model_copy(update={"available_actions": ["UP"]}), previous
    ) == "next_action_unavailable"
    assert release_failure(
        pending_batch(expected_grid="0" * 64), current, previous
    ) == "expected_frame_mismatch"
    assert release_failure(
        pending_batch(),
        game_turn(0, 1, previous=up),
        action_result(0, up, before, before),
    ) == "previous_action_no_change"

    exhausted = pending_batch()
    exhausted.next_offset = 2
    assert release_failure(exhausted, current, previous) == "batch_exhausted"


@pytest.mark.asyncio
async def test_batch_stops_after_a_no_change_successor_and_returns_to_actor(
    tmp_path: Path,
) -> None:
    llm = RecordingFakeLLM(
        [
            response(
                "take_actions",
                batch_arguments({"action": "UP"}, {"action": "DOWN"}),
                "batch-1",
            ),
            response(
                "take_actions",
                batch_arguments({"action": "LEFT"}),
                "fallback",
            ),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="batch-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN", "LEFT", "CLICK"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    first = await agent.act(game_turn(0, 0))
    unchanged = observation(0)
    await agent.observe(action_result(0, first, unchanged, unchanged))

    assert await agent.act(game_turn(0, 1, previous=first)) == ActionDecision(
        action="LEFT"
    )
    assert len(llm.requests) == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "action-batch-trace.jsonl").read_text().splitlines()
    ]
    invalidated = next(record for record in records if record["event"] == "invalidated")
    assert invalidated["reason"] == "previous_action_no_change"
    assert invalidated["offset"] == 1


@pytest.mark.asyncio
async def test_batch_stops_when_an_exact_successor_guard_does_not_match(
    tmp_path: Path,
) -> None:
    llm = RecordingFakeLLM(
        [
            response(
                "take_actions",
                batch_arguments(
                    {"action": "UP"},
                    {
                        "action": "DOWN",
                        "expected_before_grid_sha256": "0" * 64,
                    },
                    stop_on_no_change=False,
                ),
                "batch-1",
            ),
            response(
                "take_actions",
                batch_arguments({"action": "LEFT"}),
                "fallback",
            ),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="batch-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN", "LEFT", "CLICK"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    first = await agent.act(game_turn(0, 0))
    await agent.observe(action_result(0, first, observation(0), observation(1)))

    assert await agent.act(game_turn(1, 1, previous=first)) == ActionDecision(
        action="LEFT"
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "action-batch-trace.jsonl").read_text().splitlines()
    ]
    invalidated = next(record for record in records if record["event"] == "invalidated")
    assert invalidated["reason"] == "expected_frame_mismatch"


@pytest.mark.asyncio
async def test_batch_validation_uses_current_actions_and_remaining_horizon(
    tmp_path: Path,
) -> None:
    llm = RecordingFakeLLM(
        [
            response(
                "take_actions",
                batch_arguments(
                    {"action": "UP"},
                    {"action": "DOWN"},
                    {"action": "UP"},
                ),
                "too-long",
            ),
            response(
                "take_actions",
                batch_arguments({"action": "UP"}),
                "fallback",
            ),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="batch-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN", "CLICK"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    turn = GameTurn(
        grid=[[0]],
        available_actions=["UP", "DOWN", "CLICK"],
        sequence=6,
        action_limit=8,
        actions_used=6,
        actions_remaining=2,
        final_permitted_action=False,
        level=0,
        turn_in_level=6,
    )

    assert await agent.act(turn) == ActionDecision(action="UP")
    assert len(llm.requests) == 2
    first_schema = llm.requests[0]["schemas"]["take_actions"]  # type: ignore[index]
    assert first_schema["properties"]["actions"]["maxItems"] == 2  # type: ignore[index]
    item_schema = first_schema["properties"]["actions"]["items"]  # type: ignore[index]
    assert all(
        "expected_before_grid_sha256" in branch["properties"]
        for branch in item_schema["oneOf"]
    )
    trace = [json.loads(line) for line in (tmp_path / "tool-trace.jsonl").read_text().splitlines()]
    rejected = next(record for record in trace if record["tool_call_id"] == "too-long")
    assert rejected["status"] == "error"
    assert rejected["error_stage"] == "validation"


@pytest.mark.asyncio
async def test_worker_protocol_releases_the_same_batch_without_a_second_model_call(
    tmp_path: Path,
) -> None:
    llm = RecordingFakeLLM(
        [
            response(
                "take_actions",
                batch_arguments({"action": "UP"}, {"action": "DOWN"}),
                "batch-1",
            )
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="batch-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN", "LEFT", "CLICK"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    session = WorkerSession()
    session.agent = agent
    before = observation(0)
    session.initial_observation = before

    first_reply = await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=0,
            action_limit=8,
            actions_used=0,
            actions_remaining=8,
            final_permitted_action=False,
            level=0,
            turn_in_level=0,
            decision_timeout_seconds=600,
            observation=before,
            previous_decision=None,
        )
    )
    first = ActionDecision.model_validate(first_reply["decision"])
    after = observation(5)
    await session.observe(action_result(0, first, before, after))

    second_reply = await session.decide(
        TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=1,
            action_limit=8,
            actions_used=1,
            actions_remaining=7,
            final_permitted_action=False,
            level=0,
            turn_in_level=1,
            decision_timeout_seconds=600,
            observation=after,
            previous_decision=first,
        )
    )

    assert first_reply["decision"] == {"action": "UP", "row": None, "col": None}
    assert second_reply["decision"] == {
        "action": "DOWN",
        "row": None,
        "col": None,
    }
    assert len(llm.requests) == 1
