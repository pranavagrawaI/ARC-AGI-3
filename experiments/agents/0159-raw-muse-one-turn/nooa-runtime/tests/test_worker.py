from __future__ import annotations

import asyncio
import copy
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from nooa.errors import GenerationError
from nooa.events import Feedback
from nooa.media import Image
from nooa.runtime.sandbox.errors import SandboxUnavailable
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from PIL import Image as PILImage

from worker import (
    _CODEACT_CONFIG,
    ANALYSIS_MAX_ITERATIONS,
    COMMIT_MAX_ATTEMPTS,
    CURRENT_SEQUENCE,
    AuditLog,
    CompactBeliefsGameSolver,
    Decision,
    DurableBeliefsGameSolver,
    GameSolver,
    History,
    ImplicitContextArchivalDisabled,
    ProviderPoolClient,
    TrajectoryGameSolver,
    _failure_stage,
    _find_implicit_context_archival_error,
    _forbidden_cell_primitive,
    _response,
    _write,
    disable_implicit_context_archival,
    install_provider_logging,
    retry_config,
)


def _codeact_response(code: str, *, reasoning: str | None = None) -> LLMResponse:
    arguments = json.dumps({"code": code})
    call = ToolCall(id="analysis-1", name="execute_python", arguments=arguments)
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[call],
        finish_reason="tool_calls",
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments},
                }
            ],
        },
        reasoning=reasoning,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _tool_response(
    name: str, arguments: dict, *, reasoning: str | None = None
) -> LLMResponse:
    encoded = json.dumps(arguments)
    call = ToolCall(id="commit-1", name=name, arguments=encoded)
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[call],
        finish_reason="tool_calls",
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": name, "arguments": encoded},
                }
            ],
        },
        reasoning=reasoning,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


class _HangingClient:
    def __init__(self):
        self.calls = 0

    async def acall(self, *args, **kwargs):
        self.calls += 1
        await asyncio.sleep(60)


class _RecordingClient:
    def __init__(self, response: LLMResponse):
        self.response = response
        self.calls = 0

    async def acall(self, *args, **kwargs):
        self.calls += 1
        return self.response


class _FailingClient:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    async def acall(self, *args, **kwargs):
        self.calls += 1
        raise self.error


@pytest.mark.asyncio
async def test_provider_pool_fails_over_then_sticks_to_successor(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    hanging = _HangingClient()
    successor = _RecordingClient(_response("ok"))
    pool = ProviderPoolClient(
        [("slow", hanging), ("successor", successor)],
        AuditLog(audit_path),
        attempt_timeout_seconds=0.01,
    )

    assert (await pool.acall([])).content == "ok"
    assert (await pool.acall([])).content == "ok"

    assert hanging.calls == 1
    assert successor.calls == 2
    events = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "provider_attempt_started",
        "provider_attempt_failed",
        "provider_attempt_started",
        "provider_attempt_completed",
        "provider_attempt_started",
        "provider_attempt_completed",
    ]
    assert events[1]["provider"] == "slow"
    assert events[3]["provider"] == "successor"


@pytest.mark.asyncio
async def test_provider_pool_does_not_fallback_on_model_length_finish():
    length_response = _response("")
    length_response.finish_reason = "length"
    primary = _RecordingClient(length_response)
    fallback = _RecordingClient(_response("fallback"))
    pool = ProviderPoolClient(
        [("primary", primary), ("fallback", fallback)],
        AuditLog(None),
        attempt_timeout_seconds=0.1,
    )

    response = await pool.acall([])

    assert response.finish_reason == "length"
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_provider_pool_does_not_fallback_or_poison_on_context_error():
    context_error = RuntimeError("This model's maximum context length is 131072 tokens")
    primary = _FailingClient(context_error)
    fallback = _RecordingClient(_response("fallback"))
    pool = ProviderPoolClient(
        [("primary", primary), ("fallback", fallback)],
        AuditLog(None),
        attempt_timeout_seconds=0.1,
    )

    with pytest.raises(RuntimeError, match="maximum context length"):
        await pool.acall([])
    with pytest.raises(RuntimeError, match="maximum context length"):
        await pool.acall([])

    assert primary.calls == 2
    assert fallback.calls == 0
    assert pool.unhealthy == set()


class RecordingFakeLLMClient(FakeLLMClient):
    def __init__(self, responses: list[LLMResponse]):
        super().__init__(responses)
        self.requests: list[dict] = []

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.requests.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "output_model": output_model,
                "kwargs": copy.deepcopy(kwargs),
            }
        )
        return await super().acall(
            messages,
            tools=tools,
            output_model=output_model,
            **kwargs,
        )


def test_protocol_write_uses_captured_file_descriptor(monkeypatch):
    read_fd, write_fd = os.pipe()
    redirected = io.StringIO()
    monkeypatch.setattr("worker._PROTOCOL_FD", write_fd)
    monkeypatch.setattr(sys, "stdout", redirected)

    try:
        _write("PREFIX\t", {"action": 2})
        os.close(write_fd)
        payload = os.read(read_fd, 1_000).decode()
    finally:
        os.close(read_fd)

    assert payload == 'PREFIX\t{"action":2}\n'
    assert redirected.getvalue() == ""


def test_decision_uses_human_control_labels_and_validates_click_coordinates():
    assert Decision(action="UP", explanation="move").model_dump() == {
        "action": "UP",
        "x": None,
        "y": None,
        "explanation": "move",
    }
    assert Decision(action="CLICK", x=12, y=34, explanation="select").model_dump() == {
        "action": "CLICK",
        "x": 12,
        "y": 34,
        "explanation": "select",
    }
    with pytest.raises(ValueError, match="CLICK requires"):
        Decision(action="CLICK", explanation="missing point")
    with pytest.raises(ValueError, match="does not accept"):
        Decision(action="LEFT", x=1, y=2, explanation="not a click")
    with pytest.raises(ValueError):
        Decision(action=1, explanation="SDK id is not model-visible")


@pytest.mark.asyncio
async def test_same_agent_replays_reasoning_into_next_game_turn(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    marker = "FIRST_TURN_REASONING_MARKER_0159"
    llm = FakeLLMClient(
        [
            _codeact_response(
                "return_result(action='DOWN', explanation='probe down')",
                reasoning=marker,
            ),
            _codeact_response(
                "return_result(action='RIGHT', explanation='continue right')"
            ),
        ]
    )
    solver = TrajectoryGameSolver(llm=llm)

    await solver.analyze_trajectory_and_choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action=None,
        turn=0,
        levels_completed=0,
    )
    await solver.analyze_trajectory_and_choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action="DOWN",
        turn=1,
        levels_completed=0,
    )

    replayed = [
        message for message in llm.last_messages if message.get("reasoning") == marker
    ]
    assert len(replayed) == 1
    assert replayed[0]["role"] == "assistant"
    assert replayed[0]["tool_calls"]


@pytest.mark.asyncio
async def test_codeact_replays_reasoning_after_tool_result(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    marker = "FIRST_CELL_REASONING_MARKER_0159"
    llm = FakeLLMClient(
        [
            _codeact_response("print('inspect trajectory')", reasoning=marker),
            _codeact_response("return_result(action='DOWN', explanation='probe down')"),
        ]
    )
    solver = TrajectoryGameSolver(llm=llm)

    result = await solver.analyze_trajectory_and_choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action=None,
        turn=0,
        levels_completed=0,
    )

    assert result.action == "DOWN"
    replayed = [
        message for message in llm.last_messages if message.get("reasoning") == marker
    ]
    assert len(replayed) == 1
    assert replayed[0]["tool_calls"][0]["function"]["name"] == "execute_python"


@pytest.mark.asyncio
async def test_budget_exhaustion_reserves_one_action_only_commit_call(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    analysis = [
        _codeact_response(f"print('analysis {index}')")
        for index in range(ANALYSIS_MAX_ITERATIONS)
    ]
    analysis[-1].reasoning = "FINAL_ANALYSIS_REASONING_MARKER_0159"
    final = _tool_response(
        "return_result",
        {"result": {"action": "LEFT", "explanation": "commit left"}},
        reasoning="FINAL_COMMIT_REASONING_MARKER_0159",
    )
    next_turn = _codeact_response(
        "return_result(action='DOWN', explanation='continue down')"
    )
    llm = RecordingFakeLLMClient([*analysis, final, next_turn])
    solver = TrajectoryGameSolver(llm=llm)

    result = await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action=None,
        turn=0,
        levels_completed=0,
    )

    assert result == Decision(action="LEFT", explanation="commit left")
    assert llm.call_count == ANALYSIS_MAX_ITERATIONS + 1
    assert [tool.name for tool in llm.last_tools or []] == ["return_result"]
    assert llm.requests[-1]["kwargs"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "return_result"},
    }
    transcript = json.dumps(llm.last_messages)
    assert "Analysis calls remaining: 0" in transcript
    assert "reserved action-commitment inference" in transcript
    assert "FINAL_ANALYSIS_REASONING_MARKER_0159" in transcript

    await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action="LEFT",
        turn=1,
        levels_completed=0,
    )

    replayed = json.dumps(llm.last_messages)
    assert "FINAL_COMMIT_REASONING_MARKER_0159" in replayed
    assert "return_result" in replayed


@pytest.mark.asyncio
async def test_reserved_commit_reprompts_after_unavailable_analysis_tool(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    analysis = [
        _codeact_response(f"print('analysis {index}')")
        for index in range(ANALYSIS_MAX_ITERATIONS)
    ]
    wrong_tool = _tool_response(
        "execute_python",
        {"code": "print('one more analysis')"},
    )
    corrected = _tool_response(
        "return_result",
        {"result": {"action": "RIGHT", "explanation": "corrected commitment"}},
    )
    llm = RecordingFakeLLMClient([*analysis, wrong_tool, corrected])
    solver = TrajectoryGameSolver(llm=llm)

    result = await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action=None,
        turn=0,
        levels_completed=0,
    )

    assert COMMIT_MAX_ATTEMPTS == 3
    assert result == Decision(action="RIGHT", explanation="corrected commitment")
    assert llm.call_count == ANALYSIS_MAX_ITERATIONS + 2
    assert [tool.name for tool in llm.requests[-2]["tools"]] == ["return_result"]
    assert [tool.name for tool in llm.requests[-1]["tools"]] == ["return_result"]
    assert "received ['execute_python']" in json.dumps(llm.requests[-1]["messages"])


@pytest.mark.asyncio
async def test_reserved_commit_accepts_exact_json_decision_when_provider_omits_tool(
    tmp_path,
):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    analysis = [
        _codeact_response(f"print('analysis {index}')")
        for index in range(ANALYSIS_MAX_ITERATIONS)
    ]
    textual_decision = _response(
        json.dumps({"action": "LEFT", "explanation": "strict transport fallback"})
    )
    llm = FakeLLMClient([*analysis, textual_decision])
    solver = TrajectoryGameSolver(llm=llm)

    result = await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action=None,
        turn=0,
        levels_completed=0,
    )

    assert result == Decision(action="LEFT", explanation="strict transport fallback")
    assert llm.call_count == ANALYSIS_MAX_ITERATIONS + 1


@pytest.mark.asyncio
async def test_reserved_commit_retry_feedback_names_the_commit_interface(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    analysis = [
        _codeact_response(f"print('analysis {index}')")
        for index in range(ANALYSIS_MAX_ITERATIONS)
    ]
    malformed_text = _response("I should probably move upward.")
    corrected = _tool_response(
        "return_result",
        {"result": {"action": "UP", "explanation": "corrected commitment"}},
    )
    llm = RecordingFakeLLMClient([*analysis, malformed_text, corrected])
    audit_path = tmp_path / "provider-events.jsonl"
    solver = TrajectoryGameSolver(llm=llm, audit=AuditLog(audit_path))

    result = await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action=None,
        turn=0,
        levels_completed=0,
    )

    assert result.action == "UP"
    correction = json.dumps(llm.requests[-1]["messages"])
    assert "ACTION COMMITMENT REJECTED" in correction
    assert "Invoke `return_result`" in correction
    assert "Ensure your JSON output" not in correction
    responses = [
        row
        for row in map(json.loads, audit_path.read_text().splitlines())
        if row["event"] == "action_commitment_response"
    ]
    assert responses[0]["content"] == "I should probably move upward."
    assert responses[0]["tool_calls"] == []
    assert responses[1]["tool_calls"][0]["name"] == "return_result"


@pytest.mark.asyncio
async def test_each_analysis_round_reports_exact_remaining_budget(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    llm = RecordingFakeLLMClient(
        [
            _codeact_response("print('first inspection')"),
            _codeact_response("print('second inspection')"),
            _codeact_response("return_result(action='DOWN', explanation='probe down')"),
        ]
    )
    solver = TrajectoryGameSolver(llm=llm)

    result = await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action=None,
        turn=0,
        levels_completed=0,
    )

    assert result.action == "DOWN"
    first_request = json.dumps(llm.requests[0]["messages"])
    second_request = json.dumps(llm.requests[1]["messages"])
    third_request = json.dumps(llm.requests[2]["messages"])
    assert "up to 11 analysis calls" in first_request
    assert "one reserved action-only commitment inference" in first_request
    assert "Analysis calls used: 1 of 11" in second_request
    assert "Analysis calls remaining: 10" in second_request
    assert "Analysis calls used: 2 of 11" in third_request
    assert "Analysis calls remaining: 9" in third_request


@pytest.mark.asyncio
async def test_invalid_reserved_commit_is_model_failure_without_fallback(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    analysis = [
        _codeact_response(f"print('analysis {index}')")
        for index in range(ANALYSIS_MAX_ITERATIONS)
    ]
    invalid = _tool_response(
        "return_result",
        {"result": {"action": "SPACEBAR", "explanation": "invalid"}},
    )
    llm = FakeLLMClient([*analysis, invalid])
    solver = TrajectoryGameSolver(llm=llm)

    with pytest.raises(GenerationError, match="not one of the legal actions"):
        await solver.choose_action(
            image=Image.from_file(frame),
            trajectory=[],
            legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
            previous_action=None,
            turn=0,
            levels_completed=0,
        )

    assert llm.call_count == ANALYSIS_MAX_ITERATIONS + 1


@pytest.mark.asyncio
async def test_illegal_analysis_commit_uses_constrained_commitment_reprompt(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    illegal_analysis = _codeact_response(
        "return_result(action='RESET', explanation='restart')"
    )
    corrected = _tool_response(
        "return_result",
        {"result": {"action": "RIGHT", "explanation": "legal correction"}},
    )
    llm = RecordingFakeLLMClient([illegal_analysis, corrected])
    audit_path = tmp_path / "provider-events.jsonl"
    solver = TrajectoryGameSolver(llm=llm, audit=AuditLog(audit_path))

    result = await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action=None,
        turn=0,
        levels_completed=0,
    )

    assert result == Decision(action="RIGHT", explanation="legal correction")
    assert llm.call_count == 2
    assert [tool.name for tool in llm.requests[-1]["tools"]] == ["return_result"]
    starts = [
        row
        for row in map(json.loads, audit_path.read_text().splitlines())
        if row["event"] == "action_commitment_started"
    ]
    assert len(starts) == 1
    assert starts[0]["reason"] == "analysis_selected_illegal_action"
    assert starts[0]["rejected_action"] == "RESET"
    assert starts[0]["legal_actions"] == ["UP", "DOWN", "LEFT", "RIGHT"]


@pytest.mark.asyncio
async def test_same_agent_carries_first_decision_into_second_turn(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    first = json.dumps({"action": "DOWN", "explanation": "probe down"})
    second = json.dumps(
        {"action": "RIGHT", "explanation": "action 2 moved down; go right"}
    )
    llm = FakeLLMClient([_response(first), _response(second)])
    solver = GameSolver(llm=llm)

    await solver.next_action(
        image=Image.from_file(frame),
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action=None,
        turn=0,
        levels_completed=0,
    )
    result = await solver.next_action(
        image=Image.from_file(frame),
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action="DOWN",
        turn=1,
        levels_completed=0,
    )

    assert result.action == "RIGHT"
    assert llm.call_count == 2
    assert "probe down" in json.dumps(llm.last_messages)


@pytest.mark.asyncio
async def test_predict_replays_reasoning_into_next_game_turn(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    marker = "PREDICT_REASONING_MARKER_0159"
    first = _response(json.dumps({"action": "DOWN", "explanation": "probe down"}))
    first.reasoning = marker
    second = _response(json.dumps({"action": "RIGHT", "explanation": "continue right"}))
    llm = FakeLLMClient([first, second])
    solver = GameSolver(llm=llm)

    await solver.next_action(
        image=Image.from_file(frame),
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action=None,
        turn=0,
        levels_completed=0,
    )
    await solver.next_action(
        image=Image.from_file(frame),
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action="DOWN",
        turn=1,
        levels_completed=0,
    )

    replayed = [
        message for message in llm.last_messages if message.get("reasoning") == marker
    ]
    assert len(replayed) == 1
    assert replayed[0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_provider_log_distinguishes_waiting_from_completion(tmp_path):
    path = tmp_path / "provider-events.jsonl"
    llm = FakeLLMClient([_response('{"action":1,"explanation":"up"}')])
    original_acall = llm.acall

    async def slow_acall(*args, **kwargs):
        await asyncio.sleep(0.025)
        return await original_acall(*args, **kwargs)

    llm.acall = slow_acall
    install_provider_logging(llm, AuditLog(path), heartbeat_seconds=0.005)
    token = CURRENT_SEQUENCE.set(7)
    try:
        await llm.acall([{"role": "user", "content": "next"}])
    finally:
        CURRENT_SEQUENCE.reset(token)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    events = [row["event"] for row in rows]
    assert events[0] == "provider_call_started"
    assert "provider_call_waiting" in events
    assert events[-1] == "provider_call_completed"
    assert {row["sequence"] for row in rows} == {7}


@pytest.mark.asyncio
async def test_provider_log_accepts_structured_response_content(tmp_path):
    path = tmp_path / "provider-events.jsonl"
    response = _response("placeholder")
    response.content = {"action": "UP", "explanation": "up"}
    llm = FakeLLMClient([response])
    install_provider_logging(llm, AuditLog(path), heartbeat_seconds=0.005)

    result = await llm.acall([{"role": "user", "content": "next"}])

    assert result.content == {"action": "UP", "explanation": "up"}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]["event"] == "provider_call_completed"
    assert rows[-1]["response_chars"] > 0


@pytest.mark.asyncio
async def test_provider_terminal_timeout_is_failure_not_endless_heartbeat(tmp_path):
    path = tmp_path / "provider-events.jsonl"
    llm = FakeLLMClient()

    async def timeout(*args, **kwargs):
        raise TimeoutError("provider deadline")

    llm.acall = timeout
    install_provider_logging(llm, AuditLog(path), heartbeat_seconds=0.005)

    with pytest.raises(TimeoutError, match="provider deadline"):
        await llm.acall([{"role": "user", "content": "next"}])

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "provider_call_started",
        "provider_call_failed",
    ]


def test_retry_callback_records_429_and_backoff(tmp_path):
    class RateLimitError(RuntimeError):
        status_code = 429

    path = tmp_path / "provider-events.jsonl"
    config = retry_config(AuditLog(path))
    assert config.on_retry is not None
    config.on_retry(2, RateLimitError("too many requests"), 9.5)

    row = json.loads(path.read_text())
    assert row["event"] == "provider_retry_scheduled"
    assert row["retry_number"] == 2
    assert row["status_code"] == 429
    assert row["rate_limited"] is True
    assert row["delay_seconds"] == 9.5


@pytest.mark.asyncio
async def test_codeact_can_compute_over_trajectory_and_return_one_decision(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    code = """
before = np.array(trajectory[-1]["before"])
after = np.array(trajectory[-1]["after"])
changed = int(np.count_nonzero(before != after))
return_result(action='DOWN', explanation=f"observed {changed} changed cell")
"""
    llm = FakeLLMClient([_codeact_response(code)])
    solver = TrajectoryGameSolver(llm=llm)
    trajectory = [
        {
            "turn": 0,
            "action": "UP",
            "x": None,
            "y": None,
            "before": [[0, 0], [0, 0]],
            "after": [[0, 1], [0, 0]],
            "levels_completed": 0,
            "state": "NOT_FINISHED",
            "full_reset": False,
        }
    ]

    result = await solver.analyze_trajectory_and_choose_action(
        image=Image.from_file(frame),
        trajectory=trajectory,
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action="UP",
        turn=1,
        levels_completed=0,
    )

    assert result.action == "DOWN"
    assert result.explanation == "observed 1 changed cell"
    assert trajectory[0]["after"] == [[0, 1], [0, 0]]
    rendered_messages = json.dumps(llm.last_messages, default=str)
    assert "data:image/" in rendered_messages
    assert "BLOCKED: __name__" not in rendered_messages
    system_prompt = llm.last_messages[0]["content"]
    assert "build_llm" not in system_prompt
    assert "Fan-out generation" not in system_prompt
    assert "Existing methods on `self`" not in system_prompt


@pytest.mark.asyncio
async def test_durable_beliefs_survive_between_game_turns(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    llm = RecordingFakeLLMClient(
        [
            _codeact_response(
                "self.write_notes('hypotheses.md', "
                "'DOWN moved the object down (confirmed after turn 0)')\n"
                "return_result(action='DOWN', explanation='recorded the tested mapping')"
            ),
            _codeact_response(
                "known = self.read_notes('hypotheses.md')\n"
                "return_result(action='RIGHT', explanation=known)"
            ),
        ]
    )
    solver = DurableBeliefsGameSolver(llm=llm, knowledge_dir=tmp_path / "knowledge")

    await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action=None,
        turn=0,
        levels_completed=0,
    )
    result = await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action="DOWN",
        turn=1,
        levels_completed=0,
    )

    assert result.action == "RIGHT"
    assert result.explanation == "DOWN moved the object down (confirmed after turn 0)"
    assert (tmp_path / "knowledge" / "hypotheses.md").read_text() == result.explanation
    prompt = json.dumps(llm.requests[0]["messages"])
    assert "self.read_notes" in prompt
    assert "self.write_notes" in prompt
    assert "curated semantic knowledge" in prompt
    assert str(tmp_path / "knowledge") not in prompt
    assert "self.choose_action" not in prompt


@pytest.mark.asyncio
async def test_compact_solver_exposes_namespaced_history_without_old_trajectory(
    tmp_path,
):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    snapshot = {
        "schema": "0159-history-snapshot-v1",
        "events": [
            {
                "event_id": "trajectory:000003",
                "stream": "trajectory",
                "ordinal": 3,
                "turn": 3,
                "kind": "record",
                "record": {"turn": 3, "action": "DOWN"},
            }
        ],
        "turn_views": {
            "3": {
                "schema": "0159-history-turn-view-v1",
                "turn": 3,
                "transcript": [],
                "mechanical_transitions": [{"action": "DOWN", "changed_cell_count": 1}],
                "omitted_streams": ["provider"],
                "truncated": False,
            }
        },
        "provider_episodes": {},
        "source_manifest": {},
    }
    llm = RecordingFakeLLMClient(
        [
            _codeact_response(
                "old = history.turn(3)\n"
                "return_result(action='LEFT', "
                "explanation=old['mechanical_transitions'][0]['action'])"
            )
        ]
    )
    solver = CompactBeliefsGameSolver(llm=llm, knowledge_dir=tmp_path / "knowledge")
    solver.install_history(snapshot)

    result = await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action="DOWN",
        turn=8,
        levels_completed=0,
    )

    assert result == Decision(action="LEFT", explanation="DOWN")
    prompt = json.dumps(llm.requests[0]["messages"])
    assert "history.turn(turn: int)" in prompt
    assert "history_turn" not in prompt
    assert "recent immutable-copy `trajectory`" in prompt


@pytest.mark.asyncio
async def test_nooa_summary_explicitly_marks_live_context_recovery(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    marker = "CONTEXT RECOVERY — SAME GAME, SAME AGENT — NOT A RESET"
    llm = RecordingFakeLLMClient(
        [
            _codeact_response("return_result(action='UP', explanation='zero')"),
            _codeact_response("return_result(action='DOWN', explanation='one')"),
            _codeact_response("return_result(action='LEFT', explanation='two')"),
        ]
    )
    solver = CompactBeliefsGameSolver(llm=llm, knowledge_dir=tmp_path / "knowledge")

    keys_before = set(solver.event_manager.keys())
    await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action=None,
        turn=0,
        levels_completed=0,
    )
    first_turn_tags = [
        tag
        for tag in solver.event_manager.keys()  # noqa: SIM118 - custom API
        if tag not in keys_before
    ]
    await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action="UP",
        turn=1,
        levels_completed=0,
    )
    solver.event_manager.collapse(first_turn_tags[0], first_turn_tags[-1], marker)
    await solver.choose_action(
        image=Image.from_file(frame),
        trajectory=[],
        legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        previous_action="DOWN",
        turn=2,
        levels_completed=0,
    )

    rendered = json.dumps(llm.requests[-1]["messages"])
    assert "CONTEXT RECOVERY" in rendered
    assert "SAME GAME, SAME AGENT" in rendered
    assert "NOT A RESET" in rendered


def test_history_namespace_is_bounded_read_only_and_exact():
    history = History(
        {
            "schema": "0159-history-snapshot-v1",
            "events": [
                {
                    "event_id": "transcript:000001",
                    "stream": "transcript",
                    "ordinal": 1,
                    "turn": 0,
                    "kind": "decision",
                    "record": {"action": "UP", "api_key": "[REDACTED]"},
                }
            ],
            "turn_views": {"0": {"turn": 0}},
            "provider_episodes": {},
            "source_manifest": {},
        }
    )

    assert history.turn(0) == {"turn": 0}
    assert history.get(["transcript:000001"])[0]["record"]["action"] == "UP"
    assert history.turns(0, 0, ["transcript"])[0]["event_id"] == ("transcript:000001")
    assert history.find("action up", {"limit": 1})[0]["event_id"] == (
        "transcript:000001"
    )
    with pytest.raises(ValueError, match="1 through 20"):
        history.find("up", {"limit": 100})


def test_compact_variant_blocks_nooa_implicit_archival_without_mutating_events(
    tmp_path,
):
    audit_path = tmp_path / "audit.jsonl"
    solver = CompactBeliefsGameSolver(
        llm=FakeLLMClient([]), knowledge_dir=tmp_path / "knowledge"
    )
    solver.event_manager.add(Feedback(content="evidence that must survive"))
    tags_before = list(solver.event_manager.keys())
    disable_implicit_context_archival(solver, AuditLog(audit_path))

    provider_error = RuntimeError("maximum context length is 131072 tokens")
    with pytest.raises(ImplicitContextArchivalDisabled) as caught:
        solver.runtime._archive_on_context_error(131_072, provider_error)

    assert caught.value.__cause__ is provider_error
    assert list(solver.event_manager.keys()) == tags_before
    event = json.loads(audit_path.read_text())
    assert event["event"] == "implicit_context_archival_blocked"
    assert event["context_window"] == 131_072


@pytest.mark.asyncio
async def test_real_nooa_context_error_surfaces_compaction_request_without_collapse(
    tmp_path,
):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    rejecting = _FailingClient(
        RuntimeError(
            "request contains 130000 tokens; maximum context length is 131072 tokens"
        )
    )
    rejecting.context_window = 131_072
    solver = CompactBeliefsGameSolver(
        llm=rejecting, knowledge_dir=tmp_path / "knowledge"
    )
    disable_implicit_context_archival(solver, AuditLog(None))

    with pytest.raises(Exception) as caught:  # NOOA may wrap the causal error
        await solver.choose_action(
            image=Image.from_file(frame),
            trajectory=[],
            legal_actions=["UP", "DOWN", "LEFT", "RIGHT"],
            previous_action=None,
            turn=3,
            levels_completed=0,
        )

    assert _find_implicit_context_archival_error(caught.value) is not None
    assert all(
        ".." not in tag
        for tag in solver.event_manager.keys()  # noqa: SIM118 - custom mapping API
    )


def test_worker_protocol_collapses_complete_turn_and_resumes_with_history(tmp_path):
    frame = tmp_path / "frame.png"
    PILImage.new("RGB", (2, 2), "white").save(frame)
    snapshot = {
        "schema": "0159-history-snapshot-v1",
        "events": [],
        "turn_views": {"0": {"turn": 0}},
        "provider_episodes": {},
        "source_manifest": {},
    }
    requests = [
        {
            "type": "turn",
            "sequence": 0,
            "image_path": str(frame),
            "legal_actions": ["UP", "DOWN", "LEFT", "RIGHT"],
            "previous_action": None,
            "levels_completed": 0,
            "trajectory": [],
            "history_snapshot": snapshot,
        },
        {
            "type": "turn",
            "sequence": 1,
            "image_path": str(frame),
            "legal_actions": ["UP", "DOWN", "LEFT", "RIGHT"],
            "previous_action": "UP",
            "levels_completed": 0,
            "trajectory": [],
            "history_snapshot": snapshot,
        },
        {
            "type": "compact",
            "sequence": 2,
            "retain_from_turn": 1,
            "summary_text": "CONTEXT RECOVERY SAME GAME SAME AGENT",
            "history_snapshot": snapshot,
        },
        {
            "type": "turn",
            "sequence": 2,
            "image_path": str(frame),
            "legal_actions": ["UP", "DOWN", "LEFT", "RIGHT"],
            "previous_action": "DOWN",
            "levels_completed": 0,
            "trajectory": [],
            "history_snapshot": snapshot,
        },
        {
            "type": "turn",
            "sequence": 3,
            "image_path": str(frame),
            "legal_actions": ["UP", "DOWN", "LEFT", "RIGHT"],
            "previous_action": "LEFT",
            "levels_completed": 0,
            "trajectory": [],
            "history_snapshot": snapshot,
        },
        {
            "type": "compact",
            "sequence": 4,
            "retain_from_turn": 3,
            "summary_text": "SECOND RECOVERY REPLACES THE FIRST",
            "history_snapshot": snapshot,
        },
        {"type": "close"},
    ]
    environment = {
        **os.environ,
        "NOOA_0159_VARIANT": "codeact_notes_compact",
        "NOOA_0159_KNOWLEDGE_DIR": str(tmp_path / "knowledge"),
        "NOOA_0159_SCRIPTED_CODEACT_CELLS": json.dumps(
            [
                "return_result(action='UP', explanation='zero')",
                "return_result(action='DOWN', explanation='one')",
                (
                    "old=history.turn(0); "
                    "return_result(action='LEFT', explanation=str(old['turn']))"
                ),
                "return_result(action='RIGHT', explanation='three')",
            ]
        ),
    }

    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "worker.py")],
        input="".join(json.dumps(request) + "\n" for request in requests),
        text=True,
        capture_output=True,
        env=environment,
        timeout=30,
        check=True,
    )

    responses = [
        json.loads(line.removeprefix("0159_RESPONSE\t"))
        for line in process.stdout.splitlines()
        if line.startswith("0159_RESPONSE\t")
    ]
    assert [response["type"] for response in responses] == [
        "decision",
        "decision",
        "compaction_applied",
        "decision",
        "decision",
        "compaction_applied",
    ]
    assert responses[2]["collapsed_turns"] == [0]
    assert responses[2]["active_events_after"] < responses[2]["active_events_before"]
    assert responses[3]["action"] == "LEFT"
    assert responses[3]["explanation"] == "0"
    assert responses[4]["action"] == "RIGHT"
    assert responses[5]["collapsed_turns"] == [1, 2]
    assert responses[5]["active_events_after"] < responses[5]["active_events_before"]


def test_codeact_guard_rejects_self_and_file_access():
    assert _forbidden_cell_primitive("print(self._llm)") == "self"
    assert _forbidden_cell_primitive("open('/proc/self/environ').read()") == "open"
    assert _forbidden_cell_primitive("print(type(trajectory).__name__)") is None
    assert _forbidden_cell_primitive("print(build_llm.__globals__)") == "__globals__"
    assert _forbidden_cell_primitive("np.array(trajectory[-1]['after'])") is None


def test_durable_beliefs_guard_allows_only_direct_notes_calls():
    allowed = frozenset({"read_notes", "write_notes"})

    assert (
        _forbidden_cell_primitive(
            "self.write_notes('hypotheses.md', 'action 2 = down')",
            allowed_self_methods=allowed,
        )
        is None
    )
    assert (
        _forbidden_cell_primitive("print(self._llm)", allowed_self_methods=allowed)
        == "self._llm"
    )
    assert (
        _forbidden_cell_primitive(
            "getattr(self, 'write_notes')('x.md', 'x')",
            allowed_self_methods=allowed,
        )
        == "self"
    )


def test_compact_guard_allows_only_direct_namespaced_history_calls():
    allowed = History._METHODS

    assert (
        _forbidden_cell_primitive(
            "evidence = history.turn(3)",
            allowed_history_methods=allowed,
        )
        is None
    )
    assert (
        _forbidden_cell_primitive(
            "print(history._events)",
            allowed_history_methods=allowed,
        )
        == "history._events"
    )
    assert (
        _forbidden_cell_primitive(
            "method = history.turn",
            allowed_history_methods=allowed,
        )
        == "history"
    )


def test_failure_stage_preserves_sandbox_failures():
    wrapped = RuntimeError("generation stopped")
    wrapped.__cause__ = SandboxUnavailable("network guard unavailable")

    assert _failure_stage(wrapped) == "sandbox"


def test_failure_stage_distinguishes_provider_and_model_failures():
    assert _failure_stage(TimeoutError("provider deadline")) == "provider"
    assert _failure_stage(GenerationError("invalid final answer")) == "model"


@pytest.mark.asyncio
async def test_os_sandbox_denies_network_and_unlisted_files(tmp_path):
    private_file = tmp_path / "host-only.txt"
    private_file.write_text("host evidence")
    solver = TrajectoryGameSolver(llm=FakeLLMClient())
    executor = SandboxedExecutor(
        solver,
        _CODEACT_CONFIG.sandbox,
        cell_timeout=2.0,
    )
    try:
        network = await executor.run_cell(
            "import socket\nsocket.socket(socket.AF_INET, socket.SOCK_STREAM)"
        )
        filesystem = await executor.run_cell(f"open({str(private_file)!r}).read()")
    finally:
        await executor.aclose()

    assert network.success is False
    assert "permission" in str(network.error).lower()
    assert filesystem.success is False
    assert "permission" in str(filesystem.error).lower()
