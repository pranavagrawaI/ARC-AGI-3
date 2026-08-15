from __future__ import annotations

import asyncio
import copy
import io
import json
import os
import sys

import pytest
from nooa.errors import GenerationError
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
    Decision,
    DurableBeliefsGameSolver,
    GameSolver,
    TrajectoryGameSolver,
    _failure_stage,
    _forbidden_cell_primitive,
    _response,
    _write,
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
