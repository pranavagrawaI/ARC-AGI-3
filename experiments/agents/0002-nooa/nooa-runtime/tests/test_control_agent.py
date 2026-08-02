# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from agent import ActionDecision, NooaControlAgent, PublicActionPoint, validate_decision


@pytest.fixture(autouse=True)
def _current_public_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Image.new("RGB", (1, 1)).save(tmp_path / "current_frame.png")


def _execute(code: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="execute_python",
                arguments=json.dumps({"code": code}),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
    )


def _return_via_tool(payload: dict[str, Any], call_id: str) -> LLMResponse:
    """The model answering through the return_result tool rather than inline."""

    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="return_result",
                arguments=json.dumps({"result": payload}),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
    )


class _RecordingFakeLLM(FakeLLMClient):
    def __init__(self, *, scripted_responses: list[LLMResponse]) -> None:
        super().__init__(scripted_responses=scripted_responses)
        self.calls: list[list[dict[str, Any]]] = []

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        output_model: type[Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(json.loads(json.dumps(messages)))
        return await super().acall(
            messages,
            tools=tools,
            output_model=output_model,
            **kwargs,
        )


@pytest.mark.asyncio
async def test_model_can_compute_write_a_file_and_return_one_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "counter = sum([20, 22])\n"
                "open('notes.txt', 'w').write(str(counter))\n"
                "print(counter)",
                "round-1",
            ),
            _execute(
                "assert counter == 42\n"
                "assert open('notes.txt').read() == '42'\n"
                "return_result({'action': 1, 'data': None, 'reason': 'fixture choice'})",
                "round-2",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)
    action_point = PublicActionPoint(
        observation={"step": 0, "pixels": [[[0]]], "state": "NOT_FINISHED"},
        legal_actions=(1, 2),
    )

    decision = await agent.choose_action(action_point)

    assert decision == ActionDecision(action=1, data=None, reason="fixture choice")
    assert (tmp_path / "notes.txt").read_text() == "42"
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_string_null_data_is_normalized_for_an_action_without_data() -> None:
    llm = _RecordingFakeLLM(
        scripted_responses=[
            _execute(
                "return_result({'action': 1, 'data': 'null', 'reason': 'move'})",
                "string-null",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1, 2))
    )

    assert decision == ActionDecision(action=1, data=None, reason="move")
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_string_null_data_does_not_bypass_coordinate_requirements() -> None:
    llm = _RecordingFakeLLM(
        scripted_responses=[
            _execute(
                "return_result({'action': 6, 'data': 'null', 'reason': 'missing'})",
                "string-null",
            ),
            _execute(
                "return_result({'action': 6, 'data': {'x': 2, 'y': 3}, "
                "'reason': 'repaired'})",
                "coordinate-repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(
            observation={"step": 0},
            legal_actions=(6,),
            coordinate_actions=(6,),
            frame_width=8,
            frame_height=6,
        )
    )

    assert decision == ActionDecision(
        action=6,
        data={"x": 2, "y": 3},
        reason="repaired",
    )
    assert llm.call_count == 2
    assert "requires integer x and y data" in json.dumps(llm.calls[1]).lower()


@pytest.mark.parametrize("data", ["None", "NULL", "{}", ""])
def test_action_decision_rejects_other_data_strings(data: str) -> None:
    with pytest.raises(ValueError, match="data"):
        ActionDecision(action=1, data=data, reason="invalid")


def test_host_rejects_an_action_outside_the_current_public_legal_set() -> None:
    action_point = PublicActionPoint(
        observation={"step": 3, "pixels": [[[0]]], "state": "NOT_FINISHED"},
        legal_actions=(1, 2),
    )

    with pytest.raises(ValueError, match="not legal"):
        validate_decision(
            action_point,
            ActionDecision(action=99, data=None, reason="illegal fixture"),
        )


def test_simple_action_data_matches_the_frozen_control_parser() -> None:
    action_point = PublicActionPoint(
        observation={"step": 0},
        legal_actions=(1,),
    )

    decision = validate_decision(
        action_point,
        ActionDecision(
            action=1,
            data={"unused": "preserved"},
            reason="control-compatible payload",
        ),
    )

    assert decision.data == {"unused": "preserved"}


@pytest.mark.asyncio
async def test_illegal_final_action_is_visible_and_requires_model_repair() -> None:
    llm = _RecordingFakeLLM(
        scripted_responses=[
            _execute(
                "return_result({'action': 99, 'data': None, 'reason': 'illegal'})",
                "illegal",
            ),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'repaired'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1, 2))
    )

    assert decision.reason == "repaired"
    assert llm.call_count == 2
    assert "action 99 is not legal" in json.dumps(llm.calls[1])


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (None, "requires integer x and y"),
        ({"x": 8, "y": 2}, "outside width=8, height=6"),
        ({"x": 2, "y": 6}, "outside width=8, height=6"),
    ],
)
def test_host_rejects_missing_or_out_of_bounds_coordinates(
    data: dict[str, int] | None, message: str
) -> None:
    action_point = PublicActionPoint(
        observation={"step": 0},
        legal_actions=(6,),
        coordinate_actions=(6,),
        frame_width=8,
        frame_height=6,
    )

    with pytest.raises(ValueError, match=message):
        validate_decision(
            action_point,
            ActionDecision(action=6, data=data, reason="invalid coordinates"),
        )


def test_coordinate_action_extra_data_matches_the_frozen_control_parser() -> None:
    action_point = PublicActionPoint(
        observation={"step": 0},
        legal_actions=(6,),
        coordinate_actions=(6,),
        frame_width=8,
        frame_height=6,
    )

    decision = validate_decision(
        action_point,
        ActionDecision(
            action=6,
            data={"x": 2, "y": 2, "extra": "preserved"},
            reason="control-compatible payload",
        ),
    )

    assert decision.data == {"x": 2, "y": 2, "extra": "preserved"}


@pytest.mark.asyncio
async def test_python_memory_and_files_survive_the_environment_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "remembered = 41\n"
                "open('ledger.txt', 'w').write('first action')\n"
                "return_result({'action': 1, 'data': None, 'reason': 'first'})",
                "action-1",
            ),
            _execute(
                "assert remembered == 41\n"
                "assert action_point['observation']['step'] == 1\n"
                "assert open('ledger.txt').read() == 'first action'\n"
                "remembered += 1\n"
                "return_result({'action': 2, 'data': None, 'reason': str(remembered)})",
                "action-2",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    first = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1, 2))
    )
    second = await agent.choose_action(
        PublicActionPoint(observation={"step": 1}, legal_actions=(1, 2))
    )
    await agent.aclose()

    assert first.action == 1
    assert second == ActionDecision(action=2, data=None, reason="42")
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_consumed_return_result_call_is_not_replayed_as_an_open_tool_call() -> None:
    """The host answers return_result, so history must not leave it open.

    return_result ends an action point: the host takes its argument and releases
    the environment action, and no tool message ever replies. Replaying the call
    left an unanswered tool call in the next request, which ModelRun refuses with
    "every tool_call must be answered by a tool message before the conversation
    continues". Other providers accept it, so load balancing hid the fault until
    a run pinned one endpoint.
    """

    llm = FakeLLMClient(
        scripted_responses=[
            _execute("marker = 1", "python-call"),
            _return_via_tool({"action": 1, "data": None, "reason": "first"}, "ret-1"),
            _return_via_tool({"action": 2, "data": None, "reason": "second"}, "ret-2"),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1, 2))
    )
    second = await agent.choose_action(
        PublicActionPoint(observation={"step": 1}, legal_actions=(1, 2))
    )
    await agent.aclose()

    assert second.action == 2
    _assert_every_replayed_tool_call_is_answered(llm.last_messages)
    # The answered execute_python call and its output must survive untouched.
    assert any(
        call.get("function", {}).get("name") == "execute_python"
        for message in llm.last_messages
        for call in (message.get("tool_calls") or [])
    )


def _assert_every_replayed_tool_call_is_answered(
    messages: list[dict[str, Any]],
) -> None:
    answered = {
        message.get("tool_call_id")
        for message in messages
        if message.get("role") == "tool"
    }
    open_calls = [
        call
        for message in messages
        for call in (message.get("tool_calls") or [])
        if call.get("id") not in answered
    ]
    assert open_calls == [], open_calls


@pytest.mark.asyncio
async def test_inline_return_leaves_no_unanswered_execute_python_in_history() -> None:
    """The inline route ends the turn too, so its execute_python is unanswered.

    Dropping only ``return_result`` was not enough. When the cell itself ends in
    an inline ``return_result``, the host consumes the result and the enclosing
    ``execute_python`` call never receives its output either. ModelRun refuses
    the next request for exactly the same reason.
    """

    llm = FakeLLMClient(
        scripted_responses=[
            _execute("marker = 1", "answered-call"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'first'})",
                "consumed-call",
            ),
            _execute(
                "return_result({'action': 2, 'data': None, 'reason': 'second'})",
                "consumed-call-2",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1, 2))
    )
    second = await agent.choose_action(
        PublicActionPoint(observation={"step": 1}, legal_actions=(1, 2))
    )
    await agent.aclose()

    assert second.action == 2
    _assert_every_replayed_tool_call_is_answered(llm.last_messages)
    # The answered call and its output are untouched.
    assert any(
        message.get("tool_call_id") == "answered-call"
        for message in llm.last_messages
    )


@pytest.mark.asyncio
async def test_a_reissued_action_point_carries_no_unanswered_call_either() -> None:
    """A repair crossing reissues one action point, and it must stay clean.

    Filtering only the retained turns passed both earlier regressions and still
    failed live: a reissued action point reaches the provider through the live
    turn, which retention never touches. Run 20260802T144344Z refused a
    commitment, reissued the crossing, and ModelRun rejected the repair with the
    same "every tool_call must be answered" error, naming the consumed call from
    the crossing before it.
    """

    llm = FakeLLMClient(
        scripted_responses=[
            _execute("marker = 1", "answered-call"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'refused'})",
                "consumed-first",
            ),
            _execute(
                "return_result({'action': 2, 'data': None, 'reason': 'repaired'})",
                "consumed-repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    # The same step twice: the host refused the first commitment and reissued.
    await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1, 2))
    )
    repaired = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1, 2))
    )
    await agent.aclose()

    assert repaired.action == 2
    _assert_every_replayed_tool_call_is_answered(llm.last_messages)


@pytest.mark.asyncio
async def test_context_evicts_complete_action_turns_with_only_a_pointer_notice() -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "old_private_marker = 1\n"
                "return_result({'action': 1, 'data': None, 'reason': 'first'})",
                "first",
            ),
            _execute(
                "retained_marker = 2\n"
                "return_result({'action': 1, 'data': None, 'reason': 'second'})",
                "second",
            ),
            _execute(
                "retained_marker = 3\n"
                "return_result({'action': 1, 'data': None, 'reason': 'third'})",
                "third",
            ),
            _execute("print(variables())", "hidden-history-query"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'fourth'})",
                "fourth",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm, max_context_turns=1)

    for step in range(4):
        await agent.choose_action(
            PublicActionPoint(
                observation={"step": step},
                legal_actions=(1,),
            )
        )

    visible = json.dumps(llm.last_messages)
    assert "Transcript omission." in visible
    assert "action steps 0" in visible
    assert "observations/step-*.json" in visible
    assert "retained_marker" in visible
    assert "old_private_marker" not in visible
    assert "NameError: name 'variables' is not defined" in visible
    host_audit_events = list(agent.event_manager._backend.all_events())
    assert "old_private_marker" in repr(host_audit_events)


@pytest.mark.asyncio
async def test_total_model_call_budget_stops_before_an_extra_provider_call() -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute("print('one permitted call')", "first"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'too late'})",
                "second",
            ),
        ]
    )
    agent = NooaControlAgent(
        llm=llm,
        max_model_calls=1,
        max_decision_repairs=1,
    )

    with pytest.raises(Exception, match="model-call budget exhausted"):
        await agent.choose_action(
            PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
        )

    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_syntax_and_runtime_errors_are_visible_and_require_model_repair() -> None:
    llm = _RecordingFakeLLM(
        scripted_responses=[
            _execute("broken =", "syntax"),
            _execute("1 / 0", "runtime"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'repaired'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    assert decision.reason == "repaired"
    assert llm.call_count == 3
    assert "RestrictedCodeError: Syntax error" in json.dumps(llm.calls[1])
    assert "ZeroDivisionError" in json.dumps(llm.calls[2])


@pytest.mark.asyncio
async def test_malformed_final_action_is_rejected_before_model_repair() -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "return_result({'action': 'one', 'data': None, 'reason': 'bad'})",
                "malformed",
            ),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'repaired'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    assert decision.reason == "repaired"
    assert llm.call_count == 2
    assert "validation" in json.dumps(llm.last_messages).lower()


@pytest.mark.asyncio
async def test_oversized_python_output_is_bounded_before_model_repair() -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute("print('x' * 100_000)", "oversized"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'bounded'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    visible = json.dumps(llm.last_messages)
    assert decision.reason == "bounded"
    assert len(visible) < 30_000
    assert "truncat" in visible.lower()


@pytest.mark.asyncio
async def test_valid_action_closes_the_temporal_seam_before_later_model_work(
    tmp_path: Path,
) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'accepted'})",
                "accepted",
            ),
            _execute(
                "open('post-accept-mutation.txt', 'w').write('forbidden')",
                "must-not-run",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm, workspace=str(tmp_path))

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    assert decision.reason == "accepted"
    assert llm.call_count == 1
    assert not (tmp_path / "post-accept-mutation.txt").exists()


@pytest.mark.asyncio
async def test_timeout_restarts_memory_preserves_files_and_requires_model_repair(
    tmp_path: Path,
) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "memory_before_timeout = 7\n"
                "open('durable.txt', 'w').write('kept')\n"
                "return_result({'action': 1, 'data': None, 'reason': 'seeded'})",
                "seed",
            ),
            _execute("while True:\n    pass", "timeout"),
            _execute(
                "try:\n"
                "    memory_before_timeout\n"
                "except NameError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('memory survived timeout restart')\n"
                "assert action_point['observation']['step'] == 1\n"
                "assert tuple(action_point['legal_actions']) == (1,)\n"
                "assert open('durable.txt').read() == 'kept'\n"
                "for leaked_name in ('self', 'CodeActStrategy', 'variables'):\n"
                "    if leaked_name in dir():\n"
                "        raise AssertionError('FRAMEWORK_RESTART_LEAK')\n"
                "return_result({'action': 1, 'data': None, 'reason': 'restarted'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(
        llm=llm,
        workspace=str(tmp_path),
        workspace_timeout_seconds=0.1,
    )

    await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )
    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 1}, legal_actions=(1,))
    )

    assert decision.reason == "restarted"
    feedback = json.dumps(llm.last_messages).lower()
    assert "was restarted, so variables/functions defined in earlier cells are gone" in feedback
    assert "framework_restart_leak" not in feedback


@pytest.mark.asyncio
async def test_background_worker_cannot_outlive_action_acceptance(tmp_path: Path) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "import threading, time\n"
                "def delayed_mutation():\n"
                "    time.sleep(0.05)\n"
                "    open('post-accept.txt', 'w').write('mutated')\n"
                "threading.Thread(target=delayed_mutation).start()\n"
                "return_result({'action': 1, 'data': None, 'reason': 'unsafe'})",
                "background-worker",
            ),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'repaired'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm, workspace=str(tmp_path))

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )
    await asyncio.sleep(0.1)

    assert decision.reason == "repaired"
    assert llm.call_count == 2
    assert not (tmp_path / "post-accept.txt").exists()


@pytest.mark.asyncio
async def test_explicit_restart_clears_python_memory_but_keeps_workspace_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    first_llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "memory = 7\n"
                "open('ledger.txt', 'w').write('durable')\n"
                "return_result({'action': 1, 'data': None, 'reason': 'first'})",
                "first",
            )
        ]
    )
    first_agent = NooaControlAgent(llm=first_llm, workspace=str(tmp_path))
    await first_agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )
    await first_agent.aclose()

    second_llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "try:\n"
                "    memory\n"
                "except NameError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('memory crossed restart')\n"
                "assert open('ledger.txt').read() == 'durable'\n"
                "return_result({'action': 1, 'data': None, 'reason': 'restarted'})",
                "second",
            )
        ]
    )
    second_agent = NooaControlAgent(llm=second_llm, workspace=str(tmp_path))
    decision = await second_agent.choose_action(
        PublicActionPoint(observation={"step": 1}, legal_actions=(1,))
    )
    await second_agent.aclose()

    assert decision.reason == "restarted"


@pytest.mark.asyncio
async def test_generated_python_cannot_reference_the_nooa_agent() -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute("print(self)", "forbidden-self"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'repaired'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    assert decision.action == 1
    assert llm.call_count == 2
    feedback = json.dumps(llm.last_messages)
    assert "NameError" in feedback
    assert "sandbox self-proxy" not in feedback


@pytest.mark.asyncio
async def test_generated_python_cannot_reach_hidden_objects_through_globals() -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute("print(globals()['self'])", "forbidden-globals"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'repaired'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    assert decision.action == 1
    assert llm.call_count == 2
    feedback = json.dumps(llm.last_messages)
    assert "globals() is forbidden" in feedback
    assert "sandbox self-proxy" not in feedback


@pytest.mark.asyncio
async def test_generic_object_introspection_remains_available_after_runtime_pruning() -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "payload = {'answer': 42}\n"
                "assert getattr(payload, 'get')('answer') == 42\n"
                "assert 'get' in dir(payload)\n"
                "return_result({'action': 1, 'data': None, 'reason': 'generic'})",
                "generic-introspection",
            )
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    assert decision.reason == "generic"
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_generated_python_cannot_import_the_nooa_runtime() -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute("import nooa\nprint(nooa)", "forbidden-import"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'repaired'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    assert decision.action == 1
    assert llm.call_count == 2
    feedback = json.dumps(llm.last_messages[-1])
    assert "nooa" in feedback
    assert "is restricted" in feedback


@pytest.mark.asyncio
async def test_model_written_module_cannot_bypass_blocked_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARC_LLM_API_KEY", "workspace-import-secret")
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "open('escape_module.py', 'w').write("
                "\"import os\\nopen('leaked.txt', 'w').write("
                "os.environ.get('ARC_LLM_API_KEY', 'missing'))\\n\")\n"
                "try:\n"
                "    import escape_module\n"
                "except ModuleNotFoundError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('workspace module became importable')\n"
                "return_result({'action': 1, 'data': None, 'reason': 'contained'})",
                "workspace-import-probe",
            )
        ]
    )
    agent = NooaControlAgent(llm=llm, workspace=str(tmp_path))

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )
    await agent.aclose()

    assert decision.reason == "contained"
    assert not (tmp_path / "leaked.txt").exists()
    assert "workspace-import-secret" not in json.dumps(llm.last_messages)


@pytest.mark.asyncio
async def test_public_action_point_is_plain_data_not_an_agent_runtime_type() -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute("print(type(action_point))", "type-probe"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'plain data'})",
                "decision",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    assert decision.action == 1
    visible = json.dumps(llm.last_messages)
    assert "<class 'dict'>" in visible
    assert "agent.PublicActionPoint" not in visible


@pytest.mark.asyncio
async def test_generated_python_cannot_import_an_undeclared_distribution() -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute("import pydantic\nprint(pydantic)", "undeclared-package"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'repaired'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    assert decision.action == 1
    assert "is restricted" in json.dumps(llm.last_messages)


@pytest.mark.asyncio
async def test_declared_numpy_and_pillow_packages_remain_available() -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "import numpy as np\n"
                "from PIL import Image as PillowImage\n"
                "assert int(np.array([20, 22]).sum()) == 42\n"
                "assert PillowImage.new('RGB', (1, 1)).size == (1, 1)\n"
                "return_result({'action': 1, 'data': None, 'reason': 'available'})",
                "declared",
            ),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'unexpected repair'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    assert decision.reason == "available", json.dumps(llm.last_messages)
    assert llm.call_count == 1, json.dumps(llm.last_messages)


@pytest.mark.asyncio
async def test_model_visible_prompt_does_not_advertise_runtime_objects() -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'fixture'})",
                "decision",
            )
        ]
    )
    agent = NooaControlAgent(llm=llm)

    await agent.choose_action(PublicActionPoint(observation={"step": 0}, legal_actions=(1,)))

    visible = json.dumps(llm.last_messages).lower()
    for forbidden in (
        "nooa",
        "agent instance",
        "self.",
        "runtime",
        "provider",
        "controller",
        "context",
        "trace",
        "ipc",
        "process",
        "host",
        "context_manager",
        "event_manager",
    ):
        assert forbidden not in visible


@pytest.mark.asyncio
async def test_os_sandbox_denies_files_outside_the_run_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    Image.new("RGB", (1, 1)).save(workspace / "current_frame.png")
    secret = tmp_path / "host-only.txt"
    secret.write_text("must-not-cross")
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(f"print(open({str(secret)!r}).read())", "escape"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'repaired'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm, workspace=str(workspace))

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    assert decision.action == 1
    assert llm.call_count == 2
    feedback = json.dumps(llm.last_messages)
    assert "PermissionError" in feedback
    assert "must-not-cross" not in feedback


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact", ("ipc/requests.jsonl", "process.jsonl"))
async def test_generated_python_cannot_read_sibling_ipc_or_run_artifacts(
    tmp_path: Path,
    artifact: str,
) -> None:
    run_root = tmp_path / "run"
    workspace = run_root / "workspace"
    workspace.mkdir(parents=True)
    Image.new("RGB", (1, 1)).save(workspace / "current_frame.png")
    protected = run_root / artifact
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_text("HIDDEN_RUN_RECORD")
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(f"print(open({str(protected)!r}).read())", "sibling-read"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'repaired'})",
                "repair",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm, workspace=str(workspace))

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    feedback = json.dumps(llm.last_messages[-1])
    assert decision.reason == "repaired"
    assert llm.call_count == 2
    assert "PermissionError" in feedback
    assert "HIDDEN_RUN_RECORD" not in feedback


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "probe", "leak_marker"),
    [
        (
            "framework namespace",
            "try:\n"
            "    CodeActStrategy\n"
            "except NameError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('FRAMEWORK_NAMESPACE_LEAK')",
            "FRAMEWORK_NAMESPACE_LEAK",
        ),
        (
            "framework introspection helper",
            "try:\n"
            "    variables()\n"
            "except NameError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('FRAMEWORK_HELPER_LEAK')",
            "FRAMEWORK_HELPER_LEAK",
        ),
        (
            "process arguments",
            "from pathlib import Path\n"
            "Path('/proc/self/cmdline').read_bytes()\n"
            "raise AssertionError('PROCESS_ARGUMENT_LEAK')",
            "PROCESS_ARGUMENT_LEAK",
        ),
        (
            "process environment",
            "from pathlib import Path\n"
            "Path('/proc/self/environ').read_bytes()\n"
            "raise AssertionError('PROCESS_ENVIRONMENT_LEAK')",
            "PROCESS_ENVIRONMENT_LEAK",
        ),
        (
            "parent process",
            "from pathlib import Path\n"
            "Path('/proc/1/cmdline').read_bytes()\n"
            "raise AssertionError('PARENT_PROCESS_LEAK')",
            "PARENT_PROCESS_LEAK",
        ),
        (
            "NOOA package discovery",
            "import pkgutil\n"
            "if any(item.name == 'nooa' for item in pkgutil.iter_modules()):\n"
            "    raise AssertionError('NOOA_PACKAGE_DISCOVERY_LEAK')",
            "NOOA_PACKAGE_DISCOVERY_LEAK",
        ),
        (
            "garbage-collector object enumeration",
            "import gc\n"
            "runtime_objects = [repr(type(item)) for item in gc.get_objects() "
            "if 'nooa' in repr(type(item)).lower()]\n"
            "if runtime_objects:\n"
            "    raise AssertionError('GC_RUNTIME_LEAK')",
            "GC_RUNTIME_LEAK",
        ),
        (
            "NOOA source file",
            "from pathlib import Path\n"
            f"Path({str(Path(__file__).parents[1] / 'agent.py')!r}).read_bytes()\n"
            "raise AssertionError('NOOA_SOURCE_LEAK')",
            "NOOA_SOURCE_LEAK",
        ),
        (
            "host cache",
            "from pathlib import Path\n"
            "tuple(Path('/home/hipra/.cache').iterdir())\n"
            "raise AssertionError('HOST_CACHE_LEAK')",
            "HOST_CACHE_LEAK",
        ),
    ],
)
async def test_generated_python_cannot_reach_runtime_or_process_surfaces(
    route: str,
    probe: str,
    leak_marker: str,
) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _execute(probe, f"probe-{route}"),
            _execute(
                "return_result({'action': 1, 'data': None, 'reason': 'repaired'})",
                f"repair-{route}",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm)

    decision = await agent.choose_action(
        PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
    )

    execution_feedback = json.dumps(llm.last_messages[-1])
    assert decision.reason == "repaired"
    assert llm.call_count == 2
    assert leak_marker not in execution_feedback
    if route in {
        "framework namespace",
        "framework introspection helper",
        "NOOA package discovery",
    }:
        assert "ResultStatus.COMPLETE" in execution_feedback
    else:
        assert any(
            marker in execution_feedback
            for marker in (
                "RUNTIME_ACCESS_DENIED",
                "PermissionError",
                "is restricted",
                "is blocked",
            )
        ), execution_feedback
