# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
                "return_result({'action': 1, 'data': None, 'reason': 'third'})",
                "third",
            ),
        ]
    )
    agent = NooaControlAgent(llm=llm, max_context_turns=1)

    for step in range(3):
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
                "assert open('durable.txt').read() == 'kept'\n"
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
    assert "RUNTIME_ACCESS_DENIED" in json.dumps(llm.last_messages)


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
    assert "RUNTIME_ACCESS_DENIED" in json.dumps(llm.last_messages)


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
    feedback = json.dumps(llm.last_messages)
    assert "nooa" in feedback
    assert "is restricted" in feedback


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
    assert "nooa" not in visible
    assert "agent instance" not in visible
    assert "self." not in visible
    assert "context_manager" not in visible
    assert "event_manager" not in visible


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
