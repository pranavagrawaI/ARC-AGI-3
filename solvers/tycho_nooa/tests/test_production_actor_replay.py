from __future__ import annotations

import json
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa import GameTurn, TychoNooaAgent
from tycho_nooa.models import ActionDecision
from tycho_nooa.production_actor_replay import (
    FAILURE_CLASSES,
    ProductionReplayOracle,
    execute_production_replay,
    prepare_production_replay_packet,
    validate_production_replay_packet,
)


@pytest.fixture(autouse=True)
def trusted_test_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")


def response(tool: str, arguments: dict, call_id: str) -> LLMResponse:
    if tool == "take_actions" and "actions" not in arguments:
        arguments = {
            "actions": [arguments],
            "rationale": "test one-action commitment",
        }
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=tool, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    )


def checkpoint(tmp_path: Path) -> tuple[Path, GameTurn, GameTurn]:
    root = tmp_path / "checkpoint"
    agent = TychoNooaAgent(
        llm=FakeLLMClient(),
        trigger_builder=False,
        game_id="game",
        workspace_root=root,
        available_actions=["UP", "DOWN"],
    )
    previous = GameTurn(
        grid=[[0, 0]],
        available_actions=["UP", "DOWN"],
        sequence=0,
        level=0,
        turn_in_level=0,
    )
    replay = GameTurn(
        grid=[[0, 1]],
        available_actions=["UP", "DOWN"],
        sequence=1,
        level=0,
        turn_in_level=1,
        previous_action="UP",
    )
    agent._workspace.record(
        previous.grid,
        level=0,
        turn_in_level=0,
        available=previous.available_actions,
    )
    agent._workspace.record(
        replay.grid,
        level=0,
        turn_in_level=1,
        action="UP",
        available=replay.available_actions,
    )
    return agent.workspace_dir, previous, replay


def oracle(failure_class: str = "remote_contradiction") -> ProductionReplayOracle:
    return ProductionReplayOracle(
        failure_class=failure_class,
        acceptable_terminal_outcomes=(ActionDecision(action="DOWN"),),
        acceptable_support_sets=(("L0:T0",),),
        known_relevant_evidence=("L0:T0",),
        known_distractor_evidence=("L0:T1",),
        required_computation="none",
        historical_failure="The historical Actor failed to use the old observation.",
    )


def prepare(tmp_path: Path, *, failure_class: str = "remote_contradiction") -> Path:
    workspace, previous, replay = checkpoint(tmp_path)
    packet = tmp_path / "packet"
    prepare_production_replay_packet(
        source_workspace=workspace,
        game_id="game",
        previous_turn=previous,
        replay_turn=replay,
        oracle=oracle(failure_class),
        destination=packet,
        mode="single",
        treatment="control",
    )
    return packet


def test_packet_commits_hidden_evaluator_key_and_detects_drift(tmp_path: Path) -> None:
    packet = prepare(tmp_path)

    prepared = validate_production_replay_packet(packet)

    assert prepared.oracle.failure_class == "remote_contradiction"
    assert not (packet / "source" / "workspace" / "evaluator").exists()
    manifest_text = (packet / "manifest.json").read_text()
    assert "historical Actor failed" not in manifest_text
    (packet / "evaluator" / "key.json").write_text("{}")
    with pytest.raises(ValueError, match="evaluator_key_commitment"):
        validate_production_replay_packet(packet)


@pytest.mark.asyncio
async def test_actual_production_controller_stops_at_replay_consequence_boundary(
    tmp_path: Path,
) -> None:
    packet = prepare(tmp_path)
    llm = FakeLLMClient(
        scripted_responses=[
            response(
                "read_file",
                {"path": "level_0/turn_000.json"},
                "read-old",
            ),
            response("take_actions", {"action": "DOWN"}, "terminal"),
        ]
    )

    report = await execute_production_replay(
        packet,
        tmp_path / "result",
        llm=llm,
    )

    assert report["controller"] == "tycho_nooa.agent.TychoNooaAgent"
    assert report["environment_actions_executed"] == 0
    assert report["commit"]["terminal_correct"] is True
    assert report["commit"]["grounding"] == "observed"
    assert report["select"] == {
        "observation_completeness": "exact_for_native_file_reads",
        "observed_relevant_disclosures": 1,
        "observed_distractor_disclosures": 1,
        "observed_unique_disclosures": 2,
        "reacquisitions": 0,
        "useful_evidence_fraction": 0.5,
        "evidence_reuse_rate": None,
    }
    assert report["commit"]["post_sufficiency_tool_calls"] == 0
    assert report["resource"]["provider_calls"] == 2
    runtime = tmp_path / "result" / "runtime" / "game"
    assert not (runtime / "evaluator").exists()
    assert not (runtime / "key.json").exists()


@pytest.mark.asyncio
async def test_python_history_access_is_reported_as_partial_not_inferred(
    tmp_path: Path,
) -> None:
    packet = prepare(tmp_path)
    llm = FakeLLMClient(
        scripted_responses=[
            response(
                "run_python",
                {"code": ("import wmlib\nprint(sorted(wmlib.frames()))\n")},
                "python",
            ),
            response("take_actions", {"action": "DOWN"}, "terminal"),
        ]
    )

    report = await execute_production_replay(
        packet,
        tmp_path / "result",
        llm=llm,
    )

    assert report["select"]["observation_completeness"] == "partial_python_access"
    assert report["commit"]["grounding"] == "unobservable"
    assert report["commit"]["premature_commitment"] is None
    assert report["compute"]["python_calls"] == 1


@pytest.mark.parametrize("failure_class", FAILURE_CLASSES)
def test_every_phase_one_failure_class_is_admitted(
    tmp_path: Path,
    failure_class: str,
) -> None:
    packet = prepare(tmp_path, failure_class=failure_class)

    prepared = validate_production_replay_packet(packet)

    assert prepared.oracle.failure_class == failure_class
