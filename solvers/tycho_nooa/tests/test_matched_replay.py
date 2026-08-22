from __future__ import annotations

import json
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa import TychoNooaAgent
from tycho_nooa.matched_replay import (
    ARMS,
    THEORY_TRANSACTION_ARMS,
    execute_matched_replay,
    prepare_legacy_run_packet,
    validate_packet,
)
from tycho_nooa.trigger_builder import TriggerDecision


def response(tool: str, arguments: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=tool, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def prose(content: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def make_legacy_run(tmp_path: Path) -> Path:
    run = tmp_path / "legacy-run"
    workspace_root = run / "workspace"
    agent = TychoNooaAgent(
        llm=FakeLLMClient(),
        trigger_builder=False,
        game_id="game",
        workspace_root=workspace_root,
        available_actions=["UP", "DOWN"],
    )
    agent._workspace.record(
        [[0, 0]],
        level=0,
        turn_in_level=0,
        available=["UP", "DOWN"],
    )
    agent._workspace.record(
        [[0, 1]],
        level=0,
        turn_in_level=1,
        action="UP",
        available=["UP", "DOWN"],
    )
    before = {
        "grid": [[0, 1]],
        "available_actions": ["UP", "DOWN"],
        "levels_completed": 0,
        "state": "NOT_FINISHED",
        "full_reset": False,
    }
    after = {**before, "grid": [[0, 2]]}
    action_result = {
        "type": "action_result",
        "sequence": 1,
        "level": 0,
        "turn_in_level": 1,
        "decision": {"action": "UP", "row": None, "col": None},
        "before": before,
        "after": after,
    }
    finish = {
        "type": "finish",
        "actions": 2,
        "levels_completed": 0,
        "stop_reason": "action_budget",
        "final_observation": after,
    }
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text(json.dumps({"game_id": "game"}))
    (run / "transcript.jsonl").write_text(
        json.dumps(action_result) + "\n" + json.dumps(finish) + "\n"
    )
    return run


def test_legacy_packet_names_its_lifetime_limit_and_detects_drift(
    tmp_path: Path,
) -> None:
    packet = tmp_path / "packet"
    manifest = prepare_legacy_run_packet(make_legacy_run(tmp_path), packet)

    assert "not exact continuation" in manifest["claim_ceiling"]
    assert manifest["arms"] == [arm.__dict__ for arm in ARMS]

    (packet / "source" / "workspace" / "world_model.py").write_text("drift")
    with pytest.raises(ValueError, match="workspace_commitment"):
        validate_packet(packet)


@pytest.mark.asyncio
async def test_matched_warm_restart_exercises_all_three_authority_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = tmp_path / "packet"
    prepare_legacy_run_packet(make_legacy_run(tmp_path), packet)
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(False, None, None),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] replay fixture", None),
    )
    builder_tail = [
        response(
            "write_file",
            {"path": "notes/world_model.md", "content": "inspect objective"},
            "builder-notes",
        ),
        response(
            "write_file",
            {
                "path": "notes/world_model_report.md",
                "content": (
                    "candidate_goal_assessment: inconclusive\n"
                    "candidate_goal_reason: replay fixture has no terminal evidence\n"
                    "candidate_goal_route: DOWN\n"
                    "recommended_action: DOWN\n"
                ),
            },
            "builder-report",
        ),
        prose("builder done"),
        response("take_action", {"action": "DOWN"}, "action"),
    ]
    llms = {
        "trigger": FakeLLMClient(
            scripted_responses=[response("take_action", {"action": "UP"}, "action")]
        ),
        "exact_orchestrator": FakeLLMClient(
            scripted_responses=[
                response(
                    "invoke_builder",
                    {"reason": "objective unknown"},
                    "builder-request",
                ),
                *builder_tail,
            ]
        ),
        "orchestrator_typed_candidate_goals": FakeLLMClient(
            scripted_responses=[
                response(
                    "propose_goal",
                    {
                        "predicate": "reach colour 2",
                        "evidence_refs": ["level_0/turn_002.txt"],
                        "falsifier": "reaching colour 2 does not complete the level",
                    },
                    "goal",
                ),
                response(
                    "request_builder",
                    {
                        "focus": "objective",
                        "evidence_refs": ["level_0/turn_002.txt"],
                        "candidate_goal_id": "goal-0001",
                    },
                    "builder-request",
                ),
                *builder_tail[:-1],
                response(
                    "update_goal",
                    {
                        "goal_id": "goal-0001",
                        "status": "proposed",
                        "evidence_refs": ["level_0/turn_002.txt"],
                        "note": "builder review was inconclusive; retain hypothesis",
                    },
                    "goal-update",
                ),
                builder_tail[-1],
            ]
        ),
    }

    report = await execute_matched_replay(packet, tmp_path / "result", llms=llms)

    by_arm = {row["arm"]["name"]: row for row in report["observations"]}
    assert by_arm["trigger"]["decision"] == {
        "action": "UP",
        "row": None,
        "col": None,
    }
    assert "actor_request" in by_arm["exact_orchestrator"]["builder_events"]
    typed = by_arm["orchestrator_typed_candidate_goals"]
    assert typed["goal_events"] == ["goal_proposed", "goal_updated"]
    assert "request_builder" in typed["tool_names"]
    assert "actor_uptake" in typed["builder_events"]


@pytest.mark.asyncio
async def test_transactional_replay_matches_control_and_treatment_on_one_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")
    packet = tmp_path / "packet"
    manifest = prepare_legacy_run_packet(
        make_legacy_run(tmp_path),
        packet,
        arms=THEORY_TRANSACTION_ARMS,
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            True, "wm_signal", "matched transaction replay"
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] matched transaction fixture", None),
    )
    builder_turn = [
        response(
            "write_file",
            {"path": "world_model.py", "content": "# candidate body\n"},
            "builder-edit",
        ),
        response(
            "write_file",
            {"path": "notes/world_model.md", "content": "candidate finding"},
            "builder-notes",
        ),
        response(
            "write_file",
            {
                "path": "notes/world_model_report.md",
                "content": "recommended_action: UP\nnote: matched replay",
            },
            "builder-report",
        ),
        prose("builder complete"),
        response("take_action", {"action": "UP"}, "actor-action"),
    ]
    llms = {
        arm.name: FakeLLMClient(scripted_responses=list(builder_turn))
        for arm in THEORY_TRANSACTION_ARMS
    }

    report = await execute_matched_replay(
        packet,
        tmp_path / "result",
        llms=llms,
        arms=THEORY_TRANSACTION_ARMS,
    )

    assert manifest["arms"] == [arm.__dict__ for arm in THEORY_TRANSACTION_ARMS]
    by_arm = {row["arm"]["name"]: row for row in report["observations"]}
    assert by_arm["builder_control"]["world_model_changed"] is True
    treatment = by_arm["builder_transactional"]
    assert treatment["world_model_changed"] is False
    assert treatment["theory_publication"]["disposition"] == "canonical_restored"
    assert "transaction_finished" in treatment["theory_ledger_events"]
