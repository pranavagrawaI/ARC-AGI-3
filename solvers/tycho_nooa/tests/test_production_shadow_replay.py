from __future__ import annotations

import json
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.production_shadow_replay import execute_production_shadow_replay


@pytest.fixture(autouse=True)
def trusted_test_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")


def response(action: str, call_id: str) -> LLMResponse:
    call = ToolCall(
        id=call_id,
        name="take_action",
        arguments=json.dumps({"action": action}),
    )
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[call],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )


def write_transcript(path: Path) -> None:
    observations = [
        _observation([[0, 0]]),
        _observation([[0, 1]]),
        _observation([[1, 1]]),
        _observation([[1, 0]]),
    ]
    records = []
    for sequence, action in enumerate(("UP", "DOWN", "UP")):
        records.append(
            {
                "type": "action_result",
                "sequence": sequence,
                "level": 0,
                "turn_in_level": sequence,
                "decision": {"action": action, "row": None, "col": None},
                "before": observations[sequence],
                "after": observations[sequence + 1],
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def write_long_transcript(path: Path, turns: int) -> None:
    records = []
    for sequence in range(turns):
        records.append(
            {
                "type": "action_result",
                "sequence": sequence,
                "level": 0,
                "turn_in_level": sequence,
                "decision": {"action": "UP", "row": None, "col": None},
                "before": _observation([[sequence % 2]]),
                "after": _observation([[(sequence + 1) % 2]]),
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _observation(grid: list[list[int]]) -> dict:
    return {
        "grid": grid,
        "available_actions": ["UP", "DOWN"],
        "levels_completed": 0,
        "state": "NOT_FINISHED",
    }


@pytest.mark.asyncio
async def test_long_shadow_uses_one_persistent_production_actor_without_arc_calls(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    write_transcript(transcript)
    llm = FakeLLMClient(
        scripted_responses=[
            response("UP", "a"),
            response("DOWN", "b"),
            response("UP", "c"),
        ]
    )

    report = await execute_production_shadow_replay(
        transcript=transcript,
        destination=tmp_path / "result",
        llm=llm,
        game_id="game",
        mode="single",
        treatment="control",
    )

    assert report["controller"] == "tycho_nooa.agent.TychoNooaAgent"
    assert report["actor_instances"] == 1
    assert report["environment_actions_executed"] == 0
    assert report["turns"] == 3
    assert report["historical_action_agreement"] == 3
    assert [row["provider_calls"] for row in report["trajectory"]] == [1, 1, 1]
    assert [row["input_tokens"] for row in report["trajectory"]] == [10, 10, 10]
    assert all(row["provider_request_bytes"] > 0 for row in report["trajectory"])
    evidence = [row["available_exact_evidence_bytes"] for row in report["trajectory"]]
    assert evidence == sorted(evidence)
    assert len(set(evidence)) == 3


@pytest.mark.asyncio
async def test_shadow_marks_action_divergence_without_fabricating_counterfactuals(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    write_transcript(transcript)
    llm = FakeLLMClient(
        scripted_responses=[
            response("DOWN", "a"),
            response("DOWN", "b"),
            response("UP", "c"),
        ]
    )

    report = await execute_production_shadow_replay(
        transcript=transcript,
        destination=tmp_path / "result",
        llm=llm,
        game_id="game",
        mode="single",
        treatment="control",
    )

    assert report["first_action_divergence_sequence"] == 0
    assert report["claim_ceiling"].startswith("off-policy shadow")
    assert report["trajectory"][1]["after_action_divergence"] is True


@pytest.mark.asyncio
async def test_old_turn_projection_is_bounded_while_exact_evidence_keeps_growing(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    write_long_transcript(transcript, 12)
    llm = FakeLLMClient(
        scripted_responses=[response("UP", f"call-{index}") for index in range(12)]
    )

    report = await execute_production_shadow_replay(
        transcript=transcript,
        destination=tmp_path / "result",
        llm=llm,
        game_id="game",
        mode="single",
        treatment="control",
    )

    request_sizes = [row["provider_request_bytes"] for row in report["trajectory"]]
    bounded_tail = request_sizes[4:]
    evidence = [row["available_exact_evidence_bytes"] for row in report["trajectory"]]
    assert max(bounded_tail) - min(bounded_tail) < 1_000
    assert evidence[-1] > evidence[4]


@pytest.mark.asyncio
async def test_shadow_can_freeze_a_bounded_prefix(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    write_long_transcript(transcript, 5)
    llm = FakeLLMClient(
        scripted_responses=[response("UP", f"call-{index}") for index in range(2)]
    )

    report = await execute_production_shadow_replay(
        transcript=transcript,
        destination=tmp_path / "result",
        llm=llm,
        game_id="game",
        mode="single",
        treatment="control",
        max_turns=2,
    )

    assert report["turns"] == 2
    assert [row["sequence"] for row in report["trajectory"]] == [0, 1]
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_shadow_rejects_nonpositive_prefix_before_model_calls(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    write_long_transcript(transcript, 2)
    llm = FakeLLMClient(scripted_responses=[])

    with pytest.raises(ValueError, match="max_turns must be positive"):
        await execute_production_shadow_replay(
            transcript=transcript,
            destination=tmp_path / "result",
            llm=llm,
            game_id="game",
            mode="single",
            treatment="control",
            max_turns=0,
        )

    assert llm.call_count == 0
