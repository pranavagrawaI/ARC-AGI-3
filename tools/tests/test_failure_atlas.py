from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.failure_atlas_core import (
    Action,
    Observation,
    WorldModel,
    commitment_json,
    mine_builder_refreshes,
    mine_corpus,
    replay_packet,
    score_grid,
)

MODEL = """\
import numpy as np

def init_state(grid, level):
    return {"grid": np.asarray(grid).copy()}

def transition(state, action):
    result = {"grid": state["grid"].copy()}
    result["grid"][0, 0] += 1
    return result

def render(state):
    return state["grid"]

def outcome(state):
    return "ongoing"

def actions(state):
    return [{"action": "RIGHT"}, {"action": "LEFT"}]
"""


class TinyEnvironment:
    def __init__(
        self, *, corrupt_prefix: bool = False, initial_full_reset: bool = False
    ) -> None:
        self.corrupt_prefix = corrupt_prefix
        self.initial_full_reset = initial_full_reset
        self.runs: list[tuple[Action, ...]] = []

    def run(self, actions: tuple[Action, ...]) -> tuple[Observation, ...]:
        self.runs.append(actions)
        value = 0
        initial = observation(value)
        if self.initial_full_reset:
            initial = Observation(
                grid=initial.grid,
                available_actions=initial.available_actions,
                levels_completed=initial.levels_completed,
                state=initial.state,
                full_reset=True,
            )
        observations = [initial]
        for index, action in enumerate(actions):
            value += 1 if action.action == "RIGHT" else 2
            if self.corrupt_prefix and index == 0:
                value += 10
            observations.append(observation(value))
        return tuple(observations)


def observation(value: int) -> Observation:
    return Observation(
        grid=((value,),),
        available_actions=("RIGHT", "LEFT"),
        levels_completed=0,
        state="NOT_FINISHED",
        full_reset=False,
    )


def packet(sequence: int = 0) -> dict:
    prefix = [Action("RIGHT").to_json()] * sequence
    observations = [observation(index).to_json() for index in range(sequence + 1)]
    return {
        "packet_id": f"packet-{sequence}",
        "sequence": sequence,
        "level": 0,
        "initial_observation": observation(0).to_json(),
        "action_prefix": prefix,
        "observation_prefix": observations,
        "decision_observation": observation(sequence).to_json(),
        "available_action_labels": ["RIGHT", "LEFT"],
        "executed_action": Action("RIGHT").to_json(),
    }


def write_model(path: Path, source: str = MODEL) -> WorldModel:
    path.mkdir(parents=True)
    (path / "world_model.py").write_text(source)
    return WorldModel(path)


def write_refresh_call(
    root: Path,
    *,
    call_index: int,
    sequence: int,
    source: str,
    report: str | None = None,
    cost: float,
    tool_name: str | None = None,
    tool_arguments: dict | None = None,
) -> None:
    body = {
        "model": "fake-model",
        "messages": (
            [{"role": "user", "content": f"prefix\n{report}\nsuffix"}]
            if report is not None
            else []
        ),
        "tools": [],
        "output_model": None,
        "parameters": {},
    }
    request = {
        "call_index": call_index,
        "source": source,
        "decision_point": {
            "sequence": sequence,
            "level": 0,
            "turn_in_level": sequence,
            "phase": "decision",
        },
        "request_payload_commitment": commitment_json(body),
        "request_body": body,
    }
    tool_calls = []
    if tool_name is not None:
        tool_calls.append(
            {
                "name": tool_name,
                "arguments": json.dumps(tool_arguments or {}),
            }
        )
    response = {
        "status": "complete",
        "usage": {
            "cost": cost,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
        "tool_calls": tool_calls,
    }
    call = root / "calls" / f"{call_index:06d}"
    call.mkdir(parents=True)
    (call / "request.json").write_text(json.dumps(request))
    (call / "response.json").write_text(json.dumps(response))


def builder_report(*, model: str, action: str) -> str:
    return "\n".join(
        (
            "automatic world-model builder report (advisory; inspect or test it):",
            "confidence: high",
            f"model: {model}",
            "dynamics: simulation_accuracy=1.00",
            "outcome: ongoing",
            "outcome_verified: no",
            "plan: validated=no",
            f"recommended_action: {action}",
            "note: bounded fixture''')",
        )
    )


def test_unknown_cells_are_penalized_instead_of_counted_as_correct() -> None:
    score = score_grid(
        np.asarray([[1, -1], [2, 3]]),
        np.asarray([[1, 9], [2, 4]]),
    )

    assert score["claimed_cells"] == 3
    assert score["unknown_cells"] == 1
    assert score["prediction_coverage"] == pytest.approx(0.75)
    assert score["known_cell_accuracy"] == pytest.approx(2 / 3)
    assert score["unknown_penalized_cell_accuracy"] == pytest.approx(0.5)
    assert score["differing_claimed_cells"] == 1
    assert score["canonical_exact"] is False


def test_builder_refresh_miner_ranks_repeated_and_actionless_reports_without_calls(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    report_a = builder_report(model="movement", action="UP")
    report_b = builder_report(model="collision", action="none")
    for call_index, sequence, source, report, cost, tool, arguments in (
        (1, 1, "builder", None, 0.01, None, None),
        (2, 1, "actor", report_a, 0.001, "take_action", {"action": "UP"}),
        (3, 2, "builder", None, 0.02, None, None),
        (4, 2, "actor", report_a, 0.001, "take_action", {"action": "DOWN"}),
        (5, 3, "builder", None, 0.03, None, None),
        (6, 3, "actor", report_b, 0.001, "run_python", {"code": "pass"}),
    ):
        write_refresh_call(
            root,
            call_index=call_index,
            sequence=sequence,
            source=source,
            report=report,
            cost=cost,
            tool_name=tool,
            tool_arguments=arguments,
        )

    result = mine_builder_refreshes([root])

    assert result["model_calls_made"] == 0
    assert result["refresh_count"] == 3
    assert result["total_builder_cost_usd"] == pytest.approx(0.06)
    assert [row["candidate_signal"] for row in result["candidates"]] == [
        "exact_content_repeat",
        "no_action_recommendation",
    ]
    assert result["candidates"][0]["sequence"] == 2
    assert result["candidates"][0]["builder"] == {
        "call_count": 1,
        "cost_usd": 0.02,
        "missing_cost_calls": 0,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }
    assert result["candidates"][0]["immediate_actor_response"] == {
        "call_index": 4,
        "first_tool_call": "take_action",
        "take_action": "DOWN",
        "recommendation_relation": "diverged",
    }
    assert result["refreshes"][0]["report_fields"]["note"] == "bounded fixture"


def test_branch_replay_freshly_executes_every_simple_legal_action(tmp_path: Path) -> None:
    environment = TinyEnvironment()

    result = replay_packet(packet(), write_model(tmp_path / "model"), environment)

    branches = {item["action"]["action"]: item for item in result["branches"]}
    assert result["engine_prefix_parity"]["exact"] is True
    assert result["model_prefix"]["status"] == "exact"
    assert branches["RIGHT"]["score"]["failure"] is False
    assert branches["LEFT"]["score"]["failure"] is True
    assert branches["LEFT"]["score"]["grid"]["first_claimed_difference"] == {
        "row": 0,
        "col": 0,
        "predicted": 1,
        "actual": 2,
    }
    assert environment.runs == [(), (Action("RIGHT"),), (Action("LEFT"),)]


def test_full_recorded_prefix_must_match_before_a_branch_is_scored(
    tmp_path: Path,
) -> None:
    result = replay_packet(
        packet(sequence=1),
        write_model(tmp_path / "model"),
        TinyEnvironment(corrupt_prefix=True),
    )

    assert result["status"] == "invalid_engine_prefix_parity"
    assert result["engine_prefix_parity"]["first_difference"] == 1
    assert result["branches"] == []


def test_online_offline_initial_reset_marker_is_reported_not_scored_as_mechanics(
    tmp_path: Path,
) -> None:
    result = replay_packet(
        packet(),
        write_model(tmp_path / "model"),
        TinyEnvironment(initial_full_reset=True),
    )

    assert result["status"] == "scored"
    assert result["engine_prefix_parity"]["exact"] is True
    assert result["engine_prefix_parity"]["ignored_metadata_differences"] == [
        {
            "index": 0,
            "field": "full_reset",
            "recorded": False,
            "replayed": True,
            "reason": "online_offline_initial_reset_marker",
        }
    ]


def test_unknown_model_prefix_remains_partial_not_a_claimed_divergence(
    tmp_path: Path,
) -> None:
    source = MODEL.replace(
        'return {"grid": np.asarray(grid).copy()}',
        'return {"grid": np.full_like(np.asarray(grid), -1)}',
    ).replace('result["grid"][0, 0] += 1', "pass")

    result = replay_packet(
        packet(), write_model(tmp_path / "unknown-model", source), TinyEnvironment()
    )

    assert result["model_prefix"] == {
        "status": "partial_unknown",
        "first_divergence": None,
    }
    assert all(branch["status"] == "scored" for branch in result["branches"])
    assert all(branch["score"]["failure"] for branch in result["branches"])


def test_miner_physically_separates_predecision_packet_from_successor(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    workspace = run / "workspace" / "test-game"
    workspace.mkdir(parents=True)
    (workspace / "world_model.py").write_text(MODEL)
    initial = observation(0).to_json()
    after = observation(9).to_json()
    rows = [
        {
            "type": "start_game",
            "game_id": "test-game",
            "seed": 3,
            "initial_observation": initial,
        },
        {"type": "decision", "sequence": 0, "decision": Action("RIGHT").to_json()},
        {
            "type": "action_result",
            "sequence": 0,
            "level": 0,
            "turn_in_level": 0,
            "decision": Action("RIGHT").to_json(),
            "before": initial,
            "after": after,
        },
    ]
    (run / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    output = tmp_path / "corpus"
    manifest = mine_corpus(run, output)

    entry = manifest["packets"][0]
    admissible = json.loads((output / entry["admissible"]).read_text())
    oracle = json.loads((output / entry["recorded_oracle"]).read_text())
    assert "recorded_successor" not in admissible
    assert admissible["observation_prefix"] == [initial]
    assert oracle["recorded_successor"] == after
    assert admissible["model_source"] == "retrospective_final_workspace"
    assert (
        admissible["model_temporal_validity"]
        == "retrospective_final_candidate_not_historical"
    )


def test_miner_prefers_the_last_decision_phase_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    workspace = run / "workspace" / "test-game"
    workspace.mkdir(parents=True)
    (workspace / "world_model.py").write_text(MODEL + "\nFINAL = True\n")
    initial = observation(0).to_json()
    rows = [
        {
            "type": "start_game",
            "game_id": "test-game",
            "seed": 3,
            "initial_observation": initial,
        },
        {"type": "decision", "sequence": 0, "decision": Action("RIGHT").to_json()},
        {
            "type": "action_result",
            "sequence": 0,
            "level": 0,
            "turn_in_level": 0,
            "decision": Action("RIGHT").to_json(),
            "before": initial,
            "after": observation(1).to_json(),
        },
    ]
    (run / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    checkpoint = run / "decision-checkpoints"
    call = checkpoint / "calls" / "000001"
    call.mkdir(parents=True)
    content = MODEL.encode()
    digest = hashlib.sha256(content).hexdigest()
    blob = checkpoint / "blobs" / "sha256" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content)
    request = {
        "source": "actor",
        "decision_point": {
            "sequence": 0,
            "level": 0,
            "turn_in_level": 0,
            "phase": "decision",
        },
        "request_payload_commitment": "sha256:request",
        "workspace_commitment": "sha256:workspace",
    }
    manifest = {
        "directories": [],
        "files": [
            {
                "path": "world_model.py",
                "sha256": digest,
                "size": len(content),
            }
        ],
    }
    (call / "request.json").write_text(json.dumps(request))
    (call / "response.json").write_text(json.dumps({"status": "complete"}))
    (call / "workspace.json").write_text(json.dumps(manifest))

    output = tmp_path / "corpus"
    mined = mine_corpus(run, output)
    admissible = json.loads(
        (output / mined["packets"][0]["admissible"]).read_text()
    )

    assert admissible["model_source"] == "decision_checkpoint"
    assert admissible["model_temporal_validity"] == "exact_predecision"
    assert [ref["call_index"] for ref in admissible["checkpoint_refs"]] == [1]
    assert (output / "checkpoints" / "blobs" / "sha256" / digest).read_bytes() == content
