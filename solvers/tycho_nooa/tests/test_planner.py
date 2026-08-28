from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.planner import PlannerIntegration, _valid_action, planner_command


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("python plan.py", "python plan.py"),
        ("python3 plan.py bfs", "python plan.py bfs"),
        ('subprocess.run(["python", "plan.py", "astar"])', "python plan.py astar"),
        ('runpy.run_path("plan.py")', "python plan.py"),
        ("python notes/plan.py auto", None),
        ("python plan.py auto unexpected", None),
        ("python 'plan.py", None),
        ('runpy.run_path("notes/plan.py")', None),
        ('subprocess.run(["python", "other.py"])', None),
        ("subprocess.run(", None),
        ("", None),
    ],
)
def test_planner_command_recognizes_only_seeded_planner_execution(
    source: str, expected: str | None
) -> None:
    assert planner_command(source) == expected


def test_t1_rejects_unanchored_candidate_and_revokes_stale_plan(
    tmp_path: Path,
) -> None:
    workspace = GameWorkspace(
        "test-game",
        root=str(tmp_path / "canonical"),
        available_actions=["UP"],
    )
    workspace.record(
        [[0, 1]],
        level=0,
        turn_in_level=0,
        available=["UP"],
    )
    canonical = workspace.dir / "notes" / "validated_plan.json"
    canonical.write_text('{"status":"stale"}', encoding="utf-8")

    candidate_workspace = tmp_path / "candidate"
    candidate = candidate_workspace / "notes" / "validated_plan.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "validated",
                "method": "astar",
                "start": {
                    "level": 0,
                    "turn": 0,
                    "grid_sha256": "0" * 64,
                },
                "world_model_sha256": hashlib.sha256(
                    (workspace.dir / "world_model.py").read_bytes()
                ).hexdigest(),
                "plan_length": 1,
                "actions": [{"action": "UP", "row": None, "col": None}],
                "expected_grid_sha256": ["1" * 64],
                "validation": "canonical world_model replay reached level_complete",
            }
        ),
        encoding="utf-8",
    )

    result = PlannerIntegration(workspace).publish_candidate(
        candidate_workspace,
        "PLANNER_STATUS: plan_found (elapsed_s=0.01, length=1)",
    )

    assert result == (
        "PLANNER_PUBLICATION: rejected "
        "(validated artifact has the wrong current-grid hash)"
    )
    assert not canonical.exists()


@pytest.mark.parametrize("candidate_kind", ["absent", "oversized"])
def test_t1_rejects_unreadable_or_oversized_candidate_before_publication(
    tmp_path: Path,
    candidate_kind: str,
) -> None:
    workspace = GameWorkspace(
        "test-game",
        root=str(tmp_path / "canonical"),
        available_actions=["UP"],
    )
    workspace.record([[0]], level=0, turn_in_level=0, available=["UP"])
    canonical = workspace.dir / "notes" / "validated_plan.json"
    canonical.write_text('{"status":"stale"}', encoding="utf-8")
    candidate_workspace = tmp_path / "candidate"
    candidate = candidate_workspace / "notes" / "validated_plan.json"
    if candidate_kind == "oversized":
        candidate.parent.mkdir(parents=True)
        with candidate.open("wb") as stream:
            stream.seek(1_000_000)
            stream.write(b"x")

    result = PlannerIntegration(workspace).publish_candidate(
        candidate_workspace,
        "PLANNER_STATUS: plan_found (elapsed_s=0.01, length=1)",
    )

    assert result.startswith("PLANNER_PUBLICATION: rejected (")
    assert not canonical.exists()


def test_planner_refuses_a_workspace_script_that_differs_from_the_seed(
    tmp_path: Path,
) -> None:
    workspace = GameWorkspace(
        "test-game",
        root=str(tmp_path),
        available_actions=["UP"],
    )
    (workspace.dir / "plan.py").write_text("print('forged')", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the pinned seeded planner"):
        PlannerIntegration(workspace).seeded_plan_bytes()


@pytest.mark.parametrize(
    ("action", "available", "expected"),
    [
        (None, {"UP"}, False),
        ({"action": 1}, {"UP"}, False),
        ({"action": ""}, {"UP"}, False),
        ({"action": "UP", "row": None, "col": None}, {"UP"}, True),
        ({"action": "UP", "row": 0, "col": 0}, {"UP"}, False),
        ({"action": "DOWN", "row": None, "col": None}, {"UP"}, False),
        ({"action": "CLICK", "row": 0, "col": 63}, {"CLICK"}, True),
        ({"action": "CLICK", "row": -1, "col": 63}, {"CLICK"}, False),
        ({"action": "CLICK", "row": 0, "col": 64}, {"CLICK"}, False),
        ({"action": "CLICK", "row": None, "col": None}, {"CLICK"}, False),
    ],
)
def test_t1_action_validation_matches_host_legality_contract(
    action: Any,
    available: set[str],
    expected: bool,
) -> None:
    assert _valid_action(action, frozenset(available)) is expected


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("not_object", "must be an object"),
        ("schema", "wrong schema or status"),
        ("status", "wrong schema or status"),
        ("method", "unknown planning method"),
        ("validation", "no canonical replay claim"),
        ("missing_start", "no start anchor"),
        ("malformed_start", "malformed start anchor"),
        ("wrong_turn", "not anchored to the current frame"),
        ("wrong_grid", "wrong current-grid hash"),
        ("wrong_model", "wrong world-model hash"),
        ("length", "inconsistent plan lengths"),
        ("action", "invalid action"),
        ("expected_grid", "invalid grid hash"),
    ],
)
def test_t1_rejects_each_invalid_plan_contract(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    workspace = GameWorkspace(
        "test-game",
        root=str(tmp_path / "canonical"),
        available_actions=["UP"],
    )
    workspace.record([[0]], level=0, turn_in_level=0, available=["UP"])
    artifact: Any = _valid_artifact(workspace)
    if case == "not_object":
        artifact = []
    else:
        assert isinstance(artifact, dict)
        if case == "schema":
            artifact["schema_version"] = 2
        elif case == "status":
            artifact["status"] = "candidate"
        elif case == "method":
            artifact["method"] = "unknown"
        elif case == "validation":
            artifact["validation"] = "claimed"
        elif case == "missing_start":
            artifact["start"] = None
        elif case == "malformed_start":
            artifact["start"]["turn"] = "0"
        elif case == "wrong_turn":
            artifact["start"]["turn"] = 1
        elif case == "wrong_grid":
            artifact["start"]["grid_sha256"] = "0" * 64
        elif case == "wrong_model":
            artifact["world_model_sha256"] = "0" * 64
        elif case == "length":
            artifact["plan_length"] = 0
        elif case == "action":
            artifact["actions"] = [{"action": "DOWN", "row": None, "col": None}]
        elif case == "expected_grid":
            artifact["expected_grid_sha256"] = ["not-a-digest"]
    candidate = tmp_path / "candidate" / "notes" / "validated_plan.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps(artifact))

    result = PlannerIntegration(workspace).publish_candidate(
        candidate.parents[1],
        "PLANNER_STATUS: plan_found (elapsed_s=0.01, length=1)",
    )

    assert result.startswith("PLANNER_PUBLICATION: rejected")
    assert message in result
    assert not (workspace.dir / "notes" / "validated_plan.json").exists()


def test_t1_accepts_complete_current_plan_contract(tmp_path: Path) -> None:
    workspace = GameWorkspace(
        "test-game",
        root=str(tmp_path / "canonical"),
        available_actions=["UP"],
    )
    workspace.record([[0]], level=0, turn_in_level=0, available=["UP"])
    artifact = _valid_artifact(workspace)
    candidate = tmp_path / "candidate" / "notes" / "validated_plan.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps(artifact))

    result = PlannerIntegration(workspace).publish_candidate(
        candidate.parents[1],
        "PLANNER_STATUS: plan_found (elapsed_s=0.01, length=1)",
    )

    assert result == "PLANNER_PUBLICATION: accepted by T1"
    assert json.loads(
        (workspace.dir / "notes" / "validated_plan.json").read_text()
    ) == artifact


@pytest.mark.parametrize(
    "pointer",
    [
        {"path": "/tmp/outside.txt", "level": 0, "turn": 0},
        {"path": "../outside.txt", "level": 0, "turn": 0},
        {"path": "missing.txt", "level": 0, "turn": 0},
    ],
)
def test_t1_rejects_unavailable_current_frame(
    tmp_path: Path, pointer: dict[str, object]
) -> None:
    workspace = GameWorkspace(
        "test-game",
        root=str(tmp_path / "canonical"),
        available_actions=["UP"],
    )
    workspace.record([[0]], level=0, turn_in_level=0, available=["UP"])
    (workspace.dir / ".current_frame.json").write_text(json.dumps(pointer))
    candidate = tmp_path / "candidate" / "notes" / "validated_plan.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps(_valid_artifact(workspace)))

    result = PlannerIntegration(workspace).publish_candidate(
        candidate.parents[1],
        "PLANNER_STATUS: plan_found (elapsed_s=0.01, length=1)",
    )

    assert result == "PLANNER_PUBLICATION: rejected (current frame is unavailable)"


def _valid_artifact(workspace: GameWorkspace) -> dict[str, object]:
    grid_hash = hashlib.sha256(
        np.asarray([[0]], dtype=np.int16).tobytes()
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": "validated",
        "method": "astar",
        "start": {"level": 0, "turn": 0, "grid_sha256": grid_hash},
        "world_model_sha256": hashlib.sha256(
            (workspace.dir / "world_model.py").read_bytes()
        ).hexdigest(),
        "plan_length": 1,
        "actions": [{"action": "UP", "row": None, "col": None}],
        "expected_grid_sha256": ["1" * 64],
        "validation": "canonical world_model replay reached level_complete",
    }
