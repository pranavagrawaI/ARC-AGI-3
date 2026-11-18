from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from prefix import PrefixError, project_transitions  # noqa: E402  # identity path

FIXTURES = HERE / "fixtures"


def write_transition(path: Path, **row) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def test_projects_host_contract_transitions(tmp_path: Path) -> None:
    path = tmp_path / "ipc" / "transitions.jsonl"
    write_transition(
        path,
        step=1,
        turn=0,
        action="RIGHT",
        before={"grid": [[12, 4, 4]], "state": "NOT_FINISHED", "levels_completed": 0, "available_actions": ["UP", "DOWN", "LEFT", "RIGHT"]},
        after={"grid": [[4, 12, 4]], "state": "NOT_FINISHED", "levels_completed": 0, "available_actions": ["UP", "DOWN", "LEFT", "RIGHT"]},
    )
    case = project_transitions(path)
    view = case.inspection_view()
    assert view["frontier"] == 1
    assert view["visible_transitions"][0]["action"] == "RIGHT"
    assert view["visible_transitions"][0]["before"]["grid"] == [[12, 4, 4]]
    assert "grid_rows" not in json.dumps(view)
    assert "collision" not in json.dumps(view)


def test_rejects_empty_and_non_chronology(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(PrefixError):
        project_transitions(empty)
    bad = tmp_path / "bad.jsonl"
    write_transition(bad, step=2, turn=0, action="UP", before={"grid": [[1]]}, after={"grid": [[1]]})
    with pytest.raises(PrefixError):
        project_transitions(bad)
