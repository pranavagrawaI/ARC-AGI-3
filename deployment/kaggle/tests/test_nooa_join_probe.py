from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

CANDIDATE = Path(__file__).resolve().parents[1] / "s0-nooa-join-candidate"
PROBE = CANDIDATE / "nooa_public_probe.py"


def load_probe(monkeypatch, *, games: str = "ls20,s5i5,cd82") -> ModuleType:
    monkeypatch.setenv("TYCHO_PUBLIC_GAME_IDS", games)
    monkeypatch.setenv("TYCHO_KAGGLE_MAX_ACTIONS", "8")
    spec = importlib.util.spec_from_file_location(
        f"nooa_public_probe_{games.replace(',', '_')}", PROBE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_recording(path: Path, *, game: str, frames: list[list[list[int]]]) -> None:
    lines = []
    for frame in frames:
        lines.append(
            json.dumps(
                {
                    "timestamp": "2026-09-04T00:00:00+00:00",
                    "data": {
                        "game_id": game,
                        "frame": frame,
                        "state": "NOT_FINISHED",
                        "levels_completed": 0,
                        "action_input": {"id": 0, "data": {}, "reasoning": None},
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")


def write_session(
    root: Path, *, game: str, actions: list[str], suffix: str = "session"
) -> None:
    session = root / f"{game}-{suffix}"
    (session / "workspace" / game).mkdir(parents=True)
    ipc = session / "ipc"
    ipc.mkdir()
    (session / "worker-metadata.json").write_text(
        json.dumps({"join": "s0-nooa-join", "chassis": "q3c-mdfiles-after"})
    )
    (ipc / "actions.jsonl").write_text(json.dumps({"actions": actions}) + "\n")
    (ipc / "states.jsonl").write_text(json.dumps({"grid_rows": ["01"]}) + "\n")


def test_recording_id_zero_with_grid_change_is_play(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_probe(monkeypatch, games="ls20")
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    write_recording(
        recordings / "ls20.myagent.2.rec.recording.jsonl",
        game="ls20",
        frames=[[[0, 1], [0, 0]], [[0, 0], [0, 1]]],
    )
    summary = module.summarize_recordings(recordings, require_complete=True)
    assert summary["total_recorded_action_results"] == 2
    assert summary["total_grid_hash_changes"] == 1
    assert summary["games"]["ls20"]["recording_action_input_ids"] == [0, 0]
    assert "solver_action_ids" not in summary["games"]["ls20"]
    assert summary["missing_games"] == []


def test_partial_recordings_survive_when_complete_is_not_required(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_probe(monkeypatch, games="ls20,s5i5")
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    write_recording(
        recordings / "ls20.myagent.2.rec.recording.jsonl",
        game="ls20",
        frames=[[[0]], [[1]]],
    )
    summary = module.summarize_recordings(recordings, require_complete=False)
    assert summary["missing_games"] == ["s5i5"]
    assert summary["game_count"] == 1


def test_complete_run_still_requires_every_expected_game(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_probe(monkeypatch, games="ls20,s5i5")
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    write_recording(
        recordings / "ls20.myagent.2.rec.recording.jsonl",
        game="ls20",
        frames=[[[0]], [[1]]],
    )
    try:
        module.summarize_recordings(recordings, require_complete=True)
    except RuntimeError as error:
        assert "recording game IDs differ" in str(error)
    else:
        raise AssertionError("missing games must fail a complete run")


def test_ipc_counts_submitted_env_actions(tmp_path: Path, monkeypatch) -> None:
    module = load_probe(monkeypatch, games="ls20")
    write_session(tmp_path, game="ls20", actions=["LEFT", "UP"])
    summary = module.summarize_nooa_sessions(tmp_path, require_complete=True)
    assert summary["total_submitted_env_actions"] == 2
    assert summary["games"]["ls20"]["submitted_action_names"] == ["LEFT", "UP"]
    assert summary["missing_games"] == []
    assert summary["games"]["ls20"]["episode_count"] == 1


def test_multiple_sessions_after_game_over_are_one_game(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_probe(monkeypatch, games="ls20")
    write_session(tmp_path, game="ls20", actions=["LEFT"], suffix="ep0")
    write_session(tmp_path, game="ls20", actions=["UP", "DOWN"], suffix="ep1")
    summary = module.summarize_nooa_sessions(tmp_path, require_complete=True)
    assert summary["games"]["ls20"]["episode_count"] == 2
    assert summary["total_submitted_env_actions"] == 3
    assert summary["games"]["ls20"]["submitted_action_names"] == [
        "LEFT",
        "UP",
        "DOWN",
    ]
