from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from arc_host import cli
from arc_host.cli import _load_environment
from arc_host.process import ProtocolError


def test_explicit_environment_file_loads_without_overriding_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / ".env"
    environment.write_text(
        "ARC_API_KEY=from-file\nARC_LLM_MODEL=from-file-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ARC_API_KEY", "from-shell")
    monkeypatch.delenv("ARC_LLM_MODEL", raising=False)

    _load_environment(environment)

    assert os.environ["ARC_API_KEY"] == "from-shell"
    assert os.environ["ARC_LLM_MODEL"] == "from-file-model"


def test_missing_explicit_environment_file_fails_before_a_run(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="environment file does not exist"):
        _load_environment(tmp_path / "missing.env")


def test_primary_run_error_is_preserved_and_summarized_without_environment_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EnvironmentWithoutClose:
        pass

    class FakeArcade:
        def __init__(self, **_: object) -> None:
            pass

        def make(self, *_: object, **__: object) -> EnvironmentWithoutClose:
            return EnvironmentWithoutClose()

    class FakeWorker:
        closed = False

        def __init__(self, *_: object, **__: object) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    run_dir = tmp_path / "run"
    worker = FakeWorker()
    monkeypatch.setenv("ARC_API_KEY", "test-key")
    monkeypatch.setattr(cli.arc_agi, "Arcade", FakeArcade)
    monkeypatch.setattr(cli, "JsonlWorker", lambda *_args, **_kwargs: worker)
    def fail_after_progress(_environment: object, solver: object, **_kwargs: object):
        solver.actions_recorded = 21
        solver.latest_levels_completed = 0
        raise ProtocolError("primary")

    monkeypatch.setattr(cli, "run_episode", fail_after_progress)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arc-host",
            "--game",
            "fake-game",
            "--run-dir",
            str(run_dir),
            "--",
            "fake-worker",
        ],
    )

    with pytest.raises(ProtocolError, match="primary"):
        cli.main()

    assert worker.closed is True
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "exception"
    assert summary["actions"] == 21
    assert summary["levels_completed"] == 0
    assert summary["error"] == {
        "type": "ProtocolError",
        "message": "primary",
    }
    transcript = [
        json.loads(line)
        for line in (run_dir / "transcript.jsonl").read_text().splitlines()
    ]
    assert transcript == [
        {
            "actions": 21,
            "error": {"message": "primary", "type": "ProtocolError"},
            "levels_completed": 0,
            "type": "error",
        }
    ]
