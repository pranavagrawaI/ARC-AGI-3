"""Hermetic Python 3.14 host -> Python 3.12 Tycho-NOOA worker smoke."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from arc_host.process import JsonlWorker
from arc_host.runner import run_episode

ROOT = Path(__file__).resolve().parents[3]
SOLVER = ROOT / "solvers/tycho_nooa"


def frame(value: int, *, actions: tuple[int, ...], levels: int = 0):
    return SimpleNamespace(
        frame=np.asarray([[[value, value + 1], [value + 2, value + 3]]]),
        available_actions=actions,
        levels_completed=levels,
        state="NOT_FINISHED",
        full_reset=False,
    )


class Environment:
    def __init__(self):
        self.frames = iter([frame(0, actions=(1,)), frame(4, actions=(), levels=1)])

    def reset(self):
        return next(self.frames)

    def step(self, action, data=None, reasoning=None):
        assert action.value == 1
        assert data is None
        return next(self.frames)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tycho-nooa-ipc-") as directory:
        root = Path(directory)
        environment = {
            key: os.environ[key]
            for key in ("HOME", "LANG", "PATH", "SSL_CERT_FILE")
            if key in os.environ
        }
        environment["TYCHO_NOOA_SCRIPTED_TOOL_CALLS"] = json.dumps(
            [
                {
                    "tool": "write_file",
                    "arguments": {
                        "path": "notes/ipc.md",
                        "content": "This crossed the Python 3.14/3.12 boundary.",
                    },
                },
                {"tool": "take_action", "arguments": {"action": "UP"}},
            ]
        )
        with JsonlWorker(
            [
                "uv",
                "run",
                "--frozen",
                "--project",
                str(SOLVER),
                "tycho-nooa-worker",
            ],
            stdout_log=root / "worker-stdout.log",
            stderr_log=root / "worker-stderr.log",
            timeout_seconds=30,
            environment=environment,
        ) as worker:
            result = run_episode(
                Environment(),
                worker,
                game_id="fake-game",
                seed=0,
                workspace_root=root / "workspace",
                max_actions=2,
                stop_after_levels=1,
            )

        note = root / "workspace/fake/notes/ipc.md"
        assert result.stop_reason == "level_limit"
        assert result.actions == 1
        assert note.read_text() == "This crossed the Python 3.14/3.12 boundary."
        assert not (root / "worker-stderr.log").read_text()
        print("host-worker IPC smoke passed: 1 action, persistent note preserved")


if __name__ == "__main__":
    main()
