"""Exercise level-boundary batching through the real deployment join.

From deployment/kaggle:
  uv run --frozen python ../../analysis/local-probes/20260905T-stall-authority-q6/review/repro_join.py

Exits 1 if either invariant fails. Uses the existing stub only in place of the
LLM agent; MyAgent, LocalSolver, and the subprocess worker are production code.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "deployment/kaggle/tests"))

import pytest
from arcengine import GameAction

from test_join_qualification import MOVES, STUB, build_agent, drive, frame


def main() -> int:
    failures: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch, tempfile.TemporaryDirectory() as temp:
        for key, value in {
            "NOOA_AGENT_COMMAND": json.dumps([sys.executable, str(STUB)]),
            "NOOA_AGENT_READY_TIMEOUT_SECONDS": "10",
            "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS": "15",
            "NOOA_STUB_BATCH": "UP,DOWN,LEFT",
        }.items():
            monkeypatch.setenv(key, value)
        root = Path(temp)
        first = frame([[0, 1], [0, 0]], actions=MOVES)
        mid = frame([[0, 1], [1, 0]], actions=MOVES)
        boundary = frame([[3, 3], [3, 3]], levels_completed=1, actions=MOVES)
        agent = build_agent(monkeypatch, root, first)
        try:
            emitted = drive(agent, [first, mid])
            assert emitted == [GameAction.ACTION1, GameAction.ACTION2]
            agent.frames.append(boundary)
            assert not agent.is_done(agent.frames, boundary)
            states_path = next((root / "workspaces").glob("*/ipc/states.jsonl"))
            states = [json.loads(line) for line in states_path.read_text().splitlines()]
            print("state steps immediately after boundary observe:", [s["step"] for s in states])
            try:
                assert len(states) == 2 and states[-1]["step"] == 2 and states[-1]["levels_completed"] == 1, (
                    "boundary observation must be published before the next decide"
                )
            except AssertionError as error:
                failures.append(str(error))

            next_action = agent.choose_action(agent.frames, boundary)
            agent.action_counter += 1
            print("first action on new level:", next_action.name)
            try:
                assert next_action is GameAction.ACTION1, (
                    "new level must receive fresh-batch UP, not queued old-level LEFT"
                )
            except AssertionError as error:
                failures.append(str(error))
        finally:
            agent.cleanup(scorecard=None)
    for failure in failures:
        print("FAIL:", failure)
    if not failures:
        print("PASS: boundary publication and queued-tail interruption")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
