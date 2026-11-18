#!/usr/bin/env python3
"""Protocol-v3 worker that plays NOOA's harness role against a CodeAct agent.

The official MyAgent loop owns the environment. This process translates
Turn/ActionResult messages into ``ipc/states.jsonl`` and reads
``ipc/actions.jsonl``. A submitted batch is drained one official decision at a
time; the next state line is written only after the batch is fully observed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from tycho_nooa.models import ActionDecision
from tycho_nooa.protocol import (
    HOST_PREFIX,
    PROTOCOL_VERSION,
    WORKER_PREFIX,
    ActionResultMessage,
    CloseMessage,
    FinishMessage,
    StartGameMessage,
    TurnMessage,
    parse_host_message,
)

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / "upstream"
HEADLESS = HERE / "nooa_headless.py"
STUB = HERE / "stub_agent.py"
GRID_HEX = "0123456789abcdef"
CLICK = re.compile(r"^CLICK\s+(\d+)\s+(\d+)$")
AGENT_COMMAND_ENV = "NOOA_AGENT_COMMAND"
STUB_BATCH_ENV = "NOOA_STUB_BATCH"
READY_TIMEOUT_ENV = "NOOA_AGENT_READY_TIMEOUT_SECONDS"
DEFAULT_READY_TIMEOUT = 60.0


class WorkerError(RuntimeError):
    """Protocol or IPC failure visible to LocalSolver as type=error."""


def _write(stdout: TextIO, kind: str, **fields: Any) -> None:
    payload = {"protocol_version": PROTOCOL_VERSION, "type": kind, **fields}
    stdout.write(WORKER_PREFIX + json.dumps(payload, separators=(",", ":")) + "\n")
    stdout.flush()


def _grid_rows(grid: list[list[int]]) -> list[str]:
    return ["".join(GRID_HEX[v] if 0 <= v <= 15 else "?" for v in row) for row in grid]


def _decision(action: str) -> ActionDecision:
    raw = action.strip()
    match = CLICK.match(raw)
    if match:
        col, row = int(match.group(1)), int(match.group(2))
        return ActionDecision(action="CLICK", row=row, col=col)
    name = raw.split()[0] if raw else ""
    if name == "USE":
        name = "SPACEBAR"
    return ActionDecision(action=name)


def _outcome(before: ActionResultMessage) -> str:
    after = before.after
    if after.levels_completed > before.before.levels_completed:
        return "level_up"
    state = after.state.upper()
    if state in {"WIN", "GAME_OVER"}:
        return "terminal"
    if after.grid == before.before.grid:
        return "no_impact"
    return "move"


def _agent_command(run_dir: Path) -> list[str]:
    raw = os.environ.get(AGENT_COMMAND_ENV)
    if raw:
        command = json.loads(raw)
        if not isinstance(command, list) or not command:
            raise WorkerError(f"{AGENT_COMMAND_ENV} must be a JSON list of argv")
        return [str(part) for part in command]
    if HEADLESS.is_file():
        return [sys.executable, str(HEADLESS), "--run-dir", str(run_dir)]
    return [sys.executable, str(STUB)]


class HostSession:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.ipc = run_dir / "ipc"
        self.states_path = self.ipc / "states.jsonl"
        self.actions_path = self.ipc / "actions.jsonl"
        self.observations_path = self.ipc / "observations.json"
        self.ipc.mkdir(parents=True, exist_ok=True)
        self.states_path.touch()
        self.actions_path.touch()
        self._actions_pos = 0
        self._queue: list[str] = []
        self._batch_results: list[dict[str, Any]] = []
        self._nooa_turn = 0
        self._env_steps = 0
        self._process: subprocess.Popen[str] | None = None
        self.finished = False

    def start(self, message: StartGameMessage) -> dict[str, Any]:
        if self._process is not None:
            raise WorkerError("a game is already active in this worker")
        command = _agent_command(self.run_dir)
        Path(message.workspace_root).mkdir(parents=True, exist_ok=True)
        (Path(message.workspace_root) / message.game_id).mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            command,
            cwd=self.run_dir,
            env=dict(os.environ),
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
            text=True,
        )
        self._wait_ready()
        return {
            "worker_id": str(os.getpid()),
            "metadata": {
                "agent": "nooa.MdArcSolverAgent",
                "chassis": "nooa-pin upstream examples/arc_agi_3 (mdfiles variant)",
                "join": "s0-nooa-upstream-join",
                "command": command,
                "workspace": str(message.workspace_root),
            },
        }

    def decide(self, message: TurnMessage) -> dict[str, Any]:
        if self.finished:
            raise WorkerError("the game has already finished")
        if self._queue:
            action = self._queue.pop(0)
            return {
                "sequence": message.sequence,
                "decision": _decision(action).model_dump(),
            }
        self._write_state(message)
        batch = self._wait_batch(message.decision_timeout_seconds or 30.0)
        actions = list(batch.get("actions") or [])
        if not actions:
            raise WorkerError("agent submitted an empty action batch")
        first, rest = actions[0], actions[1:]
        self._queue = rest
        return {
            "sequence": message.sequence,
            "decision": _decision(first).model_dump(),
        }

    def observe(self, message: ActionResultMessage) -> dict[str, Any]:
        self._env_steps += 1
        self._batch_results.append(
            {
                "action": message.decision.action,
                "outcome": _outcome(message),
                "levels_completed": message.after.levels_completed,
                "state": message.after.state,
            }
        )
        if not self._queue:
            self._nooa_turn += 1
        return {"sequence": message.sequence}

    def finish(self, _message: FinishMessage) -> dict[str, Any]:
        self.finished = True
        return {}

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._process = None

    def _wait_ready(self) -> None:
        timeout = float(os.environ.get(READY_TIMEOUT_ENV, DEFAULT_READY_TIMEOUT))
        marker = self.ipc / "agent_ready"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if marker.exists():
                return
            if self._process is not None and self._process.poll() is not None:
                raise WorkerError(
                    f"agent exited {self._process.returncode} before agent_ready"
                )
            time.sleep(0.02)
        raise WorkerError(f"timed out waiting for agent_ready ({timeout}s)")

    def _wait_batch(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.actions_path.open() as handle:
                handle.seek(self._actions_pos)
                line = handle.readline()
                if line.endswith("\n"):
                    self._actions_pos = handle.tell()
                    if line.strip():
                        return json.loads(line)
            if self._process is not None and self._process.poll() is not None:
                raise WorkerError(
                    f"agent exited {self._process.returncode} before submit_actions"
                )
            time.sleep(0.02)
        raise WorkerError(f"timed out waiting for submit_actions ({timeout}s)")

    def _write_state(self, message: TurnMessage) -> None:
        observation = message.observation
        entry = {
            "turn": self._nooa_turn,
            "step": self._env_steps,
            "level": message.level,
            "state": observation.state,
            "levels_completed": observation.levels_completed,
            "available_actions": list(observation.available_actions),
            "action_results": list(self._batch_results),
            "grid_rows": _grid_rows(observation.grid),
            "diff_summary": "first state"
            if self._nooa_turn == 0
            else f"turn {self._nooa_turn}",
        }
        with self.states_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        self.observations_path.write_text(
            json.dumps(
                {
                    "turn": self._nooa_turn,
                    "step": self._env_steps,
                    "levels_completed": observation.levels_completed,
                    "known_from_here": {},
                    "last_retrodiction": None,
                },
                indent=2,
            )
        )
        self._batch_results = []


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    session: HostSession | None = None
    try:
        while line := stdin.readline():
            if not line.startswith(HOST_PREFIX):
                continue
            sequence: int | None = None
            try:
                message = parse_host_message(line[len(HOST_PREFIX) :])
                sequence = getattr(message, "sequence", None)
                if isinstance(message, CloseMessage):
                    return
                if isinstance(message, StartGameMessage):
                    session = HostSession(Path.cwd())
                    _write(stdout, "ready", **session.start(message))
                elif session is None:
                    raise WorkerError("start_game is required before this message")
                elif isinstance(message, TurnMessage):
                    _write(stdout, "decision", **session.decide(message))
                elif isinstance(message, ActionResultMessage):
                    _write(stdout, "observed", **session.observe(message))
                elif isinstance(message, FinishMessage):
                    _write(stdout, "finished", **session.finish(message))
            except Exception as error:  # noqa: BLE001 - protocol errors must be returned
                payload: dict[str, Any] = {
                    "message": f"{type(error).__name__}: {error}"
                }
                if sequence is not None:
                    payload["sequence"] = sequence
                _write(stdout, "error", **payload)
                return
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    serve()
