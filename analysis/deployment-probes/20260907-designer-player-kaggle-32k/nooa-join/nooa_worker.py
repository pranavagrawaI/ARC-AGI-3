#!/usr/bin/env python3
"""Protocol-v3 worker that plays NOOA's harness role against a CodeAct agent.

The official MyAgent loop owns the environment. This process translates
Turn/ActionResult messages into ``ipc/states.jsonl`` and reads
``ipc/actions.jsonl``. A submitted batch is drained one official decision at a
time; the next state line is written only after the batch is fully observed.
"""

from __future__ import annotations

import json
import hashlib
import ctypes
import signal
import os
import re
import subprocess
import sys
import time
import uuid
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
    ObservationPayload,
    StartGameMessage,
    TurnMessage,
    parse_host_message,
)

HERE = Path(__file__).resolve().parent
CHASSIS = HERE / "chassis"
HEADLESS = HERE / "headless.py"
STUB = HERE / "stub_agent.py"
sys.path.insert(0, str(CHASSIS))
from host_contract import HostEvidence, score_predicted_path  # noqa: E402
from step_check import bind_prediction, check_step  # noqa: E402

CLICK = re.compile(r"^CLICK\s+(\d+)\s+(\d+)$")
AGENT_COMMAND_ENV = "NOOA_AGENT_COMMAND"
STUB_BATCH_ENV = "NOOA_STUB_BATCH"
READY_TIMEOUT_ENV = "NOOA_AGENT_READY_TIMEOUT_SECONDS"
DEFAULT_READY_TIMEOUT = 60.0
WORKER_PID = os.getpid()


class WorkerError(RuntimeError):
    """Protocol or IPC failure visible to LocalSolver as type=error."""


def _write(stdout: TextIO, kind: str, **fields: Any) -> None:
    payload = {"protocol_version": PROTOCOL_VERSION, "type": kind, **fields}
    stdout.write(WORKER_PREFIX + json.dumps(payload, separators=(",", ":")) + "\n")
    stdout.flush()


def _observation(value) -> dict:
    return {
        "grid": value.grid,
        "state": value.state,
        "levels_completed": value.levels_completed,
        "available_actions": list(value.available_actions),
    }


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


def _die_with_worker() -> None:
    # Kaggle and local qualification run on Linux. Even SIGKILL of the protocol
    # worker must terminate the Actor rather than leave model calls orphaned.
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError("could not set parent-death signal")
    if os.getppid() != WORKER_PID:
        os.kill(os.getpid(), signal.SIGTERM)


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
        # Whether the state line for the current NOOA turn is already on disk.
        self._state_written = False
        self._nooa_turn = 0
        self._process: subprocess.Popen[str] | None = None
        self.finished = False
        self.evidence = HostEvidence(self.ipc)
        self._pending_action: str | None = None
        arm = os.environ.get('ARC_SOLVER_ARM', 'loop')
        if arm not in ('baseline', 'loop', 'mindset'):
            raise WorkerError('invalid ARC_SOLVER_ARM')
        self._step_checks = arm in ('loop', 'mindset')
        self._prediction: dict = {}

    def start(self, message: StartGameMessage) -> dict[str, Any]:
        if self._process is not None:
            raise WorkerError("a game is already active in this worker")
        manifest = json.loads((HERE / "CHASSIS.json").read_text())
        for relative, expected in manifest["files"].items():
            if hashlib.sha256((HERE / relative).read_bytes()).hexdigest() != expected:
                raise WorkerError(f"candidate drift: {relative}")
        flag = os.environ.get("ARC_STALL_REVIEW", "0")
        if flag not in ("0", "1"):
            raise WorkerError("ARC_STALL_REVIEW must be 0 or 1")
        (self.run_dir / "launch.json").write_text(
            json.dumps(
                {
                    "arm": "treatment" if flag == "1" else "control",
                    "solver_arm": os.environ.get('ARC_SOLVER_ARM', 'loop'),
                    "platform": "kaggle",
                    "contract_sha256": manifest["contract_sha256"],
                    "manifest": manifest,
                    "game_id": message.game_id,
                    "started_at": time.time(),
                    "run_id": os.environ.get("ARC_Q6_RUN_ID"),
                    "session_id": uuid.uuid4().hex,
                    "model_config": {
                        key: os.environ.get(key)
                        for key in (
                            "ARC_LLM_MODEL",
                            "ARC_LLM_MAX_TOKENS",
                            "ARC_LLM_EXTRA_BODY",
                            "ARC_LLM_CONTEXT_WINDOW",
                            "ARC_LLM_TIMEOUT_SECONDS",
                        )
                    },
                },
                indent=2,
            )
        )
        command = _agent_command(self.run_dir)
        Path(message.workspace_root).mkdir(parents=True, exist_ok=True)
        (Path(message.workspace_root) / message.game_id).mkdir(
            parents=True, exist_ok=True
        )
        self._process = subprocess.Popen(
            command,
            cwd=self.run_dir,
            env=dict(os.environ),
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
            text=True,
            preexec_fn=_die_with_worker,
        )
        self._wait_ready()
        return {
            "worker_id": str(os.getpid()),
            "metadata": {
                "agent": "nooa.MdArcSolverAgent",
                "chassis": manifest["identity"],
                "contract_sha256": manifest["contract_sha256"],
                "join": "q6-nooa-join",
                "command": command,
                "agent_pid": self._process.pid,
                "workspace": str(message.workspace_root),
            },
        }

    def decide(self, message: TurnMessage) -> dict[str, Any]:
        if self.finished:
            raise WorkerError("the game has already finished")
        if self._queue:
            action = self._queue.pop(0)
            self._pending_action = action
            return {
                "sequence": message.sequence,
                "decision": _decision(action).model_dump(),
            }
        # Normally ``observe`` has already published this turn's state the moment
        # the previous batch finished, so the agent's WAIT ended then rather than
        # here. This covers turn 0 and any path that did not reach ``observe``.
        if not self._state_written:
            self._write_state(message.observation, message.level)
        batch = self._wait_batch(message.decision_timeout_seconds or 30.0)
        if batch.get("turn") != self._nooa_turn:
            raise WorkerError("submitted batch does not match the current turn")
        actions = list(batch.get("actions") or [])
        if not actions:
            raise WorkerError("agent submitted an empty action batch")
        first, rest = actions[0], actions[1:]
        self.evidence.begin(batch)
        if self._step_checks:
            self._prediction = bind_prediction(self.ipc, batch)
        self._queue = rest
        self._pending_action = first
        self._state_written = False
        return {
            "sequence": message.sequence,
            "decision": _decision(first).model_dump(),
        }

    def observe(self, message: ActionResultMessage) -> dict[str, Any]:
        if self._pending_action is None:
            raise WorkerError("observation has no pending action")
        stop = self.evidence.observe(
            _observation(message.before),
            _observation(message.after),
            self._pending_action,
            len(self._queue),
        )
        if self._step_checks and not stop:
            transition = {'turn': self._nooa_turn, 'step': self.evidence.step,
                          'action': self._pending_action, 'before': _observation(message.before),
                          'after': _observation(message.after)}
            checked = check_step(self._prediction, transition)
            checked['skipped_actions'] = len(self._queue) if checked['status'] != 'matched' else 0
            with (self.ipc / 'step_checks.jsonl').open('a') as handle:
                handle.write(json.dumps(checked) + '\n')
            if checked['status'] != 'matched':
                stop = True
                self.evidence.last_retrodiction = score_predicted_path(
                    self.evidence.batch.get('predicted_path'), self.evidence.results)
                if self.evidence.last_retrodiction is not None:
                    self.evidence.last_retrodiction['turn'] = self._nooa_turn
                self.evidence.note = (f"Frozen prediction {checked['status']} after "
                    f"{self._pending_action!r}; {len(self._queue)} queued actions skipped. "
                    "See step_checks.jsonl; a match would establish consistency only.")
                with (self.ipc / 'batch_stops.jsonl').open('a') as handle:
                    handle.write(json.dumps({**checked, 'reason': 'prediction_' + checked['status']}) + '\n')
        self._pending_action = None
        if stop:
            self._queue.clear()
        if not self._queue:
            self._nooa_turn += 1
            self._write_state(message.after, message.after.levels_completed)
        return {"sequence": message.sequence}

    def finish(self, _message: FinishMessage) -> dict[str, Any]:
        self.finished = True
        (self.run_dir / "stop.json").write_text(
            json.dumps(
                {
                    "reason": _message.stop_reason,
                    "steps": self.evidence.step,
                    "ended_at": time.time(),
                }
            )
        )
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
            if (self.ipc / "ABORT").exists():
                raise WorkerError((self.ipc / "ABORT").read_text())
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

    def _write_state(self, observation: ObservationPayload, level: int) -> None:
        self.evidence.publish(_observation(observation), self._nooa_turn)
        self._state_written = True


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
