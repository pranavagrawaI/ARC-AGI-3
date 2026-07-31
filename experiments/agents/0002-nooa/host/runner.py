"""Host-side process and append-only file bridge for the 0002-NOOA control."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

from arcengine import GameAction

from experiments.harness import Decision, Observation, Transition

_BASELINE_AGENT_ROOT = (
    Path(__file__).resolve().parents[2] / "0002-llm-repl-agent"
)
_P9_PARENT = Path(__file__).resolve().parents[2] / "0006-capability-environment"
for _source_root in (_BASELINE_AGENT_ROOT, _P9_PARENT):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from llm_repl_agent.observations import ColorPermutation, ModelObservation, ObservationStore
from p9.agent import NOTES_FILE, WORKSPACE_ADDENDUM
from p9.frame_tools import FRAME_TOOLS_FILE, frame_tools_source

from .qualification import PairedRunSpec

PROTOCOL = "0002-nooa-ipc-v1"


class NooaWorkerError(RuntimeError):
    """The isolated control process failed or violated the IPC contract."""


FixturePolicy = Callable[[int, Observation], list[str]]


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _redact_images(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_images(item) for item in value]
    if not isinstance(value, dict):
        return value
    image = value.get("image_url")
    if isinstance(image, dict):
        url = image.get("url")
        if isinstance(url, str) and url.startswith("data:image/"):
            return {
                **value,
                "image_url": {
                    "redacted": True,
                    "sha256": sha256(url.encode("utf-8")).hexdigest(),
                    "chars": len(url),
                },
            }
    return {key: _redact_images(item) for key, item in value.items()}


def _read_tail(path: Path, *, limit: int = 4_000) -> str:
    if not path.exists():
        return ""
    byte_limit = limit * 4
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - byte_limit))
        value = handle.read(byte_limit).decode("utf-8", errors="replace")
    return value[-limit:]


class NooaProcessAgent:
    """Synchronous harness adapter around one long-lived Python 3.12 worker."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        run_root: Path,
        fixture_policy: FixturePolicy,
        paired_run_spec: PairedRunSpec | None = None,
        response_timeout: float = 30.0,
    ) -> None:
        self.runtime_root = runtime_root.resolve()
        self.run_root = run_root.resolve()
        self.run_root.mkdir(parents=True, exist_ok=False)
        self.requests_path = self.run_root / "ipc" / "requests.jsonl"
        self.responses_path = self.run_root / "ipc" / "responses.jsonl"
        self.process_path = self.run_root / "process.jsonl"
        self.worker_stdout_path = self.run_root / "worker.stdout.log"
        self.worker_stderr_path = self.run_root / "worker.stderr.log"
        self.workspace = self.run_root / "workspace"
        self.requests_path.parent.mkdir(parents=True)
        self.requests_path.touch()
        self.workspace.mkdir()
        self.observation_store = ObservationStore(
            self.workspace,
            colors=ColorPermutation.identity(),
        )
        with (self.workspace / "WORKSPACE.md").open("a", encoding="utf-8") as contract:
            contract.write(WORKSPACE_ADDENDUM)
        (self.workspace / "public_history.jsonl").touch()
        (self.workspace / NOTES_FILE).touch()
        (self.workspace / FRAME_TOOLS_FILE).write_text(
            frame_tools_source(components=False),
            encoding="utf-8",
        )
        self.run_id = uuid4().hex
        self.fixture_policy = fixture_policy
        self.paired_run_spec = paired_run_spec or PairedRunSpec.initial()
        self.response_timeout = response_timeout
        self._step = 0
        self._current_projection: ModelObservation | None = None
        self._history: list[dict[str, Any]] = []
        self._pending_decision: Decision | None = None
        self._closed = False
        self._response_offset = 0
        self._worker_stdout = self.worker_stdout_path.open("w", encoding="utf-8")
        self._worker_stderr = self.worker_stderr_path.open("w", encoding="utf-8")
        self._worker_logs_closed = False
        self._process = subprocess.Popen(
            [
                str(self.runtime_root / ".venv" / "bin" / "python"),
                str(self.runtime_root / "launcher.py"),
                "--requests",
                str(self.requests_path),
                "--responses",
                str(self.responses_path),
                "--workspace",
                str(self.workspace),
                "--offline-fixture",
                "--serve",
            ],
            cwd=self.runtime_root,
            stdin=subprocess.DEVNULL,
            stdout=self._worker_stdout,
            stderr=self._worker_stderr,
            text=True,
            start_new_session=True,
        )
        self._write_process(
            "worker_started",
            pid=self._process.pid,
            runtime_python=str(self.runtime_root / ".venv" / "bin" / "python"),
        )

    def _write_process(self, kind: str, **payload: Any) -> None:
        _append_jsonl(
            self.process_path,
            {
                "protocol": PROTOCOL,
                "kind": kind,
                "run_id": self.run_id,
                **payload,
            },
        )

    def act(self, observation: Observation) -> Decision:
        if self._closed:
            raise NooaWorkerError("worker is closed")
        if self._pending_decision is not None:
            raise NooaWorkerError("observe must close the previous action before act")
        if self._step >= self.paired_run_spec.max_actions:
            raise NooaWorkerError(
                "environment-action budget exhausted "
                f"({self._step}/{self.paired_run_spec.max_actions})"
            )
        if self._current_projection is None:
            self._current_projection = self.observation_store.write_observation(
                observation,
                step=self._step,
            )
        current = self._current_projection
        height, width = observation.pixels.shape[-2:]
        coordinate_actions = [
            action
            for action in observation.available_actions
            if GameAction.from_id(action).is_complex()
        ]
        request = {
            "protocol": PROTOCOL,
            "run_id": self.run_id,
            "action_point_id": self._step,
            "observation": {
                "step": self._step,
                "current_metadata": current.metadata,
                "representations": current.representations,
                "recent_transitions": self._history[
                    -self.paired_run_spec.max_recent_transitions:
                ],
                "full_history_file": "public_history.jsonl",
                "current_observation_file": "current_observation.json",
                "observations_glob": "observations/step-*.json",
            },
            "legal_actions": list(observation.available_actions),
            "coordinate_actions": coordinate_actions,
            "frame_width": int(width),
            "frame_height": int(height),
            "paired_run_spec": self.paired_run_spec.as_record(),
            "fixture_python_cells": self.fixture_policy(self._step, observation),
        }
        _append_jsonl(self.requests_path, request)
        self._write_process(
            "action_request",
            action_point_id=self._step,
            request_sha256=sha256(
                json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            request=request,
        )
        response = self._wait_for_response(self._step)
        audit = response.get("audit")
        self._write_process(
            "model_crossing",
            action_point_id=self._step,
            audit=_redact_images(audit if isinstance(audit, dict) else {}),
            worker_decision=response.get("decision"),
        )
        model_calls = audit.get("model_calls") if isinstance(audit, dict) else None
        if (
            not isinstance(model_calls, list)
            or not model_calls
            or any(
                not isinstance(call, dict)
                or not isinstance(call.get("messages"), list)
                or not call["messages"]
                or not isinstance(call.get("tools"), list)
                for call in model_calls
            )
        ):
            raise NooaWorkerError("worker response has no valid model-call audit")
        payload = response.get("decision")
        if not isinstance(payload, dict):
            raise NooaWorkerError("worker response has no decision object")
        action = payload.get("action")
        data = payload.get("data")
        reason = payload.get("reason")
        if type(action) is not int or action not in observation.available_actions:
            raise NooaWorkerError(f"worker returned illegal action: {action!r}")
        if data is not None and not isinstance(data, dict):
            raise NooaWorkerError("worker decision data must be an object or null")
        if not isinstance(reason, str):
            raise NooaWorkerError("worker decision reason must be a string")
        if action in coordinate_actions:
            if (
                not isinstance(data, dict)
                or any(
                    not isinstance(data.get(axis), int)
                    or isinstance(data.get(axis), bool)
                    for axis in ("x", "y")
                )
            ):
                raise NooaWorkerError(
                    f"action {action} requires integer x and y data"
                )
            if not (0 <= data["x"] < width and 0 <= data["y"] < height):
                raise NooaWorkerError(
                    f"action {action} coordinates {(data['x'], data['y'])} are outside "
                    f"width={width}, height={height}"
                )
        reasoning = {"reason": reason}
        decision = Decision(
            action=action,
            data=data,
            reasoning=reasoning,
        )
        self._write_process(
            "decision_validated",
            action_point_id=self._step,
            decision={
                "action": decision.action,
                "data": dict(decision.data) if decision.data else None,
                "reasoning": (
                    dict(decision.reasoning) if decision.reasoning else None
                ),
            },
        )
        self._pending_decision = decision
        return decision

    def observe(self, transition: Transition) -> None:
        if self._pending_decision is None:
            raise NooaWorkerError("observe has no pending action")
        if transition.decision != self._pending_decision:
            raise NooaWorkerError("observe transition does not match the pending action")
        if self._current_projection is None:
            raise NooaWorkerError("observe has no projected before observation")
        after = self.observation_store.write_observation(
            transition.after,
            step=self._step + 1,
        )
        transition_record = self.observation_store.append_transition(
            transition,
            step=self._step + 1,
            before=self._current_projection,
            after=after,
        )
        self._history.append(transition_record)
        self._write_process(
            "transition_recorded",
            action_point_id=self._step,
            transition=transition_record,
        )
        self._current_projection = after
        self._pending_decision = None
        self._step += 1

    def _wait_for_response(self, action_point_id: int) -> dict[str, Any]:
        deadline = time.monotonic() + self.response_timeout
        pending = ""
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self._close_worker_logs()
                stderr = _read_tail(self.worker_stderr_path)
                raise NooaWorkerError(
                    f"worker exited {self._process.returncode} before response: {stderr.strip()}"
                )
            if self.responses_path.exists():
                with self.responses_path.open("r", encoding="utf-8") as handle:
                    handle.seek(self._response_offset)
                    chunk = handle.read()
                    self._response_offset = handle.tell()
                pending += chunk
                lines = pending.splitlines(keepends=True)
                pending = ""
                if lines and not lines[-1].endswith("\n"):
                    pending = lines.pop()
                for line in lines:
                    try:
                        response = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise NooaWorkerError(
                            f"worker wrote malformed response JSON: {error}"
                        ) from error
                    if not isinstance(response, dict):
                        raise NooaWorkerError(
                            "worker response JSON must be an object"
                        )
                    identity = (response.get("run_id"), response.get("action_point_id"))
                    expected = (self.run_id, action_point_id)
                    if identity != expected or response.get("protocol") != PROTOCOL:
                        raise NooaWorkerError(
                            f"stale or mismatched response {identity!r}; expected {expected!r}"
                        )
                    if response.get("complete") is not True:
                        raise NooaWorkerError("worker response is not complete")
                    return response
            time.sleep(0.01)
        raise NooaWorkerError(f"worker response timed out after {self.response_timeout}s")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            _append_jsonl(self.requests_path, {"protocol": PROTOCOL, "kind": "stop"})
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._kill_worker_tree()
            self._process.wait(timeout=5)
        self._close_worker_logs()
        stderr = _read_tail(self.worker_stderr_path)
        self._write_process(
            "worker_stopped",
            returncode=self._process.returncode,
            stderr=stderr,
        )
        if self._process.returncode:
            raise NooaWorkerError(f"worker exited {self._process.returncode}: {stderr.strip()}")

    def _kill_worker_tree(self) -> None:
        if os.name == "posix":
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
                return
            except ProcessLookupError:
                return
        self._process.kill()

    def _close_worker_logs(self) -> None:
        if self._worker_logs_closed:
            return
        self._worker_logs_closed = True
        self._worker_stdout.close()
        self._worker_stderr.close()

    def __enter__(self) -> "NooaProcessAgent":
        return self

    def __exit__(self, error_type: object, *_: object) -> None:
        try:
            self.close()
        except NooaWorkerError:
            if error_type is None:
                raise
