"""Persistent prefixed-JSONL transport for a solver subprocess."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import IO, Any, Self

from arc_host.types import (
    ActionDecision,
    EpisodeResult,
    GameStart,
    JsonValue,
    Transition,
    Turn,
    WorkerReady,
)

HOST_PREFIX = "ARC_HOST\t"
WORKER_PREFIX = "ARC_AGENT\t"
PROTOCOL_VERSION = 1
WORKER_ENVIRONMENT_KEYS = frozenset(
    {
        "ARC_LLM_API_KEY",
        "ARC_LLM_BASE_URL",
        "ARC_LLM_CONTEXT_WINDOW",
        "ARC_LLM_MAX_TOKENS",
        "ARC_LLM_MODEL",
        "ARC_LLM_USE_RESPONSES",
        "HOME",
        "LANG",
        "LC_ALL",
        "NOOA_LLM_CONFIG",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "TYCHO_MODE",
        "TYCHO_NOOA_CAPTURE_REASONING",
        "TZ",
    }
)


class ProtocolError(RuntimeError):
    pass


def worker_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """Pass runtime and model credentials, never the ARC environment credential."""
    source = dict(os.environ) if source is None else source
    return {key: source[key] for key in WORKER_ENVIRONMENT_KEYS if key in source}


class JsonlWorker:
    """One process and therefore one solver identity for an entire game."""

    def __init__(
        self,
        command: list[str],
        *,
        stdout_log: Path,
        stderr_log: Path,
        timeout_seconds: float = 3_600,
        environment: dict[str, str] | None = None,
    ):
        if not command:
            raise ValueError("worker command must be non-empty")
        self.command = list(command)
        self.stdout_log = stdout_log
        self.timeout_seconds = timeout_seconds
        self._stderr: IO[str] = stderr_log.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
            env=worker_environment() if environment is None else environment,
        )
        self._stdout_lines: queue.Queue[str | None] = queue.Queue()
        self._stdout_reader = threading.Thread(
            target=self._read_stdout,
            name="arc-host-worker-stdout",
            daemon=True,
        )
        self._stdout_reader.start()
        self._closed = False

    def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            self._stdout_lines.put(None)
            return
        for line in stdout:
            self._stdout_lines.put(line)
        self._stdout_lines.put(None)

    def start_game(self, start: GameStart) -> WorkerReady:
        self._send("start_game", start.to_json())
        payload = self._receive("ready")
        worker_id = payload.get("worker_id")
        metadata = payload.get("metadata")
        if not isinstance(worker_id, str) or not worker_id:
            raise ProtocolError("ready.worker_id must be non-empty text")
        if not isinstance(metadata, dict):
            raise ProtocolError("ready.metadata must be an object")
        return WorkerReady(worker_id=worker_id, metadata=metadata)

    def decide(self, turn: Turn) -> ActionDecision:
        self._send("turn", turn.to_json())
        payload = self._receive("decision", sequence=turn.sequence)
        return ActionDecision.from_json(payload.get("decision"))

    def observe(self, transition: Transition) -> None:
        self._send("action_result", transition.to_json())
        self._receive("observed", sequence=transition.turn.sequence)

    def finish(self, result: EpisodeResult) -> None:
        self._send(
            "finish",
            {
                "stop_reason": result.stop_reason,
                "actions": result.actions,
                "levels_completed": result.levels_completed,
                "final_observation": result.final.to_json(),
            },
        )
        self._receive("finished")

    def _send(self, message_type: str, payload: dict[str, JsonValue]) -> None:
        process = self._process
        if process.poll() is not None or process.stdin is None:
            raise ProtocolError(f"worker exited with code {process.poll()}")
        message: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "type": message_type,
            **payload,
        }
        process.stdin.write(
            HOST_PREFIX + json.dumps(message, separators=(",", ":")) + "\n"
        )
        process.stdin.flush()

    def _receive(
        self, expected_type: str, *, sequence: int | None = None
    ) -> dict[str, JsonValue]:
        process = self._process
        while True:
            try:
                line = self._stdout_lines.get(timeout=self.timeout_seconds)
            except queue.Empty:
                raise TimeoutError(f"worker timed out waiting for {expected_type}")
            if line is None:
                raise ProtocolError(f"worker exited with code {process.poll()}")
            if not line.startswith(WORKER_PREFIX):
                with self.stdout_log.open("a", encoding="utf-8") as log:
                    log.write(line)
                continue
            try:
                payload = json.loads(line[len(WORKER_PREFIX) :])
            except json.JSONDecodeError as error:
                raise ProtocolError("worker emitted malformed JSON") from error
            if not isinstance(payload, dict):
                raise ProtocolError("worker response must be a JSON object")
            if payload.get("protocol_version") != PROTOCOL_VERSION:
                raise ProtocolError("worker protocol version mismatch")
            if payload.get("type") == "error":
                raise ProtocolError(str(payload.get("message", "worker error")))
            if payload.get("type") != expected_type:
                raise ProtocolError(
                    f"expected worker message {expected_type!r}, got {payload.get('type')!r}"
                )
            if sequence is not None and payload.get("sequence") != sequence:
                raise ProtocolError(
                    f"expected sequence {sequence}, got {payload.get('sequence')!r}"
                )
            return payload

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process.poll() is None and process.stdin is not None:
            try:
                self._send("close", {})
                process.wait(timeout=10)
            except BrokenPipeError, OSError, ProtocolError, subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        self._stderr.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
