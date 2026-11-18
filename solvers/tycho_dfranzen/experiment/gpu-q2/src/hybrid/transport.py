"""Reuse canonical continuity checks with bounded writes and response deadlines."""
from __future__ import annotations

import json
import os
import queue
import select
import time

from hybrid_local_solver import LocalSolver, LocalSolverError
from tycho_nooa.protocol import HOST_PREFIX, WORKER_PREFIX, PROTOCOL_VERSION


class HybridLocalSolver(LocalSolver):
    stats = None

    def __init__(self, *, cancellation=None, **kwargs):
        self.cancellation = cancellation
        super().__init__(**kwargs)

    def _check_cancelled(self):
        if self.cancellation is not None and self.cancellation.is_set():
            raise LocalSolverError("benchmark cancelled")

    def close(self):
        # Cleanup must not inherit a long model/pipe deadline.
        self.timeout_seconds = min(self.timeout_seconds, 1)
        self.close_timeout_seconds = min(self.close_timeout_seconds, 1)
        super().close()

    def _send_payload(self, payload):
        process = self._process
        if process.poll() is not None or process.stdin is None:
            raise LocalSolverError(f"worker exited with code {process.poll()}")
        wire = (HOST_PREFIX + json.dumps({"protocol_version": PROTOCOL_VERSION, **payload},
                    separators=(",", ":")) + "\n").encode()
        fd = process.stdin.fileno()
        os.set_blocking(fd, False)
        deadline = time.monotonic() + self.timeout_seconds
        offset = 0
        while offset < len(wire):
            self._check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalSolverError("worker input write deadline")
            if not select.select([], [fd], [], min(remaining, 0.2))[1]:
                continue
            try:
                offset += os.write(fd, wire[offset:])
            except BlockingIOError:
                continue

    def _receive(self, expected_type, *, sequence=None):
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            self._check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"worker deadline waiting for {expected_type}")
            try:
                line = self._stdout_lines.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                continue
            if line is None:
                raise LocalSolverError(f"worker exited with code {self._process.poll()}")
            if not line.startswith(WORKER_PREFIX):
                with self.stdout_log.open("a") as stream:
                    stream.write(line)
                continue
            try:
                payload = json.loads(line[len(WORKER_PREFIX):])
            except json.JSONDecodeError as error:
                raise LocalSolverError("worker emitted malformed JSON") from error
            if not isinstance(payload, dict) or payload.get("protocol_version") != PROTOCOL_VERSION:
                raise LocalSolverError("worker protocol version mismatch")
            self.stats = payload.get("hybrid", {})
            if payload.get("type") == "error":
                raise LocalSolverError(str(payload.get("message", "worker error")))
            if payload.get("type") != expected_type or (sequence is not None and payload.get("sequence") != sequence):
                raise LocalSolverError("unexpected worker response or sequence")
            return payload
