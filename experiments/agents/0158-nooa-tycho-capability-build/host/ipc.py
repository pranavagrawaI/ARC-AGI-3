"""Fail-closed host authority for the 0158 append-only artifact IPC."""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .canonical import append_jsonl, content_sha256, read_jsonl
from .contract import MODEL_CONTRACT_SHA256, model_contract

PROTOCOL = "0158-grounding-ipc-v1"
REVIEW_STOP_POLICY = {
    "mode": "review_stop",
    "ordinary_cycle_stop": "terminate_run",
    "g2": "stop_for_review",
}
AUTONOMOUS_44_POLICY = {
    "mode": "autonomous_44_action",
    "ordinary_cycle_stop": "record_and_reset_without_human",
    "g2": "record_reset_without_new_api",
}


class ArtifactWorkerError(RuntimeError):
    """The isolated artifact worker failed or violated its bound response."""


class ArtifactGenerationError(ArtifactWorkerError):
    """The provider completed a call without producing a usable artifact."""


FixturePolicy = Callable[[str, dict[str, Any]], list[str]]


def _workspace_commitment(root: Path) -> tuple[str, int]:
    records: list[dict[str, Any]] = []
    if root.exists():
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            base = Path(directory)
            for name in sorted([*dirnames, *filenames]):
                path = base / name
                relative = path.relative_to(root).as_posix()
                mode = os.lstat(path).st_mode
                if stat.S_ISLNK(mode):
                    record = {"path": relative, "kind": "symlink", "target": os.readlink(path)}
                elif stat.S_ISDIR(mode):
                    record = {"path": relative, "kind": "directory"}
                elif stat.S_ISREG(mode):
                    payload = path.read_bytes()
                    record = {"path": relative, "kind": "file", "bytes": len(payload), "sha256": sha256(payload).hexdigest()}
                else:
                    record = {"path": relative, "kind": "special", "mode": stat.S_IFMT(mode)}
                records.append(record)
    return content_sha256(records), len(records)


def validate_response_record(request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    unhashed = dict(request)
    claimed_request_hash = unhashed.pop("request_sha256", None)
    if claimed_request_hash != content_sha256(unhashed):
        raise ArtifactWorkerError("request content hash mismatch")
    bound = (
        "protocol", "run_id", "artifact_id", "sequence", "phase", "observation_id",
        "evidence_sha256", "model_contract_sha256", "request_sha256",
    )
    mismatched = [key for key in bound if response.get(key) != request.get(key)]
    if mismatched:
        raise ArtifactWorkerError(f"stale or mismatched response fields: {mismatched}")
    if response.get("complete") is not True:
        raise ArtifactWorkerError("response is not complete")
    if "error" in response:
        error = response["error"]
        error_type = error.get("type")
        failure = f"worker error: {error_type}: {error.get('message')}"
        if error_type == "GenerationError":
            raise ArtifactGenerationError(failure)
        raise ArtifactWorkerError(failure)
    artifact = response.get("artifact")
    if not isinstance(artifact, dict):
        raise ArtifactWorkerError("response artifact must be an object")
    if response.get("artifact_sha256") != content_sha256(artifact):
        raise ArtifactWorkerError("artifact content hash mismatch")
    return artifact


def select_exact_response(records: list[dict[str, Any]], artifact_id: str) -> dict[str, Any]:
    matching = [item for item in records if item.get("artifact_id") == artifact_id]
    if len(matching) != 1:
        raise ArtifactWorkerError(f"expected exactly one response, found {len(matching)}")
    return matching[0]


class ArtifactBridge:
    """Launch one Python 3.12 worker and one fresh CodeAct sandbox per artifact."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        run_root: Path,
        fixture_policy: FixturePolicy | None = None,
        provider_config: dict[str, Any] | None = None,
        worker_env: dict[str, str] | None = None,
        python_executable: Path | None = None,
        timeout_seconds: float = 60.0,
        run_policy: dict[str, str] | None = None,
    ) -> None:
        self.runtime_root = runtime_root.resolve()
        self.run_root = run_root.resolve()
        self.fixture_policy = fixture_policy
        self.provider_config = provider_config
        self.run_policy = dict(run_policy or REVIEW_STOP_POLICY)
        if self.run_policy not in (REVIEW_STOP_POLICY, AUTONOMOUS_44_POLICY):
            raise ValueError("run_policy must be one frozen 0158 policy")
        if (fixture_policy is None) == (provider_config is None):
            raise ValueError("select exactly one artifact worker mode")
        if provider_config is not None and set(provider_config) != {
            "provider_request_profile",
            "provider_route",
            "runtime_limits",
            "transport_max_retries",
        }:
            raise ValueError("provider config has unknown or missing fields")
        self.worker_env = dict(worker_env) if worker_env is not None else None
        selected_python = python_executable or (self.runtime_root / ".venv" / "bin" / "python")
        self.python_executable = selected_python.absolute()
        if not self.python_executable.is_file():
            raise FileNotFoundError(f"Python 3.12 runtime is absent: {self.python_executable}")
        self.timeout_seconds = timeout_seconds
        self.run_root.mkdir(parents=True, exist_ok=False)
        self.requests_path = self.run_root / "ipc" / "requests.jsonl"
        self.responses_path = self.run_root / "ipc" / "responses.jsonl"
        self.trace_path = self.run_root / "trace.jsonl"
        self.requests_path.parent.mkdir(parents=True)
        self.requests_path.touch()
        self.responses_path.touch()
        self.run_id = uuid4().hex
        self._sequence = 0
        self.latest_response: dict[str, Any] | None = None

    def request(
        self,
        *,
        phase: str,
        observation_id: str,
        evidence: dict[str, Any],
        legal_actions: list[int],
        accepted_claim: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if phase not in {"claim", "commitment"}:
            raise ValueError("phase must be claim or commitment")
        artifact_input = {
            "phase": phase,
            "current_observation_id": observation_id,
            "legal_actions": legal_actions,
            "accepted_claim": accepted_claim,
            "evidence": evidence,
            "model_contract": model_contract(),
            "run_policy": dict(self.run_policy),
        }
        request: dict[str, Any] = {
            "protocol": PROTOCOL,
            "run_id": self.run_id,
            "artifact_id": uuid4().hex,
            "sequence": self._sequence,
            "phase": phase,
            "observation_id": observation_id,
            "evidence_sha256": content_sha256(evidence),
            "model_contract_sha256": MODEL_CONTRACT_SHA256,
            "artifact_input": artifact_input,
        }
        if self.fixture_policy is not None:
            request.update(
                {
                    "worker_mode": "fixture",
                    "fixture_python_cells": self.fixture_policy(phase, artifact_input),
                }
            )
        else:
            assert self.provider_config is not None
            request.update({"worker_mode": "provider", **self.provider_config})
        request["request_sha256"] = content_sha256(request)
        append_jsonl(self.requests_path, request)
        append_jsonl(self.trace_path, {"kind": "artifact_requested", "sequence": self._sequence, "request_sha256": request["request_sha256"]})
        workspace = self.run_root / "artifact-workspaces" / f"{self._sequence:04d}"
        try:
            completed = self._run_worker(workspace)
            responses = read_jsonl(self.responses_path)
            try:
                response = select_exact_response(responses, request["artifact_id"])
            except ArtifactWorkerError as error:
                raise ArtifactWorkerError(f"{error}; stderr={completed.stderr[-2000:]}") from error
            self.latest_response = response
            try:
                artifact = validate_response_record(request, response)
            except ArtifactWorkerError:
                append_jsonl(self.trace_path, {"kind": "artifact_rejected", "sequence": self._sequence})
                raise
            if completed.returncode != 0:
                raise ArtifactWorkerError(f"worker exited {completed.returncode}: {completed.stderr[-2000:]}")
            append_jsonl(self.trace_path, {"kind": "artifact_accepted", "sequence": self._sequence, "artifact_sha256": content_sha256(artifact)})
            self._sequence += 1
            return artifact
        finally:
            commitment, entries = _workspace_commitment(workspace)
            append_jsonl(
                self.trace_path,
                {
                    "kind": "artifact_workspace_retired",
                    "sequence": request["sequence"],
                    "workspace_sha256": commitment,
                    "entries": entries,
                },
            )
            shutil.rmtree(workspace, ignore_errors=True)

    def _run_worker(self, workspace: Path) -> subprocess.CompletedProcess[str]:
        command = [
            str(self.python_executable),
            str(self.runtime_root / "launcher.py"),
            "--requests", str(self.requests_path),
            "--responses", str(self.responses_path),
            "--request-index", str(self._sequence),
            "--workspace", str(workspace),
        ]
        process = subprocess.Popen(
            command,
            cwd=self.runtime_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=self.worker_env,
        )
        append_jsonl(
            self.trace_path,
            {"kind": "artifact_worker_started", "sequence": self._sequence, "pid": process.pid},
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            append_jsonl(self.trace_path, {"kind": "artifact_timeout", "sequence": self._sequence})
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
            raise ArtifactWorkerError(f"worker timed out after {self.timeout_seconds} seconds") from error
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
