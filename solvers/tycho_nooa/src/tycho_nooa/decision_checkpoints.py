"""Replayable snapshots at Tycho-NOOA's model-call boundary.

Each call records the exact JSON-safe request handed to NOOA's ``llm_call``
middleware, the standardized response, and a content-addressed snapshot of the
durable game workspace as it existed before the call.  The commitment is scoped
to that boundary: provider formatters may subsequently translate the request to
Chat Completions or Responses wire syntax.

Workspace files are stored once by SHA-256 rather than copied into every call.
This matters in a run with hundreds of calls: the growing manifests stay cheap
while every historical workspace remains byte-restorable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = 2
REQUEST_BOUNDARY = "nooa.llm_call"


class DecisionCheckpointRecorder:
    """One run-wide, ordered recorder shared by actor and fresh builders."""

    def __init__(
        self,
        root: Path | None,
        *,
        workspace_dir: Path,
        llm: Any,
    ) -> None:
        self._root = root
        self._workspace_dir = workspace_dir.resolve()
        self._llm = llm
        self._call_index = 0
        self._decision_point: dict[str, Any] | None = None
        self._source_override: str | None = None
        if root is not None and _is_within(root.resolve(), self._workspace_dir):
            raise ValueError(
                "decision checkpoint storage cannot be inside the workspace"
            )

    def set_decision_point(
        self,
        *,
        sequence: int | None,
        level: int,
        turn_in_level: int,
        phase: str,
    ) -> None:
        """Associate subsequent calls with a public host decision boundary."""

        if phase not in {"decision", "post_action"}:
            raise ValueError(f"invalid checkpoint phase: {phase}")
        self._decision_point = {
            "sequence": sequence,
            "level": level,
            "turn_in_level": turn_in_level,
            "phase": phase,
        }

    @contextmanager
    def source_scope(self, source: str) -> Iterator[None]:
        """Attribute one bounded strategy invocation and restore prior context."""
        previous = self._source_override
        self._source_override = source
        try:
            yield
        finally:
            self._source_override = previous

    def attach(self, event_manager: Any, *, source: str) -> None:
        """Capture calls on one agent stream while preserving a run-wide order."""

        if self._root is None:
            return

        async def capture(ctx: Any, nxt: Any) -> Any:
            call_index, call_dir = self._claim_call_directory()
            workspace, workspace_commitment = self._snapshot_workspace()
            _write_json(call_dir / "workspace.json", workspace)
            decision_point = (
                dict(self._decision_point)
                if self._decision_point is not None
                else None
            )
            request_body = _request_body(
                model=self._llm.model,
                messages=ctx.messages,
                params=ctx.params,
            )
            request = {
                "schema_version": SCHEMA_VERSION,
                "call_index": call_index,
                "source": self._source_override or source,
                "request_boundary": REQUEST_BOUNDARY,
                "commitment_scope": (
                    "exact credential-free request at the NOOA llm_call boundary; "
                    "provider wire formatting is downstream"
                ),
                "request_payload_commitment": _commitment(request_body),
                "workspace_commitment": workspace_commitment,
                "decision_point": decision_point,
                "replay_context": _replay_context(self._llm),
                "request_body": request_body,
            }
            _write_json(call_dir / "request.json", request)

            try:
                result = await nxt(ctx)
            except BaseException as error:
                response = _committed_response(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
                _write_json(
                    call_dir / "response.json",
                    response,
                )
                raise

            _write_json(
                call_dir / "response.json",
                _committed_response(_response_record(result.response)),
            )
            return result

        event_manager.intercept("llm_call", capture)

    def _claim_call_directory(self) -> tuple[int, Path]:
        """Atomically append when several workers share one run directory."""

        assert self._root is not None
        while True:
            self._call_index += 1
            call_dir = self._root / "calls" / f"{self._call_index:06d}"
            try:
                call_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return self._call_index, call_dir

    def _snapshot_workspace(self) -> tuple[dict[str, Any], str]:
        assert self._root is not None
        blob_root = self._root / "blobs" / "sha256"
        directories: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []

        for path in sorted(self._workspace_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"workspace snapshots do not permit symlinks: {path}")
            relative = path.relative_to(self._workspace_dir).as_posix()
            mode = stat.S_IMODE(path.stat().st_mode)
            if path.is_dir():
                directories.append({"path": relative, "mode": mode})
                continue
            if not path.is_file():
                raise ValueError(f"unsupported workspace entry: {path}")

            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            _store_blob(blob_root, digest, content)
            files.append(
                {
                    "path": relative,
                    "sha256": digest,
                    "size": len(content),
                    "mode": mode,
                }
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "directories": directories,
            "files": files,
        }
        return manifest, _commitment(manifest)


def restore_workspace(
    checkpoint_root: Path,
    *,
    call_index: int,
    destination: Path,
) -> None:
    """Restore one pre-call workspace without overwriting existing evidence."""

    if destination.exists():
        raise FileExistsError(destination)
    call_dir = checkpoint_root / "calls" / f"{call_index:06d}"
    manifest = json.loads((call_dir / "workspace.json").read_text())
    temporary = destination.with_name(f".{destination.name}.restore-{uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        for entry in manifest["directories"]:
            relative = _safe_relative(entry["path"])
            path = temporary / relative
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(entry["mode"])
        for entry in manifest["files"]:
            relative = _safe_relative(entry["path"])
            digest = entry["sha256"]
            blob = checkpoint_root / "blobs" / "sha256" / digest
            content = blob.read_bytes()
            if (
                len(content) != entry["size"]
                or hashlib.sha256(content).hexdigest() != digest
            ):
                raise ValueError(f"corrupt checkpoint blob: {digest}")
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(entry["mode"])
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _request_body(
    *,
    model: str,
    messages: list[dict[str, Any]],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    tools = params.get("tools")
    output_model = params.get("output_model")
    parameters = {
        key: value
        for key, value in params.items()
        if key not in {"tools", "output_model"}
    }
    return _jsonable(
        {
            "model": model,
            "messages": messages,
            "tools": [_tool_record(tool) for tool in tools] if tools else [],
            "output_model": _output_model_record(output_model),
            "parameters": parameters,
        }
    )


def _tool_record(tool: Any) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.get_parameter_schema(),
    }


def _replay_context(llm: Any) -> dict[str, Any]:
    config = getattr(llm, "config", {})
    safe_config, omitted = _without_credentials(config)
    return _jsonable(
        {
            "client_type": f"{type(llm).__module__}.{type(llm).__qualname__}",
            "client_config": safe_config,
            "omitted_credential_keys": omitted,
            "cache_control_injection_points": getattr(
                llm, "cache_control_injection_points", []
            ),
        }
    )


def _without_credentials(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    sensitive_fragments = (
        "api_key",
        "authorization",
        "credential",
        "password",
        "secret",
    )
    safe: dict[str, Any] = {}
    omitted: list[str] = []
    for key, value in config.items():
        lowered = key.lower()
        if any(fragment in lowered for fragment in sensitive_fragments):
            omitted.append(key)
        else:
            safe[key] = value
    return safe, sorted(omitted)


def _output_model_record(output_model: Any) -> dict[str, Any] | None:
    if output_model is None:
        return None
    return {
        "qualified_name": f"{output_model.__module__}.{output_model.__qualname__}",
        "json_schema": output_model.model_json_schema(),
    }


def _response_record(response: Any) -> dict[str, Any]:
    if response is None:
        raise RuntimeError("llm_call completed without a response")
    return _jsonable(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "content": response.content,
            "reasoning": response.reasoning,
            "finish_reason": response.finish_reason,
            "assistant_message": response.assistant_message,
            "usage": response.usage,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in response.tool_calls
            ],
        }
    )


def _committed_response(response: dict[str, Any]) -> dict[str, Any]:
    committed = dict(response)
    committed["response_payload_commitment"] = _commitment(response)
    return committed


def _jsonable(value: Any) -> Any:
    if isinstance(value, (type(None), bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if _is_json_sequence(value):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    raise TypeError(
        f"request checkpoint value is not JSON-safe: {type(value).__name__}"
    )


def _is_json_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _commitment(value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value)).hexdigest()
    return f"sha256:{digest}"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


def _store_blob(root: Path, digest: str, content: bytes) -> None:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / digest
    if destination.exists():
        return
    temporary = root / f".{digest}.{uuid4().hex}.tmp"
    temporary.write_bytes(content)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe workspace path: {value!r}")
    return Path(*path.parts)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
