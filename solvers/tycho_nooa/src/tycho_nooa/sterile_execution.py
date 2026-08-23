"""Exec-backed Python cells with invocation-local state and data-only results."""

from __future__ import annotations

import asyncio
import json
import keyword
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
)

ReturnT = TypeVar("ReturnT")

_PROTOCOL_VERSION = 1
_MAX_PROTOCOL_BYTES = 1_048_576


class ExecutionManifest(BaseModel):
    """Complete data supplied to one disposable Python execution domain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal[1] = _PROTOCOL_VERSION
    invocation_id: str = Field(min_length=1, max_length=128)
    role: Literal["actor", "builder"]
    input_values: dict[str, JsonValue] = Field(default_factory=dict)
    wall_time_seconds: float = Field(default=5.0, gt=0, le=300)
    output_bytes: int = Field(default=65_536, ge=1_024, le=1_048_576)
    locals_preview_items: int = Field(default=64, ge=0, le=256)
    memory_headroom_mb: int = Field(default=256, ge=32, le=4_096)
    cpu_seconds: int = Field(default=10, ge=1, le=300)
    file_size_bytes: int = Field(default=1_048_576, ge=1_024, le=16_777_216)
    open_files: int = Field(default=64, ge=16, le=256)

    @field_validator("input_values")
    @classmethod
    def input_names_are_explicit_namespace_values(
        cls, values: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        reserved = {"self", "return_result", "__builtins__"}
        for name in values:
            if name in reserved or not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(f"input_values name is unavailable: {name!r}")
        return values


@dataclass(frozen=True)
class ExecutionException:
    type: str
    message: str


class SterileReturnValidationError(ValueError):
    """T2 proposed a value that does not satisfy T1's return contract."""


class SterileCellTimeoutError(TimeoutError):
    """A model-generated cell exceeded its invocation wall-time bound."""


@dataclass(frozen=True)
class SterileExecutionResult[ReturnT]:
    """Bounded observations of a cell; the child namespace never crosses T2."""

    status: Literal["complete", "error", "returned"]
    stdout: str
    stderr: str
    locals_preview: dict[str, str]
    return_value: ReturnT | None = None
    exception: ExecutionException | None = None


class SterileExecutor[ReturnT]:
    """Own one clean interpreter for the lifetime of a CodeAct invocation."""

    def __init__(self, manifest: ExecutionManifest, *, return_type: type[ReturnT]):
        self._manifest = manifest
        self._return_adapter = TypeAdapter(return_type)
        self._process: asyncio.subprocess.Process | None = None
        self._scratch: tempfile.TemporaryDirectory[str] | None = None
        self._request_id = 0
        self._returned = False

    async def __aenter__(self) -> SterileExecutor[ReturnT]:
        if self._process is not None:
            raise RuntimeError("sterile executor is already running")
        self._scratch = tempfile.TemporaryDirectory(prefix="tycho-codeact-")
        worker = Path(__file__).with_name("_sterile_worker.py")
        environment = {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.defpath,
            "PYTHONHASHSEED": "0",
            "TMPDIR": self._scratch.name,
        }
        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            str(worker),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._scratch.name,
            env=environment,
            close_fds=True,
        )
        await self._send(
            {
                "protocol_version": _PROTOCOL_VERSION,
                "type": "initialize",
                "manifest": self._manifest.model_dump(mode="json"),
            }
        )
        response = await self._receive()
        if response.get("type") != "ready":
            await self.aclose()
            raise RuntimeError(f"sterile worker initialization failed: {response}")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def run_cell(self, code: str) -> SterileExecutionResult[ReturnT]:
        if self._process is None:
            raise RuntimeError("sterile executor is not running")
        if self._returned:
            raise RuntimeError("sterile invocation already returned a valid result")
        self._request_id += 1
        request_id = self._request_id
        response = await self._execute_request(code, request_id)
        return self._result_from_response(response, request_id)

    async def _execute_request(self, code: str, request_id: int) -> dict[str, Any]:
        await self._send(
            {
                "protocol_version": _PROTOCOL_VERSION,
                "type": "execute",
                "invocation_id": self._manifest.invocation_id,
                "request_id": request_id,
                "code": code,
            }
        )
        try:
            response = await asyncio.wait_for(
                self._receive(), timeout=self._manifest.wall_time_seconds
            )
        except TimeoutError as error:
            await self._kill_process()
            raise SterileCellTimeoutError(
                f"cell exceeded {self._manifest.wall_time_seconds} seconds"
            ) from error
        return response

    def _result_from_response(
        self, response: dict[str, Any], request_id: int
    ) -> SterileExecutionResult[ReturnT]:
        if response.get("request_id") != request_id:
            raise RuntimeError("sterile worker response did not match its request")
        status = response.get("status")
        if status not in {"complete", "error", "returned"}:
            raise RuntimeError(f"invalid sterile worker status: {status!r}")
        return SterileExecutionResult(
            status=status,
            stdout=str(response.get("stdout", "")),
            stderr=str(response.get("stderr", "")),
            locals_preview={
                str(name): str(preview)
                for name, preview in dict(response.get("locals_preview", {})).items()
            },
            return_value=self._validated_return(response, status),
            exception=self._execution_exception(response.get("exception")),
        )

    def _validated_return(
        self, response: dict[str, Any], status: str
    ) -> ReturnT | None:
        if status != "returned":
            return None
        try:
            value = self._return_adapter.validate_python(response.get("return_payload"))
        except ValidationError as error:
            raise SterileReturnValidationError(str(error)) from error
        self._returned = True
        return value

    @staticmethod
    def _execution_exception(error: Any) -> ExecutionException | None:
        if not isinstance(error, dict):
            return None
        return ExecutionException(type=str(error["type"]), message=str(error["message"]))

    async def aclose(self) -> None:
        process = self._process
        self._process = None
        try:
            if process is not None and process.returncode is None:
                await self._request_shutdown(process)
            if process is not None:
                await self._raise_on_unsuccessful_exit(process)
        finally:
            if self._scratch is not None:
                self._scratch.cleanup()
                self._scratch = None

    async def _request_shutdown(self, process: asyncio.subprocess.Process) -> None:
        try:
            await self._send_to(process, {"type": "shutdown"})
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except (BrokenPipeError, ConnectionError, TimeoutError):
            process.kill()
            await process.wait()

    @staticmethod
    async def _raise_on_unsuccessful_exit(
        process: asyncio.subprocess.Process,
    ) -> None:
        if process.returncode in {None, 0}:
            return
        stderr = await process.stderr.read() if process.stderr is not None else b""
        if stderr:
            raise RuntimeError(
                "sterile worker exited unsuccessfully: "
                + stderr.decode("utf-8", errors="replace")[:2_000]
            )

    async def _kill_process(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()

    async def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None:
            raise RuntimeError("sterile executor is not running")
        await self._send_to(process, payload)

    @staticmethod
    async def _send_to(
        process: asyncio.subprocess.Process, payload: dict[str, Any]
    ) -> None:
        if process.stdin is None:
            raise RuntimeError("sterile worker stdin is unavailable")
        encoded = json.dumps(
            payload, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > _MAX_PROTOCOL_BYTES:
            raise ValueError("sterile protocol request exceeds its byte limit")
        process.stdin.write(struct.pack(">I", len(encoded)) + encoded)
        await process.stdin.drain()

    async def _receive(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("sterile worker stdout is unavailable")
        header = await process.stdout.readexactly(4)
        size = struct.unpack(">I", header)[0]
        if size > _MAX_PROTOCOL_BYTES:
            raise RuntimeError("sterile protocol response exceeds its byte limit")
        payload = await process.stdout.readexactly(size)
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise TypeError("sterile protocol response must be an object")
        return parsed
