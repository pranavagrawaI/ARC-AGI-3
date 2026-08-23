"""Exec-backed Python cells with invocation-local state and data-only results."""

from __future__ import annotations

import asyncio
import json
import keyword
import os
import struct
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from tycho_nooa.execution_events import InvocationProvenance

ReturnT = TypeVar("ReturnT")
CapabilityHandler = Callable[[dict[str, JsonValue]], JsonValue]
CapabilityAuditor = Callable[[], dict[str, Any] | None]

_PROTOCOL_VERSION = 1
_MAX_PROTOCOL_BYTES = 1_048_576


class ExecutionManifest(BaseModel):
    """Complete data supplied to one disposable Python execution domain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal[1] = _PROTOCOL_VERSION
    invocation_id: str = Field(min_length=1, max_length=128)
    role: Literal["actor", "builder"]
    capabilities: tuple[Literal["evidence.query"], ...] = ()
    capability_contracts: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict
    )
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
        reserved = {"self", "help", "return_result", "evidence", "__builtins__"}
        for name in values:
            if name in reserved or not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(f"input_values name is unavailable: {name!r}")
        return values

    @field_validator("capabilities")
    @classmethod
    def capabilities_are_unique(
        cls, capabilities: tuple[str, ...]
    ) -> tuple[str, ...]:
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("capabilities must be unique")
        return capabilities

    @model_validator(mode="after")
    def contracts_describe_only_granted_capabilities(self) -> ExecutionManifest:
        unknown = set(self.capability_contracts) - set(self.capabilities)
        if unknown:
            raise ValueError(
                f"capability_contracts describe ungranted capabilities: {sorted(unknown)!r}"
            )
        return self


@dataclass(frozen=True)
class ExecutionException:
    type: str
    message: str


@dataclass(frozen=True)
class CapabilityRecord:
    name: str
    arguments: dict[str, JsonValue]
    status: Literal["complete", "error"]
    result: JsonValue | None = None
    exception: ExecutionException | None = None


class SterileReturnValidationError(ValueError):
    """T2 proposed a value that does not satisfy T1's return contract."""


class SterileCellTimeoutError(TimeoutError):
    """A model-generated cell exceeded its invocation wall-time bound."""


class SterileWorkerLostError(RuntimeError):
    """The disposable interpreter died, taking invocation-local state with it."""


@dataclass(frozen=True)
class SterileExecutionResult[ReturnT]:
    """Bounded observations of a cell; the child namespace never crosses T2."""

    status: Literal["complete", "error", "returned"]
    stdout: str
    stderr: str
    locals_preview: dict[str, str]
    return_value: ReturnT | None = None
    exception: ExecutionException | None = None
    capability_records: tuple[CapabilityRecord, ...] = ()


class SterileExecutor[ReturnT]:
    """Own one clean interpreter for the lifetime of a CodeAct invocation."""

    def __init__(
        self,
        manifest: ExecutionManifest,
        *,
        return_type: type[ReturnT],
        capability_handlers: dict[str, CapabilityHandler] | None = None,
        capability_auditors: dict[str, CapabilityAuditor] | None = None,
        provenance: InvocationProvenance | None = None,
    ):
        self._manifest = manifest
        self._return_adapter = TypeAdapter(return_type)
        self._capability_handlers = dict(capability_handlers or {})
        self._capability_auditors = dict(capability_auditors or {})
        self._provenance = provenance
        if (
            provenance is not None
            and provenance.invocation_id != manifest.invocation_id
        ):
            raise ValueError("provenance invocation must match the execution manifest")
        if set(manifest.capabilities) != set(self._capability_handlers):
            raise ValueError(
                "manifest capabilities must exactly match capability_handlers"
            )
        if not set(self._capability_auditors).issubset(self._capability_handlers):
            raise ValueError("capability_auditors must correspond to handlers")
        self._process: asyncio.subprocess.Process | None = None
        self._scratch: tempfile.TemporaryDirectory[str] | None = None
        self._request_id = 0
        self._provenance_request_id = 0
        self._capability_request_ids: set[int] = set()
        self._returned = False

    async def __aenter__(self) -> SterileExecutor[ReturnT]:
        if self._process is not None:
            raise RuntimeError("sterile executor is already running")
        if self._provenance is not None:
            self._provenance.start(self._manifest.model_dump(mode="json"))
        self._scratch = tempfile.TemporaryDirectory(prefix="tycho-codeact-")
        worker = Path(__file__).with_name("_sterile_worker.py")
        environment = {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.defpath,
            "PYTHONHASHSEED": "0",
            "TMPDIR": self._scratch.name,
        }
        try:
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
                raise RuntimeError(
                    f"sterile worker initialization failed: {response}"
                )
        except Exception:
            if self._provenance is not None:
                self._provenance.fail("worker_start_failed", locals_lost=True)
            await self.aclose()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def run_cell(
        self,
        code: str,
        *,
        tool_call_id: str | None = None,
        tool_name: str = "execute_python",
        tool_arguments: dict[str, Any] | None = None,
        tool_reasoning: str | None = None,
    ) -> SterileExecutionResult[ReturnT]:
        if self._process is None:
            raise RuntimeError("sterile executor is not running")
        if self._returned:
            raise RuntimeError("sterile invocation already returned a valid result")
        self._request_id += 1
        request_id = self._request_id
        if self._provenance is not None:
            self._provenance.cell_dispatched(
                request_id,
                code,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                tool_reasoning=tool_reasoning,
            )
        response, records = await self._execute_request(code, request_id)
        return self._result_from_response(response, request_id, records)

    async def _execute_request(
        self, code: str, request_id: int
    ) -> tuple[dict[str, Any], tuple[CapabilityRecord, ...]]:
        records: list[CapabilityRecord] = []
        try:
            async with asyncio.timeout(self._manifest.wall_time_seconds):
                await self._send(
                    {
                        "protocol_version": _PROTOCOL_VERSION,
                        "type": "execute",
                        "invocation_id": self._manifest.invocation_id,
                        "request_id": request_id,
                        "code": code,
                    }
                )
                while True:
                    response = await self._receive()
                    if response.get("type") != "capability_request":
                        return response, tuple(records)
                    records.append(
                        await self._handle_capability_request(response, request_id)
                    )
        except TimeoutError as error:
            await self._kill_process()
            if self._provenance is not None:
                self._provenance.cell_timed_out(
                    request_id, self._manifest.wall_time_seconds
                )
            raise SterileCellTimeoutError(
                f"cell exceeded {self._manifest.wall_time_seconds} seconds"
            ) from error
        except (asyncio.IncompleteReadError, BrokenPipeError, ConnectionError) as error:
            await self._kill_process()
            if self._provenance is not None:
                self._provenance.worker_lost(request_id, type(error).__name__)
            raise SterileWorkerLostError(
                "sterile worker was lost; invocation-local state did not survive"
            ) from error

    async def _handle_capability_request(
        self, request: dict[str, Any], cell_seq: int
    ) -> CapabilityRecord:
        name = str(request.get("name", ""))
        arguments = request.get("arguments")
        request_id = request.get("capability_request_id")
        self._provenance_request_id += 1
        provenance_request_id = self._provenance_request_id
        if self._provenance is not None:
            self._provenance.capability_requested(
                cell_seq,
                provenance_request_id,
                name,
                arguments,
            )
        try:
            self._validate_capability_request(request, name, arguments, request_id)
            handler = self._capability_handlers[name]
            result = _json_copy(handler(arguments))
        except Exception as error:  # noqa: BLE001 - errors become data for T2
            audit = self._take_capability_audit(name)
            self._record_capability_normalization(
                cell_seq, provenance_request_id, audit
            )
            exception = ExecutionException(
                type=type(error).__name__, message=str(error)[:2_000]
            )
            await self._send_capability_result(request_id, exception=exception)
            if self._provenance is not None:
                self._provenance.capability_resolved(
                    cell_seq,
                    provenance_request_id,
                    exception={
                        "type": exception.type,
                        "message": exception.message,
                    },
                )
            return CapabilityRecord(
                name=name,
                arguments=arguments if isinstance(arguments, dict) else {},
                status="error",
                exception=exception,
            )
        audit = self._take_capability_audit(name)
        self._record_capability_normalization(cell_seq, provenance_request_id, audit)
        await self._send_capability_result(request_id, result=result)
        if self._provenance is not None:
            self._provenance.capability_resolved(
                cell_seq,
                provenance_request_id,
                result=result,
            )
            if audit is not None and audit.get("disclosed") is True:
                self._provenance.capability_disclosed(
                    cell_seq, provenance_request_id
                )
        return CapabilityRecord(
            name=name,
            arguments=arguments,
            status="complete",
            result=result,
        )

    def _take_capability_audit(self, name: str) -> dict[str, Any] | None:
        auditor = self._capability_auditors.get(name)
        return None if auditor is None else auditor()

    def _record_capability_normalization(
        self,
        cell_seq: int,
        request_id: int,
        audit: dict[str, Any] | None,
    ) -> None:
        if self._provenance is None or audit is None:
            return
        semantic_query = audit.get("semantic_query")
        semantic_sha256 = audit.get("semantic_sha256")
        if semantic_query is None or semantic_sha256 is None:
            return
        if not isinstance(semantic_query, dict) or not isinstance(
            semantic_sha256, str
        ):
            raise TypeError("capability audit semantic record is malformed")
        self._provenance.capability_normalized(
            cell_seq,
            request_id,
            semantic_query,
            semantic_sha256,
        )

    def _validate_capability_request(
        self,
        request: dict[str, Any],
        name: str,
        arguments: Any,
        request_id: Any,
    ) -> None:
        self._validate_capability_envelope(request)
        self._validate_capability_request_id(request_id)
        self._validate_capability_operation(name, arguments)

    def _validate_capability_envelope(self, request: dict[str, Any]) -> None:
        if request.get("protocol_version") != _PROTOCOL_VERSION:
            raise ValueError("capability protocol version mismatch")
        if request.get("invocation_id") != self._manifest.invocation_id:
            raise PermissionError("capability invocation mismatch")

    def _validate_capability_request_id(self, request_id: Any) -> None:
        if request_id is None or type(request_id) is not int:
            raise ValueError("capability_request_id must be an integer")
        if request_id in self._capability_request_ids:
            raise ValueError("capability_request_id was already used")
        self._capability_request_ids.add(request_id)

    def _validate_capability_operation(self, name: str, arguments: Any) -> None:
        if name not in self._manifest.capabilities:
            raise PermissionError(f"capability is not in the manifest: {name!r}")
        if not isinstance(arguments, dict):
            raise TypeError("capability arguments must be an object")

    async def _send_capability_result(
        self,
        request_id: Any,
        *,
        result: JsonValue | None = None,
        exception: ExecutionException | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "protocol_version": _PROTOCOL_VERSION,
            "type": "capability_result",
            "capability_request_id": request_id,
        }
        if exception is None:
            payload.update(status="complete", result=result)
        else:
            payload.update(
                status="error",
                exception={"type": exception.type, "message": exception.message},
            )
        await self._send(payload)

    def _result_from_response(
        self,
        response: dict[str, Any],
        request_id: int,
        records: tuple[CapabilityRecord, ...],
    ) -> SterileExecutionResult[ReturnT]:
        if response.get("type") != "execution_result":
            raise RuntimeError("sterile worker did not return an execution result")
        if response.get("request_id") != request_id:
            raise RuntimeError("sterile worker response did not match its request")
        status = response.get("status")
        if status not in {"complete", "error", "returned"}:
            raise RuntimeError(f"invalid sterile worker status: {status!r}")
        stdout = str(response.get("stdout", ""))
        stderr = str(response.get("stderr", ""))
        locals_preview = {
            str(name): str(preview)
            for name, preview in dict(response.get("locals_preview", {})).items()
        }
        exception = self._execution_exception(response.get("exception"))
        if self._provenance is not None:
            self._provenance.cell_finished(
                request_id,
                status=status,
                stdout=stdout,
                stderr=stderr,
                locals_preview=locals_preview,
                exception=(
                    {"type": exception.type, "message": exception.message}
                    if exception is not None
                    else None
                ),
            )
        return_value = self._validated_return(response, status, request_id)
        return SterileExecutionResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            locals_preview=locals_preview,
            return_value=return_value,
            exception=exception,
            capability_records=records,
        )

    def _validated_return(
        self, response: dict[str, Any], status: str, cell_seq: int
    ) -> ReturnT | None:
        if status != "returned":
            return None
        proposed = response.get("return_payload")
        if self._provenance is not None:
            self._provenance.return_proposed(cell_seq, proposed)
        try:
            value = self._return_adapter.validate_python(proposed)
        except ValidationError as error:
            if self._provenance is not None:
                self._provenance.return_rejected(cell_seq, str(error))
            raise SterileReturnValidationError(str(error)) from error
        self._returned = True
        if self._provenance is not None:
            accepted = self._return_adapter.dump_python(value, mode="json")
            self._provenance.return_accepted(cell_seq, accepted)
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
            if self._provenance is not None and self._provenance.state == "running":
                self._provenance.fail(
                    "closed_without_terminal_return",
                    locals_lost=True,
                )

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


def _json_copy(value: Any) -> JsonValue:
    """Collapse handler output to the only representation allowed across T1/T2."""
    return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))
