"""T1-owned CodeAct provenance and its bounded NOOA-visible projection."""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any, ClassVar, Literal
from uuid import NAMESPACE_URL, uuid5

from nooa.context_blocks.formatter import OpenAIProviderFormatter, XMLBlockFormatter
from nooa.context_blocks.models import Role
from nooa.context_blocks.renderer import render_context
from nooa.events import EventBase
from nooa.runtime.context_builder import _phase_events
from nooa.runtime.event_manager import EventManager
from pydantic import BaseModel, ConfigDict, Field, JsonValue

ExecutionEventType = Literal[
    "InvocationStarted",
    "ProviderTurnRejected",
    "CellDispatched",
    "CapabilityRequested",
    "CapabilityReturned",
    "CapabilityRejected",
    "CellCompleted",
    "CellFailed",
    "CellTimedOut",
    "WorkerLost",
    "ReturnProposed",
    "ReturnAccepted",
    "ReturnRejected",
    "InvocationCompleted",
    "InvocationFailed",
]


class ExecutionProvenanceEvent(BaseModel):
    """One lossless fact assigned and committed by the trusted control plane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    event_id: str
    event_type: ExecutionEventType
    actor_session_id: str
    invocation_id: str
    event_seq: int = Field(ge=1)
    cell_seq: int | None = Field(default=None, ge=1)
    request_id: int | None = Field(default=None, ge=1)
    parent_event_id: str | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    payload_sha256: str


class ExecutionProvenanceLog:
    """Append-only JSONL with validated identity, sequence, and payload hashes."""

    def __init__(self, path: Path, *, actor_session_id: str):
        self.path = Path(path)
        self.actor_session_id = actor_session_id
        self._events = self._load()

    @property
    def events(self) -> tuple[ExecutionProvenanceEvent, ...]:
        return tuple(self._events)

    def append(
        self,
        event_type: ExecutionEventType,
        *,
        invocation_id: str,
        cell_seq: int | None = None,
        request_id: int | None = None,
        parent_event_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ExecutionProvenanceEvent:
        value = _json_value(payload or {})
        if not isinstance(value, dict):
            raise TypeError("provenance payload must be an object")
        event_seq = len(self._events) + 1
        event = ExecutionProvenanceEvent(
            event_id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"tycho:{self.actor_session_id}:{event_seq}",
                )
            ),
            event_type=event_type,
            actor_session_id=self.actor_session_id,
            invocation_id=invocation_id,
            event_seq=event_seq,
            cell_seq=cell_seq,
            request_id=request_id,
            parent_event_id=parent_event_id,
            payload=value,
            payload_sha256=_payload_sha256(value),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._events.append(event)
        return event

    def _load(self) -> list[ExecutionProvenanceEvent]:
        if not self.path.exists():
            return []
        events = [
            ExecutionProvenanceEvent.model_validate_json(line)
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]
        _validate_loaded_events(events, actor_session_id=self.actor_session_id)
        return events


def _validate_loaded_events(
    events: list[ExecutionProvenanceEvent], *, actor_session_id: str
) -> None:
    """Reject a structurally forged or partially rewritten durable history."""
    known_ids: set[str] = set()
    for expected_seq, event in enumerate(events, start=1):
        _validate_actor_session(event, actor_session_id)
        _validate_event_sequence(event, expected_seq)
        _validate_event_identity(event, known_ids)
        _validate_event_parent(event, known_ids)
        _validate_payload_hash(event)
        known_ids.add(event.event_id)


def _validate_actor_session(
    event: ExecutionProvenanceEvent, actor_session_id: str
) -> None:
    if event.actor_session_id != actor_session_id:
        raise ValueError("provenance actor_session_id mismatch")


def _validate_event_sequence(
    event: ExecutionProvenanceEvent, expected_seq: int
) -> None:
    if event.event_seq != expected_seq:
        raise ValueError("provenance event sequence is not contiguous")


def _validate_event_identity(
    event: ExecutionProvenanceEvent, known_ids: set[str]
) -> None:
    if event.event_id in known_ids:
        raise ValueError("provenance event identity is duplicated")


def _validate_event_parent(
    event: ExecutionProvenanceEvent, known_ids: set[str]
) -> None:
    if event.parent_event_id is not None and event.parent_event_id not in known_ids:
        raise ValueError("provenance parent must precede its child")


def _validate_payload_hash(event: ExecutionProvenanceEvent) -> None:
    if event.payload_sha256 != _payload_sha256(event.payload):
        raise ValueError("provenance payload hash mismatch")


class _CellState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dispatched_event_id: str
    outcome_event_id: str | None = None
    requests: dict[int, str] = Field(default_factory=dict)
    resolved_requests: set[int] = Field(default_factory=set)
    proposal_event_id: str | None = None
    proposal_resolved: bool = False


class InvocationProvenance:
    """State machine for one CodeAct invocation within an Actor session log."""

    def __init__(self, log: ExecutionProvenanceLog, *, invocation_id: str):
        self.log = log
        self.invocation_id = invocation_id
        self._state: Literal["created", "running", "completed", "failed"] = "created"
        self._root_event_id: str | None = None
        self._cells: dict[int, _CellState] = {}

    @property
    def state(self) -> str:
        return self._state

    def start(self, manifest: dict[str, Any]) -> None:
        self._require_state("created")
        event = self.log.append(
            "InvocationStarted",
            invocation_id=self.invocation_id,
            payload={"manifest": manifest},
        )
        self._root_event_id = event.event_id
        self._state = "running"

    def cell_dispatched(self, cell_seq: int, source: str) -> None:
        self._require_running()
        if cell_seq in self._cells:
            raise RuntimeError(f"cell {cell_seq} was already dispatched")
        event = self.log.append(
            "CellDispatched",
            invocation_id=self.invocation_id,
            cell_seq=cell_seq,
            parent_event_id=self._root_event_id,
            payload={"source": source, "source_sha256": _text_sha256(source)},
        )
        self._cells[cell_seq] = _CellState(dispatched_event_id=event.event_id)

    def provider_turn_rejected(self, reason: str) -> None:
        self._require_running()
        self.log.append(
            "ProviderTurnRejected",
            invocation_id=self.invocation_id,
            parent_event_id=self._root_event_id,
            payload={
                "reason": reason,
                "instruction": (
                    "Retry using exactly one execute_python or return_result tool call."
                ),
            },
        )

    def capability_requested(
        self,
        cell_seq: int,
        request_id: int,
        name: str,
        arguments: Any,
    ) -> None:
        cell = self._open_cell(cell_seq)
        if request_id in cell.requests:
            raise RuntimeError(f"capability request {request_id} was already recorded")
        event = self.log.append(
            "CapabilityRequested",
            invocation_id=self.invocation_id,
            cell_seq=cell_seq,
            request_id=request_id,
            parent_event_id=cell.dispatched_event_id,
            payload={"name": name, "arguments": arguments},
        )
        cell.requests[request_id] = event.event_id

    def capability_resolved(
        self,
        cell_seq: int,
        request_id: int,
        *,
        result: Any = None,
        exception: dict[str, Any] | None = None,
    ) -> None:
        cell = self._open_cell(cell_seq)
        parent = cell.requests.get(request_id)
        if parent is None or request_id in cell.resolved_requests:
            raise RuntimeError(f"capability request {request_id} is not open")
        rejected = exception is not None
        payload = {"result": result} if not rejected else {"exception": exception}
        self.log.append(
            "CapabilityRejected" if rejected else "CapabilityReturned",
            invocation_id=self.invocation_id,
            cell_seq=cell_seq,
            request_id=request_id,
            parent_event_id=parent,
            payload=payload,
        )
        cell.resolved_requests.add(request_id)

    def cell_finished(
        self,
        cell_seq: int,
        *,
        status: Literal["complete", "error", "returned"],
        stdout: str,
        stderr: str,
        locals_preview: dict[str, str],
        exception: dict[str, Any] | None,
    ) -> None:
        cell = self._open_cell(cell_seq)
        if set(cell.requests) != cell.resolved_requests:
            raise RuntimeError("cell cannot finish with unresolved capability requests")
        failed = status == "error"
        payload: dict[str, Any] = {
            "status": status,
            "stdout": stdout,
            "stderr": stderr,
            "locals_preview": locals_preview,
        }
        if exception is not None:
            payload["exception"] = exception
        event = self.log.append(
            "CellFailed" if failed else "CellCompleted",
            invocation_id=self.invocation_id,
            cell_seq=cell_seq,
            parent_event_id=cell.dispatched_event_id,
            payload=payload,
        )
        cell.outcome_event_id = event.event_id

    def return_proposed(self, cell_seq: int, value: Any) -> None:
        cell = self._closed_cell(cell_seq)
        if cell.proposal_event_id is not None:
            raise RuntimeError("cell already proposed a return")
        event = self.log.append(
            "ReturnProposed",
            invocation_id=self.invocation_id,
            cell_seq=cell_seq,
            parent_event_id=cell.outcome_event_id,
            payload={"value": value},
        )
        cell.proposal_event_id = event.event_id

    def return_rejected(self, cell_seq: int, error: str) -> None:
        cell = self._open_proposal(cell_seq)
        self.log.append(
            "ReturnRejected",
            invocation_id=self.invocation_id,
            cell_seq=cell_seq,
            parent_event_id=cell.proposal_event_id,
            payload={"error": error},
        )
        cell.proposal_resolved = True

    def return_accepted(self, cell_seq: int, value: Any) -> None:
        cell = self._open_proposal(cell_seq)
        accepted = self.log.append(
            "ReturnAccepted",
            invocation_id=self.invocation_id,
            cell_seq=cell_seq,
            parent_event_id=cell.proposal_event_id,
            payload={"value": value},
        )
        cell.proposal_resolved = True
        self.log.append(
            "InvocationCompleted",
            invocation_id=self.invocation_id,
            parent_event_id=accepted.event_id,
            payload={"terminal_cell_seq": cell_seq},
        )
        self._state = "completed"

    def cell_timed_out(self, cell_seq: int, wall_time_seconds: float) -> None:
        cell = self._open_cell(cell_seq)
        event = self.log.append(
            "CellTimedOut",
            invocation_id=self.invocation_id,
            cell_seq=cell_seq,
            parent_event_id=cell.dispatched_event_id,
            payload={
                "wall_time_seconds": wall_time_seconds,
                "locals_lost": True,
            },
        )
        cell.outcome_event_id = event.event_id
        self.fail("cell_timeout", parent_event_id=event.event_id, locals_lost=True)

    def worker_lost(self, cell_seq: int, reason: str) -> None:
        cell = self._open_cell(cell_seq)
        event = self.log.append(
            "WorkerLost",
            invocation_id=self.invocation_id,
            cell_seq=cell_seq,
            parent_event_id=cell.dispatched_event_id,
            payload={"reason": reason, "locals_lost": True},
        )
        cell.outcome_event_id = event.event_id
        self.fail("worker_lost", parent_event_id=event.event_id, locals_lost=True)

    def fail(
        self,
        reason: str,
        *,
        parent_event_id: str | None = None,
        locals_lost: bool,
    ) -> None:
        if self._state in {"completed", "failed"}:
            return
        self._require_running()
        self.log.append(
            "InvocationFailed",
            invocation_id=self.invocation_id,
            parent_event_id=parent_event_id or self._root_event_id,
            payload={"reason": reason, "locals_lost": locals_lost},
        )
        self._state = "failed"

    def _require_running(self) -> None:
        self._require_state("running")

    def _require_state(self, expected: str) -> None:
        if self._state != expected:
            raise RuntimeError(f"invocation is {self._state}, expected {expected}")

    def _open_cell(self, cell_seq: int) -> _CellState:
        self._require_running()
        cell = self._cells.get(cell_seq)
        if cell is None or cell.outcome_event_id is not None:
            raise RuntimeError(f"cell {cell_seq} is not open")
        return cell

    def _closed_cell(self, cell_seq: int) -> _CellState:
        self._require_running()
        cell = self._cells.get(cell_seq)
        if cell is None or cell.outcome_event_id is None:
            raise RuntimeError(f"cell {cell_seq} has not completed")
        return cell

    def _open_proposal(self, cell_seq: int) -> _CellState:
        cell = self._closed_cell(cell_seq)
        if cell.proposal_event_id is None or cell.proposal_resolved:
            raise RuntimeError(f"cell {cell_seq} has no open return proposal")
        return cell


class CodeActCellHistory(EventBase):
    _role: ClassVar[Role] = Role.ASSISTANT

    content: str


class CodeActResultHistory(EventBase):
    _role: ClassVar[Role] = Role.USER

    content: str


class CodeActLifecycleHistory(EventBase):
    _role: ClassVar[Role] = Role.USER

    content: str


class NooaExecutionProjection:
    """Deterministically render semantically useful history from raw provenance."""

    def __init__(self, events: tuple[ExecutionProvenanceEvent, ...]):
        self._events = events

    def event_manager(self) -> EventManager:
        manager = EventManager()
        for event in self._project():
            manager.add(event)
        return manager

    def _project(self) -> list[EventBase]:
        projected: list[EventBase] = []
        invocations = _group_by(self._events, lambda event: event.invocation_id)
        for invocation_events in invocations:
            projected.extend(self._project_invocation(invocation_events))
        return projected

    def _project_invocation(
        self, events: list[ExecutionProvenanceEvent]
    ) -> list[EventBase]:
        started = events[0]
        projected: list[EventBase] = [
            _projection_event(
                CodeActLifecycleHistory,
                started,
                "CodeAct invocation started "
                f"(invocation_id={started.invocation_id!r}).",
                [started],
            )
        ]
        for _, chunk in _projection_chunks(events):
            projected.extend(chunk)
        projected.extend(_terminal_projection(events))
        return projected


def _project_provider_rejection(event: ExecutionProvenanceEvent) -> EventBase:
    return _projection_event(
        CodeActResultHistory,
        event,
        _safe_text("ProviderTurnRejected: " + _canonical_json(event.payload)),
        [event],
    )


def _projection_chunks(
    events: list[ExecutionProvenanceEvent],
) -> list[tuple[int, list[EventBase]]]:
    cells = _group_by(
        [event for event in events if event.cell_seq is not None],
        lambda event: event.cell_seq,
    )
    chunks = [
        (cell_events[0].event_seq, _project_cell(cell_events))
        for cell_events in cells
    ]
    chunks.extend(
        (event.event_seq, [_project_provider_rejection(event)])
        for event in events
        if event.event_type == "ProviderTurnRejected"
    )
    return sorted(chunks, key=lambda item: item[0])


def _terminal_projection(
    events: list[ExecutionProvenanceEvent],
) -> list[EventBase]:
    terminal = next(
        (
            event
            for event in reversed(events)
            if event.event_type in {"InvocationCompleted", "InvocationFailed"}
        ),
        None,
    )
    if terminal is None:
        return []
    return [
        _projection_event(
            CodeActLifecycleHistory,
            terminal,
            _safe_text(
                f"CodeAct {terminal.event_type}: "
                + _canonical_json(terminal.payload)
            ),
            [terminal],
        )
    ]


def _project_cell(events: list[ExecutionProvenanceEvent]) -> list[EventBase]:
    dispatched = next(event for event in events if event.event_type == "CellDispatched")
    source = str(dispatched.payload["source"])
    cell_event = _projection_event(
        CodeActCellHistory,
        dispatched,
        (
            f"# CodeAct cell {dispatched.cell_seq}\n"
            f"# source_sha256={dispatched.payload['source_sha256']}\n"
            + _safe_text(source)
        ),
        [dispatched],
    )
    outcome = next(
        (
            event
            for event in events
            if event.event_type
            in {"CellCompleted", "CellFailed", "CellTimedOut", "WorkerLost"}
        ),
        None,
    )
    if outcome is None:
        return [cell_event]
    result_content = {
        "cell_seq": dispatched.cell_seq,
        "outcome": outcome.event_type,
        "execution": outcome.payload,
        "capabilities": _capability_summaries(events),
        "return": _return_summary(events),
    }
    result_event = _projection_event(
        CodeActResultHistory,
        outcome,
        _safe_text(_canonical_json(result_content)),
        events,
    )
    return [cell_event, result_event]


def _capability_summaries(
    events: list[ExecutionProvenanceEvent],
) -> list[dict[str, Any]]:
    requests = {
        event.request_id: event
        for event in events
        if event.event_type == "CapabilityRequested"
    }
    resolutions = {
        event.request_id: event
        for event in events
        if event.event_type in {"CapabilityReturned", "CapabilityRejected"}
    }
    return [
        _capability_summary(request_id, request, resolutions.get(request_id))
        for request_id, request in requests.items()
    ]


def _capability_summary(
    request_id: int | None,
    request: ExecutionProvenanceEvent,
    resolution: ExecutionProvenanceEvent | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "request_id": request_id,
        "name": request.payload.get("name"),
        "arguments": request.payload.get("arguments"),
        "status": resolution.event_type if resolution is not None else "unresolved",
    }
    if resolution is None:
        return summary
    summary["payload_sha256"] = resolution.payload_sha256
    summary["result_shape"] = _result_shape(resolution.payload)
    return summary


def _return_summary(events: list[ExecutionProvenanceEvent]) -> dict[str, Any] | None:
    proposed = _first_event(events, {"ReturnProposed"})
    if proposed is None:
        return None
    resolution = _first_event(events, {"ReturnAccepted", "ReturnRejected"})
    return {
        "proposed": proposed.payload,
        "resolution": resolution.event_type if resolution is not None else "unresolved",
        "detail": resolution.payload if resolution is not None else None,
    }


def _first_event(
    events: list[ExecutionProvenanceEvent], event_types: set[ExecutionEventType]
) -> ExecutionProvenanceEvent | None:
    return next(
        (event for event in events if event.event_type in event_types),
        None,
    )


def _result_shape(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    result = payload.get("result")
    if isinstance(result, dict):
        shape: dict[str, JsonValue] = {"keys": sorted(result)}
        records = result.get("records")
        if isinstance(records, list):
            shape["records"] = len(records)
        if "frontier" in result:
            shape["frontier"] = result["frontier"]
        return shape
    if isinstance(result, list):
        return {"type": "list", "items": len(result)}
    return {"type": type(result).__name__}


def _projection_event(
    cls: type[EventBase],
    identity: ExecutionProvenanceEvent,
    content: str,
    raw_events: list[ExecutionProvenanceEvent],
) -> EventBase:
    return cls(
        id=f"{identity.event_id}:projection",
        content=content,
        metadata={
            "invocation_id": identity.invocation_id,
            "raw_event_ids": [event.event_id for event in raw_events],
        },
    )


def render_execution_context(manager: EventManager) -> list[dict[str, Any]]:
    """Render the exact OpenAI-compatible context derived from visible events."""
    blocks = _phase_events([], manager)
    rendered = render_context(
        blocks,
        block_formatter=XMLBlockFormatter(),
        provider_formatter=OpenAIProviderFormatter(),
    )
    if not isinstance(rendered.output, list):
        raise TypeError("OpenAI provider formatter returned a non-list context")
    return rendered.output


def _group_by(items: Any, key: Any) -> list[list[Any]]:
    groups: list[list[Any]] = []
    identities: list[Any] = []
    for item in items:
        identity = key(item)
        if identity not in identities:
            identities.append(identity)
            groups.append([])
        groups[identities.index(identity)].append(item)
    return groups


def _json_value(value: Any) -> JsonValue:
    return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _payload_sha256(payload: dict[str, JsonValue]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _safe_text(value: str) -> str:
    """Keep untrusted bytes legible without allowing XML-like role injection."""
    return html.escape(value, quote=True)
