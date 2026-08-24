"""Immutable evidence memory with names separated from read authority."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, NewType, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

DEFAULT_GRANT_LIFETIME = timedelta(minutes=10)
DEFAULT_VIEW_BYTES = 2_048
MAX_VIEW_BYTES = 8_192
EvidenceHandle = NewType("EvidenceHandle", str)


class EvidenceAccessDenied(PermissionError):
    """A name was valid, but the current principal has no matching read grant."""


class EvidenceIntegrityError(RuntimeError):
    """Cached bytes no longer match the immutable entry they claim to represent."""


class EvidenceAccessContext(BaseModel):
    """The complete principal identity considered by the reference monitor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1, max_length=128)
    invocation_id: str = Field(min_length=1, max_length=128)
    role: Literal["actor", "builder"]
    frontier: JsonValue


class EvidenceEntry(BaseModel):
    """Stable identity and structural metadata for one immutable JSON value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content_hash: str
    frontier_hash: str
    schema_version: str
    serialized_bytes: int = Field(ge=0)


class EvidenceGrant(BaseModel):
    """Short-lived read permission; neither the handle nor hash grants access."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    handle: EvidenceHandle
    content_hash: str
    session_id: str
    invocation_id: str
    role: Literal["actor", "builder"]
    frontier_hash: str
    expires_at: datetime
    operations: tuple[Literal["read", "view"], ...] = ("read", "view")


class EvidenceReceipt(BaseModel):
    """Small non-semantic description safe to place in provider context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["evidence_receipt.v1"] = "evidence_receipt.v1"
    handle: EvidenceHandle
    content_hash: str
    query_hash: str
    refs: tuple[str, ...]
    record_count: int = Field(ge=0)
    record_fields: tuple[str, ...]
    encoding: str | None
    frontier: JsonValue
    serialized_bytes: int = Field(ge=0)


class EvidenceViewRequest(BaseModel):
    """A bounded projection of records already admitted under an existing grant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    handle: EvidenceHandle
    refs: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    rows: tuple[int, ...] = ()
    max_bytes: int = Field(default=DEFAULT_VIEW_BYTES, ge=64, le=MAX_VIEW_BYTES)


class EvidenceSource(Protocol):
    def query(self, raw_query: dict[str, Any]) -> dict[str, Any]: ...


class EvidenceCache:
    """Content-addressed immutable values retained independently of grant expiry."""

    def __init__(self) -> None:
        self._entries: dict[str, EvidenceEntry] = {}
        self._payloads: dict[str, bytes] = {}

    def store(
        self,
        payload: dict[str, Any],
        *,
        frontier: JsonValue,
        schema_version: str = "evidence_page.v1",
    ) -> EvidenceEntry:
        encoded = _canonical_json_bytes(payload)
        content_hash = _sha256(encoded)
        entry = EvidenceEntry(
            content_hash=content_hash,
            frontier_hash=_json_hash(frontier),
            schema_version=schema_version,
            serialized_bytes=len(encoded),
        )
        incumbent = self._entries.get(content_hash)
        if incumbent is not None and incumbent != entry:
            raise EvidenceIntegrityError("content identity has inconsistent metadata")
        self._entries[content_hash] = entry
        self._payloads[content_hash] = encoded
        return entry

    def read(self, content_hash: str) -> dict[str, Any]:
        entry = self._entries.get(content_hash)
        encoded = self._payloads.get(content_hash)
        if entry is None or encoded is None:
            raise EvidenceAccessDenied("evidence entry is unavailable")
        if _sha256(encoded) != entry.content_hash:
            raise EvidenceIntegrityError("cached payload content hash mismatch")
        value = json.loads(encoded)
        if not isinstance(value, dict):
            raise EvidenceIntegrityError("cached evidence is not an object")
        return value


class EvidenceGrantRegistry:
    """T1-owned bindings shared by admission and the read reference monitor."""

    def __init__(self) -> None:
        self._handles: dict[EvidenceHandle, str] = {}
        self._grants: dict[tuple[str, str, str, str], EvidenceGrant] = {}

    def bind(
        self,
        handle: EvidenceHandle,
        entry: EvidenceEntry,
        grant: EvidenceGrant,
        context: EvidenceAccessContext,
    ) -> None:
        if handle in self._handles:
            raise RuntimeError("evidence handle is already bound")
        self._handles[handle] = entry.content_hash
        self._grants[_grant_key(handle, context)] = grant

    def resolve(self, handle: str) -> str | None:
        return self._handles.get(handle)

    def grant_for(
        self, handle: str, context: EvidenceAccessContext
    ) -> EvidenceGrant | None:
        return self._grants.get(_grant_key(handle, context))

    def revoke_invocation(self, context: EvidenceAccessContext) -> None:
        prefix = (context.session_id, context.invocation_id, context.role)
        self._grants = {
            key: grant for key, grant in self._grants.items() if key[:3] != prefix
        }


class EvidenceAdmissionAuthority:
    """Acquire, name, and grant evidence without exposing a dereference path."""

    def __init__(
        self,
        cache: EvidenceCache,
        grants: EvidenceGrantRegistry,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        handle_factory: Callable[[], str] = lambda: f"ev_{uuid4().hex}",
    ) -> None:
        self._cache = cache
        self._grants = grants
        self._clock = clock
        self._handle_factory = handle_factory

    def admit(
        self,
        payload: dict[str, Any],
        *,
        query: dict[str, Any],
        context: EvidenceAccessContext,
        grant_lifetime: timedelta = DEFAULT_GRANT_LIFETIME,
    ) -> EvidenceReceipt:
        if grant_lifetime <= timedelta(0):
            raise ValueError("grant lifetime must be positive")
        frontier = _json_value(payload.get("frontier"))
        if _json_hash(frontier) != _json_hash(context.frontier):
            raise EvidenceAccessDenied(
                "evidence frontier does not match the invocation"
            )
        entry = self._cache.store(payload, frontier=frontier)
        handle = self._new_handle()
        grant = EvidenceGrant(
            handle=handle,
            content_hash=entry.content_hash,
            session_id=context.session_id,
            invocation_id=context.invocation_id,
            role=context.role,
            frontier_hash=entry.frontier_hash,
            expires_at=self._now() + grant_lifetime,
        )
        self._grants.bind(handle, entry, grant, context)
        return _receipt(handle, entry, payload, query, frontier)

    def revoke_invocation(self, context: EvidenceAccessContext) -> None:
        self._grants.revoke_invocation(context)

    def _new_handle(self) -> EvidenceHandle:
        handle = EvidenceHandle(self._handle_factory())
        if not handle.startswith("ev_") or self._grants.resolve(handle) is not None:
            raise RuntimeError("handle factory did not produce a fresh evidence name")
        return handle

    def _now(self) -> datetime:
        return _aware_now(self._clock)


class EvidenceReadBroker:
    """Reference monitor: read or view one granted immutable value, nothing else."""

    def __init__(
        self,
        cache: EvidenceCache,
        grants: EvidenceGrantRegistry,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._cache = cache
        self._grants = grants
        self._clock = clock

    def read(self, handle: str, *, context: EvidenceAccessContext) -> dict[str, Any]:
        grant = self._authorized_grant(handle, context, operation="read")
        payload = self._cache.read(grant.content_hash)
        if _json_hash(payload.get("frontier")) != grant.frontier_hash:
            raise EvidenceIntegrityError("cached evidence frontier hash mismatch")
        return payload

    def view(
        self,
        request: EvidenceViewRequest,
        *,
        context: EvidenceAccessContext,
    ) -> dict[str, Any]:
        grant = self._authorized_grant(request.handle, context, operation="view")
        payload = self._cache.read(grant.content_hash)
        records = _selected_records(payload, request)
        value = {"handle": request.handle, "records": records}
        if len(_canonical_json_bytes(value)) > request.max_bytes:
            raise ValueError(
                f"evidence view exceeds {request.max_bytes} bytes; select fewer values"
            )
        return value

    def _authorized_grant(
        self,
        handle: str,
        context: EvidenceAccessContext,
        *,
        operation: Literal["read", "view"],
    ) -> EvidenceGrant:
        resolved_hash = self._grants.resolve(handle)
        grant = self._grants.grant_for(handle, context)
        if resolved_hash is None or grant is None:
            raise EvidenceAccessDenied("evidence handle is not granted")
        if operation not in grant.operations:
            raise EvidenceAccessDenied(
                f"evidence operation {operation!r} is not granted"
            )
        if grant.expires_at <= _aware_now(self._clock):
            raise EvidenceAccessDenied("evidence grant expired")
        if grant.content_hash != resolved_hash:
            raise EvidenceAccessDenied(
                "evidence handle no longer resolves to its grant"
            )
        if grant.frontier_hash != _json_hash(context.frontier):
            raise EvidenceAccessDenied("evidence grant frontier mismatch")
        return grant


class ContextEvidencePlane:
    """Acquire through one source; reread only through its scoped broker grant."""

    def __init__(
        self,
        source: EvidenceSource,
        admission: EvidenceAdmissionAuthority,
        broker: EvidenceReadBroker,
        context: EvidenceAccessContext,
    ) -> None:
        self._source = source
        self._admission = admission
        self._broker = broker
        self._context = context

    def query(self, raw_query: dict[str, Any]) -> dict[str, JsonValue]:
        payload = self._source.query(raw_query)
        receipt = self._admission.admit(
            payload,
            query=raw_query,
            context=self._context,
        )
        return receipt.model_dump(mode="json")

    def read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        handle = _single_handle(arguments)
        return self._broker.read(handle, context=self._context)

    def view(self, arguments: dict[str, Any]) -> dict[str, Any]:
        request = EvidenceViewRequest.model_validate(arguments)
        return self._broker.view(request, context=self._context)

    def close(self) -> None:
        self._admission.revoke_invocation(self._context)


def _single_handle(arguments: dict[str, Any]) -> str:
    if set(arguments) != {"handle"} or not isinstance(arguments["handle"], str):
        raise TypeError("evidence.read requires exactly one text handle")
    return arguments["handle"]


def _receipt(
    handle: str,
    entry: EvidenceEntry,
    payload: dict[str, Any],
    query: dict[str, Any],
    frontier: JsonValue,
) -> EvidenceReceipt:
    records = payload.get("records", [])
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise TypeError("evidence payload records must be objects")
    fields = sorted({str(field) for record in records for field in record})
    return EvidenceReceipt(
        handle=handle,
        content_hash=entry.content_hash,
        query_hash=_json_hash(query),
        refs=tuple(_record_ref(record) for record in records),
        record_count=len(records),
        record_fields=tuple(fields),
        encoding=(
            str(payload["grid_encoding"])
            if payload.get("grid_encoding") is not None
            else None
        ),
        frontier=frontier,
        serialized_bytes=entry.serialized_bytes,
    )


def _selected_records(
    payload: dict[str, Any], request: EvidenceViewRequest
) -> list[dict[str, Any]]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise EvidenceIntegrityError("cached evidence records are not a list")
    by_ref = {_record_ref(record): record for record in records}
    refs = request.refs or tuple(by_ref)
    missing = [ref for ref in refs if ref not in by_ref]
    if missing:
        raise EvidenceAccessDenied(
            f"requested evidence is not present in the granted entry: {missing!r}"
        )
    return [_project_record(by_ref[ref], request) for ref in refs]


def _project_record(
    record: dict[str, Any], request: EvidenceViewRequest
) -> dict[str, Any]:
    fields = request.fields or tuple(str(field) for field in record)
    unknown = [field for field in fields if field not in record]
    if unknown:
        raise ValueError(f"evidence fields are unavailable: {unknown!r}")
    return {field: _selected_rows(record[field], request.rows) for field in fields}


def _selected_rows(value: Any, rows: tuple[int, ...]) -> Any:
    if not rows or not isinstance(value, list):
        return value
    try:
        return [value[index] for index in rows]
    except IndexError as error:
        raise ValueError("evidence row index is unavailable") from error


def _record_ref(record: dict[str, Any]) -> str:
    direct = record.get("ref")
    if isinstance(direct, str):
        return direct
    level, turn = record.get("level"), record.get("turn")
    if isinstance(level, int) and isinstance(turn, int):
        return f"L{level}:T{turn}"
    raise TypeError("evidence record has no canonical ref")


def _grant_key(
    handle: str, context: EvidenceAccessContext
) -> tuple[str, str, str, str]:
    return (context.session_id, context.invocation_id, context.role, handle)


def _aware_now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("evidence grant clock must return a timezone-aware value")
    return value


def _json_hash(value: Any) -> str:
    return _sha256(_canonical_json_bytes(value))


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_value(value: Any) -> JsonValue:
    return json.loads(_canonical_json_bytes(value))
