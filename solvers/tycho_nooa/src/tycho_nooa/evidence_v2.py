"""Read-only value transport over Tycho's authoritative observation history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, computed_field
from tycho.workspace import wmlib_template
from tycho.workspace.workspace import GameWorkspace

Grid = tuple[tuple[int, ...], ...]
DEFAULT_EVIDENCE_PAYLOAD_BYTES = 8_192


class EvidenceFrontierMismatchError(RuntimeError):
    """A paginated query no longer names the current authoritative frontier."""


class EvidenceRecordTooLargeError(ValueError):
    """One atomic record cannot cross the configured disclosure boundary."""


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: int = Field(ge=0)
    turn: int = Field(ge=0)


class EvidenceRecordIdentity(BaseModel):
    """A record carries its canonical citation; cursors deliberately do not."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: int = Field(ge=0)
    turn: int = Field(ge=0)

    @computed_field
    @property
    def ref(self) -> str:
        return f"L{self.level}:T{self.turn}"


class FrameEvidence(EvidenceRecordIdentity):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["frame"] = "frame"
    grid: Grid


class TransitionEvidence(EvidenceRecordIdentity):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["transition"] = "transition"
    turn: int = Field(ge=1)
    prev: Grid
    next: Grid
    action: str | None
    row: int | None
    col: int | None


EvidenceRecord = Annotated[
    FrameEvidence | TransitionEvidence,
    Field(discriminator="type"),
]


class FrameMetadataEvidence(EvidenceRecordIdentity):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["frame_metadata"] = "frame_metadata"


class TransitionMetadataEvidence(EvidenceRecordIdentity):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["transition_metadata"] = "transition_metadata"
    turn: int = Field(ge=1)
    action: str | None
    row: int | None
    col: int | None


EvidenceMetadataRecord = Annotated[
    FrameMetadataEvidence | TransitionMetadataEvidence,
    Field(discriminator="type"),
]
TransportEvidenceRecord = Annotated[
    FrameEvidence
    | TransitionEvidence
    | FrameMetadataEvidence
    | TransitionMetadataEvidence,
    Field(discriminator="type"),
]


class EvidenceQuery(BaseModel):
    """A bounded page over current-attempt frames or ordinary transitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["frames", "transitions"]
    level: int | None = Field(default=None, ge=0)
    after: EvidenceRef | None = None
    frontier: EvidenceRef | None = None
    limit: int = Field(default=16, ge=1, le=32)
    grid_encoding: Literal["raw", "rle"] = "raw"
    projection: Literal["full", "metadata"] = "full"


class EvidenceQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["frames", "transitions"]
    frontier: EvidenceRef | None
    records: tuple[TransportEvidenceRecord, ...]
    next_after: EvidenceRef | None
    truncated: bool
    truncation_reason: Literal["record_limit", "payload_limit"] | None = None
    grid_encoding: Literal["raw", "rle"]
    projection: Literal["full", "metadata"]
    encoding_legend: str | None = None


class EvidenceCapability(Protocol):
    def query(self, raw_query: dict[str, Any] | EvidenceQuery) -> dict[str, Any]: ...


class WorkspaceEvidenceCapability:
    """Translate a live GameWorkspace into bounded immutable evidence DTOs."""

    def __init__(
        self,
        workspace: GameWorkspace,
        *,
        max_payload_bytes: int = DEFAULT_EVIDENCE_PAYLOAD_BYTES,
    ):
        self._root = workspace.dir
        self._max_payload_bytes = _payload_limit(max_payload_bytes)

    @classmethod
    def from_root(
        cls,
        root: Path,
        *,
        max_payload_bytes: int = DEFAULT_EVIDENCE_PAYLOAD_BYTES,
    ) -> WorkspaceEvidenceCapability:
        """Open an offline frozen workspace without constructing or mutating it."""
        capability = cls.__new__(cls)
        capability._root = Path(root)
        capability._max_payload_bytes = _payload_limit(max_payload_bytes)
        return capability

    @property
    def max_payload_bytes(self) -> int:
        return self._max_payload_bytes

    def query(self, raw_query: dict[str, Any] | EvidenceQuery) -> dict[str, Any]:
        query = (
            raw_query
            if isinstance(raw_query, EvidenceQuery)
            else EvidenceQuery.model_validate(raw_query)
        )
        root = str(self._root)
        frontier = _evidence_ref(wmlib_template.current_frame_key(root))
        if query.frontier is not None and query.frontier != frontier:
            raise EvidenceFrontierMismatchError(
                f"requested frontier {query.frontier} does not match current {frontier}"
            )
        records = self._records(query, root)
        page = self._bounded_page(records, query, frontier)
        return page

    def _bounded_page(
        self,
        records: list[EvidenceRecord],
        query: EvidenceQuery,
        frontier: EvidenceRef | None,
    ) -> dict[str, Any]:
        semantic_page = tuple(
            _project_record(record, query.projection)
            for record in records[: query.limit]
        )
        accepted: tuple[EvidenceRecord | EvidenceMetadataRecord, ...] = ()
        for record in semantic_page:
            trial = (*accepted, record)
            trial_reason = "payload_limit" if len(records) > len(trial) else None
            trial_value = _evidence_page_value(
                query=query,
                frontier=frontier,
                records=trial,
                truncation_reason=trial_reason,
            )
            if serialized_evidence_size(trial_value) > self._max_payload_bytes:
                if not accepted:
                    raise EvidenceRecordTooLargeError(
                        f"evidence record {_record_ref_text(record)} exceeds "
                        f"payload limit {self._max_payload_bytes} bytes"
                    )
                break
            accepted = trial
        if len(accepted) < len(semantic_page):
            reason: Literal["record_limit", "payload_limit"] | None = "payload_limit"
        elif len(records) > len(accepted):
            reason = "record_limit"
        else:
            reason = None
        value = _evidence_page_value(
            query=query,
            frontier=frontier,
            records=accepted,
            truncation_reason=reason,
        )
        if serialized_evidence_size(value) > self._max_payload_bytes:
            raise RuntimeError("evidence payload accounting is inconsistent")
        return value

    @staticmethod
    def _records(query: EvidenceQuery, root: str) -> list[EvidenceRecord]:
        if query.kind == "frames":
            records: list[EvidenceRecord] = [
                FrameEvidence(level=level, turn=turn, grid=_grid(grid))
                for (level, turn), grid in sorted(wmlib_template.frames(root).items())
            ]
        else:
            records = [
                TransitionEvidence(
                    level=int(item["level"]),
                    turn=int(item["turn"]),
                    prev=_grid(item["prev"]),
                    next=_grid(item["next"]),
                    action=item.get("action"),
                    row=item.get("row"),
                    col=item.get("col"),
                )
                for item in wmlib_template.transitions(root)
            ]
        return [record for record in records if _selected(record, query)]


def _evidence_page_value(
    *,
    query: EvidenceQuery,
    frontier: EvidenceRef | None,
    records: tuple[EvidenceRecord | EvidenceMetadataRecord, ...],
    truncation_reason: Literal["record_limit", "payload_limit"] | None,
) -> dict[str, Any]:
    result = EvidenceQueryResult(
        kind=query.kind,
        frontier=frontier,
        records=records,
        next_after=(
            _record_ref(records[-1]) if truncation_reason is not None else None
        ),
        truncated=truncation_reason is not None,
        truncation_reason=truncation_reason,
        grid_encoding=query.grid_encoding,
        projection=query.projection,
        encoding_legend=_encoding_legend(query),
    )
    value = result.model_dump(mode="json")
    _encode_query_grids(value, query)
    return value


def serialized_evidence_size(value: dict[str, Any]) -> int:
    """Measure the exact compact JSON value disclosed across either Actor surface."""
    return len(
        json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    )


def _payload_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("evidence payload limit must be a positive integer")
    return value


def _record_ref_text(record: EvidenceRecord | EvidenceMetadataRecord) -> str:
    return f"L{record.level}:T{record.turn}"


def _grid(value: Any) -> Grid:
    return tuple(tuple(int(cell) for cell in row) for row in value)


def _project_record(
    record: EvidenceRecord, projection: Literal["full", "metadata"]
) -> EvidenceRecord | EvidenceMetadataRecord:
    if projection == "full":
        return record
    if isinstance(record, FrameEvidence):
        return FrameMetadataEvidence(level=record.level, turn=record.turn)
    return TransitionMetadataEvidence(
        level=record.level,
        turn=record.turn,
        action=record.action,
        row=record.row,
        col=record.col,
    )


def _encode_record_grids(records: list[dict[str, Any]]) -> None:
    for record in records:
        for field in ("grid", "prev", "next"):
            grid = record.get(field)
            if grid is not None:
                record[field] = [_rle_row(row) for row in grid]


def _encoding_legend(query: EvidenceQuery) -> str | None:
    if not _uses_rle(query):
        return None
    return "Each grid row is comma-separated <hex count>:<hex cell> runs."


def _encode_query_grids(value: dict[str, Any], query: EvidenceQuery) -> None:
    if _uses_rle(query):
        _encode_record_grids(value["records"])


def _uses_rle(query: EvidenceQuery) -> bool:
    return query.projection == "full" and query.grid_encoding == "rle"


def _rle_row(row: list[int]) -> str:
    runs: list[str] = []
    current = row[0]
    count = 0
    for cell in row:
        if cell == current:
            count += 1
            continue
        runs.append(f"{count:x}:{current:x}")
        current = cell
        count = 1
    runs.append(f"{count:x}:{current:x}")
    return ",".join(runs)


def _evidence_ref(key: tuple[int, int] | None) -> EvidenceRef | None:
    return None if key is None else EvidenceRef(level=key[0], turn=key[1])


def _record_ref(record: EvidenceRecord | EvidenceMetadataRecord) -> EvidenceRef:
    return EvidenceRef(level=record.level, turn=record.turn)


def _selected(record: EvidenceRecord, query: EvidenceQuery) -> bool:
    if query.level is not None and record.level != query.level:
        return False
    if query.after is None:
        return True
    return (record.level, record.turn) > (query.after.level, query.after.turn)
