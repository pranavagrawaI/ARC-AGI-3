"""Read-only value transport over Tycho's authoritative observation history."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from tycho.workspace import wmlib_template
from tycho.workspace.workspace import GameWorkspace

Grid = tuple[tuple[int, ...], ...]


class EvidenceFrontierMismatchError(RuntimeError):
    """A paginated query no longer names the current authoritative frontier."""


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: int = Field(ge=0)
    turn: int = Field(ge=0)


class FrameEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["frame"] = "frame"
    level: int = Field(ge=0)
    turn: int = Field(ge=0)
    grid: Grid


class TransitionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["transition"] = "transition"
    level: int = Field(ge=0)
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


class FrameMetadataEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["frame_metadata"] = "frame_metadata"
    level: int = Field(ge=0)
    turn: int = Field(ge=0)


class TransitionMetadataEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["transition_metadata"] = "transition_metadata"
    level: int = Field(ge=0)
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
    grid_encoding: Literal["raw", "rle"]
    projection: Literal["full", "metadata"]
    encoding_legend: str | None = None


class WorkspaceEvidenceCapability:
    """Translate a live GameWorkspace into bounded immutable evidence DTOs."""

    def __init__(self, workspace: GameWorkspace):
        self._root = workspace.dir

    @classmethod
    def from_root(cls, root: Path) -> WorkspaceEvidenceCapability:
        """Open an offline frozen workspace without constructing or mutating it."""
        capability = cls.__new__(cls)
        capability._root = Path(root)
        return capability

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
        selected_page = records[: query.limit]
        page = tuple(_project_record(record, query.projection) for record in selected_page)
        truncated = len(records) > len(selected_page)
        result = EvidenceQueryResult(
            kind=query.kind,
            frontier=frontier,
            records=page,
            next_after=_record_ref(page[-1]) if truncated else None,
            truncated=truncated,
            grid_encoding=query.grid_encoding,
            projection=query.projection,
            encoding_legend=(
                "Each grid row is comma-separated <hex count>:<hex cell> runs."
                if query.projection == "full" and query.grid_encoding == "rle"
                else None
            ),
        )
        value = result.model_dump(mode="json")
        if query.projection == "full" and query.grid_encoding == "rle":
            _encode_record_grids(value["records"])
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
