"""Authority-free evidence values and exact row decoding for model computation."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any, Literal

MAX_DECODED_ROW_CELLS = 4_096
RLE_HELPER_SURFACE = {
    "namespace": "rle",
    "method": "decode_row",
    "signature": "decode_row(encoded: str) -> tuple[int, ...]",
    "max_decoded_cells": MAX_DECODED_ROW_CELLS,
}
RLEDecodeErrorCode = Literal[
    "empty_encoding",
    "invalid_encoding",
    "invalid_run_length",
    "decoded_size_exceeded",
]


class RLEDecodeError(ValueError):
    """Stable semantic rejection shared by every decoder surface."""

    def __init__(self, code: RLEDecodeErrorCode, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


def decode_rle_row(encoded: str) -> tuple[int, ...]:
    """Decode comma-separated ``<hex count>:<hex cell>`` runs exactly."""
    if not isinstance(encoded, str):
        raise RLEDecodeError("invalid_encoding", "RLE row must be text")
    if not encoded:
        raise RLEDecodeError("empty_encoding", "RLE row must not be empty")
    cells: list[int] = []
    for run in encoded.split(","):
        try:
            count_text, cell_text = run.split(":")
            count, cell = int(count_text, 16), int(cell_text, 16)
        except (ValueError, TypeError) as error:
            raise RLEDecodeError(
                "invalid_encoding", f"invalid RLE run: {run!r}"
            ) from error
        if cell < 0:
            raise RLEDecodeError(
                "invalid_encoding", f"cell values must be non-negative: {run!r}"
            )
        if count < 1:
            raise RLEDecodeError(
                "invalid_run_length", f"run length must be positive: {run!r}"
            )
        if len(cells) + count > MAX_DECODED_ROW_CELLS:
            raise RLEDecodeError(
                "decoded_size_exceeded",
                f"decoded row exceeds {MAX_DECODED_ROW_CELLS} cells",
            )
        cells.extend([cell] * count)
    return tuple(cells)


def describe_evidence_page(value: dict[str, Any]) -> dict[str, Any]:
    """Attach a compact, serializable contract without changing record values."""
    result = json.loads(json.dumps(value))
    result["schema"] = "evidence_page.v1"
    result["usage"] = {
        "records": "page.records or page['records']",
        "record_ref": "record.ref or record['ref']",
        "rle_grids": "record.prev_rows/next_rows; decode with rle.decode_row(row)",
    }
    return result


class EvidenceRecordView(Mapping[str, Any]):
    """Read-only record view whose attribute names match its serialized fields."""

    def __init__(self, value: dict[str, Any]):
        self._value = value

    def __getitem__(self, key: str) -> Any:
        return self._value[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __getattr__(self, name: str) -> Any:
        if name == "prev_rows":
            return self._value["prev"]
        if name == "next_rows":
            return self._value["next"]
        try:
            return self._value[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __repr__(self) -> str:
        fields = ", ".join(sorted(self._value))
        return f"EvidenceRecord(ref={self._value.get('ref')!r}, fields=[{fields}])"


class EvidencePageView(Mapping[str, Any]):
    """Read-only page view with discoverable records and representation metadata."""

    def __init__(self, value: dict[str, Any]):
        if value.get("schema") != "evidence_page.v1":
            raise ValueError("unsupported evidence page schema")
        self._value = value
        self.records = tuple(EvidenceRecordView(row) for row in value["records"])

    def __getitem__(self, key: str) -> Any:
        return self.records if key == "records" else self._value[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._value[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __repr__(self) -> str:
        return (
            f"EvidencePage(schema='evidence_page.v1', records={len(self.records)}, "
            f"grid_encoding={self._value.get('grid_encoding')!r}, "
            f"usage={self._value.get('usage')!r})"
        )


class RLEHelpers:
    """Small model-facing namespace backed by the same audited pure function."""

    @staticmethod
    def decode_row(encoded: str) -> tuple[int, ...]:
        return decode_rle_row(encoded)

    def __repr__(self) -> str:
        return (
            "RLEHelpers(decode_row(encoded: str) -> tuple[int, ...], "
            f"max_decoded_cells={MAX_DECODED_ROW_CELLS})"
        )
