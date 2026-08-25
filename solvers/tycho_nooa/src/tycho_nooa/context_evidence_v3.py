"""Bulk acquisition over Context/Evidence Plane v2 immutable grants."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tycho_nooa.context_evidence_v2 import ContextEvidencePlane as _V2Plane
from tycho_nooa.evidence import EvidenceRef

ACQUISITION_PAGE_RECORDS = 32
MAX_ACQUIRED_RECORDS = 256
MAX_ACQUIRED_BYTES = 524_288


class EvidenceAcquireArguments(BaseModel):
    """Select one complete bounded evidence set for external-memory admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["frames", "transitions"]
    level: int | None = Field(default=None, ge=0)
    frontier: EvidenceRef | None = None
    grid_encoding: Literal["raw", "rle"] = "rle"
    projection: Literal["full", "metadata"] = "full"


class ContextEvidencePlane(_V2Plane):
    """Add complete-set acquisition without adding a new dereference authority."""

    def acquire(self, raw_request: dict[str, Any]) -> dict[str, Any]:
        request = EvidenceAcquireArguments.model_validate(raw_request)
        payload = self._materialize(request)
        receipt = self._admission.admit(
            payload,
            query={"operation": "acquire", **request.model_dump(mode="json")},
            context=self._context,
        )
        return receipt.model_dump(mode="json")

    def _materialize(self, request: EvidenceAcquireArguments) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        after: dict[str, int] | None = None
        seen_cursors: set[tuple[int, int]] = set()
        first_page: dict[str, Any] | None = None
        while True:
            query = request.model_dump(mode="json", exclude_none=True)
            query["limit"] = ACQUISITION_PAGE_RECORDS
            if after is not None:
                query["after"] = after
            page = self._source.query(query)
            _validate_page(page, request)
            if first_page is None:
                first_page = page
            elif page.get("frontier") != first_page.get("frontier"):
                raise RuntimeError("evidence frontier changed during acquisition")
            page_records = page["records"]
            records.extend(page_records)
            _enforce_acquisition_bounds(first_page, records)
            if not page["truncated"]:
                break
            after = _next_cursor(page)
            cursor = (after["level"], after["turn"])
            if cursor in seen_cursors:
                raise RuntimeError("evidence acquisition cursor did not advance")
            seen_cursors.add(cursor)
        if first_page is None:
            raise RuntimeError("evidence acquisition returned no page")
        return {
            **first_page,
            "records": records,
            "next_after": None,
            "truncated": False,
            "truncation_reason": None,
        }


def _validate_page(page: dict[str, Any], request: EvidenceAcquireArguments) -> None:
    if not isinstance(page, dict) or not isinstance(page.get("records"), list):
        raise TypeError("evidence acquisition page is malformed")
    expected = request.model_dump(mode="json")
    for field in ("kind", "grid_encoding", "projection"):
        if page.get(field) != expected[field]:
            raise RuntimeError(f"evidence acquisition page changed {field}")
    if not isinstance(page.get("truncated"), bool):
        raise TypeError("evidence acquisition page omits truncation state")


def _next_cursor(page: dict[str, Any]) -> dict[str, int]:
    cursor = page.get("next_after")
    if not isinstance(cursor, dict) or set(cursor) != {"level", "turn"}:
        raise RuntimeError("truncated evidence page has no complete cursor")
    if not all(isinstance(cursor[name], int) for name in ("level", "turn")):
        raise TypeError("evidence acquisition cursor fields must be integers")
    return {"level": cursor["level"], "turn": cursor["turn"]}


def _enforce_acquisition_bounds(
    first_page: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    if len(records) > MAX_ACQUIRED_RECORDS:
        raise ValueError(
            f"evidence acquisition exceeds {MAX_ACQUIRED_RECORDS} records"
        )
    candidate = {
        **first_page,
        "records": records,
        "next_after": None,
        "truncated": False,
        "truncation_reason": None,
    }
    size = len(
        json.dumps(candidate, allow_nan=False, separators=(",", ":")).encode("utf-8")
    )
    if size > MAX_ACQUIRED_BYTES:
        raise ValueError(
            f"evidence acquisition exceeds {MAX_ACQUIRED_BYTES} serialized bytes"
        )
