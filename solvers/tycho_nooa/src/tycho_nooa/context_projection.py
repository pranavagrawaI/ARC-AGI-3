"""Bounded provider views derived from lossless Native and CodeAct histories."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from tycho_nooa.execution_events import (
    ExecutionProvenanceEvent,
    NooaExecutionProjection,
    render_execution_context,
)


class ContextProjector:
    """Keep recent control topology exact and replace older spans with facts."""

    def __init__(self, *, recent_exchanges: int = 2, receipt_limit: int = 16):
        if recent_exchanges < 1 or receipt_limit < 1:
            raise ValueError("context projection limits must be positive")
        self._recent_exchanges = recent_exchanges
        self._receipt_limit = receipt_limit

    def native(
        self,
        messages: list[dict[str, Any]],
        *,
        receipts: tuple[dict[str, Any], ...] = (),
    ) -> list[dict[str, Any]]:
        if len(messages) <= 2:
            return _json_copy(messages)
        exchanges = _native_exchanges(messages[2:])
        if len(exchanges) <= self._recent_exchanges:
            return _json_copy(messages)
        old = exchanges[: -self._recent_exchanges]
        recent = exchanges[-self._recent_exchanges :]
        summary = _native_summary(old, receipts[-self._receipt_limit :])
        return _json_copy(
            [
                *messages[:2],
                {
                    "role": "user",
                    "content": "CONTEXT_HISTORY_COLLAPSED: " + _canonical_json(summary),
                },
                *(message for exchange in recent for message in exchange),
            ]
        )

    def codeact(
        self, events: tuple[ExecutionProvenanceEvent, ...]
    ) -> list[dict[str, Any]]:
        if not events:
            return []
        manager = NooaExecutionProjection(events).event_manager()
        chunk_starts = _execution_chunk_starts(events)
        collapsed_count = len(chunk_starts) - self._recent_exchanges
        if collapsed_count > 0:
            cutoff = chunk_starts[collapsed_count]
            collapsed = [
                event
                for event in events
                if events[0].event_seq < event.event_seq < cutoff
            ]
            keys = manager.keys()
            manager.collapse(
                keys[1],
                keys[collapsed_count],
                summary_text=_canonical_json(_codeact_summary(collapsed)),
            )
        return [
            message
            for message in render_execution_context(manager)
            if message.get("content") or message.get("role") != "system"
        ]


def _native_exchanges(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    exchanges: list[list[dict[str, Any]]] = []
    for message in messages:
        if message.get("role") == "assistant" or not exchanges:
            exchanges.append([])
        exchanges[-1].append(message)
    return exchanges


def _execution_chunk_starts(events: tuple[ExecutionProvenanceEvent, ...]) -> list[int]:
    cells = {
        event.cell_seq: event.event_seq
        for event in events
        if event.event_type == "CellDispatched"
    }
    rejections = [
        event.event_seq
        for event in events
        if event.event_type == "ProviderTurnRejected"
    ]
    return sorted([*cells.values(), *rejections])


def _codeact_summary(events: list[ExecutionProvenanceEvent]) -> dict[str, Any]:
    event_counts = _event_counts(events)
    capability_counts = _capability_counts(events)
    cells = _cell_numbers(events)
    outcomes = _outcomes(events)
    handles = _receipt_handles(events)
    return {
        "schema": "codeact_history_collapse.v1",
        "collapsed_cells": len(cells),
        "cell_range": [cells[0], cells[-1]] if cells else None,
        "event_counts": dict(sorted(event_counts.items())),
        "capability_counts": dict(sorted(capability_counts.items())),
        "stdout_bytes": _output_bytes(outcomes, "stdout"),
        "stderr_bytes": _output_bytes(outcomes, "stderr"),
        "recent_receipt_handles": handles[-16:],
        "raw_events_sha256": _sha256([event.payload_sha256 for event in events]),
    }


def _event_counts(events: list[ExecutionProvenanceEvent]) -> Counter[str]:
    return Counter(event.event_type for event in events)


def _capability_counts(events: list[ExecutionProvenanceEvent]) -> Counter[str]:
    return Counter(
        str(event.payload.get("name"))
        for event in events
        if event.event_type == "CapabilityRequested"
    )


def _cell_numbers(events: list[ExecutionProvenanceEvent]) -> list[int]:
    return sorted({event.cell_seq for event in events if event.cell_seq is not None})


def _outcomes(
    events: list[ExecutionProvenanceEvent],
) -> list[ExecutionProvenanceEvent]:
    return [
        event for event in events if event.event_type in {"CellCompleted", "CellFailed"}
    ]


def _receipt_handles(events: list[ExecutionProvenanceEvent]) -> list[str]:
    handles = (
        _receipt_handle(event.payload)
        for event in events
        if event.event_type == "CapabilityReturned"
    )
    return [handle for handle in handles if handle is not None]


def _output_bytes(events: list[ExecutionProvenanceEvent], field: str) -> int:
    return sum(len(str(event.payload.get(field, "")).encode()) for event in events)


def _receipt_handle(payload: dict[str, Any]) -> str | None:
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    handle = result.get("handle")
    return handle if isinstance(handle, str) else None


def _native_summary(
    exchanges: list[list[dict[str, Any]]],
    receipts: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    messages = [message for exchange in exchanges for message in exchange]
    tools = Counter(
        str(call.get("function", {}).get("name"))
        for message in messages
        for call in message.get("tool_calls", [])
    )
    return {
        "schema": "native_history_collapse.v1",
        "collapsed_exchanges": len(exchanges),
        "message_counts": dict(
            sorted(Counter(str(message.get("role")) for message in messages).items())
        ),
        "tool_counts": dict(sorted(tools.items())),
        "active_receipts": [_receipt_orientation(value) for value in receipts],
        "raw_messages_sha256": _sha256(messages),
    }


def _receipt_orientation(receipt: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "handle",
        "content_hash",
        "query_hash",
        "refs",
        "record_count",
        "encoding",
        "frontier",
    )
    return {field: receipt.get(field) for field in fields if field in receipt}


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))
