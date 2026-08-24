#!/usr/bin/env python3
"""Replay the preserved Native call-40 and six-cell CodeAct context growth."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tycho_nooa.cognitive_parity import evaluate_answer, load_packet
from tycho_nooa.compatibility_authorization_v2 import measure_provider_request
from tycho_nooa.computational_evidence import decode_rle_row
from tycho_nooa.context_evidence import (
    ContextEvidencePlane,
    EvidenceAccessContext,
    EvidenceAdmissionAuthority,
    EvidenceCache,
    EvidenceGrantRegistry,
    EvidenceReadBroker,
)
from tycho_nooa.context_projection import ContextProjector
from tycho_nooa.evidence_v2 import WorkspaceEvidenceCapability
from tycho_nooa.execution_events import ExecutionProvenanceEvent

ROOT = Path(__file__).resolve().parents[2]
V1_RUN = ROOT / (
    "analysis/matched-actor-v1-pilot/runs/20260824T140538Z-deepseek-v4-flash-deepinfra"
)
V3_RUN = ROOT / (
    "analysis/matched-actor-v3-pilot/runs/20260824T160655Z-deepseek-v4-flash-deepinfra"
)
OUTPUT = Path(__file__).with_name("replay-report.json")
NATIVE_PACKET = ROOT / (
    "analysis/codeact-cognitive-parity/v1/packets/objective-success-level-boundary.json"
)
CODEACT_PACKET = ROOT / (
    "analysis/codeact-cognitive-parity/v1/packets/history-success-level1-noop.json"
)
CODEACT_VALUE_REPLAY = ROOT / (
    "analysis/matched-actor-v3-pilot/offline-replay/"
    "history-success-level1-noop/codeact/provenance.jsonl"
)


def main() -> None:
    native = _native_call_40_replay()
    codeact = _codeact_six_cell_replay()
    codeact_value = _codeact_value_answer_replay()
    report = {
        "schema_version": 1,
        "native_call_40": native,
        "codeact_six_cell": codeact,
        "codeact_value_answer": codeact_value,
        "success": {
            "native_request_admissible": native["new"]["input_token_bound"] <= 16_384,
            "native_exact_values_preserved": native["exact_values_preserved"],
            "native_answer_attainable": native["answer_attainability"]["correct"],
            "native_payload_growth_externalized": (
                native["new"]["serialized_bytes"]
                < native["old"]["serialized_bytes"] // 2
            ),
            "codeact_request_admissible": (
                codeact["new"]["input_token_bound"] <= 16_384
            ),
            "codeact_recent_history_preserved": codeact["recent_history_preserved"],
            "codeact_exact_values_preserved": codeact_value["exact_values_preserved"],
            "codeact_answer_attainable": codeact_value["answer_attainability"][
                "correct"
            ],
            "codeact_old_history_collapsed": (
                codeact["new"]["serialized_bytes"] < codeact["old"]["serialized_bytes"]
            ),
        },
    }
    if not all(report["success"].values()):
        raise RuntimeError(f"context/evidence replay failed: {report['success']}")
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def _native_call_40_replay() -> dict[str, Any]:
    request = _planned_request(V1_RUN / "authorization-events.jsonl", "call-0040")
    projected = deepcopy(request)
    messages = projected["messages"]
    query_by_call = _query_arguments(messages)
    frontier = _first_evidence_payload(messages)["frontier"]
    context = EvidenceAccessContext(
        session_id="replay-native",
        invocation_id="call-0040",
        role="actor",
        frontier=frontier,
    )
    cache = EvidenceCache()
    grants = EvidenceGrantRegistry()
    names = iter(f"ev_replay_{index}" for index in range(1, 100))
    admission = EvidenceAdmissionAuthority(
        cache,
        grants,
        clock=lambda: datetime(2026, 8, 24, tzinfo=UTC),
        handle_factory=lambda: next(names),
    )
    broker = EvidenceReadBroker(
        cache,
        grants,
        clock=lambda: datetime(2026, 8, 24, tzinfo=UTC),
    )
    receipts: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    exact_values_preserved = True
    for message in messages:
        if message.get("role") != "tool":
            continue
        payload = _evidence_payload(message.get("content"))
        if payload is None:
            continue
        query = query_by_call[str(message["tool_call_id"])]
        receipt = admission.admit(payload, query=query, context=context)
        exact_values_preserved &= cache.read(receipt.content_hash) == payload
        pages.append(broker.read(receipt.handle, context=context))
        value = receipt.model_dump(mode="json")
        receipts.append(value)
        message["content"] = _canonical_json(value)
    projected["messages"] = ContextProjector(recent_exchanges=2).native(
        messages,
        receipts=tuple(receipts),
    )
    return {
        "source_call_id": "call-0040",
        "old": _measurement(request),
        "new": _measurement(projected),
        "externalized_entries": [
            {
                "handle": receipt["handle"],
                "content_hash": receipt["content_hash"],
                "serialized_bytes": receipt["serialized_bytes"],
                "refs": receipt["refs"],
            }
            for receipt in receipts
        ],
        "exact_values_preserved": exact_values_preserved,
        "answer_attainability": _native_answer_attainability(pages),
        "projected_request_sha256": _sha256(projected),
    }


def _codeact_six_cell_replay() -> dict[str, Any]:
    request = _planned_request(V3_RUN / "authorization-events.jsonl", "call-0012")
    provenance_path = V3_RUN / (
        "pairs/action-success-grounded/codeact/provenance.jsonl"
    )
    events = tuple(
        ExecutionProvenanceEvent.model_validate_json(line)
        for line in provenance_path.read_text().splitlines()
        if line.strip()
    )
    # call-0012 was planned after cells 1-5 and before its response became cell 6.
    provider_event_frontier = tuple(
        event
        for event in events
        if event.event_type == "InvocationStarted"
        or (event.cell_seq is not None and event.cell_seq <= 5)
    )
    projected = deepcopy(request)
    bounded_history = ContextProjector(recent_exchanges=2).codeact(
        provider_event_frontier
    )
    projected["messages"] = [*request["messages"][:2], *bounded_history]
    old_tail = request["messages"][-4:]
    new_tail = projected["messages"][-4:]
    return {
        "source_call_id": "call-0012",
        "source_cells": len(
            {event.cell_seq for event in events if event.cell_seq is not None}
        ),
        "old": _measurement(request),
        "new": _measurement(projected),
        "recent_history_preserved": [_semantic_message(message) for message in old_tail]
        == [_semantic_message(message) for message in new_tail],
        "projected_request_sha256": _sha256(projected),
    }


def _native_answer_attainability(pages: list[dict[str, Any]]) -> dict[str, Any]:
    packet = load_packet(NATIVE_PACKET, repository_root=ROOT)
    records = [record for page in pages for record in page["records"]]
    transitions = [
        record
        for record in records
        if record.get("type") == "transition" and record.get("level") == 0
    ]
    frames = [
        record
        for record in records
        if record.get("type") == "frame" and record.get("level") == 1
    ]
    last_transition = max(transitions, key=lambda record: record["turn"])
    first_frame = min(frames, key=lambda record: record["turn"])
    supported = "H_LEFT" if last_transition["action"] == "LEFT" else "H_OTHER"
    answer = {
        "task_kind": "discriminate",
        "supported": supported,
        "falsified": ["H_OTHER" if supported == "H_LEFT" else "H_LEFT"],
        "evidence_refs": [
            {"level": last_transition["level"], "turn": last_transition["turn"]},
            {"level": first_frame["level"], "turn": first_frame["turn"]},
        ],
        "conclusion": (
            f"{last_transition['action']} at {last_transition['ref']} immediately "
            f"precedes the first level-1 frame {first_frame['ref']}."
        ),
    }
    evaluation = evaluate_answer(packet, answer).model_dump(mode="json")
    return {
        "answer": answer,
        "derived_from_refs": [last_transition["ref"], first_frame["ref"]],
        "evaluation": evaluation,
        "correct": (
            evaluation["verdict_correct"] is True
            and evaluation["falsification_correct"] is True
            and evaluation["evidence_recall"] == 1.0
        ),
    }


def _codeact_value_answer_replay() -> dict[str, Any]:
    packet = load_packet(CODEACT_PACKET, repository_root=ROOT)
    events = tuple(
        ExecutionProvenanceEvent.model_validate_json(line)
        for line in CODEACT_VALUE_REPLAY.read_text().splitlines()
        if line.strip()
    )
    request = next(
        event.payload["arguments"]
        for event in events
        if event.event_type == "CapabilityRequested"
        and event.payload.get("name") == "evidence.query"
    )
    preserved = next(
        event.payload["result"]
        for event in events
        if event.event_type == "CapabilityReturned"
    )
    preserved_evidence = _semantic_evidence_payload(preserved)
    accepted = next(
        event.payload for event in events if event.event_type == "ReturnAccepted"
    )
    source = WorkspaceEvidenceCapability.from_root(ROOT / packet.checkpoint_root)
    context = EvidenceAccessContext(
        session_id="replay-codeact",
        invocation_id="history-success-level1-noop:codeact",
        role="actor",
        frontier=packet.task.evidence_frontier.model_dump(mode="json"),
    )
    cache = EvidenceCache()
    grants = EvidenceGrantRegistry()
    admission = EvidenceAdmissionAuthority(
        cache,
        grants,
        handle_factory=lambda: "ev_replay_codeact",
    )
    broker = EvidenceReadBroker(cache, grants)
    plane = ContextEvidencePlane(source, admission, broker, context)
    replayed = source.query(request)
    receipt = admission.admit(preserved_evidence, query=request, context=context)
    page = plane.read({"handle": receipt.handle})
    target = next(record for record in page["records"] if record["ref"] == "L1:T1")
    unchanged = [decode_rle_row(row) for row in target["prev"]] == [
        decode_rle_row(row) for row in target["next"]
    ]
    answer = {
        "task_kind": "discriminate",
        "supported": "H_history" if unchanged else "H_visual",
        "falsified": ["H_visual" if unchanged else "H_history"],
        "evidence_refs": [{"level": target["level"], "turn": target["turn"]}],
        "conclusion": (
            f"{target['ref']} records {target['action']} with "
            f"{'identical' if unchanged else 'changed'} before and after grids."
        ),
    }
    evaluation = evaluate_answer(packet, answer).model_dump(mode="json")
    plane.close()
    return {
        "source_provenance": str(CODEACT_VALUE_REPLAY.relative_to(ROOT)),
        "query": request,
        "preserved_envelope_sha256": _sha256(preserved),
        "preserved_evidence_sha256": _sha256(preserved_evidence),
        "replayed_payload_sha256": _sha256(replayed),
        "brokered_payload_sha256": _sha256(page),
        "exact_values_preserved": preserved_evidence == replayed == page,
        "historical_return_accepted": accepted,
        "answer": answer,
        "derived_from_refs": [target["ref"]],
        "receipt": receipt.model_dump(mode="json"),
        "evaluation": evaluation,
        "answer_attainability": {
            "correct": (
                unchanged
                and evaluation["verdict_correct"] is True
                and evaluation["falsification_correct"] is True
                and evaluation["evidence_recall"] == 1.0
            )
        },
    }


def _semantic_evidence_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Remove only the old model-facing envelope around exact evidence values."""
    return {key: item for key, item in value.items() if key not in {"schema", "usage"}}


def _planned_request(path: Path, call_id: str) -> dict[str, Any]:
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if (
            event.get("event_type") == "CallPlanned"
            and event.get("payload", {}).get("call_id") == call_id
        ):
            return event["payload"]["request"]
    raise RuntimeError(f"preserved provider request is unavailable: {call_id}")


def _query_arguments(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(call["id"]): json.loads(call["function"]["arguments"])
        for message in messages
        for call in message.get("tool_calls", [])
        if call.get("function", {}).get("name") == "evidence_query"
    }


def _first_evidence_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in messages:
        payload = _evidence_payload(message.get("content"))
        if payload is not None:
            return payload
    raise RuntimeError("preserved request contains no evidence payload")


def _evidence_payload(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        return None
    return value


def _measurement(request: dict[str, Any]) -> dict[str, int]:
    measured = measure_provider_request(request, default_max_output_tokens=4_096)
    return {
        "serialized_bytes": measured.request_serialized_bytes,
        "input_token_bound": measured.request_input_token_bound,
        "total_token_bound": measured.request_total_token_bound,
        "messages": len(request["messages"]),
    }


def _semantic_message(message: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(message)
    content = value.get("content")
    if (
        isinstance(content, list)
        and len(content) == 1
        and isinstance(content[0], dict)
        and isinstance(content[0].get("text"), str)
    ):
        value["content"] = content[0]["text"]
    return value


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
