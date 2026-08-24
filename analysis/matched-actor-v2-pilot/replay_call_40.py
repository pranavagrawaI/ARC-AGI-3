"""Zero-provider-call proof over the preserved V1 call-40 failure."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from tycho_nooa.compatibility_authorization_v2 import (
    CompatibilityAuthorization,
    DurableAuthorizationLedger,
    PricingSchedule,
    authorization_sha256,
    measure_provider_request,
)
from tycho_nooa.evidence_v2 import (
    WorkspaceEvidenceCapability,
    serialized_evidence_size,
)

REPO = Path(__file__).resolve().parents[2]
BASE = Path(__file__).parent
V1_RUN = (
    REPO
    / "analysis/matched-actor-v1-pilot/runs/20260824T140538Z-deepseek-v4-flash-deepinfra"
)
PACKET = (
    REPO
    / "analysis/codeact-cognitive-parity/v1/packets/objective-success-level-boundary.json"
)


def main() -> None:
    request = _preserved_request()
    measurement = measure_provider_request(request, default_max_output_tokens=4096)
    authorization = _authorization()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "events.jsonl"
        ledger = DurableAuthorizationLedger.create(
            path,
            authorization=authorization,
            authorization_sha256=authorization_sha256(authorization),
        )
        rejected = False
        try:
            ledger.plan_call(
                gate_id="objective-success-level-boundary",
                arm="native",
                request=request,
            )
        except ValueError:
            rejected = True
        events = [json.loads(line) for line in path.read_text().splitlines()]

    packet = json.loads(PACKET.read_text())
    evidence = WorkspaceEvidenceCapability.from_root(REPO / packet["checkpoint_root"])
    page = evidence.query({"kind": "transitions", "level": 0, "grid_encoding": "rle"})
    report = {
        "schema_version": 1,
        "provider_calls": 0,
        "preserved_request": {
            "serialized_bytes": measurement.request_serialized_bytes,
            "input_token_bound": measurement.request_input_token_bound,
            "authorized_input_ceiling": authorization.max_input_tokens_per_call,
            "rejected_before_call_planned": rejected
            and not any(event["event_type"] == "CallPlanned" for event in events),
            "dispatched": any(
                event["event_type"] == "CallDispatched" for event in events
            ),
        },
        "bounded_evidence": {
            "records": len(page["records"]),
            "serialized_bytes": serialized_evidence_size(page),
            "payload_limit_bytes": evidence.max_payload_bytes,
            "truncated": page["truncated"],
            "truncation_reason": page["truncation_reason"],
            "next_after": page["next_after"],
        },
    }
    if not (
        report["preserved_request"]["rejected_before_call_planned"]
        and not report["preserved_request"]["dispatched"]
        and report["bounded_evidence"]["serialized_bytes"] <= evidence.max_payload_bytes
        and report["bounded_evidence"]["truncation_reason"] == "payload_limit"
    ):
        raise RuntimeError("V2 call-40 containment proof failed")
    (BASE / "call-40-replay.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def _preserved_request() -> dict:
    events = [
        json.loads(line)
        for line in (V1_RUN / "authorization-events.jsonl").read_text().splitlines()
    ]
    return next(
        event["payload"]["request"]
        for event in events
        if event["event_type"] == "CallPlanned"
        and event["payload"]["call_id"] == "call-0040"
    )


def _authorization() -> CompatibilityAuthorization:
    return CompatibilityAuthorization(
        authorization_id="call-40-offline-replay",
        experiment_manifest_sha256="sha256:" + "1" * 64,
        allowed_provider="DeepInfra",
        requested_model="openai/deepseek/deepseek-v4-flash-0731",
        expected_reported_model="deepseek/deepseek-v4-flash-0731",
        allowed_reasoning_effort="enabled",
        maximum_calls=30,
        maximum_spend_usd=Decimal(3),
        max_input_tokens_per_call=16384,
        max_output_tokens_per_call=4096,
        expires_at=datetime.now(UTC) + timedelta(days=1),
        pricing=PricingSchedule(
            provider="DeepInfra",
            model="openai/deepseek/deepseek-v4-flash-0731",
            input_usd_per_million=Decimal("0.08"),
            cached_input_usd_per_million=Decimal("0.016"),
            output_usd_per_million=Decimal("0.18"),
            source="preserved V1 schedule",
            effective_at=datetime(2026, 8, 24, tzinfo=UTC),
        ),
    )


if __name__ == "__main__":
    main()
