from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

import tycho_nooa.matched_actor_pilot_run as pilot_v1
import tycho_nooa.matched_actor_pilot_run_v2 as pilot_v2
from tycho_nooa.compatibility_authorization_v2 import (
    CompatibilityAuthorization,
    DurableAuthorizationLedger,
    PricingSchedule,
    authorization_sha256,
)
from tycho_nooa.evidence_v2 import WorkspaceEvidenceCapability, serialized_evidence_size

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
V1_RUN = REPOSITORY_ROOT / (
    "analysis/matched-actor-v1-pilot/runs/20260824T140538Z-deepseek-v4-flash-deepinfra"
)


def _authorization() -> CompatibilityAuthorization:
    return CompatibilityAuthorization(
        authorization_id="call-40-replay",
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


def _preserved_call_40_request() -> dict:
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


def test_preserved_call_40_is_rejected_before_dispatch(tmp_path: Path) -> None:
    auth = _authorization()
    path = tmp_path / "call-40-v2-replay-events.jsonl"
    ledger = DurableAuthorizationLedger.create(
        path,
        authorization=auth,
        authorization_sha256=authorization_sha256(auth),
    )
    with pytest.raises(ValueError, match="pre-dispatch input-token bound"):
        ledger.plan_call(
            gate_id="objective-success-level-boundary",
            arm="native",
            request=_preserved_call_40_request(),
        )

    assert ledger.state.calls == {}
    assert '"event_type":"CallDispatched"' not in path.read_text()


def test_call_40_evidence_query_is_payload_bounded_before_request_admission() -> None:
    packet = json.loads(
        (
            REPOSITORY_ROOT / "analysis/codeact-cognitive-parity/v1/packets/"
            "objective-success-level-boundary.json"
        ).read_text()
    )
    capability = WorkspaceEvidenceCapability.from_root(
        REPOSITORY_ROOT / packet["checkpoint_root"]
    )

    page = capability.query({"kind": "transitions", "level": 0, "grid_encoding": "rle"})

    assert len(page["records"]) < 16
    assert page["truncated"] is True
    assert page["truncation_reason"] == "payload_limit"
    assert serialized_evidence_size(page) <= capability.max_payload_bytes


def test_native_and_codeact_receive_identical_bounded_evidence_values() -> None:
    packet = json.loads(
        (
            REPOSITORY_ROOT / "analysis/codeact-cognitive-parity/v1/packets/"
            "objective-success-level-boundary.json"
        ).read_text()
    )
    root = REPOSITORY_ROOT / packet["checkpoint_root"]
    query = {"kind": "transitions", "level": 0, "grid_encoding": "rle"}

    native_value = WorkspaceEvidenceCapability.from_root(root).query(query)
    codeact_value = WorkspaceEvidenceCapability.from_root(root).query(query)

    assert native_value == codeact_value
    assert native_value["truncation_reason"] == "payload_limit"


def test_v2_reuses_v1_assessment_and_advancement_semantics() -> None:
    assert pilot_v2.PilotAssessment is pilot_v1.PilotAssessment
    assert pilot_v2.assess_pilot is pilot_v1.assess_pilot
    assert pilot_v2._codeact_advances is pilot_v1._codeact_advances
    assert pilot_v2._disposition is pilot_v1._disposition
