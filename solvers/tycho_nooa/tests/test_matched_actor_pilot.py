from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.compatibility_authorization import (
    DurableAuthorizationLedger,
    PricingSchedule,
    authorization_sha256,
)
from tycho_nooa.matched_actor_audit import _flat_expected_answer
from tycho_nooa.matched_actor_manifest import sha256_file
from tycho_nooa.matched_actor_pilot import load_matched_actor_pilot_sample
from tycho_nooa.matched_actor_pilot_authorization import (
    MatchedActorPilotAuthorization,
)
from tycho_nooa.matched_actor_pilot_launch import (
    execute_authorized_matched_actor_pilot,
    preflight_matched_actor_pilot,
)
from tycho_nooa.matched_actor_pilot_manifest import (
    load_matched_actor_pilot_manifest,
)
from tycho_nooa.matched_actor_pilot_run import (
    _codeact_advances,
    _disposition,
    execute_pilot_pair,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASE = REPOSITORY_ROOT / "analysis/matched-actor-v1-pilot"
SAMPLE = BASE / "sample.json"
MANIFEST = BASE / "experiment.json"


def response(name: str, arguments: dict, *, provider: str = "provider-a") -> LLMResponse:
    return LLMResponse(
        raw_response={"provider": provider, "model": "reported-fake-model"},
        content="",
        tool_calls=[
            ToolCall(id=f"{name}-id", name=name, arguments=json.dumps(arguments))
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 20, "completion_tokens": 10},
    )


def configured_fake(*responses: LLMResponse) -> FakeLLMClient:
    client = FakeLLMClient(scripted_responses=list(responses))
    client.config["reasoning"] = {"enabled": True}
    return client


def authorized_inputs(
    tmp_path: Path, *, maximum_spend: Decimal = Decimal(3)
) -> tuple[Path, Path]:
    raw = json.loads(MANIFEST.read_text())
    raw["execution_authorized"] = True
    raw["model_route"] = {
        "model": "fake-model",
        "reported_model": "reported-fake-model",
        "provider": "provider-a",
        "reasoning_effort": "enabled",
        "temperature": None,
    }
    raw["caps"]["max_cost_usd"] = float(maximum_spend)
    manifest_path = tmp_path / "experiment.json"
    manifest_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    authorization = MatchedActorPilotAuthorization(
        authorization_id="matched-pilot-test",
        experiment_manifest_sha256=sha256_file(manifest_path),
        allowed_provider="provider-a",
        requested_model="fake-model",
        expected_reported_model="reported-fake-model",
        allowed_reasoning_effort="enabled",
        maximum_calls=72,
        maximum_spend_usd=maximum_spend,
        max_input_tokens_per_call=16384,
        max_output_tokens_per_call=4096,
        expires_at=datetime.now(UTC) + timedelta(days=1),
        pricing=PricingSchedule(
            provider="provider-a",
            model="fake-model",
            input_usd_per_million=Decimal("0.1"),
            cached_input_usd_per_million=Decimal("0.05"),
            output_usd_per_million=Decimal("0.2"),
            source="test schedule",
            effective_at=datetime(2026, 8, 24, tzinfo=UTC),
        ),
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(authorization.model_dump_json(indent=2) + "\n")
    return manifest_path, authorization_path


def test_pilot_is_a_balanced_counterbalanced_view_of_frozen_v1() -> None:
    sample, packets = load_matched_actor_pilot_sample(
        SAMPLE, repository_root=REPOSITORY_ROOT
    )

    assert len(packets) == 6
    assert [record.arm_order for record in sample.packets] == [
        ("native", "codeact"),
        ("codeact", "native"),
    ] * 3
    assert {record.stratum for record in sample.packets} == {
        "computation_unnecessary",
        "computation_useful",
        "composition",
    }


def test_disabled_pilot_refuses_paid_execution() -> None:
    manifest, packets = load_matched_actor_pilot_manifest(
        MANIFEST, repository_root=REPOSITORY_ROOT
    )

    assert len(packets) == 6
    with pytest.raises(RuntimeError, match="execution is disabled"):
        manifest.require_execution_authorized()


async def test_pair_failure_still_runs_counterpart_when_integrity_survives(
    tmp_path: Path,
) -> None:
    sample, packets = load_matched_actor_pilot_sample(
        SAMPLE, repository_root=REPOSITORY_ROOT
    )
    packet = packets[0]
    native = configured_fake(*[response("wrong", {}) for _ in range(6)])
    codeact = configured_fake(response("return_result", _flat_expected_answer(packet)))

    report = await execute_pilot_pair(
        packet,
        stratum=sample.packets[0].stratum,
        arm_order=sample.packets[0].arm_order,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "pair",
        native_llm=native,
        codeact_llm=codeact,
        max_provider_calls=6,
        max_output_tokens=4096,
        session_timeout_seconds=20,
        cell_timeout_seconds=5,
    )

    assert [arm.status for arm in report.arms] == ["failed", "completed"]
    assert native.call_count == 6
    assert codeact.call_count == 1


async def test_authorized_pilot_obeys_frozen_counterbalanced_schedule(
    tmp_path: Path,
) -> None:
    manifest_path, authorization_path = authorized_inputs(tmp_path)
    sample, packets = load_matched_actor_pilot_sample(
        SAMPLE, repository_root=REPOSITORY_ROOT
    )
    native_answers = [
        response("return_result", _flat_expected_answer(packet)) for packet in packets
    ]
    codeact_answers = [
        response("return_result", _flat_expected_answer(packet)) for packet in packets
    ]

    assessment = await execute_authorized_matched_actor_pilot(
        manifest_path,
        authorization_path,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "run",
        native_llm=configured_fake(*native_answers),
        codeact_llm=configured_fake(*codeact_answers),
    )

    assert assessment.complete_pairs == 6
    assert assessment.native_correct == assessment.codeact_correct == 6
    assert assessment.native_cost_usd > 0
    assert assessment.codeact_cost_usd > 0
    assert assessment.disposition == "select_native"
    events = [
        json.loads(line)
        for line in (tmp_path / "run/authorization-events.jsonl").read_text().splitlines()
    ]
    planned = [event for event in events if event["event_type"] == "CallPlanned"]
    assert [event["payload"]["arm"] for event in planned] == [
        arm for record in sample.packets for arm in record.arm_order
    ]
    assert events[-1]["event_type"] == "RunCompleted"


async def test_route_drift_is_preserved_accounted_and_stops_pilot(
    tmp_path: Path,
) -> None:
    manifest_path, authorization_path = authorized_inputs(tmp_path)
    native = configured_fake(response("return_result", {}, provider="provider-b"))
    codeact = configured_fake()

    with pytest.raises(RuntimeError, match="route_violation"):
        await execute_authorized_matched_actor_pilot(
            manifest_path,
            authorization_path,
            repository_root=REPOSITORY_ROOT,
            destination=tmp_path / "run",
            native_llm=native,
            codeact_llm=codeact,
        )

    events = (tmp_path / "run/authorization-events.jsonl").read_text()
    assert '"event_type":"CallReceived"' in events
    assert '"event_type":"RouteViolated"' in events
    assert '"event_type":"CallAccounted"' in events
    assert native.call_count == 1
    assert codeact.call_count == 0


async def test_worst_case_cost_is_refused_before_provider_dispatch(
    tmp_path: Path,
) -> None:
    manifest_path, authorization_path = authorized_inputs(
        tmp_path, maximum_spend=Decimal("0.001")
    )
    native = configured_fake()
    codeact = configured_fake()

    with pytest.raises(RuntimeError, match="cost_reservation"):
        await execute_authorized_matched_actor_pilot(
            manifest_path,
            authorization_path,
            repository_root=REPOSITORY_ROOT,
            destination=tmp_path / "run",
            native_llm=native,
            codeact_llm=codeact,
        )

    assert native.call_count == codeact.call_count == 0
    events = (tmp_path / "run/authorization-events.jsonl").read_text()
    assert '"event_type":"CallDispatched"' not in events


def test_preflight_manifest_drift_makes_zero_calls(tmp_path: Path) -> None:
    manifest_path, authorization_path = authorized_inputs(tmp_path)
    raw = json.loads(manifest_path.read_text())
    raw["experiment_id"] = "drifted"
    manifest_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    native = configured_fake()
    codeact = configured_fake()

    with pytest.raises(ValueError, match="does not bind"):
        preflight_matched_actor_pilot(
            manifest_path,
            authorization_path,
            repository_root=REPOSITORY_ROOT,
            native_llm=native,
            codeact_llm=codeact,
        )

    assert native.call_count == codeact.call_count == 0


def test_pilot_ledger_blocks_the_seventy_third_call(tmp_path: Path) -> None:
    _, authorization_path = authorized_inputs(tmp_path)
    authorization = MatchedActorPilotAuthorization.model_validate_json(
        authorization_path.read_text()
    )
    ledger = DurableAuthorizationLedger.create(
        tmp_path / "events.jsonl",
        authorization=authorization,
        authorization_sha256=authorization_sha256(authorization),
    )
    for index in range(72):
        call = ledger.plan_call(gate_id=f"p{index}", arm="native", request={})
        ledger.dispatch(call.call_id)
        ledger.receive(
            call.call_id,
            response={},
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            observed_provider="provider-a",
            observed_model="reported-fake-model",
            observed_reasoning_effort="enabled",
        )
        ledger.reconcile(call.call_id)

    with pytest.raises(RuntimeError, match="call cap"):
        ledger.plan_call(gate_id="p72", arm="native", request={})

    assert ledger.state.committed_calls == 72


def test_pilot_recovery_stops_an_unknown_dispatched_call(tmp_path: Path) -> None:
    _, authorization_path = authorized_inputs(tmp_path)
    authorization = MatchedActorPilotAuthorization.model_validate_json(
        authorization_path.read_text()
    )
    authorization_hash = authorization_sha256(authorization)
    path = tmp_path / "events.jsonl"
    ledger = DurableAuthorizationLedger.create(
        path,
        authorization=authorization,
        authorization_sha256=authorization_hash,
    )
    call = ledger.plan_call(gate_id="p0", arm="native", request={})
    ledger.dispatch(call.call_id)

    reopened = DurableAuthorizationLedger(
        path,
        authorization=authorization,
        authorization_sha256=authorization_hash,
    )
    reopened.recover()

    assert reopened.state.run_state == "stopped"
    assert reopened.state.stop_reason == "unknown_dispatched_call"


def test_advancement_is_asymmetric_and_precommitted() -> None:
    assert _codeact_advances(
        complete_pairs=6,
        wins=2,
        losses=1,
        native_correct=4,
        codeact_correct=5,
        native_calls=20,
        codeact_calls=30,
        native_cost=1,
        codeact_cost=2,
    )
    assert _codeact_advances(
        complete_pairs=6,
        wins=0,
        losses=0,
        native_correct=6,
        codeact_correct=6,
        native_calls=20,
        codeact_calls=16,
        native_cost=1,
        codeact_cost=1,
    )
    assert not _codeact_advances(
        complete_pairs=5,
        wins=3,
        losses=0,
        native_correct=2,
        codeact_correct=5,
        native_calls=20,
        codeact_calls=10,
        native_cost=1,
        codeact_cost=0.5,
    )
    assert _disposition(
        complete_pairs=6, native_correct=3, codeact_correct=3, advances=False
    ) == "model_unreliable"
