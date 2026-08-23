from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tycho_nooa.compatibility_authorization import (
    CompatibilityAuthorization,
    DurableAuthorizationLedger,
    PricingSchedule,
    authorization_sha256,
)


def authorization(
    *, maximum_calls: int = 30, maximum_spend: str = "3.00"
) -> CompatibilityAuthorization:
    return CompatibilityAuthorization(
        authorization_id="authorization-test",
        experiment_manifest_sha256="sha256:" + "1" * 64,
        allowed_provider="test-provider",
        requested_model="test/model",
        expected_reported_model="test-model-snapshot",
        allowed_reasoning_effort="enabled",
        maximum_calls=maximum_calls,
        maximum_spend_usd=Decimal(maximum_spend),
        max_input_tokens_per_call=1000,
        max_output_tokens_per_call=100,
        expires_at=datetime.now(UTC) + timedelta(days=1),
        pricing=PricingSchedule(
            provider="test-provider",
            model="test/model",
            input_usd_per_million=Decimal(10),
            cached_input_usd_per_million=Decimal(2),
            output_usd_per_million=Decimal(20),
            source="test schedule v1",
            effective_at=datetime(2026, 8, 23, tzinfo=UTC),
        ),
    )


def ledger(tmp_path: Path, auth: CompatibilityAuthorization) -> DurableAuthorizationLedger:
    return DurableAuthorizationLedger.create(
        tmp_path / "authorization-events.jsonl",
        authorization=auth,
        authorization_sha256=authorization_sha256(auth),
    )


def finish_call(
    value: DurableAuthorizationLedger,
    call_id: str,
    *,
    provider: str = "test-provider",
    model: str = "test-model-snapshot",
) -> Decimal:
    value.dispatch(call_id)
    value.receive(
        call_id,
        response={"content": "ok"},
        usage={
            "prompt_tokens": 500,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 100},
        },
        observed_provider=provider,
        observed_model=model,
    )
    return value.reconcile(call_id)


def test_worst_case_reservation_reconciles_to_frozen_usage_cost(
    tmp_path: Path,
) -> None:
    auth = authorization()
    value = ledger(tmp_path, auth)

    call = value.plan_call(gate_id="g0", arm="native", request={"prompt": "x"})

    assert call.reservation_usd == Decimal("0.012")
    assert value.state.committed_spend_usd == Decimal("0.012")
    actual = finish_call(value, call.call_id)
    assert actual == Decimal("0.0052")
    assert value.state.committed_spend_usd == Decimal("0.0052")

    reopened = DurableAuthorizationLedger(
        value.path,
        authorization=auth,
        authorization_sha256=authorization_sha256(auth),
    )
    assert reopened.state.committed_calls == 1
    assert reopened.state.committed_spend_usd == Decimal("0.0052")


def test_call_cap_blocks_before_an_additional_plan(tmp_path: Path) -> None:
    auth = authorization(maximum_calls=1)
    value = ledger(tmp_path, auth)
    call = value.plan_call(gate_id="g0", arm="native", request={})
    finish_call(value, call.call_id)

    with pytest.raises(RuntimeError, match="call cap"):
        value.plan_call(gate_id="g0", arm="codeact", request={})

    assert value.state.run_state == "stopped"
    assert len(value.state.calls) == 1


def test_thirty_first_call_is_blocked_by_the_frozen_global_cap(
    tmp_path: Path,
) -> None:
    auth = authorization()
    value = ledger(tmp_path, auth)
    for index in range(30):
        call = value.plan_call(gate_id=f"g{index}", arm="native", request={})
        finish_call(value, call.call_id)

    with pytest.raises(RuntimeError, match="call cap"):
        value.plan_call(gate_id="g31", arm="native", request={})

    assert value.state.committed_calls == 30
    assert value.state.stop_reason == "call_cap"


def test_worst_case_cost_blocks_before_dispatch(tmp_path: Path) -> None:
    auth = authorization(maximum_spend="0.011")
    value = ledger(tmp_path, auth)

    with pytest.raises(RuntimeError, match="cost reservation"):
        value.plan_call(gate_id="g0", arm="native", request={})

    assert value.state.run_state == "stopped"
    assert value.state.calls == {}


def test_route_violation_is_accounted_then_stops(tmp_path: Path) -> None:
    auth = authorization()
    value = ledger(tmp_path, auth)
    call = value.plan_call(gate_id="g0", arm="native", request={})

    actual = finish_call(value, call.call_id, model="wrong-model")

    assert actual == Decimal("0.0052")
    assert value.state.calls[call.call_id].state == "accounted"
    assert value.state.committed_spend_usd == actual
    assert value.state.run_state == "stopped"
    assert value.state.stop_reason == f"route_violation:{call.call_id}"


def test_recovery_accounts_a_received_response_without_retry(tmp_path: Path) -> None:
    auth = authorization()
    value = ledger(tmp_path, auth)
    call = value.plan_call(gate_id="g0", arm="native", request={})
    value.dispatch(call.call_id)
    value.receive(
        call.call_id,
        response={"content": "arrived"},
        usage={"prompt_tokens": 500, "completion_tokens": 50},
        observed_provider="test-provider",
        observed_model="test-model-snapshot",
    )

    reopened = DurableAuthorizationLedger(
        value.path,
        authorization=auth,
        authorization_sha256=authorization_sha256(auth),
    )
    reopened.recover()

    recovered = reopened.state.calls[call.call_id]
    assert recovered.state == "accounted"
    assert recovered.actual_cost_usd == Decimal("0.006")
    assert len(reopened.state.calls) == 1


def test_recovery_stops_on_a_call_that_may_have_escaped(tmp_path: Path) -> None:
    auth = authorization()
    value = ledger(tmp_path, auth)
    call = value.plan_call(gate_id="g0", arm="native", request={})
    value.dispatch(call.call_id)

    reopened = DurableAuthorizationLedger(
        value.path,
        authorization=auth,
        authorization_sha256=authorization_sha256(auth),
    )
    reopened.recover()

    assert reopened.state.run_state == "stopped"
    assert reopened.state.stop_reason == "unknown_dispatched_call"
    assert reopened.state.committed_spend_usd == auth.maximum_call_cost_usd


def test_failed_call_preserves_a_partial_provider_response(tmp_path: Path) -> None:
    auth = authorization()
    value = ledger(tmp_path, auth)
    call = value.plan_call(gate_id="g0", arm="native", request={})
    value.dispatch(call.call_id)

    value.fail(
        call.call_id,
        reason="partial stream",
        response={"partial_response": {"content": "half"}},
    )

    failed = value.state.calls[call.call_id]
    assert failed.response == {"partial_response": {"content": "half"}}
    assert failed.failure == "partial stream"
    assert value.state.run_state == "stopped"


def test_embedded_provider_failure_is_accounted_then_stops(tmp_path: Path) -> None:
    auth = authorization()
    value = ledger(tmp_path, auth)
    call = value.plan_call(gate_id="g0", arm="native", request={})
    value.dispatch(call.call_id)
    value.receive(
        call.call_id,
        response={"content": "partial"},
        usage={"prompt_tokens": 500, "completion_tokens": 50},
        observed_provider="test-provider",
        observed_model="test-model-snapshot",
    )

    actual = value.reconcile_provider_failure(
        call.call_id, reason="provider error 502: partial stream"
    )

    failed = value.state.calls[call.call_id]
    assert actual == Decimal("0.006")
    assert failed.state == "accounted"
    assert failed.failure == "provider error 502: partial stream"
    assert value.state.run_state == "stopped"
    assert value.state.stop_reason == f"provider_failure:{call.call_id}"


def test_recovery_finishes_accounting_an_observed_provider_failure(
    tmp_path: Path,
) -> None:
    auth = authorization()
    value = ledger(tmp_path, auth)
    call = value.plan_call(gate_id="g0", arm="native", request={})
    value.dispatch(call.call_id)
    value.receive(
        call.call_id,
        response={"content": "partial"},
        usage={"prompt_tokens": 500, "completion_tokens": 50},
        observed_provider="test-provider",
        observed_model="test-model-snapshot",
    )
    value._append(
        "ProviderFailureObserved",
        {"call_id": call.call_id, "failure": "provider error 502: partial"},
    )

    reopened = DurableAuthorizationLedger(
        value.path,
        authorization=auth,
        authorization_sha256=authorization_sha256(auth),
    )
    reopened.recover()

    recovered = reopened.state.calls[call.call_id]
    assert recovered.state == "accounted"
    assert recovered.actual_cost_usd == Decimal("0.006")
    assert recovered.failure == "provider error 502: partial"
    assert reopened.state.stop_reason == f"provider_failure:{call.call_id}"


def test_recovery_marks_an_interrupted_gate_without_replaying_it(
    tmp_path: Path,
) -> None:
    auth = authorization()
    value = ledger(tmp_path, auth)
    value.start_gate("g0")

    reopened = DurableAuthorizationLedger(
        value.path,
        authorization=auth,
        authorization_sha256=authorization_sha256(auth),
    )
    reopened.recover()

    assert reopened.state.run_state == "stopped"
    assert reopened.state.stop_reason == "interrupted_gate:g0"


def test_recovery_cancels_a_never_dispatched_plan(tmp_path: Path) -> None:
    auth = authorization(maximum_calls=1)
    value = ledger(tmp_path, auth)
    call = value.plan_call(gate_id="g0", arm="native", request={})

    reopened = DurableAuthorizationLedger(
        value.path,
        authorization=auth,
        authorization_sha256=authorization_sha256(auth),
    )
    reopened.recover()

    assert reopened.state.calls[call.call_id].state == "cancelled"
    assert reopened.state.committed_calls == 0
    assert reopened.state.committed_spend_usd == 0
    retry = reopened.plan_call(gate_id="g0", arm="native", request={})
    assert retry.call_id == "call-0002"


def test_completed_authorization_cannot_be_reused(tmp_path: Path) -> None:
    auth = authorization()
    value = ledger(tmp_path, auth)
    value.complete()

    reopened = DurableAuthorizationLedger(
        value.path,
        authorization=auth,
        authorization_sha256=authorization_sha256(auth),
    )
    with pytest.raises(RuntimeError, match="completed"):
        reopened.plan_call(gate_id="g0", arm="native", request={})


def test_journal_is_bound_to_the_exact_authorization(tmp_path: Path) -> None:
    auth = authorization()
    value = ledger(tmp_path, auth)

    with pytest.raises(ValueError, match="commitment does not match"):
        DurableAuthorizationLedger(
            value.path,
            authorization=auth,
            authorization_sha256="sha256:" + "f" * 64,
        )
