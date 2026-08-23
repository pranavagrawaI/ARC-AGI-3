from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.compatibility_authorization import (
    CompatibilityAuthorization,
    PricingSchedule,
)
from tycho_nooa.compatibility_launch import (
    execute_authorized_compatibility_ladder,
    preflight_authorized_compatibility_ladder,
)
from tycho_nooa.compatibility_manifest import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASE_MANIFEST = REPOSITORY_ROOT / "analysis/parity-compat-v2/experiment.json"


def response(
    conclusion: str,
    *,
    provider: str = "provider-a",
    model: str = "reported-fake-model",
) -> LLMResponse:
    result = {
        "task_kind": "retrieve",
        "evidence_refs": [],
        "conclusion": conclusion,
    }
    return LLMResponse(
        raw_response={"provider": provider, "model": model},
        content="",
        tool_calls=[
            ToolCall(
                id="return_result",
                name="return_result",
                arguments=json.dumps({"result": result}),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 20, "completion_tokens": 10},
    )


def provider_failure_response() -> LLMResponse:
    value = response("READY")
    value.raw_response = SimpleNamespace(
        provider="provider-a",
        model="reported-fake-model",
        choices=[
            SimpleNamespace(
                provider_specific_fields={
                    "error": {
                        "code": 502,
                        "message": "generation ended before tool call completed",
                    }
                },
            )
        ],
    )
    return value


def configured_fake(*responses: LLMResponse) -> FakeLLMClient:
    client = FakeLLMClient(scripted_responses=list(responses))
    client.config["reasoning"] = {"enabled": True}
    return client


def authorized_inputs(
    tmp_path: Path, *, maximum_spend: float = 3.0
) -> tuple[Path, Path]:
    raw = json.loads(BASE_MANIFEST.read_text())
    raw["execution_authorized"] = True
    raw["model_route"] = {
        "model": "fake-model",
        "reported_model": "reported-fake-model",
        "provider": "provider-a",
        "reasoning_effort": "enabled",
        "temperature": None,
    }
    raw["caps"].update(
        {
            "input_tokens_per_call": 1000,
            "output_tokens_per_call": 100,
            "max_cost_usd": maximum_spend,
        }
    )
    manifest_path = tmp_path / "experiment.json"
    manifest_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    authorization = CompatibilityAuthorization(
        authorization_id="parity-compat-test",
        experiment_manifest_sha256=sha256_file(manifest_path),
        allowed_provider="provider-a",
        requested_model="fake-model",
        expected_reported_model="reported-fake-model",
        allowed_reasoning_effort="enabled",
        maximum_calls=30,
        maximum_spend_usd=maximum_spend,
        max_input_tokens_per_call=1000,
        max_output_tokens_per_call=100,
        expires_at=datetime.now(UTC) + timedelta(days=1),
        pricing=PricingSchedule(
            provider="provider-a",
            model="fake-model",
            input_usd_per_million=10,
            cached_input_usd_per_million=2,
            output_usd_per_million=20,
            source="test schedule v1",
            effective_at=datetime(2026, 8, 23, tzinfo=UTC),
        ),
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        authorization.model_dump_json(indent=2) + "\n"
    )
    return manifest_path, authorization_path


async def test_one_arm_gate_failure_still_completes_the_matched_pair(
    tmp_path: Path,
) -> None:
    manifest, authorization = authorized_inputs(tmp_path)
    native = configured_fake(response("WRONG"))
    codeact = configured_fake(response("READY"))

    report = await execute_authorized_compatibility_ladder(
        manifest,
        authorization,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "run",
        native_llm=native,
        codeact_llm=codeact,
    )

    assert report.disposition == "rejected_model_protocol"
    assert native.call_count == 1
    assert codeact.call_count == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "run/authorization-events.jsonl").read_text().splitlines()
    ]
    assert [event["event_type"] for event in events].count("CallAccounted") == 2
    assert events[-1]["event_type"] == "RunCompleted"


def test_preflight_validates_without_creating_a_run_or_calling_provider(
    tmp_path: Path,
) -> None:
    manifest, authorization = authorized_inputs(tmp_path)
    native = configured_fake(response("READY"))
    codeact = configured_fake(response("READY"))

    loaded, ladder, _, _ = preflight_authorized_compatibility_ladder(
        manifest,
        authorization,
        repository_root=REPOSITORY_ROOT,
        native_llm=native,
        codeact_llm=codeact,
    )

    assert loaded.execution_authorized is True
    assert ladder.termination.task.gate_id == "g0-termination"
    assert native.call_count == 0
    assert codeact.call_count == 0
    assert not (tmp_path / "run").exists()


async def test_manifest_change_after_authorization_makes_zero_calls(
    tmp_path: Path,
) -> None:
    manifest, authorization = authorized_inputs(tmp_path)
    raw = json.loads(manifest.read_text())
    raw["experiment_id"] = "changed-after-authorization"
    manifest.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    native = configured_fake(response("READY"))
    codeact = configured_fake(response("READY"))

    with pytest.raises(ValueError, match="does not bind"):
        await execute_authorized_compatibility_ladder(
            manifest,
            authorization,
            repository_root=REPOSITORY_ROOT,
            destination=tmp_path / "run",
            native_llm=native,
            codeact_llm=codeact,
        )

    assert native.call_count == 0
    assert codeact.call_count == 0
    assert not (tmp_path / "run").exists()


async def test_wrong_requested_model_makes_zero_calls(tmp_path: Path) -> None:
    manifest, authorization = authorized_inputs(tmp_path)
    native = configured_fake(response("READY"))
    native.model = "wrong-model"
    codeact = configured_fake(response("READY"))

    with pytest.raises(ValueError, match="configured model"):
        await execute_authorized_compatibility_ladder(
            manifest,
            authorization,
            repository_root=REPOSITORY_ROOT,
            destination=tmp_path / "run",
            native_llm=native,
            codeact_llm=codeact,
        )

    assert native.call_count == 0
    assert codeact.call_count == 0
    assert not (tmp_path / "run").exists()


async def test_wrong_reasoning_route_makes_zero_calls(tmp_path: Path) -> None:
    manifest, authorization = authorized_inputs(tmp_path)
    native = configured_fake(response("READY"))
    native.config["reasoning"] = {"enabled": False}
    codeact = configured_fake(response("READY"))

    with pytest.raises(ValueError, match="configured reasoning"):
        await execute_authorized_compatibility_ladder(
            manifest,
            authorization,
            repository_root=REPOSITORY_ROOT,
            destination=tmp_path / "run",
            native_llm=native,
            codeact_llm=codeact,
        )

    assert native.call_count == 0
    assert codeact.call_count == 0
    assert not (tmp_path / "run").exists()


async def test_worst_case_cost_refusal_occurs_before_dispatch(tmp_path: Path) -> None:
    manifest, authorization = authorized_inputs(tmp_path, maximum_spend=0.011)
    native = configured_fake(response("READY"))
    codeact = configured_fake(response("READY"))

    with pytest.raises(RuntimeError, match="cost reservation"):
        await execute_authorized_compatibility_ladder(
            manifest,
            authorization,
            repository_root=REPOSITORY_ROOT,
            destination=tmp_path / "run",
            native_llm=native,
            codeact_llm=codeact,
        )

    assert native.call_count == 0
    assert codeact.call_count == 0
    events = (tmp_path / "run/authorization-events.jsonl").read_text()
    assert '"event_type":"CallDispatched"' not in events
    assert '"reason":"cost_reservation"' in events


async def test_response_route_drift_is_preserved_and_accounted(
    tmp_path: Path,
) -> None:
    manifest, authorization = authorized_inputs(tmp_path)
    native = configured_fake(response("READY", provider="provider-b"))
    codeact = configured_fake(response("READY"))

    with pytest.raises(RuntimeError, match="route_violation"):
        await execute_authorized_compatibility_ladder(
            manifest,
            authorization,
            repository_root=REPOSITORY_ROOT,
            destination=tmp_path / "run",
            native_llm=native,
            codeact_llm=codeact,
        )

    assert native.call_count == 1
    assert codeact.call_count == 0
    events = (tmp_path / "run/authorization-events.jsonl").read_text()
    assert '"event_type":"CallReceived"' in events
    assert '"event_type":"RouteViolated"' in events
    assert '"event_type":"CallAccounted"' in events
    assert '"provider-b"' in events


async def test_embedded_provider_failure_is_preserved_accounted_and_stops(
    tmp_path: Path,
) -> None:
    manifest, authorization = authorized_inputs(tmp_path)
    native = configured_fake(provider_failure_response())
    codeact = configured_fake(response("READY"))

    with pytest.raises(RuntimeError, match="provider_failure:call-0001"):
        await execute_authorized_compatibility_ladder(
            manifest,
            authorization,
            repository_root=REPOSITORY_ROOT,
            destination=tmp_path / "run",
            native_llm=native,
            codeact_llm=codeact,
        )

    assert native.call_count == 1
    assert codeact.call_count == 0
    events = (tmp_path / "run/authorization-events.jsonl").read_text()
    assert '"event_type":"CallReceived"' in events
    assert '"event_type":"ProviderFailureObserved"' in events
    assert '"event_type":"CallAccounted"' in events
    assert '"event_type":"RunStopped"' in events
    assert "generation ended before tool call completed" in events
    assert not (tmp_path / "run/ladder/ladder-report.json").exists()


async def test_completed_authorization_is_not_silently_replayed(
    tmp_path: Path,
) -> None:
    manifest, authorization = authorized_inputs(tmp_path)
    destination = tmp_path / "run"
    await execute_authorized_compatibility_ladder(
        manifest,
        authorization,
        repository_root=REPOSITORY_ROOT,
        destination=destination,
        native_llm=configured_fake(response("WRONG")),
        codeact_llm=configured_fake(response("READY")),
    )
    native = configured_fake(response("READY"))
    codeact = configured_fake(response("READY"))

    with pytest.raises(RuntimeError, match="already exists with state completed"):
        await execute_authorized_compatibility_ladder(
            manifest,
            authorization,
            repository_root=REPOSITORY_ROOT,
            destination=destination,
            native_llm=native,
            codeact_llm=codeact,
        )

    assert native.call_count == 0
    assert codeact.call_count == 0
