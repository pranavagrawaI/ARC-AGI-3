from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.shadow_eval import (
    SPEC_SCHEMA,
    ShadowValidationError,
    _jsonable,
    canonical_commitment,
    execute_shadow,
    inspect_checkpoint,
    main,
    prepare_shadow_packet,
    validate_shadow_packet,
)

BUILDER_REPORT = "BUILDER REPORT: choose UP\n"
TEST_MAX_COST_USD = 1.0
TEST_COST_RESERVE_USD = 0.25


class ConditionalClient(FakeLLMClient):
    def __init__(self, behavior: str = "sensitive") -> None:
        super().__init__()
        self.behavior = behavior
        self.calls: list[dict[str, Any]] = []

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.call_count += 1
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "output_model": output_model,
                "parameters": kwargs,
            }
        )
        text = json.dumps(messages)
        if self.behavior == "unique":
            return prose_response(f"sample-{self.call_count}")
        if self.behavior == "constant":
            action = "UP"
        else:
            action = "UP" if BUILDER_REPORT.strip() in text else "DOWN"
        return action_response(action)


class CostedClient(ConditionalClient):
    def __init__(self, costs: list[float | None]) -> None:
        super().__init__("constant")
        self.costs = costs

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        response = await super().acall(
            messages,
            tools=tools,
            output_model=output_model,
            **kwargs,
        )
        cost = self.costs[self.call_count - 1]
        assert response.usage is not None
        if cost is None:
            response.usage.pop("cost", None)
        else:
            response.usage["cost"] = cost
        return response


class RoutedClient(ConditionalClient):
    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        response = await super().acall(
            messages,
            tools=tools,
            output_model=output_model,
            **kwargs,
        )
        response.raw_response = {
            "provider": "NebulaCompute",
            "model": "google/gemma-4-31b-it",
            "system_fingerprint": "route-fingerprint",
            "_hidden_params": {
                "custom_llm_provider": "openai",
                "model_id": "backend-route-17",
            },
        }
        return response


class DriftingRouteClient(RoutedClient):
    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        response = await super().acall(
            messages,
            tools=tools,
            output_model=output_model,
            **kwargs,
        )
        response.raw_response["provider"] = f"route-{self.call_count % 2}"
        return response


def action_response(action: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=f"dynamic-{action}",
                name="take_actions",
                arguments=json.dumps(
                    {
                        "actions": [{"action": action}],
                        "rationale": "test one-action commitment",
                    }
                ),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        reasoning=None,
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cost": 0.0,
        },
    )


def prose_response(content: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
        reasoning=None,
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cost": 0.0,
        },
    )


def write_checkpoint(root: Path, *, output_model: object = None) -> dict[str, Any]:
    content = b"MODEL = 'candidate'\n"
    digest = hashlib.sha256(content).hexdigest()
    blob = root / "blobs" / "sha256" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content)
    workspace = {
        "schema_version": 2,
        "directories": [],
        "files": [
            {
                "path": "world_model.py",
                "sha256": digest,
                "size": len(content),
                "mode": 0o644,
            }
        ],
    }
    body = {
        "model": "fake-model",
        "messages": [
            {"role": "system", "content": "choose one action"},
            {
                "role": "user",
                "content": f"observation\n{BUILDER_REPORT}end",
            },
        ],
        "tools": [
            {
                "name": "take_actions",
                "description": "commit one or more actions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"action": {"type": "string"}},
                                "required": ["action"],
                            },
                            "minItems": 1,
                            "maxItems": 4,
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["actions", "rationale"],
                    "additionalProperties": False,
                },
            }
        ],
        "output_model": output_model,
        "parameters": {"max_tokens": 100},
    }
    request = {
        "schema_version": 2,
        "call_index": 1,
        "source": "actor",
        "request_boundary": "nooa.llm_call",
        "request_payload_commitment": canonical_commitment(body),
        "workspace_commitment": canonical_commitment(workspace),
        "decision_point": {
            "sequence": 7,
            "level": 0,
            "turn_in_level": 3,
            "phase": "decision",
        },
        "replay_context": {
            "client_type": "test.expected.Client",
            "client_config": {},
            "omitted_credential_keys": [],
            "cache_control_injection_points": [],
        },
        "request_body": body,
    }
    response = {
        "schema_version": 2,
        "status": "complete",
        "content": "",
        "reasoning": None,
        "finish_reason": "tool_calls",
        "assistant_message": {"role": "assistant", "content": None},
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        "tool_calls": [
            {
                "id": "recorded-id",
                "name": "take_actions",
                "arguments": json.dumps(
                    {
                        "actions": [{"action": "UP"}],
                        "rationale": "test one-action commitment",
                    }
                ),
            }
        ],
    }
    response["response_payload_commitment"] = canonical_commitment(response)
    call = root / "calls" / "000001"
    call.mkdir(parents=True)
    for name, value in (
        ("request.json", request),
        ("response.json", response),
        ("workspace.json", workspace),
    ):
        (call / name).write_text(json.dumps(value))
    return request


def perturbation(request: dict[str, Any], *, effect: str = "take_actions") -> dict:
    target = request["request_body"]["messages"][1]["content"]
    return {
        "schema": SPEC_SCHEMA,
        "id": "remove-builder-report",
        "question": "Does the actor's immediate action depend on builder advice?",
        "hypothesis": "Removing the advice changes take_actions.",
        "expected_effect": effect,
        "invalid_if": "Any request leaf besides the target message changes.",
        "source_request_payload_commitment": request["request_payload_commitment"],
        "operation": {
            "type": "replace_exact_text",
            "pointer": "/messages/1/content",
            "target_value_commitment": canonical_commitment(target),
            "old": BUILDER_REPORT,
            "new": "",
        },
    }


def prepared_packet(tmp_path: Path, *, effect: str = "take_actions") -> Path:
    root = tmp_path / "checkpoints"
    request = write_checkpoint(root)
    packet = tmp_path / "packet"
    prepare_shadow_packet(
        root,
        call_index=1,
        perturbation=perturbation(request, effect=effect),
        destination=packet,
    )
    return packet


def test_prepare_makes_a_self_contained_one_leaf_packet(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    request = write_checkpoint(root)
    packet = tmp_path / "packet"

    manifest = prepare_shadow_packet(
        root,
        call_index=1,
        perturbation=perturbation(request),
        destination=packet,
    )
    prepared = validate_shadow_packet(packet)

    assert manifest["changed_paths"] == ["/messages/1/content"]
    assert BUILDER_REPORT in prepared.control["messages"][1]["content"]
    assert BUILDER_REPORT not in prepared.treatment["messages"][1]["content"]
    assert list((packet / "blobs" / "sha256").iterdir())


def test_inspect_exposes_exact_targets_needed_for_precommitment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    request = write_checkpoint(root)

    result = inspect_checkpoint(root, call_index=1)

    target = next(
        row
        for row in result["message_text_targets"]
        if row["pointer"] == "/messages/1/content"
    )
    assert (
        result["source_request_payload_commitment"]
        == request["request_payload_commitment"]
    )
    assert target["target_value_commitment"] == canonical_commitment(
        request["request_body"]["messages"][1]["content"]
    )


def test_prepare_rejects_a_spec_bound_to_another_request(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    request = write_checkpoint(root)
    spec = perturbation(request)
    spec["source_request_payload_commitment"] = "sha256:wrong"

    with pytest.raises(ShadowValidationError, match="different source request"):
        prepare_shadow_packet(
            root,
            call_index=1,
            perturbation=spec,
            destination=tmp_path / "packet",
        )


def test_schema_two_checkpoint_cannot_downgrade_its_response(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    write_checkpoint(root)
    call = root / "calls" / "000001"
    response_path = call / "response.json"
    response = json.loads(response_path.read_text())
    inspect_checkpoint(root, call_index=1)

    response.pop("schema_version")
    response.pop("response_payload_commitment")
    response_path.write_text(json.dumps(response))
    with pytest.raises(ShadowValidationError, match="response commitment"):
        inspect_checkpoint(root, call_index=1)

    write_checkpoint(tmp_path / "downgraded")
    downgraded = tmp_path / "downgraded" / "calls" / "000001" / "request.json"
    request = json.loads(downgraded.read_text())
    request.pop("schema_version")
    downgraded.write_text(json.dumps(request))
    with pytest.raises(ShadowValidationError, match="not promotion-grade"):
        inspect_checkpoint(tmp_path / "downgraded", call_index=1)


def test_packet_tampering_is_detected_before_execution(tmp_path: Path) -> None:
    packet = prepared_packet(tmp_path)
    treatment = json.loads((packet / "treatment-request.json").read_text())
    treatment["messages"][0]["content"] = "silent second change"
    (packet / "treatment-request.json").write_text(json.dumps(treatment))

    with pytest.raises(ShadowValidationError, match="stale packet commitment"):
        validate_shadow_packet(packet)


def test_source_response_tampering_is_detected_before_execution(
    tmp_path: Path,
) -> None:
    packet = prepared_packet(tmp_path)
    response_path = packet / "source" / "response.json"
    response = json.loads(response_path.read_text())
    response["tool_calls"][0]["arguments"] = json.dumps({"action": "DOWN"})
    response_path.write_text(json.dumps(response))

    with pytest.raises(
        ShadowValidationError,
        match="source_response_commitment",
    ):
        validate_shadow_packet(packet)


@pytest.mark.asyncio
async def test_stable_paired_difference_supports_immediate_dependence(
    tmp_path: Path,
) -> None:
    client = ConditionalClient()
    packet = prepared_packet(tmp_path)

    report = await execute_shadow(
        packet,
        llm=client,
        repeats=2,
        max_calls=4,
        schedule_seed=7,
        max_cost_usd=TEST_MAX_COST_USD,
        cost_reserve_per_call_usd=TEST_COST_RESERVE_USD,
        enforce_replay_context=False,
    )

    assert report["executed_calls"] == 4
    assert report["assessment"]["judgment"] == "supported"
    assert report["promotion_allowed"] is True
    assert report["assessment"]["paired_change_rates"]["take_actions_changed"] == 1
    assert report["assessment"]["control_recorded_response_match_rate"] == 1
    assert all(call["output_model"] is None for call in client.calls)


@pytest.mark.asyncio
async def test_stable_same_response_contradicts_the_expected_effect(
    tmp_path: Path,
) -> None:
    report = await execute_shadow(
        prepared_packet(tmp_path),
        llm=ConditionalClient("constant"),
        repeats=2,
        max_calls=4,
        schedule_seed=0,
        max_cost_usd=TEST_MAX_COST_USD,
        cost_reserve_per_call_usd=TEST_COST_RESERVE_USD,
        enforce_replay_context=False,
    )

    assert report["assessment"]["judgment"] == "contradicted"
    assert report["promotion_allowed"] is False


@pytest.mark.asyncio
async def test_within_arm_variability_is_inconclusive(tmp_path: Path) -> None:
    report = await execute_shadow(
        prepared_packet(tmp_path, effect="content"),
        llm=ConditionalClient("unique"),
        repeats=3,
        max_calls=6,
        schedule_seed=0,
        max_cost_usd=TEST_MAX_COST_USD,
        cost_reserve_per_call_usd=TEST_COST_RESERVE_USD,
        enforce_replay_context=False,
    )

    assert report["assessment"]["judgment"] == "inconclusive"
    assert "within-arm variability" in report["assessment"]["reason"]


@pytest.mark.asyncio
async def test_call_cap_is_checked_before_the_first_model_call(tmp_path: Path) -> None:
    client = ConditionalClient()

    with pytest.raises(ShadowValidationError, match="exceed hard cap"):
        await execute_shadow(
            prepared_packet(tmp_path),
            llm=client,
            repeats=3,
            max_calls=5,
            schedule_seed=0,
            enforce_replay_context=False,
        )

    assert client.call_count == 0


@pytest.mark.asyncio
async def test_direct_execution_refuses_an_unmetered_schedule(tmp_path: Path) -> None:
    client = ConditionalClient()

    with pytest.raises(ShadowValidationError, match="requires.*monetary limit"):
        await execute_shadow(
            prepared_packet(tmp_path),
            llm=client,
            repeats=2,
            max_calls=4,
            schedule_seed=0,
            enforce_replay_context=False,
        )

    assert client.call_count == 0


@pytest.mark.asyncio
async def test_cost_reservation_stops_before_a_call_that_cannot_fit(
    tmp_path: Path,
) -> None:
    client = CostedClient([0.015, 0.001])

    report = await execute_shadow(
        prepared_packet(tmp_path),
        llm=client,
        repeats=2,
        max_calls=4,
        schedule_seed=0,
        max_cost_usd=0.02,
        cost_reserve_per_call_usd=0.01,
        enforce_replay_context=False,
    )

    assert client.call_count == 1
    assert report["status"] == "stopped_cost_limit"
    assert report["executed_calls"] == 1
    assert report["assessment"]["judgment"] == "inconclusive"
    assert report["cost_budget"] == {
        "max_cost_usd": 0.02,
        "reserve_per_call_usd": 0.01,
        "observed_cost_usd": 0.015,
        "remaining_cost_usd": 0.005,
        "calls_withheld": 3,
        "stop_reason": "insufficient_reserved_cost",
    }


@pytest.mark.asyncio
async def test_observed_cost_overshoot_withholds_every_later_call(
    tmp_path: Path,
) -> None:
    client = CostedClient([0.03])

    report = await execute_shadow(
        prepared_packet(tmp_path),
        llm=client,
        repeats=2,
        max_calls=4,
        schedule_seed=0,
        max_cost_usd=0.02,
        cost_reserve_per_call_usd=0.01,
        enforce_replay_context=False,
    )

    assert client.call_count == 1
    assert report["status"] == "stopped_cost_limit"
    assert report["cost_budget"]["observed_cost_usd"] == 0.03
    assert report["cost_budget"]["calls_withheld"] == 3
    assert report["cost_budget"]["stop_reason"] == "observed_cost_exceeded"
    assert report["assessment"]["judgment"] == "inconclusive"


@pytest.mark.asyncio
async def test_missing_response_cost_withholds_every_later_call(tmp_path: Path) -> None:
    client = CostedClient([None])

    report = await execute_shadow(
        prepared_packet(tmp_path),
        llm=client,
        repeats=2,
        max_calls=4,
        schedule_seed=0,
        max_cost_usd=0.02,
        cost_reserve_per_call_usd=0.01,
        enforce_replay_context=False,
    )

    assert client.call_count == 1
    assert report["status"] == "stopped_cost_limit"
    assert report["cost_budget"]["observed_cost_usd"] == 0.0
    assert report["cost_budget"]["calls_withheld"] == 3
    assert report["cost_budget"]["stop_reason"] == "missing_observed_cost"
    assert report["assessment"]["judgment"] == "inconclusive"


@pytest.mark.asyncio
async def test_provider_route_fields_are_preserved_when_the_response_exposes_them(
    tmp_path: Path,
) -> None:
    report = await execute_shadow(
        prepared_packet(tmp_path),
        llm=RoutedClient("constant"),
        repeats=1,
        max_calls=2,
        schedule_seed=0,
        max_cost_usd=TEST_MAX_COST_USD,
        cost_reserve_per_call_usd=TEST_COST_RESERVE_USD,
        enforce_replay_context=False,
    )

    assert report["observations"][0]["response"]["provider_route"] == {
        "reported_provider": "NebulaCompute",
        "reported_model": "google/gemma-4-31b-it",
        "system_fingerprint": "route-fingerprint",
        "litellm_provider": "openai",
        "model_id": "backend-route-17",
    }


@pytest.mark.asyncio
async def test_observed_provider_route_drift_makes_the_shadow_inconclusive(
    tmp_path: Path,
) -> None:
    report = await execute_shadow(
        prepared_packet(tmp_path),
        llm=DriftingRouteClient("constant"),
        repeats=2,
        max_calls=4,
        schedule_seed=0,
        max_cost_usd=TEST_MAX_COST_USD,
        cost_reserve_per_call_usd=TEST_COST_RESERVE_USD,
        enforce_replay_context=False,
    )

    assert report["assessment"]["judgment"] == "inconclusive"
    assert report["assessment"]["provider_route"] == {
        "verifiable": True,
        "stable": False,
        "missing_calls": 0,
        "identity_counts": [
            {
                "signature": (
                    '{"litellm_provider":"openai",'
                    '"model_id":"backend-route-17",'
                    '"reported_model":"google/gemma-4-31b-it",'
                    '"reported_provider":"route-0",'
                    '"system_fingerprint":"route-fingerprint"}'
                ),
                "count": 2,
            },
            {
                "signature": (
                    '{"litellm_provider":"openai",'
                    '"model_id":"backend-route-17",'
                    '"reported_model":"google/gemma-4-31b-it",'
                    '"reported_provider":"route-1",'
                    '"system_fingerprint":"route-fingerprint"}'
                ),
                "count": 2,
            },
        ],
    }


@pytest.mark.asyncio
async def test_client_drift_is_checked_before_the_first_model_call(
    tmp_path: Path,
) -> None:
    client = ConditionalClient()

    with pytest.raises(ShadowValidationError, match="configuration drift"):
        await execute_shadow(
            prepared_packet(tmp_path),
            llm=client,
            repeats=2,
            max_calls=4,
            schedule_seed=0,
            max_cost_usd=TEST_MAX_COST_USD,
            cost_reserve_per_call_usd=TEST_COST_RESERVE_USD,
        )

    assert client.call_count == 0


def test_structured_output_checkpoint_is_explicitly_unsupported(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    request = write_checkpoint(root, output_model={"qualified_name": "Example"})

    with pytest.raises(ShadowValidationError, match="structured output"):
        prepare_shadow_packet(
            root,
            call_index=1,
            perturbation=perturbation(request),
            destination=tmp_path / "packet",
        )


def test_cli_run_without_execute_is_a_zero_call_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    packet = prepared_packet(tmp_path)
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shadow_eval",
            "run",
            str(packet),
            str(output),
            "--repeats",
            "2",
            "--max-calls",
            "4",
        ],
    )

    assert main() == 2
    assert '"status": "planned"' in capsys.readouterr().out
    assert not output.exists()


def test_cli_refuses_paid_execution_without_an_explicit_monetary_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = prepared_packet(tmp_path)
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shadow_eval",
            "run",
            str(packet),
            str(output),
            "--repeats",
            "2",
            "--max-calls",
            "4",
            "--execute",
        ],
    )

    with pytest.raises(ShadowValidationError, match="explicit monetary limit"):
        main()

    assert not output.exists()


def test_jsonable_handles_every_supported_shape_and_rejects_opaque_values() -> None:
    class ModelValue:
        def model_dump(self, *, mode: str) -> dict[str, str]:
            assert mode == "json"
            return {"value": "model"}

    assert _jsonable(None) is None
    assert _jsonable({1: [True, ModelValue()]}) == {
        "1": [True, {"value": "model"}]
    }
    with pytest.raises(TypeError, match="not JSON-safe"):
        _jsonable(object())


def test_cli_paid_path_persists_the_precommitted_report_without_provider_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = prepared_packet(tmp_path)
    output = tmp_path / "result"
    client = ConditionalClient("constant")
    calls: list[Path] = []

    async def fake_execute(packet_path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append(packet_path)
        return {"schema": "test", "status": "complete"}

    monkeypatch.setattr("tycho_nooa.shadow_eval.build_llm", lambda: (client, "live"))
    monkeypatch.setattr("tycho_nooa.shadow_eval.execute_shadow", fake_execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shadow_eval",
            "run",
            str(packet),
            str(output),
            "--repeats",
            "2",
            "--max-calls",
            "4",
            "--max-cost-usd",
            "1.0",
            "--cost-reserve-per-call-usd",
            "0.25",
            "--execute",
        ],
    )

    assert main() == 0
    assert calls == [packet]
    assert json.loads((output / "report.json").read_text()) == {
        "schema": "test",
        "status": "complete",
    }
    assert client.call_count == 0
