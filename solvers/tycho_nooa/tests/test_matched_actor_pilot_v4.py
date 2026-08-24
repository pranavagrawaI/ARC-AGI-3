from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.compatibility_authorization_v2 import PricingSchedule
from tycho_nooa.matched_actor_audit import _flat_expected_answer
from tycho_nooa.matched_actor_manifest import sha256_file
from tycho_nooa.matched_actor_pilot import load_matched_actor_pilot_sample
from tycho_nooa.matched_actor_pilot_authorization_v2 import (
    MatchedActorPilotAuthorizationV2,
)
from tycho_nooa.matched_actor_pilot_launch_v4 import (
    execute_authorized_matched_actor_pilot_v4,
    preflight_matched_actor_pilot_v4,
)
from tycho_nooa.matched_actor_pilot_manifest_v4 import (
    load_matched_actor_pilot_manifest_v4,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASE = REPOSITORY_ROOT / "analysis/matched-actor-v4-pilot"
MANIFEST = BASE / "experiment.json"


def _response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        raw_response={"provider": "provider-a", "model": "reported-fake-model"},
        content="",
        tool_calls=[
            ToolCall(id=f"{name}-id", name=name, arguments=json.dumps(arguments))
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 20, "completion_tokens": 10},
    )


def _configured_fake(*responses: LLMResponse) -> FakeLLMClient:
    client = FakeLLMClient(scripted_responses=list(responses))
    client.config["reasoning"] = {"enabled": True}
    return client


def _authorized_inputs(tmp_path: Path) -> tuple[Path, Path]:
    raw = json.loads(MANIFEST.read_text())
    raw["execution_authorized"] = True
    raw["model_route"] = {
        "model": "fake-model",
        "reported_model": "reported-fake-model",
        "provider": "provider-a",
        "reasoning_effort": "enabled",
        "temperature": None,
    }
    manifest_path = tmp_path / "experiment.json"
    manifest_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    authorization = MatchedActorPilotAuthorizationV2(
        authorization_id="matched-v4-pilot-test",
        experiment_manifest_sha256=sha256_file(manifest_path),
        allowed_provider="provider-a",
        requested_model="fake-model",
        expected_reported_model="reported-fake-model",
        allowed_reasoning_effort="enabled",
        maximum_calls=72,
        maximum_spend_usd=Decimal(3),
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


def test_disabled_v4_package_loads_and_binds_context_plane() -> None:
    manifest, packets = load_matched_actor_pilot_manifest_v4(
        MANIFEST, repository_root=REPOSITORY_ROOT
    )

    assert manifest.schema_version == 4
    assert manifest.context_plane_base_revision == "9747693"
    assert manifest.execution_authorized is False
    assert len(packets) == 6
    with pytest.raises(RuntimeError, match="execution is disabled"):
        manifest.require_execution_authorized()


async def test_v4_authorized_launcher_completes_frozen_schedule(
    tmp_path: Path,
) -> None:
    manifest_path, authorization_path = _authorized_inputs(tmp_path)
    manifest, _ = load_matched_actor_pilot_manifest_v4(
        MANIFEST, repository_root=REPOSITORY_ROOT
    )
    sample, packets = load_matched_actor_pilot_sample(
        REPOSITORY_ROOT / manifest.sample.path,
        repository_root=REPOSITORY_ROOT,
    )
    native = _configured_fake(
        *[_response("return_result", _flat_expected_answer(packet)) for packet in packets]
    )
    codeact = _configured_fake(
        *[_response("return_result", _flat_expected_answer(packet)) for packet in packets]
    )

    assessment = await execute_authorized_matched_actor_pilot_v4(
        manifest_path,
        authorization_path,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "run",
        native_llm=native,
        codeact_llm=codeact,
    )

    assert assessment.complete_pairs == 6
    assert assessment.native_correct == assessment.codeact_correct == 6
    events = [
        json.loads(line)
        for line in (tmp_path / "run/authorization-events.jsonl")
        .read_text()
        .splitlines()
    ]
    planned = [event for event in events if event["event_type"] == "CallPlanned"]
    assert [event["payload"]["arm"] for event in planned] == [
        arm for record in sample.packets for arm in record.arm_order
    ]
    assert events[-1]["event_type"] == "RunCompleted"


def test_v4_preflight_manifest_drift_makes_zero_calls(tmp_path: Path) -> None:
    manifest_path, authorization_path = _authorized_inputs(tmp_path)
    raw = json.loads(manifest_path.read_text())
    raw["experiment_id"] = "drifted"
    manifest_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    native = _configured_fake()
    codeact = _configured_fake()

    with pytest.raises(ValueError, match="does not bind"):
        preflight_matched_actor_pilot_v4(
            manifest_path,
            authorization_path,
            repository_root=REPOSITORY_ROOT,
            native_llm=native,
            codeact_llm=codeact,
        )

    assert native.call_count == codeact.call_count == 0
