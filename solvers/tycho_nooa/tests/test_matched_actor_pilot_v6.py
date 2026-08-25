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
from tycho_nooa.matched_actor_pilot_authorization_v3 import (
    MatchedActorPilotAuthorizationV3,
)
from tycho_nooa.matched_actor_pilot_launch_v6 import (
    execute_authorized_matched_actor_pilot_v6,
)
from tycho_nooa.matched_actor_pilot_manifest_v6 import (
    load_matched_actor_pilot_manifest_v6,
)
from tycho_nooa.matched_actor_pilot_v2 import load_matched_actor_pilot_sample_v2

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASE = REPOSITORY_ROOT / "analysis/matched-actor-v2-five-packet-pilot"
MANIFEST = BASE / "experiment.json"


def _response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        raw_response={"provider": "provider-a", "model": "reported-fake-model"},
        content="",
        tool_calls=[ToolCall(id=f"{name}-id", name=name, arguments=json.dumps(arguments))],
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
    authorization = MatchedActorPilotAuthorizationV3(
        authorization_id="matched-v2-pilot-test",
        experiment_manifest_sha256=sha256_file(manifest_path),
        allowed_provider="provider-a",
        requested_model="fake-model",
        expected_reported_model="reported-fake-model",
        allowed_reasoning_effort="enabled",
        maximum_calls=60,
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
            effective_at=datetime(2026, 8, 25, tzinfo=UTC),
        ),
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(authorization.model_dump_json(indent=2) + "\n")
    return manifest_path, authorization_path


def test_disabled_v2_package_loads_without_the_remote_stress_packet() -> None:
    manifest, packets = load_matched_actor_pilot_manifest_v6(
        MANIFEST, repository_root=REPOSITORY_ROOT
    )

    assert manifest.execution_authorized is False
    assert manifest.caps.sessions == 10
    assert manifest.caps.calls_total == 60
    assert len(packets) == 5
    assert "remote-failure-composition" not in {
        packet.task.packet_id for packet in packets
    }
    with pytest.raises(RuntimeError, match="execution is disabled"):
        manifest.require_execution_authorized()


@pytest.mark.asyncio
async def test_authorized_launcher_completes_the_five_pair_schedule(
    tmp_path: Path,
) -> None:
    manifest_path, authorization_path = _authorized_inputs(tmp_path)
    sample, packets = load_matched_actor_pilot_sample_v2(
        BASE / "sample.json", repository_root=REPOSITORY_ROOT
    )
    native = _configured_fake(
        *[_response("return_result", _flat_expected_answer(packet)) for packet in packets]
    )
    codeact = _configured_fake(
        *[_response("return_result", _flat_expected_answer(packet)) for packet in packets]
    )

    assessment = await execute_authorized_matched_actor_pilot_v6(
        manifest_path,
        authorization_path,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "run",
        native_llm=native,
        codeact_llm=codeact,
    )

    assert assessment.complete_pairs == 5
    assert assessment.native_correct == assessment.codeact_correct == 5
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
