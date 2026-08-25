from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

import tycho_nooa.production_shadow_launch as shadow_launch
from tycho_nooa.compatibility_authorization_v2 import (
    CompatibilityAuthorization,
    PricingSchedule,
)
from tycho_nooa.matched_actor_manifest import sha256_file
from tycho_nooa.production_shadow_launch import (
    execute_authorized_production_shadow,
    preflight_production_shadow,
    runtime_bundle_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def trusted_test_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "host")
    monkeypatch.setenv("ARC_LLM_MAX_TOKENS", "4096")
    monkeypatch.setattr(shadow_launch, "_validate_import_origin", lambda _root: None)
    monkeypatch.setattr(shadow_launch.shutil, "which", lambda _name: "/usr/bin/bwrap")


def _response(action: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response={"provider": "provider-a", "model": "reported-fake-model"},
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="take_action",
                arguments=json.dumps({"action": action}),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 20, "completion_tokens": 10},
    )


def _configured_fake(*responses: LLMResponse) -> FakeLLMClient:
    client = FakeLLMClient(scripted_responses=list(responses))
    client.config["reasoning"] = {"enabled": True}
    return client


def _observation(value: int) -> dict:
    return {
        "grid": [[value]],
        "available_actions": ["UP", "DOWN"],
        "levels_completed": 0,
        "state": "NOT_FINISHED",
    }


def _write_transcript(path: Path, turns: int = 2) -> None:
    records = []
    for sequence in range(turns):
        records.append(
            {
                "type": "action_result",
                "sequence": sequence,
                "level": 0,
                "turn_in_level": sequence,
                "decision": {"action": "UP", "row": None, "col": None},
                "before": _observation(sequence % 2),
                "after": _observation((sequence + 1) % 2),
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _authorized_inputs(
    tmp_path: Path,
    *,
    phase: str = "short",
    prerequisite: dict | None = None,
) -> tuple[Path, Path]:
    directory = tmp_path / phase
    directory.mkdir(parents=True, exist_ok=True)
    transcript = directory / "transcript.jsonl"
    controller = directory / "controller.py"
    launcher = directory / "launcher.py"
    card = directory / "card.md"
    pricing_source = directory / "pricing.json"
    _write_transcript(transcript)
    controller.write_text("controller\n")
    launcher.write_text("launcher\n")
    card.write_text("card\n")
    pricing_source.write_text("{}\n")
    runtime = tmp_path / "solvers/tycho_nooa/src/tycho_nooa/runtime.py"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text("runtime\n")
    for relative_path in shadow_launch.RUNTIME_FIXED_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"{relative_path}\n")
    relative = lambda path: str(path.relative_to(tmp_path))
    manifest = {
        "schema_version": 1,
        "experiment_id": f"production-shadow-{phase}-test",
        "execution_authorized": True,
        "phase": phase,
        "prerequisite": prerequisite,
        "sandbox_runtime": "host",
        "runtime_bundle_sha256": runtime_bundle_sha256(tmp_path),
        "transcript_path": relative(transcript),
        "transcript_sha256": sha256_file(transcript),
        "controller_path": relative(controller),
        "controller_sha256": sha256_file(controller),
        "launcher_path": relative(launcher),
        "launcher_sha256": sha256_file(launcher),
        "experiment_card_path": relative(card),
        "experiment_card_sha256": sha256_file(card),
        "pricing_source_path": relative(pricing_source),
        "pricing_source_sha256": sha256_file(pricing_source),
        "model_route": {
            "model": "fake-model",
            "reported_model": "reported-fake-model",
            "provider": "provider-a",
            "reasoning_effort": "enabled",
            "temperature": None,
        },
        "caps": {
            "turns": 2,
            "calls_total": 4,
            "input_tokens_per_call": 32768,
            "output_tokens_per_call": 4096,
            "max_cost_usd": 1.0,
        },
        "output_root": "runs",
        "advancement_rule": "short pass advances",
        "invalidity_conditions": ["route drift"],
    }
    manifest_path = directory / "experiment.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    authorization = CompatibilityAuthorization(
        authorization_id=f"production-shadow-{phase}-test",
        experiment_manifest_sha256=sha256_file(manifest_path),
        allowed_provider="provider-a",
        requested_model="fake-model",
        expected_reported_model="reported-fake-model",
        allowed_reasoning_effort="enabled",
        maximum_calls=4,
        maximum_spend_usd=Decimal(1),
        max_input_tokens_per_call=32768,
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
    authorization_path = directory / "authorization.json"
    authorization_path.write_text(authorization.model_dump_json(indent=2) + "\n")
    return manifest_path, authorization_path


@pytest.mark.asyncio
async def test_authorized_shadow_completes_without_arc_actions(
    tmp_path: Path,
) -> None:
    manifest_path, authorization_path = _authorized_inputs(tmp_path)
    llm = _configured_fake(_response("UP", "a"), _response("UP", "b"))

    assessment = await execute_authorized_production_shadow(
        manifest_path,
        authorization_path,
        repository_root=tmp_path,
        destination=tmp_path / "result",
        llm=llm,
    )

    assert assessment["passed"] is True
    assert assessment["advance"] is True
    assert assessment["turns"] == 2
    assert assessment["provider_calls"] == 2
    assert assessment["environment_actions_executed"] == 0
    assert (tmp_path / "result/advancement-certificate.json").is_file()
    events = [
        json.loads(line)
        for line in (tmp_path / "result/authorization-events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert events[-1]["event_type"] == "RunCompleted"

    prerequisite = {
        "certificate_path": str(
            (tmp_path / "result/advancement-certificate.json").relative_to(tmp_path)
        ),
        "short_manifest_path": str(manifest_path.relative_to(tmp_path)),
        "short_manifest_sha256": sha256_file(manifest_path),
        "short_authorization_path": str(authorization_path.relative_to(tmp_path)),
        "short_authorization_sha256": sha256_file(authorization_path),
    }
    long_manifest, long_authorization = _authorized_inputs(
        tmp_path,
        phase="long",
        prerequisite=prerequisite,
    )
    long_llm = _configured_fake()

    manifest, _, _ = preflight_production_shadow(
        long_manifest,
        long_authorization,
        repository_root=tmp_path,
        llm=long_llm,
    )

    assert manifest.phase == "long"
    assert long_llm.call_count == 0


def test_preflight_hash_drift_makes_zero_calls(tmp_path: Path) -> None:
    manifest_path, authorization_path = _authorized_inputs(tmp_path)
    raw = json.loads(manifest_path.read_text())
    raw["transcript_sha256"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    llm = _configured_fake()

    with pytest.raises(ValueError, match="committed file drift"):
        preflight_production_shadow(
            manifest_path,
            authorization_path,
            repository_root=tmp_path,
            llm=llm,
        )

    assert llm.call_count == 0


def test_preflight_runtime_bundle_drift_makes_zero_calls(tmp_path: Path) -> None:
    manifest_path, authorization_path = _authorized_inputs(tmp_path)
    (tmp_path / "solvers/tycho_nooa/src/tycho_nooa/runtime.py").write_text(
        "drifted runtime\n"
    )
    llm = _configured_fake()

    with pytest.raises(ValueError, match="runtime bundle drift"):
        preflight_production_shadow(
            manifest_path,
            authorization_path,
            repository_root=tmp_path,
            llm=llm,
        )

    assert llm.call_count == 0


def test_preflight_sandbox_mismatch_makes_zero_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, authorization_path = _authorized_inputs(tmp_path)
    monkeypatch.setenv("TYCHO_SANDBOX_RUNTIME", "auto")
    llm = _configured_fake()

    with pytest.raises(ValueError, match="sandbox runtime mismatch"):
        preflight_production_shadow(
            manifest_path,
            authorization_path,
            repository_root=tmp_path,
            llm=llm,
        )

    assert llm.call_count == 0


def test_long_preflight_requires_completed_identity_bound_short_run(
    tmp_path: Path,
) -> None:
    short_manifest, short_authorization = _authorized_inputs(tmp_path)
    run = tmp_path / "runs/short"
    run.mkdir(parents=True)
    assessment = run / "assessment.json"
    ledger = run / "authorization-events.jsonl"
    assessment.write_text('{"passed": false, "advance": false}\n')
    ledger.write_text('{"event_type":"RunCompleted"}\n')
    certificate = run / "advancement-certificate.json"
    certificate.write_text(
        json.dumps(
            {
                "schema": "tycho-nooa-production-shadow-advancement-v1",
                "authorization_id": "production-shadow-short-test",
                "short_manifest_sha256": sha256_file(short_manifest),
                "short_authorization_sha256": sha256_file(short_authorization),
                "assessment_path": str(assessment.relative_to(tmp_path)),
                "assessment_sha256": sha256_file(assessment),
                "ledger_path": str(ledger.relative_to(tmp_path)),
                "ledger_sha256": sha256_file(ledger),
                "advance": True,
            }
        )
        + "\n"
    )
    prerequisite = {
        "certificate_path": str(certificate.relative_to(tmp_path)),
        "short_manifest_path": str(short_manifest.relative_to(tmp_path)),
        "short_manifest_sha256": sha256_file(short_manifest),
        "short_authorization_path": str(short_authorization.relative_to(tmp_path)),
        "short_authorization_sha256": sha256_file(short_authorization),
    }
    manifest_path, authorization_path = _authorized_inputs(
        tmp_path,
        phase="long",
        prerequisite=prerequisite,
    )
    llm = _configured_fake()

    with pytest.raises(RuntimeError, match="did not authorize advancement"):
        preflight_production_shadow(
            manifest_path,
            authorization_path,
            repository_root=tmp_path,
            llm=llm,
        )

    assert llm.call_count == 0
