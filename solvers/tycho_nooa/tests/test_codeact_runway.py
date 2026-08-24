from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.codeact_runway import (
    CodeActRunwayManifest,
    CodeActRunwayMilestones,
    _interpret,
    execute_authorized_codeact_runway,
    execute_codeact_runway_arm,
    extract_runway_milestones,
    load_codeact_runway_manifest,
)
from tycho_nooa.compatibility_authorization import (
    CompatibilityAuthorization,
    PricingSchedule,
)
from tycho_nooa.compatibility_experiment import CompatibilityArmObservation
from tycho_nooa.compatibility_manifest import sha256_file, sha256_text
from tycho_nooa.parity_compatibility import load_compatibility_gate

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GATE_PATH = REPOSITORY_ROOT / "analysis/parity-compat-v12/gates/g3-composition.json"
NATIVE_CONTROL = REPOSITORY_ROOT / (
    "analysis/parity-compat-v12/runs/"
    "20260824T112416Z-gemma-4-31b-openinference/ladder/"
    "g3-composition/native/result.json"
)
CODEACT_TRACE = REPOSITORY_ROOT / (
    "analysis/parity-compat-v12/runs/"
    "20260824T112416Z-gemma-4-31b-openinference/ladder/"
    "g3-composition/codeact/provenance.jsonl"
)
V13_MANIFEST = REPOSITORY_ROOT / "analysis/parity-compat-v13/experiment.json"


def response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        raw_response={"provider": "provider-a", "model": "fake-model"},
        content="",
        tool_calls=[ToolCall(id=name, name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 2, "completion_tokens": 1},
    )


def commitment(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def diagnostic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    protocol = tmp_path / "PROTOCOL.md"
    card = tmp_path / "EXPERIMENT-CARD.md"
    protocol.write_text("exact V12 G3, CodeAct only, six calls\n")
    card.write_text("Question: does six-call runway rescue exact V12 G3?\n")
    runner = REPOSITORY_ROOT / "solvers/tycho_nooa/src/tycho_nooa/codeact_runway.py"
    executor = REPOSITORY_ROOT / (
        "solvers/tycho_nooa/src/tycho_nooa/compatibility_experiment.py"
    )
    authorization_schema = REPOSITORY_ROOT / (
        "solvers/tycho_nooa/src/tycho_nooa/compatibility_authorization.py"
    )
    module_root = REPOSITORY_ROOT / "solvers/tycho_nooa/src/tycho_nooa"
    raw = {
        "schema_version": 1,
        "experiment_id": "test-runway",
        "execution_authorized": True,
        "runtime_commit": "8d79177",
        "package_commit": "8d79177",
        "gate": commitment(GATE_PATH),
        "prior_native_control": commitment(NATIVE_CONTROL),
        "prior_codeact_trace": commitment(CODEACT_TRACE),
        "runner": commitment(runner),
        "gate_executor": commitment(executor),
        "launch_wrapper": commitment(module_root / "compatibility_launch.py"),
        "authorization_schema": commitment(authorization_schema),
        "codeact_controller": commitment(module_root / "sterile_codeact.py"),
        "execution_projection": commitment(module_root / "execution_events.py"),
        "sterile_executor": commitment(module_root / "sterile_execution.py"),
        "sterile_worker": commitment(module_root / "_sterile_worker.py"),
        "return_contract_module": commitment(module_root / "compatibility_return.py"),
        "evidence_module": commitment(module_root / "evidence.py"),
        "protocol": commitment(protocol),
        "experiment_card": commitment(card),
        "codeact_prompt_sha256": sha256_text(
            "You are completing a protocol-compatibility eligibility gate, not a reasoning "
            "benchmark. Use only execute_python(code) and the exact gate-specific "
            "return_result schema shown by the provider. Python locals persist across cells. "
            "evidence.query exists only when the visible contract enables it; use that exact "
            "query shape. Finish with the typed result."
        ),
        "model_route": {
            "model": "fake-model",
            "reported_model": "fake-model",
            "provider": "provider-a",
            "reasoning_effort": "enabled",
            "temperature": None,
        },
        "caps": {
            "calls_total": 6,
            "input_tokens_per_call": 1000,
            "output_tokens_per_call": 100,
            "wall_seconds": 30,
            "max_cost_usd": 3.0,
        },
        "output_root": str(tmp_path / "run"),
    }
    manifest_path = tmp_path / "experiment.json"
    manifest_path.write_text(json.dumps(raw, indent=2) + "\n")
    authorization = CompatibilityAuthorization(
        authorization_id="test-runway",
        experiment_manifest_sha256=sha256_file(manifest_path),
        allowed_provider="provider-a",
        requested_model="fake-model",
        expected_reported_model="fake-model",
        allowed_reasoning_effort="enabled",
        maximum_calls=6,
        maximum_spend_usd=Decimal("3.00"),
        max_input_tokens_per_call=1000,
        max_output_tokens_per_call=100,
        expires_at=datetime.now(UTC) + timedelta(days=1),
        pricing=PricingSchedule(
            provider="provider-a",
            model="fake-model",
            input_usd_per_million=1,
            cached_input_usd_per_million=0,
            output_usd_per_million=1,
            source="test",
            effective_at=datetime.now(UTC),
        ),
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(authorization.model_dump_json(indent=2) + "\n")
    return manifest_path, authorization_path


async def test_six_call_arm_records_earliest_fourth_call_completion(
    tmp_path: Path,
) -> None:
    gate = load_compatibility_gate(GATE_PATH, repository_root=REPOSITORY_ROOT)
    query = (
        "res = evidence.query(kind='transitions', level=0, limit=1, "
        "projection='full', grid_encoding='rle')\nprint(res)"
    )
    llm = FakeLLMClient(
        scripted_responses=[
            response("execute_python", {"code": query}),
            response(
                "execute_python",
                {"code": "answer = '11->5'; ref = 'L0:T1'; print(ref, answer)"},
            ),
            response("execute_python", {"code": "print(answer, ref)"}),
            response(
                "return_result",
                {"conclusion": "11->5", "evidence_ref": "L0:T1"},
            ),
        ]
    )

    observation = await execute_codeact_runway_arm(
        gate,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "codeact",
        llm=llm,
        max_llm_calls=6,
    )
    milestones = extract_runway_milestones(tmp_path / "codeact/provenance.jsonl")

    assert observation.passed is True
    assert observation.metrics.return_call_index == 4
    assert milestones.capability_discovered_call == 1
    assert milestones.evidence_disclosed_call == 1
    assert milestones.computation_completed_call == 2
    assert milestones.correct_answer_observed_call == 2
    assert milestones.return_result_call == 4


async def test_six_call_arm_exhaustion_is_preserved(tmp_path: Path) -> None:
    gate = load_compatibility_gate(GATE_PATH, repository_root=REPOSITORY_ROOT)
    inspection = response("execute_python", {"code": "print(repr(evidence))"})

    observation = await execute_codeact_runway_arm(
        gate,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "codeact",
        llm=FakeLLMClient(scripted_responses=[inspection] * 6),
        max_llm_calls=6,
    )

    assert observation.passed is False
    assert observation.metrics.provider_calls == 6
    assert observation.metrics.return_call_index is None


def test_provider_call_ceiling_is_not_rendered_into_codeact_provenance(
    tmp_path: Path,
) -> None:
    fixture = REPOSITORY_ROOT / (
        "analysis/parity-compat-v12/runs/"
        "20260824T112416Z-gemma-4-31b-openinference/ladder/"
        "g3-composition/codeact/provenance.jsonl"
    )
    text = fixture.read_text()

    assert "max_provider_calls" not in text
    assert "max_llm_calls" not in text


def test_frozen_v13_package_is_committed_but_disabled() -> None:
    manifest, gate = load_codeact_runway_manifest(
        V13_MANIFEST, repository_root=REPOSITORY_ROOT
    )

    assert manifest.execution_authorized is False
    assert manifest.caps.calls_total == 6
    assert gate.task.gate_id == "g3-composition"
    with pytest.raises(RuntimeError, match="disabled"):
        manifest.require_execution_authorized()


async def test_authorized_diagnostic_executes_only_codeact(tmp_path: Path) -> None:
    manifest, authorization = diagnostic_inputs(tmp_path)
    query = (
        "res = evidence.query(kind='transitions', level=0, limit=1, "
        "projection='full', grid_encoding='rle'); print(res)"
    )
    llm = FakeLLMClient(
        scripted_responses=[
            response("execute_python", {"code": query}),
            response(
                "execute_python",
                {"code": "answer = '11->5'; ref = 'L0:T1'; print(ref, answer)"},
            ),
            response("execute_python", {"code": "print(answer, ref)"}),
            response(
                "return_result",
                {"conclusion": "11->5", "evidence_ref": "L0:T1"},
            ),
        ],
    )
    llm.model = "fake-model"
    llm.config["reasoning"] = {"enabled": True}

    report = await execute_authorized_codeact_runway(
        manifest,
        authorization,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "run",
        llm=llm,
    )

    assert report.interpretation == "deadline_truncation"
    assert report.prior_native_control["arm"] == "native"
    assert llm.call_count == 4
    assert not (tmp_path / "run/native").exists()


def test_runway_manifest_rejects_disabled_execution_and_drift(tmp_path: Path) -> None:
    manifest_path, _ = diagnostic_inputs(tmp_path)
    raw = json.loads(manifest_path.read_text())
    raw["execution_authorized"] = False
    raw["model_route"] = {}
    raw["caps"]["max_cost_usd"] = None
    manifest_path.write_text(json.dumps(raw))

    manifest, gate = load_codeact_runway_manifest(
        manifest_path, repository_root=REPOSITORY_ROOT
    )
    assert gate.task.gate_id == "g3-composition"
    with pytest.raises(RuntimeError, match="disabled"):
        manifest.require_execution_authorized()

    raw["gate"]["sha256"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="file commitment mismatch"):
        load_codeact_runway_manifest(manifest_path, repository_root=REPOSITORY_ROOT)


def test_enabled_manifest_requires_route_and_cost(tmp_path: Path) -> None:
    manifest_path, _ = diagnostic_inputs(tmp_path)
    raw = json.loads(manifest_path.read_text())
    raw["model_route"]["provider"] = None

    with pytest.raises(ValueError, match="route and cost"):
        CodeActRunwayManifest.model_validate(raw)


@pytest.mark.parametrize(
    ("passed", "terminal", "correct", "expected"),
    [
        (True, 4, None, "deadline_truncation"),
        (True, 5, None, "compatible_turn_hungry"),
        (True, 6, None, "compatible_fragile"),
        (True, 2, None, "compatible_before_prior_ceiling"),
        (False, None, 3, "interface_policy_mismatch"),
        (False, None, None, "no_compatible_completion"),
    ],
)
def test_runway_interpretations(
    passed: bool, terminal: int | None, correct: int | None, expected: str
) -> None:
    observation = CompatibilityArmObservation.model_construct(passed=passed)
    milestones = CodeActRunwayMilestones(
        return_result_call=terminal,
        correct_answer_observed_call=correct,
    )

    assert _interpret(observation, milestones) == expected
