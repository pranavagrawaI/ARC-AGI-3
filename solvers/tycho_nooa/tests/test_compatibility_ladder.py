from __future__ import annotations

import json
from pathlib import Path

import pytest

from tycho_nooa.compatibility_experiment import (
    CompatibilityArmObservation,
    CompatibilityGateReport,
    CompatibilityMetrics,
)
from tycho_nooa.compatibility_ladder import (
    CompatibilityLadder,
    CompatibilityRunAssessment,
    run_compatibility_ladder,
)
from tycho_nooa.parity_compatibility import (
    CompatibilityEvidencePolicy,
    CompatibilityExpected,
    CompatibilityGate,
    CompatibilityTask,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def gate(
    gate_id: str,
    gate_kind: str,
    *,
    encoding: str = "raw",
    projection: str = "full",
) -> CompatibilityGate:
    if gate_kind == "termination":
        return CompatibilityGate.model_validate(
            {
                "task": {
                    "gate_id": gate_id,
                    "gate_kind": gate_kind,
                    "instruction": "Return READY.",
                },
                "evidence": {"enabled": False, "max_queries": 0},
                "expected": {"conclusion": "READY"},
            }
        )
    return CompatibilityGate(
        checkpoint_root="checkpoint",
        checkpoint_sha256="sha256:" + "0" * 64,
        task=CompatibilityTask.model_validate(
            {
                "gate_id": gate_id,
                "gate_kind": gate_kind,
                "instruction": "Query once and return the exact answer.",
            }
        ),
        evidence=CompatibilityEvidencePolicy.model_validate(
            {
                "enabled": True,
                "kind": "transitions",
                "level": 0,
                "limit": 1,
                "projection": projection,
                "grid_encoding": encoding,
                "max_queries": 1,
            }
        ),
        expected=CompatibilityExpected(conclusion="ANSWER"),
    )


def ladder() -> CompatibilityLadder:
    return CompatibilityLadder(
        termination=gate("g0", "termination"),
        metadata_query=gate("g1", "metadata_query", projection="metadata"),
        representation_rle=gate("g2-rle", "representation_rle", encoding="rle"),
        representation_raw=gate("g2-raw", "representation_raw", encoding="raw"),
        easy_positive_rle=gate("g3-rle", "easy_positive", encoding="rle"),
        easy_positive_raw=gate("g3-raw", "easy_positive", encoding="raw"),
    )


def report(gate_value: CompatibilityGate, *, passed: bool) -> CompatibilityGateReport:
    metrics = CompatibilityMetrics(
        llm_calls=2,
        input_tokens=10,
        output_tokens=2,
        evidence_queries=int(gate_value.evidence.enabled),
        python_cells=0,
        invalid_interactions=0,
        wall_seconds=0.1,
    )
    native = CompatibilityArmObservation(
        gate_id=gate_value.task.gate_id,
        arm="native",
        passed=passed,
        metrics=metrics,
    )
    codeact = CompatibilityArmObservation(
        gate_id=gate_value.task.gate_id,
        arm="codeact",
        passed=passed,
        metrics=metrics,
    )
    return CompatibilityGateReport(
        gate_id=gate_value.task.gate_id,
        gate_kind=gate_value.task.gate_kind,
        passed=passed,
        native=native,
        codeact=codeact,
    )


async def test_rle_success_skips_both_raw_alternatives(tmp_path: Path) -> None:
    gates_run: list[str] = []

    async def execute(gate_value: CompatibilityGate, _: Path):
        gates_run.append(gate_value.task.gate_id)
        return report(gate_value, passed=True)

    result = await run_compatibility_ladder(
        ladder(), destination=tmp_path / "run", execute_gate=execute
    )

    assert result.disposition == "eligible_rle"
    assert result.selected_encoding == "rle"
    assert gates_run == ["g0", "g1", "g2-rle", "g3-rle"]
    assert result.skipped_gate_ids == ("g2-raw", "g3-raw")
    assert result.native_llm_calls == 8
    assert result.codeact_llm_calls == 8
    assert (tmp_path / "run/ladder-report.json").is_file()


async def test_rle_failure_uses_raw_for_representation_and_cognition(
    tmp_path: Path,
) -> None:
    gates_run: list[str] = []

    async def execute(gate_value: CompatibilityGate, _: Path):
        gates_run.append(gate_value.task.gate_id)
        return report(gate_value, passed=gate_value.task.gate_id != "g2-rle")

    result = await run_compatibility_ladder(
        ladder(), destination=tmp_path / "run", execute_gate=execute
    )

    assert result.disposition == "eligible_raw"
    assert result.selected_encoding == "raw"
    assert gates_run == ["g0", "g1", "g2-rle", "g2-raw", "g3-raw"]
    assert result.skipped_gate_ids == ("g3-rle",)


@pytest.mark.parametrize(
    ("fail_gate", "disposition", "expected_run", "encoding"),
    [
        ("g0", "not_eligible_termination", ["g0"], None),
        ("g1", "rejected_capability_pairing", ["g0", "g1"], None),
        (
            "g2-raw",
            "rejected_representation",
            ["g0", "g1", "g2-rle", "g2-raw"],
            None,
        ),
        (
            "g3-rle",
            "rejected_cognitive_eligibility",
            ["g0", "g1", "g2-rle", "g3-rle"],
            "rle",
        ),
    ],
)
async def test_failure_stops_at_predeclared_boundary(
    tmp_path: Path,
    fail_gate: str,
    disposition: str,
    expected_run: list[str],
    encoding: str | None,
) -> None:
    gates_run: list[str] = []

    async def execute(gate_value: CompatibilityGate, _: Path):
        gates_run.append(gate_value.task.gate_id)
        should_fail = gate_value.task.gate_id == fail_gate or (
            fail_gate == "g2-raw" and gate_value.task.gate_id == "g2-rle"
        )
        return report(gate_value, passed=not should_fail)

    result = await run_compatibility_ladder(
        ladder(), destination=tmp_path / "run", execute_gate=execute
    )

    assert result.disposition == disposition
    assert result.selected_encoding == encoding
    assert gates_run == expected_run


def test_ladder_rejects_unmatched_representation_alternatives() -> None:
    value = ladder().model_dump(mode="json")
    value["representation_raw"]["expected"]["conclusion"] = "DIFFERENT"

    with pytest.raises(ValueError, match="differ only"):
        CompatibilityLadder.model_validate(value)


def test_paid_run_assessment_preserves_valid_partial_evidence() -> None:
    path = (
        REPOSITORY_ROOT
        / "analysis/parity-compat-v2/runs"
        / "20260823T150413Z-deepseek-v4-flash-sail/RESULTS.json"
    )

    assessment = CompatibilityRunAssessment.model_validate(
        json.loads(path.read_text())["assessment"]
    )

    assert assessment.integrity.native == "valid"
    assert assessment.integrity.codeact == "provider_censored"
    assert assessment.integrity.pair == "incomplete"
    assert assessment.arm_outcomes.native == "failed_termination"
    assert assessment.arm_outcomes.codeact == "unknown"
    assert assessment.advancement.eligible is False
    assert assessment.advancement.reason == "native_prerequisite_failed"
