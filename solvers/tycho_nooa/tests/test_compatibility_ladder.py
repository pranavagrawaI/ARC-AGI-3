from __future__ import annotations

from pathlib import Path

import pytest

from tycho_nooa.compatibility_experiment import (
    CompatibilityArmObservation,
    CompatibilityGateReport,
    CompatibilityMetrics,
)
from tycho_nooa.compatibility_ladder import (
    CompatibilityLadder,
    run_compatibility_ladder,
)
from tycho_nooa.parity_compatibility import (
    CompatibilityEvidencePolicy,
    CompatibilityExpected,
    CompatibilityGate,
    CompatibilityTask,
)


def disabled_gate(gate_id: str, gate_kind: str, expected: dict) -> CompatibilityGate:
    return CompatibilityGate.model_validate({
        "task": {"gate_id": gate_id, "gate_kind": gate_kind, "instruction": "Do it."},
        "evidence": {"enabled": False, "max_queries": 0},
        "expected": expected,
    })


def evidence_gate(gate_id: str, gate_kind: str, expected: dict, *, projection: str) -> CompatibilityGate:
    return CompatibilityGate(
        checkpoint_root="checkpoint",
        checkpoint_sha256="sha256:" + "0" * 64,
        task=CompatibilityTask(gate_id=gate_id, gate_kind=gate_kind, instruction="Do it."),
        evidence=CompatibilityEvidencePolicy(
            enabled=True, kind="transitions", level=0, limit=1,
            projection=projection, grid_encoding="rle", max_queries=1,
        ),
        expected=CompatibilityExpected.model_validate(expected),
    )


def ladder() -> CompatibilityLadder:
    return CompatibilityLadder(
        termination=disabled_gate("g0", "termination", {"ok": True}),
        capability_retrieval=evidence_gate(
            "g1", "capability_retrieval", {"evidence_ref": "L0:T1"}, projection="metadata"
        ),
        identifier_binding=CompatibilityGate(
            task=CompatibilityTask(
                gate_id="g2a", gate_kind="identifier_binding", instruction="Do it.",
                inputs={"record": {"ref": "L0:T1"}, "frontier": "L1:T2", "next_after": "L0:T1"},
            ),
            evidence=CompatibilityEvidencePolicy(enabled=False, max_queries=0),
            expected=CompatibilityExpected(
                evidence_ref="L0:T1", frontier_substitution="L1:T2",
                turn_number_leakage="L0:T2",
            ),
        ),
        representation_rle=disabled_gate(
            "g2b", "representation_rle", {"decoded_value": 5}
        ),
        composition=evidence_gate(
            "g3", "composition", {"conclusion": "11->5", "evidence_ref": "L0:T1"},
            projection="full",
        ),
    )


def report(gate: CompatibilityGate, passed: bool) -> CompatibilityGateReport:
    metrics = CompatibilityMetrics(
        llm_calls=2, input_tokens=10, output_tokens=2,
        evidence_queries=int(gate.evidence.enabled), python_cells=0,
        invalid_interactions=0, wall_seconds=0.1,
    )
    arms = [CompatibilityArmObservation(
        gate_id=gate.task.gate_id, arm=arm, passed=passed, metrics=metrics
    ) for arm in ("native", "codeact")]
    return CompatibilityGateReport(
        gate_id=gate.task.gate_id, gate_kind=gate.task.gate_kind, passed=passed,
        native=arms[0], codeact=arms[1],
    )


async def test_all_five_gates_pass_sequentially(tmp_path: Path) -> None:
    run: list[str] = []
    async def execute(gate: CompatibilityGate, _: Path):
        run.append(gate.task.gate_id)
        return report(gate, True)

    result = await run_compatibility_ladder(
        ladder(), destination=tmp_path / "run", execute_gate=execute
    )

    assert result.disposition == "eligible"
    assert run == ["g0", "g1", "g2a", "g2b", "g3"]
    assert result.skipped_gate_ids == ()


@pytest.mark.parametrize(("failed", "disposition", "run"), [
    ("g0", "not_eligible_termination", ["g0"]),
    ("g1", "rejected_capability_pairing", ["g0", "g1"]),
    ("g2a", "rejected_identifier_binding", ["g0", "g1", "g2a"]),
    ("g2b", "rejected_representation", ["g0", "g1", "g2a", "g2b"]),
    ("g3", "rejected_composition", ["g0", "g1", "g2a", "g2b", "g3"]),
])
async def test_failure_stops_at_first_unearned_burden(
    tmp_path: Path, failed: str, disposition: str, run: list[str]
) -> None:
    observed: list[str] = []
    async def execute(gate: CompatibilityGate, _: Path):
        observed.append(gate.task.gate_id)
        return report(gate, gate.task.gate_id != failed)

    result = await run_compatibility_ladder(
        ladder(), destination=tmp_path / "run", execute_gate=execute
    )

    assert result.disposition == disposition
    assert observed == run
