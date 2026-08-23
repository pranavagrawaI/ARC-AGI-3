"""Sequential eligibility ladder for matched Native and sterile CodeAct arms."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, Protocol

from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, ConfigDict, model_validator

from tycho_nooa.compatibility_experiment import (
    CompatibilityGateReport,
    execute_compatibility_gate,
)
from tycho_nooa.parity_compatibility import CompatibilityGate

CompatibilityDisposition = Literal[
    "not_eligible_termination",
    "rejected_capability_pairing",
    "rejected_representation",
    "rejected_cognitive_eligibility",
    "eligible_rle",
    "eligible_raw",
]
CompatibilityEncoding = Literal["rle", "raw"]
ArmIntegrity = Literal[
    "valid",
    "provider_censored",
    "execution_failed",
    "not_run",
]
PairIntegrity = Literal["complete", "incomplete"]
ArmOutcome = Literal[
    "passed",
    "failed_termination",
    "failed_gate",
    "unknown",
    "not_run",
]
AdvancementReason = Literal[
    "all_prerequisites_passed",
    "native_prerequisite_failed",
    "codeact_prerequisite_failed",
    "pair_incomplete",
    "later_gate_failed",
]
GateExecutor = Callable[
    [CompatibilityGate, Path], Awaitable[CompatibilityGateReport]
]


class CompatibilityLadderObserver(Protocol):
    def gate_started(self, gate: CompatibilityGate) -> None: ...

    def gate_completed(
        self, gate: CompatibilityGate, report: CompatibilityGateReport
    ) -> None: ...

    def ladder_completed(self, report: CompatibilityLadderReport) -> None: ...


class CompatibilityLadder(BaseModel):
    """The complete frozen gate set, including conditional alternatives."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    termination: CompatibilityGate
    metadata_query: CompatibilityGate
    representation_rle: CompatibilityGate
    representation_raw: CompatibilityGate
    easy_positive_rle: CompatibilityGate
    easy_positive_raw: CompatibilityGate

    @model_validator(mode="after")
    def gate_roles_and_matched_alternatives_are_exact(self) -> CompatibilityLadder:
        expected_kinds = {
            "termination": "termination",
            "metadata_query": "metadata_query",
            "representation_rle": "representation_rle",
            "representation_raw": "representation_raw",
            "easy_positive_rle": "easy_positive",
            "easy_positive_raw": "easy_positive",
        }
        for name, expected_kind in expected_kinds.items():
            gate = getattr(self, name)
            if gate.task.gate_kind != expected_kind:
                raise ValueError(f"{name} must contain a {expected_kind} gate")
        gate_ids = [gate.task.gate_id for gate in self.ordered_gates()]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("compatibility ladder gate ids must be unique")
        _validate_projection(self.metadata_query, projection="metadata")
        _validate_encoding_pair(self.representation_rle, self.representation_raw)
        _validate_encoding_pair(self.easy_positive_rle, self.easy_positive_raw)
        return self

    def ordered_gates(self) -> tuple[CompatibilityGate, ...]:
        return (
            self.termination,
            self.metadata_query,
            self.representation_rle,
            self.representation_raw,
            self.easy_positive_rle,
            self.easy_positive_raw,
        )


class CompatibilityLadderReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    disposition: CompatibilityDisposition
    selected_encoding: CompatibilityEncoding | None
    stopped_after: str
    gate_reports: tuple[CompatibilityGateReport, ...]
    skipped_gate_ids: tuple[str, ...]
    native_llm_calls: int
    codeact_llm_calls: int


class CompatibilityIntegrity(BaseModel):
    """Validity of each arm and of the matched pair as distinct facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    native: ArmIntegrity
    codeact: ArmIntegrity
    pair: PairIntegrity


class CompatibilityArmOutcomes(BaseModel):
    """Observed arm results without smuggling in pair-level conclusions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    native: ArmOutcome
    codeact: ArmOutcome


class CompatibilityAdvancement(BaseModel):
    """Whether the conjunctive eligibility rule permits advancement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    eligible: bool
    reason: AdvancementReason


class CompatibilityRunAssessment(BaseModel):
    """Scientific interpretation kept separate from raw machine disposition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    integrity: CompatibilityIntegrity
    arm_outcomes: CompatibilityArmOutcomes
    advancement: CompatibilityAdvancement


async def execute_compatibility_ladder(
    ladder: CompatibilityLadder,
    *,
    repository_root: Path,
    destination: Path,
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
    max_output_tokens: int = 4096,
    session_timeout_seconds: float = 300.0,
    cell_timeout_seconds: float = 5.0,
) -> CompatibilityLadderReport:
    """Run the paid-compatible ladder with a fixed three-call ceiling per arm."""

    async def execute(gate: CompatibilityGate, gate_destination: Path):
        return await execute_compatibility_gate(
            gate,
            repository_root=repository_root,
            destination=gate_destination,
            native_llm=native_llm,
            codeact_llm=codeact_llm,
            max_llm_calls=3,
            max_output_tokens=max_output_tokens,
            session_timeout_seconds=session_timeout_seconds,
            cell_timeout_seconds=cell_timeout_seconds,
        )

    return await run_compatibility_ladder(
        ladder, destination=destination, execute_gate=execute
    )


async def run_compatibility_ladder(
    ladder: CompatibilityLadder,
    *,
    destination: Path,
    execute_gate: GateExecutor,
    observer: CompatibilityLadderObserver | None = None,
) -> CompatibilityLadderReport:
    """Apply predeclared stop/branch rules to matched gate reports."""
    destination.mkdir(parents=True, exist_ok=False)
    reports: list[CompatibilityGateReport] = []

    termination = await _run(
        ladder.termination, destination, execute_gate, reports, observer
    )
    if not termination.passed:
        return _finish(
            ladder,
            destination,
            reports,
            "not_eligible_termination",
            None,
            observer,
        )

    metadata = await _run(
        ladder.metadata_query, destination, execute_gate, reports, observer
    )
    if not metadata.passed:
        return _finish(
            ladder,
            destination,
            reports,
            "rejected_capability_pairing",
            None,
            observer,
        )

    representation = await _run(
        ladder.representation_rle, destination, execute_gate, reports, observer
    )
    if representation.passed:
        selected: CompatibilityEncoding = "rle"
        easy_gate = ladder.easy_positive_rle
    else:
        raw = await _run(
            ladder.representation_raw, destination, execute_gate, reports, observer
        )
        if not raw.passed:
            return _finish(
                ladder,
                destination,
                reports,
                "rejected_representation",
                None,
                observer,
            )
        selected = "raw"
        easy_gate = ladder.easy_positive_raw

    easy = await _run(easy_gate, destination, execute_gate, reports, observer)
    if not easy.passed:
        return _finish(
            ladder,
            destination,
            reports,
            "rejected_cognitive_eligibility",
            selected,
            observer,
        )
    disposition: CompatibilityDisposition = f"eligible_{selected}"
    return _finish(ladder, destination, reports, disposition, selected, observer)


async def _run(
    gate: CompatibilityGate,
    destination: Path,
    execute_gate: GateExecutor,
    reports: list[CompatibilityGateReport],
    observer: CompatibilityLadderObserver | None,
) -> CompatibilityGateReport:
    if observer is not None:
        observer.gate_started(gate)
    report = await execute_gate(gate, destination / gate.task.gate_id)
    if report.gate_id != gate.task.gate_id:
        raise ValueError("gate executor returned a report for a different gate")
    reports.append(report)
    if observer is not None:
        observer.gate_completed(gate, report)
    return report


def _finish(
    ladder: CompatibilityLadder,
    destination: Path,
    reports: list[CompatibilityGateReport],
    disposition: CompatibilityDisposition,
    selected_encoding: CompatibilityEncoding | None,
    observer: CompatibilityLadderObserver | None,
) -> CompatibilityLadderReport:
    run_ids = {report.gate_id for report in reports}
    report = CompatibilityLadderReport(
        disposition=disposition,
        selected_encoding=selected_encoding,
        stopped_after=reports[-1].gate_id,
        gate_reports=tuple(reports),
        skipped_gate_ids=tuple(
            gate.task.gate_id
            for gate in ladder.ordered_gates()
            if gate.task.gate_id not in run_ids
        ),
        native_llm_calls=sum(row.native.metrics.llm_calls for row in reports),
        codeact_llm_calls=sum(row.codeact.metrics.llm_calls for row in reports),
    )
    _write_json(destination / "ladder-report.json", report.model_dump(mode="json"))
    if observer is not None:
        observer.ladder_completed(report)
    return report


def _validate_projection(
    gate: CompatibilityGate, *, projection: Literal["full", "metadata"]
) -> None:
    if gate.evidence.projection != projection:
        raise ValueError(f"{gate.task.gate_id} must use {projection} evidence")


def _validate_encoding_pair(
    rle_gate: CompatibilityGate, raw_gate: CompatibilityGate
) -> None:
    _validate_projection(rle_gate, projection="full")
    _validate_projection(raw_gate, projection="full")
    if rle_gate.evidence.grid_encoding != "rle":
        raise ValueError(f"{rle_gate.task.gate_id} must use RLE evidence")
    if raw_gate.evidence.grid_encoding != "raw":
        raise ValueError(f"{raw_gate.task.gate_id} must use raw evidence")
    if _matched_gate_value(rle_gate) != _matched_gate_value(raw_gate):
        raise ValueError("RLE and raw alternatives must differ only by identity and encoding")


def _matched_gate_value(gate: CompatibilityGate) -> dict:
    value = gate.model_dump(mode="json")
    value["task"].pop("gate_id")
    value["task"].pop("gate_kind")
    value["evidence"].pop("grid_encoding")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
