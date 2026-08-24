"""Counterbalanced pair runner and precommitted pilot assessment."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, ConfigDict

from tycho_nooa.cognitive_parity import CognitiveParityPacket
from tycho_nooa.matched_actor_corpus import ActorCognitiveStratum
from tycho_nooa.matched_actor_experiment import (
    ActorTreatmentObservation,
    _run_codeact,
    _run_native,
)
from tycho_nooa.matched_actor_pilot import PilotArm


class PilotArmResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: PilotArm
    status: Literal["completed", "failed"]
    observation: ActorTreatmentObservation | None = None
    error_type: str | None = None
    error_message: str | None = None


class PilotPairReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str
    stratum: ActorCognitiveStratum
    arm_order: tuple[PilotArm, PilotArm]
    arms: tuple[PilotArmResult, ...]

    @property
    def complete(self) -> bool:
        return len(self.arms) == 2 and all(arm.status == "completed" for arm in self.arms)


class PilotAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    complete_pairs: int
    native_correct: int
    codeact_correct: int
    codeact_packet_wins: int
    codeact_packet_losses: int
    native_provider_calls: int
    codeact_provider_calls: int
    native_cost_usd: float
    codeact_cost_usd: float
    native_control_tax_calls: tuple[int | None, ...]
    codeact_control_tax_calls: tuple[int | None, ...]
    codeact_advances: bool
    disposition: Literal[
        "advance_codeact_full_treatment",
        "select_native",
        "model_unreliable",
        "incomplete_pilot",
    ]


@dataclass(frozen=True)
class _PilotTotals:
    complete_pairs: int
    native_correct: int
    codeact_correct: int
    wins: int
    losses: int
    native_calls: int
    codeact_calls: int
    native_cost: float
    codeact_cost: float
    native_tax: tuple[int | None, ...]
    codeact_tax: tuple[int | None, ...]


@dataclass(frozen=True)
class _CorrectnessTotals:
    native: int
    codeact: int
    wins: int
    losses: int


async def execute_pilot_pair(
    packet: CognitiveParityPacket,
    *,
    stratum: ActorCognitiveStratum,
    arm_order: tuple[PilotArm, PilotArm],
    repository_root: Path,
    destination: Path,
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
    max_provider_calls: int,
    max_output_tokens: int,
    session_timeout_seconds: float,
    cell_timeout_seconds: float,
    continuation_allowed: Callable[[], bool] = lambda: True,
) -> PilotPairReport:
    destination.mkdir(parents=True, exist_ok=False)
    llms = {"native": native_llm, "codeact": codeact_llm}
    results: list[PilotArmResult] = []
    for arm in arm_order:
        if not continuation_allowed():
            break
        arm_destination = destination / arm
        try:
            async with asyncio.timeout(session_timeout_seconds):
                observation = await _execute_arm(
                    arm,
                    packet,
                    repository_root=repository_root,
                    destination=arm_destination,
                    llm=llms[arm],
                    max_provider_calls=max_provider_calls,
                    max_output_tokens=max_output_tokens,
                    cell_timeout_seconds=cell_timeout_seconds,
                )
            results.append(
                PilotArmResult(
                    arm=arm, status="completed", observation=observation
                )
            )
        # Model/provider adapters fail heterogeneously; the pair artifact must retain
        # any ordinary failure while still allowing the clean counterpart to run.
        except Exception as error:  # noqa: BLE001
            arm_destination.mkdir(exist_ok=True)
            failure = PilotArmResult(
                arm=arm,
                status="failed",
                error_type=type(error).__name__,
                error_message=str(error),
            )
            results.append(failure)
            _write_json(
                arm_destination / "failure.json", failure.model_dump(mode="json")
            )
    report = PilotPairReport(
        packet_id=packet.task.packet_id,
        stratum=stratum,
        arm_order=arm_order,
        arms=tuple(results),
    )
    _write_json(destination / "pair-report.json", report.model_dump(mode="json"))
    return report


def assess_pilot(reports: tuple[PilotPairReport, ...]) -> PilotAssessment:
    totals = _pilot_totals(reports)
    advance = _codeact_advances(
        complete_pairs=totals.complete_pairs,
        wins=totals.wins,
        losses=totals.losses,
        native_correct=totals.native_correct,
        codeact_correct=totals.codeact_correct,
        native_calls=totals.native_calls,
        codeact_calls=totals.codeact_calls,
        native_cost=totals.native_cost,
        codeact_cost=totals.codeact_cost,
    )
    return PilotAssessment(
        complete_pairs=totals.complete_pairs,
        native_correct=totals.native_correct,
        codeact_correct=totals.codeact_correct,
        codeact_packet_wins=totals.wins,
        codeact_packet_losses=totals.losses,
        native_provider_calls=totals.native_calls,
        codeact_provider_calls=totals.codeact_calls,
        native_cost_usd=totals.native_cost,
        codeact_cost_usd=totals.codeact_cost,
        native_control_tax_calls=totals.native_tax,
        codeact_control_tax_calls=totals.codeact_tax,
        codeact_advances=advance,
        disposition=_disposition(
            complete_pairs=totals.complete_pairs,
            native_correct=totals.native_correct,
            codeact_correct=totals.codeact_correct,
            advances=advance,
        ),
    )


def _pilot_totals(reports: tuple[PilotPairReport, ...]) -> _PilotTotals:
    complete = tuple(report for report in reports if report.complete)
    native = tuple(_observation(report, "native") for report in complete)
    codeact = tuple(_observation(report, "codeact") for report in complete)
    correctness = _correctness_totals(native, codeact)
    return _PilotTotals(
        complete_pairs=len(complete),
        native_correct=correctness.native,
        codeact_correct=correctness.codeact,
        wins=correctness.wins,
        losses=correctness.losses,
        native_calls=_provider_calls(native),
        codeact_calls=_provider_calls(codeact),
        native_cost=_dollar_cost(native),
        codeact_cost=_dollar_cost(codeact),
        native_tax=_control_taxes(native),
        codeact_tax=_control_taxes(codeact),
    )


def _correctness_totals(
    native: tuple[ActorTreatmentObservation, ...],
    codeact: tuple[ActorTreatmentObservation, ...],
) -> _CorrectnessTotals:
    native_flags = [_correct(value) for value in native]
    codeact_flags = [_correct(value) for value in codeact]
    pairs = tuple(zip(native_flags, codeact_flags, strict=True))
    return _CorrectnessTotals(
        native=sum(native_flags),
        codeact=sum(codeact_flags),
        wins=sum(codeact_won and not native_won for native_won, codeact_won in pairs),
        losses=sum(native_won and not codeact_won for native_won, codeact_won in pairs),
    )


def _provider_calls(values: tuple[ActorTreatmentObservation, ...]) -> int:
    return sum(value.resources.provider_calls for value in values)


def _dollar_cost(values: tuple[ActorTreatmentObservation, ...]) -> float:
    return sum(value.resources.dollar_cost for value in values)


def _control_taxes(
    values: tuple[ActorTreatmentObservation, ...],
) -> tuple[int | None, ...]:
    return tuple(_control_tax(value) for value in values)


async def _execute_arm(
    arm: PilotArm,
    packet: CognitiveParityPacket,
    *,
    repository_root: Path,
    destination: Path,
    llm: UnifiedLLM,
    max_provider_calls: int,
    max_output_tokens: int,
    cell_timeout_seconds: float,
) -> ActorTreatmentObservation:
    evidence_root = repository_root / packet.checkpoint_root
    kwargs = {
        "evidence_root": evidence_root,
        "destination": destination,
        "llm": llm,
        "max_provider_calls": max_provider_calls,
        "max_output_tokens": max_output_tokens,
        "cell_timeout_seconds": cell_timeout_seconds,
    }
    if arm == "native":
        return await _run_native(packet, **kwargs)
    return await _run_codeact(packet, **kwargs)


def _observation(
    report: PilotPairReport, arm: PilotArm
) -> ActorTreatmentObservation:
    result = next(item for item in report.arms if item.arm == arm)
    assert result.observation is not None
    return result.observation


def _correct(observation: ActorTreatmentObservation) -> bool:
    value = observation.evaluation
    checks = [value.evidence_precision == 1, value.evidence_recall == 1]
    checks.extend(
        item is not False
        for item in (
            value.verdict_correct,
            value.falsification_correct,
            value.invariant_correct,
            value.scope_correct,
            value.action_legal,
        )
    )
    if value.invariant_atom_recall is not None:
        checks.append(value.invariant_atom_recall == 1)
    return all(checks)


def _codeact_advances(
    *,
    complete_pairs: int,
    wins: int,
    losses: int,
    native_correct: int,
    codeact_correct: int,
    native_calls: int,
    codeact_calls: int,
    native_cost: float,
    codeact_cost: float,
) -> bool:
    if complete_pairs != 6:
        return False
    if wins >= 2 and losses <= wins and codeact_correct >= native_correct:
        return True
    if codeact_correct != native_correct:
        return False
    return _materially_lower(codeact_calls, native_calls) or _materially_lower(
        codeact_cost, native_cost
    )


def _materially_lower(candidate: float, incumbent: float) -> bool:
    return incumbent > 0 and candidate <= incumbent * 0.8


def _control_tax(observation: ActorTreatmentObservation) -> int | None:
    formed = observation.milestones.candidate_answer_formed
    accepted = observation.milestones.terminal_result_accepted
    if formed is None or accepted is None:
        return None
    return accepted - formed


def _disposition(
    *, complete_pairs: int, native_correct: int, codeact_correct: int, advances: bool
) -> str:
    if complete_pairs != 6:
        return "incomplete_pilot"
    if native_correct < 4 and codeact_correct < 4:
        return "model_unreliable"
    if advances:
        return "advance_codeact_full_treatment"
    return "select_native"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
