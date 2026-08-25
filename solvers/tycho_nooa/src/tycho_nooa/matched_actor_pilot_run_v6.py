"""Five-packet V2 pilot pair seam over scoped immutable evidence handles."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from nooa.unifiedllm import UnifiedLLM

from tycho_nooa import matched_actor_pilot_run as _v1
from tycho_nooa.cognitive_parity import CognitiveParityPacket
from tycho_nooa.matched_actor_corpus import ActorCognitiveStratum
from tycho_nooa.matched_actor_experiment import ActorTreatmentObservation
from tycho_nooa.matched_actor_experiment_v6 import _run_codeact, _run_native
from tycho_nooa.matched_actor_pilot import PilotArm

PilotArmResult = _v1.PilotArmResult
PilotAssessment = _v1.PilotAssessment
PilotPairReport = _v1.PilotPairReport
_write_json = _v1._write_json


def assess_pilot(reports: tuple[PilotPairReport, ...]) -> PilotAssessment:
    """Apply the precommitted advancement rule to five complete pairs."""
    totals = _v1._pilot_totals(reports)
    advance = _codeact_advances(totals)
    if totals.complete_pairs != 5:
        disposition = "incomplete_pilot"
    elif totals.native_correct < 3 and totals.codeact_correct < 3:
        disposition = "model_unreliable"
    elif advance:
        disposition = "advance_codeact_full_treatment"
    else:
        disposition = "select_native"
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
        disposition=disposition,
    )


def _codeact_advances(totals) -> bool:
    if totals.complete_pairs != 5:
        return False
    if (
        totals.wins >= 2
        and totals.losses <= totals.wins
        and totals.codeact_correct >= totals.native_correct
    ):
        return True
    if totals.codeact_correct != totals.native_correct:
        return False
    return _v1._materially_lower(
        totals.codeact_calls, totals.native_calls
    ) or _v1._materially_lower(totals.codeact_cost, totals.native_cost)


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
                    arm=arm,
                    status="completed",
                    observation=observation,
                )
            )
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
    kwargs = {
        "evidence_root": repository_root / packet.checkpoint_root,
        "destination": destination,
        "llm": llm,
        "max_provider_calls": max_provider_calls,
        "max_output_tokens": max_output_tokens,
        "cell_timeout_seconds": cell_timeout_seconds,
    }
    if arm == "native":
        return await _run_native(packet, **kwargs)
    return await _run_codeact(packet, **kwargs)
