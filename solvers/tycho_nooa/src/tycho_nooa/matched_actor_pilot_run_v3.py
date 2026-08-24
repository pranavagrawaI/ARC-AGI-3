"""V3 pilot pair seam: V1 assessment over value-adapted evidence runs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from nooa.unifiedllm import UnifiedLLM

from tycho_nooa import matched_actor_pilot_run as _v1
from tycho_nooa.cognitive_parity import CognitiveParityPacket
from tycho_nooa.matched_actor_corpus import ActorCognitiveStratum
from tycho_nooa.matched_actor_experiment import ActorTreatmentObservation
from tycho_nooa.matched_actor_experiment_v3 import _run_codeact, _run_native
from tycho_nooa.matched_actor_pilot import PilotArm

PilotArmResult = _v1.PilotArmResult
PilotAssessment = _v1.PilotAssessment
PilotPairReport = _v1.PilotPairReport
assess_pilot = _v1.assess_pilot
_codeact_advances = _v1._codeact_advances
_disposition = _v1._disposition
_write_json = _v1._write_json


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
        # Preserve the V1 paired-failure policy; only the evidence value
        # boundary changed.
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
                arm_destination / "failure.json",
                failure.model_dump(mode="json"),
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
