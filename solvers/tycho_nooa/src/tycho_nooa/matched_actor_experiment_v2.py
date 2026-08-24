"""V2 Actor construction seam: V1 cognition over bounded evidence."""

from __future__ import annotations

import time
from pathlib import Path

from nooa.unifiedllm import UnifiedLLM

from tycho_nooa.cognitive_parity import CognitiveParityPacket
from tycho_nooa.evidence_v2 import WorkspaceEvidenceCapability
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.matched_actor import (
    MatchedCodeActController,
    MatchedNativeActorController,
    cognitive_answer,
)
from tycho_nooa.matched_actor_experiment import (
    ActorTreatmentObservation,
    RecordingEvidenceCapability,
    _codeact_successful_decode_calls,
    _observation,
    _write_json,
)


async def _run_native(
    packet: CognitiveParityPacket,
    *,
    evidence_root: Path,
    destination: Path,
    llm: UnifiedLLM,
    max_provider_calls: int,
    max_output_tokens: int,
    cell_timeout_seconds: float,
) -> ActorTreatmentObservation:
    destination.mkdir()
    evidence = RecordingEvidenceCapability(
        WorkspaceEvidenceCapability.from_root(evidence_root)
    )
    controller = MatchedNativeActorController(
        llm=llm,
        packet=packet,
        evidence=evidence,
        max_llm_calls=max_provider_calls,
        max_output_tokens=max_output_tokens,
        cell_timeout_seconds=cell_timeout_seconds,
    )
    started = time.monotonic()
    matched = await controller.run()
    latency = time.monotonic() - started
    answer = cognitive_answer(packet, matched)
    observation = _observation(
        packet,
        "native",
        answer.model_dump(mode="json"),
        usage=[turn.usage for turn in controller.provider_turns],
        actions=controller.model_actions,
        evidence=evidence,
        python_cells=controller.python_cells,
        invalid_interactions=controller.invalid_returns,
        latency_seconds=latency,
        terminal_call=controller.terminal_return_call,
        successful_decode_calls=controller.successful_decode_calls,
    )
    _write_json(destination / "result.json", observation.model_dump(mode="json"))
    return observation


async def _run_codeact(
    packet: CognitiveParityPacket,
    *,
    evidence_root: Path,
    destination: Path,
    llm: UnifiedLLM,
    max_provider_calls: int,
    max_output_tokens: int,
    cell_timeout_seconds: float,
) -> ActorTreatmentObservation:
    destination.mkdir()
    evidence = RecordingEvidenceCapability(
        WorkspaceEvidenceCapability.from_root(evidence_root)
    )
    provenance_path = destination / "provenance.jsonl"
    log = ExecutionProvenanceLog(
        provenance_path,
        actor_session_id=f"matched:{packet.task.packet_id}:codeact",
    )
    controller = MatchedCodeActController(
        llm=llm,
        packet=packet,
        evidence=evidence,
        provenance=InvocationProvenance(
            log, invocation_id=f"{packet.task.packet_id}:codeact"
        ),
        max_llm_calls=max_provider_calls,
        max_output_tokens=max_output_tokens,
        cell_timeout_seconds=cell_timeout_seconds,
    )
    started = time.monotonic()
    matched = await controller.run()
    latency = time.monotonic() - started
    answer = cognitive_answer(packet, matched)
    invalid = sum(
        event.event_type
        in {
            "ProviderTurnRejected",
            "CapabilityRejected",
            "CellFailed",
            "ReturnRejected",
        }
        for event in log.events
    )
    successful_decode_calls = _codeact_successful_decode_calls(
        controller.model_actions, log.events
    )
    observation = _observation(
        packet,
        "codeact",
        answer.model_dump(mode="json"),
        usage=[turn.usage for turn in controller.provider_turns],
        actions=controller.model_actions,
        evidence=evidence,
        python_cells=sum(
            action.name == "execute_python" for action in controller.model_actions
        ),
        invalid_interactions=invalid,
        latency_seconds=latency,
        terminal_call=controller.terminal_return_call,
        successful_decode_calls=successful_decode_calls,
        provenance_path="provenance.jsonl",
    )
    _write_json(destination / "result.json", observation.model_dump(mode="json"))
    return observation
