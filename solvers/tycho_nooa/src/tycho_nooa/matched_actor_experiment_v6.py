"""Matched Actor treatment over repaired Context/Evidence Plane v2."""

from __future__ import annotations

import time
from itertools import count
from pathlib import Path

from nooa.unifiedllm import UnifiedLLM

from tycho_nooa.cognitive_parity import CognitiveParityPacket
from tycho_nooa.context_evidence_v2 import (
    ContextEvidencePlane,
    EvidenceAccessContext,
    EvidenceAdmissionAuthority,
    EvidenceCache,
    EvidenceGrantRegistry,
    EvidenceReadBroker,
)
from tycho_nooa.context_projection import ContextProjector
from tycho_nooa.evidence_v2 import WorkspaceEvidenceCapability
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.matched_actor import cognitive_answer
from tycho_nooa.matched_actor_context_v2 import (
    ContextCodeActController,
    ContextNativeActorController,
)
from tycho_nooa.matched_actor_experiment import (
    ActorTreatmentObservation,
    RecordingEvidenceCapability,
    _codeact_successful_decode_calls,
    _observation,
    _write_json,
)


def _context_evidence(
    packet: CognitiveParityPacket,
    evidence_root: Path,
    *,
    arm: str,
) -> tuple[ContextEvidencePlane, RecordingEvidenceCapability]:
    """Give one invocation names and grants over its own immutable evidence."""
    source = RecordingEvidenceCapability(
        WorkspaceEvidenceCapability.from_root(evidence_root)
    )
    context = EvidenceAccessContext(
        session_id=f"matched:{packet.task.packet_id}:{arm}",
        invocation_id=f"{packet.task.packet_id}:{arm}",
        role="actor",
        frontier=packet.task.evidence_frontier.model_dump(mode="json"),
    )
    cache = EvidenceCache()
    grants = EvidenceGrantRegistry()
    names = count(1)
    admission = EvidenceAdmissionAuthority(
        cache,
        grants,
        handle_factory=lambda: f"ev_{arm}_{next(names):04d}",
    )
    broker = EvidenceReadBroker(cache, grants)
    return ContextEvidencePlane(source, admission, broker, context), source


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
    plane, evidence = _context_evidence(packet, evidence_root, arm="native")
    controller = ContextNativeActorController(
        llm=llm,
        packet=packet,
        evidence_plane=plane,
        projector=ContextProjector(recent_exchanges=2),
        max_llm_calls=max_provider_calls,
        max_output_tokens=max_output_tokens,
        cell_timeout_seconds=cell_timeout_seconds,
    )
    started = time.monotonic()
    matched = await controller.run()
    observation = _observation(
        packet,
        "native",
        cognitive_answer(packet, matched).model_dump(mode="json"),
        usage=[turn.usage for turn in controller.provider_turns],
        actions=controller.model_actions,
        evidence=evidence,
        python_cells=controller.python_cells,
        invalid_interactions=controller.invalid_returns,
        latency_seconds=time.monotonic() - started,
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
    plane, evidence = _context_evidence(packet, evidence_root, arm="codeact")
    provenance_path = destination / "provenance.jsonl"
    log = ExecutionProvenanceLog(
        provenance_path,
        actor_session_id=f"matched:{packet.task.packet_id}:codeact",
    )
    controller = ContextCodeActController(
        llm=llm,
        packet=packet,
        evidence_plane=plane,
        projector=ContextProjector(recent_exchanges=2),
        provenance=InvocationProvenance(
            log, invocation_id=f"{packet.task.packet_id}:codeact"
        ),
        max_llm_calls=max_provider_calls,
        max_output_tokens=max_output_tokens,
        cell_timeout_seconds=cell_timeout_seconds,
    )
    started = time.monotonic()
    matched = await controller.run()
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
    observation = _observation(
        packet,
        "codeact",
        cognitive_answer(packet, matched).model_dump(mode="json"),
        usage=[turn.usage for turn in controller.provider_turns],
        actions=controller.model_actions,
        evidence=evidence,
        python_cells=sum(
            action.name == "execute_python" for action in controller.model_actions
        ),
        invalid_interactions=invalid,
        latency_seconds=time.monotonic() - started,
        terminal_call=controller.terminal_return_call,
        successful_decode_calls=_codeact_successful_decode_calls(
            controller.model_actions, log.events
        ),
        provenance_path="provenance.jsonl",
    )
    _write_json(destination / "result.json", observation.model_dump(mode="json"))
    return observation
