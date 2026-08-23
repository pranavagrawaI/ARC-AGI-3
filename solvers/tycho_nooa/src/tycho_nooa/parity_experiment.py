"""Matched, mutation-free execution of one frozen cognitive-parity packet."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, ConfigDict, Field

from tycho_nooa.cognitive_parity import (
    CognitiveAnswer,
    CognitiveParityPacket,
    ParityEvaluation,
    evaluate_answer,
)
from tycho_nooa.evidence import WorkspaceEvidenceCapability
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.native_parity import NativeParityController
from tycho_nooa.sterile_codeact import SterileCodeActController

TreatmentArm = Literal["native", "codeact"]


class TreatmentMetrics(BaseModel):
    """Mechanism and cost observations, never a synthetic quality score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    evidence_queries: int = Field(ge=0)
    python_cells: int = Field(ge=0)
    invalid_returns: int = Field(ge=0)
    wall_seconds: float = Field(ge=0)


class TreatmentObservation(BaseModel):
    """One arm's raw typed answer, factored evaluation, and resource use."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    packet_id: str
    arm: TreatmentArm
    answer: dict[str, Any]
    evaluation: ParityEvaluation
    metrics: TreatmentMetrics
    provenance_path: str | None = None


class MatchedParityReport(BaseModel):
    """Two observations over one packet; action agreement remains descriptive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    packet_id: str
    native: TreatmentObservation
    codeact: TreatmentObservation


async def execute_matched_packet(
    packet: CognitiveParityPacket,
    *,
    repository_root: Path,
    destination: Path,
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
    max_llm_calls: int,
    max_output_tokens: int,
    session_timeout_seconds: float,
    cell_timeout_seconds: float,
) -> MatchedParityReport:
    """Run both austere arms once and preserve every completed observation."""
    destination.mkdir(parents=True, exist_ok=False)
    evidence_root = repository_root / packet.checkpoint_root
    try:
        async with asyncio.timeout(session_timeout_seconds):
            native = await _run_native(
                packet,
                evidence_root=evidence_root,
                destination=destination / "native",
                llm=native_llm,
                max_llm_calls=max_llm_calls,
                max_output_tokens=max_output_tokens,
            )
        async with asyncio.timeout(session_timeout_seconds):
            codeact = await _run_codeact(
                packet,
                evidence_root=evidence_root,
                destination=destination / "codeact",
                llm=codeact_llm,
                max_llm_calls=max_llm_calls,
                max_output_tokens=max_output_tokens,
                cell_timeout_seconds=cell_timeout_seconds,
            )
    except BaseException as error:
        _write_json(
            destination / "failure.json",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        raise
    report = MatchedParityReport(
        packet_id=packet.task.packet_id,
        native=native,
        codeact=codeact,
    )
    _write_json(destination / "report.json", report.model_dump(mode="json"))
    return report


async def _run_native(
    packet: CognitiveParityPacket,
    *,
    evidence_root: Path,
    destination: Path,
    llm: UnifiedLLM,
    max_llm_calls: int,
    max_output_tokens: int,
) -> TreatmentObservation:
    destination.mkdir()
    controller = NativeParityController(
        llm=llm,
        packet=packet,
        evidence=WorkspaceEvidenceCapability.from_root(evidence_root),
        max_llm_calls=max_llm_calls,
        max_output_tokens=max_output_tokens,
    )
    started = time.monotonic()
    answer = await controller.run()
    observation = _observation(
        packet,
        "native",
        answer,
        usage=[turn.usage for turn in controller.provider_turns],
        evidence_queries=controller.evidence_queries,
        python_cells=0,
        invalid_returns=controller.invalid_returns,
        wall_seconds=time.monotonic() - started,
    )
    _write_json(destination / "result.json", observation.model_dump(mode="json"))
    return observation


async def _run_codeact(
    packet: CognitiveParityPacket,
    *,
    evidence_root: Path,
    destination: Path,
    llm: UnifiedLLM,
    max_llm_calls: int,
    max_output_tokens: int,
    cell_timeout_seconds: float,
) -> TreatmentObservation:
    destination.mkdir()
    provenance_path = destination / "provenance.jsonl"
    log = ExecutionProvenanceLog(
        provenance_path,
        actor_session_id=f"parity:{packet.task.packet_id}:codeact",
    )
    controller = SterileCodeActController(
        llm=llm,
        packet=packet,
        evidence=WorkspaceEvidenceCapability.from_root(evidence_root),
        provenance=InvocationProvenance(
            log, invocation_id=f"{packet.task.packet_id}:codeact:1"
        ),
        max_llm_calls=max_llm_calls,
        max_output_tokens=max_output_tokens,
        cell_timeout_seconds=cell_timeout_seconds,
    )
    started = time.monotonic()
    answer = await controller.run()
    event_counts = Counter(event.event_type for event in log.events)
    evidence_queries = sum(
        event.event_type == "CapabilityRequested"
        and event.payload.get("name") == "evidence.query"
        for event in log.events
    )
    python_cells = sum(
        "execute_python" in turn.tool_names for turn in controller.provider_turns
    )
    observation = _observation(
        packet,
        "codeact",
        answer,
        usage=[turn.usage for turn in controller.provider_turns],
        evidence_queries=evidence_queries,
        python_cells=python_cells,
        invalid_returns=event_counts["ReturnRejected"],
        wall_seconds=time.monotonic() - started,
        provenance_path="provenance.jsonl",
    )
    _write_json(destination / "result.json", observation.model_dump(mode="json"))
    return observation


def _observation(
    packet: CognitiveParityPacket,
    arm: TreatmentArm,
    answer: CognitiveAnswer,
    *,
    usage: list[dict[str, int] | None],
    evidence_queries: int,
    python_cells: int,
    invalid_returns: int,
    wall_seconds: float,
    provenance_path: str | None = None,
) -> TreatmentObservation:
    answer_value = answer.model_dump(mode="json")
    tokens = _token_totals(usage)
    return TreatmentObservation(
        packet_id=packet.task.packet_id,
        arm=arm,
        answer=answer_value,
        evaluation=evaluate_answer(packet, answer_value),
        metrics=TreatmentMetrics(
            llm_calls=len(usage),
            input_tokens=tokens["input"],
            output_tokens=tokens["output"],
            evidence_queries=evidence_queries,
            python_cells=python_cells,
            invalid_returns=invalid_returns,
            wall_seconds=wall_seconds,
        ),
        provenance_path=provenance_path,
    )


def _token_totals(rows: list[dict[str, int] | None]) -> dict[str, int]:
    totals = {"input": 0, "output": 0}
    for usage in rows:
        if not usage:
            continue
        totals["input"] += _usage_value(usage, "prompt_tokens", "input_tokens")
        totals["output"] += _usage_value(usage, "completion_tokens", "output_tokens")
    return totals


def _usage_value(usage: dict[str, int], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
    return 0


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
