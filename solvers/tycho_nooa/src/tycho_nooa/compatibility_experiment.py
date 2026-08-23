"""Matched execution of one protocol-compatibility eligibility gate."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Literal

from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, ConfigDict, Field

from tycho_nooa.cognitive_parity import RetrievalAnswer
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.native_parity import NativeParityController
from tycho_nooa.parity_compatibility import (
    CompatibilityEvaluation,
    CompatibilityEvidenceCapability,
    CompatibilityGate,
    evaluate_compatibility_answer,
)
from tycho_nooa.sterile_codeact import SterileCodeActController

CompatibilityArm = Literal["native", "codeact"]

NATIVE_COMPATIBILITY_PROMPT = (
    "You are completing a protocol-compatibility eligibility gate, not a reasoning "
    "benchmark. Follow the visible evidence contract exactly when it is enabled. "
    "Use only the offered tools and finish with one typed return_result."
)
CODEACT_COMPATIBILITY_PROMPT = (
    "You are completing a protocol-compatibility eligibility gate, not a reasoning "
    "benchmark. Use only execute_python(code) and return_result(result). Python "
    "locals persist across cells. evidence.query exists only when the visible "
    "contract enables it; use that exact query shape. Finish with the typed result."
)


class CompatibilityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    evidence_queries: int = Field(ge=0)
    query_attempts: int = Field(default=0, ge=0)
    python_cells: int = Field(ge=0)
    invalid_interactions: int = Field(ge=0)
    transport_adherence: dict[str, int] = Field(default_factory=dict)
    wall_seconds: float = Field(ge=0)


class CompatibilityArmObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    gate_id: str
    arm: CompatibilityArm
    passed: bool
    answer: dict[str, Any] | None = None
    evaluation: CompatibilityEvaluation | None = None
    failure: str | None = None
    metrics: CompatibilityMetrics
    provenance_path: str | None = None


class CompatibilityGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    gate_id: str
    gate_kind: str
    passed: bool
    native: CompatibilityArmObservation
    codeact: CompatibilityArmObservation


async def execute_compatibility_gate(
    gate: CompatibilityGate,
    *,
    repository_root: Path,
    destination: Path,
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
    max_llm_calls: int = 3,
    max_output_tokens: int = 4096,
    session_timeout_seconds: float = 300.0,
    cell_timeout_seconds: float = 5.0,
) -> CompatibilityGateReport:
    """Run both arms even when one fails its eligibility gate."""
    destination.mkdir(parents=True, exist_ok=False)
    try:
        async with asyncio.timeout(session_timeout_seconds):
            native = await _run_native(
                gate,
                repository_root=repository_root,
                destination=destination / "native",
                llm=native_llm,
                max_llm_calls=max_llm_calls,
                max_output_tokens=max_output_tokens,
            )
        async with asyncio.timeout(session_timeout_seconds):
            codeact = await _run_codeact(
                gate,
                repository_root=repository_root,
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
    report = CompatibilityGateReport(
        gate_id=gate.task.gate_id,
        gate_kind=gate.task.gate_kind,
        passed=native.passed and codeact.passed,
        native=native,
        codeact=codeact,
    )
    _write_json(destination / "report.json", report.model_dump(mode="json"))
    return report


async def _run_native(
    gate: CompatibilityGate,
    *,
    repository_root: Path,
    destination: Path,
    llm: UnifiedLLM,
    max_llm_calls: int,
    max_output_tokens: int,
) -> CompatibilityArmObservation:
    destination.mkdir()
    evidence = CompatibilityEvidenceCapability.from_gate(
        gate, repository_root=repository_root
    )
    controller = NativeParityController(
        llm=llm,
        packet=gate,
        evidence=evidence,
        max_llm_calls=max_llm_calls,
        max_output_tokens=max_output_tokens,
        system_prompt=NATIVE_COMPATIBILITY_PROMPT,
    )
    started = time.monotonic()
    answer, failure = await _answer_or_exhaustion(controller.run())
    metrics = _metrics(
        usage=[turn.usage for turn in controller.provider_turns],
        evidence_queries=controller.evidence_queries,
        query_attempts=(0 if evidence is None else evidence.query_attempts),
        python_cells=0,
        invalid_interactions=controller.invalid_returns,
        transport_adherence={},
        wall_seconds=time.monotonic() - started,
    )
    observation = _observation(gate, "native", answer, failure, metrics)
    _write_json(destination / "result.json", observation.model_dump(mode="json"))
    return observation


async def _run_codeact(
    gate: CompatibilityGate,
    *,
    repository_root: Path,
    destination: Path,
    llm: UnifiedLLM,
    max_llm_calls: int,
    max_output_tokens: int,
    cell_timeout_seconds: float,
) -> CompatibilityArmObservation:
    destination.mkdir()
    evidence = CompatibilityEvidenceCapability.from_gate(
        gate, repository_root=repository_root
    )
    provenance_path = destination / "provenance.jsonl"
    log = ExecutionProvenanceLog(
        provenance_path,
        actor_session_id=f"compatibility:{gate.task.gate_id}:codeact",
    )
    controller = SterileCodeActController(
        llm=llm,
        packet=gate,
        evidence=evidence,
        provenance=InvocationProvenance(
            log, invocation_id=f"{gate.task.gate_id}:codeact:1"
        ),
        max_llm_calls=max_llm_calls,
        max_output_tokens=max_output_tokens,
        cell_timeout_seconds=cell_timeout_seconds,
        system_prompt=CODEACT_COMPATIBILITY_PROMPT,
    )
    started = time.monotonic()
    answer, failure = await _answer_or_exhaustion(controller.run())
    counts = Counter(event.event_type for event in log.events)
    evidence_queries = counts["CapabilityReturned"]
    python_cells = sum(
        "execute_python" in turn.tool_names for turn in controller.provider_turns
    )
    metrics = _metrics(
        usage=[turn.usage for turn in controller.provider_turns],
        evidence_queries=evidence_queries,
        query_attempts=(0 if evidence is None else evidence.query_attempts),
        python_cells=python_cells,
        invalid_interactions=(
            counts["ReturnRejected"]
            + counts["ProviderTurnRejected"]
            + counts["CapabilityRejected"]
            + counts["CellFailed"]
        ),
        transport_adherence=controller.transport_adherence,
        wall_seconds=time.monotonic() - started,
    )
    observation = _observation(
        gate,
        "codeact",
        answer,
        failure,
        metrics,
        provenance_path="provenance.jsonl",
    )
    _write_json(destination / "result.json", observation.model_dump(mode="json"))
    return observation


async def _answer_or_exhaustion(
    awaitable: Awaitable[object],
) -> tuple[RetrievalAnswer | None, str | None]:
    try:
        answer = await awaitable
    except RuntimeError as error:
        if "exhausted its LLM-call budget" not in str(error):
            raise
        return None, str(error)
    if not isinstance(answer, RetrievalAnswer):
        raise TypeError("compatibility gate requires a retrieval answer")
    return answer, None


def _observation(
    gate: CompatibilityGate,
    arm: CompatibilityArm,
    answer: RetrievalAnswer | None,
    failure: str | None,
    metrics: CompatibilityMetrics,
    *,
    provenance_path: str | None = None,
) -> CompatibilityArmObservation:
    evaluation = (
        None
        if answer is None
        else evaluate_compatibility_answer(
            gate, answer, evidence_queries=metrics.evidence_queries
        )
    )
    passed = evaluation.passed if evaluation is not None else False
    return CompatibilityArmObservation(
        gate_id=gate.task.gate_id,
        arm=arm,
        passed=passed,
        answer=None if answer is None else answer.model_dump(mode="json"),
        evaluation=evaluation,
        failure=failure,
        metrics=metrics,
        provenance_path=provenance_path,
    )


def _metrics(
    *,
    usage: list[dict[str, int] | None],
    evidence_queries: int,
    query_attempts: int,
    python_cells: int,
    invalid_interactions: int,
    transport_adherence: dict[str, int],
    wall_seconds: float,
) -> CompatibilityMetrics:
    input_tokens = sum(_usage_value(row, "prompt_tokens", "input_tokens") for row in usage)
    output_tokens = sum(
        _usage_value(row, "completion_tokens", "output_tokens") for row in usage
    )
    return CompatibilityMetrics(
        llm_calls=len(usage),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        evidence_queries=evidence_queries,
        query_attempts=query_attempts,
        python_cells=python_cells,
        invalid_interactions=invalid_interactions,
        transport_adherence=transport_adherence,
        wall_seconds=wall_seconds,
    )


def _usage_value(usage: dict[str, int] | None, *names: str) -> int:
    if not usage:
        return 0
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
    return 0


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
