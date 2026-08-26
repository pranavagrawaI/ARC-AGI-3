"""Zero-ARC matched Native/CodeAct replay over one frozen Builder case."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, ConfigDict

from tycho_nooa.builder_proposal_adapters import (
    BuilderCodeActController,
    BuilderNativeController,
)
from tycho_nooa.builder_proposal_protocol import (
    FrozenBuildCase,
    JsonValue,
    ProposalEvaluator,
    ProposalLedger,
    canonical_json,
    object_sha256,
)
from tycho_nooa.execution_events import (
    ExecutionProvenanceLog,
    InvocationProvenance,
)
from tycho_nooa.theory_evaluation import PublicationDisposition, compare


class BuilderReplayMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_calls: int
    input_tokens: int
    output_tokens: int
    python_calls: int
    evaluation_attempts: int
    evaluated_candidates: int
    protected_gain_candidates: int
    candidate_yield: str
    first_protected_gain_provider_call: int | None
    first_pareto_safe_provider_call: int | None
    total_wall_seconds: str
    evaluator_wall_seconds: str
    model_cost: str | None
    evaluator_cost: str | None
    total_cost_to_first_pareto_safe: str | None


class BuilderArmObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: Literal["native", "codeact"]
    case_sha256: str
    report_sha256: str
    nominated_candidate_source_sha256: str | None
    semantic_outcomes: tuple[str, ...]
    metrics: BuilderReplayMetrics


class BuilderMatchedReplayReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    case_sha256: str
    provider_settings_sha256: str
    native: BuilderArmObservation
    codeact: BuilderArmObservation


async def execute_zero_arc_builder_replay(
    *,
    case: FrozenBuildCase,
    evaluator_factory: Callable[[], ProposalEvaluator],
    provider_settings: dict[str, JsonValue],
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
    destination: Path,
    max_llm_calls: int = 8,
    max_output_tokens: int | None = None,
    cell_timeout_seconds: float = 10.0,
) -> BuilderMatchedReplayReport:
    """Run both execution modes without transaction finish or ARC authority."""
    destination.mkdir(parents=True, exist_ok=False)
    provider_settings_sha256 = object_sha256(
        "BuilderProviderSettings", provider_settings
    )
    (destination / "provider-settings.json").write_bytes(
        canonical_json(provider_settings) + b"\n"
    )
    (destination / "native").mkdir()
    (destination / "codeact").mkdir()
    native_evaluator = evaluator_factory()
    codeact_evaluator = evaluator_factory()
    if native_evaluator is codeact_evaluator:
        raise ValueError("matched replay requires isolated evaluator instances")
    if (
        native_evaluator.evaluator_sha256 != case.evaluator_sha256
        or codeact_evaluator.evaluator_sha256 != case.evaluator_sha256
    ):
        raise ValueError("matched replay evaluator commitment differs from the case")
    native_ledger = ProposalLedger(
        case, native_evaluator, destination / "native" / "ledger"
    )
    codeact_ledger = ProposalLedger(
        case, codeact_evaluator, destination / "codeact" / "ledger"
    )
    native = BuilderNativeController(
        llm=native_llm,
        case=case,
        ledger=native_ledger,
        max_llm_calls=max_llm_calls,
        max_output_tokens=max_output_tokens,
        cell_timeout_seconds=cell_timeout_seconds,
    )
    native_started = time.monotonic()
    native_report = await native.run()
    native_wall_seconds = time.monotonic() - native_started

    provenance_log = ExecutionProvenanceLog(
        destination / "codeact" / "provenance.jsonl",
        actor_session_id=f"builder-replay:{case.case_sha256}",
    )
    codeact = BuilderCodeActController(
        llm=codeact_llm,
        case=case,
        ledger=codeact_ledger,
        provenance=InvocationProvenance(
            provenance_log, invocation_id="builder-codeact-replay"
        ),
        max_llm_calls=max_llm_calls,
        max_output_tokens=max_output_tokens,
        cell_timeout_seconds=cell_timeout_seconds,
    )
    codeact_started = time.monotonic()
    codeact_report = await codeact.run()
    codeact_wall_seconds = time.monotonic() - codeact_started

    native_observation = _observation(
        arm="native",
        case=case,
        report=native_report,
        ledger=native_ledger,
        usage=native.provider_usage,
        python_calls=native.python_calls,
        receipt_provider_calls=native.receipt_provider_calls,
        total_wall_seconds=native_wall_seconds,
    )
    codeact_usage = [turn.usage for turn in codeact.provider_turns]
    codeact_observation = _observation(
        arm="codeact",
        case=case,
        report=codeact_report,
        ledger=codeact_ledger,
        usage=codeact_usage,
        python_calls=sum(
            "execute_python" in turn.tool_names for turn in codeact.provider_turns
        ),
        receipt_provider_calls=_codeact_receipt_calls(
            codeact.provider_turns, provenance_log.events
        ),
        total_wall_seconds=codeact_wall_seconds,
    )
    result = BuilderMatchedReplayReport(
        case_sha256=case.case_sha256,
        provider_settings_sha256=provider_settings_sha256,
        native=native_observation,
        codeact=codeact_observation,
    )
    (destination / "report.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _observation(
    *,
    arm: Literal["native", "codeact"],
    case: FrozenBuildCase,
    report: Any,
    ledger: ProposalLedger,
    usage: list[dict[str, Any] | None],
    python_calls: int,
    receipt_provider_calls: list[int],
    total_wall_seconds: float,
) -> BuilderArmObservation:
    eligible = ledger.eligible_receipts()
    eligible_ids = {id(receipt) for receipt in eligible}
    comparisons = [
        compare(case.incumbent_evaluation, receipt.semantic.evaluation)
        for receipt in ledger.receipts
    ]
    first_protected_index = next(
        (
            index
            for index, (receipt, decision) in enumerate(
                zip(ledger.receipts, comparisons, strict=True)
            )
            if id(receipt) in eligible_ids and decision.gained
        ),
        None,
    )
    first_receipt_index = next(
        (
            index
            for index, (receipt, decision) in enumerate(
                zip(ledger.receipts, comparisons, strict=True)
            )
            if id(receipt) in eligible_ids
            and decision.disposition is PublicationDisposition.PUBLISHABLE
        ),
        None,
    )
    first_provider_call = (
        None
        if first_receipt_index is None
        or first_receipt_index >= len(receipt_provider_calls)
        else receipt_provider_calls[first_receipt_index]
    )
    first_protected_call = (
        None
        if first_protected_index is None
        or first_protected_index >= len(receipt_provider_calls)
        else receipt_provider_calls[first_protected_index]
    )
    model_cost_known = all(
        row is not None and any(name in row for name in ("cost", "cost_usd"))
        for row in usage
    )
    model_cost = sum((_usage_decimal(row, "cost", "cost_usd") for row in usage), Decimal())
    evaluator_cost_known = hasattr(ledger.evaluator, "last_cost")
    evaluator_cost = sum(
        (Decimal(receipt.evaluator_cost) for receipt in ledger.receipts), Decimal()
    )
    cost_to_first = None
    if first_provider_call is not None and model_cost_known and evaluator_cost_known:
        prefix_model_cost = sum(
            (
                _usage_decimal(row, "cost", "cost_usd")
                for row in usage[:first_provider_call]
            ),
            Decimal(),
        )
        prefix_evaluator_cost = sum(
            (
                Decimal(receipt.evaluator_cost)
                for receipt in ledger.receipts[: first_receipt_index + 1]
            ),
            Decimal(),
        )
        cost_to_first = _decimal(prefix_model_cost + prefix_evaluator_cost)
    return BuilderArmObservation(
        arm=arm,
        case_sha256=case.case_sha256,
        report_sha256=report.report_sha256,
        nominated_candidate_source_sha256=(
            report.nominated_candidate_source_sha256
        ),
        semantic_outcomes=tuple(
            receipt.semantic.obligation_outcome_sha256
            for receipt in ledger.eligible_receipts()
        ),
        metrics=BuilderReplayMetrics(
            provider_calls=len(usage),
            input_tokens=sum(_usage_int(row, "prompt_tokens", "input_tokens") for row in usage),
            output_tokens=sum(
                _usage_int(row, "completion_tokens", "output_tokens") for row in usage
            ),
            python_calls=python_calls,
            evaluation_attempts=len(ledger.receipts),
            evaluated_candidates=len(eligible),
            protected_gain_candidates=sum(
                bool(decision.gained)
                for receipt, decision in zip(
                    ledger.receipts, comparisons, strict=True
                )
                if id(receipt) in eligible_ids
            ),
            candidate_yield=_decimal(
                Decimal(
                    sum(
                        decision.disposition is PublicationDisposition.PUBLISHABLE
                        for receipt, decision in zip(
                            ledger.receipts, comparisons, strict=True
                        )
                        if id(receipt) in eligible_ids
                    )
                )
                / Decimal(len(eligible) or 1)
            ),
            first_protected_gain_provider_call=first_protected_call,
            first_pareto_safe_provider_call=first_provider_call,
            total_wall_seconds=_decimal(Decimal(str(total_wall_seconds))),
            evaluator_wall_seconds=_decimal(
                sum(
                    (Decimal(receipt.evaluation_wall_seconds) for receipt in ledger.receipts),
                    Decimal(),
                )
            ),
            model_cost=_decimal(model_cost) if model_cost_known else None,
            evaluator_cost=(
                _decimal(evaluator_cost) if evaluator_cost_known else None
            ),
            total_cost_to_first_pareto_safe=cost_to_first,
        ),
    )


def _codeact_receipt_calls(provider_turns: tuple[Any, ...], events: tuple[Any, ...]) -> list[int]:
    boundaries = [turn.event_seq for turn in provider_turns]
    calls: list[int] = []
    for event_index, event in enumerate(events):
        if event.event_type != "CapabilityReturned":
            continue
        call = sum(boundary <= event_index for boundary in boundaries)
        calls.append(max(1, call))
    return calls


def _usage_int(row: dict[str, Any] | None, *names: str) -> int:
    if row is None:
        return 0
    return next((int(row[name]) for name in names if name in row), 0)


def _usage_decimal(row: dict[str, Any] | None, *names: str) -> Decimal:
    if row is None:
        return Decimal()
    return next((Decimal(str(row[name])) for name in names if name in row), Decimal())


def _decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"
