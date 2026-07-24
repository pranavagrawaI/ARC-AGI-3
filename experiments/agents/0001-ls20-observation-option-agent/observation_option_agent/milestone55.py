"""Evidence probe for contact/event option predicates.

Milestone 5.5 is intentionally a probe, not a predicate upgrade.  It asks
whether the existing public trace/event artifacts contain honest level-3 or
level-4 termination evidence for a trace-induced option.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal

from .milestone4 import OptionCandidate, predicate_expression_hash
from .transition_learning import (
    contact_effect_symbols_from_record,
    contact_signature_from_record,
)


ProbeStatus = Literal[
    "available_event_predicate_evidence",
    "partial_ambiguous_evidence",
    "no_clean_event_predicate_evidence",
]
ProbeRole = Literal["termination", "guard", "rejected"]
EventOptionExecutionStatus = Literal["COMPLETED", "REFUSED"]


@dataclass(frozen=True)
class EventPredicateProbeCandidate:
    predicate_id: str
    option_id: str
    kind: str
    abstraction_level: int
    mode: Literal["state", "transition"]
    role: ProbeRole
    expression: dict[str, Any]
    evidence: tuple[dict[str, Any], ...]
    true_at_initiation: bool
    true_at_terminal: bool
    earlier_true_rate: float
    terminal_selective: bool
    validates_m5_route_context: bool
    rejection_reason: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "option_id": self.option_id,
            "kind": self.kind,
            "abstraction_level": self.abstraction_level,
            "mode": self.mode,
            "role": self.role,
            "expression": self.expression,
            "expression_hash": predicate_expression_hash(self.expression),
            "evidence": list(self.evidence),
            "true_at_initiation": self.true_at_initiation,
            "true_at_terminal": self.true_at_terminal,
            "earlier_true_rate": self.earlier_true_rate,
            "terminal_selective": self.terminal_selective,
            "validates_m5_route_context": self.validates_m5_route_context,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class EventOptionExecutionResult:
    status: EventOptionExecutionStatus
    completed: bool
    termination_predicate_satisfied: bool
    raw_trace_actions_masked: bool
    termination_by_event_predicate: bool
    termination_by_trace_index: bool
    semantic_guard_used: bool
    failure_reason: str | None
    records: tuple[dict[str, Any], ...]


class EventPredicateEvidenceProbe:
    """Scan public event records for higher-level option termination evidence."""

    def probe(
        self,
        *,
        option: OptionCandidate,
        event_records: tuple[dict[str, Any], ...],
    ) -> tuple[EventPredicateProbeCandidate, ...]:
        start, end = option.segment_range
        by_step = {
            int(record.get("step", -1)): record
            for record in event_records
        }
        terminal = by_step.get(end)
        if terminal is None:
            return (
                _missing_terminal_record_candidate(
                    option=option,
                    end=end,
                ),
            )
        initiation = by_step.get(start)
        earlier = tuple(
            by_step[index]
            for index in range(start + 1, end)
            if index in by_step
        )

        candidates: list[EventPredicateProbeCandidate] = []
        contact = self._contact_candidate(
            option=option,
            terminal=terminal,
            initiation=initiation,
            earlier=earlier,
        )
        if contact is not None:
            candidates.append(contact)
        candidates.extend(
            self._semantic_delta_candidates(
                option=option,
                terminal=terminal,
                initiation=initiation,
                earlier=earlier,
            )
        )
        candidates.append(
            self._semantic_guard_candidate(
                option=option,
                terminal=terminal,
            )
        )
        return tuple(candidates)

    def _contact_candidate(
        self,
        *,
        option: OptionCandidate,
        terminal: dict[str, Any],
        initiation: dict[str, Any] | None,
        earlier: tuple[dict[str, Any], ...],
    ) -> EventPredicateProbeCandidate | None:
        signature = contact_signature_from_record(terminal)
        blocked = bool((terminal.get("interaction") or {}).get("blocked"))
        if not signature and not blocked:
            return None
        expression = {
            "type": "contact_or_boundary_event",
            "contact_signature": [
                [value, area, shape_id]
                for value, area, shape_id in signature
            ],
            "blocked": blocked,
            "forbidden_form": "trace_index == segment_end",
        }
        return self._candidate(
            option=option,
            predicate_id=f"{option.option_id}:probe_contact_or_boundary",
            kind="contact_or_boundary_event",
            abstraction_level=3,
            expression=expression,
            evidence=(
                {
                    "kind": "terminal_contact_or_boundary",
                    "step": terminal.get("step"),
                    "field_path": "interaction.contact_candidates|interaction.blocked",
                    "contact_signature": expression["contact_signature"],
                    "blocked": blocked,
                },
            ),
            terminal=terminal,
            initiation=initiation,
            earlier=earlier,
            symbol=event_symbol(expression),
        )

    def _semantic_delta_candidates(
        self,
        *,
        option: OptionCandidate,
        terminal: dict[str, Any],
        initiation: dict[str, Any] | None,
        earlier: tuple[dict[str, Any], ...],
    ) -> tuple[EventPredicateProbeCandidate, ...]:
        candidates = []
        for symbol in sorted(_semantic_delta_symbols(terminal)):
            expression = {
                "type": "semantic_delta_event",
                "symbol": symbol,
                **_semantic_delta_expression_details(symbol, terminal),
                "forbidden_form": "trace_index == segment_end",
            }
            candidates.append(self._candidate(
                option=option,
                predicate_id=f"{option.option_id}:probe_semantic_delta:{_safe_symbol(symbol)}",
                kind="semantic_delta_event",
                abstraction_level=4,
                expression=expression,
                evidence=(
                    {
                        "kind": "terminal_semantic_delta",
                        "step": terminal.get("step"),
                        "field_path": _field_path_for_symbol(symbol),
                        "symbol": symbol,
                    },
                ),
                terminal=terminal,
                initiation=initiation,
                earlier=earlier,
                symbol=symbol,
            ))
        return tuple(candidates)

    def _semantic_guard_candidate(
        self,
        *,
        option: OptionCandidate,
        terminal: dict[str, Any],
    ) -> EventPredicateProbeCandidate:
        expression = {
            "type": "semantic_signature_compatible",
            "role": "guard_not_termination",
            "forbidden_form": "trace_index == segment_end",
        }
        return EventPredicateProbeCandidate(
            predicate_id=f"{option.option_id}:probe_semantic_guard",
            option_id=option.option_id,
            kind="semantic_signature_compatibility",
            abstraction_level=4,
            mode="state",
            role="guard",
            expression=expression,
            evidence=(
                {
                    "kind": "semantic_guard_context",
                    "step": terminal.get("step"),
                    "field_path": "option.expected_semantic_delta",
                },
            ),
            true_at_initiation=True,
            true_at_terminal=True,
            earlier_true_rate=1.0,
            terminal_selective=False,
            validates_m5_route_context=True,
            rejection_reason="classified_as_guard_not_termination",
        )

    def _candidate(
        self,
        *,
        option: OptionCandidate,
        predicate_id: str,
        kind: str,
        abstraction_level: int,
        expression: dict[str, Any],
        evidence: tuple[dict[str, Any], ...],
        terminal: dict[str, Any],
        initiation: dict[str, Any] | None,
        earlier: tuple[dict[str, Any], ...],
        symbol: str,
    ) -> EventPredicateProbeCandidate:
        true_at_initiation = _record_has_symbol(initiation, symbol) if initiation else False
        true_at_terminal = _record_has_symbol(terminal, symbol)
        earlier_true = sum(_record_has_symbol(record, symbol) for record in earlier)
        earlier_rate = earlier_true / len(earlier) if earlier else 0.0
        terminal_selective = (
            not true_at_initiation
            and true_at_terminal
            and earlier_rate < 0.5
        )
        rejection = _probe_rejection(
            expression=expression,
            evidence=evidence,
            true_at_initiation=true_at_initiation,
            true_at_terminal=true_at_terminal,
            terminal_selective=terminal_selective,
            validates_m5_route_context=True,
        )
        return EventPredicateProbeCandidate(
            predicate_id=predicate_id,
            option_id=option.option_id,
            kind=kind,
            abstraction_level=abstraction_level,
            mode="transition",
            role="termination" if rejection is None else "rejected",
            expression=expression,
            evidence=evidence,
            true_at_initiation=true_at_initiation,
            true_at_terminal=true_at_terminal,
            earlier_true_rate=round(earlier_rate, 4),
            terminal_selective=terminal_selective,
            validates_m5_route_context=True,
            rejection_reason=rejection,
        )


class EventOptionExecutor:
    """Validate event-terminated options without raw segment-action access."""

    def execute(
        self,
        *,
        option: OptionCandidate,
        predicate: EventPredicateProbeCandidate,
        transition_record: dict[str, Any],
        semantic_guard_valid: bool = True,
    ) -> EventOptionExecutionResult:
        if predicate.mode != "transition" or predicate.role != "termination":
            return self._result(
                option=option,
                predicate=predicate,
                transition_record=transition_record,
                semantic_guard_valid=semantic_guard_valid,
                satisfied=False,
                failure_reason="PREDICATE_NOT_EXECUTABLE_TERMINATION",
            )
        if not semantic_guard_valid:
            return self._result(
                option=option,
                predicate=predicate,
                transition_record=transition_record,
                semantic_guard_valid=semantic_guard_valid,
                satisfied=False,
                failure_reason="SEMANTIC_GUARD_FAILED",
            )
        satisfied = _record_has_symbol(
            transition_record,
            event_symbol(predicate.expression),
        )
        return self._result(
            option=option,
            predicate=predicate,
            transition_record=transition_record,
            semantic_guard_valid=semantic_guard_valid,
            satisfied=satisfied,
            failure_reason=None if satisfied else "TERMINATION_EVENT_ABSENT",
        )

    def _result(
        self,
        *,
        option: OptionCandidate,
        predicate: EventPredicateProbeCandidate,
        transition_record: dict[str, Any],
        semantic_guard_valid: bool,
        satisfied: bool,
        failure_reason: str | None,
    ) -> EventOptionExecutionResult:
        record = _execution_record(
            option=option,
            predicate=predicate,
            transition_record=transition_record,
            semantic_guard_valid=semantic_guard_valid,
            satisfied=satisfied,
            failure_reason=failure_reason,
        )
        return EventOptionExecutionResult(
            status="COMPLETED" if satisfied else "REFUSED",
            completed=satisfied,
            termination_predicate_satisfied=satisfied,
            raw_trace_actions_masked=True,
            termination_by_event_predicate=satisfied,
            termination_by_trace_index=False,
            semantic_guard_used=True,
            failure_reason=failure_reason,
            records=(record,),
        )


def summarize_probe(
    candidates: tuple[EventPredicateProbeCandidate, ...],
) -> dict[str, Any]:
    termination_candidates = tuple(
        candidate for candidate in candidates if candidate.role == "termination"
    )
    rejected_event_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.role == "rejected" and candidate.abstraction_level >= 3
    )
    if termination_candidates:
        status: ProbeStatus = "available_event_predicate_evidence"
    elif rejected_event_candidates:
        status = "partial_ambiguous_evidence"
    else:
        status = "no_clean_event_predicate_evidence"
    return {
        "status": status,
        "candidate_count": len(candidates),
        "termination_candidate_count": len(termination_candidates),
        "rejected_event_candidate_count": len(rejected_event_candidates),
        "guard_candidate_count": sum(candidate.role == "guard" for candidate in candidates),
        "available_predicate_ids": [
            candidate.predicate_id for candidate in termination_candidates
        ],
        "max_available_abstraction_level": max(
            (candidate.abstraction_level for candidate in termination_candidates),
            default=0,
        ),
        "max_observed_abstraction_level": max(
            (candidate.abstraction_level for candidate in candidates),
            default=0,
        ),
    }


def event_symbol(expression: dict[str, Any]) -> str:
    kind = expression.get("type")
    if kind == "contact_or_boundary_event":
        signature = tuple(
            tuple(item)
            for item in expression.get("contact_signature", ())
        )
        return "contact_or_boundary:" + json.dumps({
            "blocked": bool(expression.get("blocked")),
            "signature": signature,
        }, sort_keys=True, separators=(",", ":"))
    if kind == "semantic_delta_event":
        return str(expression.get("symbol", ""))
    return str(kind)


def _record_has_symbol(record: dict[str, Any] | None, symbol: str) -> bool:
    if record is None:
        return False
    if symbol.startswith("contact_or_boundary:"):
        payload = json.loads(symbol.split(":", 1)[1])
        signature = [
            list(item)
            for item in contact_signature_from_record(record)
        ]
        blocked = bool((record.get("interaction") or {}).get("blocked"))
        return payload == {"blocked": blocked, "signature": signature}
    return symbol in _semantic_delta_symbols(record)


def _semantic_delta_symbols(record: dict[str, Any]) -> frozenset[str]:
    symbols: set[str] = set()
    level_delta = int(record.get("levels_after", 0)) - int(record.get("levels_before", 0))
    if level_delta:
        symbols.add(f"level_delta:{level_delta:+d}")
    if record.get("full_reset"):
        symbols.add("full_reset")

    interaction = record.get("interaction") or {}
    for symbol in contact_effect_symbols_from_record(record):
        symbols.add(symbol)
    if record.get("created_object_ids") or interaction.get("created_other_object_ids"):
        symbols.add("created_other_object")
    if record.get("disappeared_object_ids") or interaction.get("disappeared_other_object_ids"):
        symbols.add("disappeared_other_object")

    # Generic object-delta noise is deliberately not promoted here.  The trace
    # contains continuous animation residue during normal movement; treating it
    # as semantic termination would create fake abstraction.
    return frozenset(symbols)


def _semantic_delta_expression_details(
    symbol: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    if symbol == "created_other_object":
        return {
            "event_type": "non_controlled_object_created",
            "created_object_signatures": _created_object_signatures(record),
            "before_condition": "object_absent",
            "after_condition": "object_present",
            "excluded_noise_classes": [
                "animation_residue",
                "controlled_entity_artifact",
                "generic_shape_area_delta",
            ],
        }
    if symbol == "disappeared_other_object":
        return {
            "event_type": "non_controlled_object_disappeared",
            "before_condition": "object_present",
            "after_condition": "object_absent",
            "excluded_noise_classes": [
                "animation_residue",
                "controlled_entity_artifact",
                "generic_shape_area_delta",
            ],
        }
    if symbol.startswith("level_delta"):
        return {"event_type": "public_level_delta"}
    return {"event_type": symbol}


def _created_object_signatures(record: dict[str, Any]) -> list[dict[str, Any]]:
    created_ids = set(record.get("created_object_ids") or ())
    interaction = record.get("interaction") or {}
    created_ids.update(interaction.get("created_other_object_ids") or ())
    signatures = []
    for delta in record.get("object_deltas", ()):
        object_id = delta.get("object_id")
        if object_id not in created_ids:
            continue
        signatures.append({
            "area": delta.get("area_after"),
            "bbox": delta.get("bbox_after"),
            "value": delta.get("value_after"),
        })
    if signatures:
        return signatures
    return [{"object_id": object_id} for object_id in sorted(created_ids)]


def _execution_record(
    *,
    option: OptionCandidate,
    predicate: EventPredicateProbeCandidate,
    transition_record: dict[str, Any],
    semantic_guard_valid: bool,
    satisfied: bool,
    failure_reason: str | None,
) -> dict[str, Any]:
    created_signatures = _created_object_signatures(transition_record)
    return {
        "phase": "OPTION_EXECUTION",
        "option_id": option.option_id,
        "action_source": "OPTION_PLANNER",
        "used_raw_trace_action": False,
        "selected_predicate_id": predicate.predicate_id,
        "selected_predicate_kind": predicate.kind,
        "selected_predicate_abstraction_level": predicate.abstraction_level,
        "predicate_mode": predicate.mode,
        "event_type": predicate.expression.get("event_type"),
        "semantic_guard_valid": semantic_guard_valid,
        "semantic_guard_used": True,
        "before": {
            "non_controlled_object_signature_present": False,
        },
        "after": {
            "non_controlled_object_signature_present": bool(created_signatures),
            "created_object_signatures": created_signatures,
        },
        "semantic_delta_event": {
            "type": predicate.expression.get("event_type"),
            "symbol": predicate.expression.get("symbol"),
            "validated": satisfied,
        },
        "event_predicate_satisfied": satisfied,
        "termination_by_event_predicate": satisfied,
        "termination_by_trace_index": False,
        "stop_reason": (
            "TERMINATION_PREDICATE_SATISFIED"
            if satisfied
            else failure_reason
        ),
        "rejected_noise": [],
    }


def _field_path_for_symbol(symbol: str) -> str:
    if symbol.startswith("level_delta"):
        return "levels_before|levels_after"
    if symbol == "full_reset":
        return "full_reset"
    if symbol == "created_other_object":
        return "interaction.created_other_object_ids"
    if symbol == "disappeared_other_object":
        return "interaction.disappeared_other_object_ids"
    if symbol.startswith("contact_") or symbol.startswith("effect_"):
        return "interaction.contact_candidates|object_deltas"
    return "event_record"


def _probe_rejection(
    *,
    expression: dict[str, Any],
    evidence: tuple[dict[str, Any], ...],
    true_at_initiation: bool,
    true_at_terminal: bool,
    terminal_selective: bool,
    validates_m5_route_context: bool,
) -> str | None:
    if not evidence:
        return "missing_evidence_provenance"
    if _is_index_based(expression):
        return "trace_index_based"
    if _is_raw_action_based(expression):
        return "raw_action_based"
    if true_at_initiation:
        return "true_at_initiation"
    if not true_at_terminal:
        return "not_true_at_terminal"
    if not validates_m5_route_context:
        return "does_not_validate_m5_route_context"
    if not terminal_selective:
        return "not_terminal_selective_or_overbroad"
    return None


def _is_index_based(expression: dict[str, Any]) -> bool:
    if expression.get("type") == "trace_index":
        return True
    return any(
        key not in {"forbidden_form"}
        and ("trace_index" in key or "segment_end" in key)
        for key in expression
    )


def _is_raw_action_based(expression: dict[str, Any]) -> bool:
    encoded = json.dumps(expression, sort_keys=True)
    return any(
        marker in encoded
        for marker in (
            "raw_action",
            "human_action",
            "next_action",
            "expected_action",
            "action_at_index",
        )
    )


def _safe_symbol(symbol: str) -> str:
    return (
        symbol
        .replace(":", "_")
        .replace("+", "plus")
        .replace("-", "minus")
        .replace("/", "_")
        .replace("|", "_")
    )


def _missing_terminal_record_candidate(
    *,
    option: OptionCandidate,
    end: int,
) -> EventPredicateProbeCandidate:
    return EventPredicateProbeCandidate(
        predicate_id=f"{option.option_id}:probe_missing_terminal_record",
        option_id=option.option_id,
        kind="missing_terminal_record",
        abstraction_level=0,
        mode="transition",
        role="rejected",
        expression={"type": "missing_terminal_record"},
        evidence=({"kind": "missing_terminal_record", "step": end},),
        true_at_initiation=False,
        true_at_terminal=False,
        earlier_true_rate=0.0,
        terminal_selective=False,
        validates_m5_route_context=False,
        rejection_reason="missing_terminal_record",
    )
