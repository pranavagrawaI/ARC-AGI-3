"""Public factor counterexample assay for 0001 M12.R1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .milestone12r0 import (
    PublicFactorTransition,
    factor_transition_outcome_symbol,
    forbidden_factor_discovery_findings,
)


M12R1Status = Literal[
    "complete_with_known_limits",
    "partial_no_reused_factor_contexts",
    "partial_no_counterexamples_found",
    "partial_no_stable_contexts",
    "failed_contradicted_context_promoted",
    "failed_no_counterexample_surface",
    "failed_named_object_ontology",
    "failed_oracle_or_private_state_used",
]


@dataclass(frozen=True)
class FactorContextEvidence:
    context_id: str
    factor_state_key: str
    action: int
    support: int
    outcomes: tuple[tuple[str, int], ...]
    transition_outcomes: tuple[tuple[str, str], ...]
    majority_outcome: str
    status: Literal["stable", "contradicted", "unsupported"]
    promoted: bool
    counterexample_count: int

    def to_record(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "factor_state_key": self.factor_state_key,
            "action": self.action,
            "support": self.support,
            "outcomes": [
                {"outcome": outcome, "count": count}
                for outcome, count in self.outcomes
            ],
            "transition_outcomes": [
                {"transition_id": transition_id, "outcome": outcome}
                for transition_id, outcome in self.transition_outcomes
            ],
            "majority_outcome": self.majority_outcome,
            "status": self.status,
            "promoted": self.promoted,
            "counterexample_count": self.counterexample_count,
        }


@dataclass(frozen=True)
class FactorCounterexampleAudit:
    transition_count: int
    context_count: int
    reused_context_count: int
    stable_context_count: int
    contradicted_context_count: int
    unsupported_context_count: int
    promoted_context_count: int
    blocked_promotion_count: int
    counterexample_count: int
    no_op_contradiction_count: int
    contexts: tuple[FactorContextEvidence, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "transition_count": self.transition_count,
            "context_count": self.context_count,
            "reused_context_count": self.reused_context_count,
            "stable_context_count": self.stable_context_count,
            "contradicted_context_count": self.contradicted_context_count,
            "unsupported_context_count": self.unsupported_context_count,
            "promoted_context_count": self.promoted_context_count,
            "blocked_promotion_count": self.blocked_promotion_count,
            "counterexample_count": self.counterexample_count,
            "no_op_contradiction_count": self.no_op_contradiction_count,
            "contexts": [context.to_record() for context in self.contexts],
        }


class FactorCounterexampleAssay:
    def __init__(self, *, minimum_support: int = 2) -> None:
        self.minimum_support = minimum_support

    def evaluate(
        self,
        transitions: tuple[PublicFactorTransition, ...],
    ) -> FactorCounterexampleAudit:
        grouped: dict[tuple[str, int], list[tuple[str, str]]] = {}
        for transition in transitions:
            context = (transition.factor_state_key, transition.action)
            grouped.setdefault(context, []).append((
                transition.transition_id,
                factor_transition_outcome_symbol(transition),
            ))

        contexts = []
        for index, ((factor_state_key, action), records) in enumerate(
            sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
        ):
            outcome_counts = _outcome_counts(records)
            majority_outcome, majority_count = _majority(outcome_counts)
            support = len(records)
            if support < self.minimum_support:
                status: Literal["stable", "contradicted", "unsupported"] = "unsupported"
            elif len(outcome_counts) == 1:
                status = "stable"
            else:
                status = "contradicted"
            contexts.append(FactorContextEvidence(
                context_id=f"factor_context_{index:03d}",
                factor_state_key=factor_state_key,
                action=action,
                support=support,
                outcomes=tuple(sorted(
                    outcome_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )),
                transition_outcomes=tuple(records),
                majority_outcome=majority_outcome,
                status=status,
                promoted=status == "stable",
                counterexample_count=(
                    support - majority_count if status == "contradicted" else 0
                ),
            ))

        return FactorCounterexampleAudit(
            transition_count=len(transitions),
            context_count=len(contexts),
            reused_context_count=sum(
                1 for context in contexts if context.support >= self.minimum_support
            ),
            stable_context_count=sum(
                1 for context in contexts if context.status == "stable"
            ),
            contradicted_context_count=sum(
                1 for context in contexts if context.status == "contradicted"
            ),
            unsupported_context_count=sum(
                1 for context in contexts if context.status == "unsupported"
            ),
            promoted_context_count=sum(1 for context in contexts if context.promoted),
            blocked_promotion_count=sum(
                1 for context in contexts if context.status == "contradicted"
            ),
            counterexample_count=sum(context.counterexample_count for context in contexts),
            no_op_contradiction_count=sum(
                1
                for context in contexts
                if context.status == "contradicted"
                and any(outcome == "no_op" for outcome, _ in context.outcomes)
            ),
            contexts=tuple(contexts),
        )


def m12r1_status(
    *,
    reused_context_count: int,
    contradicted_context_count: int,
    stable_context_count: int,
    promoted_context_count: int,
    blocked_promotion_count: int,
    named_object_ontology: bool,
    oracle_or_private_state_used: bool,
) -> M12R1Status:
    if oracle_or_private_state_used:
        return "failed_oracle_or_private_state_used"
    if named_object_ontology:
        return "failed_named_object_ontology"
    if reused_context_count == 0:
        return "partial_no_reused_factor_contexts"
    if contradicted_context_count == 0:
        return "partial_no_counterexamples_found"
    if blocked_promotion_count < contradicted_context_count:
        return "failed_contradicted_context_promoted"
    if promoted_context_count > stable_context_count:
        return "failed_contradicted_context_promoted"
    if stable_context_count == 0:
        return "partial_no_stable_contexts"
    return "complete_with_known_limits"


def forbidden_factor_counterexample_findings(payload: dict[str, Any]) -> tuple[str, ...]:
    return forbidden_factor_discovery_findings(payload)


def _outcome_counts(records: list[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, outcome in records:
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def _majority(counts: dict[str, int]) -> tuple[str, int]:
    if not counts:
        return "unknown", 0
    return max(counts.items(), key=lambda item: (item[1], item[0]))
