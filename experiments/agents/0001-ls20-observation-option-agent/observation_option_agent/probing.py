"""Generic experiment proposals for learned contact/effect relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .transition_learning import (
    ContactSignature,
    RelationHypothesis,
    contact_effect_symbols_from_record,
    contact_signature_from_record,
)


@dataclass(frozen=True)
class ProbeProposal:
    contact_signature: ContactSignature
    predicted_effect: str
    support: int
    effect_rate: float
    lift: float


@dataclass(frozen=True)
class InterventionResult:
    proposal: ProbeProposal
    positive_attempts: int
    positive_successes: int
    negative_controls: int
    negative_control_violations: int

    @property
    def reproduced(self) -> bool:
        return self.positive_attempts > 0 and self.positive_successes == self.positive_attempts

    @property
    def negative_control_passed(self) -> bool:
        return (
            self.negative_controls > 0
            and self.negative_control_violations == 0
        )


class ProbePolicy:
    """Select neutral relation hypotheses worth actively testing."""

    def __init__(
        self,
        *,
        minimum_support: int = 3,
        minimum_lift: float = 1.5,
    ) -> None:
        self.minimum_support = minimum_support
        self.minimum_lift = minimum_lift

    def propose(
        self,
        hypotheses: Iterable[RelationHypothesis],
    ) -> tuple[ProbeProposal, ...]:
        proposals = []
        for hypothesis in hypotheses:
            if hypothesis.support < self.minimum_support:
                continue
            for effect, rate, lift in hypothesis.top_effects:
                if lift < self.minimum_lift:
                    continue
                proposals.append(ProbeProposal(
                    contact_signature=hypothesis.contact_signature,
                    predicted_effect=effect,
                    support=hypothesis.support,
                    effect_rate=rate,
                    lift=lift,
                ))
        proposals.sort(
            key=lambda item: (item.lift, item.effect_rate, item.support),
            reverse=True,
        )
        return tuple(proposals)


class ReplayInterventionAssay:
    """Use held-out observations as a proxy falsifier for proposed probes.

    This is not a live intervention. It estimates whether a proposed relation
    reproduces on later matching contacts and stays absent on non-matching
    contacts, using only neutral event records.
    """

    def evaluate(
        self,
        proposal: ProbeProposal,
        records: Iterable[dict[str, object]],
    ) -> InterventionResult:
        positive_attempts = 0
        positive_successes = 0
        negative_controls = 0
        negative_control_violations = 0
        for record in records:
            contact_signature = contact_signature_from_record(record)
            if not contact_signature:
                continue
            effects = contact_effect_symbols_from_record(record)
            if contact_signature == proposal.contact_signature:
                positive_attempts += 1
                if proposal.predicted_effect in effects:
                    positive_successes += 1
                continue
            negative_controls += 1
            if proposal.predicted_effect in effects:
                negative_control_violations += 1

        return InterventionResult(
            proposal=proposal,
            positive_attempts=positive_attempts,
            positive_successes=positive_successes,
            negative_controls=negative_controls,
            negative_control_violations=negative_control_violations,
        )
