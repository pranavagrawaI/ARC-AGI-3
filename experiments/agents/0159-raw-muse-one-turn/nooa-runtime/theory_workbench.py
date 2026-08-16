"""Deterministic WM2 theory workbench over E0 evidence and lane state."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

WORKBENCH_SCHEMA = "0159-theory-workbench-v1"
MAX_RENDER_BYTES = 6_144
MAX_RECENT_TRANSITIONS = 3
MAX_INTERPRETATION_CARDS = 3

ClaimKind = Literal[
    "whole_action_abstention", "no_cell_claim", "partial_cell_claim", "full_cell_claim"
]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TransitionEvaluationFact:
    source_event_id: str
    action: str
    claim: ClaimKind
    known_cells: int
    total_cells: int
    matched: bool | None


@dataclass(frozen=True)
class TheoryArtifactSnapshot:
    snapshot_id: str
    installed_evidence_snapshot_id: str
    current_model_version_id: str | None
    contract_kind: str | None
    evaluation_evidence_snapshot_id: str | None
    evaluation_stale: bool
    candidate_status: str | None
    operational_status: str | None
    prediction_coverage: float | None
    transition_results: tuple[TransitionEvaluationFact, ...]
    first_divergence_event_id: str | None
    latest_edit_status: str | None

    @classmethod
    def create(cls, **fields: Any) -> TheoryArtifactSnapshot:
        """Create a content-committed immutable projection."""

        transition_results = tuple(fields.pop("transition_results"))
        logical = {
            **fields,
            "transition_results": [asdict(item) for item in transition_results],
        }
        return cls(
            snapshot_id="theory-snapshot-" + _digest(logical),
            transition_results=transition_results,
            **fields,
        )


def theory_snapshot_from_lane_view(
    lane_view: dict[str, Any], *, latest_edit_status: str | None = None
) -> TheoryArtifactSnapshot:
    """Freeze the narrow lane projection consumed by the workbench builder."""

    evaluation = lane_view.get("latest_evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
    dynamics = evaluation.get("dynamics")
    if not isinstance(dynamics, dict):
        dynamics = {}
    raw_results = dynamics.get("transition_results", [])
    if not isinstance(raw_results, list):
        raise TypeError("lane transition_results must be a list")
    transition_results = tuple(
        TransitionEvaluationFact(
            source_event_id=str(item["source_event_id"]),
            action=str(item["action"]),
            claim=item["claim"],
            known_cells=int(item["known_cells"]),
            total_cells=int(item["total_cells"]),
            matched=item.get("matched"),
        )
        for item in raw_results
    )
    operational = evaluation.get("operational")
    operational_status = (
        str(operational.get("status")) if isinstance(operational, dict) else None
    )
    divergence = evaluation.get("first_divergence")
    divergence_event = (
        str(divergence.get("source_event_id"))
        if isinstance(divergence, dict) and divergence.get("source_event_id")
        else None
    )
    coverage = dynamics.get("prediction_coverage")
    return TheoryArtifactSnapshot.create(
        installed_evidence_snapshot_id=str(
            lane_view["installed_evidence_snapshot_id"]
        ),
        current_model_version_id=lane_view.get("current_model_version_id"),
        contract_kind=lane_view.get("contract_kind"),
        evaluation_evidence_snapshot_id=lane_view.get(
            "evaluation_evidence_snapshot_id"
        ),
        evaluation_stale=bool(lane_view.get("evaluation_stale")),
        candidate_status=evaluation.get("candidate_status"),
        operational_status=operational_status,
        prediction_coverage=float(coverage) if coverage is not None else None,
        transition_results=transition_results,
        first_divergence_event_id=divergence_event,
        latest_edit_status=latest_edit_status,
    )


@dataclass(frozen=True)
class WorkbenchTransitionFact:
    source_event_id: str
    action: str
    boundary: str
    changed_cell_count: int
    changed_bounds: tuple[int, int, int, int] | None
    claim: ClaimKind | None
    evaluation_matched: bool | None
    evaluation_reference: str | None
    divergence_reference: str | None

    def render(self) -> str:
        change = "no changed cells"
        if self.changed_bounds is not None:
            left, top, right, bottom = self.changed_bounds
            change = (
                f"{self.changed_cell_count} changed cells; "
                f"bounds x=[{left},{right}), y=[{top},{bottom})"
            )
        claim = "not evaluated by the active theory"
        if self.claim == "whole_action_abstention":
            claim = "predict(...) returned None"
        elif self.claim == "no_cell_claim":
            claim = "the prediction claimed no cells"
        elif self.claim == "partial_cell_claim":
            claim = "the prediction claimed some cells"
        elif self.claim == "full_cell_claim":
            claim = "the prediction claimed every cell"
        evaluation = "no replay result"
        if self.evaluation_matched is True:
            evaluation = "replay matched"
        elif self.evaluation_matched is False:
            evaluation = "replay diverged"
        references = tuple(
            reference
            for reference in (
                self.evaluation_reference,
                self.divergence_reference,
            )
            if reference is not None
        )
        if references:
            evaluation += "; reference " + ", ".join(references)
        return (
            f"- {self.source_event_id}: {self.action}; boundary {self.boundary}; "
            f"{change}; {claim}; {evaluation}"
        )


@dataclass(frozen=True)
class InterpretationCard:
    code: str
    what_happened: str
    mechanical_meaning: str
    does_not_imply: str
    possible_responses: tuple[str, ...]
    evidence_snapshot_id: str
    theory_snapshot_id: str
    evidence_event_ids: tuple[str, ...]

    def render(self) -> str:
        lines = [
            self.what_happened,
            f"Meaning: {self.mechanical_meaning}",
            f"It does not imply: {self.does_not_imply}",
            "Possible responses:",
            *(f"- {response}" for response in self.possible_responses),
            "Nothing here requires an edit.",
        ]
        if self.evidence_event_ids:
            lines.append("Evidence: " + ", ".join(self.evidence_event_ids))
        return "\n".join(lines)


@dataclass(frozen=True)
class TheoryWorkbenchView:
    schema: str
    view_digest: str
    evidence_snapshot_id: str
    theory_snapshot_id: str
    current_model_version_id: str | None
    contract_kind: str | None
    evaluation_evidence_snapshot_id: str | None
    evaluation_stale: bool
    candidate_status: str | None
    operational_status: str | None
    prediction_coverage: float | None
    claimed_cell_count: int
    claimable_cell_count: int
    whole_action_abstention_count: int
    ordinary_transition_count: int
    unevaluated_transition_count: int
    action_counts: tuple[tuple[str, int], ...]
    recent_transitions: tuple[WorkbenchTransitionFact, ...]
    condition_codes: tuple[str, ...]
    cards: tuple[InterpretationCard, ...]
    omitted_card_count: int
    omitted_card_codes: tuple[str, ...]

    def render(self) -> str:
        model = "none"
        if self.current_model_version_id is not None:
            model = self.current_model_version_id
            if self.contract_kind:
                model += f" ({self.contract_kind})"
        actions = (
            ", ".join(f"{action}: {count}" for action, count in self.action_counts)
            if self.action_counts
            else "none"
        )
        lines = [
            "## Executable theory workbench",
            f"Installed evidence frontier: {self.evidence_snapshot_id}",
            f"Active commitment: {model}",
            f"Ordinary transitions in the current evidence segment: {self.ordinary_transition_count}",
            f"Ordinary transitions outside the current evaluation: {self.unevaluated_transition_count}",
            f"Observed ordinary actions: {actions}",
        ]
        if self.current_model_version_id is not None:
            lines.extend(
                [
                    f"Evaluation snapshot: {self.evaluation_evidence_snapshot_id or 'none'}",
                    "Latest candidate status: "
                    + _candidate_status_text(self.candidate_status),
                    f"Prediction coverage: {_coverage_text(self.prediction_coverage)}",
                    (
                        "Claimed cells in evaluated current-segment transitions: "
                        f"{self.claimed_cell_count} of {self.claimable_cell_count}"
                    ),
                    (
                        "Whole-action abstentions in evaluated current-segment "
                        f"transitions: {self.whole_action_abstention_count}"
                    ),
                ]
            )
        if self.recent_transitions:
            lines.extend(
                [
                    "",
                    "Recent ordinary transition facts:",
                    *(item.render() for item in self.recent_transitions),
                ]
            )
        if self.cards:
            lines.extend(["", "Current interpretations of the mechanical status:"])
            for card in self.cards:
                lines.extend(["", card.render()])
        if self.omitted_card_count:
            lines.extend(
                [
                    "",
                    f"{self.omitted_card_count} additional mechanical condition(s) are retained in the audit.",
                ]
            )
        rendered = "\n".join(lines)
        if len(rendered.encode("utf-8")) > MAX_RENDER_BYTES:
            raise ValueError("theory workbench rendering exceeds its frozen byte bound")
        return rendered


def _coverage_text(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value:.4f}; greater coverage is not necessarily better"


def _candidate_status_text(value: str | None) -> str:
    return {
        None: "none",
        "active_supported": "Operationally valid; replay supported its claimed behavior so far.",
        "active_inconclusive": "Operationally valid; replay was inconclusive for its claimed behavior.",
        "active_contradicted": "Operationally valid; replay contradicted at least one claimed behavior.",
        "rejected_operationally": "Operationally rejected; the prior active version was preserved.",
    }.get(value, "Unknown status; inspect the structured lane result.")


class TheoryWorkbenchViewBuilder:
    """Build one immutable, read-only workbench projection."""

    def __init__(self, *, read_grid: Callable[[str], tuple[tuple[int, ...], ...]]):
        self._read_grid = read_grid

    def build(
        self, evidence: Any, theory: TheoryArtifactSnapshot
    ) -> TheoryWorkbenchView:
        records = json.loads(evidence.occurrence_records_json)
        current_attempt = _latest_attempt(records)
        ordinary = tuple(
            transition
            for transition in evidence.transitions
            if transition.boundary == "none"
            and transition.outcome == "ongoing"
            and transition.attempt_before == current_attempt
            and transition.attempt_after == current_attempt
        )
        evaluations = {
            item.source_event_id: item for item in theory.transition_results
        }
        current_event_ids = {item.source_event_id for item in ordinary}
        current_transition_results = tuple(
            item
            for item in theory.transition_results
            if item.source_event_id in current_event_ids
        )
        recent = tuple(
            self._transition_fact(
                transition,
                evaluations.get(transition.source_event_id),
                evaluation_reference=theory.evaluation_evidence_snapshot_id,
                divergence_reference=(
                    transition.source_event_id
                    if transition.source_event_id
                    == theory.first_divergence_event_id
                    else None
                ),
            )
            for transition in ordinary[-MAX_RECENT_TRANSITIONS:]
        )
        counts = Counter(transition.action.label for transition in ordinary)
        action_counts = tuple(sorted(counts.items()))
        evaluated_ids = set(evaluations)
        unevaluated = sum(
            transition.source_event_id not in evaluated_ids for transition in ordinary
        )
        claimed_cells = sum(item.known_cells for item in current_transition_results)
        claimable_cells = sum(item.total_cells for item in current_transition_results)
        whole_action_abstentions = sum(
            item.claim == "whole_action_abstention"
            for item in current_transition_results
        )
        latest_nonordinary = _latest_nonordinary(evidence.transitions, records)
        condition_codes = _condition_codes(
            ordinary=ordinary,
            counts=counts,
            theory=theory,
            transition_results=current_transition_results,
            latest_nonordinary=latest_nonordinary,
        )
        all_cards = tuple(
            _card(
                code,
                evidence_snapshot_id=evidence.snapshot_id,
                theory=theory,
                transition_results=current_transition_results,
                ordinary=ordinary,
                counts=counts,
                latest_nonordinary=latest_nonordinary,
            )
            for code in sorted(condition_codes, key=_card_priority)
        )
        cards = all_cards[:MAX_INTERPRETATION_CARDS]
        logical = {
            "schema": WORKBENCH_SCHEMA,
            "evidence_snapshot_id": evidence.snapshot_id,
            "theory_snapshot_id": theory.snapshot_id,
            "current_model_version_id": theory.current_model_version_id,
            "contract_kind": theory.contract_kind,
            "evaluation_evidence_snapshot_id": theory.evaluation_evidence_snapshot_id,
            "evaluation_stale": theory.evaluation_stale,
            "candidate_status": theory.candidate_status,
            "operational_status": theory.operational_status,
            "prediction_coverage": theory.prediction_coverage,
            "claimed_cell_count": claimed_cells,
            "claimable_cell_count": claimable_cells,
            "whole_action_abstention_count": whole_action_abstentions,
            "ordinary_transition_count": len(ordinary),
            "unevaluated_transition_count": unevaluated,
            "action_counts": action_counts,
            "recent_transitions": [asdict(item) for item in recent],
            "condition_codes": condition_codes,
            "cards": [asdict(item) for item in cards],
            "omitted_card_count": len(all_cards) - len(cards),
            "omitted_card_codes": tuple(card.code for card in all_cards[len(cards) :]),
        }
        view = TheoryWorkbenchView(
            schema=WORKBENCH_SCHEMA,
            view_digest=_digest(logical),
            evidence_snapshot_id=evidence.snapshot_id,
            theory_snapshot_id=theory.snapshot_id,
            current_model_version_id=theory.current_model_version_id,
            contract_kind=theory.contract_kind,
            evaluation_evidence_snapshot_id=theory.evaluation_evidence_snapshot_id,
            evaluation_stale=theory.evaluation_stale,
            candidate_status=theory.candidate_status,
            operational_status=theory.operational_status,
            prediction_coverage=theory.prediction_coverage,
            claimed_cell_count=claimed_cells,
            claimable_cell_count=claimable_cells,
            whole_action_abstention_count=whole_action_abstentions,
            ordinary_transition_count=len(ordinary),
            unevaluated_transition_count=unevaluated,
            action_counts=action_counts,
            recent_transitions=recent,
            condition_codes=condition_codes,
            cards=cards,
            omitted_card_count=len(all_cards) - len(cards),
            omitted_card_codes=tuple(card.code for card in all_cards[len(cards) :]),
        )
        view.render()
        return view

    def _transition_fact(
        self,
        transition: Any,
        evaluation: TransitionEvaluationFact | None,
        *,
        evaluation_reference: str | None,
        divergence_reference: str | None,
    ) -> WorkbenchTransitionFact:
        before = self._read_grid(transition.before_grid_sha256)
        after = self._read_grid(transition.after_grid_sha256)
        changed = [
            (x, y)
            for y, (before_row, after_row) in enumerate(zip(before, after, strict=True))
            for x, (before_cell, after_cell) in enumerate(
                zip(before_row, after_row, strict=True)
            )
            if before_cell != after_cell
        ]
        bounds = None
        if changed:
            xs, ys = zip(*changed, strict=True)
            bounds = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
        return WorkbenchTransitionFact(
            source_event_id=transition.source_event_id,
            action=transition.action.label,
            boundary=transition.boundary,
            changed_cell_count=len(changed),
            changed_bounds=bounds,
            claim=evaluation.claim if evaluation else None,
            evaluation_matched=evaluation.matched if evaluation else None,
            evaluation_reference=(evaluation_reference if evaluation else None),
            divergence_reference=divergence_reference,
        )


def _latest_attempt(records: list[dict[str, Any]]) -> str:
    for record in reversed(records):
        observation = record.get("after_observation") or record.get("observation")
        if isinstance(observation, dict) and isinstance(observation.get("attempt_id"), str):
            return observation["attempt_id"]
    raise ValueError("E0 snapshot contains no current observation attempt")


def _latest_nonordinary(
    transitions: tuple[Any, ...], records: list[dict[str, Any]]
) -> Any | str | None:
    if records and records[-1].get("type") in {
        "initial_reset",
        "host_reset",
        "decision_observation",
    }:
        return str(records[-1]["type"])
    if not transitions:
        return None
    latest = transitions[-1]
    if latest.boundary != "none" or latest.outcome != "ongoing":
        return latest
    return None


def _condition_codes(
    *,
    ordinary: tuple[Any, ...],
    counts: Counter[str],
    theory: TheoryArtifactSnapshot,
    transition_results: tuple[TransitionEvaluationFact, ...],
    latest_nonordinary: Any | str | None,
) -> tuple[str, ...]:
    codes: list[str] = []
    if not ordinary:
        codes.append("no_transition_evidence")
    if ordinary and theory.current_model_version_id is None:
        codes.append("evidence_without_commitment")
    if any(count >= 2 for count in counts.values()):
        codes.append("repeated_action_evidence")
    if theory.current_model_version_id is not None and theory.evaluation_stale:
        codes.append("new_evidence_since_evaluation")
    if any(item.claim == "whole_action_abstention" for item in transition_results):
        codes.append("whole_action_abstention")
    if any(item.claim == "partial_cell_claim" for item in transition_results):
        codes.append("partial_cell_claim")
    if theory.candidate_status == "active_contradicted":
        codes.append("empirical_contradiction")
    if theory.latest_edit_status == "rejected_operationally":
        codes.append("operational_rejection")
    if latest_nonordinary is not None:
        codes.append("nonordinary_boundary")
    return tuple(codes)


_CARD_PRIORITY = {
    "operational_rejection": 0,
    "empirical_contradiction": 0,
    "nonordinary_boundary": 1,
    "whole_action_abstention": 2,
    "partial_cell_claim": 2,
    "new_evidence_since_evaluation": 3,
    "evidence_without_commitment": 4,
    "repeated_action_evidence": 4,
    "no_transition_evidence": 5,
}


def _card_priority(code: str) -> tuple[int, str]:
    return (_CARD_PRIORITY[code], code)


def _card(
    code: str,
    *,
    evidence_snapshot_id: str,
    theory: TheoryArtifactSnapshot,
    transition_results: tuple[TransitionEvaluationFact, ...],
    ordinary: tuple[Any, ...],
    counts: Counter[str],
    latest_nonordinary: Any | str | None,
) -> InterpretationCard:
    event_ids: tuple[str, ...] = ()
    responses = (
        "retain the current state if no executable commitment is useful now",
        "keep unresolved or competing explanations in durable prose notes",
    )
    if code == "no_transition_evidence":
        happened = "No ordinary action result is available yet in this evidence segment."
        meaning = "there is no observed successor transition here to replay"
        nonimplication = "an executable theory is impossible or will never be useful"
    elif code == "evidence_without_commitment":
        event_ids = tuple(item.source_event_id for item in ordinary)
        happened = (
            f"{len(ordinary)} ordinary action result(s) are available, and no executable "
            "commitment is active."
        )
        meaning = "public transition evidence exists outside executable memory"
        nonimplication = "those transitions contain one repeatable or decision-relevant rule"
        responses += ("encode only a narrow rule you can presently defend",)
    elif code == "repeated_action_evidence":
        action, count = next(
            (action, count) for action, count in sorted(counts.items()) if count >= 2
        )
        event_ids = tuple(
            item.source_event_id for item in ordinary if item.action.label == action
        )
        happened = f"{action} appears in {count} ordinary transitions."
        meaning = "multiple public consequences of the same action label are available to inspect"
        nonimplication = "the transitions follow the same rule"
        responses += ("compare the cited transitions before deciding whether to model them",)
    elif code == "new_evidence_since_evaluation":
        happened = "New public evidence exists beyond the active model's evaluation frontier."
        meaning = "the saved evaluation does not include every currently installed transition"
        nonimplication = "the active theory is contradicted or must be revised"
        responses += ("read or evaluate the theory if the newer evidence matters now",)
    elif code == "whole_action_abstention":
        fact = next(
            item
            for item in transition_results
            if item.claim == "whole_action_abstention"
        )
        event_ids = (fact.source_event_id,)
        happened = (
            f"On transition {fact.source_event_id}, predict(...) returned None for {fact.action}."
        )
        meaning = "no cells were predicted, so this observation neither supported nor contradicted the theory"
        nonimplication = f"{fact.action} is unpredictable or the theory is wrong"
        responses += ("retain the gap or extend the theory if you can defend a prediction",)
    elif code == "partial_cell_claim":
        fact = next(
            item for item in transition_results if item.claim == "partial_cell_claim"
        )
        event_ids = (fact.source_event_id,)
        happened = (
            f"The prediction for {fact.source_event_id} claimed {fact.known_cells} of "
            f"{fact.total_cells} cells."
        )
        meaning = "only claimed cells participated in empirical scoring"
        nonimplication = "greater coverage would make the theory more useful or more correct"
        responses += ("retain deliberate partial coverage or revise it for your own reasons",)
    elif code == "empirical_contradiction":
        if theory.first_divergence_event_id:
            event_ids = (theory.first_divergence_event_id,)
        happened = "At least one claimed prediction differs from installed public evidence."
        meaning = "the operationally valid source remains active and has a recorded divergence"
        nonimplication = "which hypothesis, representation, or source function should change"
        responses += ("inspect the first divergence and decide whether another edit is warranted",)
    elif code == "operational_rejection":
        happened = "The latest model-authored edit failed an operational check and did not activate."
        meaning = "the previous active version, if any, was preserved"
        nonimplication = "the edit's game hypothesis was empirically false"
        responses += ("inspect the structured failure and submit another explicit edit if desired",)
    elif code == "nonordinary_boundary":
        event_ids = (
            (latest_nonordinary.source_event_id,)
            if not isinstance(latest_nonordinary, str)
            else ()
        )
        happened = "The latest evidence marks a reset, terminal, or level boundary."
        meaning = "evidence before and after this boundary must not be replayed as one ordinary transition segment"
        nonimplication = "any game mechanic, objective, or next action"
    else:  # pragma: no cover - closed registry makes this an integrity failure
        raise ValueError(f"unsupported opportunity condition {code!r}")
    return InterpretationCard(
        code=code,
        what_happened=happened,
        mechanical_meaning=meaning,
        does_not_imply=nonimplication,
        possible_responses=responses,
        evidence_snapshot_id=evidence_snapshot_id,
        theory_snapshot_id=theory.snapshot_id,
        evidence_event_ids=event_ids,
    )
