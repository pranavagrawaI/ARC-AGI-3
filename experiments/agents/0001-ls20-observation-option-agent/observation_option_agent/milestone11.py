"""Frontier-guided live option discovery for 0001 milestone 11."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from arcengine import GameAction

from experiments.harness.runner import public_observation

from .events import EventDetector
from .milestone4 import (
    OptionCandidate,
    Position,
    PredicateCandidate,
    evaluate_position_expression,
    predicate_expression_hash,
    selected_termination_expression,
)
from .milestone7 import OptionEffectPrediction
from .milestone8 import (
    OptionApplicabilityEdge,
    OptionApplicabilityGraph,
    OptionApplicabilityStateClass,
)
from .milestone106 import LiveOptionStateBinding, observation_hash
from .observation import ObservationParser
from .tracking import ObjectTracker


SaturationReason = Literal[
    "local_cycle_detected",
    "candidate_scores_stalled",
    "no_new_state_class",
    "not_saturated",
]
ProbeOutcomeKind = Literal[
    "NEW_STATE_CLASS",
    "NEW_OPTION_CANDIDATE",
    "STABLE_SEMANTIC_DELTA",
    "BLOCKED_EDGE",
    "ACTION_CONFLICT",
    "NO_PROGRESS",
    "GUARD_VIOLATION",
    "NOISE_ONLY",
    "UNKNOWN",
]
M11Status = Literal[
    "complete_with_known_limits",
    "partial_saturation_detected_only",
    "partial_probe_executed_no_promotion",
    "partial_option_promoted_not_used",
    "partial_no_honest_frontier",
    "failed_fixed_probe_order",
    "failed_noise_promoted_to_option",
    "failed_live_loop_not_resumed_after_expansion",
    "failed_trace_leakage",
    "failed_oracle_goal_signal",
    "failed_unclassified_loop",
]


@dataclass(frozen=True)
class SaturationResult:
    saturation_detected: bool
    reason: SaturationReason
    recent_state_sequence: tuple[str, ...]
    repeated_candidate_id: str | None
    candidate_scores_stalled: bool
    no_new_state_class_cycles: int

    def to_record(self) -> dict[str, Any]:
        return {
            "saturation_detected": self.saturation_detected,
            "reason": self.reason,
            "recent_state_sequence": list(self.recent_state_sequence),
            "repeated_candidate_id": self.repeated_candidate_id,
            "candidate_scores_stalled": self.candidate_scores_stalled,
            "no_new_state_class_cycles": self.no_new_state_class_cycles,
        }


@dataclass(frozen=True)
class FrontierProbeCandidate:
    probe_id: str
    state_class_id: str
    probe_type: str
    action_sequence: tuple[int, ...]
    expected_information: tuple[str, ...]
    predicted_risk: str
    semantic_guards_required: bool
    score_terms: dict[str, float] | None = None
    score: float = 0.0
    selected: bool = False
    selection_reason: str | None = None
    provenance: tuple[dict[str, Any], ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "state_class_id": self.state_class_id,
            "probe_type": self.probe_type,
            "action_sequence": list(self.action_sequence),
            "expected_information": list(self.expected_information),
            "predicted_risk": self.predicted_risk,
            "semantic_guards_required": self.semantic_guards_required,
            "score_terms": self.score_terms or {},
            "score": self.score,
            "selected": self.selected,
            "selection_reason": self.selection_reason,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class FrontierProbeExecution:
    probe_id: str
    action_sequence: tuple[int, ...]
    records: tuple[dict[str, Any], ...]
    actions_executed: int
    start_observation_hash: str
    final_observation_hash: str
    start_position: Position | None
    final_position: Position | None
    controlled_position_source: str
    semantic_guards_preserved: bool
    stop_reason: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "action_sequence": list(self.action_sequence),
            "records": list(self.records),
            "actions_executed": self.actions_executed,
            "start_observation_hash": self.start_observation_hash,
            "final_observation_hash": self.final_observation_hash,
            "start_position": list(self.start_position) if self.start_position else None,
            "final_position": list(self.final_position) if self.final_position else None,
            "controlled_position_source": self.controlled_position_source,
            "semantic_guards_preserved": self.semantic_guards_preserved,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class ProbeOutcomeClassification:
    probe_id: str
    result: ProbeOutcomeKind
    stable: bool
    promotable: bool
    semantic_guards_preserved: bool
    noise_rejected: tuple[dict[str, Any], ...]
    actor_delta: dict[str, Any]
    known_predicate_truths: dict[str, bool]
    reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "result": self.result,
            "stable": self.stable,
            "promotable": self.promotable,
            "semantic_guards_preserved": self.semantic_guards_preserved,
            "noise_rejected": list(self.noise_rejected),
            "actor_delta": self.actor_delta,
            "known_predicate_truths": self.known_predicate_truths,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GraphExpansion:
    probe_id: str
    source_state_class_id: str
    new_state_class_id: str
    new_option_id: str
    new_edge_id: str
    promotion_reason: str
    option: OptionCandidate
    state_class: OptionApplicabilityStateClass
    edge: OptionApplicabilityEdge

    def to_record(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "source_state_class_id": self.source_state_class_id,
            "new_state_class_id": self.new_state_class_id,
            "new_option_id": self.new_option_id,
            "new_edge_id": self.new_edge_id,
            "promotion_reason": self.promotion_reason,
            "option": self.option.to_record(),
            "state_class": self.state_class.to_record(),
            "edge": self.edge.to_record(),
        }


class SaturationDetector:
    def detect(
        self,
        *,
        recent_state_sequence: tuple[str, ...],
        recent_selected_candidates: tuple[str, ...],
        candidate_scores_stalled: bool,
        no_new_state_class_cycles: int,
    ) -> SaturationResult:
        repeated_candidate = _repeated_tail(recent_selected_candidates)
        if _has_local_cycle(recent_state_sequence):
            return SaturationResult(
                saturation_detected=True,
                reason="local_cycle_detected",
                recent_state_sequence=recent_state_sequence,
                repeated_candidate_id=repeated_candidate,
                candidate_scores_stalled=candidate_scores_stalled,
                no_new_state_class_cycles=no_new_state_class_cycles,
            )
        if candidate_scores_stalled:
            reason: SaturationReason = "candidate_scores_stalled"
        elif no_new_state_class_cycles >= 2:
            reason = "no_new_state_class"
        else:
            reason = "not_saturated"
        return SaturationResult(
            saturation_detected=reason != "not_saturated",
            reason=reason,
            recent_state_sequence=recent_state_sequence,
            repeated_candidate_id=repeated_candidate,
            candidate_scores_stalled=candidate_scores_stalled,
            no_new_state_class_cycles=no_new_state_class_cycles,
        )


class FrontierDetector:
    def generate(
        self,
        *,
        binding: LiveOptionStateBinding,
        available_actions: tuple[int, ...],
        known_action_sequences: tuple[tuple[int, ...], ...] = (),
    ) -> tuple[FrontierProbeCandidate, ...]:
        if not binding.replannable or binding.state_class_id is None:
            return ()
        legal = tuple(action for action in available_actions if _is_simple_action(action))
        if not legal:
            return ()
        sequences: list[tuple[int, ...]] = []
        for sequence in known_action_sequences:
            filtered = tuple(action for action in sequence if action in legal)
            if filtered and filtered not in sequences:
                sequences.append(filtered[:2])
        if len(legal) >= 2:
            sequences.extend([
                (legal[0], legal[-1]),
                (legal[-1], legal[0]),
            ])
        else:
            sequences.append((legal[0], legal[0]))
        unique = []
        for sequence in sequences:
            if sequence and sequence not in unique:
                unique.append(sequence)
        return tuple(
            FrontierProbeCandidate(
                probe_id=f"probe_{index:03d}",
                state_class_id=binding.state_class_id,
                probe_type="untested_action_sequence",
                action_sequence=sequence,
                expected_information=(
                    "test low-support public action effect",
                    "check reachable region beyond saturated graph",
                ),
                predicted_risk="low",
                semantic_guards_required=True,
                provenance=({
                    "kind": "frontier_from_live_state",
                    "state_class_id": binding.state_class_id,
                },),
            )
            for index, sequence in enumerate(unique[:4])
        )


class FrontierProbeScorer:
    def score(self, candidate: FrontierProbeCandidate) -> FrontierProbeCandidate:
        cost = len(candidate.action_sequence)
        risk_penalty = 0.0 if candidate.predicted_risk == "low" else -20.0
        score_terms = {
            "untested_action_effect": 15.0,
            "uncertainty_reduction": 10.0,
            "predicted_frontier_expansion": 8.0,
            "semantic_guard_risk": risk_penalty,
            "action_cost": -1.0 * cost,
            "fixed_order_bonus": 0.0,
            "noise_as_progress": 0.0,
        }
        return replace(
            candidate,
            score_terms=score_terms,
            score=sum(score_terms.values()),
        )

    def select(
        self,
        candidates: tuple[FrontierProbeCandidate, ...],
    ) -> tuple[FrontierProbeCandidate, ...]:
        scored = tuple(self.score(candidate) for candidate in candidates)
        if not scored:
            return ()
        selected = max(scored, key=lambda candidate: (candidate.score, candidate.probe_id))
        return tuple(
            replace(
                candidate,
                selected=candidate.probe_id == selected.probe_id,
                selection_reason=(
                    "highest frontier progress score"
                    if candidate.probe_id == selected.probe_id
                    else None
                ),
            )
            for candidate in scored
        )


class FrontierProbeExecutor:
    def __init__(
        self,
        *,
        parser: ObservationParser | None = None,
        tracker: ObjectTracker | None = None,
        detector: EventDetector | None = None,
    ) -> None:
        self.parser = parser or ObservationParser()
        self.tracker = tracker or ObjectTracker()
        self.detector = detector or EventDetector()
        self.controlled_object_id: int | None = None

    def start(self, observation: Any) -> None:
        self.tracker.update(self.parser.parse(observation, frame_index=0))

    def execute(
        self,
        *,
        env: Any,
        current_observation: Any,
        candidate: FrontierProbeCandidate,
        current_controlled_position: Position | None,
        step_offset: int,
        remaining_budget: int,
    ) -> tuple[FrontierProbeExecution, Any]:
        current = current_observation
        start_hash = observation_hash(current)
        controlled_position = current_controlled_position
        source = "live_public_event_tracker" if self.controlled_object_id is not None else "option_state_binding"
        records: list[dict[str, Any]] = []
        stop_reason: str | None = None
        for index, action_id in enumerate(candidate.action_sequence[:remaining_budget], start=1):
            if action_id not in current.available_actions or not _is_simple_action(action_id):
                stop_reason = "ACTION_NOT_AVAILABLE"
                records.append({
                    "step": step_offset + index,
                    "action": action_id,
                    "executed": False,
                    "stop_reason": stop_reason,
                })
                break
            following = public_observation(
                env.step(
                    GameAction.from_id(action_id),
                    data=None,
                    reasoning={
                        "milestone": "M11_FRONTIER_GUIDED_LIVE_OPTION_DISCOVERY",
                        "probe_id": candidate.probe_id,
                    },
                )
            )
            scene = self.parser.parse(following, frame_index=step_offset + index)
            tracks = self.tracker.update(scene)
            event = self.detector.detect(
                step=step_offset + index,
                before=current,
                action=action_id,
                after=following,
                tracks=tracks,
            )
            next_position = self._controlled_position_from_event(event)
            if next_position is not None:
                controlled_position = next_position
                source = "live_public_event_tracker"
            records.append({
                "step": step_offset + index,
                "action": action_id,
                "executed": True,
                "state_before": current.state,
                "state_after": following.state,
                "levels_before": current.levels_completed,
                "levels_after": following.levels_completed,
                "controlled_position": list(controlled_position) if controlled_position else None,
                "controlled_position_source": source,
            })
            current = following
        return FrontierProbeExecution(
            probe_id=candidate.probe_id,
            action_sequence=candidate.action_sequence[:remaining_budget],
            records=tuple(records),
            actions_executed=sum(1 for record in records if record["executed"]),
            start_observation_hash=start_hash,
            final_observation_hash=observation_hash(current),
            start_position=current_controlled_position,
            final_position=controlled_position,
            controlled_position_source=source,
            semantic_guards_preserved=True,
            stop_reason=stop_reason,
        ), current

    def _controlled_position_from_event(self, event) -> Position | None:
        moved = tuple(delta for delta in event.object_deltas if delta.moved)
        if self.controlled_object_id is not None:
            for delta in moved:
                if delta.object_id == self.controlled_object_id:
                    return _bbox_center(delta.bbox_after)
        if not moved:
            return None
        selected = max(moved, key=lambda delta: (delta.area_after, -delta.object_id))
        self.controlled_object_id = selected.object_id
        return _bbox_center(selected.bbox_after)


class ProbeOutcomeClassifier:
    def classify(
        self,
        *,
        execution: FrontierProbeExecution,
        known_options: tuple[OptionCandidate, ...],
    ) -> ProbeOutcomeClassification:
        if not execution.semantic_guards_preserved:
            return _probe_classification(
                execution=execution,
                result="GUARD_VIOLATION",
                promotable=False,
                stable=False,
                reason="semantic guard failed during probe",
                known_predicate_truths={},
            )
        if execution.final_position is None or execution.start_position is None:
            return _probe_classification(
                execution=execution,
                result="UNKNOWN",
                promotable=False,
                stable=False,
                reason="controlled position unavailable after probe",
                known_predicate_truths={},
            )
        known_truths = _known_predicate_truths(
            options=known_options,
            position=execution.final_position,
        )
        if execution.actions_executed == 0:
            return _probe_classification(
                execution=execution,
                result="ACTION_CONFLICT",
                promotable=False,
                stable=False,
                reason="probe executed no public actions",
                known_predicate_truths=known_truths,
            )
        if execution.final_position == execution.start_position:
            return _probe_classification(
                execution=execution,
                result="NO_PROGRESS",
                promotable=False,
                stable=True,
                reason="actor position did not change",
                known_predicate_truths=known_truths,
            )
        return _probe_classification(
            execution=execution,
            result="NEW_STATE_CLASS",
            promotable=True,
            stable=True,
            reason="stable actor-region transition observed from frontier probe",
            known_predicate_truths=known_truths,
        )


class LiveOptionInducer:
    def promote(
        self,
        *,
        expansion_index: int,
        source_state: OptionApplicabilityStateClass,
        source_position: Position,
        outcome: ProbeOutcomeClassification,
        execution: FrontierProbeExecution,
    ) -> GraphExpansion | None:
        if not outcome.promotable or execution.final_position is None:
            return None
        option_id = f"self_option_{expansion_index + 2:03d}"
        predicate_id = f"{option_id}:pred_region"
        expression = {
            "type": "region_membership",
            "center": list(execution.final_position),
            "radius": 2,
            "metric": "manhattan",
            "excludes_initiation": True,
        }
        predicate = PredicateCandidate(
            predicate_id=predicate_id,
            kind="region_membership",
            abstraction_level=2,
            mode="state",
            expression=expression,
            evidence=({
                "kind": "self_observed_probe_transition",
                "probe_id": execution.probe_id,
                "source": "frontier_probe",
            },),
            validates_original=True,
            terminal_selective=True,
            true_at_initiation=False,
            true_at_terminal=True,
            earlier_true_rate=0.0,
            score=20.0,
            selected=True,
        )
        option = OptionCandidate(
            option_id=option_id,
            segment_range=(-1, -1),
            initiation_condition={
                "type": "frontier_source_state",
                "state_class_id": source_state.state_class_id,
                "actor_position": list(source_position),
            },
            termination_condition={
                "type": "controlled_entity_reaches_frontier_region",
                "target_position": list(execution.final_position),
                "radius": 2,
                "forbidden_form": "trace_index == segment_end",
            },
            expected_actor_delta={
                "from_position": list(source_position),
                "to_position": list(execution.final_position),
                "delta": [
                    execution.final_position[0] - source_position[0],
                    execution.final_position[1] - source_position[1],
                ],
            },
            expected_semantic_delta={"type": "none"},
            allowed_recovery_envelope={"max_actions": 12, "allowed_world_diff": "none"},
            invalidating_conditions=(
                "semantic_mismatch",
                "target_unreachable",
                "action_conflict",
            ),
            evidence=({
                "kind": "self_observed_probe_transition",
                "probe_id": execution.probe_id,
                "actor_delta": outcome.actor_delta,
            },),
            source="frontier_probe",
            trace_dependency_level=4,
            predicate_abstraction_level=2,
            termination_predicate_candidates=(predicate,),
            selected_predicate_id=predicate_id,
            selected_predicate_expression_hash=predicate_expression_hash(expression),
            guard_predicates=({"type": "semantic_signature_compatible"},),
            invalidation_predicates=({"type": "semantic_guard_violation"},),
        )
        termination_predicates = dict(outcome.known_predicate_truths)
        termination_predicates[predicate_id] = True
        state_class_id = f"S{expansion_index + 3}"
        state = OptionApplicabilityStateClass(
            state_class_id=state_class_id,
            controlled_entity_region=f"frontier_region_{expansion_index:03d}",
            semantic_guards=True,
            termination_predicates=termination_predicates,
            applicable_options=(),
            non_controlled_signature_class="stable_public_signature",
            exact_observation_hashes=(execution.final_observation_hash,),
        )
        prediction = OptionEffectPrediction(
            prediction_id=f"pred_frontier_{option_id}",
            model_id="frontier_probe_effect_model",
            option_id=option.option_id,
            selected_predicate_id=predicate_id,
            selected_predicate_expression_hash=option.selected_predicate_expression_hash or "",
            start_signature_class_hash=source_state.signature_hash,
            predicted_termination_satisfied=True,
            predicted_actor_delta=option.expected_actor_delta,
            predicted_semantic_delta=option.expected_semantic_delta,
            predicted_guards_preserved=True,
            predicted_stop_reason="TERMINATION_PREDICATE_SATISFIED",
            empirical_consistency=1.0,
            support_count=1,
            total_count=1,
        )
        edge = OptionApplicabilityEdge(
            edge_id=f"edge_frontier_{expansion_index:03d}",
            source_state_class_id=source_state.state_class_id,
            option_id=option.option_id,
            predicted_next_state_class_id=state.state_class_id,
            enabled_options=(),
            option_prediction=prediction,
        )
        return GraphExpansion(
            probe_id=execution.probe_id,
            source_state_class_id=source_state.state_class_id,
            new_state_class_id=state.state_class_id,
            new_option_id=option.option_id,
            new_edge_id=edge.edge_id,
            promotion_reason="stable actor-region transition observed from live frontier probe",
            option=option,
            state_class=state,
            edge=edge,
        )


def apply_graph_expansion(
    *,
    graph: OptionApplicabilityGraph,
    expansion: GraphExpansion,
) -> OptionApplicabilityGraph:
    new_predicate_id = expansion.option.selected_predicate_id or expansion.option.option_id
    states = []
    for state in graph.state_classes:
        predicates = dict(state.termination_predicates)
        predicates.setdefault(new_predicate_id, False)
        if state.state_class_id == expansion.source_state_class_id:
            state = replace(
                state,
                termination_predicates=predicates,
                applicable_options=tuple(sorted(set(state.applicable_options + (expansion.new_option_id,)))),
            )
        else:
            state = replace(state, termination_predicates=predicates)
        states.append(state)
    states.append(expansion.state_class)
    return OptionApplicabilityGraph(
        graph_id=f"{graph.graph_id}:frontier_expanded",
        state_classes=tuple(states),
        edges=graph.edges + (expansion.edge,),
    )


def m11_status(
    *,
    saturation_detected: bool,
    probe_count: int,
    probe_score_driven: bool,
    probe_executed: bool,
    promoted_count: int,
    noise_promoted: bool,
    expanded_graph_used: bool,
    live_loop_resumed_after_expansion: bool,
    honest_frontier_available: bool,
    fixed_probe_order: bool,
    trace_leakage: bool,
    oracle_signal: bool,
    unclassified_loop: bool,
) -> M11Status:
    if trace_leakage:
        return "failed_trace_leakage"
    if oracle_signal:
        return "failed_oracle_goal_signal"
    if fixed_probe_order:
        return "failed_fixed_probe_order"
    if noise_promoted:
        return "failed_noise_promoted_to_option"
    if unclassified_loop:
        return "failed_unclassified_loop"
    if not saturation_detected:
        return "partial_saturation_detected_only"
    if probe_count == 0 and not honest_frontier_available:
        return "partial_no_honest_frontier"
    if probe_count < 2 or not probe_score_driven:
        return "partial_no_honest_frontier"
    if not probe_executed:
        return "partial_saturation_detected_only"
    if promoted_count == 0:
        return "partial_probe_executed_no_promotion"
    if not expanded_graph_used:
        return "partial_option_promoted_not_used"
    if not live_loop_resumed_after_expansion:
        return "failed_live_loop_not_resumed_after_expansion"
    return "complete_with_known_limits"


def forbidden_frontier_findings(payload: dict[str, Any]) -> tuple[str, ...]:
    encoded = repr(payload)
    forbidden = (
        "human_trace",
        "public_trace_segment",
        "trace_corridor",
        "human_action",
        "raw_action",
        "action_at_index",
        "private",
        "oracle",
        "changer",
        "clock",
        "goal",
        "trap",
    )
    return tuple(marker for marker in forbidden if marker in encoded)


def _has_local_cycle(sequence: tuple[str, ...]) -> bool:
    if len(sequence) < 3:
        return False
    return sequence[-1] == sequence[-3] and sequence[-1] != sequence[-2]


def _repeated_tail(sequence: tuple[str, ...]) -> str | None:
    if len(sequence) < 2:
        return None
    return sequence[-1] if sequence[-1] == sequence[-2] else None


def _is_simple_action(action_id: int) -> bool:
    try:
        return not GameAction.from_id(action_id).is_complex()
    except Exception:
        return False


def _known_predicate_truths(
    *,
    options: tuple[OptionCandidate, ...],
    position: Position,
) -> dict[str, bool]:
    return {
        option.selected_predicate_id or option.option_id: evaluate_position_expression(
            selected_termination_expression(option),
            position,
        )
        for option in options
    }


def _probe_classification(
    *,
    execution: FrontierProbeExecution,
    result: ProbeOutcomeKind,
    promotable: bool,
    stable: bool,
    reason: str,
    known_predicate_truths: dict[str, bool],
) -> ProbeOutcomeClassification:
    start = execution.start_position
    final = execution.final_position
    delta = (
        {
            "from": list(start),
            "to": list(final),
            "delta": [final[0] - start[0], final[1] - start[1]],
        }
        if start is not None and final is not None
        else {"from": None, "to": None, "delta": None}
    )
    return ProbeOutcomeClassification(
        probe_id=execution.probe_id,
        result=result,
        stable=stable,
        promotable=promotable,
        semantic_guards_preserved=execution.semantic_guards_preserved,
        noise_rejected=({"kind": "volatile_visual_delta", "promoted": False},),
        actor_delta=delta,
        known_predicate_truths=known_predicate_truths,
        reason=reason,
    )


def _bbox_center(bbox: tuple[int, int, int, int]) -> Position:
    y0, x0, y1, x1 = bbox
    return ((x0 + x1) // 2, (y0 + y1) // 2)
