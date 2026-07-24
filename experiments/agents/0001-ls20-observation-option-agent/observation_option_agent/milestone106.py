"""Live option-state binding and replanning for 0001 M10.6."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

from arcengine import GameAction

from experiments.harness.runner import public_observation

from .events import EventDetector
from .milestone3 import SemanticSignature
from .milestone4 import (
    OptionCandidate,
    Position,
    evaluate_position_expression,
    selected_termination_expression,
)
from .milestone8 import OptionApplicabilityGraph, OptionApplicabilityStateClass
from .milestone9 import ChainCandidate
from .observation import ObservationParser
from .tracking import ObjectTracker


LiveClassificationStatus = Literal[
    "MATCHED_KNOWN_STATE_CLASS",
    "NEW_STATE_CLASS",
    "AMBIGUOUS_STATE_CLASS",
    "UNKNOWN_STATE_CLASS",
    "SEMANTIC_GUARD_VIOLATION",
    "CONTROLLED_ENTITY_LOST",
]
M106Status = Literal[
    "complete_with_known_limits",
    "partial_state_classification_only",
    "partial_replanner_not_wired_to_actuator",
    "partial_single_cycle_only",
    "partial_counterfactual_selection_only",
    "failed_live_observation_unclassified",
    "failed_no_live_candidate_generation",
    "failed_selected_candidate_not_executed",
    "failed_trace_leakage",
    "failed_oracle_goal_signal",
]


@dataclass(frozen=True)
class LiveOptionStateBinding:
    observation_hash: str
    classification: LiveClassificationStatus
    state_class_id: str | None
    confidence: float
    controlled_position: Position | None
    controlled_position_source: str
    predicate_truths: dict[str, bool]
    true_predicates: tuple[str, ...]
    false_predicates: tuple[str, ...]
    applicable_options: tuple[str, ...]
    semantic_guards_valid: bool
    reason: str
    exact_hash_used_for_match: bool = False

    @property
    def replannable(self) -> bool:
        return self.classification in {
            "MATCHED_KNOWN_STATE_CLASS",
            "NEW_STATE_CLASS",
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "observation_hash": self.observation_hash,
            "classification": self.classification,
            "state_class_id": self.state_class_id,
            "confidence": self.confidence,
            "controlled_position": (
                list(self.controlled_position)
                if self.controlled_position is not None
                else None
            ),
            "controlled_position_source": self.controlled_position_source,
            "predicate_truths": self.predicate_truths,
            "true_predicates": list(self.true_predicates),
            "false_predicates": list(self.false_predicates),
            "applicable_options": list(self.applicable_options),
            "semantic_guards_valid": self.semantic_guards_valid,
            "reason": self.reason,
            "exact_hash_used_for_match": self.exact_hash_used_for_match,
            "replannable": self.replannable,
        }


@dataclass(frozen=True)
class LiveActuationResult:
    candidate_id: str
    executed_option_id: str | None
    public_action_plan: tuple[int, ...]
    records: tuple[dict[str, Any], ...]
    actions_executed: int
    final_observation_hash: str
    final_state: str
    levels_completed: int
    controlled_position: Position | None
    controlled_position_source: str
    selected_candidate_sent_to_actuator: bool
    stop_reason: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "executed_option_id": self.executed_option_id,
            "public_action_plan": list(self.public_action_plan),
            "records": list(self.records),
            "actions_executed": self.actions_executed,
            "final_observation_hash": self.final_observation_hash,
            "final_state": self.final_state,
            "levels_completed": self.levels_completed,
            "controlled_position": (
                list(self.controlled_position)
                if self.controlled_position is not None
                else None
            ),
            "controlled_position_source": self.controlled_position_source,
            "selected_candidate_sent_to_actuator": self.selected_candidate_sent_to_actuator,
            "stop_reason": self.stop_reason,
        }


class LiveObservationStateClassifier:
    def classify(
        self,
        *,
        observation_hash: str,
        graph: OptionApplicabilityGraph,
        options: tuple[OptionCandidate, ...],
        controlled_position: Position | None,
        controlled_position_source: str,
        semantic_guards_valid: bool = True,
        allow_new_state_class: bool = False,
    ) -> LiveOptionStateBinding:
        if controlled_position is None:
            return LiveOptionStateBinding(
                observation_hash=observation_hash,
                classification="CONTROLLED_ENTITY_LOST",
                state_class_id=None,
                confidence=0.0,
                controlled_position=None,
                controlled_position_source=controlled_position_source,
                predicate_truths={},
                true_predicates=(),
                false_predicates=(),
                applicable_options=(),
                semantic_guards_valid=semantic_guards_valid,
                reason="controlled entity position unavailable",
            )
        predicate_truths = _predicate_truths(
            options=options,
            controlled_position=controlled_position,
        )
        true_predicates = tuple(
            predicate_id
            for predicate_id, value in sorted(predicate_truths.items())
            if value
        )
        false_predicates = tuple(
            predicate_id
            for predicate_id, value in sorted(predicate_truths.items())
            if not value
        )
        if not semantic_guards_valid:
            return LiveOptionStateBinding(
                observation_hash=observation_hash,
                classification="SEMANTIC_GUARD_VIOLATION",
                state_class_id=None,
                confidence=0.0,
                controlled_position=controlled_position,
                controlled_position_source=controlled_position_source,
                predicate_truths=predicate_truths,
                true_predicates=true_predicates,
                false_predicates=false_predicates,
                applicable_options=(),
                semantic_guards_valid=False,
                reason="semantic guard failed",
            )

        matches = tuple(
            state
            for state in graph.state_classes
            if _state_matches_predicates(state, predicate_truths)
            and state.semantic_guards
        )
        if len(matches) == 1:
            state = matches[0]
            return LiveOptionStateBinding(
                observation_hash=observation_hash,
                classification="MATCHED_KNOWN_STATE_CLASS",
                state_class_id=state.state_class_id,
                confidence=1.0,
                controlled_position=controlled_position,
                controlled_position_source=controlled_position_source,
                predicate_truths=predicate_truths,
                true_predicates=true_predicates,
                false_predicates=false_predicates,
                applicable_options=state.applicable_options,
                semantic_guards_valid=True,
                reason="matched predicate/applicability signature",
            )
        if len(matches) > 1:
            return LiveOptionStateBinding(
                observation_hash=observation_hash,
                classification="AMBIGUOUS_STATE_CLASS",
                state_class_id=None,
                confidence=0.0,
                controlled_position=controlled_position,
                controlled_position_source=controlled_position_source,
                predicate_truths=predicate_truths,
                true_predicates=true_predicates,
                false_predicates=false_predicates,
                applicable_options=(),
                semantic_guards_valid=True,
                reason="multiple state classes share predicate signature",
            )
        if allow_new_state_class:
            return LiveOptionStateBinding(
                observation_hash=observation_hash,
                classification="NEW_STATE_CLASS",
                state_class_id="S_NEW",
                confidence=0.5,
                controlled_position=controlled_position,
                controlled_position_source=controlled_position_source,
                predicate_truths=predicate_truths,
                true_predicates=true_predicates,
                false_predicates=false_predicates,
                applicable_options=(),
                semantic_guards_valid=True,
                reason="valid unmatched predicate signature",
            )
        return LiveOptionStateBinding(
            observation_hash=observation_hash,
            classification="UNKNOWN_STATE_CLASS",
            state_class_id=None,
            confidence=0.0,
            controlled_position=controlled_position,
            controlled_position_source=controlled_position_source,
            predicate_truths=predicate_truths,
            true_predicates=true_predicates,
            false_predicates=false_predicates,
            applicable_options=(),
            semantic_guards_valid=True,
            reason="no known state class matches predicate signature",
        )


class LiveCandidateGenerator:
    def generate(
        self,
        *,
        graph: OptionApplicabilityGraph,
        binding: LiveOptionStateBinding,
        options: tuple[OptionCandidate, ...],
        action_plans: dict[str, tuple[int, ...]],
        max_depth: int = 2,
    ) -> tuple[ChainCandidate, ...]:
        if not binding.replannable or binding.state_class_id is None:
            return ()
        option_by_id = {option.option_id: option for option in options}
        state = graph.state(binding.state_class_id)
        if state is None:
            return ()
        candidates: list[ChainCandidate] = []
        for option_id in state.applicable_options:
            option = option_by_id.get(option_id)
            if option is None:
                continue
            single = graph.predict_chain(
                prediction_id=f"pred_live_{option_id}",
                start_state_class_id=state.state_class_id,
                options=(option,),
            )
            if single is not None:
                candidates.append(_candidate_from_live_prediction(
                    candidate_id=_candidate_id_for((option_id,)),
                    option_ids=(option_id,),
                    start_state_class_id=state.state_class_id,
                    predicted_terminal_state_class_id=(
                        single.links[-1].predicted_next_state_class_id
                    ),
                    action_cost=_action_cost(
                        option_ids=(option_id,),
                        action_plans=action_plans,
                    ),
                    new_option_enabled=bool(single.links[-1].enabled_options),
                    uncertainty_reduction=False,
                ))
            if max_depth < 2 or single is None:
                continue
            next_state = graph.state(single.links[-1].predicted_next_state_class_id)
            if next_state is None:
                continue
            for next_option_id in next_state.applicable_options:
                next_option = option_by_id.get(next_option_id)
                if next_option is None:
                    continue
                chain_options = (option, next_option)
                chain = graph.predict_chain(
                    prediction_id=f"pred_live_{option_id}_{next_option_id}",
                    start_state_class_id=state.state_class_id,
                    options=chain_options,
                )
                if chain is None:
                    continue
                ids = (option_id, next_option_id)
                candidates.append(_candidate_from_live_prediction(
                    candidate_id=_candidate_id_for(ids),
                    option_ids=ids,
                    start_state_class_id=state.state_class_id,
                    predicted_terminal_state_class_id=(
                        chain.links[-1].predicted_next_state_class_id
                    ),
                    action_cost=_action_cost(
                        option_ids=ids,
                        action_plans=action_plans,
                    ),
                    new_option_enabled=any(link.enabled_options for link in chain.links),
                    uncertainty_reduction=True,
                ))
        return tuple(candidates)


class LiveCandidateActuator:
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

    def execute_first_link(
        self,
        *,
        env: Any,
        current_observation: Any,
        candidate: ChainCandidate,
        action_plans: dict[str, tuple[int, ...]],
        current_controlled_position: Position | None,
        step_offset: int,
        remaining_budget: int,
    ) -> tuple[LiveActuationResult, Any]:
        option_id = candidate.option_ids[0] if candidate.option_ids else None
        plan = action_plans.get(option_id or "", ())[:remaining_budget]
        records: list[dict[str, Any]] = []
        controlled_position = current_controlled_position
        controlled_position_source = (
            "live_public_event_tracker"
            if self.controlled_object_id is not None
            else "option_state_binding"
        )
        current = current_observation
        stop_reason: str | None = None
        for index, action_id in enumerate(plan, start=1):
            if current.state in {"WIN", "GAME_OVER"}:
                stop_reason = "TERMINAL_STATE"
                break
            if action_id not in current.available_actions:
                stop_reason = "ACTION_NOT_AVAILABLE"
                records.append(_action_record(
                    step=step_offset + index,
                    action_id=action_id,
                    before=current,
                    after=None,
                    executed=False,
                    stop_reason=stop_reason,
                ))
                break
            action = GameAction.from_id(action_id)
            if action.is_complex():
                stop_reason = "ACTION_NOT_AVAILABLE"
                records.append(_action_record(
                    step=step_offset + index,
                    action_id=action_id,
                    before=current,
                    after=None,
                    executed=False,
                    stop_reason="COMPLEX_ACTION_NOT_SUPPORTED",
                ))
                break
            following = public_observation(
                env.step(
                    action,
                    data=None,
                    reasoning={
                        "milestone": "M10_6_LIVE_OPTION_STATE_BINDING",
                        "candidate_id": candidate.candidate_id,
                        "option_id": option_id,
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
                controlled_position_source = "live_public_event_tracker"
            records.append(_action_record(
                step=step_offset + index,
                action_id=action_id,
                before=current,
                after=following,
                executed=True,
                stop_reason=None,
                controlled_position=controlled_position,
                controlled_position_source=controlled_position_source,
            ))
            current = following
        return LiveActuationResult(
            candidate_id=candidate.candidate_id,
            executed_option_id=option_id,
            public_action_plan=plan,
            records=tuple(records),
            actions_executed=sum(1 for record in records if record["executed"]),
            final_observation_hash=observation_hash(current),
            final_state=current.state,
            levels_completed=current.levels_completed,
            controlled_position=controlled_position,
            controlled_position_source=controlled_position_source,
            selected_candidate_sent_to_actuator=True,
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


def observation_hash(observation: Any) -> str:
    digest = hashlib.sha256()
    digest.update(observation.pixels.tobytes())
    digest.update(json.dumps({
        "available_actions": observation.available_actions,
        "levels_completed": observation.levels_completed,
        "state": observation.state,
        "full_reset": observation.full_reset,
    }, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def m106_status(
    *,
    classification_count: int,
    post_execution_classified: bool,
    candidate_generation_from_live_state: bool,
    learned_progress_used: bool,
    candidate_sent_to_actuator: bool,
    actions_executed: int,
    replan_cycles_completed: int,
    valid_stop_reason: bool,
    counterfactual_only: bool,
    selected_candidate_not_executed: bool,
    trace_leakage: bool,
    oracle_signal: bool,
    live_observation_unclassified: bool,
    no_live_candidate_generation: bool,
) -> M106Status:
    if trace_leakage:
        return "failed_trace_leakage"
    if oracle_signal:
        return "failed_oracle_goal_signal"
    if selected_candidate_not_executed:
        return "failed_selected_candidate_not_executed"
    if live_observation_unclassified:
        return "failed_live_observation_unclassified"
    if no_live_candidate_generation:
        return "failed_no_live_candidate_generation"
    if counterfactual_only:
        return "partial_counterfactual_selection_only"
    if classification_count == 0 or not post_execution_classified:
        return "partial_state_classification_only"
    if not candidate_generation_from_live_state or not learned_progress_used:
        return "partial_state_classification_only"
    if not candidate_sent_to_actuator or actions_executed == 0:
        return "partial_replanner_not_wired_to_actuator"
    if replan_cycles_completed >= 2 or valid_stop_reason:
        return "complete_with_known_limits"
    return "partial_single_cycle_only"


def forbidden_live_replanning_findings(payload: dict[str, Any]) -> tuple[str, ...]:
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


def _predicate_truths(
    *,
    options: tuple[OptionCandidate, ...],
    controlled_position: Position,
) -> dict[str, bool]:
    truths = {}
    for option in options:
        predicate_id = option.selected_predicate_id or option.option_id
        truths[predicate_id] = evaluate_position_expression(
            selected_termination_expression(option),
            controlled_position,
        )
    return dict(sorted(truths.items()))


def _state_matches_predicates(
    state: OptionApplicabilityStateClass,
    predicate_truths: dict[str, bool],
) -> bool:
    return all(
        predicate_truths.get(predicate_id) == value
        for predicate_id, value in state.termination_predicates.items()
    )


def _candidate_from_live_prediction(
    *,
    candidate_id: str,
    option_ids: tuple[str, ...],
    start_state_class_id: str,
    predicted_terminal_state_class_id: str | None,
    action_cost: int,
    new_option_enabled: bool,
    uncertainty_reduction: bool,
) -> ChainCandidate:
    return ChainCandidate(
        candidate_id=candidate_id,
        candidate_type="option_chain" if len(option_ids) > 1 else "single_option",
        option_ids=option_ids,
        start_state_class_id=start_state_class_id,
        predicted_terminal_state_class_id=predicted_terminal_state_class_id,
        predicted_progress_features={
            "level_completion": False,
            "stable_semantic_delta": False,
            "new_option_enabled": new_option_enabled,
            "new_state_class": predicted_terminal_state_class_id != start_state_class_id,
            "uncertainty_reduction": uncertainty_reduction,
            "volatile_visual_delta": False,
        },
        cost_features={
            "predicted_action_cost": action_cost,
            "empirical_consistency": 1.0,
        },
        risk_features={"guard_risk": 0},
        provenance=({
            "kind": "live_option_graph_prediction",
            "start_state_class_id": start_state_class_id,
        },),
    )


def _candidate_id_for(option_ids: tuple[str, ...]) -> str:
    if option_ids == ("self_option_000",):
        return "option_000"
    if option_ids == ("self_option_001",):
        return "option_001"
    if option_ids == ("self_option_000", "self_option_001"):
        return "chain_000"
    return "chain_" + "_".join(option_ids)


def _action_cost(
    *,
    option_ids: tuple[str, ...],
    action_plans: dict[str, tuple[int, ...]],
) -> int:
    return sum(len(action_plans.get(option_id, ())) for option_id in option_ids)


def _action_record(
    *,
    step: int,
    action_id: int,
    before: Any,
    after: Any | None,
    executed: bool,
    stop_reason: str | None,
    controlled_position: Position | None = None,
    controlled_position_source: str | None = None,
) -> dict[str, Any]:
    record = {
        "step": step,
        "action": action_id,
        "phase": "LIVE_ACTUATION",
        "executed": executed,
        "levels_before": before.levels_completed,
        "state_before": before.state,
        "available_actions_before": list(before.available_actions),
        "stop_reason": stop_reason,
    }
    if after is not None:
        record.update({
            "levels_after": after.levels_completed,
            "state_after": after.state,
            "available_actions_after": list(after.available_actions),
        })
    if controlled_position is not None:
        record["controlled_position"] = list(controlled_position)
    if controlled_position_source is not None:
        record["controlled_position_source"] = controlled_position_source
    return record


def _bbox_center(bbox: tuple[int, int, int, int]) -> Position:
    min_x, min_y, max_x, max_y = bbox
    return (
        int(round((min_x + max_x) / 2)),
        int(round((min_y + max_y) / 2)),
    )
