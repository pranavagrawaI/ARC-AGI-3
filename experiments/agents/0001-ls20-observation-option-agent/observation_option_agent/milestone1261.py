"""Public motor-constraint learning for 0001 M12.6.1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Literal

from .milestone9 import (
    AttemptMemory,
    ChainCandidate,
    OptionSelectionLedger,
    ProgressModel,
)
from .milestone10 import LearnedProgressModel, ProgressHypothesisSet


M1261Status = Literal[
    "complete_with_known_limits",
    "partial_motor_audit_only",
    "partial_constraints_without_selection_effect",
    "failed_no_public_no_op_measurement",
    "failed_repeated_no_op_not_deprioritized",
    "failed_named_wall_detector",
    "failed_oracle_or_private_state_used",
]


Position = tuple[int, int]


@dataclass(frozen=True)
class ActionEffectObservation:
    observation_id: str
    state_class_id: str
    action: int
    before_position: Position
    after_position: Position
    levels_before: int
    levels_after: int
    source: str

    @property
    def moved(self) -> bool:
        return self.before_position != self.after_position

    @property
    def level_delta(self) -> int:
        return self.levels_after - self.levels_before

    @property
    def no_op(self) -> bool:
        return not self.moved and self.level_delta == 0

    def to_record(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "state_class_id": self.state_class_id,
            "action": self.action,
            "before_position": list(self.before_position),
            "after_position": list(self.after_position),
            "moved": self.moved,
            "levels_before": self.levels_before,
            "levels_after": self.levels_after,
            "level_delta": self.level_delta,
            "no_op": self.no_op,
            "source": self.source,
        }


@dataclass(frozen=True)
class ActionEffectConstraint:
    state_class_id: str
    action: int
    support: int
    no_op_count: int
    moved_count: int
    level_delta_count: int
    no_op_rate: float
    status: str

    @property
    def constrained(self) -> bool:
        return self.status == "CONSTRAINED_NO_OP"

    def to_record(self) -> dict[str, Any]:
        return {
            "state_class_id": self.state_class_id,
            "action": self.action,
            "support": self.support,
            "no_op_count": self.no_op_count,
            "moved_count": self.moved_count,
            "level_delta_count": self.level_delta_count,
            "no_op_rate": self.no_op_rate,
            "status": self.status,
            "constrained": self.constrained,
        }


class ActionEffectConstraintModel:
    def __init__(
        self,
        *,
        minimum_support: int = 2,
        no_op_rate_threshold: float = 0.75,
    ) -> None:
        self.minimum_support = minimum_support
        self.no_op_rate_threshold = no_op_rate_threshold
        self._observations: list[ActionEffectObservation] = []

    def observe(self, observation: ActionEffectObservation) -> None:
        self._observations.append(observation)

    def observations(self) -> tuple[ActionEffectObservation, ...]:
        return tuple(self._observations)

    def constraints(self) -> tuple[ActionEffectConstraint, ...]:
        buckets: dict[tuple[str, int], list[ActionEffectObservation]] = defaultdict(list)
        for observation in self._observations:
            buckets[(observation.state_class_id, observation.action)].append(observation)
        constraints = []
        for (state_class_id, action), records in sorted(buckets.items()):
            no_op_count = sum(1 for record in records if record.no_op)
            moved_count = sum(1 for record in records if record.moved)
            level_delta_count = sum(1 for record in records if record.level_delta != 0)
            support = len(records)
            no_op_rate = round(no_op_count / support, 4) if support else 0.0
            status = "INSUFFICIENT_SUPPORT"
            if support >= self.minimum_support:
                status = (
                    "CONSTRAINED_NO_OP"
                    if no_op_rate >= self.no_op_rate_threshold
                    else "ACTION_EFFECT_OBSERVED"
                )
            constraints.append(ActionEffectConstraint(
                state_class_id=state_class_id,
                action=action,
                support=support,
                no_op_count=no_op_count,
                moved_count=moved_count,
                level_delta_count=level_delta_count,
                no_op_rate=no_op_rate,
                status=status,
            ))
        return tuple(constraints)

    def constraint_for(
        self,
        *,
        state_class_id: str,
        action: int,
    ) -> ActionEffectConstraint | None:
        for constraint in self.constraints():
            if constraint.state_class_id == state_class_id and constraint.action == action:
                return constraint
        return None

    def penalty_for_plan(
        self,
        *,
        state_class_id: str,
        action_plan: tuple[int, ...],
    ) -> tuple[float, tuple[dict[str, Any], ...]]:
        penalty = 0.0
        evidence = []
        for index, action in enumerate(action_plan, start=1):
            constraint = self.constraint_for(
                state_class_id=state_class_id,
                action=action,
            )
            if constraint is None or not constraint.constrained:
                continue
            action_penalty = -80.0 * constraint.no_op_rate
            penalty += action_penalty
            evidence.append({
                "plan_index": index,
                "action": action,
                "penalty": round(action_penalty, 4),
                "constraint": constraint.to_record(),
            })
        return round(penalty, 4), tuple(evidence)

    def to_record(self) -> dict[str, Any]:
        return {
            "minimum_support": self.minimum_support,
            "no_op_rate_threshold": self.no_op_rate_threshold,
            "observations": [observation.to_record() for observation in self._observations],
            "constraints": [constraint.to_record() for constraint in self.constraints()],
            "interpretation": "public action-effect constraints, not named object categories",
        }


class MotorAwareSelector:
    def __init__(
        self,
        *,
        motor_model: ActionEffectConstraintModel,
        progress_model: ProgressModel | None = None,
    ) -> None:
        self.motor_model = motor_model
        self.progress_model = progress_model or LearnedProgressModel(
            ProgressHypothesisSet.from_m9_priors()
        )

    def select(
        self,
        *,
        round_index: int,
        state_class_id: str,
        candidates: tuple[ChainCandidate, ...],
        action_plans: dict[str, tuple[int, ...]],
        attempt_memory: AttemptMemory,
    ) -> OptionSelectionLedger:
        scored = []
        for candidate in candidates:
            base = self.progress_model.score_candidate(
                candidate=candidate,
                attempt_memory=attempt_memory,
            )
            plan = _action_plan_for_candidate(candidate, action_plans)
            motor_penalty, evidence = self.motor_model.penalty_for_plan(
                state_class_id=state_class_id,
                action_plan=plan,
            )
            terms = dict(base.score_terms or {})
            terms["motor_no_op_penalty"] = motor_penalty
            rejection_reason = base.rejection_reason
            applicable = base.applicable
            provenance = base.provenance
            if evidence:
                rejection_reason = _append_reason(rejection_reason, "motor_constraint_no_op")
                applicable = False
                provenance = provenance + ({
                    "kind": "motor_constraint_evidence",
                    "evidence": list(evidence),
                },)
            scored.append(replace(
                base,
                score_terms=terms,
                score=sum(terms.values()),
                applicable=applicable,
                rejection_reason=rejection_reason,
                provenance=provenance,
            ))
        viable = tuple(candidate for candidate in scored if candidate.applicable)
        pool = viable if viable else tuple(scored)
        selected = max(
            pool,
            key=lambda candidate: (candidate.score, candidate.candidate_id),
        ) if pool else None
        decorated = tuple(
            replace(
                candidate,
                selected=selected is not None and candidate.candidate_id == selected.candidate_id,
                selection_reason=(
                    "highest motor-aware progress score among applicable candidates"
                    if selected is not None and candidate.candidate_id == selected.candidate_id
                    else None
                ),
            )
            for candidate in scored
        )
        return OptionSelectionLedger(
            round_index=round_index,
            state_class_id=state_class_id,
            candidates=decorated,
            selected_candidate_id=selected.candidate_id if selected else None,
            selection_generated_before_execution=True,
            candidate_scores_generated_before_execution=True,
            selection_policy="deterministic_progress_score_with_public_motor_constraints",
        )


def action_effect_observations_from_events(
    events: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    default_state_class_id: str = "public_context",
) -> tuple[ActionEffectObservation, ...]:
    observations: list[ActionEffectObservation] = []
    seen: set[tuple[int, int, Position, Position]] = set()
    for event_index, event in enumerate(events):
        state_class_id = _state_class_id_from_event(
            event,
            default_state_class_id=default_state_class_id,
        )
        for container_index, container in enumerate(_iter_action_record_containers(event)):
            before_position = _position_or_none(container.get("start_position"))
            records = [
                record
                for record in container.get("records", ())
                if isinstance(record, dict)
            ]
            for record in sorted(records, key=lambda item: int(item.get("step", 0))):
                if not record.get("executed"):
                    continue
                after_position = _position_or_none(record.get("controlled_position"))
                if after_position is None:
                    continue
                if before_position is not None:
                    step = int(record.get("step", len(observations) + 1))
                    action = int(record["action"])
                    key = (step, action, before_position, after_position)
                    if key not in seen:
                        seen.add(key)
                        observations.append(ActionEffectObservation(
                            observation_id=(
                                f"action_effect_{event_index}_{container_index}_{step}"
                            ),
                            state_class_id=state_class_id,
                            action=action,
                            before_position=before_position,
                            after_position=after_position,
                            levels_before=int(record.get("levels_before", 0)),
                            levels_after=int(record.get("levels_after", 0)),
                            source="public_milestone_action_record",
                        ))
                before_position = after_position
    return tuple(observations)


def m1261_status(
    *,
    observation_count: int,
    constrained_action_count: int,
    selection_changed: bool,
    repeated_no_op_deprioritized: bool,
    named_wall_detector: bool,
    oracle_or_private_state_used: bool,
) -> M1261Status:
    if oracle_or_private_state_used:
        return "failed_oracle_or_private_state_used"
    if named_wall_detector:
        return "failed_named_wall_detector"
    if observation_count == 0:
        return "failed_no_public_no_op_measurement"
    if constrained_action_count == 0:
        return "partial_motor_audit_only"
    if not repeated_no_op_deprioritized:
        return "failed_repeated_no_op_not_deprioritized"
    if not selection_changed:
        return "partial_constraints_without_selection_effect"
    return "complete_with_known_limits"


def forbidden_motor_constraint_findings(payload: dict[str, Any]) -> tuple[str, ...]:
    encoded = repr(payload)
    forbidden = (
        "private",
        "oracle",
        "wall",
        "changer",
        "clock",
        "goal",
        "trap",
        "human_trace",
        "role",
    )
    return tuple(marker for marker in forbidden if marker in encoded)


def _action_plan_for_candidate(
    candidate: ChainCandidate,
    action_plans: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    actions: list[int] = []
    for option_id in candidate.option_ids:
        actions.extend(action_plans.get(option_id, ()))
    return tuple(actions)


def _append_reason(current: str | None, reason: str) -> str:
    if current is None:
        return reason
    if reason in current:
        return current
    return f"{current};{reason}"


def _iter_action_record_containers(value: Any):
    if isinstance(value, dict):
        records = value.get("records")
        if isinstance(records, list) and any(
            isinstance(record, dict) and {"step", "action", "executed"}.issubset(record)
            for record in records
        ):
            yield value
        for item in value.values():
            yield from _iter_action_record_containers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_action_record_containers(item)


def _position_or_none(value: Any) -> Position | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return (int(value[0]), int(value[1]))


def _state_class_id_from_event(
    event: dict[str, Any],
    *,
    default_state_class_id: str,
) -> str:
    if isinstance(event.get("state_class_id"), str):
        return str(event["state_class_id"])
    binding = event.get("binding")
    if isinstance(binding, dict) and isinstance(binding.get("state_class_id"), str):
        return str(binding["state_class_id"])
    return default_state_class_id
