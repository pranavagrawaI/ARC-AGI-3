"""Level-transition context binding for 0001 milestone 11.6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .milestone8 import OptionApplicabilityGraph, OptionApplicabilityStateClass
from .milestone11 import FrontierProbeExecution


M116Status = Literal[
    "complete_with_known_limits",
    "partial_transition_detected_only",
    "partial_context_created_no_seeding",
    "partial_post_transition_seeding_no_action",
    "failed_cross_level_delta_promoted",
    "failed_no_new_level_context",
    "failed_transferable_knowledge_dropped",
    "failed_stale_spatial_options_reused",
    "failed_post_transition_state_unclassified",
    "failed_trace_leakage",
    "failed_oracle_goal_signal",
]


@dataclass(frozen=True)
class LevelTransitionEvent:
    event_type: str
    from_level_index: int
    to_level_index: int
    trigger_action: int
    public_step: int
    probe_id: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "from_level_index": self.from_level_index,
            "to_level_index": self.to_level_index,
            "trigger_action": self.trigger_action,
            "public_step": self.public_step,
            "probe_id": self.probe_id,
        }


@dataclass(frozen=True)
class BoundaryProbeSlice:
    probe_id: str
    original_actions: tuple[int, ...]
    pre_transition_actions: tuple[int, ...]
    post_transition_actions: tuple[int, ...]
    transition_event: LevelTransitionEvent
    cross_level_option_training_allowed: bool
    cross_level_delta_promoted: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "original_actions": list(self.original_actions),
            "pre_transition_actions": list(self.pre_transition_actions),
            "post_transition_actions": list(self.post_transition_actions),
            "transition_event": self.transition_event.to_record(),
            "cross_level_option_training_allowed": self.cross_level_option_training_allowed,
            "cross_level_delta_promoted": self.cross_level_delta_promoted,
        }


@dataclass(frozen=True)
class LevelContext:
    context_id: str
    level_index: int
    initial_state_class_id: str
    graph: OptionApplicabilityGraph
    carried_transferables: tuple[str, ...]
    invalidated_level_local: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "level_index": self.level_index,
            "initial_state_class_id": self.initial_state_class_id,
            "graph": self.graph.to_record(),
            "carried_transferables": list(self.carried_transferables),
            "invalidated_level_local": list(self.invalidated_level_local),
        }


@dataclass(frozen=True)
class TerminalEvidenceRecord:
    terminal_evidence_id: str
    source: str
    level_completed: bool
    from_level_index: int
    to_level_index: int
    trigger_action: int
    public_step: int
    pre_terminal_state_class: str | None
    trigger_probe_id: str | None
    known_limit: str

    def to_record(self) -> dict[str, Any]:
        return {
            "terminal_evidence_id": self.terminal_evidence_id,
            "source": self.source,
            "level_completed": self.level_completed,
            "from_level_index": self.from_level_index,
            "to_level_index": self.to_level_index,
            "trigger_action": self.trigger_action,
            "public_step": self.public_step,
            "pre_terminal_state_class": self.pre_terminal_state_class,
            "trigger_probe_id": self.trigger_probe_id,
            "known_limit": self.known_limit,
        }


class LevelTransitionDetector:
    def detect(
        self,
        *,
        execution: FrontierProbeExecution,
    ) -> LevelTransitionEvent | None:
        for record in execution.records:
            before = record.get("levels_before")
            after = record.get("levels_after")
            if isinstance(before, int) and isinstance(after, int) and after > before:
                return LevelTransitionEvent(
                    event_type="LEVEL_TRANSITION",
                    from_level_index=before,
                    to_level_index=after,
                    trigger_action=int(record["action"]),
                    public_step=int(record["step"]),
                    probe_id=execution.probe_id,
                )
        return None


class BoundaryProbeSlicer:
    def slice(
        self,
        *,
        execution: FrontierProbeExecution,
        transition: LevelTransitionEvent,
    ) -> BoundaryProbeSlice:
        pre: list[int] = []
        post: list[int] = []
        crossed = False
        for record in execution.records:
            if not record.get("executed"):
                continue
            action = int(record["action"])
            if not crossed:
                pre.append(action)
            else:
                post.append(action)
            if int(record["step"]) == transition.public_step:
                crossed = True
        return BoundaryProbeSlice(
            probe_id=execution.probe_id,
            original_actions=execution.action_sequence,
            pre_transition_actions=tuple(pre),
            post_transition_actions=tuple(post),
            transition_event=transition,
            cross_level_option_training_allowed=False,
            cross_level_delta_promoted=False,
        )


class LevelContextManager:
    def create_context(
        self,
        *,
        level_index: int,
        observation_hash: str,
    ) -> LevelContext:
        state = OptionApplicabilityStateClass(
            state_class_id=f"L{level_index}_S0",
            controlled_entity_region=f"level_{level_index}_initial_region",
            semantic_guards=True,
            termination_predicates={},
            applicable_options=(),
            non_controlled_signature_class=f"level_{level_index}_public_signature",
            exact_observation_hashes=(observation_hash,),
        )
        graph = OptionApplicabilityGraph(
            graph_id=f"level_{level_index}_option_graph",
            state_classes=(state,),
            edges=(),
        )
        return LevelContext(
            context_id=f"level_{level_index}_context",
            level_index=level_index,
            initial_state_class_id=state.state_class_id,
            graph=graph,
            carried_transferables=(
                "live_public_event_tracker",
                "action_delta_model",
                "predicate_templates",
                "progress_weights",
                "noise_filters",
                "frontier_scorer",
            ),
            invalidated_level_local=(
                "previous_level_spatial_options",
                "previous_level_state_classes",
                "previous_level_edges",
                "previous_level_saturation_memory",
                "previous_level_attempt_memory",
            ),
        )


def terminal_evidence_from_transition(
    *,
    transition: LevelTransitionEvent,
    pre_terminal_state_class: str | None,
) -> TerminalEvidenceRecord:
    return TerminalEvidenceRecord(
        terminal_evidence_id="terminal_000",
        source="live_public_run",
        level_completed=True,
        from_level_index=transition.from_level_index,
        to_level_index=transition.to_level_index,
        trigger_action=transition.trigger_action,
        public_step=transition.public_step,
        pre_terminal_state_class=pre_terminal_state_class,
        trigger_probe_id=transition.probe_id,
        known_limit=(
            "terminal reached during frontier probing, not yet modeled as final "
            "objective condition"
        ),
    )


def m116_status(
    *,
    transition_detected: bool,
    probe_split: bool,
    cross_level_delta_promoted: bool,
    new_context_created: bool,
    transferables_carried: bool,
    stale_spatial_invalidated: bool,
    post_transition_state_bound: bool,
    post_transition_probe_count: int,
    post_transition_actions_executed: int,
    trace_leakage: bool,
    oracle_signal: bool,
) -> M116Status:
    if trace_leakage:
        return "failed_trace_leakage"
    if oracle_signal:
        return "failed_oracle_goal_signal"
    if cross_level_delta_promoted:
        return "failed_cross_level_delta_promoted"
    if not transition_detected:
        return "partial_transition_detected_only"
    if not probe_split:
        return "partial_transition_detected_only"
    if not new_context_created:
        return "failed_no_new_level_context"
    if not transferables_carried:
        return "failed_transferable_knowledge_dropped"
    if not stale_spatial_invalidated:
        return "failed_stale_spatial_options_reused"
    if not post_transition_state_bound:
        return "failed_post_transition_state_unclassified"
    if post_transition_probe_count == 0:
        return "partial_context_created_no_seeding"
    if post_transition_actions_executed == 0:
        return "partial_post_transition_seeding_no_action"
    return "complete_with_known_limits"


def forbidden_level_transition_findings(payload: dict[str, Any]) -> tuple[str, ...]:
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
