"""Passively extract neutral visual events from an admissible trace."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from observation_option_agent import (
    ActionCouplingMemory,
    ControlledEntityBelief,
    ConditionalEffectMemory,
    EventDetector,
    InteractionDetector,
    ObjectTracker,
    ObservationParser,
    iter_trace_steps,
    extract_effect_evidence,
)


DEFAULT_TRACE = (
    REPO
    / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710"
    / "ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "human-trace-events.jsonl"
)
DEFAULT_SUMMARY = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "human-trace-summary.json"
)


def _control_record(control) -> dict[str, object] | None:
    if control is None:
        return None
    return {
        "level": control.level,
        "object_ids": list(control.object_ids),
        "support": control.support,
        "distinct_actions": control.distinct_actions,
        "consistency": control.consistency,
        "confidence": control.confidence,
        "source": control.source,
        "action_model": {
            str(action): list(delta)
            for action, delta in control.action_model.items()
        },
        "discovered_step": control.discovered_step,
        "actions_into_level": control.actions_into_level,
    }


def _interaction_record(interaction) -> dict[str, object] | None:
    if interaction is None:
        return None
    return {
        "step": interaction.step,
        "action": interaction.action,
        "controlled_object_ids": list(interaction.controlled_object_ids),
        "expected_translation": list(interaction.expected_translation),
        "observed_translation": list(interaction.observed_translation),
        "blocked": interaction.blocked,
        "contact_candidates": [
            {
                "object_id": item.object_id,
                "value": item.value,
                "shape_id": item.shape_id,
                "overlap_pixels": item.overlap_pixels,
                "object_area": item.object_area,
                "object_coverage": item.object_coverage,
            }
            for item in interaction.contact_candidates
        ],
        "changed_other_object_ids": list(interaction.changed_other_object_ids),
        "created_other_object_ids": list(interaction.created_other_object_ids),
        "disappeared_other_object_ids": list(interaction.disappeared_other_object_ids),
    }


def _affordance_record(affordance) -> dict[str, object]:
    return {
        "level": affordance.level,
        "contact": [
            {
                "value": item.value,
                "area": item.area,
                "shape_id": item.shape_id,
            }
            for item in affordance.contacts
        ],
        "effect_kind": affordance.effect_kind,
        "target_object_ids": list(affordance.target_object_ids),
        "targets": [
            {
                "object_id": item.object_id,
                "value": item.value,
                "baseline_support": item.baseline_support,
                "baseline_opportunities": item.baseline_opportunities,
                "baseline_rate": item.baseline_rate,
                "rate_excess": item.rate_excess,
            }
            for item in affordance.targets
        ],
        "contact_support": affordance.contact_support,
        "contact_opportunities": affordance.contact_opportunities,
        "contact_rate": round(affordance.contact_rate, 4),
        "state_model": {
            "observed_state_count": affordance.state_model.observed_state_count,
            "deterministic": affordance.state_model.deterministic,
            "continuous": affordance.state_model.continuous,
            "closed_cycle": affordance.state_model.closed_cycle,
            "cycle_length": affordance.state_model.cycle_length,
            "current_state": (
                [list(item) for item in affordance.state_model.current_state]
                if affordance.state_model.current_state is not None
                else None
            ),
            "transitions": [
                {
                    "step": transition.step,
                    "before": [list(item) for item in transition.before],
                    "after": [list(item) for item in transition.after],
                }
                for transition in affordance.state_model.transitions
            ],
        },
    }


def _affordance_key(affordance) -> tuple:
    return (
        affordance.level,
        affordance.contacts,
        affordance.effect_kind,
        affordance.target_object_ids,
    )


def _event_record(
    event,
    *,
    detections: int,
    control,
    interaction,
    affordance_changes,
) -> dict[str, object]:
    return {
        "step": event.step,
        "action": event.action,
        "detections": detections,
        "levels_before": event.levels_before,
        "levels_after": event.levels_after,
        "state_after": event.state_after,
        "full_reset": event.full_reset,
        "changed_pixels": event.changed_pixels,
        "scene_change_fraction": round(event.scene_change_fraction, 6),
        "created_object_ids": list(event.created_object_ids),
        "disappeared_object_ids": list(event.disappeared_object_ids),
        "reappeared_object_ids": list(event.reappeared_object_ids),
        "object_deltas": [
            {
                "object_id": delta.object_id,
                "translation": list(delta.translation),
                "value_before": delta.value_before,
                "value_after": delta.value_after,
                "value_changed": delta.value_changed,
                "shape_changed": delta.shape_changed,
                "shape_transform": delta.shape_transform,
                "area_before": delta.area_before,
                "area_after": delta.area_after,
                "area_delta": delta.area_delta,
                "bbox_before": list(delta.bbox_before),
                "bbox_after": list(delta.bbox_after),
            }
            for delta in event.object_deltas
        ],
        "motion_groups": [
            {
                "translation": list(group.translation),
                "object_ids": list(group.object_ids),
            }
            for group in event.motion_groups
        ],
        "control": _control_record(control),
        "interaction": _interaction_record(interaction),
        "affordance_changes": [
            {
                "status": status,
                "affordance": _affordance_record(item),
            }
            for status, item in affordance_changes
        ],
    }


def _group_conditional_effects(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple, dict[str, object]] = {}
    for candidate in candidates:
        contact_key = tuple(
            (item["value"], item["area"], item["shape_id"])
            for item in candidate["contact"]
        )
        key = (
            candidate["level"],
            contact_key,
            candidate["effect"]["kind"],
            candidate["contact_support"],
            candidate["contact_opportunities"],
            candidate["contact_rate"],
        )
        group = groups.setdefault(key, {
            "level": candidate["level"],
            "contact": candidate["contact"],
            "effect_kind": candidate["effect"]["kind"],
            "contact_support": candidate["contact_support"],
            "contact_opportunities": candidate["contact_opportunities"],
            "contact_rate": candidate["contact_rate"],
            "targets": [],
        })
        group["targets"].append({
            "object_id": candidate["effect"]["target_object_id"],
            "value": candidate["effect"]["target_value"],
            "baseline_support": candidate["baseline_support"],
            "baseline_opportunities": candidate["baseline_opportunities"],
            "baseline_rate": candidate["baseline_rate"],
            "rate_excess": candidate["rate_excess"],
        })
    output = list(groups.values())
    for group in output:
        group["targets"].sort(key=lambda item: item["object_id"])
    output.sort(
        key=lambda item: (
            item["effect_kind"] != "unclassified_form_change",
            item["contact_support"],
            len(item["targets"]),
        ),
        reverse=True,
    )
    return output


def evaluate(trace_path: Path, output_path: Path) -> dict[str, object]:
    parser = ObservationParser()
    tracker = ObjectTracker()
    detector = EventDetector()
    coupling = ActionCouplingMemory()
    control_belief = ControlledEntityBelief()
    interaction_detector = InteractionDetector()
    effect_memory = ConditionalEffectMemory()
    events = []
    interactions = []
    detection_counts = []
    active_affordances: dict[tuple, object] = {}
    affordance_activations = 0
    affordance_retractions = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output:
        for index, trace_step in enumerate(iter_trace_steps(trace_path), start=1):
            if index == 1:
                initial_scene = parser.parse(trace_step.before, frame_index=0)
                tracker.update(initial_scene)
                detection_counts.append(len(initial_scene.detections))
            scene = parser.parse(trace_step.after, frame_index=index)
            force_new = (
                trace_step.after.levels_completed != trace_step.before.levels_completed
                or trace_step.after.full_reset
            )
            tracks = tracker.update(scene, force_new=force_new)
            event = detector.detect(
                step=index,
                before=trace_step.before,
                action=trace_step.action,
                after=trace_step.after,
                tracks=tracks,
            )
            coupling.observe(event)
            prior_control = control_belief.active
            expected_translation = control_belief.expected_delta(event.action)
            interaction = interaction_detector.detect(
                event=event,
                tracks=tracks,
                control=prior_control,
                expected_translation=expected_translation,
            )
            effect_evidence = None
            if interaction is not None:
                effect_evidence = extract_effect_evidence(event, tracks, interaction)
                if effect_evidence is not None:
                    effect_memory.observe(effect_evidence)
            current_affordances = effect_memory.affordances()
            current_by_key = {
                _affordance_key(item): item for item in current_affordances
            }
            affordance_changes = tuple(
                ("retracted", active_affordances[key])
                for key in sorted(
                    active_affordances.keys() - current_by_key.keys(),
                    key=repr,
                )
            ) + tuple(
                ("activated", current_by_key[key])
                for key in sorted(
                    current_by_key.keys() - active_affordances.keys(),
                    key=repr,
                )
            )
            affordance_activations += sum(
                status == "activated" for status, _item in affordance_changes
            )
            affordance_retractions += sum(
                status == "retracted" for status, _item in affordance_changes
            )
            active_affordances = current_by_key
            control = control_belief.observe(event)
            events.append(event)
            if interaction is not None:
                interactions.append(interaction)
            detection_counts.append(len(scene.detections))
            output.write(json.dumps(
                _event_record(
                    event,
                    detections=len(scene.detections),
                    control=control,
                    interaction=interaction,
                    affordance_changes=affordance_changes,
                ),
                separators=(",", ":"),
                sort_keys=True,
            ))
            output.write("\n")

    summaries = coupling.summaries(minimum_moves=8)
    group_summaries = coupling.co_motion_summaries(minimum_moves=8)
    control_discoveries = [_control_record(item) for item in control_belief.discoveries]
    conditional_effects = effect_memory.summaries()
    conditional_effect_groups = _group_conditional_effects(conditional_effects)
    affordances = effect_memory.affordances()
    affordance_records = [_affordance_record(item) for item in affordances]
    return {
        "trace": str(trace_path),
        "event_artifact": str(output_path),
        "transitions": len(events),
        "final_state": events[-1].state_after if events else None,
        "levels_completed": events[-1].levels_after if events else 0,
        "completion_events": sum(event.level_delta > 0 for event in events),
        "full_reset_events": sum(event.full_reset for event in events),
        "events_with_pixel_change": sum(event.changed_pixels > 0 for event in events),
        "events_with_motion": sum(bool(event.motion_groups) for event in events),
        "events_with_appearance_change": sum(
            any(delta.appearance_changed for delta in event.object_deltas)
            for event in events
        ),
        "events_with_stationary_appearance_change": sum(
            any(delta.appearance_changed and not delta.moved for delta in event.object_deltas)
            for event in events
        ),
        "events_with_reappearance": sum(
            bool(event.reappeared_object_ids) for event in events
        ),
        "detections": {
            "minimum": min(detection_counts, default=0),
            "median": statistics.median(detection_counts) if detection_counts else 0,
            "maximum": max(detection_counts, default=0),
        },
        "action_coupled_objects": len(summaries),
        "top_action_coupled_objects": summaries[:12],
        "action_coupled_groups": len(group_summaries),
        "top_action_coupled_groups": group_summaries[:12],
        "control_discoveries": control_discoveries,
        "levels_with_control_discovery": sorted({
            item.level for item in control_belief.discoveries
        }),
        "interaction_events": len(interactions),
        "contact_candidate_events": sum(
            bool(item.contact_candidates) for item in interactions
        ),
        "blocked_interaction_events": sum(item.blocked for item in interactions),
        "interaction_events_with_other_change": sum(
            bool(item.changed_other_object_ids) for item in interactions
        ),
        "conditional_effect_candidates": conditional_effects,
        "conditional_effect_candidate_count": len(conditional_effects),
        "classified_conditional_effect_candidate_count": sum(
            item["effect"]["kind"] != "unclassified_form_change"
            for item in conditional_effects
        ),
        "conditional_effect_groups": conditional_effect_groups,
        "conditional_effect_group_count": len(conditional_effect_groups),
        "affordances": affordance_records,
        "affordance_count": len(affordances),
        "affordance_activations": affordance_activations,
        "affordance_retractions": affordance_retractions,
        "classified_affordance_count": sum(
            item.effect_kind != "unclassified_form_change"
            for item in affordances
        ),
        "affordances_with_state_transitions": sum(
            bool(item.state_model.transitions) for item in affordances
        ),
        "observed_closed_state_cycles": sum(
            item.state_model.closed_cycle for item in affordances
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    summary = evaluate(args.trace, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.summary.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
