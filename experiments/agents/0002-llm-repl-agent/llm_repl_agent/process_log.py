"""Policy-independent process evidence and human annotation scaffolding."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


DISCOVERY_KINDS = (
    "controllable_entity",
    "relevant_state_variables",
    "action_semantics",
    "goal_and_failure_conditions",
    "compact_predictive_world_model",
    "reusable_procedure",
    "final_successful_plan",
)

WASTED_ACTION_KINDS = (
    "perception_failure",
    "object_identity_drift",
    "rediscovering_known_mechanics",
    "maintaining_contradictory_hypotheses",
    "premature_commitment",
    "continued_exploration_after_sufficient_understanding",
    "correct_theory_faulty_execution",
    "correct_local_mechanics_failed_cross_level_composition",
)


def _without_data_urls(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("data:image/"):
        return f"<image-data-url:{len(value)}-chars>"
    if isinstance(value, Mapping):
        return {str(key): _without_data_urls(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_without_data_urls(item) for item in value]
    return value


class ProcessLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("w", encoding="utf-8")
        self._event_index = 0

    def write(self, event_type: str, **payload: Any) -> int:
        self._event_index += 1
        record = {
            "event_index": self._event_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            **_without_data_urls(payload),
        }
        self._stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        self._stream.write("\n")
        self._stream.flush()
        return self._event_index

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "ProcessLog":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def annotation_template(*, actions: int) -> dict[str, Any]:
    return {
        "instructions": (
            "Annotate from process.jsonl and the workspace snapshot after the policy run. "
            "Do not infer discovery from the final score alone. Evidence must name event indices."
        ),
        "human_compact_description": None,
        "discoveries": {
            kind: {
                "status": "unreviewed",
                "first_action": None,
                "evidence_event_indices": [],
                "description": None,
            }
            for kind in DISCOVERY_KINDS
        },
        "representation_snapshots": [],
        "representation_latency_actions": None,
        "representation_stability": {
            "status": "unreviewed",
            "unsupported_mutation_count": None,
            "mutations": [],
        },
        "wasted_action_categories": list(WASTED_ACTION_KINDS),
        "wasted_actions": [
            {
                "action": index,
                "classification": "unreviewed",
                "category": None,
                "evidence_event_indices": [],
                "notes": None,
            }
            for index in range(1, actions + 1)
        ],
    }


def write_annotation_template(path: Path, *, actions: int) -> None:
    path.write_text(
        json.dumps(annotation_template(actions=actions), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def measurement_summary(
    annotation: Mapping[str, Any],
    *,
    actions: int,
    event_indices: set[int],
) -> dict[str, Any]:
    """Validate a completed human annotation and derive compact run measurements."""

    human_description = annotation.get("human_compact_description")
    if not isinstance(human_description, str) or not human_description.strip():
        raise ValueError("human_compact_description must be filled")
    raw_discoveries = annotation.get("discoveries")
    if not isinstance(raw_discoveries, Mapping):
        raise ValueError("discoveries must be an object")
    discoveries: dict[str, Any] = {}
    for kind in DISCOVERY_KINDS:
        record = raw_discoveries.get(kind)
        if not isinstance(record, Mapping):
            raise ValueError(f"missing discovery annotation: {kind}")
        status = record.get("status")
        if status not in {"observed", "not_observed"}:
            raise ValueError(f"discovery {kind} is still unreviewed")
        first_action = record.get("first_action")
        if status == "observed" and (
            isinstance(first_action, bool) or not isinstance(first_action, int)
        ):
            raise ValueError(f"observed discovery {kind} requires first_action")
        if status == "observed":
            if not 0 <= first_action <= actions:
                raise ValueError(f"discovery {kind} first_action is outside the run")
            description = record.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"observed discovery {kind} requires a description")
            _validate_evidence(
                record.get("evidence_event_indices"),
                event_indices=event_indices,
                label=f"discovery {kind}",
            )
        elif first_action is not None:
            raise ValueError(f"not-observed discovery {kind} cannot have first_action")
        discoveries[kind] = {"status": status, "first_action": first_action}

    stability = annotation.get("representation_stability")
    if not isinstance(stability, Mapping) or stability.get("status") not in {
        "stable",
        "unstable",
        "not_reached",
    }:
        raise ValueError("representation_stability must be reviewed")
    latency = annotation.get("representation_latency_actions")
    compact_observed = discoveries["compact_predictive_world_model"]["status"] == "observed"
    if compact_observed and (isinstance(latency, bool) or not isinstance(latency, int)):
        raise ValueError("an observed compact representation requires latency in actions")
    if compact_observed and latency != discoveries["compact_predictive_world_model"]["first_action"]:
        raise ValueError("representation latency must equal compact representation first_action")
    if not compact_observed and latency is not None:
        raise ValueError("representation latency must be null when no compact representation appeared")

    mutation_count = stability.get("unsupported_mutation_count")
    if stability["status"] in {"stable", "unstable"} and (
        isinstance(mutation_count, bool)
        or not isinstance(mutation_count, int)
        or mutation_count < 0
    ):
        raise ValueError("reviewed representation stability requires a nonnegative mutation count")
    if stability["status"] == "stable" and mutation_count != 0:
        raise ValueError("stable representation must have zero unsupported mutations")
    if stability["status"] == "unstable" and mutation_count == 0:
        raise ValueError("unstable representation must have unsupported mutations")
    if stability["status"] == "not_reached" and compact_observed:
        raise ValueError("representation stability cannot be not_reached after compact discovery")
    if stability["status"] != "not_reached" and not compact_observed:
        raise ValueError("representation stability cannot be measured before compact discovery")
    mutations = stability.get("mutations")
    if not isinstance(mutations, list):
        raise ValueError("representation_stability.mutations must be a list")
    if stability["status"] == "unstable" and len(mutations) < mutation_count:
        raise ValueError("unstable representation must document each unsupported mutation")
    for index, mutation in enumerate(mutations):
        if not isinstance(mutation, Mapping):
            raise ValueError("each representation mutation must be an object")
        _validate_evidence(
            mutation.get("evidence_event_indices"),
            event_indices=event_indices,
            label=f"representation mutation {index}",
        )

    snapshots = annotation.get("representation_snapshots")
    if not isinstance(snapshots, list):
        raise ValueError("representation_snapshots must be a list")
    if compact_observed and not snapshots:
        raise ValueError("an observed compact representation requires representation snapshots")
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, Mapping):
            raise ValueError("each representation snapshot must be an object")
        action = snapshot.get("action")
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action <= actions:
            raise ValueError(f"representation snapshot {index} has invalid action")
        description = snapshot.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"representation snapshot {index} requires a description")
        _validate_evidence(
            snapshot.get("evidence_event_indices"),
            event_indices=event_indices,
            label=f"representation snapshot {index}",
        )

    raw_actions = annotation.get("wasted_actions")
    if not isinstance(raw_actions, list):
        raise ValueError("wasted_actions must be a list")
    action_ids = [record.get("action") for record in raw_actions if isinstance(record, Mapping)]
    if (
        len(action_ids) != actions
        or any(isinstance(item, bool) or not isinstance(item, int) for item in action_ids)
        or sorted(action_ids) != list(range(1, actions + 1))
    ):
        raise ValueError("wasted_actions must contain each run action exactly once")
    wasted_counts = {kind: 0 for kind in WASTED_ACTION_KINDS}
    useful_count = 0
    for record in raw_actions:
        if not isinstance(record, Mapping):
            raise ValueError("each wasted-action annotation must be an object")
        classification = record.get("classification")
        if classification == "useful":
            useful_count += 1
            continue
        if classification != "wasted":
            raise ValueError(f"action {record.get('action')} is still unreviewed")
        category = record.get("category")
        if category not in wasted_counts:
            raise ValueError(f"action {record.get('action')} has invalid wasted category")
        _validate_evidence(
            record.get("evidence_event_indices"),
            event_indices=event_indices,
            label=f"wasted action {record.get('action')}",
        )
        wasted_counts[str(category)] += 1

    return {
        "representation_latency_actions": latency,
        "representation_stability": {
            "status": stability["status"],
            "unsupported_mutation_count": mutation_count,
        },
        "discoveries": discoveries,
        "actions_reviewed": len(raw_actions),
        "useful_action_count": useful_count,
        "wasted_action_count": sum(wasted_counts.values()),
        "wasted_action_counts": wasted_counts,
    }


def _validate_evidence(value: Any, *, event_indices: set[int], label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} requires evidence_event_indices")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{label} evidence indices must be integers")
    unknown = sorted(set(value) - event_indices)
    if unknown:
        raise ValueError(f"{label} references unknown process events: {unknown}")
