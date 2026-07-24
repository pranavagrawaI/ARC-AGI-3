"""Mechanically assess whether executable reconstruction was actually used."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, Sequence


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_ENTRYPOINTS = {"parse_observation", "predict", "update"}


def assess_reconstruction_compliance(
    events: Sequence[Mapping[str, Any]],
    *,
    eligible_steps: Iterable[int],
) -> dict[str, Any]:
    """Return the frozen activation conjunction for one persisted replay run."""

    steps = tuple(eligible_steps)
    if not steps or len(set(steps)) != len(steps) or any(
        isinstance(step, bool) or not isinstance(step, int) or step < 1 for step in steps
    ):
        raise ValueError("eligible_steps must be unique positive integers")
    if tuple(sorted(steps)) != steps:
        raise ValueError("eligible_steps must be sorted")

    ordered = list(events)
    raw_indexes = [event.get("event_index") for event in ordered]
    indexes: list[int] = []
    for index in raw_indexes:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("every compliance event requires an integer event_index")
        indexes.append(index)
    if indexes != sorted(indexes) or len(set(indexes)) != len(indexes):
        raise ValueError("compliance event_index values must be unique and increasing")

    loaded = [
        event
        for event in ordered
        if event.get("type") == "artifact_loaded"
        and event.get("ok") is True
        and isinstance(event.get("artifact_sha256"), str)
        and _SHA256.fullmatch(event["artifact_sha256"])
    ]
    loaded_digests = {event["artifact_sha256"] for event in loaded}
    artifact_loaded = bool(loaded)
    required_entrypoints = any(
        isinstance(event.get("entrypoints"), list)
        and _REQUIRED_ENTRYPOINTS.issubset(event["entrypoints"])
        for event in loaded
    )

    commitments = _by_step(ordered, "prediction_committed", steps)
    reveals = _by_step(ordered, "observation_revealed", steps)
    updates = _by_step(ordered, "model_updated", steps)

    prediction_coverage = all(commitments[step] for step in steps)
    prospective_commitments = all(
        commitments[step]
        and reveals[step]
        and min(event["event_index"] for event in commitments[step])
        < min(event["event_index"] for event in reveals[step])
        for step in steps
    )
    nonempty_predictions = prediction_coverage and all(
        isinstance(event.get("prediction"), Mapping) and bool(event["prediction"])
        for step in steps
        for event in commitments[step]
    )
    update_coverage = all(
        reveals[step]
        and updates[step]
        and max(event["event_index"] for event in updates[step])
        > min(event["event_index"] for event in reveals[step])
        and any(event.get("ok") is True for event in updates[step])
        for step in steps
    )
    artifact_identity_bound = prediction_coverage and bool(loaded_digests) and all(
        event.get("artifact_sha256") in loaded_digests
        for step in steps
        for event in commitments[step]
    )

    action_sensitive_predictions = False
    for event in ordered:
        if event.get("type") != "candidate_predictions":
            continue
        predictions = event.get("predictions")
        if not isinstance(predictions, Mapping) or len(predictions) < 2:
            continue
        canonical = {
            json.dumps(value, separators=(",", ":"), sort_keys=True)
            for value in predictions.values()
        }
        if len(canonical) >= 2:
            action_sensitive_predictions = True
            break

    criteria = {
        "artifact_loaded": artifact_loaded,
        "required_entrypoints": required_entrypoints,
        "prospective_commitments": prospective_commitments,
        "prediction_coverage": prediction_coverage,
        "nonempty_predictions": nonempty_predictions,
        "action_sensitive_predictions": action_sensitive_predictions,
        "update_coverage": update_coverage,
        "artifact_identity_bound": artifact_identity_bound,
    }
    return {
        "schema": "reconstruction-uptake-compliance-v1",
        "eligible_steps": list(steps),
        "criteria": criteria,
        "activated": all(criteria.values()),
    }


def _by_step(
    events: Sequence[Mapping[str, Any]],
    event_type: str,
    steps: tuple[int, ...],
) -> dict[int, list[Mapping[str, Any]]]:
    result = {step: [] for step in steps}
    for event in events:
        step = event.get("step")
        if event.get("type") == event_type and step in result:
            result[step].append(event)
    return result
