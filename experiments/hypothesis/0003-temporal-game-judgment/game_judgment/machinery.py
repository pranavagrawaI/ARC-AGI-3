"""Prefix-only replay and planning machinery for the 0003 decision cases."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any


class MachineryError(ValueError):
    pass


def discriminator_is_decision_relevant(discriminator: dict[str, Any]) -> bool:
    """Return whether the candidate predicts at least two distinct outcomes."""

    outcomes = discriminator.get("predicted_outcomes")
    if not isinstance(outcomes, dict) or len(outcomes) < 2:
        return False
    predictions = list(outcomes.values())
    return any(prediction != predictions[0] for prediction in predictions[1:])


def action_id(row: dict[str, Any]) -> int:
    raw = str(row["action_input"]["id"])
    if not raw.startswith("ACTION") or not raw[6:].isdigit():
        raise MachineryError(f"invalid action id {raw!r}")
    return int(raw[6:])


def _bbox(value: Any) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise MachineryError(f"invalid bbox {value!r}")
    return tuple(value)


def _position(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return bbox[0], bbox[2]


def _patch(frame: list[list[int]], bbox: tuple[int, int, int, int]) -> list[list[int]]:
    top, bottom, left, right = bbox
    if not (0 <= top < bottom <= len(frame) and 0 <= left < right <= len(frame[0])):
        raise MachineryError(f"bbox outside frame: {bbox}")
    return [row[left:right] for row in frame[top:bottom]]


def _tracked_positions(
    prefix: list[dict[str, Any]], proposal: dict[str, Any]
) -> tuple[dict[int, tuple[int, int]], dict[str, Any]]:
    tracked = [row for row in proposal["groundings"] if row["kind"] == "tracked_candidate"]
    if len(tracked) != 1:
        raise MachineryError("exactly one tracked candidate is required")
    grounding = tracked[0]
    positions: dict[int, tuple[int, int]] = {}
    signatures: set[tuple[tuple[int, ...], ...]] = set()
    for step in range(1, len(prefix)):
        try:
            bbox = _bbox(grounding["occurrences"][str(step)])
        except KeyError as exc:
            raise MachineryError(f"tracked candidate has no prefix occurrence at step {step}") from exc
        patch = _patch(prefix[step]["frame"][-1], bbox)
        if not any(cell != 0 for row in patch for cell in row):
            raise MachineryError(f"tracked candidate at step {step} is empty")
        signatures.add(tuple(tuple(row) for row in patch))
        positions[step] = _position(bbox)
    return positions, {
        "track_id": "track-0",
        "occurrences": [
            {"step": step, "position": list(position)}
            for step, position in sorted(positions.items())
        ],
        "observed_patch_signatures": len(signatures),
    }


def _translation_program(proposal: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in proposal["transitions"] if row["kind"] == "action_translation"]
    if len(rows) != 1:
        raise MachineryError("exactly one action_translation program is required")
    program = deepcopy(rows[0])
    program["action_deltas"] = {
        str(action): [int(delta[0]), int(delta[1])]
        for action, delta in program["action_deltas"].items()
    }
    return program


def _replay_translation(
    prefix: list[dict[str, Any]],
    positions: dict[int, tuple[int, int]],
    program: dict[str, Any],
) -> dict[str, Any]:
    supports: list[int] = []
    contradictions: list[int] = []
    rows: list[dict[str, Any]] = []
    for step in range(2, len(prefix)):
        action = action_id(prefix[step])
        delta = program["action_deltas"].get(str(action))
        if delta is None:
            contradictions.append(step)
            continue
        before = positions[step - 1]
        predicted = (before[0] + delta[0], before[1] + delta[1])
        actual = positions[step]
        matched = predicted == actual
        (supports if matched else contradictions).append(step)
        rows.append(
            {
                "after_step": step,
                "action": action,
                "before_position": list(before),
                "predicted_position": list(predicted),
                "actual_position": list(actual),
                "result": "support" if matched else "contradiction",
            }
        )
    return {
        "status": "working" if supports and not contradictions else "contradicted",
        "supports": supports,
        "contradictions": contradictions,
        "uncovered": [],
        "covered_actions": sorted({row["action"] for row in rows if row["result"] == "support"}),
        "rows": rows,
    }


def _replay_public_persistence(prefix: list[dict[str, Any]]) -> dict[str, Any]:
    supports: list[int] = []
    contradictions: list[int] = []
    for step in range(2, len(prefix)):
        before = prefix[step - 1]["levels_completed"]
        after = prefix[step]["levels_completed"]
        (supports if before == after else contradictions).append(step)
    return {
        "status": "working" if supports and not contradictions else "contradicted",
        "supports": supports,
        "contradictions": contradictions,
        "uncovered": [],
    }


def _unresolved_programs(
    prefix: list[dict[str, Any]], proposal: dict[str, Any]
) -> list[dict[str, Any]]:
    cycle = next(row for row in proposal["transitions"] if row["kind"] == "advance_on_entry")
    competitor = proposal["competitor"]
    objective = next(row for row in proposal["objectives"] if row["kind"] == "entry_with_equality")
    event_step = 7 if len(prefix) > 7 else len(prefix) - 1
    return [
        {
            "id": cycle["id"],
            "class": "transition",
            "program": deepcopy(cycle),
            "replay": {"status": "unresolved", "supports": [], "contradictions": [], "uncovered": [event_step]},
        },
        {
            "id": competitor["id"],
            "class": "transition",
            "program": deepcopy(competitor),
            "replay": {"status": "unresolved", "supports": [], "contradictions": [], "uncovered": [event_step]},
        },
        {
            "id": objective["id"],
            "class": "objective",
            "program": deepcopy(objective),
            "replay": {"status": "unresolved", "supports": [], "contradictions": [], "uncovered": [len(prefix) - 1]},
        },
    ]


def _route(
    start: tuple[int, int],
    target: tuple[int, int],
    legal_actions: list[int],
    action_deltas: dict[str, list[int]],
    max_depth: int,
    max_expansions: int,
) -> tuple[list[int], list[list[int]], int]:
    queue = deque([(start, [], [list(start)])])
    seen = {start}
    expansions = 0
    while queue:
        position, actions, states = queue.popleft()
        if position == target and actions:
            return actions, states, expansions
        if len(actions) >= max_depth:
            continue
        for action in sorted(legal_actions):
            delta = action_deltas.get(str(action))
            if delta is None:
                continue
            if expansions >= max_expansions:
                raise MachineryError("route search exceeded expansion budget")
            expansions += 1
            successor = position[0] + delta[0], position[1] + delta[1]
            if successor in seen:
                continue
            seen.add(successor)
            queue.append((successor, [*actions, action], [*states, list(successor)]))
    raise MachineryError("no route found within the declared budget")


def compile_prefix(
    prefix: list[dict[str, Any]], proposal: dict[str, Any], case_spec: dict[str, Any]
) -> dict[str, Any]:
    """Compile and execute only against observations at or before the supplied frontier."""

    if len(prefix) < 4:
        raise MachineryError("at least four prefix observations are required")
    positions, track = _tracked_positions(prefix, proposal)
    translation = _translation_program(proposal)
    translation_replay = _replay_translation(prefix, positions, translation)
    persistence_replay = _replay_public_persistence(prefix)
    conjectures = [
        {
            "id": translation["id"],
            "class": "transition",
            "program": translation,
            "replay": translation_replay,
        },
        {
            "id": "C-public-persistence",
            "class": "transition",
            "program": {"kind": "public_field_persistence", "field": "levels_completed"},
            "replay": persistence_replay,
        },
        *_unresolved_programs(prefix, proposal),
    ]
    status = {row["id"]: row["replay"]["status"] for row in conjectures}
    current = prefix[-1]
    plans: list[dict[str, Any]] = []
    goal = case_spec.get("goal")
    if goal is not None:
        start = positions[len(prefix) - 1]
        if goal["kind"] == "relative_position":
            target = start[0] + goal["delta"][0], start[1] + goal["delta"][1]
        elif goal["kind"] == "absolute_position":
            target = tuple(goal["position"])
        else:
            raise MachineryError(f"unknown goal kind {goal['kind']!r}")
        dependencies = (translation["id"], "C-public-persistence")
        if any(status[dependency] != "working" for dependency in dependencies):
            raise MachineryError("plan dependency is not replay-working")
        actions, states, expansions = _route(
            start,
            target,
            current["available_actions"],
            translation["action_deltas"],
            int(case_spec["max_depth"]),
            int(case_spec["max_expansions"]),
        )
        uncovered_actions = sorted(set(actions) - set(translation_replay["covered_actions"]))
        if uncovered_actions:
            raise MachineryError(f"route uses uncovered actions: {uncovered_actions}")
        plans.append(
            {
                "plan_id": case_spec["plan_id"],
                "actions": actions,
                "predicted_positions": states,
                "predicted_public_fields": {"public.levels_completed": current["levels_completed"]},
                "conjecture_ids": list(dependencies),
                "dependency_statuses": {dependency: status[dependency] for dependency in dependencies},
                "search": {"max_depth": case_spec["max_depth"], "max_expansions": case_spec["max_expansions"], "realized_expansions": expansions},
            }
        )

    discriminators: list[dict[str, Any]] = []
    if case_spec.get("discriminator"):
        cycle = next(row for row in proposal["transitions"] if row["kind"] == "advance_on_entry")
        competitor = proposal["competitor"]
        objective = next(row for row in proposal["objectives"] if row["kind"] == "entry_with_equality")
        entry_count = 1 if len(prefix) > 7 else 0
        values = {
            cycle["id"]: (cycle["initial"] + entry_count * cycle["delta"]) % cycle["modulus"],
            competitor["id"]: (competitor["initial"] + entry_count * competitor["delta"]) % competitor["modulus"],
        }
        outcomes = {
            conjecture_id: {
                "public.levels_completed": current["levels_completed"]
                + (objective["public_effect"]["delta"] if value == objective["right"] else 0)
            }
            for conjecture_id, value in values.items()
        }
        candidate_discriminator = {
            "discriminator_id": case_spec["discriminator"],
            "distinguishes": [cycle["id"], competitor["id"]],
            "predicted_outcomes": outcomes,
        }
        if not discriminator_is_decision_relevant(candidate_discriminator):
            raise MachineryError("declared discriminator does not distinguish outcomes")
        target = _position(_bbox(next(row for row in proposal["groundings"] if row["symbol"] == objective["region"])["bbox"]))
        actions, _, _ = _route(
            positions[len(prefix) - 1], target, current["available_actions"], translation["action_deltas"], 1, 4
        )
        candidate_discriminator["action"] = actions[0]
        discriminators.append(candidate_discriminator)

    return {
        "reconstruction_authority": "oracle_coach_prefix_proposal",
        "candidate_scene": {
            "schema": "0003-candidate-scene-v1",
            "evidence_frontier": len(prefix) - 1,
            "neutral_tracks": [track],
            "regions": [
                {"region_id": f"region-{index}", "bbox": grounding["bbox"]}
                for index, grounding in enumerate(
                    row for row in proposal["groundings"] if row["kind"] == "region"
                )
            ],
        },
        "executable_conjectures": conjectures,
        "replay_status": status,
        "candidate_plans": plans,
        "verified_route": plans[0] if plans else None,
        "discriminators": discriminators,
    }
