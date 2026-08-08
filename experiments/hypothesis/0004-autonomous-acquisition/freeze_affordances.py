"""Compile prior replay-supported trigger effects without exposing a route."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SUBSTRATE = REPO / "experiments/hypothesis/0001-inverted-authority"
RECORDING = REPO / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710/ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"


def main() -> int:
    sys.path.insert(0, str(SUBSTRATE))
    from digest_probe import digest, lenses, level_slice, load  # noqa: PLC0415
    from select_probe import oracle_pair  # noqa: PLC0415

    steps = load(RECORDING.parent)
    prior = level_slice(steps, 1)
    prior_digest = digest(prior, "colour", len(prior))
    truth = oracle_pair(prior_digest)
    if truth is None:
        raise RuntimeError("prior completed level has no unique objective pair")
    tracks = {track.tid: track for track in prior_digest["_tracks"]["all"]}
    controlled = prior_digest["_tracks"]["ctrl"]
    workspace = tracks[truth[1]]
    supports = []
    trigger_patches = []
    clockwise_turns = []
    for observation in sorted(workspace.sigs):
        if observation - 1 not in workspace.sigs or workspace.sigs[observation] == workspace.sigs[observation - 1]:
            continue
        destination = controlled.sprites[observation].bbox
        row, column = destination[0], destination[2]
        trigger = prior[observation - 1].grid[row:row + 5, column:column + 5]
        before = workspace.sprites[observation - 1].paint(prior[observation - 1].grid)
        after = workspace.sprites[observation].paint(prior[observation].grid)
        rotations = [turns for turns in (1, 2, 3) if np.array_equal(np.rot90(before, turns), after)]
        if len(rotations) != 1:
            continue
        trigger_patches.append(trigger)
        clockwise_turns.append((4 - rotations[0]) % 4)
        supports.append({"prior_level": 1, "observation": observation + 1, "action": prior[observation].action})
    if len(supports) != 3 or set(clockwise_turns) != {1} or any(not np.array_equal(trigger_patches[0], patch) for patch in trigger_patches[1:]):
        raise RuntimeError("rotation affordance no longer has three consistent public supports")

    case = json.loads((ROOT / "fixtures/planning-case.json").read_text())
    current = level_slice(steps, 2)[23].grid
    trigger = trigger_patches[0]
    positions = [
        [row, column]
        for row in range(current.shape[0] - 4)
        for column in range(current.shape[1] - 4)
        if np.array_equal(current[row:row + 5, column:column + 5], trigger)
    ]
    if positions != [[10, 49]]:
        raise RuntimeError(f"current trigger match drift: {positions}")
    target = np.asarray(case["acquired_objective"]["invariant"]["patch"])
    workspace_now = np.asarray(case["acquired_objective"]["mutable"]["current_patch"])
    applications = [
        turns for turns in range(1, 4)
        if lenses(target, np.rot90(workspace_now, -turns))["scale_normalised"]
    ]
    if applications != [2]:
        raise RuntimeError(f"objective application count drift: {applications}")
    enriched = deepcopy(case)
    enriched["authority"] = "prefix_objective_plus_prior_replay_supported_affordances_no_route"
    enriched["causal_affordances"] = [{
        "affordance_id": "affordance-0",
        "trigger_patch": trigger.tolist(),
        "current_trigger_top_left": positions[0],
        "effect_program": {"kind": "rotate_workspace_clockwise_quarter_turn", "turns_per_contact": 1},
        "objective_applications_required": applications[0],
        "replay": {"status": "working", "supports": supports, "contradictions": []},
    }]
    path = ROOT / "fixtures/planning-affordance-case.json"
    path.write_text(json.dumps(enriched, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"affordance supports={len(supports)} position={positions[0]} applications={applications[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
