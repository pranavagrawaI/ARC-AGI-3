from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SOURCE = (
    REPO
    / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710/"
    "ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
)
EXPECTED = "434bd2f66b5e77b6aaa3bc1ed9cb0d5e28b0070ccaf93c5ebb077eacec5bf5b4"
PROPOSAL_SOURCE = (
    REPO
    / "experiments/hypothesis/0002-decision-relevant-reconstruction/fixtures/"
    "ls20-level1-step13-v1.json"
)

sys.path.insert(0, str(ROOT))
from game_judgment import CONTRACT, canonical_bytes  # noqa: E402
from game_judgment.machinery import action_id, compile_prefix  # noqa: E402


PALETTE = [
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (170, 170, 170),
    (240, 18, 190),
    (255, 133, 27),
    (127, 219, 255),
    (135, 12, 37),
]

CASE_SPECS = [
    {
        "case_id": "execution-control",
        "frontier": 3,
        "burden": "execution",
        "goal": {"kind": "relative_position", "delta": [0, -5]},
        "plan_id": "route-left-control",
        "max_depth": 1,
        "max_expansions": 4,
        "reflection": (
            "A verified one-step translation route is supplied; serialize its legal head "
            "and immediate public prediction."
        ),
    },
    {
        "case_id": "planning-control",
        "frontier": 11,
        "burden": "planning",
        "goal": {"kind": "absolute_position", "position": [10, 34]},
        "plan_id": "route-to-supplied-goal",
        "max_depth": 3,
        "max_expansions": 64,
        "reflection": (
            "The current tracked position and target position are supplied; construct a "
            "replay-covered route."
        ),
    },
    {
        "case_id": "game-judgment",
        "frontier": 13,
        "burden": "judgment",
        "discriminator": "D-cycle-delta-at-objective-entry",
        "reflection": (
            "Two replay-plausible latent update programs disagree on public progress at the "
            "next objective entry; preserve both and probe."
        ),
    },
]

WIDENED_CASE_SPECS = [
    *CASE_SPECS,
    {
        "case_id": "execution-control-left",
        "frontier": 5,
        "burden": "execution",
        "goal": {"kind": "relative_position", "delta": [-5, 0]},
        "plan_id": "route-left-control-frontier-5",
        "max_depth": 1,
        "max_expansions": 4,
        "reflection": "A replay-covered one-step left route is supplied at a later frontier.",
    },
    {
        "case_id": "planning-control-down",
        "frontier": 9,
        "burden": "planning",
        "goal": {"kind": "absolute_position", "position": [25, 34]},
        "plan_id": "route-down-to-supplied-goal",
        "max_depth": 2,
        "max_expansions": 32,
        "reflection": "Construct a two-step route using a separately replay-covered action.",
    },
]


def write(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def render_grid(grid: list[list[int]], path: Path) -> tuple[int, int]:
    image = Image.new("RGB", (len(grid[0]) * 5, len(grid) * 5))
    pixels = image.load()
    for grid_y, row in enumerate(grid):
        for grid_x, color in enumerate(row):
            rgb = PALETTE[int(color) % len(PALETTE)]
            for image_y in range(grid_y * 5, grid_y * 5 + 5):
                for image_x in range(grid_x * 5, grid_x * 5 + 5):
                    pixels[image_x, image_y] = rgb
    image.save(path)
    return image.size


def changed_cells(before: list[list[int]], after: list[list[int]]) -> int:
    return sum(
        before_cell != after_cell
        for before_row, after_row in zip(before, after)
        for before_cell, after_cell in zip(before_row, after_row)
    )


def frame_digest(frame: list[list[list[int]]]) -> str:
    return sha256(canonical_bytes(frame)).hexdigest()


def build_live_initial_case(
    rows: list[dict[str, Any]], proposal: dict[str, Any], source: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    current, successor = rows[1], rows[2]
    grounding = next(
        row for row in proposal["groundings"] if row["kind"] == "tracked_candidate"
    )
    current_bbox = grounding["occurrences"]["1"]
    successor_bbox = grounding["occurrences"]["2"]
    current_position = [current_bbox[0], current_bbox[2]]
    successor_position = [successor_bbox[0], successor_bbox[2]]
    if action_id(successor) != 3 or successor_position != [45, 29]:
        raise RuntimeError("frozen live successor no longer matches ACTION3 certificate")
    image_path = ROOT / "fixtures/images/live-initial-step-1.png"
    image_size = render_grid(current["frame"][-1], image_path)
    dependencies = ["C-live-translation", "C-live-public-persistence"]
    case = {
        "schema": "0003-decision-case-v2",
        "case_id": "live-initial-control",
        "burden": "execution",
        "source": source,
        "proposal_provenance": {
            "authority_label": "historical_replay_certificate",
            "frontier": 1,
        },
        "legal_actions": current["available_actions"],
        "current_public": {
            field: current[field]
            for field in ("state", "levels_completed", "win_levels", "available_actions")
        },
        "images": [{
            "step": 1,
            "path": str(image_path.relative_to(ROOT)),
            "sha256": sha256(image_path.read_bytes()).hexdigest(),
            "size": list(image_size),
        }],
        "step_labels": [1],
        "numeric_transitions": [],
        "reflection": "Select the head of a separately replay-verified initial-state route.",
        "reconstruction_authority": "historical_replay_certificate",
        "candidate_scene": {
            "schema": "0003-candidate-scene-v1",
            "evidence_frontier": 1,
            "neutral_tracks": [{"track_id": "track-0", "occurrences": [{"step": 1, "position": current_position}], "observed_patch_signatures": 1}],
            "regions": [],
        },
        "executable_conjectures": [
            {"id": dependencies[0], "class": "transition", "program": {"kind": "action_translation", "action_deltas": {"3": [0, -5]}}, "replay": {"status": "working", "supports": [2], "contradictions": [], "uncovered": [], "covered_actions": [3]}},
            {"id": dependencies[1], "class": "transition", "program": {"kind": "public_field_persistence", "field": "levels_completed"}, "replay": {"status": "working", "supports": [2], "contradictions": [], "uncovered": []}},
        ],
        "replay_status": {dependency: "working" for dependency in dependencies},
        "candidate_plans": [{
            "plan_id": "live-initial-action3",
            "actions": [3],
            "predicted_positions": [current_position, successor_position],
            "predicted_public_fields": {"public.levels_completed": successor["levels_completed"]},
            "conjecture_ids": dependencies,
            "dependency_statuses": {dependency: "working" for dependency in dependencies},
        }],
        "verified_route": None,
        "discriminators": [],
        "contract": CONTRACT,
    }
    label = {
        "case_id": case["case_id"],
        "protected_successor": {"public.levels_completed": successor["levels_completed"]},
        "valid_decision_signatures": [["EXECUTE", "live-initial-action3", 3]],
        "valid_abstentions": [],
        "expected_current_frame_sha256": frame_digest(current["frame"]),
        "expected_successor_frame_sha256": frame_digest(successor["frame"]),
    }
    return case, label


def build_route_ceiling_case(
    rows: list[dict[str, Any]],
    step: int,
    source: dict[str, Any],
    namespace: str = "route-ceiling",
) -> tuple[dict[str, Any], dict[str, Any]]:
    current, successor = rows[step], rows[step + 1]
    action = action_id(successor)
    plan_id = f"{namespace}-step-{step}"
    conjecture_id = (
        f"C-historical-replay-step-{step}"
        if namespace == "route-ceiling"
        else f"C-{namespace}-historical-replay-step-{step}"
    )
    image_path = ROOT / f"fixtures/images/{namespace}-step-{step}.png"
    image_size = render_grid(current["frame"][-1], image_path)
    case = {
        "schema": "0003-decision-case-v2",
        "case_id": f"{namespace}-{step}",
        "burden": "execution",
        "source": source,
        "proposal_provenance": {
            "authority_label": "historical_replay_certificate",
            "frontier": step,
        },
        "legal_actions": current["available_actions"],
        "current_public": {
            field: current[field]
            for field in ("state", "levels_completed", "win_levels", "available_actions")
        },
        "images": [{
            "step": step,
            "path": str(image_path.relative_to(ROOT)),
            "sha256": sha256(image_path.read_bytes()).hexdigest(),
            "size": list(image_size),
        }],
        "step_labels": [step],
        "numeric_transitions": [],
        "reflection": "Select the sole head of the separately replay-verified route ceiling.",
        "reconstruction_authority": "historical_replay_certificate",
        "candidate_scene": {
            "schema": "0003-candidate-scene-v1",
            "evidence_frontier": step,
            "neutral_tracks": [],
            "regions": [],
        },
        "executable_conjectures": [{
            "id": conjecture_id,
            "class": "transition",
            "program": {"kind": "historical_state_action_replay", "action": action},
            "replay": {
                "status": "working",
                "supports": [step + 1],
                "contradictions": [],
                "uncovered": [],
                "covered_actions": [action],
            },
        }],
        "replay_status": {conjecture_id: "working"},
        "candidate_plans": [{
            "plan_id": plan_id,
            "actions": [action],
            "predicted_positions": [],
            "predicted_public_fields": {
                "public.levels_completed": successor["levels_completed"]
            },
            "conjecture_ids": [conjecture_id],
            "dependency_statuses": {conjecture_id: "working"},
        }],
        "verified_route": None,
        "discriminators": [],
        "contract": CONTRACT,
    }
    label = {
        "case_id": case["case_id"],
        "protected_successor": {
            "public.levels_completed": successor["levels_completed"]
        },
        "valid_decision_signatures": [["EXECUTE", plan_id, action]],
        "valid_abstentions": [],
        "expected_current_frame_sha256": frame_digest(current["frame"]),
        "expected_successor_frame_sha256": frame_digest(successor["frame"]),
    }
    return case, label


def build_case(
    prefix: list[dict[str, Any]],
    spec: dict[str, Any],
    source: dict[str, Any],
    proposal: dict[str, Any],
    proposal_provenance: dict[str, Any],
) -> dict[str, Any]:
    step = len(prefix) - 1
    current = prefix[-1]
    case_id = spec["case_id"]
    image_refs = []
    for image_step in range(step - 3, step + 1):
        path = ROOT / "fixtures" / "images" / f"{case_id}-step-{image_step}.png"
        size = render_grid(prefix[image_step]["frame"][-1], path)
        image_refs.append(
            {
                "step": image_step,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "size": list(size),
            }
        )

    transitions = []
    for after_step in range(max(2, step - 2), step + 1):
        transitions.append(
            {
                "before_step": after_step - 1,
                "after_step": after_step,
                "action": action_id(prefix[after_step]),
                "changed_cells": changed_cells(
                    prefix[after_step - 1]["frame"][-1],
                    prefix[after_step]["frame"][-1],
                ),
            }
        )

    compiled = compile_prefix(prefix, proposal, spec)
    return {
        "schema": "0003-decision-case-v2",
        "case_id": case_id,
        "burden": spec["burden"],
        "source": source,
        "proposal_provenance": proposal_provenance,
        "legal_actions": current["available_actions"],
        "current_public": {
            field: current[field]
            for field in ("state", "levels_completed", "win_levels", "available_actions")
        },
        "images": image_refs,
        "step_labels": [row["step"] for row in image_refs],
        "numeric_transitions": transitions,
        "reflection": spec["reflection"],
        **compiled,
        "contract": CONTRACT,
    }


def build_label(case: dict[str, Any], successor: dict[str, Any]) -> dict[str, Any]:
    successor_action = action_id(successor)
    signatures = [
        ["EXECUTE", plan["plan_id"], successor_action]
        for plan in case["candidate_plans"]
        if plan["actions"][0] == successor_action
    ]
    signatures.extend(
        ["PROBE", discriminator["discriminator_id"], successor_action]
        for discriminator in case["discriminators"]
        if discriminator["action"] == successor_action
    )

    valid_abstentions = []
    if case["burden"] == "judgment":
        abstention = {
            "reason_code": "insufficient_evidence",
            "missing_evidence": ["cycle_delta"],
        }
        valid_abstentions.append(abstention)
        signatures.append(["ABSTAIN", abstention["reason_code"]])
    return {
        "case_id": case["case_id"],
        "protected_successor": {
            "public.levels_completed": successor["levels_completed"]
        },
        "valid_decision_signatures": signatures,
        "valid_abstentions": valid_abstentions,
    }


def main() -> int:
    if sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError("source digest mismatch")
    rows = [json.loads(line)["data"] for line in SOURCE.read_text().splitlines()]
    images = ROOT / "fixtures" / "images"
    scorer = ROOT / "scorer"
    images.mkdir(parents=True, exist_ok=True)
    scorer.mkdir(parents=True, exist_ok=True)
    expected_images = {
        f"{spec['case_id']}-step-{step}.png"
        for spec in WIDENED_CASE_SPECS
        for step in range(spec["frontier"] - 3, spec["frontier"] + 1)
    }
    for existing in images.glob("*.png"):
        if existing.name not in expected_images:
            existing.unlink()

    proposal_document = json.loads(PROPOSAL_SOURCE.read_text())
    proposal = proposal_document["oracle_proposal"]
    proposal_provenance = {
        "authority_label": "oracle_coach_prefix_proposal",
        "path": str(PROPOSAL_SOURCE.relative_to(REPO)),
        "sha256": sha256(PROPOSAL_SOURCE.read_bytes()).hexdigest(),
    }
    for specs, batch_name, labels_name in (
        (CASE_SPECS, "batch.json", "labels.json"),
        (WIDENED_CASE_SPECS, "widened-batch.json", "widened-labels.json"),
    ):
        cases = []
        labels = []
        for spec in specs:
            step = spec["frontier"]
            source = {
                "path": str(SOURCE.relative_to(REPO)),
                "sha256": EXPECTED,
                "frontier": step,
            }
            case = build_case(
                rows[: step + 1], spec, source, proposal, proposal_provenance
            )
            cases.append(case)
            labels.append(build_label(case, rows[step + 1]))
        write(
            ROOT / "fixtures" / batch_name,
            {
                "schema": "0003-frozen-batch-v2",
                "source_sha256": EXPECTED,
                "proposal_provenance": proposal_provenance,
                "cases": cases,
            },
        )
        write(
            ROOT / "scorer" / labels_name,
            {"schema": "0003-scorer-labels-v2", "labels": labels},
        )
    live_case, live_label = build_live_initial_case(
        rows,
        proposal,
        {"path": str(SOURCE.relative_to(REPO)), "sha256": EXPECTED, "frontier": 1},
    )
    write(ROOT / "fixtures/live-initial-case.json", live_case)
    write(ROOT / "scorer/live-initial-label.json", live_label)
    route_cases = []
    route_labels = []
    for step in range(1, 14):
        route_case, route_label = build_route_ceiling_case(
            rows,
            step,
            {"path": str(SOURCE.relative_to(REPO)), "sha256": EXPECTED, "frontier": step},
        )
        route_cases.append(route_case)
        route_labels.append(route_label)
    write(ROOT / "fixtures/live-route-ceiling.json", {
        "schema": "0003-live-route-ceiling-v1",
        "authority": "historical_replay_ceiling_not_autonomous_reconstruction",
        "cases": route_cases,
    })
    write(ROOT / "scorer/live-route-ceiling-labels.json", {
        "schema": "0003-live-route-ceiling-labels-v1",
        "labels": route_labels,
    })
    full_route_cases = []
    full_route_labels = []
    for step in range(1, len(rows) - 1):
        route_case, route_label = build_route_ceiling_case(
            rows,
            step,
            {"path": str(SOURCE.relative_to(REPO)), "sha256": EXPECTED, "frontier": step},
            namespace="full-route-ceiling",
        )
        full_route_cases.append(route_case)
        full_route_labels.append(route_label)
    write(ROOT / "fixtures/live-full-route-ceiling.json", {
        "schema": "0003-live-full-route-ceiling-v1",
        "authority": "historical_replay_ceiling_not_autonomous_reconstruction",
        "natural_stop": "WIN",
        "cases": full_route_cases,
    })
    write(ROOT / "scorer/live-full-route-ceiling-labels.json", {
        "schema": "0003-live-full-route-ceiling-labels-v1",
        "labels": full_route_labels,
    })
    print(ROOT / "fixtures" / "batch.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
