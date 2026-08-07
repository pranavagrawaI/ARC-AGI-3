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
        for spec in CASE_SPECS
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
    cases = []
    labels = []
    for spec in CASE_SPECS:
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
        ROOT / "fixtures" / "batch.json",
        {
            "schema": "0003-frozen-batch-v2",
            "source_sha256": EXPECTED,
            "proposal_provenance": proposal_provenance,
            "cases": cases,
        },
    )
    write(
        ROOT / "scorer" / "labels.json",
        {"schema": "0003-scorer-labels-v2", "labels": labels},
    )
    print(ROOT / "fixtures" / "batch.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
