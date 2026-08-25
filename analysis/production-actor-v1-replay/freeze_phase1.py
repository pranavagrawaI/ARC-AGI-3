#!/usr/bin/env python3
"""Freeze six real-workspace replay packets for the zero-cost engineering probe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.production_actor_replay import (
    ProductionReplayOracle,
    prepare_production_replay_packet,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
PACKETS = OUTPUT / "packets"


@dataclass(frozen=True)
class Case:
    packet_id: str
    failure_class: str
    workspace: str
    game_id: str
    support_sets: tuple[tuple[str, ...], ...]
    relevant: tuple[str, ...]
    distractors: tuple[str, ...]
    required_computation: str
    historical_failure: str


CASES = (
    Case(
        "remote-contradiction",
        "remote_contradiction",
        "analysis/candidate-retention/20260822-turn18/workspace/ar25",
        "ar25",
        (("L0:T3", "L0:T4"),),
        ("L0:T3", "L0:T4"),
        ("L0:T5", "L0:T11", "L0:T12"),
        "none",
        "The historical reasoning retained a fixed RIGHT-response hypothesis despite unequal old transitions.",
    ),
    Case(
        "repeated-state-action",
        "repeated_state_action",
        "runs/tycho-nooa-level-start-ar25-20260818T175557Z/workspace/ar25",
        "ar25",
        (("L0:T1",),),
        ("L0:T1",),
        ("L0:T4", "L0:T39", "L0:T56"),
        "none",
        "The long trajectory contains visually identical states whose action histories differ.",
    ),
    Case(
        "multi-transition-computation",
        "multi_transition_computation",
        "runs/tycho-nooa-symmetry-uptake-ar25-20260820T101248Z/workspace/ar25",
        "ar25",
        (("L0:T16", "L1:T2"),),
        ("L0:T16", "L1:T2"),
        ("L0:T14", "L0:T15"),
        "python",
        "The decision requires comparing changed-cell counts across two levels.",
    ),
    Case(
        "goal-discrimination",
        "goal_discrimination",
        "runs/tycho-nooa-symmetry-uptake-ar25-20260820T101248Z/workspace/ar25",
        "ar25",
        (("L0:T16", "L1:T0"),),
        ("L0:T16", "L1:T0"),
        ("L0:T14", "L0:T15"),
        "none",
        "The level boundary can distinguish whether LEFT or another action completed level zero.",
    ),
    Case(
        "plan-assumption-repair",
        "plan_assumption_repair",
        "runs/tycho-nooa-tb-first-live-20260818T103742Z/workspace/ls20",
        "ls20",
        (("L0:T13", "L0:T19"),),
        ("L0:T13", "L0:T19"),
        ("L0:T18",),
        "none",
        "A no-plan-within-budget result at turn thirteen did not identify which plan assumption failed.",
    ),
    Case(
        "evidence-sufficiency",
        "evidence_sufficiency",
        "analysis/candidate-retention/20260822-turn18/workspace/ar25",
        "ar25",
        (("L0:T18",),),
        ("L0:T18",),
        ("L0:T3", "L0:T4", "L0:T5"),
        "none",
        "The replay asks whether the Actor stops when the current observation already suffices.",
    ),
)


def main() -> None:
    if PACKETS.exists():
        raise FileExistsError(PACKETS)
    records = []
    for case in CASES:
        workspace = ROOT / case.workspace
        previous, replay = _current_turn_pair(workspace)
        action = _plain_action(replay.available_actions)
        packet = PACKETS / case.packet_id
        manifest = prepare_production_replay_packet(
            source_workspace=workspace,
            game_id=case.game_id,
            previous_turn=previous,
            replay_turn=replay,
            oracle=ProductionReplayOracle(
                evaluation_purpose="scripted_reachability",
                failure_class=case.failure_class,
                acceptable_terminal_outcomes=(ActionDecision(action=action),),
                acceptable_support_sets=case.support_sets,
                known_relevant_evidence=case.relevant,
                known_distractor_evidence=case.distractors,
                required_computation=case.required_computation,
                historical_failure=case.historical_failure,
            ),
            destination=packet,
            mode="single",
            treatment="control",
        )
        records.append(
            {
                "packet_id": case.packet_id,
                "failure_class": case.failure_class,
                "source_workspace": case.workspace,
                "packet_commitment": manifest["workspace_commitment"],
                "scripted_action": action,
            }
        )
    _write_json(
        OUTPUT / "corpus.json",
        {
            "schema": "tycho-nooa-production-actor-phase1-scripted-corpus-v1",
            "purpose": (
                "zero-provider reachability and evaluator-isolation proof; not a "
                "measurement of Actor cognitive quality"
            ),
            "records": records,
        },
    )


def _current_turn_pair(workspace: Path) -> tuple[GameTurn, GameTurn]:
    marker = _read_json(workspace / ".current_frame.json")
    level = int(marker["level"])
    turn = int(marker["turn"])
    if turn < 1:
        raise ValueError(f"workspace lacks a within-level predecessor: {workspace}")
    previous = _turn(workspace, level, turn - 1, sequence=turn - 1)
    replay = _turn(workspace, level, turn, sequence=turn)
    return previous, replay


def _turn(workspace: Path, level: int, turn: int, *, sequence: int) -> GameTurn:
    base = workspace / f"level_{level}" / f"turn_{turn:03d}"
    metadata = _read_json(base.with_suffix(".json"))
    return GameTurn(
        grid=_read_grid(base.with_suffix(".txt")),
        available_actions=metadata["available"],
        sequence=sequence,
        level=level,
        turn_in_level=turn,
        levels_completed=level,
        state=metadata.get("state") or "NOT_FINISHED",
        previous_action=metadata.get("action") or "",
        previous_row=metadata.get("row"),
        previous_col=metadata.get("col"),
    )


def _read_grid(path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("x:"):
            continue
        body = stripped.split(":", 1)[1] if ":" in stripped else stripped
        tokens = body.replace("|", " ").split()
        if tokens and all(len(token) == 1 for token in tokens):
            rows.append([int(token, 16) for token in tokens])
    if not rows:
        raise ValueError(f"no grid rows in {path}")
    return rows


def _plain_action(actions: list[str]) -> str:
    for action in actions:
        if action != "CLICK":
            return action
    raise ValueError("scripted reachability requires one coordinate-free action")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
