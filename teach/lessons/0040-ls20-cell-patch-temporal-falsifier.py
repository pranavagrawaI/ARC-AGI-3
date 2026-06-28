"""0040 - Cell-patch temporal falsification for ls20.

Run from the repo root:
    uv run python teach/lessons/0040-ls20-cell-patch-temporal-falsifier.py

0039 showed that connected components are the wrong object primitive for ls20.
This lesson starts from 5x5 cell-patch candidates and assigns only labels that
interaction justifies:

  - controlled: the same patch signature relocates after an action,
  - static: the patch remains in place across probes,
  - blocked action: no patch changes after the action from the start state,
  - action delta: the displacement observed for the controlled patch.

The private game state is used only to score the diagnosis at the end, not as
input to perception or falsification.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys as _sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import arc_agi
import numpy as np
from arc_agi.base import OperationMode
from arcengine import GameAction


REPO = Path(__file__).resolve().parents[2]


def _load_0039():
    path = REPO / "teach/lessons/0039-ls20-object-perception-diagnostic.py"
    spec = _ilu.spec_from_file_location("cpff40_diag39", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


diag39 = _load_0039()


@dataclass(frozen=True)
class Candidate:
    cid: str
    cell: tuple[int, int]
    signature: tuple[tuple[int, ...], ...]
    colors: tuple[tuple[int, int], ...]
    non_background: int
    distinct_colors: int


@dataclass
class CandidateEvidence:
    candidate: Candidate
    moved_by_action: dict[int, tuple[int, int]] = field(default_factory=dict)
    static_actions: set[int] = field(default_factory=set)
    changed_without_relocation: set[int] = field(default_factory=set)

    @property
    def controlled_votes(self) -> int:
        return len(self.moved_by_action)

    @property
    def static_votes(self) -> int:
        return len(self.static_actions)


@dataclass(frozen=True)
class ActionProbe:
    action: int
    changed_cells: tuple[tuple[int, int], ...]
    blocked: bool


def _candidate_score(patch) -> tuple[int, int, int, int]:
    hud_penalty = -1 if patch.row >= 50 else 0
    return (
        patch.distinct_colors,
        patch.non_background,
        hud_penalty,
        -(patch.row + patch.col),
    )


def initial_candidates(grid: np.ndarray, limit: int) -> list[Candidate]:
    patches = [
        patch
        for patch in diag39._cell_patches(grid)
        if patch.non_background > 0 and patch.distinct_colors >= 2
    ]
    patches.sort(key=_candidate_score, reverse=True)
    candidates = []
    for index, patch in enumerate(patches[:limit]):
        candidates.append(
            Candidate(
                cid=f"cand_{index:02d}",
                cell=(patch.row, patch.col),
                signature=patch.signature,
                colors=patch.colors,
                non_background=patch.non_background,
                distinct_colors=patch.distinct_colors,
            )
        )
    return candidates


def _changed_lattice_cells(grid0: np.ndarray, grid1: np.ndarray) -> tuple[tuple[int, int], ...]:
    cells = []
    for patch in diag39._cell_patches(grid0):
        after = diag39._patch(grid1, patch.row, patch.col)
        before = diag39._patch(grid0, patch.row, patch.col)
        if np.any(before != after):
            cells.append((patch.row, patch.col))
    return tuple(cells)


def _signature_locations(
    grid: np.ndarray, signature: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, int], ...]:
    return tuple(diag39._find_signature(grid, signature))


def run_single_action_probe(
    env,
    action: int,
) -> tuple[np.ndarray, np.ndarray, list, ActionProbe]:
    frame0 = env.reset()
    game = env._game
    oracle_objects = diag39._oracle_objects(game)
    grid0 = diag39._grid(frame0)
    frame1 = env.step(GameAction.from_id(action))
    grid1 = diag39._grid(frame1)
    changed = _changed_lattice_cells(grid0, grid1)
    return (
        grid0,
        grid1,
        oracle_objects,
        ActionProbe(action=action, changed_cells=changed, blocked=not changed),
    )


def falsify_from_start(actions: tuple[int, ...], limit: int) -> dict[str, object]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    grid0, _grid1, oracle_objects, _probe = run_single_action_probe(env, actions[0])
    candidates = initial_candidates(grid0, limit)
    evidence = {candidate.cid: CandidateEvidence(candidate) for candidate in candidates}
    probes: list[ActionProbe] = []

    for action in actions:
        base, after, _oracle_objects, probe = run_single_action_probe(env, action)
        probes.append(probe)
        changed_set = set(probe.changed_cells)
        for candidate in candidates:
            ev = evidence[candidate.cid]
            locs_after = _signature_locations(after, candidate.signature)
            original_present = candidate.cell in locs_after
            relocated = [cell for cell in locs_after if cell != candidate.cell]
            original_changed = candidate.cell in changed_set
            if original_changed and not original_present and len(relocated) == 1:
                new_cell = relocated[0]
                ev.moved_by_action[action] = (
                    new_cell[0] - candidate.cell[0],
                    new_cell[1] - candidate.cell[1],
                )
            elif not original_changed and original_present:
                ev.static_actions.add(action)
            elif original_changed:
                ev.changed_without_relocation.add(action)

    return {
        "grid0": grid0,
        "oracle_objects": oracle_objects,
        "candidates": candidates,
        "evidence": evidence,
        "probes": probes,
    }


def _oracle_cell_labels(oracle_objects: list) -> dict[tuple[int, int], str]:
    return {obj.cell: obj.name for obj in oracle_objects}


def _summarize_evidence(result: dict[str, object]) -> list[str]:
    candidates: list[Candidate] = result["candidates"]
    evidence: dict[str, CandidateEvidence] = result["evidence"]
    probes: list[ActionProbe] = result["probes"]
    oracle_objects = result["oracle_objects"]
    oracle_labels = _oracle_cell_labels(oracle_objects)

    lines = ["ls20 cell-patch temporal falsification", ""]
    lines.append("Initial object-like cell candidates:")
    for candidate in candidates:
        label = oracle_labels.get(candidate.cell, "")
        suffix = f" oracle={label}" if label else ""
        lines.append(
            f"  {candidate.cid} cell={candidate.cell} colors={candidate.colors} "
            f"non_bg={candidate.non_background}{suffix}"
        )

    lines.extend(["", "Action probes from the first frame:"])
    for probe in probes:
        state = "blocked/no-op" if probe.blocked else "changed"
        lines.append(
            f"  A{probe.action}: {state} changed_cells={list(probe.changed_cells)}"
        )

    controlled = [
        ev for ev in evidence.values()
        if ev.controlled_votes >= 2
    ]
    controlled.sort(key=lambda ev: (-ev.controlled_votes, ev.candidate.cid))
    lines.extend(["", "Falsified roles from temporal evidence:"])
    if controlled:
        for ev in controlled:
            learned = ", ".join(
                f"A{action}->{delta}"
                for action, delta in sorted(ev.moved_by_action.items())
            )
            label = oracle_labels.get(ev.candidate.cell, "unknown")
            lines.append(
                f"  controlled {ev.candidate.cid} cell={ev.candidate.cell} "
                f"votes={ev.controlled_votes} learned={learned} oracle={label}"
            )
    else:
        lines.append("  no controlled candidate reached the vote threshold")

    static = [
        ev for ev in evidence.values()
        if ev.static_votes == len(probes) and ev.controlled_votes == 0
    ]
    static.sort(key=lambda ev: ev.candidate.cid)
    lines.append("  stable/static candidates:")
    for ev in static[:12]:
        label = oracle_labels.get(ev.candidate.cell, "")
        suffix = f" oracle={label}" if label else ""
        lines.append(f"    {ev.candidate.cid} cell={ev.candidate.cell}{suffix}")

    action_delta_votes: dict[int, Counter] = defaultdict(Counter)
    for ev in evidence.values():
        for action, delta in ev.moved_by_action.items():
            action_delta_votes[action][delta] += 1
    lines.extend(["", "Learned action model from controlled-patch relocation:"])
    for probe in probes:
        votes = action_delta_votes.get(probe.action, Counter())
        if votes:
            delta, count = votes.most_common(1)[0]
            lines.append(f"  A{probe.action}: delta={delta} votes={count}")
        elif probe.blocked:
            lines.append(f"  A{probe.action}: blocked at start")
        else:
            lines.append(f"  A{probe.action}: changed but no clean relocation")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--actions", default="1,2,3,4")
    args = parser.parse_args()
    actions = tuple(int(part) for part in args.actions.split(",") if part)

    result = falsify_from_start(actions, args.limit)
    for line in _summarize_evidence(result):
        print(line)


if __name__ == "__main__":
    main()
