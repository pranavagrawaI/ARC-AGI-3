"""0027 - Primitive planner over learned remote trigger effects.

Run from the repo root:
    uv run python teach/lessons/0027-primitive-planner.py

0025 learned the attribute descriptor:

    shape_match AND color_match AND orientation_match

0026 learned the terminal predicate:

    next_overlap_smaller

This lesson is the cheapest planner that connects them: use supported remote
trigger effects to make the descriptor true, then execute a primitive action
whose shifted actor pixels overlap the smaller matched object.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import re
import sys as _sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _load_0026():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0026-terminal-predicate-discovery.py"
    spec = _ilu.spec_from_file_location("pp27_tp26", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tp26 = _load_0026()
am24 = tp26.am24


@dataclass(frozen=True)
class PrimitiveAction:
    name: str
    dr: int
    dc: int
    local_facts: frozenset[str]


@dataclass(frozen=True)
class LearnedTriggerEffect:
    cause_clause: tuple[str, ...]
    object_id: str
    slot: str
    delta: str

    @classmethod
    def from_remote_hypothesis(cls, hypothesis) -> "LearnedTriggerEffect":
        object_id, slot, delta = hypothesis.effect
        return cls(hypothesis.cause_clause, object_id, slot, delta)


@dataclass(frozen=True)
class PrimitivePlan:
    actions: tuple[str, ...]
    descriptor_actions: tuple[str, ...]
    terminal_action: str
    final_descriptor: tuple[tuple[str, object], ...]


def _normalise_value(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    return value


def _apply_delta(before: object, delta: str) -> object:
    """Apply the compact remote-effect delta format used by 0017/0018."""

    if "->" in delta:
        _left, right = delta.split("->", 1)
        if re.fullmatch(r"rot[+-]\d+(?:mod\d+)?", right):
            return _apply_delta(before, right)
        if isinstance(before, int) and re.fullmatch(r"-?\d+", right):
            return int(right)
        return right

    rot = re.fullmatch(r"rot([+-]\d+)(?:mod(\d+))?", delta)
    if rot and isinstance(before, int):
        step = int(rot.group(1))
        mod = int(rot.group(2) or 4)
        return (before + step) % mod

    return delta


def _descriptor_satisfied(state: dict[str, object], target: dict[str, object]) -> bool:
    return all(state.get(slot) == value for slot, value in target.items())


def _next_state(
    state: dict[str, object],
    action: PrimitiveAction,
    effects: tuple[LearnedTriggerEffect, ...],
    target_object: str,
) -> dict[str, object]:
    nxt = dict(state)
    for effect in effects:
        if effect.object_id != target_object:
            continue
        if not set(effect.cause_clause) <= action.local_facts:
            continue
        nxt[effect.slot] = _normalise_value(_apply_delta(nxt.get(effect.slot), effect.delta))
    return nxt


def plan_to_descriptor(
    initial: dict[str, object],
    target: dict[str, object],
    effects: tuple[LearnedTriggerEffect, ...],
    actions: tuple[PrimitiveAction, ...],
    *,
    target_object: str,
    max_depth: int = 8,
) -> tuple[str, ...] | None:
    if _descriptor_satisfied(initial, target):
        return ()

    slots = tuple(sorted(set(initial) | set(target) | {effect.slot for effect in effects}))

    def key(state: dict[str, object]) -> tuple[tuple[str, object], ...]:
        return tuple((slot, state.get(slot)) for slot in slots)

    seen = {key(initial)}
    q = deque([(dict(initial), ())])
    while q:
        state, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for action in actions:
            nxt = _next_state(state, action, effects, target_object)
            if nxt == state:
                continue
            nkey = key(nxt)
            if nkey in seen:
                continue
            npath = path + (action.name,)
            if _descriptor_satisfied(nxt, target):
                return npath
            seen.add(nkey)
            q.append((nxt, npath))
    return None


def terminal_overlap_action(actors: list, small, actions: tuple[PrimitiveAction, ...]) -> str | None:
    for action in actions:
        facts = tp26.terminal_facts(actors, small, small, action.dr, action.dc)
        if "next_overlap_smaller" in facts:
            return action.name
    return None


def primitive_plan(
    initial: dict[str, object],
    target: dict[str, object],
    effects: tuple[LearnedTriggerEffect, ...],
    actions: tuple[PrimitiveAction, ...],
    *,
    target_object: str,
    actors: list,
    small,
    max_depth: int = 8,
) -> PrimitivePlan | None:
    descriptor_actions = plan_to_descriptor(
        initial,
        target,
        effects,
        actions,
        target_object=target_object,
        max_depth=max_depth,
    )
    if descriptor_actions is None:
        return None

    terminal_action = terminal_overlap_action(actors, small, actions)
    if terminal_action is None:
        return None

    final = dict(initial)
    by_name = {action.name: action for action in actions}
    for action_name in descriptor_actions:
        final = _next_state(final, by_name[action_name], effects, target_object)
    return PrimitivePlan(
        actions=descriptor_actions + (terminal_action,),
        descriptor_actions=descriptor_actions,
        terminal_action=terminal_action,
        final_descriptor=tuple(sorted(final.items())),
    )


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    actions = (
        PrimitiveAction("ACTION1", -1, 0, frozenset({"contact", "trigger=color"})),
        PrimitiveAction("ACTION2", 0, 1, frozenset({"contact", "trigger=orientation"})),
        PrimitiveAction("ACTION3", 1, 0, frozenset({"contact", "noise"})),
    )
    effects = (
        LearnedTriggerEffect(("contact", "trigger=color"), "workspace", "color", "red->blue"),
        LearnedTriggerEffect(("contact", "trigger=orientation"), "workspace", "orientation", "rot+1mod4"),
    )
    plan = plan_to_descriptor(
        {"shape": "L", "color": "red", "orientation": 0},
        {"shape": "L", "color": "blue", "orientation": 1},
        effects,
        actions,
        target_object="workspace",
    )
    checks.append(_check(
        "1 remote trigger effects reach descriptor",
        plan == ("ACTION1", "ACTION2"),
        f"plan={plan}",
    ))

    actor_grid = np.zeros((5, 5), dtype=bool)
    actor_grid[2, 2] = True
    small_grid = np.zeros((5, 5), dtype=bool)
    small_grid[1, 2] = True
    actor = am24._FakeView(actor_grid, 9, 0)
    small = am24._FakeView(small_grid, 9, 0)
    terminal = terminal_overlap_action([actor], small, actions)
    checks.append(_check(
        "2 terminal action overlaps smaller matched object",
        terminal == "ACTION1",
        f"terminal={terminal}",
    ))

    full = primitive_plan(
        {"shape": "L", "color": "red", "orientation": 0},
        {"shape": "L", "color": "blue", "orientation": 1},
        effects,
        actions,
        target_object="workspace",
        actors=[actor],
        small=small,
    )
    checks.append(_check(
        "3 composed plan reaches descriptor then enters smaller object",
        full is not None
        and full.actions == ("ACTION1", "ACTION2", "ACTION1")
        and dict(full.final_descriptor)["orientation"] == 1,
        f"plan={full}",
    ))

    blocked = plan_to_descriptor(
        {"shape": "L", "color": "red", "orientation": 0},
        {"shape": "L", "color": "blue", "orientation": 2},
        effects,
        actions,
        target_object="workspace",
        max_depth=1,
    )
    checks.append(_check(
        "4 fails loud when descriptor is unreachable within budget",
        blocked is None,
        f"plan={blocked}",
    ))

    live_delta_plan = plan_to_descriptor(
        {"orientation": 0},
        {"orientation": 1},
        (LearnedTriggerEffect(("contact", "trigger=orientation"), "workspace", "orientation", "any->rot+1mod4"),),
        actions,
        target_object="workspace",
    )
    checks.append(_check(
        "5 applies live remote rotation delta format",
        live_delta_plan == ("ACTION2",),
        f"plan={live_delta_plan}",
    ))
    return all(checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    print("Offline checks (primitive planner):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before using primitive planner.")
    print("\nPrimitive planner can chain learned remote effects into the learned terminal action predicate.")


if __name__ == "__main__":
    main()
