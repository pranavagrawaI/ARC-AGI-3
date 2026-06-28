"""0030 - Closed-loop toy agent.

Run from the repo root:
    uv run python teach/lessons/0030-closed-loop-toy-agent.py

0028 and 0029 were replay checks. This lesson is the first closed-loop check in
a controlled toy world: route to trigger cells, observe learned remote effects,
replan after perturbation, then enter the smaller matched object.

It is not ls20 control. It is a cheap sanity check that the planner loop has the
right shape before live routing and perception are connected.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys as _sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


def _load_0027():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0027-primitive-planner.py"
    spec = _ilu.spec_from_file_location("clta30_pp27", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pp27 = _load_0027()

MOVE_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


@dataclass(frozen=True)
class ToyTrigger:
    cell: tuple[int, int]
    action: pp27.PrimitiveAction
    effect: pp27.LearnedTriggerEffect


@dataclass
class ToyWorld:
    height: int
    width: int
    actor: tuple[int, int]
    small: tuple[int, int]
    descriptor: dict[str, object]
    target: dict[str, object]
    triggers: tuple[ToyTrigger, ...]
    walls: set[tuple[int, int]] = field(default_factory=set)
    done: bool = False
    history: list[str] = field(default_factory=list)

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width and cell not in self.walls

    def step(self, move: str) -> None:
        if self.done:
            return
        dr, dc = MOVE_DELTAS[move]
        nxt = (self.actor[0] + dr, self.actor[1] + dc)
        if not self.in_bounds(nxt):
            self.history.append(f"blocked:{move}")
            return
        self.actor = nxt
        self.history.append(move)
        for trigger in self.triggers:
            if trigger.cell == self.actor:
                self.descriptor = pp27._next_state(
                    self.descriptor,
                    trigger.action,
                    (trigger.effect,),
                    trigger.effect.object_id,
                )
        if self.actor == self.small and pp27._descriptor_satisfied(self.descriptor, self.target):
            self.done = True


def _bfs(world: ToyWorld, goals: set[tuple[int, int]]) -> list[str] | None:
    if world.actor in goals:
        return []
    seen = {world.actor}
    q = deque([(world.actor, [])])
    while q:
        cell, path = q.popleft()
        for move, (dr, dc) in MOVE_DELTAS.items():
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in seen or not world.in_bounds(nxt):
                continue
            npath = path + [move]
            if nxt in goals:
                return npath
            seen.add(nxt)
            q.append((nxt, npath))
    return None


class ToyAgent:
    def choose(self, world: ToyWorld) -> str | None:
        if pp27._descriptor_satisfied(world.descriptor, world.target):
            path = _bfs(world, {world.small})
            return path[0] if path else None

        actions = tuple(trigger.action for trigger in world.triggers)
        effects = tuple(trigger.effect for trigger in world.triggers)
        target_object = world.triggers[0].effect.object_id if world.triggers else "workspace"
        plan = pp27.plan_to_descriptor(
            world.descriptor,
            world.target,
            effects,
            actions,
            target_object=target_object,
            max_depth=4,
        )
        if not plan:
            return None
        desired_action = plan[0]
        goals = {trigger.cell for trigger in world.triggers if trigger.action.name == desired_action}
        path = _bfs(world, goals)
        return path[0] if path else None


def run_agent(world: ToyWorld, *, max_steps: int = 40, perturb_at: int | None = None) -> ToyWorld:
    agent = ToyAgent()
    for step in range(max_steps):
        if world.done:
            return world
        if perturb_at is not None and step == perturb_at:
            world.descriptor["orientation"] = 0
            world.history.append("perturb:orientation=0")
        move = agent.choose(world)
        if move is None:
            world.history.append("no_plan")
            return world
        world.step(move)
    world.history.append("timeout")
    return world


def _base_world() -> ToyWorld:
    return ToyWorld(
        height=5,
        width=6,
        actor=(4, 0),
        small=(0, 5),
        descriptor={"shape": "L", "color": "red", "orientation": 0},
        target={"shape": "L", "color": "blue", "orientation": 1},
        triggers=(
            ToyTrigger(
                (4, 2),
                pp27.PrimitiveAction("COLOR_TRIGGER", 0, 0, frozenset({"contact", "trigger=color"})),
                pp27.LearnedTriggerEffect(("contact", "trigger=color"), "workspace", "color", "red->blue"),
            ),
            ToyTrigger(
                (2, 2),
                pp27.PrimitiveAction("ORIENTATION_TRIGGER", 0, 0, frozenset({"contact", "trigger=orientation"})),
                pp27.LearnedTriggerEffect(("contact", "trigger=orientation"), "workspace", "orientation", "any->rot+1mod4"),
            ),
        ),
    )


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []
    solved = run_agent(_base_world())
    checks.append(_check(
        "1 routes to triggers then enters smaller matched object",
        solved.done and solved.actor == solved.small and solved.descriptor == solved.target,
        f"done={solved.done} actor={solved.actor} descriptor={solved.descriptor} history={solved.history}",
    ))

    perturbed = run_agent(_base_world(), perturb_at=5)
    checks.append(_check(
        "2 replans after descriptor perturbation",
        perturbed.done and "perturb:orientation=0" in perturbed.history,
        f"done={perturbed.done} descriptor={perturbed.descriptor} history={perturbed.history}",
    ))

    blocked = _base_world()
    blocked.walls = {(3, 2), (2, 1), (2, 3), (1, 2)}
    failed = run_agent(blocked)
    checks.append(_check(
        "3 fails loud when required trigger is unreachable",
        not failed.done and "no_plan" in failed.history,
        f"done={failed.done} actor={failed.actor} history={failed.history}",
    ))
    return all(checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    print("Offline checks (closed-loop toy agent):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before live routing work.")
    print("\nClosed-loop toy agent can route, trigger learned effects, replan, and execute terminal entry.")


if __name__ == "__main__":
    main()
