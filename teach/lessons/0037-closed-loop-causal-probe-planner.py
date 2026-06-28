"""0037 - Closed-loop causal probe planner.

Run from the repo root:
    uv run python teach/lessons/0037-closed-loop-causal-probe-planner.py

0032-0036 built replay falsifiers:
  - route to a known causal trigger,
  - remember causal triggers,
  - test cold-start probe heuristics.

This lesson is the first closed-loop planner-shaped check after those replays. It
uses a tiny grid world with hidden object outcomes:

  USEFUL trigger    -> changes the descriptor toward the target
  TELEPORT trigger  -> sends the actor far away
  NEUTRAL trigger   -> no useful effect

The agent starts without causal memory. Each turn it:
  1. chooses one action,
  2. observes the outcome,
  3. updates causal memory / risk memory,
  4. replans.

The critical test is not shortest path. A teleporter lies on the geometric
shortcut in one scenario; after learning it is risky, the planner must prefer a
longer route around it.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field


MOVE_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
USEFUL_EFFECT = "descriptor=color:blue"
TELEPORT_EFFECT = "teleport"
NEUTRAL_EFFECT = "neutral"


@dataclass(frozen=True)
class ToyObject:
    object_id: str
    kind: str
    cell: tuple[int, int]


@dataclass
class PlannerMemory:
    useful: dict[str, ToyObject] = field(default_factory=dict)
    contacts: Counter[str] = field(default_factory=Counter)
    effects: dict[str, Counter[str]] = field(default_factory=dict)

    def observe(self, obj: ToyObject | None, effect: str) -> None:
        if obj is None:
            return
        self.contacts[obj.object_id] += 1
        if obj.object_id not in self.effects:
            self.effects[obj.object_id] = Counter()
        self.effects[obj.object_id][effect] += 1
        if effect == USEFUL_EFFECT:
            self.useful[obj.object_id] = obj

    def risk(self, obj: ToyObject | None) -> float:
        if obj is None:
            return 0.0
        effects = self.effects.get(obj.object_id)
        if not effects:
            return 0.0
        total = sum(effects.values())
        return effects[TELEPORT_EFFECT] / total if total else 0.0

    def is_known_neutral(self, obj: ToyObject | None) -> bool:
        if obj is None:
            return False
        effects = self.effects.get(obj.object_id)
        return bool(effects and effects[NEUTRAL_EFFECT] > 0 and effects[USEFUL_EFFECT] == 0)


@dataclass
class ToyWorld:
    height: int
    width: int
    actor: tuple[int, int]
    terminal: tuple[int, int]
    descriptor: dict[str, str]
    target: dict[str, str]
    objects: dict[str, ToyObject]
    walls: set[tuple[int, int]] = field(default_factory=set)
    teleporter_exit: tuple[int, int] = (0, 0)
    done: bool = False
    history: list[str] = field(default_factory=list)

    def object_at(self, cell: tuple[int, int]) -> ToyObject | None:
        for obj in self.objects.values():
            if obj.cell == cell:
                return obj
        return None

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width and cell not in self.walls

    def step(self, action: str) -> tuple[ToyObject | None, str]:
        if self.done:
            return None, "done"
        dr, dc = MOVE_DELTAS[action]
        nxt = (self.actor[0] + dr, self.actor[1] + dc)
        if not self.in_bounds(nxt):
            self.history.append(f"blocked:{action}")
            return None, "blocked"

        self.actor = nxt
        obj = self.object_at(nxt)
        effect = "move"
        if obj is not None:
            if obj.kind == "USEFUL":
                self.descriptor["color"] = "blue"
                effect = USEFUL_EFFECT
            elif obj.kind == "TELEPORT":
                self.actor = self.teleporter_exit
                effect = TELEPORT_EFFECT
            else:
                effect = NEUTRAL_EFFECT

        if self.actor == self.terminal and self.descriptor == self.target:
            self.done = True
        self.history.append(f"{action}:{effect}@{self.actor}")
        return obj, effect


class CausalProbePlanner:
    def __init__(self) -> None:
        self.memory = PlannerMemory()

    def choose(self, world: ToyWorld) -> str | None:
        if world.descriptor == world.target:
            return self._first_action_to(world, {world.terminal})

        useful_goals = {obj.cell for obj in self.memory.useful.values()}
        if useful_goals:
            return self._first_action_to(world, useful_goals)

        probe = self._choose_probe(world)
        if probe is None:
            return None
        return self._first_action_to(world, {probe.cell})

    def observe(self, obj: ToyObject | None, effect: str) -> None:
        self.memory.observe(obj, effect)

    def _choose_probe(self, world: ToyWorld) -> ToyObject | None:
        candidates = [
            obj for obj in world.objects.values()
            if obj.object_id not in self.memory.useful
            and not self.memory.is_known_neutral(obj)
            and self.memory.risk(obj) < 0.5
        ]
        routed = []
        for obj in candidates:
            path = self._path_to(world, {obj.cell})
            if path is None:
                continue
            novelty = 1.0 / (1 + self.memory.contacts[obj.object_id])
            # A probe is valuable when it is unknown, cheap, and not known risky.
            score = novelty - (0.05 * len(path)) - (2.0 * self.memory.risk(obj))
            routed.append((score, -len(path), obj.object_id, obj))
        if not routed:
            return None
        _score, _neg_len, _obj_id, obj = max(routed)
        return obj

    def _first_action_to(self, world: ToyWorld, goals: set[tuple[int, int]]) -> str | None:
        path = self._path_to(world, goals)
        if not path:
            return None
        return path[0]

    def _path_to(self, world: ToyWorld, goals: set[tuple[int, int]]) -> tuple[str, ...] | None:
        if world.actor in goals:
            return ()
        seen = {world.actor}
        q = deque([(world.actor, ())])
        while q:
            cell, path = q.popleft()
            for action, (dr, dc) in MOVE_DELTAS.items():
                nxt = (cell[0] + dr, cell[1] + dc)
                if nxt in seen or not world.in_bounds(nxt):
                    continue
                obj = world.object_at(nxt)
                if self.memory.risk(obj) >= 0.5 and nxt not in goals:
                    continue
                npath = path + (action,)
                if nxt in goals:
                    return npath
                seen.add(nxt)
                q.append((nxt, npath))
        return None


def run_agent(world: ToyWorld, *, max_steps: int = 60) -> tuple[ToyWorld, CausalProbePlanner]:
    agent = CausalProbePlanner()
    for _ in range(max_steps):
        if world.done:
            return world, agent
        action = agent.choose(world)
        if action is None:
            world.history.append("no_plan")
            return world, agent
        obj, effect = world.step(action)
        agent.observe(obj, effect)
    world.history.append("timeout")
    return world, agent


def discovery_world() -> ToyWorld:
    return ToyWorld(
        height=5,
        width=7,
        actor=(4, 0),
        terminal=(0, 6),
        descriptor={"color": "red"},
        target={"color": "blue"},
        objects={
            "neutral": ToyObject("neutral", "NEUTRAL", (4, 1)),
            "useful": ToyObject("useful", "USEFUL", (4, 3)),
        },
    )


def teleporter_world() -> ToyWorld:
    return ToyWorld(
        height=5,
        width=8,
        actor=(4, 0),
        terminal=(0, 7),
        descriptor={"color": "red"},
        target={"color": "blue"},
        objects={
            "teleporter": ToyObject("teleporter", "TELEPORT", (4, 1)),
            "useful": ToyObject("useful", "USEFUL", (2, 4)),
        },
        walls={(3, 2), (4, 2)},
        teleporter_exit=(4, 7),
    )


def terminal_teleporter_world() -> ToyWorld:
    return ToyWorld(
        height=4,
        width=6,
        actor=(3, 0),
        terminal=(3, 5),
        descriptor={"color": "red"},
        target={"color": "blue"},
        objects={
            "useful": ToyObject("useful", "USEFUL", (3, 1)),
            "teleporter": ToyObject("teleporter", "TELEPORT", (3, 3)),
        },
        teleporter_exit=(0, 0),
    )


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    checks = []

    discovered, agent = run_agent(discovery_world())
    checks.append(_check(
        "1 probes neutral, discovers useful trigger, then reaches terminal",
        discovered.done
        and agent.memory.effects["neutral"][NEUTRAL_EFFECT] == 1
        and agent.memory.effects["useful"][USEFUL_EFFECT] == 1,
        f"done={discovered.done} descriptor={discovered.descriptor} history={discovered.history}",
    ))

    teleported, agent_t = run_agent(teleporter_world())
    checks.append(_check(
        "2 learns teleporter risk and still solves by replanning around it",
        teleported.done
        and agent_t.memory.effects["teleporter"][TELEPORT_EFFECT] == 1
        and agent_t.memory.effects["useful"][USEFUL_EFFECT] == 1,
        f"done={teleported.done} risk={agent_t.memory.risk(teleported.objects['teleporter'])} history={teleported.history}",
    ))

    checks.append(_check(
        "3 does not step on known teleporter after learning risk",
        sum(1 for item in teleported.history if TELEPORT_EFFECT in item) == 1,
        f"history={teleported.history}",
    ))

    terminal_world = terminal_teleporter_world()
    terminal_agent = CausalProbePlanner()
    terminal_agent.memory.observe(terminal_world.objects["useful"], USEFUL_EFFECT)
    terminal_agent.memory.observe(terminal_world.objects["teleporter"], TELEPORT_EFFECT)
    terminal_actions = []
    for _ in range(12):
        if terminal_world.done:
            break
        action = terminal_agent.choose(terminal_world)
        if action is None:
            break
        terminal_actions.append(action)
        obj, effect = terminal_world.step(action)
        terminal_agent.observe(obj, effect)
    checks.append(_check(
        "4 terminal route avoids a known teleporter even when it is the geometric shortcut",
        terminal_world.done and all(TELEPORT_EFFECT not in item for item in terminal_world.history),
        f"done={terminal_world.done} actions={terminal_actions} history={terminal_world.history}",
    ))
    return all(checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    print("Offline checks (closed-loop causal probe planner):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before live planning work.")
    print("\nClosed-loop planner can probe, learn useful/risky outcomes, replan, and avoid a known teleporter.")


if __name__ == "__main__":
    main()
