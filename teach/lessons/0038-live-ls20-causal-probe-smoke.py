"""0038 - Live ls20 causal probe smoke test.

Run from the repo root:
    uv run python teach/lessons/0038-live-ls20-causal-probe-smoke.py --steps 120

0037 proved the closed-loop planner shape in a toy world. This lesson runs the
same style of loop against real offline ls20 frames:

  1. bootstrap controllability until the avatar can be perceived,
  2. route to low-risk / high-information small objects,
  3. observe whether contact caused remote changes or teleport-like motion,
  4. update shape-level causal/risk memory,
  5. replan one action at a time.

This is intentionally a smoke test, not a claim of competence. It measures where
the current stack breaks first when replay oracles are removed.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys as _sys
from collections import Counter, defaultdict
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import arc_agi
import numpy as np
from arc_agi.base import OperationMode
from arcengine import GameAction


def _load_0036():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0036-info-risk-cold-start-probe-replay.py"
    spec = _ilu.spec_from_file_location("llcp38_ircsp36", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ircsp36 = _load_0036()
cstp35 = ircsp36.cstp35
ewr32 = ircsp36.ewr32
oc16 = ircsp36.rcf18.rca17._load_oc16()


@dataclass
class ShapeStats:
    contacts: int = 0
    remote_effects: int = 0
    teleport_like: int = 0
    neutral: int = 0

    @property
    def risk_rate(self) -> float:
        return self.teleport_like / self.contacts if self.contacts else 0.0

    @property
    def remote_rate(self) -> float:
        return self.remote_effects / self.contacts if self.contacts else 0.0


def _grid(frame_data) -> np.ndarray:
    arr = np.asarray(frame_data.frame, dtype=np.int8)
    return arr[-1] if arr.ndim == 3 else arr


def _actor_centroid(actor_views: list) -> tuple[float, float] | None:
    if not actor_views:
        return None
    return (
        sum(view.centroid[0] for view in actor_views) / len(actor_views),
        sum(view.centroid[1] for view in actor_views) / len(actor_views),
    )


def _view_ahead(parts: list, actor_views: list, action: int):
    center = _actor_centroid(actor_views)
    if center is None or action not in oc16.ACTION_DELTAS:
        return None
    dr, dc = oc16.ACTION_DELTAS[action]
    cell = (round(center[0] + dr), round(center[1] + dc))
    for view in parts:
        if cell in view.pixels:
            return view
    return None


def _actor_shift(prev_actor_views: list, next_parts: list, controlled_sigs: set[tuple[int, int]]) -> float | None:
    prev = _actor_centroid(prev_actor_views)
    next_actor_views = oc16._actor_views(next_parts, controlled_sigs)
    nxt = _actor_centroid(next_actor_views)
    if prev is None or nxt is None:
        return None
    return abs(prev[0] - nxt[0]) + abs(prev[1] - nxt[1])


class LiveCausalProbeAgent:
    def __init__(self) -> None:
        self.belief = oc16.AgentBelief()
        self.stats: dict[str, ShapeStats] = defaultdict(ShapeStats)
        self.bootstrap_actions = (1, 4, 2, 3)
        self.bootstrap_i = 0
        self.frame_idx = 0
        self.views = []
        self.parts = []
        self.prev_parts = []
        self.actor_views = []
        self.stable_shapes: Counter[str] = Counter()
        self.active_target = None
        self.active_shape: str | None = None
        self.active_age = 0
        self.blocked_edges: set[tuple[tuple[int, int], int]] = set()
        self.last_reason = "init"
        self.last_observation = "obs:init"
        self.counts: Counter[str] = Counter()

    def begin_frame(self, grid: np.ndarray) -> None:
        self.views = self.belief.perceive(grid)
        _bg, self.parts = oc16.il.views_from_grid(grid, frame=self.frame_idx)
        self.actor_views = oc16._actor_views(self.parts, self.belief.controlled_sigs)
        if self.prev_parts:
            self._update_stability(self.parts, self.prev_parts)
        if self.active_target is not None:
            self.active_target = ewr32._match_overlap(self.active_target, self.parts)
            if self.active_target is None:
                self.active_shape = None
                self.active_age = 0
        self.prev_parts = self.parts

    def _update_stability(self, parts: list, prev_parts: list) -> None:
        actor_ids = {id(view) for view in self.actor_views}
        max_area = 4 * oc16.CELL ** 2
        for view in parts:
            if id(view) in actor_ids or view.area > max_area:
                continue
            shape = oc16.ig.shape_literal(view)
            for prev in prev_parts:
                if not (view.pixels & prev.pixels):
                    continue
                if oc16.ig.shape_literal(prev) == shape:
                    self.stable_shapes[shape] += 1
                    break

    def choose(self, grid: np.ndarray, available_actions: list[int]) -> int:
        if not self.actor_views:
            action = self.bootstrap_actions[self.bootstrap_i % len(self.bootstrap_actions)]
            self.bootstrap_i += 1
            self.last_reason = "bootstrap_control"
            return action if action in available_actions else available_actions[0]

        if self.active_target is not None:
            route = self._route_to(grid, self.active_target, available_actions)
            if route is not None and self.active_age <= 20:
                self.active_age += 1
                self.last_reason = f"continue:{self.active_shape}:len={len(route.actions)}"
                return route.first_action
            self.active_target = None
            self.active_shape = None
            self.active_age = 0

        candidates = cstp35._small_non_actor_candidates(self.parts, self.actor_views, oc16)
        ranked = []
        for order, view in enumerate(candidates):
            shape = oc16.ig.shape_literal(view)
            stats = self.stats.get(shape, ShapeStats())
            if stats.risk_rate >= 0.5:
                continue
            if stats.contacts and stats.remote_effects == 0 and stats.teleport_like == 0:
                continue
            stability = self.stable_shapes.get(shape, 0)
            if self.frame_idx > 8 and stability == 0:
                continue
            route = self._route_to(grid, view, available_actions)
            if route is None:
                continue
            novelty = 1.0 / (1 + stats.contacts)
            stability_bonus = min(stability, 5) * 0.08
            score = stats.remote_rate + novelty + stability_bonus - (2.0 * stats.risk_rate) - (0.03 * len(route.actions))
            ranked.append((score, -len(route.actions), -view.area, -order, route.first_action, shape))

        if ranked:
            score, neg_len, _area, _order, action, shape = max(ranked)
            # Re-find the selected view by shape and route length. This keeps the
            # ranked tuple numeric-only while committing to one probe across frames.
            for view in candidates:
                if oc16.ig.shape_literal(view) != shape:
                    continue
                route = self._route_to(grid, view, available_actions)
                if route is not None and route.first_action == action and len(route.actions) == -neg_len:
                    self.active_target = view
                    self.active_shape = shape
                    self.active_age = 1
                    break
            self.last_reason = f"probe:{shape}:score={score:.2f}:len={-neg_len}"
            return action

        action = self.bootstrap_actions[self.bootstrap_i % len(self.bootstrap_actions)]
        self.bootstrap_i += 1
        self.last_reason = "fallback_sweep"
        return action if action in available_actions else available_actions[0]

    def _route_to(self, grid: np.ndarray, target_view, available_actions: list[int]):
        start = ewr32._to_cell(_actor_centroid(self.actor_views), oc16.CELL)
        bg = ewr32.ltr31._background(grid)
        body = [view for view in self.actor_views if getattr(view, "area", 0) > 1] or self.actor_views
        actor_cells = {ewr32._to_cell(pixel, oc16.CELL) for view in body for pixel in view.pixels}
        grid_deltas = {
            action: (int(round(dr / oc16.CELL)), int(round(dc / oc16.CELL)))
            for action, (dr, dc) in oc16.ACTION_DELTAS.items()
        }
        trigger_cells = {ewr32._to_cell(pixel, oc16.CELL) for pixel in target_view.pixels}
        height, width = grid.shape

        def free(cell: tuple[int, int]) -> bool:
            if cell in actor_cells:
                return True
            pr, pc = cell[0] * oc16.CELL + oc16.CELL // 2, cell[1] * oc16.CELL + oc16.CELL // 2
            if not (0 <= pr < height and 0 <= pc < width):
                return False
            return int(grid[pr, pc]) == bg

        goals: dict[tuple[int, int], int] = {}
        for final_action, (gdr, gdc) in grid_deltas.items():
            if final_action not in available_actions:
                continue
            for target_cell in trigger_cells:
                approach = (target_cell[0] - gdr, target_cell[1] - gdc)
                if free(approach):
                    goals.setdefault(approach, final_action)
        if start in goals and (start, goals[start]) not in self.blocked_edges:
            return ewr32.ltr31.RoutePlan((), goals[start], start)

        seen = {start}
        q = deque([(start, ())])
        while q:
            cell, path = q.popleft()
            for action in available_actions:
                if action not in grid_deltas or (cell, action) in self.blocked_edges:
                    continue
                gdr, gdc = grid_deltas[action]
                nxt = (cell[0] + gdr, cell[1] + gdc)
                if nxt in seen or not free(nxt):
                    continue
                npath = path + (action,)
                if nxt in goals and (nxt, goals[nxt]) not in self.blocked_edges:
                    return ewr32.ltr31.RoutePlan(npath, goals[nxt], nxt)
                seen.add(nxt)
                q.append((nxt, npath))
        return None

    def observe(self, prev_grid: np.ndarray, action: int, next_grid: np.ndarray) -> None:
        target = _view_ahead(self.parts, self.actor_views, action)
        predicted = oc16.predict_frame(prev_grid, action, self.views, self.belief)
        _bg_next, parts_next = oc16.il.views_from_grid(next_grid, frame=self.frame_idx + 1)
        shift = _actor_shift(self.actor_views, parts_next, self.belief.controlled_sigs)
        remote_clusters = []
        for cluster in oc16.diff_clusters(predicted != next_grid):
            kind = oc16._change_kind(prev_grid, next_grid, cluster, self.actor_views)
            if kind == "remote_attribute_change":
                remote_clusters.append(cluster)

        if target is not None and not oc16.ig.is_actor_view(target, self.belief.controlled_sigs):
            max_area = 4 * oc16.CELL ** 2
            if target.area <= max_area:
                shape = oc16.ig.shape_literal(target)
                stats = self.stats[shape]
                stats.contacts += 1
                teleport = shift is None or shift > oc16.CELL * 1.6
                stats.teleport_like += int(teleport)
                stats.remote_effects += int(bool(remote_clusters))
                stats.neutral += int(not teleport and not remote_clusters)
                self.counts["contacts"] += 1
                self.counts["remote_contacts"] += int(bool(remote_clusters))
                self.counts["teleport_like"] += int(teleport)
                self.counts[f"shape:{shape}"] += 1
                self.last_observation = (
                    f"contact:{shape}:remote={bool(remote_clusters)}:teleport={teleport}:shift={shift}"
                )
                if self.active_shape == shape:
                    self.active_target = None
                    self.active_shape = None
                    self.active_age = 0
            else:
                self.last_observation = f"large_target:area={target.area}:shift={shift}"
        else:
            self.counts["no_contact_ahead"] += 1
            if shift is None:
                self.counts["actor_shift_unknown"] += 1
            elif shift < oc16.CELL * 0.5:
                self.counts["no_motion"] += 1
                center = _actor_centroid(self.actor_views)
                if center is not None:
                    self.blocked_edges.add((ewr32._to_cell(center, oc16.CELL), action))
            self.last_observation = f"no_contact_ahead:shift={shift}:actor={_actor_centroid(self.actor_views)}"

        self.belief.learn_remote_dynamics(self.parts, parts_next)
        self.belief.observe_transition(prev_grid, action, next_grid)
        self.belief.frame_idx += 1
        self.frame_idx += 1
        self.belief.perceive(next_grid)


def run_live(steps: int) -> dict[str, object]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    frame = env.reset()
    agent = LiveCausalProbeAgent()
    action_log: list[str] = []
    levels_completed = frame.levels_completed

    for step in range(steps):
        grid = _grid(frame)
        agent.begin_frame(grid)
        action = agent.choose(grid, list(frame.available_actions))
        next_frame = env.step(GameAction.from_id(action))
        next_grid = _grid(next_frame)
        agent.observe(grid, action, next_grid)
        action_log.append(
            f"{step}:A{action}:{agent.last_reason}:{agent.last_observation}:L{next_frame.levels_completed + 1}:state={next_frame.state.value}"
        )
        frame = next_frame
        levels_completed = frame.levels_completed
        if frame.state.value in {"WIN", "GAME_OVER"}:
            break

    return {
        "steps": len(action_log),
        "levels_completed": levels_completed,
        "state": frame.state.value,
        "controlled_sigs": sorted(agent.belief.controlled_sigs),
        "counts": dict(agent.counts),
        "known_shapes": {
            shape: {
                "contacts": stats.contacts,
                "remote_effects": stats.remote_effects,
                "teleport_like": stats.teleport_like,
                "neutral": stats.neutral,
                "risk": round(stats.risk_rate, 3),
                "remote_rate": round(stats.remote_rate, 3),
            }
            for shape, stats in sorted(agent.stats.items())
        },
        "actions": action_log[:80],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    result = run_live(args.steps)
    print("\nLive ls20 causal probe smoke:")
    print(f"  steps={result['steps']} state={result['state']} levels_completed={result['levels_completed']}")
    print(f"  controlled_sigs={result['controlled_sigs']}")
    print(f"  counts={result['counts']}")
    print("  known_shapes:")
    for shape, stats in result["known_shapes"].items():
        print(f"    {shape}: {stats}")
    print("  first_actions:")
    for item in result["actions"][:40]:
        print(f"    {item}")


if __name__ == "__main__":
    main()
