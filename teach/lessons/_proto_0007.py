"""Prototype v4 — reactive matching planner. Replans from live state every step."""
from collections import deque
import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

S = 5
DELTA = {1: (0, -S), 2: (0, S), 3: (-S, 0), 4: (S, 0)}
CYCLER_TAGS = {"shape": "ttfwljgohq", "color": "soyhouuebz", "rotation": "rhsxkxzdjz"}


def bfs(G, start, goals, avoid=frozenset()):
    if start in goals:
        return []
    seen, q = {start}, deque([(start, [])])
    while q:
        pos, path = q.popleft()
        for a, nb in G.get(pos, {}).items():
            if nb in seen or nb in avoid:
                continue
            if nb in goals:
                return path + [a]
            seen.add(nb); q.append((nb, path + [a]))
    return None


def build_graph(walls, start, blocked):
    G, seen, q = {}, {start}, deque([start])
    while q:
        c = q.popleft(); G[c] = {}
        for a, (dx, dy) in DELTA.items():
            nb = (c[0] + dx, c[1] + dy)
            if nb in walls or nb in blocked:
                continue
            G[c][a] = nb
            if nb not in seen:
                seen.add(nb); q.append(nb)
    return G


class MatchingPlanner:
    """Reactive: each step, read live map+goal from the engine and emit one action."""

    def __init__(self, game):
        self.g = game

    def choose(self, frame):
        g = self.g
        pos = (g.gudziatsk.x, g.gudziatsk.y)
        attrs = {"shape": g.fwckfzsyc, "color": g.hiaauhahz, "rotation": g.cklxociuu}
        walls = {(s.x, s.y) for s in g.current_level.get_sprites_by_tag("ihdgageizm")}
        cyclers = {k: [(s.x, s.y) for s in g.current_level.get_sprites_by_tag(t)]
                   for k, t in CYCLER_TAGS.items()}
        all_cyclers = {c for v in cyclers.values() for c in v}
        unsolved = [i for i in range(len(g.plrpelhym)) if not g.lvrnuajbl[i]]
        if not unsolved:
            return frame.available_actions[0]
        door_cells = {(g.plrpelhym[i].x, g.plrpelhym[i].y) for i in unsolved}
        G = build_graph(walls, pos, blocked=door_cells)

        def door_target(i):
            """(shortest path to a neighbour, step-in action) for door i, avoiding cyclers."""
            d = (g.plrpelhym[i].x, g.plrpelhym[i].y)
            best = None
            for a, (dx, dy) in DELTA.items():
                nb = (d[0] - dx, d[1] - dy)
                if nb not in G:
                    continue
                p = bfs(G, pos, {nb}, avoid=all_cyclers - {pos})
                if p is not None and (best is None or len(p) < len(best[0])):
                    best = (p, a)
            return best

        # choose the nearest unsolved door to commit to this step
        target = min(((i, door_target(i)) for i in unsolved),
                     key=lambda t: 1e9 if t[1] is None else len(t[1][0]))
        i, dt = target
        req = {"shape": g.ldxlnycps[i], "color": g.yjdexjsoa[i], "rotation": g.ehwheiwsk[i]}

        # 1. fix the first mismatched attribute by tapping its cycler
        for a_name in ("shape", "color", "rotation"):
            if attrs[a_name] == req[a_name] or not cyclers[a_name]:
                continue
            C = cyclers[a_name][0]
            if pos == C:  # already on it; step off so we can re-enter for another tap
                off = next((a for a, nb in G[C].items() if nb not in all_cyclers), None)
                return off if off is not None else frame.available_actions[0]
            path = bfs(G, pos, {C})  # walking onto C taps it once
            return path[0] if path else frame.available_actions[0]

        # 2. attributes match: walk to the door and step in
        if dt is None:
            return frame.available_actions[0]
        path, a_in = dt
        return a_in if not path else path[0]


def run():
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make("ls20"); frame = env.reset(); g = env._game
    planner = MatchingPlanner(g)
    total = 0
    marks = {0: 0}
    while frame.state.value not in ("WIN", "GAME_OVER") and total < 2000:
        lvl = frame.levels_completed
        frame = env.step(GameAction.from_id(planner.choose(frame))); total += 1
        if frame.levels_completed > lvl:
            marks[frame.levels_completed] = total
            print(f"  level {frame.levels_completed} CLEARED at action {total} "
                  f"(+{total - marks[lvl]} moves)")
    print(f"RESULT levels={frame.levels_completed}/7 state={frame.state.value} total_actions={total}")


if __name__ == "__main__":
    run()
