"""Verified routes for the A2 arm and the A2 text built from them.

  routes.py check            play every route in hints.json from its level entry on the engine; all must complete
  routes.py fill             for hints without a route, take the shortest sibling segment that solves from our prefix

A route is a list of model-facing actions exactly as action() accepts them: "UP"/"DOWN"/"LEFT"/"RIGHT"/"SPACE"/"UNDO"/
"RESET" or {"action": "MOUSE", "row": r, "col": c}.
"""
import json, sys
from pathlib import Path
import look as L
from arcengine import ActionInput, GameAction, GameState

HERE = Path(__file__).resolve().parent
HINTS = HERE / "hints.json"
INV = {v: k for k, v in L.NAMES.items()}


def to_model(a):
    return {"action": "MOUSE", "row": a["data"]["y"], "col": a["data"]["x"]} if a["id"] == "ACTION6" else L.NAMES[a["id"]]


def to_engine(m):
    if isinstance(m, dict):
        return ActionInput(id=GameAction.ACTION6, data={"x": m["col"], "y": m["row"]})
    return ActionInput(id=GameAction[INV[m]])


def plays(key, route):
    game, lv = key.rsplit("-L", 1)
    s, g, fr = L.entry(game, int(lv))
    for i, m in enumerate(route):
        fr = g.perform_action(to_engine(m))
        if fr.levels_completed > int(lv):
            return i + 1 == len(route)  # must complete on exactly the last action
        if fr.state == GameState.GAME_OVER:
            return False
    return False


def plan_text(goal, route):
    return (goal.rstrip() + "\n\nA verified route that completes this level from its starting position. Pass it to "
            f"action() exactly as written, in order: {json.dumps(route)}")


def main():
    h = json.loads(HINTS.read_text())
    if sys.argv[1] == "fill":
        for key, e in h.items():
            if e.get("route"):
                continue
            game, lv = key.rsplit("-L", 1)
            st = L.ST[(game, int(lv))]
            def clean(r):  # a RESET restores the level entry, so the solve is the part after the last RESET
                seg = L.segment(game, int(lv), r)[1]
                k = max([i for i, a in enumerate(seg) if a["id"] == "RESET"], default=-1)
                return seg[k + 1:]
            segs = sorted((len(clean(r)), r) for r in st["solved_runs"])
            e["route"] = [to_model(a) for a in clean(segs[0][1])]
            e["route_source"] = f"sibling {segs[0][1]} segment after its last RESET"
        HINTS.write_text(json.dumps(h, indent=1, ensure_ascii=False) + "\n")
    bad = [k for k, e in h.items() if not e.get("route") or not plays(k, e["route"])]
    for k, e in h.items():
        print(f"{k:10s} route {len(e.get('route') or []):4d} {'OK' if k not in bad else 'FAIL'}  ({e.get('route_source', '')})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
