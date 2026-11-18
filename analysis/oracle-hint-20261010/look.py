"""Inspection lever for writing and verifying hints. Uses the model's own color letters (WwgGcBMPRbSYOrNp) and row/col.

  look.py frame <game> <level>                 level-entry board (ascii) + connected components
  look.py seg <game> <level> [run]             replay a sibling's solving segment from our prefix: per-action diff summary
  look.py talk <game> <level> [run] [chars]    the solving run's own reasoning just before it completed the level
  look.py src <game> [context]                 source lines around the win/next_level calls
  look.py try <game> <level> <actions.json>    play actions (model names: UP/DOWN/LEFT/RIGHT/SPACE/UNDO/RESET or
                                               [row,col] for MOUSE) from level entry; report whether the level completes
"""
import json, re, sys
from pathlib import Path
import numpy as np
from arcengine import ActionInput, GameAction, GameState

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import starts as S  # noqa: E402

CH = "WwgGcBMPRbSYOrNp"
NAMES = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE", "ACTION7": "UNDO", "RESET": "RESET"}
ST = {(s["game"], s["level"]): s for s in json.loads((HERE / "starts.json").read_text())}
TRANSCRIPTS = {"dfz-control": "out-control", "dfz-r2": "out-control-r2", "dfz-s1": "out-s1"}


def grid(fr):
    return np.array(fr.frame[-1], dtype=int)


def ascii_(g):
    return "\n".join("".join(CH[v] for v in row) for row in g)


def components(g, min_size=1):
    seen = np.zeros_like(g, bool); out = []
    H, W = g.shape
    bg = np.bincount(g.ravel()).argmax()
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            col = g[r, c]; stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True; stack.append((ny, nx))
            if len(cells) >= min_size:
                ys, xs = zip(*cells)
                out.append((CH[col], len(cells), min(ys), min(xs), max(ys), max(xs)))
    return CH[bg], out


def entry(game, level):
    s = ST[(game, level)]
    g, fr = S.play(s["gid"], s["prefix"])
    return s, g, fr


def act(g, a):
    if isinstance(a, list):
        return g.perform_action(ActionInput(id=GameAction.ACTION6, data={"x": a[1], "y": a[0]}))
    inv = {v: k for k, v in NAMES.items()}
    return g.perform_action(ActionInput(id=GameAction[inv[a]]))


def show(a):
    return f"MOUSE(row={a['data']['y']},col={a['data']['x']})" if a["id"] == "ACTION6" else NAMES[a["id"]]


def segment(game, level, run=None):
    s = ST[(game, level)]
    run = run or s["solved_runs"][0]
    R = S.runs()[(run, game)]
    k = sum(R["apl"][:level])
    return run, R["h"][k:k + R["apl"][level]]


def cmd_frame(game, level):
    s, g, fr = entry(game, level)
    G = grid(fr)
    print(f"{game} L{level} entry ({s['klass']}); available actions {sorted(int(a) for a in fr.available_actions)}; state {fr.state.name}")
    print(ascii_(G))
    bg, comps = components(G)
    print(f"background {bg}; {len(comps)} components (color, cells, r0, c0, r1, c1):")
    for c in sorted(comps, key=lambda c: (c[2], c[3])):
        print("  ", c)


def cmd_seg(game, level, run=None):
    s, g, fr = entry(game, level)
    run, seg = segment(game, level, run)
    print(f"{game} L{level}: {run} solved in {len(seg)} actions")
    prev = grid(fr)
    for i, a in enumerate(seg):
        fr = g.perform_action(ActionInput(id=GameAction[a["id"]], data=a.get("data") or {}))
        cur = grid(fr)
        d = np.argwhere(prev != cur)
        trans = {}
        for y, x in d:
            key = f"{CH[prev[y, x]]}->{CH[cur[y, x]]}"
            trans[key] = trans.get(key, 0) + 1
        box = f"r{d[:, 0].min()}-{d[:, 0].max()} c{d[:, 1].min()}-{d[:, 1].max()}" if len(d) else "-"
        print(f"{i:3d} {show(a):24s} changed={len(d):4d} {box:18s} {dict(sorted(trans.items(), key=lambda t: -t[1])[:5])}"
              f"{'  LEVEL COMPLETE' if fr.levels_completed > level else ''}{'  GAME_OVER' if fr.state == GameState.GAME_OVER else ''}")
        prev = cur


def thinking(chunk, per=700, last=8):
    """[THINKING] blocks (and tool results' first line) in order, truncated: the model's own account of the level."""
    out = []
    for m in re.finditer(r"\[THINKING\]\n(.*?)(?=\n\[TOOL CALL|\n\[USER PROMPT\]|\n\[MODEL RESPONSE|\Z)", chunk, re.S):
        out.append(re.sub(r"\s+", " ", m.group(1)).strip()[:per])
    return out[-int(last):]


def cmd_think(game, level, run=None, last=8, per=700):
    """Thinking blocks of the turns between the previous completion and this level's completion in a run."""
    s = ST[(game, level)]
    run = run or (s["solved_runs"][0] if s["solved_runs"] else s["stuck_runs"][0])
    d = HERE.parent / "m2-winners-20261001/runs" / TRANSCRIPTS[run] / "transcripts"
    t = next(d.glob(f"{game}-*_p0.txt")).read_text(errors="replace")
    # level index L is shown to the model as "level L+1"; the region runs to the first header of the next level
    heads = [(m.start(), int(m.group(1))) for m in re.finditer(r"Current state: step \d+, level (\d+)\.", t)]
    mine = [p for p, k in heads if k == level + 1]
    nxt = [p for p, k in heads if k > level + 1]
    start = mine[0] if mine else 0
    end = nxt[0] if nxt else len(t)
    hits = [None] * (level + 1 if nxt else level)
    blocks = thinking(t[start:end], int(per), int(last))
    print(f"--- {run} {game} L{level} ({'solved' if len(hits) > level else 'NOT solved in this run'}): last {len(blocks)} thinking blocks ---")
    for b in blocks:
        print("*", b)


def cmd_talk(game, level, run=None, chars=6000):
    s = ST[(game, level)]
    run = run or s["solved_runs"][0]
    d = HERE.parent / "m2-winners-20261001/runs" / TRANSCRIPTS[run] / "transcripts"
    t = next(d.glob(f"{game}-*_p0.txt")).read_text(errors="replace")
    hits = [m.start() for m in re.finditer(r"-> LEVEL COMPLETED", t)]
    if len(hits) <= level:
        print(f"only {len(hits)} completions in {run} transcript"); return
    end = hits[level]
    start = hits[level - 1] if level else 0
    chunk = t[max(start, end - int(chars)):end + 200]
    chunk = "\n".join(l for l in chunk.splitlines() if not l.startswith("- ") and len(l) < 3000)
    print(f"--- {run} {game} L{level}: last {len(chunk)} chars before completion #{level + 1} ---")
    print(chunk)


def cmd_src(game, context=40):
    f = next((S.ENV / game).glob(f"*/{game}.py"))
    L = f.read_text().splitlines()
    for i, l in enumerate(L):
        if "self.win()" in l or "next_level()" in l:
            print(f"--- {f.name}:{i + 1} ---")
            print("\n".join(f"{j + 1}: {L[j]}" for j in range(max(0, i - int(context)), min(len(L), i + 8))))


def cmd_try(game, level, path):
    s, g, fr = entry(game, level)
    acts = json.loads(Path(path).read_text()) if Path(path).exists() else json.loads(path)
    for i, a in enumerate(acts):
        fr = act(g, a)
        if fr.levels_completed > level:
            print(f"LEVEL COMPLETE after {i + 1} actions"); return True
        if fr.state == GameState.GAME_OVER:
            print(f"GAME_OVER after {i + 1} actions"); return False
    print(f"not complete after {len(acts)} actions"); print(ascii_(grid(fr))); return False


if __name__ == "__main__":
    c, *a = sys.argv[1:]
    {"frame": lambda: cmd_frame(a[0], int(a[1])), "seg": lambda: cmd_seg(a[0], int(a[1]), *a[2:3]),
     "talk": lambda: cmd_talk(a[0], int(a[1]), *a[2:4]),
     "think": lambda: cmd_think(a[0], int(a[1]), *[x or None for x in a[2:3]], *a[3:5]),
     "both": lambda: (cmd_seg(a[0], int(a[1])), cmd_think(a[0], int(a[1]), None, *(a[2:4] or [7, 700]))), "src": lambda: cmd_src(a[0], *a[1:2]),
     "try": lambda: cmd_try(a[0], int(a[1]), a[2])}[c]()
