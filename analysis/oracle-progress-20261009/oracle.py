"""Oracle progress: exact distance-to-next-level for every state a logged run visited, by exhaustive local search.

  .venv-python -I oracle.py build   -> graphs/<game>-L<k>.json  (state-hash graph + distance-to-goal per state)
  .venv-python -I oracle.py score   -> per run x level rows in rows.jsonl

Per (game, level): start from the level's first state (replayed from one run), BFS over keyboard actions + one click per
colour component + every click any logged run made on that level, deduplicating by last-frame hash. If the reachable set
is exhausted within the cap, distance-to-goal is exact for every state; otherwise the level is marked partial.
Then every run is replayed and each state is looked up: d(s) = actions still needed under perfect play.
"""
import collections, copy, hashlib, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from replay import game, GameAction, ActionInput, P
from arcengine import GameState

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "metric-validation-20261008"))
from validate import RUNS, HIDDEN  # noqa: E402

NODE_CAP, TIME_CAP, LEVELS = 40000, 240, 3


def H(fr):
    return hashlib.md5(np.asarray(fr.frame[-1]).tobytes()).hexdigest()


def components(fr):
    F = np.asarray(fr.frame[-1]); seen = np.zeros_like(F, bool); out = []
    for y in range(64):
        for x in range(64):
            if seen[y, x]: continue
            col = F[y, x]; st = [(y, x)]; seen[y, x] = True; cells = []
            while st:
                yy, xx = st.pop(); cells.append((yy, xx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < 64 and 0 <= nx < 64 and not seen[ny, nx] and F[ny, nx] == col:
                        seen[ny, nx] = True; st.append((ny, nx))
            if len(cells) < 1500:
                yy, xx = cells[len(cells) // 2]; out.append((xx, yy))
    return out


def act(a):
    i, d = a
    return ActionInput(id=GameAction[i], data=d or {})


def run_actions():
    """(game, level) -> set of actions any run took there; plus one replay prefix reaching each level start."""
    acts, prefix = collections.defaultdict(set), {}
    for name, d in RUNS.items():
        b = json.loads((d / "benchmark.json").read_text())
        for r in b["game_runs"]:
            g = r["game_id"]; h = P(r["history"]); apl = P(r["actions_per_level"]); k = 0; done = int(r["levels_completed"] or 0)
            for L, n in enumerate(apl[:LEVELS]):
                for x in h[k:k + n]:
                    a = x["action"]; acts[(g, L)].add((a["id"], tuple(sorted((a.get("data") or {}).items()))))
                if (g, L) not in prefix and L <= done: prefix[(g, L)] = [x["action"] for x in h[:k]]
                k += n
    return acts, prefix


def build(job):
    (g, L), prefix, extra = job
    out = HERE / "graphs" / f"{g}-L{L}.json"
    if out.exists(): return json.loads(out.read_text())["summary"]
    env = game(g); fr = env.perform_action(ActionInput(id=GameAction.RESET))
    for a in prefix: fr = env.perform_action(ActionInput(id=GameAction[a["id"]], data=a.get("data") or {}))
    if fr.levels_completed != L: return dict(game=g, level=L, error="prefix did not reach level")
    keyb = [(f"ACTION{a}", None) for a in fr.available_actions if a != 6]
    clicks = 6 in fr.available_actions
    extra = [(i, dict(d)) for i, d in extra if i == "ACTION6"]
    # frontier holds action paths from the level root (a game copy is 0.2-1.2 MB; 40k of them do not fit in RAM)
    root = H(fr); edges = collections.defaultdict(set); goal = set(); seen = {root}
    q = collections.deque([()]); t0 = time.time(); n = 0
    while q and len(seen) < NODE_CAP and time.time() - t0 < TIME_CAP:
        path = q.popleft(); s = copy.deepcopy(env); f0 = fr
        for a in path: f0 = s.perform_action(act(a))
        k0 = H(f0)
        cand = keyb + ([("ACTION6", {"x": x, "y": y}) for x, y in components(f0)] + extra if clicks else [])
        for a in cand:
            c = copy.deepcopy(s); f = c.perform_action(act(a)); n += 1
            if f.levels_completed > L: goal.add(k0); continue
            if f.state == GameState.GAME_OVER: continue
            k = H(f)
            if k == k0: continue
            edges[k].add(k0)                       # reverse edge
            if k not in seen: seen.add(k); q.append(path + (a,))
    exhausted = not q
    dist = {k: 1 for k in goal}; dq = collections.deque(goal)
    while dq:
        k = dq.popleft()
        for p in edges.get(k, ()):
            if p not in dist: dist[p] = dist[k] + 1; dq.append(p)
    summ = dict(game=g, level=L, states=len(seen), exhausted=exhausted, d_root=dist.get(root), secs=round(time.time() - t0), steps=n)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(summary=summ, root=root, dist=dist, seen=sorted(seen))))
    return summ


def score():
    rows = []
    graphs = {}
    for f in (HERE / "graphs").glob("*.json"):
        j = json.loads(f.read_text()); s = j["summary"]
        graphs[(s["game"], s["level"])] = (s, j["dist"], set(j["seen"]))
    for name, d in RUNS.items():
        b = json.loads((d / "benchmark.json").read_text())
        for r in b["game_runs"]:
            g = r["game_id"]; h = P(r["history"]); apl = P(r["actions_per_level"])
            if not any((g, L) in graphs for L in range(LEVELS)): continue
            env = game(g); fr = env.perform_action(ActionInput(id=GameAction.RESET)); k = 0; done = int(r["levels_completed"] or 0)
            for L, n in enumerate(apl[:LEVELS]):
                if (g, L) in graphs and (n or L == done):
                    s, dist, seen = graphs[(g, L)]
                    ds = [dist.get(H(fr))]; cov = [H(fr) in seen]; t = [h[k - 1]["wallclock_seconds"] if k else 0.0]; tok = [0]
                    for x in h[k:k + n]:
                        fr = env.perform_action(ActionInput(id=GameAction[x["action"]["id"]], data=x["action"].get("data") or {}))
                        hk = H(fr); ds.append(0 if fr.levels_completed > L else dist.get(hk)); cov.append(hk in seen or fr.levels_completed > L)
                        t.append(x["wallclock_seconds"]); tok.append(x["generated_tokens"])
                    rows.append(dict(run=name, game=g[:4], level=L, solved=L < done, exact=s["exhausted"], d0=s["d_root"],
                                     d=ds, covered=sum(cov) / len(cov), t=t, tok=tok, base=P(r["base_actions_per_level"])[L]))
                else:
                    for x in h[k:k + n]:
                        fr = env.perform_action(ActionInput(id=GameAction[x["action"]["id"]], data=x["action"].get("data") or {}))
                k += n
                if L >= done: break
    with open(HERE / "rows.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} run-level rows")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        acts, prefix = run_actions()
        jobs = [((g, L), prefix[(g, L)], sorted(acts[(g, L)])) for (g, L) in sorted(prefix)]
        with mp.Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as pool:
            for s in pool.imap_unordered(build, jobs):
                print(json.dumps(s), flush=True)
    else:
        score()
