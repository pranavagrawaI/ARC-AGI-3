"""Build the 40 hard level-entry start states for the oracle-hint experiment.

  solvers/tycho_nooa/.venv/bin/python -B starts.py      -> starts.json

A hard level is one reached in a dfranzen run (control, r2, s1) and left unsolved by at least one run.
For each, the prefix is the logged history up to the level boundary of a run that reached it (a stuck run when
one exists, since that is the state it failed from). Checks, on the raw engine:
  1. the prefix lands on level L, not GAME_OVER;
  2. every run that reached L lands on the same frame (level entry is path-independent);
  3. where a sibling solved L, its own level-L segment, played from OUR prefix's state, completes L.
"""
import hashlib, importlib.util, json, sys
from pathlib import Path
from arcengine import ActionInput, ARCBaseGame, GameAction, GameState

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "metric-validation-20261008"))
from validate import RUNS, P  # noqa: E402

ENV = Path.home() / "projects/arc-agi-3-reference/competition-env/environment_files"
NAMES = ["dfz-control", "dfz-r2", "dfz-s1"]


def engine(gid):
    f = next((ENV / gid[:4]).glob(f"*/{gid[:4]}.py"))
    spec = importlib.util.spec_from_file_location(gid[:4], f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cls = [v for v in vars(m).values() if isinstance(v, type) and issubclass(v, ARCBaseGame) and v is not ARCBaseGame][0]
    return cls()


def play(gid, actions):
    g = engine(gid)
    fr = g.perform_action(ActionInput(id=GameAction.RESET))
    for a in actions:
        fr = g.perform_action(ActionInput(id=GameAction[a["id"]], data=a.get("data") or {}))
    return g, fr


def fhash(fr):
    return hashlib.sha256(json.dumps([[list(map(int, row)) for row in f] for f in fr.frame[-1:]]).encode()).hexdigest()[:16]


def runs():
    out = {}
    for n in NAMES:
        b = json.loads((RUNS[n] / "benchmark.json").read_text())
        for r in b["game_runs"]:
            h, apl = P(r["history"]), P(r["actions_per_level"])
            out[(n, r["game_id"][:4])] = dict(gid=r["game_id"], h=[x["action"] for x in h], apl=apl, done=int(r["levels_completed"] or 0))
    return out


def main():
    R = runs()
    keys = sorted({(g, L) for (n, g), r in R.items() for L in range(min(r["done"], len(r["apl"]) - 1) + 1)})
    starts = []
    for g, L in keys:
        reached = [n for n in NAMES if (n, g) in R and R[(n, g)]["done"] >= L]
        stuck = [n for n in reached if R[(n, g)]["done"] == L]
        solved = [n for n in reached if R[(n, g)]["done"] > L]
        if not stuck or L >= len(R[(stuck[0], g)]["apl"]):
            continue
        hashes, ok = {}, True
        for n in reached:
            r = R[(n, g)]
            k = sum(r["apl"][:L])
            _, fr = play(r["gid"], r["h"][:k])
            hashes[n] = fhash(fr)
            if fr.levels_completed != L or fr.state == GameState.GAME_OVER:
                ok = False
        src = stuck[0]
        r = R[(src, g)]
        prefix = r["h"][:sum(r["apl"][:L])]
        sib = []
        for n in solved:
            s = R[(n, g)]
            k = sum(s["apl"][:L])
            seg = s["h"][k:k + s["apl"][L]]
            _, fr = play(r["gid"], prefix + seg)
            sib.append(dict(run=n, actions=len(seg), solves_from_our_prefix=fr.levels_completed > L))
        starts.append(dict(game=g, gid=r["gid"], level=L, prefix_run=src, prefix=prefix, prefix_len=len(prefix),
                           lands_ok=ok, entry_hashes=hashes, same_entry=len(set(hashes.values())) == 1,
                           stuck_runs=stuck, solved_runs=solved, sibling_check=sib,
                           klass="sibling" if solved else "never"))
        print(f"{g} L{L} {starts[-1]['klass']:7s} prefix={len(prefix):4d} lands={ok} same_entry={starts[-1]['same_entry']} "
              f"sib={[(x['run'][4:], x['actions'], x['solves_from_our_prefix']) for x in sib]}")
    (HERE / "starts.json").write_text(json.dumps(starts, indent=1))
    print(len(starts), "starts;", sum(s["klass"] == "sibling" for s in starts), "sibling,", sum(s["klass"] == "never" for s in starts), "never")


if __name__ == "__main__":
    main()
