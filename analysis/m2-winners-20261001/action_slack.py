"""Are actions (free up to ~0.93x human baseline) being hoarded while tokens (the binding constraint) are spent?

Per level attempt: actions used, human baseline, generated tokens spent on that level, completed or not.
Requires per-request logs (<game>_p0_requests.jsonl) for token attribution; falls back to action counts only.
"""
import json, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from wastewalk import walk

def levels(out):
    b = json.loads((out / "benchmark.json").read_text())
    G = walk(out) if list(out.glob("*_p0_requests.jsonl")) else {}
    rows = []
    for r in b["game_runs"]:
        g = r["game_id"][:4]; L = r["levels_completed"] or 0
        tok = {}
        for q in G.get(g, []):
            tok[q["level"]] = tok.get(q["level"], 0) + q["gen"]
        for i, (a, base) in enumerate(zip(r["actions_per_level"], r["base_actions_per_level"])):
            if i > L: break
            rows.append(dict(game=g, level=i + 1, done=i < L, actions=a, base=base, tokens=tok.get(i, None)))
    return rows

if __name__ == "__main__":
    rows = levels(Path(sys.argv[1]))
    done = [r for r in rows if r["done"]]; stuck = [r for r in rows if not r["done"]]
    under = [r for r in done if r["actions"] < 0.93 * r["base"]]
    print(f"completed levels {len(done)}: median actions/baseline {statistics.median(r['actions']/r['base'] for r in done):.2f}; "
          f"{len(under)} solved under 0.93x baseline (credit capped, slack unused)")
    print(f"  unused free actions on completed levels: {sum(0.93*r['base']-r['actions'] for r in under):.0f} "
          f"(vs {sum(r['actions'] for r in done)} actions used)")
    print(f"stuck (never-completed) levels {len(stuck)}:")
    for r in stuck:
        t = f"{r['tokens']:,}" if r["tokens"] is not None else "?"
        print(f"  {r['game']} L{r['level']}: actions {r['actions']:4d} / baseline {r['base']:4d} ({r['actions']/r['base']:4.2f}x)  tokens on level {t}")
    if any(r["tokens"] for r in done):
        tl = [(r["tokens"], r["actions"] / r["base"]) for r in done if r["tokens"]]
        print("completed levels, tokens spent vs action ratio (sorted by tokens):")
        for r in sorted(done, key=lambda r: -(r["tokens"] or 0))[:12]:
            print(f"  {r['game']} L{r['level']}: tokens {r['tokens']:7,}  actions {r['actions']:4d} / baseline {r['base']:4d} ({r['actions']/r['base']:.2f}x)")
