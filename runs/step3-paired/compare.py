"""Paired readback for Step 3: per-episode levels/actions/turns per arm, firings for the critic arm.

usage: compare.py incumbent/kaggle-output critic/kaggle-output
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def episodes(out: Path) -> dict[str, dict]:
    rows = {}
    for ev_path in sorted(out.glob("artifacts/*_events.jsonl")):
        stem = ev_path.name.replace("_events.jsonl", "")
        ev = [json.loads(l) for l in ev_path.read_text().splitlines() if l.strip()]
        acts = [e for e in ev if e.get("type") == "action"]
        levels = max((int(e.get("level") or 1) for e in ev), default=1) - 1
        turns = len({e.get("analysis_step") for e in acts})
        first_l2 = next((i for i, e in enumerate(acts) if int(e.get("level") or 1) >= 2), None)
        trace = out / "transcripts" / f"{stem}_critic.jsonl"
        firings = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()] if trace.exists() else []
        uptake = 0
        for f in firings:
            rec = (f.get("recommended_actions") or [None])[0]
            n = f["action_num"]
            if rec and n < len(acts):
                a = acts[n]
                disp = a.get("action_display") or ""
                if rec.get("action") == "MOUSE":
                    if disp == f"MOUSE(row={rec.get('row')}, col={rec.get('col')})":
                        uptake += 1
                elif disp == rec.get("action"):
                    uptake += 1
        rows[stem] = dict(
            game=stem.split("_")[0], levels=levels, actions=len(acts), turns=turns,
            actions_to_l1=first_l2, firings=len(firings),
            causes=dict(Counter(f["cause"] for f in firings)),
            critic_s=round(sum(f["seconds"] for f in firings), 1),
            parsed=sum(1 for f in firings if f["parsed"]), uptake=uptake,
            errors=sum(1 for f in firings if f.get("error")),
        )
    return rows


arms = {name: episodes(Path(p)) for name, p in zip(("incumbent", "critic"), sys.argv[1:3])}
print(f"{'episode':<20}" + "".join(f"{a:>28}" for a in arms))
games = sorted({s for r in arms.values() for s in r})
for s in games:
    cells = []
    for a in arms:
        r = arms[a].get(s)
        cells.append("(missing)" if r is None else f"L{r['levels']} {r['actions']:>3}a {r['turns']:>2}t" + (f" f{r['firings']} u{r['uptake']}" if a == "critic" else ""))
    print(f"{s:<20}" + "".join(f"{c:>28}" for c in cells))
print()
for a, rows in arms.items():
    tot = sum(r["levels"] for r in rows.values())
    any_l = sum(1 for r in rows.values() if r["levels"] > 0)
    by_game = defaultdict(int)
    for r in rows.values():
        by_game[r["game"]] += r["levels"]
    print(f"{a}: episodes={len(rows)} TOTAL LEVELS={tot} episodes>=1 level={any_l} by game={dict(by_game)} "
          f"mean turns={sum(r['turns'] for r in rows.values())/max(1,len(rows)):.1f}")
    if a == "critic":
        f = sum(r["firings"] for r in rows.values()); p = sum(r["parsed"] for r in rows.values())
        u = sum(r["uptake"] for r in rows.values()); e = sum(r["errors"] for r in rows.values())
        cs = sum(r["critic_s"] for r in rows.values())
        causes = Counter()
        for r in rows.values():
            causes.update(r["causes"])
        print(f"   firings={f} parsed={p} uptake={u} errors={e} critic wall={cs:.0f}s ({100*cs/max(1,1200*len(rows)):.1f}% of cap) causes={dict(causes)}")
