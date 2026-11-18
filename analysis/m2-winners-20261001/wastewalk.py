"""Waste walk over a Duck/TAAF run that saved per-request logs (<game>_p0_requests.jsonl + artifacts/<game>_p0_events.jsonl).

Classifies every generated token by what it bought:
  productive   - generated on a level that the game later completed
  stuck        - generated on the level the game was on when the run ended (never completed)
and per request: did the turn move the game (action counter advanced) or not.
"""
import json, sys, collections
from pathlib import Path

def level_by_action(events):
    lv = [0]
    for line in events.open():
        e = json.loads(line)
        if e["type"] == "action":
            lv.append(int(e.get("score") or 0))
    return lv  # lv[k] = levels completed after k actions

def walk(out: Path):
    G = {}
    for req in sorted(out.glob("*_p0_requests.jsonl")):
        g = req.name[:4]
        lv = level_by_action(out / "artifacts" / req.name.replace("_requests.jsonl", "_events.jsonl"))
        final = lv[-1]
        resp = [json.loads(l) for l in req.open()]
        resp = [r for r in resp if r["event"] == "response"]
        rows = []
        for i, r in enumerate(resp):
            u = r.get("usage") or {}
            a0 = r["action"]
            a1 = resp[i + 1]["action"] if i + 1 < len(resp) else len(lv) - 1
            lvl = lv[min(a0 - 1, len(lv) - 1)] if a0 >= 1 else 0
            rows.append(dict(game=g, step=r["analysis_step"], idx=r["request_index_within_turn"], a0=a0, a1=a1,
                             level=lvl, final=final, finish=r.get("finish_reason"),
                             gen=u.get("completion_tokens", 0), think=u.get("reasoning_tokens", 0),
                             prompt=u.get("prompt_tokens", 0),
                             cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                             image=(u.get("prompt_tokens_details") or {}).get("image_tokens", 0)))
        G[g] = rows
    return G

def report(G):
    allr = [r for rows in G.values() for r in rows]
    gen = sum(r["gen"] for r in allr); think = sum(r["think"] for r in allr)
    prompt = sum(r["prompt"] for r in allr); cached = sum(r["cached"] for r in allr)
    moved = [r for r in allr if r["a1"] > r["a0"]]
    stuck = [r for r in allr if r["level"] == r["final"] and r["final"] < 99]
    print(f"requests={len(allr)} generated={gen:,} thinking={think/gen:.0%} prompt={prompt:,} cached={cached/prompt:.0%}")
    print(f"finish_reason: {collections.Counter(r['finish'] for r in allr)}")
    print(f"requests that moved the game: {len(moved)}/{len(allr)} ({len(moved)/len(allr):.0%}); "
          f"their tokens {sum(r['gen'] for r in moved)/gen:.0%} of generation")
    print(f"tokens on the never-completed final level: {sum(r['gen'] for r in stuck)/gen:.0%}")
    print("\nper game: final levels | gen tokens | % on unfinished final level | requests | moved% | tokens/action")
    for g, rows in sorted(G.items()):
        gg = sum(r["gen"] for r in rows); st = sum(r["gen"] for r in rows if r["level"] == r["final"])
        acts = max(r["a1"] for r in rows)
        mv = sum(1 for r in rows if r["a1"] > r["a0"])
        print(f"  {g} L{rows[0]['final']} gen={gg:7,} stuck={st/gg:4.0%} req={len(rows):3d} moved={mv/len(rows):4.0%} tok/act={gg/max(acts,1):5.0f}")

if __name__ == "__main__":
    report(walk(Path(sys.argv[1])))
