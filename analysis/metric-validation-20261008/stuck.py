"""Two offline checks on stuck levels, pooled over replicate families.

  python3 -B stuck.py

A. Decisive-action contact (Makarov 743723): for a level solved in run B but never solved in run A (same game),
   did A ever take B's final, level-completing action (same id; clicks within 2 cells)? And how long had A held it?
   Recognition latency in solved levels: actions from first contact with the decisive action to the solve.
B. Solve hazard by time-in-level: of attempts still open at minute m of the level, the share solved in the next 10 min,
   and the share of all tokens spent past each minute. Sizes the "free the stuck streams" lever.
"""
import ast, json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from validate import RUNS, P

FAMILIES = {"dfranzen": ["dfz-control", "dfz-r2", "dfz-s1"], "flash-next": ["kf2", "c1", "p1", "p2", "r1", "g1", "g2"]}


def levels(d):
    b = json.loads((d / "benchmark.json").read_text()); out = {}
    for r in b["game_runs"]:
        h, apl = P(r["history"]), P(r["actions_per_level"]); done = int(r["levels_completed"] or 0)
        k = 0
        for L, n in enumerate(apl):
            seg = h[k:k + n]; t0 = h[k - 1]["wallclock_seconds"] if k else 0.0
            if seg or L == done:
                out[(r["game_id"][:4], L)] = dict(done=L < done, seq=[(x["action"]["id"], x["action"].get("data") or {}, x["wallclock_seconds"] - t0, x["generated_tokens"]) for x in seg],
                                                  sec=(seg[-1][ "wallclock_seconds"] if L < done else float(r["final_wallclock_seconds"])) - t0 if seg else float(r["final_wallclock_seconds"]) - t0)
            k += n
    return out


def same(a, b):
    if a[0] != b[0]: return False
    if a[0] != "ACTION6": return True
    return abs(a[1].get("x", -99) - b[1].get("x", 99)) <= 2 and abs(a[1].get("y", -99) - b[1].get("y", 99)) <= 2


if __name__ == "__main__":
    for fam, names in FAMILIES.items():
        Lv = {n: levels(RUNS[n]) for n in names}
        print(f"\n## {fam} ({len(names)} runs)")
        # A: contact
        rows = []
        for a in names:
            for b in names:
                if a == b: continue
                for key, la in Lv[a].items():
                    lb = Lv[b].get(key)
                    if la["done"] or not lb or not lb["done"] or not lb["seq"]: continue
                    dec = lb["seq"][-1]
                    if dec[0] in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"): continue   # a move is always "tried"; not discriminating
                    hit = next((i for i, x in enumerate(la["seq"]) if same(x, dec)), None)
                    rows.append((key, dec[0], hit is not None, (la["sec"] - la["seq"][hit][2]) / 60 if hit is not None else None, len(la["seq"])))
        uniq = {(k, d): (h, m) for k, d, h, m, n in rows}
        hits = [v for v in uniq.values() if v[0]]
        print(f"A. stuck-in-A / solved-in-B level pairs with a non-move decisive action: {len(rows)} pairs, {len(uniq)} distinct levels")
        print(f"   A had already taken B's decisive action in {sum(r[2] for r in rows)}/{len(rows)} pairs; "
              f"median minutes held after first contact {statistics.median([r[3] for r in rows if r[2]]) if any(r[2] for r in rows) else float('nan'):.0f}")
        lat = []
        for n in names:
            for key, l in Lv[n].items():
                if not l["done"] or not l["seq"] or l["seq"][-1][0] in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"): continue
                dec = l["seq"][-1]; i = next(i for i, x in enumerate(l["seq"]) if same(x, dec))
                lat.append(((len(l["seq"]) - 1 - i) / len(l["seq"]), (l["seq"][-1][2] - l["seq"][i][2]) / max(l["seq"][-1][2], 1)))
        if lat:
            print(f"   solved levels (non-move finish, n={len(lat)}): first contact with the decisive action comes, median, after "
                  f"{1 - statistics.median(x for x, _ in lat):.0%} of the level's actions and {1 - statistics.median(y for _, y in lat):.0%} of its time")
        # B: hazard by time-in-level
        att = [l for n in names for l in Lv[n].values()]
        print("B. minute-in-level | open | solved in next 10 min | share of all tokens spent past this minute")
        tot = sum(x[3] for l in att for x in l["seq"]) or 1
        for m in (0, 5, 10, 20, 30, 45, 60, 90):
            open_ = [l for l in att if l["sec"] > m * 60]
            sol = [l for l in open_ if l["done"] and l["sec"] <= (m + 10) * 60]
            past = sum(x[3] for l in att for x in l["seq"] if x[2] > m * 60)
            print(f"   {m:3d} | {len(open_):4d} | {len(sol) / max(len(open_), 1):5.0%} | {past / tot:5.0%}")
