"""Readout of the oracle-hint experiment by the pre-registered rule (EXPERIMENT-CARD.md, amendment 1).

  python3 -B analyze.py            -> RESULT tables on stdout + readout.json

solve = events.jsonl kind=solved for the attempt, or benchmark.json state 'won'. Completing a game's final level ends
the game as WIN without a solved event (8 attempts on lp85 L7, tu93 L8, cn04 L5, sc25 L5); 'won' covers those.
Per level: p0 = mean(A0 r1, r2); p1 = mean(A1 r1, r2); p2 = A2. d1 = p1 - p0, d2 = p2 - p0, over hinted (sibling) levels.
"""
import json, random, statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(session):
    out = HERE / "sessions" / session / "output"
    ev_path = next(out.rglob("events.jsonl"), None)
    bm_path = next(out.rglob("benchmark.json"), None)
    attempts = json.loads((HERE / "sessions" / session / "attempts.json").read_text())
    ev = [json.loads(l) for l in ev_path.read_text().splitlines()] if ev_path else []
    runs = {r["game_id"]: r for r in json.loads(bm_path.read_text())["game_runs"]} if bm_path else {}
    first = lambda name, kind: next((e for e in ev if e["attempt"] == name and e["kind"] == kind), None)
    rows = {}
    for a in attempts:
        n = a["name"]
        adm, sol = first(n, "admitted"), first(n, "solved")
        r = runs.get(n, {})
        won = r.get("state") == "won"
        hist = r.get("history") or []
        rows[n] = dict(key=a["key"], arm=a["arm"], admitted=adm is not None, hinted=bool(adm and adm.get("hinted")),
                       solved=sol is not None or won, solved_event=sol is not None, won=won,
                       t_solve=(sol["t"] - adm["t"]) if (sol and adm) else None,
                       actions=len(hist) if isinstance(hist, list) else None, state=r.get("state"), note=r.get("solver_note"))
    return rows


def boot(vals, n=10000, seed=1):
    rnd = random.Random(seed)
    ms = sorted(statistics.mean(rnd.choice(vals) for _ in vals) for _ in range(n))
    return ms[int(0.05 * n)], ms[int(0.95 * n)]


def verdict(d1, d2):
    if d1 >= 0.30:
        return "JUDGMENT"
    if d1 < 0.15 and d2 >= 0.30:
        return "EXECUTION"
    if d2 < 0.15:
        return "PERCEPTION/INTERFACE"
    return "INCONCLUSIVE"


def main():
    rows = {**load("a0"), **load("hints")}
    starts = {f"{s['game']}-L{s['level']}": s for s in json.loads((HERE / "starts.json").read_text())}
    by = defaultdict(lambda: defaultdict(list))
    for r in rows.values():
        if r["admitted"]:
            by[r["key"]][r["arm"]].append(r)
    problems = [n for n, r in rows.items() if r["admitted"] and r["solved_event"] and not r["won"]]
    final_level_wins = [n for n, r in rows.items() if r["admitted"] and r["won"] and not r["solved_event"]]
    problems += [n for n, r in rows.items() if r["admitted"] and (r["arm"] != "A0") != r["hinted"]]
    p = lambda key, arm: statistics.mean(x["solved"] for x in by[key][arm]) if by[key][arm] else None
    table = []
    for key, s in starts.items():
        table.append(dict(key=key, klass=s["klass"], p_inrun=len(s["solved_runs"]) / (len(s["solved_runs"]) + len(s["stuck_runs"])),
                          p0=p(key, "A0"), p1=p(key, "A1"), p2=p(key, "A2"),
                          n0=len(by[key]["A0"]), n1=len(by[key]["A1"]), n2=len(by[key]["A2"])))
    hinted = [t for t in table if t["p1"] is not None and t["p0"] is not None and t["p2"] is not None]
    d1 = [t["p1"] - t["p0"] for t in hinted]
    d2 = [t["p2"] - t["p0"] for t in hinted]
    res = dict(n_hinted_levels=len(hinted), mean_p0=statistics.mean(t["p0"] for t in hinted) if hinted else None,
               mean_p1=statistics.mean(t["p1"] for t in hinted) if hinted else None,
               mean_p2=statistics.mean(t["p2"] for t in hinted) if hinted else None,
               d1=statistics.mean(d1) if d1 else None, d1_ci90=boot(d1) if d1 else None,
               d2=statistics.mean(d2) if d2 else None, d2_ci90=boot(d2) if d2 else None)
    res["verdict"] = verdict(res["d1"], res["d2"]) if d1 else "NO DATA"
    # fresh-context cost: in-run solve fraction (3 dfranzen runs, with cross-level context) vs the A0 sprint rate
    res["mean_p_inrun_hinted"] = statistics.mean(t["p_inrun"] for t in hinted) if hinted else None
    never = [t for t in table if t["klass"] == "never"]
    res["never_A0_rate"] = statistics.mean(t["p0"] for t in never if t["p0"] is not None) if never else None
    a0 = [(by[k]["A0"][0]["solved"], by[k]["A0"][1]["solved"]) for k in by if len(by[k]["A0"]) == 2]
    res["A0_replicate_agreement"] = (sum(a == b for a, b in a0), len(a0))
    res["admitted"] = {arm: sum(r["admitted"] for r in rows.values() if r["arm"] == arm) for arm in ("A0", "A1", "A2")}
    res["integrity_problems"] = problems
    res["final_level_wins_without_event"] = final_level_wins
    print(f"{'level':10s} {'class':7s} inrun  p0    p1    p2   (n0,n1,n2)")
    for t in table:
        f = lambda v: "  -  " if v is None else f"{v:4.2f} "
        print(f"{t['key']:10s} {t['klass']:7s} {f(t['p_inrun'])} {f(t['p0'])} {f(t['p1'])} {f(t['p2'])} ({t['n0']},{t['n1']},{t['n2']})")
    print(json.dumps(res, indent=1))
    (HERE / "readout.json").write_text(json.dumps(dict(result=res, table=table, rows=rows), indent=1))


if __name__ == "__main__":
    main()
