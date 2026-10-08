"""Candidate speed/intelligence metrics per game, scored for (a) noise between identical runs and (b) early prediction of outcome.

  python3 -B speed_metrics.py runs/out-control runs/out-control-r2 runs/out-s1

Per action we have cumulative wall seconds and generated tokens (benchmark.json history) and the board after it (events.jsonl).
Early-window metrics use only actions taken before EARLY_MIN of each game's own clock.
"""
import ast, hashlib, json, math, statistics, sys, itertools
from pathlib import Path

EARLY_MIN = 30


def load(out):
    b = json.loads((out / "benchmark.json").read_text())
    games = {}
    for r in b["game_runs"]:
        g = r["game_id"]
        h = r["history"] if isinstance(r["history"], list) else ast.literal_eval(r["history"])
        base = r["base_actions_per_level"] if isinstance(r["base_actions_per_level"], list) else ast.literal_eval(r["base_actions_per_level"])
        ev = [json.loads(l) for l in (out / "artifacts" / f"{g}_p0_events.jsonl").open()]
        acts = [e for e in ev if e["type"] == "action"]
        init = next(e for e in ev if e["type"] == "initial")
        assert len(acts) == len(h), (g, len(acts), len(h))
        steps = []
        for x, e in zip(h, acts):
            steps.append(dict(t=x["wallclock_seconds"], tok=x["generated_tokens"], lvl=int(e.get("score") or 0),
                              changed=str(e.get("board_changed")) == "True", over=str(e.get("game_over")) == "True",
                              board=hashlib.md5(json.dumps(e["board"]).encode()).hexdigest(), aid=x["action"]["id"]))
        games[g[:4]] = dict(steps=steps, base=base, init=hashlib.md5(json.dumps(init["board"]).encode()).hexdigest(),
                            wall=float(r["final_wallclock_seconds"]), score=float(r["final_score"] or 0))
    return games


def metrics(G, horizon_s=None):
    s = [x for x in G["steps"] if horizon_s is None or x["t"] <= horizon_s]
    wall = horizon_s or G["wall"]
    m = {}
    m["levels"] = s[-1]["lvl"] if s else 0
    n = len(s); tok = sum(x["tok"] for x in s) or 1
    m["actions"] = n
    m["s_per_action"] = (s[-1]["t"] / n) if n else wall
    m["tok_per_action"] = tok / max(n, 1)
    m["batched_frac"] = sum(x["tok"] == 0 for x in s) / max(n, 1)   # actions issued inside a multi-action call
    m["noop_frac"] = sum(not x["changed"] for x in s) / max(n, 1)
    # novelty: board never seen before on this level attempt
    seen, lvl, novel, orient, lvl_tok, lvl_n = {G["init"]}, 0, 0, 0, {}, {}
    first = True
    for x in s:
        if first: orient += x["tok"]; first = False
        lvl_tok[lvl] = lvl_tok.get(lvl, 0) + x["tok"]; lvl_n[lvl] = lvl_n.get(lvl, 0) + 1
        if x["board"] not in seen: novel += 1; seen.add(x["board"])
        if x["lvl"] > lvl: lvl, seen, first = x["lvl"], {x["board"]}, True
    m["novel_frac"] = novel / max(n, 1)
    m["novel_per_ktok"] = 1000 * novel / tok
    m["novel_per_min"] = 60 * novel / wall
    m["orient_tok_frac"] = orient / tok                     # tokens on the first action of each level
    m["game_overs"] = sum(x["over"] for x in s)
    # time / tokens to first level
    t1 = next((x for x in s if x["lvl"] >= 1), None)
    m["min_to_L1"] = (t1["t"] if t1 else wall) / 60
    m["ktok_to_L1"] = sum(x["tok"] for x in s if t1 is None or x["t"] <= t1["t"]) / 1000
    # difficulty-normalised progress: human-baseline actions of completed levels per million tokens / per hour
    hb = sum(G["base"][:m["levels"]])
    m["human_actions_per_Mtok"] = 1e6 * hb / tok
    m["human_actions_per_hour"] = 3600 * hb / wall
    done = [lvl_n[i] / G["base"][i] for i in range(m["levels"]) if i in lvl_n]
    m["eff_ratio"] = statistics.median(done) if done else float("nan")
    # tokens per completed level relative to its human baseline (lower = quicker understanding)
    tb = [lvl_tok[i] / G["base"][i] for i in range(m["levels"]) if i in lvl_tok]
    m["tok_per_base_action"] = statistics.median(tb) if tb else float("nan")
    return m


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
    for k, i in enumerate(o): r[i] = k
    return r


def spearman(a, b):
    p = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(p) < 4: return float("nan")
    ra, rb = rank([x for x, _ in p]), rank([y for _, y in p])
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


if __name__ == "__main__":
    runs = {Path(p).name: load(Path(p)) for p in sys.argv[1:]}
    names = list(runs); games = sorted(set.intersection(*(set(r) for r in runs.values())))
    full = {n: {g: metrics(runs[n][g]) for g in games} for n in names}
    early = {n: {g: metrics(runs[n][g], EARLY_MIN * 60) for g in games} for n in names}
    keys = [k for k in full[names[0]][games[0]]]
    a, b = names[0], names[1]
    print(f"{len(games)} games. Columns: run means; rho_rep = per-game Spearman between identical runs {a},{b};")
    print(f"pair_win = over same-game run pairs that ended with different levels, how often the EARLY ({EARLY_MIN} min) metric picks the winner.\n")
    print(f"{'metric':24s} " + " ".join(f"{n[-10:]:>10s}" for n in names) + f" {'rho_rep':>8s} {'early~final':>11s} {'pair_win':>9s}")
    for k in keys:
        mean = lambda n: statistics.mean([v for v in (full[n][g][k] for g in games) if not math.isnan(v)] or [float('nan')])
        rho = spearman([full[a][g][k] for g in games], [full[b][g][k] for g in games])
        ef = spearman([early[n][g][k] for n in names for g in games], [full[n][g]["levels"] for n in names for g in games])
        wins = tot = 0
        for g in games:
            for x, y in itertools.combinations(names, 2):
                dl = full[x][g]["levels"] - full[y][g]["levels"]
                de = early[x][g][k] - early[y][g][k]
                if dl == 0 or de == 0 or math.isnan(de): continue
                tot += 1; wins += (dl > 0) == (de > 0)
        pw = f"{wins}/{tot}" if tot else "-"
        print(f"{k:24s} " + " ".join(f"{mean(n):10.3f}" for n in names) + f" {rho:8.2f} {ef:11.2f} {pw:>9s}")
