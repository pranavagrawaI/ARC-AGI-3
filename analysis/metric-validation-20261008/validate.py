"""Do the speed/intelligence metrics track reality? Three tests over every real run on disk.

  python3 -B validate.py            (paths are fixed below; output is the evidence table in RESULT.md)

Inputs per run: benchmark.json only (per-action cumulative wall seconds, generated tokens, actions_per_level, baselines).
  T1 hidden:  run-level metric on the public run vs the hidden leaderboard score of the same system.
  T2 noise:   paired per-level sd(log) between byte-identical reruns, on two different stacks.
  T3 known:   paired per-level metrics on pairs whose difference is already known, read at a common horizon.
"""
import ast, json, math, statistics
from pathlib import Path

HOME = Path.home() / "projects"
M2 = HOME / "arc-agi-3/analysis/m2-winners-20261001/runs"
FN = HOME / "arc-agi-3-reference/duck-qwen38/runs/flash-next"
RUNS = {
    "duck36-sub": HOME / "arc-agi-3-reference/duck-critic-adapter/runs/submission-20260918/kaggle-output",
    "q38-commit": HOME / "arc-agi-3-reference/duck-qwen38/runs/qwen38-commit/live/kaggle-output",
    "kf2": FN / "out-kf2", "c1": FN / "out-c1", "p1": FN / "out-p1", "p2": FN / "out-p2",
    "r1": FN / "out-r1", "g1": FN / "out-g1", "g2": FN / "out-g2",
    "dfz-control": M2 / "out-control", "dfz-r2": M2 / "out-control-r2", "dfz-s1": M2 / "out-s1",
    "lean-cap4": HOME / "arc-agi-3/solvers/tycho_dfranzen/evidence/readout-20261007/lean-cap4/tycho-dfranzen",
    "lean-prune": HOME / "arc-agi-3/solvers/tycho_dfranzen/evidence/readout-20261007/lean-prune/tycho-dfranzen",
}
# hidden leaderboard scores of the system each public run represents (memory: duck-submission, qwen38, kf-c1-r1, kv-lever, incumbent-dfranzen)
HIDDEN = {"duck36-sub": 0.67, "q38-commit": 1.47, "kf2": 2.96, "p1": 3.08, "dfz-control": 23.87}
P = lambda v: v if isinstance(v, list) else ast.literal_eval(v)


def load(d):
    b = json.loads((d / "benchmark.json").read_text())
    G = {}
    for r in b["game_runs"]:
        h, apl, base = P(r["history"]), P(r["actions_per_level"]), P(r["base_actions_per_level"])
        lv, cum = [], 0
        for i, n in enumerate(apl):
            lv += [i] * n
        G[r["game_id"][:4]] = dict(
            t=[x["wallclock_seconds"] for x in h], tok=[x["generated_tokens"] for x in h], lv=lv, base=base,
            wall=float(r["final_wallclock_seconds"] or 0), done=int(r["levels_completed"] or 0),
            score=float(r["final_score"] or 0))
    return G


def attempts(g, H=None):
    """Per level attempt: solved?, seconds, tokens, actions, orientation tokens. Truncated at horizon H seconds."""
    wall = min(g["wall"], H) if H else g["wall"]
    idx = [k for k in range(len(g["t"])) if g["t"][k] <= wall]
    out, start_t = {}, 0.0
    for k in idx:
        L = g["lv"][k]
        a = out.setdefault(L, dict(sec=0, tok=0, act=0, orient=g["tok"][k], done=False, base=g["base"][L]))
        a["tok"] += g["tok"][k]; a["act"] += 1
    # a level is solved at its last action if a later level exists (or the game completed it)
    solved_upto = (max(g["lv"][k] for k in idx) if idx else 0)
    if not H or wall >= g["wall"]:
        solved_upto = g["done"]
    ends = {}
    for k in idx: ends[g["lv"][k]] = g["t"][k]
    prev = 0.0
    for L in sorted(out):
        a = out[L]; a["done"] = L < solved_upto
        a["sec"] = (ends[L] if a["done"] else wall) - prev
        prev = ends[L]
    if solved_upto not in out and solved_upto < len(g["base"]):     # open level with no action yet
        out[solved_upto] = dict(sec=wall - prev, tok=0, act=0, orient=0, done=False, base=g["base"][solved_upto])
    return out


def run_metrics(G, H=None):
    att = [a for g in G.values() for a in attempts(g, H).values()]
    sol = [a for a in att if a["done"]]
    hrs = sum(a["sec"] for a in att) / 3600
    tok = sum(a["tok"] for a in att) or 1
    walls = [min(g["wall"], H) if H else g["wall"] for g in G.values()]
    return dict(
        public_score=statistics.mean(g["score"] for g in G.values()) if not H else float("nan"),
        levels=len(sol),
        levels_per_game=len(sol) / len(G),
        hazard=len(sol) / hrs,                                         # levels per attempt-hour
        stuck_share=sum(a["sec"] for a in att if not a["done"]) / (hrs * 3600),
        orient_frac=sum(a["orient"] for a in att) / tok,
        human_act_per_Mtok=1e6 * sum(a["base"] for a in sol) / tok,
        human_act_per_game_hour=3600 * sum(a["base"] for a in sol) / sum(walls),
        levels_per_Mtok=1e6 * len(sol) / tok,
        tok_per_game_s=tok / sum(walls),                               # serving speed seen by one game
        med_act_over_base=statistics.median([a["act"] / a["base"] for a in sol]) if sol else float("nan"),
    )


def paired(GA, GB, H=None):
    """Same game, same level, solved in both: log ratio B/A for tokens, actions, seconds."""
    out = {k: [] for k in ("tok", "act", "sec", "orient")}
    for g in set(GA) & set(GB):
        A, B = attempts(GA[g], H), attempts(GB[g], H)
        for L in A:
            if L in B and A[L]["done"] and B[L]["done"]:
                for k in out:
                    if A[L][k] > 0 and B[L][k] > 0: out[k].append(math.log(B[L][k] / A[L][k]))
    res = {}
    for k, d in out.items():
        if len(d) < 3: res[k] = (len(d), float("nan"), float("nan"), float("nan")); continue
        m, sd = statistics.mean(d), statistics.stdev(d)
        res[k] = (len(d), math.exp(m), m / (sd / math.sqrt(len(d))), sd)
    return res


def spearman(a, b):
    r = lambda v: [sorted(v).index(x) for x in v]
    ra, rb = r(a), r(b); n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def logfit(x, y):
    lx, ly = [math.log(v) for v in x], [math.log(v) for v in y]
    mx, my = statistics.mean(lx), statistics.mean(ly)
    b = sum((p - mx) * (q - my) for p, q in zip(lx, ly)) / sum((p - mx) ** 2 for p in lx)
    res = [q - (my + b * (p - mx)) for p, q in zip(lx, ly)]
    return b, math.sqrt(sum(e * e for e in res) / (len(res) - 2))


if __name__ == "__main__":
    R = {n: load(d) for n, d in RUNS.items()}
    M = {n: run_metrics(G) for n, G in R.items()}
    keys = list(M["kf2"])

    print("## Run-level metrics (full horizon)\n")
    print("| run | games | " + " | ".join(keys) + " |"); print("|---" * (len(keys) + 2) + "|")
    for n in R:
        print(f"| {n} | {len(R[n])} | " + " | ".join(f"{M[n][k]:.3g}" for k in keys) + " |")

    print("\n## T1: public-run metric vs hidden score (5 systems)\n")
    hs = list(HIDDEN)
    print("Order by hidden: " + ", ".join(f"{n} {HIDDEN[n]}" for n in sorted(hs, key=HIDDEN.get)) + "\n")
    print("| metric | spearman | log-log slope | residual sd (log) | KF->P1 predicted / actual hidden ratio | leave-one-out miss (" + ", ".join(hs) + ") |"); print("|---|---|---|---|---|---|")
    for k in keys:
        x = [M[n][k] for n in hs]; y = [HIDDEN[n] for n in hs]
        if min(x) <= 0: continue
        b, s = logfit(x, y)
        pr = (M["p1"][k] / M["kf2"][k]) ** b
        loo = []
        for o in hs:                                   # leave one system out, predict it from the other four
            rest = [n for n in hs if n != o]
            bb, _ = logfit([M[n][k] for n in rest], [HIDDEN[n] for n in rest])
            mx = statistics.mean(math.log(M[n][k]) for n in rest); my = statistics.mean(math.log(HIDDEN[n]) for n in rest)
            loo.append(abs(my + bb * (math.log(M[o][k]) - mx) - math.log(HIDDEN[o])))
        print(f"| {k} | {spearman(x, y):.2f} | {b:.2f} | {s:.2f} | {pr:.2f} / {HIDDEN['p1']/HIDDEN['kf2']:.2f} | "
              + " ".join(f"{math.exp(e):.2f}x" for e in loo) + " |")

    print("\n## T2: noise floor between byte-identical reruns (paired per level, solved in both)\n")
    print("| pair | k | n | geo ratio | t | sd(log) |"); print("|---|---|---|---|---|---|")
    for a, b in (("dfz-control", "dfz-r2"), ("kf2", "c1")):
        for k, (n, gr, t, sd) in paired(R[a], R[b]).items():
            print(f"| {a} vs {b} | {k} | {n} | {gr:.2f} | {t:.2f} | {sd:.2f} |")

    print("\n## T3: known differences, paired per level (B/A), common horizon\n")
    KNOWN = [("kf2", "dfz-control", None, "hidden 8.1x"), ("duck36-sub", "q38-commit", None, "hidden 2.2x"),
             ("kf2", "p1", None, "hidden 1.04x"), ("kf2", "p2", None, "same system as p1"),
             ("kf2", "g1", None, "null graft"), ("kf2", "r1", None, "null repair"),
             ("dfz-control", "dfz-s1", None, "prompt change, inconclusive"),
             ("dfz-control", "lean-cap4", 3600, "Tycho lean arm, 10 games"), ("dfz-control", "lean-prune", 3600, "Tycho lean arm, 10 games")]
    print("| A | B | horizon | known | levels A->B (shared games) | hazard B/A | tok/level geo (t) | act/level geo (t) | n |"); print("|---|---|---|---|---|---|---|---|---|")
    for a, b, H, note in KNOWN:
        sg = set(R[a]) & set(R[b])
        GA = {g: R[a][g] for g in sg}; GB = {g: R[b][g] for g in sg}
        ma, mb = run_metrics(GA, H), run_metrics(GB, H)
        p = paired(GA, GB, H)
        print(f"| {a} | {b} | {H or 'full'} | {note} | {ma['levels']}->{mb['levels']} | {mb['hazard']/ma['hazard']:.2f} | "
              f"{p['tok'][1]:.2f} ({p['tok'][2]:.1f}) | {p['act'][1]:.2f} ({p['act'][2]:.1f}) | {p['tok'][0]} |")

    print("\n## T4: early read. Run totals at 30/60 min vs final, over runs with a >=2 h horizon\n")
    long = [n for n in R if min(g["wall"] for g in R[n].values()) > 5000]
    for H in (1800, 3600):
        e = [run_metrics(R[n], H)["levels"] for n in long]; f = [M[n]["levels"] for n in long]
        print(f"- {H//60} min: spearman {spearman(e, f):.2f} over {len(long)} runs; early share of final levels median "
              f"{statistics.median(x / y for x, y in zip(e, f)):.2f}")
