"""Read the likelihood-probe results against the pre-registered checks (PLAN.md).

  python3 analyze.py items.jsonl.xz results-a.jsonl [results-b.jsonl ...]

P_v(c) = softmax over the item's candidates of the summed log-prob of candidate c's code under variant v.
  V0 instrument: R0 rows scored in two sessions agree (numeric noise floor).
  V1 policy:     P(taken) vs uniform; top-1 agreement with the action the run actually took.
  V2 quality:    oracle items: P(opt set) vs the uniform share; does P(opt) predict that the run's action was optimal (AUC)?
                 decisive items: P(decisive) vs solving within 10 min (AUC), and vs minutes-in-level.
  V3 prompts:    paired over items, log P(target) under R0 (dfranzen prompt) vs KF0 (Duck/KF prompt) vs MIN0.
"""
import json, lzma, math, statistics, sys
from collections import defaultdict


def softmax(lp):
    m = max(lp); e = [math.exp(x - m) for x in lp]; s = sum(e)
    return [x / s for x in e]


def mass(c, lp, tk, same_kind):
    """(P of targets, uniform share). same_kind: normalise only over candidates of the targets' kind (click vs key),
    which removes the string-length advantage of keyboard codes in mixed games."""
    idx = list(range(len(c)))
    if same_kind:
        kinds = {c[k]["a"] == "ACTION6" for k in tk}
        if len(kinds) == 1: idx = [k for k in idx if (c[k]["a"] == "ACTION6") in kinds]
    P = softmax([lp[k] for k in idx]); pos = {k: n for n, k in enumerate(idx)}
    return sum(P[pos[k]] for k in tk if k in pos), len(tk) / len(idx)


def auc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def tstat(d):
    if len(d) < 3: return float("nan")
    sd = statistics.stdev(d)
    return statistics.mean(d) / (sd / math.sqrt(len(d))) if sd else float("inf")


def main(items_path, *res_paths):
    items = {j["id"]: j for j in map(json.loads, lzma.open(items_path, "rt"))}
    R = defaultdict(dict); dup = defaultdict(list)
    for path in res_paths:
        for j in map(json.loads, open(path)):
            if j["variant"] in R[j["id"]]: dup[(j["id"], j["variant"])].append(j["lp"])
            else: R[j["id"]][j["variant"]] = j["lp"]
    variants = sorted({v for r in R.values() for v in r})
    print(f"items {len(items)}, scored {len(R)}, variants {variants}")
    secs = [json.loads(l)["secs"] for p in res_paths for l in open(p)]
    print(f"secs per (item, variant): median {statistics.median(secs):.1f}, total {sum(secs) / 3600:.2f} h")

    # V0: same variant scored twice (session repeat)
    rep = [(R[i][v], lp2) for (i, v), lps in dup.items() for lp2 in lps if v in R[i]]
    if rep:
        diffs = [abs(a - b) for x, y in rep for a, b in zip(softmax(x), softmax(y))]
        top = sum(max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__) for x, y in rep) / len(rep)
        print(f"\nV0 repeat: {len(rep)} items; |dP| mean {statistics.mean(diffs):.4f} max {max(diffs):.4f}; top-1 identical {top:.0%}")

    for sk in (False, True):
        tag = "same-kind" if sk else "all candidates"
        print(f"\nV1 policy [{tag}]: probe mass on the action the run actually took")
        for v in variants:
            rows = []
            for i, r in R.items():
                if v not in r: continue
                c = items[i]["cands"]; t = [k for k, x in enumerate(c) if x["taken"]]
                if not t: continue
                pm, u = mass(c, r[v], t, sk)
                rows.append((pm, u, max(range(len(c)), key=r[v].__getitem__) in t))
            if rows:
                print(f"  {v:5s} n={len(rows):3d}  P(taken) {statistics.mean(x[0] for x in rows):.3f} vs uniform {statistics.mean(x[1] for x in rows):.3f}"
                      f"  log-lift {statistics.mean(math.log(max(x[0], 1e-12) / x[1]) for x in rows):+.2f}  top-1 agree (all) {statistics.mean(x[2] for x in rows):.0%}")
        print(f"V2 quality [{tag}]: oracle items (optimal = successor distance d0-1)")
        for v in variants:
            rows = []
            for i, r in R.items():
                it = items[i]
                if v not in r or not it["d0"]: continue
                c = it["cands"]; opt = [k for k, x in enumerate(c) if x["nd"] == it["d0"] - 1]
                if not opt: continue
                pm, u = mass(c, r[v], opt, sk)
                rows.append((pm, u, any(x["taken"] and x["nd"] == it["d0"] - 1 for x in c)))
            if rows:
                print(f"  {v:5s} n={len(rows):3d}  P(opt) {statistics.mean(x[0] for x in rows):.3f} vs uniform {statistics.mean(x[1] for x in rows):.3f}"
                      f"  log-lift {statistics.mean(math.log(max(x[0], 1e-12) / x[1]) for x in rows):+.2f}"
                      f"  AUC P(opt)->run took optimal {auc([x[0] for x in rows if x[2]], [x[0] for x in rows if not x[2]]):.2f}")
        print(f"   decisive items [{tag}]: P(decisive) vs solved within 10 min")
        for v in variants:
            rows = []
            for i, r in R.items():
                it = items[i]
                if v not in r or it["d0"] is not None: continue
                c = it["cands"]; dk = [k for k, x in enumerate(c) if x["decisive"]]
                if not dk: continue
                pm, u = mass(c, r[v], dk, sk)
                rows.append((pm, u, bool(it["solved"] and it["min_to_solve"] is not None and it["min_to_solve"] <= 10)))
            if rows:
                print(f"  {v:5s} n={len(rows):3d}  P(dec) {statistics.mean(x[0] for x in rows):.3f} vs uniform {statistics.mean(x[1] for x in rows):.3f}"
                      f"  log-lift {statistics.mean(math.log(max(x[0], 1e-12) / x[1]) for x in rows):+.2f}"
                      f"  AUC P(dec)->solved<=10min {auc([x[0] for x in rows if x[2]], [x[0] for x in rows if not x[2]]):.2f} (n soon {sum(x[2] for x in rows)})")

    print("\nV3 prompts: paired log P(target) differences across items (target = opt set, else decisive)")
    def target_lp(it, lp):
        c = it["cands"]; P = softmax(lp)
        tk = [k for k, x in enumerate(c) if it["d0"] and x["nd"] == it["d0"] - 1] or [k for k, x in enumerate(c) if x["decisive"]]
        return math.log(max(sum(P[k] for k in tk), 1e-12)) if tk else None
    for a, b in [("R0", "KF0"), ("R0", "MIN0"), ("KF0", "MIN0"), ("R1", "R0")]:
        d = []
        for i, r in R.items():
            if a in r and b in r:
                x, y = target_lp(items[i], r[a]), target_lp(items[i], r[b])
                if x is not None: d.append(x - y)
        if d:
            print(f"  {a} - {b}: n={len(d)} mean {statistics.mean(d):+.3f} nats  t={tstat(d):+.1f}  share>0 {sum(x > 0 for x in d) / len(d):.0%}")
    by_game = defaultdict(list)
    for i, r in R.items():
        if "R0" in r and "KF0" in r:
            x, y = target_lp(items[i], r["R0"]), target_lp(items[i], r["KF0"])
            if x is not None: by_game[items[i]["game"]].append(x - y)
    if by_game:
        g = [statistics.mean(v) for v in by_game.values()]
        print(f"  R0 - KF0 by game: {sum(x > 0 for x in g)}/{len(g)} games positive, game-level t={tstat(g):+.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
