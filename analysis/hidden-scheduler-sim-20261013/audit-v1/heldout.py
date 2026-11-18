"""Five-fold game-held-out scheduler audit. CPU only; parent evidence is read-only.

Run python -B heldout.py --reps 50 --mixed-reps 100 --workers 8.
Clock is time since simultaneous game admission; queue waiting consumes horizon.
Every slot has the same fixed effective generation rate. No policy gets extra tokens.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import math
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[1]
sys.path.insert(0, str(PARENT))
import sim

SCENARIOS = {
    "base": {"tail_scale": 1.0, "cost_scale": 1.0},
    "tail_half": {"tail_scale": 0.5, "cost_scale": 1.0},
    "tail_zero": {"tail_scale": 0.0, "cost_scale": 1.0},
    "harder_x2": {"tail_scale": 1.0, "cost_scale": 2.0},
    "empirical_only": {"tail_scale": 0.0, "cost_scale": 1.0, "empirical_only": True},
}
SLOTS = 14
HORIZON = min(532 * 60, 9 * 3600 - 560)
RATE = 53.2
CHECKPOINTS = (15, 30, 60, 90, 121)


def folds_for(games):
    """Fixed, outcome-independent folds. Every identity is held out once."""
    ordered = sorted(games, key=lambda g: hashlib.sha256(("scheduler-audit-v1:" + g).encode()).hexdigest())
    assert len(ordered) == 25
    return [ordered[i:i + 5] for i in range(0, 25, 5)]


def select_data(data, games=None, runs=None, scale=None):
    games = set(games) if games is not None else None
    runs = set(runs) if runs is not None else None
    out = copy.deepcopy(data)
    out["runs"] = {r: m for r, m in out["runs"].items() if runs is None or r in runs}
    out["games"] = {r: {g: m for g, m in gs.items() if games is None or g in games}
                    for r, gs in out["games"].items() if r in out["runs"]}
    out["attempts"] = [dict(a, tokens=a["tokens"] * (scale or {}).get(a["run"], 1.0))
                       for a in out["attempts"] if a["run"] in out["runs"] and (games is None or a["game"] in games)]
    return out


def training_scale(data, train_games):
    """Secondary pooled-cohort scaling, estimated only from training-game solves."""
    pairs = {}
    for a in data["attempts"]:
        if a["game"] not in train_games or not a["solved"] or a["tokens"] <= 0:
            continue
        pairs.setdefault((a["game"], a["level"]), {}).setdefault(a["slots"], []).append(a["tokens"])
    ratios = [statistics.median(v[14]) / statistics.median(v[10])
              for v in pairs.values() if 10 in v and 14 in v]
    assert ratios
    return math.exp(statistics.mean(math.log(r) for r in ratios)), len(ratios)


def tail_lambda(data, start=80_000):
    att = data["attempts"]
    exposure = sum(max(0.0, a["tokens"] - start) for a in att)
    events = sum(a["solved"] and a["tokens"] > start for a in att)
    return events / exposure if exposure else 0.0


class TrainingHolds:
    def __init__(self, training_games, path=PARENT / "holds.json"):
        raw = json.loads(path.read_text())
        self.first = [v[0] for gs in raw.values() for g, v in gs.items()
                      if g in training_games and len(v) > 1]
        self.later = [x for gs in raw.values() for g, v in gs.items()
                      if g in training_games for x in v[1:-1] if x > 0]
        assert self.first and self.later and all(x > 0 for x in self.first)

    def draw(self, gid, hold_num, instance_num, seed):
        # A per-instance tape gives both policies identical hold draws even when
        # their scheduling order differs. Unused tape entries need not be generated.
        rng = random.Random(f"{seed}:{instance_num}:{gid}:{hold_num}")
        return rng.choice(self.first if hold_num == 0 else self.later)


class TaggedWorld(sim.CostModel):
    """Reuse the latent-world model, recording provenance of every level cost.

    Held-out outcomes may describe the evaluation world, but do not fit the policy.
    The extrapolation hazard is fit ONLY on training games, then sensitivity-scaled.
    """
    def __init__(self, data, train, **scenario):
        self.empirical_only = scenario.pop("empirical_only", False)
        super().__init__(data, **scenario)
        self.lam = tail_lambda(train) * scenario.get("tail_scale", 1.0)
        self.last_extrapolated = False

    def _extrapolate(self, start, g, rng):
        self.last_extrapolated = True
        return super()._extrapolate(start, g, rng)

    def draw(self, g, rng, run=None):
        n, base = self.meta[g]
        run = run or rng.choice(sorted(self.real[g]))
        rows = sorted(self.real[g][run], key=lambda a: a["level"])
        cost, acts, source = [], [], []
        censor = 0.0
        for a in rows:
            if a["solved"]:
                cost.append(a["tokens"])
                acts.append(a["actions"])
                source.append("observed_trajectory")
            else:
                censor = a["tokens"]
                break
        while len(cost) < n:
            if self.empirical_only:
                c, a, tag = math.inf, 0, "unresolved"
            else:
                self.last_extrapolated = False
                k = len(cost)
                fallback = not self.pool.get((g, k))
                c, a = self._draw_level(g, k, censor, rng)
                tag = ("extrapolated" if math.isfinite(c) else "unresolved") if self.last_extrapolated else (
                    "empirical_other_game" if fallback else "empirical_same_game")
            cost.append(c)
            acts.append(a)
            source.append(tag)
            censor = 0.0
        inst = sim.GameInst(g, n, base, [c * self.cost_scale for c in cost], acts)
        inst.sources = source
        inst.donor_run = run
        return inst


def attributed_score(g):
    """Allocate the exact game score across provenance groups, including global cap.

    The game's cap rescales contributions proportionally; groups sum to exact score.
    Downstream dependence is separate: even an empirical level behind an invented
    completion could not have been reached without that invention.
    """
    total_w = g.inst.n * (g.inst.n + 1) / 2
    pieces = []
    dependent = False
    for k in range(g.k):
        dependent |= g.inst.sources[k] == "extrapolated"
        ls = min(115.0, 100 * (g.inst.base[k] / g.apl[k]) ** 2)
        pieces.append(((k + 1) * ls / total_w, g.inst.sources[k], dependent))
    raw = sum(v for v, _, _ in pieces)
    score = sim.game_score(g.inst.n, g.inst.base, g.k, g.apl)
    factor = score / raw if raw else 0.0
    out = {s: 0.0 for s in ("observed_trajectory", "empirical_same_game", "empirical_other_game", "extrapolated")}
    for value, source, _ in pieces:
        out[source] += factor * value
    return score, out, sum(factor * v for v, _, dep in pieces if dep)


def simulate(insts, policy, *, horizon=HORIZON, slots=SLOTS, rate=RATE, holds, seed=0, record=False):
    """Fixed-rate event simulation. Games open together; initial waiting is charged.

    This fixes the parent's extra-rate startup artifact: its first few admitted
    holds ran faster than the final concurrency and retained that speed after the
    other slots filled. Here no policy receives extra token capacity.
    """
    games = [sim.GState(inst=x, idx=i, apl=[0] * x.n) for i, x in enumerate(insts)]
    waiting = list(games)
    for i, g in enumerate(games):
        g.arrival = i
    active = []
    seq = len(games)
    free = slots
    first_admission = [None] * len(games)

    def start(g, now):
        nonlocal seq
        q = holds.draw(g.inst.gid, g.holds, g.idx, seed)
        if not g.started:
            first_admission[g.idx] = now
        g.started = True
        g.holds += 1
        budget = min(q, (horizon - now) * rate)
        used = 0.0
        while used < budget and not g.won:
            need = g.inst.cost[g.k] - g.t_on
            if need <= budget - used:
                used += need
                g.apl[g.k] = g.inst.acts[g.k]
                g.solve_times.append(now + used / rate)
                g.k += 1
                g.t_on = 0.0
            else:
                g.t_on += budget - used
                used = budget
        if not g.won:
            g.apl[g.k] = g.a_on
        g.tokens += used
        seq += 1
        heapq.heappush(active, (now + max(used / rate, 1e-9), seq, g))

    def pump(now):
        nonlocal free
        ctx = sim.Ctx(now, horizon, rate)
        while free and waiting:
            g = max(waiting, key=lambda x: (sim.UNSTARTED_BASE - x.idx if not x.started else policy(x, ctx), -x.arrival))
            waiting.remove(g)
            start(g, now)
            free -= 1

    pump(0.0)
    while active:
        now, _, g = heapq.heappop(active)
        free += 1
        if now < horizon - 1e-8 and not g.won:
            seq += 1
            g.arrival = seq
            waiting.append(g)
        if now < horizon - 1e-8:
            pump(now)
    parts = [attributed_score(g) for g in games]
    out = {
        "score": statistics.mean(p[0] for p in parts),
        "levels": sum(g.k for g in games), "tokens": sum(g.tokens for g in games),
        "provenance_score": {s: statistics.mean(p[1][s] for p in parts) for s in parts[0][1]},
        "depends_on_extrapolation_score": statistics.mean(p[2] for p in parts),
        "observed_prefix_levels": sum(sum(s == "observed_trajectory" for s in g.inst.sources[:g.k]) for g in games),
        "extrapolated_levels": sum(sum(s == "extrapolated" for s in g.inst.sources[:g.k]) for g in games),
        "starved_50k": sum(g.tokens < 50_000 and not g.won for g in games),
        "zero_level_games": sum(g.k == 0 for g in games),
        "max_first_wait_s": max(t or 0.0 for t in first_admission),
        "unstarted_games": sum(t is None for t in first_admission),
    }
    if record:
        out["solve_times"] = sorted(t for g in games for t in g.solve_times)
    assert out["tokens"] <= slots * rate * horizon + 1e-5
    assert math.isclose(sum(out["provenance_score"].values()), out["score"], abs_tol=1e-8)
    return out


def fit_fold(data, folds, fold, cohort):
    test = set(folds[fold])
    train_games = set(g for fs in folds for g in fs) - test
    selected_runs = [r for r, m in data["runs"].items() if m["slots"] == 14] if cohort == "c14_only" else None
    scale_value, n_pairs = (1.0, 0) if cohort == "c14_only" else training_scale(data, train_games)
    scale = {r: scale_value for r, m in data["runs"].items() if m["slots"] == 10}
    train = select_data(data, train_games, selected_runs, scale)
    world_data = select_data(data, test, selected_runs, scale)
    hz = sim.HazardIndex(train)
    assert {a["game"] for a in train["attempts"]} == train_games
    assert not test.intersection(a["game"] for a in train["attempts"])
    return {"train": train, "world": world_data, "hazard": hz,
            "holds": TrainingHolds(train_games), "test": sorted(test),
            "scale": scale_value, "scale_pairs": n_pairs}


def summarize(rows):
    a = np.array([r["shipped"]["score"] for r in rows])
    b = np.array([r["hazard"]["score"] for r in rows])
    rel = b / a - 1
    out = {"n": len(rows), "shipped_score": float(a.mean()), "hazard_score": float(b.mean()),
           "relative_mean_gain": float(b.mean() / a.mean() - 1),
           "paired_gain_p10_p50_p90": [float(np.percentile(rel, p)) for p in (10, 50, 90)],
           "paired_win_rate": float((b > a).mean()),
           "max_paired_token_difference": max(abs(r["shipped"]["tokens"] - r["hazard"]["tokens"]) for r in rows)}
    for name in ("shipped", "hazard"):
        out[name] = {k: statistics.mean(r[name][k] for r in rows) for k in
                     ("levels", "tokens", "depends_on_extrapolation_score", "observed_prefix_levels",
                      "extrapolated_levels", "starved_50k", "zero_level_games", "max_first_wait_s", "unstarted_games")}
        out[name]["provenance_score"] = {k: statistics.mean(r[name]["provenance_score"][k] for r in rows)
                                         for k in rows[0][name]["provenance_score"]}
        out[name]["score_p10_p50_p90"] = [float(np.percentile(a if name == "shipped" else b, p)) for p in (10, 50, 90)]
    out["score_gain_without_extrapolation_dependency"] = (
        out["hazard_score"] - out["hazard"]["depends_on_extrapolation_score"] -
        out["shipped_score"] + out["shipped"]["depends_on_extrapolation_score"])
    return out


def job(args):
    cohort, scenario, fold, reps, pilot = args
    data = json.loads((PARENT / "attempts.json").read_text())
    folds = folds_for(next(iter(data["games"].values())).keys())
    fits = {f: fit_fold(data, folds, f, cohort) for f in (range(5) if fold == "mixed" else [fold])}
    worlds = {f: TaggedWorld(v["world"], v["train"], **SCENARIOS[scenario]) for f, v in fits.items()}
    game_fold = {g: f for f, gs in enumerate(folds) for g in gs}
    test_games = sorted(game_fold) if fold == "mixed" else fits[fold]["test"]

    def hazard(g, ctx):
        return fits[game_fold[g.inst.gid]]["hazard"].policy(g, ctx)

    class HoldsByGame:
        def draw(self, gid, hold_num, instance_num, seed):
            return fits[game_fold[gid]]["holds"].draw(gid, hold_num, instance_num, seed)

    results = []
    for rep in range(reps):
        rng = random.Random(f"audit-v1:{fold}:{rep}")
        insts = []
        for _ in range(110):
            gid = rng.choice(test_games)
            insts.append(worlds[game_fold[gid]].draw(gid, rng))
        seed = f"audit-v1:holds:{fold}:{rep}"
        r = {"rep": rep, "shipped": simulate(insts, sim.make_dfranzen(), holds=HoldsByGame(), seed=seed),
             "hazard": simulate(insts, hazard, holds=HoldsByGame(), seed=seed)}
        # At these costs neither arm wins every game. All capacity must be consumed.
        for arm in ("shipped", "hazard"):
            assert math.isclose(r[arm]["tokens"], SLOTS * RATE * HORIZON, abs_tol=1e-4), (cohort, scenario, fold, arm)
        results.append(r)
    result = {"cohort": cohort, "scenario": scenario, "fold": fold,
              "fit": {str(f): {"training_games": sorted({a['game'] for a in v['train']['attempts']}),
                                 "heldout_games": v["test"], "attempts": len(v["train"]["attempts"]),
                                 "tail_lambda": tail_lambda(v["train"]), "scale": v["scale"], "scale_pairs": v["scale_pairs"]}
                      for f, v in fits.items()}, "summary": summarize(results), "rows": results}
    prefix = "pilot-" if pilot else ""
    path = HERE / "runs" / f"{prefix}{cohort}-{scenario}-fold-{fold}.json"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(result, indent=2) + "\n")
    return {k: v for k, v in result.items() if k != "rows"}


def manifest(reps, mixed_reps, workers):
    data = json.loads((PARENT / "attempts.json").read_text())
    files = [PARENT / p for p in ("attempts.json", "holds.json", "sim.py", "validation.json", "policies.json")]
    return {"experiment": "scheduler-heldout-v1", "authority": "Pranav: do the test",
            "scope": "CPU only; no model calls, training, GPU launches, queue changes or submissions",
            "folds": folds_for(next(iter(data["games"].values())).keys()),
            "scenarios": SCENARIOS, "reps_per_fold": reps, "mixed_reps": mixed_reps, "workers": workers,
            "slots": SLOTS, "horizon_s": HORIZON, "slot_tok_s": RATE, "equal_token_capacity": SLOTS * RATE * HORIZON,
            "primary_cohort": "c14_only", "secondary_cohort": "all_runs_train_only_scaled",
            "candidate": "unchanged parent HazardIndex fit on 20 training game identities",
            "primary_contrast": "five-fold held-out evaluation; mixed set dispatches each game to its out-of-fold policy",
            "decision_rule": "Prepare live mechanism smoke only if primary mixed mean gain >=10%, positive in each fold, and nonnegative under half-tail and 2x-cost sensitivities. Empirical-only/zero-tail reversals require measuring tail rather than promotion.",
            "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--mixed-reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    (HERE / "runs").mkdir(exist_ok=True)
    cohorts = ("c14_only", "all_runs_train_only_scaled")
    jobs = [(cohort, sc, fold, args.mixed_reps if fold == "mixed" else args.reps, False)
            for cohort in cohorts for sc in SCENARIOS for fold in [0, 1, 2, 3, 4, "mixed"]]
    if args.pilot:
        jobs = [("c14_only", sc, "mixed", 3, True) for sc in ("base", "tail_zero")]
    target = HERE / ("PILOT.json" if args.pilot else "RESULT.json")
    if target.exists():
        raise SystemExit(f"Refusing to overwrite {target}")
    frozen = manifest(args.reps, args.mixed_reps, args.workers)
    (HERE / ("PILOT-MANIFEST.json" if args.pilot else "MANIFEST.json")).write_text(json.dumps(frozen, indent=2) + "\n")
    out = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(job, jobs):
            out.append(r)
            s = r["summary"]
            print(f"{r['cohort']:27s} {r['scenario']:14s} fold={r['fold']} n={s['n']} "
                  f"score={s['shipped_score']:.2f}->{s['hazard_score']:.2f} gain={s['relative_mean_gain']:+.2%} "
                  f"tokens_delta={s['max_paired_token_difference']:.5f}", flush=True)
    target.write_text(json.dumps({"manifest": frozen, "comparisons": out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
