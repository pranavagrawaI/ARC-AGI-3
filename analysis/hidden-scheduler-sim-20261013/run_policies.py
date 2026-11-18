"""Hidden-regime policy comparison under common random numbers.

Each bootstrap draws a hidden set of 110 games (with replacement from the 25 public games, each copy an
independent realization of that game's level costs) and plays it under every policy with the same
instances: 14 slots, horizon = min(532 min game cap, 9 h envelope - 560 s startup) = 31,840 s, the
shipped 14-stream per-slot rate. Total tokens are equal across policies by construction (same slots,
rate and horizon; slots only idle when every remaining game has won).

Usage: python -B run_policies.py [n_boot] [scenario ...]  -> policies.json
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import sim

HERE = Path(__file__).resolve().parent
N_HIDDEN = 110
SLOTS = 14
HORIZON = min(532 * 60, 32400 - 560)
R14 = 53.2           # mean effective per-slot tok/s of the three 14-stream runs (53.5, 52.1, 54.0)
C14_OVER_C10 = 1.169  # matched-level cost ratio; 10-stream donors rescaled to the shipped config

SCENARIOS = {
    "base": {},
    "hidden_harder_x2": dict(cost_scale=2.0),
    "tail_half": dict(tail_scale=0.5),
    "tail_zero": dict(tail_mode="none"),
    "tail_x1.5": dict(tail_scale=1.5),
    "c14_donors_only": dict(c14_only=True),
}


def policies(data):
    # the index is always fitted on the logged public data, whatever world the scenario simulates
    hz = sim.HazardIndex(data)
    p = {"a_dfranzen_shipped": sim.make_dfranzen(), "b_fifo": sim.fifo}
    for T in (40_000, 60_000, 80_000, 120_000):
        p[f"c_park_{T // 1000}k"] = sim.make_dfranzen(park_tokens=T)
    p["d_hazard_index"] = hz.policy
    for s in (31_000, 93_000, 124_000, 248_000, 1e12):
        p[f"e_token_scale_{'inf' if s > 1e9 else str(int(s // 1000)) + 'k'}"] = sim.make_dfranzen(token_scale=s)
    p["f_oracle"] = sim.oracle
    return p


def one_boot(args):
    b, scenario = args
    data = json.loads((HERE / "attempts.json").read_text())
    kw = dict(SCENARIOS[scenario])
    c14_only = kw.pop("c14_only", False)
    runs = [r for r, m in data["runs"].items() if m["slots"] == 14] if c14_only else None
    scale = {r: C14_OVER_C10 for r, m in data["runs"].items() if m["slots"] == 10}
    cm = sim.CostModel(data, runs=runs, run_scale=scale, **kw)
    holds = sim.Holds()
    rng = random.Random(7919 * b + 17)
    insts = [cm.draw(rng.choice(cm.games), rng) for _ in range(N_HIDDEN)]
    out = {}
    for name, pol in policies(data).items():
        r = sim.simulate(insts, pol, slots=SLOTS, horizon=HORIZON, r14=R14, holds=holds,
                         rng=random.Random(b))
        tok = np.array(r["tokens"])
        unwon = ~np.array(r["won"])
        out[name] = dict(score=r["mean_score"], levels=r["levels"], tokens=float(tok.sum()),
                         starved_50k=int(((tok < 50_000) & unwon).sum()), starved_100k=int(((tok < 100_000) & unwon).sum()),
                         zero_level_games=int(sum(1 for x in r["levels_by_game"] if x == 0)),
                         max_game_tokens=float(tok.max()))
    return out


def summarize(boots):
    names = list(boots[0])
    base = np.array([b["a_dfranzen_shipped"]["score"] for b in boots])
    rows = {}
    for n in names:
        s = np.array([b[n]["score"] for b in boots])
        rel = s / base - 1
        rows[n] = dict(mean=float(s.mean()), p10=float(np.percentile(s, 10)), p50=float(np.percentile(s, 50)),
                       p90=float(np.percentile(s, 90)),
                       rel_vs_a_mean=float(s.mean() / base.mean() - 1),
                       rel_vs_a_p10=float(np.percentile(rel, 10)), rel_vs_a_p90=float(np.percentile(rel, 90)),
                       paired_win_rate=float((s > base).mean()),
                       levels=float(np.mean([b[n]["levels"] for b in boots])),
                       tokens_M=float(np.mean([b[n]["tokens"] for b in boots]) / 1e6),
                       starved_50k=float(np.mean([b[n]["starved_50k"] for b in boots])),
                       starved_100k=float(np.mean([b[n]["starved_100k"] for b in boots])),
                       zero_level_games=float(np.mean([b[n]["zero_level_games"] for b in boots])),
                       max_game_tokens_k=float(np.mean([b[n]["max_game_tokens"] for b in boots]) / 1e3))
    return rows


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    scen = sys.argv[2:] or list(SCENARIOS)
    path = HERE / "policies.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    result["config"] = dict(n_hidden=N_HIDDEN, slots=SLOTS, horizon_s=HORIZON, r14=R14, c14_over_c10=C14_OVER_C10)
    with ProcessPoolExecutor(16) as ex:
        for sc in scen:
            nb = n_boot if sc == "base" else max(60, n_boot // 2)
            boots = list(ex.map(one_boot, [(b, sc) for b in range(nb)]))
            rows = summarize(boots)
            result[sc] = dict(n_boot=nb, policies=rows)
            print(f"\n== {sc} (n_boot={nb})")
            for n, r in rows.items():
                print(f"{n:24s} mean {r['mean']:6.2f} p10 {r['p10']:6.2f} p50 {r['p50']:6.2f} p90 {r['p90']:6.2f}"
                      f"  vs a {100 * r['rel_vs_a_mean']:+6.1f}% [{100 * r['rel_vs_a_p10']:+.1f},{100 * r['rel_vs_a_p90']:+.1f}]"
                      f" win {r['paired_win_rate']:.2f} lv {r['levels']:6.1f} tok {r['tokens_M']:.2f}M"
                      f" starved<50k {r['starved_50k']:.1f} <100k {r['starved_100k']:.1f} zero {r['zero_level_games']:.1f}"
                      f" maxg {r['max_game_tokens_k']:.0f}k", flush=True)
            path.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
