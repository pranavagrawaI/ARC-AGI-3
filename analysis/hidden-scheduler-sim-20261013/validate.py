"""Public-regime validation: leave-one-run-out.

For each logged run R, simulate its own regime (25 public games, its gate slot count, its cap, its
effective per-slot rate) with the shipped gate, drawing every game's level costs from the OTHER runs only.
Compare simulated levels, mean score, levels-over-time and per-game token share with what R observed.

Usage: python -B validate.py [n_reps]  -> validation.json
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from pathlib import Path

import sim

HERE = Path(__file__).resolve().parent
CHECKPOINTS = [15, 30, 60, 90, 121]  # minutes
C14_OVER_C10 = 1.169  # geo-mean matched (game, level) median solved cost, 127 pairs (FINDINGS)


def observed(data, run):
    att = [a for a in data["attempts"] if a["run"] == run]
    times = sorted(a["wall_end"] for a in att if a["solved"])
    games = data["games"][run]
    return dict(levels=sum(g["done"] for g in games.values()),
                mean_score=sum(g["score"] for g in games.values()) / len(games),
                curve=[sum(1 for t in times if t <= m * 60) for m in CHECKPOINTS],
                tokens={g: m["tokens"] for g, m in games.items()},
                total_tokens=sum(m["tokens"] for m in games.values()))


def run_one(data, holds, run, reps, policy=None, **cm_kw):
    meta = data["runs"][run]
    slots, cap = meta["slots"], meta["cap"]
    obs = observed(data, run)
    per_slot = obs["total_tokens"] / slots / cap
    r14 = per_slot / math.sqrt(14.0 / slots)
    others = [r for r in data["runs"] if r != run]
    # donors from the other config are rescaled to this run's config (matched-level cost ratio)
    scale = {r: (C14_OVER_C10 if slots == 14 else 1 / C14_OVER_C10)
             for r in others if data["runs"][r]["slots"] != slots}
    cm = sim.CostModel(data, runs=others, run_scale=scale, **cm_kw)
    policy = policy or sim.make_dfranzen()
    res = []
    for i in range(reps):
        rng = random.Random(1000 + i)
        insts = [cm.draw(g, rng) for g in cm.games]
        res.append(sim.simulate(insts, policy, slots=slots, horizon=cap, r14=r14, holds=holds, rng=rng, record=True))
    curve = [st.mean(sum(1 for t in r["solve_times"] if t <= m * 60) for r in res) for m in CHECKPOINTS]
    tok = {g: st.mean(r["tokens"][i] for r in res) for i, g in enumerate(cm.games)}
    gs = sorted(tok)
    o = [obs["tokens"][g] for g in gs]
    s = [tok[g] for g in gs]
    corr = st.correlation(o, s)
    lv_obs = {g: data["games"][run][g]["done"] for g in gs}
    lv_sim = {g: st.mean(r["levels_by_game"][i] for r in res) for i, g in enumerate(cm.games)}
    return dict(run=run, slots=slots, cap=cap, r14=r14,
                obs_levels=obs["levels"], sim_levels=st.mean(r["levels"] for r in res),
                sim_levels_sd=st.pstdev([r["levels"] for r in res]),
                obs_score=obs["mean_score"], sim_score=st.mean(r["mean_score"] for r in res),
                sim_score_sd=st.pstdev([r["mean_score"] for r in res]),
                obs_curve=obs["curve"], sim_curve=curve,
                obs_tokens=obs["total_tokens"], sim_tokens=st.mean(sum(r["tokens"]) for r in res),
                token_share_corr=corr,
                per_game_levels_corr=st.correlation([lv_obs[g] for g in gs], [lv_sim[g] for g in gs]))


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    data = json.loads((HERE / "attempts.json").read_text())
    holds = sim.Holds()
    rows = [run_one(data, holds, r, reps) for r in data["runs"] if data["runs"][r]["cap"] == 7254]
    for r in rows:
        print(f"{r['run']:14s} s{r['slots']} levels obs {r['obs_levels']:4d} sim {r['sim_levels']:6.1f}±{r['sim_levels_sd']:.1f}"
              f"  score obs {r['obs_score']:5.1f} sim {r['sim_score']:5.1f}  curve obs {r['obs_curve']} sim {[round(x) for x in r['sim_curve']]}"
              f"  tokcorr {r['token_share_corr']:.2f} lvcorr {r['per_game_levels_corr']:.2f}")
    err_l = [r["sim_levels"] / r["obs_levels"] - 1 for r in rows]
    err_s = [r["sim_score"] / r["obs_score"] - 1 for r in rows]
    summ = dict(level_bias=st.mean(err_l), level_mae=st.mean(abs(e) for e in err_l),
                score_bias=st.mean(err_s), score_mae=st.mean(abs(e) for e in err_s))
    print(summ)
    (HERE / "validation.json").write_text(json.dumps(dict(rows=rows, summary=summ), indent=1))


if __name__ == "__main__":
    main()
