"""Kaplan-Meier survival of 'level still unsolved' vs generated tokens on the level.

Reads attempts.json (extract.py). Writes hazard.json: pooled / by level index / by game-difficulty
KM curves, interval solve probabilities per 20K tokens, the late-tail hazard, actions-at-solve
ratios, and effective per-slot token rates.

Usage: python -B hazard.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = 20_000
GRID = list(range(0, 400_001, STEP))


def km(samples: list[tuple[float, bool]]) -> list[tuple[float, float, int]]:
    """[(t, S(t) just after t, at_risk)] at each event time."""
    samples = sorted(samples)
    n = len(samples)
    s, out, i = 1.0, [], 0
    while i < n:
        t = samples[i][0]
        d = c = 0
        while i < n and samples[i][0] == t:
            d += samples[i][1]
            c += not samples[i][1]
            i += 1
        at_risk = n - (i - d - c)
        if d:
            s *= 1 - d / at_risk
            out.append((t, s, at_risk))
    return out


def surv_at(curve, t: float) -> float:
    s = 1.0
    for tt, ss, _ in curve:
        if tt > t:
            break
        s = ss
    return s


def interval_table(samples):
    """P(solve within next STEP | unsolved at t), with number at risk."""
    curve = km(samples)
    rows = []
    for t in GRID[:-1]:
        risk = sum(1 for x, _ in samples if x > t)
        if risk < 10:
            break
        s0, s1 = surv_at(curve, t), surv_at(curve, t + STEP)
        rows.append(dict(t=t, at_risk=risk, p_next=round(1 - s1 / s0, 3) if s0 > 0 else None))
    return rows


def tail_rate(samples, start: float) -> dict:
    """Constant-hazard MLE (events per token) for attempts surviving past `start`."""
    exposure = sum(max(0.0, t - start) for t, _ in samples if t > start)
    events = sum(1 for t, e in samples if t > start and e)
    lam = events / exposure if exposure else 0.0
    return dict(start=start, events=events, exposure_tokens=exposure,
                per_token=lam, p_per_20k=1 - math.exp(-lam * STEP) if lam else 0.0)


def main() -> None:
    data = json.loads((HERE / "attempts.json").read_text())
    att = data["attempts"]
    S = [(a["tokens"], a["solved"]) for a in att]
    out = {"n_attempts": len(att), "n_solved": sum(a["solved"] for a in att)}
    out["pooled"] = interval_table(S)
    out["pooled_median_tokens"] = next((t for t, s, _ in km(S) if s <= 0.5), None)
    by_lvl = defaultdict(list)
    for a in att:
        by_lvl[min(a["level"], 4)].append((a["tokens"], a["solved"]))
    out["by_level_index"] = {str(k): dict(n=len(v), table=interval_table(v)[:8],
                                          median=next((t for t, s, _ in km(v) if s <= 0.5), None))
                             for k, v in sorted(by_lvl.items())}
    # game heterogeneity: per game, pooled over levels and runs
    by_game = defaultdict(list)
    for a in att:
        by_game[a["game"]].append((a["tokens"], a["solved"]))
    gstats = {}
    for g, v in by_game.items():
        c = km(v)
        gstats[g] = dict(n=len(v), solved=sum(e for _, e in v), p_by_60k=round(1 - surv_at(c, 60_000), 3),
                         p_by_150k=round(1 - surv_at(c, 150_000), 3),
                         tail_after_80k=tail_rate(v, 80_000)["p_per_20k"])
    out["by_game"] = gstats
    # difficulty terciles by p_by_60k
    order = sorted(gstats, key=lambda g: gstats[g]["p_by_60k"])
    terc = {g: i * 3 // len(order) for i, g in enumerate(order)}
    out["by_game_tercile"] = {}
    for t in range(3):
        v = [x for g in order if terc[g] == t for x in by_game[g]]
        out["by_game_tercile"][str(t)] = dict(games=[g for g in order if terc[g] == t],
                                              table=interval_table(v)[:10], tail_80k=tail_rate(v, 80_000))
    out["tail"] = {str(s): tail_rate(S, s) for s in (60_000, 80_000, 120_000, 160_000)}
    # actions at solve relative to the human baseline, and scored value
    ratios = sorted(a["actions"] / a["base"] for a in att if a["solved"] and a["base"])
    out["actions_ratio_at_solve"] = dict(n=len(ratios), p10=ratios[len(ratios) // 10],
                                         median=ratios[len(ratios) // 2], p90=ratios[9 * len(ratios) // 10])
    # effective per-slot generation rate per run (all games' tokens / slots / cap)
    rates = {}
    for r, meta in data["runs"].items():
        tok = sum(g["tokens"] for g in data["games"][r].values())
        rates[r] = dict(tokens=tok, slots=meta["slots"], cap=meta["cap"],
                        per_slot_tok_s=tok / meta["slots"] / meta["cap"],
                        levels=sum(g["done"] for g in data["games"][r].values()),
                        mean_score=sum(g["score"] for g in data["games"][r].values()) / 25)
    out["rates"] = rates
    (HERE / "hazard.json").write_text(json.dumps(out, indent=1))
    print("pooled P(solve next 20K | unsolved):", [(r["t"] // 1000, r["p_next"], r["at_risk"]) for r in out["pooled"]])
    for k, v in out["by_level_index"].items():
        print("level", k, v["n"], "median", v["median"], [(r["t"] // 1000, r["p_next"]) for r in v["table"]])
    for t, v in out["by_game_tercile"].items():
        print("tercile", t, v["games"], [(r["t"] // 1000, r["p_next"]) for r in v["table"]], "tail80", round(v["tail_80k"]["p_per_20k"], 3))
    print("tail", {k: round(v["p_per_20k"], 3) for k, v in out["tail"].items()})
    print("actions/base at solve", out["actions_ratio_at_solve"])
    for r, v in rates.items():
        print(r, v["slots"], round(v["per_slot_tok_s"], 1), v["levels"], round(v["mean_score"], 2))


if __name__ == "__main__":
    main()
