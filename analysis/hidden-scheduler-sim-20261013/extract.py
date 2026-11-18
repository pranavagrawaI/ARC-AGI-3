"""Extract per-level attempts (tokens, actions, solved/censored) from logged public-25 runs.

Source of truth is each run's benchmark.json (taaf GameRun.to_json_dict): `history[i].generated_tokens`
is the generation spent since the previous action, `actions_per_level[k]` the actions charged to level k
(levels < levels_completed are solved; index levels_completed is the attempt in progress), and
`solver_note` "tokens=N" the game's total generation including tokens after the last action.

Also extracts the gate's hold quantum (generated tokens between context trims, the only points where
the dfranzen gate hands a slot over) from per-game request logs of the 14-stream runs.

Usage: python -B extract.py  -> attempts.json, holds.json
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# label -> (output dir, gate slots, per-game cap seconds). Slots/caps read from each run's kernel log
# ("priority gate active: N concurrent streams", "max_runtime_s_per_game=..."). All runs: Flash-Next
# dfranzen-family harness, same priority env (tail fade observed at ~97 min of 121).
RUNS = {
    "base_c1": ("analysis/m2-winners-20261001/runs/out-control", 10, 7254),
    "base_c1_r2": ("analysis/m2-winners-20261001/runs/out-control-r2", 10, 6000),
    "s1_econ": ("analysis/m2-winners-20261001/runs/out-s1", 10, 7254),
    "m2ev_ctl": ("solvers/m2_evidence/experiment/pair-1/control/output", 10, 7254),
    "m2ev_trt": ("solvers/m2_evidence/experiment/pair-1/treatment/output", 10, 7254),
    "B1_r1": ("solvers/m2_b1/experiment/pair-1/b1-r1/output", 10, 7254),
    "B1_r2": ("solvers/m2_b1/experiment/pair-1/b1-r2/output", 10, 7254),
    "c14_E2": ("analysis/model-serving-20261011/wave-1/e2-c14/output", 14, 7254),
    "B1_c14": ("analysis/model-serving-20261011/wave-2/b1-c14/output", 14, 7254),
    "c14_prune_ctl": ("analysis/pruning-transfer-20261011/control/output", 14, 7254),
}
HOLD_RUNS = ["c14_E2", "B1_c14", "c14_prune_ctl"]
NOTE = re.compile(r"tokens=(\d+)")


def attempts_for_run(label: str) -> tuple[list[dict], dict]:
    out, slots, cap = RUNS[label]
    bench = json.loads((ROOT / out / "benchmark.json").read_text())
    rows, games = [], {}
    for g in bench["game_runs"]:
        gid = g["game_id"][:4]
        hist = g["history"]
        apl = list(g["actions_per_level"])
        done = int(g["levels_completed"])
        n = int(g["number_of_levels"])
        assert sum(apl) == len(hist), (label, gid)
        m = NOTE.search(g.get("solver_note") or "")
        total_tok = int(m.group(1)) if m else sum(h["generated_tokens"] for h in hist)
        tail_tok = max(0, total_tok - sum(h["generated_tokens"] for h in hist))
        games[gid] = dict(n_levels=n, base=list(g["base_actions_per_level"]), done=done,
                          state=g["state"], score=float(g["final_score"] or 0.0), tokens=total_tok,
                          actions=len(hist), wall=float(g["final_wallclock_seconds"] or 0.0))
        i, t_prev_wall = 0, 0.0
        for k in range(min(n, done + 1)):
            na = apl[k] if k < len(apl) else 0
            seg = hist[i:i + na]
            i += na
            tok = sum(h["generated_tokens"] for h in seg)
            solved = k < done
            if not solved:
                tok += tail_tok
            w_end = seg[-1]["wallclock_seconds"] if (seg and solved) else float(g["final_wallclock_seconds"] or 0)
            rows.append(dict(run=label, slots=slots, cap=cap, game=gid, level=k, n_levels=n,
                             base=g["base_actions_per_level"][k], tokens=tok, actions=na, solved=solved,
                             wall_start=t_prev_wall, wall_end=w_end))
            t_prev_wall = w_end
    return rows, games


def holds_for_game(path: str) -> list[int]:
    """Generated tokens between trims (message count drops), first entry = before the first trim."""
    prev, gen, holds = None, 0, []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("event") == "request":
                n = len(d["messages"])
                if prev is not None and n < prev:
                    holds.append(gen)
                    gen = 0
                prev = n
            elif d.get("event") == "response" and isinstance(d.get("usage"), dict):
                gen += int(d["usage"].get("completion_tokens") or 0)
    return holds + [-gen]  # negative = censored tail hold (game ended mid-hold)


def main() -> None:
    allrows, allgames = [], {}
    for label in RUNS:
        rows, games = attempts_for_run(label)
        allrows += rows
        allgames[label] = games
        print(label, sum(g["done"] for g in games.values()), "levels", file=sys.stderr)
    (HERE / "attempts.json").write_text(json.dumps({"runs": {k: dict(dir=v[0], slots=v[1], cap=v[2]) for k, v in RUNS.items()},
                                                   "games": allgames, "attempts": allrows}))
    if "--no-holds" in sys.argv:
        return
    jobs = []
    for label in HOLD_RUNS:
        for p in sorted((ROOT / RUNS[label][0]).glob("*_p0_requests.jsonl")):
            jobs.append((label, p.name[:4], str(p)))
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(holds_for_game, [j[2] for j in jobs]))
    holds = {}
    for (label, gid, _), h in zip(jobs, res):
        holds.setdefault(label, {})[gid] = h
    (HERE / "holds.json").write_text(json.dumps(holds))


if __name__ == "__main__":
    main()
