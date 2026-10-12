"""Offline bound on a "re-draw a stuck level" policy, from five dfranzen-family runs (no GPU).

  python3 -B redraw_bound.py

Policy R(c): once an attempt has spent c generated tokens on a level without solving it, discard the
attempt's reasoning on that level and start a fresh draw with the attempt's remaining level budget.
The fresh draw is modelled as a sibling run's attempt at the same (game, level), which is also an
independent draw that carries context from earlier levels.

Accounting covers every attempt still open at c, not just the stuck ones:
  - a stuck attempt (never solved, burned T) gains P(sibling solves within T - c);
  - an attempt that solved after c, at S tokens, loses 1 - P(sibling solves within B - c), where
    B = max(S, G) and G is the run's median burn on stuck final levels (the scheduler's give-up point).
Siblings that never solved count as failures at any budget. Siblings that never reached the level
give no information. Follow-on levels unlocked by a gain are not counted, which understates gains;
assuming a fresh draw keeps earlier-level context overstates them.
"""
import json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "c1": "analysis/m2-winners-20261001/runs/out-control",
    "c2": "analysis/m2-winners-20261001/runs/out-control-r2",
    "s1": "analysis/m2-winners-20261001/runs/out-s1",
    "pc": "solvers/m2_evidence/experiment/pair-1/control/output",
    "pt": "solvers/m2_evidence/experiment/pair-1/treatment/output",
}


def attempts(path):
    """(game, level) -> (solved, tokens on level) for every level the run entered."""
    b = json.loads((ROOT / path / "benchmark.json").read_text())
    out = {}
    for r in b["game_runs"]:
        h, apl, done = r["history"], r["actions_per_level"], int(r["levels_completed"] or 0)
        k = 0
        for L, n in enumerate(apl):
            seg = h[k:k + n]
            if L < done:
                out[(r["game_id"][:4], L)] = (True, sum(x["generated_tokens"] for x in seg))
            elif L == done:
                tail = int(r.get("final_generated_tokens") or 0)
                out[(r["game_id"][:4], L)] = (False, sum(x["generated_tokens"] for x in seg) + tail)
                break
            k += n
    return out


def p_solve(sibs, budget):
    """Share of sibling attempts that solved within budget tokens."""
    return sum(1 for s, t in sibs if s and t <= budget) / len(sibs) if sibs else None


def main():
    A = {r: attempts(p) for r, p in RUNS.items()}
    G = {r: statistics.median([t for s, t in a.values() if not s] or [0]) for r, a in A.items()}
    print("median stuck burn per run (tokens):", {r: int(g) for r, g in G.items()})
    print(f"{'c (tokens)':>10} {'open':>5} {'stuck':>5} {'gain':>6} {'loss':>6} {'net/run':>8}  (levels per 25-game pass)")
    for c in (10_000, 20_000, 30_000, 40_000, 60_000, 80_000):
        gain = loss = 0.0; n_open = n_stuck = 0
        for r, a in A.items():
            for key, (solved, t) in a.items():
                if t <= c:
                    continue
                sibs = [A[q][key] for q in A if q != r and key in A[q]]
                if not sibs:
                    continue
                n_open += 1
                if solved:
                    loss += 1 - p_solve(sibs, max(t, G[r]) - c)
                else:
                    n_stuck += 1
                    gain += p_solve(sibs, t - c)
        print(f"{c:>10} {n_open:>5} {n_stuck:>5} {gain:>6.1f} {loss:>6.1f} {(gain - loss) / len(A):>8.2f}")


if __name__ == "__main__":
    main()
