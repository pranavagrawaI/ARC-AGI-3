"""Run the inherited TAAF benchmark inside isolated pinned Solver dependencies."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

from hybrid.solver import HybridSolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument("target")
    parser.add_argument("--soft-end", default="")
    args = parser.parse_args()
    with open(args.benchmark, "rb") as stream:
        benchmark = pickle.load(stream)
    with open(args.target, "rb") as stream:
        target = pickle.load(stream)
    mode = os.environ.get("HYBRID_RUN_MODE", "smoke")
    if mode not in {"smoke", "public25"}:
        raise ValueError("HYBRID_RUN_MODE must be smoke or public25")
    if target.actual_run_as_submission:
        raise RuntimeError("This candidate is for offline qualification, not competition submission")
    donor = benchmark.solver
    benchmark.solver = HybridSolver(
        concurrency=donor.concurrency,
        max_runtime_s_per_game=donor.max_runtime_s_per_game,
        max_actions_per_game=donor.max_actions_per_game,
        analyzer_timeout=donor.analyzer_timeout,
        worker_python=__import__("sys").executable,
        runtime_root=str(Path(__file__).resolve().parents[1]))
    if mode == "smoke":
        names = {"sb26", "ls20", "ar25"}
        benchmark.games = [game for game in benchmark.games
                           if str(game.env_name).split("-")[0] in names]
        if len(benchmark.games) != 3:
            raise RuntimeError("smoke requires exactly sb26, ls20, ar25")
        benchmark.game_weights = None
        benchmark.n_passes = 1
        benchmark.solver.concurrency = 3
        benchmark.solver.max_runtime_s_per_game = 600
        benchmark.solver.max_actions_per_game = 16
    benchmark.job_dir = Path(benchmark.job_dir)/"tycho-dfranzen"
    benchmark.job_dir.mkdir(parents=True, exist_ok=True)
    (benchmark.job_dir/"candidate-config.json").write_text(json.dumps({
        "mode": mode, "tycho_mode": os.environ["TYCHO_MODE"],
        "treatment": os.environ["TYCHO_NOOA_TREATMENT"],
        "model": os.environ["ARC_LLM_MODEL"], "context_window": 128*1024,
        "max_output": 12*1024, "max_active_games": 10,
        "game_count": len(benchmark.games),
        "per_game_seconds": benchmark.solver.max_runtime_s_per_game,
        "per_game_action_cap": benchmark.solver.max_actions_per_game,
        "scheduler_calibration": "inherited Duck heuristic; not Tycho-qualified",
    }, indent=2)+"\n")
    # SIGTERM asks TAAF to cancel and close every worker, preserving partial runs.
    import signal
    signal.signal(signal.SIGTERM, lambda *_: benchmark.request_stop())
    soft_end = datetime.fromisoformat(args.soft_end) if args.soft_end else None
    asyncio.run(benchmark.run(soft_end_time=soft_end, runtime_environment=target))
    crashed = [run.game_id for run in benchmark.game_runs if run.state == "crashed"]
    if crashed:
        raise SystemExit(f"hybrid crashed games: {crashed}")


if __name__ == "__main__":
    main()
