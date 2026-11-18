"""Run the inherited TAAF benchmark inside isolated pinned Solver dependencies."""
from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import os
import pickle
from datetime import datetime
from pathlib import Path

from hybrid.solver import HybridSolver


def write_evidence_index(directory):
    """Bound the collector's inventory to this run's actual solver artifacts."""
    root=Path(directory)
    files=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name=='evidence-index.json': continue
        relative=path.relative_to(root)
        if '__pycache__' in relative.parts: continue
        files.append({'path':root.name+'/'+relative.as_posix(),
                      'critical':len(relative.parts)<=3,
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (root/'evidence-index.json').write_text(json.dumps({'files':files},indent=2)+'\n')


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
    if mode not in {"smoke", "small", "public25"}:
        raise ValueError("HYBRID_RUN_MODE must be smoke, small or public25")
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
    if mode in {"smoke", "small"}:
        names = {"sb26", "ls20", "ar25"}
        benchmark.games = [game for game in benchmark.games
                           if str(game.env_name).split("-")[0] in names]
        if len(benchmark.games) != 3:
            raise RuntimeError("small game set requires exactly sb26, ls20, ar25")
        if {str(game.env_name).split("-")[0] for game in benchmark.games} != names:
            raise RuntimeError("small game set contains duplicate or missing games")
        benchmark.game_weights = None
        benchmark.n_passes = 1
        benchmark.solver.concurrency = 3
        benchmark.solver.max_runtime_s_per_game = 600 if mode == "smoke" else 2400
        benchmark.solver.max_actions_per_game = 16 if mode == "smoke" else None
    benchmark.job_dir = Path(benchmark.job_dir)/"tycho-dfranzen"
    benchmark.job_dir.mkdir(parents=True, exist_ok=True)
    (benchmark.job_dir/"candidate-config.json").write_text(json.dumps({
        "mode": mode, "tycho_mode": os.environ["TYCHO_MODE"],
        "treatment": os.environ["TYCHO_NOOA_TREATMENT"],
        "model": os.environ["ARC_LLM_MODEL"], "context_window": 128*1024,
        "max_output": 12*1024,
        "max_active_games": int(os.environ.get("ARC3_MAX_ACTIVE_STREAMS", "0")),
        "concurrency": benchmark.solver.concurrency,
        "game_count": len(benchmark.games),
        "game_names": [str(game.env_name if hasattr(game, "env_name") else game.label)
                       for game in benchmark.games],
        "passes": benchmark.n_passes,
        "analyzer_timeout": benchmark.solver.analyzer_timeout,
        "builder_allocation": "actor-request-v3; cap 240s, Actor reserve min(120s, half usable remainder), host reserve 5s",
        "per_game_seconds": benchmark.solver.max_runtime_s_per_game,
        "per_game_action_cap": benchmark.solver.max_actions_per_game,
        "scheduler_calibration": "inherited Duck heuristic; not Tycho-qualified",
    }, indent=2)+"\n")
    # SIGTERM asks TAAF to cancel and close every worker, preserving partial runs.
    import signal
    soft_end = datetime.fromisoformat(args.soft_end) if args.soft_end else None
    async def run():
        loop=asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM,benchmark.request_stop)
        try:
            await benchmark.run(soft_end_time=soft_end,runtime_environment=target)
        finally:
            loop.remove_signal_handler(signal.SIGTERM)
    try:
        asyncio.run(run())
    finally:
        write_evidence_index(benchmark.job_dir)
    crashed = [run.game_id for run in benchmark.game_runs if run.state == "crashed"]
    if crashed:
        raise SystemExit(f"hybrid crashed games: {crashed}")


if __name__ == "__main__":
    main()
