"""Run the gameplay solver on the local grid world. No ARC service and no Kaggle."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from gameplay_solver import GameplaySolver
from local_env import LocalGridWorld

HERE = Path(__file__).resolve().parent
DEFAULT_CANDIDATE = HERE / "candidates" / "occupancy_motion.py"


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load candidate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_local(module, *, max_actions: int) -> dict:
    env = LocalGridWorld()
    solver = GameplaySolver(module, worker_id="local")
    observation = env.reset()
    solver.start_game(observation)
    for _ in range(max_actions):
        if observation.state in {"WIN", "GAME_OVER"}:
            break
        decision = solver.decide(observation, observation.available_actions)
        observation = env.step(decision.action)
        solver.observe(observation)
    return {
        "schema": "local-gameplay-v1",
        "actions": list(env.steps),
        "action_count": len(env.steps),
        "final_block": list(env.block),
        "final_state": observation.state,
        "levels_completed": observation.levels_completed,
        "stop_reason": "terminal" if observation.state in {"WIN", "GAME_OVER"} else "action_budget",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--max-actions", type=int, default=16)
    args = parser.parse_args()
    result = run_local(load_candidate(args.candidate.resolve()), max_actions=args.max_actions)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
