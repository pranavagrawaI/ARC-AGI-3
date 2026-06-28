"""Run experiment 0001 through the public-interface harness."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import arc_agi
from arc_agi.base import OperationMode

from experiments.harness import JsonlTraceWriter, run_episode

from agent import RandomAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()

    trace_path = args.trace or (
        REPO
        / "experiments/artifacts/0000-random-agent"
        / f"{args.game}-seed-{args.seed}.jsonl"
    )
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(args.game)
    agent = RandomAgent(seed=args.seed)
    with JsonlTraceWriter(trace_path) as trace:
        result = run_episode(env, agent, max_actions=args.max_actions, trace=trace)

    action_counts = Counter(
        transition.decision.action for transition in result.transitions
    )
    print(json.dumps({
        "agent": "0000-random-agent",
        "game": args.game,
        "seed": args.seed,
        "actions": result.actions,
        "action_counts": dict(sorted(action_counts.items())),
        "levels_completed": result.final.levels_completed,
        "state": result.final.state,
        "stop_reason": result.stop_reason,
        "trace": str(trace_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
