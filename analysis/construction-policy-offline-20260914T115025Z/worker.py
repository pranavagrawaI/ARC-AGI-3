"""ARC-host JSONL worker. Launch with arc-host; do not push to Kaggle from this identity.

cd arc-host && uv run --frozen arc-host --game ls20 --seed 0 --max-actions 10 \\
  --run-dir /tmp/construction-offline-game --env-file ../.env -- \\
  python /absolute/path/to/worker.py --candidate /absolute/path/to/candidates/exploratory.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gameplay_solver import GameplaySolver

HOST_PREFIX = "ARC_HOST\t"
WORKER_PREFIX = "ARC_AGENT\t"
PROTOCOL_VERSION = 3
DEFAULT_CANDIDATE = HERE / "candidates" / "exploratory.py"


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load candidate " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("initialize", "predict", "assimilate"):
        if not callable(getattr(module, name, None)):
            raise ValueError("candidate missing " + name)
    return module


def send(kind: str, **fields: Any) -> None:
    payload = {"protocol_version": PROTOCOL_VERSION, "type": kind, **fields}
    sys.stdout.write(WORKER_PREFIX + json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def envelope(message: dict[str, Any], key: str) -> dict[str, Any]:
    value = message.get(key)
    if key == "initial_observation":
        return value if isinstance(value, dict) else {}
    observation = message.get("observation")
    if isinstance(observation, dict):
        return observation
    after = message.get("after")
    if isinstance(after, dict):
        return after
    return {}


def available_actions(observation: dict[str, Any]) -> list[str]:
    actions = observation.get("available_actions")
    if not isinstance(actions, list) or not all(isinstance(x, str) for x in actions):
        raise ValueError("available_actions must be a list of strings")
    return actions


def serve(solver: GameplaySolver) -> None:
    for line in sys.stdin:
        if not line.startswith(HOST_PREFIX):
            continue
        message = json.loads(line[len(HOST_PREFIX) :])
        if message.get("protocol_version") != PROTOCOL_VERSION:
            send("error", message="protocol version mismatch")
            return
        kind = message.get("type")
        if kind == "start_game":
            ready = solver.start_game(envelope(message, "initial_observation"))
            send("ready", worker_id=ready["worker_id"], metadata=ready["metadata"])
        elif kind == "turn":
            observation = envelope(message, "observation")
            decision = solver.decide(observation, available_actions(observation))
            send("decision", sequence=message["sequence"], decision=decision.to_json())
        elif kind == "action_result":
            solver.observe(envelope(message, "after"))
            send("observed", sequence=message["sequence"])
        elif kind == "finish":
            send("finished")
        elif kind == "close":
            return
        else:
            send("error", message="unsupported type " + str(kind))
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Construction-policy gameplay worker")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    if not candidate.is_file():
        raise SystemExit("candidate file does not exist: " + str(candidate))
    solver = GameplaySolver(load_candidate(candidate), worker_id=str(os.getpid()))
    serve(solver)


if __name__ == "__main__":
    main()
