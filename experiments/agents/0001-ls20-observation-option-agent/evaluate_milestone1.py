"""Generate the Milestone 1 passive trace-understanding report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from observation_option_agent import Milestone1Analyzer


DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "human-trace-events.jsonl"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-1-summary.json"
)


def evaluate(events_path: Path) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    summary = Milestone1Analyzer().analyze(records)
    summary["event_artifact"] = str(events_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = evaluate(args.events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
