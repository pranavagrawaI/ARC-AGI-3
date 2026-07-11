"""Validate a completed process annotation and emit compact measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AGENT_ROOT = Path(__file__).resolve().parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from llm_repl_agent.process_log import measurement_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    annotation_path = args.run_dir / "process-annotation.json"
    output_path = args.run_dir / "process-measurements.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    process_events = [
        json.loads(line)
        for line in (args.run_dir / "process.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = measurement_summary(
        annotation,
        actions=int(manifest["actions"]),
        event_indices={int(event["event_index"]) for event in process_events},
    )
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
