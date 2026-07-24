"""Create the immutable post-pilot final freeze required for included gameplay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


AGENT_ROOT = Path(__file__).resolve().parent
REPO = AGENT_ROOT.parents[2]
for import_root in (REPO, AGENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from llm_repl_agent.authority import build_final_freeze
from llm_repl_agent.experiment_protocol import participating_source_hashes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--pilot-pair-dir", required=True, type=Path)
    parser.add_argument("--pilot-decision", required=True, type=Path)
    parser.add_argument("--release-gate-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise SystemExit(f"final freeze already exists: {args.output}")
    try:
        output = build_final_freeze(
            generation=args.generation,
            pilot_pair_dir=args.pilot_pair_dir,
            pilot_decision_path=args.pilot_decision,
            release_gate_dir=args.release_gate_dir,
            source_hashes=participating_source_hashes(),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(output, indent=2, sort_keys=True) + "\n"
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise SystemExit(f"final freeze already exists: {args.output}") from error
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(json.dumps({**output, "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
