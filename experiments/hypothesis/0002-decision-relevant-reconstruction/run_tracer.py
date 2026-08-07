from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reconstruction import run_operational  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline 0002 Oracle Coach tracer bullet.")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "fixtures/ls20-level1-step13-v1.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / (
        "operational-ls20-step13-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    run = run_operational(args.fixture, output)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    summary = result["summary"]
    print(f"run: {run}")
    print(f"verdict: {summary['verdict']}")
    print(f"authority: {summary['authority_label']}")
    print(f"candidate_count: {summary['candidate_count']}")
    print(f"search_expansions: {summary['search_expansions']}")
    print(f"unresolved_cases: {summary['unresolved_cases']}")
    print(f"provider_calls: {summary['provider_calls']}")
    print(f"environment_actions: {summary['environment_actions']}")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())