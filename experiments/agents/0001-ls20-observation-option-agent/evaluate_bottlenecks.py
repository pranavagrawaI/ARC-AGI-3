"""Run option-agent bottleneck diagnostics for LS20/AR25/CD82."""

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

from observation_option_agent.bottlenecks import (  # noqa: E402
    default_trace_specs,
    evaluate_suite,
    introspect_r3a_suite,
)


DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent/bottleneck-diagnostics"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("three_game",),
        default="three_game",
        help="Diagnostic suite to run. v1 supports LS20/AR25/CD82.",
    )
    parser.add_argument(
        "--codex-results-root",
        type=Path,
        default=None,
        help="Reserved for LS20 oracle integration; v1 computable diagnostics skip oracle.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-game transition limit for smoke tests.",
    )
    parser.add_argument(
        "--introspect-r3a",
        action="store_true",
        help="Generate the R3a schema introspection report instead of the numeric bottleneck suite.",
    )
    args = parser.parse_args()

    if args.introspect_r3a:
        summary = introspect_r3a_suite(
            default_trace_specs(REPO),
            output_dir=args.output,
            limit=args.limit,
        )
    else:
        summary = evaluate_suite(
            default_trace_specs(REPO),
            output_dir=args.output,
            limit=args.limit,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
