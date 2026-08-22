#!/usr/bin/env python3
"""Zero-call epistemic-funnel census for the Tycho/NOOA development corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Direct script and ``python -m tools.epistemic_funnel`` are both supported.
    from tools.epistemic_funnel_core import mine, write_csv, write_json
    from tools.epistemic_funnel_render import write_html, write_report
except ModuleNotFoundError:  # pragma: no cover - exercised by the documented script form
    from epistemic_funnel_core import mine, write_csv, write_json
    from epistemic_funnel_render import write_html, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-obligation-actions", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.goal_obligation_actions < 1:
        raise SystemExit("--goal-obligation-actions must be >= 1")
    root = args.root.resolve()
    output = args.output.resolve()
    result = mine(root, horizon=args.goal_obligation_actions)
    write_json(result, output / "funnel.json")
    write_csv(result, output / "decisions.csv")
    write_report(result, output / "REPORT.md")
    write_html(result, output / "index.html")
    print(json.dumps({
        "output": str(output),
        "runs": result["aggregate"]["runs"],
        "supported_episodes": result["aggregate"]["supported_episodes"],
        "family_counts": result["aggregate"]["family_counts"],
        "model_calls": result["execution"]["model_calls"],
        "arc_environment_calls": result["execution"]["arc_environment_calls"],
        "positive_controls_passed": all(row["pass"] for row in result["positive_controls"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
