#!/usr/bin/env python3
"""Reject protocol/evidence regressions before spending model or game budget.

Usage:
  python3 tools/hermetic_gate.py campaign [--output REPORT.json]
  python3 tools/hermetic_gate.py audit RUN_DIR [--output REPORT.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hermetic_gate_core import audit_run, run_fault_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    campaign = commands.add_parser("campaign", help="run generated fault mutations")
    campaign.add_argument("--output", type=Path)
    audit = commands.add_parser("audit", help="audit one preserved run")
    audit.add_argument("run_dir", type=Path)
    audit.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_fault_campaign() if args.command == "campaign" else audit_run(args.run_dir)
    _emit(result, args.output)
    return 0 if result["status"] == "pass" else 1


def _emit(result: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    raise SystemExit(main())
