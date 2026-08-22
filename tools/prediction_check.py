#!/usr/bin/env python3
"""Observe architectural risk proxies in preserved run evidence.

Usage:
  python3 tools/prediction_check.py RUN_DIR [RUN_DIR ...] [--output REPORT.json]
  python3 tools/prediction_check.py --corpus --manifest MANIFEST.json [--output REPORT.json]

Corpus mode without a manifest emits observations but exits with review-required
status: checkpoint presence is not scientific run eligibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # Direct script and ``python -m tools.prediction_check`` are both supported.
    from tools.prediction_check_core import (
        check_corpus,
        discover_runs,
        load_corpus_manifest,
    )
except ModuleNotFoundError:  # pragma: no cover - documented direct-script form
    from prediction_check_core import check_corpus, discover_runs, load_corpus_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs", nargs="*", type=Path, help="run directories containing game/"
    )
    parser.add_argument(
        "--corpus", action="store_true", help="inspect preserved runs under --root"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="reviewed corpus admission manifest; required for inferential tallies",
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    run_dirs = list(args.run_dirs)
    eligibility = None
    if args.manifest is not None:
        manifest_dirs, eligibility = load_corpus_manifest(
            args.manifest.resolve(), root
        )
        run_dirs.extend(manifest_dirs)
    if args.corpus:
        discovered = discover_runs(root)
        run_dirs.extend(discovered)
        if eligibility is None:
            eligibility = {
                path.resolve(): {
                    "reviewed": False,
                    "eligible": False,
                    "reason": "corpus discovery requires explicit admission review",
                }
                for path in discovered
            }
    if not run_dirs:
        raise SystemExit("give at least one RUN_DIR, or pass --corpus/--manifest")

    result = check_corpus(run_dirs, eligibility=eligibility)
    _emit(result, args.output)
    return {"pass": 0, "fail": 1, "review_required": 2}[result["status"]]


def _emit(result: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    raise SystemExit(main())
