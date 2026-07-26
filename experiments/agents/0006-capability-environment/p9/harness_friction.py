"""Read a run's process log and report where the harness, not the game, defeated the model.

The observation-filename defect sat in 58 recorded runs as the single largest
cause of tool failure and went unnoticed for months, because reading a run meant
reading a transcript by eye. This turns a run into a measurement.

It reads only ``process.jsonl``, so it works on runs that predate every module
here — including the recorded 0002 runs — and on any future 0002-R run.

Attribution is a heuristic over error text and is reported as such: ``interface``
means the model could not have known better from what the harness told it,
``model`` means the harness stated what was needed, ``unclear`` is neither. The
counts are evidence; the attribution is a reading of them.

Run::

    uv run python3 experiments/agents/0006-capability-environment/p9/harness_friction.py
    uv run python3 .../harness_friction.py --runs <dir> [<dir> ...] --json report.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "p9"

from p9.failures import FRICTION_KINDS, classify_friction


REPO = Path(__file__).resolve().parents[4]
DEFAULT_RUNS = REPO / "experiments/agents/0002-llm-repl-agent/artifacts/runs"


def read_events(run: Path) -> list[dict[str, Any]]:
    log = run / "process.jsonl"
    if not log.is_file():
        return []
    events = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def scan_run(run: Path) -> dict[str, Any]:
    events = read_events(run)
    tool_calls = 0
    frictions: list[dict[str, Any]] = []
    truncated = 0
    for event in events:
        if event.get("type") != "python_tool":
            continue
        tool_calls += 1
        try:
            output = json.loads(event.get("output") or "{}")
        except json.JSONDecodeError:
            continue
        if output.get("truncated"):
            truncated += 1
        friction = classify_friction(event.get("code"), output)
        if friction is None:
            continue
        frictions.append(
            {
                "action_step": event.get("action_step"),
                "kind": friction.kind,
                "attributable_to": friction.attributable_to,
                "detail": friction.detail,
            }
        )
    return {
        "run": run.name,
        "actions": sum(1 for e in events if e.get("type") == "transition"),
        "tool_calls": tool_calls,
        "tool_failures": len(frictions),
        "truncated_outputs": truncated,
        "decision_repairs": sum(1 for e in events if e.get("type") == "decision_repair"),
        "evictions": sum(1 for e in events if e.get("type") == "context_eviction"),
        "frictions": frictions,
    }


def report(runs: Iterable[Path]) -> dict[str, Any]:
    scanned = [scan_run(run) for run in runs]
    scanned = [item for item in scanned if item["tool_calls"] or item["actions"]]
    kinds = collections.Counter()
    attribution = collections.Counter()
    examples: dict[str, dict[str, str]] = {}
    for item in scanned:
        for friction in item["frictions"]:
            kinds[friction["kind"]] += 1
            attribution[friction["attributable_to"]] += 1
            examples.setdefault(
                friction["kind"], {"run": item["run"], "detail": friction["detail"]}
            )
    totals = {
        key: sum(item[key] for item in scanned)
        for key in ("actions", "tool_calls", "tool_failures", "truncated_outputs",
                    "decision_repairs", "evictions")
    }
    return {
        "schema": "p9-harness-friction-v1",
        "runs_scanned": len(scanned),
        "totals": totals,
        "friction_by_kind": {kind: kinds[kind] for kind in FRICTION_KINDS if kinds[kind]},
        "friction_by_attribution": dict(attribution),
        "examples": examples,
        "runs": [
            {key: item[key] for key in item if key != "frictions"} for item in scanned
        ],
        "attribution_note": (
            "Attribution is a heuristic over error text. 'interface' means the model "
            "could not have known better from what the harness told it."
        ),
    }


def render(result: dict[str, Any]) -> str:
    totals = result["totals"]
    lines = [
        f"runs scanned: {result['runs_scanned']}   actions: {totals['actions']}   "
        f"tool calls: {totals['tool_calls']}   failures: {totals['tool_failures']}"
        + (
            f" ({totals['tool_failures'] / totals['tool_calls']:.0%})"
            if totals["tool_calls"]
            else ""
        ),
        "",
    ]
    if not result["friction_by_kind"]:
        lines.append("  no tool failures recorded")
        return "\n".join(lines)
    width = max(len(kind) for kind in result["friction_by_kind"])
    for kind, count in sorted(
        result["friction_by_kind"].items(), key=lambda item: -item[1]
    ):
        example = result["examples"][kind]
        lines.append(f"  {count:4}  {kind:<{width}}")
        lines.append(f"        {example['run'][:32]}: {example['detail'][:88]}")
    lines.append("")
    for label in ("interface", "unclear", "model"):
        count = result["friction_by_attribution"].get(label, 0)
        if count:
            lines.append(f"  attributable to {label}: {count}")
    lines.append(
        f"\n  also: {totals['decision_repairs']} decision repairs, "
        f"{totals['truncated_outputs']} truncated outputs, {totals['evictions']} evictions"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="*", help="run directories to scan")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args(argv)

    runs = args.runs or sorted(
        path for path in args.runs_root.iterdir() if (path / "process.jsonl").is_file()
    )
    result = report(runs)
    print(render(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
