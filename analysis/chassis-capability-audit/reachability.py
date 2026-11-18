#!/usr/bin/env python3
"""Is a Tycho capability reachable from the NOOA actor, in the chassis that will run?

Audits at the point of use, not the point of definition. A module under
`solvers/tycho_nooa/` is not available to a NOOA CodeAct cell: the two solvers share no
runtime. The only evidence that counts is a call site in the live chassis source.

    python3 analysis/chassis-capability-audit/reachability.py
    python3 analysis/chassis-capability-audit/reachability.py --src <dir> --require symmetry,planner

`--require` exits non-zero when a named capability is absent, so a launch script can gate
on it. Ordinary runs report and exit 0.
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The live NOOA source for the q3 series. The 0158 snapshot is stale (no harness.py).
DEFAULT_SRC = REPO / "analysis/local-probes/20260904T-host-control-loop-q3/patched-source/after"

# capability -> (tokens that would appear at a call site, where it lives in Tycho)
CAPABILITIES = {
    "symmetry": (
        ("symmetry", "canon", "congruent", "transporter", "stabiliser"),
        "tycho_nooa/symmetry_template.py",
    ),
    "planner": (
        ("plan_bfs", "plan_astar", "plan_subgoals", "max_nodes", "prune_game_over"),
        "tycho/workspace/wmlib_template.py:1543-1770",
    ),
    "segmentation": (
        ("segment", "segment_summary", "connected_components"),
        "tycho/workspace/wmlib_template.py:110,180",
    ),
    "dynamics-probe": (
        ("key_counterexample", "action_laws"),
        "tycho_nooa/dynamics_template.py:41,125",
    ),
    "death-index": (
        ("death_events", "solved_actions"),
        "tycho/workspace/wmlib_template.py:893,1024",
    ),
    "terminal-evidence": (
        ("level_terminal", "terminal_events"),
        "tycho/workspace/wmlib_template.py:1038",
    ),
    "animation-frames": (
        ("event_frames", "animation"),
        "tycho/workspace/wmlib_template.py:912,989",
    ),
}


def main() -> int:
    args = _parse_args()
    sources = sorted(Path(args.src).glob("*.py"))
    if not sources:
        print(f"no python sources under {args.src}", file=sys.stderr)
        return 2
    text = "\n".join(p.read_text(errors="replace") for p in sources)

    print(f"# chassis reachability — {args.src}")
    print(f"# {len(sources)} files: {', '.join(p.name for p in sources)}\n")
    absent = set()
    for name, (tokens, site) in CAPABILITIES.items():
        hits = {t: len(re.findall(rf"\b{re.escape(t)}\b", text)) for t in tokens}
        found = {t: n for t, n in hits.items() if n}
        verdict = "REACHABLE" if found else "ABSENT   "
        if not found:
            absent.add(name)
        detail = ", ".join(f"{t}={n}" for t, n in found.items()) or site
        print(f"{verdict}  {name:20} {detail}")

    required = {r for r in args.require.split(",") if r} if args.require else set()
    missing = required & absent
    if missing:
        print(f"\nREQUIRED BUT ABSENT: {', '.join(sorted(missing))}")
        return 1
    return 0


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(DEFAULT_SRC), help="live chassis source directory")
    p.add_argument("--require", default="", help="comma-separated capabilities to gate on")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
