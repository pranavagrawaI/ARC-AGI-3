#!/usr/bin/env python3
"""Count whether `symmetry` was USED, not whether it was mentioned.

Written and calibrated against a pre-symmetry run BEFORE the run it is meant to judge, so
the outcome measure could not be chosen to suit the data. The distinction it enforces:
a call that names `symmetry` and raises NameError/ImportError is a failed reach, and a
report that pools it with a working call would turn a wiring bug into an uptake success.

Usage: python tools/symmetry_uptake.py runs/<run-dir>
"""

import json
import re
import sys
from pathlib import Path

LANES = {"actor": "tool-trace.jsonl", "builder": "builder-tool-trace.jsonl"}
USE = re.compile(r"\bsymmetry\s*\.\s*(\w+)")
# The card defines ERR as "a reach that raises". Naming individual exception classes was an
# incomplete implementation of that sentence: the first live run raised ValueError from
# inside symmetry.canon and the old list scored it as a success. A traceback is the exact,
# exception-agnostic signature of "raised", so match on that instead.
BROKEN = ("Traceback (most recent call last)",)


def _successful_reach(row: dict) -> bool:
    """A reach executed only when the durable tool trace says it completed."""
    return row.get("status") == "complete" and not any(
        broken in str(row.get("result", "")) for broken in BROKEN
    )


def main(run: Path) -> int:
    print(f"# symmetry uptake — {run.name}\n")
    total_ok = 0
    for lane, filename in LANES.items():
        path = run / filename
        if not path.is_file():
            print(f"{lane:8} (no {filename})")
            continue
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        python = [r for r in rows if r.get("name") == "run_python"]
        reaches = [r for r in python if USE.search(r["arguments"].get("code", ""))]
        ok = [r for r in reaches if _successful_reach(r)]
        total_ok += len(ok)
        names = sorted({m for r in ok for m in USE.findall(r["arguments"]["code"])})
        print(
            f"{lane:8} run_python={len(python):4}  reached={len(reaches):3}  "
            f"executed={len(ok):3}  functions={names or '-'}"
        )
        for r in reaches:
            mark = "ok " if r in ok else "ERR"
            print(
                f"           [{mark}] call {r['call_index']} level {r['level']} "
                f"turn {r['turn_in_level']}"
            )

    # Did any reach survive into a durable artifact, or only into scratch?
    ws = run / "workspace"
    print()
    for f in sorted(ws.rglob("*.py")) + sorted(ws.rglob("notes/*.md")):
        if f.name in ("wmlib.py", "symmetry.py", "verify.py", "plan.py"):
            continue
        text = f.read_text(errors="replace")
        if "symmetry" in text:
            print(
                f"durable: {f.relative_to(ws)} mentions symmetry "
                f"({text.count('symmetry')}x)"
            )
    print(f"\nPRIMARY OUTCOME: executed_symmetry_calls = {total_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
