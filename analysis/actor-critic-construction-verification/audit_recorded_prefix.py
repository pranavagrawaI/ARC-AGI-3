"""Read-only audit of real incumbent IPC for construction input checks."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "analysis/deployment-probes/20260912-objection-closure-public-v2/raw/qualification/solver-workspaces"


def rows(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit() -> dict:
    records = []
    for path in sorted(SOURCE.glob("*/ipc/transitions.jsonl")):
        transitions = rows(path)
        states = rows(path.with_name("states.jsonl"))
        per_turn = Counter(row["turn"] for row in transitions)
        records.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "executed_transitions": len(transitions),
            "actor_state_notifications": len(states),
            "multiple_executed_steps_per_actor_turn": any(n > 1 for n in per_turn.values()),
            "max_executed_steps_per_actor_turn": max(per_turn.values(), default=0),
            "steps_strictly_increasing": all(a["step"] < b["step"] for a, b in zip(transitions, transitions[1:])),
            "grids_are_integer_rows": all(
                isinstance(observation.get("grid"), list)
                and all(isinstance(row, list) and all(type(cell) is int and 0 <= cell <= 15 for cell in row)
                        for row in observation["grid"])
                for item in transitions for observation in (item["before"], item["after"])
            ),
            "transition_row_keys": sorted(transitions[0]) if transitions else [],
        })
    assert records, "no real IPC records found"
    assert all(record["steps_strictly_increasing"] and record["grids_are_integer_rows"] for record in records)
    return {"scope": "Historical real actor IPC format audit only; no constructor or prospective test.", "records": records}


if __name__ == "__main__":
    result = audit()
    output = Path(__file__).with_name("recorded-prefix-audit.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
