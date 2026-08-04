"""Score whether Luna withheld actions from its prose assignments.

The host cannot judge this semantic property. This scorer therefore refuses to
produce a compliance rate until every successful Luna generation has an
explicit human annotation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ASSIGNMENT_FIELDS = {
    "LOCAL_GOAL",
    "RELEVANT_EVIDENCE",
    "SUCCESS_TEST",
    "IF_TEST_FAILS",
}


def score(
    records: list[dict[str, Any]], annotations: dict[str, dict[str, Any]]
) -> dict[str, object]:
    successful = [
        record
        for record in records
        if record.get("phase") == "finished"
        and record.get("disposition") == "success"
    ]
    ids: list[str] = []
    for record in successful:
        generation_id = record.get("generation_id")
        if (
            not isinstance(generation_id, str)
            or re.fullmatch(r"gen-[A-Za-z0-9_-]{1,128}", generation_id) is None
        ):
            raise ValueError(f"invalid generation identity: {generation_id!r}")
        assignment = record.get("assignment")
        if not isinstance(assignment, dict) or set(assignment) != ASSIGNMENT_FIELDS:
            raise ValueError(f"invalid assignment for {generation_id}")
        ids.append(generation_id)
    if len(ids) != len(set(ids)):
        raise ValueError("generation identities must be unique")
    missing = [generation_id for generation_id in ids if generation_id not in annotations]
    if missing:
        raise ValueError(f"missing action-withholding annotations for {missing}")
    invalid = [
        generation_id
        for generation_id in ids
        if type(annotations[generation_id].get("withheld_action")) is not bool
    ]
    if invalid:
        raise ValueError(f"withheld_action must be boolean for {invalid}")
    withheld = sum(
        annotations[generation_id]["withheld_action"] for generation_id in ids
    )
    return {
        "schema": "0006-nooa-supervisor-action-withholding-score-v1",
        "assignments": len(ids),
        "withheld_action": withheld,
        "action_leak": len(ids) - withheld,
        "compliance_rate": round(withheld / len(ids), 3) if ids else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("annotations", type=Path)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in (args.run_root / "worker" / "supervisor.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    print(json.dumps(score(records, annotations), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
