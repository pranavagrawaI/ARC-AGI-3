"""Shared immutable evidence projection and task text; no game implementation."""

import hashlib
import json
from pathlib import Path

from bootstrap import HERE
from theory import digest

COMMON_TASK = (
    "RECORDED CONSTRUCTION TEST. Use the existing full Actor tool surface to inspect "
    "the supplied historical evidence and prepare ONE prospective action with "
    "self.submit_actions. No game executes. After the first proposal the host will "
    "reveal one fixed historical continuation, regardless of your proposed action. "
    "That continuation is NOT the result of your proposal. Update your operational "
    "knowledge and prepare one more decision. This tests construction completion, "
    "not gameplay utility or truth. All model calls share a six-request arm cap; "
    "at most 4096 completion tokens per request, 24576 total. No extra consolidator."
)


def evidence_rows():
    return [
        json.loads((HERE / "evidence" / name).read_text())
        for name in ("prefix.jsonl", "continuation.jsonl")
    ]


def catalog(rows):
    return {f"transition:{r['step']}": digest(r) for r in rows}


def state(observation, step, turn):
    return dict(
        turn=turn,
        step=step,
        level=observation["levels_completed"],
        state=observation["state"],
        levels_completed=observation["levels_completed"],
        available_actions=observation["available_actions"],
        grid_rows=[
            "".join("0123456789abcdef"[c] for c in row) for row in observation["grid"]
        ],
        action_results=[],
    )


def install_evidence(run, rows):
    ipc = Path(run) / "ipc"
    ipc.mkdir(parents=True, exist_ok=True)
    states = [state(rows[0]["before"], 0, 0)]
    states += [state(r["after"], r["step"], r["step"]) for r in rows]
    (ipc / "states.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states))
    (ipc / "transitions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return states[-1]


def protected_snapshot(run):
    return {
        name: hashlib.sha256((Path(run) / "ipc" / name).read_bytes()).hexdigest()
        for name in ("states.jsonl", "transitions.jsonl")
    }
