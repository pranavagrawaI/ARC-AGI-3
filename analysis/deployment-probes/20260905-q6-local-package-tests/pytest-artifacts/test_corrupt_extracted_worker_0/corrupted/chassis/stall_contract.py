"""One stall definition for the Actor and artifact scorer; no model or game rules."""

from __future__ import annotations

import json
import re
from pathlib import Path

from host_contract import ObservationRecord

WINDOW, MIN_ACTIONS, SIMILARITY = 6, 25, 0.55
OUTCOMES = frozenset({"move", "no_impact", "level_up", "terminal"})


def detect_stall(
    history: list[dict],
    state: dict,
    rationale: str,
    *,
    window: int = WINDOW,
    min_actions: int = MIN_ACTIONS,
    similarity: float = SIMILARITY,
) -> dict | None:
    if len(history) < window or state["step"] < min_actions:
        return None
    recent = history[-window:]
    if state["levels_completed"] > recent[0]["levels_completed"]:
        return None

    def tokens(text):
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    here = tokens(rationale)
    scores = []
    for row in recent:
        other = tokens(row["rationale"])
        scores.append(len(here & other) / len(here | other) if here and other else 0.0)
    best = max(scores)
    tried = {a.split()[0] for row in recent for a in row["actions"] if a.strip()}
    untried = [a for a in state["available_actions"] if a not in tried]
    if best < similarity or not untried:
        return None
    return {
        "step": state["step"],
        "turn": state["turn"],
        "thresholds": {
            "window": window,
            "min_actions": min_actions,
            "similarity": similarity,
        },
        "rationale": rationale,
        "repeated_rationales": [row["rationale"] for row in recent],
        "similarity": best,
        "untried_actions": untried,
    }


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # Writers append one newline-terminated record; an in-flight tail is not a row.
    text = path.read_text()
    return [
        json.loads(line)
        for line in text.splitlines(keepends=True)
        if line.endswith("\n") and line.strip()
    ]


def append(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def transition_outcomes(transitions: list[dict]) -> list[str]:
    record = ObservationRecord()
    outcomes = []
    for row in transitions:
        before, after = row["before"], row["after"]
        if not row.get("automatic"):
            record.observe(before["grid"], after["grid"])
        outcomes.append(
            record.outcome(
                record.classify(before["grid"], after["grid"]),
                after["state"],
                after["levels_completed"] > before["levels_completed"],
            )
        )
    return outcomes


def falsifier_satisfied(
    condition: str | None, transition: dict, outcome: str
) -> bool | None:
    condition = (condition or "").strip()
    if condition in OUTCOMES:
        return condition == outcome
    before, after = transition["before"], transition["after"]
    if condition == "grid_changed":
        return before["grid"] != after["grid"]
    if condition == "grid_unchanged":
        return before["grid"] == after["grid"]
    match = re.fullmatch(r"levels_completed\s*(==|>=|>)\s*(\d+)", condition)
    if match:
        n, value = int(match[2]), after["levels_completed"]
        return {"==": value == n, ">=": value >= n, ">": value > n}[match[1]]
    return None


def record_consequences(ipc: Path) -> None:
    """Host-side closure survives terminal cleanup with no further Actor call."""
    authority = read_rows(ipc / "authority.jsonl")
    observed = {
        r["stall_id"] for r in authority if r["event"] == "CONSEQUENCE_OBSERVED"
    }
    transitions = read_rows(ipc / "transitions.jsonl")
    indexed = {
        r["step"]: (r, outcome)
        for r, outcome in zip(transitions, transition_outcomes(transitions))
    }
    for review in authority:
        if review["event"] != "AUTHORITY_REVIEWED" or review["stall_id"] in observed:
            continue
        item = indexed.get(review["step"] + 1)
        if item is None:
            continue
        row, outcome = item
        if row["turn"] != review["turn"] or row["action"] != review["action"]:
            (ipc / "ABORT").write_text(
                "reviewed action does not match executed transition"
            )
            raise RuntimeError("reviewed action does not match executed transition")
        append(
            ipc / "authority.jsonl",
            {
                "event": "CONSEQUENCE_OBSERVED",
                "stall_id": review["stall_id"],
                "turn": row["turn"] + 1,
                "step": row["step"],
                "action": row["action"],
                "predicted": review.get("predictions"),
                "actual": outcome,
                "abandon_if_satisfied": falsifier_satisfied(
                    review.get("abandon_if"), row, outcome
                ),
            },
        )


def valid_basis(value) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, str) and v.strip() for v in value)
    )


def resolution_error(row: dict, available: list[str]) -> str | None:
    disposition = row.get("disposition")
    if disposition not in ("continue", "probe", "switch"):
        return "disposition must be continue, probe, or switch"
    if not valid_basis(row.get("previous_basis")) or not valid_basis(
        row.get("active_basis")
    ):
        return "previous_basis and active_basis must be non-empty lists of non-empty strings"
    if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
        return "rationale must be non-empty"
    action = row.get("action")
    if not isinstance(action, str) or not action.strip():
        return "action must be a single legal action string"
    parts = action.split()
    if parts[0] not in available:
        return "action is not available in this state"
    if parts[0] == "CLICK":
        if len(parts) != 3 or any(
            not p.isdigit() or not 0 <= int(p) <= 63 for p in parts[1:]
        ):
            return "CLICK requires x y coordinates from 0 through 63"
    elif len(parts) != 1:
        return "non-click action must be a single name"
    if disposition == "continue" and (
        not isinstance(row.get("abandon_if"), str) or not row["abandon_if"].strip()
    ):
        return "continue requires a non-empty abandon_if"
    if row.get("abandon_if") is not None and not isinstance(row["abandon_if"], str):
        return "abandon_if must be a string or null"
    if disposition == "switch" and row["active_basis"] == row["previous_basis"]:
        return "switch requires a changed active_basis"
    if disposition == "probe":
        predictions = row.get("predictions")
        if (
            not isinstance(predictions, dict)
            or len(predictions) < 2
            or any(
                not isinstance(k, str)
                or not k.strip()
                or not isinstance(v, str)
                or not v.strip()
                for k, v in predictions.items()
            )
            or len(set(predictions.values())) < 2
        ):
            return "probe requires at least two distinct predicted outcomes keyed by alternative"
    return None


class StallReview:
    """Detection blocks treatment submission until one reviewed action is committed.

    The next invocation observes its consequence and authors the basis that now
    governs. No additional model invocation is scheduled by this object.
    """

    def __init__(self, ipc: Path, treatment: bool):
        self.ipc = ipc
        self.treatment = treatment
        self.stall_id = 0
        self.evaluated_turn = None
        self.opened: dict | None = None
        self.staged: dict | None = None
        self.pending: dict | None = None
        self.consequence: dict | None = None
        self.rejected = 0

    def before_submit(
        self, state: dict, rationale: str, actions: list[str], history: list[dict]
    ) -> str | None:
        self.observe()
        if self.pending is not None:
            return "call finish_stall(active_basis_after=[...], rationale=...) after observing the reviewed action"
        if self.evaluated_turn != state["turn"]:
            self.evaluated_turn = state["turn"]
            append(
                self.ipc / "policy_attempts.jsonl",
                {
                    "turn": state["turn"],
                    "step": state["step"],
                    "rationale": rationale,
                    "actions": actions,
                },
            )
            detection = detect_stall(history, state, rationale)
            if detection:
                self.stall_id += 1
                self.event("STALL_DETECTED", **detection)
                if self.treatment:
                    self.opened = detection
        if self.opened is not None and self.staged is None:
            return "stall suspends this policy; call resolve_stall with one action and an authored basis"
        return None

    def stage(self, state: dict, row: dict) -> str | None:
        if self.opened is None or self.opened["turn"] != state["turn"]:
            return self.reject("no stall is open on this turn")
        error = resolution_error(row, state["available_actions"])
        if error:
            return self.reject(error)
        self.staged = row
        return None

    def commit(self, state: dict) -> None:
        if self.staged is None:
            return
        row = {
            "stall_id": self.stall_id,
            "turn": state["turn"],
            "step": state["step"],
            **self.staged,
        }
        self.event("AUTHORITY_REVIEWED", **row)
        self.pending = row
        self.consequence = None
        self.opened = None
        self.staged = None
        self.rejected = 0

    def observe(self) -> None:
        if self.pending is None or self.consequence is not None:
            return
        # The executing host owns this event; reading it cannot duplicate it.
        for row in read_rows(self.ipc / "authority.jsonl"):
            if (
                row["event"] == "CONSEQUENCE_OBSERVED"
                and row["stall_id"] == self.pending["stall_id"]
            ):
                self.consequence = row
                return

    def finish(self, active_basis_after: list[str], rationale: str) -> str:
        self.observe()
        if self.pending is None or self.consequence is None:
            return self.reject("no reviewed consequence is available")
        if (
            not valid_basis(active_basis_after)
            or not isinstance(rationale, str)
            or not rationale.strip()
        ):
            return self.reject("active_basis_after and rationale must be non-empty")
        changed = active_basis_after != self.pending["active_basis"]
        self.event(
            "AUTHORITY_RESOLVED",
            turn=self.consequence["turn"],
            step=self.consequence["step"],
            resolution="changed" if changed else "retained",
            active_basis_after=active_basis_after,
            rationale=rationale,
        )
        self.pending = None
        self.consequence = None
        self.rejected = 0
        return "authority consequence recorded; submission is open"

    def reject(self, message: str) -> str:
        self.rejected += 1
        self.event("RESOLUTION_REJECTED", reason=message, consecutive=self.rejected)
        if self.rejected >= 8:
            (self.ipc / "ABORT").write_text(
                "eight consecutive rejected stall resolutions"
            )
        return "REJECTED: " + message

    def event(self, event: str, **fields) -> None:
        append(
            self.ipc / "authority.jsonl",
            {"event": event, "stall_id": self.stall_id, **fields},
        )

    def status(self) -> str:
        self.observe()
        if self.consequence:
            return f"REVIEW CONSEQUENCE: {self.consequence!r}. Call finish_stall(active_basis_after=[...], rationale=...)."
        if self.opened:
            return f"STALL: {self.opened!r}. Call resolve_stall(disposition=..., previous_basis=[...], active_basis=[...], action=..., rationale=..., abandon_if=..., predictions=...)."
        return ""

# injected byte drift
