"""Game-independent host evidence shared by local q6 and its Kaggle join.

ObservationRecord and prediction scoring originate verbatim in the q3c harness.
"""

from __future__ import annotations

import json
from pathlib import Path


class ObservationRecord:
    """Track change frequency, exact no-change outcomes and inert-action memory.

    Frequently changing rows are diagnostic only: horizontal motion can make
    object rows look like a status band. Only identical grids establish no
    impact; terminal and level outcomes retain priority.
    """

    def __init__(self, hud_min_frac: float = 0.9, warmup: int = 5):
        self.hud_min_frac = hud_min_frac
        self.warmup = warmup
        self._row_hits: list[int] = []
        self._pairs = 0
        self.known: dict[tuple, str] = {}
        self.level_starts: list[dict] = []
        self.totals = {"no_impact": 0, "terminal": 0, "truncations": 0}

    # -- HUD discovery ----------------------------------------------------
    def observe(self, prev: list[list[int]], curr: list[list[int]]) -> None:
        if len(prev) != len(curr) or any(len(a) != len(b) for a, b in zip(prev, curr)):
            # Row frequencies have no common coordinate frame across a resize.
            self._row_hits = [0] * len(curr)
            self._pairs = 0
            return
        if not self._row_hits:
            self._row_hits = [0] * len(curr)
        self._pairs += 1
        for r, (a, b) in enumerate(zip(prev, curr)):
            if a != b:
                self._row_hits[r] += 1

    @property
    def warm(self) -> bool:
        return self._pairs >= self.warmup

    @property
    def hud_rows(self) -> list[int]:
        if not self._pairs or not self._row_hits:
            return []
        return [
            r
            for r, h in enumerate(self._row_hits)
            if h / self._pairs >= self.hud_min_frac
        ]

    # -- classification ---------------------------------------------------
    def classify(self, prev: list[list[int]], curr: list[list[int]]) -> str:
        """Classify exact equality as blocked; all other changes as playfield."""
        if prev == curr:
            return "blocked"
        # Frequently changing rows can contain a moving object. Their frequency
        # cannot establish HUD identity or authorize discarding queued actions.
        return "playfield"

    def outcome(self, label: str, state: str, level_up: bool) -> str:
        """The four classes a predicted_path entry's `expect` field may name.

        Terminal wins over everything: q2d step 103 was HUD-only AND GAME_OVER,
        and a no-impact predicate that outranks the terminal check hides a death.
        """
        if state in ("GAME_OVER", "WIN"):
            return "terminal"
        if level_up:
            return "level_up"
        return "no_impact" if label in ("hud_only", "blocked") else "move"

    # -- (playfield, action) memory ---------------------------------------
    def playfield_key(self, grid: list[list[int]]) -> tuple:
        return tuple(tuple(row) for row in grid)

    def note(self, before: list[list[int]], action: str, outcome: str) -> None:
        if outcome in ("no_impact", "terminal"):
            self.known[(self.playfield_key(before), action)] = outcome
            self.totals[outcome] += 1

    def known_from_here(self, grid: list[list[int]], actions: list[str]) -> dict:
        key = self.playfield_key(grid)
        return {a: self.known[(key, a)] for a in actions if (key, a) in self.known}

    def to_dict(self, grid: list[list[int]], actions: list[str], extra: dict) -> dict:
        d = {
            "hud_rows": self.hud_rows,
            "hud_warm": self.warm,
            "hud_authoritative": False,
            "classification_policy": "exact-grid-change-v1",
            "known_from_here": self.known_from_here(grid, actions),
            "totals": dict(self.totals),
            "level_starts": list(self.level_starts),
        }
        d.update(extra)
        return d


def score_predicted_path(path, results: list[dict]) -> dict | None:
    """Compare a submitted predicted_path's `expect` fields to what happened.

    Host-verifiable by construction: `expect` names an outcome class the harness
    computes itself, so retrodiction becomes data instead of a rationale prefix
    that can be typed without being meant.
    """
    if not isinstance(path, list) or not path or not results:
        return None
    checked = min(len(path), len(results))
    matched = 0
    for i in range(checked):
        e = path[i]
        want = e.get("expect") if isinstance(e, dict) else None
        if want is not None and want == results[i].get("outcome"):
            matched += 1
    return {
        "checked": checked,
        "matched": matched,
        "accuracy": (matched / checked) if checked else 0.0,
        "ok": checked > 0 and (matched / checked) >= RETRODICTION_MIN,
    }


RETRODICTION_MIN = 0.6


def grid_rows(grid: list[list[int]]) -> list[str]:
    return ["".join("0123456789abcdef"[v] for v in row) for row in grid]


def atomic_json(path: Path, value: dict) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(path)


class HostEvidence:
    """Record executed transitions, then publish a complete batch result.

    states.jsonl is the commit notification. All evidence read by the Actor must
    be durable before that notification is appended.
    """

    def __init__(self, ipc: Path):
        self.ipc = ipc
        self.record = ObservationRecord()
        self.step = 0
        self.level_actions = 0
        self.results: list[dict] = []
        self.batch: dict = {}
        self.last_retrodiction = None
        self.note = ""

    def begin(self, batch: dict) -> None:
        self.batch = batch
        self.results = []
        self.note = ""

    def observe(self, before: dict, after: dict, action: str, remaining: int) -> bool:
        self.step += 1
        self.level_actions += 1
        prev, curr = before["grid"], after["grid"]
        advanced = after["levels_completed"] > before["levels_completed"]
        self.record.observe(prev, curr)
        label = self.record.classify(prev, curr)
        outcome = self.record.outcome(label, after["state"], advanced)
        self.record.note(prev, action, outcome)
        result = {
            "action": action,
            "step": self.step,
            "state": after["state"],
            "levels_completed": after["levels_completed"],
            "level_completed": advanced,
            "label": label,
            "outcome": outcome,
            "diff_pixels": sum(
                a != b for ra, rb in zip(prev, curr) for a, b in zip(ra, rb)
            ) if len(prev) == len(curr) and all(len(a) == len(b) for a, b in zip(prev, curr)) else None,
        }
        self.results.append(result)
        # Full per-action frames survive batch truncation and terminal cleanup.
        with (self.ipc / "transitions.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": self.step,
                        "turn": self.batch["turn"],
                        "action": action,
                        "before": before,
                        "after": after,
                    }
                )
                + "\n"
            )
        if advanced:
            level = before["levels_completed"]
            atomic_json(
                self.ipc / f"level_{level}_terminal.json",
                {
                    "level": level,
                    "winning_action": action,
                    "step": self.step,
                    "actions_spent_on_level": self.level_actions,
                    "pre_win_grid_rows": grid_rows(prev),
                    "terminal_grid_rows": grid_rows(curr),
                },
            )
            self.level_actions = 0
            self.record.level_starts.append(
                {
                    "level": after["levels_completed"],
                    "step": self.step,
                    "turn": self.batch["turn"],
                }
            )
        stop = (
            after["state"] in ("WIN", "GAME_OVER") or advanced or outcome == "no_impact"
        )
        if (
            outcome == "no_impact"
            and not advanced
            and after["state"] not in ("WIN", "GAME_OVER")
        ):
            self.record.totals["truncations"] += 1
        if stop:
            with (self.ipc / "batch_stops.jsonl").open("a") as handle:
                handle.write(json.dumps({"turn": self.batch["turn"], "step": self.step,
                                         "reason": outcome, "skipped_actions": remaining}) + "\n")
            self.note = (
                f"{outcome} after {action!r}; {remaining} queued actions skipped."
            )
        if stop or remaining == 0:
            self.last_retrodiction = score_predicted_path(
                self.batch.get("predicted_path"), self.results
            )
            if self.last_retrodiction is not None:
                self.last_retrodiction["turn"] = self.batch["turn"]
        return stop

    def publish(self, observation: dict, turn: int) -> None:
        from stall_contract import record_consequences

        record_consequences(self.ipc)
        atomic_json(
            self.ipc / "observations.json",
            self.record.to_dict(
                observation["grid"],
                observation["available_actions"],
                {
                    "turn": turn,
                    "step": self.step,
                    "levels_completed": observation["levels_completed"],
                    "last_retrodiction": self.last_retrodiction,
                },
            ),
        )
        entry = {
            "turn": turn,
            "step": self.step,
            "level": observation["levels_completed"],
            "state": observation["state"],
            "levels_completed": observation["levels_completed"],
            "available_actions": observation["available_actions"],
            "grid_rows": grid_rows(observation["grid"]),
            "action_results": self.results,
            "note": self.note,
        }
        with (self.ipc / "states.jsonl").open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
