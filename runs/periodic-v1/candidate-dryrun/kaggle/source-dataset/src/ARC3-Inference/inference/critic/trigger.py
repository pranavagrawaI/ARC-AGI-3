"""When to invoke the critic, decided from observable Duck state only.

Duck's world model is prose, so there is no executable model to verify against
(the Tycho-side `wm_signal` trigger has no analogue here). The policy fires on:

(a) level start: once per new level, unconditionally;
(b) actor request: a `Critic:` labeled line in the actor's assistant text;
(c) stall: a run of `stall_no_change` consecutive executed actions with no board
    change, or `stall_no_progress` executed actions without a level completion.

Stall counters re-arm (reset) after any firing and after a level completion. The
no-change condition is latched: Duck batches actions, so a qualifying run that starts
and ends inside one batch must still count at the next turn start even though the
batch's last action changed the board.

`PeriodicTriggerPolicy` is the alternative scheduler for the periodic-v1 screen: no
tree, no counters, one anchor. It fires at the first turn start at which at least
`period` recorded actions (`len(game_run.history)`, automatic RESETs included) have
passed since the previous attempt, and re-anchors at the fired action count, so an
overshooting batch (18 -> 45) yields one firing at 45 with the next due at 65+. The
anchor moves in `decide()`, before any packet or model call, so a failed firing is
not retried until another `period` actions pass. Nothing fires at action 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from inference.critic.packet import TransitionRecord

CAUSE_LEVEL_START = "level_start"
CAUSE_ACTOR_REQUEST = "actor_request"
CAUSE_STALL_NO_CHANGE = "stall_no_change"
CAUSE_STALL_NO_PROGRESS = "stall_no_progress"
CAUSE_PERIODIC = "periodic"
# CAUSES enumerates the tree's causes (the gate replay tallies them); the periodic scheduler has its own.
CAUSES = (CAUSE_LEVEL_START, CAUSE_ACTOR_REQUEST, CAUSE_STALL_NO_CHANGE, CAUSE_STALL_NO_PROGRESS)

DEFAULT_STALL_NO_CHANGE = 3
DEFAULT_STALL_NO_PROGRESS = 25
DEFAULT_PERIOD = 20


@dataclass(frozen=True)
class TriggerDecision:
    fired: bool
    cause: str
    reason: str

    @classmethod
    def silent(cls, reason: str = "") -> TriggerDecision:
        return cls(fired=False, cause="", reason=reason)


class DuckTriggerPolicy:
    """Per-game trigger state machine. Feed transitions, then ask at each turn start."""

    def __init__(
        self,
        *,
        stall_no_change: int = DEFAULT_STALL_NO_CHANGE,
        stall_no_progress: int = DEFAULT_STALL_NO_PROGRESS,
    ) -> None:
        self.stall_no_change = max(1, int(stall_no_change))
        self.stall_no_progress = max(1, int(stall_no_progress))
        self._fired_levels: set[int] = set()
        self._last_observed_action_num = 0
        self._no_change_streak = 0
        self._longest_no_change_run = 0
        self._actions_since_progress = 0
        self._last_fired_action_num: int | None = None

    def observe(self, transitions: Iterable[TransitionRecord]) -> int:
        """Consume executed actions newer than any seen before; returns how many were new."""
        consumed = 0
        for record in sorted(transitions, key=lambda item: item.action_num):
            if record.action_num <= self._last_observed_action_num:
                continue
            self._last_observed_action_num = record.action_num
            consumed += 1
            if record.level_completed:
                self._rearm()
                continue
            self._actions_since_progress += 1
            if record.board_changed:
                self._no_change_streak = 0
            else:
                self._no_change_streak += 1
                self._longest_no_change_run = max(self._longest_no_change_run, self._no_change_streak)
        return consumed

    def _rearm(self) -> None:
        self._no_change_streak = 0
        self._longest_no_change_run = 0
        self._actions_since_progress = 0

    def decide(self, *, level: int, action_num: int, critic_request: str = "") -> TriggerDecision:
        """Evaluate at a turn start (before Duck builds its user prompt)."""
        decision = self._evaluate(level=level, action_num=action_num, critic_request=critic_request)
        if decision.fired:
            self._fired_levels.add(int(level))
            self._last_fired_action_num = int(action_num)
            self._rearm()
        return decision

    def _evaluate(self, *, level: int, action_num: int, critic_request: str) -> TriggerDecision:
        if int(level) not in self._fired_levels:
            return TriggerDecision(True, CAUSE_LEVEL_START, f"first turn observed on level {int(level)} (action {int(action_num)})")
        request = " ".join(str(critic_request or "").split())
        if request:
            return TriggerDecision(True, CAUSE_ACTOR_REQUEST, f"actor wrote `Critic:` {request}")
        if self._longest_no_change_run >= self.stall_no_change:
            return TriggerDecision(
                True,
                CAUSE_STALL_NO_CHANGE,
                f"a run of {self._longest_no_change_run} consecutive executed actions without a board change "
                f"since the last critic call (threshold {self.stall_no_change})",
            )
        if self._actions_since_progress >= self.stall_no_progress:
            return TriggerDecision(
                True,
                CAUSE_STALL_NO_PROGRESS,
                f"{self._actions_since_progress} executed actions without level progress (threshold {self.stall_no_progress})",
            )
        return TriggerDecision.silent(
            f"longest no-change run {self._longest_no_change_run}/{self.stall_no_change}, "
            f"since progress {self._actions_since_progress}/{self.stall_no_progress}"
        )

    @property
    def state(self) -> dict[str, int | None | list[int]]:
        return {
            "fired_levels": sorted(self._fired_levels),
            "last_observed_action_num": self._last_observed_action_num,
            "no_change_streak": self._no_change_streak,
            "longest_no_change_run": self._longest_no_change_run,
            "actions_since_progress": self._actions_since_progress,
            "last_fired_action_num": self._last_fired_action_num,
        }


class PeriodicTriggerPolicy:
    """Minimum-spacing scheduler: fire when `period` actions have passed since the last attempt."""

    def __init__(self, *, period: int = DEFAULT_PERIOD) -> None:
        self.period = max(1, int(period))
        self._anchor = 0
        self._last_fired_action_num: int | None = None
        self._attempts = 0

    def observe(self, transitions: Iterable[TransitionRecord]) -> int:
        """Transitions carry nothing this scheduler uses; the clock is `action_num` at `decide()`."""
        return sum(1 for _ in transitions)

    def decide(self, *, level: int, action_num: int, critic_request: str = "") -> TriggerDecision:
        """Evaluate at a turn start. Actor requests and level changes are ignored by design."""
        elapsed = int(action_num) - self._anchor
        if elapsed < self.period:
            return TriggerDecision.silent(f"{elapsed}/{self.period} actions since the last critic attempt")
        self._anchor = int(action_num)
        self._last_fired_action_num = int(action_num)
        self._attempts += 1
        return TriggerDecision(
            True,
            CAUSE_PERIODIC,
            f"{elapsed} recorded actions since the last critic attempt (period {self.period}); anchor now {self._anchor}",
        )

    @property
    def state(self) -> dict[str, int | None | list[int]]:
        return {
            "period": self.period,
            "anchor": self._anchor,
            "attempts": self._attempts,
            "last_fired_action_num": self._last_fired_action_num,
        }


__all__ = [
    "CAUSES",
    "CAUSE_ACTOR_REQUEST",
    "CAUSE_LEVEL_START",
    "CAUSE_PERIODIC",
    "CAUSE_STALL_NO_CHANGE",
    "CAUSE_STALL_NO_PROGRESS",
    "DEFAULT_PERIOD",
    "DEFAULT_STALL_NO_CHANGE",
    "DEFAULT_STALL_NO_PROGRESS",
    "DuckTriggerPolicy",
    "PeriodicTriggerPolicy",
    "TriggerDecision",
]
