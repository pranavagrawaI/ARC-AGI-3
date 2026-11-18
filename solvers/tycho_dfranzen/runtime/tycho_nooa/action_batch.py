# Modified for the Tycho/dfranzen candidate; see canonical-candidate.patch.
"""A bounded model-authored action queue released through ordinary host turns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.protocol import ActionResultMessage

MAX_ACTIONS_PER_BATCH = 4


def grid_sha256(grid: list[list[int]]) -> str:
    """Hash the documented compact-JSON representation of one public grid."""
    payload = json.dumps(grid, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ActionBatchStep(ActionDecision):
    """One action plus an optional exact precondition for releasing it."""

    expected_before_grid_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    def decision(self) -> ActionDecision:
        return ActionDecision(action=self.action, row=self.row, col=self.col)


class ActionBatchDecision(BaseModel):
    """One model commitment that may span several observed environment turns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: tuple[ActionBatchStep, ...] = Field(
        min_length=1,
        max_length=MAX_ACTIONS_PER_BATCH,
    )
    rationale: str = Field(min_length=1, max_length=2_000)
    stop_on_no_change: bool = True

    def identity(self) -> str:
        payload = self.model_dump_json(exclude_none=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass
class PendingActionBatch:
    """Host-owned cursor over a validated model-authored batch."""

    batch: ActionBatchDecision
    anchor_sequence: int | None
    anchor_level: int
    anchor_turn_in_level: int
    next_offset: int = 1
    model_calls_avoided: int = 0

    @property
    def batch_id(self) -> str:
        return self.batch.identity()

    @property
    def exhausted(self) -> bool:
        return self.next_offset >= len(self.batch.actions)

    @property
    def previous_action(self) -> ActionDecision:
        return self.batch.actions[self.next_offset - 1].decision()

    def release(
        self,
        turn: GameTurn,
        previous_result: ActionResultMessage | None,
    ) -> tuple[ActionDecision | None, str | None]:
        """Release the next action or return the first failed public guard."""
        if self.exhausted:
            return None, "batch_exhausted"
        failure = self._release_failure(turn, previous_result)
        if failure is not None:
            return None, failure

        step = self.batch.actions[self.next_offset]
        decision = step.decision()
        self.next_offset += 1
        self.model_calls_avoided += 1
        return decision, None

    def _release_failure(
        self,
        turn: GameTurn,
        previous_result: ActionResultMessage | None,
    ) -> str | None:
        checks = (
            self._previous_result_failure(turn, previous_result),
            self._boundary_failure(previous_result),
            self._continuity_failure(turn),
            self._step_failure(turn),
        )
        return next((failure for failure in checks if failure is not None), None)

    def _previous_result_failure(
        self,
        turn: GameTurn,
        previous_result: ActionResultMessage | None,
    ) -> str | None:
        if previous_result is None:
            return "previous_action_result_missing"
        if previous_result.decision != self.previous_action:
            return "previous_action_result_mismatch"
        if previous_result.after.grid != turn.grid:
            return "successor_observation_mismatch"
        if (
            self.batch.stop_on_no_change
            and not __import__('hybrid.policy', fromlist=['interior_changed']).interior_changed(
                previous_result.before.grid, previous_result.after.grid)
        ):
            return "previous_action_no_change"
        return None

    @staticmethod
    def _boundary_failure(previous_result: ActionResultMessage | None) -> str | None:
        if previous_result is None:
            return None
        if previous_result.after.full_reset:
            return "public_full_reset"
        if previous_result.after.levels_completed > previous_result.before.levels_completed:
            return "level_advanced"
        if previous_result.after.state.upper() != "NOT_FINISHED":
            return f"terminal_state_{previous_result.after.state.lower()}"
        return None

    def _continuity_failure(self, turn: GameTurn) -> str | None:
        if turn.level != self.anchor_level:
            return "level_changed"
        if turn.turn_in_level != self.anchor_turn_in_level + self.next_offset:
            return "turn_offset_mismatch"
        if (
            self.anchor_sequence is not None
            and turn.sequence != self.anchor_sequence + self.next_offset
        ):
            return "sequence_offset_mismatch"
        if turn.actions_remaining is not None and turn.actions_remaining <= 0:
            return "action_horizon_exhausted"
        return None

    def _step_failure(self, turn: GameTurn) -> str | None:
        step = self.batch.actions[self.next_offset]
        if step.action not in turn.available_actions:
            return "next_action_unavailable"
        expected = step.expected_before_grid_sha256
        if expected is not None and expected != grid_sha256(turn.grid):
            return "expected_frame_mismatch"
        return None
