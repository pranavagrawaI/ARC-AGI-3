"""ARC-AGI-3 competition adapter for one persistent local Solver per game.

The official framework owns environment execution. This adapter only projects
public frames into the Tycho-NOOA protocol, reconciles each chosen action with
its exact successor, and maps the next validated decision back to ``GameAction``.
"""
from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent  # type: ignore[import-not-found]

from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.protocol import (
    PROTOCOL_VERSION,
    ActionResultMessage,
    ObservationPayload,
    arc_action_payload,
    project_arc_observation,
)

WORKSPACE_ROOT_ENV = "TYCHO_KAGGLE_WORKSPACE_ROOT"


class SolverPort(Protocol):
    """Synchronous boundary owned by the later local-runtime wrapper."""

    def decide(self, turn: GameTurn) -> ActionDecision: ...

    def observe(self, result: ActionResultMessage) -> None: ...

    def finish(self, final_observation: ObservationPayload, actions: int) -> None: ...

    def close(self) -> None: ...


class SolverFactory(Protocol):
    def __call__(
        self,
        *,
        game_id: str,
        workspace_root: Path,
        initial_observation: ObservationPayload,
    ) -> SolverPort: ...


def _max_actions_from_env() -> int:
    raw = os.environ.get("TYCHO_KAGGLE_MAX_ACTIONS", "80")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("TYCHO_KAGGLE_MAX_ACTIONS must be a positive integer") from error
    if value <= 0:
        raise ValueError("TYCHO_KAGGLE_MAX_ACTIONS must be a positive integer")
    return value


def _workspace_root_from_env() -> Path:
    raw = os.environ.get(WORKSPACE_ROOT_ENV)
    if not raw:
        raise RuntimeError(
            f"{WORKSPACE_ROOT_ENV} must bind the launcher's Solver evidence root"
        )
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{WORKSPACE_ROOT_ENV} must be an absolute path")
    return path


def _default_solver_factory(
    *,
    game_id: str,
    workspace_root: Path,
    initial_observation: ObservationPayload,
) -> SolverPort:
    try:
        module = import_module("local_solver")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "local Solver is not bundled; deployment must provide "
            "local_solver.create_solver"
        ) from error
    create_solver = cast(Any, getattr(module, "create_solver"))
    return create_solver(
        game_id=game_id,
        workspace_root=workspace_root,
        initial_observation=initial_observation,
    )


class MyAgent(Agent):
    """Translate the official Agent loop into one persistent local Solver."""

    MAX_ACTIONS = 80

    def __init__(
        self,
        *args: Any,
        solver_factory: SolverFactory | None = None,
        workspace_root: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self._solver_action_limit = _max_actions_from_env()
        # The official counter includes bootstrap and GAME_OVER RESET actions.
        # One reset can precede every Solver decision, plus the initial reset,
        # so keep that safety backstop outside the exact Solver-decision cap.
        # ``is_done`` remains the normal stop condition.
        self.MAX_ACTIONS = 2 * self._solver_action_limit + 1
        super().__init__(*args, **kwargs)
        self._solver_factory = solver_factory or _default_solver_factory
        self._workspace_root = (
            _workspace_root_from_env()
            if workspace_root is None
            else Path(workspace_root)
        )
        self._solver: SolverPort | None = None
        self._last_observation: ObservationPayload | None = None
        self._pending_before: ObservationPayload | None = None
        self._pending_decision: ActionDecision | None = None
        self._pending_sequence: int | None = None
        self._pending_level: int | None = None
        self._pending_turn_in_level: int | None = None
        self._sequence = 0
        self._level = 0
        self._turn_in_level = 0
        self._previous_decision: ActionDecision | None = None
        self._solver_decisions_emitted = 0
        self._finished = False
        self._closed = False

    @property
    def name(self) -> str:
        return f"{super().name}.{self._solver_action_limit}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if self._pending_decision is not None:
            self._reconcile_pending(latest_frame)
        return (
            latest_frame.state is GameState.WIN
            or self._solver_decisions_emitted >= self._solver_action_limit
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if (
            latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER)
            and self._pending_decision is None
        ):
            if (
                latest_frame.state is GameState.GAME_OVER
                and self._solver is not None
            ):
                self._end_solver_episode(
                    self._last_observation
                    or _project_frame(latest_frame, previous_levels_completed=None)
                )
            return GameAction.RESET

        observation = self._reconcile_pending(latest_frame)
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state is GameState.GAME_OVER:
                self._end_solver_episode(observation)
            return GameAction.RESET

        solver = self._ensure_solver(observation)
        actions_used = self._solver_decisions_emitted
        actions_remaining = self._solver_action_limit - actions_used
        turn = GameTurn(
            grid=observation.grid,
            available_actions=observation.available_actions,
            sequence=self._sequence,
            action_limit=self._solver_action_limit,
            actions_used=actions_used,
            actions_remaining=actions_remaining,
            final_permitted_action=actions_remaining == 1,
            level=self._level,
            turn_in_level=self._turn_in_level,
            levels_completed=observation.levels_completed,
            state=observation.state,
            previous_action=(
                self._previous_decision.action if self._previous_decision else ""
            ),
            previous_row=(
                self._previous_decision.row if self._previous_decision else None
            ),
            previous_col=(
                self._previous_decision.col if self._previous_decision else None
            ),
        )
        decision = solver.decide(turn)
        if decision.action not in observation.available_actions:
            raise ValueError(
                f"Solver action {decision.action!r} is not available; "
                f"available actions are {observation.available_actions}"
            )
        action_id, data = arc_action_payload(decision)
        action = GameAction.from_id(action_id)
        if data is not None:
            action.set_data(data)
        self._pending_before = observation
        self._pending_decision = decision
        self._pending_sequence = self._sequence
        self._pending_level = self._level
        self._pending_turn_in_level = self._turn_in_level
        self._solver_decisions_emitted += 1
        return action

    def cleanup(self, scorecard: Any | None = None) -> None:
        try:
            try:
                if self._pending_decision is not None and self.frames:
                    self._reconcile_pending(self.frames[-1])
                if self._solver is not None and not self._finished:
                    if self._last_observation is None:
                        raise RuntimeError("local Solver has no final public observation")
                    self._solver.finish(self._last_observation, self.action_counter)
                    self._finished = True
            finally:
                if self._solver is not None and not self._closed:
                    self._solver.close()
                    self._closed = True
        finally:
            super().cleanup(scorecard)

    def _ensure_solver(self, observation: ObservationPayload) -> SolverPort:
        if self._solver is None:
            self._level = observation.levels_completed
            self._sequence = 0
            self._turn_in_level = 0
            self._previous_decision = None
            self._solver_decisions_emitted = 0
            self._solver = self._solver_factory(
                game_id=self.game_id,
                workspace_root=self._workspace_root,
                initial_observation=observation,
            )
        return self._solver

    def _end_solver_episode(self, observation: ObservationPayload) -> None:
        """GAME_OVER RESET is not a Solver decision. Close this worker so the
        next NOT_FINISHED frame cannot hit LocalSolver continuity against the
        dead grid (depth v1, s5i5 frame 49 → 50)."""
        solver = self._solver
        if solver is None:
            return
        try:
            if not self._finished:
                solver.finish(
                    observation,
                    max(self.action_counter, self._sequence, 1),
                )
                self._finished = True
        finally:
            if not self._closed:
                solver.close()
                self._closed = True
            self._solver = None
            self._finished = False
            self._closed = False
            self._sequence = 0
            self._turn_in_level = 0
            self._previous_decision = None
            self._clear_pending()

    def _reconcile_pending(self, latest_frame: FrameData) -> ObservationPayload:
        previous_levels = (
            self._pending_before.levels_completed
            if self._pending_before is not None
            else None
        )
        observation = _project_frame(
            latest_frame,
            previous_levels_completed=previous_levels,
        )
        self._last_observation = observation
        if self._pending_decision is None:
            return observation
        if self._solver is None or self._pending_before is None:
            raise RuntimeError("pending Solver decision has no owning Solver state")
        if (
            self._pending_sequence is None
            or self._pending_level is None
            or self._pending_turn_in_level is None
        ):
            raise RuntimeError("pending Solver decision is missing turn identity")
        decision = self._pending_decision
        self._solver.observe(
            ActionResultMessage(
                protocol_version=PROTOCOL_VERSION,
                type="action_result",
                sequence=self._pending_sequence,
                level=self._pending_level,
                turn_in_level=self._pending_turn_in_level,
                decision=decision,
                before=self._pending_before,
                after=observation,
            )
        )
        self._advance_turn(self._pending_before, observation, decision)
        self._clear_pending()
        return observation

    def _advance_turn(
        self,
        before: ObservationPayload,
        after: ObservationPayload,
        decision: ActionDecision,
    ) -> None:
        self._sequence += 1
        if after.levels_completed > before.levels_completed:
            self._level = after.levels_completed
            self._turn_in_level = 0
            self._previous_decision = None
        elif after.full_reset:
            self._turn_in_level = 0
            self._previous_decision = None
        else:
            self._turn_in_level += 1
            self._previous_decision = decision

    def _clear_pending(self) -> None:
        self._pending_before = None
        self._pending_decision = None
        self._pending_sequence = None
        self._pending_level = None
        self._pending_turn_in_level = None


def _project_frame(
    frame: FrameData,
    *,
    previous_levels_completed: int | None = None,
) -> ObservationPayload:
    state = frame.state.value if hasattr(frame.state, "value") else str(frame.state)
    return project_arc_observation(
        frames=frame.frame,
        available_action_ids=list(frame.available_actions),
        state=state,
        levels_completed=frame.levels_completed,
        full_reset=frame.full_reset,
        previous_levels_completed=previous_levels_completed,
    )
