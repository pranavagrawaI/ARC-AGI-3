"""Run a contestant using only the public ARC interaction surface."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np
from arcengine import GameAction

from .types import Agent, Observation, RunResult, TraceSink, Transition


TERMINAL_STATES = frozenset({"WIN", "GAME_OVER"})


class Environment(Protocol):
    """The two environment operations required by the harness."""

    def reset(self) -> Any: ...

    def step(
        self,
        action: GameAction,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> Any: ...


class IllegalActionError(ValueError):
    pass


def public_observation(frame) -> Observation:
    """Copy a raw ARC frame into the only representation an agent receives."""

    pixels = np.array(frame.frame, copy=True)
    pixels.setflags(write=False)
    state = frame.state.value if hasattr(frame.state, "value") else str(frame.state)
    return Observation(
        pixels=pixels,
        available_actions=tuple(int(action) for action in frame.available_actions),
        levels_completed=int(frame.levels_completed),
        state=state,
        full_reset=bool(frame.full_reset),
    )


def run_episode(
    env: Environment,
    agent: Agent,
    *,
    max_actions: int,
    trace: TraceSink | None = None,
    level_action_budgets: tuple[int, ...] | None = None,
    stop_after_levels: int | None = None,
    stop_on_exceptions: Mapping[type[BaseException], str] | None = None,
) -> RunResult:
    """Run one reset-to-terminal interaction episode.

    The environment remains local to this function. The agent sees copied public
    observations, and every chosen action is checked against the current public
    action set before it reaches the environment.
    """

    if max_actions < 0:
        raise ValueError("max_actions must be non-negative")
    if level_action_budgets is not None:
        if not level_action_budgets or any(limit < 0 for limit in level_action_budgets):
            raise ValueError("level action budgets must be non-empty and non-negative")
        if stop_after_levels is None:
            stop_after_levels = len(level_action_budgets)
        if len(level_action_budgets) != stop_after_levels:
            raise ValueError("provide exactly one action budget per included level")
    if stop_after_levels is not None and stop_after_levels < 1:
        raise ValueError("stop_after_levels must be positive")

    initial = public_observation(env.reset())
    current = initial
    transitions: list[Transition] = []
    level_action_counts = [0] * (stop_after_levels or 0)
    completion_action_counts: list[int] = []
    reset_count = 0
    game_over_count = 0
    last_levels_completed = initial.levels_completed
    if trace is not None:
        trace.start(initial)

    while True:
        if current.state in TERMINAL_STATES:
            stop_reason = "terminal"
            break
        if not current.available_actions:
            stop_reason = "no_available_actions"
            break
        active_level = current.levels_completed - initial.levels_completed
        if stop_after_levels is not None and active_level >= stop_after_levels:
            stop_reason = "level_limit"
            break
        if (
            level_action_budgets is not None
            and level_action_counts[active_level] >= level_action_budgets[active_level]
        ):
            stop_reason = "level_action_budget"
            break
        if len(transitions) >= max_actions:
            stop_reason = "action_budget"
            break

        try:
            decision = agent.act(current)
        except BaseException as error:
            matched_reason = next(
                (
                    reason
                    for error_type, reason in (stop_on_exceptions or {}).items()
                    if isinstance(error, error_type)
                ),
                None,
            )
            if matched_reason is None:
                raise
            stop_reason = matched_reason
            break
        if decision.action not in current.available_actions:
            raise IllegalActionError(
                f"agent chose action {decision.action}; "
                f"available actions are {current.available_actions}"
            )

        try:
            game_action = GameAction.from_id(decision.action)
        except ValueError as error:
            raise IllegalActionError(
                f"unknown ARC action id {decision.action}"
            ) from error
        action_data = dict(decision.data) if decision.data is not None else None
        reasoning = dict(decision.reasoning) if decision.reasoning is not None else None
        if game_action.is_complex() and action_data is None:
            raise IllegalActionError(
                f"action {decision.action} requires action data"
            )
        if action_data is not None:
            try:
                game_action.validate_data(action_data)
            except ValueError as error:
                raise IllegalActionError(
                    f"invalid data for action {decision.action}: {error}"
                ) from error

        next_frame = env.step(game_action, data=action_data, reasoning=reasoning)
        if next_frame is None:
            raise RuntimeError("environment step returned no frame")
        if stop_after_levels is not None:
            level_action_counts[active_level] += 1
        raw_levels_completed = int(next_frame.levels_completed)
        attempted_action_count = len(transitions) + 1
        if raw_levels_completed > last_levels_completed:
            completion_action_counts.extend(
                [attempted_action_count]
                * (raw_levels_completed - last_levels_completed)
            )
            last_levels_completed = raw_levels_completed
        completed_in_run = raw_levels_completed - initial.levels_completed
        if stop_after_levels is not None and completed_in_run >= stop_after_levels:
            following = Observation(
                pixels=current.pixels,
                available_actions=(),
                levels_completed=raw_levels_completed,
                state=current.state,
                full_reset=current.full_reset,
            )
            transition = Transition(
                current,
                decision,
                following,
                observation_suppressed=True,
            )
            transitions.append(transition)
            if trace is not None:
                trace.append_suppressed(len(transitions), transition)
            current = following
            stop_reason = "level_limit"
            break

        raw_state = (
            next_frame.state.value
            if hasattr(next_frame.state, "value")
            else str(next_frame.state)
        )
        if bool(next_frame.full_reset):
            reset_count += 1
        if raw_state == "GAME_OVER":
            game_over_count += 1
        following = public_observation(next_frame)
        transition = Transition(current, decision, following)
        agent.observe(transition)
        transitions.append(transition)
        if trace is not None:
            trace.append(len(transitions), transition)
        current = following

    result = RunResult(
        initial=initial,
        transitions=tuple(transitions),
        final=current,
        stop_reason=stop_reason,
        level_action_counts=tuple(level_action_counts),
        completion_action_counts=tuple(completion_action_counts),
        reset_count=reset_count,
        game_over_count=game_over_count,
    )
    if trace is not None:
        trace.finish(result)
    return result
