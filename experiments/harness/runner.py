"""Run a contestant using only the public ARC interaction surface."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from arcengine import GameAction

from .types import Agent, Observation, RunResult, TraceSink, Transition


TERMINAL_STATES = frozenset({"WIN", "GAME_OVER"})


class Environment(Protocol):
    """The two environment operations required by the harness."""

    def reset(self): ...

    def step(
        self,
        action: GameAction,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ): ...


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
) -> RunResult:
    """Run one reset-to-terminal interaction episode.

    The environment remains local to this function. The agent sees copied public
    observations, and every chosen action is checked against the current public
    action set before it reaches the environment.
    """

    if max_actions < 0:
        raise ValueError("max_actions must be non-negative")

    initial = public_observation(env.reset())
    current = initial
    transitions: list[Transition] = []
    if trace is not None:
        trace.start(initial)

    while True:
        if current.state in TERMINAL_STATES:
            stop_reason = "terminal"
            break
        if not current.available_actions:
            stop_reason = "no_available_actions"
            break
        if len(transitions) >= max_actions:
            stop_reason = "action_budget"
            break

        decision = agent.act(current)
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
    )
    if trace is not None:
        trace.finish(result)
    return result
