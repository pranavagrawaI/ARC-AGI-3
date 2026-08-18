"""Run one persistent solver against one ARC environment."""

from __future__ import annotations

from pathlib import Path

from arc_host.actions import game_action
from arc_host.frames import public_observation
from arc_host.types import (
    Environment,
    EpisodeResult,
    GameStart,
    SolverPort,
    Transition,
    Turn,
)

TERMINAL_STATES = frozenset({"WIN", "GAME_OVER"})


class IllegalActionError(ValueError):
    pass


def run_episode(
    environment: Environment,
    solver: SolverPort,
    *,
    game_id: str,
    seed: int,
    workspace_root: Path,
    max_actions: int,
    stop_after_levels: int | None = None,
) -> EpisodeResult:
    """Execute a reset-to-boundary episode through the narrow solver port."""
    if max_actions < 0:
        raise ValueError("max_actions must be non-negative")
    if stop_after_levels is not None and stop_after_levels < 1:
        raise ValueError("stop_after_levels must be positive")

    frame = environment.reset()
    if frame is None:
        raise RuntimeError("ARC environment reset returned no public frame")
    initial = public_observation(frame)
    solver.start_game(
        GameStart(
            game_id=game_id,
            seed=seed,
            workspace_root=workspace_root,
            initial_observation=initial,
        )
    )

    current = initial
    transitions: list[Transition] = []
    initial_levels = initial.levels_completed
    level = initial_levels
    turn_in_level = 0
    previous_decision = None

    while True:
        if current.state.upper() in TERMINAL_STATES:
            stop_reason = "terminal"
            break
        if (
            stop_after_levels is not None
            and current.levels_completed - initial_levels >= stop_after_levels
        ):
            stop_reason = "level_limit"
            break
        if not current.available_actions:
            stop_reason = "no_available_actions"
            break
        if len(transitions) >= max_actions:
            stop_reason = "action_budget"
            break

        turn = Turn(
            sequence=len(transitions),
            level=level,
            turn_in_level=turn_in_level,
            observation=current,
            previous_decision=previous_decision,
        )
        decision = solver.decide(turn)
        if decision.action not in current.available_actions:
            raise IllegalActionError(
                f"solver chose {decision.action!r}; available actions are "
                f"{current.available_actions}"
            )
        action, data = game_action(decision)
        next_frame = environment.step(action, data=data)
        if next_frame is None:
            raise RuntimeError("ARC environment step returned no public frame")
        following = public_observation(next_frame)
        transition = Transition(turn=turn, decision=decision, after=following)
        solver.observe(transition)
        transitions.append(transition)

        if following.levels_completed > current.levels_completed:
            level = following.levels_completed
            turn_in_level = 0
            previous_decision = None
        elif following.full_reset:
            turn_in_level = 0
            previous_decision = None
        else:
            turn_in_level += 1
            previous_decision = decision
        current = following

    result = EpisodeResult(
        initial=initial,
        transitions=tuple(transitions),
        final=current,
        stop_reason=stop_reason,
    )
    solver.finish(result)
    return result
