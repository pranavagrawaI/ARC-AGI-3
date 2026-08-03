"""PROTOTYPE facade for running fresh exact LS20 simulations in the model workspace."""

from __future__ import annotations

from typing import Iterable

from arcengine import ActionInput, GameAction
from ls20 import Ls20


def simulate(actions: Iterable[int]) -> dict[str, object]:
    game = Ls20()
    reset = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    final = reset
    frames: list[list[list[int]]] = []
    for action_id in actions:
        action = GameAction.from_id(int(action_id))
        if action.value not in (1, 2, 3, 4):
            raise ValueError("LS20 accepts only action IDs 1, 2, 3, and 4")
        final = game.perform_action(ActionInput(id=action), raw=True)
        frames.append(final.frame[-1].tolist())
        if final.levels_completed >= 1 or final.state.value in {"WIN", "GAME_OVER"}:
            break
    return {
        "levels_completed": int(final.levels_completed),
        "state": final.state.value,
        "frame": final.frame[-1].tolist(),
        "frames": frames,
        "actions_executed": len(frames),
    }
