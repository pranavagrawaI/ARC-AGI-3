"""Zero-ARC-action smoke for the exact pinned official Agent main loop."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from agent.my_agent import MyAgent
from tycho_nooa.models import ActionDecision


def raw_frame(
    *, value: int | None = None, state: GameState, levels_completed: int
) -> FrameDataRaw:
    raw = FrameDataRaw(
        game_id="game",
        state=state,
        levels_completed=levels_completed,
        win_levels=1,
        available_actions=[GameAction.ACTION1.value],
    )
    if value is not None:
        raw.frame = [np.array([[value]], dtype=np.int8)]
    return raw


class FakeSolver:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def decide(self, turn: Any) -> ActionDecision:
        self.events.append(("decide", turn))
        return ActionDecision(action="UP")

    def observe(self, result: Any) -> None:
        self.events.append(("observe", result))

    def finish(self, final_observation: Any, actions: int) -> None:
        self.events.append(("finish", (final_observation, actions)))

    def close(self) -> None:
        self.events.append(("close", None))


class FakeEnvironment:
    def __init__(self) -> None:
        self.observation_space = raw_frame(
            state=GameState.NOT_PLAYED,
            levels_completed=0,
        )
        self.actions: list[tuple[GameAction, Any, Any]] = []

    def step(self, action: GameAction, *, data: Any, reasoning: Any) -> FrameDataRaw:
        self.actions.append((action, data, reasoning))
        if len(self.actions) == 1:
            assert action is GameAction.RESET
            next_frame = raw_frame(
                value=1,
                state=GameState.NOT_FINISHED,
                levels_completed=0,
            )
        else:
            assert action is GameAction.ACTION1
            next_frame = raw_frame(
                value=2,
                state=GameState.WIN,
                levels_completed=1,
            )
        self.observation_space = next_frame
        return next_frame


def main() -> None:
    solver = FakeSolver()
    factory_calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> FakeSolver:
        factory_calls.append(kwargs)
        return solver

    environment = FakeEnvironment()
    agent = MyAgent(
        card_id="card",
        game_id="game",
        agent_name="myagent",
        ROOT_URL="http://gateway",
        record=False,
        arc_env=environment,
        solver_factory=factory,
        workspace_root=Path("/tmp/kaggle-agent-workspace"),
    )
    agent.main()

    assert len(factory_calls) == 1
    assert [action for action, _, _ in environment.actions] == [
        GameAction.RESET,
        GameAction.ACTION1,
    ]
    assert agent.action_counter == 2
    assert agent.frames[-1].state is GameState.WIN
    assert [kind for kind, _ in solver.events] == [
        "decide",
        "observe",
        "finish",
        "close",
    ]
    print("official_agent_pin_bootstrap_main_loop_smoke=pass")


if __name__ == "__main__":
    main()
