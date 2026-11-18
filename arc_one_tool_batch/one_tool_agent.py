"""Adapter for the official ARC-AGI-3-Agents Swarm.

Copy this and one_tool_runtime.py to agents/templates, then register OneTool.
Only uses the environment already created by Swarm. Never makes another game.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from arcengine import GameAction, GameState
from ..agent import Agent
from .one_tool_runtime import Action, Config, VLLM, play_game

_LOCK = threading.Lock()
_SHARED: tuple[Config, VLLM] | None = None


def shared() -> tuple[Config, VLLM]:
    global _SHARED
    with _LOCK:  # initialization only; never held across inference or game actions
        if _SHARED is None:
            config = Config.from_env()
            _SHARED = config, VLLM(config)
        return _SHARED


class _Environment:
    def __init__(self, owner: "OneTool"):
        self.owner = owner

    def observe(self) -> Any:
        return self.owner.arc_env.observation_space

    def step(self, action: Action) -> Any:
        # Do not mutate GameAction enum instances with set_data/reasoning:
        # enum instances are shared across threads. Data remains request-local.
        raw = self.owner.arc_env.step(GameAction[action.name], data=action.data())
        self.owner.action_counter += 1
        if raw is not None:
            self.owner.append_frame(self.owner._convert_raw_frame_data(raw))
        return raw


class OneTool(Agent):
    MAX_ACTIONS = 10000

    def main(self) -> None:
        self.timer = time.time()
        config, backend = shared()
        self.MAX_ACTIONS = config.max_game_actions
        try:
            self.result = play_game(_Environment(self), self.game_id, backend, config)
            logging.getLogger(__name__).warning("OneTool %s: %s", self.game_id, self.result)
        finally:
            backend.close_thread()
            self.cleanup()

    # Satisfy Agent's abstract interface. main above owns the complete tool loop.
    def is_done(self, frames: Any, latest_frame: Any) -> bool:
        return latest_frame.state == GameState.WIN

    def choose_action(self, frames: Any, latest_frame: Any) -> GameAction:
        raise RuntimeError("OneTool.main owns its sequential action loop")
