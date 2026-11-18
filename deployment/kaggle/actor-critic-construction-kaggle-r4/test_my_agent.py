from __future__ import annotations
import importlib.util, sys
from pathlib import Path
from types import ModuleType
from typing import Any
from arcengine import FrameData, GameAction, GameState
AGENT_PATH = Path(__file__).resolve().parent / "nooa-join/adapter.py"
class StubAgent:
    MAX_ACTIONS = 80

    def __init__(self, *, card_id: str, game_id: str, agent_name: str, ROOT_URL: str, record: bool, arc_env: object, tags: list[str] | None=None) -> None:
        self.card_id = card_id
        self.game_id = game_id
        self.agent_name = agent_name
        self.ROOT_URL = ROOT_URL
        self.arc_env = arc_env
        self.tags = tags or []
        self.frames = [FrameData(levels_completed=0, win_levels=0)]
        self.action_counter = 0
        self.base_cleanup_calls = 0

    @property
    def name(self) -> str:
        return f'{self.game_id}.{type(self).__name__.lower()}'

    def cleanup(self, scorecard: object | None=None) -> None:
        self.base_cleanup_calls += 1

def load_agent_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    agents = ModuleType('agents')
    agent_module = ModuleType('agents.agent')
    setattr(agent_module, 'Agent', StubAgent)
    setattr(agents, 'agent', agent_module)
    monkeypatch.setitem(sys.modules, 'agents', agents)
    monkeypatch.setitem(sys.modules, 'agents.agent', agent_module)
    spec = importlib.util.spec_from_file_location('tested_kaggle_agent', AGENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def frame(*grids: list[list[int]], state: GameState=GameState.NOT_FINISHED, levels_completed: int=0, actions: tuple[GameAction, ...]=(GameAction.ACTION1, GameAction.ACTION6), full_reset: bool=False) -> FrameData:
    return FrameData(game_id='game', frame=list(grids), state=state, levels_completed=levels_completed, win_levels=2, available_actions=[action.value for action in actions], full_reset=full_reset)
