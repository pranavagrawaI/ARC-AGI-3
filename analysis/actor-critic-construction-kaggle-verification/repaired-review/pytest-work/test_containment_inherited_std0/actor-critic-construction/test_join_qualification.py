from __future__ import annotations
import importlib.util, sys
from pathlib import Path
from types import ModuleType
from typing import Any
from arcengine import FrameData, GameAction, GameState
from test_my_agent import load_agent_module
from test_nooa_join import load_create_solver
def build_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, first: FrameData) -> Any:
    module = load_agent_module(monkeypatch)
    factory = load_create_solver()
    agent = module.MyAgent(card_id='card', game_id='ls20', agent_name='myagent', ROOT_URL='http://example.test', record=False, arc_env=object(), solver_factory=factory.create_solver, workspace_root=tmp_path / 'workspaces')
    agent.frames = [first]
    agent.action_counter = 1
    return agent
