"""Oracle-hint overlay for the dfranzen control notebook. Installed after the harness is importable.

install(bm, attempts, env_dir) replaces bm.games with one StartGame per attempt:
  - the game is replayed through the real engine to the attempt's level entry before the agent sees it;
  - the attempt ends (engine state forced to WIN) the moment that level is completed, so a sprint is one level;
  - attempt["hint"], when non-empty, is appended to that game's ToolAgent system prompt.
Everything else is the unchanged control configuration.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import arcengine
import taaf.game_api
import inference.framework.solver as _solver

HINT_HEADER = "\n\n## Note about this level\n"
EVENTS = Path(os.environ.get("ORACLE_EVENTS", "oracle_events.jsonl"))
_lock = threading.Lock()


def event(name, kind, **kw):
    """Append-only, flushed per line: survives a killed session. Games are copied by the Benchmark, so state lives here."""
    row = dict(t=time.time(), mono=time.monotonic(), attempt=name, kind=kind, **kw)
    with _lock, EVENTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


@dataclass
class StartGame(taaf.game_api.GameAPI):
    prefix: list = field(default_factory=list, kw_only=True)
    start_level: int = field(default=0, kw_only=True)
    hint_text: str = field(default="", kw_only=True)

    def _start_game(self, session):
        state = super()._start_game(session)
        for a in self.prefix:
            state = super()._execute_action(
                arcengine.ActionInput(id=arcengine.GameAction[a["id"]], data=dict(a.get("data") or {})))
        if state.levels_completed != self.start_level or state.raw.state == arcengine.GameState.GAME_OVER:
            event(self.external_game_id, "prefix_failed", level=state.levels_completed, state=state.raw.state.name)
            raise RuntimeError(f"{self.external_game_id}: prefix landed on level {state.levels_completed} "
                               f"state {state.raw.state.name}, expected level {self.start_level}")
        event(self.external_game_id, "landed", level=state.levels_completed, prefix_len=len(self.prefix))
        return state

    def _execute_action(self, action):
        state = super()._execute_action(action)
        if state.levels_completed > self.start_level and state.raw.state != arcengine.GameState.WIN:
            event(self.external_game_id, "solved", level=state.levels_completed)
            state.raw.state = arcengine.GameState.WIN  # one level per sprint: end the attempt here
        return state


_original_make_analyzer = _solver.HarnessSolver._make_analyzer


def _make_analyzer(self, game, index, local_server=None):
    analyzer = _original_make_analyzer(self, game, index, local_server)
    hint = getattr(game, "hint_text", "")
    if hint:
        analyzer._system_prompt = analyzer._system_prompt + HINT_HEADER + hint.strip() + "\n"
    event(getattr(game, "external_game_id", "?"), "admitted", hinted=bool(hint),
          system_prompt_tail=analyzer._system_prompt[-300:])
    return analyzer


def install(bm, attempts, env_dir):
    import arc_agi
    _solver.HarnessSolver._make_analyzer = _make_analyzer
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(env_dir))
    bm.games = [StartGame(env_name=a["gid"], arcade_spec=spec, external_game_id=a["name"],
                          prefix=a["prefix"], start_level=a["level"], hint_text=a.get("hint", ""))
                for a in attempts]
    return bm.games


def load_attempts(path):
    return json.loads(Path(path).read_text())
