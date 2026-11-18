"""Baseline R: one Retrodict run per game on the shared bridge, clock and request layer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from qwen_baselines.executor import Executor

from . import prompts
from .agent import QwenAgentClient
from .runner import GameRunner, RunnerConfig
from .search import SearchTool
from .session import ChatSession
from .thin.filesystem import FileTools
from .tools import PythonTool, lenient_edit_spec

ARCLOG = Path(__file__).with_name("arclog.py")


@dataclass(frozen=True)
class GameStatus:
    value: str


@dataclass
class Frame:
    """The env frame shape Retrodict's runner reads (arcengine's FrameData fields)."""

    frame: list
    levels_completed: int
    win_levels: int
    state: GameStatus
    available_actions: list
    full_reset: bool = False


class BridgeEnv:
    """Retrodict's ``GameEnv`` over ``qwen_baselines.bridge.Bridge``."""

    def __init__(self, bridge):
        self.bridge = bridge

    def _frame(self, meta):
        archive = self.bridge.archive
        frames = [archive.frame(meta["index"], i) for i in range(meta["frame_count"])]
        return Frame(
            frame=frames,
            levels_completed=meta["levels_completed"],
            win_levels=meta["win_levels"],
            state=GameStatus(meta["status"]),
            available_actions=[
                0 if name == "RESET" else int(name.removeprefix("ACTION"))
                for name in meta["legal_actions"]
            ],
            full_reset=bool(meta.get("full_reset", False)),
        )

    async def current(self):
        return self._frame(self.bridge.archive.metadata())

    async def reset(self):
        try:
            return self._frame(await self.bridge.step("RESET", automatic=True))
        except (TimeoutError, asyncio.CancelledError):
            raise
        except Exception as exc:
            self.bridge.archive.event("retrodict_env_error", action="RESET", error=str(exc))
            return None

    async def step(self, action, data=None):
        data = data or {}
        try:
            meta = await self.bridge.step(action, data.get("x"), data.get("y"))
        except (TimeoutError, asyncio.CancelledError):
            raise
        except Exception as exc:
            self.bridge.archive.event("retrodict_env_error", action=action, error=str(exc))
            return None
        return self._frame(meta)


class RetrodictGame:
    def __init__(self, bridge, layer, config):
        self.bridge, self.layer, self.config = bridge, layer, config
        root = bridge.archive.root
        self.workspace = root / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        # Upstream's workspace_template: arclog.py and an importable scratch/ package.
        (self.workspace / "arclog.py").write_text(ARCLOG.read_text())
        (self.workspace / "scratch").mkdir(exist_ok=True)
        (self.workspace / "scratch" / "__init__.py").touch()
        self.executor = Executor(
            bridge,
            self.workspace,
            api=False,
            evidence=False,
            scientific=config.scientific_python,
            bash_seconds=config.bash_seconds,
            bash_max_seconds=config.bash_max_seconds,
        )
        files = FileTools(self.workspace)
        specs = [
            PythonTool(self.executor).spec(),
            *(lenient_edit_spec(files) if spec.name == "edit" else spec for spec in files.specs()),
            SearchTool(files).spec(),
        ]
        session = ChatSession(
            layer,
            bridge.archive,
            prompts.system_prompt(
                gameplay_seconds=config.gameplay_seconds,
                budget_seconds=config.invocation_budget_seconds,
                requests=config.call_turns,
            ),
            specs,
            max_model_requests=config.call_turns,
            budget_seconds=config.invocation_budget_seconds,
            max_tokens=config.output_tokens_cap or None,
            clock=bridge.clock,
        )
        self.runner_config = RunnerConfig(
            game_id=bridge.label.split()[-1] if bridge.label else "game",
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            max_output_tokens=config.output_tokens_cap or 32_768,
            action_cap=config.max_actions or 2000,
            # v5.2: the fresh-session check runs between invocations only, so the context
            # may grow by an invocation's replies past the cap (v5.1 reached 80K on a 64K
            # cap and the three games preempted the pool 63 times). The threshold leaves
            # one reply of headroom below the cap.
            fresh_session_input_tokens=(
                config.compaction_tokens or config.context_window - config.reserve_tokens
            )
            - (config.output_tokens_cap or 0),
            max_model_requests=config.call_turns,
            request_timeout=int(config.gameplay_seconds),
            gameplay_seconds=config.gameplay_seconds,
        )
        self.runner = GameRunner(
            BridgeEnv(bridge),
            QwenAgentClient(session),
            self.runner_config,
            root,
            clock=bridge.clock,
            archive=bridge.archive,
        )

    async def run(self):
        try:
            async with asyncio.timeout(self.bridge.clock.remaining()):
                return await self.runner.run()
        except TimeoutError:
            self.bridge.archive.event("deadline", seconds=self.bridge.clock.elapsed())
        finally:
            await self.close()

    async def close(self):
        from qwen_baselines.sandbox import kill_group

        kill_group(self.executor.process)
