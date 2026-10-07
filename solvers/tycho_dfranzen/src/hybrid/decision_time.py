"""Opt-in Actor time allocation; keep a final action inside the host deadline."""
from __future__ import annotations

import asyncio
import json
import math
import os
from time import monotonic


class ExplorationExpired(TimeoutError):
    """The Actor must proceed to its existing action-only demand."""


def game_environment(deadline):
    environment = dict(os.environ)
    if environment.get("TYCHO_TIME_AWARE") == "1":
        environment["TYCHO_GAME_DEADLINE_MONOTONIC"] = str(deadline)
    return environment


def decision_runtime(runtime):
    """Disabled arms retain the exact original runtime and request behavior."""
    if os.environ.get("TYCHO_TIME_AWARE") != "1":
        return runtime
    deadline = runtime.agent._active_deadline
    if deadline is None:
        raise ValueError("time-aware Actor requires the host decision deadline")
    return DecisionRuntime(runtime, deadline)


class DecisionRuntime:
    """One immutable allocation shared by all calls in this decision.

    Cancellation cannot prove that the provider stopped computing. Every interrupted
    request gets an explicit unknown-usage receipt, even when the final action works.
    """

    def __init__(self, runtime, deadline):
        self.runtime = runtime
        total = deadline.total_seconds
        host_reserve = min(deadline.config.host_return_reserve_seconds, total * 0.1)
        usable = total - host_reserve
        self.action_deadline = deadline.started_at + usable
        self.action_reserve = min(120.0, usable * 0.25)
        self.exploration_deadline = self.action_deadline - self.action_reserve
        game_deadline = os.environ.get("TYCHO_GAME_DEADLINE_MONOTONIC")
        self.game_deadline = float(game_deadline) if game_deadline is not None else None
        self._observation_tag = None
        self._observation_prompt = None

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def record(self, phase, **fields):
        path = self.agent._tool_surface._trace_path
        if path is not None:
            path = path.with_name("actor-lifecycle.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps({"phase": phase, "monotonic": monotonic(), **fields}) + "\n")

    def _cue(self, final):
        from tycho_nooa.history import OBSERVATION_REF
        if self._observation_tag is None:
            for tag, event in self.event_manager.items():
                if OBSERVATION_REF in getattr(event, "metadata", {}):
                    self._observation_tag, self._observation_prompt = tag, event.prompt
        remaining = max(0.0, self.action_deadline - monotonic())
        explore = max(0.0, self.exploration_deadline - monotonic())
        cue = (f"\n\nHOST DECISION CLOCK: {remaining:.1f} seconds remain to return an action "
               f"before the host deadline; {explore:.1f} seconds remain for investigation. "
               "This is wall-clock time, separate from the game's action/HUD budget. ")
        if self.game_deadline is not None:
            cue += f"Whole-game time remaining: {max(0.0, self.game_deadline - monotonic()):.1f} seconds. "
        cue += ("Investigation is closed. Commit the best supported legal action now."
                if final else "Use a tool only if its result can change the next action; act when it cannot.")
        if self._observation_tag is not None:
            self.event_manager.update(self._observation_tag, prompt=self._observation_prompt + cue)

    async def generate(self, **kwargs):
        tools = kwargs.get("tools") or []
        final = bool(tools) and all(tool.name == "take_actions" for tool in tools)
        cutoff = self.action_deadline if final else self.exploration_deadline
        remaining = cutoff - monotonic()
        if remaining <= 0:
            if final:
                raise TimeoutError("Actor action deadline exhausted")
            raise ExplorationExpired("Actor investigation deadline exhausted")
        self._cue(final)
        # Native HTTP timeout also bounds I/O shielded from asyncio cancellation.
        kwargs["timeout"] = min(kwargs.get("timeout", remaining), remaining)
        self.record("action" if final else "exploration", remaining_seconds=remaining)
        try:
            async with asyncio.timeout_at(cutoff):
                return await self.runtime.generate(**kwargs)
        except asyncio.CancelledError:
            self.record("provider_interrupted", provider_usage="unknown", action_demand=final)
            raise
        except Exception as error:
            self.record("provider_interrupted", provider_usage="unknown", action_demand=final)
            if not final and (isinstance(error, TimeoutError) or monotonic() >= cutoff):
                raise ExplorationExpired("Actor investigation request exceeded its allocation") from error
            raise


def dispatch(runtime, surface, tool, arguments, **metadata):
    """Keep synchronous Python subprocesses inside investigation time too."""
    if isinstance(runtime, DecisionRuntime) and tool.name != "take_actions":
        remaining = runtime.exploration_deadline - monotonic()
        if remaining <= 0:
            raise ExplorationExpired("No investigation time remains for this tool")
        if tool.name == "run_python":
            seconds = math.floor(remaining)
            if seconds < 1:
                raise ExplorationExpired("Insufficient investigation time for Python")
            arguments = {**arguments, "timeout_s": min(arguments.get("timeout_s", 15), seconds)}
    return surface.dispatch(tool, arguments, **metadata)
