"""Bounded construction and handoff; canonical verification still owns validity."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import json
from time import monotonic

from nooa.events import Task
from tycho.workspace.agent_tools import canonical
from tycho.agent.builder import REPORT_FILE
from tycho_nooa.builder import (
    BuilderStrategy, COMMIT_TOOL_NAMES, FALLBACK_REPORT, WORLD_MODEL_NOTES,
    _fallback_world_model_checkpoint, _read_optional,
)


@dataclass(frozen=True)
class Allocation:
    deadline: float
    reserve: float


ALLOCATION: ContextVar[Allocation] = ContextVar("hybrid_builder_allocation")


def record(builder, phase, **fields):
    path = builder._tool_surface._trace_path
    if path is not None:
        path = path.with_name("builder-lifecycle.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps({"phase": phase, "monotonic": monotonic(), **fields}) + "\n")


class PhaseRuntime:
    """Limit every model call by this phase, including the provider HTTP timeout."""
    def __init__(self, runtime, builder, phase, deadline, *, construction=False):
        self.runtime, self.builder = runtime, builder
        self.phase, self.deadline = phase, deadline
        self.construction, self.calls = construction, 0
        self.model_before = _read_optional(builder._executor.ws.dir, "world_model.py")

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    async def generate(self, **kwargs):
        remaining = self.deadline - monotonic()
        if remaining <= 0 or (self.construction and self.calls >= 12):
            raise TimeoutError("Builder phase exhausted")
        self.calls += 1
        if self.construction and self.calls >= 5:
            unchanged = _read_optional(self.builder._executor.ws.dir, "world_model.py") == self.model_before
            if unchanged:
                self.runtime.event_manager.add(Task(prompt=(
                    "Externalize now: write the smallest falsifiable executable model edit "
                    "supported by observed transitions. Do not dump more frames. If evidence "
                    "cannot support an edit, write notes/world_model.md stating the missing "
                    "evidence and the prospective action that would distinguish the hypotheses. "
                    "Do not claim an untested model is correct."
                )))
                kwargs["tools"] = [t for t in kwargs.get("tools", []) if canonical(t.name) in COMMIT_TOOL_NAMES]
                kwargs["tool_choice"] = "required"
        # Native provider timeout contains shielded requests as well as the caller.
        kwargs["timeout"] = max(0.01, remaining)
        record(self.builder, self.phase, call=self.calls, remaining_seconds=remaining)
        try:
            async with asyncio.timeout(remaining):
                return await self.runtime.generate(**kwargs)
        except (TimeoutError, asyncio.CancelledError):
            record(self.builder, "provider_interrupted", generation_phase=self.phase,
                   call=self.calls, provider_usage="unknown")
            raise


class HybridBuilderStrategy(BuilderStrategy):
    async def execute(self, runtime, call):
        request = call.bound_parameters()["request"]
        # Host normally supplies this; retain bounded behavior for direct callers.
        deadline = request.deadline_at or monotonic() + 240
        seconds = max(0, deadline - monotonic())
        token = ALLOCATION.set(Allocation(deadline, min(90, seconds * 0.375)))
        builder = self._builder(runtime)
        record(builder, "start", allocation_seconds=seconds, handoff_reserve_seconds=ALLOCATION.get().reserve)
        runtime.event_manager.add(Task(prompt=(
            "Build one small testable mechanism, not a complete game theory. Inspect targeted "
            "evidence, externalize an executable candidate within four investigation calls, "
            "verify it against observed transitions, and report a prospective prediction. "
            "A wrong candidate is evidence to repair; unsupported certainty is not progress. "
            "Investigation ends with time reserved for notes and a compact report."
        )))
        try:
            return await super().execute(runtime, call)
        finally:
            record(builder, "end", remaining_seconds=deadline-monotonic())
            ALLOCATION.reset(token)

    async def _construction_loop(self, runtime, builder, *args, **kwargs):
        allocation = ALLOCATION.get()
        bounded = PhaseRuntime(runtime, builder, "construction", allocation.deadline-allocation.reserve,
                               construction=True)
        try:
            await super()._construction_loop(bounded, builder, *args, **kwargs)
        except TimeoutError:
            record(builder, "handoff", reason="construction phase exhausted", calls=bounded.calls)
            runtime.event_manager.add(Task(prompt="Construction phase ended. Checkpoint evidence and hand off now."))

    async def _enforce_world_model_checkpoint(self, runtime, builder, *args, **kwargs):
        deadline = ALLOCATION.get().deadline
        bounded = PhaseRuntime(runtime, builder, "checkpoint", monotonic()+max(0, deadline-monotonic()-2)/2)
        try:
            return await super()._enforce_world_model_checkpoint(bounded, builder, *args, **kwargs)
        except TimeoutError:
            before = builder._world_model_before
            if _read_optional(builder._executor.ws.dir, WORLD_MODEL_NOTES) == before:
                builder._executor.ws.write_file(WORLD_MODEL_NOTES, _fallback_world_model_checkpoint(before))
            record(builder, "checkpoint_timeout")
            return True

    async def _finish_report(self, runtime, builder, tools, config, checkpoint_forced, checkpoint, *, objective_review):
        bounded = PhaseRuntime(runtime, builder, "report", ALLOCATION.get().deadline-2)
        try:
            return await super()._finish_report(bounded, builder, tools, config, checkpoint_forced, checkpoint,
                                               objective_review=objective_review)
        except TimeoutError:
            # Honest fallback: completion of the lifecycle is not model acceptance.
            report = FALLBACK_REPORT + "\nnote: Builder report unavailable at allocated deadline.\n"
            builder._executor.ws.write_file(REPORT_FILE, report)
            record(builder, "report_timeout")
            return report, True, checkpoint_forced, checkpoint.required, checkpoint.resolved
