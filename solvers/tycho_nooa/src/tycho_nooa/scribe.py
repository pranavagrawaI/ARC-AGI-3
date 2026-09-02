"""Tycho's level-boundary scribe: a same-history consolidation pass.

Pinned `tycho/agent/agent.py::_build_agent_graph` registers the scribe as
`AgentSpec(id="scribe", trigger=on_events(Event.LEVEL_COMPLETE),
sees_full_history=True, run=lambda ctx: self._summarize_level())` — unlike the
builder (`sees_full_history=False`, a fresh transcript per fire), the scribe
reuses the actor's own growing conversation. NOOA's `Agent` owns exactly one
`event_manager` for its whole lifetime (`nooa/agent.py`), so the fidelity-
matching shape is a second `@strategy`-decorated method directly on
`TychoNooaAgent`, not a separate `FocusedBuilder`-style fresh `Agent`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nooa.context_blocks import ResultStatus, ToolCallEvent, ToolResult
from nooa.events import Error, Task
from nooa.strategies.base import GenerationStrategy, RuntimeServices
from tycho.prompts.render import render_prompt
from tycho.workspace.agent_tools import canonical
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.history import (
    bounded_context_text,
    bounded_tool_arguments,
    replayed_reasoning,
    retain_assistant_turn,
)
from tycho_nooa.modes import mode_from_env
from tycho_nooa.tools import WorkspaceToolSurface, tool_by_name

if TYPE_CHECKING:
    from nooa.strategies.current_call import CurrentCall


LEVEL_INSIGHTS_NOTES = "notes/level_{level}_insights.md"
SCRIBE_MAX_ROUNDS = 3  # tycho/agent/agent.py::_summarize_level's `for _ in range(3)`
_NON_SCRIBE_TOOL_NAMES = {"take_actions"}
_WORLD_MODEL_BASENAMES = {"world_model.py", "worldmodel.py"}
_WORLD_MODEL_EDIT_BLOCK_MESSAGE = (
    "(blocked: in this mode world_model.py is builder-owned. Write level insights "
    "to notes/level_<N>_insights.md and model-relevant observations to "
    "notes/world_model.md.)"
)
_FALLBACK_REASON = "summary pass did not create the requested note"


@dataclass(frozen=True)
class ScribeConfig:
    max_tokens: int = 16_000
    temperature: float | None = None
    max_context_value_bytes: int = 2_048


class ScribeStrategy(GenerationStrategy):
    """Consolidate a just-completed level into the actor's own event history."""

    @property
    def name(self) -> str:
        return "TYCHO_LEVEL_SCRIBE"

    def get_block_overrides(self) -> dict[str, str]:
        return {}

    async def execute(self, runtime: RuntimeServices, call: CurrentCall) -> None:
        surface = self._surface(runtime)
        workspace = self._workspace(runtime)
        config = getattr(runtime.agent, "_scribe_config", ScribeConfig())
        level = call.bound_parameters().get("level")
        if not isinstance(level, int):
            raise TypeError("ScribeStrategy requires an int argument named 'level'")

        # Tycho itself offers an action tool here and replaces its execution with
        # a skip message; tycho_nooa withholds it structurally instead.
        tools = [
            tool
            for tool in surface.native_tools()
            if tool.name not in _NON_SCRIBE_TOOL_NAMES
        ]
        tools_by_name = tool_by_name(tools)
        kickoff = render_prompt(
            "scribe.user",
            level=level,
            actor_writes_wm=False,
            wm_variant=mode_from_env().wm_variant,
        )
        runtime.event_manager.add(Task(prompt=kickoff))

        for _ in range(SCRIBE_MAX_ROUNDS):
            response, event_id = await runtime.generate(
                tools=tools,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                tool_choice="auto",
            )
            if not response.tool_calls:
                runtime.event_manager.remove(event_id)
                break
            retain_assistant_turn(
                runtime.event_manager,
                event_id,
                response,
                max_bytes=config.max_context_value_bytes,
            )
            for index, tool_call in enumerate(response.tool_calls):
                self._execute_tool_call(
                    runtime,
                    surface,
                    tools_by_name,
                    tool_call,
                    reasoning=(
                        replayed_reasoning(
                            response,
                            max_bytes=config.max_context_value_bytes,
                        )
                        if index == 0
                        else None
                    ),
                    max_context_value_bytes=config.max_context_value_bytes,
                )

        _write_level_insight_fallback(workspace, level, _FALLBACK_REASON)

    @staticmethod
    def _execute_tool_call(
        runtime: RuntimeServices,
        surface: WorkspaceToolSurface,
        tools_by_name: dict[str, Any],
        tool_call: Any,
        *,
        reasoning: str | None = None,
        max_context_value_bytes: int = 2_048,
    ) -> None:
        try:
            arguments = json.loads(tool_call.arguments)
        except json.JSONDecodeError as error:
            message = f"Invalid scribe tool call {tool_call.name!r}: {error}"
            surface.record_rejected_call(
                tool_call.name,
                {"raw_arguments": tool_call.arguments},
                message,
                tool_call_id=tool_call.id,
                raw_arguments=tool_call.arguments,
                rejection_stage="parse",
            )
            runtime.event_manager.add(Error(content=message))
            return
        try:
            tool = tools_by_name[tool_call.name]
        except KeyError as error:
            message = f"Invalid scribe tool call {tool_call.name!r}: {error}"
            surface.record_rejected_call(
                tool_call.name,
                arguments,
                message,
                tool_call_id=tool_call.id,
                raw_arguments=tool_call.arguments,
                rejection_stage="availability",
            )
            runtime.event_manager.add(Error(content=message))
            return

        event_id = runtime.event_manager.add(
            ToolCallEvent(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                arguments=bounded_tool_arguments(
                    arguments,
                    limit=max_context_value_bytes,
                    tool_name=tool_call.name,
                ),
                reasoning=reasoning,
            )
        )
        if _is_world_model_edit(tool_call.name, arguments):
            surface.record_rejected_call(
                tool_call.name,
                arguments,
                _WORLD_MODEL_EDIT_BLOCK_MESSAGE,
                tool_call_id=tool_call.id,
                raw_arguments=tool_call.arguments,
                rejection_stage="policy",
            )
            runtime.event_manager.update(
                event_id,
                result=ToolResult(
                    tool_call_id=tool_call.id,
                    content=_WORLD_MODEL_EDIT_BLOCK_MESSAGE,
                    result_status=ResultStatus.ERROR,
                ),
            )
            return
        try:
            result = surface.dispatch(
                tool,
                arguments,
                tool_call_id=tool_call.id,
                raw_arguments=tool_call.arguments,
            )
        except Exception as error:  # noqa: BLE001 - a model tool error is recoverable feedback
            runtime.event_manager.update(
                event_id,
                result=ToolResult(
                    tool_call_id=tool_call.id,
                    content=bounded_context_text(
                        f"{type(error).__name__}: {error}",
                        limit=max_context_value_bytes,
                        kind=f"{tool_call.name} result",
                    ),
                    result_status=ResultStatus.ERROR,
                ),
            )
            return
        runtime.event_manager.update(
            event_id,
            result=ToolResult(
                tool_call_id=tool_call.id,
                content=bounded_context_text(
                    str(result),
                    limit=max_context_value_bytes,
                    kind=f"{tool_call.name} result",
                ),
                result_status=ResultStatus.COMPLETE,
            ),
        )

    @staticmethod
    def _surface(runtime: RuntimeServices) -> WorkspaceToolSurface:
        surface = getattr(runtime.agent, "_tool_surface", None)
        if not isinstance(surface, WorkspaceToolSurface):
            raise TypeError("ScribeStrategy requires a WorkspaceToolSurface")
        return surface

    @staticmethod
    def _workspace(runtime: RuntimeServices) -> GameWorkspace:
        executor = getattr(runtime.agent, "_tool_executor", None)
        workspace = getattr(executor, "ws", None)
        if not isinstance(workspace, GameWorkspace):
            raise TypeError(
                "ScribeStrategy requires a ToolExecutor with a GameWorkspace"
            )
        return workspace


def _is_world_model_edit(name: str, arguments: dict[str, Any]) -> bool:
    """Port of pinned `agent.py::_is_world_model_edit`, inlined for a dict of
    already-parsed arguments rather than Tycho's own `{"input": {...}}` call
    shape. tycho_nooa's mode always has `actor_writes_wm=False`, so unlike
    `_is_builder_owned_world_model_edit` this omits the mode check — it is
    always builder-owned here."""
    if canonical(name) not in ("write_file", "edit_file", "edit_function"):
        return False
    path = str(arguments.get("path", ""))
    return path.replace("\\", "/").split("/")[-1] in _WORLD_MODEL_BASENAMES


def _write_level_insight_fallback(
    workspace: GameWorkspace, level: int, reason: str
) -> None:
    """Byte-exact port of pinned `agent.py::_write_level_insight_fallback`."""
    path = LEVEL_INSIGHTS_NOTES.format(level=level)
    existing = workspace.read_file(path, max_bytes=200)
    if not existing.startswith("(no such file"):
        return
    solved = workspace.read_file(f"level_{level}/solved.json", max_bytes=1000)
    level_complete = workspace.read_file(
        f"level_{level}/level_complete.json", max_bytes=1800
    )
    terminal = workspace.read_file(f"level_{level}/terminal.json", max_bytes=1800)
    content = (
        f"# Level {level} Insights\n\n"
        f"Deterministic fallback note: {reason}. The LLM consolidation step did not write "
        "a carry-forward note before the next level began.\n\n"
        f"solved.json:\n{solved}\n\n"
        f"level_complete.json:\n{level_complete}\n\n"
        f"terminal.json:\n{terminal}\n"
    )
    workspace.write_file(path, content)
