"""A NOOA strategy whose planning language is Tycho's native tool surface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nooa.context_blocks import ResultStatus, ToolCallEvent, ToolResult
from nooa.errors import GenerationError
from nooa.events import Error, Task
from nooa.strategies.base import GenerationStrategy, RuntimeServices

from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.planner import trigger_actor_world_model_paragraph
from tycho_nooa.tools import WorkspaceToolSurface, tool_by_name

if TYPE_CHECKING:
    from nooa.strategies.current_call import CurrentCall


@dataclass(frozen=True)
class NativeToolConfig:
    max_tool_steps: int = 25
    max_tokens: int = 16_000
    temperature: float | None = None


class NativeToolStrategy(GenerationStrategy):
    """Let the model work through native tools until it commits one action."""

    def __init__(self, config: NativeToolConfig | None = None):
        self.config = config or NativeToolConfig()

    @property
    def name(self) -> str:
        return "TYCHO_NATIVE_TOOLS"

    def get_block_overrides(self) -> dict[str, str]:
        return {
            "strategy_prompt": (
                "Work freely with the native workspace tools. Inspect the current and "
                "past evidence, and test executable ideas with run_python. End this turn "
                "only by calling take_action exactly once. The resulting observation will "
                "arrive on the next call to this same agent.\n\n"
                + trigger_actor_world_model_paragraph()
            )
        }

    async def execute(
        self,
        runtime: RuntimeServices,
        call: CurrentCall,
    ) -> ActionDecision:
        surface = self._surface(runtime)
        config = getattr(runtime.agent, "_native_tool_config", self.config)
        tools = surface.native_tools()
        tools_by_name = tool_by_name(tools)
        runtime.event_manager.add(Task(prompt=self._turn_prompt(call, surface)))

        for _ in range(config.max_tool_steps):
            response, event_id = await runtime.generate(
                tools=tools,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                tool_choice="auto",
            )
            if not response.tool_calls:
                runtime.event_manager.add(
                    Error(
                        content=(
                            "A workspace or take_action tool call is required. "
                            "Do not finish the turn with prose alone."
                        )
                    )
                )
                continue

            runtime.event_manager.remove(event_id)
            planner_in_batch = self._planner_in_batch(response.tool_calls, surface)
            for index, tool_call in enumerate(response.tool_calls):
                decision = self._execute_tool_call(
                    runtime,
                    surface,
                    tools_by_name,
                    tool_call,
                    reasoning=(response.reasoning or str(response.content) or None)
                    if index == 0
                    else None,
                    block_action=planner_in_batch,
                )
                if decision is not None:
                    return decision

        raise GenerationError(
            f"No action was committed within {config.max_tool_steps} native tool steps"
        )

    @staticmethod
    def _turn_prompt(
        call: CurrentCall,
        surface: WorkspaceToolSurface,
    ) -> str:
        turn = call.bound_parameters().get("turn")
        if not isinstance(turn, GameTurn):
            raise TypeError(
                "NativeToolStrategy requires a GameTurn argument named 'turn'"
            )
        return surface.observation_prompt(turn)

    def _execute_tool_call(
        self,
        runtime: RuntimeServices,
        surface: WorkspaceToolSurface,
        tools_by_name: dict[str, Any],
        tool_call: Any,
        *,
        reasoning: str | None,
        block_action: bool = False,
    ) -> ActionDecision | None:
        try:
            arguments = json.loads(tool_call.arguments)
            tool = tools_by_name[tool_call.name]
        except (json.JSONDecodeError, KeyError) as error:
            runtime.event_manager.add(
                Error(content=f"Invalid native tool call {tool_call.name!r}: {error}")
            )
            return None

        event_id = runtime.event_manager.add(
            ToolCallEvent(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                arguments=arguments,
                reasoning=reasoning,
            )
        )
        if block_action and tool_call.name in {"take_action", "submit_action"}:
            runtime.event_manager.update(
                event_id,
                result=ToolResult(
                    tool_call_id=tool_call.id,
                    content=(
                        "A planner call and take_action cannot share one tool batch. "
                        "Inspect the planner result, then choose an action in the next step."
                    ),
                    result_status=ResultStatus.ERROR,
                ),
            )
            return None
        try:
            result = surface.dispatch(tool, arguments)
        except Exception as error:  # noqa: BLE001 - a model tool error is recoverable feedback
            runtime.event_manager.update(
                event_id,
                result=ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"{type(error).__name__}: {error}",
                    result_status=ResultStatus.ERROR,
                ),
            )
            return None

        action = result if isinstance(result, ActionDecision) else None
        content = "Action accepted; this turn is complete." if action else str(result)
        runtime.event_manager.update(
            event_id,
            result=ToolResult(
                tool_call_id=tool_call.id,
                content=content,
                result_status=ResultStatus.COMPLETE,
            ),
        )
        return action

    @staticmethod
    def _planner_in_batch(
        tool_calls: list[Any],
        surface: WorkspaceToolSurface,
    ) -> bool:
        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call.arguments)
            except json.JSONDecodeError:
                continue
            if surface.is_planner_call(tool_call.name, arguments):
                return True
        return False

    @staticmethod
    def _surface(runtime: RuntimeServices) -> WorkspaceToolSurface:
        surface = getattr(runtime.agent, "_tool_surface", None)
        if not isinstance(surface, WorkspaceToolSurface):
            raise TypeError("NativeToolStrategy requires a WorkspaceToolSurface")
        return surface
