"""A NOOA strategy whose planning language is Tycho's native tool surface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nooa.context_blocks import ResultStatus, ToolCallEvent, ToolResult
from nooa.errors import GenerationError
from nooa.events import Error, Task
from nooa.strategies.base import GenerationStrategy, RuntimeServices
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.planner import trigger_actor_world_model_paragraph
from tycho_nooa.tools import WorkspaceToolSurface, tool_by_name

if TYPE_CHECKING:
    from nooa.strategies.current_call import CurrentCall


ACTOR_BELIEFS_NOTES = "notes/actor_beliefs.md"
ACTOR_BELIEFS_CHECKPOINT_REQUEST = (
    f"Before this turn ends, checkpoint your current hypotheses into "
    f"{ACTOR_BELIEFS_NOTES} with write_file or edit_file. Do this even if you have "
    "no confirmed change: record what you tried, what you ruled out, and what "
    "remains open, so the builder is not reading the same stale hypotheses the "
    "next time it fires."
)
_NON_CHECKPOINT_TOOL_NAMES = {"take_action", "submit_action"}


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
        workspace = self._workspace(runtime)
        config = getattr(runtime.agent, "_native_tool_config", self.config)
        tools = surface.native_tools()
        tools_by_name = tool_by_name(tools)
        actor_beliefs_before = _read_optional(workspace.dir, ACTOR_BELIEFS_NOTES)
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
                    await self._enforce_actor_beliefs_checkpoint(
                        runtime, surface, tools, config, workspace, actor_beliefs_before
                    )
                    return decision

        raise GenerationError(
            f"No action was committed within {config.max_tool_steps} native tool steps"
        )

    async def _enforce_actor_beliefs_checkpoint(
        self,
        runtime: RuntimeServices,
        surface: WorkspaceToolSurface,
        tools: list[Any],
        config: NativeToolConfig,
        workspace: GameWorkspace,
        before: str | None,
    ) -> bool:
        """Guarantee notes/actor_beliefs.md advances before this turn can end.

        Real Tycho never forces this file: it is a voluntary actor scratchpad
        that the builder treats as a disposable starting hypothesis, not
        ground truth, so a stale copy costs Tycho nothing there. This is a
        deliberate adaptation beyond Tycho's design, not a fidelity fix:
        DIAGNOSIS.md's first live run found actor_beliefs.md written once at
        turn 1 and never refreshed despite two later probing bursts, and our
        goal is reliable level solving, which benefits from the builder
        seeing the actor's current thinking rather than its turn-1 draft.
        Mirrors _enforce_world_model_checkpoint's shape: one real chance to
        write a genuine update, then a forced breadcrumb only if the model
        still leaves the file untouched. take_action is withheld from the
        offered tools here — the turn's action was already committed.
        """
        if _read_optional(workspace.dir, ACTOR_BELIEFS_NOTES) != before:
            return False

        checkpoint_tools = [
            tool for tool in tools if tool.name not in _NON_CHECKPOINT_TOOL_NAMES
        ]
        checkpoint_tools_by_name = tool_by_name(checkpoint_tools)
        runtime.event_manager.add(Task(prompt=ACTOR_BELIEFS_CHECKPOINT_REQUEST))
        response, event_id = await runtime.generate(
            tools=checkpoint_tools,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            tool_choice="auto",
        )
        if response.tool_calls:
            runtime.event_manager.remove(event_id)
            for index, tool_call in enumerate(response.tool_calls):
                self._execute_tool_call(
                    runtime,
                    surface,
                    checkpoint_tools_by_name,
                    tool_call,
                    reasoning=(response.reasoning or str(response.content) or None)
                    if index == 0
                    else None,
                )

        if _read_optional(workspace.dir, ACTOR_BELIEFS_NOTES) != before:
            return True

        fallback = _fallback_actor_beliefs_checkpoint(before)
        workspace.write_file(ACTOR_BELIEFS_NOTES, fallback)
        if _read_optional(workspace.dir, ACTOR_BELIEFS_NOTES) != fallback:
            raise RuntimeError(
                f"failed to persist mandatory {ACTOR_BELIEFS_NOTES} checkpoint"
            )
        return True

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

    @staticmethod
    def _workspace(runtime: RuntimeServices) -> GameWorkspace:
        executor = getattr(runtime.agent, "_tool_executor", None)
        workspace = getattr(executor, "ws", None)
        if not isinstance(workspace, GameWorkspace):
            raise TypeError("NativeToolStrategy requires a ToolExecutor with a GameWorkspace")
        return workspace


def _read_optional(workspace_dir: Any, relative_path: str) -> str | None:
    path = workspace_dir / relative_path
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _fallback_actor_beliefs_checkpoint(previous: str | None) -> str:
    """A guaranteed-to-advance checkpoint entry, used when the model left no edit.

    Mirrors builder.py's _fallback_world_model_checkpoint: rather than trust the
    model to always leave a real edit, force the invariant "this turn changed the
    durable hypotheses file" to hold unconditionally.
    """
    marker = "[checkpoint: turn ended with no model-authored update]"
    if previous is None:
        return f"{marker}\n(no prior notes/actor_beliefs.md content)\n"
    return f"{previous.rstrip('\n')}\n\n{marker}\n"
