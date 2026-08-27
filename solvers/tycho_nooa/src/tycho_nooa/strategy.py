"""A NOOA strategy whose planning language is Tycho's native tool surface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nooa.context_blocks import ResultStatus, ToolCallEvent, ToolResult
from nooa.errors import GenerationError
from nooa.events import Error, Task
from nooa.strategies.base import GenerationStrategy, RuntimeServices
from tycho.workspace.agent_tools import canonical
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.epistemic import RequestBuilderArguments
from tycho_nooa.history import (
    ELIDED_OBSERVATION,
    OBSERVATION_REF,
    collapse_old_actor_turns,
    evict_old_grids,
    replayed_reasoning,
    retain_assistant_turn,
)
from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.modes import mode_from_env
from tycho_nooa.planner import actor_world_model_paragraph
from tycho_nooa.tools import Observation, WorkspaceToolSurface, tool_by_name

if TYPE_CHECKING:
    from nooa.strategies.current_call import CurrentCall


ACTOR_BELIEFS_NOTES = "notes/actor_beliefs.md"
ACTION_DEMAND_REQUEST = (
    "Your workspace tool budget for this turn is spent. take_action is the only "
    "tool left. Commit the single best action the evidence you already gathered "
    "can justify, by calling take_action exactly once now."
)
# The turn-ending tools are the only ones offered by the final demand.
_ACTION_TOOL_NAMES = {"take_action", "submit_action"}


@dataclass(frozen=True)
class NativeToolConfig:
    max_tool_steps: int = 25
    # Tycho's TYCHO_GRID_KEEP default: how many recent turns keep their inlined grid.
    grid_keep: int = 3
    history_keep_turns: int = 3
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
                + actor_world_model_paragraph(mode_from_env().wm_variant)
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
        observation = self._observation(call, surface)
        turn = surface.current_turn()
        runtime.event_manager.add(
            Task(
                prompt=observation.prompt,
                metadata={
                    ELIDED_OBSERVATION: observation.elided,
                    OBSERVATION_REF: {
                        "level": turn.level,
                        "turn": turn.turn_in_level,
                    },
                },
            )
        )
        evict_old_grids(runtime.event_manager, keep=config.grid_keep)
        collapse_old_actor_turns(
            runtime.event_manager,
            keep=config.history_keep_turns,
        )

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

            retain_assistant_turn(runtime.event_manager, event_id, response)
            feedback_in_batch = self._feedback_in_batch(response.tool_calls, surface)
            for index, tool_call in enumerate(response.tool_calls):
                decision = await self._execute_tool_call(
                    runtime,
                    surface,
                    tools_by_name,
                    tool_call,
                    reasoning=replayed_reasoning(response) if index == 0 else None,
                    block_action=feedback_in_batch,
                )
                if decision is not None:
                    self._checkpoint_actor_beliefs_after_action(
                        workspace, actor_beliefs_before, decision
                    )
                    return decision

        decision = await self._demand_action(runtime, surface, tools, config)
        if decision is None:
            raise GenerationError(
                f"No action was committed within {config.max_tool_steps} native "
                "tool steps, and none was committed when take_action was the only "
                "tool left"
            )
        self._checkpoint_actor_beliefs_after_action(
            workspace, actor_beliefs_before, decision
        )
        return decision

    async def _demand_action(
        self,
        runtime: RuntimeServices,
        surface: WorkspaceToolSurface,
        tools: list[Any],
        config: NativeToolConfig,
    ) -> ActionDecision | None:
        """Spend one last step asking for the action, before losing the episode.

        Exhausting `max_tool_steps` is not the same failure as a model that will
        not act. `runs/tycho-nooa-goal-gate-ar25-20260819T120904Z` died here at
        action 0 with all 25 steps spent reading the grid and never a
        `take_action`; the episode's remaining 19 actions went with it. An ARC
        episode has no "no action" branch, so a turn that ends in silence throws
        the run either way. Restrict the offered tools to the one that must be
        called, then let the raise stand if the model still declines.
        `tool_choice` stays "auto" because
        "required" draws 500s from some gateways (nooa's own codeact says so).
        """
        action_tools = [tool for tool in tools if tool.name in _ACTION_TOOL_NAMES]
        if not action_tools:
            return

        runtime.event_manager.add(Task(prompt=ACTION_DEMAND_REQUEST))
        response, event_id = await runtime.generate(
            tools=action_tools,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            tool_choice="auto",
        )
        if not response.tool_calls:
            return

        retain_assistant_turn(runtime.event_manager, event_id, response)
        for index, tool_call in enumerate(response.tool_calls):
            decision = await self._execute_tool_call(
                runtime,
                surface,
                tool_by_name(action_tools),
                tool_call,
                reasoning=replayed_reasoning(response) if index == 0 else None,
            )
            if decision is not None:
                return decision
        return

    @staticmethod
    def _checkpoint_actor_beliefs_after_action(
        workspace: GameWorkspace,
        before: str | None,
        decision: ActionDecision,
    ) -> bool:
        """Advance the hand-off without delaying an accepted environment action.

        Real Tycho never forces this file: it is a voluntary actor scratchpad
        that the builder treats as a disposable starting hypothesis. We retain
        the local invariant introduced after DIAGNOSIS.md found stale notes, but
        never buy it with another provider call after `take_action`: the live
        reasoning-capture run had an accepted action with 27.6 seconds left and
        lost it while this post-action call took 29.9 seconds. A factual,
        provenance-marked breadcrumb is bounded local I/O and keeps stale model
        prose distinguishable from the action the model actually committed.
        """
        if _read_optional(workspace.dir, ACTOR_BELIEFS_NOTES) != before:
            return False

        fallback = _fallback_actor_beliefs_checkpoint(before, decision)
        workspace.write_file(ACTOR_BELIEFS_NOTES, fallback)
        if _read_optional(workspace.dir, ACTOR_BELIEFS_NOTES) != fallback:
            raise RuntimeError(
                f"failed to persist mandatory {ACTOR_BELIEFS_NOTES} checkpoint"
            )
        return True

    @staticmethod
    def _observation(
        call: CurrentCall,
        surface: WorkspaceToolSurface,
    ) -> Observation:
        turn = call.bound_parameters().get("turn")
        if not isinstance(turn, GameTurn):
            raise TypeError(
                "NativeToolStrategy requires a GameTurn argument named 'turn'"
            )
        return surface.observation(turn)

    async def _execute_tool_call(
        self,
        runtime: RuntimeServices,
        surface: WorkspaceToolSurface,
        tools_by_name: dict[str, Any],
        tool_call: Any,
        *,
        reasoning: str | None = None,
        block_action: bool = False,
    ) -> ActionDecision | None:
        validated = self._validated_tool_call(
            runtime, surface, tools_by_name, tool_call
        )
        if validated is None:
            return None
        arguments, tool = validated

        event_id = runtime.event_manager.add(
            ToolCallEvent(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                arguments=arguments,
                reasoning=reasoning,
            )
        )
        blocked = self._tool_block_reason(surface, tool_call.name, block_action)
        if blocked is not None:
            self._reject_tool_event(
                runtime, surface, tool_call, arguments, event_id, blocked
            )
            return None
        if tool_call.name in {"invoke_builder", "request_builder"}:
            return await self._invoke_builder(
                runtime,
                surface,
                tool,
                arguments,
                tool_call_id=tool_call.id,
                raw_arguments=tool_call.arguments,
                event_id=event_id,
            )
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
    def _validated_tool_call(
        runtime: RuntimeServices,
        surface: WorkspaceToolSurface,
        tools_by_name: dict[str, Any],
        tool_call: Any,
    ) -> tuple[dict[str, Any], Any] | None:
        try:
            arguments = json.loads(tool_call.arguments)
        except json.JSONDecodeError as error:
            message = f"Invalid native tool call {tool_call.name!r}: {error}"
            surface.record_rejected_call(
                tool_call.name,
                {"raw_arguments": tool_call.arguments},
                message,
                tool_call_id=tool_call.id,
                raw_arguments=tool_call.arguments,
                rejection_stage="parse",
            )
            runtime.event_manager.add(Error(content=message))
            return None
        try:
            return arguments, tools_by_name[tool_call.name]
        except KeyError as error:
            message = f"Invalid native tool call {tool_call.name!r}: {error}"
            surface.record_rejected_call(
                tool_call.name,
                arguments,
                message,
                tool_call_id=tool_call.id,
                raw_arguments=tool_call.arguments,
                rejection_stage="availability",
            )
            runtime.event_manager.add(Error(content=message))
            return None

    @staticmethod
    def _tool_block_reason(
        surface: WorkspaceToolSurface, name: str, block_action: bool
    ) -> str | None:
        action_names = {"take_action", "submit_action"}
        if name in action_names and surface.goal_tools is not None:
            required = surface.goal_tools.required_before_action()
            if required is not None:
                return f"Typed objective lifecycle incomplete: {required}"
        if block_action and name == "update_goal":
            return (
                "update_goal cannot share a tool batch with request_builder. Inspect "
                "the returned objective review, then resolve the candidate in the next "
                "model step."
            )
        if block_action and name in action_names:
            return (
                "A builder request, world-model edit, or planner call cannot share "
                "one tool batch with take_action. Inspect that feedback, then choose "
                "an action in the next step."
            )
        return None

    @staticmethod
    def _reject_tool_event(
        runtime: RuntimeServices,
        surface: WorkspaceToolSurface,
        tool_call: Any,
        arguments: dict[str, Any],
        event_id: Any,
        message: str,
    ) -> None:
        surface.record_rejected_call(
            tool_call.name,
            arguments,
            message,
            tool_call_id=tool_call.id,
            raw_arguments=tool_call.arguments,
            rejection_stage="policy",
        )
        runtime.event_manager.update(
            event_id,
            result=ToolResult(
                tool_call_id=tool_call.id,
                content=message,
                result_status=ResultStatus.ERROR,
            ),
        )

    async def _invoke_builder(
        self,
        runtime: RuntimeServices,
        surface: WorkspaceToolSurface,
        tool: Any,
        arguments: dict[str, Any],
        *,
        tool_call_id: str,
        raw_arguments: str,
        event_id: Any,
    ) -> None:
        model = tool.parameters_model
        assert model is not None
        try:
            validated = model.model_validate(arguments)
        except Exception as error:  # noqa: BLE001 - model receives validation feedback
            message = f"{type(error).__name__}: {error}"
            surface.record_rejected_call(
                tool.name,
                arguments,
                message,
                tool_call_id=tool_call_id,
                raw_arguments=raw_arguments,
                rejection_stage="validation",
            )
            runtime.event_manager.update(
                event_id,
                result=ToolResult(
                    tool_call_id=tool_call_id,
                    content=message,
                    result_status=ResultStatus.ERROR,
                ),
            )
            return

        normalized_arguments = validated.model_dump(exclude_none=True)
        try:
            turn = surface.current_turn()
            if tool.name == "request_builder":
                if not isinstance(validated, RequestBuilderArguments):
                    raise TypeError("request_builder has the wrong argument contract")
                result = await runtime.agent.request_builder(turn, validated)
            else:
                result = await runtime.agent.invoke_builder(turn, validated.reason)
            surface.record_external_result(
                tool.name,
                arguments,
                result,
                tool_call_id=tool_call_id,
                raw_arguments=raw_arguments,
                validated_arguments=normalized_arguments,
            )
        except Exception as error:  # noqa: BLE001 - model receives builder failure
            message = f"{type(error).__name__}: {error}"
            surface.record_rejected_call(
                tool.name,
                arguments,
                message,
                tool_call_id=tool_call_id,
                raw_arguments=raw_arguments,
                rejection_stage="execution",
                validated_arguments=normalized_arguments,
            )
            runtime.event_manager.update(
                event_id,
                result=ToolResult(
                    tool_call_id=tool_call_id,
                    content=message,
                    result_status=ResultStatus.ERROR,
                ),
            )
            return
        runtime.event_manager.update(
            event_id,
            result=ToolResult(
                tool_call_id=tool_call_id,
                content=result,
                result_status=ResultStatus.COMPLETE,
            ),
        )
        return

    @staticmethod
    def _feedback_in_batch(
        tool_calls: list[Any],
        surface: WorkspaceToolSurface,
    ) -> bool:
        for tool_call in tool_calls:
            if tool_call.name in {"invoke_builder", "request_builder"}:
                return True
            try:
                arguments = json.loads(tool_call.arguments)
            except json.JSONDecodeError:
                continue
            if surface.is_planner_call(tool_call.name, arguments) or (
                canonical(tool_call.name) in {"write_file", "edit_file", "edit_function"}
                and str(arguments.get("path", "")).replace("\\", "/").split("/")[-1]
                in {"world_model.py", "worldmodel.py"}
            ):
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
            raise TypeError(
                "NativeToolStrategy requires a ToolExecutor with a GameWorkspace"
            )
        return workspace


def _read_optional(workspace_dir: Any, relative_path: str) -> str | None:
    path = workspace_dir / relative_path
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _fallback_actor_beliefs_checkpoint(
    previous: str | None, decision: ActionDecision
) -> str:
    """A guaranteed-to-advance checkpoint entry, used when the model left no edit.

    Mirrors builder.py's _fallback_world_model_checkpoint: rather than trust the
    model to always leave a real edit, force the invariant "this turn changed the
    durable hypotheses file" to hold unconditionally.
    """
    marker = (
        "[checkpoint: model committed "
        f"{decision.action} with no model-authored beliefs update]"
    )
    if previous is None:
        return f"{marker}\n(no prior notes/actor_beliefs.md content)\n"
    return f"{previous.rstrip('\n')}\n\n{marker}\n"
