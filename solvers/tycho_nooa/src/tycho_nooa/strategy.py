"""A NOOA strategy whose planning language is Tycho's native tool surface."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from nooa.context_blocks import ResultStatus, ToolCallEvent, ToolResult
from nooa.errors import GenerationError
from nooa.events import Error, Task
from nooa.strategies.base import GenerationStrategy, RuntimeServices
from tycho.workspace.agent_tools import canonical
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.action_batch import ActionBatchDecision
from tycho_nooa.epistemic import RequestBuilderArguments
from tycho_nooa.history import (
    ELIDED_OBSERVATION,
    OBSERVATION_REF,
    bounded_context_text,
    bounded_observation,
    bounded_tool_arguments,
    collapse_old_actor_turns,
    evict_old_grids,
    replayed_reasoning,
    retain_assistant_turn,
)
from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.modes import mode_from_env
from tycho_nooa.planner import actor_world_model_paragraph
from tycho_nooa.tools import (
    ModelToolError,
    Observation,
    WorkspaceToolSurface,
    tool_by_name,
)
from tycho_nooa.treatments import (
    DECISION_DISCIPLINE_STRATEGY_PROMPT,
    treatment_from_env,
)

if TYPE_CHECKING:
    from nooa.strategies.current_call import CurrentCall


ACTOR_BELIEFS_NOTES = "notes/actor_beliefs.md"
ACTION_DEMAND_REQUEST = (
    "FINAL MODEL CALL FOR THIS TURN. There is no later request and no opportunity "
    "to continue reasoning after this response. take_actions is the only tool left and "
    "the only way to commit environment actions. Its actions array may contain one action. "
    "If this response lacks one valid take_actions tool call, the current game thread "
    "fails without an environment action. Incomplete understanding is allowed. Stop "
    "investigating and commit the best legal action or bounded action sequence the "
    "evidence already gathered can justify."
)
# The turn-ending tools are the only ones offered by the final demand.
_ACTION_TOOL_NAMES = {"take_actions"}


def _visible_tool_error(
    surface: WorkspaceToolSurface,
    *,
    stage: str,
    code: str,
    message: str,
) -> str:
    return json.dumps(
        {
            "available_actions": surface.current_turn().available_actions,
            "code": code,
            "message": message,
            "stage": stage,
            "turn_consumed": False,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _decision_cue_prompt(*, used: int, total: int, final: bool) -> str:
    total_calls = total + 1
    if final:
        return (
            "EXPLORATION CLOSED\n"
            f"Model calls used: {used} of {total_calls}. This is the final model call "
            "for the turn; no later request exists. If this response lacks one valid "
            "take_actions tool call, the game thread fails without acting. Its actions "
            "array may contain one action. "
            "Commit the best legal action or bounded action sequence supported by current "
            "evidence now. Uncertainty is allowed; each released action's consequence "
            "will become new evidence."
        )
    remaining = max(0, total - used)
    clock = (
        f"Model calls used: {used} of {total_calls}; {total_calls - used} remain, "
        "including one final action-only call. "
    )
    if used == 0:
        return (
            "DECISION CLOCK\n"
            f"{clock}Current obligation: commit one ARC action, or a bounded sequence "
            "whose later steps need no new judgment. Investigate only for a result that "
            "could change that commitment."
        )
    if remaining == 1:
        return (
            "COMMITMENT PRESSURE\n"
            f"{clock}One exploratory call remains. Use it only to resolve a decision-reversing "
            "uncertainty. Incomplete understanding is not a reason to delay by itself."
        )
    if used >= max(2, total // 2):
        return (
            "METACOGNITIVE CHECK\n"
            f"{clock}What action is currently favored? What specific uncertainty blocks it? What "
            "result from the next operation would change it? If none, act now."
        )
    return (
        "DECISION CLOCK\n"
        f"{clock}Investigate only for a result that "
        "could change the currently favored action."
    )


@dataclass(frozen=True)
class NativeToolConfig:
    max_tool_steps: int = 25
    max_tool_steps_after_initial: int | None = None
    # Tycho's TYCHO_GRID_KEEP default: how many recent turns keep their inlined grid.
    grid_keep: int = 3
    history_keep_turns: int = 3
    max_observation_bytes: int = 16_384
    max_context_value_bytes: int = 2_048
    max_tokens: int = 16_000
    max_tokens_after_initial: int | None = None
    action_demand_max_tokens: int | None = None
    action_demand_disable_thinking: bool = False
    temperature: float | None = None
    force_action_tool: bool = False


def _config_for_sequence(
    config: NativeToolConfig, sequence: int | None
) -> NativeToolConfig:
    """Apply explicit later-turn ceilings without changing the initial contract."""
    if sequence is None or sequence <= 0:
        return config
    return replace(
        config,
        max_tool_steps=(
            config.max_tool_steps
            if config.max_tool_steps_after_initial is None
            else config.max_tool_steps_after_initial
        ),
        max_tokens=(
            config.max_tokens
            if config.max_tokens_after_initial is None
            else config.max_tokens_after_initial
        ),
    )


class NativeToolStrategy(GenerationStrategy):
    """Let the model work through native tools until it commits one action."""

    def __init__(self, config: NativeToolConfig | None = None):
        self.config = config or NativeToolConfig()

    @property
    def name(self) -> str:
        return "TYCHO_NATIVE_TOOLS"

    def get_block_overrides(self) -> dict[str, str]:
        mode = mode_from_env()
        treatment = treatment_from_env(mode)
        strategy_prompt = (
            DECISION_DISCIPLINE_STRATEGY_PROMPT
            if treatment.decision_discipline
            else (
                "Work freely with the native workspace tools. Inspect the current and "
                "past evidence, and test executable ideas with run_python. End this turn "
                "only by calling take_actions exactly once. Pass one action when only one "
                "step is justified; include later steps only when they need no new judgment. "
                "Every public successor is "
                "still observed and can stop the queue. The resulting observation will "
                "arrive on the next call to this same agent."
            )
        )
        return {
            "strategy_prompt": (
                f"{strategy_prompt}\n\n"
                + actor_world_model_paragraph(
                    mode.wm_variant,
                    request_builder_instead_of_invoke=treatment.typed_candidate_goals,
                )
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
        prompt = bounded_observation(
            observation.prompt,
            observation.elided,
            limit=config.max_observation_bytes,
        )
        turn = surface.current_turn()
        config = _config_for_sequence(config, turn.sequence)
        runtime.event_manager.add(
            Task(
                prompt=prompt,
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

        for step in range(config.max_tool_steps):
            self._update_decision_cue(
                runtime,
                used=step,
                total=config.max_tool_steps,
            )
            response, event_id = await runtime.generate(
                tools=tools,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                tool_choice="auto",
            )
            retain_assistant_turn(
                runtime.event_manager,
                event_id,
                response,
                max_bytes=config.max_context_value_bytes,
            )
            if not response.tool_calls:
                runtime.event_manager.add(
                    Error(
                        content=(
                            "A workspace or take_actions tool call is required. "
                            "Do not finish the turn with prose alone."
                        )
                    )
                )
                continue

            feedback_in_batch = self._feedback_in_batch(response.tool_calls, surface)
            for index, tool_call in enumerate(response.tool_calls):
                decision = await self._execute_tool_call(
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
                    block_action=feedback_in_batch,
                )
                if decision is not None:
                    self._checkpoint_actor_beliefs_after_action(
                        workspace, actor_beliefs_before, decision
                    )
                    return decision

        self._update_decision_cue(
            runtime,
            used=config.max_tool_steps,
            total=config.max_tool_steps,
            final=True,
        )
        decision = await self._demand_action(runtime, surface, tools, config)
        if decision is None:
            raise GenerationError(
                f"No action was committed within {config.max_tool_steps} native "
                "tool steps, and none was committed when take_actions was the only "
                "tool left"
            )
        self._checkpoint_actor_beliefs_after_action(
            workspace, actor_beliefs_before, decision
        )
        return decision

    @staticmethod
    def _update_decision_cue(
        runtime: RuntimeServices,
        *,
        used: int,
        total: int,
        final: bool = False,
    ) -> None:
        treatment = runtime.agent._treatment
        if not treatment.decision_discipline:
            return
        prompt = _decision_cue_prompt(used=used, total=total, final=final)
        event_id = runtime.agent._decision_cue_event_id
        if event_id is None or event_id not in runtime.event_manager:
            runtime.agent._decision_cue_event_id = runtime.event_manager.add(
                Task(prompt=prompt)
            )
            return
        runtime.event_manager.update(event_id, prompt=prompt)

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
        action; the episode's remaining 19 actions went with it. An ARC
        episode has no "no action" branch, so a turn that ends in silence throws
        the run either way. Restrict the offered tools to the one that must be
        called, then let the raise stand if the model still declines.
        `tool_choice` remains provider-configurable because some gateways have
        historically stalled on "required". The qualified local SGLang route
        supports the explicit required choice and enables it at runtime.
        """
        action_tools = [tool for tool in tools if tool.name in _ACTION_TOOL_NAMES]
        if not action_tools:
            return

        runtime.event_manager.add(Task(prompt=ACTION_DEMAND_REQUEST))
        generation_options: dict[str, Any] = {}
        if config.action_demand_disable_thinking:
            generation_options["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        response, event_id = await runtime.generate(
            tools=action_tools,
            max_tokens=config.action_demand_max_tokens or config.max_tokens,
            temperature=config.temperature,
            tool_choice="required" if config.force_action_tool else "auto",
            **generation_options,
        )
        retain_assistant_turn(
            runtime.event_manager,
            event_id,
            response,
            max_bytes=config.max_context_value_bytes,
        )
        if not response.tool_calls:
            return
        for index, tool_call in enumerate(response.tool_calls):
            decision = await self._execute_tool_call(
                runtime,
                surface,
                tool_by_name(action_tools),
                tool_call,
                reasoning=(
                    replayed_reasoning(
                        response,
                        max_bytes=config.max_context_value_bytes,
                    )
                    if index == 0
                    else None
                ),
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
        never buy it with another provider call after an action commitment: the live
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
                arguments=self._provider_tool_arguments(
                    runtime,
                    tool_call.name,
                    arguments,
                ),
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
            if isinstance(error, ModelToolError):
                stage = error.stage
                code = error.code
                message = error.message
            else:
                stage = "execution"
                code = "execution_failed"
                message = str(error)
            runtime.event_manager.update(
                event_id,
                result=ToolResult(
                    tool_call_id=tool_call.id,
                    content=self._provider_tool_result(
                        runtime,
                        tool_call.name,
                        _visible_tool_error(
                            surface,
                            stage=stage,
                            code=code,
                            message=message,
                        ),
                    ),
                    result_status=ResultStatus.ERROR,
                ),
            )
            return None

        if isinstance(result, ActionBatchDecision):
            action = runtime.agent._stage_action_batch(result, surface.current_turn())
        else:
            action = result if isinstance(result, ActionDecision) else None
        content = (
            (
                f"Action batch accepted; releasing its first of {len(result.actions)} "
                "actions now. Each later action remains conditional on its public "
                "successor."
                if isinstance(result, ActionBatchDecision)
                else "Action accepted; this turn is complete."
            )
            if action
            else self._provider_tool_result(
                runtime,
                tool_call.name,
                str(result),
            )
        )
        runtime.event_manager.update(
            event_id,
            result=ToolResult(
                tool_call_id=tool_call.id,
                content=content,
                result_status=ResultStatus.COMPLETE,
            ),
        )
        return action

    def _provider_tool_result(
        self,
        runtime: RuntimeServices,
        tool_name: str,
        value: str,
    ) -> str:
        config = getattr(runtime.agent, "_native_tool_config", self.config)
        return bounded_context_text(
            value,
            limit=config.max_context_value_bytes,
            kind=f"{tool_name} result",
        )

    def _provider_tool_arguments(
        self,
        runtime: RuntimeServices,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        config = getattr(runtime.agent, "_native_tool_config", self.config)
        return bounded_tool_arguments(
            arguments,
            limit=config.max_context_value_bytes,
            tool_name=tool_name,
        )

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
            runtime.event_manager.add(
                Error(
                    content=_visible_tool_error(
                        surface,
                        stage="parse",
                        code="invalid_json",
                        message="Tool arguments are not valid JSON.",
                    )
                )
            )
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
            runtime.event_manager.add(
                Error(
                    content=_visible_tool_error(
                        surface,
                        stage="availability",
                        code="unavailable_tool",
                        message="That tool is not available in the current phase.",
                    )
                )
            )
            return None

    @staticmethod
    def _tool_block_reason(
        surface: WorkspaceToolSurface, name: str, block_action: bool
    ) -> str | None:
        action_names = {"take_actions"}
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
                "one tool batch with take_actions. Inspect that feedback, "
                "then choose "
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
                content=_visible_tool_error(
                    surface,
                    stage="policy",
                    code="policy_denied",
                    message=message,
                ),
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
