"""The persistent solver: a small NOOA agent over Tycho's exact workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from nooa import hidden, strategy
from nooa.agent import Agent
from nooa.agentdoc import hidden as hidden_field
from nooa.unifiedllm import UnifiedLLM
from tycho.workspace.agent_tools import ToolExecutor
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.builder import BuilderConfig
from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.protocol import ActionResultMessage
from tycho_nooa.strategy import NativeToolConfig, NativeToolStrategy
from tycho_nooa.tools import WorkspaceToolSurface
from tycho_nooa.trigger_builder import TriggerBuilderLoop


class TychoNooaAgent(Agent, context={"self": None, "state": None}):
    """You are one persistent investigator playing an unfamiliar grid game.

    Work in one durable game workspace and one growing conversation. Use the
    native file and Python tools to inspect evidence and preserve what you
    learn. Commit exactly one legal environment action per turn with
    take_action.
    """

    _workspace: Annotated[GameWorkspace, hidden_field]
    _tool_executor: Annotated[ToolExecutor, hidden_field]
    _tool_surface: Annotated[WorkspaceToolSurface, hidden_field]
    _native_tool_config: Annotated[NativeToolConfig, hidden_field]
    _trigger_builder: Annotated[TriggerBuilderLoop | None, hidden_field]

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        game_id: str,
        workspace_root: str | Path,
        available_actions: list[str],
        tool_trace_path: str | Path | None = None,
        render: bool = False,
        strategy_config: NativeToolConfig | None = None,
        builder_config: BuilderConfig | None = None,
        trigger_builder: bool = True,
        **kwargs: Any,
    ):
        super().__init__(llm=llm, **kwargs)
        self._workspace = GameWorkspace(
            game_id,
            root=str(workspace_root),
            render=render,
            available_actions=available_actions,
        )
        self._tool_executor = ToolExecutor(self._workspace)
        self._tool_surface = WorkspaceToolSurface(
            self._tool_executor,
            trace_path=Path(tool_trace_path) if tool_trace_path is not None else None,
        )
        self._native_tool_config = strategy_config or NativeToolConfig()
        trace_path = Path(tool_trace_path) if tool_trace_path is not None else None
        self._trigger_builder = (
            TriggerBuilderLoop(
                llm=llm,
                executor=self._tool_executor,
                trace_path=(
                    trace_path.with_name("trigger-builder-trace.jsonl")
                    if trace_path is not None
                    else None
                ),
                builder_tool_trace_path=(
                    trace_path.with_name("builder-tool-trace.jsonl")
                    if trace_path is not None
                    else None
                ),
                builder_config=builder_config,
            )
            if trigger_builder
            else None
        )

    @property
    @hidden
    def workspace_dir(self) -> Path:
        """Host-visible workspace location; never shown to the model."""
        return self._workspace.dir

    async def act(self, turn: GameTurn) -> ActionDecision:
        """Record one observation and obtain one environment commitment."""
        self._workspace.record(
            turn.grid,
            level=turn.level,
            turn_in_level=turn.turn_in_level,
            action=turn.previous_action,
            row=turn.previous_row,
            col=turn.previous_col,
            state=turn.state,
            available=turn.available_actions,
        )
        self._tool_surface.begin_turn(turn)
        exposure = (
            await self._trigger_builder.before_actor(turn)
            if self._trigger_builder is not None
            else None
        )
        self._tool_surface.set_builder_report(
            exposure.result.report if exposure is not None else None
        )
        try:
            decision = await self._choose_action(turn)
        except Exception as error:
            if self._trigger_builder is not None:
                self._trigger_builder.after_actor_error(
                    turn,
                    exposure,
                    error,
                    self._tool_surface.turn_tool_calls(),
                )
            raise
        if self._trigger_builder is not None:
            self._trigger_builder.after_actor(
                turn,
                exposure,
                decision,
                self._tool_surface.turn_tool_calls(),
            )
        return decision

    def observe(self, result: ActionResultMessage) -> None:
        """Preserve action outcomes that may never become another decision turn."""
        decision = result.decision
        before = result.before
        after = result.after
        if after.state.upper() == "GAME_OVER":
            self._workspace.record_game_over(
                level=result.level,
                turn_in_level=result.turn_in_level,
                action=decision.action,
                row=decision.row,
                col=decision.col,
                prev_grid=before.grid,
                game_over_grid=after.grid,
            )
        if after.levels_completed > before.levels_completed:
            self._workspace.record_solved(
                result.level,
                decision.action,
                row=decision.row,
                col=decision.col,
            )
            self._workspace.new_level()
        elif after.state.upper() == "WIN":
            self._workspace.record_terminal(
                result.level,
                after.grid,
                action=decision.action,
                row=decision.row,
                col=decision.col,
            )
            self._workspace.record_solved(
                result.level,
                decision.action,
                row=decision.row,
                col=decision.col,
            )
        elif after.full_reset:
            self._workspace.reset_level(result.level, reason="public_full_reset")

    @strategy(NativeToolStrategy())
    async def _choose_action(self, turn: GameTurn) -> ActionDecision:
        """Investigate this observation and commit the next legal action."""
        ...  # noqa: PIE790 - NOOA uses ellipsis as its generation-method marker
