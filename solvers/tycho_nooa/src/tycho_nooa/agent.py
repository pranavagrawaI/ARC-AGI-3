"""The persistent solver: a small NOOA agent over Tycho's exact workspace."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any

from nooa import hidden, strategy
from nooa.agent import Agent
from nooa.agentdoc import hidden as hidden_field
from nooa.unifiedllm import UnifiedLLM
from tycho.workspace.agent_tools import ToolExecutor
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.builder import BuilderConfig
from tycho_nooa.deadlines import DecisionDeadline, DecisionDeadlineConfig
from tycho_nooa.decision_checkpoints import DecisionCheckpointRecorder
from tycho_nooa.epistemic import (
    TYPED_GOAL_PROMPT_BRIEF,
    GoalLedger,
    GoalToolSurface,
    RequestBuilderArguments,
)
from tycho_nooa.llm import generation_max_tokens
from tycho_nooa.llm_trace import LLMTrace
from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.modes import ModePolicy, mode_from_env, world_model_enabled
from tycho_nooa.protocol import ActionResultMessage
from tycho_nooa.scribe import ScribeConfig, ScribeStrategy
from tycho_nooa.strategy import NativeToolConfig, NativeToolStrategy
from tycho_nooa.symmetry_notice import PROMPT_BRIEF
from tycho_nooa.tools import WorkspaceToolSurface, actor_tool_profile
from tycho_nooa.treatments import TreatmentPolicy, treatment_from_env
from tycho_nooa.trigger_builder import BuilderExposure, TriggerBuilderLoop
from tycho_nooa.wm_probe import GatedToolExecutor

LEVEL_COMPLETE_RECORD = "level_{level}/level_complete.json"


def _seed_symmetry(workspace_dir: Path) -> None:
    """Write `symmetry.py` next to Tycho's `wmlib.py`.

    Kept a separate file rather than folded into wmlib because wmlib is upstream's:
    `GameWorkspace._seed_worldmodel` rewrites it from `wmlib_template.py` on every fresh
    start, so anything merged in would be overwritten or would fork 1,627 vendored lines.
    Nothing is lost by the split — the two things the model needs, that the module is
    preloaded and that it is described in run_python's tool text, are both arranged below.
    """
    (workspace_dir / "symmetry.py").write_text(
        (Path(__file__).with_name("symmetry_template.py")).read_text()
    )


def _sibling_trace(path: Path | None, name: str) -> Path | None:
    return path.with_name(name) if path is not None else None


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
    _scribe_config: Annotated[ScribeConfig, hidden_field]
    _trigger_builder: Annotated[TriggerBuilderLoop | None, hidden_field]
    _mode: Annotated[ModePolicy, hidden_field]
    _treatment: Annotated[TreatmentPolicy, hidden_field]
    _goal_ledger: Annotated[GoalLedger | None, hidden_field]
    _llm_trace: Annotated[LLMTrace, hidden_field]
    _decision_checkpoints: Annotated[DecisionCheckpointRecorder, hidden_field]
    _deadline_config: Annotated[DecisionDeadlineConfig, hidden_field]
    _active_deadline: Annotated[DecisionDeadline | None, hidden_field]
    _actor_requested_exposures: Annotated[list[BuilderExposure], hidden_field]

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
        scribe_config: ScribeConfig | None = None,
        builder_config: BuilderConfig | None = None,
        deadline_config: DecisionDeadlineConfig | None = None,
        trigger_builder: bool | None = None,
        resume_workspace: bool = False,
        previous_turn: GameTurn | None = None,
        **kwargs: Any,
    ):
        super().__init__(llm=llm, **kwargs)
        self._mode = mode_from_env()
        self._treatment = treatment_from_env(self._mode)
        # Upstream ties these two to `wm_variant != "none"` together: the baseline
        # seeds no world_model.py/verify.py/plan.py, so there is nothing for the
        # automatic feedback probe (and its goal gate) to run against.
        wm_enabled = world_model_enabled(self._mode)
        self._workspace = GameWorkspace(
            game_id,
            root=str(workspace_root),
            render=render,
            available_actions=available_actions,
            seed_world_model=wm_enabled,
            resume=resume_workspace,
        )
        if not resume_workspace:
            _seed_symmetry(self._workspace.dir)
        # One executor for both the actor's tool surface and the trigger builder's,
        # so the goal gate reaches every automatic world-model probe either issues.
        self._tool_executor = GatedToolExecutor(
            self._workspace, wm_feedback_enabled=wm_enabled
        )
        trace_path = Path(tool_trace_path) if tool_trace_path is not None else None
        self._goal_ledger = (
            GoalLedger(self._workspace.dir)
            if self._treatment.typed_candidate_goals
            else None
        )
        self._tool_surface = WorkspaceToolSurface(
            self._tool_executor,
            profile=actor_tool_profile(self._mode),
            mode_policy=self._mode,
            goal_tools=(
                GoalToolSurface(self._goal_ledger)
                if self._goal_ledger is not None
                else None
            ),
            trace_path=trace_path,
        )
        max_tokens = generation_max_tokens()
        self._native_tool_config = strategy_config or NativeToolConfig(
            max_tokens=max_tokens
        )
        self._scribe_config = scribe_config or ScribeConfig(max_tokens=max_tokens)
        if builder_config is None:
            builder_config = BuilderConfig(
                max_tool_steps=(
                    self._treatment.builder_max_tool_steps
                    if self._treatment.builder_max_tool_steps is not None
                    else BuilderConfig.max_tool_steps
                ),
                max_tokens=max_tokens,
            )
        self._deadline_config = deadline_config or DecisionDeadlineConfig()
        # One trace across every agent sharing this client: the actor and scribe
        # here, each per-fire builder as the trigger loop creates it.
        self._llm_trace = LLMTrace(_sibling_trace(trace_path, "llm-trace.jsonl"))
        self._llm_trace.attach(self.event_manager, source="actor")
        self._decision_checkpoints = DecisionCheckpointRecorder(
            _sibling_trace(trace_path, "decision-checkpoints"),
            workspace_dir=self._workspace.dir,
            llm=llm,
        )
        self._decision_checkpoints.attach(self.event_manager, source="actor")
        self._trigger_builder = (
            TriggerBuilderLoop(
                llm=llm,
                executor=self._tool_executor,
                trace_path=_sibling_trace(trace_path, "trigger-builder-trace.jsonl"),
                builder_tool_trace_path=_sibling_trace(
                    trace_path, "builder-tool-trace.jsonl"
                ),
                builder_config=builder_config,
                llm_trace=self._llm_trace,
                decision_checkpoints=self._decision_checkpoints,
            )
            if (self._mode.needs_builder if trigger_builder is None else trigger_builder)
            else None
        )
        self._active_deadline = None
        self._actor_requested_exposures = []
        if previous_turn is not None:
            if not resume_workspace:
                raise ValueError("previous_turn is only valid for workspace resume")
            self._workspace.prev_grid = previous_turn.grid
            self._tool_surface.begin_turn(previous_turn)

    def _resolve_system_prompt(self) -> str:
        """This class's docstring, plus the toolbox the docstring cannot mention.

        NOOA builds the actor's system prompt from the docstring above, which is written for a
        reader of this file. `symmetry` is the one capability the actor has no other way to
        learn about at the moment it needs it: the run_python description carries it too, but a
        tool description is read when choosing a tool, not when forming a hypothesis.
        """
        treatment_prompt = (
            f"\n\n{TYPED_GOAL_PROMPT_BRIEF}"
            if self._treatment.typed_candidate_goals
            else ""
        )
        return (
            f"{super()._resolve_system_prompt()}"
            f"{treatment_prompt}\n\n{PROMPT_BRIEF}"
        )

    @property
    @hidden
    def workspace_dir(self) -> Path:
        """Host-visible workspace location; never shown to the model."""
        return self._workspace.dir

    @property
    @hidden
    def mode_name(self) -> str:
        return self._mode.name

    @property
    @hidden
    def treatment_name(self) -> str:
        return self._treatment.name

    async def act(
        self,
        turn: GameTurn,
        *,
        decision_timeout_seconds: float | None = None,
    ) -> ActionDecision:
        """Record one observation and obtain one environment commitment."""
        started_at = time.monotonic()
        deadline = (
            self._deadline_config.start(
                decision_timeout_seconds,
                now=started_at,
            )
            if decision_timeout_seconds is not None
            else None
        )
        self._decision_checkpoints.set_decision_point(
            sequence=turn.sequence,
            level=turn.level,
            turn_in_level=turn.turn_in_level,
            phase="decision",
        )
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
        self._register_evidence(
            f"level_{turn.level}/turn_{turn.turn_in_level:03d}.txt"
        )
        self._register_evidence(
            f"level_{turn.level}/turn_{turn.turn_in_level:03d}.json"
        )
        self._tool_surface.begin_turn(turn)
        exposure = (
            await self._trigger_builder.before_actor(turn, deadline=deadline)
            if self._trigger_builder is not None
            and self._mode.harness_fires_builder
            else None
        )
        self._tool_surface.set_builder_report(
            exposure.result.report if exposure is not None else None
        )
        self._active_deadline = deadline
        self._actor_requested_exposures = []
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
                for requested in self._actor_requested_exposures:
                    self._trigger_builder.after_actor_error(
                        turn,
                        requested,
                        error,
                        self._tool_surface.turn_tool_calls(),
                    )
            raise
        finally:
            self._active_deadline = None
        if self._trigger_builder is not None:
            self._trigger_builder.after_actor(
                turn,
                exposure,
                decision,
                self._tool_surface.turn_tool_calls(),
            )
            for requested in self._actor_requested_exposures:
                self._trigger_builder.after_actor(
                    turn,
                    requested,
                    decision,
                    self._tool_surface.turn_tool_calls(),
                )
        return decision

    async def invoke_builder(self, turn: GameTurn, reason: str) -> str:
        """Actor-pulled Tycho builder; its report is advisory and cannot act."""
        if not self._mode.include_builder or self._trigger_builder is None:
            raise RuntimeError("invoke_builder is unavailable outside orchestrator mode")
        exposure = await self._trigger_builder.actor_requested(
            turn,
            reason,
            deadline=self._active_deadline,
            focus=None,
        )
        self._actor_requested_exposures.append(exposure)
        return exposure.result.report

    async def request_builder(
        self,
        turn: GameTurn,
        arguments: RequestBuilderArguments,
    ) -> str:
        """Typed treatment request whose evidence is checked before construction."""
        goal_tools = self._tool_surface.goal_tools
        if goal_tools is None:
            raise RuntimeError(
                "request_builder is unavailable outside typed_candidate_goals"
            )
        if not self._mode.include_builder or self._trigger_builder is None:
            raise RuntimeError("request_builder is unavailable outside orchestrator mode")
        reason, focus = goal_tools.builder_request(arguments)
        try:
            exposure = await self._trigger_builder.actor_requested(
                turn,
                reason,
                deadline=self._active_deadline,
                focus=focus,
            )
        except Exception as error:
            if focus != "objective":
                raise
            report = f"{type(error).__name__}: {error}"
            review = goal_tools.record_builder_review(
                arguments,
                report=report,
                unavailable=True,
            )
            return json.dumps(
                {
                    "objective_review": review.model_dump(mode="json"),
                    "builder": {"status": "unavailable", "report": report},
                    "required_next_step": goal_tools.required_before_action(),
                },
                separators=(",", ":"),
            )
        self._actor_requested_exposures.append(exposure)
        if focus != "objective":
            return exposure.result.report
        review = goal_tools.record_builder_review(
            arguments,
            report=exposure.result.report,
        )
        return json.dumps(
            {
                "objective_review": review.model_dump(mode="json"),
                "builder": {
                    "status": "complete",
                    "report": exposure.result.report,
                    "recommended_action": (
                        exposure.recommended_action.model_dump(mode="json")
                        if exposure.recommended_action is not None
                        else None
                    ),
                    "recommendation_parse_error": (
                        exposure.recommendation_parse_error
                    ),
                },
                "required_next_step": goal_tools.required_before_action(),
            },
            separators=(",", ":"),
        )

    async def observe(self, result: ActionResultMessage) -> None:
        """Preserve action outcomes that may never become another decision turn."""
        self._decision_checkpoints.set_decision_point(
            sequence=result.sequence,
            level=result.level,
            turn_in_level=result.turn_in_level,
            phase="post_action",
        )
        decision = result.decision
        before = result.before
        after = result.after
        self._record_event_frames(result)
        if after.state.upper() == "GAME_OVER":
            self._workspace.record_game_over(
                level=result.level,
                turn_in_level=result.turn_in_level,
                action=decision.action,
                row=decision.row,
                col=decision.col,
                prev_grid=before.grid,
                game_over_grid=after.terminal_frame or after.grid,
            )
        if after.levels_completed > before.levels_completed:
            if after.terminal_frame is not None:
                self._workspace.record_terminal(
                    result.level,
                    after.terminal_frame,
                    action=decision.action,
                    row=decision.row,
                    col=decision.col,
                )
                self._register_evidence(f"level_{result.level}/terminal.json")
                self._register_evidence(f"level_{result.level}/terminal.txt")
            completion_evidence = {
                "source": "arc-host action_result",
                "outcome": "level_complete",
                "action": decision.action,
                "row": decision.row,
                "col": decision.col,
                "levels_completed_before": before.levels_completed,
                "levels_completed_after": after.levels_completed,
                "after_observation": (
                    "next_level_start"
                    if after.state.upper() == "NOT_FINISHED"
                    else "public_action_result"
                ),
            }
            self._workspace.write_file(
                LEVEL_COMPLETE_RECORD.format(level=result.level),
                json.dumps(completion_evidence, indent=2),
            )
            self._register_evidence(LEVEL_COMPLETE_RECORD.format(level=result.level))
            self._workspace.record_solved(
                result.level,
                decision.action,
                row=decision.row,
                col=decision.col,
            )
            if self._trigger_builder is not None and self._mode.harness_fires_builder:
                await self._trigger_builder.after_level_complete(
                    GameTurn(
                        grid=after.grid,
                        available_actions=after.available_actions,
                        sequence=result.sequence,
                        level=result.level,
                        turn_in_level=result.turn_in_level + 1,
                        levels_completed=after.levels_completed,
                        state=after.state,
                        previous_action=decision.action,
                        previous_row=decision.row,
                        previous_col=decision.col,
                    ),
                    json.dumps(
                        completion_evidence, separators=(",", ":"), sort_keys=True
                    ),
                )
            # Fires (Tycho's Event.LEVEL_COMPLETE) BEFORE the workspace advances,
            # so the scribe still sees this level's own directory as current.
            await self._summarize_level(result.level)
            self._workspace.new_level()
        elif after.state.upper() == "WIN":
            self._workspace.record_terminal(
                result.level,
                after.terminal_frame or after.grid,
                action=decision.action,
                row=decision.row,
                col=decision.col,
            )
            self._register_evidence(f"level_{result.level}/terminal.json")
            self._register_evidence(f"level_{result.level}/terminal.txt")
            self._workspace.record_solved(
                result.level,
                decision.action,
                row=decision.row,
                col=decision.col,
            )
            await self._summarize_level(result.level)
        elif after.full_reset:
            self._workspace.reset_level(result.level, reason="public_full_reset")

    def _record_event_frames(self, result: ActionResultMessage) -> None:
        """Persist transient public evidence before any level/reset lifecycle change."""
        after = result.after
        if not after.event_frames:
            return
        decision = result.decision
        if after.state.upper() == "GAME_OVER":
            terminal = "game_over"
        elif after.levels_completed > result.before.levels_completed or (
            after.state.upper() == "WIN"
        ):
            terminal = "level_complete"
        else:
            terminal = "nonterminal"
        source_indices = [event.source_index for event in after.event_frames]
        event = self._workspace.record_animation_event(
            level=result.level,
            turn_in_level=result.turn_in_level,
            action=decision.action,
            row=decision.row,
            col=decision.col,
            terminal=terminal,
            frames=[event.frame for event in after.event_frames],
            selected_indices=range(len(after.event_frames)),
            decision={
                "source_frame_indices": source_indices,
                "frame_selection_provenance": after.provenance.model_dump(),
            },
            skipped="host-retained public event frames; no summarizer run",
        )
        if event is None:
            return
        directory = event.get("directory")
        if isinstance(directory, str):
            event_dir = self._workspace.dir / directory
            for artifact in event_dir.iterdir():
                if artifact.is_file():
                    self._register_evidence(artifact.relative_to(self._workspace.dir).as_posix())

    def _register_evidence(self, relative_path: str) -> None:
        if self._goal_ledger is not None:
            self._goal_ledger.register_authoritative_evidence(relative_path)

    @strategy(NativeToolStrategy())
    async def _choose_action(self, turn: GameTurn) -> ActionDecision:
        """Investigate this observation and commit the next legal action."""
        ...  # noqa: PIE790 - NOOA uses ellipsis as its generation-method marker

    @strategy(ScribeStrategy())
    async def _summarize_level(self, level: int) -> None:
        """Consolidate the just-completed level's terminal evidence before the
        next level's transcript reuses this same growing history."""
        ...  # noqa: PIE790 - NOOA uses ellipsis as its generation-method marker
