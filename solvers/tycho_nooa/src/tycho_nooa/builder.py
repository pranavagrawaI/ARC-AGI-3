"""A fresh, bounded NOOA builder over Tycho's exact workspace contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Annotated, Any, Literal

from nooa import strategy
from nooa.agent import Agent
from nooa.agentdoc import hidden as hidden_field
from nooa.context_blocks import ResultStatus, ToolCallEvent, ToolResult
from nooa.events import Error, Task
from nooa.strategies.base import GenerationStrategy, RuntimeServices
from nooa.unifiedllm import UnifiedLLM
from tycho.agent.builder import REPORT_FILE, _verify_state
from tycho.agent.colors import COLORMAP
from tycho.prompts.render import render_prompt
from tycho.workspace.agent_tools import ToolExecutor, canonical

from tycho_nooa.history import replayed_reasoning, retain_assistant_turn
from tycho_nooa.models import GameTurn
from tycho_nooa.optionality import OPTIONALITY_PROMPT_BRIEF
from tycho_nooa.runtime_notice import prompt_brief
from tycho_nooa.theory_transaction import TheoryTransaction
from tycho_nooa.tools import BUILDER_TOOL_PROFILE, WorkspaceToolSurface, tool_by_name

if TYPE_CHECKING:
    from nooa.strategies.current_call import CurrentCall


WORLD_MODEL_PY = "world_model.py"
WORLD_MODEL_NOTES = "notes/world_model.md"
ARTIFACT_PATHS = (
    WORLD_MODEL_PY,
    WORLD_MODEL_NOTES,
    REPORT_FILE,
)
WORLD_MODEL_CHECKPOINT_REQUEST = (
    f"Before this pass ends, checkpoint your current understanding into "
    f"{WORLD_MODEL_NOTES} with write_file or edit_file. Do this even if you have no "
    "confirmed model change: record what you tried, what you ruled out, and what "
    "remains open, so the next builder invocation is not reading the same stale "
    "notes it started from."
)
RETAINED_CANDIDATE_BRIEF = """RETAINED NONCANONICAL CANDIDATES

If candidates/world_model/index.json exists, it indexes model bodies preserved from
builder passes that exceeded their deadline. Inspect a candidate when it is relevant to
the present discrepancy instead of reconstructing it from memory. These files are
evidence, not authority: they are never canonical, publication-approved, or
planner-eligible. Only a completed, verified edit to world_model.py changes the current
model.
"""
TRANSACTIONAL_CANDIDATE_BRIEF = """RETAINED NONCANONICAL CANDIDATES

The host preserves every assessed world_model.py body outside your writable workspace.
When a transaction result names a relevant record ID, use
checkout_theory_candidate(record_id) instead of reconstructing its source through file
copying. Checkout re-evaluates the immutable body and grants no publication authority.
"""
FINAL_REPORT_REQUEST = (
    "Step budget reached. Do NOT call any more tools. Right now, in your reply, give "
    "the compact report in the required format (confidence/model/dynamics/outcome/"
    "outcome_verified/plan/recommended_action+subgoal/note) summarizing what you "
    "learned and the state of world_model.py — even if it's 'still exploring, model "
    "not yet coded'. Be concrete about what you DID figure out — especially any open "
    "outcome hypotheses and the action that would disambiguate them — so the actor can "
    "use it."
)
OBJECTIVE_REVIEW_REQUEST = """OBJECTIVE REVIEW (candidate remains provisional)

Independently inspect the cited public evidence and evaluate the candidate objective in
the request. Do not promote it into world_model.py outcome() during this review. Report
the strongest discriminating evidence, a falsifying probe, and any route the actor could
execute one action at a time. Your report must include these exact fields:

candidate_goal_assessment: supported | contradicted | inconclusive
candidate_goal_reason: concise evidence-based reason
candidate_goal_route: route text, or none
recommended_action: one currently legal action, or none
"""
OBJECTIVE_FINAL_REPORT_REQUEST = (
    "Step budget reached. Do NOT call more tools. Return the compact objective review "
    "now. Include candidate_goal_assessment, candidate_goal_reason, "
    "candidate_goal_route, and recommended_action exactly as requested, even when the "
    "assessment is inconclusive."
)
FALLBACK_REPORT = (
    "confidence: none\n"
    "model: none — builder did not finish a model in its step budget\n"
    "dynamics: n/a\n"
    "outcome: unknown\n"
    "outcome_verified: no-terminal-yet\n"
    "plan: start=current; validated=no; length=none; assumptions=unknown\n"
    "recommended_action: none   subgoal: none\n"
    "note: keep exploring; re-invoke with more specific mechanics once you have them"
)
OUTCOME_CHECKPOINT_REQUEST = """HOST-CONFIRMED OUTCOME CHECKPOINT (mandatory)

The ARC host, not an actor hypothesis, observed a public levels_completed increase:
{evidence}

The verifier still says the just-completed transition is missed by outcome(). Repair that
specific false negative before exploring anything else. Preserve current non-terminal
behavior and avoid game-specific constants. Re-run verification after editing
world_model.py. Do not investigate current-level dynamics first while this checkpoint is
unresolved."""
OUTCOME_CHECKPOINT_REFUSAL = (
    "The host-confirmed outcome checkpoint is unresolved. A workspace tool call is "
    "required; prose cannot end this repair prefix."
)
COMMIT_CALLS_REMAINING = 3
COMMIT_SECONDS_REMAINING = 90
COMMIT_TOOL_NAMES = frozenset(
    {"write_file", "edit_file", "edit_function", "checkout_theory_candidate"}
)


@dataclass(frozen=True)
class BuilderConfig:
    max_tool_steps: int = 40
    max_tokens: int = 16_000
    temperature: float | None = None
    show_budget_telemetry: bool = False
    commit_seconds_remaining: int = COMMIT_SECONDS_REMAINING


@dataclass(frozen=True)
class BuilderRequest:
    turn: GameTurn
    reason: str
    outcome_evidence: str | None = None
    focus: str | None = None
    deadline_at: float | None = None
    phase: Literal["decision", "post_action"] = "decision"


def _builder_budget_notice(
    request: BuilderRequest,
    *,
    repair_calls_remaining: int,
    commit_seconds_remaining: int = COMMIT_SECONDS_REMAINING,
) -> str:
    notice, _ = _builder_budget(
        request,
        repair_calls_remaining=repair_calls_remaining,
        commit_seconds_remaining=commit_seconds_remaining,
    )
    return notice


def _builder_budget(
    request: BuilderRequest,
    *,
    repair_calls_remaining: int,
    commit_seconds_remaining: int,
) -> tuple[str, bool]:
    seconds_remaining = (
        None
        if request.deadline_at is None
        else max(0, math.floor(request.deadline_at - monotonic()))
    )
    commit = repair_calls_remaining <= COMMIT_CALLS_REMAINING or (
        seconds_remaining is not None
        and seconds_remaining <= commit_seconds_remaining
    )
    time_text = (
        "unavailable"
        if seconds_remaining is None
        else f"{seconds_remaining} seconds"
    )
    if commit:
        mode = "commit"
        instruction = (
            "Make the smallest reversible edit now and predict its verifier effect. "
            "Investigation tools are unavailable in this mode. If no falsifiable "
            "edit is possible, checkpoint the exact missing evidence."
        )
    else:
        mode = "investigate"
        instruction = (
            "Investigate the named divergence, but externalize one falsifiable "
            "hypothesis before commitment mode."
        )
    notice = (
        "BUILDER BUDGET (host-authoritative; supersedes earlier notices)\n"
        f"Time remaining: {time_text}\n"
        "Repair model calls remaining (including this call): "
        f"{repair_calls_remaining}\n"
        "Finishing model calls reserved: up to 2\n"
        f"Mode: {mode}\n"
        f"{instruction}"
    )
    return notice, commit


@dataclass(frozen=True)
class BuilderResult:
    report: str
    report_fresh: bool
    report_forced: bool
    world_model_checkpoint_forced: bool
    outcome_checkpoint_required: bool
    outcome_checkpoint_resolved: bool
    artifact_hashes: dict[str, dict[str, str | bool | None]]
    tool_calls: list[dict[str, Any]]


def verify_state(executor: ToolExecutor) -> tuple[str, tuple[int, int] | None]:
    """Call Tycho's pinned, auto-attached builder verifier unchanged."""
    return _verify_state(executor)


@dataclass
class OutcomeCheckpoint:
    """Mandatory repair state derived from host-confirmed terminal evidence."""

    evidence: str | None
    resolved: bool

    @classmethod
    def begin(cls, evidence: str | None, verifier_text: str) -> OutcomeCheckpoint:
        required = evidence is not None
        return cls(
            evidence=evidence,
            resolved=not required or not _misses_completed_level(verifier_text),
        )

    @property
    def required(self) -> bool:
        return self.evidence is not None

    def request_repair(self, runtime: RuntimeServices) -> None:
        if self.required and not self.resolved:
            runtime.event_manager.add(
                Task(prompt=OUTCOME_CHECKPOINT_REQUEST.format(evidence=self.evidence))
            )

    def reject_prose_exit(self, runtime: RuntimeServices) -> bool:
        if not self.required or self.resolved:
            return False
        runtime.event_manager.add(Error(content=OUTCOME_CHECKPOINT_REFUSAL))
        return True

    def inspect_edits(
        self,
        runtime: RuntimeServices,
        executor: ToolExecutor,
        tool_calls: list[Any],
    ) -> None:
        if not self.required or self.resolved or not _edits_world_model(tool_calls):
            return
        verifier_text, _ = verify_state(executor)
        self.resolved = not _misses_completed_level(verifier_text)
        runtime.event_manager.add(Task(prompt=verifier_text))
        self.request_repair(runtime)


class BuilderStrategy(GenerationStrategy):
    """Run only Tycho workspace tools, then guarantee a fresh compact report."""

    @property
    def name(self) -> str:
        return "TYCHO_TRIGGER_BUILDER"

    def get_block_overrides(self) -> dict[str, str]:
        return {}

    async def execute(
        self,
        runtime: RuntimeServices,
        call: CurrentCall,
    ) -> tuple[str, bool, bool, bool, bool]:
        builder = self._builder(runtime)
        request = call.bound_parameters().get("request")
        if not isinstance(request, BuilderRequest):
            raise TypeError("BuilderStrategy requires a BuilderRequest named 'request'")

        surface = builder._tool_surface
        config = builder._builder_config
        tools = surface.native_tools()
        tools_by_name = tool_by_name(tools)
        kickoff = render_prompt(
            "builder.user",
            level=request.turn.level,
            hint=request.reason,
            available_actions=request.turn.available_actions,
            report_file=REPORT_FILE,
        )
        verifier_text, _ = verify_state(builder._executor)
        runtime.event_manager.add(Task(prompt=kickoff))
        objective_review = request.focus == "objective"
        runtime.event_manager.add(
            Task(prompt=OBJECTIVE_REVIEW_REQUEST if objective_review else verifier_text)
        )
        checkpoint = OutcomeCheckpoint.begin(request.outcome_evidence, verifier_text)
        checkpoint.request_repair(runtime)
        await self._construction_loop(
            runtime,
            builder,
            request,
            surface,
            config,
            tools,
            tools_by_name,
            checkpoint,
            objective_review=objective_review,
        )
        checkpoint_forced = (
            False
            if objective_review
            else await self._enforce_world_model_checkpoint(
                runtime, builder, surface, tools_by_name, tools, config
            )
        )
        return await self._finish_report(
            runtime,
            builder,
            tools,
            config,
            checkpoint_forced,
            checkpoint,
            objective_review=objective_review,
        )

    async def _construction_loop(
        self,
        runtime: RuntimeServices,
        builder: FocusedBuilder,
        request: BuilderRequest,
        surface: WorkspaceToolSurface,
        config: BuilderConfig,
        tools: list[Any],
        tools_by_name: dict[str, Any],
        checkpoint: OutcomeCheckpoint,
        *,
        objective_review: bool,
    ) -> None:
        for round_index in range(config.max_tool_steps):
            available_tools, commit_mode = self._prepare_builder_round(
                runtime,
                request,
                config,
                tools,
                checkpoint,
                round_index=round_index,
                objective_review=objective_review,
            )
            available_tools_by_name = tool_by_name(available_tools)
            response, event_id = await runtime.generate(
                tools=available_tools,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                tool_choice="required" if commit_mode else "auto",
            )
            decision = self._construction_decision(runtime, checkpoint, response)
            if decision == "continue":
                continue
            if decision == "stop":
                return
            self._execute_builder_turn(
                runtime,
                builder,
                surface,
                available_tools_by_name,
                response,
                event_id,
                objective_review=objective_review,
                checkpoint=checkpoint,
            )

    @staticmethod
    def _prepare_builder_round(
        runtime: RuntimeServices,
        request: BuilderRequest,
        config: BuilderConfig,
        tools: list[Any],
        checkpoint: OutcomeCheckpoint,
        *,
        round_index: int,
        objective_review: bool,
    ) -> tuple[list[Any], bool]:
        if not config.show_budget_telemetry:
            return tools, False
        notice, commit_mode = _builder_budget(
            request,
            repair_calls_remaining=config.max_tool_steps - round_index,
            commit_seconds_remaining=config.commit_seconds_remaining,
        )
        runtime.event_manager.add(Task(prompt=notice))
        # Keep telemetry fresh without displacing the assignment as the final,
        # highest-priority instruction visible to the model.
        if objective_review:
            runtime.event_manager.add(Task(prompt=OBJECTIVE_REVIEW_REQUEST))
        checkpoint.request_repair(runtime)
        if not commit_mode:
            return tools, False
        available = [
            tool for tool in tools if canonical(tool.name) in COMMIT_TOOL_NAMES
        ]
        return available, True

    @staticmethod
    def _construction_decision(
        runtime: RuntimeServices,
        checkpoint: OutcomeCheckpoint,
        response: Any,
    ) -> Literal["execute", "continue", "stop"]:
        if response.tool_calls:
            return "execute"
        if response.finish_reason == "length":
            runtime.event_manager.add(
                Error(
                    content=(
                        "Builder response was truncated before a tool call; "
                        "construction is still active. Use the next call to "
                        "externalize the smallest executable edit."
                    )
                )
            )
            return "continue"
        if checkpoint.reject_prose_exit(runtime):
            return "continue"
        return "stop"

    def _execute_builder_turn(
        self,
        runtime: RuntimeServices,
        builder: FocusedBuilder,
        surface: WorkspaceToolSurface,
        available_tools_by_name: dict[str, Any],
        response: Any,
        event_id: Any,
        *,
        objective_review: bool,
        checkpoint: OutcomeCheckpoint,
    ) -> None:
        retain_assistant_turn(runtime.event_manager, event_id, response)
        reasoning = replayed_reasoning(response)
        for index, tool_call in enumerate(response.tool_calls):
            self._execute_tool_call(
                runtime,
                surface,
                available_tools_by_name,
                tool_call,
                reasoning=reasoning if index == 0 else None,
                allow_world_model_edit=not objective_review,
            )
        checkpoint.inspect_edits(runtime, builder._executor, response.tool_calls)

    async def _finish_report(
        self,
        runtime: RuntimeServices,
        builder: FocusedBuilder,
        tools: list[Any],
        config: BuilderConfig,
        checkpoint_forced: bool,
        checkpoint: OutcomeCheckpoint,
        *,
        objective_review: bool,
    ) -> tuple[str, bool, bool, bool, bool]:
        report = self._fresh_report(builder, objective_review)
        if report is not None:
            return (
                report,
                False,
                checkpoint_forced,
                checkpoint.required,
                checkpoint.resolved,
            )

        runtime.event_manager.add(
            Task(
                prompt=(
                    OBJECTIVE_FINAL_REPORT_REQUEST
                    if objective_review
                    else FINAL_REPORT_REQUEST
                )
            )
        )
        # Tycho's own forced-report call (builder.py's post-loop fallback) still
        # passes its tools_spec — it never hard-blocks the API from offering
        # tools — but its handling of that final reply only reads reply["text"]
        # and never inspects reply["tool_calls"]. Match both halves: keep tools
        # nominally available, and still only read `content` below, so any
        # tool-call attempt on this turn is not executed. Record the rejection
        # explicitly so the durable request/result graph remains total.
        response, _ = await runtime.generate(
            tools=tools,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            tool_choice="auto",
        )
        self._reject_final_tool_calls(builder, response.tool_calls)
        text = str(response.content or "").strip() or FALLBACK_REPORT
        builder._executor.ws.write_file(REPORT_FILE, text)
        if _read_report(builder._executor.ws.dir) != text:
            raise RuntimeError(f"failed to persist mandatory fresh {REPORT_FILE}")
        return text, True, checkpoint_forced, checkpoint.required, checkpoint.resolved

    @staticmethod
    def _fresh_report(builder: FocusedBuilder, objective_review: bool) -> str | None:
        report = _read_report(builder._executor.ws.dir)
        if report is None or report == builder._report_before:
            return None
        if objective_review and not _objective_report_complete(report):
            return None
        return report

    @staticmethod
    def _reject_final_tool_calls(
        builder: FocusedBuilder, tool_calls: list[Any]
    ) -> None:
        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call.arguments)
            except json.JSONDecodeError:
                arguments = {"raw_arguments": tool_call.arguments}
            builder._tool_surface.record_rejected_call(
                tool_call.name,
                arguments,
                "Builder final-report phase accepts prose only; tool call was not executed",
                tool_call_id=tool_call.id,
                raw_arguments=tool_call.arguments,
                rejection_stage="policy",
            )

    async def _enforce_world_model_checkpoint(
        self,
        runtime: RuntimeServices,
        builder: FocusedBuilder,
        surface: WorkspaceToolSurface,
        tools_by_name: dict[str, Any],
        tools: list[Any],
        config: BuilderConfig,
    ) -> bool:
        """Guarantee notes/world_model.md advances before this pass can end.

        DIAGNOSIS.md's first live run found this file read on every one of 12
        divergence-triggered passes but never advanced — each pass repeated the
        same stale turn-1 scratchpad. Give the model one real chance to write a
        genuine update (tool calls here are executed, unlike the forced-report
        call below); if it still leaves the file unchanged, force a minimal
        breadcrumb so the invariant holds unconditionally rather than trusting
        the model to remember.
        """
        workspace_dir = builder._executor.ws.dir
        before = builder._world_model_before
        if _read_optional(workspace_dir, WORLD_MODEL_NOTES) != before:
            return False

        runtime.event_manager.add(Task(prompt=WORLD_MODEL_CHECKPOINT_REQUEST))
        response, event_id = await runtime.generate(
            tools=tools,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            tool_choice="auto",
        )
        if response.tool_calls:
            retain_assistant_turn(runtime.event_manager, event_id, response)
            for index, tool_call in enumerate(response.tool_calls):
                self._execute_tool_call(
                    runtime,
                    surface,
                    tools_by_name,
                    tool_call,
                    reasoning=replayed_reasoning(response) if index == 0 else None,
                )

        if _read_optional(workspace_dir, WORLD_MODEL_NOTES) != before:
            return True

        fallback = _fallback_world_model_checkpoint(before)
        builder._executor.ws.write_file(WORLD_MODEL_NOTES, fallback)
        if _read_optional(workspace_dir, WORLD_MODEL_NOTES) != fallback:
            raise RuntimeError(
                f"failed to persist mandatory {WORLD_MODEL_NOTES} checkpoint"
            )
        return True

    @staticmethod
    def _execute_tool_call(
        runtime: RuntimeServices,
        surface: WorkspaceToolSurface,
        tools_by_name: dict[str, Any],
        tool_call: Any,
        *,
        reasoning: str | None = None,
        allow_world_model_edit: bool = True,
    ) -> None:
        validated = BuilderStrategy._validated_tool_call(
            runtime, surface, tools_by_name, tool_call
        )
        if validated is None:
            return
        arguments, tool = validated

        event_id = runtime.event_manager.add(
            ToolCallEvent(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                arguments=arguments,
                reasoning=reasoning,
            )
        )
        if not allow_world_model_edit and _edits_world_model_file(tool_call):
            message = (
                "Objective review cannot edit world_model.py before the actor resolves "
                "the candidate goal"
            )
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
            return
        try:
            result = surface.dispatch(
                tool,
                arguments,
                tool_call_id=tool_call.id,
                raw_arguments=tool_call.arguments,
            )
        except Exception as error:  # noqa: BLE001 - tool failures are builder feedback
            runtime.event_manager.update(
                event_id,
                result=ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"{type(error).__name__}: {error}",
                    result_status=ResultStatus.ERROR,
                ),
            )
            return
        runtime.event_manager.update(
            event_id,
            result=ToolResult(
                tool_call_id=tool_call.id,
                content=str(result),
                result_status=ResultStatus.COMPLETE,
            ),
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
            message = f"Invalid builder tool call {tool_call.name!r}: {error}"
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
            message = f"Invalid builder tool call {tool_call.name!r}: {error}"
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
    def _builder(runtime: RuntimeServices) -> FocusedBuilder:
        builder = runtime.agent
        if not isinstance(builder, FocusedBuilder):
            raise TypeError("BuilderStrategy requires a FocusedBuilder")
        return builder


class FocusedBuilder(Agent, context={"self": None, "state": None}):
    """One invocation of Tycho's builder prompt through a fresh NOOA history."""

    _executor: Annotated[ToolExecutor, hidden_field]
    _tool_surface: Annotated[WorkspaceToolSurface, hidden_field]
    _builder_config: Annotated[BuilderConfig, hidden_field]
    _report_before: Annotated[str | None, hidden_field]
    _world_model_before: Annotated[str | None, hidden_field]
    _builder_system_prompt: Annotated[str, hidden_field]

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        executor: ToolExecutor,
        trace_path: Path | None,
        config: BuilderConfig | None = None,
        theory_transaction: TheoryTransaction | None = None,
    ) -> None:
        # Reflection is deliberately disabled: this experiment adds only Tycho's
        # construction loop, not its separate meta-reflection mechanism.
        system_prompt = render_prompt(
            "builder.system",
            colormap=COLORMAP,
            report_file=REPORT_FILE,
            meta_reflect=False,
        )
        super().__init__(llm=llm)
        # Upstream's prompt enumerates run_python's preloads as a closed list -- `grid`, `np`,
        # `wmlib`, `frames`/`transitions` -- written before `symmetry` existed. Appending is not
        # optional politeness: an authoritative list that excludes a preload actively tells the
        # builder it is not there.
        candidate_brief = (
            TRANSACTIONAL_CANDIDATE_BRIEF
            if theory_transaction is not None
            else RETAINED_CANDIDATE_BRIEF
        )
        self._builder_system_prompt = (
            f"{system_prompt}\n\n{candidate_brief}\n\n"
            f"{OPTIONALITY_PROMPT_BRIEF}\n\n"
            f"{prompt_brief(executor.ws.dir)}"
        )
        self._executor = executor
        self._tool_surface = WorkspaceToolSurface(
            executor,
            profile=BUILDER_TOOL_PROFILE,
            trace_path=trace_path,
            trace_source="builder",
            planner_trace_path=(
                trace_path.with_name("builder-planner-trace.jsonl")
                if trace_path is not None
                else None
            ),
            theory_transaction=theory_transaction,
        )
        self._builder_config = config or BuilderConfig()
        self._report_before = None
        self._world_model_before = None

    def _resolve_system_prompt(self) -> str:
        """Tycho's prompt plus local candidate and `symmetry` capability briefs."""
        return self._builder_system_prompt

    def completed_tool_calls(self) -> list[dict[str, Any]]:
        """Expose durable partial-work evidence if the enclosing deadline cancels us."""
        return self._tool_surface.turn_tool_calls()

    async def build(self, request: BuilderRequest) -> BuilderResult:
        """Construct against the shared workspace and return fresh artifact evidence."""
        self._tool_surface.begin_turn(request.turn)
        before = _workspace_hashes(self._executor.ws.dir)
        self._report_before = _read_report(self._executor.ws.dir)
        self._world_model_before = _read_optional(
            self._executor.ws.dir, WORLD_MODEL_NOTES
        )
        with self._tool_surface.trace_scope(source="builder", phase=request.phase):
            (
                report,
                forced,
                checkpoint_forced,
                outcome_required,
                outcome_resolved,
            ) = await self._construct(request)
        after = _workspace_hashes(self._executor.ws.dir)
        artifact_paths = set(ARTIFACT_PATHS)
        artifact_paths.update(
            path
            for path in before.keys() | after.keys()
            if before.get(path) != after.get(path)
        )
        hashes = {
            path: {
                "before": before.get(path),
                "after": after.get(path),
                "changed": before.get(path) != after.get(path),
            }
            for path in sorted(artifact_paths)
        }
        return BuilderResult(
            report=report,
            # Freshness is a claim about the report's bytes, not about which code
            # path produced them: a forced final-report request can still emit the
            # same content that was already on disk (DIAGNOSIS.md turns 11 and 19),
            # and that must not read as a genuine update.
            report_fresh=hashes[REPORT_FILE]["changed"] is True,
            report_forced=forced,
            world_model_checkpoint_forced=checkpoint_forced,
            outcome_checkpoint_required=outcome_required,
            outcome_checkpoint_resolved=outcome_resolved,
            artifact_hashes=hashes,
            tool_calls=self._tool_surface.turn_tool_calls(),
        )

    @strategy(BuilderStrategy())
    async def _construct(
        self, request: BuilderRequest
    ) -> tuple[str, bool, bool, bool, bool]: ...


def _misses_completed_level(verifier_text: str) -> bool:
    return (
        "post-winning-action missed terminal" in verifier_text
        or "misses observed level_complete terminal" in verifier_text
    )


def _objective_report_complete(report: str) -> bool:
    required = (
        "candidate_goal_assessment:",
        "candidate_goal_reason:",
        "candidate_goal_route:",
        "recommended_action:",
    )
    lines = tuple(line.strip().lower() for line in report.splitlines())
    return all(any(line.startswith(field) for line in lines) for field in required)


def _edits_world_model(tool_calls: list[Any]) -> bool:
    return any(_edits_world_model_file(tool_call) for tool_call in tool_calls)


def _edits_world_model_file(tool_call: Any) -> bool:
    if canonical(tool_call.name) not in {"write_file", "edit_file", "edit_function"}:
        return False
    try:
        arguments = json.loads(tool_call.arguments)
    except json.JSONDecodeError:
        return False
    return Path(str(arguments.get("path", ""))).name == WORLD_MODEL_PY


def _workspace_hashes(workspace_dir: Path) -> dict[str, str]:
    """Hash all stable files so unexpected builder side effects remain visible."""
    return {
        path.relative_to(workspace_dir).as_posix(): digest
        for path in workspace_dir.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and (digest := _file_hash(path)) is not None
    }


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_report(workspace_dir: Path) -> str | None:
    return _read_optional(workspace_dir, REPORT_FILE)


def _read_optional(workspace_dir: Path, relative_path: str) -> str | None:
    path = workspace_dir / relative_path
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _fallback_world_model_checkpoint(previous: str | None) -> str:
    """A guaranteed-to-advance checkpoint entry, used when the model left no edit.

    DIAGNOSIS.md (turns 8-19 of the first live run) found notes/world_model.md read
    on every one of 12 divergence-triggered builder passes but never advanced, so
    every later pass repeated the same stale turn-1 scratchpad. This mirrors
    FALLBACK_REPORT: rather than trust the model to always leave a real edit, force
    the invariant "this pass changed the durable notes" to hold unconditionally.
    """
    marker = "[checkpoint: pass ended with no model-authored update]"
    if previous is None:
        return f"{marker}\n(no prior notes/world_model.md content)\n"
    return f"{previous.rstrip('\n')}\n\n{marker}\n"
