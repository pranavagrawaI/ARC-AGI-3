"""Host-side trigger, construction, delivery, and uptake instrumentation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nooa.unifiedllm import UnifiedLLM
from pydantic import ValidationError
from tycho.agent.dispatcher import TriggerDispatcher
from tycho.workspace.agent_tools import ToolExecutor, canonical

from tycho_nooa.builder import (
    BuilderConfig,
    BuilderRequest,
    BuilderResult,
    FocusedBuilder,
)
from tycho_nooa.llm_trace import LLMTrace
from tycho_nooa.models import ActionDecision, GameTurn


@dataclass(frozen=True)
class TriggerDecision:
    fired: bool
    cause: str | None
    reason: str | None


@dataclass(frozen=True)
class BuilderExposure:
    result: BuilderResult
    recommended_action: ActionDecision | None
    recommendation_parse_error: str | None = None


@dataclass(frozen=True)
class RecommendationParse:
    action: ActionDecision | None
    error: str | None


class TychoTransitionTrigger:
    """Adapt Tycho's exact TriggerDispatcher/wm_signal path to one transition."""

    def __init__(self) -> None:
        self._dispatcher = TriggerDispatcher()

    def evaluate(self, level: int, executor: ToolExecutor) -> TriggerDecision:
        fired = self._dispatcher.should_fire(level, executor)
        if not fired:
            return TriggerDecision(fired=False, cause=None, reason=None)
        reason = self._dispatcher.trigger_reason(level)
        cause = getattr(self._dispatcher, "_last_cause", None)
        self._dispatcher.note_fired(level)
        return TriggerDecision(fired=True, cause=cause, reason=reason)

    def evaluate_level_start(self, level: int) -> TriggerDecision:
        """LEVEL_START fires unconditionally (no divergence probe), mirroring
        Tycho's own `elif ... turn_in_level == 0 and level > 0` branch, which
        never calls should_fire() for this cause."""
        reason = self._dispatcher.level_start_reason(level)
        self._dispatcher.note_fired(level)
        return TriggerDecision(fired=True, cause="level_start", reason=reason)


class TriggerBuilderLoop:
    """Run a fresh builder after divergent transitions, before the actor acts."""

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        executor: ToolExecutor,
        trace_path: Path | None,
        builder_tool_trace_path: Path | None,
        builder_config: BuilderConfig | None = None,
        llm_trace: LLMTrace | None = None,
    ) -> None:
        self._llm = llm
        self._executor = executor
        self._trace_path = trace_path
        self._builder_tool_trace_path = builder_tool_trace_path
        self._builder_config = builder_config
        self._llm_trace = llm_trace
        self._trigger = TychoTransitionTrigger()
        self.builder: FocusedBuilder | None = None
        # -1 never equals a real level, so a genuine level-start check at level 0
        # (guarded out below regardless) never depends on this sentinel.
        self._last_level = -1
        self._pending_outcome_evidence: str | None = None
        self._pending_outcome_level: int | None = None
        self._resolved_outcome_levels: set[int] = set()

    async def before_actor(self, turn: GameTurn) -> BuilderExposure | None:
        """Evaluate the recorded transition into this turn, then build if signaled.

        arc-host's `full_reset` resets `turn_in_level` to 0 WITHOUT changing
        `level` (runner.py), so `turn_in_level == 0` alone cannot distinguish a
        genuine level advance from a same-level restart — Tycho's own three
        `turn_in_level = 0` call sites have the identical ambiguity, disambiguated
        there only by if/elif branch priority. `_last_level`, updated
        unconditionally before either branch below, plays that same role here:
        only a level that actually differs from the immediately preceding turn's
        level counts as a level-start.
        """
        triggered = self._evaluate_trigger(turn)
        if triggered is None:
            return None
        decision, mechanism = triggered

        self._write_trace(
            {
                "event": "mechanical_trigger",
                "mechanism": mechanism,
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "fired": decision.fired,
                "cause": decision.cause,
                "reason": decision.reason,
            }
        )
        if not decision.fired:
            return None

        self._capture_outcome_evidence(turn, decision)
        return await self._invoke_builder(turn, decision.reason or "")

    async def after_level_complete(
        self, turn: GameTurn, outcome_evidence: str
    ) -> BuilderExposure | None:
        """Repair the outcome contract while the winning transition is current.

        A level may complete on the final permitted action, in which case no
        next-level actor turn exists to fire the ordinary level-start builder.
        The host's action-result callback is therefore the only reliable place
        to make this mandatory checkpoint independent of another decision.

        Builder failure remains recorded and the evidence stays pending for a
        later level-start retry; it must not erase an already observed win.
        """
        self._pending_outcome_evidence = outcome_evidence
        self._pending_outcome_level = turn.level
        self._write_trace(
            {
                "event": "mechanical_trigger",
                "mechanism": "arc-host action_result (mandatory outcome checkpoint)",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "fired": True,
                "cause": "level_complete",
                "reason": "repair host-confirmed completed-level outcome before continuing",
            }
        )
        try:
            return await self._invoke_builder(
                turn,
                "repair host-confirmed completed-level outcome before continuing",
            )
        except Exception:  # noqa: BLE001 - preserve the completed game outcome
            return None

    async def _invoke_builder(
        self, turn: GameTurn, reason: str
    ) -> BuilderExposure:
        self.builder = self._new_builder()
        try:
            result = await self.builder.build(
                BuilderRequest(
                    turn=turn,
                    reason=reason,
                    outcome_evidence=self._pending_outcome_evidence,
                )
            )
        except Exception as error:
            self._record_builder_error(turn, error)
            raise
        if result.outcome_checkpoint_resolved:
            if self._pending_outcome_level is not None:
                self._resolved_outcome_levels.add(self._pending_outcome_level)
            self._pending_outcome_evidence = None
            self._pending_outcome_level = None
        recommendation = _recommended_action(result.report, turn.available_actions)
        self._record_builder_result(turn, result, recommendation)
        return BuilderExposure(
            result=result,
            recommended_action=recommendation.action,
            recommendation_parse_error=recommendation.error,
        )

    def _evaluate_trigger(self, turn: GameTurn) -> tuple[TriggerDecision, str] | None:
        last_level = self._last_level
        self._last_level = turn.level
        if turn.turn_in_level == 0 and turn.level > 0 and turn.level != last_level:
            return (
                self._trigger.evaluate_level_start(turn.level),
                "tycho TriggerDispatcher.level_start_reason (unconditional)",
            )
        if turn.turn_in_level <= 0 or not turn.previous_action:
            return None
        mechanism = "tycho TriggerDispatcher.should_fire -> wm_signal"
        try:
            return self._trigger.evaluate(turn.level, self._executor), mechanism
        except Exception as error:  # noqa: BLE001 - an unanswerable probe is no signal
            # wm_signal runs verify.py in Tycho's sandbox. An unavailable probe is
            # absence of signal, not a reason to end the episode; the builder itself
            # still fails loudly because unverifiable model edits are not evidence.
            self._write_trace(
                {
                    "event": "trigger_error",
                    "mechanism": mechanism,
                    "level": turn.level,
                    "turn_in_level": turn.turn_in_level,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            return None

    def _capture_outcome_evidence(
        self, turn: GameTurn, decision: TriggerDecision
    ) -> None:
        if decision.cause != "level_start":
            return
        completed_level = turn.level - 1
        if completed_level in self._resolved_outcome_levels:
            return
        evidence = _level_complete_evidence(self._executor.ws.dir, completed_level)
        if evidence is not None:
            self._pending_outcome_evidence = evidence
            self._pending_outcome_level = completed_level

    def _new_builder(self) -> FocusedBuilder:
        # A new Agent per fire is the NOOA equivalent of Tycho's local builder
        # transcript: same LLM/workspace/verifier, but no actor-history leakage.
        builder = FocusedBuilder(
            llm=self._llm,
            executor=self._executor,
            trace_path=self._builder_tool_trace_path,
            config=self._builder_config,
        )
        # Each fresh agent has a fresh event stream; attach the run-wide trace now.
        if self._llm_trace is not None:
            self._llm_trace.attach(builder.event_manager, source="builder")
        return builder

    def _record_builder_error(self, turn: GameTurn, error: Exception) -> None:
        assert self.builder is not None
        self._write_trace(
            {
                "event": "builder_error",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "builder_agent_id": self.builder.agent_id,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )

    def _record_builder_result(
        self,
        turn: GameTurn,
        result: BuilderResult,
        recommendation: RecommendationParse,
    ) -> None:
        assert self.builder is not None
        self._write_trace(
            {
                "event": "builder_result",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "builder_agent_id": self.builder.agent_id,
                "report_fresh": result.report_fresh,
                "report_forced": result.report_forced,
                "world_model_checkpoint_forced": result.world_model_checkpoint_forced,
                "outcome_checkpoint_required": result.outcome_checkpoint_required,
                "outcome_checkpoint_resolved": result.outcome_checkpoint_resolved,
                "recommended_action": (
                    recommendation.action.model_dump()
                    if recommendation.action is not None
                    else None
                ),
                "recommendation_parse_error": recommendation.error,
                "artifact_hashes": result.artifact_hashes,
                "tool_calls": [
                    {"name": call["name"], "status": call["status"]}
                    for call in result.tool_calls
                ],
            }
        )

    def after_actor(
        self,
        turn: GameTurn,
        exposure: BuilderExposure | None,
        decision: ActionDecision,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """Record syntactic follow/test evidence without guessing at model intent."""
        if exposure is None:
            return
        tested_by = [
            call["name"]
            for call in tool_calls
            if _tests_constructed_result(call["name"], call["arguments"])
        ]
        recommended = exposure.recommended_action
        self._write_trace(
            {
                "event": "actor_uptake",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "actor_action": decision.model_dump(),
                "recommended_action": (
                    recommended.model_dump() if recommended is not None else None
                ),
                "followed_recommended_action": (
                    decision == recommended if recommended is not None else None
                ),
                "tested_report": bool(tested_by),
                "test_evidence": tested_by,
            }
        )

    def after_actor_error(
        self,
        turn: GameTurn,
        exposure: BuilderExposure | None,
        error: Exception,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """Make failed post-report actor turns visible rather than silently incomplete."""
        if exposure is None:
            return
        self._write_trace(
            {
                "event": "actor_uptake",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "actor_action": None,
                "recommended_action": (
                    exposure.recommended_action.model_dump()
                    if exposure.recommended_action is not None
                    else None
                ),
                "followed_recommended_action": None,
                "tested_report": any(
                    _tests_constructed_result(call["name"], call["arguments"])
                    for call in tool_calls
                ),
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )

    def _write_trace(self, record: dict[str, Any]) -> None:
        if self._trace_path is None:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps(record, separators=(",", ":")) + "\n")


def _level_complete_evidence(workspace_dir: Path, level: int) -> str | None:
    path = workspace_dir / f"level_{level}" / "level_complete.json"
    if not path.is_file():
        return None
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if evidence.get("source") != "arc-host action_result":
        return None
    if evidence.get("outcome") != "level_complete":
        return None
    return json.dumps(evidence, separators=(",", ":"), sort_keys=True)


_RECOMMENDED_ACTION = re.compile(
    r"^recommended_action:\s*([A-Za-z0-9_]+)"
    r"(?:\(\s*(?:row\s*=\s*)?(\d+)\s*,\s*(?:col\s*=\s*)?(\d+)\s*\))?",
    re.MULTILINE,
)


def _recommended_action(
    report: str, available_actions: list[str]
) -> RecommendationParse:
    match = _RECOMMENDED_ACTION.search(report)
    if match is None or match.group(1).lower() == "none":
        return RecommendationParse(action=None, error=None)
    action = match.group(1).upper()
    if action not in available_actions:
        return RecommendationParse(
            action=None,
            error=(
                f"recommended_action {action!r} is not available; "
                f"available actions: {available_actions}"
            ),
        )
    try:
        decision = ActionDecision(
            action=action,
            row=int(match.group(2)) if match.group(2) is not None else None,
            col=int(match.group(3)) if match.group(3) is not None else None,
        )
    except ValidationError as error:
        return RecommendationParse(action=None, error=str(error))
    return RecommendationParse(action=decision, error=None)


def _tests_constructed_result(name: str, arguments: dict[str, Any]) -> bool:
    canonical_name = canonical(name)
    if canonical_name == "read_file":
        return str(arguments.get("path", "")) in {
            "notes/world_model_report.md",
            "notes/world_model.md",
            "world_model.py",
        }
    if canonical_name != "run_python":
        return False
    code = str(arguments.get("code", "")).lower()
    return any(
        marker in code for marker in ("world_model", "wmlib", "verify.py", "plan.py")
    )
