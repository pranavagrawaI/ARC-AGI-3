"""Host-side trigger, construction, delivery, and uptake instrumentation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nooa.unifiedllm import UnifiedLLM
from tycho.agent.dispatcher import TriggerDispatcher
from tycho.workspace.agent_tools import ToolExecutor, canonical

from tycho_nooa.builder import (
    BuilderConfig,
    BuilderRequest,
    BuilderResult,
    FocusedBuilder,
)
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
    ) -> None:
        self._llm = llm
        self._executor = executor
        self._trace_path = trace_path
        self._builder_tool_trace_path = builder_tool_trace_path
        self._builder_config = builder_config
        self._trigger = TychoTransitionTrigger()
        self.builder: FocusedBuilder | None = None

    async def before_actor(self, turn: GameTurn) -> BuilderExposure | None:
        """Evaluate only a recorded same-level transition, then build if signaled."""
        if turn.turn_in_level <= 0 or not turn.previous_action:
            return None

        decision = self._trigger.evaluate(turn.level, self._executor)
        self._write_trace(
            {
                "event": "mechanical_trigger",
                "mechanism": "tycho TriggerDispatcher.should_fire -> wm_signal",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "fired": decision.fired,
                "cause": decision.cause,
                "reason": decision.reason,
            }
        )
        if not decision.fired:
            return None

        # A new Agent per fire is the NOOA equivalent of Tycho's local builder
        # transcript: same LLM/workspace/verifier, but no actor-history leakage.
        self.builder = FocusedBuilder(
            llm=self._llm,
            executor=self._executor,
            trace_path=self._builder_tool_trace_path,
            config=self._builder_config,
        )
        try:
            result = await self.builder.build(
                BuilderRequest(turn=turn, reason=decision.reason or "")
            )
        except Exception as error:
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
            raise
        self._write_trace(
            {
                "event": "builder_result",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "builder_agent_id": self.builder.agent_id,
                "report_fresh": result.report_fresh,
                "report_forced": result.report_forced,
                "world_model_checkpoint_forced": result.world_model_checkpoint_forced,
                "artifact_hashes": result.artifact_hashes,
                "tool_calls": [
                    {
                        "name": call["name"],
                        "status": call["status"],
                    }
                    for call in result.tool_calls
                ],
            }
        )
        return BuilderExposure(
            result=result,
            recommended_action=_recommended_action(result.report),
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


_RECOMMENDED_ACTION = re.compile(
    r"^recommended_action:\s*([A-Za-z0-9_]+)"
    r"(?:\(\s*(?:row\s*=\s*)?(\d+)\s*,\s*(?:col\s*=\s*)?(\d+)\s*\))?",
    re.MULTILINE,
)


def _recommended_action(report: str) -> ActionDecision | None:
    match = _RECOMMENDED_ACTION.search(report)
    if match is None or match.group(1).lower() == "none":
        return None
    return ActionDecision(
        action=match.group(1),
        row=int(match.group(2)) if match.group(2) is not None else None,
        col=int(match.group(3)) if match.group(3) is not None else None,
    )


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
