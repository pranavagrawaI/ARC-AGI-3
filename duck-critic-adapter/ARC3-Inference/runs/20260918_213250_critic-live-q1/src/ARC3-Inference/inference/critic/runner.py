"""Wire trigger -> packet -> policy -> report into one Duck game session.

`CriticRunner.before_turn()` is called by `_HarnessGameSession.play()` right before
each `analyzer.analyze()`. It runs synchronously in the game thread, so every second
the critic spends is charged to the same per-game wall clock as the actor -- no
separate budget bookkeeping. Disabled (the default) it does nothing and the session
is byte-identical to upstream Duck.

Configuration is environment-driven so the Kaggle setup env can carry it:
  DUCK_CRITIC_ENABLED=1            turn the critic on
  DUCK_CRITIC_MODE=question        "question" (one call, reasoning off) or "report" (tool loop)
  DUCK_CRITIC_MAX_TOOL_STEPS=2     investigation calls per firing
  DUCK_CRITIC_MAX_SECONDS=90       wall budget per firing
  DUCK_CRITIC_MAX_OUTPUT_TOKENS=4096
  DUCK_CRITIC_STALL_NO_CHANGE=3    consecutive no-board-change actions
  DUCK_CRITIC_STALL_NO_PROGRESS=25 actions without level progress
  DUCK_CRITIC_MODEL_ID / DUCK_CRITIC_BASE_URL / DUCK_CRITIC_PROVIDER / DUCK_CRITIC_API_KEY
                                   override the actor's endpoint (default: share it)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inference.agent.runtime_state import HistoryEntry
from inference.critic.packet import CriticPacket, build_packet_from_state, transitions_from_history
from inference.critic.policy import CriticFiring, CriticModelConfig, CriticNotes, CriticPolicy, CriticPolicyConfig
from inference.critic.trigger import DuckTriggerPolicy, TriggerDecision


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return default if not raw else raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def critic_enabled() -> bool:
    return _env_flag("DUCK_CRITIC_ENABLED", False)


def policy_config_from_env() -> CriticPolicyConfig:
    return CriticPolicyConfig(
        mode=os.environ.get("DUCK_CRITIC_MODE", "").strip() or CriticPolicyConfig.mode,
        max_tool_steps=_env_int("DUCK_CRITIC_MAX_TOOL_STEPS", CriticPolicyConfig.max_tool_steps),
        max_seconds=_env_float("DUCK_CRITIC_MAX_SECONDS", CriticPolicyConfig.max_seconds),
        max_output_tokens=_env_int("DUCK_CRITIC_MAX_OUTPUT_TOKENS", CriticPolicyConfig.max_output_tokens or 4096),
    )


def model_config_from_env(actor: tuple[str, str, str, str]) -> CriticModelConfig:
    provider, base_url, model_id, api_key = actor
    return CriticModelConfig(
        provider=os.environ.get("DUCK_CRITIC_PROVIDER", "").strip() or provider,
        base_url=os.environ.get("DUCK_CRITIC_BASE_URL", "").strip() or base_url,
        model_id=os.environ.get("DUCK_CRITIC_MODEL_ID", "").strip() or model_id,
        api_key=os.environ.get("DUCK_CRITIC_API_KEY", "").strip() or api_key,
        request_timeout_seconds=_env_float("DUCK_CRITIC_REQUEST_TIMEOUT", 180.0),
    )


@dataclass
class CriticRunner:
    """One per game session. Holds the trigger state and the scratchpad."""

    policy: CriticPolicy
    trace_path: Path | None = None
    stall_no_change: int = 3
    stall_no_progress: int = 25
    trigger: DuckTriggerPolicy = field(init=False)
    notes: CriticNotes = field(default_factory=CriticNotes)
    observed: int = 0
    firings: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.trigger = DuckTriggerPolicy(
            stall_no_change=self.stall_no_change,
            stall_no_progress=self.stall_no_progress,
        )

    @classmethod
    def from_env(cls, actor_model: tuple[str, str, str, str], *, trace_path: Path | None) -> CriticRunner:
        return cls(
            policy=CriticPolicy(model_config_from_env(actor_model), policy_config_from_env()),
            trace_path=trace_path,
            stall_no_change=_env_int("DUCK_CRITIC_STALL_NO_CHANGE", 3),
            stall_no_progress=_env_int("DUCK_CRITIC_STALL_NO_PROGRESS", 25),
        )

    def decide(
        self,
        *,
        history: list[HistoryEntry],
        level: int,
        action_num: int,
        critic_request: str,
    ) -> TriggerDecision:
        transitions = transitions_from_history(history)
        new = transitions[self.observed :]
        if new:
            self.trigger.observe(new)
            self.observed = len(transitions)
        return self.trigger.decide(level=level, action_num=action_num, critic_request=critic_request)

    def fire(
        self,
        decision: TriggerDecision,
        *,
        current_frame: Any,
        history: list[HistoryEntry],
        valid_actions: list[str],
        actor_context: dict[str, Any],
    ) -> CriticFiring:
        packet: CriticPacket = build_packet_from_state(
            current_frame,
            history,
            summarized_knowledge=actor_context.get("summarized_knowledge") or {},
            last_step_summary=actor_context.get("last_step_summary"),
            last_action_result=actor_context.get("last_action_result"),
            valid_actions=valid_actions,
            trigger_cause=decision.cause or "",
            trigger_reason=decision.reason or "",
            actor_last_code=str(actor_context.get("last_tool_code") or ""),
        )
        firing = self.policy.run(packet, self.notes)
        record = {
            "time": time.time(),
            "packet_id": packet.packet_id,
            "level": packet.level,
            "action_num": packet.action_num,
            "cause": packet.trigger_cause,
            "reason": packet.trigger_reason,
            "seconds": firing.seconds,
            "model_calls": firing.model_calls,
            "tool_calls": firing.tool_calls,
            "tool_errors": firing.tool_errors,
            "prompt_tokens": firing.prompt_tokens,
            "completion_tokens": firing.completion_tokens,
            "parsed": firing.parsed,
            "fallback": firing.fallback,
            "error": firing.error,
            "recommended_actions": list(firing.report.recommended_actions),
            "recommended_legal": firing.recommended_legal,
            "confidence": firing.report.confidence,
            "report_text": firing.report_text,
            "scratchpad": firing.scratchpad,
        }
        self.firings.append(record)
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trace_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({**record, "transcript": firing.transcript}, default=str) + "\n")
        return firing

    def summary(self) -> dict[str, Any]:
        return {
            "firings": len(self.firings),
            "seconds": round(sum(float(f["seconds"]) for f in self.firings), 1),
            "by_cause": {
                cause: sum(1 for f in self.firings if f["cause"] == cause)
                for cause in sorted({f["cause"] for f in self.firings})
            },
            "parsed": sum(bool(f["parsed"]) for f in self.firings),
            "errors": sum(bool(f["error"]) for f in self.firings),
        }


__all__ = [
    "CriticRunner",
    "critic_enabled",
    "model_config_from_env",
    "policy_config_from_env",
]
