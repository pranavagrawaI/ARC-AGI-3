"""Session runner for questioner mode. Same `decide`/`fire` surface as `CriticRunner`, so the
solver's `_run_critic_if_triggered` hook drives it unchanged.

Cadence: fire after every `period_turns` analyzer turns within a level (the user's manual
critique fired every turn; the period is the experimental knob). The first firing in a level
waits `period_turns` turns so the actor has formed a belief worth questioning.

    DUCK_CRITIC_ENABLED=1 DUCK_CRITIC_MODE=questioner
    DUCK_CRITIC_PERIOD_TURNS=3          turns between questions (1 = every turn)
    DUCK_CRITIC_QUESTIONER_EFFORT=low   reasoning_effort for the question call; "none" = thinking off
    DUCK_CRITIC_QUESTIONER_MAX_TOKENS=3000
    DUCK_CRITIC_QUESTIONER_SECONDS=240  per-call timeout
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

from inference.agent.runtime_state import HistoryEntry
from inference.critic.packet import transitions_from_history
from inference.critic.policy import CriticFiring, CriticModelConfig
from inference.critic.questioner import (
    QUESTIONER_SYSTEM_PROMPT,
    QuestionLedger,
    parse_question,
    render_questioner_input,
    validate_question,
)
from inference.critic.report import CriticReport
from inference.critic.trigger import TriggerDecision
from inference.utils.openai_compat import build_chat_payload, build_headers


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


@dataclass
class QuestionerRunner:
    model: CriticModelConfig
    trace_path: Path | None = None
    period_turns: int = 3
    effort: str = "low"
    max_tokens: int = 3000
    request_seconds: float = 240.0
    post: Callable[[dict[str, Any], float], dict[str, Any]] | None = None
    ledger: QuestionLedger = field(default_factory=QuestionLedger)
    firings: list[dict[str, Any]] = field(default_factory=list)
    _level: int | None = None
    _turns_in_level: int = 0
    _turns_since_fire: int = 0
    _recent_words: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_env(cls, model: CriticModelConfig, *, trace_path: Path | None) -> QuestionerRunner:
        return cls(
            model=model,
            trace_path=trace_path,
            period_turns=max(1, _env_int("DUCK_CRITIC_PERIOD_TURNS", 3)),
            effort=os.environ.get("DUCK_CRITIC_QUESTIONER_EFFORT", "low").strip() or "low",
            max_tokens=_env_int("DUCK_CRITIC_QUESTIONER_MAX_TOKENS", 3000),
            request_seconds=float(_env_int("DUCK_CRITIC_QUESTIONER_SECONDS", 240)),
        )

    def decide(self, *, history: list[HistoryEntry], level: int, action_num: int, critic_request: str = "") -> TriggerDecision:
        if level != self._level:
            self._level, self._turns_in_level, self._turns_since_fire = level, 0, 0
        self._turns_in_level += 1
        self._turns_since_fire += 1
        if self._turns_since_fire > self.period_turns:
            return TriggerDecision(fired=True, cause="period", reason=f"{self.period_turns} turns since the last question")
        return TriggerDecision.silent("waiting")

    def fire(
        self,
        decision: TriggerDecision,
        *,
        current_frame: Any,
        history: list[HistoryEntry],
        valid_actions: list[str],
        actor_context: dict[str, Any],
    ) -> CriticFiring:
        started = time.monotonic()
        self._turns_since_fire = 1
        recent = list(actor_context.get("recent_words") or [])
        self.ledger.record_reply(recent)
        level = int(getattr(current_frame, "level", self._level or 1) or 1)
        action_num = len(history)
        transitions = [
            {"action": t.action, "board_changed": t.board_changed, "level_completed": t.level_completed, "game_over": bool(getattr(t, "game_over", False))}
            for t in transitions_from_history(history)
        ]
        prompt = render_questioner_input(
            level=level,
            action_num=action_num,
            turns_since_level=self._turns_in_level,
            recent_words=recent,
            transitions=transitions,
            ledger=self.ledger,
        )
        messages = [{"role": "system", "content": QUESTIONER_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        attempts: list[dict[str, Any]] = []
        question = rejection = error = ""
        prompt_tokens = completion_tokens = calls = 0
        for _ in range(2):
            try:
                message, usage = self._chat(messages)
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
                break
            calls += 1
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            content = str(message.get("content") or "")
            candidate = parse_question(content)
            rejection = validate_question(candidate, self.ledger)
            attempts.append({"content": content[:600], "candidate": candidate, "rejection": rejection})
            if not rejection:
                question = candidate
                break
            messages = [*messages, {"role": "assistant", "content": content}, {"role": "user", "content": f"Rejected: {rejection}. One question only."}]
        if question:
            last_seq = max((int(item.get("seq", 0)) for item in recent), default=0)
            self.ledger.add(level, action_num, question, after_seq=last_seq)
        report = CriticReport(
            packet_id=f"q-{level}-{action_num}",
            summary="",
            open_questions=(question,) if question else (),
            style="question",
        )
        firing = CriticFiring(
            packet_id=report.packet_id,
            report=report,
            report_text=question,
            scratchpad=question,
            seconds=round(time.monotonic() - started, 3),
            model_calls=calls,
            tool_calls=0,
            tool_errors=0,
            refused_action_calls=0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            parsed=bool(question),
            fallback=False,
            forced_final=False,
            recommended_legal=None,
            error=error,
            transcript=[{"role": "user", "content": prompt}, *attempts],
        )
        record = {
            "time": time.time(),
            "level": level,
            "action_num": action_num,
            "cause": decision.cause,
            "seconds": firing.seconds,
            "model_calls": calls,
            "completion_tokens": completion_tokens,
            "question": question,
            "rejections": [a["rejection"] for a in attempts if a["rejection"]],
            "error": error,
            "ledger": [e.__dict__ for e in self.ledger.entries],
        }
        self.firings.append(record)
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trace_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({**record, "input": prompt, "attempts": attempts}, default=str) + "\n")
        return firing

    def _chat(self, messages: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        thinking = self.effort.lower() != "none"
        payload = build_chat_payload(
            provider=self.model.provider,
            model=self.model.model_id,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.model.temperature,
            top_p=self.model.top_p,
            top_k=self.model.top_k,
            thinking=thinking,
        )
        if thinking and isinstance(payload.get("chat_template_kwargs"), dict):
            payload["chat_template_kwargs"]["reasoning_effort"] = self.effort
        if self.post is not None:
            body = self.post(payload, self.request_seconds)
        else:
            response = requests.post(
                f"{self.model.base_url.rstrip('/')}/chat/completions",
                headers=build_headers(provider=self.model.provider, api_key=self.model.api_key, referer=self.model.referer, title=self.model.title),
                json=payload,
                timeout=self.request_seconds,
            )
            if response.status_code >= 400:
                raise requests.RequestException(f"{response.status_code} | {response.text.strip()[:500]}")
            body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise requests.RequestException("server returned no choices")
        return choices[0].get("message", {}) or {}, body.get("usage") or {}

    def summary(self) -> dict[str, Any]:
        return {
            "firings": len(self.firings),
            "delivered": sum(bool(f["question"]) for f in self.firings),
            "seconds": round(sum(float(f["seconds"]) for f in self.firings), 1),
            "errors": sum(bool(f["error"]) for f in self.firings),
        }


__all__ = ["QuestionerRunner"]
