"""Retrodict (R) in the arena: parking at request boundaries, the invocation budget on
active seconds, a request-level compaction backstop and a model-written handoff summary
when the runner starts a fresh session."""

import asyncio
import time

from qwen_baselines.runner import retrodict_summary  # noqa: F401  (re-exported for reports)
from retrodict_qwen.agent import QwenAgentClient
from retrodict_qwen.game import BridgeEnv, RetrodictGame
from retrodict_qwen.runner import GameRunner
from retrodict_qwen.session import (
    BUDGET_CLOCK_FRACTION,
    BUDGET_FLOOR_SECONDS,
    FINAL_REQUEST_NOTICE,
    ChatSession,
    render_notice,
)

from .compaction import R_SUMMARY_PROMPT, split_at_latest_observation, summary_message, summary_text


class ArenaChatSession(ChatSession):
    def __init__(self, *args, game, window, cap, **kwargs):
        super().__init__(*args, **kwargs)
        self.game, self.window, self.cap = game, int(window), int(cap)
        self.active_started = 0.0
        self.active_notice = False
        self.compactions = 0

    def budget(self):
        return 0  # the wall-time check is replaced by the active-seconds check below

    def time_budget(self):
        if not self.budget_seconds:
            return 0
        return max(
            BUDGET_FLOOR_SECONDS,
            min(self.budget_seconds, BUDGET_CLOCK_FRACTION * self.game.clock.global_clock.remaining()),
        )

    async def run(self, prompt, resume_from=None, *, tools=True):
        self.active_started = self.game.clock.active_seconds()
        self.active_notice = False
        return await super().run(prompt, resume_from=resume_from, tools=tools)

    async def _complete(self, usage, tools):
        await self.game.clock.wait_active()
        budget = self.time_budget()
        if (
            tools is not None
            and budget
            and not self.active_notice
            and self.game.clock.active_seconds() - self.active_started >= budget
        ):
            self.active_notice = True
            self.messages.append({"role": "user", "content": render_notice(FINAL_REQUEST_NOTICE)})
            self.archive.event(
                "retrodict_budget", cause="seconds_active", requests=usage.model_requests,
                seconds=round(self.game.clock.active_seconds() - self.active_started, 1),
                budget_seconds=round(budget, 1),
            )
            tools = None
        count = await self.layer.count(self.layer.fresh(self.layer.payload(self.messages, tools)))
        if count + (self.max_tokens or 0) > self.window:
            await self.compact(count)
        return await super()._complete(usage, tools)

    async def summarize(self, messages):
        """One tool-less, thinking-off summary request over ``messages``."""
        payload = self.layer.payload(
            [*messages, {"role": "user", "content": R_SUMMARY_PROMPT}], None, summary=True
        )
        response = await self.layer.complete_payload(
            payload, self.archive, summary=True, max_tokens=self.cap
        )
        text = summary_text(response.json()["choices"][0]["message"])
        if not text:
            raise RuntimeError("Compaction summary came back empty")
        return text

    async def compact(self, count):
        started = time.monotonic()
        summary = await self.summarize(self.messages)
        system_message = self.messages[0]
        _prefix, tail = split_at_latest_observation(self.messages[1:])
        kept = tail if tail else self.messages[-1:]
        compacted = [system_message, summary_message(summary), *kept]
        check = await self.layer.count(self.layer.fresh(self.layer.payload(compacted, None)))
        if check + (self.max_tokens or 0) > self.window and len(kept) > 1:
            kept = kept[-1:] if kept[-1].get("role") == "user" else kept[:1]
            compacted = [system_message, summary_message(summary), *kept]
        self.messages = compacted
        self.compactions += 1  # Shared request layer records game count and summary latency.
        self.archive.event(
            "arena_compaction", tokens_before=count, kept=len(kept),
            summary_chars=len(summary), seconds=round(time.monotonic() - started, 3),
        )


class ArenaGameRunner(GameRunner):
    def __init__(self, *args, game, session, **kwargs):
        super().__init__(*args, **kwargs)
        self.game = game
        self.session = session
        self.pending_summary = None
        self._level_wall_start = time.monotonic()
        self._level_parked_base = 0.0

    async def _invoke(self, reason):
        await self.game.clock.wait_active()
        state = self.state
        if (
            state.invocations
            and state.resume_state is not None
            and state.context_tokens > self.cfg.fresh_session_input_tokens
        ):
            self.pending_summary = await self.session.summarize(state.resume_state["messages"])
            self.archive.event("arena_handoff_summary", chars=len(self.pending_summary))
        return await super()._invoke(reason)

    def _build_prompt(self, reason):
        before = self.state.fresh_sessions
        prompt, resume = super()._build_prompt(reason)
        if self.state.fresh_sessions > before and self.pending_summary:
            prompt += "\n\n## Your predecessor's handoff summary\n" + self.pending_summary
        self.pending_summary = None
        return prompt, resume

    def _start_level(self):
        super()._start_level()
        self._level_wall_start = self.state.level_start_time
        self._level_parked_base = self.game.clock.parked_seconds()

    def _escalation_directive(self):
        parked = self.game.clock.parked_seconds() - self._level_parked_base
        self.state.level_start_time = self._level_wall_start + parked
        return super()._escalation_directive()


class ArenaRetrodictGame(RetrodictGame):
    def __init__(self, bridge, layer, config, *, game, window):
        super().__init__(bridge, layer, config)
        old = self.runner.agent.session
        session = ArenaChatSession(
            layer,
            bridge.archive,
            old.system_prompt,
            list(old.tool_map.values()),
            max_model_requests=old.max_model_requests,
            budget_seconds=old.budget_seconds,
            max_tokens=old.max_tokens,
            clock=bridge.clock,
            tool_retries=old.tool_retries,
            game=game,
            window=window,
            cap=config.output_tokens_cap or 16384,
        )
        self.session = session
        self.game = game
        self.runner = ArenaGameRunner(
            BridgeEnv(bridge),
            QwenAgentClient(session),
            self.runner_config,
            bridge.archive.root,
            clock=bridge.clock,
            archive=bridge.archive,
            game=game,
            session=session,
        )

    async def run(self):
        try:
            return await self.runner.run()
        except (TimeoutError, asyncio.CancelledError):
            self.bridge.archive.event("deadline", seconds=self.bridge.clock.elapsed())
        finally:
            await self.close()
