"""Retrodict's model session: thinharness 0.5.4's chat-completions loop on our request layer.

``OpenRouterSession`` kept a plain chat message list (system, user, assistant, tool) and
re-sent it whole on every request; ``advance_until_terminal`` executed tool batches until a
turn had no tool calls, counted requests against ``max_model_requests`` (a HarnessError
when exhausted, a "final request" notice one request earlier) and retried a tool once on a
retryable argument error. The same loop runs here through ``qwen_baselines.model.
RequestLayer``, which supplies slot admission, the pinned chat-template settings, the
gameplay-clock feedback and the request/response archive events shared by every baseline.

v5 (18 September): an invocation budget. The last allowed request (by count, or once the
invocation has run for ``budget_seconds``, itself capped at 60% of the remaining game clock; v5 used 40%)
carries a final-request notice and no tools, so it must end in text; the request limit no
longer raises. A reply cut at ``max_tokens`` gets one continuation notice per invocation.
v5.1: a tool's retryable argument error is returned to the model every time (upstream raised
``HarnessError`` after ``max_retries=1``, which ended ar25 on Kaggle after two malformed
``edit`` calls); ``tool_retries=None`` (the default) means no limit, an int keeps the limit.

v5.2 (19 September): a request whose every tool call was rejected for its arguments or
name (ValidationError, InvalidArguments, UnknownTool, a ``[ACTIONS]`` "tool") is an error
round: it goes back to the model but does not count against ``max_model_requests``
(``ERROR_REQUEST_ALLOWANCE`` bounds the total). Calling ``[ACTIONS]`` as a tool gets a
NotATool envelope that says to write the block as reply text.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time

from .thin.providers_chat import (
    ModelTurn,
    _extract_chat_tool_calls,
    _responses_tool_to_chat,
    extract_finish_reason,
    extract_token_usage,
)
from .thin.tools_base import ToolResult, _invoke_tool
from .thin.types import HarnessError, RunUsage

FINAL_REQUEST_NOTICE = (
    "Final request: reply now with your analysis and the [ACTIONS] block; "
    "tools are unavailable for this request."
)
LENGTH_NOTICE = (
    "Your reply was cut at {max_tokens} output tokens. Continue briefly and end with the "
    "[ACTIONS] block."
)
BUDGET_FLOOR_SECONDS = 30
BUDGET_CLOCK_FRACTION = 0.6
# v5.2: error rounds are free, but never more than this many extra requests per invocation.
ERROR_REQUEST_ALLOWANCE = 4
# Tool-call rejections that make a request an error round (the tool never ran).
REJECTED_KINDS = frozenset({"ValidationError", "InvalidArguments", "UnknownTool", "NotATool"})


def render_notice(content):
    return f'<harness_notice kind="limit_warning">\n{content}\n</harness_notice>'


class SessionResult:
    def __init__(
        self, text, responses, usage, stop_reason, resume_state, *,
        budget_notices=0, length_cuts=0, error_requests=0,
    ):
        self.text, self.responses, self.usage = text, responses, usage
        self.stop_reason, self.resume_state = stop_reason, resume_state
        self.budget_notices, self.length_cuts = budget_notices, length_cuts
        self.error_requests = error_requests  # v5.2: rejected-tool rounds, not budgeted


class ChatSession:
    def __init__(
        self,
        layer,
        archive,
        system_prompt,
        specs,
        *,
        max_model_requests,
        budget_seconds=0,
        max_tokens=None,
        clock=None,
        tool_retries=None,
    ):
        self.layer, self.archive = layer, archive
        self.system_prompt = system_prompt
        self.tool_map = {spec.name: spec for spec in specs}
        self.chat_tools = [_responses_tool_to_chat(spec.response_tool()) for spec in specs]
        self.max_model_requests = max_model_requests
        self.budget_seconds = budget_seconds
        self.max_tokens = max_tokens or None
        self.clock = clock
        self.tool_retries = tool_retries
        self.messages = []

    def budget(self):
        """Seconds this invocation may spend before its final request; 0 = no time budget."""
        if not self.budget_seconds:
            return 0
        if self.clock is None:
            return self.budget_seconds
        return max(BUDGET_FLOOR_SECONDS, min(self.budget_seconds, BUDGET_CLOCK_FRACTION * self.clock.remaining()))

    async def run(self, prompt, resume_from=None, *, tools=True):
        """One agent invocation: prompt, tool rounds, final text."""
        usage = RunUsage()
        responses = []
        started = time.monotonic()
        budget = self.budget()
        budget_notices = length_cuts = error_requests = 0
        if resume_from is None:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        else:
            self.messages = copy.deepcopy(resume_from["messages"])
        self.messages.append({"role": "user", "content": prompt})
        final = not tools
        while True:
            self._check_model_limit(usage)
            reason = None
            counted = usage.model_requests - error_requests  # v5.2: error rounds are free
            if not final and (
                counted + 1 >= self.max_model_requests
                or usage.model_requests + 1 >= self.max_model_requests + ERROR_REQUEST_ALLOWANCE
            ):
                reason = "requests"
            elif not final and budget and time.monotonic() - started >= budget:
                reason = "seconds"
            if reason is not None:
                final = True
                budget_notices += 1
                self.messages.append({"role": "user", "content": render_notice(FINAL_REQUEST_NOTICE)})
                self.archive.event(
                    "retrodict_budget", cause=reason, requests=usage.model_requests,
                    seconds=round(time.monotonic() - started, 1), budget_seconds=round(budget, 1),
                )
            turn = await self._complete(usage, None if final else self.chat_tools)
            responses.append(turn.raw)
            if (
                turn.finish_reason == "length"
                and not length_cuts
                and counted + 1 < self.max_model_requests
            ):
                length_cuts += 1
                self.messages[-1].pop("tool_calls", None)  # a cut call is incomplete
                self.messages.append(
                    {"role": "user", "content": render_notice(LENGTH_NOTICE.format(max_tokens=self.max_tokens))}
                )
                self.archive.event(
                    "retrodict_budget", cause="length", requests=usage.model_requests,
                    seconds=round(time.monotonic() - started, 1), budget_seconds=round(budget, 1),
                )
                continue
            if final or not turn.tool_calls:
                return SessionResult(
                    turn.text, responses, usage, "budget" if final and tools else "end_turn",
                    {"messages": copy.deepcopy(self.messages)},
                    budget_notices=budget_notices, length_cuts=length_cuts,
                    error_requests=error_requests,
                )
            outputs, rejected = await self._execute(turn.tool_calls, usage)
            if rejected == len(turn.tool_calls):
                error_requests += 1  # an error round: the model retries for free
            for call, output in zip(turn.tool_calls, outputs, strict=True):
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    def _check_model_limit(self, usage):
        if usage.model_requests >= self.max_model_requests + ERROR_REQUEST_ALLOWANCE:
            raise HarnessError(
                f"model did not finish within max_model_requests={self.max_model_requests}"
            )

    async def _complete(self, usage, tools):
        try:
            raw = await self.layer.complete_full(self.messages, tools, self.archive, max_tokens=self.max_tokens)
        except (TimeoutError, asyncio.CancelledError):
            raise
        except Exception as exc:  # transport and serving failures: Retrodict retries once
            raise HarnessError(f"{type(exc).__name__}: {exc}") from exc
        message = (raw.get("choices") or [{}])[0].get("message") or {}
        self.messages.append(message)
        turn = ModelTurn(
            text=str(message.get("content") or ""),
            tool_calls=_extract_chat_tool_calls(message) if tools else [],
            raw=raw,
            usage=extract_token_usage(raw),
            finish_reason=extract_finish_reason(raw),
        )
        if not tools:
            message.pop("tool_calls", None)  # a tool-less request cannot call tools
        usage.model_requests += 1
        if turn.usage is not None:
            usage.input_tokens += turn.usage.input_tokens or 0
            usage.output_tokens += turn.usage.output_tokens or 0
            usage.cached_tokens += turn.usage.cached_tokens or 0
        return turn

    async def _execute(self, calls, usage):
        """Retrodict's tools are all sequential; outputs keep the model's order.

        Returns the outputs and how many calls were rejected before running (bad name or
        arguments), which is what makes a request an error round.
        """
        outputs = []
        rejected = 0
        usage.tool_calls += len(calls)
        for call in calls:
            spec = self.tool_map.get(call.name)
            if spec is None and call.name.strip("[] ").upper() == "ACTIONS":
                envelope = ToolResult(
                    False,
                    "[ACTIONS] is not a tool: end your reply with the [ACTIONS] block as plain text "
                    "(no tool call).",
                    {"error_type": "NotATool", "retry": True},
                )
            elif spec is None:
                envelope = ToolResult(
                    False, f"unknown tool: {call.name}", {"error_type": "UnknownTool", "retry": True}
                )
            else:
                envelope = await _invoke_tool(spec, call.arguments)
            if envelope.retry_kind() in REJECTED_KINDS:
                rejected += 1
            self.archive.event(
                "retrodict_tool",
                name=call.name,
                arguments=call.arguments[:4000],
                ok=envelope.ok,
                error_type=envelope.error_type(),
                chars=len(envelope.content),
            )
            if envelope.retry_kind() is not None:
                usage.tool_retries[call.name] = usage.tool_retries.get(call.name, 0) + 1
                if self.tool_retries is not None and usage.tool_retries[call.name] > self.tool_retries:
                    raise HarnessError(f"tool {call.name!r} exceeded max_retries={self.tool_retries}")
            outputs.append(envelope.to_json())
        return outputs, rejected


def last_input_tokens(responses):
    """Input tokens of the final provider request, i.e. the conversation size."""
    if not responses:
        return 0
    usage = responses[-1].get("usage") or {}
    tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    return tokens if isinstance(tokens, int) else 0


def hit_output_token_limit(responses):
    """Chat completions report a cut-off reply as finish_reason "length"."""
    return bool(responses) and extract_finish_reason(responses[-1]) == "length"


def dumps(value):
    return json.dumps(value, ensure_ascii=False)
