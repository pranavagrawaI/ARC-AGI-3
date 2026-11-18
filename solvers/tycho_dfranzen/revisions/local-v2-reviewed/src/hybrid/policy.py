"""Provider projection; canonical events and workspace evidence remain intact."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from nooa.config.truncation_config import TruncationConfig, FormatConfig
from tycho_nooa.llm_trace import LLMTrace
from inference.agent.tool_agent import _estimate_tokens, _split_messages_for_estimate
from inference.agent.runtime_state import Frame
from inference.agent.vision_context import frame_to_png_data_url, diff_image_part

# One worker owns one game. Every model-using role contributes to this ledger.
LEDGER = {"completion_tokens": 0, "prompt_tokens": 0, "cached_tokens": 0,
          "evictions": 0, "calls": 0, "roles": {}}
OBSERVATION = None
PREVIOUS = None
DEATH = None


def unbounded_render():
    # This removes NOOA's separate half-window renderer eviction. The explicit
    # provider projection below owns the actual bounded input, including tools/images.
    return TruncationConfig(max_context_tokens=10**9, response_reserve_tokens=0,
        event_format=FormatConfig(max_string=96*1024, max_length=200, max_depth=5))


def actor_images(agent=None, turn=None):
    if OBSERVATION is None:
        return []
    def frame(grid):
        return Frame(grid=tuple(tuple(row) for row in grid), step=0, level=0)
    images = [{"type": "image_url", "image_url": {
        "url": frame_to_png_data_url(frame(OBSERVATION.grid), upscale=10)}}]
    if PREVIOUS is not None and PREVIOUS.grid != OBSERVATION.grid:
        diff = diff_image_part(frame(PREVIOUS.grid), frame(OBSERVATION.grid))
        if diff is not None:
            images.append(diff)
    if DEATH is not None:
        images.append({"type": "image_url", "image_url": {
            "url": frame_to_png_data_url(frame(DEATH), upscale=10)}})
    return images


def digest(message):
    return hashlib.sha256(json.dumps(message, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ContextProjection:
    """Trim whole user-led exchanges in blocks; retain the chosen head thereafter."""
    def __init__(self, window=128*1024, drain=58*1024):
        self.window, self.drain = window, drain
        self.anchor = None
        self.chars_per_token = 3.0
        self.last_estimate = 0
        self.last_images = 0
        self.last_chars = 0

    def estimate(self, messages, tools):
        text, images = _split_messages_for_estimate(messages)
        return _estimate_tokens({"messages": text, "tools": tools}, self.chars_per_token) + images + 1024

    def project(self, messages, tools=None, output=12*1024):
        tools = [{"name": t.name, "description": t.description, "parameters": t.get_parameter_schema()}
                 if hasattr(t, "get_parameter_schema") else t for t in (tools or [])]
        prefix = 0
        while prefix < len(messages) and messages[prefix].get("role") == "system":
            prefix += 1
        head = prefix
        if self.anchor is not None:
            matches = [i for i in range(prefix, len(messages)) if digest(messages[i]) == self.anchor]
            if matches:
                head = matches[0]
            else:
                # Method/formatter changed the exchange identities. Never silently
                # claim stable cache reuse; start a new projection and record it.
                self.anchor = None
                LEDGER["projection_resets"] = LEDGER.get("projection_resets", 0) + 1
        result = messages[:prefix] + messages[head:]
        budget = self.window - output
        if budget <= 0:
            raise ValueError("output reserve leaves no input budget")
        estimate = self.estimate(result, tools)
        if estimate > budget:
            target = max(1, budget - min(self.drain, budget//2))
            # Start only at a user message, keeping assistant/tool pairs together.
            boundaries = [i for i in range(head+1, len(messages))
                          if messages[i].get("role") == "user"]
            for next_head in boundaries:
                candidate = messages[:prefix] + messages[next_head:]
                if self.estimate(candidate, tools) <= target:
                    result, head = candidate, next_head
                    break
            else:
                if boundaries:
                    head = boundaries[-1]
                    result = messages[:prefix] + messages[head:]
            if self.estimate(result, tools) > budget:
                raise ValueError("fixed prompt or newest exchange exceeds hybrid input budget")
            self.anchor = digest(messages[head])
            LEDGER["evictions"] += 1
        self.last_estimate = self.estimate(result, tools)
        text, self.last_images = _split_messages_for_estimate(result)
        self.last_chars = len(json.dumps({"messages": text, "tools": tools}, ensure_ascii=False, default=str))
        return result

    def calibrate(self, prompt_tokens):
        text_tokens = max(1, int(prompt_tokens) - self.last_images)
        if prompt_tokens and self.last_chars:
            self.chars_per_token = min(3.3, max(1.0, self.last_chars/text_tokens))


class HybridLLMTrace(LLMTrace):
    """Install the projection BEFORE canonical exact-request checkpoints."""
    def attach(self, event_manager, *, source):
        super().attach(event_manager, source=source)
        projection = ContextProjection()
        self._projections = getattr(self, "_projections", {})
        self._projections[id(event_manager)] = projection
        path = self._path.with_name("context-projection.jsonl") if self._path else None

        async def project(ctx, nxt):
            # Persist an inspectable commitment to the full rendered source as
            # well as the actual projected request captured downstream.
            full_digest = digest({"messages": ctx.messages})
            # NOOA renders reasoning under `reasoning`; the donor's Qwen chat
            # template consumes `reasoning_content` when preserve_thinking is on.
            messages = []
            for original in ctx.messages:
                message = dict(original)
                if message.get("role") == "assistant" and message.get("reasoning"):
                    message["reasoning_content"] = message.pop("reasoning")
                messages.append(message)
            ctx.messages = projection.project(messages, ctx.params.get("tools"),
                                              ctx.params.get("max_tokens", 12*1024))
            if path:
                with path.open("a") as stream:
                    stream.write(json.dumps({"source": source, "full_render_sha256": full_digest,
                        "projected_sha256": digest({"messages": ctx.messages}),
                        "estimated_input_tokens": projection.last_estimate,
                        "evictions": LEDGER["evictions"]}) + "\n")
            result = await nxt(ctx)
            usage = getattr(result.response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0)
            projection.calibrate(prompt_tokens or 0)
            return result
        event_manager.intercept("llm_call", project)

    def _begin(self, event, source):
        super()._begin(event, source)
        source = self._source_override or source
        LEDGER["calls"] += 1
        role = LEDGER["roles"].setdefault(source, {"calls": 0, "completion_tokens": 0})
        role["calls"] += 1
        role["completion_tokens"] += event.completion_tokens or 0
        for key in ("completion_tokens", "prompt_tokens", "cached_tokens"):
            LEDGER[key] += getattr(event, key, 0) or 0
