"""Actual NOOA client boundary to bounded receipt-preserving transport."""

import copy
import re

from nooa.unifiedllm import UnifiedLLM, LLMResponse, ToolCall
from transport import MODEL
from protocol import assistant_message, project_messages


class ActorClient(UnifiedLLM):
    def __init__(self, transport):
        super().__init__(model=MODEL)
        self.transport = transport
        self.prefill_ids = {}
        self._context_window = 262144

    @property
    def context_window(self):
        return self._context_window

    def count_tokens(self, text):
        # Deliberately conservative local truncation bound. Actual receipt is authoritative.
        return len(text.encode()) + 1

    def call(self, messages, tools=None, output_model=None, **kwargs):
        projected = project_messages(copy.deepcopy(messages))
        # Only framework-generated prefill IDs are alpha-renamed. Provider IDs
        # remain opaque. Apply the same stable mapping to their rendered events.
        for message in projected:
            for call in message.get("tool_calls", []):
                identity = call["id"]
                if re.fullmatch(r"prefill_[0-9a-f]{8}", identity):
                    self.prefill_ids.setdefault(
                        identity, f"prefill_{len(self.prefill_ids):08d}"
                    )

        def rename(value):
            if isinstance(value, str):
                for original, replacement in self.prefill_ids.items():
                    value = value.replace(original, replacement)
                return value
            if isinstance(value, list):
                return [rename(item) for item in value]
            if isinstance(value, dict):
                return {key: rename(item) for key, item in value.items()}
            return value

        body = {
            "model": MODEL,
            "messages": rename(projected),
            "max_tokens": min(4096, kwargs.get("max_tokens") or 4096),
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.get_parameter_schema(),
                    },
                }
                for t in tools
            ]
            body["tool_choice"] = "auto"
        response = self.transport(body)
        choice = response["choices"][0]
        message = assistant_message(choice["message"])
        calls = [
            ToolCall(
                id=c["id"],
                name=c["function"]["name"],
                arguments=c["function"]["arguments"],
            )
            for c in message.get("tool_calls", [])
        ]
        return LLMResponse(
            raw_response=response,
            content=message.get("content") or "",
            tool_calls=calls,
            finish_reason=choice["finish_reason"],
            assistant_message=message,
            usage=response["usage"],
        )

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        # Synchronous, absolute-deadline HTTP on this serial worker's main thread.
        return self.call(messages, tools, output_model, **kwargs)
