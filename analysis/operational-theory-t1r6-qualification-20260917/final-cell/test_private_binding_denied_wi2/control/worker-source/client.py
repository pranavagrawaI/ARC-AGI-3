"""Actual NOOA client boundary to bounded receipt-preserving transport."""

import copy

from nooa.unifiedllm import UnifiedLLM, LLMResponse, ToolCall
from transport import MODEL
from prefill import CallIDs
from protocol import assistant_message, project_messages


class ActorClient(UnifiedLLM):
    def __init__(self, transport):
        super().__init__(model=MODEL)
        self.transport = transport
        self.ids = CallIDs()
        self._context_window = 262144

    @property
    def context_window(self):
        return self._context_window

    def count_tokens(self, text):
        # Deliberately conservative local truncation bound. Actual receipt is authoritative.
        return len(text.encode()) + 1

    def call(self, messages, tools=None, output_model=None, **kwargs):
        projected = project_messages(copy.deepcopy(messages))
        body = {
            "model": MODEL,
            "messages": projected,
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
        self.ids.accept_provider([c["id"] for c in message.get("tool_calls", [])])
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
