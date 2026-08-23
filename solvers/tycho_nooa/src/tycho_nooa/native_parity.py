"""Austere native-tool control for frozen cognitive-parity packets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from nooa.unifiedllm import Tool, ToolCall, UnifiedLLM
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from tycho_nooa.cognitive_parity import CognitiveAnswer, CognitiveParityPacket
from tycho_nooa.evidence import EvidenceQuery, WorkspaceEvidenceCapability

ANSWER_ADAPTER = TypeAdapter(CognitiveAnswer)
SYSTEM_PROMPT = (
    "You are solving a frozen cognitive-parity task. Use only evidence_query "
    "and return_result. You have no Python, filesystem, network, Builder, planner, "
    "or environment action capability."
)


class NativeEvidenceArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["frames", "transitions"]
    level: int | None = None
    after: dict[str, int] | None = None
    frontier: dict[str, int] | None = None
    limit: int = 16
    grid_encoding: Literal["raw", "rle"] = "rle"


class NativeReturnArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: dict[str, Any]


@dataclass(frozen=True)
class NativeProviderTurn:
    messages_sha256: str
    usage: dict[str, int] | None
    finish_reason: str
    tool_names: tuple[str, ...]


class NativeParityController:
    """Offer only structured evidence retrieval and the shared typed return."""

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        packet: CognitiveParityPacket,
        evidence: WorkspaceEvidenceCapability,
        max_llm_calls: int = 8,
        max_output_tokens: int | None = None,
    ):
        if max_llm_calls < 1:
            raise ValueError("max_llm_calls must be positive")
        self._llm = llm
        self._packet = packet
        self._evidence = evidence
        self._max_llm_calls = max_llm_calls
        self._max_output_tokens = max_output_tokens
        self._messages = _base_messages(packet)
        self._provider_turns: list[NativeProviderTurn] = []
        self._evidence_queries = 0
        self._invalid_returns = 0

    @property
    def provider_turns(self) -> tuple[NativeProviderTurn, ...]:
        return tuple(self._provider_turns)

    @property
    def evidence_queries(self) -> int:
        return self._evidence_queries

    @property
    def invalid_returns(self) -> int:
        return self._invalid_returns

    async def run(self) -> CognitiveAnswer:
        for _ in range(self._max_llm_calls):
            response = await self._call_provider()
            calls = _tool_calls(response.tool_calls)
            self._record_assistant_calls(calls, response.content)
            if all(call.name == "evidence_query" for call in calls):
                for call in calls:
                    self._handle_evidence_query(call)
                continue
            answer = self._handle_return(calls[0])
            if answer is not None:
                return answer
        raise RuntimeError("native parity exhausted its LLM-call budget")

    async def _call_provider(self):
        response = await self._llm.acall(
            list(self._messages),
            tools=_native_tools(),
            output_model=None,
            **self._sampling_kwargs(),
        )
        self._provider_turns.append(
            NativeProviderTurn(
                messages_sha256=_commitment(self._messages),
                usage=response.usage,
                finish_reason=response.finish_reason,
                tool_names=tuple(call.name for call in response.tool_calls),
            )
        )
        return response

    def _record_assistant_calls(self, calls: list[ToolCall], content: object) -> None:
        self._messages.append(
            {
                "role": "assistant",
                "content": str(content) if content else "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in calls
                ],
            }
        )

    def _handle_evidence_query(self, call: ToolCall) -> None:
        raw = NativeEvidenceArguments.model_validate_json(call.arguments)
        query = EvidenceQuery.model_validate(raw.model_dump(exclude_none=True))
        result = self._evidence.query(query)
        self._evidence_queries += 1
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, separators=(",", ":")),
            }
        )

    def _handle_return(self, call: ToolCall) -> CognitiveAnswer | None:
        raw = NativeReturnArguments.model_validate_json(call.arguments)
        try:
            answer = ANSWER_ADAPTER.validate_python(raw.result)
            if answer.task_kind != self._packet.task.task_kind:
                raise ValueError("answer task_kind does not match its packet")
        except (ValidationError, ValueError) as error:
            self._invalid_returns += 1
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": f"RETURN_REJECTED: {error}",
                }
            )
            return None
        return answer

    def _sampling_kwargs(self) -> dict[str, int]:
        return (
            {}
            if self._max_output_tokens is None
            else {"max_tokens": self._max_output_tokens}
        )


def _base_messages(packet: CognitiveParityPacket) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                packet.treatment_payload(), sort_keys=True, separators=(",", ":")
            ),
        },
    ]


def _native_tools() -> list[Tool]:
    def evidence_query(**kwargs: Any) -> dict[str, Any]:
        """Read evidence; exact RLE grids use comma-separated hex count:cell runs."""
        return kwargs

    def return_result(result: dict[str, Any]) -> dict[str, Any]:
        """Submit the final typed cognitive answer."""
        return result

    return [
        Tool(
            name="evidence_query",
            description=evidence_query.__doc__ or "Query evidence.",
            callable=evidence_query,
            parameters_model=NativeEvidenceArguments,
        ),
        Tool(
            name="return_result",
            description=return_result.__doc__ or "Return result.",
            callable=return_result,
            parameters_model=NativeReturnArguments,
        ),
    ]


def _tool_calls(tool_calls: list[ToolCall]) -> list[ToolCall]:
    if not tool_calls:
        raise RuntimeError("native parity requires at least one tool call per turn")
    for call in tool_calls:
        if call.name not in {"evidence_query", "return_result"}:
            raise RuntimeError(f"unavailable native parity tool: {call.name!r}")
    returns = [call for call in tool_calls if call.name == "return_result"]
    if returns and len(tool_calls) != 1:
        raise RuntimeError("typed return cannot coexist with another tool call")
    return tool_calls


def _commitment(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
