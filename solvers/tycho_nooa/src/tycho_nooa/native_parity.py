"""Austere native-tool control for frozen cognitive-parity packets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from nooa.unifiedllm import Tool, ToolCall, UnifiedLLM
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from tycho_nooa.cognitive_parity import CognitiveAnswer
from tycho_nooa.evidence import EvidenceQuery, WorkspaceEvidenceCapability
from tycho_nooa.model_task import ModelTask

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
    projection: Literal["full", "metadata"] = "full"


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
        packet: ModelTask,
        evidence: WorkspaceEvidenceCapability | None,
        max_llm_calls: int = 8,
        max_output_tokens: int | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if max_llm_calls < 1:
            raise ValueError("max_llm_calls must be positive")
        self._llm = llm
        self._packet = packet
        self._evidence = evidence
        self._max_llm_calls = max_llm_calls
        self._max_output_tokens = max_output_tokens
        self._messages = _base_messages(packet, system_prompt)
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
            try:
                calls = _tool_calls(
                    response.tool_calls, evidence_enabled=self._evidence is not None
                )
            except RuntimeError as error:
                self._record_rejected_turn(response.content, str(error))
                continue
            self._record_assistant_calls(calls, response.content)
            if all(call.name == "evidence_query" for call in calls):
                for call in calls:
                    self._handle_evidence_query(call)
                continue
            answer = self._handle_return(calls[0])
            if answer is not None:
                return answer
        raise RuntimeError("native parity exhausted its LLM-call budget")

    def _record_rejected_turn(self, content: object, reason: str) -> None:
        self._invalid_returns += 1
        self._messages.extend(
            [
                {
                    "role": "assistant",
                    "content": str(content) if content else "",
                },
                {
                    "role": "user",
                    "content": self._rejection_feedback(reason),
                },
            ]
        )

    def _rejection_feedback(self, reason: str) -> str:
        if self._evidence is None:
            instruction = "Call exactly one return_result tool."
        else:
            instruction = (
                "Call exactly one available tool, except that parallel "
                "evidence_query calls are allowed."
            )
        return f"PROVIDER_TURN_REJECTED: {reason}. {instruction}"

    async def _call_provider(self):
        response = await self._llm.acall(
            list(self._messages),
            tools=_native_tools(evidence_enabled=self._evidence is not None),
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
        if self._evidence is None:
            raise RuntimeError("evidence capability is unavailable")
        try:
            raw = NativeEvidenceArguments.model_validate_json(call.arguments)
            query = EvidenceQuery.model_validate(raw.model_dump(exclude_none=True))
            result = self._evidence.query(query)
        except (ValidationError, ValueError, TypeError) as error:
            self._invalid_returns += 1
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": f"EVIDENCE_QUERY_REJECTED: {error}",
                }
            )
            return
        self._evidence_queries += 1
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, separators=(",", ":")),
            }
        )

    def _handle_return(self, call: ToolCall) -> CognitiveAnswer | None:
        try:
            raw = NativeReturnArguments.model_validate_json(call.arguments)
            answer = ANSWER_ADAPTER.validate_python(raw.result)
            if answer.task_kind != self._packet.task.task_kind:
                raise ValueError("answer task_kind does not match its packet")
        except (ValidationError, ValueError, TypeError) as error:
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


def _base_messages(packet: ModelTask, system_prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": json.dumps(
                packet.treatment_payload(), sort_keys=True, separators=(",", ":")
            ),
        },
    ]


def _native_tools(*, evidence_enabled: bool = True) -> list[Tool]:
    def evidence_query(**kwargs: Any) -> dict[str, Any]:
        """Read evidence; exact RLE grids use comma-separated hex count:cell runs."""
        return kwargs

    def return_result(result: dict[str, Any]) -> dict[str, Any]:
        """Submit the final typed cognitive answer."""
        return result

    tools = [
        Tool(
            name="return_result",
            description=return_result.__doc__ or "Return result.",
            callable=return_result,
            parameters_model=NativeReturnArguments,
        )
    ]
    if evidence_enabled:
        tools.insert(
            0,
            Tool(
                name="evidence_query",
                description=evidence_query.__doc__ or "Query evidence.",
                callable=evidence_query,
                parameters_model=NativeEvidenceArguments,
            ),
        )
    return tools


def _tool_calls(
    tool_calls: list[ToolCall], *, evidence_enabled: bool = True
) -> list[ToolCall]:
    if not tool_calls:
        raise RuntimeError("native parity requires at least one tool call per turn")
    available = {"return_result"}
    if evidence_enabled:
        available.add("evidence_query")
    for call in tool_calls:
        if call.name not in available:
            raise RuntimeError(f"unavailable native parity tool: {call.name!r}")
    returns = [call for call in tool_calls if call.name == "return_result"]
    if returns and len(tool_calls) != 1:
        raise RuntimeError("typed return cannot coexist with another tool call")
    return tool_calls


def _commitment(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
