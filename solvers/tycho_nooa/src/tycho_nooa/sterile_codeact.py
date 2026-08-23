"""T1-controlled CodeAct loop whose generated Python executes only in T2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from nooa.unifiedllm import Tool, ToolCall, UnifiedLLM
from pydantic import BaseModel, ConfigDict

from tycho_nooa.cognitive_parity import CognitiveAnswer, CognitiveParityPacket
from tycho_nooa.evidence import WorkspaceEvidenceCapability
from tycho_nooa.execution_events import (
    InvocationProvenance,
    NooaExecutionProjection,
    render_execution_context,
)
from tycho_nooa.sterile_execution import (
    ExecutionManifest,
    SterileExecutor,
    SterileReturnValidationError,
)


class ExecutePythonArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str


class ReturnResultArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: dict[str, Any]


@dataclass(frozen=True)
class ProviderTurn:
    """The exact reconstructed history used for one provider call."""

    event_seq: int
    messages: tuple[dict[str, Any], ...]
    messages_sha256: str
    usage: dict[str, int] | None
    finish_reason: str
    tool_names: tuple[str, ...]


class SterileCodeActController:
    """Preserve NOOA's two-tool loop while replacing ambient execution authority."""

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        packet: CognitiveParityPacket,
        evidence: WorkspaceEvidenceCapability,
        provenance: InvocationProvenance,
        max_llm_calls: int = 8,
        max_output_tokens: int | None = None,
        cell_timeout_seconds: float = 5.0,
    ):
        if max_llm_calls < 1:
            raise ValueError("max_llm_calls must be positive")
        self._llm = llm
        self._packet = packet
        self._evidence = evidence
        self._provenance = provenance
        self._max_llm_calls = max_llm_calls
        self._max_output_tokens = max_output_tokens
        self._cell_timeout_seconds = cell_timeout_seconds
        self._provider_turns: list[ProviderTurn] = []

    @property
    def provider_turns(self) -> tuple[ProviderTurn, ...]:
        return tuple(self._provider_turns)

    async def run(self) -> CognitiveAnswer:
        manifest = ExecutionManifest(
            invocation_id=self._provenance.invocation_id,
            role="actor",
            capabilities=("evidence.query",),
            input_values={"task": self._packet.treatment_payload()},
            wall_time_seconds=self._cell_timeout_seconds,
        )
        async with SterileExecutor(
            manifest,
            return_type=CognitiveAnswer,
            capability_handlers={"evidence.query": self._evidence.query},
            provenance=self._provenance,
        ) as executor:
            for _ in range(self._max_llm_calls):
                response = await self._call_provider()
                tool_call = _one_tool_call(response.tool_calls)
                code = _tool_call_code(tool_call, response.content)
                try:
                    result = await executor.run_cell(code)
                except SterileReturnValidationError:
                    continue
                if result.return_value is not None:
                    return result.return_value
        raise RuntimeError("sterile CodeAct exhausted its LLM-call budget")

    async def _call_provider(self):
        messages = self.reconstructed_messages()
        event_seq = len(self._provenance.log.events)
        response = await self._llm.acall(
            messages,
            tools=_codeact_tools(),
            output_model=None,
            **self._sampling_kwargs(),
        )
        self._provider_turns.append(
            ProviderTurn(
                event_seq=event_seq,
                messages=tuple(messages),
                messages_sha256=_commitment(messages),
                usage=response.usage,
                finish_reason=response.finish_reason,
                tool_names=tuple(call.name for call in response.tool_calls),
            )
        )
        return response

    def _sampling_kwargs(self) -> dict[str, int]:
        return (
            {}
            if self._max_output_tokens is None
            else {"max_tokens": self._max_output_tokens}
        )

    def reconstructed_messages(self, *, event_seq: int | None = None) -> list[dict[str, Any]]:
        """Build every turn from static task data plus a raw-event projection."""
        events = self._provenance.log.events
        if event_seq is not None:
            events = events[:event_seq]
        projected = render_execution_context(
            NooaExecutionProjection(events).event_manager()
        )
        visible_events = [
            message
            for message in projected
            if message.get("content") or message.get("role") != "system"
        ]
        return [*_base_messages(self._packet), *visible_events]


def _base_messages(packet: CognitiveParityPacket) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are solving a frozen cognitive-parity task. Use only "
                "execute_python(code) and return_result(result). Python locals "
                "persist across cells; evidence.query is available inside Python. "
                "There is no self, filesystem authority, network, Builder, planner, "
                "or environment action capability. Finish with the task's typed result."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                packet.treatment_payload(), sort_keys=True, separators=(",", ":")
            ),
        },
    ]


def _codeact_tools() -> list[Tool]:
    def execute_python(code: str) -> str:
        """Execute one sterile Python cell; invocation locals persist."""
        return code

    def return_result(result: dict[str, Any]) -> dict[str, Any]:
        """Submit the final typed cognitive answer."""
        return result

    return [
        Tool(
            name="execute_python",
            description=execute_python.__doc__ or "Execute Python.",
            callable=execute_python,
            parameters_model=ExecutePythonArguments,
        ),
        Tool(
            name="return_result",
            description=return_result.__doc__ or "Return a result.",
            callable=return_result,
            parameters_model=ReturnResultArguments,
        ),
    ]


def _one_tool_call(tool_calls: list[ToolCall]) -> ToolCall:
    if len(tool_calls) != 1:
        raise RuntimeError("sterile CodeAct requires exactly one tool call per turn")
    call = tool_calls[0]
    if call.name not in {"execute_python", "return_result"}:
        raise RuntimeError(f"unavailable sterile CodeAct tool: {call.name!r}")
    return call


def _tool_call_code(tool_call: ToolCall, content: object) -> str:
    arguments = json.loads(tool_call.arguments)
    if not isinstance(arguments, dict):
        raise TypeError("tool arguments must be an object")
    prefix = _comment(str(content)) if content else ""
    if tool_call.name == "execute_python":
        code = ExecutePythonArguments.model_validate(arguments).code
    else:
        result = ReturnResultArguments.model_validate(arguments).result
        code = f"return_result({result!r})"
    return f"{prefix}\n{code}" if prefix else code


def _comment(value: str) -> str:
    return "\n".join(f"# {line}" if line else "#" for line in value.splitlines())


def _commitment(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
