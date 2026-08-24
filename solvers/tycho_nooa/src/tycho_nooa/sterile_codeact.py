"""T1-controlled CodeAct loop whose generated Python executes only in T2."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from nooa.unifiedllm import Tool, ToolCall, UnifiedLLM
from pydantic import BaseModel, ConfigDict

from tycho_nooa.cognitive_parity import (
    CognitiveAnswer,
    CognitiveReturnArguments,
    RetrievalAnswer,
)
from tycho_nooa.compatibility_return import RetrievalReturnArguments
from tycho_nooa.evidence import EvidenceCapability
from tycho_nooa.execution_events import (
    InvocationProvenance,
    NooaExecutionProjection,
    render_execution_context,
)
from tycho_nooa.model_task import ModelTask
from tycho_nooa.sterile_execution import (
    ExecutionManifest,
    SterileExecutor,
    SterileReturnValidationError,
)

SYSTEM_PROMPT = (
    "You are solving a frozen cognitive-parity task. Use only "
    "execute_python(code) and return_result(result). Python locals "
    "persist across cells; evidence.query is available inside Python. Its exact "
    "grid encoding defaults to RLE: each row is comma-separated hex count:cell runs. "
    "There is no self, filesystem authority, network, Builder, planner, "
    "or environment action capability. Finish with the task's typed result."
)


class ExecutePythonArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str


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
        packet: ModelTask,
        evidence: EvidenceCapability | None,
        provenance: InvocationProvenance,
        max_llm_calls: int = 8,
        max_output_tokens: int | None = None,
        cell_timeout_seconds: float = 5.0,
        system_prompt: str = SYSTEM_PROMPT,
        flat_retrieval_return: bool = False,
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
        self._system_prompt = system_prompt
        self._flat_retrieval_return = flat_retrieval_return
        self._provider_turns: list[ProviderTurn] = []
        self._transport_adherence: Counter[str] = Counter()

    @property
    def provider_turns(self) -> tuple[ProviderTurn, ...]:
        return tuple(self._provider_turns)

    @property
    def transport_adherence(self) -> dict[str, int]:
        return dict(self._transport_adherence)

    async def run(self) -> CognitiveAnswer:
        async with SterileExecutor(
            self._execution_manifest(),
            return_type=(
                RetrievalAnswer if self._flat_retrieval_return else CognitiveAnswer
            ),
            capability_handlers=self._capability_handlers(),
            capability_auditors=self._capability_auditors(),
            provenance=self._provenance,
        ) as executor:
            for _ in range(self._max_llm_calls):
                answer = await self._run_provider_turn(executor)
                if answer is not None:
                    return answer
        raise RuntimeError("sterile CodeAct exhausted its LLM-call budget")

    def _execution_manifest(self) -> ExecutionManifest:
        payload = self._packet.treatment_payload()
        capabilities = ("evidence.query",) if self._evidence is not None else ()
        evidence_contract = payload.get("evidence_contract")
        return ExecutionManifest(
            invocation_id=self._provenance.invocation_id,
            role="actor",
            capabilities=capabilities,
            capability_contracts=(
                {"evidence.query": evidence_contract}
                if capabilities and isinstance(evidence_contract, dict)
                else {}
            ),
            input_values={"task": payload},
            wall_time_seconds=self._cell_timeout_seconds,
        )

    def _capability_handlers(self) -> dict[str, Any]:
        return (
            {"evidence.query": self._model_evidence_query}
            if self._evidence is not None
            else {}
        )

    def _capability_auditors(self) -> dict[str, Any]:
        if self._evidence is None or not hasattr(self._evidence, "pop_audit"):
            return {}
        return {"evidence.query": self._evidence.pop_audit}

    async def _run_provider_turn(
        self, executor: SterileExecutor[CognitiveAnswer]
    ) -> CognitiveAnswer | None:
        response = await self._call_provider()
        try:
            tool_call = _one_tool_call(response.tool_calls)
            code, arguments = _tool_call_execution(
                tool_call,
                response.content,
                flat_retrieval_return=self._flat_retrieval_return,
            )
        except (RuntimeError, ValueError, TypeError) as error:
            category = _rejected_transport_class(
                response.tool_calls, response.content
            )
            self._transport_adherence[category] += 1
            self._provenance.provider_turn_rejected(str(error))
            return None
        self._transport_adherence[f"valid_{tool_call.name}"] += 1
        try:
            result = await executor.run_cell(
                code,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                tool_arguments=arguments,
                tool_reasoning=(
                    None if response.content is None else str(response.content)
                ),
            )
        except SterileReturnValidationError:
            return None
        return result.return_value

    def _model_evidence_query(self, raw_query: dict[str, Any]) -> dict[str, Any]:
        if self._evidence is None:
            raise RuntimeError("evidence capability is unavailable")
        query = {**raw_query}
        query.setdefault("grid_encoding", "rle")
        return self._evidence.query(query)

    async def _call_provider(self):
        messages = self.reconstructed_messages()
        event_seq = len(self._provenance.log.events)
        response = await self._llm.acall(
            messages,
            tools=_codeact_tools(
                flat_retrieval_return=self._flat_retrieval_return
            ),
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

    def reconstructed_messages(
        self, *, event_seq: int | None = None
    ) -> list[dict[str, Any]]:
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
        return [*_base_messages(self._packet, self._system_prompt), *visible_events]


def _base_messages(packet: ModelTask, system_prompt: str) -> list[dict[str, str]]:
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


def _codeact_tools(*, flat_retrieval_return: bool = False) -> list[Tool]:
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
            parameters_model=(
                RetrievalReturnArguments
                if flat_retrieval_return
                else CognitiveReturnArguments
            ),
        ),
    ]


def _one_tool_call(tool_calls: list[ToolCall]) -> ToolCall:
    if len(tool_calls) != 1:
        raise RuntimeError("sterile CodeAct requires exactly one tool call per turn")
    call = tool_calls[0]
    if call.name not in {"execute_python", "return_result"}:
        raise RuntimeError(f"unavailable sterile CodeAct tool: {call.name!r}")
    return call


def _tool_call_execution(
    tool_call: ToolCall,
    content: object,
    *,
    flat_retrieval_return: bool = False,
) -> tuple[str, dict[str, Any]]:
    arguments = json.loads(tool_call.arguments)
    if not isinstance(arguments, dict):
        raise TypeError("tool arguments must be an object")
    prefix = _comment(str(content)) if content else ""
    if tool_call.name == "execute_python":
        code = ExecutePythonArguments.model_validate(arguments).code
    else:
        result = (
            RetrievalReturnArguments.model_validate(arguments).model_dump(mode="json")
            if flat_retrieval_return
            else _return_result_value(arguments)
        )
        code = f"return_result({result!r})"
    return (f"{prefix}\n{code}" if prefix else code), arguments


def _rejected_transport_class(
    tool_calls: list[ToolCall], content: object
) -> str:
    if not tool_calls and content:
        return "plain_prose"
    if len(tool_calls) == 1 and tool_calls[0].name not in {
        "execute_python",
        "return_result",
    }:
        return "wrong_tool"
    return "malformed_tool_call"


def _return_result_value(arguments: dict[str, Any]) -> dict[str, Any]:
    """Preserve T2 return validation while T1 advertises the exact schema."""
    if set(arguments) != {"result"} or not isinstance(arguments["result"], dict):
        raise TypeError("return_result requires exactly one object-valued result")
    return arguments["result"]


def _comment(value: str) -> str:
    return "\n".join(f"# {line}" if line else "#" for line in value.splitlines())


def _commitment(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
