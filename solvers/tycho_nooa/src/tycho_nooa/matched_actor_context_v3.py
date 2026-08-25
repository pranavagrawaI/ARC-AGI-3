"""Native Actor with generic bulk evidence acquisition into granted memory."""

from __future__ import annotations

from typing import Any

from nooa.unifiedllm import Tool, ToolCall
from pydantic import BaseModel, ValidationError

from tycho_nooa.context_evidence_v3 import (
    ContextEvidencePlane,
    EvidenceAcquireArguments,
)
from tycho_nooa.matched_actor import _commitment
from tycho_nooa.matched_actor_context_v2 import (
    ContextNativeActorController as _V2NativeController,
)
from tycho_nooa.matched_actor_context_v2 import _context_native_tools
from tycho_nooa.native_parity_v2 import NativeProviderTurn

CONTEXT_NATIVE_ACQUISITION_SYSTEM_PROMPT = (
    "Solve the frozen Actor task using provider-native tools. evidence_query "
    "acquires one bounded page. evidence_acquire materializes the complete bounded "
    "set selected by kind, level, projection, and encoding into immutable external "
    "memory and returns a compact receipt. run_python is computation-only; inside "
    "it evidence_cache.load(handle) reads an already granted exact value. No "
    "operation can mutate canonical state or ARC. Call exactly one tool per "
    "provider turn. Finish with return_result."
)


class ContextNativeAcquisitionController(_V2NativeController):
    """Native control whose one new power is generic complete-set acquisition."""

    def __init__(
        self,
        *,
        evidence_plane: ContextEvidencePlane,
        **kwargs: Any,
    ) -> None:
        super().__init__(evidence_plane=evidence_plane, **kwargs)
        self._messages[0]["content"] = CONTEXT_NATIVE_ACQUISITION_SYSTEM_PROMPT

    async def _run_context_loop(self) -> BaseModel:
        for _ in range(self._max_llm_calls):
            response = await self._call_provider()
            try:
                call = _one_native_acquisition_call(response.tool_calls)
            except RuntimeError as error:
                self._record_rejected_turn(response.content, str(error))
                continue
            self._record_assistant_calls([call], response.content)
            self._record_action(call)
            if call.name == "evidence_acquire":
                self._handle_evidence_acquire(call)
                continue
            if call.name == "evidence_query":
                self._handle_evidence_query(call)
                continue
            if call.name == "evidence_view":
                self._handle_evidence_view(call)
                continue
            if call.name == "run_python":
                await self._handle_run_python(call)
                continue
            if call.name == "rle_decode_row":
                self._handle_rle_decode(call)
                continue
            answer = self._handle_return(call)
            if answer is not None:
                self._terminal_return_call = len(self.provider_turns)
                return answer
        raise RuntimeError("Native acquisition Actor exhausted its LLM-call budget")

    async def _call_provider(self):
        messages = self._projector.native(
            self._messages,
            receipts=tuple(self._receipts),
        )
        response = await self._llm.acall(
            messages,
            tools=_native_acquisition_tools(return_model=self._return_model),
            output_model=None,
            **self._sampling_kwargs(),
        )
        self._provider_turns.append(
            NativeProviderTurn(
                messages_sha256=_commitment(messages),
                usage=response.usage,
                finish_reason=response.finish_reason,
                tool_names=tuple(call.name for call in response.tool_calls),
            )
        )
        return response

    def _handle_evidence_acquire(self, call: ToolCall) -> None:
        try:
            arguments = EvidenceAcquireArguments.model_validate_json(call.arguments)
            receipt = self._evidence_plane.acquire(
                arguments.model_dump(mode="json", exclude_none=True)
            )
        except (ValidationError, ValueError, TypeError, RuntimeError) as error:
            self._append_tool_error(call, "EVIDENCE_ACQUIRE_REJECTED", error)
            return
        self._evidence_queries += 1
        self._receipts.append(receipt)
        self._append_tool_value(call, receipt)


def _native_acquisition_tools(*, return_model: type[BaseModel]) -> list[Tool]:
    def evidence_acquire(**kwargs: Any) -> dict[str, Any]:
        """Materialize one complete bounded evidence set into immutable memory."""
        return kwargs

    tools = _context_native_tools(return_model=return_model)
    tools.insert(
        1,
        Tool(
            name="evidence_acquire",
            description=evidence_acquire.__doc__ or "Acquire evidence.",
            callable=evidence_acquire,
            parameters_model=EvidenceAcquireArguments,
        ),
    )
    return tools


def _one_native_acquisition_call(tool_calls: list[ToolCall]) -> ToolCall:
    if len(tool_calls) != 1:
        raise RuntimeError("Native acquisition Actor requires exactly one tool call")
    call = tool_calls[0]
    available = {
        "evidence_query",
        "evidence_acquire",
        "evidence_view",
        "run_python",
        "rle_decode_row",
        "return_result",
    }
    if call.name not in available:
        raise RuntimeError(f"unavailable Native acquisition tool: {call.name!r}")
    return call
