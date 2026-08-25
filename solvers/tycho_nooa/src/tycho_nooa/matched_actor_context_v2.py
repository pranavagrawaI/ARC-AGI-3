"""Matched Actor controllers over Context/Evidence Plane v1."""

from __future__ import annotations

import json
from typing import Any

from nooa.unifiedllm import Tool, ToolCall
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from tycho_nooa.context_evidence_v2 import ContextEvidencePlane, EvidenceViewRequest
from tycho_nooa.context_execution_v2 import (
    ContextComputationalSterileExecutor,
    ContextExecutionManifest,
)
from tycho_nooa.context_projection import ContextProjector
from tycho_nooa.matched_actor import (
    MatchedCodeActController,
    MatchedNativeActorController,
    NativeComputationRecord,
    _commitment,
)
from tycho_nooa.native_computation import (
    NativeRLEDecodeArguments,
    rle_decode_row_adapter,
)
from tycho_nooa.native_parity_v2 import NativeEvidenceArguments, NativeProviderTurn

CONTEXT_NATIVE_SYSTEM_PROMPT = (
    "Solve the frozen Actor task using provider-native tools. evidence_query "
    "acquires reality and returns a compact immutable receipt. evidence_view shows "
    "a bounded part of an already granted receipt. run_python is computation-only; "
    "inside it evidence_cache.load(handle) reads an already granted exact value. "
    "No operation can mutate canonical state or ARC. Call exactly one tool per "
    "provider turn. Finish with return_result."
)
CONTEXT_CODEACT_SYSTEM_PROMPT = (
    "Solve the frozen Actor task using execute_python and return_result. Python "
    "locals persist. evidence.query acquires reality and returns a compact receipt; "
    "evidence.view shows a bounded granted view; evidence_cache.load(handle) reads "
    "the exact already granted value. No operation can mutate canonical state or "
    "ARC. Finish with return_result."
)


class ContextRunPythonArguments(BaseModel):
    """Fresh Native computation with no authority to acquire new evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    values: dict[str, JsonValue] = Field(default_factory=dict)


class ContextNativeActorController(MatchedNativeActorController):
    """Native semantic control with exact evidence outside provider history."""

    def __init__(
        self,
        *,
        evidence_plane: ContextEvidencePlane,
        projector: ContextProjector,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            evidence=evidence_plane,
            **kwargs,
        )
        self._messages[0]["content"] = CONTEXT_NATIVE_SYSTEM_PROMPT
        self._evidence_plane = evidence_plane
        self._projector = projector
        self._receipts: list[dict[str, Any]] = []

    async def run(self) -> BaseModel:
        try:
            return await self._run_context_loop()
        finally:
            self._evidence_plane.close()

    async def _run_context_loop(self) -> BaseModel:
        for _ in range(self._max_llm_calls):
            response = await self._call_provider()
            try:
                call = _one_context_native_call(response.tool_calls)
            except RuntimeError as error:
                self._record_rejected_turn(response.content, str(error))
                continue
            self._record_assistant_calls([call], response.content)
            self._record_action(call)
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
        raise RuntimeError("context-plane Native Actor exhausted its LLM-call budget")

    async def _call_provider(self):
        messages = self._projector.native(
            self._messages,
            receipts=tuple(self._receipts),
        )
        response = await self._llm.acall(
            messages,
            tools=_context_native_tools(return_model=self._return_model),
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

    def _handle_evidence_query(self, call: ToolCall) -> None:
        try:
            raw = NativeEvidenceArguments.model_validate_json(call.arguments)
            receipt = self._evidence_plane.query(
                raw.model_dump(mode="json", exclude_none=True)
            )
        except (ValidationError, ValueError, TypeError) as error:
            self._append_tool_error(call, "EVIDENCE_QUERY_REJECTED", error)
            return
        self._evidence_queries += 1
        self._receipts.append(receipt)
        self._append_tool_value(call, receipt)

    def _handle_evidence_view(self, call: ToolCall) -> None:
        try:
            arguments = EvidenceViewRequest.model_validate_json(call.arguments)
            value = self._evidence_plane.view(arguments.model_dump(mode="json"))
        except (ValidationError, ValueError, TypeError) as error:
            self._append_tool_error(call, "EVIDENCE_VIEW_REJECTED", error)
            return
        self._append_tool_value(call, value)

    async def _handle_run_python(self, call: ToolCall) -> None:
        try:
            arguments = ContextRunPythonArguments.model_validate_json(call.arguments)
            manifest = ContextExecutionManifest(
                invocation_id=(
                    f"context-native:{self._packet.task.packet_id}:"
                    f"{len(self.provider_turns)}"
                ),
                role="actor",
                capabilities=("evidence.read",),
                input_values=arguments.values,
                wall_time_seconds=self._cell_timeout_seconds,
                output_bytes=4_096,
            )
            async with ContextComputationalSterileExecutor(
                manifest,
                return_type=dict[str, JsonValue],
                capability_handlers={"evidence.read": self._evidence_plane.read},
            ) as executor:
                result = await executor.run_cell(arguments.code)
        except (ValidationError, ValueError, TypeError, RuntimeError) as error:
            self._append_tool_error(call, "PYTHON_REJECTED", error)
            return
        self._record_context_computation(arguments.code, result)
        self._append_tool_value(
            call,
            {
                "status": result.status,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exception": (
                    None
                    if result.exception is None
                    else {
                        "type": result.exception.type,
                        "message": result.exception.message,
                    }
                ),
            },
        )

    def _record_context_computation(self, code: str, result: Any) -> None:
        self._python_cells += 1
        self._computation_records.append(
            NativeComputationRecord(
                call_index=len(self.provider_turns),
                code=code,
                status=result.status,
                stdout=result.stdout,
                error_code=(
                    None if result.exception is None else result.exception.type
                ),
            )
        )


class ContextCodeActController(MatchedCodeActController):
    """CodeAct control with the same scoped cache and bounded projection."""

    def __init__(
        self,
        *,
        evidence_plane: ContextEvidencePlane,
        projector: ContextProjector,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            evidence=evidence_plane,
            **kwargs,
        )
        self._system_prompt = CONTEXT_CODEACT_SYSTEM_PROMPT
        self._evidence_plane = evidence_plane
        self._projector = projector

    async def run(self) -> BaseModel:
        try:
            async with ContextComputationalSterileExecutor(
                self._execution_manifest(),
                return_type=self._return_model,
                capability_handlers=self._capability_handlers(),
                capability_auditors=self._capability_auditors(),
                provenance=self._provenance,
            ) as executor:
                for _ in range(self._max_llm_calls):
                    answer = await self._run_provider_turn(executor)
                    if answer is not None:
                        self._terminal_return_call = len(self.provider_turns)
                        return answer
            raise RuntimeError(
                "context-plane CodeAct Actor exhausted its LLM-call budget"
            )
        finally:
            self._evidence_plane.close()

    def _execution_manifest(self) -> ContextExecutionManifest:
        manifest = super()._execution_manifest()
        contracts = dict(manifest.capability_contracts)
        contracts["evidence.query"] = _context_evidence_query_contract()
        value = manifest.model_dump(mode="json")
        value.update(
            {
                "capabilities": (
                    "evidence.query",
                    "evidence.read",
                    "evidence.view",
                ),
                "capability_contracts": contracts,
                "output_bytes": 4_096,
            }
        )
        return ContextExecutionManifest.model_validate(value)

    def _capability_handlers(self) -> dict[str, Any]:
        return {
            "evidence.query": self._model_evidence_query,
            "evidence.read": self._evidence_plane.read,
            "evidence.view": self._evidence_plane.view,
        }

    def _capability_auditors(self) -> dict[str, Any]:
        return {}

    def reconstructed_messages(
        self, *, event_seq: int | None = None
    ) -> list[dict[str, Any]]:
        events = self._provenance.log.events
        if event_seq is not None:
            events = events[:event_seq]
        return [
            *self._base_context_messages(),
            *self._projector.codeact(events),
        ]

    def _base_context_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    self._packet.treatment_payload(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ]

    def _model_evidence_query(self, raw_query: dict[str, Any]) -> dict[str, Any]:
        query = dict(raw_query)
        query.setdefault("grid_encoding", "rle")
        return self._evidence_plane.query(query)


def _context_native_tools(*, return_model: type[BaseModel]) -> list[Tool]:
    def evidence_query(**kwargs: Any) -> dict[str, Any]:
        """Acquire evidence and return a compact receipt, not the bulk value."""
        return kwargs

    def evidence_view(**kwargs: Any) -> dict[str, Any]:
        """Show a bounded projection from an existing granted receipt."""
        return kwargs

    def run_python(code: str, values: dict[str, JsonValue]) -> str:
        """Compute over values or evidence_cache.load(handle); cannot query reality."""
        return code + repr(values)

    def return_result(**kwargs: Any) -> dict[str, Any]:
        """Submit the task's small typed answer."""
        return kwargs

    return [
        Tool(
            name="evidence_query",
            description=evidence_query.__doc__ or "Query evidence.",
            callable=evidence_query,
            parameters_model=NativeEvidenceArguments,
        ),
        Tool(
            name="evidence_view",
            description=evidence_view.__doc__ or "View granted evidence.",
            callable=evidence_view,
            parameters_model=EvidenceViewRequest,
        ),
        Tool(
            name="run_python",
            description=run_python.__doc__ or "Run Python.",
            callable=run_python,
            parameters_model=ContextRunPythonArguments,
        ),
        Tool(
            name="rle_decode_row",
            description="Decode one supplied RLE row without retrieving evidence.",
            callable=rle_decode_row_adapter,
            parameters_model=NativeRLEDecodeArguments,
        ),
        Tool(
            name="return_result",
            description=return_result.__doc__ or "Return result.",
            callable=return_result,
            parameters_model=return_model,
        ),
    ]


def _context_evidence_query_contract() -> dict[str, Any]:
    """Derive both control surfaces from the admitted provider query model."""
    return {
        "schema_version": "evidence_query.v1",
        "parameters": NativeEvidenceArguments.model_json_schema(),
        "example": {
            "kind": "transitions",
            "level": 0,
            "limit": 16,
            "projection": "metadata",
        },
        "result_schema": "evidence_receipt.v2",
    }


def _one_context_native_call(tool_calls: list[ToolCall]) -> ToolCall:
    if len(tool_calls) != 1:
        raise RuntimeError("context-plane Native Actor requires exactly one tool call")
    call = tool_calls[0]
    available = {
        "evidence_query",
        "evidence_view",
        "run_python",
        "rle_decode_row",
        "return_result",
    }
    if call.name not in available:
        raise RuntimeError(f"unavailable context-plane Native tool: {call.name!r}")
    return call
