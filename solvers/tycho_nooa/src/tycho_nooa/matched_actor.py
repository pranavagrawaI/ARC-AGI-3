"""Matched control-language treatments over one authority-free substrate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from nooa.unifiedllm import Tool, ToolCall, UnifiedLLM
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from tycho_nooa.cognitive_parity import (
    ActionProposal,
    CognitiveAnswer,
    HypothesisAnswer,
    InvariantAnswer,
    RetrievalAnswer,
    Scope,
)
from tycho_nooa.computational_evidence import describe_evidence_page
from tycho_nooa.computational_execution import ComputationalSterileExecutor
from tycho_nooa.evidence import EvidenceCapability, EvidenceQuery, EvidenceRef
from tycho_nooa.execution_events import InvocationProvenance
from tycho_nooa.model_task import ModelTask
from tycho_nooa.models import ActionDecision
from tycho_nooa.native_computation import (
    NativeRLEDecodeArguments,
    dispatch_native_rle_decode,
    rle_decode_row_adapter,
)
from tycho_nooa.native_parity import (
    NativeEvidenceArguments,
    NativeParityController,
    NativeProviderTurn,
)
from tycho_nooa.sterile_codeact import SterileCodeActController
from tycho_nooa.sterile_execution import ExecutionManifest

NATIVE_SYSTEM_PROMPT = (
    "Solve the frozen Actor task using provider-native tools. evidence_query reads "
    "reality. run_python computes only over explicit values and starts a fresh "
    "sterile workspace each call. rle_decode_row is the shared pure decoder. "
    "No tool can mutate canonical state or ARC. Finish with return_result."
)
CODEACT_SYSTEM_PROMPT = (
    "Solve the frozen Actor task using execute_python and return_result. Python "
    "locals persist; evidence.query reads reality and rle.decode_row is the shared "
    "pure decoder. There is no canonical-state or ARC mutation authority. Finish "
    "with return_result."
)


class _FlatReturn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MatchedRetrievalReturn(_FlatReturn):
    conclusion: str
    evidence_refs: tuple[EvidenceRef, ...]


class MatchedDiscriminationReturn(_FlatReturn):
    conclusion: str
    verdict: str
    falsified: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]


class MatchedInvariantReturn(_FlatReturn):
    invariant_id: str
    invariant: str
    invariant_atoms: dict[str, bool | int | float | str]
    evidence_refs: tuple[EvidenceRef, ...]
    scope: Scope


class MatchedActionReturn(_FlatReturn):
    decision: ActionDecision
    evidence_refs: tuple[EvidenceRef, ...]
    expected_effect: str
    information_goal: str | None = None


type MatchedReturn = (
    MatchedRetrievalReturn
    | MatchedDiscriminationReturn
    | MatchedInvariantReturn
    | MatchedActionReturn
)

RETURN_MODELS: dict[str, type[_FlatReturn]] = {
    "retrieve": MatchedRetrievalReturn,
    "discriminate": MatchedDiscriminationReturn,
    "recover_invariant": MatchedInvariantReturn,
    "propose_action": MatchedActionReturn,
}


class RunPythonArguments(BaseModel):
    """Computation-only Native Python; reality must arrive as explicit values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    values: dict[str, JsonValue]


@dataclass(frozen=True)
class ModelAction:
    call_index: int
    tool_call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class NativeComputationRecord:
    call_index: int
    code: str
    status: str
    stdout: str
    error_code: str | None


def return_model_for_task(packet: ModelTask) -> type[_FlatReturn]:
    try:
        return RETURN_MODELS[packet.task.task_kind]
    except KeyError as error:
        raise ValueError(f"unsupported matched task kind: {packet.task.task_kind}") from error


def cognitive_answer(packet: ModelTask, value: MatchedReturn) -> CognitiveAnswer:
    """Promote a small model-facing value into the evaluator's rich type."""
    kind = packet.task.task_kind
    if kind == "retrieve" and isinstance(value, MatchedRetrievalReturn):
        return RetrievalAnswer(
            conclusion=value.conclusion,
            evidence_refs=value.evidence_refs,
        )
    if kind == "discriminate" and isinstance(value, MatchedDiscriminationReturn):
        return HypothesisAnswer(
            supported=value.verdict,
            falsified=value.falsified,
            evidence_refs=value.evidence_refs,
            conclusion=value.conclusion,
        )
    if kind == "recover_invariant" and isinstance(value, MatchedInvariantReturn):
        return InvariantAnswer(
            invariant_id=value.invariant_id,
            invariant=value.invariant,
            invariant_atoms=value.invariant_atoms,
            evidence_refs=value.evidence_refs,
            scope=value.scope,
        )
    if kind == "propose_action" and isinstance(value, MatchedActionReturn):
        return ActionProposal(
            decision=value.decision,
            evidence_refs=value.evidence_refs,
            expected_effect=value.expected_effect,
            information_goal=value.information_goal,
        )
    raise TypeError("matched return does not belong to the packet task kind")


class MatchedNativeActorController(NativeParityController):
    """Native semantic tools plus fresh computation-only Python calls."""

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        packet: ModelTask,
        evidence: EvidenceCapability,
        max_llm_calls: int,
        max_output_tokens: int,
        cell_timeout_seconds: float,
    ):
        super().__init__(
            llm=llm,
            packet=packet,
            evidence=evidence,
            max_llm_calls=max_llm_calls,
            max_output_tokens=max_output_tokens,
            system_prompt=NATIVE_SYSTEM_PROMPT,
            return_model=return_model_for_task(packet),
        )
        self._cell_timeout_seconds = cell_timeout_seconds
        self._python_cells = 0
        self._computation_records: list[NativeComputationRecord] = []
        self._model_actions: list[ModelAction] = []
        self._terminal_return_call: int | None = None
        self._successful_decode_calls: list[int] = []

    @property
    def python_cells(self) -> int:
        return self._python_cells

    @property
    def computation_records(self) -> tuple[NativeComputationRecord, ...]:
        return tuple(self._computation_records)

    @property
    def model_actions(self) -> tuple[ModelAction, ...]:
        return tuple(self._model_actions)

    @property
    def terminal_return_call(self) -> int | None:
        return self._terminal_return_call

    @property
    def successful_decode_calls(self) -> tuple[int, ...]:
        calls = list(self._successful_decode_calls)
        calls.extend(
            record.call_index
            for record in self._computation_records
            if record.status == "complete" and "rle.decode_row" in record.code
        )
        return tuple(sorted(set(calls)))

    async def run(self) -> BaseModel:
        for _ in range(self._max_llm_calls):
            response = await self._call_provider()
            try:
                call = _one_native_tool_call(response.tool_calls)
            except RuntimeError as error:
                self._record_rejected_turn(response.content, str(error))
                continue
            self._record_assistant_calls([call], response.content)
            self._record_action(call)
            if call.name == "evidence_query":
                self._handle_evidence_query(call)
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
        raise RuntimeError("matched Native Actor exhausted its LLM-call budget")

    async def _call_provider(self):
        response = await self._llm.acall(
            list(self._messages),
            tools=_native_actor_tools(return_model=self._return_model),
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

    def _handle_evidence_query(self, call: ToolCall) -> None:
        try:
            raw = NativeEvidenceArguments.model_validate_json(call.arguments)
            query = EvidenceQuery.model_validate(raw.model_dump(exclude_none=True))
            result = describe_evidence_page(self._evidence.query(query))
        except (ValidationError, ValueError, TypeError) as error:
            self._append_tool_error(call, "EVIDENCE_QUERY_REJECTED", error)
            return
        self._evidence_queries += 1
        self._append_tool_value(call, result)

    async def _handle_run_python(self, call: ToolCall) -> None:
        try:
            arguments = RunPythonArguments.model_validate_json(call.arguments)
            manifest = ExecutionManifest(
                invocation_id=(
                    f"matched-native:{self._packet.task.packet_id}:"
                    f"{len(self.provider_turns)}"
                ),
                role="actor",
                input_values=arguments.values,
                wall_time_seconds=self._cell_timeout_seconds,
            )
            async with ComputationalSterileExecutor(
                manifest,
                return_type=dict[str, JsonValue],
            ) as executor:
                result = await executor.run_cell(arguments.code)
        except (ValidationError, ValueError, TypeError, RuntimeError) as error:
            self._append_tool_error(call, "PYTHON_REJECTED", error)
            return
        self._python_cells += 1
        error_code = None if result.exception is None else result.exception.type
        self._computation_records.append(
            NativeComputationRecord(
                call_index=len(self.provider_turns),
                code=arguments.code,
                status=result.status,
                stdout=result.stdout,
                error_code=error_code,
            )
        )
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

    def _handle_rle_decode(self, call: ToolCall) -> None:
        try:
            arguments = json.loads(call.arguments)
            result = dispatch_native_rle_decode(arguments)
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            self._append_tool_error(call, "RLE_DECODE_REJECTED", error)
            return
        self._successful_decode_calls.append(len(self.provider_turns))
        self._append_tool_value(call, result)

    def _append_tool_error(
        self, call: ToolCall, label: str, error: BaseException
    ) -> None:
        self._invalid_returns += 1
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": f"{label}: {error}",
            }
        )

    def _append_tool_value(self, call: ToolCall, value: object) -> None:
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(value, separators=(",", ":")),
            }
        )

    def _record_action(self, call: ToolCall) -> None:
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError:
            arguments = {"malformed": call.arguments}
        self._model_actions.append(
            ModelAction(len(self.provider_turns), call.id, call.name, arguments)
        )


class MatchedCodeActController(SterileCodeActController):
    """Persistent CodeAct orchestration over the shared pure helper surface."""

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        packet: ModelTask,
        evidence: EvidenceCapability,
        provenance: InvocationProvenance,
        max_llm_calls: int,
        max_output_tokens: int,
        cell_timeout_seconds: float,
    ):
        super().__init__(
            llm=llm,
            packet=packet,
            evidence=evidence,
            provenance=provenance,
            max_llm_calls=max_llm_calls,
            max_output_tokens=max_output_tokens,
            cell_timeout_seconds=cell_timeout_seconds,
            system_prompt=CODEACT_SYSTEM_PROMPT,
            return_model=return_model_for_task(packet),
        )
        self._model_actions: list[ModelAction] = []
        self._terminal_return_call: int | None = None

    @property
    def model_actions(self) -> tuple[ModelAction, ...]:
        return tuple(self._model_actions)

    @property
    def terminal_return_call(self) -> int | None:
        return self._terminal_return_call

    async def run(self) -> BaseModel:
        async with ComputationalSterileExecutor(
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
        raise RuntimeError("matched CodeAct Actor exhausted its LLM-call budget")

    async def _call_provider(self):
        response = await super()._call_provider()
        call_index = len(self.provider_turns)
        for call in response.tool_calls:
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                arguments = {"malformed": call.arguments}
            self._model_actions.append(
                ModelAction(call_index, call.id, call.name, arguments)
            )
        return response

    def _model_evidence_query(self, raw_query: dict[str, Any]) -> dict[str, Any]:
        return describe_evidence_page(super()._model_evidence_query(raw_query))


def _native_actor_tools(*, return_model: type[BaseModel]) -> list[Tool]:
    def evidence_query(**kwargs: Any) -> dict[str, Any]:
        """Retrieve immutable authoritative evidence as a self-describing page."""
        return kwargs

    def run_python(code: str, values: dict[str, JsonValue]) -> str:
        """Run fresh sterile Python over explicit values; rle.decode_row is preloaded."""
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
            name="run_python",
            description=run_python.__doc__ or "Run Python.",
            callable=run_python,
            parameters_model=RunPythonArguments,
        ),
        Tool(
            name="rle_decode_row",
            description=(
                "Decode one supplied RLE row using the shared pure decoder; this "
                "cannot retrieve evidence."
            ),
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


def _one_native_tool_call(tool_calls: list[ToolCall]) -> ToolCall:
    if len(tool_calls) != 1:
        raise RuntimeError("matched Native Actor requires exactly one tool call")
    call = tool_calls[0]
    if call.name not in {
        "evidence_query",
        "run_python",
        "rle_decode_row",
        "return_result",
    }:
        raise RuntimeError(f"unavailable matched Native tool: {call.name!r}")
    return call


def _commitment(value: object) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
