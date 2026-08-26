"""Matched Native and sterile-CodeAct adapters over the Builder proposal protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from nooa.unifiedllm import Tool, UnifiedLLM
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from tycho_nooa.builder_proposal_protocol import (
    BuilderEvaluationCapability,
    BuilderReport,
    FrozenBuildCase,
    ProposalLedger,
)
from tycho_nooa.builder_sterile_execution import (
    BuilderExecutionManifest,
    BuilderSterileExecutor,
)
from tycho_nooa.execution_events import InvocationProvenance
from tycho_nooa.sterile_codeact import SterileCodeActController

COMMON_BUILDER_TASK = (
    "Use the frozen evidence, incumbent theory, and obligation table to propose "
    "an evidence-supported executable-theory improvement. Evaluation is "
    "observational. Preserve every protected obligation and do not claim "
    "unobserved generality. Return an advisory report; the host alone decides "
    "publication."
)

NATIVE_MECHANICS = (
    "Use provider-native evaluate_candidate, run_python, and return_result tools. "
    "Each run_python call starts with a fresh local namespace."
)

CODEACT_MECHANICS = (
    "Use execute_python and return_result. Python locals persist across cells. "
    "Inside Python, builder.evaluate is the only capability that can leave the "
    "realm."
)


class BuilderReportReturn(BaseModel):
    """The report value shared by both execution modes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_sha256: str
    nominated_candidate_source_sha256: str | None = None
    judgment: str = Field(min_length=1)
    evidence_references: tuple[str, ...] = ()
    open_uncertainties: tuple[str, ...] = ()

    def to_protocol(self, case: FrozenBuildCase) -> BuilderReport:
        if self.case_sha256 != case.case_sha256:
            raise ValueError("Builder report return belongs to another frozen case")
        return BuilderReport.create(
            case=case,
            nominated_candidate_source_sha256=(
                self.nominated_candidate_source_sha256
            ),
            judgment=self.judgment,
            evidence_references=self.evidence_references,
            open_uncertainties=self.open_uncertainties,
        )


class EvaluateCandidateArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_source: str
    parent_source: str
    declared_parent_source_sha256: str
    provenance: dict[str, JsonValue] = Field(default_factory=dict)


class NativeRunPythonArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    values: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass(frozen=True)
class _BuilderPacket:
    case: FrozenBuildCase

    def treatment_payload(self) -> dict[str, JsonValue]:
        return {"case": self.case.prompt_preview()}


class BuilderCodeActController(SterileCodeActController):
    """Run a Builder in a clean process with one typed evaluation egress."""

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        case: FrozenBuildCase,
        ledger: ProposalLedger,
        provenance: InvocationProvenance,
        max_llm_calls: int = 8,
        max_output_tokens: int | None = None,
        cell_timeout_seconds: float = 10.0,
    ) -> None:
        self._builder_case = case
        self._proposal_ledger = ledger
        super().__init__(
            llm=llm,
            packet=_BuilderPacket(case),  # type: ignore[arg-type]
            evidence=None,
            provenance=provenance,
            max_llm_calls=max_llm_calls,
            max_output_tokens=max_output_tokens,
            cell_timeout_seconds=cell_timeout_seconds,
            system_prompt=f"{COMMON_BUILDER_TASK}\n\n{CODEACT_MECHANICS}",
            return_model=BuilderReportReturn,
        )

    async def run(self) -> BuilderReport:
        async with BuilderSterileExecutor(
            self._execution_manifest(),
            return_type=BuilderReportReturn,
            capability_handlers=self._capability_handlers(),
            capability_auditors=self._capability_auditors(),
            provenance=self._provenance,
        ) as executor:
            for _ in range(self._max_llm_calls):
                self._proposal_ledger.begin_provider_turn()
                returned = await self._run_provider_turn(executor)
                if returned is not None:
                    if not isinstance(returned, BuilderReportReturn):
                        raise TypeError(
                            "sterile CodeAct returned the wrong Builder report type"
                        )
                    return returned.to_protocol(self._builder_case)
        raise RuntimeError("sterile CodeAct exhausted its LLM-call budget")

    def _execution_manifest(self) -> BuilderExecutionManifest:
        return BuilderExecutionManifest(
            invocation_id=self._provenance.invocation_id,
            capabilities=("builder.evaluate",),
            input_values={"case": self._builder_case.builder_view()},
            wall_time_seconds=self._cell_timeout_seconds,
        )

    def _capability_handlers(self) -> dict[str, Any]:
        return {
            "builder.evaluate": BuilderEvaluationCapability(self._proposal_ledger)
        }


class BuilderNativeController:
    """Provider-native control with fresh, sterile Python on every computation."""

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        case: FrozenBuildCase,
        ledger: ProposalLedger,
        max_llm_calls: int = 8,
        max_output_tokens: int | None = None,
        cell_timeout_seconds: float = 10.0,
    ) -> None:
        self._llm = llm
        self._case = case
        self._ledger = ledger
        self._max_llm_calls = max_llm_calls
        self._max_output_tokens = max_output_tokens
        self._cell_timeout_seconds = cell_timeout_seconds
        self._messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": f"{COMMON_BUILDER_TASK}\n\n{NATIVE_MECHANICS}",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"case": case.prompt_preview()},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ]
        self.provider_usage: list[dict[str, int] | None] = []
        self.python_calls = 0
        self.invalid_interactions = 0
        self.receipt_provider_calls: list[int] = []

    async def run(self) -> BuilderReport:
        tools = _native_tools()
        for _ in range(self._max_llm_calls):
            self._ledger.begin_provider_turn()
            response = await self._llm.acall(
                list(self._messages),
                tools=tools,
                output_model=None,
                **(
                    {}
                    if self._max_output_tokens is None
                    else {"max_tokens": self._max_output_tokens}
                ),
            )
            self.provider_usage.append(response.usage)
            if len(response.tool_calls) != 1:
                self.invalid_interactions += 1
                self._messages.append(
                    {
                        "role": "user",
                        "content": "Emit exactly one offered tool call.",
                    }
                )
                continue
            call = response.tool_calls[0]
            self._messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                    ],
                }
            )
            try:
                result = await self._dispatch(call.name, json.loads(call.arguments))
            except Exception as error:  # noqa: BLE001 - return typed tool failure
                self.invalid_interactions += 1
                self._record_tool(call.id, {"error": f"{type(error).__name__}: {error}"})
                continue
            if isinstance(result, BuilderReportReturn):
                return result.to_protocol(self._case)
            self._record_tool(call.id, result)
        raise RuntimeError("Native Builder exhausted its LLM-call budget")

    async def _dispatch(self, name: str, arguments: object) -> object:
        if name == "evaluate_candidate":
            parsed = EvaluateCandidateArguments.model_validate(arguments)
            result = BuilderEvaluationCapability(self._ledger)(
                parsed.model_dump(mode="json")
            )
            self.receipt_provider_calls.append(len(self.provider_usage))
            return result
        if name == "run_python":
            parsed = NativeRunPythonArguments.model_validate(arguments)
            self.python_calls += 1
            manifest = BuilderExecutionManifest(
                invocation_id=f"native-builder-python-{self.python_calls}",
                input_values={
                    "case": self._case.builder_view(),
                    **parsed.values,
                },
                wall_time_seconds=self._cell_timeout_seconds,
            )
            async with BuilderSterileExecutor(manifest, return_type=dict) as executor:
                cell = await executor.run_cell(parsed.code)
            return {
                "status": cell.status,
                "stdout": cell.stdout,
                "stderr": cell.stderr,
                "locals_preview": cell.locals_preview,
                "return_value": cell.return_value,
                "exception": (
                    None
                    if cell.exception is None
                    else {
                        "type": cell.exception.type,
                        "message": cell.exception.message,
                    }
                ),
            }
        if name == "return_result":
            return BuilderReportReturn.model_validate(arguments)
        raise ValueError(f"unavailable Native Builder tool: {name!r}")

    def _record_tool(self, call_id: str, result: object) -> None:
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    result,
                    default=_json_default,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )


def _native_tools() -> list[Tool]:
    def evaluate_candidate(**arguments: Any) -> dict[str, Any]:
        """Evaluate a parent-bound candidate without retaining or publishing it."""
        return arguments

    def run_python(code: str, values: dict[str, JsonValue]) -> dict[str, Any]:
        """Run one fresh sterile Python computation over the frozen case and values."""
        return {"code": code, "values": values}

    def return_result(result: BuilderReportReturn) -> BuilderReportReturn:
        """Return the final advisory Builder report."""
        return result

    return [
        Tool(
            name="evaluate_candidate",
            description=evaluate_candidate.__doc__ or "Evaluate candidate.",
            callable=evaluate_candidate,
            parameters_model=EvaluateCandidateArguments,
        ),
        Tool(
            name="run_python",
            description=run_python.__doc__ or "Run Python.",
            callable=run_python,
            parameters_model=NativeRunPythonArguments,
        ),
        Tool(
            name="return_result",
            description=return_result.__doc__ or "Return report.",
            callable=return_result,
            parameters_model=BuilderReportReturn,
        ),
    ]


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
