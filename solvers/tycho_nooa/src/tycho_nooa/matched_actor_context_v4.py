"""Forward Actor controllers using the congruent evidence receipt worker."""

from __future__ import annotations

from typing import Any

from pydantic import JsonValue, ValidationError

from tycho_nooa.context_execution_v3 import (
    ContextComputationalSterileExecutor,
    ContextExecutionManifest,
)
from tycho_nooa.matched_actor import NativeComputationRecord
from tycho_nooa.matched_actor_context_v2 import (
    ContextCodeActController as _V2CodeActController,
)
from tycho_nooa.matched_actor_context_v2 import (
    ContextNativeActorController as _V2NativeController,
)
from tycho_nooa.matched_actor_context_v2 import ContextRunPythonArguments
from tycho_nooa.matched_actor_context_v3 import (
    ContextNativeAcquisitionController as _V3AcquisitionController,
)


class _CongruentNativeExecution:
    async def _handle_run_python(self, call: Any) -> None:
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
        self._python_cells += 1
        self._computation_records.append(
            NativeComputationRecord(
                call_index=len(self.provider_turns),
                code=arguments.code,
                status=result.status,
                stdout=result.stdout,
                error_code=(
                    None if result.exception is None else result.exception.type
                ),
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


class ContextNativeActorController(_CongruentNativeExecution, _V2NativeController):
    """Native Actor whose bounded Python receives the congruent receipt."""


class ContextNativeAcquisitionController(
    _CongruentNativeExecution,
    _V3AcquisitionController,
):
    """Bulk-acquisition Native Actor using the congruent receipt worker."""


class ContextCodeActController(_V2CodeActController):
    """CodeAct Actor using the congruent receipt worker."""

    async def run(self):
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
