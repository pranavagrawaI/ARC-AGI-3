"""Adapter from Tycho's workspace tools to NOOA native tools."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from nooa.unifiedllm import Tool
from pydantic import BaseModel, ConfigDict, Field
from tycho.workspace.agent_tools import ToolExecutor, canonical, tool_specs
from tycho.workspace.workspace import diff_text, grid_text

from tycho_nooa.epistemic import GoalToolSurface
from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.modes import ModePolicy
from tycho_nooa.planner import PlannerIntegration, planner_command
from tycho_nooa.runtime_notice import tool_note
from tycho_nooa.theory_transaction import TheoryTransaction
from tycho_nooa.trusted_runtime import TRUSTED_PATHS


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ListArguments(_Arguments):
    path: str = "."


class _ReadArguments(_Arguments):
    path: str


class _WriteArguments(_Arguments):
    path: str
    content: str


class _EditArguments(_Arguments):
    path: str
    old: str
    new: str


class _EditFunctionArguments(_Arguments):
    path: str
    name: str
    code: str


class _RunPythonArguments(_Arguments):
    code: str
    timeout_s: int = Field(default=15, ge=1, le=300)


class _VerbosityArguments(_Arguments):
    grid: Literal["on", "off"] | None = None
    diff: Literal["on", "summary", "off"] | None = None


class _InvokeBuilderArguments(_Arguments):
    reason: str = ""


class _ActionArguments(_Arguments):
    action: str
    row: int | None = Field(default=None, ge=0, le=63)
    col: int | None = Field(default=None, ge=0, le=63)


class _CheckoutTheoryArguments(_Arguments):
    record_id: str


_ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    "ls": _ListArguments,
    "read_file": _ReadArguments,
    "write_file": _WriteArguments,
    "edit_file": _EditArguments,
    "edit_function": _EditFunctionArguments,
    "run_python": _RunPythonArguments,
    "set_verbosity": _VerbosityArguments,
    "invoke_builder": _InvokeBuilderArguments,
    "take_action": _ActionArguments,
}


@dataclass(frozen=True)
class ToolProfile:
    """Select one Tycho-owned tool surface without duplicating its schemas."""

    include_edit_function: bool = False
    include_builder: bool = False
    include_take_action: bool = True
    include_set_verbosity: bool = True


@dataclass(frozen=True)
class Observation:
    """One turn's evidence in the two sizes the actor's conversation needs.

    `prompt` inlines the grid and the transition; `elided` keeps every other
    part of the same message and replaces those two payloads with a pointer to
    the frame on disk. Both are built once, at the turn they describe, so
    eviction later is a swap rather than a re-parse of rendered text.
    """

    prompt: str
    elided: str


ACTOR_TOOL_PROFILE = ToolProfile()
BUILDER_TOOL_PROFILE = ToolProfile(
    include_edit_function=True,
    include_take_action=False,
    include_set_verbosity=False,
)


def actor_tool_profile(policy: ModePolicy) -> ToolProfile:
    """Expose function-level model editing only when the actor owns the model."""
    return ToolProfile(
        include_edit_function=policy.actor_writes_wm,
        include_builder=policy.include_builder,
    )


# What replaces the two heavy payloads once newer turns have arrived. The frame
# and its diff stay on disk, so this is a pointer, not a loss of evidence.
_ELIDED_PAYLOAD = [
    "",
    (
        "grid and transition elided to bound this conversation. Re-read the frame "
        "named above with read_file, or reconstruct any turn or diff with "
        "run_python."
    ),
]


class WorkspaceToolSurface:
    """One turn's native tool interface over a game-long Tycho workspace."""

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        profile: ToolProfile = ACTOR_TOOL_PROFILE,
        mode_policy: ModePolicy | None = None,
        goal_tools: GoalToolSurface | None = None,
        trace_path: Path | None = None,
        trace_source: str = "actor",
        planner_trace_path: Path | None = None,
        theory_transaction: TheoryTransaction | None = None,
    ):
        self._executor = executor
        self._profile = profile
        self._mode_policy = mode_policy
        self._goal_tools = goal_tools
        self._trace_path = trace_path
        self._trace_source = trace_source
        self._trace_phase = "decision"
        self._theory_transaction = theory_transaction
        if planner_trace_path is None and trace_path is not None:
            planner_trace_path = trace_path.with_name("planner-trace.jsonl")
        self._planner = PlannerIntegration(executor.ws, planner_trace_path)
        self._tool_call_index = 0
        self._turn_tool_calls: list[dict[str, Any]] = []
        self._turn: GameTurn | None = None
        self._previous_grid: list[list[int]] | None = None
        self._builder_report: str | None = None
        self.grid_verbosity = "on"
        self.diff_verbosity = "on"

    @contextmanager
    def trace_scope(self, *, source: str, phase: str) -> Iterator[None]:
        """Attribute tool calls made by one bounded strategy invocation."""
        previous_source = self._trace_source
        previous_phase = self._trace_phase
        self._trace_source = source
        self._trace_phase = phase
        try:
            yield
        finally:
            self._trace_source = previous_source
            self._trace_phase = previous_phase

    def begin_turn(self, turn: GameTurn) -> None:
        previous = self._turn
        self._previous_grid = (
            previous.grid
            if previous is not None
            and previous.level == turn.level
            and turn.turn_in_level > 0
            else None
        )
        self._turn = turn
        self._turn_tool_calls = []
        self._builder_report = None
        self._executor.cur_grid = turn.grid
        self._executor.available = turn.available_actions
        if self._goal_tools is not None:
            self._goal_tools.begin_turn(turn)

    def set_builder_report(self, report: str | None) -> None:
        """Expose a just-built advisory report in the actor's next task message."""
        self._builder_report = report

    def turn_tool_calls(self) -> list[dict[str, Any]]:
        """Return a copy of the current turn's observable tool-use evidence."""
        return [dict(call) for call in self._turn_tool_calls]

    def observation(self, turn: GameTurn) -> Observation:
        """Render evidence selected by the actor's persistent verbosity state."""
        frame = f"level_{turn.level}/turn_{turn.turn_in_level:03d}.txt"
        head = self._observation_head(turn, frame)
        advisory = self._observation_advisory(turn)
        return Observation(
            prompt="\n".join(head + self._observation_payload(turn) + advisory),
            elided="\n".join(head + _ELIDED_PAYLOAD + advisory),
        )

    def _observation_head(self, turn: GameTurn, frame: str) -> list[str]:
        metadata = turn.model_dump(exclude={"grid"})
        return [
            (
                "Investigate the following new public observation. It has already "
                "been recorded in the durable Tycho workspace."
            ),
            "",
            f"turn = {metadata!r}",
            "",
            f"frame on disk = {frame}",
            "",
            (
                "display = "
                f"grid={self.grid_verbosity!r}, diff={self.diff_verbosity!r}; "
                "the full grid and exact transition remain available through run_python"
            ),
        ]

    def _observation_payload(self, turn: GameTurn) -> list[str]:
        lines: list[str] = []
        if self.grid_verbosity == "on":
            lines.extend(
                [
                    "",
                    "grid (row labels at left, column labels in header):",
                    grid_text(turn.grid),
                ]
            )

        if self._previous_grid is not None:
            difference = diff_text(self._previous_grid, turn.grid)
            no_change = difference.startswith("no cells changed")
            if no_change or self.diff_verbosity != "off":
                if self.diff_verbosity == "summary" and not no_change:
                    difference = difference.splitlines()[0]
                lines.extend(["", "change from the previous turn:", difference])

        if self.grid_verbosity == "off" and self.diff_verbosity == "off":
            lines.extend(
                [
                    "",
                    (
                        "No grid payload is inlined. Inspect the current frame with "
                        "run_python when needed."
                    ),
                ]
            )
        return lines

    def _observation_advisory(self, turn: GameTurn) -> list[str]:
        lines: list[str] = []
        planner_hint = self._planner.hint(turn)
        if planner_hint is not None:
            lines.extend(
                [
                    "",
                    "validated planner feedback:",
                    planner_hint,
                ]
            )
        if self._builder_report is not None:
            lines.extend(
                [
                    "",
                    "automatic world-model builder report (advisory; inspect or test it):",
                    self._builder_report,
                ]
            )
        return lines

    def native_tools(self) -> list[Tool]:
        """Return Tycho's generic tool names/descriptions as NOOA tools."""
        tools: list[Tool] = []
        runtime_note = tool_note(self._executor.ws.dir)
        dispatch: dict[str, Callable[..., str | ActionDecision]] = {
            "ls": self.ls,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "edit_function": self.edit_function,
            "run_python": self.run_python,
            "set_verbosity": self.set_verbosity,
            "invoke_builder": self.invoke_builder,
            "take_action": self.take_action,
        }
        for spec in tool_specs(
            "generic",
            # The named typed-goal treatment replaces the untyped actor request;
            # leaving both visible lets the model bypass evidence validation.
            include_builder=(
                self._profile.include_builder and self._goal_tools is None
            ),
            include_edit_func=self._profile.include_edit_function,
            include_take_action=self._profile.include_take_action,
            include_set_verbosity=self._profile.include_set_verbosity,
        ):
            name = spec["name"]
            tools.append(
                Tool(
                    name=name,
                    description=spec["description"] + runtime_note.get(name, ""),
                    callable=dispatch[name],
                    parameters_model=_ARGUMENT_MODELS[name],
                )
            )
        if self._goal_tools is not None:
            tools.extend(self._goal_tools.native_tools())
        if self._theory_transaction is not None:
            tools.append(
                Tool(
                    name="checkout_theory_candidate",
                    description=(
                        "Checkout an immutable retained theory by record ID. The host "
                        "re-evaluates it against the current frozen frontier and working "
                        "incumbent; checkout never grants publication authority."
                    ),
                    callable=self.checkout_theory_candidate,
                    parameters_model=_CheckoutTheoryArguments,
                )
            )
        return tools

    def dispatch(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        *,
        tool_call_id: str | None = None,
        raw_arguments: str | None = None,
    ) -> str | ActionDecision:
        model = tool.parameters_model
        assert model is not None
        try:
            validated = model.model_validate(arguments)
        except Exception as error:
            self._record_tool_call(
                tool.name,
                arguments,
                status="error",
                result=f"{type(error).__name__}: {error}",
                tool_call_id=tool_call_id,
                raw_arguments=raw_arguments,
                validation_status="error",
                execution_status="not_attempted",
                error_stage="validation",
            )
            raise
        normalized_arguments = validated.model_dump(exclude_none=True)
        try:
            if self._goal_tools is not None and tool.name in self._goal_tools.TOOL_NAMES:
                result = self._goal_tools.dispatch(tool, arguments)
            else:
                result = tool.callable(**normalized_arguments)
        except Exception as error:
            self._record_tool_call(
                tool.name,
                arguments,
                status="error",
                result=f"{type(error).__name__}: {error}",
                tool_call_id=tool_call_id,
                raw_arguments=raw_arguments,
                validated_arguments=normalized_arguments,
                execution_status="error",
                error_stage="execution",
            )
            raise
        self._record_tool_call(
            tool.name,
            arguments,
            status="complete",
            result=result,
            tool_call_id=tool_call_id,
            raw_arguments=raw_arguments,
            validated_arguments=normalized_arguments,
        )
        return result

    def record_rejected_call(
        self,
        name: str,
        arguments: dict[str, Any],
        message: str,
        *,
        tool_call_id: str | None = None,
        raw_arguments: str | None = None,
        rejection_stage: Literal[
            "parse", "availability", "policy", "validation", "execution"
        ] = "policy",
        validated_arguments: dict[str, Any] | None = None,
    ) -> None:
        """Keep malformed or unavailable model tool attempts in the durable trace."""
        self._record_tool_call(
            name,
            arguments,
            status="error",
            result=message,
            tool_call_id=tool_call_id,
            raw_arguments=raw_arguments,
            parse_status="error" if rejection_stage == "parse" else "complete",
            validation_status=(
                "error"
                if rejection_stage == "validation"
                else "complete"
                if rejection_stage == "execution"
                else "not_attempted"
            ),
            execution_status=(
                "error" if rejection_stage == "execution" else "not_attempted"
            ),
            validated_arguments=validated_arguments,
            error_stage=rejection_stage,
        )

    def record_external_result(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
        *,
        tool_call_id: str | None = None,
        raw_arguments: str | None = None,
        validated_arguments: dict[str, Any] | None = None,
    ) -> None:
        """Record an async tool whose implementation lives above this surface."""
        self._record_tool_call(
            name,
            arguments,
            status="complete",
            result=result,
            tool_call_id=tool_call_id,
            raw_arguments=raw_arguments,
            validated_arguments=validated_arguments,
        )

    def current_turn(self) -> GameTurn:
        """Expose the active public turn to higher-level async tool adapters."""
        return self._require_turn()

    @property
    def goal_tools(self) -> GoalToolSurface | None:
        return self._goal_tools

    def _record_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        status: Literal["complete", "error"],
        result: str | ActionDecision,
        tool_call_id: str | None = None,
        raw_arguments: str | None = None,
        parse_status: Literal["complete", "error"] = "complete",
        validation_status: Literal["complete", "error", "not_attempted"] = "complete",
        execution_status: Literal["complete", "error", "not_attempted"] = "complete",
        validated_arguments: dict[str, Any] | None = None,
        error_stage: str | None = None,
    ) -> None:
        self._tool_call_index += 1
        turn = self._require_turn()
        record = {
            "schema_version": 2,
            "call_index": self._tool_call_index,
            "tool_call_id": tool_call_id,
            "source": self._trace_source,
            "phase": self._trace_phase,
            "sequence": turn.sequence,
            "level": turn.level,
            "turn_in_level": turn.turn_in_level,
            "name": name,
            "raw_arguments": raw_arguments,
            "arguments": arguments,
            "parse_status": parse_status,
            "validation_status": validation_status,
            "validated_arguments": validated_arguments,
            "execution_status": execution_status,
            "error_stage": error_stage,
            "status": status,
            "result": (
                result.model_dump() if isinstance(result, ActionDecision) else result
            ),
        }
        self._turn_tool_calls.append(record)
        if self._trace_path is None:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps(record, separators=(",", ":")) + "\n")

    def ls(self, path: str = ".") -> str:
        return self._executor.execute("ls", {"path": path})

    def read_file(self, path: str) -> str:
        return self._executor.execute("read_file", {"path": path})

    def write_file(self, path: str, content: str) -> str:
        self._authorize_transactional_builder_write(path)
        self._authorize_trusted_runtime_write(path)
        self._authorize_actor_write(path)
        return self._execute_theory_edit(
            "write_file",
            path,
            lambda: self._executor.execute(
                "write_file", {"path": path, "content": content}
            ),
        )

    def edit_file(self, path: str, old: str, new: str) -> str:
        self._authorize_transactional_builder_write(path)
        self._authorize_trusted_runtime_write(path)
        self._authorize_actor_write(path)
        return self._execute_theory_edit(
            "edit_file",
            path,
            lambda: self._executor.execute(
                "edit_file", {"path": path, "old": old, "new": new}
            ),
        )

    def edit_function(self, path: str, name: str, code: str) -> str:
        self._authorize_transactional_builder_write(path)
        self._authorize_trusted_runtime_write(path)
        self._authorize_actor_write(path)
        return self._execute_theory_edit(
            "edit_function",
            path,
            lambda: self._executor.execute(
                "edit_function", {"path": path, "name": name, "code": code}
            ),
        )

    def _authorize_trusted_runtime_write(self, path: str) -> None:
        requested = Path(path.replace("\\", "/"))
        workspace = self._executor.ws.dir.resolve()
        candidate = (
            requested.resolve()
            if requested.is_absolute()
            else (workspace / requested).resolve()
        )
        protected = {(workspace / name).resolve() for name in TRUSTED_PATHS}
        if candidate in protected:
            raise PermissionError(
                f"cannot modify trusted runtime path {path!r}; runtime files are read-only"
            )

    def _authorize_actor_write(self, path: str) -> None:
        if self._mode_policy is not None:
            self._mode_policy.authorize_actor_write(path)

    def _authorize_transactional_builder_write(self, path: str) -> None:
        if self._theory_transaction is None:
            return
        normalized = path.replace("\\", "/")
        parsed = PurePosixPath(normalized)
        allowed = {
            "world_model.py",
            "notes/world_model.md",
            "notes/world_model_report.md",
        }
        if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() not in allowed:
            raise PermissionError(
                f"transactional Builder cannot write {path!r}; owned paths are "
                f"{sorted(allowed)}"
            )

    def _execute_theory_edit(
        self,
        operation: str,
        path: str,
        mutate: Callable[[], str],
    ) -> str:
        if self._theory_transaction is None or PurePosixPath(
            path.replace("\\", "/")
        ).as_posix() != "world_model.py":
            return mutate()
        return self._theory_transaction.apply_edit(operation, mutate).render()

    def checkout_theory_candidate(self, record_id: str) -> str:
        if self._theory_transaction is None:
            raise RuntimeError("no active theory transaction")
        return self._theory_transaction.checkout(record_id).render()

    def run_python(self, code: str, timeout_s: int = 15) -> str:
        invocation = self._planner.begin_invocation(code)
        result = self._executor.execute(
            "run_python", {"code": code, "timeout_s": timeout_s}
        )
        self._planner.finish_invocation(invocation, result)
        return result

    @staticmethod
    def is_planner_call(name: str, arguments: dict[str, Any]) -> bool:
        """Whether a native tool call invokes Tycho's seeded planner wrapper."""
        return (
            canonical(name) == "run_python"
            and planner_command(str(arguments.get("code", ""))) is not None
        )

    def set_verbosity(
        self,
        grid: Literal["on", "off"] | None = None,
        diff: Literal["on", "summary", "off"] | None = None,
    ) -> str:
        changed: list[str] = []
        if grid is not None:
            self.grid_verbosity = grid
            changed.append(f"grid={grid}")
        if diff is not None:
            self.diff_verbosity = diff
            changed.append(f"diff={diff}")
        if not changed:
            return (
                "(set_verbosity: pass grid='on'|'off' and/or diff='on'|'summary'|'off')"
            )
        return (
            f"verbosity updated ({', '.join(changed)}). Now: full grid inlined="
            f"{self.grid_verbosity == 'on'}, diff mode={self.diff_verbosity}. "
            "The full grid and exact diff remain available through run_python."
        )

    @staticmethod
    def invoke_builder(reason: str = "") -> str:
        """Marker callable; NativeToolStrategy performs this async handoff."""
        raise RuntimeError(
            "invoke_builder must be dispatched by NativeToolStrategy"
        )

    def take_action(
        self,
        action: str,
        row: int | None = None,
        col: int | None = None,
    ) -> ActionDecision:
        turn = self._require_turn()
        if action not in turn.available_actions:
            raise ValueError(
                f"illegal action {action!r}; available actions: {turn.available_actions}"
            )
        return ActionDecision(action=action, row=row, col=col)

    def _require_turn(self) -> GameTurn:
        if self._turn is None:
            raise RuntimeError("no game turn is active")
        return self._turn


def tool_by_name(tools: list[Tool]) -> dict[str, Tool]:
    return {canonical(tool.name): tool for tool in tools}
