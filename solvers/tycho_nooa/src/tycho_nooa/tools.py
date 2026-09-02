"""Adapter from Tycho's workspace tools to NOOA native tools."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from nooa.unifiedllm import Tool
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from tycho.workspace.agent_tools import ToolExecutor, canonical, tool_specs
from tycho.workspace.workspace import diff_text, grid_text

from tycho_nooa.action_batch import (
    MAX_ACTIONS_PER_BATCH,
    ActionBatchDecision,
    grid_sha256,
)
from tycho_nooa.epistemic import GoalToolSurface
from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.modes import ModePolicy
from tycho_nooa.planner import (
    PlannerIntegration,
    is_host_owned_planner_path,
    planner_command,
)
from tycho_nooa.runtime_notice import tool_note
from tycho_nooa.theory_transaction import TheoryTransaction
from tycho_nooa.trusted_runtime import TRUSTED_PATHS
from tycho_nooa.wm_probe import TrustedRuntimePythonSandbox


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


def _action_horizon_prompt(turn: GameTurn) -> str:
    if turn.action_limit is None:
        return ""
    prompt = (
        f"Action horizon: {turn.actions_used} of {turn.action_limit} permitted "
        f"environment actions have been used; {turn.actions_remaining} remain "
        "including this decision."
    )
    if turn.final_permitted_action:
        prompt += (
            " This is the final permitted environment action. Its result will be "
            "recorded, but no next model decision is guaranteed."
        )
    return prompt


def _action_schema_extra(schema: dict[str, Any]) -> None:
    """Expose the same action-dependent shape that request validation enforces."""
    properties = schema["properties"]
    actions = properties["action"].get("enum")
    if not isinstance(actions, list):
        return
    coordinate = {"maximum": 63, "minimum": 0, "type": "integer"}
    shared = {
        name: value
        for name, value in properties.items()
        if name not in {"action", "row", "col"}
    }
    non_click_actions = [action for action in actions if action != "CLICK"]
    if "CLICK" in actions and non_click_actions:
        schema["oneOf"] = [
            {
                "additionalProperties": False,
                "properties": {
                    "action": {"const": "CLICK"},
                    "row": coordinate,
                    "col": coordinate,
                    **shared,
                },
                "required": ["action", "row", "col"],
                "type": "object",
            },
            {
                "additionalProperties": False,
                "properties": {
                    "action": {"enum": non_click_actions},
                    **shared,
                },
                "required": ["action"],
                "type": "object",
            },
        ]
    elif actions == ["CLICK"]:
        properties["row"] = coordinate
        properties["col"] = coordinate
        schema["required"] = ["action", "row", "col"]
    else:
        properties.pop("row", None)
        properties.pop("col", None)


class _ActionArguments(_Arguments):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_action_schema_extra)

    action: str
    row: int | None = Field(default=None, ge=0, le=63)
    col: int | None = Field(default=None, ge=0, le=63)

    @model_validator(mode="after")
    def validate_action_shape(self) -> _ActionArguments:
        supplied_coordinates = self.model_fields_set.intersection({"row", "col"})
        if self.action == "CLICK":
            if self.row is None or self.col is None:
                raise ValueError("CLICK requires integer row and col")
        elif supplied_coordinates:
            raise ValueError(f"{self.action} does not accept row or col")
        return self


class _BatchActionArguments(_ActionArguments):
    expected_before_grid_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class _BatchArguments(_Arguments):
    actions: list[_BatchActionArguments] = Field(
        min_length=1,
        max_length=MAX_ACTIONS_PER_BATCH,
    )
    rationale: str = Field(min_length=1, max_length=2_000)
    stop_on_no_change: bool = True


class _ActionTool(Tool):
    """Retain action-shape composition that NOOA's generic cleaner discards."""

    def get_parameter_schema(self, *, strict: bool = False) -> dict[str, Any]:
        model = self.parameters_model
        assert model is not None
        raw_schema = model.model_json_schema()
        branches = raw_schema.get("oneOf")
        schema = super().get_parameter_schema(
            strict=False if isinstance(branches, list) else strict
        )
        if raw_schema.get("additionalProperties") is False:
            schema["additionalProperties"] = False
        if isinstance(branches, list):
            schema["oneOf"] = branches
        return schema


class ModelToolError(RuntimeError):
    """A stable model-visible rejection backed by a detailed raw trace."""

    def __init__(self, *, stage: str, code: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message


def _current_batch_arguments(
    actions: list[str], *, max_actions: int
) -> type[BaseModel]:
    """Bind every step and the batch length to this public turn."""
    values = tuple(dict.fromkeys(actions))
    if not values:
        raise RuntimeError("current turn advertises no available actions")
    action_type = cast(Any, Literal)[values]
    step_model = create_model(
        "_CurrentBatchActionArguments",
        __base__=_BatchActionArguments,
        action=(action_type, ...),
    )
    return create_model(
        "_CurrentBatchArguments",
        __base__=_Arguments,
        actions=(
            list[step_model],  # type: ignore[valid-type]
            Field(min_length=1, max_length=max_actions),
        ),
        rationale=(str, Field(min_length=1, max_length=2_000)),
        stop_on_no_change=(bool, True),
    )


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
    "take_actions": _BatchArguments,
}


@dataclass(frozen=True)
class ToolProfile:
    """Select one Tycho-owned tool surface without duplicating its schemas."""

    include_edit_function: bool = False
    include_builder: bool = False
    include_take_actions: bool = True
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
    include_take_actions=False,
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
        lines = [
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
        if horizon := _action_horizon_prompt(turn):
            lines.extend(("", horizon))
        return lines

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
        if self._goal_tools is not None:
            lines.extend(self._goal_tools.observation_advisory())
        return lines

    def native_tools(self) -> list[Tool]:
        """Return Tycho's generic tool names/descriptions as NOOA tools."""
        runtime_note = tool_note(self._executor.ws.dir)
        dispatch: dict[
            str, Callable[..., str | ActionDecision | ActionBatchDecision]
        ] = {
            "ls": self.ls,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "edit_function": self.edit_function,
            "run_python": self.run_python,
            "set_verbosity": self.set_verbosity,
            "invoke_builder": self.invoke_builder,
            "take_actions": self.take_actions,
        }
        tools = [
            self._native_tool(spec, dispatch, runtime_note)
            for spec in self._generic_tool_specs()
        ]
        if self._profile.include_take_actions:
            tools.append(self._take_actions_tool(dispatch["take_actions"]))
        if self._goal_tools is not None:
            tools.extend(self._goal_tools.native_tools())
        if self._theory_transaction is not None:
            tools.append(self._checkout_theory_tool())
        return tools

    def _generic_tool_specs(self) -> list[dict[str, Any]]:
        # The named typed-goal treatment replaces the untyped actor request;
        # leaving both visible lets the model bypass evidence validation.
        include_builder = self._profile.include_builder and self._goal_tools is None
        specs = tool_specs(
            "generic",
            include_builder=include_builder,
            include_edit_func=self._profile.include_edit_function,
            # Pinned Tycho couples invoke_builder to its singular action spec.
            # Request it only when needed to obtain invoke_builder, then remove
            # the action spec before it reaches the model.
            include_take_action=include_builder,
            include_set_verbosity=self._profile.include_set_verbosity,
        )
        return [spec for spec in specs if spec["name"] != "take_action"]

    def _native_tool(
        self,
        spec: dict[str, Any],
        dispatch: dict[str, Callable[..., str | ActionDecision | ActionBatchDecision]],
        runtime_note: dict[str, str],
    ) -> Tool:
        name = spec["name"]
        parameters_model = _ARGUMENT_MODELS[name]
        description = spec["description"] + runtime_note.get(name, "")
        return Tool(
            name=name,
            description=description,
            callable=dispatch[name],
            parameters_model=parameters_model,
        )

    def _take_actions_tool(
        self,
        callable: Callable[..., str | ActionDecision | ActionBatchDecision],
    ) -> Tool:
        turn = self._turn
        available_actions = (
            turn.available_actions if turn is not None else self._executor.available
        )
        horizon_cap = (
            turn.actions_remaining
            if turn is not None and turn.actions_remaining is not None
            else MAX_ACTIONS_PER_BATCH
        )
        batch_cap = min(MAX_ACTIONS_PER_BATCH, horizon_cap)
        parameters_model = (
            _current_batch_arguments(available_actions, max_actions=batch_cap)
            if available_actions
            else _BatchArguments
        )
        return _ActionTool(
            name="take_actions",
            description=(
                f"The only action-commitment tool. Commit 1-{batch_cap} ordered "
                "environment actions from the current "
                f"available_actions {available_actions} in one model call. The Solver "
                "returns only the first action now, observes its public result, then "
                "releases each successor without another model call while the queue guards "
                "hold. By default a no-change result stops the queue. A later step may add "
                "expected_before_grid_sha256, computed as SHA-256 of the compact JSON grid "
                "(json.dumps(grid,separators=(',',':'))), to require an exact predicted "
                "frame. Boundaries, terminal states, resets, action illegality, horizon "
                "exhaustion, or failed guards return control to the Actor. Pass row and col "
                "only for CLICK. State a short predicted effect in rationale."
            ),
            callable=callable,
            parameters_model=parameters_model,
        )

    def _checkout_theory_tool(self) -> Tool:
        return Tool(
            name="checkout_theory_candidate",
            description=(
                "Checkout an immutable retained theory by record ID. The host re-evaluates "
                "it against the current frozen frontier and working incumbent; checkout "
                "never grants publication authority."
            ),
            callable=self.checkout_theory_candidate,
            parameters_model=_CheckoutTheoryArguments,
        )

    def dispatch(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        *,
        tool_call_id: str | None = None,
        raw_arguments: str | None = None,
    ) -> str | ActionDecision | ActionBatchDecision:
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
            raise ModelToolError(
                stage="validation",
                code="invalid_arguments",
                message="Tool arguments failed validation.",
            ) from error
        normalized_arguments = validated.model_dump(exclude_none=True)
        try:
            if self._goal_tools is not None and tool.name in self._goal_tools.TOOL_NAMES:
                result = self._goal_tools.dispatch(tool, arguments)
            else:
                result = tool.callable(**normalized_arguments)
        except ModelToolError as error:
            self._record_tool_call(
                tool.name,
                arguments,
                status="error",
                result=error.message,
                tool_call_id=tool_call_id,
                raw_arguments=raw_arguments,
                validated_arguments=normalized_arguments,
                execution_status=(
                    "not_attempted" if error.stage == "policy" else "error"
                ),
                error_stage=error.stage,
            )
            raise
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
        result: str | ActionDecision | ActionBatchDecision,
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
                result.model_dump()
                if isinstance(result, (ActionDecision, ActionBatchDecision))
                else result
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
        self._authorize_planner_write(path)
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
        self._authorize_planner_write(path)
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
        self._authorize_planner_write(path)
        self._authorize_trusted_runtime_write(path)
        self._authorize_actor_write(path)
        return self._execute_theory_edit(
            "edit_function",
            path,
            lambda: self._executor.execute(
                "edit_function", {"path": path, "name": name, "code": code}
            ),
        )

    def _authorize_planner_write(self, path: str) -> None:
        if is_host_owned_planner_path(self._executor.ws.dir, path):
            raise PermissionError(
                f"cannot modify host-owned planner path {path!r}; "
                "the seeded planner and validated artifacts are host-owned"
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
        if invocation.command is None:
            result = self._executor.execute(
                "run_python", {"code": code, "timeout_s": timeout_s}
            )
        else:
            result = self._run_disposable_planner(invocation.command, timeout_s)
        self._planner.finish_invocation(invocation, result)
        return result

    def _run_disposable_planner(self, command: str, timeout_s: int) -> str:
        """Run seeded planner code in writable scratch, then validate its output."""
        canonical_workspace = self._executor.ws
        canonical = canonical_workspace.dir.resolve()
        seeded_plan = self._planner.seeded_plan_bytes()
        with tempfile.TemporaryDirectory(prefix="tycho-plan-") as temporary_root:
            candidate = Path(temporary_root) / canonical.name
            shutil.copytree(canonical, candidate, symlinks=True)
            candidate_plan = candidate / "plan.py"
            candidate_plan.unlink(missing_ok=True)
            candidate_plan.write_bytes(seeded_plan)

            candidate_workspace = copy.copy(canonical_workspace)
            candidate_workspace.dir = candidate
            candidate_executor = copy.copy(self._executor)
            candidate_executor.ws = candidate_workspace
            candidate_executor._python_sandbox = TrustedRuntimePythonSandbox(
                workspace_read_only=False
            )
            result = candidate_executor.execute(
                "run_python",
                {"code": command, "timeout_s": timeout_s},
            )
            publication = self._planner.publish_candidate(candidate, result)
        return f"{result.rstrip()}\n{publication}"

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

    def take_actions(
        self,
        actions: list[dict[str, Any]],
        rationale: str,
        stop_on_no_change: bool = True,
    ) -> ActionBatchDecision:
        """Validate one bounded batch against the current public turn."""
        turn = self._require_turn()
        batch = ActionBatchDecision.model_validate(
            {
                "actions": actions,
                "rationale": rationale,
                "stop_on_no_change": stop_on_no_change,
            }
        )
        if turn.actions_remaining is not None and len(batch.actions) > turn.actions_remaining:
            raise ValueError(
                f"batch length {len(batch.actions)} exceeds the "
                f"{turn.actions_remaining} remaining environment actions"
            )
        illegal = [
            step.action
            for step in batch.actions
            if step.action not in turn.available_actions
        ]
        if illegal:
            raise ValueError(
                f"illegal batch actions {illegal!r}; available actions: "
                f"{turn.available_actions}"
            )
        first_guard = batch.actions[0].expected_before_grid_sha256
        if first_guard is not None and first_guard != grid_sha256(turn.grid):
            raise ValueError("first batch action's expected frame does not match this turn")
        return batch

    def _require_turn(self) -> GameTurn:
        if self._turn is None:
            raise RuntimeError("no game turn is active")
        return self._turn


def tool_by_name(tools: list[Tool]) -> dict[str, Tool]:
    return {canonical(tool.name): tool for tool in tools}
