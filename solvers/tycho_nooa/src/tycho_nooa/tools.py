"""Adapter from Tycho's workspace tools to NOOA native tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nooa.unifiedllm import Tool
from pydantic import BaseModel, ConfigDict, Field
from tycho.workspace.agent_tools import ToolExecutor, canonical, tool_specs
from tycho.workspace.workspace import diff_text, grid_text

from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.planner import PlannerIntegration, planner_command


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


class _ActionArguments(_Arguments):
    action: str
    row: int | None = Field(default=None, ge=0, le=63)
    col: int | None = Field(default=None, ge=0, le=63)


_ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    "ls": _ListArguments,
    "read_file": _ReadArguments,
    "write_file": _WriteArguments,
    "edit_file": _EditArguments,
    "edit_function": _EditFunctionArguments,
    "run_python": _RunPythonArguments,
    "set_verbosity": _VerbosityArguments,
    "take_action": _ActionArguments,
}


@dataclass(frozen=True)
class ToolProfile:
    """Select one Tycho-owned tool surface without duplicating its schemas."""

    include_edit_function: bool = False
    include_take_action: bool = True
    include_set_verbosity: bool = True


ACTOR_TOOL_PROFILE = ToolProfile()
BUILDER_TOOL_PROFILE = ToolProfile(
    include_edit_function=True,
    include_take_action=False,
    include_set_verbosity=False,
)


class WorkspaceToolSurface:
    """One turn's native tool interface over a game-long Tycho workspace."""

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        profile: ToolProfile = ACTOR_TOOL_PROFILE,
        trace_path: Path | None = None,
        planner_trace_path: Path | None = None,
    ):
        self._executor = executor
        self._profile = profile
        self._trace_path = trace_path
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

    def set_builder_report(self, report: str | None) -> None:
        """Expose a just-built advisory report in the actor's next task message."""
        self._builder_report = report

    def turn_tool_calls(self) -> list[dict[str, Any]]:
        """Return a copy of the current turn's observable tool-use evidence."""
        return [dict(call) for call in self._turn_tool_calls]

    def observation_prompt(self, turn: GameTurn) -> str:
        """Render evidence selected by the actor's persistent verbosity state."""
        metadata = turn.model_dump(exclude={"grid"})
        lines = [
            (
                "Investigate the following new public observation. It has already "
                "been recorded in the durable Tycho workspace."
            ),
            "",
            f"turn = {metadata!r}",
            "",
            (
                "display = "
                f"grid={self.grid_verbosity!r}, diff={self.diff_verbosity!r}; "
                "the full grid and exact transition remain available through run_python"
            ),
        ]
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
        return "\n".join(lines)

    def native_tools(self) -> list[Tool]:
        """Return Tycho's generic tool names/descriptions as NOOA tools."""
        tools: list[Tool] = []
        dispatch: dict[str, Callable[..., str | ActionDecision]] = {
            "ls": self.ls,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "edit_function": self.edit_function,
            "run_python": self.run_python,
            "set_verbosity": self.set_verbosity,
            "take_action": self.take_action,
        }
        for spec in tool_specs(
            "generic",
            include_edit_func=self._profile.include_edit_function,
            include_take_action=self._profile.include_take_action,
            include_set_verbosity=self._profile.include_set_verbosity,
        ):
            name = spec["name"]
            tools.append(
                Tool(
                    name=name,
                    description=spec["description"],
                    callable=dispatch[name],
                    parameters_model=_ARGUMENT_MODELS[name],
                )
            )
        return tools

    def dispatch(self, tool: Tool, arguments: dict[str, Any]) -> str | ActionDecision:
        model = tool.parameters_model
        assert model is not None
        try:
            validated = model.model_validate(arguments)
            result = tool.callable(**validated.model_dump(exclude_none=True))
        except Exception as error:
            self._record_tool_call(
                tool.name,
                arguments,
                status="error",
                result=f"{type(error).__name__}: {error}",
            )
            raise
        self._record_tool_call(tool.name, arguments, status="complete", result=result)
        return result

    def record_rejected_call(
        self,
        name: str,
        arguments: dict[str, Any],
        message: str,
    ) -> None:
        """Keep malformed or unavailable model tool attempts in the durable trace."""
        self._record_tool_call(name, arguments, status="error", result=message)

    def _record_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        status: Literal["complete", "error"],
        result: str | ActionDecision,
    ) -> None:
        self._tool_call_index += 1
        turn = self._require_turn()
        record = {
            "call_index": self._tool_call_index,
            "level": turn.level,
            "turn_in_level": turn.turn_in_level,
            "name": name,
            "arguments": arguments,
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
        return self._executor.execute("write_file", {"path": path, "content": content})

    def edit_file(self, path: str, old: str, new: str) -> str:
        return self._executor.execute(
            "edit_file", {"path": path, "old": old, "new": new}
        )

    def edit_function(self, path: str, name: str, code: str) -> str:
        return self._executor.execute(
            "edit_function", {"path": path, "name": name, "code": code}
        )

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
