"""Retrodict's ``python`` tool (upstream ``arc3/tools.py``) executed through our sandbox.

Upstream runs ``[analysis_python, "-c", code]`` in a separate interpreter with the
workspace as cwd; here the same command runs through ``qwen_baselines.executor.Executor``
(the jailed shell child: workspace cwd, numpy on the path, no game engine, no network),
with the same 30 s cap, 40,000-character per-stream limit and timed-out preamble.
"""

from __future__ import annotations

import shlex
import time

from pydantic import Field, model_validator

from .thin.bash_helpers import _format_output, _truncate_text
from .thin.defaults import DEFAULT_EDIT_DESCRIPTION, DEFAULT_EDIT_INSTRUCTIONS
from .thin.filesystem import EditArgs
from .thin.tools_base import StrictArgs, ToolResult, ToolSpec, coerce_args

PYTHON_DESCRIPTION = (
    "Run one self-contained python3 script (pass the source as `code`; it is executed with `python3 -c`). "
    "The interpreter has numpy, scipy, and networkx. It runs with the workspace as cwd, so open('log.txt') works. "
    "Print what you want to see; only stdout/stderr come back."
)


class PythonArgs(StrictArgs):
    """Arguments for python."""

    code: str
    timeout: int = Field(default=30, ge=1, le=30)


class PythonTool:
    """Retrodict's analysis tool over the jailed executor."""

    def __init__(self, executor, *, max_tool_chars: int = 40_000) -> None:
        self.executor = executor
        self.max_tool_chars = max_tool_chars

    def spec(self) -> ToolSpec:
        """Return the python tool spec."""
        return ToolSpec("python", PYTHON_DESCRIPTION, PythonArgs, self.run, sequential=True)

    async def run(self, args):
        """Run one python script in the sandbox; stdout and stderr come back merged."""
        args = coerce_args(args, PythonArgs)
        start = time.perf_counter()
        # The shell child merges stderr into stdout (shell_child.py); upstream returned
        # the two streams separately with the same per-stream cap.
        archive = self.executor.bridge.archive
        try:
            result = await self.executor.execute(
                "bash", command="python3 -c " + shlex.quote(args.code), timeout=args.timeout
            )
            exit_code, output = result["exitCode"], result["output"]
        except TimeoutError:
            if self.executor.bridge.clock.remaining() <= 0:
                raise  # the gameplay deadline, not the script's budget
            # The executor kills the process group on its timeout (upstream: TimeoutExpired)
            # and raises; the output streamed so far is in the call transcript.
            exit_code, output = None, archive.transcript(archive.count("calls") - 1)["output"]
        duration = time.perf_counter() - start
        timed_out = exit_code is None
        stdout, stdout_truncated = _truncate_text(output, self.max_tool_chars)
        ok = not timed_out and exit_code == 0
        metadata = {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_seconds": round(duration, 3),
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": False,
        }
        if timed_out:
            metadata["error_type"] = "Timeout"
        elif exit_code != 0:
            metadata["error_type"] = "NonZeroExit"
        content = _format_output(stdout, "")
        if timed_out:
            content = (
                f"TIMED OUT after {metadata['duration_seconds']}s (budget {args.timeout}s): the computation was too large to "
                "finish. Shrink the search space or switch to a cheaper method before retrying — a similar-sized retry will "
                "time out the same way. A timeout does not mean no solution exists.\n\n" + content
            )
        return ToolResult(ok, content, metadata)


# --- Retrodict v5.2: the edit tool accepts a top-level path -----------------------------
#
# thinharness's EditArgs forbids a top-level ``path`` (``extra_forbidden``); Qwen kept
# sending ``{"path": ..., "edits": [{"old_string", "new_string"}]}`` and lost 2-6 requests
# per game to it in v3-v5.1. The lenient shape fills each edit's path from the top level.


class LenientEditOperation(StrictArgs):
    """One exact replacement; ``path`` may come from the top level."""

    path: str | None = None
    old_string: str
    new_string: str
    all: bool = False
    expected_replacements: int | None = Field(default=None, ge=1)


class LenientEditArgs(StrictArgs):
    """``EditArgs`` plus an optional top-level ``path`` applied to every edit without one."""

    edits: list[LenientEditOperation] = Field(min_length=1)
    path: str | None = None

    @model_validator(mode="after")
    def _fill_paths(self):
        for index, edit in enumerate(self.edits):
            if edit.path is None:
                if self.path is None:
                    raise ValueError(f"edits.{index}.path is required (or give one top-level path)")
                edit.path = self.path
        return self

    def strict(self) -> EditArgs:
        return EditArgs.model_validate(
            {"edits": [edit.model_dump(exclude_none=True) for edit in self.edits]}
        )


def lenient_edit_spec(files) -> ToolSpec:
    """The thinharness edit spec with the lenient argument shape (same handler)."""

    def handler(args):
        args = coerce_args(args, LenientEditArgs)
        return files.edit(args.strict())

    return ToolSpec(
        "edit",
        DEFAULT_EDIT_DESCRIPTION + " A top-level path applies to every edit that has none.",
        LenientEditArgs,
        handler,
        sequential=True,
        instructions=DEFAULT_EDIT_INSTRUCTIONS,
    )
