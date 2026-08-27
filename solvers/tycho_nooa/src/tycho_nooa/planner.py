"""Tycho's single-agent planner contract and its observable lifecycle."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tycho.agent.colors import COLORMAP
from tycho.prompts.render import render_prompt
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa.models import GameTurn

# wm_trigger.j2's own words claim all three of Tycho's real trigger paths
# (divergence, level start, GAME_OVER reset). This slice's TriggerBuilderLoop
# (trigger_builder.py before_actor) implements divergence and level start;
# GAME_OVER-reset is structurally unreachable under arc-host's TERMINAL_STATES
# design, not merely unimplemented — see PROVENANCE.md. Left unedited, the
# byte-exact sentence would promise a GAME_OVER-reset refresh that can never
# happen. Adapt this one sentence to the two implemented triggers only;
# everything else stays byte-exact.
_LIFECYCLE_CLAIM = (
    "The builder fires when the model is detectably wrong on dynamics or "
    "outcome, at level starts, and after GAME_OVER resets."
)
_IMPLEMENTED_TRIGGERS_CLAIM = (
    "The builder fires when the model is detectably wrong on dynamics or "
    "outcome, and at level starts."
)


_INVOKE_BUILDER_CLAUSE = (
    "call `invoke_builder` and put your beliefs in the `reason`"
)
_REQUEST_BUILDER_CLAUSE = (
    "call `request_builder` and put your beliefs in the focused request"
)


def actor_world_model_paragraph(
    wm_variant: str,
    *,
    request_builder_instead_of_invoke: bool = False,
) -> str:
    """Return the actor's world-model paragraph for one of Tycho's modes.

    `none` is the direct-reasoning baseline and carries wm_none.j2 alone.
    actor.system.j2 gates its actor_beliefs.md paragraph on the builder modes
    (`orch`/`trigger`) because that paragraph is addressed to an actor whose
    notes a builder will read; wm_none.j2 names the same file in its own words,
    so the baseline still keeps durable hypotheses and is not simply the trigger
    prompt minus a sentence.
    """
    if wm_variant == "trigger":
        return trigger_actor_world_model_paragraph()
    if wm_variant == "single":
        return render_prompt("wm_single").strip()
    if wm_variant == "orch":
        text = render_prompt("wm_orch").strip()
        if not request_builder_instead_of_invoke:
            return text
        if _INVOKE_BUILDER_CLAUSE not in text:
            raise RuntimeError(
                "wm_orch.j2 no longer contains the invoke_builder clause; "
                "re-check the typed-goal adaptation against the new text"
            )
        return text.replace(_INVOKE_BUILDER_CLAUSE, _REQUEST_BUILDER_CLAUSE)
    if wm_variant == "none":
        return render_prompt("wm_none").strip()
    raise NotImplementedError(
        f"no actor world-model paragraph is wired for wm_variant={wm_variant!r}"
    )


def trigger_actor_world_model_paragraph() -> str:
    """Return Tycho's trigger-mode world-model paragraph plus its actor_beliefs.md channel.

    Byte-sourced from wm_trigger.j2 and the wm_variant == 'trigger' block of
    actor.system.j2 — the same two places real Tycho's actor.system prompt draws
    this text from in trigger mode, with one adaptation (see _LIFECYCLE_CLAIM).
    Trigger-mode actors are never taught Planner mechanics/plan.py: only the
    builder (which owns world_model.py) invokes it, via its own byte-identical
    builder.system.j2 render in builder.py.
    """
    wm_trigger = render_prompt("wm_trigger")
    if _LIFECYCLE_CLAIM not in wm_trigger:
        raise RuntimeError(
            "wm_trigger.j2's builder-fires sentence changed upstream; "
            "re-check the divergence-only adaptation against the new text"
        )
    wm_trigger = wm_trigger.replace(_LIFECYCLE_CLAIM, _IMPLEMENTED_TRIGGERS_CLAIM)
    system_prompt = render_prompt(
        "actor.system",
        colormap=COLORMAP,
        wm_variant="trigger",
        vision=False,
        show_grid=True,
        show_diff=True,
        lossless_cells=False,
        meta_reflect=False,
    )
    start = system_prompt.index("Keep your evolving hypotheses")
    beliefs_paragraph = system_prompt[start:].strip()
    return wm_trigger + "\n\n" + beliefs_paragraph


@dataclass(frozen=True)
class PlannerInvocation:
    """A Python call plus the artifact present before it ran."""

    command: str | None
    artifact_sha256_before: str | None


class PlannerIntegration:
    """Connect planner execution, guarded continuation, and durable diagnostics."""

    def __init__(self, workspace: GameWorkspace, trace_path: Path | None = None):
        self.workspace = workspace
        self.trace_path = trace_path
        self._event_index = 0
        self._level: int | None = None
        self._turn_in_level: int | None = None

    def hint(self, turn: GameTurn) -> str | None:
        """Return and trace Tycho's guarded next-action feedback for this frame."""
        self._level = turn.level
        self._turn_in_level = turn.turn_in_level
        hint = self.workspace.validated_plan_hint(
            turn.grid,
            level=turn.level,
            turn=turn.turn_in_level,
            available=turn.available_actions,
        )
        hint = self._add_click_coordinates(hint, turn)
        if hint is not None:
            self._append(
                {
                    "event": "planner_hint",
                    "level": turn.level,
                    "turn_in_level": turn.turn_in_level,
                    "status": _hint_status(hint),
                    "hint": hint,
                }
            )
        return hint

    def begin_invocation(self, code: str) -> PlannerInvocation:
        """Capture pre-execution state before a possibly indirect planner call."""
        return PlannerInvocation(
            command=planner_command(code),
            artifact_sha256_before=self._artifact_sha256(),
        )

    def finish_invocation(self, invocation: PlannerInvocation, result: str) -> None:
        """Trace the planner result and any newly produced validated artifact."""
        command = invocation.command or _planner_command_from_result(result)
        if command is None:
            return
        status = _planner_status(result)
        self._append(
            {
                "event": "planner_invocation",
                "level": self._level,
                "turn_in_level": self._turn_in_level,
                "command": command,
                "status": status,
                "result": result,
            }
        )
        artifact_sha256 = self._artifact_sha256()
        if artifact_sha256 is None:
            return
        if status != "plan_found" and artifact_sha256 == invocation.artifact_sha256_before:
            return
        artifact = self._read_artifact()
        self._append(
            {
                "event": "planner_artifact",
                "level": self._level,
                "turn_in_level": self._turn_in_level,
                "path": "notes/validated_plan.json",
                "sha256": artifact_sha256,
                "status": artifact.get("status"),
                "method": artifact.get("method"),
                "start": artifact.get("start"),
                "plan_length": artifact.get("plan_length"),
                "world_model_sha256": artifact.get("world_model_sha256"),
            }
        )

    def _artifact_sha256(self) -> str | None:
        try:
            data = self._artifact_path().read_bytes()
        except OSError:
            return None
        return hashlib.sha256(data).hexdigest()

    def _read_artifact(self) -> dict[str, Any]:
        try:
            value = json.loads(self._artifact_path().read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _artifact_path(self) -> Path:
        return self.workspace.dir / "notes" / "validated_plan.json"

    def _add_click_coordinates(self, hint: str | None, turn: GameTurn) -> str | None:
        """Adapt Tycho's ACTION6-only hint formatting to the host's CLICK label."""
        if hint is None or "next action is CLICK." not in hint:
            return hint
        artifact = self._read_artifact()
        try:
            offset = turn.turn_in_level - int(artifact["start"]["turn"])
            action = artifact["actions"][offset]
            row = int(action["row"])
            col = int(action["col"])
        except (KeyError, IndexError, TypeError, ValueError):
            return hint
        return hint.replace(
            "next action is CLICK.",
            f"next action is CLICK(row={row},col={col}).",
        )

    def _append(self, record: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        self._event_index += 1
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"event_index": self._event_index, **record}
        with self.trace_path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps(payload, separators=(",", ":")) + "\n")


def planner_command(code: str) -> str | None:
    """Normalize Tycho's direct or Python-wrapped planner command, if present."""
    source = code.strip()
    if not source:
        return None
    if "\n" not in source:
        try:
            parts = shlex.split(source)
        except ValueError:
            parts = []
        if parts and parts[0] in {"python", "python3"}:
            parts = parts[1:]
        if parts and Path(parts[0]).name == "plan.py":
            return shlex.join(["python", *parts])
    return _python_wrapped_planner_command(source)


def _python_wrapped_planner_command(source: str) -> str | None:
    """Recognize bounded standard-library ways to execute the seeded plan.py."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    runners = {
        "os.popen",
        "os.system",
        "runpy.run_path",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.run",
    }
    methods = {"auto", "astar", "bfs", "subgoals"}
    for node in ast.walk(tree):
        strings = _runner_strings(node, runners)
        if strings is None:
            continue
        if not any(Path(value).name == "plan.py" for value in strings):
            continue
        method = next((value for value in strings if value in methods), None)
        return shlex.join(["python", "plan.py", *([method] if method else [])])
    return None


def _runner_strings(node: ast.AST, runners: set[str]) -> list[str] | None:
    if not isinstance(node, ast.Call) or _dotted_name(node.func) not in runners:
        return None
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _planner_command_from_result(result: str) -> str | None:
    match = re.search(
        r"^PLANNER_COMMAND:\s+python(?:3)?\s+plan\.py(?:\s+(auto|astar|bfs|subgoals))?",
        result,
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    method = match.group(1)
    return shlex.join(["python", "plan.py", *([method] if method else [])])


def _planner_status(result: str) -> str:
    match = re.search(r"^PLANNER_STATUS:\s+([^\s(]+)", result, flags=re.MULTILINE)
    if match is not None:
        return match.group(1)
    if "error" in result.lower() or "traceback" in result.lower():
        return "error"
    return "unknown"


def _hint_status(hint: str) -> str:
    if hint.startswith("Validated plan continuation:"):
        return "continuation"
    markers = (
        ("world_model.py changed", "invalidated_model_changed"),
        ("observed frame diverged", "invalidated_observation_diverged"),
        ("is not available", "invalidated_action_unavailable"),
        ("matched all", "trajectory_complete"),
    )
    for marker, status in markers:
        if marker in hint:
            return status
    return "unknown"
