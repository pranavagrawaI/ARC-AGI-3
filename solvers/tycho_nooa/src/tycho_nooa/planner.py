"""Tycho's single-agent planner contract and its observable lifecycle."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tycho.agent.colors import COLORMAP
from tycho.prompts.render import render_prompt
from tycho.workspace import wm_templates
from tycho.workspace.workspace import GameWorkspace, _parse_grid_text

from tycho_nooa.models import GameTurn, tycho_workspace_available_actions

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
HOST_OWNED_PLANNER_PATHS = frozenset(
    {
        "plan.py",
        "verify.py",
        "notes/validated_plan.json",
        "notes/validated_plan_expected.npz",
    }
)
_SEEDED_PLAN_BYTES = wm_templates.PLAN.encode("utf-8")
_MAX_PLAN_ARTIFACT_BYTES = 1_000_000
_PLAN_METHODS = frozenset({"auto", "astar", "bfs", "subgoals"})


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

    def publish_candidate(self, candidate_workspace: Path, result: str) -> str:
        """Validate a disposable planner artifact before publishing it atomically."""
        candidate = candidate_workspace / "notes" / "validated_plan.json"
        canonical = self._artifact_path()
        try:
            payload = self._validated_candidate(candidate, result)
        except (OSError, TypeError, ValueError) as error:
            canonical.unlink(missing_ok=True)
            return f"PLANNER_PUBLICATION: rejected ({error})"

        canonical.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=canonical.parent,
            prefix=".validated_plan.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, canonical)
        finally:
            temporary.unlink(missing_ok=True)
        return "PLANNER_PUBLICATION: accepted by T1"

    def _validated_candidate(self, candidate: Path, result: str) -> bytes:
        if _planner_status(result) != "plan_found":
            raise ValueError("planner did not produce plan_found")
        payload = _read_bounded_regular_file(candidate, _MAX_PLAN_ARTIFACT_BYTES)
        try:
            artifact = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("validated artifact is not JSON") from error
        self._validate_artifact_value(artifact)
        return json.dumps(artifact, separators=(",", ":")).encode("utf-8")

    def seeded_plan_bytes(self) -> bytes:
        """Return pinned seed bytes only while canonical plan.py is unchanged."""
        plan = self.workspace.dir / "plan.py"
        observed = _read_bounded_regular_file(plan, len(_SEEDED_PLAN_BYTES))
        if observed != _SEEDED_PLAN_BYTES:
            raise ValueError("canonical plan.py differs from the pinned seeded planner")
        return _SEEDED_PLAN_BYTES

    def _validate_artifact_value(self, artifact: Any) -> None:
        if not isinstance(artifact, dict):
            raise TypeError("validated artifact must be an object")
        _validate_artifact_header(artifact)
        frame = self._current_frame()
        _validate_start_anchor(artifact.get("start"), frame)
        model_hash = hashlib.sha256(
            (self.workspace.dir / "world_model.py").read_bytes()
        ).hexdigest()
        _validate_model_commitment(artifact.get("world_model_sha256"), model_hash)
        _validate_plan_payload(artifact, frame["available_actions"])

    def _current_frame(self) -> dict[str, Any]:
        try:
            pointer = json.loads(
                (self.workspace.dir / ".current_frame.json").read_text(encoding="utf-8")
            )
            relative = Path(str(pointer["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError
            rows = _parse_grid_text(
                (self.workspace.dir / relative).read_text(encoding="utf-8")
            )
            grid = np.asarray(rows, dtype=np.int16)
            if grid.ndim != 2 or grid.size == 0:
                raise ValueError
            metadata = json.loads(
                (self.workspace.dir / relative.with_suffix(".json")).read_text(
                    encoding="utf-8"
                )
            )
            available_actions = tycho_workspace_available_actions(metadata)
            return {
                "level": int(pointer["level"]),
                "turn": int(pointer["turn"]),
                "grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest(),
                "available_actions": frozenset(available_actions),
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("current frame is unavailable") from error

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
    direct = _direct_planner_command(source)
    if direct is not None:
        return direct
    return _python_wrapped_planner_command(source)


def _direct_planner_command(source: str) -> str | None:
    try:
        parts = shlex.split(source)
    except ValueError:
        return None
    if parts and parts[0] in {"python", "python3"}:
        parts = parts[1:]
    if parts == ["plan.py"]:
        return "python plan.py"
    if len(parts) == 2 and parts[0] == "plan.py" and parts[1] in _PLAN_METHODS:
        return shlex.join(["python", *parts])
    return None


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
    for node in ast.walk(tree):
        strings = _runner_strings(node, runners)
        if strings is None:
            continue
        if "plan.py" not in strings:
            continue
        method = next((value for value in strings if value in _PLAN_METHODS), None)
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


def _validate_artifact_header(artifact: dict[str, Any]) -> None:
    if (
        type(artifact.get("schema_version")) is not int
        or artifact.get("schema_version") != 1
        or artifact.get("status") != "validated"
    ):
        raise ValueError("validated artifact has the wrong schema or status")
    if artifact.get("method") not in {"astar", "bfs", "subgoals", "planner.py"}:
        raise ValueError("validated artifact has an unknown planning method")
    if artifact.get("validation") != (
        "canonical world_model replay reached level_complete"
    ):
        raise ValueError("validated artifact has no canonical replay claim")


def _validate_start_anchor(start: Any, frame: dict[str, Any]) -> None:
    if not isinstance(start, dict):
        raise TypeError("validated artifact has no start anchor")
    if type(start.get("level")) is not int or type(start.get("turn")) is not int:
        raise ValueError("validated artifact has a malformed start anchor")
    if (start.get("level"), start.get("turn")) != (
        frame["level"],
        frame["turn"],
    ):
        raise ValueError("validated artifact is not anchored to the current frame")
    if start.get("grid_sha256") != frame["grid_sha256"]:
        raise ValueError("validated artifact has the wrong current-grid hash")


def _validate_model_commitment(value: Any, expected: str) -> None:
    if value != expected:
        raise ValueError("validated artifact has the wrong world-model hash")


def _validate_plan_payload(
    artifact: dict[str, Any], available: frozenset[str]
) -> None:
    actions, expected = _plan_sequences(artifact)
    _validate_plan_actions(actions, available)
    _validate_expected_grids(expected)


def _plan_sequences(artifact: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    actions = artifact.get("actions")
    expected = artifact.get("expected_grid_sha256")
    length = artifact.get("plan_length")
    if (
        type(length) is not int
        or length <= 0
        or length > 10_000
        or not isinstance(actions, list)
        or not isinstance(expected, list)
        or length != len(actions)
        or length != len(expected)
    ):
        raise ValueError("validated artifact has inconsistent plan lengths")
    return actions, expected


def _validate_plan_actions(actions: list[Any], available: frozenset[str]) -> None:
    if not all(_valid_action(action, available) for action in actions):
        raise ValueError("validated artifact contains an invalid action")


def _validate_expected_grids(expected: list[Any]) -> None:
    if not all(_is_sha256(value) for value in expected):
        raise ValueError("validated artifact contains an invalid grid hash")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_action(value: Any, available: frozenset[str]) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("action"), str):
        return False
    action = value["action"]
    if not action or action not in available:
        return False
    row = value.get("row")
    col = value.get("col")
    if action in {"CLICK", "ACTION6"}:
        return _valid_click_coordinates(row, col)
    return row is None and col is None


def _valid_click_coordinates(row: Any, col: Any) -> bool:
    return (
        type(row) is int
        and type(col) is int
        and 0 <= row <= 63
        and 0 <= col <= 63
    )


def is_host_owned_planner_path(workspace: Path, requested: str) -> bool:
    """Return whether a file operation targets the host-owned planner namespace."""
    raw = Path(requested.replace("\\", "/"))
    root = workspace.resolve()
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    protected = {(root / path).resolve() for path in HOST_OWNED_PLANNER_PATHS}
    if candidate in protected:
        return True
    notes = (root / "notes").resolve()
    return candidate.parent == notes and candidate.name.startswith(".validated_plan.")


def _read_bounded_regular_file(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{path.name} is not a regular file")
        if metadata.st_size > limit:
            raise ValueError(f"{path.name} exceeds {limit} bytes")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(limit + 1)
        if len(payload) > limit:
            raise ValueError(f"{path.name} exceeds {limit} bytes")
        return payload
    finally:
        os.close(descriptor)


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
