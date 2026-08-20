"""Tycho's world-model feedback probe, with a goal-satisfiability gate spliced in.

Adaptation of Tycho's `wm_feedback_probe.py.tmpl` (pinned; see PROVENANCE.md).
Upstream runs its inline planner whenever the verifier's dynamics, initial-render,
outcome, and terminal-render preconditions hold. None of those preconditions asks
whether a goal state exists, so a model whose `outcome()` can never return
"level_complete" -- the seeded stub, which a builder pass may leave untouched --
satisfies all of them, and the search that follows can only report
`no_plan_within_budget` or `planner_timeout`. Both read as "search was too slow",
and on such a model neither is true.

The splice adds one precondition and one audit:

- `not_run_goal_unsatisfiable` replaces the search when no state can satisfy the
  goal. That is decided statically, so it costs nothing and cannot be wrong about
  a model it does block.
- `PLANNER_CONTRACT` reports what each entry of `actions()` does to the planner's
  own key, and whether the key ever merges two paths. Reported, never enforced.

`probe_gate` carries both and is inlined here as source text, because the probe
executes inside Tycho's sandbox where `tycho_nooa` is not importable.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

from tycho.workspace.agent_tools import ToolExecutor

from tycho_nooa import probe_gate

# Upstream's planner configuration block: the last statement before the probe's
# planner machinery, and below the point where `wmlib` and `world_model` are both
# imported and known good. The gate's source goes immediately above it.
_CONFIG_ANCHOR = '_PLAN_METHOD = os.environ.get("TYCHO_AUTO_PLAN_METHOD", "astar")'

# Upstream's unconditional entry into the search. Replaced rather than appended
# to: appending would leave the misleading "search was too slow" reading of the
# planner's failure standing next to its correction. Replacing only these two
# lines also leaves upstream's long result-printing block un-reindented.
_SEARCH_ANCHOR = """    if planner_ready:
        pr = _planner_probe()
"""

_GATED_SEARCH = """    if planner_ready and not _wmgate()["goal_satisfiable"]:
        for _line in wmgate_blocked_lines(_wmgate()):
            print(_line)
    elif planner_ready:
        for _line in wmgate_audit_lines(_wmgate()):
            print(_line)
        pr = _planner_probe()
"""

# Deferred and memoized: the gate must not run for a model that never reaches the
# planner preconditions, and must not run twice for one that does.
_GATE_GLUE = '''

_WMGATE_CACHE = []


def _wmgate():
    """This probe run's gate report: computed on first use, at most once."""
    if not _WMGATE_CACHE:
        _WMGATE_CACHE.append(wmgate_report(world_model, wmlib))
    return _WMGATE_CACHE[0]


'''

# Upstream `ToolExecutor._wm_feedback`, which GatedToolExecutor reimplements. A
# change to it is a change to the behaviour being duplicated, so it must be read
# before it is copied forward.
_UPSTREAM_WM_FEEDBACK_SHA256 = (
    "44e9dac2216ae6b3d3a70f60d9f61e8e9814520031d177516e804c47f0bd1b11"
)


def gated_probe_source() -> str:
    """Return Tycho's probe template with the goal gate and operator audit spliced in."""
    template = _upstream_template()
    _require_once(template, _CONFIG_ANCHOR, "planner configuration block")
    _require_once(template, _SEARCH_ANCHOR, "unconditional planner entry")
    gate = _gate_source()
    _require_no_shadowing(template, gate)
    spliced = template.replace(_CONFIG_ANCHOR, gate + _CONFIG_ANCHOR, 1)
    spliced = spliced.replace(_SEARCH_ANCHOR, _GATED_SEARCH, 1)
    ast.parse(spliced)  # the sandbox would report a SyntaxError as "model does not load"
    return spliced


class GatedToolExecutor(ToolExecutor):
    """Tycho's tool executor, running the gated probe in place of the upstream one."""

    # `symmetry` joins upstream's preloaded `np`/`wmlib`/`grid`. A helper the model has to
    # remember to import is a helper it does not use; the guard keeps a workspace that was
    # never seeded (tests, bare dirs) from breaking every run_python call.
    _PREAMBLE = ToolExecutor._PREAMBLE + (
        "try:\n"
        "    import symmetry\n"
        "except Exception:\n"
        "    symmetry = None\n"
    )

    def _wm_feedback(
        self, path: str, before_ast: tuple[bool, str | None] | None = None
    ) -> str:
        """Upstream `ToolExecutor._wm_feedback` against `gated_probe_source()`.

        Duplicated rather than wrapped: upstream reads the probe text from a
        module-level constant, and the skip-on-unchanged-AST shortcut in front of
        it is behaviour to preserve exactly rather than re-derive. The hash guard
        makes an upstream change to it loud instead of silent.
        """
        _require_upstream_wm_feedback()
        if Path(path).name not in self._WM_FILES:
            return ""
        if not self.wm_feedback_enabled:
            return ""
        after_ast = self._wm_ast(path)
        if before_ast and before_ast[0] and after_ast[0] and before_ast[1] == after_ast[1]:
            return (
                "\n\n[auto world-model feedback]\n"
                "skipped: Python syntax tree is unchanged (comments/formatting only)"
            )
        out = self._run_python(gated_probe_source(), timeout=20)
        return "\n\n[auto world-model feedback]\n" + out.strip()


def _gate_source() -> str:
    """The gate module's own text, minus its module docstring, plus the glue.

    The probe is one file: a second module docstring would silently become a
    no-op expression, and a `__future__` import below the file's first statement
    would be a SyntaxError.
    """
    source = Path(probe_gate.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    if not tree.body:
        raise RuntimeError("probe_gate.py is empty")
    body = tree.body[1:] if ast.get_docstring(tree) is not None else tree.body
    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            raise RuntimeError(
                "probe_gate.py grew a __future__ import; it cannot have one, "
                "because its source is inlined into the middle of the probe template"
            )
    lines = source.splitlines(keepends=True)
    return "".join(lines[body[0].lineno - 1 :]) + _GATE_GLUE


def _upstream_template() -> str:
    from tycho.workspace import agent_tools

    template = agent_tools._WM_FEEDBACK_PROBE
    if not isinstance(template, str) or not template.strip():
        raise RuntimeError(
            "tycho.workspace.agent_tools._WM_FEEDBACK_PROBE is no longer the probe "
            "source text; re-check the goal-gate splice against the new shape"
        )
    return template


def _require_once(template: str, anchor: str, description: str) -> None:
    count = template.count(anchor)
    if count == 1:
        return
    raise RuntimeError(
        f"wm_feedback_probe.py.tmpl's {description} appears {count} times upstream, "
        f"expected exactly once; re-check the goal-gate splice against the new "
        f"text. Anchor: {anchor!r}"
    )


def _require_no_shadowing(template: str, gate: str) -> None:
    """Refuse a splice where the gate would rebind a name the probe already uses.

    Both halves are module-level code in one file, so a shared helper name would
    silently redefine the probe's own function and change results it prints
    rather than failing.
    """
    collisions = sorted(_bound_names(template) & _bound_names(gate))
    if collisions:
        raise RuntimeError(
            "the goal-gate splice would rebind names the probe template already "
            f"defines: {', '.join(collisions)}; rename them in probe_gate.py"
        )


def _bound_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _require_upstream_wm_feedback() -> None:
    digest = hashlib.sha256(
        inspect.getsource(ToolExecutor._wm_feedback).encode("utf-8")
    ).hexdigest()
    if digest != _UPSTREAM_WM_FEEDBACK_SHA256:
        raise RuntimeError(
            "tycho ToolExecutor._wm_feedback changed upstream "
            f"(sha256 {digest}); GatedToolExecutor._wm_feedback duplicates it and "
            "must be re-checked against the new body"
        )
