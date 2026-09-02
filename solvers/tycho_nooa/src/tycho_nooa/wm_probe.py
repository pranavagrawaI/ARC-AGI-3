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

- `not_run_goal_unsatisfiable` replaces the search only for the restricted proof
  that `outcome()` is one unconditional non-goal string return. Literal absence
  and sampled non-goal outcomes remain diagnostics, never blockers.
- `PLANNER_CONTRACT` reports what each entry of `actions()` does to the planner's
  own key, and whether the key ever merges two paths. Reported, never enforced.

`probe_gate` carries both and is inlined here as source text, because the probe
executes inside Tycho's sandbox where `tycho_nooa` is not importable.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import shutil
import sys
import tempfile
from contextvars import ContextVar
from pathlib import Path

from tycho.workspace.agent_tools import ToolExecutor
from tycho.workspace.sandbox import PythonSandbox, SandboxError

from tycho_nooa import probe_gate
from tycho_nooa.hardened_host import command as hardened_host_command
from tycho_nooa.hardened_host import enabled as hardened_host_enabled
from tycho_nooa.trusted_runtime import MANIFEST_PATH, inspect_runtime

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

_GATED_SEARCH = """    if planner_ready and _wmgate()["goal_status"] == "proven_unsatisfiable":
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

# Upstream's reply when an edit leaves the syntax tree intact: the verifier is
# skipped, so the tool result carries no metrics. A reader downstream cannot tell
# that from a verifier that ran and found nothing unless it matches this text, so
# the notice is named here rather than re-spelled at each use.
AST_UNCHANGED_NOTICE = "skipped: Python syntax tree is unchanged"

# Upstream `ToolExecutor._wm_feedback`, which GatedToolExecutor reimplements. A
# change to it is a change to the behaviour being duplicated, so it must be read
# before it is copied forward.
_UPSTREAM_WM_FEEDBACK_SHA256 = (
    "44e9dac2216ae6b3d3a70f60d9f61e8e9814520031d177516e804c47f0bd1b11"
)


class ReadOnlyWorkspacePythonSandbox(PythonSandbox):
    """Builder Python may inspect the workspace but never mutate its authority."""

    def command(self, workspace, script, *, name, args=()):
        if self.runtime == "host":
            raise SandboxError(
                "transactional Builder run_python requires container isolation; "
                "host runtime cannot enforce a read-only workspace"
            )
        command = super().command(workspace, script, name=name, args=args)
        mount_index = command.index("--mount") + 1
        command[mount_index] = command[mount_index] + ",readonly"
        return command


class TrustedRuntimePythonSandbox(PythonSandbox):
    """Expose exact workspace values without granting publication authority."""

    _GENERATED_SOURCE_ROOT = "/tycho-generated"
    _GENERATED_SOURCE_PATH = f"{_GENERATED_SOURCE_ROOT}/source.py"
    _GENERATED_SOURCE_RUNNER = (
        "import runpy,sys;"
        f"sys.argv[0]={_GENERATED_SOURCE_PATH!r};"
        f"runpy.run_path({_GENERATED_SOURCE_PATH!r},run_name='__main__')"
    )

    def __init__(
        self,
        runtime: str | None = None,
        image: str | None = None,
        *,
        workspace_read_only: bool = True,
    ) -> None:
        super().__init__(runtime=runtime, image=image)
        self.workspace_read_only = workspace_read_only
        self._generated_source: ContextVar[Path | None] = ContextVar(
            "tycho_nooa_generated_source", default=None
        )

    def run_source(self, workspace, source, *, timeout, args=()):
        """Run generated source outside the actor-visible evidence namespace."""
        root = self._workspace(workspace)
        with tempfile.TemporaryDirectory(prefix="tycho-generated-source-") as directory:
            script = Path(directory) / "source.py"
            script.write_text(source, encoding="utf-8")
            token = self._generated_source.set(script.resolve(strict=True))
            try:
                return self.run_script(root, script, timeout=timeout, args=args)
            finally:
                self._generated_source.reset(token)

    def command(self, workspace, script, *, name, args=()):
        root = Path(workspace).resolve()
        generated_source = self._generated_source.get()
        script_path = Path(script).resolve(strict=True)
        if generated_source is not None and script_path == generated_source:
            command = self._generated_source_command(root, name=name, args=args)
        else:
            command = super().command(root, script_path, name=name, args=args)
        runtime = inspect_runtime(root).runtime
        paths = (
            ()
            if runtime is None
            else (
                root / MANIFEST_PATH,
                *(root / module.path for module in runtime.modules),
            )
        )
        if self.runtime == "host":
            return _host_runtime_command(
                root,
                paths,
                command,
                workspace_read_only=self.workspace_read_only,
                generated_source=generated_source,
            )
        return _container_runtime_command(
            self.image,
            paths,
            command,
            workspace_read_only=self.workspace_read_only,
            generated_source=generated_source,
        )

    def _generated_source_command(
        self, workspace: Path, *, name: str, args: tuple[str, ...]
    ) -> list[str]:
        """Build upstream's command without leaving its placeholder in the workspace."""
        with tempfile.NamedTemporaryFile(
            suffix=".py", prefix=".tycho-command-", dir=workspace
        ) as handle:
            placeholder = Path(handle.name)
            command = super().command(
                workspace, placeholder, name=name, args=args
            )
        placeholder_argument = (
            str(placeholder)
            if self.runtime == "host"
            else f"/workspace/{placeholder.name}"
        )
        argument_index = command.index(placeholder_argument)
        command[argument_index : argument_index + 1] = [
            "-c",
            self._GENERATED_SOURCE_RUNNER,
        ]
        return command


def _host_runtime_command(
    workspace: Path,
    protected_paths: tuple[Path, ...],
    command: list[str],
    *,
    workspace_read_only: bool = True,
    generated_source: Path | None = None,
) -> list[str]:
    if hardened_host_enabled():
        return hardened_host_command(
            workspace,
            protected_paths,
            command,
            workspace_read_only=workspace_read_only,
            generated_source=generated_source,
        )
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SandboxError(
            "host run_python requires bubblewrap to protect the trusted runtime"
        )
    return _bubblewrap_command(
        bwrap,
        workspace,
        protected_paths,
        command,
        workspace_read_only=workspace_read_only,
        generated_source=generated_source,
    )


def _container_runtime_command(
    image: str,
    protected_paths: tuple[Path, ...],
    command: list[str],
    *,
    workspace_read_only: bool = True,
    generated_source: Path | None = None,
) -> list[str]:
    if workspace_read_only and "--mount" in command:
        workspace_mount = command.index("--mount") + 1
        command[workspace_mount] = command[workspace_mount] + ",readonly"
    image_index = command.index(image)
    mounts = [
        argument
        for path in protected_paths
        for argument in (
            "--mount",
            f"type=bind,src={path},dst=/workspace/{path.name},readonly",
        )
    ]
    if generated_source is not None:
        mounts.extend(
            (
                "--mount",
                (
                    "type=bind,"
                    f"src={generated_source.parent},"
                    f"dst={TrustedRuntimePythonSandbox._GENERATED_SOURCE_ROOT},readonly"
                ),
            )
        )
    command[image_index:image_index] = mounts
    return command


def _bubblewrap_command(
    bwrap: str,
    workspace: Path,
    protected_paths: tuple[Path, ...],
    command: list[str],
    *,
    workspace_read_only: bool = True,
    generated_source: Path | None = None,
) -> list[str]:
    venv = Path(sys.prefix).resolve(strict=True)
    directories = _mount_parent_directories(workspace, venv)
    mounts = [
        argument
        for path in protected_paths
        for argument in ("--ro-bind", str(path), str(path))
    ]
    generated_source_mount = (
        []
        if generated_source is None
        else [
            "--dir",
            TrustedRuntimePythonSandbox._GENERATED_SOURCE_ROOT,
            "--ro-bind",
            str(generated_source.parent),
            TrustedRuntimePythonSandbox._GENERATED_SOURCE_ROOT,
        ]
    )
    return [
        bwrap,
        # Make generated Python PID 1 rather than leaving bubblewrap as PID 1.
        # `--clearenv` scrubs the generated process, not bubblewrap itself; when
        # bubblewrap remains visible as /proc/1, model-authored code can read its
        # inherited host environment through /proc/1/environ.
        "--as-pid-1",
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--disable-userns",
        "--clearenv",
        *directories,
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--ro-bind",
        str(venv),
        str(venv),
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--ro-bind" if workspace_read_only else "--bind",
        str(workspace),
        str(workspace),
        *mounts,
        *generated_source_mount,
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PATH",
        f"{venv}/bin:/usr/bin",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
        "--setenv",
        "OMP_NUM_THREADS",
        "1",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "1",
        "--setenv",
        "MKL_NUM_THREADS",
        "1",
        "--chdir",
        str(workspace),
        "--",
        *command,
    ]


def _mount_parent_directories(*paths: Path) -> list[str]:
    parents = {
        parent
        for path in paths
        for parent in path.parents
        if parent != Path("/")
    }
    arguments: list[str] = []
    for parent in sorted(parents, key=lambda value: (len(value.parts), str(value))):
        arguments.extend(("--dir", str(parent)))
    return arguments


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

    # Workspace mathematics joins upstream's preloaded `np`/`wmlib`/`grid`. A helper the
    # model has to remember to import is a helper it does not use; the guards also preserve
    # exact historical runtimes that predate one of the current modules.
    _PREAMBLE = ToolExecutor._PREAMBLE + (
        "try:\n"
        "    import symmetry\n"
        "except Exception:\n"
        "    symmetry = None\n"
        "try:\n"
        "    import dynamics\n"
        "except Exception:\n"
        "    dynamics = None\n"
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
                f"{AST_UNCHANGED_NOTICE} (comments/formatting only)"
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
