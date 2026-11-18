"""Process-local, hash-pinned CodeAct binding persistence; installed NOOA is untouched.

Ownership is a set of model-assigned module-scope identifiers, never a dump of
runtime locals/globals. The vendored execution boundary captures only that set.
The existing OS/pipe authority remains the containment boundary; the AST contract
below prevents direct/reflection-based use of private interpreter state, not
arbitrary-code security in the Python process.
"""

import ast
from contextvars import ContextVar
import hashlib
import inspect
import json
from pathlib import Path
import symtable

_OWNED = ContextVar("model_cell_owned", default=None)
# Names injected by the wrapper or owned by the host are not model variables.
RESERVED = frozenset(
    {"self", "Out", "__builtins__", "__repl_captured_locals__", "__repl_wrapper__"}
)
REFLECTION = frozenset(
    {
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "eval",
        "exec",
        "compile",
        "__import__",
    }
)
PRIVATE_MODULES = frozenset(
    {
        "nooa",
        "actor",
        "worker",
        "cell_state",
        "evidence_api",
        "prefill",
        "solver_agent",
        "sys",
        "gc",
        "inspect",
        "importlib",
        "builtins",
        "types",
        "ctypes",
    }
)
CONTRACT = (
    "Python cell bindings persist within this Actor session, including underscore-prefixed "
    "variables, imports and functions. Host input bindings (self, Out, notification and "
    "other call parameters) are read-only; their public APIs remain usable. "
    "Runtime private names, __repl_* names, private object attributes and interpreter "
    "reflection are reserved: do not read, replace or copy them. Runtime/reflection "
    "modules, wildcard/relative imports and global/nonlocal declarations are unavailable. "
    "Use public tools and direct attributes; assign cell names directly. "
    "A CELL_STATE_REPAIR reports a rejected cell or missing binding; no hidden execution "
    "or model retry occurs. Calls still count toward the same request cap."
)


class _Inputs(dict):
    """Track actual call parameters separately from convenience module imports."""

    def __init__(self, values, protected):
        super().__init__(values)
        self.protected = frozenset(protected)


def private_host_names(agent):
    names = set()
    for cls in type(agent).__mro__:
        module = inspect.getmodule(cls)
        if module is not None:
            names.update(n for n in vars(module) if n.startswith("_"))
    return names


def bindings(code):
    """Python's own symbol table covers imports, unpacking, walrus, match and loops."""
    table = symtable.symtable(code, "<Actor cell>", "exec")
    return {
        s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()
    }


def missing_private(code, owned):
    tables = [symtable.symtable(code, "<Actor cell>", "exec")]
    while tables:
        table = tables.pop()
        for symbol in table.get_symbols():
            name = symbol.get_name()
            if (
                symbol.is_referenced()
                and symbol.is_global()
                and name.startswith("_")
                and name not in owned
            ):
                return "binding unavailable: " + name
        tables.extend(table.get_children())
    return None


def violation(code, created, host_inputs):
    if created & (RESERVED | host_inputs | REFLECTION) or any(
        n.startswith("__repl_") for n in created
    ):
        return "host/runtime binding is read-only"
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return "global/nonlocal declarations are runtime-owned; assign cell names directly"
        if isinstance(node, ast.Name):
            name = node.id
            if (
                name in REFLECTION
                or name.startswith("__repl_")
                or name == "__builtins__"
            ):
                return "runtime namespace/reflection is private"
            if (
                isinstance(node.ctx, (ast.Store, ast.Del))
                and name in RESERVED | host_inputs
            ):
                return "host input binding is read-only: " + name
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return "private attributes are outside the Actor cell contract"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(m.split(".")[0] in PRIVATE_MODULES for m in modules):
                return "runtime/reflection modules are private"
            if isinstance(node, ast.ImportFrom) and (
                node.level
                or any(a.name.startswith("_") or a.name == "*" for a in node.names)
            ):
                return (
                    "private and wildcard imports are outside the Actor cell contract"
                )
    return None


def install(ids):
    """Install once in each sterile worker, before any Actor generation/prefill."""
    from nooa.runtime.actor import ActorRuntime, _extract_captured_locals
    from nooa.strategies.codeact import CodeActStrategy
    from nooa.events import ExecutionResult

    here = Path(__file__).parent
    pin = json.loads((here / "runtime-boundary.json").read_text())
    original = ActorRuntime.execute_code
    if (
        hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
        != pin["original_method_sha256"]
    ):
        raise RuntimeError("pinned Actor execution boundary drift")
    if (
        hashlib.sha256(Path(inspect.getfile(ActorRuntime)).read_bytes()).hexdigest()
        != pin["original_module_sha256"]
    ):
        raise RuntimeError("pinned Actor runtime module drift")

    def capture(globals_):
        owned = _OWNED.get()
        if owned is None:
            return _extract_captured_locals(globals_)
        return {
            k: v
            for k, v in globals_.get("__repl_captured_locals__", {}).items()
            if k in owned
        }

    namespace = dict(original.__globals__, _cell_owned=_OWNED, _capture_owned=capture)
    source = here / "actor_execute.py.txt"
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    ActorRuntime.execute_code = namespace["_Vendored"].execute_code
    execute = CodeActStrategy._execute_code
    build = CodeActStrategy._build_builtins

    def build_inputs(strategy, runtime, call):
        return _Inputs(
            build(strategy, runtime, call), set(call.kwargs) | {"return_result"}
        )

    CodeActStrategy._build_builtins = build_inputs

    async def execute_owned(
        strategy,
        runtime,
        code,
        builtins,
        session,
        target_method_name,
        tool_call_id=None,
    ):
        # Only IDs minted by the trusted prefill allocator bypass model policy.
        # Provider IDs are checked against that registry before tool execution.
        if tool_call_id in ids.local:
            token = _OWNED.set(None)
            try:
                return await execute(
                    strategy,
                    runtime,
                    code,
                    builtins,
                    session,
                    target_method_name,
                    tool_call_id,
                )
            finally:
                _OWNED.reset(token)
        try:
            created = bindings(code)
            if not isinstance(builtins, _Inputs):
                raise RuntimeError("CodeAct call-input ownership missing")
            protected = builtins.protected | private_host_names(runtime.agent)
            prior = set(session.session_locals) - RESERVED - protected
            owned = prior | created
            problem = violation(code, created, protected) or missing_private(
                code, owned
            )
        except SyntaxError:
            return await execute(
                strategy,
                runtime,
                code,
                builtins,
                session,
                target_method_name,
                tool_call_id,
            )
        if problem:
            return ExecutionResult(
                stdout="CELL_STATE_REPAIR: " + problem + "; cell not executed.\n",
                error=None,
                defined_methods={},
            )
        # Private module convenience imports are not inputs to model execution.
        public_builtins = {k: v for k, v in builtins.items() if not k.startswith("_")}
        token = _OWNED.set(owned)
        try:
            result = await execute(
                strategy,
                runtime,
                code,
                public_builtins,
                session,
                target_method_name,
                tool_call_id,
            )
        finally:
            _OWNED.reset(token)
        # Replace only owned session entries, respecting del and exception-target cleanup.
        if result.wrapper_line_offset and result.captured_locals is not None:
            for name in owned:
                session.session_locals.pop(name, None)
            session.session_locals.update(result.captured_locals)
        if isinstance(result.error, NameError):
            result.stdout += (
                "\nCELL_STATE_REPAIR: binding unavailable: "
                + str(result.error)[:240]
                + ". Define it before use. Prior cell effects are retained; no retry performed.\n"
            )
            result.error = None
        return result

    CodeActStrategy._execute_code = execute_owned
