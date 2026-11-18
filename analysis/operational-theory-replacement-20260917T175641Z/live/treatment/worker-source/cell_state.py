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
import types

_OWNED = ContextVar("model_cell_owned", default=None)
_FACADE = ContextVar("model_cell_facade", default=None)
_SHARED = ContextVar("model_cell_shared_globals", default=None)
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
        "object",
        "type",
        "super",
        "dir",
        "id",
        "hash",
        "repr",
        "format",
        "help",
        "isinstance",
        "issubclass",
        "copy",
        "deepcopy",
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
        "operator",
        "pickle",
        "dill",
        "marshal",
        "cloudpickle",
        "functools",
        "__future__",
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


class _OpaqueCallable:
    """Callable capability with no reference to the backing agent in its state."""

    __slots__ = ("__token",)

    def __init__(self, token):
        object.__setattr__(self, "_OpaqueCallable__token", token)

    def __call__(self, *args, **kwargs):
        return _CAPABILITY_FUNCS[
            object.__getattribute__(self, "_OpaqueCallable__token")
        ](*args, **kwargs)

    def __getattribute__(self, name):
        if name in {"__call__", "__class__", "__slots__"}:
            return object.__getattribute__(self, name)
        raise AttributeError("opaque capability attribute unavailable")


_CAPABILITY_FUNCS = {}
_NEXT_CAPABILITY = 0


def _capability(fn):
    token = _register(fn)
    return _OpaqueCallable(token)


def _register(fn):
    global _NEXT_CAPABILITY
    token = f"cap-{_NEXT_CAPABILITY}"
    _NEXT_CAPABILITY += 1
    _CAPABILITY_FUNCS[token] = fn
    return token


class ActorFacade:
    """Small public capability surface; the raw Actor never enters cell globals."""

    __slots__ = ("__token",)

    def __init__(self, agent):
        object.__setattr__(self, "_ActorFacade__token", _register(agent))

    def __getattribute__(self, name):
        if name.startswith("_") or name in {"__class__", "__dict__", "__weakref__"}:
            raise AttributeError("private Actor capability unavailable")
        approved = {
            "latest_state",
            "observations",
            "grid_array",
            "grid_summary",
            "format_game_context",
            "submit_actions",
            "list_notes",
            "read_notes",
            "append_notes",
            "write_notes",
            "decision_packet",
            "last_transition",
            "workflow_skill",
            "operational_cycle",
            "operational_hash",
            "register_operational_artifact",
            "write_helper",
            "load_helpers",
            "queue_manager",
            "skills",
            "h",
            "vars",
            "v",
        }
        if name not in approved:
            raise AttributeError("Actor capability is not public")
        token = object.__getattribute__(self, "_ActorFacade__token")
        agent = _CAPABILITY_FUNCS[token]
        value = getattr(agent, name)
        if name == "queue_manager":
            return QueueFacade(value)
        if name in {"skills", "h", "vars", "v"}:
            return _sanitize(value)
        return _capability(
            lambda *args, _value=value, **kwargs: _sanitize(_value(*args, **kwargs))
        )

    def __setattr__(self, name, value):
        raise AttributeError("Actor capabilities are read-only")


def _sanitize(value):
    if callable(value):
        return _capability(value)
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize(v) for v in value)
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    return PublicObject(value)


class PublicObject:
    __slots__ = ("__token",)

    def __init__(self, value):
        object.__setattr__(self, "_PublicObject__token", _register(value))

    def __getattribute__(self, name):
        if name.startswith("_"):
            raise AttributeError("private capability unavailable")
        token = object.__getattribute__(self, "_PublicObject__token")
        value = getattr(_CAPABILITY_FUNCS[token], name)
        return _sanitize(value)

    def __setattr__(self, name, value):
        raise AttributeError("capability objects are read-only")


class QueueFacade:
    __slots__ = ("__token",)

    def __init__(self, value):
        object.__setattr__(self, "_QueueFacade__token", _register(value))

    def __getattribute__(self, name):
        if name.startswith("_") or name not in {
            "status",
            "spawn",
            "shutdown",
            "cancel",
        }:
            raise AttributeError("queue capability unavailable")
        token = object.__getattribute__(self, "_QueueFacade__token")
        value = getattr(_CAPABILITY_FUNCS[token], name)
        return _capability(value) if callable(value) else _sanitize(value)

    def __setattr__(self, name, value):
        raise AttributeError("queue capability is read-only")


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
        if isinstance(node, ast.Attribute) and node.attr in {
            "__class__",
            "__dict__",
            "__mro__",
            "__bases__",
            "__base__",
            "__globals__",
            "__closure__",
            "__code__",
            "__func__",
            "__self__",
            "__reduce__",
            "__reduce_ex__",
            "__getattribute__",
            "__getattr__",
        }:
            return "object traversal is outside the Actor cell contract"
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

    namespace = dict(
        original.__globals__,
        _cell_owned=_OWNED,
        _cell_facade=_FACADE,
        _cell_shared=_SHARED,
        _capture_owned=capture,
    )
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
        facade = ActorFacade(runtime.agent)
        if tool_call_id in ids.local:
            token = _OWNED.set(None)
            # Trusted framework prefill code runs against the real runtime object;
            # only model-authored cells receive the facade.
            facade_token = _FACADE.set(None)
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
                _FACADE.reset(facade_token)
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
        for _name in ("locals", "globals"):
            if _name in builtins:
                public_builtins[_name] = builtins[_name]
        token = _OWNED.set(owned)
        facade_token = _FACADE.set(facade)
        if not hasattr(session, "_owned_globals"):
            session._owned_globals = {"self": facade}
        shared_globals = session._owned_globals
        shared_globals["self"] = facade
        shared_globals.update(
            {
                name: session.session_locals[name]
                for name in owned
                if name in session.session_locals
            }
        )
        for name, value in list(session.session_locals.items()):
            if name in owned and callable(value) and hasattr(value, "__code__"):
                value = types.FunctionType(
                    value.__code__,
                    shared_globals,
                    value.__name__,
                    value.__defaults__,
                    value.__closure__,
                )
                session.session_locals[name] = value
                shared_globals[name] = value
        shared_token = _SHARED.set(shared_globals)
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
            _FACADE.reset(facade_token)
            _SHARED.reset(shared_token)
            _OWNED.reset(token)
        # Replace only owned session entries, respecting del and exception-target cleanup.
        if result.wrapper_line_offset and result.captured_locals is not None:
            for name in owned:
                session.session_locals.pop(name, None)
            session.session_locals.update(result.captured_locals)
            for name in owned:
                if name in session.session_locals:
                    shared_globals[name] = session.session_locals[name]
            for name, value in list(session.session_locals.items()):
                if callable(value) and name in owned and hasattr(value, "__code__"):
                    value.__globals__.update(shared_globals)
                    session.session_locals[name] = types.FunctionType(
                        value.__code__,
                        shared_globals,
                        value.__name__,
                        value.__defaults__,
                        value.__closure__,
                    )
                    shared_globals[name] = session.session_locals[name]
        if isinstance(result.error, NameError):
            result.stdout += (
                "\nCELL_STATE_REPAIR: binding unavailable: "
                + str(result.error)[:240]
                + ". Define it before use. Prior cell effects are retained; no retry performed.\n"
            )
            result.error = None
        return result

    CodeActStrategy._execute_code = execute_owned
