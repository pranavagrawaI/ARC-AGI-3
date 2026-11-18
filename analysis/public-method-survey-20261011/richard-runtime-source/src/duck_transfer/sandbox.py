"""One persistent Python interpreter per game for the corrected Duck agent.

The public ``run_sandboxed_python`` spawns a fresh child for every tool call, so helper
functions and parsed board models vanish between calls (the v3 ft09 run re-pasted the
same 40-line parser in 8 of 29 calls and hit NameErrors). ``StatefulSandbox`` keeps the
public bootstrap and its restrictions, but the child loops over ``{"type": "code"}``
requests: user definitions stay in ``runtime_globals`` while the runtime variables
(``current_frame``, ``history``, ``transitions``, ...) are re-injected on every call. A
timeout or child exit kills the interpreter and the next call starts a fresh one; the
model is told that its state was reset.
"""

import collections
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time

LOOP = '''
    request = initial
    while True:
        del action_results[:]
        stdout = io.StringIO()
        runtime_globals["result"] = None
        _refresh_state(request.get("state") or {})
        try:
            compiled = compile(str(request.get("code", "")), "<python_tool>", "exec")
            with contextlib.redirect_stdout(stdout):
                exec(compiled, runtime_globals, runtime_globals)
            _send(
                {
                    "type": "final",
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                    "segmentation": dict(_SEGMENTATION_STATS),
                    "history_size": len(retained_history),
                }
            )
        except Exception as exc:
            _send(
                {
                    "type": "error",
                    "error": _sanitize_exception(exc),
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                }
            )
        try:
            request = _recv()
        except EOFError:
            break
        if request.get("type") != "code":
            break
'''

REFRESH_HEAD = (
    "    def _refresh_state(state_payload):\n"
    '        current_frame = _frame_from_payload(state_payload.get("current_frame"))\n'
    '        history = _history_from_payload(state_payload.get("history"))\n'
)
RETAINED_REFRESH = (
    "    def _refresh_state(state_payload):\n"
    '        current_frame = _frame_from_payload(state_payload.get("current_frame"))\n'
    '        incoming = _history_from_payload(state_payload.get("history"))\n'
    '        base = state_payload.get("history_base")\n'
    "        if not base:\n"
    "            retained_history[:] = incoming\n"
    "        elif int(base) == len(retained_history):\n"
    "            retained_history.extend(incoming)\n"
    "        else:\n"
    '            raise RuntimeError("history sync mismatch: %s held, base %s" % (len(retained_history), base))\n'
    "        history = list(retained_history)\n"
)
SEGMENTATION_MEMO = """
_segment_layer_uncached = segment_layer
_SEGMENTATION_CACHE = {}
_SEGMENTATION_STATS = {"hits": 0, "misses": 0}


def segment_layer(layer, color_chars):
    key = (tuple(tuple(row) for row in layer), str(color_chars))
    cached = _SEGMENTATION_CACHE.get(key)
    if cached is not None:
        _SEGMENTATION_STATS["hits"] += 1
        return cached
    _SEGMENTATION_STATS["misses"] += 1
    if len(_SEGMENTATION_CACHE) >= 64:
        del _SEGMENTATION_CACHE[next(iter(_SEGMENTATION_CACHE))]
    cached = _SEGMENTATION_CACHE[key] = _segment_layer_uncached(layer, color_chars)
    return cached
"""

RESET_AFTER_TIMEOUT = "python state was reset after a timeout (redefine helpers before reuse)"
RESET_AFTER_EXIT = "Sandbox process exited unexpectedly; python state was reset"
INTERRUPTED_AFTER_TIMEOUT = (
    "the running statement was interrupted and python state was kept (functions and variables defined so far are "
    "still available; the interrupted call may have left partial results)"
)

# --- arena harness_fixes (29 September): interruptible calls, the inventory request, the wider allow-list ---------

# Top-level child code: the interrupt handler (SIGUSR1 raises ``ToolTimeout`` in the running statement while a call
# is armed) and the source registry behind ``inspect.getsource`` and the inventory.
FIXES_CORE = r'''
import inspect as _inspect_real
import reprlib as _reprlib
import signal as _signal
import types as _types


class ToolTimeout(BaseException):
    """Raised in the running statement when the host's tool timeout expires."""


_INTERRUPT = {"armed": False, "pending": False, "seconds": 30}


def _on_interrupt(signum, frame):
    if _INTERRUPT["armed"]:
        raise ToolTimeout(
            "the python call ran past the %ss tool timeout and was interrupted here" % _INTERRUPT["seconds"])
    _INTERRUPT["pending"] = True


_signal.signal(_signal.SIGUSR1, _on_interrupt)
_SOURCES = {}


def _register_source(code, text):
    stack = [code]
    while stack:
        item = stack.pop()
        _SOURCES[item] = text
        stack.extend(c for c in item.co_consts if isinstance(c, _types.CodeType))
    while len(_SOURCES) > 4096:
        del _SOURCES[next(iter(_SOURCES))]


def _getsourcelines(obj):
    obj = getattr(obj, "__func__", obj)
    obj = getattr(obj, "__wrapped__", obj)
    code = getattr(obj, "__code__", None)
    text = _SOURCES.get(code) if code is not None else None
    if text is None:
        raise OSError("source is available only for functions defined in this python session")
    lines = text.splitlines(True)
    return _inspect_real.getblock(lines[code.co_firstlineno - 1:]), code.co_firstlineno


def _getsource(obj):
    return "".join(_getsourcelines(obj)[0])

'''

# Top-level child code (``sandbox`` fix): restricted ``time``, ``pickle`` and ``inspect`` modules, three more
# pure-computation modules and four introspection builtins. File access stays closed (no ``open``, no ``io``/``os``);
# ``pickle.loads`` rebuilds only plain containers, and the ``time`` shim has clocks but no ``sleep``.
FIXES_MODULES = r'''
import pickle as _pickle_real
import time as _time_real


def _shim(name, doc, **attrs):
    module = _types.ModuleType(name, doc)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _no_sleep(*args, **kwargs):
    raise RuntimeError("time.sleep is disabled in the sandbox: the game advances only when you call action(...)")


_PICKLE_CLASSES = {("builtins", n) for n in ("set", "frozenset", "complex", "bytearray", "slice", "range")} | {
    ("collections", n) for n in ("deque", "Counter", "OrderedDict", "defaultdict")}


class _SafeUnpickler(_pickle_real.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _PICKLE_CLASSES:
            return super().find_class(module, name)
        raise _pickle_real.UnpicklingError("pickle.loads may not load %s.%s in the sandbox" % (module, name))


def _pickle_loads(data, **kwargs):
    return _SafeUnpickler(io.BytesIO(data), **kwargs).load()


SHIM_MODULES = {
    "time": _shim("time", "Clocks only (no sleep).", **{n: getattr(_time_real, n) for n in (
        "time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns", "process_time",
        "process_time_ns")}, sleep=_no_sleep),
    "pickle": _shim("pickle", "dumps/loads of plain data.", dumps=_pickle_real.dumps, loads=_pickle_loads,
                    HIGHEST_PROTOCOL=_pickle_real.HIGHEST_PROTOCOL, DEFAULT_PROTOCOL=_pickle_real.DEFAULT_PROTOCOL,
                    PickleError=_pickle_real.PickleError, PicklingError=_pickle_real.PicklingError,
                    UnpicklingError=_pickle_real.UnpicklingError),
    "inspect": _shim("inspect", "Signatures and the source of functions defined in this session.",
                     getsource=_getsource, getsourcelines=_getsourcelines,
                     **{n: getattr(_inspect_real, n) for n in (
                         "signature", "Signature", "Parameter", "getdoc", "cleandoc", "getfullargspec",
                         "getmembers", "isfunction", "isclass", "ismethod", "isroutine", "isbuiltin",
                         "isgenerator", "isgeneratorfunction")}),
}
SAFE_MODULES = SAFE_MODULES | {"hashlib", "textwrap", "traceback"}
SAFE_BUILTINS = SAFE_BUILTINS | {"globals", "id", "locals", "vars"}
_NOT_INSTALLED = {"numpy", "scipy", "pandas", "sklearn", "torch", "PIL", "cv2", "networkx", "sympy"}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name or "").split(".", 1)[0]
    if root in SHIM_MODULES and root == name:
        return SHIM_MODULES[root]
    if root not in SAFE_MODULES:
        why = ("is not installed in the sandbox (standard library only)" if root in _NOT_INSTALLED
               else "is not allowed in the sandbox")
        raise ImportError("Module '%s' %s. Importable modules: %s." % (
            name, why, ", ".join(sorted(SAFE_MODULES | set(SHIM_MODULES)))))
    return builtins.__import__(name, globals, locals, fromlist, level)
'''

PUBLIC_SAFE_IMPORT = '''
def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name or "").split(".", 1)[0]
    if root not in SAFE_MODULES:
        raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
    return builtins.__import__(name, globals, locals, fromlist, level)
'''

# The request loop with the fixes: calls are armed for the interrupt (``action`` disarms while it talks to the host,
# so an interrupt never lands inside the protocol), sources are registered, and an ``{"type": "inventory"}`` request
# returns the live user definitions without running code.
LOOP_FIXES = '''
    _raw_action = action

    def action(actions):
        armed = _INTERRUPT["armed"]
        _INTERRUPT["armed"] = False
        try:
            return _raw_action(actions)
        finally:
            _INTERRUPT["armed"] = armed
            if armed and _INTERRUPT["pending"]:
                _INTERRUPT["pending"] = False
                raise ToolTimeout("the python call ran past the %ss tool timeout; interrupted after this action"
                                  % _INTERRUPT["seconds"])

    runtime_globals["action"] = action
    _INTERRUPT["seconds"] = timeout_seconds
    _RUNTIME_NAMES = {"current_frame", "latest_frame", "history", "transitions", "last_transition", "previous_frame",
                      "last_action_frame", "last_action", "valid_actions", "last_action_result", "action", "result"}
    _short = _reprlib.Repr()
    _short.maxstring = _short.maxother = 48
    _short.maxlist = _short.maxtuple = _short.maxset = _short.maxfrozenset = _short.maxdict = 4
    _short.maxdeque = _short.maxarray = 4
    _short.maxlevel = 2

    def _inventory(limit=120):
        functions, classes, modules, values = [], [], [], []
        for name, value in list(runtime_globals.items()):
            if name in _RUNTIME_NAMES or name.startswith("__"):
                continue
            try:
                function = getattr(value, "__wrapped__", value)  # functools.lru_cache/wraps around a helper
                own = isinstance(function, _types.FunctionType) and function.__code__.co_filename == "<python_tool>"
                if own:
                    try:
                        signature = str(_inspect_real.signature(function))
                    except (TypeError, ValueError):
                        signature = "(...)"
                    doc = (function.__doc__ or "").strip().split("\\n", 1)[0]
                    functions.append([name, signature, doc])
                elif isinstance(value, type):
                    classes.append(name)
                elif isinstance(value, _types.ModuleType) or callable(value):
                    modules.append(name)  # imported modules and names (e.g. ``from collections import deque``)
                else:
                    try:
                        size = len(value)
                    except Exception:
                        size = None
                    values.append([name, type(value).__name__, size, _short.repr(value)[:80]])
            except Exception:
                values.append([name, "?", None, "<unavailable>"])
        return {"functions": functions[-limit:], "classes": classes[-limit:], "modules": modules[-limit:],
                "values": values[-limit:],
                "counts": {"functions": len(functions), "classes": len(classes), "values": len(values)}}

    request = initial
    while True:
        if request.get("type") == "inventory":
            _INTERRUPT["pending"] = False
            _INTERRUPT["armed"] = True
            try:
                payload = _inventory()
            except BaseException as exc:
                payload = {"error": "%s: %s" % (exc.__class__.__name__, exc)}
            finally:
                _INTERRUPT["armed"] = False
            _send({"type": "inventory", **payload})
        else:
            del action_results[:]
            stdout = io.StringIO()
            runtime_globals["result"] = None
            _refresh_state(request.get("state") or {})
            try:
                code = str(request.get("code", ""))
                compiled = compile(code, "<python_tool>", "exec")
                _register_source(compiled, code)
                _INTERRUPT["pending"] = False
                _INTERRUPT["armed"] = True
                try:
                    with contextlib.redirect_stdout(stdout):
                        exec(compiled, runtime_globals, runtime_globals)
                finally:
                    _INTERRUPT["armed"] = False
                _send(
                    {
                        "type": "final",
                        "stdout": stdout.getvalue(),
                        "result": _json_safe(runtime_globals.get("result")),
                        "action_results": _json_safe(action_results),
                        "segmentation": dict(_SEGMENTATION_STATS),
                        "history_size": len(retained_history),
                    }
                )
            except (Exception, ToolTimeout) as exc:
                _send(
                    {
                        "type": "error",
                        "error": _sanitize_exception(exc),
                        "stdout": stdout.getvalue(),
                        "action_results": _json_safe(action_results),
                        "interrupted": isinstance(exc, ToolTimeout),
                    }
                )
        try:
            request = _recv()
        except EOFError:
            break
        if request.get("type") not in ("code", "inventory"):
            break
'''

SANDBOX_FIXES = ("modules",)


def stateful_bootstrap(bootstrap, *, fixes=None):
    """Turn the public one-shot sandbox bootstrap into a request loop.

    ``fixes`` (the arena's ``harness_fixes``): None keeps the loop of every earlier run; a collection of names from
    ``SANDBOX_FIXES`` switches to the interruptible loop with the inventory request, plus the wider allow-list when
    it has "modules"."""
    head, sep, rest = bootstrap.partition('    try:\n        compiled = compile(str(initial.get("code", ""))')
    body, tail_sep, tail = rest.partition('\n\nif __name__ == "__main__":')
    if not sep or not tail_sep or bootstrap.count(sep) != 1:
        raise ValueError("Pinned Duck sandbox bootstrap changed")
    if fixes is not None and set(fixes) - set(SANDBOX_FIXES):
        raise ValueError(f"Unknown sandbox fixes {sorted(set(fixes) - set(SANDBOX_FIXES))}")
    loop = LOOP if fixes is None else LOOP_FIXES
    text = head + loop.strip("\n") + "\n" + tail_sep + tail
    replacements = {
        # The loop refreshes the runtime variables before every request.
        '    runtime_globals["action"] = action\n    _refresh_state(initial.get("state") or {})\n':
            '    runtime_globals["action"] = action\n',
        # Delta history: the host sends only entries the child has not seen (history_base =
        # entries already held); FrameView objects and their segmentation memo persist.
        REFRESH_HEAD: RETAINED_REFRESH,
        "    action_results = []\n    stdout = io.StringIO()\n":
            "    action_results = []\n    retained_history = []\n    stdout = io.StringIO()\n",
        # Segmentation memo keyed by the grid: current_frame is rebuilt every call and
        # containment is a full-grid flood fill per component.
        "\nHOST_STDOUT = sys.stdout\n": SEGMENTATION_MEMO + "\nHOST_STDOUT = sys.stdout\n",
        # RLIMIT_CPU counts cumulative CPU time; a long-lived child needs the game's budget,
        # which also bounds an orphan should the fleet process die.
        "    _set_limits(timeout_seconds)\n":
            '    _set_limits(int(initial.get("cpu_seconds", 1200)))\n',
    }
    if fixes is not None:
        import_hook = FIXES_CORE + (FIXES_MODULES if "modules" in fixes else PUBLIC_SAFE_IMPORT)
        replacements[PUBLIC_SAFE_IMPORT] = import_hook
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError("Pinned Duck sandbox bootstrap changed")
        text = text.replace(old, new, 1)
    return text


PROMPT_REPLACEMENTS = {
    "- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\n":
        "- Python state persists across `python` calls within this game: imports, functions and "
        "variables you define stay available until the game ends or a timeout resets the "
        "interpreter (the tool result says so). The runtime variables are refreshed on every call.\n",
    "- The only tool is `python`; call it with one ephemeral `code` string.\n":
        "- The only tool is `python`; call it with one `code` string.\n",
    "- The `python` tool code is not saved between calls, so rewrite any custom utility logic you still need.\n":
        "- Definitions from earlier `python` calls stay available; after a reported reset, redefine what you need.\n",
}

DESCRIPTION_REPLACEMENTS = {
    "Run one ephemeral Python snippet against preloaded ASCII game state.":
        "Run a Python snippet in this game's persistent interpreter against preloaded ASCII game state.",
}

PARAMETER_REPLACEMENTS = {
    "Python code to run. The snippet is ephemeral and is not saved across tool calls.":
        "Python code to run in this game's persistent session; definitions persist across tool calls.",
}


def _replace_all(text, replacements, what):
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f"Pinned Duck {what} changed")
        text = text.replace(old, new, 1)
    return text


def stateful_system_prompt(text):
    return _replace_all(text, PROMPT_REPLACEMENTS, "system prompt")


def stateful_tools(tools):
    """The public tool list with the python tool described as persistent."""
    patched = []
    for tool in tools:
        tool = json.loads(json.dumps(tool))
        function = tool.get("function") or {}
        if function.get("name") == "python":
            function["description"] = _replace_all(
                function["description"], DESCRIPTION_REPLACEMENTS, "tool description"
            )
            code = function["parameters"]["properties"]["code"]
            code["description"] = _replace_all(
                code["description"], PARAMETER_REPLACEMENTS, "tool parameter"
            )
        patched.append(tool)
    return patched


def _kill(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


class StatefulSandbox:
    """Drop-in for ``run_sandboxed_python`` that keeps one child alive per game."""

    def __init__(self, bootstrap, *, color_chars, on_event=None, label=""):
        self.bootstrap, self.color_chars = bootstrap, color_chars
        self.on_event, self.label = on_event, label
        self.process = None
        self.tempdir = None
        self.lines = None
        self.stderr = collections.deque(maxlen=50)
        self.calls = self.restarts = 0
        self.closed = False
        self.cpu_seconds = 1200
        self.lock = threading.Lock()
        # History entries the live child already holds (delta history).
        self.synced = 0
        self.delta_entries = 0
        # arena harness_fixes (``configure_fixes``): interrupt a call past its timeout instead of killing the child,
        # and answer inventory requests; both need the fixes bootstrap.
        self.interrupt_on_timeout = False
        self.interrupt_grace = 3.0
        self.inventory_supported = False
        self.interrupts = 0

    def configure_fixes(self, bootstrap, *, interrupt, grace=3.0):
        """Switch to a fixes bootstrap (``stateful_bootstrap(..., fixes=...)``) before the first call."""
        if self.process is not None:
            raise RuntimeError("Sandbox fixes must be configured before the interpreter starts")
        self.bootstrap = bootstrap
        self.inventory_supported = True
        self.interrupt_on_timeout, self.interrupt_grace = bool(interrupt), float(grace)

    def _interrupt(self):
        try:
            os.kill(self.process.pid, signal.SIGUSR1)
        except (OSError, AttributeError):
            pass

    def _event(self, kind, **data):
        if self.on_event is not None:
            self.on_event(kind, **data)

    def _spawn(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="duck_python_")
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", self.bootstrap],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=self.tempdir.name,
            env={
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": "/tmp",
                "TMPDIR": "/tmp",
                "PATH": os.environ.get("PATH", ""),
            },
            start_new_session=True,
        )
        lines = queue.Queue()

        def read_stdout():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        def read_stderr():
            for line in process.stderr:
                self.stderr.append(line)

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()
        self.process, self.lines = process, lines
        self.synced = 0

    def _delta(self, state):
        """State with only the history entries the child has not received yet."""
        state = dict(state or {})
        history = list(state.get("history") or [])
        base = self.synced if len(history) >= self.synced else 0
        state["history"], state["history_base"] = history[base:], base
        self.synced = len(history)
        self.delta_entries += len(history) - base
        return state

    def _discard(self, reason):
        process, self.process, self.lines = self.process, None, None
        self.synced = 0
        if process is not None:
            _kill(process)
            try:
                process.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                pass
        if self.tempdir is not None:
            self.tempdir.cleanup()
            self.tempdir = None
        if reason:
            self.restarts += 1
            self._event("duck_python_reset", reason=reason, restarts=self.restarts)

    def _send(self, payload):
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def run(self, *, code, timeout_seconds, initial_state, action_handler):
        with self.lock:
            if self.closed:
                return {"error": "Sandbox closed", "stdout": "", "action_results": []}
            self.calls += 1
            results = []
            try:
                if self.process is None:
                    self._spawn()
                    self._send(
                        {
                            "code": code,
                            "timeout_seconds": timeout_seconds,
                            "sandbox_cwd": self.tempdir.name,
                            "state": self._delta(initial_state),
                            "color_chars": self.color_chars,
                            "cpu_seconds": int(self.cpu_seconds),
                        }
                    )
                else:
                    self._send({"type": "code", "code": code, "state": self._delta(initial_state)})
            except OSError:
                self._discard("start_failed")
                return {"error": "Sandbox process could not start.", "stdout": "", "action_results": []}
            if self.closed:
                self._discard(None)
                return {"error": "Sandbox closed", "stdout": "", "action_results": []}
            deadline = time.monotonic() + max(1, int(timeout_seconds))
            interrupted_at = None  # harness_fixes: the time the interrupt was first sent
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if (self.interrupt_on_timeout and interrupted_at is None and self.process is not None
                            and self.process.poll() is None):
                        interrupted_at = time.monotonic()
                        deadline = interrupted_at + self.interrupt_grace
                        self._interrupt()
                        self._event("duck_python_interrupt", timeout=timeout_seconds)
                        continue
                    self._discard("timeout")
                    return {
                        "error": f"Tool timed out after {timeout_seconds}s; {RESET_AFTER_TIMEOUT}",
                        "stdout": "",
                        "action_results": list(results),
                    }
                try:
                    line = self.lines.get(timeout=min(remaining, 0.5) if interrupted_at is not None else remaining)
                except queue.Empty:
                    if interrupted_at is not None:
                        self._interrupt()  # a bare ``except:`` in the call may have swallowed the first one
                    continue
                if line is None:
                    self._discard(None if self.closed else "exit")
                    return {
                        "error": "Sandbox closed" if self.closed else RESET_AFTER_EXIT,
                        "stdout": "",
                        "action_results": list(results),
                    }
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._discard("invalid_response")
                    return {
                        "error": f"Sandbox process returned an invalid response; {RESET_AFTER_EXIT}",
                        "stdout": "",
                        "action_results": list(results),
                    }
                kind = str(message.get("type", "")).strip()
                if kind == "action" and interrupted_at is not None:
                    # Sent before the interrupt landed: refused, never executed (the child raises on the reply).
                    self._send({"type": "action_error", "error": "tool timeout: action not executed"})
                    continue
                if kind == "action":
                    try:
                        reply = action_handler(list(message.get("actions") or []))
                    except Exception:  # noqa: BLE001
                        self._send({"type": "action_error", "error": "action failed in sandbox host."})
                        continue
                    raw = reply.get("action_result") or {}
                    if isinstance(raw, dict):
                        results.append(dict(raw))
                    self._send(
                        {
                            "type": "action_result",
                            "action_result": raw,
                            "state": self._delta(reply.get("state") or {}),
                        }
                    )
                    continue
                if kind in {"final", "error"}:
                    output = {
                        "stdout": str(message.get("stdout", "") or ""),
                        "result": message.get("result"),
                        "error": str(message.get("error", "") or ""),
                        "action_results": list(message.get("action_results") or results),
                        "segmentation": message.get("segmentation"),
                        "history_size": message.get("history_size"),
                    }
                    if interrupted_at is not None:
                        # Resend the whole history next time: the child's copy is intact, but a full resync costs
                        # one larger message and rules out a drift after an interrupted call.
                        self.synced = 0
                        if kind == "error":
                            self.interrupts += 1
                            self._event("duck_python_interrupted", interrupts=self.interrupts,
                                        seconds=round(time.monotonic() - interrupted_at, 3))
                            detail = output["error"]
                            output["error"] = (f"Tool timed out after {timeout_seconds}s; {INTERRUPTED_AFTER_TIMEOUT}"
                                               + (f"\n{detail}" if detail else ""))
                            output["interrupted"] = True
                    return output
                self._discard("unknown_message")
                return {
                    "error": f"Sandbox process returned an unknown message type; {RESET_AFTER_EXIT}",
                    "stdout": "",
                    "action_results": list(results),
                }

    def inventory(self, timeout=5.0):
        """The live user definitions (``LOOP_FIXES`` inventory request), or None when there is no interpreter or it
        cannot answer. A child that does not answer within ``timeout`` is interrupted, then discarded."""
        with self.lock:
            if self.closed or self.process is None or not self.inventory_supported:
                return None
            try:
                self._send({"type": "inventory"})
            except OSError:
                self._discard("inventory_failed")
                return None
            deadline = time.monotonic() + timeout
            interrupted = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not interrupted:
                        interrupted = True
                        self._interrupt()
                        deadline = time.monotonic() + self.interrupt_grace
                        continue
                    self._discard("inventory_timeout")
                    return None
                try:
                    line = self.lines.get(timeout=remaining)
                except queue.Empty:
                    continue
                if line is None:
                    self._discard(None if self.closed else "exit")
                    return None
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._discard("invalid_response")
                    return None
                if message.get("type") != "inventory":
                    self._discard("unknown_message")
                    return None
                message.pop("type", None)
                return message

    def close(self):
        """Kill the child (unblocking a call in progress) and drop the working directory."""
        self.closed = True
        process = self.process
        if process is not None:
            _kill(process)
        # A call in progress observes EOF and discards; otherwise clean up here.
        if not self.lock.locked():
            with self.lock:
                self._discard(None)
