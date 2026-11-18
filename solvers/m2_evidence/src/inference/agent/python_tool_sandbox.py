# Modified for M2 basics, 2026-10-01. See solvers/m2_basics/NOTICE.md and candidate.patch.
"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
import logging
import os
import selectors
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from copy import deepcopy
import textwrap
import time
from typing import Any, Callable

from inference.utils import frame_diff as _frame_diff_module
from inference.utils import segmentation as _segmentation
from inference.utils import retained_functions as _retained_functions
from inference.utils.grid_utils import ARC_COLOR_CHARS


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import ast
    import builtins
    import contextlib
    import io
    import json
    import os
    import sys
    import traceback

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""

    __SEGMENTATION_SOURCE__

    __FRAME_DIFF_SOURCE__

    __RETAINED_FUNCTIONS_SOURCE__

    HOST_STDOUT = sys.stdout

    SAFE_MODULES = {
        "bisect",
        "collections",
        "copy",
        # pure-Python sequence comparison, no I/O. Measured across 32
        # transcripts it was imported 9 times and USED none of them - the model
        # reaches for it on "compare two things" and then writes the comparison
        # by hand. Allowing it therefore buys robustness, not capability: a
        # stray import line no longer costs the whole snippet.
        "difflib",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
    SAFE_BUILTINS = {
        # Exception types. Only four were present, so anything narrower than
        # `except Exception` failed at the except clause and took the snippet
        # with it. These are inert classes and expose nothing; SystemExit and
        # KeyboardInterrupt stay out because the sandbox uses them for its own
        # timeout and shutdown handling.
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "EOFError",
        "IndexError",
        "KeyError",
        "LookupError",
        "NameError",
        "NotImplementedError",
        "OverflowError",
        "RecursionError",
        "StopIteration",
        "UnboundLocalError",
        "ZeroDivisionError",
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "zip",
    }


    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None
            self._rows = None
            self._numeric_grid = None
            self._cells = None

        @property
        def rows(self):
            if self._rows is None:
                self._rows = tuple(self.ascii.splitlines())
            return self._rows

        @property
        def grid(self):
            if self._numeric_grid is None:
                self._numeric_grid = tuple(tuple(row) for row in self._grid)
            return self._numeric_grid

        def _index_cells(self):
            if self._cells is None:
                cells = {}
                for r, row in enumerate(self._grid):
                    for c, color in enumerate(row):
                        cells.setdefault(color, []).append((r, c))
                self._cells = {color: tuple(positions) for color, positions in cells.items()}

        @property
        def counts(self):
            self._index_cells()
            return {color: len(cells) for color, cells in self._cells.items()}

        def cells(self, color):
            self._index_cells()
            return self._cells.get(color, ())

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class HistoryEntryView:
        def __init__(self, *, action, frame, result=None, animation=None):
            self.action = action
            self.frame = frame
            self.result = dict(result) if isinstance(result, dict) else {}
            self._animation = dict(animation or {})

        @property
        def step(self):
            return self.frame.step

        @property
        def level(self):
            return self.frame.level

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result, animation=None):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}
            self._animation = dict(animation or {})
            self._frames = None

        @property
        def animation_info(self):
            return {k: v for k, v in self._animation.items() if k != "frames"}

        @property
        def frames(self):
            if self._frames is not None:
                return self._frames
            views = []
            for encoded in self._animation.get("frames", []):
                grid = [[int(ch, 16) for ch in row] for row in encoded.split("/")]
                views.append(FrameView(
                    ascii="\n".join("".join(COLOR_CHARS[v] for v in row) for row in grid),
                    step=self.after_frame.step, level=self.after_frame.level,
                    shape=(len(grid), len(grid[0]) if grid else 0), grid=grid))
            self._frames = views
            return self._frames

        intermediate_frames = frames
        animation_frames = frames

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            ascii=str(payload.get("ascii", "")),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
        )


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=_frame_from_payload(entry.get("frame")),
                    result=entry.get("result"),
                    animation=entry.get("animation"),
                )
            )
        return items


    def _transitions_from_history(history):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result=entry.result,
                    animation=entry._animation,
                )
            )
        return transitions


    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


    def _sanitize_exception(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
        lines.append(f"{exc.__class__.__name__}: {exc}")
        return "\n".join(lines)


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)


    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ):
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def main():
        initial = _recv()
        global COLOR_CHARS
        COLOR_CHARS = str(initial.get("color_chars") or "")
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        _set_limits(timeout_seconds)

        action_results = []
        stdout = io.StringIO()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "result": None,
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import
        _animation_views = []

        def frame_diff(before=None, after=None):
            # Object-level diff between two frames. Defaults: previous_frame -> current_frame.
            # Returns only what changed; unchanged objects are omitted.
            if before is None:
                before = runtime_globals.get("previous_frame")
            if after is None:
                after = runtime_globals.get("current_frame")
            if before is None or after is None:
                return {"error": "need two frames (no previous_frame available?)"}
            if before.shape != after.shape:
                return {"changed_cell_count": -1, "note": "frame shapes differ; full redraw",
                        "before_shape": before.shape, "after_shape": after.shape}
            return compute_frame_diff(
                before._grid,
                after._grid,
                before.segmentation.get("nodes", []),
                after.segmentation.get("nodes", []),
            )

        def _refresh_state(state_payload):
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(state_payload.get("history"))
            last_action_call_result = state_payload.get("last_action_call_result")
            action_result = (
                dict(last_action_call_result) if isinstance(last_action_call_result, dict) else {}
            )
            # action_echo is printed by action(), then removed from its return value.
            # Keep the exposed call result identical across subsequent inspections.
            action_result.pop("action_echo", None)
            transitions = _transitions_from_history(history)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["latest_frame"] = current_frame
            runtime_globals["history"] = history
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["last_action_frame"] = (
                last_transition.after_frame if last_transition is not None else None
            )
            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
            runtime_globals["frame_diff"] = frame_diff
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["death_ledger"] = state_payload.get("death_ledger") or {"attempts": []}
            runtime_globals["last_action_call_result"] = action_result

        # Both refusals derive from BaseException, not Exception, so a
        # defensive `except Exception:` around action(...) - which models write
        # routinely - cannot swallow them. A refusal means the action did not
        # run at all, so the snippet's assumed state is already wrong and every
        # line after it rests on a false premise. One run treated 1500
        # consecutive refusals as proof that every cell on the board was inert.
        # StaleStateActionError below is deliberately catchable: it reports a
        # belief that MAY be stale rather than a verdict on the action.
        class KnownNoOpActionError(BaseException):
            pass  # raised INSTEAD of executing an action already proven inert here

        class KnownDeathActionError(BaseException):
            pass  # raised INSTEAD of executing an action that ended a prior attempt

        # Raised INSTEAD of executing an action this snippet has already run
        # from this exact board. Buggy loop code can spend a whole life bar
        # getting nowhere - a cursor that wraps turns RIGHT into a short cycle,
        # a mistaken `while pos < j: LEFT` walks away from its target forever -
        # and the terminal latch only stops that once the game actually ends,
        # measured at 20 to 30 wasted actions. BaseException like the two above:
        # the shape that produces these loops is `try: action(...) except
        # Exception:` inside a while, which would otherwise swallow the verdict
        # and keep going. No override slot: the snippet is not reading the
        # refusal, its code is running, and writing a new snippet IS the
        # response - the map is rebuilt per execution.
        class RepeatedActionInStateError(BaseException):
            pass

        # Raised when a PREVIOUS action in this snippet changed nothing, and the
        # snippet then tries to act again. Deliberately a plain Exception, not
        # BaseException: unlike the two refusals above this is a warning about
        # the snippet's beliefs rather than a verdict on the action, and a model
        # that has re-read current_frame may legitimately want to proceed.
        class StaleStateActionError(Exception):
            pass

        runtime_globals["KnownNoOpActionError"] = KnownNoOpActionError
        runtime_globals["KnownDeathActionError"] = KnownDeathActionError
        runtime_globals["StaleStateActionError"] = StaleStateActionError
        runtime_globals["RepeatedActionInStateError"] = RepeatedActionInStateError
        # armed when an action changes nothing; scoped to THIS snippet because
        # the closure is rebuilt per execution, so no reset path can be missed
        _stale_state: list[str] = []

        # Raised when a snippet tries to act after the attempt or level ended.
        # Derives from BaseException, not Exception, so a defensive
        # `try: action(...) except Exception:` around the call - which models
        # write routinely - cannot swallow it. Continuing to execute a path
        # computed against a board that no longer exists is never correct.
        class TerminalStateActionError(BaseException):
            pass

        runtime_globals["TerminalStateActionError"] = TerminalStateActionError
        # latched when an action reports a terminal result; scoped to THIS
        # snippet because these closures are rebuilt for every execution, so
        # no reset path can be missed and the next snippet always starts clear
        _terminal_state = []

        _TERMINAL_DETAIL = {
            "game_over": (
                "the previous action reported game_over. The level has been reset to its "
                "start, so the board you planned against is gone and this action was not "
                "executed. The next turn begins from the new state."
            ),
            "level_completed": (
                "the previous action reported level_completed. A new level has been loaded, "
                "so the board you planned against is gone and this action was not executed. "
                "The next turn begins from the new state."
            ),
            "run_complete": (
                "the previous action reported run_complete. The run is finished and this "
                "action was not executed."
            ),
        }

        def action(actions):
            if _terminal_state:
                raise TerminalStateActionError(_TERMINAL_DETAIL[_terminal_state[0]])
            normalized_actions = _normalize_actions(actions)
            _send({
                "type": "action",
                "actions": normalized_actions,
                # the solver decides precedence: a known death or a known no-op
                # is a sharper message than "your state may be stale", and it
                # carries a concrete override
                "stale_after": _stale_state[0] if _stale_state else None,
            })
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            # Cached animations describe the previous action. Invalidate the
            # existing views so aliases also refresh, even if a guard raises.
            for view in _animation_views:
                view.invalidate()
            _refresh_state(reply.get("state") or {})
            if action_result.get("stop_reason") == "reset_not_first":
                raise RuntimeError(str(action_result.get("stop_detail")))
            if action_result.get("stop_reason") == "stale_state":
                raise StaleStateActionError(
                    str(action_result.get("stop_detail") or "stale state")
                )
            if action_result.get("stop_reason") == "known_death":
                raise KnownDeathActionError(
                    str(action_result.get("stop_detail") or "known fatal action was not executed.")
                )
            if action_result.get("stop_reason") == "repeated_action_in_state":
                raise RepeatedActionInStateError(
                    str(action_result.get("stop_detail") or "")
                    or "this action was already run from this board in this snippet"
                )
            if action_result.get("stop_reason") == "known_noop":
                # nothing executed and no action was spent; halt the snippet so
                # the model re-plans instead of continuing on a false premise
                raise KnownNoOpActionError(
                    str(action_result.get("stop_detail") or "known no-op action was not executed.")
                )
            for _flag in ("run_complete", "done", "level_completed", "game_over"):
                if action_result.get(_flag):
                    # `done` means the run was won; report it as run_complete
                    _terminal_state.append(
                        "run_complete" if _flag == "done" else _flag
                    )
                    break
            _echo = action_result.pop("action_echo", None)
            if _echo:
                # into the snippet's own stdout, so a loop can see what its
                # call did without waiting for the next turn's opener
                print(_echo)
            gameplay_changed = action_result.get("gameplay_changed")
            if gameplay_changed is True:
                # a real change means the snippet's beliefs are current again
                del _stale_state[:]
            if gameplay_changed is False and action_result.get("executed"):
                _executed = action_result.get("executed_actions")
                _name = (
                    _executed[-1]
                    if isinstance(_executed, list) and _executed
                    else action_result.get("action_display")
                    or action_result.get("action_name")
                    or "?"
                )
                # Arm rather than raise. The action executed and changed
                # nothing, which is often exactly what the snippet went looking
                # for - it printed the board before, and wants to print it
                # after. Killing it here throws away the second half of the
                # measurement AND the batch stop's own trace, which is computed
                # and then discarded. So let the snippet finish observing, and
                # stop it only if it tries to ACT on the stale belief.
                if not _stale_state:
                    _stale_state.append(str(_name))
            return action_result

        def _animation_fetch(kind):
            _send({"type": "animation", "request": {"kind": kind}})
            reply = _recv()
            if reply.get("type") == "animation_error":
                raise RuntimeError(str(reply.get("error", "animation failed")))
            if reply.get("type") != "animation_result":
                raise RuntimeError("Invalid animation response from sandbox host.")
            payload = reply.get("animation")
            if kind != "frames":
                return payload or {}
            return [
                FrameView(
                    ascii=item.get("ascii", ""),
                    step=item.get("step", 0),
                    level=item.get("level", 0),
                    shape=item.get("shape", (0, 0)),
                    grid=item.get("grid", ()),
                )
                for item in (payload or [])
            ]

        runtime_globals["action"] = action
        def _data_request(operation, **kwargs):
            _send({"type": "data", "operation": operation, **kwargs})
            reply = _recv()
            if reply.get("error"):
                raise ValueError(reply["error"])
            return reply.get("result")

        def save_data(name, value):
            # JSON must be lossless: reject tuples, non-string keys, NaN, etc.
            # Check before transport; json.dumps would silently change some types.
            def check(item):
                if item is None or type(item) in (bool, int, str):
                    return
                if type(item) is float:
                    if item != item or item in (float("inf"), -float("inf")):
                        raise ValueError("Saved data must contain finite numbers")
                    return
                if type(item) is list:
                    for v in item:
                        check(v)
                    return
                if type(item) is dict and all(type(k) is str for k in item):
                    for v in item.values():
                        check(v)
                    return
                raise ValueError("Saved data accepts JSON values only: string keys and lists, not tuples/sets/objects")
            check(value)
            encoded = json.dumps(value, allow_nan=False)
            if len(encoded.encode("utf-8")) > 262144:
                raise ValueError("Saved item exceeds 256 KiB")
            return _data_request("save", name=name, value=value)

        def load_data(name):
            return _data_request("load", name=name)

        def list_data():
            return _data_request("list")

        def delete_data(name):
            return _data_request("delete", name=name)

        def record_evidence(*args, claim=None, prediction=None, test=None, observed=None,
                            interpretation=None, scope="level", revises=None):
            # Append-only: a correction is a new entry naming the one it revises.
            # Keyword-only: in the smoke run the model passed scope positionally,
            # which landed in interpretation and erased it.
            if args:
                raise TypeError("record_evidence takes keyword arguments only: record_evidence("
                                "claim=..., prediction=..., test=..., observed=..., "
                                "interpretation=..., scope='level', revises=None)")
            entry = {"claim": claim, "prediction": prediction, "test": test,
                     "observed": observed, "interpretation": interpretation,
                     "scope": scope, "revises": revises}
            return _data_request("evidence_add", entry=entry)

        def evidence(scope=None, include_revised=False):
            return _data_request("evidence_list", scope=scope,
                                 include_revised=include_revised)

        runtime_globals.update(save_data=save_data, load_data=load_data,
                               list_data=list_data, delete_data=delete_data,
                               record_evidence=record_evidence, evidence=evidence)
        if initial.get("animation_enabled"):
            # Globals rather than attributes on last_transition: after a death
            # the harness appends a RESET entry, so last_transition is the
            # reset and not the action that animated. The record is simply
            # "the last executed action that animated", and naming it that way
            # removes the mismatch. Empty when the last executed action
            # returned a single frame.
            class _LazyAnimation:
                def __init__(self, kind, empty):
                    self._kind = kind
                    self._empty = empty
                    self._value = None

                def _get(self):
                    if self._value is None:
                        self._value = _animation_fetch(self._kind) or self._empty
                    return self._value

                def invalidate(self):
                    self._value = None

                def __len__(self):
                    return len(self._get())

                def __iter__(self):
                    return iter(self._get())

                def __getitem__(self, key):
                    return self._get()[key]

                def __bool__(self):
                    return bool(self._get())

                def __contains__(self, key):
                    return key in self._get()

                def keys(self):
                    return self._get().keys()

                def items(self):
                    return self._get().items()

                def values(self):
                    return self._get().values()

                def get(self, key, default=None):
                    return self._get().get(key, default)

                def __repr__(self):
                    return repr(self._get())

            runtime_globals["last_animation_frames"] = _LazyAnimation("frames", [])
            runtime_globals["last_animation_timeline"] = _LazyAnimation("timeline", {})
            _animation_views.extend((runtime_globals["last_animation_frames"],
                                     runtime_globals["last_animation_timeline"]))

        _refresh_state(initial.get("state") or {})

        _persistence_enabled = bool(initial.get("persistence_enabled"))
        _retention_options = {
            "retain_imports": bool(initial.get("retain_imports")),
            "repair_hints": bool(initial.get("repair_hints")),
        }
        _kept_pool = {}
        _retention_rejected = []
        _available_names = set(runtime_globals) | set(runtime_globals["__builtins__"])
        if _persistence_enabled:
            _before_restore = dict(runtime_globals)
            try:
                _kept_pool, _retention_rejected = restore_functions(
                    initial.get("kept_functions") or [], runtime_globals, _available_names,
                    **_retention_options)
            except Exception:
                # Optional cache maintenance must never prevent this snippet running.
                runtime_globals.clear()
                runtime_globals.update(_before_restore)
                _kept_pool = {}
                _retention_rejected = [{"name": "all", "reason": "restoration failed; redefine needed functions"}]

        def _collect_retention_payload():
            if not _persistence_enabled:
                return {}
            try:
                kept, rejected = collect_functions(
                    _code_text, _kept_pool, runtime_globals, _available_names,
                    **_retention_options)
                return {
                    "keepable_functions": [{"name": n, "source": src} for n, src in kept.items()],
                    "retention_rejected": _retention_rejected + rejected,
                }
            except Exception:
                # Missing snapshot preserves the last committed pool on the host.
                # Never replace the original error/output or replay its actions.
                return {"retention_rejected": [
                    {"name": "all", "reason": "retention bookkeeping failed; previous code preserved"}]}

        try:
            _code_text = str(initial.get("code", ""))
            _tree = ast.parse(_code_text, "<python_tool>")
            _last_expr = None
            if _tree.body and isinstance(_tree.body[-1], ast.Expr):
                _last_expr = ast.Expression(_tree.body[-1].value)
                ast.copy_location(_last_expr, _tree.body[-1])
                _tree.body = _tree.body[:-1]
            with contextlib.redirect_stdout(stdout):
                if _tree.body:
                    exec(compile(_tree, "<python_tool>", "exec"), runtime_globals, runtime_globals)
                if _last_expr is not None:
                    _value = eval(compile(_last_expr, "<python_tool>", "eval"), runtime_globals, runtime_globals)
                    if _value is not None:
                        print(repr(_value))
                        if runtime_globals.get("result") is None:
                            runtime_globals["result"] = _value
            _send(
                {
                    "type": "final",
                    **_collect_retention_payload(),
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                }
            )
        except (
            KnownNoOpActionError,
            RepeatedActionInStateError,
            KnownDeathActionError,
            TerminalStateActionError,
            Exception,
        ) as exc:
            # The three refusals derive from BaseException so a model's own
            # `except Exception:` cannot swallow them - but THIS handler is the
            # same construct, so without naming them explicitly they escaped it
            # too, killed the child process, and the host reported a bare
            # "Sandbox process exited unexpectedly". The refusal messages - the
            # action name, that nothing was spent, how to override - were built
            # and then never delivered. They stop the snippet either way; the
            # point is that the model is told why.
            _send(
                {
                    "type": "error",
                    **_collect_retention_payload(),
                    "error": _sanitize_exception(exc),
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                }
            )


    if __name__ == "__main__":
        main()
    """
).replace("__RETAINED_FUNCTIONS_SOURCE__\n", inspect.getsource(_retained_functions)).replace(
    "__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation)
).replace(
    "__FRAME_DIFF_SOURCE__\n", inspect.getsource(_frame_diff_module)
)


log = logging.getLogger(__name__)


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }


def _wait_for_pipe(selector: selectors.BaseSelector, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not selector.select(timeout=remaining):
        raise TimeoutError

def _send_json_line(handle: Any, payload: dict[str, Any], *, deadline: float) -> None:
    pending = memoryview((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    with selectors.DefaultSelector() as selector:
        selector.register(handle, selectors.EVENT_WRITE)
        while pending:
            _wait_for_pipe(selector, deadline)
            try:
                written = os.write(handle.fileno(), pending)
            except BlockingIOError:
                continue
            pending = pending[written:]

def _read_sandbox_stdout(handle: Any, pending: bytearray, *, deadline: float) -> bytes | None:
    search_from = 0
    with selectors.DefaultSelector() as selector:
        selector.register(handle, selectors.EVENT_READ)
        while True:
            separator = pending.find(b"\n", search_from)
            if separator >= 0:
                line = bytes(pending[:separator])
                del pending[:separator + 1]

                return line
            _wait_for_pipe(selector, deadline)
            try:
                chunk = os.read(handle.fileno(), 65536)
            except BlockingIOError:
                continue
            if not chunk:
                line = bytes(pending) if pending else None
                pending.clear()

                return line
            search_from = len(pending)
            pending.extend(chunk)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[bytes], *, timeout: float = 1.0) -> None:
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    except OSError:
        return

    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


EVIDENCE_KEY = "__evidence__"
EVIDENCE_FIELDS = ("claim", "prediction", "test", "observed", "interpretation")
EVIDENCE_FIELD_CHARS = 400
EVIDENCE_MAX_ENTRIES = 256


def evidence_entries(store):
    """The game's evidence record, oldest first. Lives beside saved data, outside its limits."""
    return store.setdefault(EVIDENCE_KEY, [])


def current_evidence(entries):
    """Entries not superseded by a later revision."""
    revised = {e["revises"] for e in entries if e.get("revises") is not None}
    return [e for e in entries if e["id"] not in revised]


def _evidence_operation(store, request, frame):
    entries = evidence_entries(store)
    if request.get("operation") == "evidence_list":
        scope = request.get("scope")
        if scope not in (None, "level", "game"):
            raise ValueError("evidence(scope=...) accepts None, 'level' or 'game'")
        shown = entries if request.get("include_revised") else current_evidence(entries)
        level = frame.get("level")
        if scope == "level":
            shown = [e for e in shown if e["scope"] == "level" and e["level"] == level]
        elif scope == "game":
            shown = [e for e in shown if e["scope"] == "game"]
        return deepcopy(shown)
    entry = request.get("entry")
    if not isinstance(entry, dict):
        raise ValueError("record_evidence needs claim, prediction, test, observed, interpretation")
    for field in EVIDENCE_FIELDS:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"record_evidence: {field} must be a nonempty string")
        if len(value) > EVIDENCE_FIELD_CHARS:
            raise ValueError(f"record_evidence: {field} exceeds {EVIDENCE_FIELD_CHARS} characters; state it briefly")
    scope = entry.get("scope", "level")
    if scope not in ("level", "game"):
        raise ValueError("record_evidence: scope must be 'level' or 'game'")
    revises = entry.get("revises")
    if revises is not None:
        if type(revises) is not int or not any(e["id"] == revises for e in entries):
            raise ValueError("record_evidence: revises must be the id of an existing entry")
        if any(e.get("revises") == revises for e in entries):
            raise ValueError(f"record_evidence: entry {revises} was already revised; revise its newest revision instead")
    if len(entries) >= EVIDENCE_MAX_ENTRIES:
        raise ValueError(f"Evidence record is full ({EVIDENCE_MAX_ENTRIES} entries)")
    record = {"id": len(entries) + 1, "scope": scope, "level": frame.get("level"),
              "step": frame.get("step"), "revises": revises,
              **{field: entry[field].strip() for field in EVIDENCE_FIELDS}}
    entries.append(record)
    return {"id": record["id"], "level": record["level"], "step": record["step"]}


def _data_operation(store, request, frame):
    """Explicit per-game checkpoints. Validate completely before replacing a key."""
    operation = request.get("operation")
    if operation in ("evidence_add", "evidence_list"):
        return _evidence_operation(store, request, frame)
    if operation == "list":
        return {key: {k: v for k, v in record.items() if k != "value"}
                for key, record in store.items() if key != EVIDENCE_KEY}
    name = request.get("name")
    if not isinstance(name, str) or not name or len(name) > 80:
        raise ValueError("Saved data names must be nonempty strings of at most 80 characters")
    if name == EVIDENCE_KEY:
        raise ValueError("That name is reserved for the evidence record")
    if operation == "load":
        if name not in store:
            raise ValueError("No saved data named " + name)
        return deepcopy(store[name])
    if operation == "delete":
        return store.pop(name, None) is not None
    if operation != "save":
        raise ValueError("Unknown saved data operation")
    value = request.get("value")
    encoded = json.dumps(value, allow_nan=False).encode("utf-8")
    if len(encoded) > 262144:
        raise ValueError("Saved item exceeds 256 KiB")
    saved = {key: item for key, item in store.items() if key != EVIDENCE_KEY}
    if name not in saved and len(saved) >= 64:
        raise ValueError("Saved data has 64 items; delete an unused item first")
    total = sum(item["bytes"] for key, item in saved.items() if key != name) + len(encoded)
    if total > 1048576:
        raise ValueError("Saved data exceeds 1 MiB; delete unused items first")
    record = {"value": json.loads(encoded), "saved_step": frame.get("step"),
              "saved_level": frame.get("level"), "bytes": len(encoded)}
    store[name] = record
    return {k: v for k, v in record.items() if k != "value"}


def run_sandboxed_python(
    *, code: str, timeout_seconds: float, initial_state: dict[str, Any],
    action_handler: Callable[..., dict[str, Any]],
    animation_handler: Callable[[dict[str, Any]], Any] | None = None,
    kept_functions: list[str] | None = None,
    retain_imports: bool = False, repair_hints: bool = False,
    saved_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bound pipe I/O and always reap the child; host receipts are authoritative.

    The synchronous environment callback must enforce its own game deadline.
    Interrupting an in-flight engine mutation in another thread is not safe.
    """
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        data_store = saved_data if saved_data is not None else {}
        observed_frame = initial_state.get("current_frame") or {}
        receipts: list[dict[str, Any]] = []
        def failed(error, **extra):
            # No retention snapshot means rollback to previously committed code.
            return {"error": error, "stdout": "", "action_results": receipts, **extra}
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0, cwd=sandbox_dir, env=_sandbox_env(), start_new_session=True,
            )
        except OSError:
            return failed("Sandbox process could not start.")
        try:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))
            os.set_blocking(process.stdin.fileno(), False)
            os.set_blocking(process.stdout.fileno(), False)
            def send(payload):
                _send_json_line(process.stdin, payload, deadline=deadline)
            send({"code": code, "kept_functions": list(kept_functions or []),
                  "persistence_enabled": kept_functions is not None,
                  "retain_imports": retain_imports, "repair_hints": repair_hints,
                  "animation_enabled": animation_handler is not None,
                  "timeout_seconds": timeout_seconds, "sandbox_cwd": sandbox_dir,
                  "state": initial_state, "color_chars": ARC_COLOR_CHARS})
            pending = bytearray()
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError
                line = _read_sandbox_stdout(process.stdout, pending, deadline=deadline)
                if line is None:
                    return failed("Sandbox process exited unexpectedly.")
                try:
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise ValueError("Expected a protocol object")
                except (ValueError, UnicodeDecodeError):
                    return failed("Sandbox process returned an invalid response.")
                kind = str(message.get("type", "")).strip()
                if kind == "data":
                    try:
                        result = _data_operation(data_store, message, observed_frame)
                    except (ValueError, TypeError, KeyError, RecursionError) as exc:
                        send({"type": "data_result", "error": str(exc)})
                    else:
                        send({"type": "data_result", "result": result})
                    continue
                if kind == "animation":
                    if animation_handler is None:
                        send({"type": "animation_error", "error": "Animation frames are not available in this session."})
                        continue
                    try:
                        payload = animation_handler(dict(message.get("request") or {}))
                    except Exception:
                        send({"type": "animation_error", "error": "animation failed in sandbox host."})
                    else:
                        send({"type": "animation_result", "animation": payload})
                    continue
                if kind == "action":
                    try:
                        payload = action_handler(list(message.get("actions") or []),
                                                 stale_after=message.get("stale_after"))
                    except Exception:
                        send({"type": "action_error", "error": "action failed in sandbox host."})
                        continue
                    receipt = payload.get("action_result") or {}
                    observed_frame = (payload.get("state") or {}).get("current_frame") or observed_frame
                    if isinstance(receipt, dict):
                        owned = deepcopy(receipt)
                        # Already printed inside action(); do not echo it twice.
                        owned.pop("action_echo", None)
                        receipts.append(owned)
                    send({"type": "action_result", "action_result": receipt,
                          "state": payload.get("state") or {}})
                    continue
                if kind in {"final", "error"}:
                    return {"stdout": str(message.get("stdout") or ""),
                            "result": message.get("result"), "error": str(message.get("error") or ""),
                            "action_results": receipts,
                            **({"keepable_functions": list(message["keepable_functions"] or [])}
                               if "keepable_functions" in message else {}),
                            **({"retention_rejected": message.get("retention_rejected", [])}
                               if kept_functions is not None else {})}
                return failed("Sandbox process returned an unknown message type.")
        except TimeoutError:
            return failed(f"Tool timed out after {timeout_seconds}s", timed_out=True)
        except (BrokenPipeError, ConnectionResetError):
            return failed("Sandbox process disconnected before the tool finished.")
        finally:
            _kill_process_group(process)
            _wait_for_process_exit(process)
            # Keep crash diagnostics without ever blocking on an open stderr pipe.
            if process.stderr is not None:
                with suppress(OSError):
                    os.set_blocking(process.stderr.fileno(), False)
                    detail = os.read(process.stderr.fileno(), 4000).decode("utf-8", errors="replace")
                    if detail.strip():
                        log.warning("sandbox stderr: %s", detail)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    with suppress(OSError):
                        stream.close()


def _clip(text, limit=160):
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_evidence_digest(store, level, *, include_previous_level=False, max_chars=4000):
    """Current game-scope and level-scope evidence, newest first, for re-showing after
    context trimming or at a level start. Empty when there is nothing to show."""
    entries = current_evidence((store or {}).get(EVIDENCE_KEY) or [])
    levels = {level, level - 1} if include_previous_level and isinstance(level, int) else {level}
    relevant = [e for e in entries if e["scope"] == "game" or e["level"] in levels]
    lines, used = [], 0
    for e in reversed(relevant):
        where = "game" if e["scope"] == "game" else f"level {e['level']}"
        revises = f", revises E{e['revises']}" if e.get("revises") else ""
        line = (f"- E{e['id']} ({where}{revises}) claim: {_clip(e['claim'])} | predicted: {_clip(e['prediction'])}"
                f" | test: {_clip(e['test'])} | observed: {_clip(e['observed'])} | now: {_clip(e['interpretation'])}")
        if lines and used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return ""
    hidden = len(entries) - len(lines)
    head = ("Your evidence record (current entries, newest first). A failed test rejects that prediction, "
            "not necessarily every version of the idea; revise an entry with record_evidence(..., revises=id).")
    tail = [f"({hidden} other current entries; call evidence() or evidence(include_revised=True).)"] if hidden else []
    return "\n".join([head, *lines, *tail])
