# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ARC-AGI-3 single-agent solver built on the normal nemo-oo TUI agent.

One agent (no team) plays an ARC-AGI-3 game through a file-IPC harness:

- the harness appends game states to ``ipc/states.jsonl``; the agent tails that
  file into a ``game_states`` queue, so each new state wakes ``handle()``;
- the agent analyzes the state in CodeAct (numpy on the grid, persistent helper
  modules) and answers by calling ``submit_actions([...], rationale)``, which
  appends to ``ipc/actions.jsonl`` for the harness to execute.

Two knowledge variants share this base:

- :class:`MemArcSolverAgent` — the nemo-oo **memory system** (``self.memory.*``,
  installed as a skill by the launcher) holds accumulated knowledge, hypotheses
  and reflections;
- :class:`MdArcSolverAgent` — the knowledge-file baseline: knowledge lives in
  ``knowledge/*.md`` files via ``read/append/write_notes``.
"""

from __future__ import annotations

# Module-level imports are visible to LLM-generated CodeAct cells.
import asyncio  # noqa: F401
import collections  # noqa: F401
import itertools  # noqa: F401
import json  # noqa: F401
import math  # noqa: F401
import re  # noqa: F401
from pathlib import Path  # noqa: F401
from types import SimpleNamespace

import numpy as np  # noqa: F401

from nooa import hidden, strategy
from nooa.interactive import InteractiveAgent, RespondReason, RespondResult  # noqa: F401
from nooa.media import Image  # visual grid input (show()n to the LLM)
from nooa.runtime import show  # CodeAct builtin — attach an image to the turn

with hidden:
    from stall_contract import StallReview, read_rows, append
    from integrity_guard import IntegrityGuard
    from prediction_feedback import PredictionFeedback
    from epistemic_critic import (
        actor_evidence as _critic_actor_evidence,
        actor_injection as _critic_actor_injection,
        bind_claim_ref as _critic_bind_claim_ref,
        budget_view as _critic_budget_view,
        commitment_claims as _critic_commitment_claims,
        count_mismatches as _critic_count_mismatches,
        critic_messages as _critic_messages,
        critic_packet as _critic_packet,
        review_trigger as _critic_review_trigger,
        validate as _validate_question,
    )
    from host_contract import atomic_json
    from nooa.context_blocks.formatter import XMLBlockFormatter
    from nooa.context_blocks import Context
    import importlib.util
    import hashlib
    import io
    import os  # module-level only (hidden from cells; cells' os access stays blocked)
    import time
    import resource
    from typing import Annotated, Any

    from PIL import Image as _PILImage  # PNG rendering only

    from nooa.agentdoc import pformat, spec
    from nooa.config import CodeActConfig
    from nooa.interactive import SummarizationConfig, install_summarizer
    from nooa.runtime.channels import Channel, _ChannelReader
    from nooa.runtime.restrictions import (
        DEFAULT_BLOCKED_MODULES,
        RestrictionsConfig,
    )
    from nooa.runtime.sandbox import SandboxConfig
    from nooa.skill_registry import SkillRegistry
    from nooa.storage.markers import nosnapshot
    from nooa.strategies import CodeActStrategy

    # _ReturnResultSignal is the internal signal raised by return_result(); no public
    # re-export exists yet. If nooa ever exposes a public path, update this import.
    from nooa.strategies.codeact import _ReturnResultSignal
    from nooa.strategy_validation import InvariantError

# L2 guard (isolation plan §L2): deny file/network/process modules in generated
# CodeAct cells. The agent's own tools (which use pathlib internally) are
# unaffected — restrictions apply only to LLM-generated code. `open()` on a
# forbidden path is neutralised structurally by L3 (the path is absent from the
# sandbox mount); L2 closes the import-based vectors below.
_ARC_RESTRICTIONS = RestrictionsConfig(
    blocked_modules=DEFAULT_BLOCKED_MODULES
    | frozenset(
        {
            # file/dir access (incl. alternate open() entry points io/codecs/linecache)
            "os",
            "shutil",
            "tempfile",
            "glob",
            "fileinput",
            "pathlib",
            "io",
            "codecs",
            "linecache",
            # C-level file access that bypasses the open() jail
            "sqlite3",
            "dbm",
            "shelve",
            # network
            "urllib",
            "http",
            "requests",
            "httpx",
            "aiohttp",
            "socketserver",
            # process / low-level escape (multiprocessing 'spawn' starts a fresh
            # interpreter WITHOUT this denylist — a hard escape, so block it)
            "ctypes",
            "mmap",
            "pty",
            "fcntl",
            "multiprocessing",
            "concurrent",
            # dynamic import / exec-by-path (partial bypass of the name denylist)
            "importlib",
            "runpy",
            "pickle",
            "marshal",
            "webbrowser",
            # `sys` closes the sys.modules['os'] cached-module bypass (see _scan_cell)
            "sys",
        }
    ),
    # `asyncio` stays allowed (cells use `await self...` / gather); its network
    # ENTRYPOINTS are blocked at the AST layer by _scan_cell below (open_connection
    # / create_connection / …). NB: a module denylist is best-effort — the hard
    # boundary for the filesystem is the external uid-drop (uid_sandbox / run_solver
    # --sandbox drop), which holds regardless of any in-process import trick.
)


def _wait_requires_submission(agent, result, call) -> None:
    """Postcondition on ``handle``: a WAIT turn-result must be backed by a
    submission for the current turn (method-local; not LLM-visible).

    In the 20260716 fleet the agent 13 times ended a turn claiming "submitted N
    actions" while nothing reached ``ipc/actions.jsonl`` (implicit-return bug,
    worker restarts) — and both sides idled the full 900s nudge timer. Raising
    ``InvariantError`` routes the unbacked WAIT through the standard
    validation-retry channel: an immediate same-turn resubmit instead of a stall.
    """
    kind = getattr(result, "kind", None)
    if kind is None and isinstance(result, dict):
        kind = result.get("kind")
    if kind is None or str(kind).upper() != "WAIT":
        return
    state = agent._latest_state()
    # The host may already have published the successor before this method's
    # WAIT postcondition runs. Validate the dispatch that just submitted, not
    # the mutable newest state, or validation retries swallow the next turn.
    turn = getattr(agent, '_active_dispatch_turn', None)
    if turn is None:
        turn = state.get("turn") if state else None
    if turn is None or agent._last_submitted_turn() == turn:
        return
    raise InvariantError(
        f"result kind WAIT, but no actions were submitted for turn {turn} — "
        "nothing reached the harness. Call self.submit_actions([...], rationale) now; "
        "a successful submit ends the turn by itself."
    )


def _arc_cell_config() -> CodeActConfig:
    """Build the per-cell CodeAct config, optionally with the OS sandbox backend.

    Default keeps the historical in-process backend. The protocol supervisor
    enforces ``cell_timeout`` using the per-cell IPC execution record, including
    synchronous loops that block asyncio cancellation. Enable
    the per-cell OS sandbox — where ``cell_timeout`` becomes a *hard* kill and the
    filesystem / network / memory / CPU limits are kernel-enforced (Landlock +
    seccomp + rlimits) — via ``run_solver.py --sandbox-cells`` (or a run_multi
    ``sandbox:`` config block), which set the ``ARC_SANDBOX_*`` env this reads. The
    cell's ``self.*`` tool calls broker back to this live agent, so it needs no
    network of its own.
    """
    base = {
        "restrictions": _ARC_RESTRICTIONS,
        "cell_timeout": 60.0,
        "postconditions": (_wait_requires_submission,),
    }
    if os.environ.get("ARC_SANDBOX_CELLS") != "1":
        return CodeActConfig(**base)
    workspace = os.environ.get("ARC_SANDBOX_WORKSPACE") or os.getcwd()
    return CodeActConfig(
        **base,
        execution_backend="sandbox",
        sandbox=SandboxConfig(
            workspace=workspace,
            network=False,
            max_memory_mb=int(os.environ.get("ARC_SANDBOX_MEM_MB", "4096")),
            max_cpu_seconds=int(os.environ.get("ARC_SANDBOX_CPU_S", "90")),
            # Fail open in the example so a host lacking Landlock/seccomp still runs;
            # production callers should keep the default require=True.
            require=os.environ.get("ARC_SANDBOX_REQUIRE", "0") == "1",
        ),
    )


_ARC_CELL_CONFIG = _arc_cell_config()

# ── Cell guard (B-lite): AST-level block of the escape / network / denylist-bypass
# primitives a generated cell could otherwise use, which a module denylist alone
# cannot stop (__import__('os'), sys.modules['os'], subclass gadgets all reach
# blocked modules). Best-effort defense-in-depth ON TOP of the external uid-drop —
# a determined model can still build names dynamically, so this is a speed bump,
# not a boundary. Applied to every cell via an execute_python guard (see __init__).
_CELL_BANNED_ATTRS = frozenset(
    {
        # host escape tools (network + host FS; no legitimate use for a grid solver)
        "shell",
        "repo",
        "pyp",
        "web",
        "mcp",
        "libwriting",
        "tui_config",
        # denylist-bypass gadgets
        "__subclasses__",
        "__bases__",
        "__base__",
        "__mro__",
        "__globals__",
        "__builtins__",
        "__loader__",
        "__spec__",
        "__code__",
        # asyncio network entrypoints (the path the import denylist misses)
        "open_connection",
        "create_connection",
        "create_server",
        "getaddrinfo",
        "sock_connect",
        "sock_accept",
        "start_server",
        "open_unix_connection",
    }
)
_CELL_BANNED_NAMES = frozenset({"__import__", "eval", "exec", "compile", "breakpoint"})


@hidden
def _scan_cell(code: str) -> str | None:
    """Return the first banned primitive a cell references, or None. AST-based, so
    it ignores matches inside strings/comments (e.g. helper source passed to
    write_helper). Syntax errors fall through to the real executor to report."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _CELL_BANNED_ATTRS:
            return (
                f"self.{node.attr}"
                if node.attr in ("shell", "repo", "pyp", "web", "mcp")
                else node.attr
            )
        if isinstance(node, ast.Name) and node.id in _CELL_BANNED_NAMES:
            return node.id
    return None


_NOTE_NAME = re.compile(r"^[a-z0-9_\-]+\.md$")
_HELPER_NAME = re.compile(r"^[a-z0-9_]+\.py$")
_CLICK = re.compile(r"^CLICK\s+(\d{1,2})\s+(\d{1,2})$")

MAX_ACTIONS_PER_TURN = 20

# Consecutive REJECTED submits with no committed env action before the run aborts.
# See EXPERIMENT-CARD.md Stop: the host gates can starve a run and this is their
# kill switch.
REJECTION_ABORT = 8

# Scribe (ported from tycho/agent/agent.py:652 + prompts/scribe.user.j2): a completed
# level is consolidated before the next is played. After this many refusals at one
# boundary the HOST writes the carry-forward note itself and play resumes — a scribe
# performs the consolidation, it does not merely forbid skipping it.
SCRIBE_MAX_REFUSALS = 2

# ── L1.5 in-process path jail (inspired by beam/agent/repl.py `_safe_open`) ────
# Generated CodeAct cells inherit THIS module's globals, so a module-level `open`
# shadows the builtin for agent-generated code. We route it through a workspace
# jail: reads/writes are allowed ONLY under the registered roots (the agent's own
# workspace + IPC). This closes the vector L2 cannot — a bare `open()` on an
# absolute path to the game source or another run — WITHOUT needing OS namespaces
# (which the L3 sandbox provides where available). io/codecs/linecache (alternate
# open() entry points) are blocked by L2 above so this is the only file gateway.
_REAL_OPEN = open
_JAIL_ROOTS: list[Path] = []


def _register_jail_roots(*roots: Path) -> None:
    """Set the allowlisted roots for the shadowed open (one agent per process)."""
    _JAIL_ROOTS.clear()
    _JAIL_ROOTS.extend(r.resolve() for r in roots)


def _jailed_open(file, mode="r", *args, **kwargs):
    """Path-jailed replacement for builtin open. Only paths under a registered
    root are permitted; anything else raises PermissionError. Falls back to the
    real open when no jail is set (framework internals before agent __init__)."""
    if not _JAIL_ROOTS:
        return _REAL_OPEN(file, mode, *args, **kwargs)
    try:
        resolved = Path(file).resolve()
    except (TypeError, ValueError):
        raise PermissionError(f"open({file!r}) — invalid path") from None
    for root in _JAIL_ROOTS:
        try:
            resolved.relative_to(root)
            return _REAL_OPEN(file, mode, *args, **kwargs)
        except ValueError:
            continue
    raise PermissionError(
        f"open({file!r}) denied — outside the agent workspace. Use "
        f"self.read_notes/read_helper/latest_state/trajectory for data."
    )


open = _jailed_open  # noqa: A001  shadow builtin in this module's exec namespace


def _build_redactions(run_dir: Path) -> list[tuple[str, str]]:
    """Identity substrings to scrub from cell output (isolation §anonymization).

    Cell stdout/stderr can echo absolute paths (tracebacks, helper-import errors,
    reprs) that embed `results/arc_agi_3/…/<game>_<variant>/…` — leaking the
    benchmark + game id into the next prompt. Replace the run-dir path prefixes
    and identity tokens with neutral placeholders (longest-first)."""
    reps: list[tuple[str, str]] = []
    p = run_dir.resolve()
    for _ in range(5):
        # A root prefix matches every slash, including actor-authored prose.
        if p.parent == p:
            break
        reps.append((str(p), "/workspace"))
        if p.name == "arc_agi_3" or p.parent == p:
            break
        p = p.parent
    reps.append(("arc_agi_3", "env"))
    # game/variant tokens from the run-dir name: <ts>_<game>_<variant>[_tag]
    parts = run_dir.name.split("_")
    if len(parts) >= 4:
        game = parts[2]
        reps.append((f"{game}_", "game_"))
        reps.append((game, "game"))
    reps.sort(key=lambda kv: -len(kv[0]))
    return reps


def _redact(text: str, reps: list[tuple[str, str]]) -> str:
    for find, rep in reps:
        if find and find in text:
            text = text.replace(find, rep)
    return text


@hidden
def _request_evidence(messages: list[dict], llm) -> tuple[list[dict], list[dict]]:
    """Freeze the JSON request messages and their rendered tool results.

    Apply the client's cache-control rendering before taking the copy; the
    provider sees that prepared list, not the interceptor's source list. A
    non-JSON message is a qualification failure: silently stringifying it would
    not preserve the prompt that the provider received.
    """
    injection_points = getattr(llm, 'cache_control_injection_points', [])
    prepare = getattr(llm, '_inject_cache_control', None)
    prepared = prepare(messages, injection_points) if prepare is not None else messages

    def without_none(value):
        if isinstance(value, dict):
            return {key: without_none(item) for key, item in value.items()
                    if item is not None}
        if isinstance(value, list):
            return [without_none(item) for item in value]
        return value

    # LiteLLM's OpenAI request serializer excludes null dictionary fields. The
    # local fixture observes that transmitted JSON, so mirror the same lossless
    # normalization rather than claiming the pre-serialization object is exact.
    frozen = json.loads(json.dumps(
        without_none(prepared), ensure_ascii=False, allow_nan=False))
    tool_results = []
    for message in frozen:
        if not isinstance(message, dict):
            continue
        content = message.get('content')
        if (message.get('role') == 'tool'
                or (message.get('role') == 'user' and isinstance(content, str)
                    and 'PythonOutput(tool_call_id=' in content)):
            tool_results.append(message)
    return frozen, tool_results


# L1 guard (isolation plan §L1): helper modules are importlib-loaded, so their
# module-level code runs OUTSIDE the CodeAct cell sandbox — the one in-process
# path that could `import urllib`/`open(...)` a game-source file. Restrict helper
# source to pure computation: only these imports, no file/network/process/dunder.
_HELPER_IMPORT_ALLOW = frozenset(
    {
        "numpy",
        "np",
        "math",
        "json",
        "re",
        "collections",
        "itertools",
        "statistics",
        "functools",
        "operator",
        "heapq",
        "bisect",
        "dataclasses",
        "typing",
        "enum",
        "fractions",
        "copy",
    }
)
_HELPER_BANNED_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "memoryview",
        "getattr",
        "setattr",
        "delattr",
    }
)
_HELPER_BANNED_DUNDERS = frozenset(
    {
        "__globals__",
        "__builtins__",
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__class__",
        "__dict__",
        "__getattribute__",
        "__import__",
        "__loader__",
        "__code__",
        "__closure__",
        "__self__",
        "__module__",
        "__base__",
    }
)


def _check_helper_ast(source: str) -> str | None:
    """Return an error string if the helper source uses anything beyond pure
    computation (disallowed import / open / dunder / banned builtin), else None."""
    import ast
    import textwrap

    # Models routinely paste uniformly indented source (e.g. lifted out of a
    # docstring or an if-block); that's valid code — dedent before parsing
    # instead of bouncing it with a spurious IndentationError (~220 wasted
    # rounds across the 20260716 fleet).
    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"syntax error — {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top not in _HELPER_IMPORT_ALLOW:
                    return f"import {a.name!r} not allowed (helpers are pure-compute only)"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in _HELPER_IMPORT_ALLOW:
                return f"import from {node.module!r} not allowed (helpers are pure-compute only)"
        elif isinstance(node, ast.Name) and node.id in _HELPER_BANNED_NAMES:
            return f"use of {node.id!r} not allowed in a helper module"
        elif isinstance(node, ast.Attribute) and node.attr in _HELPER_BANNED_DUNDERS:
            return f"attribute {node.attr!r} not allowed in a helper module"
    return None


@hidden
async def _tail_from_start(path: str, poll_interval: float = 0.5):
    """Like runtime.producers.tail() but starts at position 0, not EOF.

    The harness may write state 0 between the agent_ready marker and the tail
    task's first run — seeking to EOF (as producers.tail does) would silently
    drop it and deadlock the turn loop. Reading from the start makes the
    attach timing irrelevant (the file is always fresh per run).
    """
    fh = open(path)
    try:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if line.endswith("\n"):
                if line.strip():
                    yield line.rstrip("\n")
            else:
                fh.seek(pos)  # partial or no line — wait for the writer
                await asyncio.sleep(poll_interval)
    finally:
        fh.close()


@hidden
def _effort_for_elapsed(elapsed: float, ladder: list[tuple[float, str]]) -> str | None:
    """Pick the reasoning effort for a turn that has been running ``elapsed`` seconds.

    ``ladder`` is ascending ``[(after_seconds, effort), …]`` (rung 0 is the
    primary effort the client was built with). Returns the effort of the last
    rung whose threshold ``elapsed`` has crossed — so a turn spends its later
    minutes at a cheaper effort. Mirrors beam's ``REASONING_EFFORT_LEVELS``
    ladder, but stepped by wall-clock rather than by retry count.
    """
    effort = ladder[0][1] if ladder else None
    for after_s, level in ladder:
        if elapsed >= after_s:
            effort = level
        else:
            break
    return effort


# The environment's canonical 16-color palette (index -> RGB), the same colors the
# reference renders grid PNGs with. Hidden from exec_globals: the name is only used
# by the @hidden renderer below and must not surface on the agent's tool surface.
with hidden:
    _PALETTE_16 = (
        (255, 255, 255),
        (204, 204, 204),
        (153, 153, 153),
        (102, 102, 102),
        (51, 51, 51),
        (0, 0, 0),
        (229, 58, 163),
        (255, 123, 204),
        (249, 60, 49),
        (30, 147, 255),
        (136, 216, 241),
        (255, 220, 0),
        (255, 133, 27),
        (146, 18, 49),
        (79, 204, 48),
        (163, 86, 214),
    )


@hidden
def _grid_png_bytes(grid, scale: int) -> bytes:
    """Render a 2D palette-index grid to PNG bytes at ``scale``x — a
    (scale*H)x(scale*W) RGB image (out-of-range colors -> black, index 5),
    matching image_artifacts.grid_to_rgb_array. For a 64x64 grid at scale s the
    PNG is (s*64)x(s*64)."""
    s = max(int(scale or 1), 1)
    arr = np.asarray(grid, dtype=np.int16)
    palette = np.asarray(_PALETTE_16, dtype=np.uint8)
    safe = np.where((0 <= arr) & (arr < len(palette)), arr, 5).astype(np.intp)
    rgb = palette[safe]
    if s > 1:
        rgb = np.repeat(np.repeat(rgb, s, axis=0), s, axis=1)
    buf = io.BytesIO()
    _PILImage.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class _StateAccess:
    """Game notifications belong to the dispatcher; Actor reads are synchronous."""

    def get(self, *args, **kwargs):
        raise RuntimeError(
            "Do not consume the dispatcher queue. Use self.latest_state() for the "
            "current observation or self.trajectory() for executed history; neither needs await."
        )


class ArcSolverBase(InteractiveAgent):
    """You are an interactive grid-game-solving agent. You discover the unknown rules of an
    interactive 64x64 grid game by experimenting, then solve all its levels.
    Follow the <arc_skill> context block: observe with code (never squint), and
    explore without inventing a goal or competing hypotheses. Keep partial models
    with tested scope and open conditions. End a turn by submitting actions.
    """

    # Producer-side channel (hidden) + LLM-facing reader for harness states.
    _game_states_in: Annotated[Channel, hidden, nosnapshot]
    game_states: Annotated[_StateAccess, hidden, nosnapshot]
    # Namespace of loaded helper modules: after load_helpers(), call e.g.
    # self.h.world_model.predict(...). HIDDEN from the rendered <state> block:
    # a loaded module's repr is "<module 'x' from '/abs/run_dir/.../x.py'>", and
    # the run-dir path embeds the game code + benchmark name — an identity leak
    # into the agent's prompt every turn once helpers are loaded. `nosnapshot`
    # alone did NOT hide it from state; the agent still learns about self.h.<name>
    # from load_helpers()'s report (which prints no path) and the method docs.
    h: Annotated[SimpleNamespace, hidden, nosnapshot]
    skills: Annotated[SkillRegistry, nosnapshot]
    # Filesystem paths are hidden from the LLM surface: the run dir embeds
    # `results/arc_agi_3/…/<game>_<variant>/`, which would leak the benchmark
    # name and the real game id into the rendered <self>/<state> blocks.
    run_dir: Annotated[Path, hidden]
    workspace: Annotated[Path, hidden]
    helpers_dir: Annotated[Path, hidden]
    ipc_dir: Annotated[Path, hidden]

    _stall: Annotated[StallReview, hidden]
    _integrity: Annotated[IntegrityGuard, hidden]
    _model_feedback: Annotated[PredictionFeedback, hidden, nosnapshot]
    _search_results: Annotated[dict, hidden, nosnapshot]
    _tool_seconds_since_model: Annotated[float, hidden, nosnapshot]
    _deep_investigations: Annotated[int, hidden, nosnapshot]
    _critic_actor_calls: Annotated[int, hidden, nosnapshot]
    _critic_reviews: Annotated[int, hidden, nosnapshot]
    _objections: Annotated[Any, hidden, nosnapshot]
    _critic_last_actor_calls: Annotated[int, hidden, nosnapshot]
    _critic_last_step: Annotated[int, hidden, nosnapshot]
    _critic_last_levels_completed: Annotated[int, hidden, nosnapshot]
    _critic_last_mismatch_count: Annotated[int, hidden, nosnapshot]
    _critic_pending_sequence_length: Annotated[int, hidden, nosnapshot]
    _critic_pending_sequence_step: Annotated[int | None, hidden, nosnapshot]

    def __init__(
        self,
        llm=None,
        *,
        run_dir: str | Path,
        game_id: str = "",
        alias: str = "",
        reflect_every: int = 8,
        skill_path: str | Path | None = None,
        effort_ladder: list[tuple[float, str]] | None = None,
        visual: str = "off",
        png_scale: int = 8,
        max_actions_per_turn: int = MAX_ACTIONS_PER_TURN,
        enable_summarizer: bool = True,
        **kwargs,
    ):
        super().__init__(llm=llm, **kwargs)
        # Per-turn action batch cap enforced by submit_actions. <=0 disables the
        # cap entirely (the harness must be launched with the same setting so its
        # backstop truncation is off too — run_multi forwards one config key to both).
        self._max_actions_per_turn = int(max_actions_per_turn)
        # Visual input mode (off|only|additive): 'only'/'additive' auto-show the
        # current settled grid to the LLM each turn as a color PNG
        # ((png_scale*64)x(png_scale*64), 16-color palette) — the picture the
        # reference solver sends, last grid only (no event frames). 'only' also drops the hex
        # grid_rows from the state (harness side) so the image REPLACES the grid;
        # 'additive' keeps both.
        self._visual = str(visual or "off")
        self._png_scale = int(png_scale)
        # Effort ladder: a turn's reasoning_effort steps DOWN this ascending
        # [(after_seconds, effort), …] ladder (rung 0 = the client's primary
        # effort) as its wall-clock grows. The clock is "time since we last
        # submitted an action batch" — the same silence the harness times for its
        # nudge/kill — so a slow multi-call turn spends its later minutes cheaper.
        self._turn_started_at = time.monotonic()
        # Whole-game clock, used to price the critic's questions in seconds per
        # executed action rather than leaving affordability implicit.
        self._game_started_at = self._turn_started_at
        self._effort_ladder: list[tuple[float, str]] = sorted(effort_ladder or [])
        self._ladder_active_effort: str | None = (
            self._effort_ladder[0][1] if self._effort_ladder else None
        )
        self._ladder_downshifts = 0
        self.run_dir = Path(run_dir)
        # game_id is the real id (kept off the agent surface). The agent only
        # ever sees `alias` — an opaque per-run handle carrying no game identity
        # — so it cannot recognise the game and recall training-data knowledge.
        self.game_id = game_id
        self.alias = alias or "the game"
        self.reflect_every = reflect_every
        self.workspace = self.run_dir / "team_nemo" / "shared"
        self.helpers_dir = self.workspace / "helpers"
        self.ipc_dir = self.run_dir / "ipc"
        self._states_path = self.ipc_dir / "states.jsonl"
        self._actions_path = self.ipc_dir / "actions.jsonl"
        self._observations_path = self.ipc_dir / "observations.json"
        self._rejections_path = self.ipc_dir / "rejections.jsonl"
        self._tool_seconds_since_model = 0.0
        self._deep_investigations = 0
        self._search_results = {}
        self._critic_actor_calls = 0
        self._critic_reviews = 0
        from objection_ledger import ObjectionLedger
        # Host-owned pending-objection state. Without an owner the 2026-09-12 run
        # re-derived one S5I5 objection five times at 93s of pure repetition.
        self._objections = ObjectionLedger()
        self._critic_last_actor_calls = 0
        self._critic_last_step = 0
        self._critic_last_levels_completed = 0
        self._critic_last_mismatch_count = 0
        self._critic_pending_sequence_length = 0
        self._critic_pending_sequence_step = None
        for d in (self.workspace, self.helpers_dir, self.ipc_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._states_path.touch()
        self._model_feedback = PredictionFeedback(self.ipc_dir, self.helpers_dir, _check_helper_ast)
        self.context["prediction_workflow"] = Context((
            "REPAIR THEN TEST: When prediction_feedback reports a mismatch, your next "
            "goal is a small executable repair and a fresh test, not reconstructing the whole game. "
            "Within the next THREE execute_python calls: inspect only the needed helper and "
            "failing transition, save the smallest defensible correction, and submit ONE test action. "
            "Combine these operations in one cell when possible. Reuse measurements already computed; "
            "do not repeatedly print unchanged trajectory summaries. Correct an independently "
            "disproven part now rather than waiting to explain every other discrepancy. "
            "Keep unresolved parts explicitly provisional. If no repair is defensible yet, "
            "record what remains unknown and take one discriminating action instead. "
            "Never hardcode one observed frame just to make a check pass. "
            "A partial model is useful when its limits guide the next experiment."
        ), prefix=True)
        self.context["epistemic_contract"] = Context((
            "LOCAL COMMITMENT: You may take one exploratory action without a goal, helper, "
            "or hypothesis pair. State the observation sought, not a fabricated prediction. "
            "Use set_experiment only for real alternatives with different observable predictions; "
            "none may match, and confounders may leave the interpretation unresolved. "
            "Keep observation, mechanics, objective, and conditional plan distinct. A matching "
            "prediction supports its measured consequence in those tested conditions only. "
            "A level transition demonstrates that an executed sequence succeeded in that starting "
            "state, not that every step was necessary or every explanation correct. "
            "Resolve uncertainty when plausible alternatives change the next commitment; "
            "otherwise retain it and act. Carry evidence step references, tested scope, and "
            "open conditions in existing notes; summaries must preserve qualifications. "
            "Unexpected outcomes can introduce new hypotheses. Mismatch challenges the state "
            "reading, measurement and dynamics jointly; test before attributing it. "
            "Search failure inside a partial model does not disprove a game objective. "
            "A spending deadline may stop a policy without making its hypothesis false. "
            "No global mechanics-solved claim: only locally tested rules."
        ), prefix=True)
        flag = os.environ.get("ARC_STALL_REVIEW", "0")
        if flag not in ("0", "1"):
            raise ValueError("ARC_STALL_REVIEW must be 0 or 1")
        self._stall = StallReview(self.ipc_dir, treatment=flag == "1")
        self._actions_path.touch()

        # Jail the shadowed open() to this agent's own workspace + IPC, so a
        # generated cell cannot open the game source or another run (L1.5).
        _register_jail_roots(self.workspace, self.ipc_dir)

        # Scrub identity-bearing paths from cell output before the agent sees it:
        # tracebacks/reprs echo the run-dir absolute path, which embeds
        # arc_agi_3 + the game id. Redact them in an execute_python middleware.
        self._redactions = _build_redactions(self.run_dir)
        self._integrity = IntegrityGuard(self.ipc_dir, getattr(self, "_block_formatter", None) or XMLBlockFormatter())
        self.event_manager.on("PythonOutput", self._integrity.output)

        async def _redact_cell_output(ctx, nxt):
            self._integrity.require_clear()
            ctx = await nxt(ctx)
            r = getattr(ctx, "result", None)
            if r is not None:
                upd = {}
                for field in ("stdout", "stderr", "error"):
                    val = getattr(r, field, None)
                    if isinstance(val, str) and val:
                        red = _redact(val, self._redactions)
                        if red != val:
                            upd[field] = red
                self._integrity.capture(ctx.params.get("tool_call_id"), r.stdout, upd.get("stdout", r.stdout))
                if upd:
                    ctx.result = r.model_copy(update=upd)
            return ctx

        self.event_manager.intercept("execute_python", _redact_cell_output)

        # Cell guard (B-lite): if a cell reaches for an escape tool, a network
        # entrypoint, or a denylist-bypass gadget, replace the cell with a refusal
        # so the model gets feedback and nothing runs. No runtime network patching
        # (that would also block concurrent background reflection LLM calls) — the
        # block is purely at the cell-code layer. Defense-in-depth over the uid-drop.
        async def _guard_cell(ctx, nxt):
            started = time.monotonic()
            execution = {
                'tool_call_id': ctx.params.get('tool_call_id'),
                'started_at_monotonic': started,
                'timeout_seconds': _ARC_CELL_CONFIG.cell_timeout,
                'finished_at_monotonic': None,
            }
            atomic_json(self.ipc_dir / 'cell_execution.json', execution)
            code = getattr(ctx, "code", None)
            if isinstance(code, str):
                bad = _scan_cell(code)
                if bad:
                    ctx.code = (
                        f"print('BLOCKED: {bad} is not permitted in agent cells "
                        "(host tool / network / bypass primitive). Use self.memory, "
                        "the grid methods and numpy only.')"
                    )
            try:
                return await nxt(ctx)
            finally:
                finished = time.monotonic()
                elapsed = finished - started
                execution['finished_at_monotonic'] = finished
                atomic_json(self.ipc_dir / 'cell_execution.json', execution)
                if elapsed >= execution['timeout_seconds']:
                    (self.ipc_dir / 'ABORT').write_text('Python cell timed out')
                self._tool_seconds_since_model += elapsed
                append(self.ipc_dir / "tool_calls.jsonl", {
                    "turn": (self._latest_state() or {}).get("turn"),
                    "step": (self._latest_state() or {}).get("step"),
                    "tool": "execute_python", "wall_seconds": elapsed,
                })

        self.event_manager.intercept("execute_python", _guard_cell)

        self.h = SimpleNamespace()

        # Defense in depth: the framework renders the <state> block as
        # pformat(self, ...), which bypasses the cell-output redaction above — a
        # value repr (e.g. a loaded module's path, a stray Path) can re-expose the
        # run-dir path (game code + benchmark). Override the protected block with a
        # redacted renderer so ANY such leak is scrubbed, not just today's known one.
        self.context_manager.set_dynamic_protected("state", "self._render_state()")

        # Harness states arrive on their own queue: tail the JSONL file the
        # harness appends to, so every new state wakes the dispatcher.
        self._game_states_in = self.queue_manager.queue("game_states")
        self.game_states = _StateAccess()
        self.queue_manager.spawn(
            _tail_from_start(str(self._states_path)),
            channel="game_states",
            label="game-state tail",
        )

        # Skill registry (bootstrap and the memory skill require it) + LLM
        # access to context/events, mirroring TUIAgent.
        self.skills = SkillRegistry(self)
        spec(self, "context", hidden=False)
        spec(self, "events", hidden=False)
        # Long games overflow the context window — compress older history.
        # Compact before the Actor grows into the expensive long-prefill regime.
        # preserve_recent keeps the live CodeAct tail; target_chars bounds the
        # summary itself. The resulting total tokens are measured, not assumed.
        if enable_summarizer:
            install_summarizer(SummarizationConfig(
                max_tokens=19_000, preserve_recent=6, target_chars=6_000
            ), agent=self)

        if skill_path is not None:
            self.context["arc_skill"] = Context(Path(skill_path).read_text(), prefix=True)
        self.context["arc_game"] = Context(expr="self.format_game_context()")

        # Step reasoning_effort down the ladder as a turn runs long (wraps
        # self._llm.acall — must come after super().__init__ set self._llm).
        self._install_effort_ladder()

        def _guard(*, summarizer: bool):
            async def _guarded_llm(ctx, nxt):
                if summarizer:
                    self._integrity.check_summary_messages(ctx.messages, ctx.agent)
                else:
                    active_outputs = {
                        event.tool_call_id: event.tag for tag in self.event_manager.keys()
                        if type(event := self.event_manager[tag]).__name__ == "PythonOutput"
                    }
                    self._integrity.check_messages(ctx.messages, active_outputs)
                started = time.monotonic()
                if (self.ipc_dir / 'request_failure.json').exists():
                    # Failure is terminal for this run, even if NOOA catches
                    # the prior exception and attempts another model call.
                    await asyncio.Future()
                deadline = float(os.environ.get('ARC_GAME_MODEL_ADMISSION_DEADLINE', 'inf'))
                if started >= deadline:
                    # No HTTP request is active here. Park until the host has
                    # acknowledged exhaustion and finishes this game normally.
                    (self.ipc_dir / 'budget_exhausted.json').write_text(json.dumps({
                        'reason': 'model_admission_budget', 'at_monotonic': started,
                        'admission_deadline': deadline, 'active_requests': 0,
                        'worker_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    }))
                    await asyncio.Future()
                request_id = time.monotonic_ns()
                append(self.ipc_dir / 'request_lifecycle.jsonl', {
                    'request_id': request_id, 'event': 'admitted', 'at_monotonic': started,
                    'admission_deadline': deadline if math.isfinite(deadline) else None,
                    'worker_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                })
                try:
                    if not summarizer:
                        state_before = self._latest_state() or {}
                        step_before = int(state_before.get('step') or 0)
                        levels_before = int(state_before.get('levels_completed') or 0)
                        step_checks = read_rows(self.ipc_dir / 'step_checks.jsonl')
                        mismatch_count = _critic_count_mismatches(step_checks)
                        review_trigger = _critic_review_trigger(
                            actor_calls=self._critic_actor_calls,
                            step=step_before,
                            last_actor_calls=self._critic_last_actor_calls,
                            last_step=self._critic_last_step,
                            levels_completed=levels_before,
                            last_levels_completed=self._critic_last_levels_completed,
                            mismatch_count=mismatch_count,
                            last_mismatch_count=self._critic_last_mismatch_count,
                            pending_sequence_length=self._critic_pending_sequence_length,
                            pending_sequence_step=self._critic_pending_sequence_step,
                        )
                        wall_remaining = (max(0.0, deadline - time.monotonic())
                                          if math.isfinite(deadline) else None)
                        spent_probe = self._objections.probe_due(step_before)
                        if spent_probe is not None:
                            from objection_ledger import probe_guidance
                            append(self.ipc_dir / 'objections.jsonl', {
                                'event': 'probe_exhausted', 'step': step_before, **spent_probe})
                            ctx.messages.append({'role': 'user', 'content': probe_guidance(spent_probe)})
                            self._integrity.check_messages(ctx.messages, active_outputs)
                        if review_trigger is not None and not self._objections.accepts_review():
                            # A pending objection makes a second model-generated
                            # variant of it pure cost. Restate it locally instead.
                            pending = self._objections.restate()
                            append(self.ipc_dir / 'critic_reviews.jsonl', {
                                'review_id': pending['review_id'], 'status': 'restated',
                                'review_trigger': review_trigger, 'reminders': pending['reminders'],
                                'turn': state_before.get('turn'), 'step': step_before,
                            })
                            self._critic_last_actor_calls = self._critic_actor_calls
                            self._critic_last_step = step_before
                            self._critic_last_levels_completed = levels_before
                            self._critic_last_mismatch_count = mismatch_count
                            self._critic_pending_sequence_length = 0
                            self._critic_pending_sequence_step = None
                            from objection_ledger import restatement_injection
                            ctx.messages.append(restatement_injection(pending))
                            self._integrity.check_messages(ctx.messages, active_outputs)
                            review_trigger = None
                        if review_trigger is not None:
                            self._critic_reviews += 1
                            review_id = f'critic:{self._critic_reviews}'
                            critic_request_id = time.monotonic_ns()
                            critic_started = time.monotonic()
                            append(self.ipc_dir / 'request_lifecycle.jsonl', {
                                'request_id': critic_request_id, 'event': 'admitted',
                                'request_purpose': 'critic', 'review_id': review_id,
                                'review_trigger': review_trigger,
                                'at_monotonic': critic_started,
                                'admission_deadline': deadline if math.isfinite(deadline) else None,
                                'worker_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            })
                            critic_request_messages = _critic_messages(
                                self._critic_packet_for(
                                    ctx.messages, review_trigger, state_before,
                                    wall_remaining))
                            critic_messages, critic_tool_results = _request_evidence(
                                critic_request_messages, self._llm)
                            try:
                                critic_response = await self._llm.acall(
                                    critic_request_messages,
                                    tools=None,
                                    max_tokens=192,
                                    prompt_cache_key=f'{self._agent_id}-epistemic-critic',
                                )
                            except BaseException as error:
                                append(self.ipc_dir / 'request_lifecycle.jsonl', {
                                    'request_id': critic_request_id, 'event': 'failed',
                                    'request_purpose': 'critic', 'review_id': review_id,
                                    'error': type(error).__name__, 'at_monotonic': time.monotonic(),
                                })
                                raise
                            critic_finished = time.monotonic()
                            append(self.ipc_dir / 'request_lifecycle.jsonl', {
                                'request_id': critic_request_id, 'event': 'completed',
                                'request_purpose': 'critic', 'review_id': review_id,
                                'at_monotonic': critic_finished,
                            })
                            critic_usage = getattr(critic_response, 'usage', None) or {}
                            critic_prompt_details = critic_usage.get('prompt_tokens_details') or {}
                            critic_details = critic_usage.get('completion_tokens_details') or {}
                            critic_reasoning = critic_usage.get('reasoning_tokens', 0)
                            if isinstance(critic_details, dict):
                                critic_reasoning = max(
                                    critic_reasoning,
                                    critic_details.get('reasoning_tokens', 0),
                                )
                            critic_content = str(getattr(critic_response, 'content', '') or '')
                            append(self.ipc_dir / 'model_calls.jsonl', {
                                'request_id': critic_request_id,
                                'turn': state_before.get('turn'), 'step': state_before.get('step'),
                                'wall_seconds': critic_finished - critic_started,
                                'request_purpose': 'critic', 'review_id': review_id,
                                'review_trigger': review_trigger,
                                'prompt_tokens': critic_usage.get('prompt_tokens'),
                                'cached_prompt_tokens': (
                                    critic_prompt_details.get('cached_tokens')
                                    if isinstance(critic_prompt_details, dict) else None
                                ),
                                'completion_tokens': critic_usage.get('completion_tokens'),
                                'usage': critic_usage, 'reasoning_tokens': critic_reasoning,
                                'reasoning_present': bool(str(
                                    getattr(critic_response, 'reasoning', None) or ''
                                ).strip()),
                                'finish_reason': getattr(critic_response, 'finish_reason', None),
                                'request_messages': critic_messages,
                                'tool_results': critic_tool_results,
                                'response_content': critic_content,
                                'tool_calls': [
                                    {'name': call.name, 'arguments': call.arguments}
                                    for call in (getattr(critic_response, 'tool_calls', None) or [])
                                ],
                                'summarizer': False,
                            })
                            try:
                                if getattr(critic_response, 'finish_reason', None) != 'stop':
                                    raise ValueError('critic did not finish with stop')
                                if getattr(critic_response, 'tool_calls', None):
                                    raise ValueError('critic attempted a tool call')
                                question = _validate_question(critic_content)
                                evidence = (_critic_actor_evidence(ctx.messages)
                                            + _critic_commitment_claims(
                                                read_rows(self.ipc_dir / 'actions.jsonl')))
                                binding = _critic_bind_claim_ref(question, evidence)
                            except ValueError as error:
                                append(self.ipc_dir / 'critic_reviews.jsonl', {
                                    'review_id': review_id, 'request_id': critic_request_id,
                                    'status': 'invalid_output', 'error': str(error),
                                    'review_trigger': review_trigger,
                                    'turn': state_before.get('turn'), 'step': state_before.get('step'),
                                })
                            else:
                                # Only a completed, source-valid review consumes its
                                # boundary. Malformed output leaves the trigger live.
                                self._critic_last_actor_calls = self._critic_actor_calls
                                self._critic_last_step = step_before
                                self._critic_last_levels_completed = levels_before
                                self._critic_last_mismatch_count = mismatch_count
                                self._critic_pending_sequence_length = 0
                                self._critic_pending_sequence_step = None
                                append(self.ipc_dir / 'critic_reviews.jsonl', {
                                    'review_id': review_id, 'request_id': critic_request_id,
                                    'status': 'completed', 'question': question,
                                    'source_binding': binding,
                                    'review_trigger': review_trigger,
                                    'turn': state_before.get('turn'), 'step': state_before.get('step'),
                                })
                                if question['question']:
                                    self._objections.open(
                                        review_id=review_id, question=question,
                                        step=step_before, wall_remaining=wall_remaining)
                                    ctx.messages.append(_critic_actor_injection(review_id, question))
                                    self._integrity.check_messages(ctx.messages, active_outputs)
                    request_messages, tool_results = _request_evidence(ctx.messages, self._llm)
                    ctx = await nxt(ctx)
                except BaseException as error:
                    append(self.ipc_dir / 'request_lifecycle.jsonl', {
                        'request_id': request_id, 'event': 'failed',
                        'error': type(error).__name__, 'at_monotonic': time.monotonic(),
                    })
                    (self.ipc_dir / 'request_failure.json').write_text(json.dumps({
                        'request_id': request_id, 'error_type': type(error).__name__,
                        'at_monotonic': time.monotonic(),
                    }))
                    raise
                append(self.ipc_dir / 'request_lifecycle.jsonl', {
                    'request_id': request_id, 'event': 'completed', 'at_monotonic': time.monotonic(),
                })
                response = ctx.response
                usage = getattr(response, "usage", None) or {}
                prompt_details = usage.get("prompt_tokens_details") or {}
                cached_tokens = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
                reasoning = usage.get("reasoning_tokens", 0)
                details = usage.get("completion_tokens_details") or {}
                if isinstance(details, dict):
                    reasoning = max(reasoning, details.get("reasoning_tokens", 0))
                # A thinking-off turn can still emit an empty <think> block, whose
                # whitespace is truthy. Strip before recording presence.
                reasoning_present = bool(str(getattr(response, "reasoning", None) or "").strip())
                state = self._latest_state() or {}
                append(self.ipc_dir / "model_calls.jsonl", {
                    "request_id": request_id,
                    "turn": state.get("turn"), "step": state.get("step"),
                    "wall_seconds": time.monotonic() - started,
                    "request_purpose": "summarizer" if summarizer else "actor",
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_prompt_tokens": cached_tokens,
                    "completion_tokens": usage.get("completion_tokens"),
                    "ttft_seconds": None,
                    "ttft_status": "unavailable_nonstreaming",
                    "tool_seconds_since_previous_request": self._tool_seconds_since_model,
                    "usage": usage, "reasoning_tokens": reasoning,
                    "reasoning_present": reasoning_present,
                    "finish_reason": getattr(response, "finish_reason", None),
                    "request_messages": request_messages,
                    "tool_results": tool_results,
                    "response_content": str(getattr(response, "content", "")),
                    "tool_calls": [{"name": call.name, "arguments": call.arguments}
                                   for call in (getattr(response, "tool_calls", None) or [])],
                    "summarizer": summarizer,
                })
                self._tool_seconds_since_model = 0.0
                if not summarizer:
                    self._critic_actor_calls += 1
                return ctx

            return _guarded_llm

        self.event_manager.intercept("llm_call", _guard(summarizer=False))
        # install_summarizer builds a separate Agent that inherits this LLM but
        # owns its own event manager, so context compression reaches SGLang
        # without passing the interceptor above. Left alone it writes no
        # model_calls record — breaking the one-request-per-recorded-call
        # reconciliation. Record reported reasoning counts separately from text
        # presence; never invent a token count when the provider omits usage.
        for summarizer in getattr(self, "_summarizers", ()):
            summarizer.event_manager.intercept("agent_call", self._integrity.guard_summary_call)
            summarizer.event_manager.intercept("llm_call", _guard(summarizer=True))

        # tail() starts at end-of-file — the harness must not write state 0
        # before the tail producer above is attached. This marker releases it.
        (self.ipc_dir / "agent_ready").write_text(str(time.time()))

    @hidden
    def _install_effort_ladder(self) -> None:
        """Wrap ``self._llm.acall`` so each call's ``reasoning_effort`` follows the
        ladder for how long the current turn has run.

        A single already-running call can't be downshifted mid-flight — only the
        NEXT call picks a lower effort — so this bounds slow *multi-call* turns,
        not a single hung request (for that, see the harness kill / retry bounds).
        Injects ``reasoning_effort`` as a per-call kwarg, which overrides the
        client's built-in effort (``api_params = {**self.config, **kwargs}``)
        without building or mutating a second client.
        """
        ladder = self._effort_ladder
        if len(ladder) < 2:
            return  # only the primary rung — nothing to step down to
        llm = self._llm
        orig_acall = getattr(llm, "acall", None)
        if orig_acall is None:
            return
        primary = ladder[0][1]

        async def _laddered_acall(*args, **kwargs):
            if "reasoning_effort" not in kwargs:
                elapsed = time.monotonic() - self._turn_started_at
                effort = _effort_for_elapsed(elapsed, ladder)
                if effort is not None and effort != primary:
                    kwargs["reasoning_effort"] = effort
                    if effort != self._ladder_active_effort:
                        self._ladder_active_effort = effort
                        self._ladder_downshifts += 1
            return await orig_acall(*args, **kwargs)

        llm.acall = _laddered_acall

    # ------------------------------------------------------------------ state

    @hidden
    def _render_state(self) -> str:
        """Redacted replacement for the framework's default ``<state>`` block
        (``pformat(self, …)``). Scrubs run-dir path prefixes + the game/benchmark
        tokens so no value repr leaks the game identity into the prompt."""
        return _redact(
            pformat(self, max_length=50, max_string=500, max_depth=4),
            self._redactions,
        )

    @hidden
    def _latest_state(self) -> dict | None:
        lines = self._states_path.read_text().strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line:
                return json.loads(line)
        return None

    def observations(self) -> dict:
        """What the HOST has already worked out for you — read it before planning.

        Bookkeeping you would otherwise pay tokens to rebuild each turn:

        - ``hud_rows`` / ``hud_warm`` — rows that change on almost every step
          whatever you do. This is diagnostic only, not proof of a status band.
        - ``known_from_here`` — actions already shown to be inert or fatal FROM
          THE CURRENT STATE, e.g. ``{"LEFT": "no_impact", "UP": "terminal"}``.
          Submitting one of these as your first action is REJECTED.
        - ``level_starts`` — the step and turn each level began on.
        - ``last_retrodiction`` — how your last ``predicted_path`` scored, and
          whether batching is currently open to you.
        - ``totals`` — no-impact steps, terminal states, and batches truncated.
        """
        try:
            result = json.loads(self._observations_path.read_text())
        except Exception:
            result = {}
        result["prediction_feedback"] = self.prediction_feedback()
        return result

    def level_terminal(self, level: int | None = None) -> dict:
        """The winning evidence for a completed level — read this at a level boundary.

        Returns ``{level, winning_action, step, actions_spent_on_level,
        pre_win_grid_rows, terminal_grid_rows}``. The pre-win frame is the state one
        action BEFORE the level completed; comparing it with the terminal frame and
        the winning action records one successful boundary transition. Parse either
        grid with ``self.grid_array(...)``.

        Defaults to the most recently completed level.
        """
        if level is None:
            level = max(0, (self._latest_state() or {}).get("levels_completed", 1) - 1)
        try:
            return json.loads((self.ipc_dir / f"level_{level}_terminal.json").read_text())
        except Exception:
            return {"error": f"no terminal evidence recorded for level {level}"}

    @hidden
    def _consolidation_path(self, level: int):
        kd = getattr(self, "knowledge_dir", None)
        return None if kd is None else kd / f"level_{level}_insights.md"

    @hidden
    def _write_consolidation_fallback(self, level: int, reason: str) -> None:
        """Deterministic carry-forward if the model did not write one.

        Mirrors tycho's _write_level_insight_fallback: the boundary must leave a
        durable record even when the model spends its turns elsewhere.
        """
        p = self._consolidation_path(level)
        if p is None or p.exists():
            return
        ev = self.level_terminal(level)
        p.write_text(
            f"# Level {level} insights\n\n"
            f"Deterministic fallback note: {reason}. The model did not write a "
            "carry-forward note before the next level began.\n\n"
            f"winning action: {ev.get('winning_action')!r}\n"
            f"actions spent on this level: {ev.get('actions_spent_on_level')}\n"
            f"completed at env step: {ev.get('step')}\n\n"
            "pre-win frame and terminal frame are in "
            f"ipc/level_{level}_terminal.json (self.level_terminal({level})); "
            "they were NOT compared by the model.\n"
        )

    @hidden
    def _scribe_refusals(self, level: int) -> int:
        n = 0
        try:
            for line in self._rejections_path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("reason") == "scribe" and r.get("level") == level:
                    n += 1
        except Exception:
            pass
        return n

    @hidden
    def _log_rejection(self, reason: str, actions: list[str], detail: str) -> None:
        """Record a refused submit. Rejections never reach actions.jsonl, so
        without this the run cannot be scored on attempts the host blocked."""
        state = self._latest_state() or {}
        try:
            with self._rejections_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "turn": state.get("turn"),
                            "step": state.get("step"),
                            "reason": reason,
                            "actions": actions,
                            "level": state.get("levels_completed"),
                            "detail": detail[:500],
                        }
                    )
                    + "\n"
                )
        except Exception:
            return
        # Card Stop condition: refusal deadlock. The gates are new machinery and
        # can starve a run; this is their kill switch. Counted since the last
        # batch that actually reached the harness.
        try:
            committed = self._last_submitted_turn()
            since = 0
            for line in reversed(self._rejections_path.read_text().splitlines()):
                if not line.strip():
                    continue
                r = json.loads(line)
                rejected_turn = r.get("turn")
                if committed is not None and rejected_turn is not None and rejected_turn <= committed:
                    break
                since += 1
            if since >= REJECTION_ABORT:
                (self.ipc_dir / "ABORT").write_text(
                    f"refusal deadlock: {since} consecutive rejected submits "
                    f"with no committed action"
                )
        except Exception:
            pass

    @hidden
    def _last_action_entry(self) -> dict | None:
        for line in reversed(self._actions_path.read_text().strip().splitlines()):
            if line.strip():
                return json.loads(line)
        return None

    @hidden
    def _last_submitted_turn(self) -> int | None:
        lines = self._actions_path.read_text().strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line:
                return json.loads(line).get("turn")
        return None

    def trajectory(self, per_step: bool = True, last_n: int | None = None,
                   include_grids: bool = True, include_events: bool = False) -> list[dict]:
        """One entry per TRANSITION, not per frame. With one action len==1:
        t=trajectory()[0]; compare t['before_grid_rows'] with t['grid_rows'].
        grid_rows is the AFTER grid; comparing first/last entries with len==1
        compares the successor to itself. Use last_transition() for measured
        changes; last_transition(include_grids=True) for explicit before/after.
        per_step=False returns state snapshots including the initial state.
        Animation is unavailable in this runtime.
        """
        from stall_contract import read_rows
        entries = []
        if not per_step:
            entries = read_rows(self._states_path)
            if not include_grids:
                entries = [{k: v for k, v in row.items() if k != "grid_rows"} for row in entries]
        else:
            for row in read_rows(self.ipc_dir / "transitions.jsonl"):
                a, b = row["before"]["grid"], row["after"]["grid"]
                entry = {"step": row["step"], "turn": row["turn"], "action": row["action"],
                         "levels_completed": row["after"]["levels_completed"],
                         "diff_pixels": sum(x != y for ra, rb in zip(a,b) for x,y in zip(ra,rb))
                         if len(a) == len(b) and all(len(x) == len(y) for x,y in zip(a,b)) else None,
                         "event_frame_count": None}
                if include_grids:
                    entry["before_grid_rows"] = ["".join(format(v,"x") for v in r) for r in a]
                    entry["grid_rows"] = ["".join(format(v,"x") for v in r) for r in b]
                if include_events:
                    entry.update(event_frames=[], animation_available=False)
                entries.append(entry)
        return entries if last_n is None else entries[-last_n:] if last_n > 0 else []

    def last_transition(self, include_grids: bool = False) -> dict | None:
        """Latest MEASURED action evidence: step, hashes, changed-cell count,
        bbox, color changes, and up to64 [row,column,before_color,after_color]
        changes (changes_omitted explicitly counts the rest). None before any
        action. include_grids=True adds before_grid_rows and after_grid_rows.
        Use this to test a hypothesis; pixel changes do not establish its cause.
        Never reconstruct an old grid from prose: both exact grids are here.
        """
        from decision_packet import transition_view
        from prediction_feedback import rows
        transitions = rows(self.ipc_dir / 'transitions.jsonl')
        return transition_view(transitions[-1], include_grids) if transitions else None

    def changed_region(self, padding: int = 1, include_grids: bool = True,
                       max_cells: int = 4096) -> dict | None:
        """Return a bounded crop around the latest measured change.

        ``padding`` is added on every side of the measured bounding box.
        ``before_region_rows`` and ``after_region_rows`` are half-open crops
        only when ``include_grids`` is true and their area is at most
        ``max_cells``.  The result remains measured evidence: it does not name
        an object or explain the cause of a change.  ``None`` means that no
        action transition exists yet.  A zero-change transition returns its
        compact evidence with ``region_bbox=None``.
        """
        view = self.last_transition(include_grids=True)
        if view is None:
            return None
        bbox = view.get('changed_bbox')
        if bbox is None:
            view['region_bbox'] = None
            view.pop('before_grid_rows', None)
            view.pop('after_grid_rows', None)
            return view
        if not isinstance(padding, int) or padding < 0:
            raise ValueError('padding must be a nonnegative integer')
        if not isinstance(max_cells, int) or max_cells <= 0:
            raise ValueError('max_cells must be a positive integer')
        before = view.pop('before_grid_rows')
        after = view.pop('after_grid_rows')
        h, w = len(after), max((len(row) for row in after), default=0)
        r0 = max(0, bbox[0] - padding); r1 = min(h, bbox[2] + padding + 1)
        c0 = max(0, bbox[1] - padding); c1 = min(w, bbox[3] + padding + 1)
        view['region_bbox'] = [r0, c0, r1, c1]
        view['region_shape'] = [r1 - r0, c1 - c0]
        view['region_omitted'] = (r1 - r0) * (c1 - c0) > max_cells
        if include_grids and not view['region_omitted']:
            view['before_region_rows'] = [row[c0:c1] for row in before[r0:r1]]
            view['after_region_rows'] = [row[c0:c1] for row in after[r0:r1]]
        return view

    def grid_summary(self, grid_rows: list[str] | None = None) -> dict:
        """Return bounded measured shape, color counts, and color bboxes.

        No object or goal labels are inferred.  The full grid remains available
        from ``latest_state()``; this summary is intended to replace repeated
        full-grid prints during initial inspection.
        """
        if grid_rows is None:
            state = self._latest_state() or {}
            grid_rows = state.get('grid_rows')
        if not grid_rows:
            return {'kind': 'grid_summary', 'status': 'unavailable',
                    'reason': 'no current grid'}
        counts = {}
        positions = {}
        for r, row in enumerate(grid_rows):
            for c, value in enumerate(row):
                counts[value] = counts.get(value, 0) + 1
                positions.setdefault(value, []).append((r, c))
        bboxes = {}
        for value, cells in positions.items():
            rs = [cell[0] for cell in cells]; cs = [cell[1] for cell in cells]
            bboxes[value] = [min(rs), min(cs), max(rs), max(cs)]
        return {'kind': 'grid_summary', 'status': 'measured',
                'shape': [len(grid_rows), max((len(row) for row in grid_rows), default=0)],
                'color_counts': dict(sorted(counts.items())),
                'color_bboxes': dict(sorted(bboxes.items()))}

    def preflight_model(self, actions: list[str]) -> dict:
        """Test the selected world-model helper using actual runtime preparation.
        Use ``self.model_grid()`` for direct helper calls. encode receives a NumPy grid;
        predict receives a JSON-normalized latent
        (tuples become lists). Latents must be JSON-safe and <=16000 characters.
        After writing/loading a helper, preflight_model(['UP']) in the SAME cell
        before a proposed action detects execution errors that direct tests miss.
        Ready means executable, not true; this neither commits a prediction nor
        earns batching. An unavailable model still permits one exploratory action.
        """
        return self._model_feedback.preflight(self._latest_state(), actions)

    def model_grid(self) -> np.ndarray:
        """Return the current grid in the exact helper input representation.

        This is a detached ``int64`` NumPy array.  It avoids passing the
        serialized hexadecimal ``latest_state()['grid_rows']`` directly to
        ``encode``.  A missing observation is an error, not an empty grid.
        """
        state = self._latest_state()
        if not state:
            raise ValueError('no current observation')
        grid = state.get('grid')
        if grid is None:
            grid = self.grid_array(state.get('grid_rows') or [])
        elif isinstance(grid, list) and grid and isinstance(grid[0], str):
            grid = self.grid_array(grid)
        return np.asarray(grid, dtype=int).copy()

    def game_status(self) -> dict:
        """Current game status: latest state header (without the grid) plus whether
        an action batch was already submitted for this turn."""
        state = self._latest_state()
        if state is None:
            return {"status": "no game state received yet — wait for the harness"}
        header = {k: v for k, v in state.items() if k not in ("grid_rows",)}
        header["submitted_this_turn"] = self._last_submitted_turn() == state.get("turn")
        return header

    def prediction_feedback(self) -> dict:
        """Compare the frozen encode/predict committed before actions with actual
        successor frames. Field errors are evidence against your model. Fix its
        code, then predict one fresh action successfully before batching again.
        A changed helper version needs its own check. No model or unresolved
        alternatives still permit one exploratory action. Matching a partial
        encoder does not establish object identity or a causal explanation.
        """
        return self._model_feedback.observe()

    def latest_state(self) -> dict:
        """Return the FULL latest game state (including grid_rows), re-read from
        the harness state file. Use whenever the game_states notification is
        missing or truncated — never wait for a state you can fetch yourself."""
        state = self._latest_state()
        if state is not None:
            state["prediction_feedback"] = self.prediction_feedback()
        return state if state is not None else {"status": "no game state yet"}

    def decision_packet(self) -> dict:
        """Read current metadata, last three exact-grid changes, frozen prediction
        feedback, helper versions, and the latest labelled experiment together.
        Full raw observation and history remain available, without compression.
        """
        from decision_packet import packet
        from prediction_feedback import rows
        state = self._latest_state() or {}
        obs = json.loads(self._observations_path.read_text()) if self._observations_path.exists() else {}
        result = packet(state, rows(self.ipc_dir / 'transitions.jsonl'),
                        self.prediction_feedback(), obs, self.helpers_dir, self.ipc_dir)
        from contingent_controller import enabled
        if enabled():
            events = rows(self.ipc_dir / 'contingent_events.jsonl')
            result['contingent_controller'] = events[-1] if events else None
        return result

    @hidden
    def _critic_packet_for(self, messages, trigger, state, wall_remaining):
        """Build the critic's compact input from host records, not the transcript.

        Reuses ``decision_packet``'s ``recent_transitions``, which report changed-cell
        counts and regions but never cell values: the critic can still ask whether a
        proxy connects to level completion, and cannot slide into asking which pixel
        moved.
        """
        from decision_packet import packet as _decision_packet
        from prediction_feedback import rows as _rows
        observations = (json.loads(self._observations_path.read_text())
                        if self._observations_path.exists() else {})
        evidence = _decision_packet(
            state, _rows(self.ipc_dir / 'transitions.jsonl'), self.prediction_feedback(),
            observations, self.helpers_dir, self.ipc_dir)
        submitted = _rows(self.ipc_dir / 'actions.jsonl')
        commitment = None
        if submitted:
            last = submitted[-1]
            commitment = {k: last.get(k) for k in
                          ('turn', 'step', 'actions', 'rationale', 'predicted_path')}
        executed = len(_rows(self.ipc_dir / 'transitions.jsonl'))
        elapsed = time.monotonic() - self._game_started_at
        commitments = _critic_commitment_claims(submitted)
        return _critic_packet(
            trigger=trigger, evidence=_critic_actor_evidence(messages),
            commitment=commitment, transitions=evidence['recent_transitions'],
            budget=_critic_budget_view(wall_remaining=wall_remaining,
                                       actions_executed=executed, wall_elapsed=elapsed),
            ledger=self._objections.view(), state=state, commitments=commitments)

    def measured_evidence(self) -> dict:
        """Return the current host-measured evidence block, separate from beliefs."""
        from evidence_renderer import measured_evidence
        return measured_evidence(self)

    @hidden
    def render_measured_evidence(self) -> str:
        """Render measured evidence for the prompt without adding interpretation."""
        from evidence_renderer import render_measured_evidence
        return render_measured_evidence(self)

    def plan_actions(self, target: dict, actions: list[str] | None = None,
                     limits: dict | None = None) -> dict:
        """Search your partial model for a LOCAL target; exploration needs no model.

        target={"path":["x"],"op":"eq","value":3}; ops eq/lt/le/gt/ge,
        or {"all":[predicates]} / {"any":[predicates]}. Missing fields are errors.
        Default actions exclude CLICK; supply explicit "CLICK x y" candidates.
        Limits may lower wall_seconds=5, callback_seconds=.25,
        generated_states=5000, frontier=1000, depth=12, action_count=64.

        path_found returns plan, predicted_latents, and search_id. Pass search_id
        to submit_actions when using a nonempty prefix of this plan. Author host
        outcome predicted_path separately: latent changes are not outcome labels.
        Existing batch gates still apply. This commits no action or prediction.
        already_satisfied means only this target holds in the encoded state.
        model_exhausted concerns ONLY this model/action set. limited is unresolved
        work, model_error is broken execution, invalid_request needs corrected
        inputs, unavailable means execution could not safely be provided.
        """
        import latent_planner
        import uuid
        from prediction_feedback import write_row
        state = self._latest_state()
        result = latent_planner.run(state, self.helpers_dir, _check_helper_ast,
                                    target, actions, limits)
        if result['status'] == 'path_found':
            search_id = uuid.uuid4().hex
            result['search_id'] = search_id
            # Keep a detached authority record; returned dictionaries are mutable.
            if len(self._search_results) >= 16:
                self._search_results.pop(next(iter(self._search_results)))
            self._search_results[search_id] = json.loads(json.dumps(result))
        write_row(self.ipc_dir / 'search_results.jsonl', result)
        result.pop('model_source', None)  # durable diagnostics retain the frozen source
        return result

    def set_experiment(self, question: str, hypotheses: list[str], action: str,
                       outcomes: dict) -> dict:
        """Record a changed prospective question in the same cell as the action.
        outcomes must name an epistemic target and its compatible host metric, then
        map every hypothesis to a distinct typed prediction and knowledge update,
        plus an `otherwise` update. The host scores only the typed prediction
        against the next matching transition; hypothesis labels and proposed
        updates remain model conjectures, not host-certified facts.
        Supported metrics: grid_changed, changed_bbox, shape_changed, after_shape,
        changed_cells, color_changes, state, and levels_completed_delta.
        Unchanged experiments remain in decision_packet with their original step.
        """
        from decision_packet import record_experiment
        return record_experiment(self.ipc_dir, self._latest_state(), question,
                                 hypotheses, action, outcomes)

    def compare_contingent_plans(self, proposal: dict) -> dict:
        """Compare up to four two-decision candidates across up to four weighted
        hypotheses. No action is submitted. Estimates are conjectures, not reward.
        proposal={"hypotheses":{"a":0.5,"b":0.5}, "candidates":[
          {"id":"probe", "action":"UP", "metric":"changed_bbox",
           "predictions":{"a":[0,0,1,0],"b":[1,0,1,1]},
           "assumptions":["These alternatives cover the decision"],
           "continuations":[{"action":"LEFT", "outcomes":{
             "a":{"completion":1,"remaining_actions":1,"recovery_actions":0},
             "b":{"completion":0,"remaining_actions":1,"recovery_actions":2}}}]}]}
        Add other roots and continuation actions to compare choices. A common
        observation must use one shared next action across all matching worlds.
        Costs include the next primitive and any further estimated actions.
        Completion is a provisional probability in [0,1]. Selection maximizes
        expected completion, then minimizes action plus recovery cost. null next
        action returns control to the Actor. No entropy bonus or hidden-world policy.
        """
        from contingent_controller import compile_plan, enabled
        if not enabled():
            raise RuntimeError('The contingent interface is off in this arm; '
                               'use ordinary single-action exploration')
        return compile_plan(proposal)

    def submit_contingent_plan(self, proposal: dict) -> str:
        """Choose and submit one root action through ordinary gates. After its
        authoritative successor, the dispatcher can submit at most ONE matching
        continuation without a model call. Unknown outcomes, boundaries, frozen
        model failures, new user guidance or rejected actions return to the Actor.
        No long batch or verified world-model status is earned by this plan.
        """
        from contingent_controller import enabled
        from prediction_feedback import write_row
        if not enabled():
            return 'REJECTED: the contingent interface is off in this arm'
        compiled = self.compare_contingent_plans(proposal)
        state = self._latest_state()
        if not state:
            return "REJECTED: no current observation"
        pending = self.ipc_dir / 'contingent_pending.json'
        if pending.exists():
            return "REJECTED: a contingent decision is already pending"
        record = {"turn":state['turn'], "step":state['step'],
                  "levels_completed":state.get('levels_completed',0),
                  "grid":self.model_grid().tolist(), "plan":compiled}
        pending.write_text(json.dumps(record, allow_nan=False))
        write_row(self.ipc_dir / 'contingent_events.jsonl',
                  {"status":"proposed", **record})
        try:
            result = self.submit_actions([compiled['selected']['action']],
                rationale="predict: contingent root " + compiled['selected']['id'])
        except _ReturnResultSignal:
            raise
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
        pending.unlink(missing_ok=True)
        write_row(self.ipc_dir / 'contingent_events.jsonl',
                  {"status":"root_rejected", "turn":state['turn'], "reason":result})
        return result

    @hidden
    def _continue_contingent_plan(self, notification: dict) -> bool:
        """Dispatcher-only fast path; existing submit_actions owns every gate."""
        from contingent_controller import enabled, select_branch
        from decision_packet import _observed_metric
        from prediction_feedback import rows, write_row
        if not enabled():
            return False
        pending = self.ipc_dir / 'contingent_pending.json'
        if not pending.exists():
            return False
        record = json.loads(pending.read_text())
        state = self._latest_state() or {}
        event = {"turn":state.get('turn'), "proposal_sha256":record['plan']['proposal_sha256']}
        def stop(reason):
            pending.unlink(missing_ok=True)
            write_row(self.ipc_dir / 'contingent_events.jsonl',
                      {**event, "status":"stopped", "reason":reason})
            return False
        if notification.get('user_messages'):
            return stop('user_guidance')
        # Never consume a prospective plan on a repeated root notification.
        if state.get('turn') == record['turn'] and state.get('step') == record['step']:
            return False
        if state.get('turn') != record['turn'] + 1 or state.get('step') != record['step'] + 1:
            return stop('stale_or_skipped_observation')
        transitions = rows(self.ipc_dir / 'transitions.jsonl')
        transition = transitions[-1] if transitions else {}
        selected = record['plan']['selected']
        submitted = self._last_action_entry() or {}
        if (transition.get('turn') != record['turn'] or transition.get('step') != state['step']
                or transition.get('action') != selected['action']
                or transition.get('before', {}).get('grid') != record['grid']
                or transition.get('after', {}).get('grid') != self.model_grid().tolist()
                or submitted.get('turn') != record['turn']
                or submitted.get('step') != record['step']
                or submitted.get('actions') != [selected['action']]):
            return stop('transition_identity_mismatch')
        after = transition['after']
        if (after.get('state') in ('WIN','GAME_OVER')
                or after.get('levels_completed',0) != record['levels_completed']
                or state.get('levels_completed',0) != record['levels_completed']):
            return stop('protocol_boundary')
        if transition['before']['grid'] == after['grid']:
            return stop('unchanged_grid')
        checks = rows(self.ipc_dir / 'step_checks.jsonl')
        commits = rows(self.ipc_dir / 'model_predictions.jsonl')
        has_model = any(c.get('turn') == record['turn'] and c.get('step') == record['step']
                        and c.get('status') == 'committed' for c in commits)
        current_checks = [c for c in checks if c.get('turn') == record['turn']
                          and c.get('step') == state['step']]
        if (has_model and (not current_checks or any(c.get('status') != 'matched' for c in current_checks))
                or any(c.get('status') in ('mismatch','error','timeout') for c in current_checks)):
            return stop('frozen_prediction_failed')
        observed = _observed_metric(selected['metric'], transition, record)
        branch = select_branch(record['plan'], observed)
        event.update(metric=selected['metric'], observed=observed)
        if branch is None:
            return stop('none_of_predictions')
        if branch['action'] is None:
            return stop('planned_handoff')
        # Consume before publishing. A failed submission hands back to the Actor;
        # it cannot silently retry or extend this two-decision lease.
        pending.unlink()
        write_row(self.ipc_dir / 'contingent_events.jsonl',
                  {**event, "status":"branch_selected", "action":branch['action'],
                   "estimates":branch, "assumptions":selected['assumptions']})
        try:
            result = self.submit_actions([branch['action']],
                rationale="predict: observation-contingent continuation for " + selected['id'])
        except _ReturnResultSignal:
            write_row(self.ipc_dir / 'contingent_events.jsonl',
                      {**event, "status":"continuation_submitted", "action":branch['action']})
            return True
        write_row(self.ipc_dir / 'contingent_events.jsonl',
                  {**event, "status":"continuation_rejected", "reason":result})
        return False

    async def deep_investigate(self, question: str, evidence_refs: list[str]) -> dict:
        """Use one explicitly scoped reasoning-on call for a hard abstraction gap.

        Routine Actor calls stay reasoning-off. This path is deliberately sparse,
        bounded, and labelled; it returns analysis only and cannot submit actions.
        Call it only when ordinary inspection cannot discriminate the live models,
        then turn any useful result into a tested experiment or procedure.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("deep investigation needs one concrete question")
        if not isinstance(evidence_refs, list) or not 1 <= len(evidence_refs) <= 8:
            raise ValueError("supply one to eight existing evidence references")
        if any(not isinstance(ref, str) or not ref.strip() for ref in evidence_refs):
            raise ValueError("evidence references must be nonempty strings")
        maximum = int(os.environ.get("ARC_DEEP_MAX_CALLS", "1"))
        if self._deep_investigations >= maximum:
            raise RuntimeError("deep investigation budget exhausted")
        self._deep_investigations += 1
        deep_id = f"deep-{self._deep_investigations}"
        started = time.monotonic()
        response = await self._llm.acall(
            [{"role": "system", "content": (
                "Analyze only the stated abstraction failure. Separate measured facts "
                "from interpretations, name the best discriminating test, and propose "
                "reusable code only when its applicability and stop condition are clear."
            )}, {"role": "user", "content": json.dumps({
                "question": question, "evidence_refs": evidence_refs,
                "measured_evidence": self.measured_evidence(),
            }, sort_keys=True)}],
            max_tokens=int(os.environ.get("ARC_DEEP_MAX_TOKENS", "32768")),
            extra_body={
                "reasoning": {"enabled": True},
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        usage = getattr(response, "usage", None) or {}
        record = {
            "schema": "deep-investigation-v1", "id": deep_id,
            "question": question, "evidence_refs": evidence_refs,
            "wall_seconds": time.monotonic() - started,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "result": str(getattr(response, "content", "")),
            "actions_enabled": None,
        }
        append(self.ipc_dir / "deep_investigations.jsonl", record)
        return record

    def record_deep_outcome(self, investigation_id: str, actions_enabled: int) -> dict:
        """Record measured action amortization for a prior deep investigation."""
        if not re.fullmatch(r"deep-[1-9][0-9]*", investigation_id):
            raise ValueError("invalid investigation id")
        if not isinstance(actions_enabled, int) or not 0 <= actions_enabled <= 1000:
            raise ValueError("actions_enabled must be an integer from 0 through 1000")
        record = {"schema": "deep-outcome-v1", "id": investigation_id,
                  "actions_enabled": actions_enabled,
                  "turn": (self._latest_state() or {}).get("turn")}
        append(self.ipc_dir / "deep_outcomes.jsonl", record)
        return record

    @hidden
    def format_game_context(self) -> str:
        state = self._latest_state()
        if state is None:
            return f"game={self.alias} — waiting for the first state from the harness."
        submitted = self._last_submitted_turn() == state.get("turn")
        parts = [
            f"game={self.alias}",
            f"turn={state.get('turn')}",
            f"env_step={state.get('step')}",
            f"level={state.get('level')}",
            f"state={state.get('state')}",
            f"levels_completed={state.get('levels_completed')}",
            f"available_actions={state.get('available_actions')}",
            f"submitted_this_turn={submitted}",
        ]
        deadline = float(os.environ.get('ARC_GAME_MODEL_ADMISSION_DEADLINE', 'inf'))
        if math.isfinite(deadline):
            parts.append(f"model_admission_seconds_remaining={max(0.0, deadline-time.monotonic()):.1f}")
            parts.append('Budget stops further model calls, including analysis calls; prioritize executable progress.')
        line = " | ".join(str(p) for p in parts)
        hints = [
            self._stall_status_line(state),
            self._scribe_status_line(state),
            self._helpers_status_line(),
            self._knowledge_status_line(),
        ]
        return "\n".join([line] + [h for h in hints if h])

    @hidden
    def _scribe_status_line(self, state: dict) -> str:
        """Announce a pending level consolidation before the model tries to act,
        so the scribe reads as a step in the turn rather than as a surprise refusal."""
        cur = state.get("levels_completed", 0) or 0
        if cur <= 0:
            return ""
        p = self._consolidation_path(cur - 1)
        if p is None or p.exists():
            return ""
        return (
            f"SCRIBE PENDING: level {cur - 1} is complete and not yet consolidated. "
            f"self.level_terminal({cur - 1}) has the pre-win frame, the winning action and "
            f"the terminal frame; write level_{cur - 1}_insights.md before acting."
        )

    @hidden
    def _stall_status_line(self, state: dict) -> str:
        status = self._stall.status()
        if self._stall.opened and getattr(self, "knowledge_dir", None):
            hypotheses = self.knowledge_dir / "hypotheses.md"
            if hypotheses.exists():
                status += "\nCurrent authored hypotheses:\n" + hypotheses.read_text()
        return status

    def resolve_stall(
        self, disposition: str, previous_basis: list[str], active_basis: list[str],
        action: str, rationale: str, abandon_if: str | None = None,
        predictions: dict[str, str] | None = None,
    ) -> str:
        """Resolve an open stall and submit exactly ONE legal action in this cell.

        continue requires abandon_if; probe requires two distinct predicted outcomes
        keyed by alternative; switch requires a different active_basis. Host-checkable
        falsifiers are move/no_impact/level_up/terminal, grid_changed/grid_unchanged,
        or levels_completed ==/>=/> N. Other prose is retained but scores unknown.
        After its result, call finish_stall with the basis that now governs.
        """
        state = self._latest_state() or {}
        row = dict(disposition=disposition, previous_basis=previous_basis,
                   active_basis=active_basis, action=action, rationale=rationale,
                   abandon_if=abandon_if, predictions=predictions)
        error = self._stall.stage(state, row)
        if error:
            return error
        try:
            result = self.submit_actions([action], rationale=rationale)
            # A valid resolution may still violate an existing gameplay gate.
            return self._stall.reject(result)
        finally:
            self._stall.staged = None

    def finish_stall(self, active_basis_after: list[str], rationale: str) -> str:
        """After the reviewed action, author the basis retained or changed by its result."""
        return self._stall.finish(active_basis_after, rationale)

    @hidden
    def _helpers_status_line(self) -> str:
        """A3: the world model on the dashboard beside the notes. It used to be
        invisible (self.h is hidden from <state> so the run dir cannot leak the
        game id), which left markdown as the only signposted memory."""
        mods = []
        for p in sorted(self.helpers_dir.glob("*.py")):
            try:
                src = p.read_text()
            except Exception:
                continue
            fns = [
                ln.split("def ", 1)[1].split("(")[0]
                for ln in src.splitlines()
                if ln.startswith("def ")
            ]
            mods.append(f"{p.stem}.{{{','.join(fns)}}}" if fns else p.stem)
        if not mods:
            return "helpers: NONE YET — the plan has to live in code before it can constrain a batch"
        from prediction_feedback import select_model
        try:
            selected = select_model(self.helpers_dir)
        except OSError:
            selected = 'helper unreadable'
        hint = ""
        if not isinstance(selected, str) and all('def ' + name + '(' in selected['source'] for name in ('encode', 'predict')):
            hint = " | plan_actions(target) searches this partial model (bounded; no batch permission)"
        return "helpers: " + " | ".join(mods) + "  (self.h.<module>.<fn> after load_helpers())" + hint

    @hidden
    def _knowledge_status_line(self) -> str:
        return ""

    # ------------------------------------------------------------------ grid

    def grid_array(self, grid_rows: list[str]) -> np.ndarray:
        """Parse the state's ``grid_rows`` (64 hex strings) into a 64x64 numpy int
        array. Index as arr[row, col]; row 0 is the top of the screen."""
        return np.array([[int(c, 16) for c in row] for row in grid_rows], dtype=int)

    def render_grid(self, grid_rows: list[str], every: int = 4,
                    bbox: tuple[int, int, int, int] | None = None,
                    max_chars: int = 8000) -> str:
        """Render a bounded grid crop with explicit truncation metadata.

        ``bbox`` is half-open ``(row_start, row_end, col_start, col_end)``.
        The default covers the whole grid but caps the returned text at
        ``max_chars``; request a smaller crop for targeted inspection.  A cap
        never silently discards data: the suffix reports omitted characters
        and rows.  Exact cells remain available through ``model_grid()``.
        """
        if not grid_rows:
            return '[grid unavailable]'
        if not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError('max_chars must be a positive integer')
        h, w = len(grid_rows), max((len(row) for row in grid_rows), default=0)
        if bbox is None:
            r0, r1, c0, c1 = 0, h, 0, w
        else:
            if len(bbox) != 4:
                raise ValueError('bbox must be (row_start,row_end,col_start,col_end)')
            r0, r1, c0, c1 = bbox
            if not (0 <= r0 <= r1 <= h and 0 <= c0 <= c1 <= w):
                raise ValueError('bbox is outside grid bounds')
        header = "    " + "".join(f"{c:<{every}}" for c in range(c0, c1, every))
        lines = [header]
        for r in range(r0, r1):
            lines.append(f"{r:>3} {grid_rows[r][c0:c1]}")
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        kept = text[:max_chars].rsplit('\n', 1)[0]
        omitted = len(text) - len(kept)
        omitted_rows = max(0, text[len(kept):].count('\n'))
        return (kept + f"\n[TRUNCATED: omitted_chars={omitted}; "
                f"omitted_rows_at_least={omitted_rows}; request a smaller bbox]")

    def grid_image(self, grid_rows: list[str] | None = None) -> Image:
        """The current settled grid as a color PNG Image (standard 16-color palette,
        each cell an N×N block so a 64×64 grid is (N·64)×(N·64) px). Pass grid_rows to render
        a specific grid; default is the latest state's grid. Look at it with
        show(self.grid_image()); when visual input is enabled it is auto-shown each
        turn. Colors match the hex digits in grid_rows (parse those with
        self.grid_array for exact values)."""
        if grid_rows is None:
            grid_rows = (self._latest_state() or {}).get("grid_rows") or []
            if not grid_rows:  # 'only' mode drops grid_rows from state — use trajectory
                traj = self.trajectory(last_n=1, include_grids=True)
                grid_rows = traj[-1].get("grid_rows") if traj else []
        arr = self.grid_array(grid_rows) if grid_rows else np.zeros((64, 64), dtype=int)
        return Image.from_bytes(_grid_png_bytes(arr, self._png_scale), media_type="image/png")

    # ---------------------------------------------------------------- actions

    def submit_actions(
        self, actions: list[str], rationale: str, predicted_path: list[dict] | None = None,
        *, search_id: str | None = None, closes: dict | None = None
    ) -> str:
        """Submit the action sequence for the CURRENT turn — this is how you play.

        ``actions``: the actions to execute, e.g. ``["UP"]``. **Batch length is
        earned, not chosen.** A single action is always allowed. More than one is
        allowed only when you submit a ``predicted_path`` and your last one was
        right, with a successful frozen-model check of the same helper version.
        A search result alone never earns batch permission.

        ``search_id``: pass the reference returned by plan_actions when using its
        plan. Current state/model and a nonempty action prefix must still match.

        ``closes``: required while a HOST EPISTEMIC OBJECTION is open, as
        ``{"review_id": …, "disposition": …, "evidence": …}``. Dispositions:
        ``defend`` (cite the observation that warrants the claim), ``narrow``
        (add ``narrowed_claim``), ``withdraw`` (add ``retire_actions``: the
        actions you will stop repeating), or ``probe`` (add ``actions_budgeted``:
        settle it by acting, and the host will ask what those actions decided).
        Defending and probing leave this commitment unchanged — dispose of the
        objection honestly rather than agreeing to clear the check.

        ``predicted_path``: one entry per action, in order, each
        ``{"expect": <class>, "z": <your latent, or None>}`` where ``<class>`` is
        one of:

        - ``"move"`` — the playfield changes
        - ``"no_impact"`` — the whole grid is unchanged
        - ``"level_up"`` — this action completes the level
        - ``"terminal"`` — this action ends the game (GAME_OVER or WIN)

        The harness scores ``expect`` against what actually happens and reports it
        in the next state's note and in ``self.observations()``. This is your
        retrodiction, and it is checked rather than asserted.

        ``rationale``: one or two lines: a conditional prediction or the observation
        sought by exploration. No objective or hypothesis pair is required.

        Read ``self.observations()`` first: an action already known to be inert or
        fatal from the current state is REJECTED, and so is a batch longer than 1
        while your last retrodiction failed. To reopen batching, submit ONE action
        with a ``predicted_path`` and be right about it.

        The harness executes the batch in order, stopping early on level
        completion, WIN, GAME_OVER, or the first step with an unchanged whole grid. A successful submit ENDS your turn, so do any
        knowledge/helper writes BEFORE calling it. On invalid input it returns a
        ``"REJECTED: …"`` string (no yield) so you can fix it in the same turn.
        """
        state = self._latest_state()
        if state is None:
            return "REJECTED: no game state received yet."
        if state.get("state") == "WIN":
            return "REJECTED: game already won — no more actions needed."
        if not actions:
            return "REJECTED: empty action list."
        available = set(state.get("available_actions", []))
        for a in actions:
            base = a.split()[0] if a.strip() else ""
            if base not in available:
                return (
                    f"REJECTED: {a!r} — base action {base!r} not in available {sorted(available)}."
                )
            if base == "CLICK":
                m = _CLICK.match(a.strip())
                if not m or not all(0 <= int(g) <= 63 for g in m.groups()):
                    return f"REJECTED: {a!r} — CLICK needs 'CLICK x y' with 0<=x,y<=63."
        turn = state.get("turn")
        if self._last_submitted_turn() == turn:
            return f"REJECTED: already submitted for turn {turn} — wait for the next state."

        from objection_ledger import (ObjectionError, recurrence_guidance,
                                      submit_guidance)
        step_now = int(state.get("step") or 0)
        closure = None
        if closes is not None:
            recurred = self._objections.recurrence(actions)
            try:
                if self._objections.blocking():
                    closure = self._objections.close(closes, step=step_now)
                elif recurred is not None:
                    closure = self._objections.reopen_for_probe(
                        recurred, closes, step=step_now)
                else:
                    return ("REJECTED: closes= was supplied with no objection open and "
                            "no retired action resumed. Submit without it.")
            except ObjectionError as error:
                return f"REJECTED: {error}"
            append(self.ipc_dir / 'objections.jsonl',
                   {'event': 'closed', 'turn': turn, **closure})
        elif self._objections.blocking():
            expired = self._objections.block_submit()
            if expired is None:
                return submit_guidance(self._objections.pending)
            append(self.ipc_dir / 'objections.jsonl',
                   {'event': 'expired', 'turn': turn, 'step': step_now, **expired})
        else:
            recurred = self._objections.recurrence(actions)
            if recurred is not None:
                append(self.ipc_dir / 'objections.jsonl',
                       {'event': 'recurrence_rejected', 'turn': turn, 'step': step_now,
                        'claim_ref': recurred['claim_ref'],
                        'submitted_actions': recurred['submitted_actions'],
                        'matched_actions': recurred['matched_actions']})
                return recurrence_guidance(recurred)

        if search_id is not None:
            from latent_planner import identity
            from prediction_feedback import select_model
            stored = self._search_results.get(search_id) if isinstance(search_id, str) else None
            try:
                selected = select_model(self.helpers_dir)
                valid = (stored is not None and stored['status'] == 'path_found'
                         and not isinstance(selected, str)
                         and stored['snapshot'] == identity(state, selected)
                         and actions == stored['plan'][:len(actions)])
            except Exception:
                valid = False
            if not valid:
                return "REJECTED: stale, unknown, or mismatched search_id; search again or submit an ordinary exploratory action."

        stall_error = self._stall.before_submit(
            state, rationale, actions, read_rows(self._actions_path))
        if stall_error:
            self._log_rejection("stall_unresolved", actions, stall_error)
            return "REJECTED: " + stall_error + "\n" + self._stall_status_line(state)

        # ---- host gates. Each refuses a move the run has already paid to learn
        # is wrong, so the failure it prevents cannot be re-entered by fluency.
        obs = self.observations()
        _EXPECT = {"move", "no_impact", "level_up", "terminal"}

        def _bad_path(p) -> str | None:
            if not isinstance(p, list) or len(p) != len(actions):
                return (
                    f"predicted_path must be a list of {len(actions)} entries, one per "
                    f"action, each {{'expect': one of {sorted(_EXPECT)}, 'z': ...}}"
                )
            for i, e in enumerate(p):
                if not isinstance(e, dict) or e.get("expect") not in _EXPECT:
                    return f"predicted_path[{i}] needs an 'expect' in {sorted(_EXPECT)}"
            return None

        # G1 — re-entry. Only the FIRST action is judged: later ones start from
        # states we have not seen. If every available action is already known bad
        # we do not gate, or the run would starve.
        known = obs.get("known_from_here") or {}
        avail_known_bad = [a for a in state.get("available_actions", []) if a in known]
        base0 = actions[0].split()[0] if actions[0].strip() else ""
        if base0 in known and len(avail_known_bad) < len(state.get("available_actions", [])):
            why = known[base0]
            msg = (
                f"REJECTED: {actions[0]!r} is already recorded as {why!r} from this exact "
                f"playfield state (see self.observations()['known_from_here']). "
                f"Repeating it spends an action to re-learn something the run already knows. "
                f"Choose a different action, or change the state first."
            )
            self._log_rejection("reentry", actions, msg)
            return msg

        # G2 — SCRIBE. A completed level is consolidated before the next is played.
        # Ported from tycho (agent.py:652, _summarize_level, prompts/scribe.user.j2).
        # q3a put a refusal here instead and P2 failed: a refusal can make the
        # omission illegal, it cannot perform the consolidation. So this hands over
        # the evidence, and after SCRIBE_MAX_REFUSALS the host writes the note itself.
        cur_levels = state.get("levels_completed", 0)
        done_level = cur_levels - 1
        note_path = self._consolidation_path(done_level) if cur_levels > 0 else None
        if note_path is not None and not note_path.exists():
            refusals = self._scribe_refusals(cur_levels)
            if refusals >= SCRIBE_MAX_REFUSALS:
                self._write_consolidation_fallback(
                    done_level,
                    f"model did not consolidate after {refusals} boundary turns",
                )
            else:
                msg = (
                    f"REJECTED: you just COMPLETED level {done_level}. This turn is for "
                    "consolidation, not for acting.\n\n"
                    f"Call self.level_terminal({done_level}) — it returns the pre-win frame, "
                    "the action that won, and the terminal frame. Compare those three: they "
                    "record one successful boundary transition; the transcript of "
                    "this level will not be this legible again.\n\n"
                    "Distil supported conditions, unresolved goal/mechanics claims, and what may "
                    f"vary on the next level, then write it with "
                    f"self.write_notes('level_{done_level}_insights.md', ...). If the evidence "
                    "changes your executable model, rewrite the helper too — the next level's "
                    "spawn and object roles may already have moved.\n\n"
                    "Then submit your action for the new level."
                )
                self._log_rejection("scribe", actions, msg)
                return msg

        # Coarse frame change cannot validate an executable state prediction.
        # Snapshot the Actor's own helper, never a host-authored game model.
        proposal = self._model_feedback.prepare(state, actions)
        if len(actions) > 1:
            error = self._model_feedback.batch_error(proposal, self.prediction_feedback())
            if error:
                self._log_rejection("model_prediction_unverified", actions, error)
                return "REJECTED: " + error

        # G3/G4 — batch length is earned by a scored prediction, never by tone.
        if len(actions) > 1:
            err = _bad_path(predicted_path)
            if err:
                msg = (
                    f"REJECTED: a batch of {len(actions)} needs a predicted_path — {err}. "
                    "Submit one action, or compute the path over predict() first."
                )
                self._log_rejection("no_path", actions, msg)
                return msg
            retro = obs.get("last_retrodiction")
            if retro is not None and not retro.get("ok"):
                msg = (
                    f"REJECTED: your last retrodiction scored "
                    f"{retro.get('matched')}/{retro.get('checked')}, so batching is closed. "
                    "Submit ONE action with a predicted_path; get it right and batching reopens."
                )
                self._log_rejection("retrodiction_closed", actions, msg)
                return msg
        # Over-cap batches execute their first ``cap`` actions instead of bouncing:
        # a REJECTED round-trip costs a full model call, while the useful prefix
        # can play now. The drop is LOUD — flagged here in the yield explanation
        # and relayed by the harness as a note on the next state (truncated_from).
        cap = getattr(self, "_max_actions_per_turn", MAX_ACTIONS_PER_TURN)
        requested = len(actions)
        truncation_warning = ""
        if cap > 0 and requested > cap:
            actions = actions[:cap]
            truncation_warning = (
                f"; WARNING: batch had {requested} actions — only the first {cap} were "
                f"submitted, actions {cap + 1}..{requested} were NOT executed"
            )
        entry = {
            "turn": turn,
            "actions": actions,
            "rationale": rationale,
            "step": state["step"],
            "predicted_path": predicted_path,
            "search_id": search_id,
            "levels_completed": state.get("levels_completed", 0),
        }
        if truncation_warning:
            entry["truncated_from"] = requested
        # Only the admitted action prefix belongs to the prospective record.
        proposal["actions"] = actions
        if "predicted" in proposal:
            proposal["predicted"] = proposal["predicted"][:len(actions)]
        self._model_feedback.commit(proposal)
        # Publish review evidence before the action file wakes the executing host.
        self._stall.commit(state)
        try:
            with self._actions_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            (self.ipc_dir / "ABORT").write_text("action publication failed after review")
            raise
        # The next Actor request occurs only after the host has drained or
        # stopped this admitted sequence, so it is the first useful boundary at
        # which the critic can compare the commitment with environmental evidence.
        self._critic_pending_sequence_length = max(
            self._critic_pending_sequence_length, len(actions)
        )
        self._critic_pending_sequence_step = int(state["step"])
        # New turn starts now: reset the effort-ladder clock (mirrors the harness
        # resetting its silence timer when it receives this batch) and rewind to
        # the primary effort.
        self._turn_started_at = time.monotonic()
        self._ladder_active_effort = self._effort_ladder[0][1] if self._effort_ladder else None
        raise _ReturnResultSignal(
            result={
                "kind": RespondReason.WAIT,
                "explanation": f"submitted {len(actions)} action(s) for turn {turn}"
                f"{truncation_warning}; waiting for the next state",
            }
        )

    # ---------------------------------------------------------------- helpers

    def write_helper(self, filename: str, source: str) -> str:
        """Create or overwrite a persistent python helper module in the workspace
        (``helpers/<filename>``). Use for your evolving world model: parsing,
        entity extraction, predict(state, action) functions. Syntax-checked on
        write; call load_helpers() afterwards to (re)load into ``self.h``.
        Use preflight_model(actions) to test the exact runtime JSON-latent path;
        direct predict(encode(grid)) alone does not test that contract."""
        import textwrap

        if not _HELPER_NAME.match(filename):
            return f"REJECTED: filename {filename!r} must match [a-z0-9_]+.py"
        source = textwrap.dedent(source)  # keep the written file importable (see _check_helper_ast)
        err = _check_helper_ast(source)
        if err is not None:
            return f"REJECTED: {err}"
        (self.helpers_dir / filename).write_text(source)
        return f"wrote helpers/{filename} ({len(source)} chars). Call load_helpers() to use it."

    def save_procedure(self, filename: str, source: str, applicability: str,
                       predicted_effect: str, stop_condition: str,
                       evidence_refs: list[str]) -> dict:
        """Compile a supported discovery into guarded executable state.

        The source passes the same AST jail as every helper. Metadata states
        when it applies, what it predicts, and when execution must stop.
        """
        fields = (applicability, predicted_effect, stop_condition)
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise ValueError("applicability, predicted_effect, and stop_condition are required")
        if not isinstance(evidence_refs, list) or not 1 <= len(evidence_refs) <= 12:
            raise ValueError("supply one to twelve evidence references")
        outcome = self.write_helper(filename, source)
        if outcome.startswith("REJECTED:"):
            raise ValueError(outcome)
        path = self.helpers_dir / filename
        record = {
            "schema": "executable-procedure-v1", "filename": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "applicability": applicability, "predicted_effect": predicted_effect,
            "stop_condition": stop_condition, "evidence_refs": evidence_refs,
            "turn": (self._latest_state() or {}).get("turn"),
        }
        append(self.ipc_dir / "procedures.jsonl", record)
        return record

    def read_helper(self, filename: str) -> str:
        """Read back a helper module's source from the workspace."""
        p = self.helpers_dir / filename
        if not _HELPER_NAME.match(filename) or not p.exists():
            return f"no such helper: {filename!r} (have: {[q.name for q in self.helpers_dir.glob('*.py')]})"
        return p.read_text()

    def load_helpers(self) -> str:
        """(Re)load every ``helpers/*.py`` module into ``self.h.<module_name>``.
        Returns the loaded modules and their public functions. Call at the start
        of a turn to reuse the world model you built in earlier turns."""
        report = []
        for f in sorted(self.helpers_dir.glob("*.py")):
            # Re-check before executing: a seeded/older helper could predate the
            # write-time guard, so importing it unchecked would reopen the hole.
            err = _check_helper_ast(f.read_text())
            if err is not None:
                report.append(f"{f.name}: BLOCKED — {err}")
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"arc_helper_{f.stem}", f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception as e:
                report.append(f"{f.name}: FAILED to load — {e!r}")
                continue
            setattr(self.h, f.stem, mod)
            fns = [n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n))]
            report.append(f"{f.name} -> self.h.{f.stem} ({', '.join(fns) or 'no functions'})")
        return "\n".join(report) or "no helper modules yet — write_helper() to create one."

    # ----------------------------------------------------------------- turn

    @hidden
    @strategy(CodeActStrategy(config=_ARC_CELL_CONFIG))
    async def handle(self, notification: dict[str, list]) -> RespondResult:
        """One interactive-grid-game solving turn. Follow the <arc_skill> context block.

        ``notification`` maps channel name to new items:

        - "game_states": raw JSON lines from the harness — parse with
          json.loads(line). Fields: turn, step, level, state, levels_completed,
          available_actions, action_results (what each action of your previous
          submission actually did: action, reward, diff_pixels, level_completed,
          state, event_frame_count), grid_rows (64 strings of 64 hex digits,
          row 0 = top; parse with self.grid_array), optionally note (harness
          warnings) and prediction_feedback. diff_summary and animation frames
          are NOT supplied here: use self.last_transition() for measured changes
          and self.last_transition(include_grids=True) for exact before/after.
          Multiple
          queued states mean your view was stale — act on the LAST one. If the
          notification content is missing or truncated, call
          self.latest_state() — it always returns the full current state.
        - "user_messages": operational guidance injected into the run.
          It OVERRIDES your current plan — acknowledge it with message() and
          follow it.

        Per-turn workflow:
        1. Read self.observations() — what the host has already established:
           the status band, actions already known to be inert or fatal FROM HERE,
           and how your last predicted_path scored. It costs no tokens to rebuild.
        2. Parse the newest state; check your last predicted_path against
           self.last_transition(). Keep measured facts (step/hash/cells) separate
           from hypotheses about objects, dynamics, or goals. A passing helper
           test is not a confirmed rule. When measurements contradict a claim,
           correct or withdraw that claim in the helper/notes before relying on it.
        3. Analyze the grid in code (self.grid_array + numpy; self.load_helpers()
           to reuse your world model; self.trajectory() for the full history of
           executed actions: each entry has before_grid_rows and AFTER grid_rows).
           After a helper change, use preflight_model(actions) in the same cell
           as the next test when practical. Once the decisive evidence is
           measured, take a justified discriminating action instead of repeatedly
           printing the same observation. Unexplained details may stay provisional.
        4. Record what you observed per <knowledge_api> — the lab book, not the plan.
        5. END the turn by calling self.submit_actions([...],
           rationale="predict: ...", predicted_path=[...]) — one entry per action
           naming the outcome you expect. A single action is always allowed; a
           batch has to be earned. A successful submit yields until the next state
           arrives (no separate return_result call is needed; do steps 1-4 first).

        The host ends a game: on WIN or GAME_OVER it stops sending states, and
        when the model budget is spent it stops admitting calls. There is no
        "stuck" verdict for you to reach. Every turn ends with submit_actions;
        never call return_result.
        """
        # Prefill: when visual input is on (only|additive), show the current grid
        # PNG so the LLM sees the colored picture this turn. In 'only' the harness
        # has dropped the hex grid_rows from the state (image replaces the grid); in
        # 'additive' both are present.
        if self._visual != "off":
            show(self.grid_image())
        ...


class MemArcSolverAgent(ArcSolverBase):
    """You are an interactive grid-game-solving agent. You discover the unknown rules of an
    interactive 64x64 grid game by experimenting, then solve all its levels.
    Follow the <arc_skill> context block. Your accumulated knowledge lives in
    your long-term MEMORY (self.memory.*) — recall before deciding, remember
    after observing, reflect to consolidate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The launcher registers MemorySkill (self.memory) after bootstrap —
        # this block documents the contract the skill expects the LLM to follow.
        self.context["knowledge_api"] = (
            "KNOWLEDGE API (memory variant):\n"
            "- self.memory.recall(query, k=5) — associative recall (also happens\n"
            "  spontaneously each turn: see <recalled_memories>).\n"
            "- self.memory.search(query, k=5) — term-focused lookup.\n"
            "- self.memory.remember(content, type=..., importance=..., tags=[...]) — write one\n"
            "  durable fact. Types: 'info' (observations, action semantics),\n"
            "  'skill' (verified reusable procedure/strategy: include applicability),\n"
            "  'todo' (open experiment to run; close with update_memory(id, status='DONE')),\n"
            "  'episode' (what happened in a level attempt).\n"
            "- self.memory.update_memory(id, ...) — revise/close; use for hypothesis\n"
            "  status changes (confirmed/contradicted — say which evidence).\n"
            "- self.memory.associate(a_id, b_id, relation) — link related facts.\n"
            f"- self.memory.reflect() — consolidate. Call after each level completion\n"
            f"  and roughly every {self.reflect_every} turns; set self.v.last_reflect_turn.\n"
            "Store CURATED knowledge, not raw grids: one fact per memory, with the\n"
            "evidence. Tag with the game id and level."
        )

    @hidden
    def _knowledge_status_line(self) -> str:
        last = self.vars.get("last_reflect_turn") if hasattr(self.vars, "get") else None
        state = self._latest_state() or {}
        turn = state.get("turn", 0) or 0
        if last is None:
            return f"memory: no reflection yet (reflect every ~{self.reflect_every} turns)"
        return f"memory: last reflection at turn {last} (now {turn}; reflect every ~{self.reflect_every})"


class MdArcSolverAgent(ArcSolverBase):
    """You are an interactive grid-game-solving agent. You discover the unknown rules of an
    interactive 64x64 grid game by experimenting, then solve all its levels.
    Follow the <arc_skill> context block.

    Your PLAN lives in code: encode(grid)->z and predict(z, action)->z\' in a
    helper module (write_helper / load_helpers / self.h.*). Only the helper can
    constrain what you submit. Your markdown files (read_notes / append_notes /
    write_notes) are the lab book — they record what you observed and why, and a
    belief that stays there cannot stop a bad batch. When a note and the helper
    disagree, the helper is what the game obeys, so fix the helper.

    WHERE THE GENERIC CODEACT INSTRUCTIONS BELOW DISAGREE WITH THIS, THIS WINS:

    - You end a turn with self.submit_actions(...), never with return_result.
      The host decides when a game is over; you do not.
    - "Use LLM reasoning, answer directly" does not apply here. Every claim about
      this grid is settled by numpy in a cell, not by looking at hex and deciding.
    - "Keep cells short" is about the 60s kill, not a style rule. A search over
      predict() is exactly what a cell is for; write it.
    - "Helpers" in the CodeAct text means throwaway REPL locals. YOUR helpers are
      persistent modules written with write_helper and reloaded with
      load_helpers — those are the world model.
    - self.grid_array, self.submit_actions and self.observations are synchronous.
      Do not await them.
    """

    knowledge_dir: Annotated[Path, hidden]  # path embeds the run dir — hide it

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge_dir = self.workspace / "knowledge"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.context["knowledge_api"] = (
            "KNOWLEDGE API (markdown variant):\n"
            "- self.list_notes() / self.read_notes(name) — read before deciding.\n"
            "- self.append_notes(name, content) — add entries after observing.\n"
            "- self.write_notes(name, content) — rewrite a file when curating.\n"
            "Keep these files (create on demand):\n"
            "- step_log.md — per-turn record: state, hypothesis checked, result,\n"
            "  and a Carry-Forward section with everything worth preserving.\n"
            "- hypotheses.md — claim | evidence steps | tested scope | open conditions.\n"
            "  Revise dependent claims after contradictory evidence; retain qualifications.\n"
            "- level_log.md — on completion: successful starting conditions and sequence,\n"
            f"  supported rules and untested generalization. Curate/rewrite files roughly every {self.reflect_every}\n"
            "  turns (this is your reflection); set self.v.last_reflect_turn."
        )

    @hidden
    def _knowledge_status_line(self) -> str:
        names = [p.name for p in sorted(self.knowledge_dir.glob("*.md"))]
        last = self.vars.get("last_reflect_turn") if hasattr(self.vars, "get") else None
        return f"notes: {names or 'none yet'} | last curation turn: {last}"

    def list_notes(self) -> list[str]:
        """List the knowledge markdown files in the workspace."""
        return [p.name for p in sorted(self.knowledge_dir.glob("*.md"))]

    def read_notes(self, name: str) -> str:
        """Read one knowledge file (e.g. 'hypotheses.md')."""
        if not _NOTE_NAME.match(name):
            return f"REJECTED: name {name!r} must match [a-z0-9_-]+.md"
        p = self.knowledge_dir / name
        if not p.exists():
            return f"(no file {name} yet — have {self.list_notes()})"
        body = p.read_text()
        self._integrity.read(name, body)
        return body

    def append_notes(self, name: str, content: str) -> str:
        """Append an entry to a knowledge file (created if missing)."""
        if not _NOTE_NAME.match(name):
            return f"REJECTED: name {name!r} must match [a-z0-9_-]+.md"
        p = self.knowledge_dir / name
        expected = (p.read_text() if p.exists() else "") + content.rstrip() + "\n\n"
        with p.open("a") as f:
            f.write(content.rstrip() + "\n\n")
        self._integrity.written(name, expected, p.read_text())
        return f"appended {len(content)} chars to {name}"

    def write_notes(self, name: str, content: str) -> str:
        """Overwrite a knowledge file — use when curating/consolidating."""
        if not _NOTE_NAME.match(name):
            return f"REJECTED: name {name!r} must match [a-z0-9_-]+.md"
        p = self.knowledge_dir / name
        p.write_text(content)
        self._integrity.written(name, content, p.read_text())
        return f"wrote {name} ({len(content)} chars)"
