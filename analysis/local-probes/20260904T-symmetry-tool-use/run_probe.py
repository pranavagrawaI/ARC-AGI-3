#!/usr/bin/env python3
"""q6-tool-use-contact. Helper in the cell. Two frames. OpenRouter, thinking off."""
from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import shutil
import sys
import time
import traceback
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROBE = Path(__file__).resolve().parent
SEED = PROBE.parent / "20260904T-symmetry-q5" / "seed"
sys.path.insert(0, str(SEED))
import symmetry as S  # noqa: E402

DIGEST = "7e0c1f1cdcd0d594b1bd21dee0fd74c9624a5a66141575d2d9f947fc298b433f"
MODEL = "qwen/qwen3.8-27b"
CAP_USD = 0.12
ROUNDS = 8
CELL_SEC = 15
STDOUT_CAP = 8000
PROMPT_USD = 0.42e-6
COMPLETION_USD = 3.0e-6
CACHE_USD = 0.085e-6
MIN_REMAINING = 0.02
MIN_FOR_B = 0.04

WRAP_NAMES = (
    "similar",
    "congruent",
    "directly_congruent",
    "canon",
    "transporter",
    "orbit",
    "primitive",
)
ALLOWED_IMPORTS = frozenset(
    {"numpy", "collections", "itertools", "math", "functools", "copy", "re", "heapq", "typing"}
)
LEAK = (
    "ls20",
    "ar25",
    "hud",
    "door",
    "level 1",
    "level_1",
    "n=45",
    "n=24",
    "rows 39",
    "colour-4",
    "colour-11",
    "playfield",
    "transporter onto",
)

# Scoring only. Never serialised into a request.
ORACLE = {
    "A": {
        "id": "ar25-turn0",
        "frame": "frames/ar25_turn0.json",
        "pair": ((45, (4,)), (45, (11,))),
        "need": "true_or_transporter",
    },
    "B": {
        "id": "ls20-l1-step32",
        "frame": "frames/ls20_l1_step32.json",
        "pair": ((24, (9,)), (6, (9,))),
        "need": "similar",
    },
}

CALLS: list[dict] = []


class CellTimeout(Exception):
    pass


def remaining(key: str) -> float:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.load(r).get("data", {})
    return float(d["limit_remaining"])


def cost_of(usage: dict) -> float:
    if not usage:
        return 0.0
    if usage.get("cost") is not None:
        return float(usage["cost"])
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = 0
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    billed = max(prompt - cached, 0)
    return billed * PROMPT_USD + cached * CACHE_USD + completion * COMPLETION_USD


def fig_summary(fig) -> dict | None:
    if not isinstance(fig, dict) or not fig:
        return None
    colours = tuple(sorted({int(v) for v in fig.values()}))
    return {"n": len(fig), "colours": colours}


def wrap(fn, name):
    def inner(*a, **k):
        rec = {
            "fn": name,
            "kwargs": {
                kk: (vv if isinstance(vv, (str, bool, int, float)) else repr(vv)[:80])
                for kk, vv in k.items()
                if kk in ("translate", "scale", "colours", "point")
            },
            "figures": [fig_summary(x) for x in a[:2]],
        }
        out = fn(*a, **k)
        if name == "canon":
            rec["result"] = "canon"
        elif name == "primitive":
            core, kk = out
            rec["result"] = {"core_n": len(core) if isinstance(core, dict) else None, "k": kk}
        elif name == "transporter":
            rec["result"] = "nonempty" if out else "empty"
            rec["truth"] = bool(out)
        elif name == "orbit":
            rec["result"] = len(out) if hasattr(out, "__len__") else repr(out)[:80]
        else:
            rec["result"] = out
            rec["truth"] = bool(out) if isinstance(out, bool) else None
        CALLS.append(rec)
        return out

    inner.__name__ = name
    inner.__doc__ = fn.__doc__
    return inner


def install_wraps() -> None:
    for name in WRAP_NAMES:
        setattr(S, name, wrap(getattr(S, name), name))


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"{name} is not available in this cell")
    return __import__(name, globals, locals, fromlist, level)


def cell_ns(grid) -> dict:
    import builtins

    b = {
        k: getattr(builtins, k)
        for k in dir(builtins)
        if not k.startswith("_")
    }
    b["__import__"] = safe_import
    b.pop("open", None)
    b.pop("exec", None)
    b.pop("eval", None)
    b.pop("compile", None)
    b.pop("input", None)
    b.pop("breakpoint", None)
    b.pop("exit", None)
    b.pop("quit", None)
    return {
        "__builtins__": b,
        "GRID": np.array(grid, dtype=int),
        "np": np,
        "numpy": np,
        "symmetry": S,
    }


def run_cell(code: str, ns: dict) -> dict:
    def on_alarm(signum, frame):
        raise CellTimeout(f"cell exceeded {CELL_SEC}s")

    buf = io.StringIO()
    err = io.StringIO()
    t0 = time.time()
    old = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(CELL_SEC)
    try:
        with redirect_stdout(buf), redirect_stderr(err):
            exec(compile(code, "<cell>", "exec"), ns, ns)
        status = "ok"
        tb = ""
    except CellTimeout as e:
        status = "timeout"
        tb = str(e)
    except Exception:
        status = "error"
        tb = traceback.format_exc(limit=8)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    out = buf.getvalue()
    if len(out) > STDOUT_CAP:
        out = out[:STDOUT_CAP] + f"\n<truncated-output n={len(out)}>"
    return {
        "status": status,
        "stdout": out,
        "stderr": err.getvalue()[:2000],
        "traceback": tb[-4000:],
        "seconds": round(time.time() - t0, 3),
    }


def hex_grid(grid) -> str:
    alphabet = "0123456789abcdef"
    lines = []
    for row in grid:
        lines.append("".join(alphabet[int(c)] if 0 <= int(c) < 16 else "?" for c in row))
    return "\n".join(lines)


def sent_source(source: str) -> str:
    # The seeded file names a game in a comment. The card forbids game names
    # in the model prompt. Disk digest is unchanged.
    return source.replace("ar25's equal-armed piece", "an equal-armed piece")


def system_prompt(source: str) -> str:
    return (
        "You have one tool, execute_python. In the interpreter:\n"
        "- GRID is a 2-d numpy array of ints (the frame)\n"
        "- np is numpy\n"
        "- symmetry is the module whose source follows. You do not edit it.\n"
        "\n"
        "A figure is a dict {(y, x): colour}. The module compares figures under "
        "four quotients you choose (point, translate, scale, colours).\n"
        "\n"
        "Determine whether any two figures on GRID are similar or congruent. "
        "Call the module. Print what you compared and what it returned. "
        "Do not claim a pair without a call.\n"
        "When finished, stop calling tools and write:\n"
        "PAIR: yes|no\n"
        "FIGURES: <what you compared>\n"
        "QUOTIENTS: <point, translate, scale, colours>\n"
        "\n"
        "Module source:\n"
        "```python\n"
        f"{sent_source(source)}"
        "\n```\n"
    )


def user_prompt(grid) -> str:
    h, w = len(grid), len(grid[0])
    return (
        f"GRID shape {h}x{w} is already in the interpreter.\n"
        "Digits 0-9 then a=10, b=11, ... one character per cell, one row per line:\n"
        f"{hex_grid(grid)}\n"
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Run Python. GRID, np, and symmetry are in scope.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }
]


def leak_in(obj) -> list[str]:
    blob = json.dumps(obj).lower()
    return [w for w in LEAK if w in blob]


def chat(key: str, messages: list, timeout: int = 120) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0,
        "max_tokens": 2048,
        "reasoning": {"enabled": False},
    }
    leaked = leak_in({"messages": messages})
    if leaked:
        raise RuntimeError(f"oracle leak in request: {leaked}")
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"openrouter HTTP {e.code}: {body}") from e


def pair_hit(calls: list[dict], spec: dict) -> bool:
    (n1, c1), (n2, c2) = spec["pair"]
    need = spec["need"]

    def match_figs(figs):
        if len(figs) != 2 or any(f is None for f in figs):
            return False
        a, b = figs
        left = (a["n"], tuple(a["colours"]))
        right = (b["n"], tuple(b["colours"]))
        return {left, right} == {(n1, tuple(c1)), (n2, tuple(c2))}

    for c in calls:
        if not match_figs(c.get("figures") or []):
            continue
        if need == "similar":
            if c["fn"] == "similar" and c.get("truth") is True:
                return True
        elif need == "true_or_transporter":
            if c["fn"] in ("similar", "congruent", "directly_congruent") and c.get("truth") is True:
                return True
            if c["fn"] == "transporter" and c.get("truth") is True:
                return True
    return False


def extraction_of(codes: list[str]) -> str:
    blob = "\n".join(codes)
    window = any(
        x in blob
        for x in ("GRID[", "grid[", "[:, ", "[r0", "bbox", "window", "slice(")
    )
    cc = any(
        x in blob
        for x in (
            "nonzero",
            "connected",
            "ndimage",
            "stack",
            "deque",
            "neighbors",
            "N4",
            "flood",
            "component",
        )
    )
    if window and cc:
        return "cc+window"
    if window:
        return "window"
    if cc:
        return "cc"
    return "other"


def score_trial(label: str, cells: list[dict], calls: list[dict]) -> dict:
    codes = [c["code"] for c in cells]
    return {
        "label": label,
        "oracle_id": ORACLE[label]["id"],
        "U1": sum(1 for c in calls if c["fn"] in WRAP_NAMES),
        "U2": pair_hit(calls, ORACLE[label]),
        "extraction": extraction_of(codes),
        "n_cells": len(cells),
        "n_calls": len(calls),
        "fns": sorted({c["fn"] for c in calls}),
        "true_calls": [c for c in calls if c.get("truth") is True],
        "calls": calls,
    }


def run_trial(key: str, label: str, grid, source: str, out_dir: Path, spend: SimpleNamespace) -> dict:
    ns = cell_ns(grid)
    messages = [
        {"role": "system", "content": system_prompt(source)},
        {"role": "user", "content": user_prompt(grid)},
    ]
    cells = []
    trial_calls_before = len(CALLS)
    raw = []
    for rnd in range(ROUNDS):
        rem = remaining(key)
        spend.remaining = rem
        if rem < MIN_REMAINING or spend.usd >= CAP_USD:
            raw.append({"stop": "credit", "remaining": rem, "usd": spend.usd})
            break
        payload = chat(key, messages)
        raw.append(payload)
        usage = payload.get("usage") or {}
        spend.usd += cost_of(usage)
        spend.calls += 1
        rt = int(usage.get("reasoning_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        rt += int(details.get("reasoning_tokens") or 0)
        if rt > 0:
            raise RuntimeError(f"reasoning_tokens={rt}")
        msg = payload["choices"][0]["message"]
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name")
            if name != "execute_python":
                result = {"status": "error", "stdout": "", "traceback": f"unknown tool {name}"}
                code = ""
            else:
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"code": fn.get("arguments") or ""}
                code = args.get("code") or ""
                mark = len(CALLS)
                result = run_cell(code, ns)
                result["new_calls"] = CALLS[mark:]
            cells.append({"round": rnd, "code": code, **{k: result[k] for k in result if k != "new_calls"}, "new_calls": result.get("new_calls", [])})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": json.dumps(
                        {
                            "status": result["status"],
                            "stdout": result.get("stdout", ""),
                            "stderr": result.get("stderr", ""),
                            "traceback": result.get("traceback", ""),
                        }
                    )[: STDOUT_CAP + 500],
                }
            )
    trial_calls = deepcopy(CALLS[trial_calls_before:])
    (out_dir / f"{label}-raw.json").write_text(json.dumps(raw, indent=2) + "\n")
    (out_dir / f"{label}-cells.json").write_text(json.dumps(cells, indent=2) + "\n")
    (out_dir / f"{label}-messages.json").write_text(
        json.dumps(
            [
                {k: v for k, v in m.items() if k != "reasoning"}
                for m in messages
            ],
            indent=2,
        )
        + "\n"
    )
    scored = score_trial(label, cells, trial_calls)
    scored["spend_usd"] = spend.usd
    scored["remaining"] = spend.remaining
    scored["llm_calls"] = spend.calls
    (out_dir / f"{label}-score.json").write_text(json.dumps(scored, indent=2, default=str) + "\n")
    return scored


def components(grid, skip_colours=(), skip_cell=None):
    h, w = len(grid), len(grid[0])
    seen, out = set(), []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            if skip_cell and skip_cell(r, c):
                continue
            col = int(grid[r][c])
            if col in skip_colours:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells = {}
            while stack:
                y, x = stack.pop()
                cells[(y, x)] = col
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < h and 0 <= nx < w) or (ny, nx) in seen:
                        continue
                    if skip_cell and skip_cell(ny, nx):
                        continue
                    if int(grid[ny][nx]) != col:
                        continue
                    seen.add((ny, nx))
                    stack.append((ny, nx))
            out.append(cells)
    return out


def self_check(frames) -> None:
    CALLS.clear()
    g = frames["A"]
    ccs = components(g, skip_colours={9, 10}, skip_cell=lambda r, c: c == 63 or r == 63)
    by = {}
    for cc in ccs:
        by.setdefault(next(iter(cc.values())), []).append(cc)
    fig4 = max(by[4], key=len)
    fig11 = max(by[11], key=len)
    assert S.congruent(fig4, fig11, translate=True, colours="any") is True
    assert pair_hit(CALLS, ORACLE["A"]) is True
    CALLS.clear()
    g = np.array(frames["B"])
    hud = {(y, x): 9 for y in range(54, 64) for x in range(0, 12) if g[y, x] == 9}
    play = {(y, x): 9 for y in range(39, 46) for x in range(13, 21) if g[y, x] == 9}
    assert len(hud) == 24 and len(play) == 6
    assert S.similar(hud, play, colours="any") is True
    assert pair_hit(CALLS, ORACLE["B"]) is True
    CALLS.clear()
    src = (SEED / "symmetry.py").read_text()
    leaked = leak_in({"messages": [{"role": "system", "content": system_prompt(src)}]})
    if leaked:
        raise RuntimeError(f"self-check leak {leaked}")


def main() -> int:
    key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("source .env first")
    source = (SEED / "symmetry.py").read_text()
    got = hashlib.sha256(source.encode()).hexdigest()
    if got != DIGEST:
        raise SystemExit(f"digest mismatch {got}")
    frames = {
        "A": json.loads((PROBE / "frames" / "ar25_turn0.json").read_text()),
        "B": json.loads((PROBE / "frames" / "ls20_l1_step32.json").read_text()),
    }
    install_wraps()
    self_check(frames)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = PROBE / "results" / stamp
    out.mkdir(parents=True)
    shutil.copy(PROBE / "EXPERIMENT-CARD.md", out / "EXPERIMENT-CARD.md")
    start = remaining(key)
    spend = SimpleNamespace(usd=0.0, remaining=start, calls=0)
    (out / "spend-start.json").write_text(
        json.dumps({"t": time.time(), "remaining": start, "cap": CAP_USD}, indent=2) + "\n"
    )
    if start < MIN_FOR_B:
        raise SystemExit(f"remaining {start:.4f} below {MIN_FOR_B}")
    results = []
    for label in ("A", "B"):
        if label == "B" and remaining(key) < MIN_FOR_B:
            results.append({"label": "B", "skipped": "credit"})
            break
        results.append(run_trial(key, label, frames[label], source, out, spend))
    end = remaining(key)
    summary = {
        "digest": got,
        "start_remaining": start,
        "end_remaining": end,
        "spent_credit": start - end,
        "spent_accounted": spend.usd,
        "llm_calls": spend.calls,
        "trials": [
            {
                "label": r.get("label"),
                "oracle_id": r.get("oracle_id"),
                "U1": r.get("U1"),
                "U2": r.get("U2"),
                "extraction": r.get("extraction"),
                "fns": r.get("fns"),
                "n_cells": r.get("n_cells"),
                "skipped": r.get("skipped"),
            }
            for r in results
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("run:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
