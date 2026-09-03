"""Measure whether the served model can read a 64x64 ARC frame from a rendered image.

The V15 deployment sends the actor a pipe-delimited text grid and nothing else, and the
model spent 48,000 reasoning tokens hand-parsing it without emitting a tool call. Tycho's
`agent/vision.py` claims Qwen is the one model family whose encoder resolves such a grid
exactly: patch 16 x spatial-merge 2 = 32 px per merged visual token, so a 64x64 grid drawn
at 32 px/cell is 2048x2048 -> 4,096 visual tokens, one per ARC cell. That claim is
conditional on the serving processor admitting 4,194,304 pixels without downscaling, which
is a property of *our* SGLang deployment and has never been measured.

This probe measures it two ways against the same pinned frames:

  * directly - how many prompt tokens each rendered scale actually costs, by differencing
    an identical request with and without the image (`measure_visual_tokens`);
  * behaviourally - per-cell readback, palette recovery, and bounding-box accuracy across
    input modalities, scored against the frame array (`run_readback_tasks`).

It answers one question and refuses the neighbouring ones: it does not play, does not
score RHAE, and says nothing about whether an image-fed actor plays better.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROBE_SCHEMA = "tycho-nooa-vision-readback-v1"

# Tycho's actor prompt names cell values by these colors, so the probe must speak the same
# language when it asks the model about a color.
COLORMAP = (
    "Color palette (cell value -> rendered color): 0=white, 1=light-grey, 2=grey, "
    "3=dark-grey, 4=charcoal, 5=black, 6=magenta, 7=pink, 8=red, 9=blue, a=cyan, "
    "b=yellow, c=orange, d=maroon, e=green, f=purple. Treat them as opaque IDs; the "
    "names are just for reference."
)

# The environment's canonical 16-color palette, byte-identical to the reference renderer
# (nooa examples/arc_agi_3/solver_agent.py _PALETTE_16). Indices 0-5 are a greyscale ramp,
# which is what makes ls20 a fair perception test rather than a high-contrast one.
PALETTE_16 = (
    (255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102),
    (51, 51, 51), (0, 0, 0), (229, 58, 163), (255, 123, 204),
    (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0),
    (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214),
)

RENDER_SCALES = (6, 16, 32)
# Serving stacks wrap an image in a couple of delimiter tokens; see measure_visual_tokens.
VISUAL_FRAMING_TOKEN_ALLOWANCE = 8
CELL_SAMPLE_SIZE = 12


# --------------------------------------------------------------------------------------
# Conditions and tasks
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One input modality under one generation contract.

    `text_think` is the incumbent: exactly what V15 sent the actor. Every other condition
    disables thinking and caps output at 2,048 tokens, which is the regime the deployment
    can actually afford at ~41 tok/s.
    """

    name: str
    show_text: bool
    render_scale: int | None
    disable_thinking: bool
    max_tokens: int
    frame_limit: int | None = None  # None means every pinned frame


CONDITIONS = (
    Condition("text_nothink", True, None, True, 2048),
    Condition("image06_nothink", False, 6, True, 2048),
    Condition("image16_nothink", False, 16, True, 2048),
    Condition("image32_nothink", False, 32, True, 2048),
    Condition("both32_nothink", True, 32, True, 2048),
    # The incumbent arm is expensive (up to 390 s per request at 41 tok/s), so it runs on
    # the first frame only. It establishes what the current deployment pays for an answer.
    Condition("text_think", True, None, False, 16000, frame_limit=1),
)

TASK_NAMES = ("cells", "palette", "bbox")


def sample_cells(grid: list[list[int]]) -> list[list[int]]:
    """Pick coordinates that separate the easy claim from the real one.

    Reading a cell in the middle of a 2,609-cell charcoal field proves nothing. The sample
    deliberately mixes the dominant field, the three rarest colors, the HUD rows that V15's
    actor misread as terrain, and the frame's corners.
    """
    counts = Counter(value for row in grid for value in row)
    by_frequency = [color for color, _ in counts.most_common()]
    picks: list[tuple[int, int]] = []

    def add_color(color: int, wanted: int) -> None:
        """Take `wanted` occurrences spread evenly through the colour's scan order.

        Taking the first N instead would cluster them: the dominant colour's first three
        cells are adjacent, and a model that simply guesses the dominant colour would
        collect them for free.
        """
        occurrences = [
            (r, c)
            for r, row in enumerate(grid)
            for c, value in enumerate(row)
            if value == color and (r, c) not in picks
        ]
        if not occurrences:
            return
        stride = max(1, len(occurrences) // wanted)
        for index in range(wanted):
            position = min(index * stride, len(occurrences) - 1)
            cell = occurrences[position]
            if cell not in picks:
                picks.append(cell)

    add_color(by_frequency[0], 3)
    for color in by_frequency[-3:]:
        add_color(color, 1)
    for c in (8, 32, 56):  # the colour-11 bar rows ls20 depletes one column per action
        if (61, c) not in picks:
            picks.append((61, c))
    for cell in ((0, 0), (63, 63), (31, 31)):
        if cell not in picks:
            picks.append(cell)

    return [[r, c] for r, c in sorted(dict.fromkeys(picks))[:CELL_SAMPLE_SIZE]]


def bbox_color(grid: list[list[int]]) -> int:
    """The rarest color with at least four cells: a real object, not a stray pixel."""
    counts = Counter(value for row in grid for value in row)
    eligible = [(count, color) for color, count in counts.items() if count >= 4]
    return min(eligible)[1]


def truth_for(grid: list[list[int]], task: str) -> Any:
    if task == "cells":
        return [grid[r][c] for r, c in sample_cells(grid)]
    if task == "palette":
        return sorted({value for row in grid for value in row})
    if task == "bbox":
        color = bbox_color(grid)
        cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == color]
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        return [min(rows), min(cols), max(rows), max(cols)]
    raise ValueError(f"unknown task {task!r}")


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def grid_text(grid: list[list[int]], col_stride: int = 8) -> str:
    """The exact `spaced_grid8` text V15 inlined in every actor turn.

    Byte-compatible with tycho/workspace/workspace.py grid_text so the text arm measures
    the incumbent input, not a convenient reconstruction of it.
    """
    width = len(grid[0]) if grid else 0
    blocks = [(s, min(s + col_stride - 1, width - 1)) for s in range(0, width, col_stride)]
    block_width = col_stride * 2 - 1
    lines = ["x:    " + " | ".join(f"{a:02d}-{b:02d}".ljust(block_width) for a, b in blocks)]
    for i, row in enumerate(grid):
        parts = [" ".join(f"{row[j]:x}" for j in range(a, b + 1)) for a, b in blocks]
        lines.append(f"y{i:02d}: " + " | ".join(parts))
    return "\n".join(lines)


def grid_png(grid: list[list[int]], scale: int) -> bytes:
    """Render the grid as a lossless PNG at `scale` pixels per cell.

    Written against the stdlib alone: the probe runs inside the frozen Kaggle runtime,
    where adding Pillow would change the package closure the route already qualified.
    """
    height = len(grid) * scale
    width = len(grid[0]) * scale
    rows = bytearray()
    for row in grid:
        line = bytearray([0])  # PNG filter type 0 (None) for each scanline
        for value in row:
            r, g, b = PALETTE_16[value if 0 <= value < 16 else 5]
            line.extend(bytes((r, g, b)) * scale)
        rows.extend(line * scale)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + chunk(b"IEND", b"")
    )


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# --------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------

TOOLS: dict[str, dict[str, Any]] = {
    "cells": {
        "type": "function",
        "function": {
            "name": "report_cells",
            "description": "Report the color value at each requested coordinate, in order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 15},
                        "description": "One value per requested coordinate, same order.",
                    }
                },
                "required": ["values"],
            },
        },
    },
    "palette": {
        "type": "function",
        "function": {
            "name": "report_palette",
            "description": "Report every distinct color value present in the grid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 15},
                    }
                },
                "required": ["colors"],
            },
        },
    },
    "bbox": {
        "type": "function",
        "function": {
            "name": "report_bbox",
            "description": "Report the inclusive bounding box of every cell of one color.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_row": {"type": "integer", "minimum": 0, "maximum": 63},
                    "min_col": {"type": "integer", "minimum": 0, "maximum": 63},
                    "max_row": {"type": "integer", "minimum": 0, "maximum": 63},
                    "max_col": {"type": "integer", "minimum": 0, "maximum": 63},
                },
                "required": ["min_row", "min_col", "max_row", "max_col"],
            },
        },
    },
}

SYSTEM_PROMPT = (
    "You are reading a 64x64 grid of colored cells from an interactive game. Coordinates "
    "are (row, col) with row 0 at the top and col 0 at the left, both 0-63. Answer only "
    "from what you can see in this message. Report your reading with exactly one tool "
    f"call.\n{COLORMAP}"
)


def task_instruction(task: str, grid: list[list[int]]) -> str:
    if task == "cells":
        coords = ", ".join(f"({r},{c})" for r, c in sample_cells(grid))
        return (
            f"Report the color value at each of these {CELL_SAMPLE_SIZE} coordinates, in "
            f"the order given: {coords}. Call report_cells with exactly "
            f"{CELL_SAMPLE_SIZE} values."
        )
    if task == "palette":
        return (
            "Report every distinct color value that appears anywhere in the grid. Call "
            "report_palette once."
        )
    if task == "bbox":
        color = bbox_color(grid)
        return (
            f"Find every cell whose color value is {color} (hex {color:x}) and report the "
            "inclusive bounding box that contains all of them. Call report_bbox once."
        )
    raise ValueError(f"unknown task {task!r}")


def build_request(
    model: str,
    condition: Condition,
    grid: list[list[int]],
    task: str,
    thinking_control: str = "chat_template_kwargs",
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": task_instruction(task, grid)}]
    if condition.show_text:
        content.append({"type": "text", "text": "Grid:\n" + grid_text(grid)})
    if condition.render_scale is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url(grid_png(grid, condition.render_scale))},
            }
        )

    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "tools": [TOOLS[task]],
        "tool_choice": "required",
        "max_tokens": condition.max_tokens,
        "temperature": 0.0,
    }
    if condition.disable_thinking:
        request.update(THINKING_CONTROLS[thinking_control])
    return request


# How to suppress reasoning, per route. `extra_body` is an OpenAI *SDK* concept: the SDK
# flattens it into the top-level body before sending, so a raw HTTP client that ships a
# literal "extra_body" key sends something no server reads. Measured against
# qwen/qwen3.8-27b on OpenRouter: nested extra_body and top-level chat_template_kwargs
# both left reasoning on (43 completion tokens, 129 reasoning chars, identical to
# baseline); `reasoning: {enabled: false}` cut it to 3 tokens and no reasoning.
THINKING_CONTROLS: dict[str, dict[str, Any]] = {
    # SGLang serving Qwen3.x honours this at the top level; V15's Solver reached it
    # through LiteLLM, which flattens extra_body for the same effect.
    "chat_template_kwargs": {"chat_template_kwargs": {"enable_thinking": False}},
    # OpenRouter's own control, needed because upstream providers ignore the above.
    "openrouter_reasoning": {"reasoning": {"enabled": False}},
}


REQUESTS_ISSUED = Counter()


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST one chat completion, counting every attempt.

    The outer runtime reconciles this count against SGLang's own access log, so an
    unreported retry would show up as a discrepancy rather than pass unnoticed.
    """
    REQUESTS_ISSUED["attempted"] += 1
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    REQUESTS_ISSUED["succeeded"] += 1
    return parsed


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def parse_tool_arguments(response: dict[str, Any]) -> dict[str, Any] | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    arguments = calls[0].get("function", {}).get("arguments")
    if not isinstance(arguments, str):
        return None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def score(task: str, truth: Any, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Score one answer. A malformed or absent tool call scores zero, never None."""
    if arguments is None:
        return {"answered": False, "exact": False, "accuracy": 0.0}

    if task == "cells":
        values = arguments.get("values")
        if not isinstance(values, list) or len(values) != len(truth):
            return {"answered": True, "exact": False, "accuracy": 0.0, "malformed": True}
        hits = sum(1 for got, want in zip(values, truth) if got == want)
        return {
            "answered": True,
            "exact": hits == len(truth),
            "accuracy": round(hits / len(truth), 4),
            "reported": values,
        }

    if task == "palette":
        colors = arguments.get("colors")
        if not isinstance(colors, list):
            return {"answered": True, "exact": False, "accuracy": 0.0, "malformed": True}
        got, want = set(colors), set(truth)
        union = got | want
        return {
            "answered": True,
            "exact": got == want,
            "accuracy": round(len(got & want) / len(union), 4) if union else 1.0,
            "reported": sorted(got),
        }

    if task == "bbox":
        keys = ("min_row", "min_col", "max_row", "max_col")
        if not all(isinstance(arguments.get(k), int) for k in keys):
            return {"answered": True, "exact": False, "accuracy": 0.0, "malformed": True}
        reported = [arguments[k] for k in keys]
        return {
            "answered": True,
            "exact": reported == list(truth),
            "accuracy": round(box_iou(reported, list(truth)), 4),
            "reported": reported,
        }

    raise ValueError(f"unknown task {task!r}")


def box_iou(a: list[int], b: list[int]) -> float:
    r0, c0 = max(a[0], b[0]), max(a[1], b[1])
    r1, c1 = min(a[2], b[2]), min(a[3], b[3])
    if r1 < r0 or c1 < c0:
        return 0.0
    overlap = (r1 - r0 + 1) * (c1 - c0 + 1)
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return overlap / (area_a + area_b - overlap)


# --------------------------------------------------------------------------------------
# The two measurements
# --------------------------------------------------------------------------------------


def measure_visual_tokens(
    url: str,
    model: str,
    grid: list[list[int]],
    timeout_s: float,
    *,
    headers: dict[str, str] | None = None,
    thinking_control: str = "chat_template_kwargs",
) -> dict[str, Any]:
    """Difference identical prompts with and without the image to price each render scale.

    This is the gating measurement. If scale 32 does not cost about 4,096 prompt tokens,
    SGLang's processor downscaled the 2048x2048 render and `lossless_cells` is false on
    this route regardless of how the behavioural arms score.
    """
    probe_text = "Reply with report_palette([0])."

    def cost(scale: int | None) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": probe_text}]
        if scale is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url(grid_png(grid, scale))},
                }
            )
        response = post_json(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "tools": [TOOLS["palette"]],
                "tool_choice": "required",
                "max_tokens": 16,
                "temperature": 0.0,
                **THINKING_CONTROLS[thinking_control],
            },
            timeout_s,
            headers,
        )
        usage = response.get("usage") or {}
        return {"prompt_tokens": usage.get("prompt_tokens")}

    baseline = cost(None)["prompt_tokens"]
    measurements = []
    for scale in RENDER_SCALES:
        pixels = (64 * scale) ** 2
        prompt_tokens = cost(scale)["prompt_tokens"]
        observed = None
        if isinstance(prompt_tokens, int) and isinstance(baseline, int):
            observed = prompt_tokens - baseline
        # Qwen3.x merges a 32x32 pixel block into one visual token.
        expected = (64 * scale // 32) ** 2 if scale >= 32 else None
        measurements.append(
            {
                "render_scale": scale,
                "image_pixels": pixels,
                "image_side_px": 64 * scale,
                "prompt_tokens": prompt_tokens,
                "observed_visual_tokens": observed,
                "expected_visual_tokens_if_lossless": expected,
                # Allow a few tokens of image framing: measured against
                # qwen/qwen3.8-27b, scale 6/16/32 cost 146/1026/4098 tokens for an
                # expected 144/1024/4096, i.e. exactly one merged visual token per ARC
                # cell plus two delimiters. Demanding == 4096 called that a downscale.
                "framing_token_allowance": VISUAL_FRAMING_TOKEN_ALLOWANCE,
                "one_token_per_cell": (
                    None
                    if scale != 32 or observed is None
                    else 0 <= observed - 4096 <= VISUAL_FRAMING_TOKEN_ALLOWANCE
                ),
            }
        )
    return {"text_only_prompt_tokens": baseline, "scales": measurements}


def run_readback_tasks(
    url: str,
    model: str,
    frames: list[dict[str, Any]],
    timeout_s: float,
    deadline: float,
    *,
    headers: dict[str, str] | None = None,
    thinking_control: str = "chat_template_kwargs",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        used_frames = frames[: condition.frame_limit] if condition.frame_limit else frames
        for frame in used_frames:
            grid = frame["grid"]
            for task in TASK_NAMES:
                if time.monotonic() > deadline:
                    records.append(
                        {
                            "condition": condition.name,
                            "record_index": frame["record_index"],
                            "task": task,
                            "skipped": "probe deadline reached",
                        }
                    )
                    continue
                records.append(
                    run_one(
                        url, model, condition, frame, task, grid, timeout_s,
                        headers=headers, thinking_control=thinking_control,
                    )
                )
    return records


def run_one(
    url: str,
    model: str,
    condition: Condition,
    frame: dict[str, Any],
    task: str,
    grid: list[list[int]],
    timeout_s: float,
    *,
    headers: dict[str, str] | None = None,
    thinking_control: str = "chat_template_kwargs",
) -> dict[str, Any]:
    request = build_request(model, condition, grid, task, thinking_control)
    truth = truth_for(grid, task)
    started = time.monotonic()
    record: dict[str, Any] = {
        "condition": condition.name,
        "record_index": frame["record_index"],
        "level": frame["level"],
        "task": task,
        "render_scale": condition.render_scale,
        "show_text": condition.show_text,
        "disable_thinking": condition.disable_thinking,
        "max_tokens": condition.max_tokens,
        "truth": truth,
    }
    try:
        response = post_json(url, request, timeout_s, headers)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record["score"] = {"answered": False, "exact": False, "accuracy": 0.0}
        record["elapsed_s"] = round(time.monotonic() - started, 3)
        return record

    usage = response.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    record["elapsed_s"] = round(time.monotonic() - started, 3)
    record["prompt_tokens"] = usage.get("prompt_tokens")
    record["completion_tokens"] = usage.get("completion_tokens")
    record["reasoning_tokens"] = completion_details.get("reasoning_tokens")
    record["finish_reason"] = (response.get("choices") or [{}])[0].get("finish_reason")
    record["score"] = score(task, truth, parse_tool_arguments(response))
    # Assume nothing about the wire format taking effect: a "no thinking" condition that
    # still spends reasoning tokens is not the condition it claims to be, and comparing
    # it against the incumbent would be comparing two thinking arms.
    if condition.disable_thinking:
        reasoning = record["reasoning_tokens"]
        record["thinking_suppression_verified"] = reasoning == 0
        if reasoning:
            record["contract_violation"] = (
                f"disable_thinking condition spent {reasoning} reasoning tokens"
            )
    return record


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition.name and "score" in r]
        if not rows:
            continue
        scored = [r["score"] for r in rows]
        completion = [r.get("completion_tokens") or 0 for r in rows]
        summary[condition.name] = {
            "requests": len(rows),
            "answered": sum(1 for s in scored if s["answered"]),
            "exact": sum(1 for s in scored if s["exact"]),
            "mean_accuracy": round(sum(s["accuracy"] for s in scored) / len(scored), 4),
            "completion_tokens_total": sum(completion),
            "elapsed_s_total": round(sum(r.get("elapsed_s") or 0 for r in rows), 3),
            "by_task": {
                task: round(
                    sum(
                        r["score"]["accuracy"] for r in rows if r["task"] == task
                    ) / max(1, sum(1 for r in rows if r["task"] == task)),
                    4,
                )
                for task in TASK_NAMES
            },
        }
    return summary


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--probe-timeout-s", type=float, default=5400.0)
    # Read the credential from the environment, never from argv: argv is visible in the
    # preserved command record and in any process listing.
    parser.add_argument("--api-key-env", help="env var holding a bearer token")
    parser.add_argument(
        "--thinking-control",
        choices=sorted(THINKING_CONTROLS),
        default="chat_template_kwargs",
        help="how this route suppresses reasoning; see THINKING_CONTROLS",
    )
    args = parser.parse_args()

    started = time.monotonic()
    deadline = started + args.probe_timeout_s
    frames_doc = json.loads(args.frames.read_text())
    if frames_doc.get("schema") != "tycho-nooa-vision-readback-frames-v1":
        raise RuntimeError("pinned frame asset has an unexpected schema")
    frames = frames_doc["frames"]
    url = args.base_url.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {}
    if args.api_key_env:
        token = os.environ.get(args.api_key_env)
        if not token:
            raise RuntimeError(f"{args.api_key_env} is unset or empty")
        headers["Authorization"] = f"Bearer {token}"

    result: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "status": "failed",
        "model": args.model,
        "game_id": frames_doc["game_id"],
        "frames": [
            {"record_index": f["record_index"], "level": f["level"]} for f in frames
        ],
        "conditions": [c.name for c in CONDITIONS],
        "thinking_control": args.thinking_control,
        "base_url": args.base_url,
    }
    try:
        result["visual_token_measurement"] = measure_visual_tokens(
            url, args.model, frames[0]["grid"], args.request_timeout_s,
            headers=headers, thinking_control=args.thinking_control,
        )
        records = run_readback_tasks(
            url, args.model, frames, args.request_timeout_s, deadline,
            headers=headers, thinking_control=args.thinking_control,
        )
        result["records"] = records
        result["summary"] = aggregate(records)
        result["status"] = "passed"
    except Exception as error:  # noqa: BLE001 - preserved, not swallowed: the outer runtime reads this
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    result["request_counts"] = {
        "attempted": REQUESTS_ISSUED["attempted"],
        "succeeded": REQUESTS_ISSUED["succeeded"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result.get("summary", {}), indent=1, sort_keys=True))
    print(json.dumps(result.get("visual_token_measurement", {}), indent=1, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
