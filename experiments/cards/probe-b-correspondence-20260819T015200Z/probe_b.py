"""Probe B: can the model name the correspondence when handed the frame directly?

The live run enumerated three goal hypotheses and bound every one of them to
Shape A, which is five cells short of covering the target and is its mirror.
Probe A showed the dynamics were solve-ready. So the open question is whether
the model *cannot* see the correspondence, or merely never looked.

Two conditions separate perception from comparison:
  raw       - the opening frame exactly as the agent's actor saw it
  segmented - the three structures pre-extracted as small bitmaps

Correct answer in both: the body of 4s. No ARC actions, no game session.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import urllib.request

REPO = Path("/home/hipra/projects/arc-agi-3-tycho-nooa-refactor")
FRAME = REPO / "runs/tycho-nooa-level-start-ar25-20260818T175557Z/workspace/ar25/level_0/turn_000.txt"
OUT = Path(__file__).parent / "responses.jsonl"
TRIALS = 4

HEX = {"0": 0, "4": 4, "5": 5, "9": 9, "a": 10, "b": 11}


def load_env() -> dict[str, str]:
    """Read credentials from the repo .env without echoing them anywhere."""
    env = {}
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip("'\"")
    return env


def load_grid() -> np.ndarray:
    rows = []
    for line in FRAME.read_text().splitlines():
        m = re.match(r"y(\d\d): (.*)", line)
        if m:
            rows.append([HEX[c] for c in m.group(2).replace("|", "").split()])
    return np.array(rows, dtype=np.int16)


def body(grid: np.ndarray, colour: int) -> np.ndarray:
    """The one compact component of `colour`.

    Colour is not a body identifier here: row 63 is a full-width bar of 5s and
    column 63 a full-height bar of 11s, both status indicators sharing a colour
    with a body. Segment, then drop anything spanning the grid.
    """
    from scipy import ndimage

    labels, n = ndimage.label(grid == colour)
    keep = []
    for i in range(1, n + 1):
        r, c = np.where(labels == i)
        if np.ptp(r) < 60 and np.ptp(c) < 60:
            keep.append(i)
    if len(keep) != 1:
        raise AssertionError(f"colour {colour}: expected 1 compact component, got {len(keep)}")
    return labels == keep[0]


def bitmap(grid: np.ndarray, mask: np.ndarray, name: str) -> str:
    """Crop `mask` to its bounding box and render as a '#'/'.' bitmap."""
    r, c = np.where(mask)
    sub = grid[r.min(): r.max() + 1, c.min(): c.max() + 1]
    m = mask[r.min(): r.max() + 1, c.min(): c.max() + 1]
    lines = ["".join("#" if v else "." for v in row) for row in m]
    body = "\n".join(lines)
    return (
        f"{name}: rows {r.min()}-{r.max()}, cols {c.min()}-{c.max()}, "
        f"{int(mask.sum())} filled cells\n{body}"
    )


QUESTION = """\
This is one frame of a grid puzzle. The grid holds:
  - a vertical wall you cannot pass through
  - two movable bodies, which you can slide up/down/left/right
  - one stationary target region

Exactly one of the two movable bodies can be slid so that it covers the target
region exactly: every target cell filled, and no part of the body hanging off.

Which of the two movable bodies is it? Answer with the colour value of that
body, and state in one or two sentences the geometric fact that rules the other
one out.
"""


def prompt_raw(grid: np.ndarray) -> str:
    # Deliberately unsegmented: this is the frame exactly as the actor saw it.
    # The one concession is naming the two status bars, which the live agent had
    # already identified and which are not what this probe is testing.
    return (
        "Cell values: 9 = empty background, a = wall, b = target region, "
        "and 5 and 4 are the two movable bodies (0 marks a hole inside a body).\n"
        "Row y63 and column x63 are status indicators, not part of any body.\n\n"
        + FRAME.read_text() + "\n" + QUESTION
    )


def prompt_segmented(grid: np.ndarray) -> str:
    a = bitmap(grid, body(grid, 5), "Movable body, colour 5")
    b = bitmap(grid, body(grid, 4), "Movable body, colour 4")
    t = bitmap(grid, body(grid, 11), "Target region, colour b")
    return (
        "Three structures have already been extracted from the frame and are "
        "drawn here as bitmaps, cropped to their bounding boxes. '#' is a filled "
        "cell of that structure, '.' is not.\n\n"
        f"{a}\n\n{b}\n\n{t}\n\n"
        "A vertical wall sits between colour 5 (left of it) and colour 4 and the "
        "target (right of it); bodies cannot cross it.\n" + QUESTION
    )


def ask(env: dict[str, str], prompt: str) -> dict:
    """One call. This is a reasoning model: content and reasoning arrive apart,
    and a budget too small to finish thinking returns content=None."""
    url = env["ARC_LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": env["ARC_LLM_MODEL"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "max_tokens": 8000,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {env['ARC_LLM_API_KEY']}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    choice = body["choices"][0]
    usage = body.get("usage", {})
    return {
        "answer": choice["message"].get("content"),
        "reasoning": choice["message"].get("reasoning"),
        "finish_reason": choice.get("finish_reason"),
        "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
        "seconds": round(time.time() - t0, 1),
    }


def grade(answer: str | None) -> str:
    """Correct = names colour 4. Scored on the verdict, not the reasoning."""
    if not answer:
        return "truncated"
    tail = answer[-400:].lower()
    says4 = bool(re.search(r"\b(colou?r\s*)?4\b", tail))
    says5 = bool(re.search(r"\b(colou?r\s*)?5\b", tail))
    if says4 and not says5:
        return "correct"
    if says5 and not says4:
        return "wrong(5)"
    return "ambiguous"


def main() -> None:
    env = load_env()
    grid = load_grid()
    conditions = {"raw": prompt_raw(grid), "segmented": prompt_segmented(grid)}

    if "--show" in sys.argv:
        print(conditions["segmented"])
        return

    print(f"model: {env['ARC_LLM_MODEL']}   trials: {TRIALS} per condition\n")
    with OUT.open("w") as fh:
        for name, prompt in conditions.items():
            scores = []
            errors = 0
            for i in range(1, TRIALS + 1):
                try:
                    rec = ask(env, prompt)
                    errors = 0
                except Exception as exc:                  # noqa: BLE001
                    errors += 1
                    print(f"  {name} trial {i}: ERROR {exc}")
                    if errors >= 2:                       # card's Stop condition
                        print("  stopping: 2 consecutive API errors")
                        return
                    continue
                rec["verdict"] = grade(rec["answer"])
                rec["condition"], rec["trial"] = name, i
                scores.append(rec["verdict"])
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"  {name} trial {i}: {rec['verdict']:<10} "
                      f"({rec['seconds']}s, {rec['reasoning_tokens']} reasoning tokens)")
            n = sum(s == "correct" for s in scores)
            print(f"  -> {name}: {n}/{len(scores)} correct\n")
    print(f"full answers: {OUT}")


if __name__ == "__main__":
    main()
