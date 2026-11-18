"""v5 feedback for the Retrodict runner: HUD mask, object summaries and mismatch receipts.

Retrodict stopped a plan on any mismatched ``expect`` cell, including cells of the timer or
step-budget strip that its own prompt tells the model to ignore (ar25 lost two v3
invocations to column x=63), and reported a mismatch as five cells. Diffs above 40 cells
were logged as a bounding box only. These helpers keep a per-level change count per cell
so border cells that change on most steps are treated as HUD, render an object-level
summary (``arc_infra.transition.summarize``) in Retrodict's x,y convention for the log's
``[OBJECTS]`` line and for the mismatch receipt, and keep the cached segmentation.
"""

from __future__ import annotations

import numpy as np

from arc_infra.transition import TransitionCache, summarize

__all__ = ["LevelChanges", "TransitionCache", "format_objects", "receipt_text", "transition"]

SIZE = 64
HUD_MARGIN = 2  # cells from any edge that may belong to a timer or step-budget strip
HUD_MIN_STEPS = 3  # a border cell must have changed this often ...
HUD_FRACTION = 0.4  # ... and on at least this share of the level's steps
LARGE_DIFF = 40  # upstream's listed-diff limit; larger diffs get an [OBJECTS] line


class LevelChanges:
    """Per-level change counts per cell behind the HUD mask."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.counts = np.zeros((SIZE, SIZE), dtype=int)
        self.steps = 0

    def note(self, diff) -> None:
        """Record one executed step's settled diff ``[(x, y, old, new), ...]``."""
        self.steps += 1
        for x, y, _, _ in diff:
            self.counts[y, x] += 1

    def hud(self, x: int, y: int) -> bool:
        """A border cell that changes on most of this level's steps."""
        near_edge = x < HUD_MARGIN or y < HUD_MARGIN or x >= SIZE - HUD_MARGIN or y >= SIZE - HUD_MARGIN
        return bool(near_edge and self.counts[y, x] >= max(HUD_MIN_STEPS, HUD_FRACTION * self.steps))


def transition(cache: TransitionCache, before, after) -> dict:
    """Transition summary with objects for two settled boards."""
    return summarize(
        before, after, before_objects=cache.objects(before), after_objects=cache.objects(after)
    )


def _span(bbox) -> str:
    r0, c0, r1, c1 = bbox
    return f"(x={c0}..{c1 - 1},y={r0}..{r1 - 1})"


def _object(entry) -> str:
    return f"c={entry['color']} px={entry['pixels']} at {_span(entry['bbox'])}"


def _moved(entry) -> str:
    return f"c={entry['color']} px={entry['pixels']} {_span(entry['from'])}->{_span(entry['bbox'])}"


def format_objects(summary: dict) -> str:
    """One line for the log and prompts: what appeared, disappeared or moved (x,y terms)."""
    objects = summary.get("objects")
    if not objects:
        return "objects: segmentation unavailable (too many components)"
    parts = []
    for key, render in (("appeared", _object), ("disappeared", _object), ("moved", _moved)):
        listed = objects.get(key) or []
        count = objects["counts"][key]
        if count:
            text = "; ".join(render(entry) for entry in listed)
            more = f" (+{count - len(listed)} more)" if count > len(listed) else ""
            parts.append(f"{key}: {text}{more}")
    if objects.get("background_reshaped"):
        parts.append("background reshaped")
    return "; ".join(parts) if parts else "objects: none appeared, disappeared or moved"


def receipt_text(action_name: str, mismatches: list[str], summary: dict) -> str:
    """The surprise note for a real expectation failure: cells plus the board receipt."""
    bbox = summary.get("change_bbox")
    changed = summary.get("changed_cells")
    board = "the board did not change" if not changed else f"{changed} cells changed in {_span(bbox)}"
    return f"after {action_name}: " + "; ".join(mismatches[:5]) + f" | {board}; {format_objects(summary)}"
