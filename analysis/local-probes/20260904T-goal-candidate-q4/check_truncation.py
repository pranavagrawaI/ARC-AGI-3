#!/usr/bin/env python3
"""Pre-launch context-budget check. No paid calls.

q3c set max_context_tokens=32_000, left max_event_tokens=None, and observed
mean prompt tokens ~55,883 with zero <truncated> markers. The card requires
setting max_event_tokens and confirming the marker before spending.

This script records what the framework actually does with those knobs.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

NOOA_SRC = Path("/home/pranav/projects/arc-agi-3-reference/nooa/src")
sys.path.insert(0, str(NOOA_SRC))

from nooa.agentdoc._truncating_stream import TruncatingStringIO  # noqa: E402
from nooa.config.truncation_config import TruncationConfig, CaptureConfig, FormatConfig  # noqa: E402
from nooa.context_blocks import renderer as R  # noqa: E402


def capture_marker() -> tuple[bool, bool, str]:
    buf = TruncatingStringIO(limit=20)
    buf.write("abcdefghijklmnopqrstuvwxyz")
    text = buf.getvalue()
    return "<truncated>" in text, "not shown" in text.lower(), text


def main() -> int:
    has_tag, has_notice, sample = capture_marker()
    src = inspect.getsource(R.render_context)
    events_dropped_hardcoded = "events_dropped=0" in src
    applies_limit_to_system_only = (
        "_apply_context_total_limit(\n            system_blocks" in src
        or "_apply_context_total_limit(\n        system_blocks" in src
    )
    max_event_in_renderer = "max_event_tokens" in src

    actor = (NOOA_SRC / "nooa/runtime/actor.py").read_text()
    actor_reads_max_event = "tc.max_event_tokens" in actor or "max_event_tokens" in actor

    cfg = TruncationConfig(
        max_context_tokens=32_000,
        max_event_tokens=32_000,
        capture=CaptureConfig(max_stdout=6_000),
        event_format=FormatConfig(max_string=2_000, max_length=100, max_depth=4),
    )

    report = {
        "capture_produces_<truncated>": has_tag,
        "capture_produces_not_shown_notice": has_notice,
        "capture_sample": sample[:120],
        "render_context_hardcodes_events_dropped_0": events_dropped_hardcoded,
        "render_context_evicts_system_blocks_only": applies_limit_to_system_only,
        "render_context_mentions_max_event_tokens": max_event_in_renderer,
        "actor_reads_max_event_tokens": actor_reads_max_event,
        "config_accepts_max_event_tokens": cfg.max_event_tokens == 32_000,
    }
    out = Path(__file__).resolve().parent / "truncation-check.json"
    import json

    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    # Card: if the marker still does not appear from event-history eviction,
    # ship q4 at q3c prompt size. Do not add a third truncation knob.
    event_budget_binds = actor_reads_max_event and max_event_in_renderer
    if not event_budget_binds:
        print(
            "SHIP_ANYWAY: max_event_tokens is accepted by TruncationConfig but "
            "is not applied to event history. Capture-time <truncated> works "
            "on stdout only. q4 ships at q3c prompt size."
        )
        return 0
    print("EVENT_BUDGET_BINDS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
