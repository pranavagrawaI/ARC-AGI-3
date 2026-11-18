#!/usr/bin/env python3
"""Deterministic NOOA-shaped agent for join tests. Not the q3c chassis."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

IPC = Path("ipc")
STATES = IPC / "states.jsonl"
ACTIONS = IPC / "actions.jsonl"
READY = IPC / "agent_ready"


def main() -> None:
    IPC.mkdir(parents=True, exist_ok=True)
    STATES.touch()
    ACTIONS.touch()
    READY.write_text(str(time.time()))
    batch_raw = os.environ.get("NOOA_STUB_BATCH", "UP")
    batch = [part.strip() for part in batch_raw.split(",") if part.strip()]
    if not batch:
        batch = ["UP"]
    pos = 0
    while True:
        with STATES.open() as handle:
            handle.seek(pos)
            line = handle.readline()
            if line.endswith("\n"):
                pos = handle.tell()
                if line.strip():
                    state = json.loads(line)
                    available = state.get("available_actions") or []
                    actions = [item for item in batch if item in available] or [
                        next(
                            (name for name in available if name != "RESET"),
                            "UP",
                        )
                    ]
                    ACTIONS.open("a").write(
                        json.dumps(
                            {
                                "turn": state.get("turn"),
                                "actions": actions,
                                "rationale": "predict: stub",
                            }
                        )
                        + "\n"
                    )
        time.sleep(0.02)


if __name__ == "__main__":
    main()
