"""Check notification selection in the candidate; fake queues, no model."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

SOURCE = Path(__file__).parent / "candidate/headless.py"


async def main():
    source = SOURCE.read_bytes()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_headless_dispatch"
    )
    namespace = {"asyncio": asyncio, "sys": sys, "RespondReason": NS(DONE="DONE")}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    cases = iter(
        [
            (0, [("game_states", "already committed")]),
            (0, [("user_messages", "explicit guidance")]),
            (0, [("game_states", "old"), ("user_messages", "explicit guidance")]),
            (1, [("game_states", "successor")]),
        ]
    )
    seen = []
    agent = NS(turn=0)

    async def race():
        try:
            agent.turn, notification = next(cases)
        except StopIteration:
            raise ValueError("end of bounded control") from None
        return notification

    async def handle(notification):
        seen.append({"turn": agent.turn, "channels": list(notification)})
        return NS(kind="WAIT")

    agent.queue_manager = NS(race=race)
    agent.handle = handle
    agent.latest_state = lambda: {"turn": agent.turn}
    agent._last_submitted_turn = lambda: 0
    await namespace["_headless_dispatch"](agent)
    assert seen == [
        {"turn": 0, "channels": ["user_messages"]},
        {"turn": 0, "channels": ["game_states", "user_messages"]},
        {"turn": 1, "channels": ["game_states"]},
    ], seen
    print(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "scope": "candidate dispatcher with fake notifications",
                "handled": seen,
                "stale_state_only_skipped": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
