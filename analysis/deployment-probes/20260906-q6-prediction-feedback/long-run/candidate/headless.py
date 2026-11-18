#!/usr/bin/env python3
"""Headless q6 MdArcSolverAgent for the Kaggle join. No TUI, no viewer exporters."""

from __future__ import annotations

import argparse
import asyncio
import os
import json
import sys
import resource

from pathlib import Path

CHASSIS = Path(__file__).resolve().parent / "chassis"
sys.path.insert(0, str(CHASSIS))

from state_delivery import StateDelivery

from arc_llm import build_llm  # noqa: E402
from solver_agent import MdArcSolverAgent  # noqa: E402

from nooa.config.truncation_config import (  # noqa: E402
    CaptureConfig,
    FormatConfig,
    TruncationConfig,
)
from nooa.interactive import RespondReason  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--skill", default="interactive-game-solver")
    parser.add_argument("--visual", default="off")
    parser.add_argument("--png-scale", type=int, default=8)
    parser.add_argument("--max-actions-per-turn", type=int, default=20)
    parser.add_argument("--reflect-every", type=int, default=8)
    return parser.parse_args()


def _skill_path(name: str) -> Path:
    path = CHASSIS / "skills" / name / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"skill missing: {path}")
    return path


async def _headless_dispatch(agent: MdArcSolverAgent) -> None:
    consecutive_errors = 0
    delivery = StateDelivery()
    while True:
        if (agent.ipc_dir / "ABORT").exists():
            raise RuntimeError((agent.ipc_dir / "ABORT").read_text())
        try:
            wins = await agent.queue_manager.race()
        except (asyncio.CancelledError, ValueError):
            break
        notification: dict[str, list] = {}
        for name, item in wins:
            notification.setdefault(name, []).append(item)
        incoming = notification.get("game_states", [])
        latest = agent.latest_state()
        notification = delivery.prepare(notification, latest)
        with (agent.ipc_dir / "state_delivery.jsonl").open("a") as log:
            log.write(json.dumps({"latest": {"turn": latest.get("turn"), "step": latest.get("step")}, "incoming": incoming, "delivered": notification.get("game_states", []) if notification else [], "suppressed": notification is None}) + "\n")
        if notification is None:
            continue
        try:
            result = await agent.handle(notification)
            consecutive_errors = 0
        except asyncio.CancelledError:
            break
        except Exception as exc:
            if (agent.ipc_dir / "ABORT").exists():
                raise
            consecutive_errors += 1
            backoff = min(2 ** min(consecutive_errors, 6), 60)
            print(
                f"[headless] handle() raised {exc!r} "
                f"(consecutive={consecutive_errors}) — retrying in {backoff}s",
                file=sys.stderr,
            )
            await asyncio.sleep(backoff)
            continue
        if getattr(result, "kind", None) == RespondReason.DONE:
            print(
                f"[headless] agent returned DONE: {result.explanation}", file=sys.stderr
            )


def _limit_worker_memory() -> None:
    # Actor code runs in this CPU worker, separate from the GPU model server.
    # Bound allocation failures such as accidentally accumulating awaitables.
    _worker_memory = 4 * 1024**3
    _memory_soft, _memory_hard = resource.getrlimit(resource.RLIMIT_AS)
    _worker_memory = min(_worker_memory, _memory_hard) if _memory_hard != resource.RLIM_INFINITY else _worker_memory
    resource.setrlimit(resource.RLIMIT_AS, (_worker_memory, _worker_memory))


async def run() -> None:
    _limit_worker_memory()
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    (run_dir / "team_nemo" / "shared").mkdir(parents=True, exist_ok=True)
    llm = build_llm()
    truncation = TruncationConfig(
        max_context_tokens=32_000,
        capture=CaptureConfig(max_stdout=6_000),
        event_format=FormatConfig(max_string=2_000, max_length=100, max_depth=4),
    )
    agent = MdArcSolverAgent(
        llm=llm,
        truncation=truncation,
        run_dir=run_dir,
        game_id="the game",
        alias="the game",
        reflect_every=args.reflect_every,
        visual=args.visual,
        png_scale=args.png_scale,
        skill_path=_skill_path(args.skill),
        max_actions_per_turn=args.max_actions_per_turn,
    )
    states_path = run_dir / "ipc" / "states.jsonl"

    async def _kickoff() -> None:
        while True:
            try:
                if states_path.stat().st_size > 0:
                    break
            except OSError:
                pass
            await asyncio.sleep(0.05)
        agent._user_messages_in.put(
            "Start solving the game. Read self.latest_state() synchronously for the current "
            "observation; the dispatcher owns the notification queue. Follow the arc_skill context block."
        )

    kickoff = asyncio.create_task(_kickoff())
    try:
        await _headless_dispatch(agent)
    finally:
        kickoff.cancel()
        await asyncio.gather(kickoff, return_exceptions=True)
        await agent.queue_manager.shutdown()


if __name__ == "__main__":
    if os.environ.get("ARC_LLM_MODEL"):
        os.environ.setdefault("ARC_LLM_MODEL", os.environ["ARC_LLM_MODEL"])
    asyncio.run(run())
