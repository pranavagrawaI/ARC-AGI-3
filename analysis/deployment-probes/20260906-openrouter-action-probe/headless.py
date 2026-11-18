#!/usr/bin/env python3
"""Headless q6 MdArcSolverAgent for the Kaggle join. No TUI, no viewer exporters."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

CHASSIS = Path('/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/deployment/kaggle/q6-nooa-candidate/chassis')
sys.path.insert(0, str(CHASSIS))

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


async def run() -> None:
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
            "Start solving the game. Use self.latest_state() to read the current state; "
            "do not wait on the game_states queue for a state already delivered. "
            "Within your first two model calls, submit one legal movement action with "
            "a concrete falsifiable prediction. Then compare the next observed grid "
            "with that prediction and revise your hypothesis before the next action. "
            "Use small informative experiments, not extended static inspection. "
            "Follow the arc_skill and existing submission gates."
        )

    kickoff = asyncio.create_task(_kickoff())
    try:
        await _headless_dispatch(agent)
    finally:
        kickoff.cancel()


if __name__ == "__main__":
    if os.environ.get("ARC_LLM_MODEL"):
        os.environ.setdefault("ARC_LLM_MODEL", os.environ["ARC_LLM_MODEL"])
    asyncio.run(run())
