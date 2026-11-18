#!/usr/bin/env python3
"""Headless launcher for the pristine upstream MdArcSolverAgent.

Upstream's own entry points (``run_solver.py`` -> ``launcher.py``) open a tmux
session and a Rich TUI, own the game side through ``harness.py``, and mint a
competition scorecard. On Kaggle the official ``MyAgent`` seam owns the game,
there is no terminal, and the run is Phase A only, so this file supplies just the
construction and dispatch that ``launcher.py`` would have supplied.

Every agent argument below comes from upstream's own competition config
(``configs/competition.yaml``): ``skill: interactive-game-solver``,
``visual: additive``, ``png_scale: 8``, ``reasoning_effort: xhigh``,
``effort_ladder: "600:medium"``, ``reflect_every: 8``, and upstream's
``MAX_ACTIONS_PER_TURN`` default. Two things this file must change, both forced by
the deployment rather than chosen:

* **Variant.** The config runs ``memory``. That needs the ``nooa-memory``
  distribution and an embeddings gateway, and an offline Kaggle session has
  neither, so this runs ``mdfiles`` — the paper's own ablation arm.
* **Bounds.** A frontier gateway's defaults do not transfer to one local SGLang
  process. ``--max-context-tokens`` and ``--max-tokens`` bound the context and the
  per-call decode; both read the environment so the two reference arms face the
  same budget.

Nothing here changes agent policy.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parent.parent / "upstream"
sys.path.insert(0, str(UPSTREAM))

from arc_llm import build_llm  # noqa: E402
from solver_agent import MAX_ACTIONS_PER_TURN, MdArcSolverAgent  # noqa: E402

from nooa.config.truncation_config import (  # noqa: E402
    CaptureConfig,
    FormatConfig,
    TruncationConfig,
)
from nooa.interactive import RespondReason  # noqa: E402


def parse_effort_ladder(
    spec: str | None, primary_effort: str | None
) -> list[tuple[float, str]] | None:
    """Upstream ``launcher.parse_effort_ladder``, kept identical."""
    rungs: list[tuple[float, str]] = [(0.0, primary_effort or "")]
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        after, _, effort = part.partition(":")
        rungs.append((float(after), effort.strip()))
    return sorted(rungs) if len(rungs) > 1 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    # Defaults below are configs/competition.yaml's values.
    parser.add_argument("--skill", default="interactive-game-solver")
    parser.add_argument("--visual", default="additive")
    parser.add_argument("--png-scale", dest="png_scale", type=int, default=8)
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--effort-ladder", default="600:medium")
    parser.add_argument("--reflect-every", type=int, default=8)
    parser.add_argument("--max-actions-per-turn", type=int, default=MAX_ACTIONS_PER_TURN)
    # Deployment bounds, not upstream settings.
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=int(os.environ.get("ARC_LLM_CONTEXT_WINDOW", "32768")),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("ARC_LLM_MAX_TOKENS", "8192")),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.environ.get("ARC_LLM_TIMEOUT_SECONDS", "170")),
    )
    return parser.parse_args()


def _skill_path(name: str) -> Path:
    path = UPSTREAM / "skills" / name / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"skill missing: {path}")
    return path


async def _headless_dispatch(agent: MdArcSolverAgent) -> None:
    """The dispatch loop upstream's TUI runs, with no terminal attached."""
    consecutive_errors = 0
    while True:
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
            print(f"[headless] agent returned DONE: {result.explanation}", file=sys.stderr)


async def run() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    (run_dir / "team_nemo" / "shared").mkdir(parents=True, exist_ok=True)
    llm = build_llm(
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        request_timeout=args.request_timeout,
    )
    truncation = TruncationConfig(
        max_context_tokens=args.max_context_tokens,
        capture=CaptureConfig(max_stdout=6_000),
        event_format=FormatConfig(max_string=2_000, max_length=100, max_depth=4),
    )
    agent = MdArcSolverAgent(
        llm=llm,
        truncation=truncation,
        run_dir=run_dir,
        # Upstream passes the opaque alias as both, so no path can leak the real
        # game id into the agent's own surface.
        game_id="the game",
        alias="the game",
        reflect_every=args.reflect_every,
        effort_ladder=parse_effort_ladder(args.effort_ladder, args.reasoning_effort),
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
            "Start solving the game. The first game state is on your "
            "game_states queue. Follow the arc_skill context block."
        )

    kickoff = asyncio.create_task(_kickoff())
    try:
        await _headless_dispatch(agent)
    finally:
        kickoff.cancel()


if __name__ == "__main__":
    asyncio.run(run())
