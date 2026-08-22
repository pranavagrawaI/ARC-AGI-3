#!/usr/bin/env python3
"""Mine and branch-replay Tycho-NOOA failure packets without model calls.

Usage:
  arc-host/.venv/bin/python tools/failure_atlas.py mine RUN_DIR OUTPUT
  arc-host/.venv/bin/python tools/failure_atlas.py replay CORPUS OUTPUT \
      --environment-files environment_files [--sequence 18]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import arc_agi
from arc_agi.base import OperationMode
from arc_host.actions import game_action
from arc_host.frames import public_observation
from arc_host.types import ActionDecision
from failure_atlas_core import (
    Action,
    Observation,
    ReplayEnvironment,
    engine_commitment,
    mine_builder_refreshes,
    mine_corpus,
    replay_corpus,
    write_json,
)


class LocalArcEnvironment(ReplayEnvironment):
    """Create a fresh offline game instance for every candidate sequence."""

    def __init__(self, *, game_id: str, seed: int, environment_files: Path) -> None:
        self.game_id = game_id
        self.seed = seed
        self.environment_files = environment_files.resolve()
        self._recordings = TemporaryDirectory(prefix="failure-atlas-arc-")
        logger = logging.getLogger("failure_atlas.arc")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        self.arcade = arc_agi.Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(self.environment_files),
            recordings_dir=self._recordings.name,
            logger=logger,
        )

    def run(self, actions: tuple[Action, ...]) -> tuple[Observation, ...]:
        environment = self.arcade.make(self.game_id, seed=self.seed, save_recording=False)
        if environment is None:
            raise RuntimeError(f"offline ARC engine could not create {self.game_id!r}")
        frame = environment.reset()
        if frame is None:
            raise RuntimeError("offline ARC engine reset returned no frame")
        observations = [_observation(frame)]
        for action in actions:
            sdk_action, data = game_action(
                ActionDecision(action=action.action, row=action.row, col=action.col)
            )
            frame = environment.step(sdk_action, data=data)
            if frame is None:
                raise RuntimeError(f"offline ARC engine returned no frame after {action.action}")
            observations.append(_observation(frame))
        return tuple(observations)

    def close(self) -> None:
        self.arcade.close_scorecard()
        self._recordings.cleanup()


def _observation(frame: object) -> Observation:
    value = public_observation(frame)
    return Observation(
        grid=value.grid,
        available_actions=value.available_actions,
        levels_completed=value.levels_completed,
        state=value.state,
        full_reset=value.full_reset,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mine = commands.add_parser("mine", help="split one run into replayable packets")
    mine.add_argument("run_dir", type=Path)
    mine.add_argument("output", type=Path)

    replay = commands.add_parser("replay", help="branch packets through a local engine")
    replay.add_argument("corpus", type=Path)
    replay.add_argument("output", type=Path)
    replay.add_argument("--environment-files", type=Path, required=True)
    replay.add_argument("--sequence", type=int, action="append")
    replay.add_argument("--max-packets", type=int)
    refreshes = commands.add_parser(
        "builder-refreshes",
        help="rank saved builder report refreshes without model calls",
    )
    refreshes.add_argument("checkpoint_root", type=Path, nargs="+")
    refreshes.add_argument("--output", type=Path, required=True)
    refreshes.add_argument("--shadow-root", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "mine":
        result = mine_corpus(args.run_dir, args.output)
    elif args.command == "replay":
        manifest = json.loads((args.corpus / "manifest.json").read_text())
        identity = engine_commitment(
            args.environment_files.resolve(), str(manifest["game_id"])
        )
        environment = LocalArcEnvironment(
            game_id=str(manifest["game_id"]),
            seed=int(manifest["seed"]),
            environment_files=args.environment_files,
        )
        try:
            result = replay_corpus(
                args.corpus,
                args.output,
                environment,
                sequences=set(args.sequence) if args.sequence else None,
                max_packets=args.max_packets,
                engine_identity=identity,
            )
        finally:
            environment.close()
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        result = mine_builder_refreshes(
            args.checkpoint_root,
            shadow_root=args.shadow_root,
        )
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
