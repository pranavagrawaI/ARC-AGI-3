"""Cover the pre-registered primary measure.

The measure decides the comparison, so the cases that matter are the ones where
a plausible implementation would get it wrong: a counter-only change scored as
activity, and a multi-layer frame read on the wrong layer.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inert_action_rate import HUD_FIRST_ROW, measure

SIZE = 64


def blank() -> list[list[int]]:
    return [[0] * SIZE for _ in range(SIZE)]


def with_counter(frame: list[list[int]], column: int) -> list[list[int]]:
    frame = [row[:] for row in frame]
    for row in range(HUD_FIRST_ROW, HUD_FIRST_ROW + 2):
        frame[row][column] = 3
    return frame


def with_sprite(frame: list[list[int]], column: int) -> list[list[int]]:
    frame = [row[:] for row in frame]
    for row in range(40, 45):
        for offset in range(10):
            frame[row][column + offset] = 12
    return frame


def write_run(root: Path, frames: list[list[list[int]]], actions: list[int]) -> Path:
    observations = root / "worker" / "host-observations"
    observations.mkdir(parents=True)
    for step, frame in enumerate(frames):
        (observations / f"step-{step:05d}.json").write_text(
            json.dumps({"pixels": [frame]}), encoding="utf-8"
        )
    process = [
        {"kind": "decision_validated", "action_point_id": step, "decision": {"action": action}}
        for step, action in enumerate(actions)
    ]
    (root / "worker" / "process.jsonl").write_text(
        "\n".join(json.dumps(record) for record in process), encoding="utf-8"
    )
    return root


class InertActionRateTests(unittest.TestCase):
    def test_a_counter_only_change_is_inert(self) -> None:
        """The counter ticks on nearly every action, so counting any change at all
        scores 43 of 44 control transitions as activity and measures nothing."""

        first = with_counter(blank(), 16)
        second = with_counter(blank(), 17)
        with tempfile.TemporaryDirectory() as directory:
            root = write_run(Path(directory), [first, second], [2])
            result = measure(root)
        self.assertEqual(result["inert_actions"], 1)
        self.assertEqual(result["scored_transitions"], 1)
        self.assertEqual(result["inert_rate"], 1.0)

    def test_a_play_area_change_is_not_inert(self) -> None:
        first = with_counter(with_sprite(blank(), 30), 16)
        second = with_counter(with_sprite(blank(), 34), 17)
        with tempfile.TemporaryDirectory() as directory:
            root = write_run(Path(directory), [first, second], [1])
            result = measure(root)
        self.assertEqual(result["inert_actions"], 0)
        self.assertGreater(result["transitions"][0]["play_area_cells"], 0)

    def test_the_measure_reads_the_layer_the_game_draws_on(self) -> None:
        """Reading pixels[0] measures a layer the environment is not drawing on.

        An earlier probe walked 404 of 591 tool calls into exactly this trap, so
        the mutant is worth pinning: here the first layer is frozen and only the
        last one moves, and a first-layer read would call the action inert.
        """

        stale = with_sprite(blank(), 30)
        moved = with_sprite(blank(), 34)
        observations = {
            0: [stale, with_counter(stale, 16)],
            1: [stale, with_counter(moved, 17)],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "worker" / "host-observations").mkdir(parents=True)
            for step, layers in observations.items():
                (root / "worker" / "host-observations" / f"step-{step:05d}.json").write_text(
                    json.dumps({"pixels": layers}), encoding="utf-8"
                )
            (root / "worker" / "process.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "decision_validated",
                        "action_point_id": 0,
                        "decision": {"action": 1},
                    }
                ),
                encoding="utf-8",
            )
            result = measure(root)
        self.assertEqual(result["inert_actions"], 0)

    def test_an_off_pattern_transition_is_reported_not_absorbed(self) -> None:
        """Play-area change was 0 or 50 in every archived control transition. If a
        run breaks that, the measure's premise is gone and must say so rather than
        rounding the transition onto one side."""

        first = with_counter(blank(), 16)
        second = with_counter(blank(), 17)
        second[10][10] = 7
        with tempfile.TemporaryDirectory() as directory:
            root = write_run(Path(directory), [first, second], [1])
            result = measure(root)
        self.assertEqual(result["unexpected_play_area_sizes"], [1])
        self.assertEqual(result["inert_actions"], 0)


if __name__ == "__main__":
    unittest.main()
