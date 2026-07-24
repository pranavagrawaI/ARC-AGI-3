from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from experiments.harness.viewer import color_for_value, layer_grid, load_play_record


REPO = Path(__file__).resolve().parents[3]
M117_RECORDER = (
    REPO
    / "experiments/agents/0001-ls20-observation-option-agent"
    / "record_milestone117_replay.py"
)


class ViewerTraceTests(unittest.TestCase):
    def test_loads_harness_trace_frames(self) -> None:
        records = [
            {
                "type": "observation",
                "step": 0,
                "pixels": [[[0, 1], [2, 3]]],
                "available_actions": [1, 2],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
                "full_reset": True,
            },
            {
                "type": "transition",
                "step": 1,
                "action": 2,
                "action_data": None,
                "reasoning": {"source": "test"},
                "observation": {
                    "pixels": [[[4, 5], [6, 7]]],
                    "available_actions": [1],
                    "levels_completed": 1,
                    "state": "WIN",
                    "full_reset": False,
                },
            },
            {"type": "end", "actions": 1, "stop_reason": "terminal"},
        ]

        frames = load_play_record(_write_jsonl(records))

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].action, "START")
        self.assertEqual(frames[0].available_actions, (1, 2))
        self.assertEqual(frames[1].source_step, 1)
        self.assertEqual(frames[1].action, "2")
        self.assertEqual(frames[1].state, "WIN")
        self.assertEqual(layer_grid(frames[1], 0), ((4, 5), (6, 7)))

    def test_loads_raw_arc_recording_frames(self) -> None:
        records = [
            {
                "timestamp": "2026-06-10T06:39:34+00:00",
                "data": {
                    "game_id": "ls20-9607627b",
                    "state": "NOT_FINISHED",
                    "levels_completed": 0,
                    "action_input": {"id": "ACTION3", "data": {}, "reasoning": None},
                    "full_reset": False,
                    "available_actions": [1, 2, 3, 4],
                    "frame": [[[9, 9], [5, 5]]],
                },
            }
        ]

        frames = load_play_record(_write_jsonl(records))

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].action, "ACTION3")
        self.assertEqual(frames[0].timestamp, "2026-06-10T06:39:34+00:00")
        self.assertEqual(frames[0].width, 2)
        self.assertEqual(frames[0].height, 2)
        self.assertEqual(layer_grid(frames[0], 0), ((9, 9), (5, 5)))

    def test_accepts_single_2d_frame_as_one_layer(self) -> None:
        frames = load_play_record(
            _write_jsonl([
                {
                    "type": "observation",
                    "step": 0,
                    "pixels": [[1, 2], [3, 4]],
                }
            ])
        )

        self.assertEqual(frames[0].layers, 1)
        self.assertEqual(layer_grid(frames[0], 0), ((1, 2), (3, 4)))

    def test_rejects_non_rectangular_frame(self) -> None:
        path = _write_jsonl([
            {
                "type": "observation",
                "step": 0,
                "pixels": [[[1, 2], [3]]],
            }
        ])

        with self.assertRaisesRegex(ValueError, "rectangular"):
            load_play_record(path)

    def test_palette_handles_arc_and_out_of_range_values(self) -> None:
        self.assertEqual(color_for_value(0), "#000000")
        self.assertRegex(color_for_value(99), r"^#[0-9a-f]{6}$")

    def test_extracts_milestone_replay_actions_by_public_step(self) -> None:
        module = _load_module(M117_RECORDER)
        path = _write_jsonl([
            {
                "phase": "LIVE_OPTION_ACTUATION",
                "actuation": {
                    "records": [
                        {"step": 2, "action": 3, "executed": True},
                        {"step": 1, "action": 4, "executed": True},
                        {"step": 3, "action": 1, "executed": False},
                    ]
                },
            },
            {
                "phase": "DUPLICATE_NESTED_VIEW",
                "execution": {
                    "records": [
                        {"step": 2, "action": 3, "executed": True},
                        {"step": 4, "action": 1, "executed": True},
                    ]
                },
            },
        ])

        actions = module.extract_executed_actions(path)
        truncated = module.extract_executed_actions(path, stop_after_actions=2)

        self.assertEqual(actions, (4, 3, 1))
        self.assertEqual(truncated, (4, 3))


def _write_jsonl(records: list[dict]) -> Path:
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "record.jsonl"
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    _TEMP_DIRS.append(directory)
    return path


_TEMP_DIRS: list[tempfile.TemporaryDirectory] = []


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("record_milestone117_replay", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
