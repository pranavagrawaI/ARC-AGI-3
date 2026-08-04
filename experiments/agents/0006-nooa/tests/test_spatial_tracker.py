"""Regression for the tracker bug exposed by the completed experiment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spatial_tracker import AVATAR_TEMPLATE, track_translation


class SpatialTrackerTests(unittest.TestCase):
    def test_source_template_resolves_an_up_left_move(self) -> None:
        before = np.zeros((20, 20), dtype=int)
        after = np.zeros((20, 20), dtype=int)
        before[10:15, 10:15] = AVATAR_TEMPLATE
        after[5:10, 5:10] = AVATAR_TEMPLATE

        self.assertEqual(
            track_translation(before, after),
            {"source": (10, 10), "destination": (5, 5), "delta": (-5, -5)},
        )


if __name__ == "__main__":
    unittest.main()
