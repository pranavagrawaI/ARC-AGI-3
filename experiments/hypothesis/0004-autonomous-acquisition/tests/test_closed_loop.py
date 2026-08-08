import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from closed_loop import parse_response
from run_closed_loop import ACTION_CAP, assemble_request, checkpoint, public_case


def response(content, finish_reason="stop"):
    return {
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": content},
        }]
    }


class ClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import freeze_planning
        import freeze_affordances
        freeze_planning.main()
        freeze_affordances.main()
        cls.case = json.loads((ROOT / "fixtures/planning-affordance-compact-case.json").read_text())
        cls.label = json.loads((ROOT / "scorer/planning-label.json").read_text())
        runtime = json.loads((ROOT.parents[2] / "experiments/hypothesis/0003-temporal-game-judgment/runtime-config.json").read_text())
        cls.config = dict(runtime)
        cls.config["model"] = runtime["transfer"]["model"]

    def test_action_contract_is_single_and_closed(self):
        self.assertEqual(parse_response(response("3")).action, 3)
        self.assertEqual(parse_response(response(" 3\n")).action, 3)
        for content in ("0", "True", '{"action": 1}', "TAKE_ACTION{action:3}"):
            with self.assertRaises(ValueError):
                parse_response(response(content))
        with self.assertRaises(ValueError):
            parse_response(response("1", finish_reason="length"))

    def test_visible_case_has_no_route_or_protected_successor(self):
        visible = public_case(self.case)
        self.assertIn("checkpoint_state", visible)
        self.assertNotIn("current_state", visible)
        self.assertTrue({"candidate_plans", "verified_route", "protected_successor", "checkpoint_prefix_actions"}.isdisjoint(visible))

    def test_request_contains_only_current_frame_before_first_action(self):
        arcade, _, frame = checkpoint(self.label)
        try:
            grid = np.asarray(frame.frame[-1])
            request, rows = assemble_request(self.case, self.config, step=0, before_grid=None, current_grid=grid, history=[])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "current")
            self.assertEqual(request["max_tokens"], 8)
            self.assertNotIn("tools", request)
            self.assertNotIn("tool_choice", request)
        finally:
            arcade.close_scorecard()

    def test_successor_request_exposes_feedback_without_future_content(self):
        before = np.zeros((2, 2), dtype=int)
        current = before.copy()
        current[0, 1] = 1
        history = [{"step": 0, "action": 4, "changed_cells": 1, "levels_completed": 0, "state": "NOT_FINISHED", "grid_sha256": "x"}]
        request, rows = assemble_request(self.case, self.config, step=1, before_grid=before, current_grid=current, history=history)
        self.assertEqual([row[0] for row in rows], ["previous", "current"])
        text = request["messages"][1]["content"][-1]["text"]
        self.assertIn('"actions_remaining": 19', text)
        self.assertIn('"changed_cells": 1', text)
        self.assertNotIn("verified_route", text)
        self.assertEqual(ACTION_CAP, 20)


if __name__ == "__main__":
    unittest.main()
