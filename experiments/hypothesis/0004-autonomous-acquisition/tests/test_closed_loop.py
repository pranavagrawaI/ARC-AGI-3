import json
from pathlib import Path
import sys
import unittest

import numpy as np
from arcengine import GameAction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from closed_loop import DELIBERATIVE_SYSTEM_PROMPT, parse_deliberative_response, parse_response
from grounding import PublicGrounder
from run_closed_loop import ACTION_CAP, DELIBERATIVE_MAX_TOKENS, assemble_request, checkpoint, public_case, response_cost_usd


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

    def test_deliberative_contract_allows_reply_but_closes_action(self):
        value = response("I will route upward first.\nACTION: 1")
        self.assertEqual(parse_deliberative_response(value).action, 1)
        for content in ("ACTION: 1\nextra", "I choose 1", "ACTION: 1\nACTION: 2"):
            with self.assertRaises(ValueError):
                parse_deliberative_response(response(content))

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

    def test_primary_request_keeps_provider_pin(self):
        grid = np.zeros((2, 2), dtype=int)
        request, _ = assemble_request(
            self.case, self.config, step=0, before_grid=None, current_grid=grid,
            history=[], include_provider=True,
        )
        self.assertEqual(request["provider"], self.config["provider"])

    def test_deliberative_request_has_room_and_prior_reply_memory(self):
        grid = np.zeros((2, 2), dtype=int)
        config = dict(self.config)
        config.update({
            "reasoning": {"enabled": True, "exclude": False},
            "decision_system_prompt": DELIBERATIVE_SYSTEM_PROMPT,
            "decision_max_tokens": DELIBERATIVE_MAX_TOKENS,
        })
        history = [{"step": 0, "action": 1, "model_reply": "plan\nACTION: 1"}]
        request, _ = assemble_request(
            self.case, config, step=1, before_grid=grid, current_grid=grid,
            history=history, include_provider=True,
        )
        self.assertEqual(request["max_tokens"], 8192)
        self.assertTrue(request["reasoning"]["enabled"])
        self.assertIn('"model_reply": "plan\\nACTION: 1"', request["messages"][1]["content"][-1]["text"])

    def test_public_grounder_stays_exact_through_blocks_and_reset(self):
        actions = [1, 1, 1, 1, 4, 4, 4, 4, 4, 2, 1, 2, 4]
        arcade, environment, frame = checkpoint(self.label)
        grounder = PublicGrounder.from_frozen_prefix()
        observation = grounder.observe(frame.frame[-1])
        try:
            self.assertEqual(observation["controlled_cell"], [30, 24])
            self.assertEqual(observation["controlled_candidate_count"], 1)
            classes = []
            for action in actions:
                frame = environment.step(GameAction.from_id(action))
                observation = grounder.observe(
                    frame.frame[-1], previous=observation, action=action
                )
                classes.append(observation["transition_class"])
                self.assertEqual(
                    observation["controlled_cell"],
                    [environment._game.gudziatsk.y, environment._game.gudziatsk.x],
                )
                self.assertEqual(observation["controlled_candidate_count"], 1)
            self.assertIn("blocked_or_no_translation", classes)
            self.assertIn("unexpected_transition", classes)
        finally:
            arcade.close_scorecard()

    def test_grounded_request_contains_overlay_and_measured_state(self):
        arcade, _, frame = checkpoint(self.label)
        grounder = PublicGrounder.from_frozen_prefix()
        grid = np.asarray(frame.frame[-1])
        grounding = grounder.observe(grid)
        try:
            request, rows = assemble_request(
                self.case, self.config, step=0, before_grid=None, current_grid=grid,
                history=[], include_provider=True, grounder=grounder, grounding=grounding,
            )
            self.assertEqual([row[0] for row in rows], ["current", "grounding"])
            text = request["messages"][1]["content"][-1]["text"]
            self.assertIn('"controlled_cell": [', text)
            self.assertIn('"controlled_track_status": "unique"', text)
        finally:
            arcade.close_scorecard()

    def test_paid_cost_uses_reported_or_audited_usage(self):
        audit = {"endpoint": {"pricing": {"prompt": "0.1", "completion": "0.2"}}}
        self.assertEqual(response_cost_usd({"usage": {"cost": 0.25}}, audit), 0.25)
        self.assertAlmostEqual(
            response_cost_usd({"usage": {"prompt_tokens": 2, "completion_tokens": 3}}, audit),
            0.8,
        )


if __name__ == "__main__":
    unittest.main()
