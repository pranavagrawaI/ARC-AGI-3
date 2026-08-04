"""First-loop contract tests for issue #11's minimum 0006 treatment."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1] / "0006-capability-environment"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from p9 import run_arm
from p9.learning_loop import (
    DecisionContractError,
    LearningLoopRuntime,
    evaluate_prediction,
    validate_commitment,
)


class CommitmentTests(unittest.TestCase):
    def commitment(self, **overrides: object) -> dict[str, object]:
        value: dict[str, object] = {
            "working_state": {"beliefs": ["blue may move the avatar"]},
            "test": "test whether the frame changes",
            "prediction": [
                {"observable": "changed_pixels", "op": ">", "value": 0}
            ],
            "objective": {
                "claim": "the tested mechanism advances the game objective",
                "progress": {
                    "observable": "levels_completed",
                    "op": ">",
                    "value": 0,
                },
                "horizon": 2,
            },
            "history_steps": [],
            "reconciliation": None,
        }
        value.update(overrides)
        return value

    def test_model_owns_the_working_state_schema(self) -> None:
        commitment = validate_commitment(
            self.commitment(working_state={"model_chosen": {"anything": [1, "x"]}}),
            available_history_steps=(),
            reconciliation_required=False,
            objective_active=False,
        )
        self.assertEqual(
            commitment["working_state"],
            {"model_chosen": {"anything": [1, "x"]}},
        )

    def test_reconciliation_is_required_only_after_a_mismatch(self) -> None:
        with self.assertRaisesRegex(DecisionContractError, "reconciliation"):
            validate_commitment(
                self.commitment(),
                available_history_steps=(1,),
                reconciliation_required=True,
                objective_active=False,
            )
        accepted = validate_commitment(
            self.commitment(reconciliation="The level prediction was contradicted."),
            available_history_steps=(1,),
            reconciliation_required=True,
            objective_active=False,
        )
        self.assertIn("contradicted", accepted["reconciliation"])

    def test_history_selection_can_only_name_recorded_steps(self) -> None:
        with self.assertRaisesRegex(DecisionContractError, "history_steps"):
            validate_commitment(
                self.commitment(history_steps=[2]),
                available_history_steps=(1,),
                reconciliation_required=False,
                objective_active=False,
            )

    def test_active_objective_cannot_be_replaced_to_reset_its_horizon(self) -> None:
        with self.assertRaisesRegex(DecisionContractError, "active_objective"):
            validate_commitment(
                self.commitment(),
                available_history_steps=(),
                reconciliation_required=False,
                objective_active=True,
            )
        accepted = validate_commitment(
            self.commitment(objective=None),
            available_history_steps=(),
            reconciliation_required=False,
            objective_active=True,
        )
        self.assertIsNone(accepted["objective"])

    def test_objective_progress_cannot_be_a_local_pixel_effect(self) -> None:
        objective = {
            "claim": "changing a pixel is the objective",
            "progress": {"observable": "changed_pixels", "op": ">", "value": 0},
            "horizon": 2,
        }
        with self.assertRaisesRegex(DecisionContractError, "local visual effect"):
            validate_commitment(
                self.commitment(objective=objective),
                available_history_steps=(),
                reconciliation_required=False,
                objective_active=False,
            )

    def test_objective_must_predict_an_increase_from_current_level_count(self) -> None:
        objective = {
            "claim": "remaining on level zero is progress",
            "progress": {
                "observable": "levels_completed",
                "op": "==",
                "value": 0,
            },
            "horizon": 2,
        }
        with self.assertRaisesRegex(DecisionContractError, "increase from 0"):
            validate_commitment(
                self.commitment(objective=objective),
                available_history_steps=(),
                reconciliation_required=False,
                objective_active=False,
                objective_level_baseline=0,
            )


class PredictionTests(unittest.TestCase):
    def test_result_names_each_public_observation_and_the_mismatch(self) -> None:
        result = evaluate_prediction(
            [
                {"observable": "changed_pixels", "op": "==", "value": 2},
                {"observable": "levels_completed", "op": ">", "value": 0},
                {"observable": "state", "op": "==", "value": "NOT_FINISHED"},
                {"observable": "cell", "x": 1, "y": 0, "op": "==", "value": 7},
            ],
            transition={
                "changed_pixels": 2,
                "levels_after": 0,
                "state_after": "NOT_FINISHED",
            },
            final_grid=np.asarray([[3, 7], [1, 1]]),
        )
        self.assertEqual(result["outcome"], "mismatch")
        self.assertEqual(
            [check["observed"] for check in result["checks"]],
            [2, 0, "NOT_FINISHED", 7],
        )
        self.assertEqual(
            [check["matched"] for check in result["checks"]],
            [True, False, True, True],
        )


class ObjectiveHorizonTests(unittest.TestCase):
    def test_identical_pending_commitment_is_idempotent_before_transition(self) -> None:
        with TemporaryDirectory(prefix="p11-idempotent-") as scratch:
            runtime = LearningLoopRuntime(Path(scratch))
            commitment = CommitmentTests().commitment()
            first = runtime.accept(
                commitment,
                history=[],
                frame_shape=(2, 2),
                levels_completed=0,
            )
            duplicate = runtime.accept(
                commitment,
                history=[],
                frame_shape=(2, 2),
                levels_completed=0,
            )
            self.assertEqual(duplicate, first)
            self.assertEqual(runtime.active_objective["remaining_steps"], 2)

    def test_local_matches_do_not_protect_an_expired_objective_claim(self) -> None:
        with TemporaryDirectory(prefix="p11-objective-") as scratch:
            runtime = LearningLoopRuntime(Path(scratch))
            first = {
                "working_state": {},
                "test": "confirm that clicking changes a pixel",
                "prediction": [
                    {"observable": "changed_pixels", "op": ">", "value": 0}
                ],
                "objective": {
                    "claim": "clicking arbitrary cells is the game objective",
                    "progress": {
                        "observable": "levels_completed",
                        "op": ">",
                        "value": 0,
                    },
                    "horizon": 2,
                },
                "history_steps": [],
                "reconciliation": None,
            }
            runtime.accept(
                first,
                history=[],
                frame_shape=(2, 2),
                levels_completed=0,
            )
            result = runtime.observe(
                {
                    "changed_pixels": 1,
                    "levels_after": 0,
                    "state_after": "NOT_FINISHED",
                },
                final_grid=np.zeros((2, 2), dtype=int),
            )
            self.assertEqual(result["outcome"], "match")
            self.assertEqual(result["objective_result"]["outcome"], "pending")
            self.assertEqual(result["objective_result"]["remaining_steps"], 1)

            second = {**first, "objective": None}
            runtime.accept(
                second,
                history=[],
                frame_shape=(2, 2),
                levels_completed=0,
            )
            result = runtime.observe(
                {
                    "changed_pixels": 1,
                    "levels_after": 0,
                    "state_after": "NOT_FINISHED",
                },
                final_grid=np.zeros((2, 2), dtype=int),
            )
            self.assertEqual(result["outcome"], "match")
            self.assertEqual(result["objective_result"]["outcome"], "falsified")
            self.assertTrue(runtime.reconciliation_required)


class VerticalRunTests(unittest.TestCase):
    def test_offline_pair_exercises_contradiction_reconciliation_and_history(self) -> None:
        with TemporaryDirectory(prefix="p11-loop-") as scratch:
            root = Path(scratch) / "pair"
            self.assertEqual(
                run_arm.main(
                    [
                        "--offline",
                        "--pair",
                        "--capability",
                        "learning_loop",
                        "--max-actions",
                        "3",
                        "--run-dir",
                        str(root),
                    ]
                ),
                0,
            )
            diff = json.loads((root / "pair-diff.json").read_text())
            self.assertEqual(diff["diff"]["unauthorized"], [])
            self.assertEqual(diff["diff"]["authorized_unused"], [])

            events = [
                json.loads(line)
                for line in (root / "treatment" / "process.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            results = [event for event in events if event["type"] == "prediction_result"]
            self.assertEqual([event["outcome"] for event in results], ["match", "match", "match"])
            self.assertEqual(
                [event["objective_result"]["outcome"] for event in results],
                ["pending", "falsified", "pending"],
            )

            decisions = [event for event in events if event["type"] == "decision"]
            self.assertIsNone(decisions[0]["reasoning"]["commitment"]["reconciliation"])
            self.assertIsNone(decisions[1]["reasoning"]["commitment"]["reconciliation"])
            self.assertTrue(decisions[2]["reasoning"]["commitment"]["reconciliation"])

            model_responses = [event for event in events if event["type"] == "model_response"]
            second_payload = model_responses[-2]["request_messages"][-1]["content"][0]["text"]
            packet = json.loads(second_payload)["capability_packet"]
            self.assertEqual(packet["selected_history"][0]["step"], 1)
            public_history = [
                json.loads(line)
                for line in (root / "treatment" / "workspace" / "public_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                packet["selected_history"][0],
                next(item for item in public_history if item["step"] == 1),
            )


if __name__ == "__main__":
    unittest.main()
