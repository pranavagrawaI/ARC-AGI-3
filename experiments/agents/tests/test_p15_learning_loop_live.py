"""Contract tests for the single learning_loop treatment live run.

The treatment is only worth running if it differs from the recorded control in
exactly one way. These tests check that claim against the specs themselves rather
than against the prose that asserts it.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


EXPERIMENT_ROOT = (
    Path(__file__).resolve().parents[1] / "0006-capability-environment"
)
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from p9 import freeze
from uniform_action_live import corrected_live_spec
import learning_loop_live
from learning_loop_live import (
    CAPABILITY,
    CONFIRM_EXECUTION,
    TREATMENT_ARM_ID,
    _learning_loop_measures,
    _shared_field_diff,
    build_plan,
    require_confirmation,
    require_game_files,
    require_unmoved_control,
    treatment_live_spec,
)


ENDPOINT = {
    "name": "ModelRun | google/gemma-4-31b-it-20260402",
    "provider_name": "ModelRun",
    "tag": "modelrun/fp4",
    "quantization": "fp4",
}


class TreatmentPairTests(unittest.TestCase):
    def test_treatment_differs_from_control_in_exactly_one_way(self) -> None:
        control = corrected_live_spec()
        treatment = treatment_live_spec()

        self.assertEqual(control.treatment_capabilities, ())
        self.assertEqual(treatment.treatment_capabilities, (CAPABILITY,))
        self.assertEqual(treatment.arm_id, TREATMENT_ARM_ID)
        self.assertNotEqual(control.arm_id, treatment.arm_id)
        self.assertEqual(_shared_field_diff(control, treatment), {})

    def test_the_shared_field_check_can_fail(self) -> None:
        """A match check that cannot detect a mismatch is decoration."""

        from dataclasses import replace

        control = corrected_live_spec()
        drifted = replace(treatment_live_spec(), max_actions=control.max_actions + 1)
        self.assertIn("max_actions", _shared_field_diff(control, drifted))

    def test_the_repair_budget_is_not_relaxed_for_the_treatment(self) -> None:
        """Buying a completed run by loosening the contract breaks the comparison."""

        self.assertEqual(
            treatment_live_spec().max_decision_repairs,
            corrected_live_spec().max_decision_repairs,
        )


class TreatmentPlanTests(unittest.TestCase):
    def test_plan_pins_provider_baseline_and_participating_sources(self) -> None:
        plan = build_plan(ENDPOINT)

        self.assertEqual(plan["endpoint"], ENDPOINT)
        self.assertEqual(plan["provider_only"], ["modelrun/fp4"])
        self.assertFalse(plan["allow_fallbacks"])
        self.assertEqual(plan["arm_id"], TREATMENT_ARM_ID)
        self.assertEqual(plan["treatment_capabilities"], [CAPABILITY])
        self.assertTrue(plan["matched_against_control"]["shared_fields_identical"])
        self.assertEqual(plan["matched_against_control"]["differing_fields"], [])
        self.assertEqual(len(plan["matched_against_control"]["baseline_summary"]["sha256"]), 64)
        for source in (
            "experiments/agents/0006-capability-environment/p9/learning_loop.py",
            "experiments/agents/0006-capability-environment/p9/learning_loop_agent.py",
            "experiments/agents/0006-capability-environment/uniform_action_control.py",
            "experiments/agents/0006-capability-environment/uniform_action_live.py",
            "experiments/agents/0006-capability-environment/learning_loop_live.py",
        ):
            self.assertIn(source, plan["source_sha256"])

    def test_plan_matches_the_recorded_control_budgets(self) -> None:
        plan = build_plan(ENDPOINT)

        self.assertEqual(plan["game"], "cd82")
        self.assertEqual(plan["model"], "google/gemma-4-31b-it")
        self.assertEqual(plan["max_actions"], 44)
        self.assertEqual(plan["env_seed"], 0)
        self.assertEqual(plan["model_seed"], 0)

    def test_plan_forbids_tuning_the_run_into_completion(self) -> None:
        self.assertIn(
            "no repair-budget or ceiling change to force completion",
            build_plan(ENDPOINT)["anti_actions"],
        )

    def test_execution_requires_exact_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "confirmation"):
            require_confirmation("yes")
        require_confirmation(CONFIRM_EXECUTION)


class FrozenControlGuardTests(unittest.TestCase):
    def test_an_unmoved_control_reports_both_digests(self) -> None:
        pinned = require_unmoved_control()
        self.assertEqual(len(pinned["runtime_digest"]), 64)
        self.assertEqual(len(pinned["control_manifest_digest"]), 64)

    def test_a_moved_control_refuses_to_plan(self) -> None:
        moved = {
            "pinned": True,
            "runtime_digest": "0" * 64,
            "control_manifest_digest": "1" * 64,
            "drift": [{"digest": "runtime", "moved": ["p9/agent.py"]}],
        }
        with patch.object(freeze, "verify", return_value=moved):
            with self.assertRaisesRegex(RuntimeError, "0002-R moved"):
                require_unmoved_control()

    def test_an_unpinned_control_refuses_to_plan(self) -> None:
        unpinned = {
            "pinned": False,
            "runtime_digest": "0" * 64,
            "control_manifest_digest": "1" * 64,
            "drift": [],
        }
        with patch.object(freeze, "verify", return_value=unpinned):
            with self.assertRaisesRegex(RuntimeError, "unpinned"):
                require_unmoved_control()

    def test_the_game_files_are_resolved_against_the_repository(self) -> None:
        """The first attempt died because Arcade scans the working directory."""

        resolved = require_game_files("cd82")
        self.assertTrue(resolved.is_dir())
        self.assertEqual(resolved.name, "cd82")
        self.assertEqual(build_plan(ENDPOINT)["game_files"], "environment_files/cd82")

    def test_a_missing_game_refuses_before_reaching_the_service(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no environment files"):
            require_game_files("not_a_game")

    def test_preparation_refuses_when_the_run_directory_exists(self) -> None:
        with patch.object(type(learning_loop_live.RUN_DIR), "exists", lambda _: True):
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                learning_loop_live.prepare()


class LearningLoopMeasureTests(unittest.TestCase):
    """The audit counts and echoes. It must not score."""

    def _events(self) -> list[dict]:
        return [
            {
                "type": "decision",
                "action": 1,
                "reasoning": {
                    "commitment": {
                        "working_state": {"note": "first"},
                        "objective": None,
                        "history_steps": [],
                        "reconciliation": None,
                    }
                },
            },
            {
                "type": "prediction_result",
                "outcome": "matched",
                "objective_result": None,
            },
            {
                "type": "decision",
                "action": 2,
                "reasoning": {
                    "commitment": {
                        "working_state": {"note": "revised"},
                        "objective": {"claim": "reach the exit"},
                        "history_steps": [0, 1],
                        "reconciliation": "the prediction missed, so I withdrew it",
                    }
                },
            },
            {
                "type": "prediction_result",
                "outcome": "mismatched",
                "objective_result": {"outcome": "falsified"},
            },
        ]

    def test_measures_count_outcomes_without_judging_them(self) -> None:
        measures = _learning_loop_measures(self._events())

        self.assertEqual(measures["commitments_recorded"], 2)
        self.assertEqual(
            measures["mechanism_outcomes"], {"matched": 1, "mismatched": 1}
        )
        self.assertEqual(measures["objective_outcomes"], {"falsified": 1})
        self.assertEqual(measures["objectives_authored"], 1)
        self.assertEqual(measures["objective_claims"], ["reach the exit"])
        self.assertEqual(measures["reconciliations_written"], 1)
        self.assertEqual(measures["history_steps_requested"], 2)
        self.assertEqual(measures["working_state_changed"], 1)

    def test_measures_survive_a_run_with_no_commitments(self) -> None:
        """A run that dies on the first decision must still produce a summary."""

        measures = _learning_loop_measures([{"type": "run_started"}])
        self.assertEqual(measures["commitments_recorded"], 0)
        self.assertEqual(measures["mechanism_outcomes"], {})
        self.assertEqual(measures["working_state_changed"], 0)

    def test_no_measure_names_a_verdict(self) -> None:
        """Nothing here may read as the host having decided the loop worked."""

        serialized = json.dumps(_learning_loop_measures(self._events()))
        for verdict in ("success", "correct", "improved", "better", "score"):
            self.assertNotIn(verdict, serialized.lower())


if __name__ == "__main__":
    unittest.main()
