"""Contract tests for the single corrected-control live run."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


EXPERIMENT_ROOT = (
    Path(__file__).resolve().parents[1] / "0006-capability-environment"
)
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from uniform_action_live import (
    CONFIRM_EXECUTION,
    PROVIDER_TAG,
    build_plan,
    corrected_live_spec,
    fixed_args,
    require_confirmation,
)


class UniformActionLiveContractTests(unittest.TestCase):
    def test_live_spec_matches_recorded_control_ceiling_and_runtime(self) -> None:
        args = fixed_args()
        spec = corrected_live_spec()

        self.assertEqual(args.game, "cd82")
        self.assertEqual(args.model, "google/gemma-4-31b-it")
        self.assertEqual(args.model_profile, "openrouter-generic")
        self.assertEqual(args.env_seed, 0)
        self.assertEqual(args.model_seed, 0)
        self.assertEqual(args.temperature, 1.0)
        self.assertEqual(args.max_completion_tokens, 4096)
        self.assertEqual(args.max_actions, 44)
        self.assertEqual(args.max_context_turns, 6)
        self.assertEqual(args.max_model_rounds_per_action, 10)
        self.assertEqual(args.max_model_calls, 400)
        self.assertEqual(args.max_reported_tokens, 12_000_000)
        self.assertEqual(args.workspace_timeout, 20.0)
        self.assertEqual(args.transport_retries, 6)
        self.assertEqual(args.transport_backoff, 4.0)
        self.assertEqual(spec.arm_id, "0006-uniform-actions-v1")
        self.assertEqual(spec.treatment_capabilities, ())

    def test_plan_pins_modelrun_and_every_participating_source(self) -> None:
        endpoint = {
            "name": "ModelRun | google/gemma-4-31b-it-20260402",
            "provider_name": "ModelRun",
            "tag": "modelrun/fp4",
            "quantization": "fp4",
        }
        plan = build_plan(endpoint)

        self.assertEqual(PROVIDER_TAG, "modelrun/fp4")
        self.assertEqual(plan["endpoint"], endpoint)
        self.assertEqual(plan["provider_only"], ["modelrun/fp4"])
        self.assertFalse(plan["allow_fallbacks"])
        self.assertEqual(plan["stage"], "one corrected-control live run")
        self.assertEqual(plan["anti_actions"], ["no control rerun", "no treatment", "no automatic successor run"])
        self.assertIn(
            "experiments/agents/0006-capability-environment/uniform_action_control.py",
            plan["source_sha256"],
        )
        self.assertIn(
            "experiments/agents/0006-capability-environment/uniform_action_live.py",
            plan["source_sha256"],
        )

    def test_execution_requires_exact_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "confirmation"):
            require_confirmation("yes")
        require_confirmation(CONFIRM_EXECUTION)


if __name__ == "__main__":
    unittest.main()
