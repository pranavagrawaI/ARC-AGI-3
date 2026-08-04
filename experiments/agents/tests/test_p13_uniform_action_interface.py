"""Contract tests for the corrected, uniformly named action interface."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


EXPERIMENT_ROOT = (
    Path(__file__).resolve().parents[1] / "0006-capability-environment"
)
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from p9 import freeze
from uniform_action_control import (
    ACTION_INPUT_CONVENTION,
    CORRECTED_CONTROL_ARM_ID,
    corrected_control_spec,
)
from p9.arm_diff import compare_arms
from p9.arms import arm_manifest, build_arm, control_spec


EXPECTED_ACTION_INPUT_CONVENTION = """\
The legal action IDs name input events only:
- 1: UP
- 2: DOWN
- 3: LEFT
- 4: RIGHT
- 5: SPACE
- 6: MOUSE, using integer grid coordinates x (column) and y (row)

These names describe only the input event sent to the game. They do not predict what that
input does in the current game or state. Infer every consequence, object role, rule, and
objective from observed action/observation transitions."""


class _SilentClient:
    def complete(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("manifest construction must not call a model")


class UniformActionInterfaceTests(unittest.TestCase):
    def test_corrected_control_adds_only_the_six_input_event_names(self) -> None:
        old = control_spec()
        corrected = corrected_control_spec(old)

        self.assertEqual(ACTION_INPUT_CONVENTION, EXPECTED_ACTION_INPUT_CONVENTION)
        self.assertEqual(
            corrected.system_prompt,
            old.system_prompt.rstrip() + "\n\n" + EXPECTED_ACTION_INPUT_CONVENTION,
        )
        self.assertEqual(corrected.arm_id, CORRECTED_CONTROL_ARM_ID)
        self.assertEqual(corrected.treatment_capabilities, ())
        self.assertNotIn("{", ACTION_INPUT_CONVENTION)
        self.assertNotIn("for example", ACTION_INPUT_CONVENTION.lower())

    def test_constructed_control_diff_is_limited_to_identity_and_prompt(self) -> None:
        with TemporaryDirectory() as scratch:
            root = Path(scratch)
            old = build_arm(
                control_spec(), run_dir=root / "old", inner_client=_SilentClient()
            )
            corrected = build_arm(
                corrected_control_spec(),
                run_dir=root / "corrected",
                inner_client=_SilentClient(),
            )
            try:
                report = compare_arms(
                    arm_manifest(old),
                    arm_manifest(corrected),
                    authorized_paths={
                        "arm_id",
                        "model_visible.system_prompt.sha256",
                        "model_visible.system_prompt.chars",
                        "host_only.spec.arm_id",
                    },
                )
            finally:
                old.close()
                corrected.close()

        self.assertTrue(report.ok, report.as_json())
        self.assertEqual(report.authorized_unused, ())

    def test_old_frozen_control_remains_unchanged(self) -> None:
        self.assertEqual(freeze.verify()["drift"], [])


if __name__ == "__main__":
    unittest.main()
