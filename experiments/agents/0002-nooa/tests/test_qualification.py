from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest


CONTROL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTROL_ROOT))

from host.qualification import PairedRunSpec, verify_source_manifest  # noqa: E402


class PairedRunSpecTests(unittest.TestCase):
    def test_pinned_upstream_and_verbatim_copies_match_the_source_manifest(
        self,
    ) -> None:
        report = verify_source_manifest()

        self.assertEqual(report["mismatches"], [])
        self.assertEqual(report["verified_sources"], 5)

    def test_one_immutable_spec_projects_the_shared_0002_r_settings(self) -> None:
        spec = PairedRunSpec.initial()

        self.assertEqual(
            spec.control_overrides(),
            {
                "max_actions": 500,
                "max_context_turns": 6,
                "max_decision_repairs": 2,
                "max_model_calls": 100,
                "max_model_rounds_per_action": 10,
                "max_reported_tokens": 250_000,
                "transport_max_retries": 2,
                "workspace_timeout_seconds": 20.0,
            },
        )
        with self.assertRaises(FrozenInstanceError):
            spec.max_actions = 1  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
