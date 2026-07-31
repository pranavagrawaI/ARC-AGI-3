from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
import unittest


CONTROL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTROL_ROOT))

from host.qualification import (  # noqa: E402
    PairedRunSpec,
    ProviderRequestProfile,
    verify_source_manifest,
)


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
        self.assertEqual(spec.max_recent_transitions, 8)
        self.assertEqual(
            spec.host_workspace_files,
            (
                "WORKSPACE.md",
                "NOTES.md",
                "frames.py",
                "public_history.jsonl",
                "observations/",
                "current_observation.json",
                "current_frame.png",
            ),
        )
        with self.assertRaisesRegex(ValueError, "unsupported paired-run surface drift"):
            replace(spec, max_recent_transitions=0)
        with self.assertRaisesRegex(ValueError, "unsupported paired-run surface drift"):
            replace(spec, allowed_distributions=("scipy",))
        with self.assertRaises(FrozenInstanceError):
            spec.max_actions = 1  # type: ignore[misc]

    def test_live_provider_profile_freezes_every_static_provider_request_field(self) -> None:
        profile = ProviderRequestProfile.responses(
            model="openai/openai/gpt-5.6-sol",
            api_base="https://gateway.invalid/v1",
            max_output_tokens=16_384,
            reasoning_effort="medium",
            request_timeout_seconds=180.0,
        )
        spec = PairedRunSpec.provider(
            profile,
            environment_profile="arc-agi-3:fixture-game",
            environment_seed=7,
        )

        self.assertEqual(spec.model_profile, profile.model)
        self.assertEqual(spec.provider_endpoint_profile, profile.api_base)
        self.assertIs(spec.provider_request_profile, profile)
        self.assertEqual(spec.environment_profile, "arc-agi-3:fixture-game")
        self.assertEqual(spec.environment_seed, 7)
        self.assertEqual(
            spec.as_record()["provider_request_profile"],
            {
                "api_base": "https://gateway.invalid/v1",
                "api_surface": "responses",
                "drop_params": True,
                "litellm_num_retries": 0,
                "max_output_tokens": 16_384,
                "model": "openai/openai/gpt-5.6-sol",
                "nooa_max_retries": 0,
                "nooa_rate_limit_extra_retries": 0,
                "parallel_tool_calls": False,
                "reasoning_effort": "medium",
                "request_timeout_seconds": 180.0,
                "schema": "0002-nooa-provider-request-v1",
                "seed": None,
                "stream": False,
                "temperature": None,
                "tool_choice": "auto",
                "truncation": "disabled",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            profile.model = "drift"  # type: ignore[misc]

    def test_provider_spec_rejects_a_profile_projection_that_does_not_match(self) -> None:
        profile = ProviderRequestProfile.responses(
            model="openai/openai/gpt-5.6-sol",
            api_base="https://gateway.invalid/v1",
            max_output_tokens=16_384,
            reasoning_effort="medium",
            request_timeout_seconds=180.0,
        )
        spec = PairedRunSpec.provider(
            profile,
            environment_profile="arc-agi-3:fixture-game",
            environment_seed=7,
        )

        with self.assertRaisesRegex(ValueError, "provider profile projection drift"):
            replace(spec, model_profile="another-model")


if __name__ == "__main__":
    unittest.main()
