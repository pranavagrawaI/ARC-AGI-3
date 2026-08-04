"""Contract tests for the cd82 information-atom shadow probe."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[3]
PROBE_ROOT = REPO / "experiments" / "probes" / "cd82-information-atoms"
MODULE_PATH = PROBE_ROOT / "probe.py"

spec = importlib.util.spec_from_file_location("cd82_information_atoms_probe", MODULE_PATH)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class _FixtureResponse:
    def __init__(self, payload: dict, call: int) -> None:
        self.message = {
            "role": "assistant",
            "content": json.dumps(
                {"action": (call % 4) + 1, "data": None, "reason": f"fixture-{call}"}
            ),
        }
        self.usage = {"prompt_tokens": call, "completion_tokens": 1, "total_tokens": call + 1}
        self.response_id = f"fixture-{call}"
        self.request_payload = deepcopy(payload)
        self.request_payload_sha256 = probe.payload_sha256(payload)
        self.request_payload_encoding = "json-utf8-sort-keys-compact-v1"
        self.response_metadata = {"model": payload["model"], "provider": "fixture"}
        self.response_body = {
            "id": self.response_id,
            "model": payload["model"],
            "provider": "fixture",
            "choices": [{"message": self.message}],
            "usage": self.usage,
        }
        self.response_body_raw = json.dumps(self.response_body, sort_keys=True)
        self.response_body_sha256 = probe.bytes_sha256(
            self.response_body_raw.encode("utf-8")
        )


class Cd82InformationAtomsContractTest(unittest.TestCase):
    def test_openrouter_environment_uses_the_official_base_url(self) -> None:
        environment = probe.resolve_openrouter_environment(
            {"OPENROUTER_API_KEY": "fixture-key"}
        )
        self.assertEqual(environment.endpoint, "https://openrouter.ai/api/v1")
        self.assertEqual(environment.model, probe.MODEL)
        self.assertEqual(environment.profile, "openrouter-generic")

    def test_frozen_branch_is_the_exact_post_noop_request(self) -> None:
        branch = probe.load_frozen_branch()
        payload = branch["source_payload"]

        self.assertEqual(branch["source_process_event"], 73)
        self.assertEqual(branch["preceding_decision_event"], 71)
        self.assertEqual(branch["preceding_transition_event"], 72)
        self.assertEqual(branch["source_payload_sha256"], probe.SOURCE_PAYLOAD_SHA256)
        self.assertEqual(probe.payload_sha256(payload), probe.SOURCE_PAYLOAD_SHA256)
        self.assertEqual(payload["provider"]["only"], ["modelrun/fp4"])
        self.assertFalse(payload["provider"]["allow_fallbacks"])
        observation = probe.current_observation_payload(payload)
        self.assertEqual(observation["step"], 13)
        transition = observation["environment_observation"]["recent_transitions"][-1]
        self.assertEqual(transition["action"], 3)
        self.assertEqual(transition["changed_pixels"], 0)
        self.assertEqual(transition["before_digest"], transition["after_digest"])

    def test_conditions_change_only_one_information_field(self) -> None:
        branch = probe.load_frozen_branch()
        source = branch["source_payload"]
        conditions = probe.load_conditions()

        self.assertEqual(list(conditions), ["c0-control", "c1-belief", "c2-linked-result", "c3-contradiction"])
        self.assertIsNone(conditions["c0-control"])
        previous_length = 0
        for condition_id, text in conditions.items():
            payload = probe.build_condition_payload(source, text)
            if condition_id == "c0-control":
                self.assertEqual(probe.payload_sha256(payload), probe.SOURCE_PAYLOAD_SHA256)
                continue
            self.assertGreater(len(text), previous_length)
            previous_length = len(text)
            self.assertEqual(
                probe.json_diff_paths(source, payload),
                ["messages[18].content[0].text.information_probe"],
            )
            self.assertEqual(
                probe.current_observation_payload(payload)["information_probe"], text
            )

    def test_fixture_bundle_makes_four_calls_and_never_steps_a_game(self) -> None:
        calls: list[dict] = []

        def complete(payload: dict):
            calls.append(deepcopy(payload))
            return _FixtureResponse(payload, len(calls))

        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "bundle"
            result = probe.execute_bundle(output_dir=output, complete_payload=complete)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["model_calls"], 4)
            self.assertEqual(result["environment_actions"], 0)
            self.assertEqual(result["conditions"], list(probe.load_conditions()))
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(list((output / "requests").glob("*.json"))), 4)
            self.assertEqual(len(list((output / "responses").glob("*.raw"))), 4)
            self.assertTrue((output / "result.json").exists())

    def test_execute_requires_exact_plan_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "confirmation"):
            probe.validate_execution_confirmation(
                confirmation="wrong",
                supplied_plan_sha256="0" * 64,
                actual_plan_sha256="1" * 64,
            )
        with self.assertRaisesRegex(ValueError, "plan sha256"):
            probe.validate_execution_confirmation(
                confirmation=probe.EXECUTION_CONFIRMATION,
                supplied_plan_sha256="0" * 64,
                actual_plan_sha256="1" * 64,
            )
        probe.validate_execution_confirmation(
            confirmation=probe.EXECUTION_CONFIRMATION,
            supplied_plan_sha256="1" * 64,
            actual_plan_sha256="1" * 64,
        )


if __name__ == "__main__":
    unittest.main()
