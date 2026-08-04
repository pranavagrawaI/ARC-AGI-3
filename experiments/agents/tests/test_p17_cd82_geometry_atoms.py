"""Contracts for the cd82 public-geometry information-atom probe."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[3]
PROBE_ROOT = REPO / "experiments" / "probes" / "cd82-geometry-atoms"
MODULE_PATH = PROBE_ROOT / "probe.py"
spec = importlib.util.spec_from_file_location("cd82_geometry_atoms_probe", MODULE_PATH)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class FixtureResponse:
    def __init__(self, payload: dict, call: int) -> None:
        self.message = {
            "role": "assistant",
            "content": json.dumps(
                {"action": call, "data": None, "reason": f"geometry-fixture-{call}"}
            ),
        }
        self.usage = {"prompt_tokens": call, "completion_tokens": 1, "total_tokens": call + 1}
        self.response_id = f"geometry-fixture-{call}"
        self.request_payload = deepcopy(payload)
        self.request_payload_sha256 = probe.payload_sha256(payload)
        self.request_payload_encoding = "json-utf8-sort-keys-compact-v1"
        self.response_metadata = {"provider": "fixture", "model": payload["model"]}
        self.response_body = {
            "id": self.response_id,
            "provider": "fixture",
            "choices": [{"message": self.message}],
            "usage": self.usage,
        }
        self.response_body_raw = json.dumps(self.response_body, sort_keys=True)
        self.response_body_sha256 = probe.bytes_sha256(self.response_body_raw.encode())


class Cd82GeometryAtomsContractTest(unittest.TestCase):
    def test_frozen_branch_and_public_measurement_are_exact(self) -> None:
        branch = probe.load_frozen_branch()
        self.assertEqual(branch["source_process_event"], 81)
        self.assertEqual(branch["preceding_decision_event"], 79)
        self.assertEqual(branch["preceding_transition_event"], 80)
        self.assertEqual(branch["source_payload_sha256"], probe.SOURCE_PAYLOAD_SHA256)
        self.assertEqual(probe.payload_sha256(branch["source_payload"]), probe.SOURCE_PAYLOAD_SHA256)
        self.assertEqual(branch["source_payload"]["provider"]["only"], ["modelrun/fp4"])
        measurement = probe.derive_public_measurement(branch)
        self.assertEqual(measurement["color_id"], 15)
        self.assertEqual(measurement["connectivity"], 4)
        self.assertEqual(measurement["selector"], "largest_component")
        self.assertEqual(measurement["before"]["extent"], [13, 13])
        self.assertEqual(measurement["after"]["extent"], [12, 7])
        self.assertEqual(measurement["before"]["area"], 85)
        self.assertEqual(measurement["after"]["area"], 84)
        self.assertNotEqual(
            measurement["before"]["normalized_shape_sha256"],
            measurement["after"]["normalized_shape_sha256"],
        )

    def test_conditions_are_cumulative_and_change_one_field_only(self) -> None:
        branch = probe.load_frozen_branch()
        source = branch["source_payload"]
        conditions = probe.load_conditions()
        self.assertEqual(
            list(conditions),
            ["c0-control", "c1-extent", "c2-component-features", "c3-nonrigid-relation"],
        )
        self.assertIsNone(conditions["c0-control"])
        previous = ""
        for condition_id, information in conditions.items():
            payload = probe.build_condition_payload(source, information)
            if condition_id == "c0-control":
                self.assertEqual(probe.payload_sha256(payload), probe.SOURCE_PAYLOAD_SHA256)
                continue
            self.assertTrue(information.startswith(previous))
            previous = information
            self.assertEqual(
                probe.json_diff_paths(source, payload),
                ["messages[16].content[0].text.information_probe"],
            )

    def test_fixture_execution_is_four_calls_and_zero_actions(self) -> None:
        calls: list[dict] = []

        def complete(payload: dict):
            calls.append(deepcopy(payload))
            return FixtureResponse(payload, len(calls))

        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "bundle"
            result = probe.execute_bundle(output_dir=output, complete_payload=complete)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["model_calls"], 4)
            self.assertEqual(result["environment_actions"], 0)
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(list((output / "requests").glob("*.json"))), 4)
            self.assertEqual(len(list((output / "responses").glob("*.raw"))), 4)

    def test_official_openrouter_environment_and_exact_confirmation(self) -> None:
        environment = probe.resolve_openrouter_environment(
            {"OPENROUTER_API_KEY": "fixture-key"}
        )
        self.assertEqual(environment.endpoint, "https://openrouter.ai/api/v1")
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
