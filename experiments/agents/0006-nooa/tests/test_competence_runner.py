"""Tests for the 0006-to-frozen-0002 host boundary."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
from arc_agi.remote_wrapper import RemoteEnvironmentWrapper

ROOT = Path(__file__).resolve().parents[1]
HOST_ROOT = ROOT / "host"
for path in (ROOT, HOST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from competence import SCHEMA, ContractError, parse_model_reason
from competence_runner import (
    EXPECTED_PROJECTION_DTYPE,
    ModelCorrectableError,
    NooaCompetenceProcessAgent,
    NooaWorkerError,
    _require_expected_projection_dtype,
    attach_competence_context,
)
from evidence_chain import canonical_digest, two_action_chain_report

from experiments.harness import Observation, Transition
from experiments.harness.runner import public_observation


class CompetenceRunnerTests(unittest.TestCase):
    def test_only_model_authored_rejections_are_marked_correctable(self) -> None:
        """Separate what the model could fix from what it could not.

        Run 20260802T120006Z lost a complete commitment to one surplus closing
        brace; run 20260802T114136Z died because the observation arrived with the
        wrong dtype. Only the first is answerable by re-authoring, and the repair
        crossing needs to tell them apart by class rather than by message.
        """

        # The harness stops on NooaWorkerError. Losing that relationship would
        # turn a clean stop into an uncaught crash.
        self.assertTrue(issubclass(ModelCorrectableError, NooaWorkerError))

        # A precondition: no commitment the model writes changes the dtype.
        with self.assertRaises(NooaWorkerError) as caught:
            _require_expected_projection_dtype(
                Observation(
                    pixels=np.zeros((1, 2, 2), dtype=np.uint8),
                    available_actions=(1,),
                    levels_completed=0,
                    state="NOT_FINISHED",
                    full_reset=False,
                )
            )
        self.assertNotIsInstance(caught.exception, ModelCorrectableError)

        # A postcondition: the exact payload from 20260802T120006Z, whose only
        # defect is a trailing brace the model could have dropped.
        live_reason = (
            '{"assertions":[{"observable":"levels_completed","op":"==","value":0}],'
            '"claim_ids":["env_layout"],"evidence_requests":[],'
            '"ledger":[{"op":"create","claim_id":"env_layout","statement":"a claim"}],'
            '"reconciliation":null,"test":"Execute action 1.",'
            '"working_state":{"observation":"initial"}}}'
        )
        with self.assertRaises(ContractError):
            parse_model_reason(live_reason, action=1, data=None)
        # Dropping the surplus brace makes the same commitment admissible.
        self.assertEqual(
            parse_model_reason(live_reason[:-1], action=1, data=None)["test"],
            "Execute action 1.",
        )

    def test_ls20_projection_dtype_is_host_enforced(self) -> None:
        def observation_with(pixels: np.ndarray) -> Observation:
            return Observation(
                pixels=pixels,
                available_actions=(1,),
                levels_completed=0,
                state="NOT_FINISHED",
                full_reset=False,
            )

        preserved_path = (
            ROOT.parents[1]
            / "artifacts/0006-nooa/provider-smoke-probes/ls20"
            / "20260801T195450Z/worker/workspace/observations/step-00000.json"
        )
        preserved = json.loads(preserved_path.read_text(encoding="utf-8"))
        producer = RemoteEnvironmentWrapper.__new__(RemoteEnvironmentWrapper)
        raw_frame = producer._convert_to_frame_data_raw(
            {
                "game_id": "ls20",
                "frame": preserved["pixels"],
                "state": preserved["state"],
                "levels_completed": preserved["levels_completed"],
                "win_levels": 0,
                "guid": "preserved-ls20-fixture",
                "full_reset": preserved["full_reset"],
                "available_actions": preserved["available_actions"],
            }
        )
        self.assertIsNotNone(raw_frame)
        produced = public_observation(raw_frame)
        self.assertEqual(produced.pixels.dtype.str, "|i1")
        _require_expected_projection_dtype(produced)

        # These alternatives can all encode the visible integers, but none is
        # the dtype emitted by the production remote wrapper.
        for wrong in (np.uint8, np.int32, np.int64):
            with self.assertRaisesRegex(NooaWorkerError, "approved ls20 contract"):
                _require_expected_projection_dtype(
                    observation_with(np.zeros((1, 2, 2), dtype=wrong))
                )

    def test_treatment_adds_a_fixed_contract_without_mutating_frozen_observation(self) -> None:
        frozen_observation = {
            "step": 3,
            "current_metadata": {"state": "NOT_FINISHED"},
            "recent_transitions": [{"levels_after": 0}],
        }
        packet = {
            "schema": SCHEMA,
            "current_observation": {"step": 3, "evidence_id": "obs-example"},
            "working_state": {"candidate": "try action 2"},
            "active_claims": [],
            "last_prediction_result": None,
            "unresolved_mismatches": [],
            "recent_continuity": [],
            "requested_evidence": [],
        }
        original = deepcopy(frozen_observation)

        attached = attach_competence_context(frozen_observation, packet)

        self.assertEqual(frozen_observation, original)
        self.assertEqual(attached["step"], 3)
        self.assertEqual(attached["competence"]["schema"], SCHEMA)
        self.assertEqual(
            attached["competence"]["context"]["current_observation"]["evidence_id"],
            "obs-example",
        )
        self.assertNotIn("action", attached["competence"]["reason_json_fields"])
        self.assertNotIn("data", attached["competence"]["reason_json_fields"])

    def test_model_visible_pattern_serializes_the_nested_commitment_mechanically(self) -> None:
        attached = attach_competence_context(
            {"step": 0},
            {"schema": SCHEMA, "current_observation": {"step": 0}},
        )
        malformed = (
            '{"working_state":{},"test":"one action","claim_ids":[],"ledger":[],'
            '"assertions":[],"evidence_requests":[],"reconciliation":"None"}"'
        )
        with self.assertRaisesRegex(ContractError, "JSON object"):
            parse_model_reason(malformed, action=1, data=None)

        captured: list[dict[str, object]] = []
        namespace = {
            "action": 1,
            "data": None,
            "commitment": {
                "working_state": {},
                "test": "one action",
                "claim_ids": [],
                "ledger": [],
                "assertions": [],
                "evidence_requests": [],
                "reconciliation": None,
            },
            "return_result": captured.append,
        }
        serialization = attached["competence"]["reason_serialization"]
        authoring_schema = attached["competence"]["reason_authoring_schema"]
        self.assertEqual(authoring_schema["test"]["type"], "string")
        self.assertEqual(authoring_schema["test"]["minLength"], 1)
        self.assertEqual(authoring_schema["test"]["maxLength"], 500)
        # Three consecutive live runs wrote a good prose test and left
        # assertions empty, losing the commitment. Neither field said what it was
        # for, so the model put its prediction in the one the host never checks.
        self.assertIn("does not evaluate", authoring_schema["test"]["purpose"])
        assertions_purpose = authoring_schema["assertions"]["purpose"]
        self.assertIn("the host actually checks", assertions_purpose)
        self.assertIn("empty list is refused", assertions_purpose)
        self.assertIn(
            "an empty list releases no action",
            attached["competence"]["instruction"],
        )
        self.assertEqual(authoring_schema["assertions"]["minItems"], 1)
        self.assertEqual(authoring_schema["assertions"]["maxItems"], 5)
        self.assertEqual(
            set(authoring_schema["assertions"]["items"]["numeric"]["observable"]),
            {"changed_cells", "levels_completed"},
        )
        # A bare null here serialized as "no specification given" and the model
        # deleted the field, killing run 20260802T052341Z. State the permission
        # in words the model cannot read as an absent rule.
        no_mismatch_rule = authoring_schema["reconciliation"][
            "when_no_unresolved_mismatches"
        ]
        self.assertIsInstance(no_mismatch_rule, str)
        self.assertIn("omit", no_mismatch_rule)
        self.assertIn("null", no_mismatch_rule)
        self.assertIn(
            "reconciliation may be omitted when there is no unresolved mismatch",
            attached["competence"]["instruction"],
        )
        # Run 20260802T124434Z spent every deliberation round deciding whether
        # return_result was a tool or an in-cell call, and emitted neither. Both
        # exist on the frozen surface, so the packet must name one route.
        instruction = attached["competence"]["instruction"]
        self.assertIn("Do not call return_result as a separate tool", instruction)
        self.assertIn("the inline call is the only supported one", instruction)
        self.assertTrue(
            serialization["python_pattern"]
            .strip()
            .endswith("return_result({'action': action, 'data': data, 'reason': reason})")
        )
        exec(  # noqa: S102 - execute the exact model-visible Python scaffold.
            serialization["python_pattern"], namespace
        )

        result = captured.pop()
        reason = result["reason"]
        self.assertIsInstance(reason, str)
        assert isinstance(reason, str)
        self.assertNotIn('"None"', reason)
        parsed = parse_model_reason(reason, action=1, data=None)
        self.assertIsNone(parsed["reconciliation"])

        with self.assertRaisesRegex(TypeError, "Python dict"):
            exec(  # noqa: S102 - execute the exact model-visible Python scaffold.
                serialization["python_pattern"],
                {
                    "action": 1,
                    "data": None,
                    "commitment": json.dumps(namespace["commitment"]),
                    "return_result": captured.append,
                },
            )

    def test_offline_worker_must_carry_and_host_admits_a_commitment(self) -> None:
        def observation(step: int) -> Observation:
            pixels = np.array([[[0, step], [0, 0]]], dtype=EXPECTED_PROJECTION_DTYPE)
            pixels.setflags(write=False)
            return Observation(
                pixels=pixels,
                available_actions=(1,),
                levels_completed=0,
                state="NOT_FINISHED",
                full_reset=False,
            )

        def fixture(step: int, _observation: Observation) -> list[str]:
            payload = {
                "working_state": {"candidate": f"action at step {step}"},
                "test": "check one public successor",
                "claim_ids": ["claim-1"],
                "ledger": (
                    [
                        {
                            "op": "create",
                            "claim_id": "claim-1",
                            "statement": "the public frame changes by one cell",
                        }
                    ]
                    if step == 0
                    else []
                ),
                "assertions": [
                    {"observable": "changed_cells", "op": "==", "value": 1},
                ],
                "evidence_requests": (
                    [{"operator": "transition", "action_id": 1}]
                    if step == 1
                    else []
                ),
                "reconciliation": None,
            }
            return [
                "return_result({'action': 1, 'data': None, 'reason': "
                + repr(json.dumps(payload, sort_keys=True, separators=(",", ":")))
                + "})"
            ]

        frozen_root = ROOT.parent / "0002-nooa"
        before, after, final = observation(0), observation(1), observation(2)
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            with NooaCompetenceProcessAgent(
                runtime_root=frozen_root / "nooa-runtime",
                run_root=run_root,
                fixture_policy=fixture,
            ) as agent:
                first = agent.act(before)
                self.assertEqual(first.action, 1)
                self.assertEqual(first.reasoning["competence"]["action_id"], 1)
                agent.observe(Transition(before=before, decision=first, after=after))
                self.assertEqual(
                    next(
                        event
                        for event in reversed(agent.competence.events)
                        if event.get("type") == "prediction_result"
                    )["outcome"],
                    "match",
                )
                second = agent.act(after)
                self.assertEqual(second.reasoning["competence"]["action_id"], 2)
                agent.observe(Transition(before=after, decision=second, after=final))

            requests = [
                json.loads(line)
                for line in (run_root / "ipc" / "requests.jsonl").read_text().splitlines()
                if line.strip()
            ]
            actions = [record for record in requests if record.get("kind") != "stop"]
            self.assertEqual(len(actions), 2)
            self.assertEqual(
                len(actions[0]["observation"]["competence"]["context"]
                    ["current_observation"]["evidence_id"].removeprefix("obs-")),
                64,
            )
            self.assertEqual(
                set(actions[0]["observation"]),
                {
                    "step",
                    "current_metadata",
                    "representations",
                    "recent_transitions",
                    "full_history_file",
                    "current_observation_file",
                    "observations_glob",
                    "competence",
                },
            )
            self.assertEqual(actions[0]["observation"]["competence"]["schema"], SCHEMA)
            self.assertEqual(
                actions[1]["observation"]["competence"]["context"]
                ["last_prediction_result"]["outcome"],
                "match",
            )
            process_kinds = [
                json.loads(line)["kind"]
                for line in (run_root / "process.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(process_kinds.count("competence_action_admitted"), 2)
            events = [
                json.loads(line)
                for line in (run_root / "competence.jsonl").read_text().splitlines()
                if line.strip()
            ]
            process_records = [
                json.loads(line)
                for line in (run_root / "process.jsonl").read_text().splitlines()
                if line.strip()
            ]
            # The fixture policy bypasses the paid model crossing. Normalize only
            # those fixture-only fields; every observation, projection, transition,
            # commitment, and successor record remains producer-authored.
            for record in process_records:
                if record.get("kind") == "action_request":
                    record["request"].pop("fixture_python_cells")
                    record["request_sha256"] = canonical_digest(record["request"])
            requests_by_action = {
                record["action_point_id"]: record["request"]
                for record in process_records
                if record.get("kind") == "action_request"
            }
            for record in process_records:
                if record.get("kind") != "model_crossing":
                    continue
                action_point = record["action_point_id"]
                worker_decision = record["worker_decision"]
                canonical_request = {
                    "provider_request_profile": requests_by_action[action_point][
                        "paired_run_spec"
                    ]["provider_request_profile"],
                    "messages": [{"role": "user", "content": "fixture"}],
                    "tools": [],
                    "kwargs": {},
                }
                crossing = {
                    "request": {},
                    "response": {},
                    "sequence": action_point + 1,
                }
                record["audit"]["model_calls"] = [
                    {
                        "schema": "0002-nooa-provider-crossing-v1",
                        "messages": canonical_request["messages"],
                        "tools": canonical_request["tools"],
                        "canonical_request": canonical_request,
                        "canonical_request_sha256": canonical_digest(
                            canonical_request
                        ),
                        "canonical_audit_request_sha256": canonical_digest(
                            canonical_request
                        ),
                        "canonicalization": "json-utf8-sort-keys-compact-v1",
                        "actual_provider_crossings": [crossing],
                        "attempts": [
                            {
                                "attempt": 1,
                                "outcome": "success",
                                "actual_provider_crossings": [crossing],
                            }
                        ],
                        "outcome": {
                            "kind": "success",
                            "response": {
                                "tool_calls": [
                                    {
                                        "arguments": json.dumps(
                                            {"result": worker_decision}
                                        ),
                                        "id": f"call-{action_point + 1}",
                                        "name": "return_result",
                                    }
                                ]
                            },
                        },
                    }
                ]
            chain_report = two_action_chain_report(events, process_records)
            event_types = [event["type"] for event in events]
            self.assertTrue(
                all(chain_report.values()),
                {
                    "report": chain_report,
                    "types": event_types,
                    "process_kinds": [record["kind"] for record in process_records],
                    "requests": [
                        record["request"]
                        for record in process_records
                        if record.get("kind") == "action_request"
                    ],
                },
            )
            observation_event = next(event for event in events if event["type"] == "observation")
            self.assertEqual(
                observation_event["record"]["evidence_id"],
                actions[0]["observation"]["competence"]["context"]["current_observation"]
                ["evidence_id"],
            )
            self.assertEqual(
                observation_event["record"]["public"]["pixels"],
                [[[0, 0], [0, 0]]],
            )
            self.assertIn("ledger", event_types)
            self.assertIn("transition_occurrence", event_types)
            self.assertIn("derived_evidence", event_types)
            derived = next(event for event in events if event["type"] == "derived_evidence")
            self.assertEqual(len(derived["record"]["evidence_id"].removeprefix("derived-")), 64)


if __name__ == "__main__":
    unittest.main()
