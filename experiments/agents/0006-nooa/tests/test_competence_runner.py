"""Tests for the 0006-to-frozen-0002 host boundary."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST_ROOT = ROOT / "host"
for path in (ROOT, HOST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from competence import SCHEMA
from competence_runner import (
    NooaCompetenceProcessAgent,
    attach_competence_context,
)

from experiments.harness import Observation, Transition


class CompetenceRunnerTests(unittest.TestCase):
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

    def test_offline_worker_must_carry_and_host_admits_a_commitment(self) -> None:
        def observation(step: int) -> Observation:
            pixels = np.array([[[0, step], [0, 0]]], dtype=int)
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
        before, after = observation(0), observation(1)
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
                    agent.competence.context_packet()["last_prediction_result"]["outcome"],
                    "match",
                )
                second = agent.act(after)
                self.assertEqual(second.reasoning["competence"]["action_id"], 2)

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
            event_types = [event["type"] for event in events]
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
