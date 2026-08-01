from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

CONTROL_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = CONTROL_ROOT / "nooa-runtime"
sys.path.insert(0, str(CONTROL_ROOT))

from host.runner import NooaProcessAgent, NooaWorkerError  # noqa: E402
from host.qualification import PairedRunSpec, qualify_model_visible_parity  # noqa: E402
from experiments.harness import Observation, Transition, run_episode  # noqa: E402


def observation(step: int, *, available_actions: tuple[int, ...] = (1, 2)) -> Observation:
    pixels = np.array([[[step, 0], [0, 0]]], dtype=int)
    pixels.setflags(write=False)
    return Observation(
        pixels=pixels,
        available_actions=available_actions,
        levels_completed=step,
        state="NOT_FINISHED",
        full_reset=False,
    )


class NooaProcessAgentTests(unittest.TestCase):
    def test_environment_failure_closes_the_accepted_action_without_a_next_call(
        self,
    ) -> None:
        class FailingEnvironment:
            def reset(self):
                return SimpleNamespace(
                    frame=np.zeros((1, 2, 2), dtype=int),
                    available_actions=(1,),
                    levels_completed=0,
                    state="NOT_FINISHED",
                    full_reset=False,
                )

            def step(self, *_args, **_kwargs):
                raise RuntimeError("injected environment failure")

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            with NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=run_root,
                fixture_policy=lambda _step, _observation: [
                    "return_result({'action': 1, 'data': None, 'reason': 'accepted'})"
                ],
            ) as agent:
                with self.assertRaisesRegex(RuntimeError, "injected environment failure"):
                    run_episode(FailingEnvironment(), agent, max_actions=2)
                self.assertIsNone(agent._pending_decision)

            process_records = [
                json.loads(line)
                for line in (run_root / "process.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["kind"] for record in process_records].count("model_crossing"),
                1,
            )
            failure = next(
                record
                for record in process_records
                if record["kind"] == "host_execution_failure"
            )
            self.assertEqual(failure["action_point_id"], 0)
            self.assertEqual(failure["error"]["type"], "RuntimeError")

    def test_paired_parity_exercises_the_same_context_eviction_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            with NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=run_root,
                fixture_policy=lambda step, _observation: [
                    "return_result({'action': 1, 'data': None, "
                    f"'reason': 'step {step}'}})"
                ],
            ) as agent:
                current = observation(0)
                for step in range(8):
                    decision = agent.act(current)
                    if step == 7:
                        break
                    successor = observation(step + 1)
                    agent.observe(
                        Transition(
                            before=current,
                            decision=decision,
                            after=successor,
                        )
                    )
                    current = successor

            model_writable_observation = (
                run_root / "workspace" / "observations" / "step-00000.json"
            )
            model_writable_observation.write_text(
                '{"pixels":[[[99]]],"available_actions":[99]}',
                encoding="utf-8",
            )
            parity = qualify_model_visible_parity(
                run_root=run_root,
                spec=PairedRunSpec.initial(),
            )

            self.assertEqual(parity["unmatched_affordances"], [])
            self.assertEqual(parity["fixture_coverage"]["candidate_actions"], 8)
            self.assertEqual(parity["fixture_coverage"]["control_actions"], 8)
            self.assertTrue(parity["fixture_coverage"]["omission_exercised"])
            omission = next(
                entry
                for entry in parity["catalogue"]
                if entry["path"] == "context.omission_notice"
            )
            self.assertEqual(omission["control"], omission["nooa"])
            omission_index = next(
                entry
                for entry in parity["catalogue"]
                if entry["path"] == "context.omission_step_index"
            )
            self.assertIn("action steps 1", omission_index["control"][0])
            self.assertIn("action steps 0", omission_index["nooa"][0])

    def test_successor_crosses_append_only_ipc_without_importing_nooa(self) -> None:
        fixtures = {
            0: [
                "memory = 40\n"
                "open('NOTES.md', 'w').write('memory graph reflection pass')\n"
                "return_result({'action': 1, 'data': None, 'reason': 'first'})"
            ],
            1: [
                "assert memory == 40\n"
                "assert action_point['observation']['recent_transitions'][-1]"
                "['levels_after'] == 1\n"
                "assert open('current_observation.json').read()\n"
                "return_result({'action': 2, 'data': None, 'reason': 'successor'})"
            ],
        }
        before = observation(0)
        after = observation(1)
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            with NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=run_root,
                fixture_policy=lambda step, _observation: fixtures[step],
            ) as agent:
                self.assertIsNone(agent._process.stdout)
                self.assertIsNone(agent._process.stderr)
                first = agent.act(before)
                self.assertEqual(first.action, 1)
                agent.observe(Transition(before=before, decision=first, after=after))
                second = agent.act(after)
                self.assertEqual(second.action, 2)

            requests = [
                json.loads(line)
                for line in (run_root / "ipc" / "requests.jsonl").read_text().splitlines()
            ]
            action_points = [record for record in requests if record.get("kind") != "stop"]
            self.assertEqual(len(action_points), 2)
            self.assertEqual(
                sorted(action_points[0]["observation"]),
                [
                    "current_metadata",
                    "current_observation_file",
                    "full_history_file",
                    "observations_glob",
                    "recent_transitions",
                    "representations",
                    "step",
                ],
            )
            self.assertEqual(action_points[0]["observation"]["recent_transitions"], [])
            self.assertEqual(
                action_points[1]["observation"]["recent_transitions"][0]["levels_after"],
                1,
            )
            self.assertTrue((run_root / "workspace" / "current_frame.png").is_file())
            self.assertTrue((run_root / "worker.stdout.log").is_file())
            self.assertTrue((run_root / "worker.stderr.log").is_file())
            self.assertTrue((run_root / "model-frame.png").is_file())
            with Image.open(run_root / "model-frame.png") as model_frame:
                self.assertEqual(model_frame.size, (38, 38))
            with Image.open(run_root / "workspace" / "current_frame.png") as raw_frame:
                self.assertEqual(raw_frame.size, (16, 16))
            self.assertTrue(
                (run_root / "workspace" / "observations" / "step-00001.json").is_file()
            )
            self.assertEqual(
                len((run_root / "workspace" / "public_history.jsonl").read_text().splitlines()),
                1,
            )
            process_records = [
                json.loads(line)
                for line in (run_root / "process.jsonl").read_text().splitlines()
            ]
            self.assertEqual(process_records[0]["kind"], "worker_started")
            self.assertEqual(
                [record["kind"] for record in process_records].count("model_crossing"),
                2,
            )
            self.assertTrue(
                all(
                    record["audit"]["model_calls"]
                    for record in process_records
                    if record["kind"] == "model_crossing"
                )
            )
            first_crossing = next(
                record for record in process_records if record["kind"] == "model_crossing"
            )
            first_call = first_crossing["audit"]["model_calls"][0]
            self.assertEqual(
                first_call["canonicalization"],
                "json-utf8-sort-keys-compact-v1",
            )
            self.assertEqual(len(first_call["canonical_request_sha256"]), 64)
            self.assertNotIn("data:image/", (run_root / "process.jsonl").read_text())
            first_observation_message = next(
                message
                for message in first_call["messages"]
                if isinstance(message.get("content"), list)
            )
            first_payload = json.loads(first_observation_message["content"][0]["text"])
            annotation = first_payload["environment_observation"]["representations"][
                "rendered_image"
            ]["annotation"]
            self.assertEqual(annotation["x_axis"], "column, labelled along the top")
            self.assertEqual(annotation["y_axis"], "row, labelled down the left")
            self.assertIn("transition_recorded", [record["kind"] for record in process_records])
            parity = qualify_model_visible_parity(
                run_root=run_root,
                spec=PairedRunSpec.initial(),
            )
            self.assertEqual(parity["unmatched_affordances"], [])
            self.assertEqual(parity["fixture_coverage"]["candidate_actions"], 2)
            self.assertEqual(parity["fixture_coverage"]["control_actions"], 2)
            self.assertTrue(parity["fixture_coverage"]["successor_exercised"])
            self.assertTrue(parity["fixture_coverage"]["canonical_request_exercised"])
            catalogue_paths = {entry["path"] for entry in parity["catalogue"]}
            self.assertTrue(
                {
                    "system_prompt.semantic_sections",
                    "system_prompt.digest",
                    "observation.payload_fields",
                    "observation.environment_fields",
                    "observation.representations",
                    "observation.public_status_fields",
                    "observation.current_image_commitment",
                    "tools.python.description",
                    "tools.python.arguments",
                    "python.builtins",
                    "python.packages",
                    "workspace.required_files",
                    "workspace.contract_digest",
                    "workspace.helper_digest",
                    "context.recent_turn_boundaries",
                    "context.omission_notice",
                    "context.transition_boundary",
                    "decision.fields",
                    "decision.transport",
                    "decision.temporal_seam",
                    "errors_and_recovery",
                    "budgets.deliberation",
                    "budgets.run",
                    "request.canonical_content",
                    "request.actual_provider_payload",
                }
                <= catalogue_paths
            )
            self.assertNotIn(
                "data:image/",
                json.dumps(parity["canonical_requests"]),
            )
            self.assertEqual(
                action_points[0]["paired_run_spec"],
                PairedRunSpec.initial().as_record(),
            )
            self.assertNotIn("nooa", sys.modules)

            original_records = [
                json.loads(line)
                for line in (run_root / "process.jsonl").read_text().splitlines()
            ]
            behavioral_mutants = (
                "runtime_introspection",
                "hidden_history_query",
                "automatic_retrieval",
                "summarization_or_reflection",
                "semantic_skill_text",
                "history_selection",
                "action_batching",
                "host_force_advance",
            )
            for mutant in behavioral_mutants:
                with self.subTest(mutant=mutant):
                    records = deepcopy(original_records)
                    crossings = [
                        record
                        for record in records
                        if record["kind"] == "model_crossing"
                    ]
                    first_call = crossings[0]["audit"]["model_calls"][0]
                    first_observation = next(
                        message
                        for message in first_call["messages"]
                        if isinstance(message.get("content"), list)
                    )
                    first_payload = json.loads(first_observation["content"][0]["text"])
                    if mutant == "runtime_introspection":
                        first_call["tools"].append(
                            {"name": "inspect_state", "parameters": {}}
                        )
                    elif mutant == "hidden_history_query":
                        first_payload["environment_observation"]["event_records"] = []
                    elif mutant == "automatic_retrieval":
                        first_call["messages"].insert(
                            1,
                            {"role": "user", "content": "Relevant prior item: record 7"},
                        )
                    elif mutant == "summarization_or_reflection":
                        first_call["messages"].insert(
                            1,
                            {"role": "user", "content": "Earlier turns imply rule alpha."},
                        )
                    elif mutant == "semantic_skill_text":
                        first_call["messages"][0]["content"] += (
                            "\nUse the supplied domain procedure alpha."
                        )
                    elif mutant == "history_selection":
                        second_call = crossings[1]["audit"]["model_calls"][0]
                        second_call["messages"] = [
                            message
                            for message in second_call["messages"]
                            if not (
                                isinstance(message.get("content"), list)
                                and json.loads(message["content"][0]["text"])["step"] == 0
                            )
                        ]
                    elif mutant == "action_batching":
                        first_call["tools"].append(
                            {"name": "submit_many", "parameters": {}}
                        )
                    elif mutant == "host_force_advance":
                        first_payload["environment_observation"]["controller_override"] = True
                    if mutant in {"hidden_history_query", "host_force_advance"}:
                        first_observation["content"][0]["text"] = json.dumps(first_payload)
                    (run_root / "process.jsonl").write_text(
                        "".join(json.dumps(record) + "\n" for record in records)
                    )
                    mutant_report = qualify_model_visible_parity(
                        run_root=run_root,
                        spec=PairedRunSpec.initial(),
                    )
                    self.assertTrue(mutant_report["unmatched_affordances"])

            (run_root / "process.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in original_records)
            )

            mutated = [
                json.loads(line)
                for line in (run_root / "process.jsonl").read_text().splitlines()
            ]
            crossing = next(
                record for record in mutated if record["kind"] == "model_crossing"
            )
            call = crossing["audit"]["model_calls"][0]
            call["messages"].append(
                {
                    "role": "assistant",
                    "content": "I considered NOOA MemorySkill and a reflection pass.",
                }
            )
            (run_root / "process.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in mutated)
            )
            model_authored_marker = qualify_model_visible_parity(
                run_root=run_root,
                spec=PairedRunSpec.initial(),
            )
            self.assertEqual(model_authored_marker["unmatched_affordances"], [])

            call["messages"][0]["content"] += "\nNOOA runtime object"
            observation_message = next(
                message
                for message in call["messages"]
                if isinstance(message.get("content"), list)
            )
            payload = json.loads(observation_message["content"][0]["text"])
            payload["environment_observation"]["hidden_history"] = ["forbidden"]
            observation_message["content"][0]["text"] = json.dumps(payload)
            (run_root / "process.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in mutated)
            )
            rejected = qualify_model_visible_parity(
                run_root=run_root,
                spec=PairedRunSpec.initial(),
            )
            self.assertTrue(rejected["unmatched_affordances"])
            self.assertTrue(
                any(
                    "environment-observation keys" in mismatch
                    for mismatch in rejected["unmatched_affordances"]
                )
            )
            self.assertTrue(
                any(
                    "system prompt exposes runtime terms" in mismatch
                    for mismatch in rejected["unmatched_affordances"]
                )
            )

    def test_host_rejects_a_decision_without_model_crossing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=Path(directory) / "run",
                fixture_policy=lambda _step, _observation: [
                    "return_result({'action': 1, 'data': None, 'reason': 'fixture'})"
                ],
            )
            response = {
                "decision": {"action": 1, "data": None, "reason": "unaudited"},
                "audit": {"model_calls": []},
            }
            try:
                with patch.object(agent, "_wait_for_response", return_value=response):
                    with self.assertRaisesRegex(NooaWorkerError, "model-call audit"):
                        agent.act(observation(0))
            finally:
                try:
                    agent.close()
                except NooaWorkerError:
                    pass

    def test_host_surfaces_structured_provider_transport_failure_without_an_action(self) -> None:
        for disposition, attempts in (("fatal", 1), ("uncertain", 1), ("retryable", 3)):
            with self.subTest(disposition=disposition):
                with tempfile.TemporaryDirectory() as directory:
                    agent = NooaProcessAgent(
                        runtime_root=RUNTIME_ROOT,
                        run_root=Path(directory) / "run",
                        fixture_policy=lambda _step, _observation: [
                            "return_result({'action': 1, 'data': None, "
                            "'reason': 'must not run'})"
                        ],
                    )
                    response = {
                        "error": {
                            "kind": "provider_transport",
                            "disposition": disposition,
                            "attempts": attempts,
                        },
                        "audit": {
                            "model_calls": [
                                {
                                    "messages": [
                                        {"role": "user", "content": "x"}
                                    ],
                                    "tools": [],
                                }
                            ]
                        },
                    }
                    try:
                        with patch.object(
                            agent,
                            "_wait_for_response",
                            return_value=response,
                        ):
                            with self.assertRaisesRegex(
                                NooaWorkerError,
                                f"provider transport {disposition} after {attempts} attempt",
                            ):
                                agent.act(observation(0))
                        self.assertIsNone(agent._pending_decision)
                    finally:
                        try:
                            agent.close()
                        except NooaWorkerError:
                            pass

    def test_host_preserves_structured_decision_failure_with_the_model_crossing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            agent = NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=run_root,
                fixture_policy=lambda _step, _observation: [
                    "return_result({'action': 1, 'data': None, 'reason': 'unused'})"
                ],
            )
            response = {
                "error": {
                    "kind": "decision_failure",
                    "type": "GenerationError",
                    "message": "invalid action payload",
                },
                "audit": {
                    "model_calls": [
                        {
                            "messages": [{"role": "user", "content": "x"}],
                            "tools": [],
                        }
                    ]
                },
            }
            try:
                with patch.object(agent, "_wait_for_response", return_value=response):
                    with self.assertRaisesRegex(
                        NooaWorkerError,
                        "decision GenerationError: invalid action payload",
                    ):
                        agent.act(observation(0))
                crossing = next(
                    json.loads(line)
                    for line in (run_root / "process.jsonl").read_text().splitlines()
                    if json.loads(line)["kind"] == "model_crossing"
                )
                self.assertEqual(crossing["worker_error"], response["error"])
            finally:
                try:
                    agent.close()
                except NooaWorkerError:
                    pass

    def test_host_surfaces_structured_budget_exhaustion_without_an_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=Path(directory) / "run",
                fixture_policy=lambda _step, _observation: [
                    "return_result({'action': 1, 'data': None, 'reason': 'unused'})"
                ],
            )
            response = {
                "error": {
                    "kind": "budget_exhausted",
                    "resource": "model_calls",
                    "used": 12,
                    "limit": 12,
                },
                "audit": {"model_calls": []},
            }
            try:
                with patch.object(agent, "_wait_for_response", return_value=response):
                    with self.assertRaisesRegex(
                        NooaWorkerError,
                        r"model_calls budget exhausted \(12/12\)",
                    ):
                        agent.act(observation(0))
            finally:
                try:
                    agent.close()
                except NooaWorkerError:
                    pass

    def test_host_preserves_simple_action_data_like_the_frozen_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=Path(directory) / "run",
                fixture_policy=lambda _step, _observation: [
                    "return_result({'action': 1, 'data': None, 'reason': 'fixture'})"
                ],
            )
            response = {
                "decision": {
                    "action": 1,
                    "data": {"unused": "preserved"},
                    "reason": "control parity",
                },
                "audit": {
                    "model_calls": [
                        {"messages": [{"role": "user", "content": "x"}], "tools": []}
                    ]
                },
            }
            try:
                with patch.object(agent, "_wait_for_response", return_value=response):
                    decision = agent.act(observation(0))
                    self.assertEqual(decision.data, {"unused": "preserved"})
            finally:
                try:
                    agent.close()
                except NooaWorkerError:
                    pass

    def test_accepted_action_cannot_be_replaced_before_its_successor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=Path(directory) / "run",
                fixture_policy=lambda _step, _observation: [
                    "return_result({'action': 1, 'data': None, 'reason': 'accepted'})"
                ],
            )
            try:
                accepted = agent.act(observation(0))
                with self.assertRaisesRegex(
                    NooaWorkerError,
                    "observe must close the previous action",
                ):
                    agent.act(observation(0))
                self.assertEqual(agent._pending_decision, accepted)
            finally:
                try:
                    agent.close()
                except NooaWorkerError:
                    pass

    def test_workspace_cannot_change_after_acceptance_before_successor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=Path(directory) / "run",
                fixture_policy=lambda _step, _observation: [
                    "open('model.txt', 'w').write('accepted state')\n"
                    "return_result({'action': 1, 'data': None, 'reason': 'accepted'})"
                ],
            )
            before = observation(0)
            after = observation(1)
            try:
                accepted = agent.act(before)
                (agent.workspace / "model.txt").write_text("post-accept mutation")
                with self.assertRaisesRegex(
                    NooaWorkerError,
                    "workspace changed after action acceptance",
                ):
                    agent.observe(
                        Transition(before=before, decision=accepted, after=after)
                    )
                self.assertEqual(agent._pending_decision, accepted)
                self.assertEqual(agent._step, 0)
                violations = [
                    json.loads(line)
                    for line in agent.process_path.read_text().splitlines()
                    if json.loads(line)["kind"] == "temporal_seam_violation"
                ]
                self.assertEqual(len(violations), 1)
                self.assertNotEqual(
                    violations[0]["accepted_workspace_commitment"],
                    violations[0]["actual_workspace_commitment"],
                )
            finally:
                try:
                    agent.close()
                except NooaWorkerError:
                    pass

    def test_host_rejects_invalid_coordinates_from_the_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=Path(directory) / "run",
                fixture_policy=lambda _step, _observation: [
                    "return_result({'action': 6, 'data': {'x': 0, 'y': 0}, "
                    "'reason': 'fixture'})"
                ],
            )
            response = {
                "decision": {
                    "action": 6,
                    "data": {"x": 2, "y": 0},
                    "reason": "bad",
                },
                "audit": {
                    "model_calls": [
                        {"messages": [{"role": "user", "content": "x"}], "tools": []}
                    ]
                },
            }
            try:
                with patch.object(agent, "_wait_for_response", return_value=response):
                    with self.assertRaisesRegex(NooaWorkerError, "outside width=2"):
                        agent.act(observation(0, available_actions=(6,)))
            finally:
                try:
                    agent.close()
                except NooaWorkerError:
                    pass

    def test_host_rejects_missing_coordinates_from_the_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=Path(directory) / "run",
                fixture_policy=lambda _step, _observation: [
                    "return_result({'action': 6, 'data': {'x': 0, 'y': 0}, "
                    "'reason': 'fixture'})"
                ],
            )
            response = {
                "decision": {
                    "action": 6,
                    "data": {"x": 0},
                    "reason": "missing y",
                },
                "audit": {
                    "model_calls": [
                        {"messages": [{"role": "user", "content": "x"}], "tools": []}
                    ]
                },
            }
            try:
                with patch.object(agent, "_wait_for_response", return_value=response):
                    with self.assertRaisesRegex(
                        NooaWorkerError,
                        "requires integer x and y",
                    ):
                        agent.act(observation(0, available_actions=(6,)))
            finally:
                try:
                    agent.close()
                except NooaWorkerError:
                    pass

    def test_host_classifies_a_non_object_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=Path(directory) / "run",
                fixture_policy=lambda _step, _observation: [
                    "return_result({'action': 1, 'data': None, 'reason': 'fixture'})"
                ],
            )
            with agent.responses_path.open("a", encoding="utf-8") as handle:
                handle.write("[]\n")
                handle.flush()
            try:
                with self.assertRaisesRegex(
                    NooaWorkerError,
                    "response JSON must be an object",
                ):
                    agent._wait_for_response(0)
            finally:
                try:
                    agent.close()
                except NooaWorkerError:
                    pass

    def test_response_timeout_terminates_the_worker_tree_without_leaking_timeout(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=Path(directory) / "run",
                fixture_policy=lambda _step, _observation: ["while True:\n    pass"],
                response_timeout=0.1,
            )
            with self.assertRaisesRegex(NooaWorkerError, "response timed out"):
                agent.act(observation(0))
            try:
                agent.close()
            except NooaWorkerError:
                pass
            except Exception as error:  # noqa: BLE001 - assert the public cleanup surface
                self.fail(f"close leaked {type(error).__name__}: {error}")

    def test_worker_process_death_never_produces_a_host_selected_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = NooaProcessAgent(
                runtime_root=RUNTIME_ROOT,
                run_root=Path(directory) / "run",
                fixture_policy=lambda _step, _observation: [
                    "return_result({'action': 1, 'data': None, 'reason': 'must not run'})"
                ],
            )
            agent._process.kill()  # noqa: SLF001 - deliberate infrastructure fault injection
            agent._process.wait(timeout=5)  # noqa: SLF001
            with self.assertRaisesRegex(NooaWorkerError, "worker exited"):
                agent.act(observation(0))
            try:
                agent.close()
            except NooaWorkerError:
                pass


if __name__ == "__main__":
    unittest.main()
