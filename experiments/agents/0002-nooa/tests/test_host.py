from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CONTROL_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = CONTROL_ROOT / "nooa-runtime"
sys.path.insert(0, str(CONTROL_ROOT))

from host.runner import NooaProcessAgent, NooaWorkerError  # noqa: E402
from host.qualification import PairedRunSpec, qualify_model_visible_parity  # noqa: E402
from experiments.harness import Observation, Transition  # noqa: E402


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
    def test_successor_crosses_append_only_ipc_without_importing_nooa(self) -> None:
        fixtures = {
            0: [
                "memory = 40\n"
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
            self.assertIn("transition_recorded", [record["kind"] for record in process_records])
            parity = qualify_model_visible_parity(
                run_root=run_root,
                spec=PairedRunSpec.initial(),
            )
            self.assertEqual(parity["unmatched_affordances"], [])
            self.assertEqual(
                action_points[0]["paired_run_spec"],
                PairedRunSpec.initial().as_record(),
            )
            self.assertNotIn("nooa", sys.modules)

            mutated = [
                json.loads(line)
                for line in (run_root / "process.jsonl").read_text().splitlines()
            ]
            crossing = next(
                record for record in mutated if record["kind"] == "model_crossing"
            )
            call = crossing["audit"]["model_calls"][0]
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
                    "environment observation keys differ" in mismatch
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
