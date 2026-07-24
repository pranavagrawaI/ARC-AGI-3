from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1] / "0003-reconstruction-uptake"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reconstruction_uptake.compliance import assess_reconstruction_compliance
from reconstruction_uptake.replay import SequentialReplay
from reconstruction_uptake.runner import (
    PolicyPrediction,
    PolicyUpdate,
    execute_replay,
)
from reconstruction_uptake.model_policy import ModelReplayPolicy


def _write_trace(path: Path) -> None:
    records = [
        {
            "type": "observation",
            "step": 0,
            "pixels": [[[0, 1], [0, 0]]],
            "available_actions": [1, 2],
            "levels_completed": 0,
            "state": "NOT_FINISHED",
            "full_reset": False,
        },
        {
            "type": "transition",
            "step": 1,
            "action": 1,
            "action_data": None,
            "reasoning": None,
            "observation": {
                "pixels": [[[0, 0], [0, 1]]],
                "available_actions": [1, 2],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
                "full_reset": False,
            },
        },
        {
            "type": "transition",
            "step": 2,
            "action": 2,
            "action_data": None,
            "reasoning": None,
            "observation": {
                "pixels": [[[1, 0], [0, 0]]],
                "available_actions": [1, 2],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
                "full_reset": False,
            },
        },
        {
            "type": "end",
            "actions": 2,
            "stop_reason": "fixture",
            "level_action_counts": [2],
            "completion_action_counts": [],
            "reset_count": 0,
            "game_over_count": 0,
            "levels_completed": 0,
            "state": "NOT_FINISHED",
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


class SequentialReplayTests(unittest.TestCase):
    def test_next_observation_is_revealed_only_after_prediction_commitment(self) -> None:
        with TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            _write_trace(trace)

            replay = SequentialReplay.from_trace(trace, first_step=1, last_step=2)

            self.assertEqual(replay.current_observation["step"], 0)
            self.assertEqual(replay.recorded_action, 1)
            with self.assertRaisesRegex(RuntimeError, "prediction commitment"):
                replay.reveal()

            revealed = replay.commit_and_reveal({"position_delta": [1, 0]})

            self.assertEqual(revealed.step, 1)
            self.assertEqual(revealed.observation["step"], 1)
            self.assertEqual(replay.current_observation["step"], 1)
            self.assertEqual(replay.recorded_action, 2)

    def test_trace_must_be_contiguous_and_cover_the_requested_window(self) -> None:
        with TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            _write_trace(trace)
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            records[2]["step"] = 3
            trace.write_text("".join(json.dumps(record) + "\n" for record in records))

            with self.assertRaisesRegex(ValueError, "contiguous"):
                SequentialReplay.from_trace(trace, first_step=1, last_step=2)


class ReconstructionComplianceTests(unittest.TestCase):
    def test_activation_requires_one_run_to_satisfy_the_full_contract(self) -> None:
        events = [
            {
                "event_index": 1,
                "type": "artifact_loaded",
                "step": 1,
                "artifact_sha256": "a" * 64,
                "entrypoints": ["parse_observation", "predict", "update"],
                "ok": True,
            },
            {
                "event_index": 2,
                "type": "candidate_predictions",
                "step": 1,
                "predictions": {"1": {"delta": [1, 0]}, "2": {"delta": [-1, 0]}},
            },
            {
                "event_index": 3,
                "type": "prediction_committed",
                "step": 1,
                "artifact_sha256": "a" * 64,
                "prediction": {"delta": [1, 0]},
            },
            {"event_index": 4, "type": "observation_revealed", "step": 1},
            {"event_index": 5, "type": "model_updated", "step": 1, "ok": True},
            {
                "event_index": 6,
                "type": "prediction_committed",
                "step": 2,
                "artifact_sha256": "a" * 64,
                "prediction": {"delta": [-1, 0]},
            },
            {"event_index": 7, "type": "observation_revealed", "step": 2},
            {"event_index": 8, "type": "model_updated", "step": 2, "ok": True},
        ]

        result = assess_reconstruction_compliance(events, eligible_steps=(1, 2))

        self.assertTrue(result["activated"])
        self.assertEqual(
            result["criteria"],
            {
                "artifact_loaded": True,
                "required_entrypoints": True,
                "prospective_commitments": True,
                "prediction_coverage": True,
                "nonempty_predictions": True,
                "action_sensitive_predictions": True,
                "update_coverage": True,
                "artifact_identity_bound": True,
            },
        )

    def test_file_creation_without_prospective_use_is_not_activation(self) -> None:
        events = [
            {
                "event_index": 1,
                "type": "artifact_loaded",
                "step": 1,
                "artifact_sha256": "a" * 64,
                "entrypoints": ["parse_observation", "predict", "update"],
                "ok": True,
            },
            {"event_index": 2, "type": "observation_revealed", "step": 1},
            {
                "event_index": 3,
                "type": "prediction_committed",
                "step": 1,
                "artifact_sha256": "a" * 64,
                "prediction": {},
            },
        ]

        result = assess_reconstruction_compliance(events, eligible_steps=(1,))

        self.assertFalse(result["activated"])
        self.assertFalse(result["criteria"]["prospective_commitments"])
        self.assertFalse(result["criteria"]["nonempty_predictions"])


class _CompliantPolicy:
    def predict(
        self,
        *,
        step: int,
        observation: dict,
        recorded_action: int,
        legal_actions: tuple[int, ...],
    ) -> PolicyPrediction:
        predictions = {
            str(action): {"next_marker": step * 10 + action}
            for action in legal_actions
        }
        return PolicyPrediction(
            prediction=predictions[str(recorded_action)],
            candidate_predictions=predictions,
            artifact_sha256="a" * 64,
            artifact_entrypoints=("parse_observation", "predict", "update"),
            artifact_loaded=True,
        )

    def update(
        self,
        *,
        step: int,
        recorded_action: int,
        observation: dict,
    ) -> PolicyUpdate:
        return PolicyUpdate(ok=True)


class _DirectPredictionPolicy:
    def predict(
        self,
        *,
        step: int,
        observation: dict,
        recorded_action: int,
        legal_actions: tuple[int, ...],
    ) -> PolicyPrediction:
        return PolicyPrediction(
            prediction={"next_marker": step},
            candidate_predictions={},
        )

    def update(
        self,
        *,
        step: int,
        recorded_action: int,
        observation: dict,
    ) -> PolicyUpdate:
        return PolicyUpdate(ok=True)


class ReplayRunnerTests(unittest.TestCase):
    def test_required_run_persists_an_activated_compliance_result(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            output = root / "run"
            _write_trace(trace)
            replay = SequentialReplay.from_trace(trace, first_step=1, last_step=2)

            result = execute_replay(
                replay=replay,
                policy=_CompliantPolicy(),
                condition="required",
                output_dir=output,
            )

            self.assertTrue(result["compliance"]["activated"])
            self.assertEqual(result["revealed_steps"], [1, 2])
            persisted = json.loads((output / "compliance.json").read_text())
            self.assertTrue(persisted["activated"])
            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text().splitlines()
            ]
            for step in (1, 2):
                commitment = next(
                    event["event_index"]
                    for event in events
                    if event["type"] == "prediction_committed" and event["step"] == step
                )
                reveal = next(
                    event["event_index"]
                    for event in events
                    if event["type"] == "observation_revealed" and event["step"] == step
                )
                self.assertLess(commitment, reveal)

    def test_required_run_refuses_to_reveal_without_an_executable_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            _write_trace(trace)
            replay = SequentialReplay.from_trace(trace, first_step=1, last_step=2)

            with self.assertRaisesRegex(RuntimeError, "required reconstruction artifact"):
                execute_replay(
                    replay=replay,
                    policy=_DirectPredictionPolicy(),
                    condition="required",
                    output_dir=root / "run",
                )

            events = [
                json.loads(line)
                for line in (root / "run" / "events.jsonl").read_text().splitlines()
            ]
            self.assertNotIn("observation_revealed", {event["type"] for event in events})


class _Result:
    def __init__(self, *, ok: bool, stdout: str = "", stderr: str = "") -> None:
        self.ok = ok
        self.stdout = stdout
        self.stderr = stderr
        self.traceback = None

    def model_text(self) -> str:
        return json.dumps({"ok": self.ok, "stdout": self.stdout, "stderr": self.stderr})


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.namespace = {"__name__": "__workspace__"}

    def execute(self, code: str) -> _Result:
        stdout = io.StringIO()
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            with contextlib.redirect_stdout(stdout):
                exec(code, self.namespace, self.namespace)
            return _Result(ok=True, stdout=stdout.getvalue())
        except BaseException as error:
            return _Result(ok=False, stdout=stdout.getvalue(), stderr=repr(error))
        finally:
            os.chdir(previous)


class _ChatResponse:
    def __init__(self, message: dict) -> None:
        self.message = message
        self.usage = {}


class _Client:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    def complete(self, *, messages, tools):
        self.requests.append({"messages": messages, "tools": tools})
        return _ChatResponse(self.responses.pop(0))


class _RequestError(RuntimeError):
    request_payload = {"model": "test", "messages": []}
    request_payload_sha256 = "b" * 64
    request_payload_encoding = "test-json"
    response_body_sha256 = "c" * 64
    response_body_raw = "failure"
    response_body_base64 = "ZmFpbHVyZQ=="


class _FailingClient:
    def complete(self, *, messages, tools):
        raise _RequestError("failed request")


class ModelReplayPolicyTests(unittest.TestCase):
    def test_required_policy_invokes_the_artifact_instead_of_trusting_prose(self) -> None:
        source = """\
import json

def parse_observation(path):
    return {"seen": 1}

def predict(state, action):
    return {"artifact_value": state["seen"] + action}

def update(state, action, observation_path):
    state["seen"] += 1
    return state
"""
        tool_code = (
            "from pathlib import Path\n"
            f"Path('world_model.py').write_text({source!r}, encoding='utf-8')"
        )
        responses = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "write-model",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": json.dumps({"code": tool_code}),
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "prediction": {"untrusted_direct_value": 999},
                        "candidate_predictions": {},
                    }
                ),
            },
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            policy = ModelReplayPolicy(
                client=_Client(responses),
                workspace=_Workspace(root),
                condition="required",
                system_prompt="Build a model.",
                max_rounds=3,
            )

            prediction = policy.predict(
                step=1,
                observation={"step": 0, "pixels": [[[0]]], "available_actions": [1, 2]},
                recorded_action=2,
                legal_actions=(1, 2),
            )
            update = policy.update(
                step=1,
                recorded_action=2,
                observation={"step": 1, "pixels": [[[1]]], "available_actions": [1, 2]},
            )

            self.assertTrue(prediction.artifact_loaded)
            self.assertEqual(prediction.prediction, {"artifact_value": 3})
            self.assertEqual(
                prediction.candidate_predictions,
                {
                    "1": {"artifact_value": 2},
                    "2": {"artifact_value": 3},
                },
            )
            self.assertTrue(update.ok)
            state = json.loads((root / "reconstruction-state.json").read_text())
            self.assertEqual(state, {"seen": 2})

    def test_failed_model_request_is_counted_and_retains_available_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            policy = ModelReplayPolicy(
                client=_FailingClient(),
                workspace=_Workspace(Path(directory)),
                condition="optional",
                system_prompt="Build a model.",
            )

            with self.assertRaisesRegex(_RequestError, "failed request"):
                policy.predict(
                    step=1,
                    observation={"step": 0, "pixels": [[[0]]], "available_actions": [1]},
                    recorded_action=1,
                    legal_actions=(1,),
                )

            self.assertEqual(policy.model_calls, 1)
            self.assertEqual(len(policy.request_evidence), 1)
            self.assertEqual(
                policy.request_evidence[0]["request_payload_sha256"],
                "b" * 64,
            )


class ReplayCliTests(unittest.TestCase):
    def test_dry_run_writes_a_replay_plan_without_model_calls(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            output = root / "output"
            _write_trace(trace)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(EXPERIMENT_ROOT / "run_replay.py"),
                    "--trace",
                    str(trace),
                    "--output",
                    str(output),
                    "--first-step",
                    "1",
                    "--last-step",
                    "2",
                    "--dry-run",
                ],
                capture_output=True,
                check=False,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            plan = [
                json.loads(line)
                for line in (output / "replay-plan.jsonl").read_text().splitlines()
            ]
            self.assertEqual(manifest["status"], "dry_run")
            self.assertEqual(manifest["model_calls"], 0)
            self.assertEqual(manifest["eligible_steps"], [1, 2])
            self.assertEqual([record["recorded_action"] for record in plan], [1, 2])
            self.assertNotIn("next_observation", plan[0])


if __name__ == "__main__":
    unittest.main()
