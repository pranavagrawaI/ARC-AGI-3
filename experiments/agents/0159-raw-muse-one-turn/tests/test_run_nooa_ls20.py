from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "run_nooa_ls20.py"
SPEC = importlib.util.spec_from_file_location("run_nooa_ls20", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class State:
    value = "NOT_FINISHED"


class Frame:
    frame = np.zeros((1, 2, 2), dtype=np.int8)
    state = State()
    levels_completed = 0
    full_reset = False
    available_actions = (1, 2, 3, 4)


def test_frame_record_contains_only_public_turn_metadata(tmp_path):
    record = MODULE.frame_record(turn=3, frame=Frame(), image_path=tmp_path / "f.png")
    assert record == {
        "type": "observation",
        "turn": 3,
        "image_path": str(tmp_path / "f.png"),
        "state": "NOT_FINISHED",
        "levels_completed": 0,
        "full_reset": False,
        "legal_actions": ["UP", "DOWN", "LEFT", "RIGHT"],
    }


def test_action_interface_uses_human_labels_and_keeps_sdk_ids_host_only():
    expected = {
        0: "RESET",
        1: "UP",
        2: "DOWN",
        3: "LEFT",
        4: "RIGHT",
        5: "SPACEBAR",
        6: "CLICK",
        7: "UNDO",
    }

    assert {
        action_id: MODULE.action_name(action_id) for action_id in expected
    } == expected
    assert {label: MODULE.game_action(label).value for label in expected.values()} == {
        label: action_id for action_id, label in expected.items()
    }


def test_worker_environment_does_not_forward_arc_key(monkeypatch):
    monkeypatch.setenv("PATH", "/test/bin")
    monkeypatch.setenv("ARC_API_KEY", "environment-secret")
    monkeypatch.setenv("ARC_LLM_API_KEY", "model-secret")
    monkeypatch.setenv("DATABASE_URL", "another-secret")
    monkeypatch.setenv("NOOA_0159_SCRIPTED_RESPONSES", '["fake"]')
    environment = MODULE.worker_environment(
        Path("/tmp/provider-events.jsonl"), variant="codeact"
    )
    assert environment["ARC_LLM_API_KEY"] == "model-secret"
    assert environment["PATH"] == "/test/bin"
    assert "ARC_API_KEY" not in environment
    assert "DATABASE_URL" not in environment
    assert "NOOA_0159_SCRIPTED_RESPONSES" not in environment
    assert environment["NOOA_0159_AUDIT_LOG"] == "/tmp/provider-events.jsonl"
    assert environment["NOOA_0159_VARIANT"] == "codeact"


def test_durable_beliefs_variant_gets_only_its_run_local_knowledge_path(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ARC_LLM_API_KEY", "model-secret")
    knowledge_dir = tmp_path / "knowledge"

    environment = MODULE.worker_environment(
        tmp_path / "provider-events.jsonl",
        variant="codeact_notes",
        knowledge_dir=knowledge_dir,
    )

    assert environment["NOOA_0159_VARIANT"] == "codeact_notes"
    assert environment["NOOA_0159_KNOWLEDGE_DIR"] == str(knowledge_dir.resolve())


def test_append_jsonl_round_trips(tmp_path):
    path = tmp_path / "trace.jsonl"
    MODULE.append_jsonl(path, {"turn": 1, "action": 2})
    assert json.loads(path.read_text()) == {"turn": 1, "action": 2}


def test_action_cap_must_be_positive(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "unused")
    with pytest.raises(ValueError, match="positive"):
        MODULE.run(tmp_path / "run", action_cap=0)


def test_transition_record_is_public_and_deterministic():
    before = Frame()
    after = Frame()
    after.frame = np.array([[[0, 1], [0, 0]]], dtype=np.int8)
    record = MODULE.transition_record(
        turn=2, action="RIGHT", x=None, y=None, before=before, after=after
    )

    assert record == {
        "turn": 2,
        "action": "RIGHT",
        "x": None,
        "y": None,
        "before": [[0, 0], [0, 0]],
        "after": [[0, 1], [0, 0]],
        "levels_completed": 0,
        "state": "NOT_FINISHED",
        "full_reset": False,
    }
    assert MODULE.trajectory_sha256([record]) == MODULE.trajectory_sha256([record])


def test_grid_values_uses_final_grid_from_arc_animation_sequence():
    frame = Frame()
    frame.frame = np.array(
        [
            [[0, 0], [0, 0]],
            [[0, 1], [0, 0]],
            [[0, 1], [2, 0]],
        ],
        dtype=np.int8,
    )

    assert MODULE.grid_values(frame) == [[0, 1], [2, 0]]


def test_animation_event_preserves_every_returned_frame(tmp_path):
    frame = Frame()
    frame.frame = np.array(
        [
            [[0, 0], [0, 0]],
            [[0, 1], [0, 0]],
            [[0, 1], [2, 0]],
        ],
        dtype=np.int8,
    )

    record = MODULE.save_animation_event(frame, turn=7, animations_dir=tmp_path)

    assert record is not None
    assert record["type"] == "animation_event"
    assert record["turn"] == 7
    assert record["frame_count"] == 3
    event_path = Path(record["event_path"])
    event = json.loads(event_path.read_text())
    assert event["frames"] == [
        [[0, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 1], [2, 0]],
    ]
    assert len(event["image_paths"]) == 3
    assert all(Path(path).is_file() for path in event["image_paths"])


@pytest.mark.parametrize("variant", ["codeact", "codeact_notes"])
def test_codeact_host_supplies_growing_trajectory_and_executes_one_action_per_turn(
    tmp_path, monkeypatch, variant
):
    class TestFrame:
        def __init__(self, value: int, levels: int = 0):
            self.frame = np.array([[[value, 0], [0, 0]]], dtype=np.int8)
            self.state = State()
            self.levels_completed = levels
            self.full_reset = False
            self.available_actions = (1, 2, 3, 4)

    class Environment:
        def __init__(self):
            self.steps = []

        def reset(self):
            return TestFrame(0)

        def step(self, action):
            self.steps.append(action)
            successor = TestFrame(
                len(self.steps), levels=1 if len(self.steps) == 2 else 0
            )
            if len(self.steps) == 1:
                successor.frame = np.array(
                    [
                        [[0, 0], [0, 0]],
                        [[1, 0], [0, 0]],
                    ],
                    dtype=np.int8,
                )
            return successor

        def close(self):
            return None

    environment = Environment()

    class Arcade:
        def __init__(self, **kwargs):
            pass

        def make(self, *args, **kwargs):
            return environment

    class Worker:
        instance = None

        def __init__(self, run_dir, *, variant):
            self.ready = {"variant": variant, "pid": 1, "model": "fake"}
            self.requests = []
            Worker.instance = self

        def decide(self, request):
            self.requests.append(json.loads(json.dumps(request)))
            return {
                "type": "decision",
                "sequence": request["sequence"],
                "action": "UP",
                "x": None,
                "y": None,
                "explanation": "probe",
            }

        def close(self):
            return None

    monkeypatch.setenv("ARC_API_KEY", "test")
    monkeypatch.setattr(MODULE.arc_agi, "Arcade", Arcade)
    monkeypatch.setattr(MODULE, "Worker", Worker)

    summary = MODULE.run(tmp_path / "run", action_cap=2, variant=variant)

    assert summary["success"] is True
    assert len(environment.steps) == 2
    assert Worker.instance is not None
    assert Worker.instance.requests[0]["trajectory"] == []
    second = Worker.instance.requests[1]["trajectory"]
    assert len(second) == 1
    assert second[0]["action"] == "UP"
    assert second[0]["x"] is None
    assert second[0]["y"] is None
    assert second[0]["before"] == [[0, 0], [0, 0]]
    assert second[0]["after"] == [[1, 0], [0, 0]]
    run_dir = tmp_path / "run"
    config = json.loads((run_dir / "run-config.json").read_text())
    assert config["schema"] == "0159-nooa-ls20-run-v15"
    assert config["action_interface"]["sdk_ids_model_visible"] is False
    assert config["action_interface"]["model_visible"] == [
        "RESET",
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT",
        "SPACEBAR",
        "CLICK x y",
        "UNDO",
    ]
    assert config["max_provider_calls_per_turn"] == 14
    assert config["analysis_max_iterations"] == 11
    assert config["commitment_max_attempts"] == 3
    assert config["commitment_interface"] == (
        "return_result(result=Decision) or exact JSON Decision"
    )
    assert config["commitment_tool_choice"] == (
        "named_return_result_requested_with_strict_json_fallback"
    )
    assert config["ipc_stdout"] == "duplicated_os_file_descriptor"
    assert config["arc_frame_sequence"] == "archive_all_use_final_as_decision_grid"
    assert config["host_action_fallback"] is False
    assert (config["durable_knowledge"] is not None) == (variant == "codeact_notes")
    expected_card = {
        "codeact": "0159-muse-glimmer-native-encoding-ls20-v12.md",
        "codeact_notes": "0159-muse-glimmer-human-action-labels-ls20-v15.md",
    }[variant]
    assert Path(config["experiment_card"]).name == expected_card
    assert set(config["source_sha256"]) == {
        "host",
        "worker",
        "worker_lock",
        "experiment_card",
        "shared_harness_runner",
        "shared_harness_types",
    }
    assert len(config["source_bundle_sha256"]) == 64
    persisted = [
        json.loads(line)
        for line in (run_dir / "trajectory.jsonl").read_text().splitlines()
    ]
    assert summary["trajectory_length"] == 2
    assert summary["trajectory_sha256"] == MODULE.trajectory_sha256(persisted)
    assert summary["animation_events"] == 1
    animation_rows = [
        row
        for row in (
            json.loads(line)
            for line in (run_dir / "transcript.jsonl").read_text().splitlines()
        )
        if row["type"] == "animation_event"
    ]
    assert len(animation_rows) == 1
    assert animation_rows[0]["frame_count"] == 2


def test_predict_request_physically_omits_trajectory(tmp_path, monkeypatch):
    class TestFrame:
        frame = np.zeros((1, 2, 2), dtype=np.int8)
        state = State()
        levels_completed = 0
        full_reset = False
        available_actions = (1, 2, 3, 4)

    class Environment:
        def reset(self):
            return TestFrame()

        def step(self, action):
            successor = TestFrame()
            successor.levels_completed = 1
            return successor

        def close(self):
            return None

    class Arcade:
        def __init__(self, **kwargs):
            pass

        def make(self, *args, **kwargs):
            return Environment()

    class Worker:
        request = None

        def __init__(self, run_dir, *, variant):
            self.ready = {
                "variant": variant,
                "pid": 1,
                "model": "fake",
                "client_kind": "live",
            }

        def decide(self, request):
            Worker.request = request
            return {
                "type": "decision",
                "sequence": request["sequence"],
                "action": "UP",
                "x": None,
                "y": None,
                "explanation": "probe",
            }

        def close(self):
            return None

    monkeypatch.setenv("ARC_API_KEY", "test")
    monkeypatch.setattr(MODULE.arc_agi, "Arcade", Arcade)
    monkeypatch.setattr(MODULE, "Worker", Worker)

    MODULE.run(tmp_path / "predict", action_cap=1, variant="predict")

    assert Worker.request is not None
    assert "trajectory" not in Worker.request


def test_step_exception_is_recorded_as_uncertain_action(tmp_path, monkeypatch):
    class TestFrame:
        frame = np.zeros((1, 2, 2), dtype=np.int8)
        state = State()
        levels_completed = 0
        full_reset = False
        available_actions = (1, 2, 3, 4)

    class Environment:
        def reset(self):
            return TestFrame()

        def step(self, action):
            raise TimeoutError("acknowledgement lost")

        def close(self):
            return None

    class Arcade:
        def __init__(self, **kwargs):
            pass

        def make(self, *args, **kwargs):
            return Environment()

    class Worker:
        def __init__(self, run_dir, *, variant):
            self.ready = {
                "variant": variant,
                "pid": 1,
                "model": "fake",
                "client_kind": "live",
            }

        def decide(self, request):
            return {
                "type": "decision",
                "sequence": request["sequence"],
                "action": "UP",
                "x": None,
                "y": None,
                "explanation": "probe",
            }

        def close(self):
            return None

    monkeypatch.setenv("ARC_API_KEY", "test")
    monkeypatch.setattr(MODULE.arc_agi, "Arcade", Arcade)
    monkeypatch.setattr(MODULE, "Worker", Worker)

    run_dir = tmp_path / "uncertain"
    summary = MODULE.run(run_dir, action_cap=1)
    rows = [
        json.loads(line)
        for line in (run_dir / "transcript.jsonl").read_text().splitlines()
    ]

    assert summary["stop_reason"] == "uncertain_environment_action"
    assert summary["valid"] is False
    assert summary["invalid_reason"] == "uncertain_environment_action"
    assert summary["actions"] == 0
    assert summary["action_attempts"] == 1
    assert [row["type"] for row in rows][-1] == "action_attempt"
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["stage"] == "environment_step"


def test_worker_startup_failure_rolls_back_process_and_log(tmp_path, monkeypatch):
    class Process:
        stdin = None
        stdout = None

        def __init__(self, *, stderr):
            self.stderr = stderr
            self.terminated = False
            self.waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            self.waited = True
            return 0

    captured = {}

    def popen(*args, **kwargs):
        process = Process(stderr=kwargs["stderr"])
        captured["process"] = process
        return process

    monkeypatch.setenv("ARC_LLM_API_KEY", "test")
    monkeypatch.setattr(MODULE.subprocess, "Popen", popen)
    monkeypatch.setattr(
        MODULE.Worker,
        "_read",
        lambda self, prefix: (_ for _ in ()).throw(RuntimeError("bad readiness")),
    )

    with pytest.raises(RuntimeError, match="bad readiness"):
        MODULE.Worker(tmp_path, variant="predict")

    process = captured["process"]
    assert process.terminated is True
    assert process.waited is True
    assert process.stderr.closed is True


def test_worker_close_kills_process_that_ignores_terminate(tmp_path):
    class Stdin:
        def write(self, value):
            return len(value)

        def flush(self):
            return None

    class Process:
        stdin = Stdin()
        wait_count = 0
        terminated = False
        killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout):
            self.wait_count += 1
            if self.wait_count < 3:
                raise subprocess.TimeoutExpired("worker", timeout)
            return 0

    worker = MODULE.Worker.__new__(MODULE.Worker)
    worker.process = Process()
    worker.stderr_handle = (tmp_path / "stderr.log").open("w", encoding="utf-8")

    worker.close()

    assert worker.process.terminated is True
    assert worker.process.killed is True
    assert worker.stderr_handle.closed is True
