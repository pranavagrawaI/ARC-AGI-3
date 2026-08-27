"""Hermetic Python 3.14 host -> Python 3.12 Tycho-NOOA worker smoke."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from arc_host.process import JsonlWorker
from arc_host.runner import run_episode
from arc_host.trace import RecordingSolver  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[3]
SOLVER = ROOT / "solvers/tycho_nooa"
sys.path.insert(0, str(ROOT))

from tools.hermetic_gate_core import audit_run  # type: ignore[import-not-found]


def commitment(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def frame(value: int, *, actions: tuple[int, ...], levels: int = 0):
    return SimpleNamespace(
        frame=np.asarray([[[value, value + 1], [value + 2, value + 3]]]),
        available_actions=actions,
        levels_completed=levels,
        state="NOT_FINISHED",
        full_reset=False,
    )


class Environment:
    def __init__(self):
        self.frames = iter(
            [
                frame(0, actions=(1,)),
                frame(4, actions=(1,)),
                frame(8, actions=(), levels=1),
            ]
        )
        self.actions = []

    def reset(self):
        return next(self.frames)

    def step(self, action, data=None, reasoning=None):
        assert action.value == 1
        assert data is None
        self.actions.append(action)
        return next(self.frames)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tycho-nooa-ipc-") as directory:
        root = Path(directory)
        environment = {
            key: os.environ[key]
            for key in ("HOME", "LANG", "PATH", "SSL_CERT_FILE")
            if key in os.environ
        }
        environment["TYCHO_NOOA_SCRIPTED_TOOL_CALLS"] = json.dumps(
            [
                {
                    "tool": "run_python",
                    "arguments": {"code": "print('sandbox ready')"},
                },
                {
                    "tool": "write_file",
                    "arguments": {
                        "path": "notes/ipc.md",
                        "content": "This crossed the Python 3.14/3.12 boundary.",
                    },
                },
                {"tool": "take_action", "arguments": {"action": "UP"}},
                {
                    "tool": "read_file",
                    "arguments": {"path": "level_0/turn_001.txt"},
                },
                {"tool": "take_action", "arguments": {"action": "UP"}},
                {
                    "tool": "write_file",
                    "arguments": {
                        "path": "notes/level_0_insights.md",
                        "content": "Two on-policy UP actions completed level 0.",
                    },
                },
            ]
        )
        environment["TYCHO_SANDBOX_RUNTIME"] = "host"
        environment["TYCHO_MODE"] = "no_world_model"
        with JsonlWorker(
            [
                "uv",
                "run",
                "--frozen",
                "--project",
                str(SOLVER),
                "tycho-nooa-worker",
            ],
            stdout_log=root / "worker-stdout.log",
            stderr_log=root / "worker-stderr.log",
            timeout_seconds=30,
            environment=environment,
        ) as worker:
            solver = RecordingSolver(worker, root / "transcript.jsonl")
            game = Environment()
            result = run_episode(
                game,
                solver,
                game_id="fake-game",
                seed=0,
                workspace_root=root / "workspace",
                max_actions=2,
                stop_after_levels=1,
            )

        note = root / "workspace/fake/notes/ipc.md"
        tool_trace = [
            json.loads(line)
            for line in (root / "tool-trace.jsonl").read_text().splitlines()
        ]
        transcript = [
            json.loads(line)
            for line in (root / "transcript.jsonl").read_text().splitlines()
        ]
        action_results = [
            record for record in transcript if record["type"] == "action_result"
        ]
        turns = [record for record in transcript if record["type"] == "turn"]
        committed_actions = [
            record for record in tool_trace if record["name"] == "take_action"
        ]
        checkpoint_requests = [
            json.loads(path.read_text())
            for path in sorted((root / "decision-checkpoints/calls").glob("*/request.json"))
        ]
        checkpoint_responses = [
            json.loads(path.read_text())
            for path in sorted((root / "decision-checkpoints/calls").glob("*/response.json"))
        ]
        requested_calls = [
            (call["id"], call["name"], json.loads(call["arguments"]))
            for response in checkpoint_responses
            for call in response["tool_calls"]
        ]
        observed_calls = [
            (record["tool_call_id"], record["name"], record["arguments"])
            for record in tool_trace
        ]
        assert result.stop_reason == "level_limit"
        assert result.actions == 2
        assert len(game.actions) == 2
        assert tool_trace[0]["name"] == "run_python"
        assert tool_trace[0]["result"].strip() == "sandbox ready"
        actor_requests = [
            request
            for request in checkpoint_requests
            if request["source"] == "actor"
            and request["decision_point"]["phase"] == "decision"
        ]
        assert actor_requests
        assert {
            request["source"]
            for request in checkpoint_requests
            if request["decision_point"]["phase"] == "decision"
        } == {"actor"}
        assert {
            (
                request["decision_point"]["sequence"],
                request["decision_point"]["turn_in_level"],
            )
            for request in actor_requests
        } == {(0, 0), (1, 1)}
        post_action_sources = {
            request["source"]
            for request in checkpoint_requests
            if request["decision_point"]["phase"] == "post_action"
        }
        assert post_action_sources == {"scribe"}
        assert len({call_id for call_id, _, _ in requested_calls}) == len(
            requested_calls
        )
        assert requested_calls == observed_calls
        assert {record["source"] for record in tool_trace[:-1]} == {"actor"}
        assert tool_trace[-1]["source"] == "scribe"
        assert tool_trace[-1]["phase"] == "post_action"
        assert all(record["execution_status"] == "complete" for record in tool_trace)
        for sequence in (0, 1):
            assert committed_actions[sequence]["sequence"] == sequence
            assert action_results[sequence]["sequence"] == sequence
            assert committed_actions[sequence]["result"] == action_results[sequence][
                "decision"
            ]
        assert action_results[0]["before"]["decision_frame"] == [[0, 1], [2, 3]]
        assert action_results[0]["after"]["decision_frame"] == [[4, 5], [6, 7]]
        assert turns[1]["observation"] == action_results[0]["after"]
        assert turns[1]["previous_decision"] == action_results[0]["decision"]
        assert action_results[1]["after"]["decision_frame"] == [[8, 9], [10, 11]]
        assert note.read_text() == "This crossed the Python 3.14/3.12 boundary."
        assert (
            root / "workspace/fake/notes/level_0_insights.md"
        ).read_text() == "Two on-policy UP actions completed level 0."
        assert not (root / "worker-stderr.log").read_text()
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "mode": "no_world_model",
                    "actions": result.actions,
                    "levels_completed": result.levels_completed,
                }
            )
        )
        gate = audit_run(root)
        assert gate["status"] == "pass", gate["violations"]

        tool_trace_path = root / "tool-trace.jsonl"
        mutated_tools = [dict(record) for record in tool_trace]
        mutated_tools[0]["raw_arguments"] = json.dumps({"code": "print('forged')"})
        tool_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in mutated_tools)
        )
        assert "checkpoint_tool_arguments_mismatch" in {
            row["code"] for row in audit_run(root)["violations"]
        }

        mutated_tools[0] = dict(tool_trace[0])
        mutated_tools[0]["validated_arguments"] = {"unexpected": True}
        tool_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in mutated_tools)
        )
        assert "validated_tool_arguments_mismatch" in {
            row["code"] for row in audit_run(root)["violations"]
        }

        mutated_tools[0] = dict(tool_trace[0])
        mutated_tools[0]["validated_arguments"] = {
            **mutated_tools[0]["validated_arguments"],
            "code": "print('different')",
        }
        tool_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in mutated_tools)
        )
        assert "validated_tool_arguments_not_derived" in {
            row["code"] for row in audit_run(root)["violations"]
        }

        mutated_tools[0] = dict(tool_trace[0])
        mutated_tools[0].pop("sequence")
        tool_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in mutated_tools)
        )
        assert "missing_tool_sequence" in {
            row["code"] for row in audit_run(root)["violations"]
        }
        tool_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in tool_trace)
        )

        first_request_path = min(
            (root / "decision-checkpoints/calls").glob("*/request.json")
        )
        first_request = json.loads(first_request_path.read_text())
        downgraded_request = dict(first_request)
        downgraded_request.pop("schema_version")
        first_request_path.write_text(json.dumps(downgraded_request))
        assert "unsupported_checkpoint_schema" in {
            row["code"] for row in audit_run(root)["violations"]
        }
        first_request_path.write_text(json.dumps(first_request))

        llm_trace_path = root / "llm-trace.jsonl"
        llm_trace = [
            json.loads(line) for line in llm_trace_path.read_text().splitlines()
        ]
        mutated_llm = [dict(record) for record in llm_trace]
        mutated_llm[0].pop("phase")
        llm_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in mutated_llm)
        )
        assert "invalid_llm_phase" in {
            row["code"] for row in audit_run(root)["violations"]
        }
        llm_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in llm_trace)
        )

        swapped_indexes = [dict(record) for record in llm_trace]
        swapped_indexes[0]["call_index"], swapped_indexes[1]["call_index"] = (
            swapped_indexes[1]["call_index"],
            swapped_indexes[0]["call_index"],
        )
        llm_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in swapped_indexes)
        )
        assert "checkpoint_llm_join_mismatch" in {
            row["code"] for row in audit_run(root)["violations"]
        }
        llm_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in llm_trace)
        )

        reordered_llm = [llm_trace[1], llm_trace[0], *llm_trace[2:]]
        llm_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in reordered_llm)
        )
        assert "checkpoint_llm_join_mismatch" in {
            row["code"] for row in audit_run(root)["violations"]
        }
        llm_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in llm_trace)
        )

        first_response_path = first_request_path.with_name("response.json")
        first_response = json.loads(first_response_path.read_text())
        undeclared_response = dict(first_response)
        undeclared_response["tool_calls"] = [
            {**first_response["tool_calls"][0], "name": "undeclared_tool"}
        ]
        undeclared_response.pop("response_payload_commitment")
        undeclared_response["response_payload_commitment"] = commitment(
            undeclared_response
        )
        undeclared_llm = [dict(record) for record in llm_trace]
        undeclared_llm[0]["tool_calls"] = ["undeclared_tool"]
        undeclared_tools = [dict(record) for record in tool_trace]
        undeclared_tools[0]["name"] = "undeclared_tool"
        first_response_path.write_text(json.dumps(undeclared_response))
        llm_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in undeclared_llm)
        )
        tool_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in undeclared_tools)
        )
        assert "unoffered_tool_execution" in {
            row["code"] for row in audit_run(root)["violations"]
        }
        first_response_path.write_text(json.dumps(first_response))
        llm_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in llm_trace)
        )
        tool_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in tool_trace)
        )

        extra_llm = [*llm_trace, {**llm_trace[-1], "call_index": 999}]
        llm_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in extra_llm)
        )
        assert "llm_checkpoint_join_missing" in {
            row["code"] for row in audit_run(root)["violations"]
        }
        llm_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in llm_trace)
        )
        print(
            "host-worker causal trace passed: the sequence-0 successor became the "
            "sequence-1 Actor observation before the terminal action"
        )


if __name__ == "__main__":
    main()
