# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from launcher import _FixtureLLM, run_offline_fixture


PROTOCOL = "0002-nooa-ipc-v1"


def _write_frame(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1)).save(workspace / "current_frame.png")


def _request(action_point_id: int, code: str) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "run_id": "fixture-run",
        "action_point_id": action_point_id,
        "observation": {"step": action_point_id},
        "legal_actions": [1, 2],
        "fixture_python_cells": [code],
    }


@pytest.mark.asyncio
async def test_fixture_audit_snapshots_messages_at_the_model_crossing() -> None:
    client = _FixtureLLM()
    client.enqueue(
        ["return_result({'action': 1, 'data': None, 'reason': 'fixture'})"]
    )
    messages = [{"role": "user", "content": "before"}]

    await client.acall(messages, tools=[])
    messages[0]["content"] = "after"

    assert client.audit_calls[0]["messages"][0]["content"] == "before"


@pytest.mark.asyncio
async def test_offline_launcher_closes_two_action_points_over_append_only_files(
    tmp_path: Path,
) -> None:
    requests = tmp_path / "ipc" / "requests.jsonl"
    responses = tmp_path / "ipc" / "responses.jsonl"
    workspace = tmp_path / "workspace"
    requests.parent.mkdir()
    _write_frame(workspace)
    requests.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                _request(
                    0,
                    "memory = 40\n"
                    "open('model.txt', 'w').write('kept')\n"
                    "return_result({'action': 1, 'data': None, 'reason': 'first'})",
                ),
                _request(
                    1,
                    "assert memory == 40\n"
                    "assert open('model.txt').read() == 'kept'\n"
                    "return_result({'action': 2, 'data': None, 'reason': 'second'})",
                ),
            )
        )
        + "\n"
    )

    completed = await run_offline_fixture(requests, responses, workspace)

    assert completed == 2
    records = [json.loads(line) for line in responses.read_text().splitlines()]
    assert [record["action_point_id"] for record in records] == [0, 1]
    assert [record["decision"]["action"] for record in records] == [1, 2]
    assert all(record["protocol"] == PROTOCOL for record in records)
    assert all(record["complete"] is True for record in records)
    first_call = records[0]["audit"]["model_calls"][0]
    assert first_call["messages"][0]["role"] == "system"
    observation_message = next(
        message
        for message in first_call["messages"]
        if isinstance(message.get("content"), list)
    )
    observation_payload = json.loads(observation_message["content"][0]["text"])
    assert observation_payload["step"] == 0
    assert "step" not in observation_payload["environment_observation"]
    assert observation_message["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert [tool["name"] for tool in first_call["tools"]] == [
        "execute_python",
        "return_result",
    ]
    assert first_call["tools"][0] == {
        "description": (
            "Execute Python 3 in the persistent isolated workspace. State and files "
            "persist for this run; the repository, game implementation, network, "
            "and prior runs are absent."
        ),
        "name": "execute_python",
        "parameters": {
            "additionalProperties": False,
            "properties": {
                "code": {
                    "description": "Python source to execute.",
                    "type": "string",
                }
            },
            "required": ["code"],
            "type": "object",
        },
    }
    assert list(
        first_call["tools"][1]["parameters"]["properties"]["result"]["properties"]
    ) == ["action", "data", "reason"]
    assert (workspace / "model.txt").read_text() == "kept"


@pytest.mark.asyncio
async def test_launcher_rejects_duplicate_action_point_identity(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    responses = tmp_path / "responses.jsonl"
    workspace = tmp_path / "workspace"
    record = _request(
        0,
        "return_result({'action': 1, 'data': None, 'reason': 'fixture'})",
    )
    requests.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="duplicate action point"):
        await run_offline_fixture(requests, responses, workspace)


@pytest.mark.asyncio
async def test_launcher_rejects_paired_spec_drift_inside_one_run(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    responses = tmp_path / "responses.jsonl"
    workspace = tmp_path / "workspace"
    first = _request(
        0,
        "return_result({'action': 1, 'data': None, 'reason': 'fixture'})",
    )
    second = _request(
        1,
        "return_result({'action': 1, 'data': None, 'reason': 'fixture'})",
    )
    first["paired_run_spec"] = {"max_context_turns": 1}
    second["paired_run_spec"] = {"max_context_turns": 2}
    requests.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")

    with pytest.raises(ValueError, match="paired_run_spec changed"):
        await run_offline_fixture(requests, responses, workspace)


def test_launcher_cli_runs_in_a_separate_python_process(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    responses = tmp_path / "responses.jsonl"
    workspace = tmp_path / "workspace"
    _write_frame(workspace)
    requests.write_text(
        json.dumps(
            _request(
                0,
                "return_result({'action': 2, 'data': None, 'reason': 'process'})",
            )
        )
        + "\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "launcher.py"),
            "--requests",
            str(requests),
            "--responses",
            str(responses),
            "--workspace",
            str(workspace),
            "--offline-fixture",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"protocol": PROTOCOL, "completed": 1}
    response = json.loads(responses.read_text())
    assert response["decision"]["action"] == 2


def test_long_lived_launcher_accepts_successor_after_first_action(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    responses = tmp_path / "responses.jsonl"
    workspace = tmp_path / "workspace"
    _write_frame(workspace)
    requests.touch()
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "launcher.py"),
            "--requests",
            str(requests),
            "--responses",
            str(responses),
            "--workspace",
            str(workspace),
            "--offline-fixture",
            "--serve",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def append(record: dict[str, object]) -> None:
        with requests.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    def wait_for_responses(count: int) -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if responses.exists() and len(responses.read_text().splitlines()) >= count:
                return
            if process.poll() is not None:
                break
            time.sleep(0.02)
        raise AssertionError(f"worker did not write response {count}; exit={process.poll()}")

    try:
        append(
            _request(
                0,
                "memory = 41\n"
                "return_result({'action': 1, 'data': None, 'reason': 'first'})",
            )
        )
        wait_for_responses(1)
        append(
            _request(
                1,
                "assert memory == 41\n"
                "return_result({'action': 2, 'data': None, 'reason': 'successor'})",
            )
        )
        wait_for_responses(2)
        append({"protocol": PROTOCOL, "kind": "stop"})
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process.returncode == 0, stderr
    assert json.loads(stdout) == {"protocol": PROTOCOL, "completed": 2}
    records = [json.loads(line) for line in responses.read_text().splitlines()]
    assert [record["decision"]["action"] for record in records] == [1, 2]
