# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from launcher import _FixtureLLM, _provider_client, run_offline_fixture, serve_provider
from llm import AuditedProviderClient, InjectedTransportFailure, TransportDisposition


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


def _provider_profile() -> dict[str, object]:
    return {
        "schema": "0002-nooa-provider-request-v2",
        "api_surface": "responses",
        "model": "openai/openai/gpt-5.6-sol",
        "wire_model": "openai/gpt-5.6-sol",
        "api_base": "https://gateway.invalid/v1",
        "max_output_tokens": 16_384,
        "reasoning_effort": "medium",
        "request_timeout_seconds": 180.0,
        "temperature": None,
        "seed": None,
        "stream": False,
        "drop_params": True,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "truncation": "disabled",
        "nooa_max_retries": 0,
        "nooa_rate_limit_extra_retries": 0,
        "litellm_num_retries": 0,
    }


@pytest.mark.parametrize("module_name", ["agent", "llm", "launcher"])
def test_fresh_runtime_import_is_local_only_and_skips_dotenv(
    tmp_path: Path,
    module_name: str,
) -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    (tmp_path / ".env").write_text("NOOA_DOTENV_SENTINEL=loaded\n")
    (tmp_path / "sitecustomize.py").write_text(
        "import httpx\n"
        "def forbidden_get(*args, **kwargs):\n"
        "    raise RuntimeError('network touched during import')\n"
        "httpx.get = forbidden_get\n"
    )
    environment = dict(os.environ)
    for name in (
        "LITELLM_LOCAL_MODEL_COST_MAP",
        "LITELLM_MODE",
        "NOOA_DOTENV_SENTINEL",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(runtime_root))
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import json, os; import {module_name}; "
                "print(json.dumps({"
                "'cost_map': os.environ.get('LITELLM_LOCAL_MODEL_COST_MAP'), "
                "'mode': os.environ.get('LITELLM_MODE'), "
                "'dotenv': os.environ.get('NOOA_DOTENV_SENTINEL')}))"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "network touched during import" not in completed.stderr
    assert json.loads(completed.stdout) == {
        "cost_map": "True",
        "mode": "PRODUCTION",
        "dotenv": None,
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


def test_provider_client_is_constructed_only_from_the_ipc_profile_and_explicit_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SimpleNamespace(
        _http=SimpleNamespace(
            httpx_async=SimpleNamespace(event_hooks={"request": [], "response": []})
        )
    )
    built: list[tuple[object, object]] = []

    def fake_build(profile: object, *, api_key: object) -> object:
        built.append((profile, api_key))
        return provider

    monkeypatch.setattr("launcher.build_llm", fake_build)
    request = {
        "paired_run_spec": {
            "provider_request_profile": _provider_profile(),
            "transport_max_retries": 2,
        }
    }

    client = _provider_client(request, api_key="explicit-secret")

    assert client.provider is provider
    assert client.transport_max_retries == 2
    assert client.known_credentials == ("explicit-secret",)
    assert built == [(_provider_profile(), "explicit-secret")]


@pytest.mark.parametrize("profile", [None, {}, "not-an-object"])
def test_provider_client_rejects_missing_profile(profile: object) -> None:
    request = {
        "paired_run_spec": {
            "provider_request_profile": profile,
            "transport_max_retries": 2,
        }
    }
    with pytest.raises(ValueError, match="provider_request_profile"):
        _provider_client(request, api_key="explicit-secret")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("disposition", "expected_calls"),
    [
        (TransportDisposition.FATAL, 1),
        (TransportDisposition.UNCERTAIN, 1),
        (TransportDisposition.RETRYABLE, 3),
    ],
)
async def test_provider_serve_returns_transport_failure_evidence_without_an_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disposition: TransportDisposition,
    expected_calls: int,
) -> None:
    class FailureProvider:
        _http = None

        def __init__(self) -> None:
            self.calls = 0

        async def acall(self, *_args: object, **_kwargs: object) -> object:
            self.calls += 1
            raise InjectedTransportFailure(disposition)

        async def aclose(self) -> None:
            return None

    provider = FailureProvider()
    audited = AuditedProviderClient(provider, _provider_profile(), transport_max_retries=2)
    monkeypatch.setattr("launcher._provider_client", lambda *_args, **_kwargs: audited)
    requests = tmp_path / "requests.jsonl"
    responses = tmp_path / "responses.jsonl"
    workspace = tmp_path / "workspace"
    _write_frame(workspace)
    request = _request(0, "unused fixture cell")
    request.pop("fixture_python_cells")
    request["paired_run_spec"] = {
        "provider_request_profile": _provider_profile(),
        "transport_max_retries": 2,
    }
    stop = {"protocol": PROTOCOL, "kind": "stop"}
    requests.write_text(json.dumps(request) + "\n" + json.dumps(stop) + "\n")

    completed = await serve_provider(
        requests,
        responses,
        workspace,
        api_key="explicit-secret",
    )

    assert completed == 0
    assert provider.calls == expected_calls
    response = json.loads(responses.read_text())
    assert "decision" not in response
    assert response["error"] == {
        "kind": "provider_transport",
        "disposition": disposition.value,
        "attempts": expected_calls,
    }
    model_call = response["audit"]["model_calls"][0]
    assert model_call["outcome"]["disposition"] == disposition.value
    assert len(model_call["attempts"]) == expected_calls
    assert all(
        attempt["outcome"] == f"{disposition.value}_failure"
        for attempt in model_call["attempts"]
    )


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
