# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from agent import NooaControlAgent, PublicActionPoint
from llm import (
    CONSUMED_CALL_OUTPUT,
    ActualCrossingRecorder,
    AuditedProviderClient,
    InjectedTransportFailure,
    ProviderTransportAbort,
    TransportDisposition,
    build_llm,
    classify_transport_failure,
    redact_for_audit,
)


def _profile(**overrides: object) -> dict[str, object]:
    profile: dict[str, object] = {
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
    profile.update(overrides)
    return profile


def test_provider_adapter_is_pruned_to_one_explicit_llm_boundary() -> None:
    path = Path(__file__).parents[1] / "llm.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "nooa_memory" not in imported_roots
    assert "dotenv" not in imported_roots
    assert "_load_dotenv" not in path.read_text(encoding="utf-8")
    assert "FakeLLMClient" not in path.read_text(encoding="utf-8")


def test_build_llm_uses_the_frozen_profile_and_disables_every_hidden_retry() -> None:
    client = build_llm(_profile(), api_key="test-secret")
    try:
        assert type(client).__name__ == "ResponsesClient"
        assert client.model == "openai/openai/gpt-5.6-sol"
        assert client.config == {
            "api_base": "https://gateway.invalid/v1",
            "api_key": "test-secret",
            "drop_params": True,
            "max_output_tokens": 16_384,
            "num_retries": 0,
            "reasoning_effort": "medium",
            "stream": False,
            "timeout": 180.0,
        }
        assert client.retry_config.max_retries == 0
        assert client.retry_config.rate_limit_extra_retries == 0
        assert client.retry_config.retry_on_empty_content is False
        assert client.cache_control_injection_points == []
    finally:
        client.close()


def test_provider_adapter_converts_chat_multimodal_blocks_to_responses_input() -> None:
    client = build_llm(_profile(), api_key="test-secret")
    try:
        transformed, instructions = client._transform_messages(
            [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "observe"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,aW1hZ2U="
                            },
                        },
                    ],
                },
            ]
        )
    finally:
        client.close()

    assert instructions == "system"
    assert transformed == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "observe"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,aW1hZ2U=",
                },
            ],
        }
    ]


def test_provider_adapter_flattens_nooa_response_batches_for_the_successor() -> None:
    client = build_llm(_profile(), api_key="test-secret")
    try:
        transformed, instructions = client._transform_messages(
            [
                {
                    "_batch": [
                        {
                            "id": "provider-item-id",
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "return_result",
                            "arguments": '{"result":{"action":1}}',
                            "status": "completed",
                        }
                    ],
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "return_result",
                                "arguments": '{"result":{"action":1}}',
                            },
                        }
                    ],
                }
            ]
        )
    finally:
        client.close()

    assert instructions is None
    # The batch is still flattened, and a call the host consumed is not replayed
    # as an open tool call. It keeps its own shape and gains the reply the host
    # owes it, so the model's record of the cell that produced its own action
    # survives without the host authoring a model turn.
    assert len(transformed) == 2
    assert transformed[0]["type"] == "function_call"
    assert transformed[0]["arguments"] == '{"result":{"action":1}}'
    assert transformed[1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": CONSUMED_CALL_OUTPUT,
    }


def test_build_llm_never_discovers_credentials_or_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARC_LLM_API_KEY", "ambient-secret")

    with pytest.raises(ValueError, match="api_key must be supplied explicitly"):
        build_llm(_profile(), api_key=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_surface", "chat-completions"),
        ("nooa_max_retries", 1),
        ("nooa_max_retries", False),
        ("nooa_rate_limit_extra_retries", 1),
        ("litellm_num_retries", 1),
        ("parallel_tool_calls", True),
        ("stream", 0),
        ("truncation", "auto"),
    ],
)
def test_build_llm_rejects_request_profile_drift(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="unsupported provider request profile"):
        build_llm(_profile(**{field: value}), api_key="test-secret")


def test_build_llm_rejects_unfrozen_extra_fields() -> None:
    with pytest.raises(ValueError, match="unknown provider request fields"):
        build_llm(_profile(extra_body={"surprise": True}), api_key="test-secret")


def test_audit_redaction_commits_credentials_and_image_bytes_without_retaining_them() -> None:
    image = b"\x89PNG\r\nsecret-image"
    image_url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    value = {
        "headers": {"Authorization": "Bearer secret-token", "x-safe": "kept"},
        "api_key": "body-secret",
        "message": "gateway rejected nvapi-12345678901234567890",
        "input": [{"type": "input_image", "image_url": image_url}],
    }

    redacted = redact_for_audit(
        value,
        known_credentials=("opaque-provider-key",),
    )
    encoded = json.dumps(redacted)

    assert "secret-token" not in encoded
    assert "body-secret" not in encoded
    assert "nvapi-12345678901234567890" not in encoded
    assert image_url not in encoded
    assert redacted["headers"]["x-safe"] == "kept"
    assert redacted["input"][0]["image_url"] == {
        "redacted": True,
        "kind": "data-image",
        "media_type": "image/png",
        "bytes": len(image),
        "sha256": hashlib.sha256(image).hexdigest(),
    }

    embedded = redact_for_audit(
        {"message": "provider echoed opaque-provider-key under an ordinary field"},
        known_credentials=("opaque-provider-key",),
    )
    assert embedded["message"]["kind"] == "known-credential-bearing-string"
    assert "opaque-provider-key" not in json.dumps(embedded)


@pytest.mark.asyncio
async def test_actual_crossing_records_http_bytes_on_both_sides_with_redaction() -> None:
    recorder = ActualCrossingRecorder(known_credentials=("opaque-provider-key",))
    image_url = "data:image/png;base64," + base64.b64encode(b"pixels").decode("ascii")

    async def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-provider-trace"] == "opaque-provider-key"
        return httpx.Response(
            200,
            json={
                "output": "ok",
                "access_token": "response-secret",
                "message": "echoed opaque-provider-key",
            },
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        event_hooks=recorder.async_event_hooks(),
    ) as client:
        response = await client.post(
            "https://user:pass@gateway.invalid/v1/responses?key=query-key&sig=query-sig",
            headers={"X-Provider-Trace": "opaque-provider-key"},
            json={"model": "m", "input": [{"image_url": image_url}]},
        )
        assert response.json()["output"] == "ok"

    crossing = recorder.records[0]
    encoded = json.dumps(crossing)
    assert crossing["request"]["body_sha256"]
    assert crossing["response"]["body_sha256"]
    assert crossing["response"]["status_code"] == 200
    assert crossing["request"]["headers"]["authorization"]["redacted"] is True
    assert crossing["request"]["headers"]["x-provider-trace"]["redacted"] is True
    assert crossing["request"]["body"]["input"][0]["image_url"]["kind"] == "data-image"
    assert "secret-token" not in encoded
    assert "response-secret" not in encoded
    assert "opaque-provider-key" not in encoded
    assert "user:pass" not in encoded
    assert "query-key" not in encoded
    assert "query-sig" not in encoded
    assert image_url not in encoded


class _FakeProvider:
    def __init__(self) -> None:
        self._http = None
        self.closed = False

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        messages[0]["content"] = "provider-mutated"
        return SimpleNamespace(
            content="ok",
            raw_response={"id": "response-1", "output": "ok"},
            usage={"total_tokens": 3},
        )

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_audited_client_snapshots_the_canonical_crossing_before_dispatch() -> None:
    provider = _FakeProvider()
    client = AuditedProviderClient(provider, _profile(), transport_max_retries=0)
    messages = [{"role": "user", "content": "before"}]

    response = await client.acall(messages, tools=[])

    assert response.content == "ok"
    assert client.audit_calls[0]["canonical_request"]["messages"] == [
        {"role": "user", "content": "before"}
    ]
    expected_canonical = {
        "provider_request_profile": _profile(),
        "messages": [{"role": "user", "content": "before"}],
        "tools": [],
        "kwargs": {},
    }
    expected_sha256 = hashlib.sha256(
        json.dumps(
            expected_canonical,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert client.audit_calls[0]["canonical_request_sha256"] == expected_sha256
    assert (
        client.audit_calls[0]["canonicalization"]
        == "json-utf8-sort-keys-compact-v1"
    )
    assert client.audit_calls[0]["actual_provider_crossings"] == []
    assert client.audit_calls[0]["outcome"]["kind"] == "success"
    assert client.audit_calls[0]["outcome"]["response"]["raw_response"]["id"] == "response-1"
    await client.aclose()
    assert provider.closed is True


@pytest.mark.asyncio
async def test_audited_client_pairs_canonical_input_with_its_owned_http_crossing() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "actual-response"}, request=request)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        headers={"Authorization": "Bearer provider-secret"},
    )

    class HttpProvider(_FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self._http = SimpleNamespace(httpx_async=http_client)

        async def acall(self, messages, tools=None, output_model=None, **kwargs):
            await http_client.post(
                "https://gateway.invalid/v1/responses",
                json={"model": "openai/openai/gpt-5.6-sol", "input": messages},
            )
            return _success()

        async def aclose(self) -> None:
            await http_client.aclose()
            await super().aclose()

    client = AuditedProviderClient(HttpProvider(), _profile(), transport_max_retries=0)

    await client.acall([{"role": "user", "content": "exact canonical input"}])

    audit = client.audit_calls[0]
    assert audit["canonical_request"]["messages"][0]["content"] == "exact canonical input"
    assert len(audit["actual_provider_crossings"]) == 1
    assert audit["actual_provider_crossings"][0]["request"]["body"]["input"] == [
        {"role": "user", "content": "exact canonical input"}
    ]
    encoded = json.dumps(audit)
    assert "provider-secret" not in encoded
    await client.aclose()


@pytest.mark.asyncio
async def test_required_actual_crossing_cannot_silently_fall_back_to_reconstruction() -> None:
    provider = _FakeProvider()
    provider._http = SimpleNamespace(
        httpx_async=SimpleNamespace(event_hooks={"request": [], "response": []})
    )
    client = AuditedProviderClient(
        provider,
        _profile(),
        transport_max_retries=0,
        require_actual_crossings=True,
    )

    with pytest.raises(ProviderTransportAbort) as caught:
        await client.acall([{"role": "user", "content": "go"}])

    assert caught.value.disposition is TransportDisposition.FATAL
    assert client.audit_calls[0]["outcome"]["error_type"] == "MissingActualProviderCrossing"


class _ScriptedProvider(_FakeProvider):
    def __init__(self, outcomes: list[object]) -> None:
        super().__init__()
        self.outcomes = outcomes
        self.calls = 0

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _success() -> SimpleNamespace:
    return SimpleNamespace(
        content="ok",
        raw_response={"id": "response-1", "output": "ok"},
        usage={"total_tokens": 3},
    )


@pytest.mark.asyncio
async def test_retryable_transport_failure_is_retried_only_by_the_audited_wrapper() -> None:
    provider = _ScriptedProvider(
        [
            InjectedTransportFailure(TransportDisposition.RETRYABLE),
            _success(),
        ]
    )
    client = AuditedProviderClient(provider, _profile(), transport_max_retries=2)

    response = await client.acall([{"role": "user", "content": "go"}])

    assert response.content == "ok"
    assert provider.calls == 2
    assert [attempt["outcome"] for attempt in client.audit_calls[0]["attempts"]] == [
        "retryable_failure",
        "success",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disposition",
    [TransportDisposition.FATAL, TransportDisposition.UNCERTAIN],
)
async def test_fatal_and_uncertain_transport_failures_are_never_retried(
    disposition: TransportDisposition,
) -> None:
    provider = _ScriptedProvider(
        [InjectedTransportFailure(disposition), _success()]
    )
    client = AuditedProviderClient(provider, _profile(), transport_max_retries=2)

    with pytest.raises(ProviderTransportAbort) as caught:
        await client.acall([{"role": "user", "content": "go"}])

    assert caught.value.disposition is disposition
    assert provider.calls == 1
    assert client.audit_calls[0]["outcome"]["kind"] == "transport_failure"
    assert client.audit_calls[0]["outcome"]["disposition"] == disposition.value


@pytest.mark.asyncio
async def test_transport_abort_bypasses_nooa_generation_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Image.new("RGB", (1, 1)).save(tmp_path / "current_frame.png")
    provider = _ScriptedProvider(
        [
            InjectedTransportFailure(TransportDisposition.UNCERTAIN),
            _success(),
        ]
    )
    client = AuditedProviderClient(provider, _profile(), transport_max_retries=2)
    agent = NooaControlAgent(llm=client, workspace=str(tmp_path), max_decision_repairs=2)

    with pytest.raises(ProviderTransportAbort):
        await agent.choose_action(
            PublicActionPoint(observation={"step": 0}, legal_actions=(1,))
        )

    assert provider.calls == 1
    await agent.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_retryable_transport_failure_stops_at_the_frozen_attempt_budget() -> None:
    provider = _ScriptedProvider(
        [InjectedTransportFailure(TransportDisposition.RETRYABLE) for _ in range(4)]
    )
    client = AuditedProviderClient(provider, _profile(), transport_max_retries=2)

    with pytest.raises(ProviderTransportAbort) as caught:
        await client.acall([{"role": "user", "content": "go"}])

    assert caught.value.disposition is TransportDisposition.RETRYABLE
    assert provider.calls == 3
    assert len(client.audit_calls[0]["attempts"]) == 3


def test_external_transport_failures_are_classified_conservatively() -> None:
    request = httpx.Request("POST", "https://gateway.invalid/v1/responses")

    assert classify_transport_failure(httpx.ConnectError("no route", request=request)) \
        is TransportDisposition.RETRYABLE
    assert classify_transport_failure(httpx.ConnectTimeout("connect", request=request)) \
        is TransportDisposition.RETRYABLE
    assert classify_transport_failure(httpx.ReadTimeout("after send", request=request)) \
        is TransportDisposition.UNCERTAIN
    assert classify_transport_failure(RuntimeError("unknown provider state")) \
        is TransportDisposition.UNCERTAIN

    fatal = RuntimeError("bad request")
    fatal.status_code = 400  # type: ignore[attr-defined]
    assert classify_transport_failure(fatal) is TransportDisposition.FATAL

    unavailable = RuntimeError("unavailable")
    unavailable.status_code = 503  # type: ignore[attr-defined]
    assert classify_transport_failure(unavailable) is TransportDisposition.UNCERTAIN


def test_live_response_batch_answers_its_consumed_call_and_keeps_the_rest() -> None:
    """Drive the real envelope from run 20260802T160203Z.

    Three earlier attempts at this fix passed against FakeLLMClient's chat shape
    -- ``{"role": "assistant", "tool_calls": [...]}`` -- which a real Responses
    call never produces. Production returns ``{"_batch": [...]}``, and the
    ``function_call`` items inside do not exist until ``_expand_response_batches``
    runs, well past where those tests were looking. This fixture is the recorded
    envelope from the run that failed, so the shape cannot drift from production
    without the test noticing.

    Past reasoning is required for model performance, so the reasoning item must
    survive exactly. So must the model's own authoring cell: a fourth attempt
    rewrote the consumed call into assistant prose, and run 20260803T062052Z then
    ended its second action point by reproducing that prose verbatim instead of
    acting. Every item the model wrote now reaches the provider unchanged, and
    the only addition is the host's reply in the host's own slot.
    """

    envelope = json.loads(
        (Path(__file__).parent / "live_batch_20260802T160203Z.json").read_text(
            encoding="utf-8"
        )
    )
    original = json.loads(json.dumps(envelope))
    consumed = [
        item["call_id"]
        for item in envelope["_batch"]
        if item.get("type") == "function_call"
    ]
    assert len(consumed) == 2, "the recorded turn issued two calls"

    client = build_llm(_profile(), api_key="test-secret")
    try:
        # Answer the first call; the second is the one the host consumed.
        transformed, _instructions = client._transform_messages(
            [
                envelope,
                {
                    "role": "tool",
                    "tool_call_id": consumed[0],
                    "content": "status: complete",
                },
            ]
        )
    finally:
        client.close()

    surviving = [
        item.get("call_id")
        for item in transformed
        if item.get("type") == "function_call"
    ]
    assert surviving == consumed, "the model's own calls must reach the provider intact"

    reasoning = [item for item in transformed if item.get("type") == "reasoning"]
    assert len(reasoning) == 1, "past reasoning must reach the provider"
    assert reasoning[0] == original["_batch"][0], "reasoning must be untouched"

    # Every call is answered, which is what ModelRun requires, and each reply
    # follows the call it answers.
    answers = {
        item["call_id"]: index
        for index, item in enumerate(transformed)
        if item.get("type") == "function_call_output"
    }
    calls = {
        item["call_id"]: index
        for index, item in enumerate(transformed)
        if item.get("type") == "function_call"
    }
    assert set(answers) == set(consumed), "every call must be answered"
    assert all(answers[call_id] > calls[call_id] for call_id in consumed)

    consumed_answer = transformed[answers[consumed[1]]]
    assert consumed_answer["output"] == CONSUMED_CALL_OUTPUT

    # The host speaks only in its own slot. Nothing the model did not write may
    # appear as a model turn, because the next action point learns from shape.
    assert not [
        item
        for item in transformed
        if item.get("role") == "assistant" and item not in original["_batch"]
    ], "the host must not author an assistant turn"
