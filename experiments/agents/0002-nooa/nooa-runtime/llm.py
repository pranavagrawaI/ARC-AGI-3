# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One explicit, frozen NOOA provider boundary for 0002-NOOA.

Derived by copy-and-prune from the pinned NVIDIA NOOA ARC example recorded in
``../SOURCE_MANIFEST.json``. Configuration and credentials are arguments: this
module does not discover ``.env`` files, ambient variables, embeddings, memory,
fallback clients, or retry policy.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import StrEnum
from hashlib import sha256
import json
import os as _dependency_os
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# NOOA imports LiteLLM. Force its bundled registry and suppress its implicit
# dotenv loader before the dependency can execute either import-time path.
_dependency_os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
_dependency_os.environ["LITELLM_MODE"] = "PRODUCTION"
del _dependency_os

import httpx
from nooa.tracing._secret_scrubber import _is_sensitive_key, scrub_string
from nooa.unifiedllm import RetryConfig
from nooa.unifiedllm.unifiedllm import ResponsesClient as _NooaResponsesClient


@dataclass(frozen=True, slots=True)
class ProviderRequestProfile:
    """Static provider fields committed by a paired run before execution."""

    schema: str
    api_surface: str
    model: str
    wire_model: str
    api_base: str
    max_output_tokens: int
    reasoning_effort: str | None
    request_timeout_seconds: float
    temperature: float | None
    seed: int | None
    stream: bool
    drop_params: bool
    parallel_tool_calls: bool
    tool_choice: str
    truncation: str
    nooa_max_retries: int
    nooa_rate_limit_extra_retries: int
    litellm_num_retries: int

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ProviderRequestProfile":
        expected = {field.name for field in fields(cls)}
        received = set(record)
        extra = sorted(received - expected)
        missing = sorted(expected - received)
        if extra:
            raise ValueError(f"unknown provider request fields: {extra!r}")
        if missing:
            raise ValueError(f"missing provider request fields: {missing!r}")
        profile = cls(**dict(record))
        profile.validate()
        return profile

    def validate(self) -> None:
        fixed = {
            "schema": "0002-nooa-provider-request-v2",
            "api_surface": "responses",
            "stream": False,
            "drop_params": True,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "truncation": "disabled",
            "nooa_max_retries": 0,
            "nooa_rate_limit_extra_retries": 0,
            "litellm_num_retries": 0,
        }
        drift = {
            key: {"required": required, "received": getattr(self, key)}
            for key, required in fixed.items()
            if type(getattr(self, key)) is not type(required)
            or getattr(self, key) != required
        }
        if drift:
            raise ValueError(f"unsupported provider request profile: {drift!r}")
        if not all(
            isinstance(value, str)
            for value in (self.model, self.wire_model, self.api_base)
        ):
            raise ValueError("provider models and api_base must be strings")
        if not self.model or not self.wire_model or not self.api_base:
            raise ValueError("provider models and api_base must be non-empty")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be positive")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
        ):
            raise ValueError("temperature must be numeric or null")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or null")
        if self.reasoning_effort is not None and not isinstance(
            self.reasoning_effort,
            str,
        ):
            raise ValueError("reasoning_effort must be a string or null")


def _responses_content(role: str, content: Any) -> Any:
    """Project Chat-style multimodal blocks onto the Responses wire schema."""

    if not isinstance(content, list):
        return content
    converted: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        kind = part.get("type")
        if kind in {
            "input_text",
            "input_image",
            "input_file",
            "input_audio",
            "input_video",
            "output_text",
        }:
            converted.append(dict(part))
        elif kind == "text":
            converted.append(
                {
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": str(part.get("text", "")),
                }
            )
        elif kind == "image_url" and role == "user":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, Mapping) else None
            if isinstance(url, str):
                converted.append({"type": "input_image", "image_url": url})
    return converted


def _expand_response_batches(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for message in messages:
        batch = message.get("_batch")
        if not isinstance(batch, list):
            expanded.append(message)
            continue
        for raw_item in batch:
            if not isinstance(raw_item, Mapping):
                continue
            item = dict(raw_item)
            if item.get("type") == "function_call":
                expanded.append(
                    {
                        "type": "function_call",
                        "call_id": str(item.get("call_id", "")),
                        "name": str(item.get("name", "")),
                        "arguments": str(item.get("arguments", "")),
                    }
                )
            elif item.get("type") == "message":
                expanded.append(
                    {
                        "role": str(item.get("role", "assistant")),
                        "content": item.get("content", ""),
                    }
                )
            else:
                expanded.append(item)
    return expanded


def _without_unanswered_tool_calls(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop tool calls the host consumed instead of answering.

    An action point ends the moment ``return_result`` supplies its argument: the
    host takes it and releases the environment action, so whichever call carried
    it never receives a reply. That is the ``return_result`` call when the model
    uses the tool directly, and the ``execute_python`` call when the cell ends in
    an inline ``return_result``. A repair crossing reissues the action point, so
    the consumed call also reaches the provider through the live turn.

    Providers disagree about whether an open call is legal. ModelRun refuses --
    "every tool_call must be answered by a tool message before the conversation
    continues" -- and CoreWeave accepts, so load balancing hid this until a run
    pinned one endpoint.

    This runs on the final Responses items, after ``_expand_response_batches``,
    because that is the only point where every route into the request has one
    shape. Filtering the agent's message list instead misses ``_batch``
    envelopes, whose ``function_call`` items do not exist yet.

    Reasoning items are never removed. Past reasoning is required for model
    performance, and only unanswered ``function_call`` items are dropped.
    """

    answered = {
        item.get("call_id")
        for item in items
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    }
    kept: list[dict[str, Any]] = []
    for item in items:
        consumed = (
            isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("call_id") not in answered
        )
        if not consumed:
            kept.append(item)
            continue
        # Keep what the call said. Dropping it outright would also drop the
        # model's record of the cell that produced its own action, which the
        # workspace keeps but the transcript would not.
        kept.append(
            {
                "role": "assistant",
                "content": (
                    f"[this action point ended inside the {item.get('name')} call "
                    "below, whose result the host consumed and released as the "
                    "environment action; it is recorded here as text because no "
                    "tool reply exists for it]\n"
                    f"{item.get('arguments')}"
                ),
            }
        )
    return kept


class ResponsesClient(_NooaResponsesClient):
    """NOOA Responses client with an explicit Chat-to-Responses content crossing."""

    def _transform_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        transformed, instructions = super()._transform_messages(
            _expand_response_batches(messages)
        )
        normalized: list[dict[str, Any]] = []
        for message in transformed:
            item = dict(message)
            role = item.get("role")
            if role in {"user", "assistant"}:
                item["content"] = _responses_content(
                    str(role),
                    item.get("content"),
                )
            normalized.append(item)
        return _without_unanswered_tool_calls(normalized), instructions


def build_llm(
    profile_record: Mapping[str, Any],
    *,
    api_key: str | None,
) -> ResponsesClient:
    """Construct the pinned NOOA Responses client with no implicit fallback.

    NOOA has separate ordinary and rate-limit retry budgets, while LiteLLM also
    accepts its own retry setting. All three are fixed to zero here. Any bounded
    recovery belongs to the audited transport wrapper above this client.
    """

    profile = ProviderRequestProfile.from_record(profile_record)
    if not api_key:
        raise ValueError("api_key must be supplied explicitly")

    config: dict[str, Any] = {
        "api_base": profile.api_base,
        "api_key": api_key,
        "drop_params": profile.drop_params,
        "max_output_tokens": profile.max_output_tokens,
        "num_retries": profile.litellm_num_retries,
        "stream": profile.stream,
        "timeout": profile.request_timeout_seconds,
    }
    if profile.reasoning_effort is not None:
        config["reasoning_effort"] = profile.reasoning_effort
    if profile.temperature is not None:
        config["temperature"] = profile.temperature
    if profile.seed is not None:
        config["seed"] = profile.seed

    return ResponsesClient(
        model=profile.model,
        retry_config=RetryConfig(
            max_retries=profile.nooa_max_retries,
            rate_limit_extra_retries=profile.nooa_rate_limit_extra_retries,
            retry_on_empty_content=False,
        ),
        cache_control_injection_points=[],
        **config,
    )


def _commit_secret(value: object, *, kind: str = "credential") -> dict[str, Any]:
    if isinstance(value, bytes):
        raw = value
    else:
        raw = str(value).encode("utf-8")
    return {
        "redacted": True,
        "kind": kind,
        "bytes": len(raw),
        "sha256": sha256(raw).hexdigest(),
    }


def _redact_data_image(value: str) -> dict[str, Any] | None:
    if not value.startswith("data:image/") or "," not in value:
        return None
    metadata, payload = value.split(",", 1)
    media_type = metadata[5:].split(";", 1)[0]
    try:
        raw = (
            base64.b64decode(payload, validate=True)
            if ";base64" in metadata
            else payload.encode()
        )
    except (ValueError, binascii.Error):
        raw = payload.encode("utf-8")
    return {
        "redacted": True,
        "kind": "data-image",
        "media_type": media_type,
        "bytes": len(raw),
        "sha256": sha256(raw).hexdigest(),
    }


_AUDIT_CREDENTIAL_QUERY_KEYS = {
    "access_key",
    "credential",
    "key",
    "secret",
    "sig",
    "signature",
    "token",
}


def _credential_marker(value: str) -> str:
    digest = sha256(value.encode("utf-8")).hexdigest()
    return f"[REDACTED sha256={digest} chars={len(value)}]"


def _redact_url(value: str) -> str:
    try:
        split = urlsplit(value)
    except ValueError:
        return value
    netloc = split.netloc
    if "@" in netloc:
        userinfo, host = netloc.rsplit("@", 1)
        netloc = f"{_credential_marker(userinfo)}@{host}"
    query = []
    for key, item in parse_qsl(split.query, keep_blank_values=True):
        sensitive = _is_sensitive_key(key) or key.casefold() in _AUDIT_CREDENTIAL_QUERY_KEYS
        query.append((key, _credential_marker(item) if sensitive else item))
    return urlunsplit((split.scheme, netloc, split.path, urlencode(query), split.fragment))


def redact_for_audit(
    value: Any,
    *,
    key: object | None = None,
    known_credentials: tuple[str, ...] = (),
) -> Any:
    """Return a JSON-safe copy with credentials and image bodies committed, not stored."""

    if key is not None and _is_sensitive_key(key):
        return _commit_secret(value)
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json")
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_for_audit(
                item,
                key=item_key,
                known_credentials=known_credentials,
            )
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            redact_for_audit(item, known_credentials=known_credentials)
            for item in value
        ]
    if isinstance(value, str):
        image = _redact_data_image(value)
        if image is not None:
            return image
        if any(secret in value for secret in known_credentials):
            return _commit_secret(value, kind="known-credential-bearing-string")
        _scrubbed, redactions = scrub_string(value)
        if redactions:
            return _commit_secret(value, kind="credential-bearing-string")
        if value.startswith(("http://", "https://")):
            return _redact_url(value)
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return _commit_secret(value, kind="opaque-bytes")
    return {"type": type(value).__name__, "repr_sha256": sha256(repr(value).encode()).hexdigest()}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _body_record(
    content: bytes,
    *,
    known_credentials: tuple[str, ...],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "body_sha256": sha256(content).hexdigest(),
        "body_bytes": len(content),
    }
    try:
        decoded = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        record["body"] = _commit_secret(content, kind="non-json-body")
    else:
        record["body"] = redact_for_audit(
            decoded,
            known_credentials=known_credentials,
        )
    return record


class ActualCrossingRecorder:
    """Capture the HTTP request/response that NOOA's owned client actually crosses."""

    def __init__(self, *, known_credentials: Iterable[str] = ()) -> None:
        self.records: list[dict[str, Any]] = []
        self.known_credentials = tuple(
            credential for credential in known_credentials if credential
        )

    def async_event_hooks(self) -> dict[str, list[Any]]:
        return {"request": [self._request], "response": [self._response]}

    async def _request(self, request: Any) -> None:
        crossing_id = len(self.records)
        request.extensions["0002_nooa_crossing_id"] = crossing_id
        content = bytes(request.content)
        self.records.append(
            {
                "sequence": crossing_id,
                "request": {
                    "method": request.method,
                    "url": redact_for_audit(
                        _redact_url(str(request.url)),
                        known_credentials=self.known_credentials,
                    ),
                    "headers": redact_for_audit(
                        dict(request.headers),
                        known_credentials=self.known_credentials,
                    ),
                    **_body_record(
                        content,
                        known_credentials=self.known_credentials,
                    ),
                },
                "response": None,
            }
        )

    async def _response(self, response: Any) -> None:
        await response.aread()
        crossing_id = response.request.extensions.get("0002_nooa_crossing_id")
        if type(crossing_id) is not int or not (0 <= crossing_id < len(self.records)):
            raise RuntimeError("provider response has no matching audited request")
        content = bytes(response.content)
        self.records[crossing_id]["response"] = {
            "status_code": response.status_code,
            "headers": redact_for_audit(
                dict(response.headers),
                known_credentials=self.known_credentials,
            ),
            **_body_record(
                content,
                known_credentials=self.known_credentials,
            ),
        }


def _tool_records(tools: list[Any] | None) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.get_parameter_schema(strict=True),
        }
        for tool in tools or ()
    ]


class TransportDisposition(StrEnum):
    """Whether a failed provider attempt may be repeated without guessing."""

    RETRYABLE = "retryable"
    FATAL = "fatal"
    UNCERTAIN = "uncertain"


class InjectedTransportFailure(Exception):
    """Deterministic fault used to exercise the public provider boundary."""

    def __init__(self, disposition: TransportDisposition) -> None:
        self.disposition = disposition
        super().__init__(f"injected {disposition.value} provider transport failure")


class ProviderTransportAbort(BaseException):
    """Control signal that bypasses NOOA's independent generation-repair loop.

    NOOA's CodeAct strategy catches ordinary ``Exception`` values and makes a new
    model call. A fatal or uncertain transport result must instead leave the
    action point without another provider crossing. The launcher catches this
    signal and returns structured failure evidence to the host.
    """

    def __init__(self, disposition: TransportDisposition, attempts: int) -> None:
        self.disposition = disposition
        self.attempts = attempts
        super().__init__(
            f"provider transport aborted: {disposition.value} after {attempts} attempt(s)"
        )


def classify_transport_failure(error: Exception) -> TransportDisposition:
    """Classify external errors conservatively at the provider-call boundary."""

    if isinstance(error, InjectedTransportFailure):
        return error.disposition
    status = getattr(error, "status_code", None)
    if type(status) is int:
        if status in {429, 500, 502, 503, 504}:
            return TransportDisposition.UNCERTAIN
        return TransportDisposition.FATAL
    if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
        return TransportDisposition.RETRYABLE
    if isinstance(
        error,
        (
            httpx.ReadError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.WriteError,
            httpx.WriteTimeout,
        ),
    ):
        return TransportDisposition.UNCERTAIN
    if isinstance(error, (TypeError, ValueError)):
        return TransportDisposition.FATAL
    return TransportDisposition.UNCERTAIN


def _failure_record(error: Exception, disposition: TransportDisposition) -> dict[str, Any]:
    encoded = str(error).encode("utf-8", errors="replace")
    status = getattr(error, "status_code", None)
    return {
        "disposition": disposition.value,
        "error_type": type(error).__name__,
        "message_chars": len(str(error)),
        "message_sha256": sha256(encoded).hexdigest(),
        "status_code": status if type(status) is int else None,
    }


class AuditedProviderClient:
    """Snapshot canonical calls and pair them with NOOA's actual HTTP crossings."""

    def __init__(
        self,
        provider: Any,
        profile_record: Mapping[str, Any],
        *,
        transport_max_retries: int,
        require_actual_crossings: bool = False,
        known_credentials: Iterable[str] = (),
    ) -> None:
        if transport_max_retries < 0:
            raise ValueError("transport_max_retries must not be negative")
        self.provider = provider
        self.profile = ProviderRequestProfile.from_record(profile_record)
        self.transport_max_retries = transport_max_retries
        self.require_actual_crossings = require_actual_crossings
        self.audit_calls: list[dict[str, Any]] = []
        self.known_credentials = tuple(
            credential for credential in known_credentials if credential
        )
        self.actual = ActualCrossingRecorder(
            known_credentials=self.known_credentials,
        )
        http_owner = getattr(provider, "_http", None)
        httpx_async = getattr(http_owner, "httpx_async", None)
        if httpx_async is not None:
            hooks = self.actual.async_event_hooks()
            for event, callbacks in hooks.items():
                httpx_async.event_hooks.setdefault(event, []).extend(callbacks)
        elif require_actual_crossings:
            raise RuntimeError("NOOA provider exposes no auditable owned async HTTP client")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        output_model: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        canonical = {
            "provider_request_profile": asdict(self.profile),
            "messages": messages,
            "tools": _tool_records(tools),
            "kwargs": kwargs,
        }
        canonical_snapshot = redact_for_audit(
            canonical,
            known_credentials=self.known_credentials,
        )
        actual_offset = len(self.actual.records)
        audit = {
            "schema": "0002-nooa-provider-crossing-v1",
            "messages": canonical_snapshot["messages"],
            "tools": canonical_snapshot["tools"],
            "canonical_request": canonical_snapshot,
            "canonical_request_sha256": sha256(_canonical_bytes(canonical)).hexdigest(),
            "canonicalization": "json-utf8-sort-keys-compact-v1",
            "actual_provider_crossings": [],
            "attempts": [],
            "outcome": None,
        }
        self.audit_calls.append(audit)
        for attempt_index in range(self.transport_max_retries + 1):
            attempt_offset = len(self.actual.records)
            try:
                response = await self.provider.acall(
                    messages,
                    tools=tools,
                    output_model=output_model,
                    **kwargs,
                )
            except Exception as error:
                disposition = classify_transport_failure(error)
                crossings = self.actual.records[attempt_offset:]
                failure = _failure_record(error, disposition)
                audit["attempts"].append(
                    {
                        "attempt": attempt_index + 1,
                        "outcome": f"{disposition.value}_failure",
                        "failure": failure,
                        "actual_provider_crossings": crossings,
                    }
                )
                audit["actual_provider_crossings"] = self.actual.records[actual_offset:]
                if (
                    disposition is TransportDisposition.RETRYABLE
                    and attempt_index < self.transport_max_retries
                ):
                    continue
                audit["outcome"] = {
                    "kind": "transport_failure",
                    **failure,
                    "attempts": attempt_index + 1,
                }
                raise ProviderTransportAbort(
                    disposition,
                    attempt_index + 1,
                ) from error
            crossings = self.actual.records[attempt_offset:]
            if self.require_actual_crossings and not crossings:
                failure = {
                    "disposition": TransportDisposition.FATAL.value,
                    "error_type": "MissingActualProviderCrossing",
                    "message_chars": 0,
                    "message_sha256": sha256(b"").hexdigest(),
                    "status_code": None,
                }
                audit["attempts"].append(
                    {
                        "attempt": attempt_index + 1,
                        "outcome": "fatal_failure",
                        "failure": failure,
                        "actual_provider_crossings": [],
                    }
                )
                audit["outcome"] = {
                    "kind": "transport_failure",
                    **failure,
                    "attempts": attempt_index + 1,
                }
                raise ProviderTransportAbort(
                    TransportDisposition.FATAL,
                    attempt_index + 1,
                )
            audit["attempts"].append(
                {
                    "attempt": attempt_index + 1,
                    "outcome": "success",
                    "actual_provider_crossings": crossings,
                }
            )
            audit["actual_provider_crossings"] = self.actual.records[actual_offset:]
            audit["outcome"] = {
                "kind": "success",
                "response": redact_for_audit(
                    {
                        "assistant_message": getattr(response, "assistant_message", None),
                        "content": getattr(response, "content", None),
                        "raw_response": getattr(response, "raw_response", None),
                        "tool_calls": getattr(response, "tool_calls", None),
                        "usage": getattr(response, "usage", None),
                    },
                    known_credentials=self.known_credentials,
                ),
            }
            return response
        raise AssertionError("transport attempt loop exited without an outcome")

    async def aclose(self) -> None:
        close = getattr(self.provider, "aclose", None)
        if close is not None:
            await close()
