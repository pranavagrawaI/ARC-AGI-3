"""Small OpenAI-compatible chat client used by the local LLM baseline."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from hashlib import sha256
from http.client import IncompleteRead
import json
import time
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


JsonObject = dict[str, Any]
REQUEST_PAYLOAD_ENCODING = "json-utf8-sort-keys-compact-v1"


def serialize_request_payload(payload: Mapping[str, Any]) -> bytes:
    """Return the canonical bytes used both for transport and replay evidence."""

    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


@dataclass(frozen=True)
class ChatResponse:
    """One assistant response plus credential-free transport evidence."""

    message: JsonObject
    usage: Mapping[str, Any]
    response_id: str | None = None
    request_payload: Mapping[str, Any] | None = None
    request_payload_sha256: str | None = None
    request_payload_encoding: str | None = None
    response_metadata: Mapping[str, Any] | None = None
    response_body: Mapping[str, Any] | None = None
    response_body_sha256: str | None = None
    response_body_raw: str | None = None
    response_body_base64: str | None = None


class ChatClient(Protocol):
    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatResponse: ...


class ModelRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        request_payload: Mapping[str, Any] | None = None,
        request_payload_sha256: str | None = None,
        response_body_raw: str | None = None,
        response_body_base64: str | None = None,
        response_body_sha256: str | None = None,
        http_status: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.request_payload = dict(request_payload) if request_payload is not None else None
        self.request_payload_sha256 = request_payload_sha256
        self.request_payload_encoding = (
            REQUEST_PAYLOAD_ENCODING if request_payload is not None else None
        )
        self.response_body_raw = response_body_raw
        self.response_body_base64 = response_body_base64
        self.response_body_sha256 = response_body_sha256
        self.http_status = http_status
        self.retryable = retryable


def _looks_rate_limited(status: int | None, detail: str) -> bool:
    if status in {408, 429, 502, 503, 504}:
        return True
    lowered = detail.lower()
    return any(
        token in lowered
        for token in (
            "rate-limited",
            "rate limit",
            "engine_overloaded",
            "temporarily rate-limited",
            "overloaded",
            "capacity",
            "operation was aborted",
            "timed out",
            "timeout",
            '"code":504',
            "'code': 504",
            '"code":429',
            "'code': 429",
        )
    )


def _provider_error_status(body: Mapping[str, Any]) -> int | None:
    """Extract a provider error code from an otherwise HTTP-200 OpenRouter body."""

    error = body.get("error")
    if not isinstance(error, Mapping):
        return None
    code = error.get("code")
    if isinstance(code, bool):
        return None
    if isinstance(code, int):
        return code
    if isinstance(code, str) and code.isdigit():
        return int(code)
    return None


class OpenAICompatibleChatClient:
    """Call a stateless OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int | None = 2048,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = 0,
        timeout_seconds: float = 300.0,
        extra_body: Mapping[str, Any] | None = None,
        rate_limit_retries: int = 0,
        rate_limit_backoff_seconds: float = 20.0,
    ) -> None:
        if (max_tokens is None) == (max_completion_tokens is None):
            raise ValueError(
                "configure exactly one of max_tokens or max_completion_tokens"
            )
        if rate_limit_retries < 0:
            raise ValueError("rate_limit_retries must be non-negative")
        if rate_limit_backoff_seconds < 0:
            raise ValueError("rate_limit_backoff_seconds must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.timeout_seconds = timeout_seconds
        self.extra_body = dict(extra_body or {})
        self.rate_limit_retries = rate_limit_retries
        self.rate_limit_backoff_seconds = rate_limit_backoff_seconds

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        payload: JsonObject = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "auto",
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.max_completion_tokens
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            payload["seed"] = self.seed
        payload.update(self.extra_body)
        return self.complete_payload(payload)

    def complete_payload(self, payload: Mapping[str, Any]) -> ChatResponse:
        """Send an already-composed credential-free payload without rewriting it."""

        if not isinstance(payload.get("messages"), list) or not isinstance(
            payload.get("model"), str
        ):
            raise ValueError("direct chat payload requires model and messages")
        attempts = 0
        while True:
            try:
                return self._complete_payload_once(payload)
            except ModelRequestError as error:
                if not error.retryable or attempts >= self.rate_limit_retries:
                    raise
                attempts += 1
                delay = self.rate_limit_backoff_seconds * (2 ** (attempts - 1))
                time.sleep(delay)

    def _complete_payload_once(self, payload: Mapping[str, Any]) -> ChatResponse:
        payload_bytes = serialize_request_payload(payload)
        payload_record = json.loads(payload_bytes.decode("utf-8"))
        payload_hash = sha256(payload_bytes).hexdigest()
        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload_bytes,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response_bytes = b""
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_bytes = response.read()
                body = json.loads(response_bytes.decode("utf-8"))
        except HTTPError as error:
            detail_bytes = error.read()
            detail = detail_bytes.decode("utf-8", errors="replace")
            raise ModelRequestError(
                f"model endpoint returned HTTP {error.code}: {detail[:2000]}",
                request_payload=payload_record,
                request_payload_sha256=payload_hash,
                response_body_raw=detail,
                response_body_base64=b64encode(detail_bytes).decode("ascii"),
                response_body_sha256=sha256(detail_bytes).hexdigest(),
                http_status=int(error.code),
                retryable=_looks_rate_limited(int(error.code), detail),
            ) from error
        except IncompleteRead as error:
            partial = error.partial if isinstance(error.partial, (bytes, bytearray)) else b""
            detail = f"IncompleteRead({len(partial)} bytes read)"
            raise ModelRequestError(
                f"model request failed: {detail}",
                request_payload=payload_record,
                request_payload_sha256=payload_hash,
                response_body_raw=partial.decode("utf-8", errors="replace"),
                response_body_base64=b64encode(bytes(partial)).decode("ascii"),
                response_body_sha256=sha256(bytes(partial)).hexdigest() if partial else None,
                retryable=True,
            ) from error
        except (URLError, TimeoutError) as error:
            raise ModelRequestError(
                f"model request failed: {error}",
                request_payload=payload_record,
                request_payload_sha256=payload_hash,
                retryable=True,
            ) from error
        except json.JSONDecodeError as error:
            raise ModelRequestError(
                f"model request failed: {error}",
                request_payload=payload_record,
                request_payload_sha256=payload_hash,
                response_body_raw=response_bytes.decode("utf-8", errors="replace"),
                response_body_base64=b64encode(response_bytes).decode("ascii"),
                response_body_sha256=sha256(response_bytes).hexdigest(),
            ) from error

        if not isinstance(body, dict):
            raise ModelRequestError(
                f"model endpoint returned a non-object chat response: {body!r}",
                request_payload=payload_record,
                request_payload_sha256=payload_hash,
                response_body_raw=response_bytes.decode("utf-8", errors="replace"),
                response_body_base64=b64encode(response_bytes).decode("ascii"),
                response_body_sha256=sha256(response_bytes).hexdigest(),
            )

        # OpenRouter sometimes returns HTTP 200 with {"error":{"code":504,...}}
        # and no choices. Treat transient provider aborts as retryable.
        provider_status = _provider_error_status(body)
        detail_text = json.dumps(body, ensure_ascii=False, sort_keys=True)
        if provider_status is not None or "choices" not in body:
            raise ModelRequestError(
                f"model endpoint returned an invalid chat response: {body!r}",
                request_payload=payload_record,
                request_payload_sha256=payload_hash,
                response_body_raw=response_bytes.decode("utf-8", errors="replace"),
                response_body_base64=b64encode(response_bytes).decode("ascii"),
                response_body_sha256=sha256(response_bytes).hexdigest(),
                http_status=provider_status,
                retryable=_looks_rate_limited(provider_status, detail_text),
            )

        try:
            message = dict(body["choices"][0]["message"])
        except (KeyError, IndexError, TypeError) as error:
            raise ModelRequestError(
                f"model endpoint returned an invalid chat response: {body!r}",
                request_payload=payload_record,
                request_payload_sha256=payload_hash,
                response_body_raw=response_bytes.decode("utf-8", errors="replace"),
                response_body_base64=b64encode(response_bytes).decode("ascii"),
                response_body_sha256=sha256(response_bytes).hexdigest(),
                http_status=provider_status,
                retryable=_looks_rate_limited(provider_status, detail_text),
            ) from error
        reasoning = message.get("reasoning")
        if "reasoning_content" not in message and isinstance(reasoning, str):
            message["reasoning_content"] = reasoning
        usage = body.get("usage")
        return ChatResponse(
            message=message,
            usage=dict(usage) if isinstance(usage, Mapping) else {},
            response_id=str(body["id"]) if body.get("id") is not None else None,
            request_payload=payload_record,
            request_payload_sha256=payload_hash,
            request_payload_encoding=REQUEST_PAYLOAD_ENCODING,
            response_metadata={
                str(key): value
                for key, value in body.items()
                if key not in {"choices", "usage"}
            },
            response_body=dict(body),
            response_body_sha256=sha256(response_bytes).hexdigest(),
            response_body_raw=response_bytes.decode("utf-8"),
            response_body_base64=b64encode(response_bytes).decode("ascii"),
        )
