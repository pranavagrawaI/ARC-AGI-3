"""Small OpenAI-compatible chat client used by the local LLM baseline."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
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


class ChatClient(Protocol):
    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatResponse: ...


class ModelRequestError(RuntimeError):
    pass


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
    ) -> None:
        if (max_tokens is None) == (max_completion_tokens is None):
            raise ValueError(
                "configure exactly one of max_tokens or max_completion_tokens"
            )
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
        payload_bytes = serialize_request_payload(payload)
        payload_record = json.loads(payload_bytes.decode("utf-8"))
        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload_bytes,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise ModelRequestError(
                f"model endpoint returned HTTP {error.code}: {detail[:2000]}"
            ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ModelRequestError(f"model request failed: {error}") from error

        try:
            message = dict(body["choices"][0]["message"])
        except (KeyError, IndexError, TypeError) as error:
            raise ModelRequestError(
                f"model endpoint returned an invalid chat response: {body!r}"
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
            request_payload_sha256=sha256(payload_bytes).hexdigest(),
            request_payload_encoding=REQUEST_PAYLOAD_ENCODING,
        )
