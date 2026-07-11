"""Small OpenAI-compatible chat client used by the local LLM baseline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ChatResponse:
    """One assistant response, preserved closely enough for tool continuation."""

    message: JsonObject
    usage: Mapping[str, Any]
    response_id: str | None = None


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
        max_tokens: int = 2048,
        seed: int | None = 0,
        timeout_seconds: float = 300.0,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
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
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        payload.update(self.extra_body)
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
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
        usage = body.get("usage")
        return ChatResponse(
            message=message,
            usage=dict(usage) if isinstance(usage, Mapping) else {},
            response_id=str(body["id"]) if body.get("id") is not None else None,
        )
