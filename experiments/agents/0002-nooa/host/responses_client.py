"""Synchronous Responses transport for the 0002-R side of the G9 pair."""

from __future__ import annotations

from base64 import b64decode, b64encode
import binascii
from hashlib import sha256
from http.client import IncompleteRead
import json
import socket
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from llm_repl_agent.client import (
    ChatResponse,
    ModelRequestError,
    serialize_request_payload,
)

from .qualification import ProviderRequestProfile


_SENSITIVE_KEY_PARTS = ("api_key", "authorization", "credential", "secret", "token")
_SENSITIVE_QUERY_KEYS = {
    "access_key",
    "credential",
    "key",
    "secret",
    "sig",
    "signature",
    "token",
}


def _credential_marker(value: str) -> str:
    return f"[REDACTED sha256={sha256(value.encode()).hexdigest()} chars={len(value)}]"


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
        sensitive = (
            key.casefold() in _SENSITIVE_QUERY_KEYS
            or any(part in key.casefold() for part in _SENSITIVE_KEY_PARTS)
        )
        query.append((key, _credential_marker(item) if sensitive else item))
    return urlunsplit((split.scheme, netloc, split.path, urlencode(query), split.fragment))


def _redact_data_images(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_data_images(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_data_images(item) for key, item in value.items()}
    if not isinstance(value, str) or not value.startswith("data:image/") or "," not in value:
        return value
    metadata, payload = value.split(",", 1)
    try:
        raw = (
            b64decode(payload, validate=True)
            if ";base64" in metadata
            else payload.encode("utf-8")
        )
    except (ValueError, binascii.Error):
        raw = payload.encode("utf-8")
    return {
        "redacted": True,
        "kind": "data-image",
        "media_type": metadata[5:].split(";", 1)[0],
        "bytes": len(raw),
        "sha256": sha256(raw).hexdigest(),
    }


def _redact_for_audit(
    value: Any,
    *,
    key: object | None = None,
    known_credentials: tuple[str, ...] = (),
) -> Any:
    if key is not None and any(
        part in str(key).casefold() for part in _SENSITIVE_KEY_PARTS
    ):
        encoded = str(value).encode("utf-8")
        return {
            "redacted": True,
            "kind": "credential",
            "bytes": len(encoded),
            "sha256": sha256(encoded).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact_for_audit(
                item,
                key=item_key,
                known_credentials=known_credentials,
            )
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _redact_for_audit(item, known_credentials=known_credentials)
            for item in value
        ]
    if isinstance(value, str):
        image = _redact_data_images(value)
        if image != value:
            return image
        if any(credential and credential in value for credential in known_credentials):
            encoded = value.encode("utf-8")
            return {
                "redacted": True,
                "kind": "credential-bearing-string",
                "bytes": len(encoded),
                "sha256": sha256(encoded).hexdigest(),
            }
        if value.startswith(("http://", "https://")):
            return _redact_url(value)
    return value


def _safe_url(value: str) -> dict[str, str]:
    split = urlsplit(value)
    return {
        "scheme": split.scheme,
        "host": split.hostname or "",
        "path": split.path,
    }


def _canonical_safe(
    value: Any,
    *,
    known_credentials: tuple[str, ...] = (),
) -> tuple[dict[str, Any], bytes]:
    safe = _redact_for_audit(value, known_credentials=known_credentials)
    if not isinstance(safe, dict):
        safe = {"value": safe}
    encoded = serialize_request_payload(safe)
    return safe, encoded


def _content_for_responses(role: str, content: Any) -> Any:
    if not isinstance(content, list):
        return "" if content is None else content
    converted: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") == "text":
            converted.append(
                {
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": str(part.get("text", "")),
                }
            )
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, Mapping) else None
            if isinstance(url, str):
                converted.append({"type": "input_image", "image_url": url})
    return converted


def _responses_input(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            if content:
                instructions.append(str(content))
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id", "")),
                    "output": str(message.get("content", "")),
                }
            )
        elif role == "assistant" and message.get("tool_calls"):
            content = message.get("content")
            if content:
                items.append({"role": "assistant", "content": content})
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {})
                items.append(
                    {
                        "type": "function_call",
                        "call_id": str(tool_call.get("id", "")),
                        "name": str(function.get("name", "")),
                        "arguments": str(function.get("arguments", "")),
                    }
                )
        elif role in {"user", "assistant"}:
            items.append(
                {
                    "role": role,
                    "content": _content_for_responses(
                        str(role),
                        message.get("content"),
                    ),
                }
            )
    return items, "\n\n".join(instructions) if instructions else None


def _responses_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, Mapping) else {}
        required = parameters.get("required", []) if isinstance(parameters, Mapping) else []
        converted.append(
            {
                "type": "function",
                "name": str(function.get("name", "")),
                "description": str(function.get("description", "")),
                "parameters": parameters,
                "strict": bool(properties) and set(required) == set(properties),
            }
        )
    return converted


class OpenAICompatibleResponsesClient:
    """Project one frozen provider profile onto a raw ``/responses`` request."""

    def __init__(self, profile: ProviderRequestProfile, api_key: str) -> None:
        if not api_key:
            raise ValueError("api_key must be explicit")
        self.profile = profile
        self.api_key = api_key
        self.actual_crossings: list[dict[str, Any]] = []

    def _payload(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        input_items, instructions = _responses_input(messages)
        payload: dict[str, Any] = {
            "model": self.profile.wire_model,
            "input": input_items,
            "max_output_tokens": self.profile.max_output_tokens,
            "stream": self.profile.stream,
            "truncation": self.profile.truncation,
        }
        if instructions is not None:
            payload["instructions"] = instructions
        converted_tools = _responses_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools
            payload["tool_choice"] = self.profile.tool_choice
            payload["parallel_tool_calls"] = self.profile.parallel_tool_calls
        if self.profile.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.profile.reasoning_effort}
        if self.profile.temperature is not None:
            payload["temperature"] = self.profile.temperature
        if self.profile.seed is not None:
            payload["seed"] = self.profile.seed
        return payload

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        payload = self._payload(messages, tools)
        payload_bytes = serialize_request_payload(payload)
        payload_sha256 = sha256(payload_bytes).hexdigest()
        safe_payload, safe_payload_bytes = _canonical_safe(
            payload,
            known_credentials=(self.api_key,),
        )
        endpoint = f"{self.profile.api_base.rstrip('/')}/responses"
        crossing: dict[str, Any] = {
            "sequence": len(self.actual_crossings),
            "request": {
                "method": "POST",
                "url": _safe_url(endpoint),
                "body_sha256": payload_sha256,
                "body_bytes": len(payload_bytes),
                "body": safe_payload,
            },
            "response": None,
        }
        self.actual_crossings.append(crossing)
        request = Request(
            endpoint,
            data=payload_bytes,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response_bytes = b""
        status: int | None = None
        try:
            with urlopen(request, timeout=self.profile.request_timeout_seconds) as response:
                status = int(response.status)
                response_bytes = response.read()
        except HTTPError as error:
            status = int(error.code)
            try:
                response_bytes = error.read()
            finally:
                error.close()
            self._record_response(crossing, status, response_bytes)
            raise self._request_error(
                f"Responses endpoint returned HTTP {status}",
                safe_payload,
                safe_payload_bytes,
                response_bytes,
                status=status,
                retryable=False,
            ) from error
        except IncompleteRead as error:
            response_bytes = bytes(error.partial or b"")
            self._record_response(crossing, status, response_bytes)
            raise self._request_error(
                "Responses endpoint returned an incomplete body",
                safe_payload,
                safe_payload_bytes,
                response_bytes,
                retryable=False,
            ) from error
        except URLError as error:
            pre_connect = isinstance(
                error.reason,
                (ConnectionRefusedError, socket.gaierror),
            )
            raise self._request_error(
                f"Responses request failed: {error}",
                safe_payload,
                safe_payload_bytes,
                b"",
                retryable=pre_connect,
            ) from error
        except TimeoutError as error:
            raise self._request_error(
                f"Responses request outcome is uncertain: {error}",
                safe_payload,
                safe_payload_bytes,
                b"",
                retryable=False,
            ) from error
        self._record_response(crossing, status, response_bytes)
        try:
            body = json.loads(response_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise self._request_error(
                f"Responses endpoint returned invalid JSON: {error}",
                safe_payload,
                safe_payload_bytes,
                response_bytes,
            ) from error
        if not isinstance(body, dict) or not isinstance(body.get("output"), list):
            raise self._request_error(
                "Responses endpoint returned no output list",
                safe_payload,
                safe_payload_bytes,
                response_bytes,
            )
        message = self._assistant_message(body["output"])
        safe_body, safe_body_bytes = _canonical_safe(
            body,
            known_credentials=(self.api_key,),
        )
        usage = body.get("usage")
        return ChatResponse(
            message=message,
            usage=dict(usage) if isinstance(usage, Mapping) else {},
            response_id=str(body["id"]) if body.get("id") is not None else None,
            request_payload=safe_payload,
            request_payload_sha256=sha256(safe_payload_bytes).hexdigest(),
            request_payload_encoding="json-utf8-sort-keys-compact-v1",
            response_metadata={
                key: value
                for key, value in safe_body.items()
                if key not in {"output", "usage"}
            },
            response_body=safe_body,
            response_body_sha256=sha256(safe_body_bytes).hexdigest(),
            response_body_raw=safe_body_bytes.decode("utf-8"),
            response_body_base64=b64encode(safe_body_bytes).decode("ascii"),
        )

    def _record_response(
        self,
        crossing: dict[str, Any],
        status: int | None,
        response_bytes: bytes,
    ) -> None:
        try:
            decoded: Any = json.loads(response_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = {"opaque_body_sha256": sha256(response_bytes).hexdigest()}
        crossing["response"] = {
            "status_code": status,
            "body_sha256": sha256(response_bytes).hexdigest(),
            "body_bytes": len(response_bytes),
            "body": _redact_for_audit(
                decoded,
                known_credentials=(self.api_key,),
            ),
        }

    @staticmethod
    def _assistant_message(output: list[Any]) -> dict[str, Any]:
        tool_calls = []
        text_parts = []
        for item in output:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "function_call":
                tool_calls.append(
                    {
                        "id": str(item.get("call_id", "")),
                        "type": "function",
                        "function": {
                            "name": str(item.get("name", "")),
                            "arguments": str(item.get("arguments", "")),
                        },
                    }
                )
            elif item.get("type") == "message":
                for content in item.get("content", []):
                    if isinstance(content, Mapping) and content.get("type") == "output_text":
                        text_parts.append(str(content.get("text", "")))
        return {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
            **({"tool_calls": tool_calls} if tool_calls else {}),
        }

    @staticmethod
    def _request_error(
        message: str,
        safe_payload: Mapping[str, Any],
        safe_payload_bytes: bytes,
        response_bytes: bytes,
        *,
        status: int | None = None,
        retryable: bool = False,
    ) -> ModelRequestError:
        return ModelRequestError(
            message,
            request_payload=safe_payload,
            request_payload_sha256=sha256(safe_payload_bytes).hexdigest(),
            response_body_raw=None,
            response_body_base64=None,
            response_body_sha256=None,
            http_status=status,
            retryable=retryable,
        )
