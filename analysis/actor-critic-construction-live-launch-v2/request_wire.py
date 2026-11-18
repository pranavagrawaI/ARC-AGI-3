"""OpenRouter wire projection for the live join. Not a generic control plane."""

from __future__ import annotations

from typing import Any

MODEL = "qwen/qwen3.8-27b"
PROVIDER_TAG = "coreweave/fp8"
MAX_OUTPUT_TOKENS = 16384
CRITIC_MAX_TOKENS = 192
PROVIDER = {
    "only": [PROVIDER_TAG],
    "allow_fallbacks": False,
    "require_parameters": True,
    "max_price": {"prompt": 0.4, "completion": 3, "request": 0},
}
STRIP_KEYS = ("parallel_tool_calls", "chat_template_kwargs")


def tool_names(payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for item in payload.get("tools") or []:
        if isinstance(item, dict):
            name = (item.get("function") or {}).get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def request_purpose(payload: dict[str, Any]) -> str:
    names = tool_names(payload)
    if "submit_candidate" in names or "request_evidence" in names:
        return "constructor"
    if payload.get("tools") is None and payload.get("max_tokens") == CRITIC_MAX_TOKENS:
        return "critic"
    if "execute_python" in names:
        return "actor"
    return "summarizer"


def project_outbound(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip unsupported optionals; pin CoreWeave; keep critic 192."""
    body = dict(payload)
    for key in STRIP_KEYS:
        body.pop(key, None)
    extra = body.get("extra_body")
    if isinstance(extra, dict):
        extra = dict(extra)
        extra.pop("chat_template_kwargs", None)
        reasoning = extra.get("reasoning")
        if isinstance(reasoning, dict):
            extra["reasoning"] = {"enabled": bool(reasoning.get("enabled"))}
        body["extra_body"] = extra
    body.pop("provider", None)
    body["model"] = MODEL
    body["stream"] = False
    body["provider"] = dict(PROVIDER)
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        body["max_tokens"] = MAX_OUTPUT_TOKENS
    elif type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive int")
    elif max_tokens > MAX_OUTPUT_TOKENS:
        raise ValueError("max_tokens exceeds 16384")
    if body.get("reasoning") is None:
        body["reasoning"] = {"enabled": False}
    elif body["reasoning"].get("enabled") is not False:
        raise ValueError("reasoning must stay disabled")
    return body


def conservative_reserve_usd(body: dict[str, Any]) -> float:
    rendered = len(__import__("json").dumps(body, ensure_ascii=False).encode())
    return rendered * 0.0000004 + int(body["max_tokens"]) * 0.000003 + 0.02


def usage_receipt(parsed: dict[str, Any]) -> dict[str, int] | None:
    """Return prompt/completion usage only when a complete receipt is present."""
    if not isinstance(parsed, dict) or parsed.get("error"):
        return None
    if not parsed.get("choices"):
        return None
    usage = parsed.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if type(prompt) is not int or prompt < 0:
        return None
    if type(completion) is not int or completion < 0:
        return None
    return {"prompt_tokens": prompt, "completion_tokens": completion}


def actual_usd(receipt: dict[str, int]) -> float:
    return receipt["prompt_tokens"] * 0.0000004 + receipt["completion_tokens"] * 0.000003

