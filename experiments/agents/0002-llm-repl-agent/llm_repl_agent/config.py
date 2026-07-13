"""Environment resolution for local and hosted OpenAI-compatible models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse


DEFAULT_LOCAL_ENDPOINT = "http://127.0.0.1:8000/v1"
QWEN36_DEFAULT_MAX_COMPLETION_TOKENS = 4096


@dataclass(frozen=True)
class ModelEnvironment:
    model: str
    endpoint: str
    profile: str
    api_key_source: str | None
    api_key: str = field(repr=False)


def resolve_model_environment(
    environ: Mapping[str, str],
    *,
    model: str | None = None,
    endpoint: str | None = None,
    profile: str | None = None,
) -> ModelEnvironment:
    """Resolve model transport without placing credentials in logs or manifests."""

    qwen_model = environ.get("QWEN_MODEL")
    resolved_model = model or qwen_model or environ.get("OPENAI_MODEL")
    if not resolved_model:
        raise ValueError("--model, QWEN_MODEL, or OPENAI_MODEL is required")
    resolved_endpoint = endpoint or environ.get("OPENAI_BASE_URL") or DEFAULT_LOCAL_ENDPOINT
    resolved_profile = profile or (
        "qwen3.6" if qwen_model and resolved_model == qwen_model else "generic"
    )
    if resolved_profile not in {"generic", "qwen3.6"}:
        raise ValueError(f"unsupported model profile: {resolved_profile}")

    if _is_local_endpoint(resolved_endpoint):
        key_source = "LOCAL_MODEL_API_KEY" if environ.get("LOCAL_MODEL_API_KEY") else None
        api_key = environ[key_source] if key_source is not None else "EMPTY"
    else:
        key_names = (
            ("QWEN_API_KEY", "OPENAI_API_KEY")
            if resolved_profile == "qwen3.6"
            else ("OPENAI_API_KEY", "QWEN_API_KEY")
        )
        key_source = next((name for name in key_names if environ.get(name)), None)
        if key_source is None:
            raise ValueError(
                "a remote model endpoint requires an API key in QWEN_API_KEY or OPENAI_API_KEY"
            )
        api_key = environ[key_source]
    return ModelEnvironment(
        model=resolved_model,
        endpoint=resolved_endpoint,
        profile=resolved_profile,
        api_key_source=key_source,
        api_key=api_key,
    )


def model_request_options(
    profile: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Return an explicit, serializable request contract for a model profile."""

    if profile == "qwen3.6":
        if max_tokens is not None:
            raise ValueError(
                "the qwen3.6 profile uses max_completion_tokens, not max_tokens"
            )
        return {
            "temperature": 1.0 if temperature is None else temperature,
            "max_tokens": None,
            "max_completion_tokens": (
                QWEN36_DEFAULT_MAX_COMPLETION_TOKENS
                if max_completion_tokens is None
                else max_completion_tokens
            ),
            "top_p": 0.95,
            "presence_penalty": 0.0,
            "extra_body": {
                "top_k": 20,
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        }
    if profile == "generic":
        if max_tokens is not None and max_completion_tokens is not None:
            raise ValueError("configure only one token limit")
        if max_completion_tokens is not None:
            resolved_max_tokens = None
            resolved_max_completion_tokens = max_completion_tokens
        else:
            resolved_max_tokens = 2048 if max_tokens is None else max_tokens
            resolved_max_completion_tokens = None
        return {
            "temperature": 0.0 if temperature is None else temperature,
            "max_tokens": resolved_max_tokens,
            "max_completion_tokens": resolved_max_completion_tokens,
            "top_p": None,
            "presence_penalty": None,
            "extra_body": {},
        }
    raise ValueError(f"unsupported model profile: {profile}")


def _is_local_endpoint(endpoint: str) -> bool:
    return urlparse(endpoint).hostname in {"127.0.0.1", "localhost", "::1"}
