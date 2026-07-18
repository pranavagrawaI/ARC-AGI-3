"""Environment resolution for local and hosted OpenAI-compatible models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse


DEFAULT_LOCAL_ENDPOINT = "http://127.0.0.1:8000/v1"
OPENROUTER_DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
OPENROUTER_QWEN36_MODEL = "qwen/qwen3.6-27b"
QWEN36_DEFAULT_MAX_COMPLETION_TOKENS = 4096
MODEL_PROFILES = ("generic", "qwen3.6", "openrouter-qwen3.6")


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

    openrouter_model = environ.get("OPENROUTER_MODEL")
    qwen_model = environ.get("QWEN_MODEL")
    resolved_model = model or openrouter_model or qwen_model or environ.get("OPENAI_MODEL")
    if not resolved_model:
        raise ValueError(
            "--model, OPENROUTER_MODEL, QWEN_MODEL, or OPENAI_MODEL is required"
        )
    inferred_openrouter = bool(
        openrouter_model and resolved_model == openrouter_model
    )
    resolved_profile = profile or (
        "openrouter-qwen3.6"
        if inferred_openrouter
        else "qwen3.6"
        if qwen_model and resolved_model == qwen_model
        else "generic"
    )
    if resolved_profile not in MODEL_PROFILES:
        raise ValueError(f"unsupported model profile: {resolved_profile}")
    if (
        resolved_profile == "openrouter-qwen3.6"
        and resolved_model != OPENROUTER_QWEN36_MODEL
    ):
        raise ValueError(
            "the openrouter-qwen3.6 profile requires model "
            f"{OPENROUTER_QWEN36_MODEL}"
        )
    if endpoint:
        resolved_endpoint = endpoint
    elif resolved_profile == "openrouter-qwen3.6":
        resolved_endpoint = OPENROUTER_DEFAULT_ENDPOINT
    else:
        resolved_endpoint = environ.get("OPENAI_BASE_URL") or DEFAULT_LOCAL_ENDPOINT
    endpoint_host_is_openrouter = _is_openrouter_host(resolved_endpoint)
    if (
        endpoint_host_is_openrouter
        and resolved_endpoint.rstrip("/") != OPENROUTER_DEFAULT_ENDPOINT
    ):
        raise ValueError(
            "OpenRouter credentials require the official HTTPS API base "
            f"{OPENROUTER_DEFAULT_ENDPOINT}"
        )
    profile_is_openrouter = resolved_profile == "openrouter-qwen3.6"
    if endpoint_host_is_openrouter != profile_is_openrouter:
        raise ValueError(
            "the official OpenRouter endpoint and openrouter-qwen3.6 profile "
            "must be used together"
        )

    if _is_local_endpoint(resolved_endpoint):
        key_source = "LOCAL_MODEL_API_KEY" if environ.get("LOCAL_MODEL_API_KEY") else None
        api_key = environ[key_source] if key_source is not None else "EMPTY"
    else:
        if resolved_profile == "openrouter-qwen3.6":
            key_names = ("OPENROUTER_API_KEY",)
        elif resolved_profile == "qwen3.6":
            key_names = ("QWEN_API_KEY", "OPENAI_API_KEY")
        else:
            key_names = ("OPENAI_API_KEY", "QWEN_API_KEY")
        key_source = next((name for name in key_names if environ.get(name)), None)
        if key_source is None:
            if resolved_profile == "openrouter-qwen3.6":
                raise ValueError(
                    "the OpenRouter model profile requires OPENROUTER_API_KEY"
                )
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
    if profile == "openrouter-qwen3.6":
        if max_tokens is not None:
            raise ValueError(
                "the openrouter-qwen3.6 profile uses max_completion_tokens, "
                "not max_tokens"
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
                "reasoning": {"enabled": True, "exclude": False},
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


def _is_openrouter_host(endpoint: str) -> bool:
    return urlparse(endpoint).hostname == "openrouter.ai"
