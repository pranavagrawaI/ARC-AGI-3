# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM + embedding builders for the ARC-AGI-3 solver example.

Configured purely from the environment (see repo-root ``.env``)::

    ARC_LLM_MODEL=openai/openai/gpt-5.4
    ARC_LLM_BASE_URL=https://inference-api.nvidia.com/v1
    ARC_LLM_API_KEY=...
    MEM_EMBED_MODEL=openai/azure/openai/text-embedding-3-large
    MEM_EMBED_BASE_URL=https://inference-api.nvidia.com/v1
    MEM_EMBED_API_KEY=...
    MEM_EMBED_DIMS=1024

Follows ``examples/memory_bench/llm.py``; falls back to ``FakeLLMClient`` /
hashing embeddings when credentials are absent so smoke tests run offline.
"""

from __future__ import annotations

import os
from pathlib import Path

from nooa.unifiedllm import CompletionClient, FakeLLMClient, RetryConfig
from nooa.unifiedllm.unifiedllm import ResponsesClient


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency): real environment wins via setdefault."""
    for d in [Path.cwd(), *Path(__file__).resolve().parents]:
        f = d / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
            return


if os.environ.get('ARC_DISABLE_DOTENV') != '1':
    _load_dotenv()


def has_llm_creds() -> bool:
    return all(os.environ.get(k) for k in ("ARC_LLM_MODEL", "ARC_LLM_BASE_URL", "ARC_LLM_API_KEY"))


def has_embedding_creds() -> bool:
    return bool(os.environ.get("MEM_EMBED_API_KEY"))


def build_llm(
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    request_timeout: float | None = 180.0,
):
    """Real gateway client when ARC_LLM_* is set, else a FakeLLMClient."""
    if not has_llm_creds():
        return FakeLLMClient()
    if max_tokens is None:
        max_tokens = int(os.environ.get('ARC_LLM_MAX_TOKENS', '16384'))
    if not 1 <= max_tokens <= 131072:
        raise ValueError('invalid output token allowance')
    retries = int(os.environ.get('ARC_LLM_RETRIES', '0'))
    if retries != 0:
        raise ValueError('This qualified runtime requires ARC_LLM_RETRIES=0')
    # Disable both LiteLLM/SDK retries and the independent NOOA retry loop.
    # One admitted call must make at most one HTTP attempt.
    extra: dict[str, object] = {'max_retries': 0, 'num_retries': 0}
    if max_tokens is not None:
        extra["max_tokens"] = max_tokens
    if request_timeout is not None:
        # Hard per-request HTTP timeout: a gateway request that hangs would otherwise
        # block the solving loop forever (litellm calls aren't cancellation-responsive).
        extra["timeout"] = request_timeout
    model = "openai/" + os.environ["ARC_LLM_MODEL"]  # litellm needs the provider prefix
    reasoning_model = "nemotron" in model.lower()
    # Reasoning-effort routing (verified live against the gateway 2026-07-16):
    #  * Anthropic Opus 4.8+ uses the *adaptive* thinking API — thinking.type=adaptive
    #    + output_config.effort. litellm's reasoning_effort maps to the OLD
    #    thinking.type=enabled, which Opus 4.8 REJECTS ("not supported for this model;
    #    use thinking.type.adaptive and output_config.effort"), and drop_params then
    #    silently drops it -> reasoning_tokens=0, NO thinking. So send the adaptive
    #    params raw via extra_body (CompletionClient forwards **config to litellm).
    #  * gpt-5.x (Responses API) and everything else keep plain reasoning_effort.
    if reasoning_effort:
        if "anthropic" in model.lower():
            extra["extra_body"] = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": reasoning_effort},
            }
        else:
            extra["reasoning_effort"] = reasoning_effort
    # Provider-specific request fields that litellm has no named parameter for.
    # OpenRouter's thinking switch is one: only a top-level ``reasoning`` object turns
    # it off. Measured 2026-09-03 on qwen/qwen3.8-27b — ``chat_template_kwargs`` (top
    # level or nested in extra_body) left reasoning ON, and a short max_tokens then
    # returned content=null with every token in ``reasoning``, which this client raises
    # as EmptyContentError. ARC_LLM_EXTRA_BODY is JSON merged into extra_body:
    #     ARC_LLM_EXTRA_BODY={"reasoning": {"enabled": false}}
    raw_extra_body = os.environ.get("ARC_LLM_EXTRA_BODY")
    if raw_extra_body:
        import json as _json

        merged = dict(extra.get("extra_body") or {})
        merged.update(_json.loads(raw_extra_body))
        extra["extra_body"] = merged
    switches = extra.setdefault('extra_body', {})
    thinking = switches.get('chat_template_kwargs', {}).get('enable_thinking')
    reasoning = switches.get('reasoning', {}).get('enabled')
    if thinking is not None and reasoning is not None and thinking != reasoning:
        raise ValueError('Conflicting reasoning switches')
    enabled = reasoning if reasoning is not None else thinking if thinking is not None else False
    if not isinstance(enabled, bool):
        raise ValueError('Reasoning switch must be boolean')
    # Set both endpoint switches explicitly. Missing settings must not inherit
    # a provider's thinking-on default. Explicit opt-in remains supported.
    switches.setdefault('reasoning', {})['enabled'] = enabled
    switches.setdefault('chat_template_kwargs', {})['enable_thinking'] = enabled

    # gpt-5.x on this gateway must use the Responses API (like the reference profiles'
    # use_responses_api profiles): chat-completions with tools + reasoning_effort
    # gets rerouted gateway-side with a mangled model id → 403 key_model_access_denied.
    # (Observed live when this matched only "gpt-5.5": a gpt-5.6-sol fleet 403'd on
    # every call. gpt-5.4 / 5.5 / 5.6-sol are all verified working via Responses.)
    use_responses = os.environ.get("ARC_LLM_USE_RESPONSES", "1" if "gpt-5" in model else "0") == "1"
    client_cls = ResponsesClient if use_responses else CompletionClient
    kwargs = dict(
        model=model,
        api_base=os.environ["ARC_LLM_BASE_URL"],
        api_key=os.environ["ARC_LLM_API_KEY"],
        # A reasoning response with null prose and valid tool calls is usable,
        # not an empty answer to regenerate. Preserve it for CodeAct parsing.
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0,
                                 retry_on_empty_content=False),
        drop_params=True,  # the gateway rejects some params on the reasoning route
        **extra,
    )
    return client_cls(**kwargs)


def llm_uri() -> str:
    """Model identifier recorded in run outputs."""
    return os.environ.get("ARC_LLM_MODEL", "fake-llm")


def build_embedding_config(force: str = "auto"):
    """Gateway embeddings when keys are set, else the deterministic hashing embedder.

    ``force`` ∈ {"auto", "litellm", "hashing"}.
    """
    from nooa_memory.config import EmbeddingConfig
    want_real = force == "litellm" or (force == "auto" and has_embedding_creds())
    if want_real:
        dims = os.environ.get("MEM_EMBED_DIMS")
        return EmbeddingConfig(
            backend="litellm",
            model=os.environ.get("MEM_EMBED_MODEL", "openai/azure/openai/text-embedding-3-large"),
            endpoint=os.environ.get("MEM_EMBED_BASE_URL", "https://inference-api.nvidia.com/v1"),
            api_key=os.environ.get("MEM_EMBED_API_KEY"),
            dimensions=int(dims) if dims else 1024,
        )
    return EmbeddingConfig(backend="hashing", dim=256)
