"""Treatment-side launcher that pins one OpenRouter provider endpoint.

The frozen 0002-NOOA runtime remains byte-for-byte unchanged. The live runner
copies this launcher into a preserved runtime overlay whose config points at the
frozen runtime and records the exact provider route added to each request.
"""

from __future__ import annotations

import asyncio
import json
import runpy
import sys
from collections.abc import Mapping
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

# The launcher stays on one audited path for all routes. The accepted payloads
# are enumerated here so a changed transport overlay cannot silently redirect an
# approved run. Anything else is refused.
EXPECTED_PROVIDER_ROUTES = (
    {
        "order": ["modelrun/fp4"],
        "only": ["modelrun/fp4"],
        "allow_fallbacks": False,
        "require_parameters": False,
        "quantizations": ["fp4"],
    },
    {},
    {
        "order": ["openai"],
        "only": ["openai"],
        "allow_fallbacks": False,
        "require_parameters": False,
    },
    {
        "order": ["modelrun/fp4"],
        "allow_fallbacks": True,
        "require_parameters": False,
    },
)
OPENAI_PROVIDER_ROUTE = EXPECTED_PROVIDER_ROUTES[2]
OPENAI_PROMPT_CACHE_KEY_MAX_CHARS = 64
RATE_LIMIT_STATUS = 429
# Applied before the error is re-raised into the frozen retry loop, which
# retries immediately. An instant retry into a pool that just said "temporarily
# rate-limited upstream" is a wasted attempt.
RATE_LIMIT_BACKOFF_SECONDS = (5.0, 15.0)


class _RateLimitBackoffClient:
    """Wait before letting a rate-limited call return to the retry loop."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._consecutive = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        try:
            response = await self._client.acall(*args, **kwargs)
        except Exception as error:
            if getattr(error, "status_code", None) == RATE_LIMIT_STATUS:
                index = min(self._consecutive, len(RATE_LIMIT_BACKOFF_SECONDS) - 1)
                self._consecutive += 1
                await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS[index])
            raise
        self._consecutive = 0
        return response


def _rate_limit_is_retryable(frozen: Any) -> Any:
    """Treat a 429 as retryable without editing the frozen runtime.

    The frozen classifier calls 429 UNCERTAIN, and uncertain results are never
    retried because the request may already have been served. That rule is right
    in general and wrong for this case: every 429 in the recorded runs arrived
    as an OpenRouter error body carrying `code: 429` and
    `limit_source: upstream_provider_shared_pool`, with **no generation id** —
    the request was refused before it reached the model, so there is no served
    response for a retry to duplicate. Three runs died at 0, 1 and 13 actions on
    a failure that nothing had consumed.

    This is an override in the treatment's own audited overlay, not an edit to
    `nooa-runtime/llm.py`, so the frozen 0002 runtime stays byte-for-byte
    unchanged and both arms travel the same launcher. Every other disposition is
    the frozen function's answer, untouched.

    Its limit, stated rather than assumed: the evidence is that OpenRouter's
    shared-pool 429 is a pre-service refusal. A provider that returned 429 after
    partially serving a request would break that reasoning, and the frozen
    `require_actual_crossings` check is what would catch it.
    """

    frozen_classify = frozen.classify_transport_failure
    retryable = frozen.TransportDisposition.RETRYABLE

    def classify(error: Exception) -> Any:
        if getattr(error, "status_code", None) == RATE_LIMIT_STATUS:
            return retryable
        return frozen_classify(error)

    return classify


class _PromptCacheKeyBoundClient:
    """Apply OpenAI's length bound to NOOA's opaque cache-shard identity."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        key = kwargs.get("prompt_cache_key")
        if isinstance(key, str) and len(key) > OPENAI_PROMPT_CACHE_KEY_MAX_CHARS:
            suffix = sha256(key.encode("utf-8")).hexdigest()[:16]
            prefix_chars = OPENAI_PROMPT_CACHE_KEY_MAX_CHARS - len(suffix) - 1
            kwargs["prompt_cache_key"] = f"{key[:prefix_chars]}-{suffix}"
        return await self._client.acall(*args, **kwargs)


def _load_overlay_config(path: Path) -> tuple[Path, dict[str, Any]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if set(record) != {"schema", "base_runtime_root", "provider"}:
        raise RuntimeError("transport overlay config has unknown or missing fields")
    if record["schema"] != "0006-nooa-provider-runtime-overlay-v1":
        raise RuntimeError("transport overlay config has the wrong schema")
    provider = record["provider"]
    if not isinstance(provider, Mapping) or dict(provider) not in EXPECTED_PROVIDER_ROUTES:
        raise RuntimeError("transport overlay config carries an unknown provider route")
    base_runtime_root = Path(record["base_runtime_root"]).resolve()
    if not (base_runtime_root / "launcher.py").is_file():
        raise RuntimeError("transport overlay base runtime is unavailable")
    return base_runtime_root, dict(provider)


def _pin_client(client: Any, provider: dict[str, Any]) -> Any:
    if "extra_body" in client.config:
        client.close()
        raise RuntimeError("frozen provider client unexpectedly supplied extra_body")
    if provider:
        client.config["extra_body"] = {"provider": provider}
    if provider == OPENAI_PROVIDER_ROUTE:
        return _RateLimitBackoffClient(_PromptCacheKeyBoundClient(client))
    return _RateLimitBackoffClient(client)


def _run() -> None:
    overlay_root = Path(__file__).resolve().parent
    base_runtime_root, provider = _load_overlay_config(
        overlay_root / "transport-config.json"
    )
    sys.path.insert(0, str(base_runtime_root))

    frozen_llm: Any = import_module("llm")

    frozen_build_llm = frozen_llm.build_llm

    def build_pinned_llm(
        profile_record: Mapping[str, Any],
        *,
        api_key: str | None,
    ) -> Any:
        client = frozen_build_llm(profile_record, api_key=api_key)
        return _pin_client(client, provider)

    frozen_llm.build_llm = build_pinned_llm
    frozen_llm.classify_transport_failure = _rate_limit_is_retryable(frozen_llm)
    runpy.run_path(str(base_runtime_root / "launcher.py"), run_name="__main__")


if __name__ == "__main__":
    _run()
