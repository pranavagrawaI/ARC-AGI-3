"""Treatment-side launcher that pins one OpenRouter provider endpoint.

The frozen 0002-NOOA runtime remains byte-for-byte unchanged. The live runner
copies this launcher into a preserved runtime overlay whose config points at the
frozen runtime and records the exact provider route added to each request.
"""

from __future__ import annotations

import json
import runpy
import sys
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

# The launcher stays on one audited path for both routes. "pinned" fixes one
# endpoint for evidence runs; "open" carries no provider block and lets
# OpenRouter route freely, for harness shakedown that must not be blocked by one
# provider's shared capacity. Anything else is refused.
EXPECTED_PROVIDER_ROUTES = (
    {
        "order": ["modelrun/fp4"],
        "only": ["modelrun/fp4"],
        "allow_fallbacks": False,
        "require_parameters": False,
        "quantizations": ["fp4"],
    },
    {},
)


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
    return client


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
    runpy.run_path(str(base_runtime_root / "launcher.py"), run_name="__main__")


if __name__ == "__main__":
    _run()
