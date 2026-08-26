"""Builder-only namespace adapter around the frozen sterile worker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import MappingProxyType
from typing import Any


def _load_base_worker():  # pragma: no cover - executed only in the child process
    path = Path(__file__).with_name("_sterile_worker.py")
    spec = importlib.util.spec_from_file_location("_tycho_base_sterile_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("base sterile worker is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base_worker()
_base._MAX_PROTOCOL_BYTES = 8 * 1024 * 1024


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


_base_initialization = _base._initialization


def _initialization() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, input_values = _base_initialization()
    return manifest, {
        key: _deep_freeze(value) for key, value in input_values.items()
    }


_base._initialization = _initialization


class _BuilderEvaluate:  # pragma: no cover - child process only
    """The sole mid-session egress from a sterile Builder realm."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def __call__(
        self,
        *,
        candidate_source: str,
        parent_source: str,
        declared_parent_source_sha256: str,
        provenance: dict[str, Any] | None = None,
    ) -> Any:
        return self._client.call(
            "builder.evaluate",
            {
                "candidate_source": candidate_source,
                "declared_parent_source_sha256": declared_parent_source_sha256,
                "parent_source": parent_source,
                "provenance": provenance or {},
            },
        )

    def __repr__(self) -> str:
        return (
            "<builder.evaluate(candidate_source=..., parent_source=..., "
            "declared_parent_source_sha256=..., provenance={...})>"
        )


class _BuilderProxy:  # pragma: no cover - child process only
    def __init__(self, client: Any) -> None:
        self.evaluate = _BuilderEvaluate(client)

    def __repr__(self) -> str:
        return "BuilderCapabilities(granted=['evaluate'])"


def _capability_namespace(manifest: dict[str, Any]) -> dict[str, Any]:
    capabilities = manifest.get("capabilities", [])
    if capabilities == []:
        return {}
    if capabilities != ["builder.evaluate"]:
        raise RuntimeError("Builder sterile realm grants at most builder.evaluate")
    contracts = manifest.get("capability_contracts", {})
    if contracts:
        raise RuntimeError("Builder evaluate has no realm-defined contract overrides")
    client = _base._CapabilityClient(str(manifest["invocation_id"]))
    return {"builder": _BuilderProxy(client)}


_base._capability_namespace = _capability_namespace


if __name__ == "__main__":  # pragma: no cover - child process only
    _base.main()
