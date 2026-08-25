"""Context-plane extension of the frozen computational worker.

The base worker remains byte-identical for frozen experiments. This executable
loads it as an implementation module, then replaces only output rendering and
the capability namespace before entering its existing protocol loop.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from inspect import Parameter, Signature
from pathlib import Path
from typing import Any


def _load_base():
    path = Path(__file__).with_name("_computational_worker.py")
    spec = importlib.util.spec_from_file_location("_tycho_context_worker_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("base computational worker is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base()


class _BoundedText(io.TextIOBase):
    """Bound output and preserve proof of the bytes that did not enter context."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._parts: list[str] = []
        self._size = 0
        self._total_size = 0
        self._digest = hashlib.sha256()

    def write(self, text: str) -> int:
        text = str(text)
        encoded = text.encode("utf-8")
        self._total_size += len(encoded)
        self._digest.update(encoded)
        remaining = max(0, self._limit - self._size)
        accepted = encoded[:remaining].decode("utf-8", errors="ignore")
        if accepted:
            self._parts.append(accepted)
            self._size += len(accepted.encode("utf-8"))
        return len(text)

    def getvalue(self) -> str:
        value = "".join(self._parts)
        if self._total_size <= self._limit:
            return value
        marker = (
            f"\n<output truncated: {self._total_size} bytes; "
            f"sha256={self._digest.hexdigest()}>\n"
        )
        prefix = value.encode("utf-8")[: max(0, self._limit - len(marker))]
        return prefix.decode("utf-8", errors="ignore") + marker


class _EvidenceView:
    def __init__(self, client: Any, view_type: type[Any]) -> None:
        self._client = client
        self._view_type = view_type

    def __call__(self, **arguments: Any) -> Any:
        return self._view_type(self._client.call("evidence.view", arguments))

    def __repr__(self) -> str:
        return "<evidence.view(*, handle, refs=(), fields=(), rows=(), max_bytes=2048)>"


class _ContextEvidenceQuery:
    """General matched-Actor query surface generated from its admitted schema."""

    def __init__(self, client: Any, contract: dict[str, Any], receipt_type: type[Any]):
        self._client = client
        self._contract = contract
        self._receipt_type = receipt_type
        self._cache: dict[str, str] = {}
        self.__signature__ = _context_query_signature(contract)
        self.__doc__ = _context_query_documentation(contract)

    def __call__(self, **arguments: Any) -> Any:
        key = json.dumps(
            arguments, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        cached = self._cache.get(key)
        if cached is None:
            value = self._client.call("evidence.query", arguments)
            cached = json.dumps(value, allow_nan=False, separators=(",", ":"))
            self._cache[key] = cached
        return self._receipt_type(json.loads(cached))

    def __repr__(self) -> str:
        properties = self._contract["parameters"]["properties"]
        limit = properties["limit"]
        return (
            "<evidence.query(*, kind: 'frames' | 'transitions', "
            "level: int | None = None, "
            "after: {'level': int, 'turn': int} | None = None, "
            "frontier: {'level': int, 'turn': int} | None = None, "
            f"limit: int = {limit['default']}, "
            "grid_encoding: 'raw' | 'rle' = 'rle', "
            "projection: 'full' | 'metadata' = 'full'; "
            f"limit minimum {limit['minimum']}, maximum {limit['maximum']}; "
            f"example={_context_query_example(self._contract)}>"
        )

    def _documentation(self) -> str:
        return self.__doc__ or "No documentation available."


class _EvidenceProxy:
    def __init__(
        self,
        client: Any,
        contract: dict[str, Any] | None,
        receipt_type: type[Any],
        view_type: type[Any] | None,
    ) -> None:
        if _has_general_query_contract(contract):
            self.query = _ContextEvidenceQuery(client, contract, receipt_type)
        else:
            documented = contract if _has_query_example(contract) else None
            self.query = _base._EvidenceQuery(client, documented, receipt_type)
        if view_type is not None:
            self.view = _EvidenceView(client, view_type)

    def __repr__(self) -> str:
        granted = ["query"]
        if hasattr(self, "view"):
            granted.append("view")
        return f"EvidenceCapabilities(granted={granted!r}, query={self.query!r})"


class _EvidenceRead:
    def __init__(self, client: Any, page_type: type[Any]) -> None:
        self._client = client
        self._page_type = page_type

    def __call__(self, handle: str) -> Any:
        return self._page_type(self._client.call("evidence.read", {"handle": handle}))

    def __repr__(self) -> str:
        return "<evidence_cache.load(handle: str) -> immutable EvidencePage>"


class _EvidenceCacheProxy:
    def __init__(self, client: Any, page_type: type[Any]) -> None:
        self.load = _EvidenceRead(client, page_type)

    def __repr__(self) -> str:
        return (
            "EvidenceCacheCapabilities(granted=['load'], load=" + repr(self.load) + ")"
        )


def _capability_namespace(manifest: dict[str, Any]) -> dict[str, Any]:
    capabilities = _validated_capabilities(manifest)
    query_contract = _query_contract(manifest)
    client = _base._CapabilityClient(str(manifest["invocation_id"]))
    pure = _load_context_evidence()
    values = {"rle": pure.RLEHelpers()}
    if "evidence.query" in capabilities:
        values["evidence"] = _EvidenceProxy(
            client,
            query_contract,
            pure.EvidenceReceiptView,
            pure.EvidenceBoundedView if "evidence.view" in capabilities else None,
        )
    if "evidence.read" in capabilities:
        values["evidence_cache"] = _EvidenceCacheProxy(client, pure.EvidencePageView)
    return values


def _validated_capabilities(manifest: dict[str, Any]) -> list[str]:
    capabilities = manifest.get("capabilities", [])
    if not isinstance(capabilities, list):
        raise TypeError("sterile worker capabilities must be a list")
    unknown = set(capabilities) - {
        "evidence.query",
        "evidence.read",
        "evidence.view",
    }
    if unknown:
        raise RuntimeError(f"unknown sterile capabilities: {sorted(unknown)!r}")
    if "evidence.view" in capabilities and "evidence.query" not in capabilities:
        raise RuntimeError("evidence.view requires the evidence query namespace")
    return capabilities


def _query_contract(manifest: dict[str, Any]) -> dict[str, Any] | None:
    contracts = manifest.get("capability_contracts", {})
    if not isinstance(contracts, dict):
        raise TypeError("sterile worker capability_contracts must be an object")
    query_contract = contracts.get("evidence.query")
    if query_contract is not None and not isinstance(query_contract, dict):
        raise TypeError("evidence.query contract must be an object")
    return query_contract


def _load_context_evidence():
    path = Path(__file__).with_name("context_computational_evidence_v2.py")
    spec = importlib.util.spec_from_file_location("_tycho_context_evidence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("context computational evidence module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _has_query_example(contract: dict[str, Any] | None) -> bool:
    return isinstance(contract, dict) and all(
        name in contract for name in ("kind", "level", "limit", "projection")
    )


def _has_general_query_contract(contract: dict[str, Any] | None) -> bool:
    if not isinstance(contract, dict):
        return False
    parameters = contract.get("parameters")
    example = contract.get("example")
    return (
        isinstance(parameters, dict)
        and isinstance(parameters.get("properties"), dict)
        and isinstance(example, dict)
        and contract.get("result_schema") == "evidence_receipt.v2"
    )


def _context_query_signature(contract: dict[str, Any]) -> Signature:
    properties = contract["parameters"]["properties"]
    parameters = []
    for name in (
        "kind",
        "level",
        "after",
        "frontier",
        "limit",
        "grid_encoding",
        "projection",
    ):
        schema = properties[name]
        default = schema.get("default", Parameter.empty)
        parameters.append(
            Parameter(name, Parameter.KEYWORD_ONLY, default=default)
        )
    return Signature(parameters)


def _context_query_documentation(contract: dict[str, Any]) -> str:
    limit = contract["parameters"]["properties"]["limit"]
    return "\n".join(
        (
            "Acquire a bounded immutable evidence page and return its receipt.",
            "Both after and frontier cursors require integer level and turn fields.",
            f"limit must be between {limit['minimum']} and {limit['maximum']}.",
            "Call: " + _context_query_example(contract),
            (
                "The receipt exposes refs, next_after, truncated, and "
                "truncation_reason; bulk records remain outside provider context."
            ),
        )
    )


def _context_query_example(contract: dict[str, Any]) -> str:
    example = contract["example"]
    arguments = ", ".join(f"{name}={value!r}" for name, value in example.items())
    return f"evidence.query({arguments})"


_base._BoundedText = _BoundedText
_base._capability_namespace = _capability_namespace

if __name__ == "__main__":
    _base.main()
