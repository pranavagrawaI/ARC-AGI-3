"""Context-plane views layered over the frozen shared computational helpers."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


def _load_base():
    path = Path(__file__).with_name("computational_evidence.py")
    spec = importlib.util.spec_from_file_location("_tycho_computational_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("base computational evidence module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base()
EvidenceRecordView = _base.EvidenceRecordView
RLEHelpers = _base.RLEHelpers


class EvidencePageView(_base.EvidencePageView):
    """The exact cached page; its receipt, rather than its payload, carries schema."""

    def __init__(self, value: dict[str, Any]):
        if not isinstance(value.get("records"), list):
            raise TypeError("unsupported evidence page schema")
        self._value = value
        self.records = tuple(EvidenceRecordView(row) for row in value["records"])


class EvidenceReceiptView(Mapping[str, Any]):
    """Small provider-safe name for exact evidence retained outside context."""

    def __init__(self, value: dict[str, Any]):
        if value.get("schema_version") != "evidence_receipt.v2":
            raise ValueError("unsupported evidence receipt schema")
        self._value = value

    def __getitem__(self, key: str) -> Any:
        return self._value[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._value[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __repr__(self) -> str:
        return (
            f"EvidenceReceipt(handle={self.handle!r}, records={self.record_count}, "
            f"refs={self.refs!r}, encoding={self.encoding!r}, "
            f"next_after={self.next_after!r}, truncated={self.truncated!r}, "
            f"truncation_reason={self.truncation_reason!r})"
        )


class EvidenceBoundedView(Mapping[str, Any]):
    """Explicit view over a bounded part of one previously granted entry."""

    def __init__(self, value: dict[str, Any]):
        self._value = value
        self.records = tuple(EvidenceRecordView(row) for row in value["records"])

    def __getitem__(self, key: str) -> Any:
        return self.records if key == "records" else self._value[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._value[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __repr__(self) -> str:
        return f"EvidenceView(handle={self.handle!r}, records={len(self.records)})"
