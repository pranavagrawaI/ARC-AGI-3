"""Congruent model-facing evidence views for post-pilot treatments."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_base():
    path = Path(__file__).with_name("context_computational_evidence_v2.py")
    spec = importlib.util.spec_from_file_location("_tycho_context_evidence_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("context evidence v2 module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base()
EvidencePageView = _base.EvidencePageView
EvidenceBoundedView = _base.EvidenceBoundedView
EvidenceRecordView = _base.EvidenceRecordView
RLEHelpers = _base.RLEHelpers


class EvidenceReceiptView(_base.EvidenceReceiptView):
    """A receipt whose display and discoverable attributes tell the same truth."""

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._value))

    def __repr__(self) -> str:
        return (
            f"EvidenceReceipt(handle={self.handle!r}, "
            f"record_count={self.record_count}, refs={self.refs!r}, "
            f"encoding={self.encoding!r}, next_after={self.next_after!r}, "
            f"truncated={self.truncated!r}, "
            f"truncation_reason={self.truncation_reason!r}, "
            f"load=evidence_cache.load({self.handle!r}))"
        )
