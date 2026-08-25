"""Context worker using the congruent post-pilot evidence receipt."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_base():
    path = Path(__file__).with_name("_context_computational_worker_v2.py")
    spec = importlib.util.spec_from_file_location("_tycho_context_worker_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("context worker v2 module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_context_evidence():
    path = Path(__file__).with_name("context_computational_evidence_v3.py")
    spec = importlib.util.spec_from_file_location("_tycho_context_evidence_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("context evidence v3 module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base()
_base._load_context_evidence = _load_context_evidence
_base._base._capability_namespace = _base._capability_namespace

if __name__ == "__main__":
    _base._base.main()
