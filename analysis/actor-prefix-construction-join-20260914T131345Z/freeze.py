"""Host-owned freeze of one candidate. Actor and critic cannot write this file."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def write_frozen(path: Path, candidate_source: str) -> dict[str, Any]:
    record = {
        "schema": "actor-prefix-frozen-v1",
        "candidate_source": candidate_source,
        "source_digest": hashlib.sha256(candidate_source.encode("utf-8")).hexdigest(),
    }
    destination = Path(path)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def load_frozen_module(path: Path):
    record = json.loads(Path(path).read_text())
    source = record["candidate_source"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("FROZEN.json missing candidate_source")
    spec = importlib.util.spec_from_loader("frozen_candidate", loader=None)
    if spec is None:
        raise ValueError("cannot load frozen candidate")
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module
