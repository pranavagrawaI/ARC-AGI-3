"""Host adapter: Actor prefix -> live constructor.run_construction -> host freeze."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent / "construction-policy-live-20260914T093351Z"
sys.path.insert(0, str(LIVE))

from constructor import ConstructionLimits, ConstructionResult, run_construction  # noqa: E402  # LIVE path
from evaluate_visible import evaluate_visible_abi  # noqa: E402

LS20_CLAUSE = (
    "ABI observations contain grid (list of integer rows), actions are strings, "
    "and singleton commitments must jointly specify block=[row,column] and bar_left=integer, not domains."
)
GENERIC_CLAUSE = (
    "ABI observations contain grid (list of integer rows). Actions are strings. "
    "Do not assume game-specific targets such as block or bar_left."
)


@contextmanager
def bound_constructor():
    import constructor as constructor_mod

    old_system = constructor_mod.SYSTEM
    old_evaluate = constructor_mod.evaluate_candidate
    constructor_mod.SYSTEM = old_system.replace(LS20_CLAUSE, GENERIC_CLAUSE)
    constructor_mod.evaluate_candidate = evaluate_visible_abi
    try:
        yield constructor_mod
    finally:
        constructor_mod.SYSTEM = old_system
        constructor_mod.evaluate_candidate = old_evaluate


class FixtureTransport:
    """Provider-shaped deterministic responses. Never interprets tool calls. SCRIPTED, not live model."""

    label = "scripted-fixture-not-live-model"

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.index = 0

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.index >= len(self.responses):
            raise RuntimeError("fixture responses exhausted")
        response = self.responses[self.index]
        self.index += 1
        return response


class HttpTransport:
    """Real HTTP POST of the constructor request body. Not a live model unless the server is one."""

    label = "real-http-not-claimed-live-model"

    def __init__(self, url: str):
        self.url = url

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        import urllib.request

        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())


def _write_exclusive(path: Path, value: dict[str, Any], mode: int = 0o444) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)


def run_episode(case: Any, transport: Any, output: Path, limits: ConstructionLimits | None = None) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    limits = limits or ConstructionLimits(
        wall_seconds=int(os.environ.get("ARC_CONSTRUCTOR_WALL_SECONDS", "45")),
        max_model_requests=int(os.environ.get("ARC_CONSTRUCTOR_MAX_REQUESTS", "2")),
    )
    with bound_constructor():
        result: ConstructionResult = run_construction(case, transport=transport, limits=limits)
    record: dict[str, Any] = {
        "construction_status": result.status,
        "transport_label": getattr(transport, "label", "unlabeled"),
        "supported_model": False,
        "visible_replay": None,
        "prospective": "not_run",
        "frozen": False,
    }
    if result.status == "CANDIDATE_READY" and isinstance(result.candidate_source, str):
        evaluation = result.evaluation or {}
        visible = (evaluation.get("visible") or {}).get("status")
        claim = evaluation.get("evidence_claim")
        record["visible_replay"] = (evaluation.get("visible") or {}).get("kind")
        record["evidence_claim"] = claim
        if visible != "passed" or claim != "ABI_VISIBLE_REPLAY_ONLY_NOT_SUPPORTED_MODEL":
            record["construction_status"] = "VISIBLE_REPLAY_INCOMPLETE"
        else:
            frozen = {
                "schema": "actor-critic-construction-frozen-v1",
                "candidate_source": result.candidate_source,
                "source_digest": hashlib.sha256(result.candidate_source.encode()).hexdigest(),
                "visible_replay": "abi_replay",
                "prospective": "not_run",
                "supported_model": False,
                "evidence_claim": claim,
            }
            _write_exclusive(output / "FROZEN.json", frozen)
            record["frozen"] = True
    elif result.status == "NEEDS_EVIDENCE":
        record["evidence_request"] = result.evidence_request
    (output / "RESULT.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
