"""Preserve constructor tool/progress that overlay run_episode drops. Overlay files are not edited."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from host import bound_constructor, _write_exclusive
from constructor import ConstructionLimits, run_construction


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def persist_episode(case: Any, transport: Any, output: Path, limits: ConstructionLimits | None = None) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    events_path = output / "tool-events.jsonl"
    progress_path = output / "CONSTRUCTOR-PROGRESS.json"
    error_path = output / "evaluation-error.json"

    def progress_sink(result: Any) -> None:
        snapshot = {
            "status": result.status,
            "model_requests": result.model_requests,
            "inspections": result.inspections,
            "submissions": result.submissions,
            "usage": result.usage,
            "frozen": False,
            "tool_event_count": len(result.tool_events),
            "transcript": result.transcript,
        }
        progress_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    def event_sink(event: dict[str, Any]) -> None:
        _append_jsonl(events_path, event)

    limits = limits or ConstructionLimits(
        wall_seconds=int(os.environ.get("ARC_CONSTRUCTOR_WALL_SECONDS", "240")),
        max_model_requests=int(os.environ.get("ARC_CONSTRUCTOR_MAX_REQUESTS", "6")),
    )
    with bound_constructor():
        result = run_construction(
            case, transport=transport, limits=limits, event_sink=event_sink, progress_sink=progress_sink
        )
    progress_sink(result)
    record: dict[str, Any] = {
        "construction_status": result.status,
        "transport_label": getattr(transport, "label", "unlabeled"),
        "supported_model": False,
        "visible_replay": None,
        "prospective": "not_run",
        "frozen": False,
        "tool_event_count": len(result.tool_events),
        "model_requests": result.model_requests,
        "inspections": result.inspections,
        "submissions": result.submissions,
        "usage": result.usage,
        "construction_repair": {"failed_submissions": sum(e["call"]["function"]["name"] == "submit_candidate" and (json.loads(e["rendered_result_message"]["content"]).get("status") == "rejected" or json.loads(e["rendered_result_message"]["content"]).get("visible_evaluation", {}).get("visible", {}).get("status") == "failed") for e in result.tool_events), "accepted_after_repair": result.status == "CANDIDATE_READY" and result.submissions > 1},
    }
    if result.status == "CANDIDATE_READY" and isinstance(result.candidate_source, str):
        evaluation = result.evaluation or {}
        visible = (evaluation.get("visible") or {}).get("status")
        claim = evaluation.get("evidence_claim")
        record["visible_replay"] = (evaluation.get("visible") or {}).get("kind")
        record["evidence_claim"] = claim
        if visible != "passed" or claim != "ABI_VISIBLE_REPLAY_ONLY_NOT_SUPPORTED_MODEL":
            record["construction_status"] = "VISIBLE_REPLAY_INCOMPLETE"
            error_path.write_text(json.dumps({"evaluation": evaluation}, indent=2, sort_keys=True) + "\n")
        else:
            frozen = {
                "schema": "actor-critic-construction-frozen-v1",
                "candidate_source": result.candidate_source,
                "source_digest": __import__("hashlib").sha256(result.candidate_source.encode()).hexdigest(),
                "visible_replay": "abi_replay",
                "prospective": "not_run",
                "supported_model": False,
                "evidence_claim": claim,
            }
            _write_exclusive(output / "FROZEN.json", frozen)
            record["frozen"] = True
    elif result.status == "NEEDS_EVIDENCE":
        record["evidence_request"] = result.evidence_request
    elif result.status not in {"CANDIDATE_READY", "BUDGET_EXHAUSTED"}:
        error_path.write_text(json.dumps({"status": result.status, "usage": result.usage}, indent=2, sort_keys=True) + "\n")
    (output / "RESULT.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record

