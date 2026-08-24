"""Observational milestones for evidence-to-computation traces."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from tycho_nooa.execution_events import ExecutionProvenanceEvent


class ComputationalMilestones(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_record_disclosed: int | None = None
    representation_decoded: int | None = None
    target_value_identified: int | None = None
    provenance_bound: int | None = None
    terminal_result_accepted: int | None = None


def extract_computational_milestones(
    path: Path,
    *,
    target_ref: str,
    before: int,
    after: int,
) -> ComputationalMilestones:
    events = [
        ExecutionProvenanceEvent.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    calls = _call_indices(events)
    disclosed = _first_matching_call(events, calls, "CapabilityReturned", target_ref)
    value = f"{before}->{after}"
    decoded = _first_matching_call(events, calls, "CellCompleted", value)
    bound = _first_matching_call(events, calls, "CellCompleted", target_ref, value)
    terminal = _first_matching_call(events, calls, "ReturnAccepted", target_ref, value)
    return ComputationalMilestones(
        target_record_disclosed=disclosed,
        representation_decoded=decoded,
        target_value_identified=decoded,
        provenance_bound=bound,
        terminal_result_accepted=terminal,
    )


def _call_indices(events: list[ExecutionProvenanceEvent]) -> dict[int, int]:
    call = 0
    result = {}
    for event in events:
        if event.event_type in {"CellDispatched", "ProviderTurnRejected"}:
            call += 1
        result[event.event_seq] = call
    return result


def _first_matching_call(
    events: list[ExecutionProvenanceEvent],
    calls: dict[int, int],
    event_type: str,
    *needles: str,
) -> int | None:
    for event in events:
        if event.event_type != event_type:
            continue
        visible = str(event.payload)
        if all(needle in visible for needle in needles):
            return calls[event.event_seq]
    return None
