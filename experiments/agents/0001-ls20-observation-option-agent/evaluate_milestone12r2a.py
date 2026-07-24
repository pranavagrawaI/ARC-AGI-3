"""Evaluate M12.R2a actor-centric local-geometry repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.harness.viewer import load_play_record  # noqa: E402
from evaluate_milestone12r0 import (  # noqa: E402
    _action_id,
    _controlled_positions_by_step,
    _observation_from_frame,
    _write_json,
    _write_jsonl,
)
from observation_option_agent import (  # noqa: E402
    ActorLocalGeometryRefiner,
    ActorLocalGeometryRepairAssay,
    PublicFactorExtractor,
    forbidden_local_geometry_findings,
    m12r2a_status,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_TRACE = (
    AGENT_ROOT
    / "artifacts/recordings"
    / "ls20-milestone-117-public-replay.jsonl"
)
DEFAULT_SOURCE_EVENTS = ARTIFACT_DIR / "milestone-117-events.jsonl"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-12r2a-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-12r2a-events.jsonl"
DEFAULT_REPAIR = ARTIFACT_DIR / "local-geometry-repair-audit.json"
DEFAULT_ALIAS_LEDGER = ARTIFACT_DIR / "alias-resolution-ledger.json"
DEFAULT_COORDINATE_CONTROL = ARTIFACT_DIR / "coordinate-control-audit.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-12r2a-freeze.md"


def evaluate(
    *,
    trace_path: Path,
    source_events: Path,
    output_events: Path,
    output_repair: Path,
    output_alias_ledger: Path,
    output_coordinate_control: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    frames = load_play_record(trace_path)
    controlled_positions = _controlled_positions_by_step(source_events)
    extractor = PublicFactorExtractor()
    refiner = ActorLocalGeometryRefiner()
    transitions = []
    records = []
    events = []
    for index in range(1, len(frames)):
        frame = frames[index]
        action = _action_id(frame)
        if action is None:
            continue
        before = _observation_from_frame(frames[index - 1])
        after = _observation_from_frame(frame)
        transition = extractor.extract(
            transition_id=f"factor_transition_{index:03d}",
            before=before,
            action=action,
            after=after,
            controlled_position=controlled_positions.get(index),
        )
        actor_position = (
            None if transition.actor_factor is not None
            else controlled_positions.get(index)
        )
        record = refiner.refine(
            transition=transition,
            before=before,
            actor_position=actor_position,
        )
        transitions.append(transition)
        records.append(record)
        events.append({
            "milestone": "M12_R2A_ACTOR_LOCAL_GEOMETRY_REPAIR",
            "phase": "LOCAL_GEOMETRY_REFINEMENT",
            "frame_index": index,
            "transition": transition.to_record(),
            "refinement": record.to_record(),
        })

    transition_tuple = tuple(transitions)
    record_tuple = tuple(records)
    audit = ActorLocalGeometryRepairAssay().evaluate(
        record_tuple,
        base_transitions=transition_tuple,
    )
    payload = {
        "trace_path": str(trace_path),
        "source_events": str(source_events),
        "records": [record.to_record() for record in record_tuple],
        "audit": audit.to_record(),
    }
    findings = forbidden_local_geometry_findings(payload)
    status = m12r2a_status(
        raw_state_count=audit.raw_state_count,
        local_state_count=audit.local_geometry.state_count,
        local_reused_context_count=audit.local_geometry.reused_context_count,
        local_stable_context_count=audit.local_geometry.stable_context_count,
        local_contradicted_context_count=audit.local_geometry.contradicted_context_count,
        local_prediction_accuracy=audit.local_geometry.prediction_accuracy,
        resolved_alias_count=audit.resolved_alias_count,
        coordinate_reused_context_count=audit.coordinate_control.reused_context_count,
        coordinate_contradicted_context_count=(
            audit.coordinate_control.contradicted_context_count
        ),
        named_object_ontology=False,
        oracle_or_private_state_used=bool(findings),
    )
    events.append({
        "milestone": "M12_R2A_ACTOR_LOCAL_GEOMETRY_REPAIR",
        "phase": "LOCAL_GEOMETRY_REPAIR_ASSAY",
        "audit": audit.to_record(),
        "forbidden_source_findings": list(findings),
    })
    _write_jsonl(output_events, events)
    _write_json(output_repair, {"audit": audit.to_record()})
    _write_json(output_alias_ledger, {
        "alias_resolutions": [
            resolution.to_record() for resolution in audit.alias_resolutions
        ],
    })
    _write_json(output_coordinate_control, {
        "coordinate_control": audit.coordinate_control.to_record(),
    })
    summary = {
        "milestone": "M12_R2A_ACTOR_LOCAL_GEOMETRY_REPAIR",
        "status": status,
        "contract_status": status,
        "trace_path": str(trace_path),
        "source_events": str(source_events),
        "transition_count": audit.transition_count,
        "raw_state_count": audit.raw_state_count,
        "base_reused_context_count": audit.base.reused_context_count,
        "base_stable_context_count": audit.base.stable_context_count,
        "base_contradicted_context_count": audit.base.contradicted_context_count,
        "local_state_count": audit.local_geometry.state_count,
        "local_reused_context_count": audit.local_geometry.reused_context_count,
        "local_stable_context_count": audit.local_geometry.stable_context_count,
        "local_contradicted_context_count": (
            audit.local_geometry.contradicted_context_count
        ),
        "local_prediction_accuracy": audit.local_geometry.prediction_accuracy,
        "coordinate_state_count": audit.coordinate_control.state_count,
        "coordinate_reused_context_count": audit.coordinate_control.reused_context_count,
        "coordinate_stable_context_count": audit.coordinate_control.stable_context_count,
        "coordinate_contradicted_context_count": (
            audit.coordinate_control.contradicted_context_count
        ),
        "coordinate_prediction_accuracy": audit.coordinate_control.prediction_accuracy,
        "resolved_alias_count": audit.resolved_alias_count,
        "terminal_alias_split_count": audit.terminal_alias_split_count,
        "forbidden_source_findings": list(findings),
        "known_limits": {
            "pre_action_local_geometry_only": True,
            "does_not_name_object_roles": True,
            "does_not_repair_terminal_causality": True,
            "coordinate_control_is_negative_control": True,
        },
        "artifacts": {
            "events": str(output_events),
            "repair": str(output_repair),
            "alias_ledger": str(output_alias_ledger),
            "coordinate_control": str(output_coordinate_control),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 12.R2a Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Lab-book line:",
        "",
        "```text",
        "M12.R2a:",
        "  I can test whether pre-action actor-centric local geometry repairs movement/no-op aliasing without collapsing back to raw-state memory.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- transitions: `{summary['transition_count']}`",
        f"- raw states: `{summary['raw_state_count']}`",
        f"- base reused contexts: `{summary['base_reused_context_count']}`",
        f"- base stable contexts: `{summary['base_stable_context_count']}`",
        f"- base contradicted contexts: `{summary['base_contradicted_context_count']}`",
        f"- local geometry states: `{summary['local_state_count']}`",
        f"- local geometry reused contexts: `{summary['local_reused_context_count']}`",
        f"- local geometry stable contexts: `{summary['local_stable_context_count']}`",
        f"- local geometry contradicted contexts: `{summary['local_contradicted_context_count']}`",
        f"- local geometry prediction accuracy: `{summary['local_prediction_accuracy']}`",
        f"- coordinate control states: `{summary['coordinate_state_count']}`",
        f"- coordinate control reused contexts: `{summary['coordinate_reused_context_count']}`",
        f"- coordinate control contradicted contexts: `{summary['coordinate_contradicted_context_count']}`",
        f"- coordinate control prediction accuracy: `{summary['coordinate_prediction_accuracy']}`",
        f"- resolved aliases: `{summary['resolved_alias_count']}`",
        f"- terminal aliases split: `{summary['terminal_alias_split_count']}`",
        f"- forbidden source findings: `{summary['forbidden_source_findings']}`",
        "",
        "Known limits:",
        "",
        "- This is R2a only: actor-centric local geometry.",
        "- It does not add contact/adjacency object factors beyond immediate local geometry.",
        "- It does not infer terminal causality.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--source-events", type=Path, default=DEFAULT_SOURCE_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--repair", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument("--alias-ledger", type=Path, default=DEFAULT_ALIAS_LEDGER)
    parser.add_argument(
        "--coordinate-control",
        type=Path,
        default=DEFAULT_COORDINATE_CONTROL,
    )
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    summary = evaluate(
        trace_path=args.trace,
        source_events=args.source_events,
        output_events=args.events,
        output_repair=args.repair,
        output_alias_ledger=args.alias_ledger,
        output_coordinate_control=args.coordinate_control,
        output_freeze=args.freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
