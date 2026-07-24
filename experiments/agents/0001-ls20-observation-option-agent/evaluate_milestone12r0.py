"""Evaluate M12.R0 public factor discovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.harness import Observation  # noqa: E402
from experiments.harness.viewer import FrameRecord, load_play_record  # noqa: E402
from observation_option_agent import (  # noqa: E402
    FactorTransitionAssay,
    PublicFactorExtractor,
    factor_hypotheses_from_transitions,
    forbidden_factor_discovery_findings,
    m12r0_status,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_TRACE = (
    AGENT_ROOT
    / "artifacts/recordings"
    / "ls20-milestone-117-public-replay.jsonl"
)
DEFAULT_SOURCE_EVENTS = ARTIFACT_DIR / "milestone-117-events.jsonl"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-12r0-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-12r0-events.jsonl"
DEFAULT_FACTORS = ARTIFACT_DIR / "factor-candidates.json"
DEFAULT_PREDICTIONS = ARTIFACT_DIR / "factor-transition-predictions.json"
DEFAULT_COMPRESSION = ARTIFACT_DIR / "factor-compression-audit.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-12r0-freeze.md"


def evaluate(
    *,
    trace_path: Path,
    source_events: Path,
    output_events: Path,
    output_factors: Path,
    output_predictions: Path,
    output_compression: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    frames = load_play_record(trace_path)
    controlled_positions = _controlled_positions_by_step(source_events)
    extractor = PublicFactorExtractor()
    transitions = []
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
        transitions.append(transition)
        events.append({
            "milestone": "M12_R0_PUBLIC_FACTOR_DISCOVERY",
            "phase": "PUBLIC_FACTOR_EXTRACTION",
            "frame_index": index,
            "transition": transition.to_record(),
        })

    transition_tuple = tuple(transitions)
    audit = FactorTransitionAssay().evaluate(transition_tuple)
    hypotheses = factor_hypotheses_from_transitions(transition_tuple)
    payload = {
        "trace_path": str(trace_path),
        "source_events": str(source_events),
        "hypotheses": list(hypotheses),
        "transitions": [transition.to_record() for transition in transition_tuple],
        "audit": audit.to_record(),
    }
    findings = forbidden_factor_discovery_findings(payload)
    actor_factor_count = sum(
        1 for transition in transition_tuple if transition.actor_factor is not None
    )
    stable_factor_transition_count = sum(
        1 for transition in transition_tuple if transition.stable_factors
    )
    changed_factor_transition_count = sum(
        1 for transition in transition_tuple if transition.changed_factors
    )
    status = m12r0_status(
        transition_count=len(transition_tuple),
        actor_factor_count=actor_factor_count,
        stable_factor_transition_count=stable_factor_transition_count,
        changed_factor_transition_count=changed_factor_transition_count,
        compression_gain=audit.compression_gain,
        factor_context_reuse_count=audit.factor_context_reuse_count,
        raw_context_reuse_count=audit.raw_context_reuse_count,
        prediction_gain=audit.prediction_gain,
        named_object_ontology=False,
        oracle_or_private_state_used=bool(findings),
    )
    events.append({
        "milestone": "M12_R0_PUBLIC_FACTOR_DISCOVERY",
        "phase": "FACTOR_TRANSITION_ASSAY",
        "audit": audit.to_record(),
        "hypotheses": list(hypotheses),
        "forbidden_source_findings": list(findings),
    })
    _write_jsonl(output_events, events)
    _write_json(output_factors, {
        "hypotheses": list(hypotheses),
        "transitions": [transition.to_record() for transition in transition_tuple],
    })
    _write_json(output_predictions, {"audit": audit.to_record()})
    _write_json(output_compression, {
        "raw_state_count": audit.raw_state_count,
        "factor_state_count": audit.factor_state_count,
        "compression_gain": audit.compression_gain,
        "raw_context_reuse_count": audit.raw_context_reuse_count,
        "factor_context_reuse_count": audit.factor_context_reuse_count,
        "raw_prediction_accuracy": audit.raw_prediction_accuracy,
        "factor_prediction_accuracy": audit.factor_prediction_accuracy,
        "prediction_gain": audit.prediction_gain,
    })
    summary = {
        "milestone": "M12_R0_PUBLIC_FACTOR_DISCOVERY",
        "status": status,
        "contract_status": status,
        "trace_path": str(trace_path),
        "source_events": str(source_events),
        "transition_count": len(transition_tuple),
        "actor_factor_count": actor_factor_count,
        "stable_factor_transition_count": stable_factor_transition_count,
        "changed_factor_transition_count": changed_factor_transition_count,
        "raw_state_count": audit.raw_state_count,
        "factor_state_count": audit.factor_state_count,
        "compression_gain": audit.compression_gain,
        "raw_context_reuse_count": audit.raw_context_reuse_count,
        "factor_context_reuse_count": audit.factor_context_reuse_count,
        "raw_prediction_accuracy": audit.raw_prediction_accuracy,
        "factor_prediction_accuracy": audit.factor_prediction_accuracy,
        "prediction_gain": audit.prediction_gain,
        "forbidden_source_findings": list(findings),
        "known_limits": {
            "factor_key_is_coarse": True,
            "not_terminal_causality": True,
            "not_full_object_discovery": True,
        },
        "artifacts": {
            "events": str(output_events),
            "factors": str(output_factors),
            "predictions": str(output_predictions),
            "compression": str(output_compression),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _observation_from_frame(frame: FrameRecord) -> Observation:
    pixels = np.asarray(frame.pixels, dtype=np.int8)
    pixels.setflags(write=False)
    return Observation(
        pixels=pixels,
        available_actions=tuple(frame.available_actions),
        levels_completed=int(frame.levels_completed or 0),
        state=frame.state or "UNKNOWN",
        full_reset=bool(frame.full_reset),
    )


def _action_id(frame: FrameRecord) -> int | None:
    if frame.action is None or frame.action == "START":
        return None
    if frame.action.startswith("ACTION"):
        return int(frame.action.removeprefix("ACTION"))
    return int(frame.action)


def _controlled_positions_by_step(events_path: Path) -> dict[int, tuple[int, int]]:
    if not events_path.exists():
        return {}
    positions = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        for record in _iter_action_records(event):
            if not record.get("executed"):
                continue
            position = record.get("controlled_position")
            if not isinstance(position, list) or len(position) != 2:
                continue
            positions[int(record["step"])] = (int(position[0]), int(position[1]))
    return positions


def _iter_action_records(value: Any):
    if isinstance(value, dict):
        if {"step", "action", "executed"}.issubset(value.keys()):
            yield value
        for item in value.values():
            yield from _iter_action_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_action_records(item)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 12.R0 Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Lab-book line:",
        "",
        "```text",
        "M12.R0:",
        "  I can extract public object/action factors from interaction evidence well enough to predict local effects better than raw frame identity.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- transitions: `{summary['transition_count']}`",
        f"- actor factor count: `{summary['actor_factor_count']}`",
        f"- stable factor transition count: `{summary['stable_factor_transition_count']}`",
        f"- changed factor transition count: `{summary['changed_factor_transition_count']}`",
        f"- raw state count: `{summary['raw_state_count']}`",
        f"- factor state count: `{summary['factor_state_count']}`",
        f"- compression gain: `{summary['compression_gain']}`",
        f"- raw context reuse count: `{summary['raw_context_reuse_count']}`",
        f"- factor context reuse count: `{summary['factor_context_reuse_count']}`",
        f"- raw prediction accuracy: `{summary['raw_prediction_accuracy']}`",
        f"- factor prediction accuracy: `{summary['factor_prediction_accuracy']}`",
        f"- forbidden source findings: `{summary['forbidden_source_findings']}`",
        "",
        "Known limits:",
        "",
        "- This is a coarse factor-discovery assay, not full object discovery.",
        "- It does not infer terminal causality or solve LS20.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--source-events", type=Path, default=DEFAULT_SOURCE_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--factors", type=Path, default=DEFAULT_FACTORS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--compression", type=Path, default=DEFAULT_COMPRESSION)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    summary = evaluate(
        trace_path=args.trace,
        source_events=args.source_events,
        output_events=args.events,
        output_factors=args.factors,
        output_predictions=args.predictions,
        output_compression=args.compression,
        output_freeze=args.freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
