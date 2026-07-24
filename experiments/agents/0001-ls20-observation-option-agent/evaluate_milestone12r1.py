"""Evaluate M12.R1 public factor counterexample assay."""

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
    FactorCounterexampleAssay,
    PublicFactorExtractor,
    forbidden_factor_counterexample_findings,
    m12r1_status,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_TRACE = (
    AGENT_ROOT
    / "artifacts/recordings"
    / "ls20-milestone-117-public-replay.jsonl"
)
DEFAULT_SOURCE_EVENTS = ARTIFACT_DIR / "milestone-117-events.jsonl"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-12r1-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-12r1-events.jsonl"
DEFAULT_COUNTEREXAMPLES = ARTIFACT_DIR / "factor-counterexamples.json"
DEFAULT_STABILITY = ARTIFACT_DIR / "factor-stability-audit.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-12r1-freeze.md"


def evaluate(
    *,
    trace_path: Path,
    source_events: Path,
    output_events: Path,
    output_counterexamples: Path,
    output_stability: Path,
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
        transition = extractor.extract(
            transition_id=f"factor_transition_{index:03d}",
            before=_observation_from_frame(frames[index - 1]),
            action=action,
            after=_observation_from_frame(frame),
            controlled_position=controlled_positions.get(index),
        )
        transitions.append(transition)
        events.append({
            "milestone": "M12_R1_PUBLIC_FACTOR_COUNTEREXAMPLES",
            "phase": "PUBLIC_FACTOR_EXTRACTION",
            "frame_index": index,
            "transition": transition.to_record(),
        })

    transition_tuple = tuple(transitions)
    audit = FactorCounterexampleAssay().evaluate(transition_tuple)
    counterexample_contexts = tuple(
        context for context in audit.contexts if context.status == "contradicted"
    )
    stable_contexts = tuple(
        context for context in audit.contexts if context.status == "stable"
    )
    payload = {
        "trace_path": str(trace_path),
        "source_events": str(source_events),
        "transitions": [transition.to_record() for transition in transition_tuple],
        "audit": audit.to_record(),
    }
    findings = forbidden_factor_counterexample_findings(payload)
    status = m12r1_status(
        reused_context_count=audit.reused_context_count,
        contradicted_context_count=audit.contradicted_context_count,
        stable_context_count=audit.stable_context_count,
        promoted_context_count=audit.promoted_context_count,
        blocked_promotion_count=audit.blocked_promotion_count,
        named_object_ontology=False,
        oracle_or_private_state_used=bool(findings),
    )
    events.append({
        "milestone": "M12_R1_PUBLIC_FACTOR_COUNTEREXAMPLES",
        "phase": "FACTOR_COUNTEREXAMPLE_ASSAY",
        "audit": audit.to_record(),
        "forbidden_source_findings": list(findings),
    })
    _write_jsonl(output_events, events)
    _write_json(output_counterexamples, {
        "counterexample_contexts": [
            context.to_record() for context in counterexample_contexts
        ],
    })
    _write_json(output_stability, {
        "audit": audit.to_record(),
        "promoted_stable_contexts": [
            context.to_record() for context in stable_contexts
        ],
    })
    summary = {
        "milestone": "M12_R1_PUBLIC_FACTOR_COUNTEREXAMPLES",
        "status": status,
        "contract_status": status,
        "trace_path": str(trace_path),
        "source_events": str(source_events),
        "transition_count": audit.transition_count,
        "context_count": audit.context_count,
        "reused_context_count": audit.reused_context_count,
        "stable_context_count": audit.stable_context_count,
        "contradicted_context_count": audit.contradicted_context_count,
        "unsupported_context_count": audit.unsupported_context_count,
        "promoted_context_count": audit.promoted_context_count,
        "blocked_promotion_count": audit.blocked_promotion_count,
        "counterexample_count": audit.counterexample_count,
        "no_op_contradiction_count": audit.no_op_contradiction_count,
        "forbidden_source_findings": list(findings),
        "known_limits": {
            "factor_key_is_coarse": True,
            "counterexamples_do_not_repair_representation": True,
            "not_terminal_causality": True,
            "not_full_object_discovery": True,
        },
        "artifacts": {
            "events": str(output_events),
            "counterexamples": str(output_counterexamples),
            "stability": str(output_stability),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 12.R1 Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Lab-book line:",
        "",
        "```text",
        "M12.R1:",
        "  I can use public interaction counterexamples to decide which factor contexts are reliable enough to promote and which show that my factor vocabulary is under-specified.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- transitions: `{summary['transition_count']}`",
        f"- contexts: `{summary['context_count']}`",
        f"- reused contexts: `{summary['reused_context_count']}`",
        f"- stable contexts: `{summary['stable_context_count']}`",
        f"- contradicted contexts: `{summary['contradicted_context_count']}`",
        f"- unsupported contexts: `{summary['unsupported_context_count']}`",
        f"- promoted contexts: `{summary['promoted_context_count']}`",
        f"- blocked promotions: `{summary['blocked_promotion_count']}`",
        f"- counterexamples: `{summary['counterexample_count']}`",
        f"- no-op contradictions: `{summary['no_op_contradiction_count']}`",
        f"- forbidden source findings: `{summary['forbidden_source_findings']}`",
        "",
        "Known limits:",
        "",
        "- This assay diagnoses coarse or missing factors; it does not repair them.",
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
    parser.add_argument("--counterexamples", type=Path, default=DEFAULT_COUNTEREXAMPLES)
    parser.add_argument("--stability", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    summary = evaluate(
        trace_path=args.trace,
        source_events=args.source_events,
        output_events=args.events,
        output_counterexamples=args.counterexamples,
        output_stability=args.stability,
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
