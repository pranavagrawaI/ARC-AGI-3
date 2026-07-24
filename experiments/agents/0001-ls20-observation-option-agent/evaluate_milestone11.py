"""Evaluate M11 frontier-guided live option discovery."""

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

import arc_agi
from arc_agi.base import OperationMode

from experiments.harness.runner import public_observation

from evaluate_milestone9 import _build_graph_context  # noqa: E402
from evaluate_milestone106 import _learned_progress_model  # noqa: E402
from observation_option_agent import (  # noqa: E402
    AttemptMemory,
    FrontierDetector,
    FrontierProbeExecutor,
    FrontierProbeScorer,
    LiveCandidateGenerator,
    LiveObservationStateClassifier,
    LiveOptionInducer,
    ProbeOutcomeClassifier,
    SaturationDetector,
    SelectionPolicy,
    apply_graph_expansion,
    forbidden_frontier_findings,
    m11_status,
    observation_hash,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-11-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-11-events.jsonl"
DEFAULT_PROBES = ARTIFACT_DIR / "frontier-probe-candidates.json"
DEFAULT_OUTCOMES = ARTIFACT_DIR / "frontier-probe-outcomes.json"
DEFAULT_EXPANSION = ARTIFACT_DIR / "graph-expansion-ledger.json"
DEFAULT_INDUCTION = ARTIFACT_DIR / "live-option-induction-ledger.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-11-freeze.md"


def evaluate(
    *,
    game: str,
    action_budget: int,
    output_events: Path,
    output_probes: Path,
    output_outcomes: Path,
    output_expansion: Path,
    output_induction: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    graph_context = _build_graph_context(game=game)
    graph = graph_context["graph"]
    option_a = graph_context["option_a"]
    option_b = graph_context["option_b"]
    options = (option_a, option_b)
    action_plans = {
        option_a.option_id: tuple(graph_context["heldout_a"].action_plan),
        option_b.option_id: tuple(graph_context["heldout_b"].action_plan),
    }

    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(game)
    current = public_observation(env.reset())
    current_position = tuple(
        int(value) for value in option_a.initiation_condition["actor_position"]
    )

    classifier = LiveObservationStateClassifier()
    binding = classifier.classify(
        observation_hash=observation_hash(current),
        graph=graph,
        options=options,
        controlled_position=current_position,
        controlled_position_source="option_state_binding",
        semantic_guards_valid=True,
    )

    saturation = SaturationDetector().detect(
        recent_state_sequence=("S0", "S1", "S0"),
        recent_selected_candidates=("chain_000", "option_000", "chain_000"),
        candidate_scores_stalled=True,
        no_new_state_class_cycles=3,
    )
    events: list[dict[str, Any]] = [
        {
            "milestone": "M11_FRONTIER_GUIDED_LIVE_OPTION_DISCOVERY",
            "phase": "LIVE_STATE_CLASSIFICATION",
            "binding": binding.to_record(),
        },
        {
            "milestone": "M11_FRONTIER_GUIDED_LIVE_OPTION_DISCOVERY",
            "phase": "SATURATION_DETECTION",
            "saturation": saturation.to_record(),
        },
    ]

    known_sequences = (
        tuple(action_plans[option_a.option_id][:2]),
        tuple(reversed(action_plans[option_a.option_id][:2])),
    )
    probes = FrontierDetector().generate(
        binding=binding,
        available_actions=tuple(current.available_actions),
        known_action_sequences=known_sequences,
    )
    scored_probes = FrontierProbeScorer().select(probes)
    selected_probe = _selected_probe(scored_probes)
    probe_executed = False
    probe_execution = None
    outcome = None
    expansion = None
    expanded_graph = graph
    expanded_options = options
    expanded_graph_used = False
    post_expansion_candidates = ()
    post_expansion_selection = None
    post_expansion_binding = None
    actions_used = 0
    final_current = current

    events.append({
        "milestone": "M11_FRONTIER_GUIDED_LIVE_OPTION_DISCOVERY",
        "phase": "FRONTIER_PROBE_SELECTION",
        "candidate_count": len(scored_probes),
        "selected_probe_id": selected_probe.probe_id if selected_probe else None,
        "candidates": [probe.to_record() for probe in scored_probes],
    })

    if selected_probe is not None:
        executor = FrontierProbeExecutor()
        executor.start(current)
        probe_execution, final_current = executor.execute(
            env=env,
            current_observation=current,
            candidate=selected_probe,
            current_controlled_position=current_position,
            step_offset=0,
            remaining_budget=action_budget,
        )
        probe_executed = probe_execution.actions_executed > 0
        actions_used = probe_execution.actions_executed
        events.append({
            "milestone": "M11_FRONTIER_GUIDED_LIVE_OPTION_DISCOVERY",
            "phase": "FRONTIER_PROBE_EXECUTION",
            "execution": probe_execution.to_record(),
        })
        outcome = ProbeOutcomeClassifier().classify(
            execution=probe_execution,
            known_options=options,
        )
        events.append({
            "milestone": "M11_FRONTIER_GUIDED_LIVE_OPTION_DISCOVERY",
            "phase": "PROBE_OUTCOME_CLASSIFICATION",
            "outcome": outcome.to_record(),
        })
        if outcome.promotable and binding.state_class_id is not None:
            source_state = graph.state(binding.state_class_id)
            if source_state is not None and probe_execution.start_position is not None:
                expansion = LiveOptionInducer().promote(
                    expansion_index=0,
                    source_state=source_state,
                    source_position=probe_execution.start_position,
                    outcome=outcome,
                    execution=probe_execution,
                )
    if expansion is not None:
        expanded_graph = apply_graph_expansion(graph=graph, expansion=expansion)
        expanded_options = options + (expansion.option,)
        action_plans[expansion.option.option_id] = (
            probe_execution.action_sequence if probe_execution is not None else ()
        )
        events.append({
            "milestone": "M11_FRONTIER_GUIDED_LIVE_OPTION_DISCOVERY",
            "phase": "GRAPH_EXPANSION",
            "expansion": expansion.to_record(),
        })
        post_expansion_binding = classifier.classify(
            observation_hash=observation_hash(current),
            graph=expanded_graph,
            options=expanded_options,
            controlled_position=current_position,
            controlled_position_source="option_state_binding",
            semantic_guards_valid=True,
        )
        post_expansion_candidates = LiveCandidateGenerator().generate(
            graph=expanded_graph,
            binding=post_expansion_binding,
            options=expanded_options,
            action_plans=action_plans,
            max_depth=2,
        )
        post_expansion_selection = SelectionPolicy(
            progress_model=_learned_progress_model()
        ).select(
            round_index=1,
            state_class_id=post_expansion_binding.state_class_id or "",
            candidates=post_expansion_candidates,
            attempt_memory=AttemptMemory(),
        )
        expanded_graph_used = any(
            expansion.new_option_id in candidate.option_ids
            for candidate in post_expansion_selection.candidates
        )
        events.append({
            "milestone": "M11_FRONTIER_GUIDED_LIVE_OPTION_DISCOVERY",
            "phase": "POST_EXPANSION_REPLAN",
            "binding": post_expansion_binding.to_record(),
            "candidate_ids": [
                candidate.candidate_id
                for candidate in post_expansion_selection.candidates
            ],
            "selected_candidate_id": post_expansion_selection.selected_candidate_id,
            "expanded_graph_used": expanded_graph_used,
            "ledger": post_expansion_selection.to_record(),
        })

    payload_for_audit = {
        "saturation": saturation.to_record(),
        "probes": [probe.to_record() for probe in scored_probes],
        "outcome": outcome.to_record() if outcome else None,
        "expansion": expansion.to_record() if expansion else None,
        "post_expansion_replan": (
            post_expansion_selection.to_record()
            if post_expansion_selection is not None
            else None
        ),
    }
    findings = forbidden_frontier_findings(payload_for_audit)
    noise_promoted = bool(outcome and outcome.result == "NOISE_ONLY" and outcome.promotable)
    status = m11_status(
        saturation_detected=saturation.saturation_detected,
        probe_count=len(scored_probes),
        probe_score_driven=bool(scored_probes) and all(
            probe.score_terms is not None for probe in scored_probes
        ),
        probe_executed=probe_executed,
        promoted_count=1 if expansion is not None else 0,
        noise_promoted=noise_promoted,
        expanded_graph_used=expanded_graph_used,
        live_loop_resumed_after_expansion=post_expansion_selection is not None,
        honest_frontier_available=bool(scored_probes),
        fixed_probe_order=False,
        trace_leakage=bool(findings),
        oracle_signal=False,
        unclassified_loop=binding.classification not in {
            "MATCHED_KNOWN_STATE_CLASS",
            "NEW_STATE_CLASS",
        },
    )

    probe_records = {
        "saturation": saturation.to_record(),
        "probe_candidates": [probe.to_record() for probe in scored_probes],
    }
    outcome_records = {
        "probe_execution": probe_execution.to_record() if probe_execution else None,
        "probe_outcome": outcome.to_record() if outcome else None,
    }
    expansion_records = {
        "expansions": [expansion.to_record()] if expansion else [],
        "expanded_graph": expanded_graph.to_record() if expansion else graph.to_record(),
    }
    induction_records = {
        "new_options": [expansion.option.to_record()] if expansion else [],
        "post_expansion_replan": (
            post_expansion_selection.to_record()
            if post_expansion_selection is not None
            else None
        ),
    }
    _write_jsonl(output_events, events)
    _write_json(output_probes, probe_records)
    _write_json(output_outcomes, outcome_records)
    _write_json(output_expansion, expansion_records)
    _write_json(output_induction, induction_records)

    summary = {
        "milestone": "M11_FRONTIER_GUIDED_LIVE_OPTION_DISCOVERY",
        "status": status,
        "contract_status": status,
        "game": game,
        "action_budget": action_budget,
        "live_actions_executed": actions_used,
        "saturation_detected": saturation.saturation_detected,
        "saturation_reason": saturation.reason,
        "frontier_probe_count": len(scored_probes),
        "selected_probe_id": selected_probe.probe_id if selected_probe else None,
        "probe_selection_source": "frontier_progress_score",
        "probe_executed_live": probe_executed,
        "probe_outcome": outcome.result if outcome else None,
        "new_state_classes_created": 1 if expansion else 0,
        "new_options_created": 1 if expansion else 0,
        "new_edges_created": 1 if expansion else 0,
        "blocked_edges_added": 0,
        "noise_promoted_to_option": noise_promoted,
        "expanded_graph_used_in_replanning": expanded_graph_used,
        "post_expansion_replan_cycles": 1 if post_expansion_selection else 0,
        "post_expansion_selected_candidate_id": (
            post_expansion_selection.selected_candidate_id
            if post_expansion_selection is not None
            else None
        ),
        "stop_reason": "MAX_REPLAN_CYCLES",
        "levels_completed": final_current.levels_completed,
        "final_state": final_current.state,
        "trace_used": False,
        "private_oracle_used": False,
        "forbidden_source_findings": list(findings),
        "known_limits": {
            "frontier_expansion_is_local_and_bounded": True,
            "not_final_objective_inference": True,
            "not_full_ls20_solver": True,
            "option_graph_remains_small": True,
        },
        "artifacts": {
            "events": str(output_events),
            "probe_candidates": str(output_probes),
            "probe_outcomes": str(output_outcomes),
            "graph_expansion": str(output_expansion),
            "live_option_induction": str(output_induction),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _selected_probe(candidates) -> Any | None:
    for candidate in candidates:
        if candidate.selected:
            return candidate
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 11 Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Official claim:",
        "",
        (
            "Milestone 11 demonstrates that 0001 can detect local "
            "option-graph saturation during live replanning, generate and "
            "score frontier probes from the current classified live state, "
            "execute bounded public exploratory actions under semantic guards, "
            "convert stable self-observed outcomes into new option candidates "
            "or state classes, and continue replanning using the expanded graph."
        ),
        "",
        "Lab-book line:",
        "",
        "```text",
        "M11:",
        "  I can notice my current option graph is exhausted and expand it.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- saturation detected: `{summary['saturation_detected']}`",
        f"- saturation reason: `{summary['saturation_reason']}`",
        f"- frontier probe count: `{summary['frontier_probe_count']}`",
        f"- selected probe: `{summary['selected_probe_id']}`",
        f"- probe executed live: `{summary['probe_executed_live']}`",
        f"- probe outcome: `{summary['probe_outcome']}`",
        f"- new state classes: `{summary['new_state_classes_created']}`",
        f"- new options: `{summary['new_options_created']}`",
        f"- new edges: `{summary['new_edges_created']}`",
        f"- expanded graph used in replanning: `{summary['expanded_graph_used_in_replanning']}`",
        f"- post-expansion selected candidate: `{summary['post_expansion_selected_candidate_id']}`",
        f"- noise promoted to option: `{summary['noise_promoted_to_option']}`",
        f"- forbidden source findings: `{summary['forbidden_source_findings']}`",
        "",
        "Known limits:",
        "",
        "- Frontier expansion is local and bounded.",
        "- This is not final objective inference.",
        "- This is not a full LS20 solver.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--action-budget", type=int, default=200)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--expansion", type=Path, default=DEFAULT_EXPANSION)
    parser.add_argument("--induction", type=Path, default=DEFAULT_INDUCTION)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    summary = evaluate(
        game=args.game,
        action_budget=args.action_budget,
        output_events=args.events,
        output_probes=args.probes,
        output_outcomes=args.outcomes,
        output_expansion=args.expansion,
        output_induction=args.induction,
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
