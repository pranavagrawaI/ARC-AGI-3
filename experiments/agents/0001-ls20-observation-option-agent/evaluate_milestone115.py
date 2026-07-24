"""M11.5 live graph-growth diagnostic for 0001.

This is not a new capability milestone. It embeds the M10.6 live replanner and
M11 frontier expansion machinery in a longer public LS20 run to find the next
integration bottleneck.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal


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
    LiveCandidateActuator,
    LiveCandidateGenerator,
    LiveObservationStateClassifier,
    LiveOptionInducer,
    ProbeOutcomeClassifier,
    SaturationDetector,
    SelectionPolicy,
    apply_graph_expansion,
    forbidden_frontier_findings,
    observation_hash,
)


DiagnosticStatus = Literal[
    "diagnostic_graph_growth_observed",
    "diagnostic_terminal_reached",
    "diagnostic_no_honest_frontier",
    "diagnostic_noise_promotion_risk",
    "diagnostic_local_probe_loop",
    "diagnostic_tracker_lost",
    "diagnostic_budget_exhausted",
    "diagnostic_no_graph_growth",
    "diagnostic_trace_or_oracle_leakage",
]


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-115-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-115-events.jsonl"
DEFAULT_GRAPH = ARTIFACT_DIR / "milestone-115-graph-growth-ledger.json"
DEFAULT_PROBES = ARTIFACT_DIR / "milestone-115-probe-ledger.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-115-freeze.md"


def evaluate(
    *,
    game: str,
    action_budget: int,
    max_cycles: int,
    output_events: Path,
    output_graph: Path,
    output_probes: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    context = _build_graph_context(game=game)
    graph = context["graph"]
    option_a = context["option_a"]
    option_b = context["option_b"]
    options = (option_a, option_b)
    action_plans: dict[str, tuple[int, ...]] = {
        option_a.option_id: tuple(context["heldout_a"].action_plan),
        option_b.option_id: tuple(context["heldout_b"].action_plan),
    }

    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(game)
    current = public_observation(env.reset())
    current_position = tuple(
        int(value) for value in option_a.initiation_condition["actor_position"]
    )
    current_position_source = "option_state_binding"

    classifier = LiveObservationStateClassifier()
    candidate_generator = LiveCandidateGenerator()
    selection_policy = SelectionPolicy(progress_model=_learned_progress_model())
    attempt_memory = AttemptMemory()
    option_actuator = LiveCandidateActuator()
    option_actuator.start(current)
    saturation_detector = SaturationDetector()
    frontier_detector = FrontierDetector()
    frontier_scorer = FrontierProbeScorer()
    outcome_classifier = ProbeOutcomeClassifier()
    inducer = LiveOptionInducer()

    events: list[dict[str, Any]] = []
    graph_ledger: list[dict[str, Any]] = []
    probe_ledger: list[dict[str, Any]] = []
    recent_states: list[str] = []
    recent_selected: list[str] = []
    stop_reason: str | None = None
    actions_used = 0
    probe_count = 0
    promoted_count = 0
    expanded_graph_used_count = 0
    post_expansion_state_rebindings = 0
    noise_promoted = False
    tracker_lost = False
    cycles_since_expansion = 0
    terminal_reached = False

    for cycle in range(max_cycles):
        binding = classifier.classify(
            observation_hash=observation_hash(current),
            graph=graph,
            options=options,
            controlled_position=current_position,
            controlled_position_source=current_position_source,
            semantic_guards_valid=True,
        )
        events.append({
            "milestone": "M11_5_LIVE_GRAPH_GROWTH_DIAGNOSTIC",
            "phase": "LIVE_STATE_CLASSIFICATION",
            "cycle": cycle,
            "binding": binding.to_record(),
        })
        if binding.classification == "CONTROLLED_ENTITY_LOST":
            tracker_lost = True
            stop_reason = "CONTROLLED_ENTITY_LOST"
            break
        if current.state in {"WIN", "GAME_OVER"} or current.levels_completed > 0:
            terminal_reached = True
            stop_reason = "TERMINAL_STATE"
            break
        if not binding.replannable:
            stop_reason = binding.classification
            break
        recent_states.append(binding.state_class_id or "UNKNOWN")

        candidates = candidate_generator.generate(
            graph=graph,
            binding=binding,
            options=options,
            action_plans=action_plans,
            max_depth=2,
        )
        saturation = saturation_detector.detect(
            recent_state_sequence=tuple(recent_states[-4:]),
            recent_selected_candidates=tuple(recent_selected[-4:]),
            candidate_scores_stalled=not candidates or cycles_since_expansion >= 2,
            no_new_state_class_cycles=cycles_since_expansion,
        )
        events.append({
            "milestone": "M11_5_LIVE_GRAPH_GROWTH_DIAGNOSTIC",
            "phase": "SATURATION_CHECK",
            "cycle": cycle,
            "candidate_count": len(candidates),
            "saturation": saturation.to_record(),
        })

        if saturation.saturation_detected or not candidates:
            probe_result = _run_frontier_probe(
                cycle=cycle,
                env=env,
                current=current,
                current_position=current_position,
                binding=binding,
                options=options,
                action_plans=action_plans,
                graph=graph,
                available_actions=tuple(current.available_actions),
                remaining_budget=action_budget - actions_used,
                step_offset=actions_used,
                frontier_detector=frontier_detector,
                frontier_scorer=frontier_scorer,
                outcome_classifier=outcome_classifier,
                inducer=inducer,
            )
            events.extend(probe_result["events"])
            probe_ledger.append(probe_result["ledger"])
            if probe_result["selected_probe_id"] is None:
                stop_reason = "NO_FRONTIER_CANDIDATE"
                break
            probe_count += 1
            actions_used += probe_result["actions_executed"]
            current = probe_result["current_observation"]
            current_position = probe_result["current_position"]
            current_position_source = probe_result["controlled_position_source"]
            if current_position is None:
                tracker_lost = True
                stop_reason = "CONTROLLED_ENTITY_LOST"
                break
            noise_promoted = noise_promoted or probe_result["noise_promoted"]
            expansion = probe_result["expansion"]
            if expansion is not None:
                graph = apply_graph_expansion(graph=graph, expansion=expansion)
                options = options + (expansion.option,)
                action_plans[expansion.option.option_id] = tuple(
                    probe_result["action_sequence"]
                )
                promoted_count += 1
                cycles_since_expansion = 0
                graph_ledger.append({
                    "cycle": cycle,
                    "graph_state_count": len(graph.state_classes),
                    "graph_edge_count": len(graph.edges),
                    "option_count": len(options),
                    "expansion": expansion.to_record(),
                })
                post_binding = classifier.classify(
                    observation_hash=observation_hash(current),
                    graph=graph,
                    options=options,
                    controlled_position=current_position,
                    controlled_position_source=current_position_source,
                    semantic_guards_valid=True,
                )
                post_candidates = candidate_generator.generate(
                    graph=graph,
                    binding=post_binding,
                    options=options,
                    action_plans=action_plans,
                    max_depth=2,
                )
                expansion_used = any(
                    expansion.new_option_id in candidate.option_ids
                    for candidate in post_candidates
                )
                expanded_graph_used_count += 1 if expansion_used else 0
                post_expansion_state_rebindings += (
                    1
                    if post_binding.state_class_id == expansion.new_state_class_id
                    else 0
                )
                events.append({
                    "milestone": "M11_5_LIVE_GRAPH_GROWTH_DIAGNOSTIC",
                    "phase": "POST_EXPANSION_CANDIDATE_GENERATION",
                    "cycle": cycle,
                    "binding": post_binding.to_record(),
                    "candidate_ids": [candidate.candidate_id for candidate in post_candidates],
                    "expanded_option_id": expansion.new_option_id,
                    "expanded_graph_used": expansion_used,
                })
            else:
                cycles_since_expansion += 1
            if actions_used >= action_budget:
                stop_reason = "ACTION_BUDGET_EXHAUSTED"
                break
            continue

        selection = selection_policy.select(
            round_index=cycle,
            state_class_id=binding.state_class_id or "",
            candidates=candidates,
            attempt_memory=attempt_memory,
        )
        events.append({
            "milestone": "M11_5_LIVE_GRAPH_GROWTH_DIAGNOSTIC",
            "phase": "LIVE_OPTION_SELECTION",
            "cycle": cycle,
            "ledger": selection.to_record(),
        })
        selected = _selected_candidate(selection)
        if selected is None:
            cycles_since_expansion += 1
            continue
        recent_selected.append(selected.candidate_id)
        result, current = option_actuator.execute_first_link(
            env=env,
            current_observation=current,
            candidate=selected,
            action_plans=action_plans,
            current_controlled_position=current_position,
            step_offset=actions_used,
            remaining_budget=action_budget - actions_used,
        )
        actions_used += result.actions_executed
        current_position = result.controlled_position
        current_position_source = result.controlled_position_source
        cycles_since_expansion += 1
        events.append({
            "milestone": "M11_5_LIVE_GRAPH_GROWTH_DIAGNOSTIC",
            "phase": "LIVE_OPTION_ACTUATION",
            "cycle": cycle,
            "actuation": result.to_record(),
        })
        if result.stop_reason:
            stop_reason = result.stop_reason
            break
        if actions_used >= action_budget:
            stop_reason = "ACTION_BUDGET_EXHAUSTED"
            break

    if stop_reason is None:
        stop_reason = "MAX_CYCLES"

    audit_payload = {
        "events": events,
        "graph_ledger": graph_ledger,
        "probe_ledger": _json_safe_probe_ledger(probe_ledger),
    }
    findings = forbidden_frontier_findings(audit_payload)
    status = _diagnostic_status(
        terminal_reached=terminal_reached,
        promoted_count=promoted_count,
        expanded_graph_used_count=expanded_graph_used_count,
        probe_count=probe_count,
        noise_promoted=noise_promoted,
        tracker_lost=tracker_lost,
        budget_exhausted=stop_reason == "ACTION_BUDGET_EXHAUSTED",
        trace_or_oracle_leakage=bool(findings),
    )
    graph_growth = [
        {
            "index": 0,
            "graph_state_count": len(context["graph"].state_classes),
            "graph_edge_count": len(context["graph"].edges),
            "option_count": 2,
        },
        *[
            {
                "index": index + 1,
                "graph_state_count": item["graph_state_count"],
                "graph_edge_count": item["graph_edge_count"],
                "option_count": item["option_count"],
                "new_option_id": item["expansion"]["new_option_id"],
            }
            for index, item in enumerate(graph_ledger)
        ],
    ]
    selected_probe_ids = [
        item.get("selected_probe_id")
        for item in probe_ledger
        if item.get("selected_probe_id") is not None
    ]
    repeated_probe_selection_count = max(
        (
            selected_probe_ids.count(probe_id)
            for probe_id in set(selected_probe_ids)
        ),
        default=0,
    )
    _write_jsonl(output_events, events)
    _write_json(output_graph, {
        "graph_growth": graph_growth,
        "expansions": [item["expansion"] for item in graph_ledger],
    })
    _write_json(output_probes, {
        "probe_count": probe_count,
        "probes": _json_safe_probe_ledger(probe_ledger),
    })
    summary = {
        "milestone": "M11_5_LIVE_GRAPH_GROWTH_DIAGNOSTIC",
        "status": status,
        "game": game,
        "action_budget": action_budget,
        "max_cycles": max_cycles,
        "live_actions_executed": actions_used,
        "cycles_attempted": len([event for event in events if event["phase"] == "LIVE_STATE_CLASSIFICATION"]),
        "frontier_probe_count": probe_count,
        "promoted_option_count": promoted_count,
        "expanded_graph_used_count": expanded_graph_used_count,
        "post_expansion_state_rebindings": post_expansion_state_rebindings,
        "initial_state_class_count": len(context["graph"].state_classes),
        "final_state_class_count": len(graph.state_classes),
        "initial_edge_count": len(context["graph"].edges),
        "final_edge_count": len(graph.edges),
        "initial_option_count": 2,
        "final_option_count": len(options),
        "noise_promoted_to_option": noise_promoted,
        "tracker_lost": tracker_lost,
        "unique_selected_probe_count": len(set(selected_probe_ids)),
        "repeated_probe_selection_count": repeated_probe_selection_count,
        "levels_completed": current.levels_completed,
        "final_state": current.state,
        "stop_reason": stop_reason,
        "forbidden_source_findings": list(findings),
        "artifacts": {
            "events": str(output_events),
            "graph_growth": str(output_graph),
            "probe_ledger": str(output_probes),
            "freeze": str(output_freeze),
        },
        "known_limits": {
            "diagnostic_not_milestone_claim": True,
            "not_full_ls20_solver": True,
            "frontier_growth_policy_is_local": True,
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _run_frontier_probe(
    *,
    cycle: int,
    env: Any,
    current: Any,
    current_position: tuple[int, int] | None,
    binding: Any,
    options: tuple[Any, ...],
    action_plans: dict[str, tuple[int, ...]],
    graph: Any,
    available_actions: tuple[int, ...],
    remaining_budget: int,
    step_offset: int,
    frontier_detector: FrontierDetector,
    frontier_scorer: FrontierProbeScorer,
    outcome_classifier: ProbeOutcomeClassifier,
    inducer: LiveOptionInducer,
) -> dict[str, Any]:
    known_sequences = tuple(action_plans[option_id][:2] for option_id in sorted(action_plans))
    probes = frontier_detector.generate(
        binding=binding,
        available_actions=available_actions,
        known_action_sequences=known_sequences,
    )
    scored = frontier_scorer.select(probes)
    selected = _selected_probe(scored)
    events: list[dict[str, Any]] = [{
        "milestone": "M11_5_LIVE_GRAPH_GROWTH_DIAGNOSTIC",
        "phase": "FRONTIER_PROBE_SELECTION",
        "cycle": cycle,
        "candidate_count": len(scored),
        "selected_probe_id": selected.probe_id if selected else None,
        "candidates": [probe.to_record() for probe in scored],
    }]
    if selected is None or remaining_budget <= 0:
        return {
            "events": events,
            "ledger": {
                "cycle": cycle,
                "candidates": [probe.to_record() for probe in scored],
                "selected_probe_id": None,
            },
            "selected_probe_id": None,
            "actions_executed": 0,
            "current_observation": current,
            "current_position": current_position,
            "controlled_position_source": binding.controlled_position_source,
            "noise_promoted": False,
            "expansion": None,
            "action_sequence": (),
        }
    executor = FrontierProbeExecutor()
    executor.start(current)
    execution, following = executor.execute(
        env=env,
        current_observation=current,
        candidate=selected,
        current_controlled_position=current_position,
        step_offset=step_offset,
        remaining_budget=remaining_budget,
    )
    outcome = outcome_classifier.classify(
        execution=execution,
        known_options=options,
    )
    expansion = None
    source_state = graph.state(binding.state_class_id or "")
    if outcome.promotable and source_state is not None and execution.start_position is not None:
        expansion = inducer.promote(
            expansion_index=len(graph.state_classes) - 3,
            source_state=source_state,
            source_position=execution.start_position,
            outcome=outcome,
            execution=execution,
        )
    events.extend([
        {
            "milestone": "M11_5_LIVE_GRAPH_GROWTH_DIAGNOSTIC",
            "phase": "FRONTIER_PROBE_EXECUTION",
            "cycle": cycle,
            "execution": execution.to_record(),
        },
        {
            "milestone": "M11_5_LIVE_GRAPH_GROWTH_DIAGNOSTIC",
            "phase": "PROBE_OUTCOME_CLASSIFICATION",
            "cycle": cycle,
            "outcome": outcome.to_record(),
        },
    ])
    if expansion is not None:
        events.append({
            "milestone": "M11_5_LIVE_GRAPH_GROWTH_DIAGNOSTIC",
            "phase": "GRAPH_EXPANSION",
            "cycle": cycle,
            "expansion": expansion.to_record(),
        })
    return {
        "events": events,
        "ledger": {
            "cycle": cycle,
            "candidates": [probe.to_record() for probe in scored],
            "selected_probe_id": selected.probe_id,
            "execution": execution.to_record(),
            "outcome": outcome.to_record(),
            "expansion": expansion.to_record() if expansion else None,
        },
        "selected_probe_id": selected.probe_id,
        "actions_executed": execution.actions_executed,
        "current_observation": following,
        "current_position": execution.final_position,
        "controlled_position_source": execution.controlled_position_source,
        "noise_promoted": bool(outcome.result == "NOISE_ONLY" and outcome.promotable),
        "expansion": expansion,
        "action_sequence": selected.action_sequence,
    }


def _diagnostic_status(
    *,
    terminal_reached: bool,
    promoted_count: int,
    expanded_graph_used_count: int,
    probe_count: int,
    noise_promoted: bool,
    tracker_lost: bool,
    budget_exhausted: bool,
    trace_or_oracle_leakage: bool,
) -> DiagnosticStatus:
    if trace_or_oracle_leakage:
        return "diagnostic_trace_or_oracle_leakage"
    if noise_promoted:
        return "diagnostic_noise_promotion_risk"
    if tracker_lost:
        return "diagnostic_tracker_lost"
    if terminal_reached:
        return "diagnostic_terminal_reached"
    if promoted_count and expanded_graph_used_count:
        return "diagnostic_graph_growth_observed"
    if probe_count and not promoted_count:
        return "diagnostic_local_probe_loop"
    if budget_exhausted:
        return "diagnostic_budget_exhausted"
    if not probe_count:
        return "diagnostic_no_honest_frontier"
    return "diagnostic_no_graph_growth"


def _selected_candidate(selection) -> Any | None:
    for candidate in selection.candidates:
        if candidate.selected:
            return candidate
    return None


def _selected_probe(candidates) -> Any | None:
    for candidate in candidates:
        if candidate.selected:
            return candidate
    return None


def _json_safe_probe_ledger(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(records, default=str))


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
        "# M11.5 Live Graph-Growth Diagnostic",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Diagnostic question:",
        "",
        "```text",
        "If M10.6 + M11 run continuously in public LS20, does the graph grow,",
        "or does the loop expose a new bottleneck?",
        "```",
        "",
        "Key evidence:",
        "",
        f"- live actions executed: `{summary['live_actions_executed']}`",
        f"- cycles attempted: `{summary['cycles_attempted']}`",
        f"- frontier probes: `{summary['frontier_probe_count']}`",
        f"- promoted options: `{summary['promoted_option_count']}`",
        f"- expanded graph used count: `{summary['expanded_graph_used_count']}`",
        f"- post-expansion state rebindings: `{summary['post_expansion_state_rebindings']}`",
        f"- graph states: `{summary['initial_state_class_count']} → {summary['final_state_class_count']}`",
        f"- graph edges: `{summary['initial_edge_count']} → {summary['final_edge_count']}`",
        f"- options: `{summary['initial_option_count']} → {summary['final_option_count']}`",
        f"- noise promoted: `{summary['noise_promoted_to_option']}`",
        f"- unique selected probes: `{summary['unique_selected_probe_count']}`",
        f"- repeated probe selection count: `{summary['repeated_probe_selection_count']}`",
        f"- stop reason: `{summary['stop_reason']}`",
        f"- levels completed: `{summary['levels_completed']}`",
        f"- forbidden source findings: `{summary['forbidden_source_findings']}`",
        "",
        "Known limits:",
        "",
        "- This is a diagnostic, not a milestone capability freeze.",
        "- This is not a full LS20 solver.",
        "- Frontier growth remains local and spatial.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--action-budget", type=int, default=500)
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    summary = evaluate(
        game=args.game,
        action_budget=args.action_budget,
        max_cycles=args.max_cycles,
        output_events=args.events,
        output_graph=args.graph,
        output_probes=args.probes,
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
