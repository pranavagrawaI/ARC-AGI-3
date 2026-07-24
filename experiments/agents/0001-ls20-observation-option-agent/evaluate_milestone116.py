"""Evaluate M11.6 level-transition context binding."""

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
    BoundaryProbeSlicer,
    FrontierDetector,
    FrontierProbeExecutor,
    FrontierProbeScorer,
    LevelContextManager,
    LevelTransitionDetector,
    LiveCandidateActuator,
    LiveCandidateGenerator,
    LiveObservationStateClassifier,
    LiveOptionInducer,
    ProbeOutcomeClassifier,
    SaturationDetector,
    SelectionPolicy,
    apply_graph_expansion,
    forbidden_level_transition_findings,
    m116_status,
    observation_hash,
    terminal_evidence_from_transition,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-116-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-116-events.jsonl"
DEFAULT_TRANSITIONS = ARTIFACT_DIR / "level-transition-events.json"
DEFAULT_CONTEXTS = ARTIFACT_DIR / "level-context-ledger.json"
DEFAULT_SLICING = ARTIFACT_DIR / "transition-slicing-ledger.json"
DEFAULT_SEEDING = ARTIFACT_DIR / "post-transition-seeding-ledger.json"
DEFAULT_TERMINAL = ARTIFACT_DIR / "terminal-evidence-ledger.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-116-freeze.md"


def evaluate(
    *,
    game: str,
    action_budget: int,
    max_cycles: int,
    output_events: Path,
    output_transitions: Path,
    output_contexts: Path,
    output_slicing: Path,
    output_seeding: Path,
    output_terminal: Path,
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
    option_actuator = LiveCandidateActuator()
    option_actuator.start(current)
    probe_executor = FrontierProbeExecutor()
    probe_executor.start(current)
    saturation_detector = SaturationDetector()
    frontier_detector = FrontierDetector()
    frontier_scorer = FrontierProbeScorer()
    outcome_classifier = ProbeOutcomeClassifier()
    inducer = LiveOptionInducer()
    transition_detector = LevelTransitionDetector()
    slicer = BoundaryProbeSlicer()

    events: list[dict[str, Any]] = []
    transition_records: list[dict[str, Any]] = []
    context_records: list[dict[str, Any]] = []
    slicing_records: list[dict[str, Any]] = []
    seeding_records: list[dict[str, Any]] = []
    terminal_records: list[dict[str, Any]] = []
    recent_states: list[str] = []
    recent_selected: list[str] = []
    actions_used = 0
    cycles_since_expansion = 0
    transition = None
    boundary_slice = None
    level_context = None
    terminal_evidence = None
    pre_terminal_state_class = None
    post_transition_binding = None
    post_transition_probe_count = 0
    post_transition_actions_executed = 0
    continued_after_transition = False
    cross_level_delta_promoted = False
    stop_reason = "MAX_CYCLES"

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
            "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
            "phase": "LIVE_STATE_CLASSIFICATION",
            "cycle": cycle,
            "binding": binding.to_record(),
        })
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
        if saturation.saturation_detected or not candidates:
            probe_result = _run_probe(
                cycle=cycle,
                env=env,
                current=current,
                current_position=current_position,
                current_position_source=current_position_source,
                binding=binding,
                options=options,
                action_plans=action_plans,
                graph=graph,
                available_actions=tuple(current.available_actions),
                remaining_budget=action_budget - actions_used,
                step_offset=actions_used,
                probe_executor=probe_executor,
                frontier_detector=frontier_detector,
                frontier_scorer=frontier_scorer,
                outcome_classifier=outcome_classifier,
                inducer=inducer,
                transition_detector=transition_detector,
            )
            events.extend(probe_result["events"])
            actions_used += probe_result["actions_executed"]
            current = probe_result["current_observation"]
            current_position = probe_result["current_position"]
            current_position_source = probe_result["controlled_position_source"]
            if probe_result["transition"] is not None:
                transition = probe_result["transition"]
                pre_terminal_state_class = binding.state_class_id
                boundary_slice = slicer.slice(
                    execution=probe_result["execution"],
                    transition=transition,
                )
                terminal_evidence = terminal_evidence_from_transition(
                    transition=transition,
                    pre_terminal_state_class=pre_terminal_state_class,
                )
                level_context = LevelContextManager().create_context(
                    level_index=transition.to_level_index,
                    observation_hash=observation_hash(current),
                )
                post_transition_binding = classifier.classify(
                    observation_hash=observation_hash(current),
                    graph=level_context.graph,
                    options=(),
                    controlled_position=current_position,
                    controlled_position_source=current_position_source,
                    semantic_guards_valid=True,
                )
                post_probes = frontier_detector.generate(
                    binding=post_transition_binding,
                    available_actions=tuple(current.available_actions),
                    known_action_sequences=(
                        tuple(boundary_slice.post_transition_actions) or (1, 4),
                        (4, 1),
                        (3, 1),
                    ),
                )
                scored_post_probes = frontier_scorer.select(post_probes)
                selected_post_probe = _selected_probe(scored_post_probes)
                post_transition_probe_count = len(scored_post_probes)
                transition_records.append(transition.to_record())
                slicing_records.append(boundary_slice.to_record())
                terminal_records.append(terminal_evidence.to_record())
                context_records.append(level_context.to_record())
                seeding_records.append({
                    "phase": "POST_TRANSITION_GRAPH_SEEDING",
                    "level_index": transition.to_level_index,
                    "new_context_id": level_context.context_id,
                    "initial_state_class_id": level_context.initial_state_class_id,
                    "post_transition_state_bound": post_transition_binding.replannable,
                    "binding": post_transition_binding.to_record(),
                    "seeded_probe_count": post_transition_probe_count,
                    "seeded_probes": [
                        probe.to_record() for probe in scored_post_probes
                    ],
                    "carried_transferables": list(level_context.carried_transferables),
                    "invalidated_level_local_structures": list(
                        level_context.invalidated_level_local
                    ),
                })
                events.extend([
                    {
                        "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
                        "phase": "LEVEL_TRANSITION_DETECTED",
                        "cycle": cycle,
                        "transition": transition.to_record(),
                    },
                    {
                        "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
                        "phase": "BOUNDARY_PROBE_SLICING",
                        "cycle": cycle,
                        "slice": boundary_slice.to_record(),
                    },
                    {
                        "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
                        "phase": "NEW_LEVEL_CONTEXT_CREATED",
                        "cycle": cycle,
                        "context": level_context.to_record(),
                    },
                    {
                        "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
                        "phase": "POST_TRANSITION_GRAPH_SEEDING",
                        "cycle": cycle,
                        "binding": post_transition_binding.to_record(),
                        "seeded_probe_count": post_transition_probe_count,
                        "seeded_probes": [probe.to_record() for probe in scored_post_probes],
                    },
                ])
                if selected_post_probe is not None and action_budget - actions_used > 0:
                    execution, current = probe_executor.execute(
                        env=env,
                        current_observation=current,
                        candidate=selected_post_probe,
                        current_controlled_position=current_position,
                        step_offset=actions_used,
                        remaining_budget=action_budget - actions_used,
                    )
                    actions_used += execution.actions_executed
                    current_position = execution.final_position
                    current_position_source = execution.controlled_position_source
                    post_transition_actions_executed = execution.actions_executed
                    continued_after_transition = execution.actions_executed > 0
                    events.append({
                        "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
                        "phase": "POST_TRANSITION_PROBE_EXECUTION",
                        "cycle": cycle,
                        "selected_probe_id": selected_post_probe.probe_id,
                        "execution": execution.to_record(),
                    })
                stop_reason = "MAX_REPLAN_CYCLES"
                break
            expansion = probe_result["expansion"]
            if expansion is not None:
                graph = apply_graph_expansion(graph=graph, expansion=expansion)
                options = options + (expansion.option,)
                action_plans[expansion.option.option_id] = tuple(
                    probe_result["action_sequence"]
                )
                cycles_since_expansion = 0
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
            attempt_memory=AttemptMemory(),
        )
        selected = _selected_candidate(selection)
        events.append({
            "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
            "phase": "LIVE_OPTION_SELECTION",
            "cycle": cycle,
            "ledger": selection.to_record(),
        })
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
            "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
            "phase": "LIVE_OPTION_ACTUATION",
            "cycle": cycle,
            "actuation": result.to_record(),
        })
        if actions_used >= action_budget:
            stop_reason = "ACTION_BUDGET_EXHAUSTED"
            break

    audit_payload = {
        "events": events,
        "transitions": transition_records,
        "contexts": context_records,
        "slicing": slicing_records,
        "terminal": terminal_records,
        "seeding": seeding_records,
    }
    findings = forbidden_level_transition_findings(audit_payload)
    transferables = tuple(level_context.carried_transferables) if level_context else ()
    invalidated = tuple(level_context.invalidated_level_local) if level_context else ()
    status = m116_status(
        transition_detected=transition is not None,
        probe_split=boundary_slice is not None,
        cross_level_delta_promoted=cross_level_delta_promoted,
        new_context_created=level_context is not None,
        transferables_carried=bool(transferables),
        stale_spatial_invalidated=bool(invalidated),
        post_transition_state_bound=bool(
            post_transition_binding and post_transition_binding.replannable
        ),
        post_transition_probe_count=post_transition_probe_count,
        post_transition_actions_executed=post_transition_actions_executed,
        trace_leakage=bool(findings),
        oracle_signal=False,
    )

    _write_jsonl(output_events, events)
    _write_json(output_transitions, {"transitions": transition_records})
    _write_json(output_contexts, {"contexts": context_records})
    _write_json(output_slicing, {"slices": slicing_records})
    _write_json(output_seeding, {"seedings": seeding_records})
    _write_json(output_terminal, {"terminal_evidence": terminal_records})
    summary = {
        "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
        "status": status,
        "contract_status": status,
        "game": game,
        "action_budget": action_budget,
        "max_cycles": max_cycles,
        "level_transition_detected": transition is not None,
        "transition_step": transition.public_step if transition else None,
        "transition_trigger_action": transition.trigger_action if transition else None,
        "from_level_index": transition.from_level_index if transition else None,
        "to_level_index": transition.to_level_index if transition else None,
        "boundary_crossing_probe_detected": boundary_slice is not None,
        "probe_split_at_transition": boundary_slice is not None,
        "cross_level_delta_promoted": cross_level_delta_promoted,
        "new_level_context_created": level_context is not None,
        "post_transition_state_bound": bool(
            post_transition_binding and post_transition_binding.replannable
        ),
        "post_transition_probe_count": post_transition_probe_count,
        "post_transition_actions_executed": post_transition_actions_executed,
        "transferable_action_model_carried": "action_delta_model" in transferables,
        "transferable_tracker_carried": "live_public_event_tracker" in transferables,
        "progress_weights_carried": "progress_weights" in transferables,
        "stale_spatial_options_invalidated": "previous_level_spatial_options" in invalidated,
        "continued_after_transition": continued_after_transition,
        "levels_completed": current.levels_completed,
        "final_state": current.state,
        "stop_reason": stop_reason,
        "forbidden_source_findings": list(findings),
        "known_limits": {
            "level_context_binding_only": True,
            "not_full_ls20_solver": True,
            "frontier_option_reuse_deferred": True,
        },
        "artifacts": {
            "events": str(output_events),
            "transitions": str(output_transitions),
            "contexts": str(output_contexts),
            "slicing": str(output_slicing),
            "seeding": str(output_seeding),
            "terminal": str(output_terminal),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _run_probe(
    *,
    cycle: int,
    env: Any,
    current: Any,
    current_position: tuple[int, int] | None,
    current_position_source: str,
    binding: Any,
    options: tuple[Any, ...],
    action_plans: dict[str, tuple[int, ...]],
    graph: Any,
    available_actions: tuple[int, ...],
    remaining_budget: int,
    step_offset: int,
    probe_executor: FrontierProbeExecutor,
    frontier_detector: FrontierDetector,
    frontier_scorer: FrontierProbeScorer,
    outcome_classifier: ProbeOutcomeClassifier,
    inducer: LiveOptionInducer,
    transition_detector: LevelTransitionDetector,
) -> dict[str, Any]:
    probes = frontier_detector.generate(
        binding=binding,
        available_actions=available_actions,
        known_action_sequences=tuple(
            action_plans[option_id][:2] for option_id in sorted(action_plans)
        ),
    )
    scored = frontier_scorer.select(probes)
    selected = _selected_probe(scored)
    events = [{
        "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
        "phase": "FRONTIER_PROBE_SELECTION",
        "cycle": cycle,
        "candidate_count": len(scored),
        "selected_probe_id": selected.probe_id if selected else None,
        "candidates": [probe.to_record() for probe in scored],
    }]
    if selected is None:
        return {
            "events": events,
            "actions_executed": 0,
            "current_observation": current,
            "current_position": current_position,
            "controlled_position_source": current_position_source,
            "transition": None,
            "execution": None,
            "expansion": None,
            "action_sequence": (),
        }
    execution, following = probe_executor.execute(
        env=env,
        current_observation=current,
        candidate=selected,
        current_controlled_position=current_position,
        step_offset=step_offset,
        remaining_budget=remaining_budget,
    )
    transition = transition_detector.detect(execution=execution)
    events.append({
        "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
        "phase": "FRONTIER_PROBE_EXECUTION",
        "cycle": cycle,
        "execution": execution.to_record(),
    })
    if transition is not None:
        return {
            "events": events,
            "actions_executed": execution.actions_executed,
            "current_observation": following,
            "current_position": execution.final_position,
            "controlled_position_source": execution.controlled_position_source,
            "transition": transition,
            "execution": execution,
            "expansion": None,
            "action_sequence": selected.action_sequence,
        }
    outcome = outcome_classifier.classify(execution=execution, known_options=options)
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
    events.append({
        "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
        "phase": "PROBE_OUTCOME_CLASSIFICATION",
        "cycle": cycle,
        "outcome": outcome.to_record(),
    })
    if expansion is not None:
        events.append({
            "milestone": "M11_6_LEVEL_TRANSITION_CONTEXT_BINDING",
            "phase": "GRAPH_EXPANSION",
            "cycle": cycle,
            "expansion": expansion.to_record(),
        })
    return {
        "events": events,
        "actions_executed": execution.actions_executed,
        "current_observation": following,
        "current_position": execution.final_position,
        "controlled_position_source": execution.controlled_position_source,
        "transition": None,
        "execution": execution,
        "expansion": expansion,
        "action_sequence": selected.action_sequence,
    }


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
        "# Milestone 11.6 Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Lab-book line:",
        "",
        "```text",
        "M11.6:",
        "  I know that a new level is a new context, not just a new state.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- transition detected: `{summary['level_transition_detected']}`",
        f"- transition step: `{summary['transition_step']}`",
        f"- trigger action: `{summary['transition_trigger_action']}`",
        f"- level index: `{summary['from_level_index']} → {summary['to_level_index']}`",
        f"- boundary probe split: `{summary['probe_split_at_transition']}`",
        f"- cross-level delta promoted: `{summary['cross_level_delta_promoted']}`",
        f"- new level context created: `{summary['new_level_context_created']}`",
        f"- post-transition state bound: `{summary['post_transition_state_bound']}`",
        f"- post-transition probe count: `{summary['post_transition_probe_count']}`",
        f"- post-transition actions executed: `{summary['post_transition_actions_executed']}`",
        f"- continued after transition: `{summary['continued_after_transition']}`",
        f"- stale spatial options invalidated: `{summary['stale_spatial_options_invalidated']}`",
        f"- forbidden source findings: `{summary['forbidden_source_findings']}`",
        "",
        "Known limits:",
        "",
        "- This binds and seeds a new level context; frontier-option reuse is deferred.",
        "- This is not a full LS20 solver.",
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
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    parser.add_argument("--slicing", type=Path, default=DEFAULT_SLICING)
    parser.add_argument("--seeding", type=Path, default=DEFAULT_SEEDING)
    parser.add_argument("--terminal", type=Path, default=DEFAULT_TERMINAL)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    summary = evaluate(
        game=args.game,
        action_budget=args.action_budget,
        max_cycles=args.max_cycles,
        output_events=args.events,
        output_transitions=args.transitions,
        output_contexts=args.contexts,
        output_slicing=args.slicing,
        output_seeding=args.seeding,
        output_terminal=args.terminal,
        output_freeze=args.freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
