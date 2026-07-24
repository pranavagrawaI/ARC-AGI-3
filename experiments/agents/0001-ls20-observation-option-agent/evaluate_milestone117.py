"""Evaluate M11.7 cross-level option-template transfer."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from evaluate_milestone116 import _run_probe  # noqa: E402
from observation_option_agent import (  # noqa: E402
    AttemptMemory,
    BoundaryProbeSlicer,
    CrossLevelTemplateRegistry,
    FrontierDetector,
    FrontierProbeCandidate,
    FrontierProbeExecutor,
    FrontierProbeScorer,
    LevelContextManager,
    LevelTemplateInstantiator,
    LevelTransitionDetector,
    LiveCandidateActuator,
    LiveCandidateGenerator,
    LiveObservationStateClassifier,
    LiveOptionInducer,
    OptionTemplateExtractor,
    ProbeOutcomeClassifier,
    SaturationDetector,
    SelectionPolicy,
    TemplateContribution,
    TemplateInstanceEvaluator,
    apply_graph_expansion,
    cross_level_transfer_findings,
    m117_status,
    observation_hash,
    promote_validated_template_instance,
    terminal_evidence_from_transition,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-117-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-117-events.jsonl"
DEFAULT_TEMPLATE_REGISTRY = ARTIFACT_DIR / "option-template-registry.json"
DEFAULT_INSTANTIATION = ARTIFACT_DIR / "template-instantiation-ledger.json"
DEFAULT_VALIDATION = ARTIFACT_DIR / "template-validation-ledger.json"
DEFAULT_AUDIT = ARTIFACT_DIR / "cross-level-transfer-audit.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-117-freeze.md"


def evaluate(
    *,
    game: str,
    action_budget: int,
    max_cycles: int,
    output_events: Path,
    output_template_registry: Path,
    output_instantiation: Path,
    output_validation: Path,
    output_audit: Path,
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
    template_records: list[dict[str, Any]] = []
    instantiation_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    recent_states: list[str] = []
    recent_selected: list[str] = []
    actions_used = 0
    cycles_since_expansion = 0
    transition_handled = False
    template_contribution = TemplateContribution(
        changed_candidate_generation=False,
        changed_candidate_score=False,
        changed_selection_surface=False,
        generic_frontier_baseline_candidate_count=0,
        template_augmented_candidate_count=0,
    )
    transition = None
    level_context = None
    stale_validation = None
    fresh_validation = None
    promoted_expansion = None
    post_promotion_selection = None
    post_transition_binding = None
    post_transition_observation = None
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
            "milestone": "M11_7_CROSS_LEVEL_OPTION_TEMPLATE_TRANSFER",
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
                boundary_slice = slicer.slice(
                    execution=probe_result["execution"],
                    transition=transition,
                )
                terminal = terminal_evidence_from_transition(
                    transition=transition,
                    pre_terminal_state_class=binding.state_class_id,
                )
                level_context = LevelContextManager().create_context(
                    level_index=transition.to_level_index,
                    observation_hash=observation_hash(current),
                )
                post_transition_observation = current
                post_transition_binding = classifier.classify(
                    observation_hash=observation_hash(current),
                    graph=level_context.graph,
                    options=(),
                    controlled_position=current_position,
                    controlled_position_source=current_position_source,
                    semantic_guards_valid=True,
                )
                events.extend([
                    {
                        "milestone": "M11_7_CROSS_LEVEL_OPTION_TEMPLATE_TRANSFER",
                        "phase": "LEVEL_TRANSITION_HANDLED",
                        "cycle": cycle,
                        "transition": transition.to_record(),
                        "slice": boundary_slice.to_record(),
                        "terminal_evidence": terminal.to_record(),
                        "context": level_context.to_record(),
                        "binding": post_transition_binding.to_record(),
                    },
                ])
                transition_handled = True
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
            "milestone": "M11_7_CROSS_LEVEL_OPTION_TEMPLATE_TRANSFER",
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
            "milestone": "M11_7_CROSS_LEVEL_OPTION_TEMPLATE_TRANSFER",
            "phase": "LIVE_OPTION_ACTUATION",
            "cycle": cycle,
            "actuation": result.to_record(),
        })
        if actions_used >= action_budget:
            stop_reason = "ACTION_BUDGET_EXHAUSTED"
            break

    templates = ()
    instances = ()
    stale_instance = None
    promoted_instance_scored = False
    expanded_level_graph_used = False
    source_probe_sequence_used_as_policy = False
    stale_instance_rejected = False
    template_contains_level_local_fact = False
    instance_binding_from_prior_level_fact = False
    validation_probe_public = False

    if transition_handled and level_context and post_transition_binding and current_position:
        templates = OptionTemplateExtractor().extract(
            options=options,
            source_level_index=transition.from_level_index if transition else 0,
        )
        registry = CrossLevelTemplateRegistry(templates=templates)
        template_records = [registry.to_record(), registry.carry_into(level_context=level_context)]
        events.append({
            "milestone": "M11_7_CROSS_LEVEL_OPTION_TEMPLATE_TRANSFER",
            "phase": "OPTION_TEMPLATE_EXTRACTION",
            "template_count": len(templates),
            "templates": [template.to_record() for template in templates],
        })
        if templates:
            template = templates[0]
            source_option = next(
                option for option in options if option.option_id == template.source_option_id
            )
            instantiator = LevelTemplateInstantiator()
            stale_instance = instantiator.stale_instance_from_option(
                option=source_option,
                template=template,
                level_context=level_context,
            )
            evaluator = TemplateInstanceEvaluator()
            stale_validation = evaluator.evaluate(instance=stale_instance, execution=None)
            stale_instance_rejected = stale_validation.result == "REJECTED_STALE_INSTANCE"
            fresh_instance = instantiator.instantiate(
                template=template,
                level_context=level_context,
                observation_hash=observation_hash(post_transition_observation or current),
                controlled_position=current_position,
                controlled_position_source=current_position_source,
                available_actions=tuple((post_transition_observation or current).available_actions),
            )
            instances = (fresh_instance,)
            instantiation_records = [
                {
                    "phase": "STALE_INSTANCE_REJECTION",
                    "instance": stale_instance.to_record(),
                    "validation": stale_validation.to_record(),
                    "promoted": False,
                    "executable": False,
                },
                {
                    "phase": "TEMPLATE_INSTANTIATION",
                    "instance": fresh_instance.to_record(),
                    "binding_inputs": {
                        "controlled_position_source": current_position_source,
                        "fresh_observation_hash": observation_hash(post_transition_observation or current),
                        "semantic_guards_valid": True,
                    },
                    "forbidden_prior_fields_present": fresh_instance.forbidden_prior_fields_present,
                },
            ]
            validation_candidate = FrontierProbeCandidate(
                probe_id="template_validation_000",
                state_class_id=level_context.initial_state_class_id,
                probe_type="template_validation",
                action_sequence=fresh_instance.validation_action_sequence,
                expected_information=(
                    "validate transferred option template instance",
                    "bind fresh level-local target region",
                ),
                predicted_risk="low",
                semantic_guards_required=True,
                provenance=({
                    "kind": "template_instance_validation",
                    "template_id": template.template_id,
                    "instance_id": fresh_instance.instance_id,
                },),
            )
            validation_execution, current = probe_executor.execute(
                env=env,
                current_observation=post_transition_observation or current,
                candidate=validation_candidate,
                current_controlled_position=current_position,
                step_offset=actions_used,
                remaining_budget=min(4, action_budget - actions_used),
            )
            actions_used += validation_execution.actions_executed
            current_position = validation_execution.final_position
            current_position_source = validation_execution.controlled_position_source
            fresh_validation = evaluator.evaluate(
                instance=fresh_instance,
                execution=validation_execution,
            )
            validation_probe_public = fresh_validation.public_actions_used
            source_state = level_context.graph.state(level_context.initial_state_class_id)
            if source_state is not None:
                promoted_expansion = promote_validated_template_instance(
                    expansion_index=0,
                    template=template,
                    instance=fresh_instance,
                    validation=fresh_validation,
                    execution=validation_execution,
                    source_state=source_state,
                )
            if promoted_expansion is not None:
                fresh_validation = replace(
                    fresh_validation,
                    promoted_option_id=promoted_expansion.option.option_id,
                )
                expanded_graph = apply_graph_expansion(
                    graph=level_context.graph,
                    expansion=promoted_expansion,
                )
                expanded_options = (promoted_expansion.option,)
                action_plans[promoted_expansion.option.option_id] = (
                    fresh_instance.validation_action_sequence
                )
                baseline_candidates = LiveCandidateGenerator().generate(
                    graph=level_context.graph,
                    binding=post_transition_binding,
                    options=(),
                    action_plans={},
                    max_depth=2,
                )
                post_promotion_candidates = LiveCandidateGenerator().generate(
                    graph=expanded_graph,
                    binding=post_transition_binding,
                    options=expanded_options,
                    action_plans=action_plans,
                    max_depth=2,
                )
                post_promotion_selection = SelectionPolicy(
                    progress_model=_learned_progress_model()
                ).select(
                    round_index=0,
                    state_class_id=post_transition_binding.state_class_id or "",
                    candidates=post_promotion_candidates,
                    attempt_memory=AttemptMemory(),
                )
                promoted_instance_scored = any(
                    promoted_expansion.option.option_id in candidate.option_ids
                    for candidate in post_promotion_selection.candidates
                )
                expanded_level_graph_used = promoted_instance_scored
                template_contribution = TemplateContribution(
                    changed_candidate_generation=(
                        len(post_promotion_candidates) > len(baseline_candidates)
                    ),
                    changed_candidate_score=promoted_instance_scored,
                    changed_selection_surface=promoted_instance_scored,
                    generic_frontier_baseline_candidate_count=len(baseline_candidates),
                    template_augmented_candidate_count=len(post_promotion_candidates),
                )
                events.append({
                    "milestone": "M11_7_CROSS_LEVEL_OPTION_TEMPLATE_TRANSFER",
                    "phase": "POST_TEMPLATE_PROMOTION_REPLAN",
                    "candidate_count": len(post_promotion_selection.candidates),
                    "selected_candidate_id": post_promotion_selection.selected_candidate_id,
                    "expanded_level_1_graph_used": expanded_level_graph_used,
                    "ledger": post_promotion_selection.to_record(),
                })
            validation_records = [
                stale_validation.to_record() if stale_validation else {},
                fresh_validation.to_record() if fresh_validation else {},
                validation_execution.to_record(),
            ]
            if promoted_expansion is not None:
                validation_records.append({
                    "phase": "TEMPLATE_GRAPH_PROMOTION",
                    "expansion": promoted_expansion.to_record(),
                })
            events.extend([
                {
                    "milestone": "M11_7_CROSS_LEVEL_OPTION_TEMPLATE_TRANSFER",
                    "phase": "STALE_INSTANCE_REJECTION",
                    "validation": stale_validation.to_record() if stale_validation else None,
                },
                {
                    "milestone": "M11_7_CROSS_LEVEL_OPTION_TEMPLATE_TRANSFER",
                    "phase": "TEMPLATE_INSTANCE_VALIDATION",
                    "validation": fresh_validation.to_record() if fresh_validation else None,
                    "execution": validation_execution.to_record(),
                },
            ])

    template_contains_level_local_fact = any(
        template.contains_level_local_fact() for template in templates
    )
    instance_binding_from_prior_level_fact = any(
        instance.forbidden_prior_fields_present for instance in instances
    )
    source_probe_sequence_used_as_policy = any(
        template.source_probe_sequence_used_as_policy for template in templates
    ) or any(instance.source_probe_sequence_used_as_policy for instance in instances)
    audit_payload = {
        "templates": [template.to_record() for template in templates],
        "instances": [instance.to_record() for instance in instances],
        "validations": validation_records,
        "contribution": template_contribution.to_record(),
    }
    findings = cross_level_transfer_findings(audit_payload)
    audit_records.append({
        "forbidden_source_findings": list(findings),
        "template_contains_level_local_fact": template_contains_level_local_fact,
        "instance_binding_from_prior_level_fact": instance_binding_from_prior_level_fact,
        "source_probe_sequence_used_as_policy": source_probe_sequence_used_as_policy,
        "stale_instance_rejected": stale_instance_rejected,
        "prior_level_coordinates_in_fresh_instance": any(
            instance.prior_level_coordinates_used for instance in instances
        ),
        "prior_level_edges_in_fresh_instance": any(
            instance.prior_level_edges_used for instance in instances
        ),
    })
    status = m117_status(
        transition_handled=transition_handled,
        template_count=len(templates),
        template_contains_level_local_fact=template_contains_level_local_fact,
        instance_count=len(instances),
        instance_binding_from_prior_level_fact=instance_binding_from_prior_level_fact,
        stale_negative_control_ran=stale_instance is not None,
        stale_instance_rejected=stale_instance_rejected,
        validation_probe_public=validation_probe_public,
        validated_instance_count=(
            1
            if fresh_validation
            and fresh_validation.result == "VALIDATED_TEMPLATE_INSTANCE"
            else 0
        ),
        promoted_instance_count=1 if promoted_expansion is not None else 0,
        promoted_instance_scored=promoted_instance_scored,
        template_contribution_detected=template_contribution.detected,
        stale_spatial_instance_reused=bool(
            stale_validation and stale_validation.result != "REJECTED_STALE_INSTANCE"
        ),
        prior_coordinate_leakage=any(
            instance.prior_level_coordinates_used for instance in instances
        ),
        prior_edge_leakage=any(instance.prior_level_edges_used for instance in instances),
        source_probe_script_reused=source_probe_sequence_used_as_policy,
        trace_leakage=bool(findings),
        oracle_signal=False,
    )

    _write_jsonl(output_events, events)
    _write_json(output_template_registry, {
        "registry": CrossLevelTemplateRegistry(templates=templates).to_record(),
        "records": template_records,
    })
    _write_json(output_instantiation, {
        "instances": [instance.to_record() for instance in instances],
        "records": instantiation_records,
    })
    _write_json(output_validation, {"records": validation_records})
    _write_json(output_audit, {"records": audit_records})
    summary = {
        "milestone": "M11_7_CROSS_LEVEL_OPTION_TEMPLATE_TRANSFER",
        "status": status,
        "contract_status": status,
        "game": game,
        "action_budget": action_budget,
        "max_cycles": max_cycles,
        "level_transition_handled": transition_handled,
        "template_count": len(templates),
        "template_kind": templates[0].kind if templates else None,
        "template_extracted_from_frontier_option": bool(templates),
        "template_contains_prior_coordinates": template_contains_level_local_fact,
        "template_contains_prior_state_ids": template_contains_level_local_fact,
        "template_contains_prior_edges": template_contains_level_local_fact,
        "template_contains_prior_observation_hashes": template_contains_level_local_fact,
        "source_probe_sequence_used_as_policy": source_probe_sequence_used_as_policy,
        "new_level_instance_count": len(instances),
        "instance_bound_from_fresh_observation": bool(instances)
        and not instance_binding_from_prior_level_fact,
        "stale_instance_rejected": stale_instance_rejected,
        "validation_probe_executed": validation_probe_public,
        "validated_template_instance_count": (
            1
            if fresh_validation
            and fresh_validation.result == "VALIDATED_TEMPLATE_INSTANCE"
            else 0
        ),
        "promoted_new_level_option_count": 1 if promoted_expansion is not None else 0,
        "promoted_instance_appeared_in_candidate_set": promoted_instance_scored,
        "promoted_instance_scored": promoted_instance_scored,
        "template_contribution_detected": template_contribution.detected,
        "template_contribution": template_contribution.to_record(),
        "expanded_level_1_graph_used_in_replanning": expanded_level_graph_used,
        "levels_completed": current.levels_completed,
        "final_state": current.state,
        "stop_reason": stop_reason,
        "forbidden_source_findings": list(findings),
        "known_limits": {
            "first_template_kind_only": "frontier_region_navigation",
            "not_full_ls20_solver": True,
            "not_cross_level_strategy_learning": True,
        },
        "artifacts": {
            "events": str(output_events),
            "template_registry": str(output_template_registry),
            "instantiation": str(output_instantiation),
            "validation": str(output_validation),
            "audit": str(output_audit),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _selected_candidate(selection) -> Any | None:
    for candidate in selection.candidates:
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
        "# Milestone 11.7 Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Lab-book line:",
        "",
        "```text",
        "M11.7:",
        "  I can reuse the kind of option I learned, not the old coordinates.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- level transition handled: `{summary['level_transition_handled']}`",
        f"- template count: `{summary['template_count']}`",
        f"- template kind: `{summary['template_kind']}`",
        f"- source probe sequence used as policy: `{summary['source_probe_sequence_used_as_policy']}`",
        f"- new level instance count: `{summary['new_level_instance_count']}`",
        f"- instance bound from fresh observation: `{summary['instance_bound_from_fresh_observation']}`",
        f"- stale instance rejected: `{summary['stale_instance_rejected']}`",
        f"- validation probe executed: `{summary['validation_probe_executed']}`",
        f"- validated template instance count: `{summary['validated_template_instance_count']}`",
        f"- promoted new-level option count: `{summary['promoted_new_level_option_count']}`",
        f"- promoted instance scored: `{summary['promoted_instance_scored']}`",
        f"- template contribution detected: `{summary['template_contribution_detected']}`",
        f"- forbidden source findings: `{summary['forbidden_source_findings']}`",
        "",
        "Known limits:",
        "",
        "- Only the frontier-region-navigation template family is exercised.",
        "- This is not cross-level strategy learning or full LS20 solving.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--action-budget", type=int, default=500)
    parser.add_argument("--max-cycles", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--template-registry", type=Path, default=DEFAULT_TEMPLATE_REGISTRY)
    parser.add_argument("--instantiation", type=Path, default=DEFAULT_INSTANTIATION)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    summary = evaluate(
        game=args.game,
        action_budget=args.action_budget,
        max_cycles=args.max_cycles,
        output_events=args.events,
        output_template_registry=args.template_registry,
        output_instantiation=args.instantiation,
        output_validation=args.validation,
        output_audit=args.audit,
        output_freeze=args.freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
