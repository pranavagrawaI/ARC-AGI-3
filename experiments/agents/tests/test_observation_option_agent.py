from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.harness import Observation
from experiments.harness.runner import public_observation


AGENT_ROOT = Path(__file__).resolve().parents[1] / "0001-ls20-observation-option-agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from observation_option_agent import (
    ActionCouplingMemory,
    ActionEffectConstraintModel,
    ActionEffectObservation,
    action_effect_observations_from_events,
    AttemptMemory,
    BoundaryProbeSlicer,
    ChainCandidate,
    ControlledEntityBelief,
    ConditionalEffectMemory,
    ContactDescriptor,
    Event,
    EventDetector,
    EventOptionExecutor,
    EventPredicateEvidenceProbe,
    EffectEvidence,
    EffectKey,
    EffectTransition,
    CrossLevelTemplateRegistry,
    FrontierDetector,
    FrontierProbeExecution,
    FrontierProbeExecutor,
    FrontierProbeScorer,
    InteractionDetector,
    LearnedProgressModel,
    LevelContextManager,
    LevelTemplateInstantiator,
    LevelTransitionDetector,
    LiveCandidateActuator,
    LiveCandidateGenerator,
    LiveContactProbeAgent,
    LiveOptionInducer,
    LiveObservationStateClassifier,
    Milestone1Analyzer,
    ControlVerifier,
    D1InjectionPolicy,
    DriftClassifier,
    DiscoveryRecord,
    LocalRecoveryPlanner,
    ModelBackedPredictionProvider,
    ModelPredictionVerifier,
    MotorAwareSelector,
    GoalAwareSelector,
    GoalEvidenceRecord,
    GoalHypothesis,
    GoalHypothesisGenerator,
    GoalHypothesisScorer,
    GoalHypothesisUpdater,
    GoalTestCandidateFactory,
    OptionApplicabilityEdge,
    OptionApplicabilityGraph,
    OptionApplicabilityStateClass,
    OptionChainVerifier,
    ProgressHypothesisSet,
    ProgressLearningLedger,
    ProgressModel,
    ProgressOutcomeRecord,
    ProbeOutcomeClassifier,
    SemanticSignature,
    SelfOptionEvaluator,
    SelfOptionExecution,
    SelfOptionInducer,
    SelectionPolicy,
    SaturationDetector,
    StructuredRecoveryAgent,
    TraceAlignment,
    TraceAssistedReplayAgent,
    TraceCorridor,
    TraceReference,
    OptionCandidate,
    OptionEffectModel,
    OptionEffectObservation,
    OptionExecutor,
    FactorTransitionAssay,
    FactorCounterexampleAssay,
    ActorLocalGeometryRefiner,
    ActorLocalGeometryRepairAssay,
    PublicFactorExtractor,
    OptionTemplateExtractor,
    PredicateCandidate,
    PredicateInducer,
    OptionSegmenter,
    ObjectDelta,
    ObjectTracker,
    ObservationParser,
    PrequentialTransitionLearner,
    RelationEffectLearner,
    choose_contact_action,
    ProbePolicy,
    ReplayInterventionAssay,
    TraceStep,
    contact_effect_symbols_from_record,
    generate_chain_candidates,
    apply_graph_expansion,
    m11_status,
    m116_status,
    m106_status,
    transition_symbols_from_record,
    iter_trace_steps,
    observation_hash,
    replay_actions_to_position,
    selected_predicate_expression_hash,
    m6_status,
    m7_status,
    m8_status,
    m9_status,
    m10_status,
    nontriviality_criteria,
    observation_from_self_execution,
    outcome_from_chain_verification,
    outcome_record_from_candidate,
    summarize_probe,
    same_option_chain_is_honest,
    start_signature_class_for_option,
    start_signature_class_hash,
    terminal_evidence_from_transition,
    TemplateInstanceEvaluator,
    cross_level_transfer_findings,
    execute_public_style_goal_test,
    forbidden_goal_hypothesis_findings,
    m117_status,
    m12_status,
    promote_validated_template_instance,
    trace_leakage_findings,
)
from observation_option_agent.events import MotionGroup


def observation(grid, *, levels=0, state="NOT_FINISHED", full_reset=False):
    pixels = np.asarray([grid], dtype=np.int8)
    pixels.setflags(write=False)
    return Observation(pixels, (1, 2, 3, 4), levels, state, full_reset)


class ObservationOptionAgentTests(unittest.TestCase):
    def test_parser_emits_neutral_connected_components(self) -> None:
        obs = observation([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2],
            [0, 1, 0, 0, 2],
            [0, 0, 0, 0, 0],
        ])

        scene = ObservationParser().parse(obs, frame_index=0)

        self.assertEqual(scene.inferred_background, 0)
        self.assertEqual([(item.value, item.area) for item in scene.detections], [(1, 3), (2, 2)])

    def test_tracker_preserves_identity_across_translation(self) -> None:
        parser = ObservationParser()
        tracker = ObjectTracker()
        before = observation([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        after = observation([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        tracker.update(parser.parse(before, frame_index=0))

        update = tracker.update(parser.parse(after, frame_index=1))
        event = EventDetector().detect(
            step=1,
            before=before,
            action=4,
            after=after,
            tracks=update,
        )

        self.assertEqual(len(update.matches), 1)
        self.assertEqual(event.object_deltas[0].translation, (0, 1))
        self.assertEqual(event.motion_groups[0].object_ids, (1,))

    def test_tracker_records_same_location_appearance_change(self) -> None:
        parser = ObservationParser()
        tracker = ObjectTracker()
        before = observation([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
        after = observation([[0, 0, 0], [0, 2, 0], [0, 2, 0]])
        tracker.update(parser.parse(before, frame_index=0))

        update = tracker.update(parser.parse(after, frame_index=1))
        event = EventDetector().detect(
            step=1,
            before=before,
            action=1,
            after=after,
            tracks=update,
        )

        self.assertEqual(len(update.matches), 1)
        self.assertTrue(event.object_deltas[0].value_changed)
        self.assertFalse(event.object_deltas[0].moved)

    def test_tracker_reacquires_after_one_hidden_frame_without_action_attribution(
        self,
    ) -> None:
        parser = ObservationParser()
        tracker = ObjectTracker(max_missing_frames=1)
        visible = observation([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        hidden = observation([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        returned = observation([
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
        ])
        tracker.update(parser.parse(visible, frame_index=0))

        missing_update = tracker.update(parser.parse(hidden, frame_index=1))
        returned_update = tracker.update(parser.parse(returned, frame_index=2))
        event = EventDetector().detect(
            step=2,
            before=hidden,
            action=4,
            after=returned,
            tracks=returned_update,
        )

        self.assertEqual([item.object_id for item in missing_update.hidden], [1])
        self.assertEqual(missing_update.disappeared, ())
        self.assertEqual(returned_update.matches[0].object_id, 1)
        self.assertEqual([item.object_id for item in returned_update.reappeared], [1])
        self.assertEqual(event.reappeared_object_ids, (1,))
        self.assertEqual(event.object_deltas, ())

    def test_tracker_expires_identity_after_occlusion_bound(self) -> None:
        parser = ObservationParser()
        tracker = ObjectTracker(max_missing_frames=1)
        visible = observation([[0, 1, 0]])
        hidden = observation([[0, 0, 0]])
        tracker.update(parser.parse(visible, frame_index=0))

        first_missing = tracker.update(parser.parse(hidden, frame_index=1))
        second_missing = tracker.update(parser.parse(hidden, frame_index=2))

        self.assertEqual([item.object_id for item in first_missing.hidden], [1])
        self.assertEqual(first_missing.disappeared, ())
        self.assertEqual(
            [item.object_id for item in second_missing.disappeared],
            [1],
        )
        self.assertEqual(second_missing.hidden, ())

    def test_event_detector_classifies_quarter_turn(self) -> None:
        parser = ObservationParser()
        tracker = ObjectTracker()
        before = observation([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        after = observation([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ])
        tracker.update(parser.parse(before, frame_index=0))
        tracks = tracker.update(parser.parse(after, frame_index=1))

        event = EventDetector().detect(
            step=1,
            before=before,
            action=1,
            after=after,
            tracks=tracks,
        )

        self.assertEqual(event.object_deltas[0].shape_transform, "rotation:1")

    def test_trace_loader_aligns_actions_and_drops_private_fields(self) -> None:
        rows = [
            {
                "data": {
                    "frame": [[[0, 0], [0, 1]]],
                    "available_actions": [1, 2],
                    "levels_completed": 0,
                    "state": "NOT_FINISHED",
                    "full_reset": True,
                    "action_input": {"id": "RESET"},
                    "game_id": "private-to-loader-output",
                    "guid": "private-to-loader-output",
                }
            },
            {
                "data": {
                    "frame": [[[0, 0], [1, 0]]],
                    "available_actions": [1, 2],
                    "levels_completed": 0,
                    "state": "NOT_FINISHED",
                    "full_reset": False,
                    "action_input": {"id": "ACTION2"},
                    "game_id": "private-to-loader-output",
                    "guid": "private-to-loader-output",
                }
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            steps = list(iter_trace_steps(path))

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].action, 2)
        self.assertFalse(hasattr(steps[0].after, "game_id"))
        self.assertFalse(hasattr(steps[0].after, "guid"))

    def test_action_coupling_rejects_action_independent_motion(self) -> None:
        memory = ActionCouplingMemory()
        for step, action in enumerate((1, 2), start=1):
            memory.observe(Event(
                step=step,
                action=action,
                levels_before=0,
                levels_after=0,
                state_after="NOT_FINISHED",
                full_reset=False,
                changed_pixels=2,
                scene_change_fraction=0.1,
                object_deltas=(
                    ObjectDelta(
                        1,
                        (-1, 0) if action == 1 else (1, 0),
                        1,
                        1,
                        False,
                        1,
                        1,
                        (1, 1, 1, 1),
                        (0, 1, 0, 1) if action == 1 else (2, 1, 2, 1),
                    ),
                    ObjectDelta(2, (0, 1), 2, 2, False, 1, 1, (0, 0, 0, 0), (0, 1, 0, 1)),
                ),
                created_object_ids=(),
                disappeared_object_ids=(),
                motion_groups=(),
            ))

        summaries = memory.summaries()

        self.assertEqual([summary["object_id"] for summary in summaries], [1])

    def test_transition_learner_predicts_contact_conditioned_visual_symbol(
        self,
    ) -> None:
        learner = PrequentialTransitionLearner()
        contact_record = self._record(
            action=2,
            contacts=((0, 3, "shape-a"),),
            transforms=("rotation:1",),
        )
        distractor = self._record(
            action=2,
            contacts=((9, 1, "shape-b"),),
            transforms=("unclassified_form_change",),
        )
        learner.observe(distractor)
        learner.observe(contact_record)
        learner.observe(contact_record)

        prediction = learner.predict(contact_record)

        self.assertGreater(
            prediction.probability("transform:rotation:1"),
            prediction.probability("transform:unclassified_form_change"),
        )
        self.assertEqual(prediction.source, "action_contact")

    def test_transition_symbol_extraction_stays_generic(self) -> None:
        record = self._record(
            action=1,
            contacts=(),
            transforms=("rotation:1",),
            level_delta=1,
            blocked=True,
        )

        symbols = transition_symbols_from_record(record)

        self.assertEqual(
            symbols,
            frozenset({
                "blocked",
                "level_delta:+1",
                "transform:rotation:1",
            }),
        )

    def test_contact_effect_symbols_focus_on_changed_noncontrolled_effects(
        self,
    ) -> None:
        record = self._record(
            action=2,
            contacts=((0, 3, "shape-a"),),
            transforms=("rotation:1",),
            blocked=False,
        )
        record["interaction"]["changed_other_object_ids"] = [99]
        record["object_deltas"][0]["object_id"] = 99

        symbols = contact_effect_symbols_from_record(record)

        self.assertEqual(symbols, frozenset({"effect_transform:rotation:1"}))

    def test_relation_effect_learner_exposes_contact_specific_hypothesis(
        self,
    ) -> None:
        learner = RelationEffectLearner()
        rotating = self._effect_record(
            action=2,
            contacts=((0, 3, "contact-a"),),
            changed_object_id=99,
            transform="rotation:1",
        )
        growing = self._effect_record(
            action=2,
            contacts=((8, 1, "contact-b"),),
            changed_object_id=77,
            transform="unclassified_form_change",
            area_delta=2,
        )
        learner.observe(growing)
        learner.observe(rotating)
        learner.observe(rotating)

        prediction = learner.predict(rotating)
        hypotheses = learner.hypotheses(minimum_support=2)

        self.assertGreater(
            prediction.probability("effect_transform:rotation:1"),
            prediction.probability("effect_area_change"),
        )
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(hypotheses[0].contact_signature, ((0, 3, "contact-a"),))
        self.assertEqual(hypotheses[0].top_effects[0][0], "effect_transform:rotation:1")
        self.assertEqual(hypotheses[0].support, 2)

    def test_relation_effect_scoring_populates_public_hypotheses(self) -> None:
        learner = RelationEffectLearner()
        records = [
            self._effect_record(
                action=2,
                contacts=((0, 3, "contact-a"),),
                changed_object_id=99,
                transform="rotation:1",
            ),
            self._effect_record(
                action=2,
                contacts=((0, 3, "contact-a"),),
                changed_object_id=99,
                transform="rotation:1",
            ),
        ]

        learner.score_records(records)

        self.assertEqual(len(learner.hypotheses(minimum_support=2)), 1)

    def test_probe_policy_selects_high_lift_relation(self) -> None:
        learner = RelationEffectLearner()
        for _ in range(3):
            learner.observe(self._effect_record(
                action=2,
                contacts=((0, 3, "contact-a"),),
                changed_object_id=99,
                transform="rotation:1",
            ))
        for _ in range(3):
            learner.observe(self._effect_record(
                action=2,
                contacts=((8, 1, "contact-b"),),
                changed_object_id=77,
                transform="unclassified_form_change",
                area_delta=2,
            ))

        proposals = ProbePolicy(minimum_support=3, minimum_lift=1.1).propose(
            learner.hypotheses(minimum_support=3)
        )

        self.assertEqual(proposals[0].predicted_effect, "effect_transform:rotation:1")
        self.assertEqual(proposals[0].contact_signature, ((0, 3, "contact-a"),))

    def test_replay_assay_scores_positive_and_negative_controls(self) -> None:
        relation_record = self._effect_record(
            action=2,
            contacts=((0, 3, "contact-a"),),
            changed_object_id=99,
            transform="rotation:1",
        )
        other_record = self._effect_record(
            action=2,
            contacts=((8, 1, "contact-b"),),
            changed_object_id=77,
            transform="unclassified_form_change",
            area_delta=2,
        )
        learner = RelationEffectLearner()
        for _ in range(3):
            learner.observe(relation_record)
        for _ in range(3):
            learner.observe(other_record)
        proposal = ProbePolicy(minimum_support=3, minimum_lift=1.1).propose(
            learner.hypotheses(minimum_support=3)
        )[0]

        result = ReplayInterventionAssay().evaluate(
            proposal,
            [relation_record, relation_record, other_record, other_record],
        )

        self.assertEqual(result.positive_attempts, 2)
        self.assertEqual(result.positive_successes, 2)
        self.assertEqual(result.negative_controls, 2)
        self.assertEqual(result.negative_control_violations, 0)
        self.assertTrue(result.reproduced)
        self.assertTrue(result.negative_control_passed)

    def test_motor_constraints_deprioritize_repeated_no_op_action(self) -> None:
        model = ActionEffectConstraintModel(minimum_support=2)
        for step in range(2):
            model.observe(ActionEffectObservation(
                observation_id=f"noop_{step}",
                state_class_id="S0",
                action=3,
                before_position=(10, 10),
                after_position=(10, 10),
                levels_before=0,
                levels_after=0,
                source="unit_public_actuation_record",
            ))
        model.observe(ActionEffectObservation(
            observation_id="move_0",
            state_class_id="S0",
            action=4,
            before_position=(10, 10),
            after_position=(10, 15),
            levels_before=0,
            levels_after=0,
            source="unit_public_actuation_record",
        ))
        blocked = ChainCandidate(
            candidate_id="blocked_candidate",
            candidate_type="single_option",
            option_ids=("blocked_option",),
            start_state_class_id="S0",
            predicted_terminal_state_class_id="S1",
            predicted_progress_features={
                "level_completion": False,
                "stable_semantic_delta": False,
                "new_option_enabled": False,
                "new_state_class": True,
                "uncertainty_reduction": False,
                "volatile_visual_delta": False,
            },
            cost_features={"predicted_action_cost": 1, "empirical_consistency": 1.0},
            risk_features={"guard_risk": 0},
            provenance=({"kind": "unit_candidate"},),
        )
        moving = replace(
            blocked,
            candidate_id="moving_candidate",
            option_ids=("moving_option",),
        )

        selection = MotorAwareSelector(motor_model=model).select(
            round_index=0,
            state_class_id="S0",
            candidates=(blocked, moving),
            action_plans={
                "blocked_option": (3,),
                "moving_option": (4,),
            },
            attempt_memory=AttemptMemory(),
        )

        self.assertEqual(selection.selected_candidate_id, "moving_candidate")
        blocked_scored = next(
            candidate for candidate in selection.candidates
            if candidate.candidate_id == "blocked_candidate"
        )
        self.assertLess(blocked_scored.score_terms["motor_no_op_penalty"], 0)
        self.assertIn("motor_constraint", blocked_scored.rejection_reason)

    def test_motor_observations_are_extracted_from_public_action_records(self) -> None:
        events = [
            {
                "phase": "FRONTIER_PROBE_EXECUTION",
                "execution": {
                    "start_position": [10, 10],
                    "records": [
                        {
                            "step": 1,
                            "action": 3,
                            "executed": True,
                            "controlled_position": [10, 10],
                            "levels_before": 0,
                            "levels_after": 0,
                        },
                        {
                            "step": 2,
                            "action": 4,
                            "executed": True,
                            "controlled_position": [10, 15],
                            "levels_before": 0,
                            "levels_after": 0,
                        },
                    ],
                },
            },
            {
                "phase": "CONTROL",
                "execution": {
                    "start_position": [99, 99],
                    "records": [
                        {
                            "step": 3,
                            "action": 1,
                            "executed": False,
                            "controlled_position": [99, 99],
                        }
                    ],
                },
            },
        ]

        observations = action_effect_observations_from_events(events)

        self.assertEqual(len(observations), 2)
        self.assertTrue(observations[0].no_op)
        self.assertEqual(observations[0].before_position, (10, 10))
        self.assertEqual(observations[0].after_position, (10, 10))
        self.assertFalse(observations[1].no_op)
        self.assertEqual(observations[1].before_position, (10, 10))
        self.assertEqual(observations[1].after_position, (10, 15))

    def test_public_factor_extractor_surfaces_actor_stable_and_changed_factors(self) -> None:
        before = observation([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 2, 0],
            [0, 0, 0, 2, 0],
            [0, 3, 0, 0, 0],
        ])
        after = observation([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 2, 0],
            [0, 0, 0, 2, 0],
            [0, 4, 0, 0, 0],
        ])

        transition = PublicFactorExtractor().extract(
            transition_id="toy_000",
            before=before,
            action=4,
            after=after,
        )

        self.assertEqual(transition.actor_factor.delta, (0, 1))
        self.assertFalse(transition.no_op)
        self.assertEqual(transition.status_delta, 0)
        self.assertEqual(
            [(factor.value, factor.area) for factor in transition.stable_factors],
            [(2, 2)],
        )
        self.assertEqual(
            [(factor.value_before, factor.value_after) for factor in transition.changed_factors],
            [(3, 4)],
        )

    def test_factor_assay_compresses_raw_states_and_improves_prediction(self) -> None:
        transitions = (
            PublicFactorExtractor().extract(
                transition_id="t0",
                before=observation([
                    [0, 0, 0, 0],
                    [0, 1, 0, 2],
                    [0, 0, 0, 2],
                ]),
                action=4,
                after=observation([
                    [0, 0, 0, 0],
                    [0, 0, 1, 2],
                    [0, 0, 0, 2],
                ]),
            ),
            PublicFactorExtractor().extract(
                transition_id="t1",
                before=observation([
                    [0, 0, 0, 0],
                    [2, 0, 1, 0],
                    [2, 0, 0, 0],
                ]),
                action=4,
                after=observation([
                    [0, 0, 0, 0],
                    [2, 0, 0, 1],
                    [2, 0, 0, 0],
                ]),
            ),
            PublicFactorExtractor().extract(
                transition_id="t2",
                before=observation([
                    [0, 0, 0, 0],
                    [0, 1, 0, 2],
                    [0, 0, 0, 2],
                ]),
                action=3,
                after=observation([
                    [0, 0, 0, 0],
                    [0, 1, 0, 2],
                    [0, 0, 0, 2],
                ]),
            ),
        )

        audit = FactorTransitionAssay().evaluate(transitions)

        self.assertLess(audit.factor_state_count, audit.raw_state_count)
        self.assertGreater(audit.factor_context_reuse_count, audit.raw_context_reuse_count)
        self.assertGreater(audit.factor_prediction_accuracy, audit.raw_prediction_accuracy)
        self.assertGreater(audit.compression_gain, 0.0)

    def test_factor_counterexample_assay_blocks_contradicted_contexts(self) -> None:
        transitions = (
            PublicFactorExtractor().extract(
                transition_id="move_0",
                before=observation([
                    [0, 0, 0, 0],
                    [0, 1, 0, 2],
                    [0, 0, 0, 2],
                ]),
                action=4,
                after=observation([
                    [0, 0, 0, 0],
                    [0, 0, 1, 2],
                    [0, 0, 0, 2],
                ]),
            ),
            PublicFactorExtractor().extract(
                transition_id="blocked_0",
                before=observation([
                    [0, 0, 0, 0],
                    [2, 0, 1, 0],
                    [2, 0, 0, 0],
                ]),
                action=4,
                after=observation([
                    [0, 0, 0, 0],
                    [2, 0, 1, 0],
                    [2, 0, 0, 0],
                ]),
            ),
            PublicFactorExtractor().extract(
                transition_id="noop_0",
                before=observation([
                    [0, 0, 0, 0],
                    [0, 1, 0, 2],
                    [0, 0, 0, 2],
                ]),
                action=3,
                after=observation([
                    [0, 0, 0, 0],
                    [0, 1, 0, 2],
                    [0, 0, 0, 2],
                ]),
            ),
            PublicFactorExtractor().extract(
                transition_id="noop_1",
                before=observation([
                    [0, 0, 0, 0],
                    [2, 0, 1, 0],
                    [2, 0, 0, 0],
                ]),
                action=3,
                after=observation([
                    [0, 0, 0, 0],
                    [2, 0, 1, 0],
                    [2, 0, 0, 0],
                ]),
            ),
        )

        audit = FactorCounterexampleAssay().evaluate(transitions)

        self.assertEqual(audit.reused_context_count, 2)
        self.assertEqual(audit.contradicted_context_count, 1)
        self.assertEqual(audit.stable_context_count, 1)
        self.assertEqual(audit.promoted_context_count, 1)
        self.assertEqual(audit.blocked_promotion_count, 1)
        self.assertEqual(audit.no_op_contradiction_count, 1)
        self.assertEqual(audit.counterexample_count, 1)

    def test_local_geometry_repair_splits_move_noop_alias_without_coordinate_memory(self) -> None:
        cases = (
            (
                "open_top",
                observation([
                    [0, 0, 0, 0],
                    [0, 1, 0, 2],
                    [0, 0, 0, 2],
                ]),
                observation([
                    [0, 0, 0, 0],
                    [0, 0, 1, 2],
                    [0, 0, 0, 2],
                ]),
                (1, 1),
            ),
            (
                "open_bottom",
                observation([
                    [0, 0, 0, 0],
                    [0, 0, 0, 2],
                    [0, 1, 0, 2],
                ]),
                observation([
                    [0, 0, 0, 0],
                    [0, 0, 0, 2],
                    [0, 0, 1, 2],
                ]),
                (2, 1),
            ),
            (
                "blocked_top",
                observation([
                    [0, 0, 0, 0],
                    [0, 0, 1, 2],
                    [0, 0, 0, 2],
                ]),
                observation([
                    [0, 0, 0, 0],
                    [0, 0, 1, 2],
                    [0, 0, 0, 2],
                ]),
                (1, 2),
            ),
            (
                "blocked_bottom",
                observation([
                    [0, 0, 0, 0],
                    [0, 0, 0, 2],
                    [0, 0, 1, 2],
                ]),
                observation([
                    [0, 0, 0, 0],
                    [0, 0, 0, 2],
                    [0, 0, 1, 2],
                ]),
                (2, 2),
            ),
        )
        refiner = ActorLocalGeometryRefiner(coordinate_bucket_size=1)
        records = []
        transitions = []
        for case_id, before, after, actor_position in cases:
            transition = PublicFactorExtractor().extract(
                transition_id=case_id,
                before=before,
                action=4,
                after=after,
                controlled_position=actor_position,
            )
            transitions.append(transition)
            records.append(refiner.refine(
                transition=transition,
                before=before,
                actor_position=actor_position,
            ))

        r1_audit = FactorCounterexampleAssay().evaluate(tuple(transitions))
        audit = ActorLocalGeometryRepairAssay().evaluate(tuple(records))

        self.assertEqual(r1_audit.contradicted_context_count, 1)
        self.assertEqual(audit.local_geometry.contradicted_context_count, 0)
        self.assertEqual(audit.local_geometry.reused_context_count, 2)
        self.assertEqual(audit.local_geometry.stable_context_count, 2)
        self.assertEqual(audit.resolved_alias_count, 1)
        self.assertEqual(audit.coordinate_control.reused_context_count, 0)

    def test_contact_action_selection_targets_matching_signature(self) -> None:
        parser = ObservationParser()
        tracker = ObjectTracker()
        obs = observation([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2],
            [0, 0, 0, 0, 0],
        ])
        scene = parser.parse(obs, frame_index=0)
        tracks = tracker.update(scene)
        control = self._control_estimate((1,), {4: (0, 2), 2: (1, 0)})
        proposal = self._probe_proposal(
            contacts=((2, 1, "7d9d4bda4403"),),
            effect="effect_transform:rotation:1",
        )

        action = choose_contact_action(
            proposal=proposal,
            tracks=tracks,
            control=control,
            available_actions=(2, 4),
        )

        self.assertEqual(action, 4)

    def test_live_probe_agent_replays_bootstrap_then_selects_contact(self) -> None:
        agent = LiveContactProbeAgent(
            proposal=self._probe_proposal(
                contacts=((2, 1, "7d9d4bda4403"),),
                effect="effect_transform:rotation:1",
            ),
            bootstrap_actions=(1,),
        )
        first = observation([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2],
            [0, 0, 0, 0, 0],
        ])
        decision = agent.act(first)
        self.assertEqual(decision.action, 1)
        agent.control_belief.active = self._control_estimate((1,), {4: (0, 2)})

        decision = agent.act(first)

        self.assertEqual(decision.action, 4)

    def test_live_probe_agent_uses_transferred_expected_delta_for_contact(self) -> None:
        agent = LiveContactProbeAgent(
            proposal=self._probe_proposal(
                contacts=((2, 1, "7d9d4bda4403"),),
                effect="effect_transform:rotation:1",
            ),
        )
        obs = observation([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2],
            [0, 0, 0, 0, 0],
        ])
        agent.act(obs)
        agent.control_belief.active = self._control_estimate((1,), {1: (0, -1)})
        agent.control_belief._transferred[4][(0, 2)] = 3

        decision = agent.act(obs)

        self.assertEqual(decision.action, 4)

    def test_milestone1_analyzer_reports_completion_contexts(self) -> None:
        records = [
            self._event_record(step=1, action=1, levels_before=0, levels_after=0),
            self._event_record(
                step=2,
                action=2,
                levels_before=0,
                levels_after=1,
                contact_signature=((5, 2, "shape-x"),),
            ),
        ]

        summary = Milestone1Analyzer().analyze(records)

        self.assertEqual(summary["completion"]["status"], "done")
        self.assertEqual(summary["completion"]["completion_events"], 1)
        self.assertEqual(
            summary["completion"]["events"][0]["contact_signature"],
            [{"value": 5, "area": 2, "shape_id": "shape-x"}],
        )

    def test_milestone1_analyzer_marks_absent_regression_evidence(self) -> None:
        summary = Milestone1Analyzer().analyze([
            self._event_record(step=1, action=1),
            self._event_record(step=2, action=2),
        ])

        self.assertEqual(summary["regression"]["status"], "unknown_absent_in_trace")

    def test_milestone1_analyzer_reports_disappearance_evidence_as_partial(
        self,
    ) -> None:
        summary = Milestone1Analyzer().analyze([
            self._event_record(
                step=1,
                action=2,
                contact_signature=((1, 1, "shape-a"),),
                disappeared=(9,),
                created=(10,),
                changed=(11,),
            )
        ])

        self.assertEqual(summary["disappearance_resource_like"]["status"], "partial")
        self.assertEqual(summary["disappearance_resource_like"]["candidate_events"], 1)

    def test_trace_reference_stops_at_first_public_completion(self) -> None:
        steps = [
            self._trace_step(
                [[0, 1, 0]],
                4,
                [[0, 0, 1]],
                levels_before=0,
                levels_after=0,
            ),
            self._trace_step(
                [[0, 0, 1]],
                1,
                [[0, 1, 0]],
                levels_before=0,
                levels_after=1,
            ),
            self._trace_step(
                [[0, 1, 0]],
                2,
                [[1, 0, 0]],
                levels_before=1,
                levels_after=1,
            ),
        ]

        reference = TraceReference.from_steps(steps)

        self.assertEqual(reference.segment_actions, (4, 1))
        self.assertEqual(reference.completion_step, 2)
        self.assertEqual(reference.initial_level, 0)

    def test_trace_alignment_reports_uncertainty_and_respects_window(self) -> None:
        reference = TraceReference.synthetic([
            ((0.0, 0.0), 1),
            ((0.0, 1.0), 2),
            ((0.0, 2.0), 3),
            ((0.0, 9.0), None),
        ])
        alignment = TraceAlignment(forward_window=1)

        result = alignment.align(
            current_center=(0.0, 9.0),
            current_signature=reference.states[3].controlled_signature,
            reference=reference,
            last_matched_step=1,
        )

        self.assertEqual(result.best_match_step, 2)
        self.assertEqual(result.window, (0, 2))
        self.assertGreaterEqual(result.alignment_entropy, 0.0)
        self.assertIsNotNone(result.second_best_step)

    def test_low_confidence_alignment_yields_no_confident_alignment(self) -> None:
        reference = TraceReference.synthetic([
            ((0.0, 0.0), 1),
            ((0.0, 0.2), 2),
            ((0.0, 0.4), None),
        ])
        alignment = TraceAlignment(confident_margin=1.0)

        result = alignment.align(
            current_center=(0.0, 0.21),
            current_signature=reference.states[1].controlled_signature,
            reference=reference,
            last_matched_step=1,
        )

        self.assertEqual(result.alignment_status, "NO_CONFIDENT_ALIGNMENT")

    def test_trace_assisted_agent_recovers_by_reducing_offset(self) -> None:
        reference = TraceReference.synthetic([
            ((0.0, 0.0), 4),
            ((0.0, 1.0), None),
        ])
        agent = TraceAssistedReplayAgent(reference=reference, recovery_budget=2)

        action = agent.choose_recovery_action(
            current_center=(0.0, -1.0),
            target_center=(0.0, 0.0),
            available_actions=(1, 4),
            verified_action_model={1: (-1, 0), 4: (0, 1)},
        )

        self.assertEqual(action, 4)

    def test_control_conflicts_suppress_bad_action_mapping(self) -> None:
        verifier = ControlVerifier(conflict_limit=2)
        verifier.seed({4: (0, 1)})

        verifier.observe(action=4, expected=(0, 1), observed=(0, 0))
        verifier.observe(action=4, expected=(0, 1), observed=(0, 0))

        self.assertNotIn(4, verifier.available_model((4,)))
        self.assertEqual(verifier.conflicts, 2)

    def test_d1_injection_records_and_invalidates_without_displacement(self) -> None:
        policy = D1InjectionPolicy(
            primary_step=2,
            backup_steps=(3,),
            alternatives={2: 3, 3: 4},
        )

        record = policy.record_for(
            live_step=2,
            original_action=1,
            controlled_displaced=False,
        )

        self.assertFalse(record.valid)
        self.assertEqual(record.injected_step, 2)
        self.assertEqual(record.injected_action, 3)
        self.assertEqual(record.reason, "harmless_displacement")

    def test_trace_corridor_rejects_semantic_false_positive(self) -> None:
        corridor = TraceCorridor.synthetic([
            (4, (2, 2), (("fixed-a",),)),
            (5, (2, 2), (("fixed-b",),)),
        ])
        classifier = DriftClassifier()

        result = classifier.classify(
            current_center=(2, 2),
            current_signature=SemanticSignature.from_labels(("fixed-b",)),
            corridor=corridor,
            action_result=None,
        )

        self.assertEqual(result.drift_type, "ON_TRACE")
        self.assertEqual(result.chosen_corridor_index, 5)
        self.assertFalse(result.corridor_candidates[0]["semantic_compatible"])

    def test_volatile_signature_difference_does_not_create_world_drift(self) -> None:
        live = SemanticSignature.from_labels(("stable", "volatile-a"))
        reference = SemanticSignature.from_labels(("stable", "volatile-b"))

        self.assertTrue(live.compatible_with(reference, volatile_prefixes=("volatile",)))

    def test_expected_future_corridor_state_prevents_false_world_drift(self) -> None:
        corridor = TraceCorridor.synthetic([
            (1, (0, 0), (("state-a",),)),
            (2, (0, 1), (("state-b",),)),
        ])
        result = DriftClassifier().classify(
            current_center=(0, 0),
            current_signature=SemanticSignature.from_labels(("state-b",)),
            corridor=corridor,
            action_result=None,
        )

        self.assertEqual(result.drift_type, "LOCAL_DRIFT")
        self.assertEqual(result.chosen_corridor_index, 2)

    def test_resume_circuit_breaker_refuses_semantic_mismatch(self) -> None:
        agent = StructuredRecoveryAgent()
        corridor = TraceCorridor.synthetic([
            (7, (1, 1), (("expected",),)),
        ])

        result = agent.validate_resume(
            current_center=(1, 1),
            current_signature=SemanticSignature.from_labels(("changed",)),
            corridor=corridor,
            resume_index=7,
        )

        self.assertEqual(result, "TRACE_INVALID")

    def test_action_conflict_precedence_over_local_drift(self) -> None:
        corridor = TraceCorridor.synthetic([
            (3, (1, 0), (("stable",),)),
        ])

        result = DriftClassifier().classify(
            current_center=(0, 0),
            current_signature=SemanticSignature.from_labels(("stable",)),
            corridor=corridor,
            action_result={
                "predicted_delta": (1, 0),
                "observed_delta": (0, 0),
                "world_diff": [],
            },
        )

        self.assertEqual(result.drift_type, "ACTION_CONFLICT")

    def test_unknown_controlled_identity_does_not_become_local_drift(self) -> None:
        corridor = TraceCorridor.synthetic([
            (3, (1, 0), (("stable",),)),
        ])

        result = DriftClassifier().classify(
            current_center=None,
            current_signature=SemanticSignature.from_labels(("stable",)),
            corridor=corridor,
            action_result=None,
        )

        self.assertEqual(result.drift_type, "UNKNOWN")

    def test_option_segmenter_generates_deterministic_public_evidence_options(self) -> None:
        reference = TraceReference.synthetic([
            ((0.0, 0.0), 4),
            ((0.0, 1.0), 4),
            ((0.0, 2.0), 2),
            ((1.0, 2.0), None),
        ])
        segmenter = OptionSegmenter()

        first = segmenter.induce(reference)
        second = segmenter.induce(reference)

        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 1)
        self.assertEqual(first[0].trace_dependency_level, 3)
        self.assertTrue(all("kind" in item for item in first[0].evidence))

    def test_option_termination_predicate_is_not_trace_index_only(self) -> None:
        option = self._option_candidate()

        self.assertEqual(option.termination_condition["type"], "actor_reaches_observable_position")
        self.assertNotEqual(option.termination_condition["type"], "trace_index")
        self.assertEqual(option.termination_condition["forbidden_form"], "trace_index == segment_end")

    def test_coordinate_option_predicate_is_flagged_low_abstraction(self) -> None:
        option = self._option_candidate()

        self.assertEqual(option.predicate_abstraction_level, 1)

    def test_option_executor_masks_action_list_and_uses_equivalent_route(self) -> None:
        option = self._option_candidate()
        executor = OptionExecutor()

        result = executor.execute(
            option=option,
            current_position=(0, 0),
            current_signature=SemanticSignature.from_labels(()),
            action_deltas={1: (0, 1), 2: (1, 0), 3: (-1, 0)},
            blocked_edges=frozenset({((0, 0), 1)}),
            valid_positions=frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)}),
        )

        self.assertTrue(result.completed)
        self.assertEqual(result.plan, (2, 1, 1, 3))
        self.assertNotEqual(result.plan, (1, 1))
        self.assertTrue(result.raw_action_list_masked)
        self.assertTrue(all(not item["used_raw_trace_action"] for item in result.records))
        self.assertFalse(hasattr(option, "segment_actions"))

    def test_option_executor_refuses_when_initiation_condition_fails(self) -> None:
        result = OptionExecutor().execute(
            option=self._option_candidate(),
            current_position=(0, 0),
            current_signature=SemanticSignature.from_labels(("unexpected",)),
            action_deltas={1: (0, 1)},
        )

        self.assertFalse(result.completed)
        self.assertEqual(result.failure_reason, "INITIATION_CONDITION_FAILED")

    def test_option_executor_refuses_unexpected_semantic_delta(self) -> None:
        option = self._option_candidate()

        result = OptionExecutor().execute(
            option=option,
            current_position=(0, 0),
            current_signature=SemanticSignature.from_labels(()),
            action_deltas={1: (0, 1)},
            observed_deltas=((0, 1), (0, 1)),
            world_diffs=((), ({"type": "unexpected_present", "label": "changed"},)),
        )

        self.assertEqual(result.status, "WORLD_DRIFT")
        self.assertEqual(result.failure_reason, "SEMANTIC_INCOMPATIBILITY")

    def test_replay_actions_to_position_exposes_baseline_script_endpoint(self) -> None:
        end = replay_actions_to_position(
            start=(0, 0),
            actions=(1, 1),
            action_deltas={1: (0, 1)},
        )

        self.assertEqual(end, (0, 2))

    def test_predicate_inducer_selects_discriminative_region_over_exact_fallback(self) -> None:
        reference = TraceReference.synthetic([
            ((0.0, 0.0), 4),
            ((0.0, 3.0), None),
        ])
        option = OptionSegmenter().induce(reference)[0]

        enriched = PredicateInducer().enrich(option=option, reference=reference)

        self.assertEqual(enriched.selected_predicate_id, "option_000_001:pred_region")
        self.assertEqual(enriched.predicate_abstraction_level, 2)
        self.assertTrue(enriched.selected_predicate_expression_hash)
        self.assertEqual(enriched.termination_condition["type"], "region_membership")

    def test_nondiscriminative_high_abstraction_predicate_is_rejected(self) -> None:
        reference = TraceReference.synthetic([
            ((0.0, 0.0), 4),
            ((0.0, 1.0), 4),
            ((0.0, 2.0), None),
        ])
        option = PredicateInducer().enrich(
            option=OptionSegmenter().induce(reference)[0],
            reference=reference,
        )
        semantic = [
            candidate
            for candidate in option.termination_predicate_candidates
            if candidate.kind == "semantic_signature_compatibility"
        ][0]

        self.assertFalse(semantic.selected)
        self.assertEqual(semantic.rejection_reason, "classified_as_guard_not_termination")

    def test_selected_predicate_is_false_at_initiation_and_true_at_terminal(self) -> None:
        reference = TraceReference.synthetic([
            ((0.0, 0.0), 4),
            ((0.0, 3.0), None),
        ])
        option = PredicateInducer().enrich(
            option=OptionSegmenter().induce(reference)[0],
            reference=reference,
        )
        selected = [
            candidate
            for candidate in option.termination_predicate_candidates
            if candidate.selected
        ][0]

        self.assertFalse(selected.true_at_initiation)
        self.assertTrue(selected.true_at_terminal)
        self.assertTrue(selected.terminal_selective)

    def test_exact_predicate_fallback_remains_available_when_abstraction_fails(self) -> None:
        reference = TraceReference.synthetic([
            ((0.0, 0.0), 4),
            ((0.0, 1.0), 4),
            ((0.0, 2.0), None),
        ])
        option = OptionSegmenter().induce(reference)[0]

        enriched = PredicateInducer().enrich(
            option=option,
            reference=reference,
            validates_original=False,
        )

        self.assertEqual(enriched.selected_predicate_id, "option_000_002:pred_exact")
        self.assertEqual(enriched.predicate_abstraction_level, 1)

    def test_selected_predicate_expression_hash_is_stable(self) -> None:
        reference = TraceReference.synthetic([
            ((0.0, 0.0), 4),
            ((0.0, 1.0), 4),
            ((0.0, 2.0), None),
        ])
        option = PredicateInducer().enrich(
            option=OptionSegmenter().induce(reference)[0],
            reference=reference,
        )

        self.assertEqual(
            selected_predicate_expression_hash(option),
            option.selected_predicate_expression_hash,
        )

    def test_event_probe_generates_terminal_contact_predicate(self) -> None:
        option = self._option_candidate()
        records = (
            self._event_record(step=0, action=1, levels_before=0, levels_after=0),
            self._event_record(step=1, action=1, levels_before=0, levels_after=0),
            self._event_record(
                step=2,
                action=1,
                levels_before=0,
                levels_after=0,
                contact_signature=((8, 5, "shape-x"),),
            ),
        )

        candidates = EventPredicateEvidenceProbe().probe(
            option=option,
            event_records=records,
        )
        contact = [
            candidate for candidate in candidates
            if candidate.kind == "contact_or_boundary_event"
        ][0]

        self.assertEqual(contact.role, "termination")
        self.assertEqual(contact.abstraction_level, 3)
        self.assertTrue(contact.terminal_selective)
        self.assertEqual(contact.rejection_reason, None)
        self.assertEqual(
            contact.evidence[0]["field_path"],
            "interaction.contact_candidates|interaction.blocked",
        )

    def test_event_probe_classifies_semantic_signature_as_guard_not_termination(self) -> None:
        candidates = EventPredicateEvidenceProbe().probe(
            option=self._option_candidate(),
            event_records=(
                self._event_record(step=0, action=1, levels_before=0, levels_after=0),
                self._event_record(step=2, action=1, levels_before=0, levels_after=0),
            ),
        )
        guard = [
            candidate for candidate in candidates
            if candidate.kind == "semantic_signature_compatibility"
        ][0]

        self.assertEqual(guard.role, "guard")
        self.assertEqual(guard.rejection_reason, "classified_as_guard_not_termination")

    def test_event_probe_rejects_nonselective_higher_level_event(self) -> None:
        records = (
            self._event_record(step=0, action=1, levels_before=0, levels_after=1),
            self._event_record(step=1, action=1, levels_before=0, levels_after=0),
            self._event_record(step=2, action=1, levels_before=0, levels_after=1),
        )

        candidates = EventPredicateEvidenceProbe().probe(
            option=self._option_candidate(),
            event_records=records,
        )
        level_delta = [
            candidate for candidate in candidates
            if candidate.kind == "semantic_delta_event"
        ][0]

        self.assertEqual(level_delta.role, "rejected")
        self.assertEqual(level_delta.rejection_reason, "true_at_initiation")
        self.assertEqual(summarize_probe(candidates)["status"], "partial_ambiguous_evidence")

    def test_event_probe_does_not_promote_generic_animation_delta(self) -> None:
        records = (
            self._event_record(step=0, action=1, levels_before=0, levels_after=0),
            {
                **self._event_record(step=2, action=1, levels_before=0, levels_after=0),
                "object_deltas": [{
                    "object_id": 99,
                    "shape_changed": True,
                    "shape_transform": "unclassified_form_change",
                    "value_changed": False,
                    "area_delta": 2,
                    "translation": [0, 0],
                }],
            },
        )

        candidates = EventPredicateEvidenceProbe().probe(
            option=self._option_candidate(),
            event_records=records,
        )

        self.assertFalse([
            candidate for candidate in candidates
            if candidate.kind == "semantic_delta_event"
        ])
        self.assertEqual(
            summarize_probe(candidates)["status"],
            "no_clean_event_predicate_evidence",
        )

    def test_event_option_executor_stops_on_transition_predicate(self) -> None:
        option = self._option_candidate()
        terminal = self._event_record(
            step=2,
            action=1,
            levels_before=0,
            levels_after=0,
            created=(20,),
        )
        predicate = [
            candidate for candidate in EventPredicateEvidenceProbe().probe(
                option=option,
                event_records=(
                    self._event_record(step=0, action=1, levels_before=0, levels_after=0),
                    terminal,
                ),
            )
            if candidate.kind == "semantic_delta_event"
        ][0]

        result = EventOptionExecutor().execute(
            option=option,
            predicate=predicate,
            transition_record=terminal,
        )

        self.assertTrue(result.completed)
        self.assertTrue(result.termination_by_event_predicate)
        self.assertFalse(result.termination_by_trace_index)
        self.assertTrue(result.raw_trace_actions_masked)
        self.assertEqual(result.records[0]["predicate_mode"], "transition")
        self.assertEqual(result.records[0]["stop_reason"], "TERMINATION_PREDICATE_SATISFIED")

    def test_event_option_executor_refuses_when_expected_event_absent(self) -> None:
        option = self._option_candidate()
        terminal = self._event_record(
            step=2,
            action=1,
            levels_before=0,
            levels_after=0,
            created=(20,),
        )
        predicate = [
            candidate for candidate in EventPredicateEvidenceProbe().probe(
                option=option,
                event_records=(
                    self._event_record(step=0, action=1, levels_before=0, levels_after=0),
                    terminal,
                ),
            )
            if candidate.kind == "semantic_delta_event"
        ][0]
        absent = self._event_record(step=2, action=1, levels_before=0, levels_after=0)

        result = EventOptionExecutor().execute(
            option=option,
            predicate=predicate,
            transition_record=absent,
        )

        self.assertFalse(result.completed)
        self.assertEqual(result.failure_reason, "TERMINATION_EVENT_ABSENT")
        self.assertFalse(result.termination_by_trace_index)

    def test_event_option_executor_refuses_when_guard_fails(self) -> None:
        option = self._option_candidate()
        terminal = self._event_record(
            step=2,
            action=1,
            levels_before=0,
            levels_after=0,
            created=(20,),
        )
        predicate = [
            candidate for candidate in EventPredicateEvidenceProbe().probe(
                option=option,
                event_records=(
                    self._event_record(step=0, action=1, levels_before=0, levels_after=0),
                    terminal,
                ),
            )
            if candidate.kind == "semantic_delta_event"
        ][0]

        result = EventOptionExecutor().execute(
            option=option,
            predicate=predicate,
            transition_record=terminal,
            semantic_guard_valid=False,
        )

        self.assertFalse(result.completed)
        self.assertEqual(result.failure_reason, "SEMANTIC_GUARD_FAILED")
        self.assertTrue(result.semantic_guard_used)

    def test_self_option_inducer_marks_self_source_and_trace_dependency(self) -> None:
        option = SelfOptionInducer().induce(self._discovery_record())

        self.assertEqual(option.source, "self_exploration")
        self.assertEqual(option.trace_dependency_level, 4)
        self.assertEqual(option.selected_predicate_id, "self_option_000:pred_region")
        self.assertEqual(trace_leakage_findings(option), ())
        self.assertTrue(all(
            str(item["kind"]).startswith("self_observed_")
            for item in option.evidence
        ))

    def test_self_option_evaluator_masks_discovery_action_sequence(self) -> None:
        discovery = self._discovery_record()
        option = SelfOptionInducer().induce(discovery)

        result = SelfOptionEvaluator().execute(
            assay_id="A",
            option=option,
            discovery=discovery,
            blocked_edges=frozenset({(discovery.start_position, 3)}),
        )

        self.assertTrue(result.completed)
        self.assertFalse(result.used_discovery_action_script)
        self.assertFalse(result.raw_trace_actions_used)
        self.assertTrue(result.execution_plan_differs_from_discovery_sequence)
        self.assertNotEqual(result.plan, discovery.action_sequence)
        self.assertTrue(all(
            not record["used_discovery_action_script"]
            for record in result.records
        ))

    def test_self_option_nontriviality_requires_more_than_one_motor_step(self) -> None:
        discovery = self._discovery_record(action_sequence=(3,), actor_delta=(0, -5))
        option = SelfOptionInducer().induce(discovery)
        execution = SelfOptionExecution(
            assay_id="A",
            completed=True,
            status="COMPLETED",
            failure_reason=None,
            plan=(3,),
            records=(),
            termination_predicate_satisfied=True,
            execution_plan_differs_from_discovery_sequence=False,
            used_discovery_action_script=False,
            raw_trace_actions_used=False,
        )

        status = m6_status(
            option=option,
            discovery=discovery,
            executions=(execution,),
            invalid_refused=True,
            expectation_refused=True,
            trace_leakage=False,
            noise_promoted=False,
        )

        self.assertIn(status, {
            "partial_motor_primitive_only",
            "partial_self_script_only",
            "partial_discovered_but_unvalidated_option",
        })

    def test_self_option_variation_preserves_same_predicate_hash(self) -> None:
        discovery = self._discovery_record()
        option = SelfOptionInducer().induce(discovery)
        evaluator = SelfOptionEvaluator()

        first = evaluator.execute(
            assay_id="A",
            option=option,
            discovery=discovery,
            blocked_edges=frozenset({(discovery.start_position, 3)}),
        )
        second = evaluator.execute(
            assay_id="B",
            option=option,
            discovery=discovery,
            blocked_edges=frozenset({(discovery.start_position, 1), ((46, 31), 3)}),
        )

        self.assertTrue(first.completed)
        self.assertTrue(second.completed)
        self.assertNotEqual(first.plan, second.plan)
        self.assertEqual(
            selected_predicate_expression_hash(option),
            option.selected_predicate_expression_hash,
        )
        self.assertTrue(nontriviality_criteria(
            discovery=discovery,
            primary_execution=first,
            variation_execution=second,
        )["same_predicate_reexecutes_under_variation"])

    def test_self_option_refuses_invalid_initiation(self) -> None:
        discovery = self._discovery_record()
        option = SelfOptionInducer().induce(discovery)

        result = SelfOptionEvaluator().execute(
            assay_id="invalid",
            option=option,
            discovery=discovery,
            current_position=(0, 0),
        )

        self.assertFalse(result.completed)
        self.assertEqual(result.failure_reason, "INVALID_INITIATION")

    def test_trace_leakage_audit_catches_human_trace_provenance(self) -> None:
        option = SelfOptionInducer().induce(self._discovery_record())
        leaked = replace(
            option,
            evidence=option.evidence + ({"kind": "public_trace_segment"},),
        )

        self.assertIn("public_trace_segment", trace_leakage_findings(leaked))

    def test_option_effect_observation_serializes_deterministically(self) -> None:
        observation = self._option_effect_observation("effect_train_000")

        record = observation.to_record()

        self.assertEqual(record["record_id"], "effect_train_000")
        self.assertEqual(record["observed_semantic_delta"], {"kind": "none", "events": []})
        self.assertTrue(record["guards_preserved"])
        self.assertEqual(record["provenance"][0]["kind"], "self_observed_option_execution")

    def test_option_effect_model_predicts_before_heldout_execution(self) -> None:
        discovery = self._discovery_record()
        option = SelfOptionInducer().induce(discovery)
        train = self._option_effect_observation("effect_train_000", option=option, discovery=discovery)
        model = OptionEffectModel.train((train,))

        prediction = model.predict(
            prediction_id="pred_000",
            option=option,
            start_signature_class_hash=train.start_signature_class_hash,
        )

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertTrue(prediction.prediction_generated_before_execution)
        self.assertEqual(prediction.prediction_source, "option_effect_model")
        self.assertEqual(prediction.selected_predicate_expression_hash, selected_predicate_expression_hash(option))
        self.assertEqual(prediction.support_count, 1)
        self.assertEqual(prediction.total_count, 1)

    def test_model_prediction_verifies_heldout_positive(self) -> None:
        discovery = self._discovery_record()
        option = SelfOptionInducer().induce(discovery)
        train = self._option_effect_observation("effect_train_000", option=option, discovery=discovery)
        heldout = self._option_effect_observation(
            "effect_test_000",
            option=option,
            discovery=discovery,
            blocked_edges=frozenset({(discovery.start_position, 1), ((46, 31), 3)}),
        )
        model = OptionEffectModel.train((train,))
        prediction = model.predict(
            prediction_id="pred_000",
            option=option,
            start_signature_class_hash=train.start_signature_class_hash,
        )

        result = ModelPredictionVerifier().verify(
            verification_id="verify_000",
            prediction=prediction,
            observation=heldout,
            training_record_ids=(train.record_id,),
            heldout_positive=True,
        )

        self.assertEqual(result.result, "PREDICTION_VALID")
        self.assertTrue(result.heldout_positive)
        self.assertNotEqual(train.record_id, heldout.record_id)

    def test_model_prediction_rejects_training_record_as_heldout(self) -> None:
        discovery = self._discovery_record()
        option = SelfOptionInducer().induce(discovery)
        train = self._option_effect_observation("effect_train_000", option=option, discovery=discovery)
        prediction = OptionEffectModel.train((train,)).predict(
            prediction_id="pred_000",
            option=option,
            start_signature_class_hash=train.start_signature_class_hash,
        )

        result = ModelPredictionVerifier().verify(
            verification_id="verify_posthoc",
            prediction=prediction,
            observation=train,
            training_record_ids=(train.record_id,),
            heldout_positive=True,
        )

        self.assertEqual(result.result, "UNKNOWN")
        self.assertIn("posthoc_or_training_record_reuse", result.mismatch_fields)

    def test_model_prediction_reports_actor_delta_mismatch_fields(self) -> None:
        observation = self._option_effect_observation("effect_train_000")
        option = SelfOptionInducer().induce(self._discovery_record())
        prediction = OptionEffectModel.train((observation,)).predict(
            prediction_id="pred_000",
            option=option,
            start_signature_class_hash=observation.start_signature_class_hash,
        )
        mismatched = replace(
            observation,
            record_id="effect_bad_actor",
            observed_actor_delta={
                "kind": "moves_into_region",
                "target_predicate_id": option.selected_predicate_id,
                "delta": [0, 0],
            },
            termination_satisfied=False,
            stop_reason="ACTION_CONFLICT",
        )

        result = ModelPredictionVerifier().verify(
            verification_id="verify_bad_actor",
            prediction=prediction,
            observation=mismatched,
            training_record_ids=(observation.record_id,),
            heldout_positive=False,
        )

        self.assertEqual(result.result, "MODEL_PREDICTION_MISMATCH")
        self.assertIn("actor_delta", result.mismatch_fields)
        self.assertIn("termination_satisfied", result.mismatch_fields)

    def test_model_prediction_reports_guard_violation(self) -> None:
        observation = self._option_effect_observation("effect_train_000")
        option = SelfOptionInducer().induce(self._discovery_record())
        prediction = OptionEffectModel.train((observation,)).predict(
            prediction_id="pred_000",
            option=option,
            start_signature_class_hash=observation.start_signature_class_hash,
        )
        guard_failure = replace(
            observation,
            record_id="effect_bad_guard",
            observed_semantic_delta={
                "kind": "unexpected_noncontrolled_change",
                "events": [{"type": "unexpected_present"}],
            },
            guards_preserved=False,
            stop_reason="WORLD_DRIFT",
        )

        result = ModelPredictionVerifier().verify(
            verification_id="verify_bad_guard",
            prediction=prediction,
            observation=guard_failure,
            training_record_ids=(observation.record_id,),
            heldout_positive=False,
        )

        self.assertEqual(result.result, "GUARD_VIOLATION")
        self.assertIn("guards_preserved", result.mismatch_fields)

    def test_unknown_option_effect_key_returns_unknown(self) -> None:
        observation = self._option_effect_observation("effect_train_000")
        option = SelfOptionInducer().induce(self._discovery_record())
        model = OptionEffectModel.train((observation,))

        prediction = model.predict(
            prediction_id="pred_unknown",
            option=option,
            start_signature_class_hash="unknown",
        )
        result = ModelPredictionVerifier().verify(
            verification_id="verify_unknown",
            prediction=prediction,
            observation=replace(observation, record_id="effect_unknown"),
            training_record_ids=(observation.record_id,),
            heldout_positive=False,
        )

        self.assertIsNone(prediction)
        self.assertEqual(result.result, "UNKNOWN")

    def test_model_backed_prediction_provider_uses_model_source(self) -> None:
        observation = self._option_effect_observation("effect_train_000")
        option = SelfOptionInducer().induce(self._discovery_record())
        provider = ModelBackedPredictionProvider(OptionEffectModel.train((observation,)))

        prediction = provider.predict(
            prediction_id="pred_000",
            option=option,
            start_signature_class_hash=observation.start_signature_class_hash,
        )

        self.assertEqual(provider.source, "option_effect_model")
        self.assertIsNotNone(prediction)
        assert prediction is not None
        encoded = repr(prediction.to_record())
        self.assertNotIn("trace_corridor", encoded)
        self.assertNotIn("human_trace", encoded)

    def test_m7_status_flags_complete_and_failed_posthoc_cases(self) -> None:
        observation = self._option_effect_observation("effect_train_000")
        positive = ModelPredictionVerifier().verify(
            verification_id="verify_positive",
            prediction=OptionEffectModel.train((observation,)).predict(
                prediction_id="pred_000",
                option=SelfOptionInducer().induce(self._discovery_record()),
                start_signature_class_hash=observation.start_signature_class_hash,
            ),
            observation=replace(observation, record_id="effect_test_000"),
            training_record_ids=(observation.record_id,),
            heldout_positive=True,
        )
        negative = replace(
            positive,
            verification_id="verify_negative",
            result="MODEL_PREDICTION_MISMATCH",
            mismatch_fields=("actor_delta",),
            heldout_positive=False,
        )
        unknown = replace(
            positive,
            verification_id="verify_unknown",
            result="UNKNOWN",
            mismatch_fields=("prediction",),
            heldout_positive=False,
        )

        self.assertEqual(
            m7_status(
                observation_count=1,
                positive=positive,
                negatives=(negative,),
                unknown=unknown,
                trace_leakage=False,
                noise_promoted=False,
                posthoc_prediction=False,
            ),
            "complete_with_known_limits",
        )
        self.assertEqual(
            m7_status(
                observation_count=1,
                positive=positive,
                negatives=(negative,),
                unknown=unknown,
                trace_leakage=False,
                noise_promoted=False,
                posthoc_prediction=True,
            ),
            "failed_posthoc_prediction",
        )

    def test_option_applicability_state_class_is_not_exact_hash_keyed(self) -> None:
        state = self._m8_state_class(
            "S0",
            applicable_options=("self_option_000",),
            termination_predicates={"self_option_000:pred_region": False},
            exact_hashes=("frame_a",),
        )
        same_signature = replace(state, exact_observation_hashes=("frame_b",))
        changed_applicability = replace(state, applicable_options=("self_option_001",))

        record = state.to_record()

        self.assertEqual(record["kind"], "option_applicability_signature")
        self.assertEqual(record["applicable_options"], ["self_option_000"])
        self.assertEqual(record["exact_observation_hashes_debug_only"], ["frame_a"])
        self.assertEqual(state.signature_hash, same_signature.signature_hash)
        self.assertNotEqual(state.signature_hash, changed_applicability.signature_hash)

    def test_option_chain_verifies_effect_before_releasing_next_option(self) -> None:
        fixture = self._m8_chain_fixture()

        verification = OptionChainVerifier().verify(
            verification_id="verify_chain_000",
            prediction=fixture["prediction"],
            observed_effects=(fixture["heldout_a"], fixture["heldout_b"]),
            observed_next_state_classes=(fixture["state_1"], fixture["state_2"]),
            training_record_ids_by_link=(
                (fixture["train_a"].record_id,),
                (fixture["train_b"].record_id,),
            ),
        )

        self.assertEqual(verification.result, "CHAIN_VERIFIED")
        self.assertEqual(verification.executed_link_count, 2)
        self.assertTrue(verification.link_verifications[0].next_link_released)
        self.assertTrue(same_option_chain_is_honest(
            chain_prediction=fixture["prediction"],
            observed_state_classes=(fixture["state_1"], fixture["state_2"]),
        ))
        self.assertEqual(
            m8_status(
                chain_prediction=fixture["prediction"],
                chain_verification=verification,
                effect_mismatch_gate_exercised=True,
                next_applicability_gate_exercised=True,
                same_option_family_chain=False,
                distinct_state_classes=True,
                posthoc_link_prediction=False,
            ),
            "complete_with_known_limits",
        )

    def test_option_chain_stops_when_link_effect_mismatches(self) -> None:
        fixture = self._m8_chain_fixture()
        mismatched_effect = replace(
            fixture["heldout_a"],
            record_id="effect_test_a_bad_actor",
            observed_actor_delta={
                "kind": "moves_into_region",
                "target_predicate_id": fixture["option_a"].selected_predicate_id,
                "delta": [0, 0],
            },
            termination_satisfied=False,
            stop_reason="ACTION_CONFLICT",
        )

        verification = OptionChainVerifier().verify(
            verification_id="verify_chain_bad_effect",
            prediction=fixture["prediction"],
            observed_effects=(mismatched_effect, fixture["heldout_b"]),
            observed_next_state_classes=(fixture["state_1"], fixture["state_2"]),
            training_record_ids_by_link=(
                (fixture["train_a"].record_id,),
                (fixture["train_b"].record_id,),
            ),
        )

        self.assertEqual(verification.result, "LINK_EFFECT_MISMATCH")
        self.assertEqual(verification.executed_link_count, 1)
        self.assertEqual(verification.stopped_at_link, 0)
        self.assertFalse(verification.link_verifications[0].next_link_released)

    def test_option_chain_stops_before_next_link_without_applicability(self) -> None:
        fixture = self._m8_chain_fixture()
        blocked_next_state = self._m8_state_class(
            "S1",
            applicable_options=(),
            termination_predicates={
                fixture["option_a"].selected_predicate_id: True,
                fixture["option_b"].selected_predicate_id: False,
            },
        )

        verification = OptionChainVerifier().verify(
            verification_id="verify_chain_bad_applicability",
            prediction=fixture["prediction"],
            observed_effects=(fixture["heldout_a"], fixture["heldout_b"]),
            observed_next_state_classes=(blocked_next_state, fixture["state_2"]),
            training_record_ids_by_link=(
                (fixture["train_a"].record_id,),
                (fixture["train_b"].record_id,),
            ),
        )

        self.assertEqual(verification.result, "NEXT_OPTION_NOT_APPLICABLE")
        self.assertEqual(verification.executed_link_count, 1)
        self.assertEqual(verification.stopped_at_link, 0)
        self.assertFalse(verification.link_verifications[0].next_link_released)

    def test_m9_generates_scored_chain_candidates_with_rejected_alternative(self) -> None:
        fixture = self._m8_chain_fixture()

        candidates = generate_chain_candidates(
            graph=fixture["graph"],
            start_state_class_id=fixture["state_0"].state_class_id,
            option_a=fixture["option_a"],
            option_b=fixture["option_b"],
        )
        ledger = SelectionPolicy().select(
            round_index=0,
            state_class_id=fixture["state_0"].state_class_id,
            candidates=candidates,
            attempt_memory=AttemptMemory(),
        )
        by_id = {candidate.candidate_id: candidate for candidate in ledger.candidates}

        self.assertEqual(set(by_id), {"option_000", "option_001", "chain_000"})
        self.assertFalse(by_id["option_001"].applicable)
        self.assertEqual(by_id["option_001"].rejection_reason, "not_applicable_in_current_state")
        self.assertEqual(ledger.selected_candidate_id, "chain_000")
        self.assertTrue(ledger.selection_generated_before_execution)
        self.assertTrue(ledger.candidate_scores_generated_before_execution)
        self.assertIn("new_option_enabled", by_id["chain_000"].score_terms or {})

    def test_m9_selection_uses_score_not_candidate_order(self) -> None:
        fixture = self._m8_chain_fixture()
        candidates = generate_chain_candidates(
            graph=fixture["graph"],
            start_state_class_id=fixture["state_0"].state_class_id,
            option_a=fixture["option_a"],
            option_b=fixture["option_b"],
        )
        policy = SelectionPolicy()

        first = policy.select(
            round_index=0,
            state_class_id="S0",
            candidates=candidates,
            attempt_memory=AttemptMemory(),
        )
        shuffled = policy.select(
            round_index=0,
            state_class_id="S0",
            candidates=tuple(reversed(candidates)),
            attempt_memory=AttemptMemory(),
        )

        self.assertEqual(first.selected_candidate_id, shuffled.selected_candidate_id)

    def test_m9_score_change_changes_selection_when_ranking_changes(self) -> None:
        low = ChainCandidate(
            candidate_id="low",
            candidate_type="single_option",
            option_ids=("o0",),
            start_state_class_id="S0",
            predicted_terminal_state_class_id="S1",
            predicted_progress_features={
                "level_completion": False,
                "stable_semantic_delta": False,
                "new_option_enabled": False,
                "new_state_class": True,
                "uncertainty_reduction": False,
            },
            cost_features={"predicted_action_cost": 4, "empirical_consistency": 1.0},
            risk_features={"guard_risk": 0},
            provenance=(),
        )
        high = replace(
            low,
            candidate_id="high",
            predicted_progress_features={
                **low.predicted_progress_features,
                "stable_semantic_delta": True,
            },
        )
        policy = SelectionPolicy()

        before = policy.select(
            round_index=0,
            state_class_id="S0",
            candidates=(low, high),
            attempt_memory=AttemptMemory(),
        )
        after = policy.select(
            round_index=0,
            state_class_id="S0",
            candidates=(replace(low, predicted_progress_features={
                **low.predicted_progress_features,
                "level_completion": True,
            }), high),
            attempt_memory=AttemptMemory(),
        )

        self.assertEqual(before.selected_candidate_id, "high")
        self.assertEqual(after.selected_candidate_id, "low")

    def test_m9_failed_candidate_penalty_changes_same_state_reselection(self) -> None:
        fixture = self._m8_chain_fixture()
        candidates = generate_chain_candidates(
            graph=fixture["graph"],
            start_state_class_id=fixture["state_0"].state_class_id,
            option_a=fixture["option_a"],
            option_b=fixture["option_b"],
        )
        memory = AttemptMemory()
        policy = SelectionPolicy()
        before = policy.select(
            round_index=0,
            state_class_id="S0",
            candidates=candidates,
            attempt_memory=memory,
        )

        update = memory.record_failure(
            candidate_id=before.selected_candidate_id or "",
            reason="NEXT_OPTION_NOT_APPLICABLE",
        )
        after = policy.select(
            round_index=1,
            state_class_id="S0",
            candidates=candidates,
            attempt_memory=memory,
        )
        before_failed = next(
            candidate for candidate in before.candidates
            if candidate.candidate_id == before.selected_candidate_id
        )
        after_failed = next(
            candidate for candidate in after.candidates
            if candidate.candidate_id == before.selected_candidate_id
        )

        self.assertEqual(update["update_type"], "FAILED_CANDIDATE_PENALTY")
        self.assertLess(after_failed.score, before_failed.score)
        self.assertNotEqual(after.selected_candidate_id, before.selected_candidate_id)

    def test_m9_outcome_records_m8_verification_and_failure_update(self) -> None:
        fixture = self._m8_chain_fixture()
        candidates = generate_chain_candidates(
            graph=fixture["graph"],
            start_state_class_id=fixture["state_0"].state_class_id,
            option_a=fixture["option_a"],
            option_b=fixture["option_b"],
        )
        selected = next(candidate for candidate in candidates if candidate.candidate_id == "chain_000")
        blocked_next_state = replace(fixture["state_1"], applicable_options=())
        verification = OptionChainVerifier().verify(
            verification_id="verify_chain_bad_applicability",
            prediction=fixture["prediction"],
            observed_effects=(fixture["heldout_a"], fixture["heldout_b"]),
            observed_next_state_classes=(blocked_next_state, fixture["state_2"]),
            training_record_ids_by_link=(
                (fixture["train_a"].record_id,),
                (fixture["train_b"].record_id,),
            ),
        )

        outcome = outcome_from_chain_verification(
            attempt_id="attempt_bad_000",
            candidate=selected,
            verification=verification,
        )

        self.assertEqual(outcome.verification_result, "NEXT_OPTION_NOT_APPLICABLE")
        self.assertEqual(outcome.model_updates[0]["update_type"], "FAILED_CANDIDATE_OBSERVED")

    def test_m9_noise_like_visual_delta_gets_no_progress_credit(self) -> None:
        candidate = ChainCandidate(
            candidate_id="noise",
            candidate_type="single_option",
            option_ids=("o0",),
            start_state_class_id="S0",
            predicted_terminal_state_class_id="S1",
            predicted_progress_features={
                "level_completion": False,
                "stable_semantic_delta": False,
                "new_option_enabled": False,
                "new_state_class": False,
                "uncertainty_reduction": False,
                "volatile_visual_delta": True,
            },
            cost_features={"predicted_action_cost": 1, "empirical_consistency": 1.0},
            risk_features={"guard_risk": 0},
            provenance=(),
        )

        scored = ProgressModel().score_candidate(
            candidate=candidate,
            attempt_memory=AttemptMemory(),
        )

        self.assertEqual((scored.score_terms or {})["noise_as_progress"], 0.0)

    def test_m9_status_labels_complete_and_blind_retry_failure(self) -> None:
        common = dict(
            candidate_count=3,
            selected_by_score=True,
            learning_update_count=1,
            second_selection_reflected_update=True,
            negative_candidate_deprioritized=True,
            hardcoded_selection=False,
            prediction_before_selection=True,
            outcome_leakage=False,
            blind_retry=False,
            oracle_signal=False,
            trace_leakage=False,
            noise_as_progress=False,
        )

        self.assertEqual(m9_status(**common), "complete_with_known_limits")
        self.assertEqual(
            m9_status(**{**common, "blind_retry": True}),
            "failed_blind_retry_after_failure",
        )

    def test_m10_hypotheses_initialize_from_m9_priors(self) -> None:
        hypotheses = ProgressHypothesisSet.from_m9_priors()

        self.assertEqual(hypotheses.hypothesis("new_option_enabled").prior_weight, 25.0)
        self.assertEqual(hypotheses.hypothesis("new_state_class").posterior_weight, 15.0)
        self.assertEqual(hypotheses.hypothesis("volatile_visual_delta").status, "NEUTRAL")

    def test_m10_positive_outcome_updates_feature_weights(self) -> None:
        candidate = self._m10_candidate(
            "candidate_positive",
            new_option_enabled=True,
            new_state_class=True,
        )
        outcome = outcome_record_from_candidate(
            outcome_id="outcome_positive",
            candidate=candidate,
            observed_features={
                "new_option_enabled": True,
                "new_state_class": True,
                "prediction_validated": True,
            },
            stop_reason="CHAIN_VERIFIED",
            provenance=({"kind": "public_option_chain_outcome"},),
        )

        updated, updates = ProgressHypothesisSet.from_m9_priors().update_from_outcomes((outcome,))

        self.assertGreater(
            updated.hypothesis("new_option_enabled").posterior_weight,
            updated.hypothesis("new_option_enabled").prior_weight,
        )
        self.assertEqual(updated.hypothesis("new_option_enabled").support_count, 1)
        self.assertEqual(updated.hypothesis("new_option_enabled").status, "ACTIVE")
        self.assertIn("new_option_enabled", {update.feature_id for update in updates})

    def test_m10_noise_outcome_suppresses_volatile_visual_delta(self) -> None:
        candidate = self._m10_candidate(
            "candidate_noise",
            volatile_visual_delta=True,
        )
        outcome = outcome_record_from_candidate(
            outcome_id="outcome_noise",
            candidate=candidate,
            observed_features={
                "new_option_enabled": False,
                "new_state_class": False,
                "stable_semantic_delta": False,
                "action_conflict_or_next_not_applicable": True,
            },
            stop_reason="NO_STABLE_PROGRESS",
            provenance=({"kind": "public_noise_suppression_assay"},),
        )

        updated, _updates = ProgressHypothesisSet.from_m9_priors().update_from_outcomes((outcome,))
        noise = updated.hypothesis("volatile_visual_delta")

        self.assertEqual(noise.status, "SUPPRESSED")
        self.assertLessEqual(noise.posterior_weight, 0)

    def test_m10_heldout_candidate_score_changes_from_feature_learning(self) -> None:
        heldout = self._m10_candidate(
            "heldout_positive_feature",
            new_option_enabled=True,
            new_state_class=True,
        )
        observed = self._m10_candidate(
            "observed_positive_feature",
            new_option_enabled=True,
            new_state_class=True,
        )
        outcome = outcome_record_from_candidate(
            outcome_id="outcome_positive",
            candidate=observed,
            observed_features={
                "new_option_enabled": True,
                "new_state_class": True,
                "prediction_validated": True,
            },
            stop_reason="CHAIN_VERIFIED",
            provenance=({"kind": "public_option_chain_outcome"},),
        )
        priors = ProgressHypothesisSet.from_m9_priors()
        learned, _updates = priors.update_from_outcomes((outcome,))

        before = LearnedProgressModel(priors).score_candidate(
            candidate=heldout,
            attempt_memory=AttemptMemory(),
        )
        after = LearnedProgressModel(learned).score_candidate(
            candidate=heldout,
            attempt_memory=AttemptMemory(),
        )

        self.assertGreater(after.score, before.score)

    def test_m10_same_state_selection_changes_due_to_progress_learning(self) -> None:
        stable_prior = self._m10_candidate(
            "candidate_stable_prior",
            stable_semantic_delta=True,
            action_cost=5,
        )
        heldout = self._m10_candidate(
            "heldout_positive_feature",
            new_option_enabled=True,
            new_state_class=True,
            action_cost=4,
        )
        observed = self._m10_candidate(
            "observed_positive_feature",
            new_option_enabled=True,
            new_state_class=True,
            action_cost=4,
        )
        outcome = outcome_record_from_candidate(
            outcome_id="outcome_positive",
            candidate=observed,
            observed_features={
                "new_option_enabled": True,
                "new_state_class": True,
                "prediction_validated": True,
            },
            stop_reason="CHAIN_VERIFIED",
            provenance=({"kind": "public_option_chain_outcome"},),
        )
        priors = ProgressHypothesisSet.from_m9_priors()
        learned, _updates = priors.update_from_outcomes((outcome,))
        candidates = (stable_prior, heldout)

        before = SelectionPolicy(progress_model=LearnedProgressModel(priors)).select(
            round_index=0,
            state_class_id="S0",
            candidates=candidates,
            attempt_memory=AttemptMemory(),
        )
        after = SelectionPolicy(progress_model=LearnedProgressModel(learned)).select(
            round_index=1,
            state_class_id="S0",
            candidates=candidates,
            attempt_memory=AttemptMemory(),
        )

        self.assertEqual(before.selected_candidate_id, "candidate_stable_prior")
        self.assertEqual(after.selected_candidate_id, "heldout_positive_feature")

    def test_m10_candidate_memory_and_feature_learning_are_distinct(self) -> None:
        candidate = self._m10_candidate("candidate_positive", new_option_enabled=True)
        outcome = outcome_record_from_candidate(
            outcome_id="outcome_positive",
            candidate=candidate,
            observed_features={"new_option_enabled": True},
            stop_reason="OPTION_VERIFIED",
            provenance=({"kind": "public_option_outcome"},),
        )
        _learned, updates = ProgressHypothesisSet.from_m9_priors().update_from_outcomes((outcome,))
        memory_update = AttemptMemory().record_failure(
            candidate_id="chain_000",
            reason="NEXT_OPTION_NOT_APPLICABLE",
        )
        ledger = ProgressLearningLedger(
            outcomes=(outcome,),
            feature_updates=updates,
            candidate_memory_updates=(memory_update,),
        )
        record = ledger.to_record()

        self.assertTrue(record["candidate_memory_separate_from_feature_learning"])
        self.assertEqual(record["feature_updates"][0]["feature_id"], "new_option_enabled")
        self.assertEqual(record["candidate_memory_updates"][0]["update_type"], "FAILED_CANDIDATE_PENALTY")

    def test_m10_status_labels_complete_and_feature_generalization_failure(self) -> None:
        common = dict(
            hypothesis_count=5,
            outcome_count=2,
            feature_update_count=2,
            heldout_score_changed=True,
            selection_changed_due_to_progress_learning=True,
            candidate_memory_only_explanation=False,
            no_feature_generalization=False,
            outcome_leakage=False,
            noise_as_progress=False,
            candidate_memory_confused=False,
            trace_leakage=False,
            oracle_signal=False,
        )

        self.assertEqual(m10_status(**common), "complete_with_known_limits")
        self.assertEqual(
            m10_status(**{**common, "no_feature_generalization": True}),
            "failed_no_feature_generalization",
        )

    def test_m106_classifier_maps_live_position_to_known_state(self) -> None:
        fixture = self._m106_live_graph_fixture()
        option_a = fixture["option_a"]
        option_b = fixture["option_b"]
        position = tuple(option_a.termination_condition["center"])

        binding = LiveObservationStateClassifier().classify(
            observation_hash="obs_after_a",
            graph=fixture["graph"],
            options=(option_a, option_b),
            controlled_position=position,
            controlled_position_source="live_public_event_tracker",
        )

        self.assertEqual(binding.classification, "MATCHED_KNOWN_STATE_CLASS")
        self.assertEqual(binding.state_class_id, "S1")
        self.assertEqual(binding.applicable_options, ("self_option_001",))
        self.assertTrue(binding.predicate_truths[option_a.selected_predicate_id])
        self.assertFalse(binding.exact_hash_used_for_match)

    def test_m106_unknown_predicate_vector_does_not_fabricate_match(self) -> None:
        fixture = self._m106_live_graph_fixture()
        graph = OptionApplicabilityGraph(
            state_classes=(fixture["state_1"],),
            edges=fixture["graph"].edges,
        )
        binding = LiveObservationStateClassifier().classify(
            observation_hash="unknown",
            graph=graph,
            options=(fixture["option_a"], fixture["option_b"]),
            controlled_position=(999, 999),
            controlled_position_source="live_public_event_tracker",
        )

        self.assertEqual(binding.classification, "UNKNOWN_STATE_CLASS")
        self.assertFalse(binding.replannable)

    def test_m106_ambiguous_duplicate_truth_vector_is_reported(self) -> None:
        fixture = self._m106_live_graph_fixture()
        duplicate = replace(fixture["state_1"], state_class_id="S1_DUP")
        graph = OptionApplicabilityGraph(
            state_classes=(fixture["state_0"], fixture["state_1"], duplicate),
            edges=fixture["graph"].edges,
        )

        binding = LiveObservationStateClassifier().classify(
            observation_hash="ambiguous",
            graph=graph,
            options=(fixture["option_a"], fixture["option_b"]),
            controlled_position=tuple(fixture["option_a"].termination_condition["center"]),
            controlled_position_source="live_public_event_tracker",
        )

        self.assertEqual(binding.classification, "AMBIGUOUS_STATE_CLASS")

    def test_m106_candidate_generator_uses_classified_s1_state(self) -> None:
        fixture = self._m106_live_graph_fixture()
        binding = LiveObservationStateClassifier().classify(
            observation_hash="obs_after_a",
            graph=fixture["graph"],
            options=(fixture["option_a"], fixture["option_b"]),
            controlled_position=tuple(fixture["option_a"].termination_condition["center"]),
            controlled_position_source="live_public_event_tracker",
        )

        candidates = LiveCandidateGenerator().generate(
            graph=fixture["graph"],
            binding=binding,
            options=(fixture["option_a"], fixture["option_b"]),
            action_plans={
                "self_option_000": (3, 1),
                "self_option_001": (3, 3),
            },
        )

        self.assertEqual([candidate.candidate_id for candidate in candidates], ["option_001"])
        self.assertEqual(candidates[0].start_state_class_id, "S1")

    def test_m106_learned_progress_weights_affect_live_selection(self) -> None:
        fixture = self._m106_live_graph_fixture()
        binding = LiveObservationStateClassifier().classify(
            observation_hash="obs_start",
            graph=fixture["graph"],
            options=(fixture["option_a"], fixture["option_b"]),
            controlled_position=tuple(fixture["option_a"].initiation_condition["actor_position"]),
            controlled_position_source="option_state_binding",
        )
        candidates = LiveCandidateGenerator().generate(
            graph=fixture["graph"],
            binding=binding,
            options=(fixture["option_a"], fixture["option_b"]),
            action_plans={
                "self_option_000": (3, 1),
                "self_option_001": (3, 3),
            },
        )
        observed = self._m10_candidate(
            "observed_positive_feature",
            new_option_enabled=True,
            new_state_class=True,
        )
        outcome = outcome_record_from_candidate(
            outcome_id="outcome_positive",
            candidate=observed,
            observed_features={
                "new_option_enabled": True,
                "new_state_class": True,
                "prediction_validated": True,
            },
            stop_reason="CHAIN_VERIFIED",
            provenance=({"kind": "public_option_chain_outcome"},),
        )
        learned, _updates = ProgressHypothesisSet.from_m9_priors().update_from_outcomes((outcome,))

        selection = SelectionPolicy(progress_model=LearnedProgressModel(learned)).select(
            round_index=0,
            state_class_id="S0",
            candidates=candidates,
            attempt_memory=AttemptMemory(),
        )
        selected = next(candidate for candidate in selection.candidates if candidate.selected)

        self.assertEqual(selection.selected_candidate_id, "chain_000")
        self.assertEqual((selected.score_terms or {})["new_option_enabled"], 35.0)

    def test_m106_actuator_executes_selected_public_action(self) -> None:
        env = _FakeLiveEnv([
            _FakeLiveFrame([[[0, 1, 0]]]),
            _FakeLiveFrame([[[0, 0, 1]]]),
        ])
        current = public_observation(env.reset())
        actuator = LiveCandidateActuator()
        actuator.start(current)
        candidate = ChainCandidate(
            candidate_id="option_000",
            candidate_type="single_option",
            option_ids=("self_option_000",),
            start_state_class_id="S0",
            predicted_terminal_state_class_id="S1",
            predicted_progress_features={
                "level_completion": False,
                "stable_semantic_delta": False,
                "new_option_enabled": True,
                "new_state_class": True,
                "uncertainty_reduction": False,
                "volatile_visual_delta": False,
            },
            cost_features={"predicted_action_cost": 1, "empirical_consistency": 1.0},
            risk_features={"guard_risk": 0},
            provenance=(),
        )

        result, _following = actuator.execute_first_link(
            env=env,
            current_observation=current,
            candidate=candidate,
            action_plans={"self_option_000": (4,)},
            current_controlled_position=(0, 1),
            step_offset=0,
            remaining_budget=10,
        )

        self.assertTrue(result.selected_candidate_sent_to_actuator)
        self.assertEqual(result.public_action_plan, (4,))
        self.assertEqual(result.actions_executed, 1)
        self.assertEqual(result.controlled_position_source, "live_public_event_tracker")

    def test_m106_status_labels_complete_and_counterfactual_partial(self) -> None:
        common = dict(
            classification_count=2,
            post_execution_classified=True,
            candidate_generation_from_live_state=True,
            learned_progress_used=True,
            candidate_sent_to_actuator=True,
            actions_executed=2,
            replan_cycles_completed=2,
            valid_stop_reason=False,
            counterfactual_only=False,
            selected_candidate_not_executed=False,
            trace_leakage=False,
            oracle_signal=False,
            live_observation_unclassified=False,
            no_live_candidate_generation=False,
        )

        self.assertEqual(m106_status(**common), "complete_with_known_limits")
        self.assertEqual(
            m106_status(**{**common, "counterfactual_only": True}),
            "partial_counterfactual_selection_only",
        )

    def test_m11_saturation_detector_detects_local_cycle(self) -> None:
        result = SaturationDetector().detect(
            recent_state_sequence=("S0", "S1", "S0"),
            recent_selected_candidates=("chain_000", "option_000", "chain_000"),
            candidate_scores_stalled=True,
            no_new_state_class_cycles=3,
        )

        self.assertTrue(result.saturation_detected)
        self.assertEqual(result.reason, "local_cycle_detected")

    def test_m11_frontier_detector_generates_scored_probe_candidates(self) -> None:
        fixture = self._m106_live_graph_fixture()
        binding = LiveObservationStateClassifier().classify(
            observation_hash="obs_start",
            graph=fixture["graph"],
            options=(fixture["option_a"], fixture["option_b"]),
            controlled_position=tuple(fixture["option_a"].initiation_condition["actor_position"]),
            controlled_position_source="option_state_binding",
        )

        probes = FrontierDetector().generate(
            binding=binding,
            available_actions=(1, 2, 3, 4),
            known_action_sequences=((3, 1), (1, 3)),
        )
        scored = FrontierProbeScorer().select(tuple(reversed(probes)))
        selected = next(probe for probe in scored if probe.selected)

        self.assertGreaterEqual(len(scored), 2)
        self.assertIsNotNone(selected.score_terms)
        self.assertEqual((selected.score_terms or {})["fixed_order_bonus"], 0.0)
        self.assertEqual((selected.score_terms or {})["noise_as_progress"], 0.0)

    def test_m11_probe_outcome_no_progress_is_not_promoted(self) -> None:
        execution = FrontierProbeExecution(
            probe_id="probe_000",
            action_sequence=(3, 1),
            records=(),
            actions_executed=2,
            start_observation_hash="before",
            final_observation_hash="after",
            start_position=(10, 10),
            final_position=(10, 10),
            controlled_position_source="live_public_event_tracker",
            semantic_guards_preserved=True,
            stop_reason=None,
        )

        outcome = ProbeOutcomeClassifier().classify(
            execution=execution,
            known_options=(),
        )

        self.assertEqual(outcome.result, "NO_PROGRESS")
        self.assertFalse(outcome.promotable)
        self.assertFalse(outcome.noise_rejected[0]["promoted"])

    def test_m11_stable_probe_transition_expands_graph_and_candidates(self) -> None:
        fixture = self._m106_live_graph_fixture()
        source_state = fixture["graph"].state("S0")
        assert source_state is not None
        execution = FrontierProbeExecution(
            probe_id="probe_001",
            action_sequence=(3, 1),
            records=(),
            actions_executed=2,
            start_observation_hash="before",
            final_observation_hash="after",
            start_position=(32, 21),
            final_position=(42, 21),
            controlled_position_source="live_public_event_tracker",
            semantic_guards_preserved=True,
            stop_reason=None,
        )
        outcome = ProbeOutcomeClassifier().classify(
            execution=execution,
            known_options=(fixture["option_a"], fixture["option_b"]),
        )

        expansion = LiveOptionInducer().promote(
            expansion_index=0,
            source_state=source_state,
            source_position=(32, 21),
            outcome=outcome,
            execution=execution,
        )
        assert expansion is not None
        expanded_graph = apply_graph_expansion(
            graph=fixture["graph"],
            expansion=expansion,
        )
        expanded_options = (
            fixture["option_a"],
            fixture["option_b"],
            expansion.option,
        )
        binding = LiveObservationStateClassifier().classify(
            observation_hash="obs_start",
            graph=expanded_graph,
            options=expanded_options,
            controlled_position=(32, 21),
            controlled_position_source="option_state_binding",
        )
        candidates = LiveCandidateGenerator().generate(
            graph=expanded_graph,
            binding=binding,
            options=expanded_options,
            action_plans={
                "self_option_000": (3, 1),
                "self_option_001": (3, 3),
                expansion.option.option_id: execution.action_sequence,
            },
        )

        self.assertEqual(expansion.option.source, "frontier_probe")
        self.assertIn(expansion.option.option_id, expanded_graph.state("S0").applicable_options)
        self.assertTrue(any(expansion.option.option_id in candidate.option_ids for candidate in candidates))

    def test_m11_status_labels_complete_and_noise_failure(self) -> None:
        common = dict(
            saturation_detected=True,
            probe_count=2,
            probe_score_driven=True,
            probe_executed=True,
            promoted_count=1,
            noise_promoted=False,
            expanded_graph_used=True,
            live_loop_resumed_after_expansion=True,
            honest_frontier_available=True,
            fixed_probe_order=False,
            trace_leakage=False,
            oracle_signal=False,
            unclassified_loop=False,
        )

        self.assertEqual(m11_status(**common), "complete_with_known_limits")
        self.assertEqual(
            m11_status(**{**common, "noise_promoted": True}),
            "failed_noise_promoted_to_option",
        )

    def test_m116_detects_and_slices_level_transition_inside_probe(self) -> None:
        execution = FrontierProbeExecution(
            probe_id="probe_002",
            action_sequence=(1, 4),
            records=(
                {
                    "step": 21,
                    "action": 1,
                    "executed": True,
                    "levels_before": 0,
                    "levels_after": 1,
                },
                {
                    "step": 22,
                    "action": 4,
                    "executed": True,
                    "levels_before": 1,
                    "levels_after": 1,
                },
            ),
            actions_executed=2,
            start_observation_hash="before",
            final_observation_hash="after",
            start_position=(36, 12),
            final_position=(34, 61),
            controlled_position_source="live_public_event_tracker",
            semantic_guards_preserved=True,
            stop_reason=None,
        )

        transition = LevelTransitionDetector().detect(execution=execution)
        assert transition is not None
        sliced = BoundaryProbeSlicer().slice(
            execution=execution,
            transition=transition,
        )

        self.assertEqual(transition.trigger_action, 1)
        self.assertEqual(transition.public_step, 21)
        self.assertEqual(sliced.pre_transition_actions, (1,))
        self.assertEqual(sliced.post_transition_actions, (4,))
        self.assertFalse(sliced.cross_level_option_training_allowed)
        self.assertFalse(sliced.cross_level_delta_promoted)

    def test_m116_level_context_seeds_empty_level_scoped_graph(self) -> None:
        context = LevelContextManager().create_context(
            level_index=1,
            observation_hash="obs_level_1",
        )

        self.assertEqual(context.context_id, "level_1_context")
        self.assertEqual(context.initial_state_class_id, "L1_S0")
        self.assertEqual(len(context.graph.state_classes), 1)
        self.assertEqual(context.graph.edges, ())
        self.assertIn("action_delta_model", context.carried_transferables)
        self.assertIn("previous_level_spatial_options", context.invalidated_level_local)

    def test_m116_status_rejects_cross_level_delta_promotion(self) -> None:
        common = dict(
            transition_detected=True,
            probe_split=True,
            cross_level_delta_promoted=False,
            new_context_created=True,
            transferables_carried=True,
            stale_spatial_invalidated=True,
            post_transition_state_bound=True,
            post_transition_probe_count=2,
            post_transition_actions_executed=1,
            trace_leakage=False,
            oracle_signal=False,
        )

        self.assertEqual(m116_status(**common), "complete_with_known_limits")
        self.assertEqual(
            m116_status(**{**common, "cross_level_delta_promoted": True}),
            "failed_cross_level_delta_promoted",
        )

    def test_m116_terminal_evidence_records_transition_trigger(self) -> None:
        execution = FrontierProbeExecution(
            probe_id="probe_002",
            action_sequence=(1,),
            records=({
                "step": 21,
                "action": 1,
                "executed": True,
                "levels_before": 0,
                "levels_after": 1,
            },),
            actions_executed=1,
            start_observation_hash="before",
            final_observation_hash="after",
            start_position=(36, 12),
            final_position=(33, 29),
            controlled_position_source="live_public_event_tracker",
            semantic_guards_preserved=True,
            stop_reason=None,
        )
        transition = LevelTransitionDetector().detect(execution=execution)
        assert transition is not None

        terminal = terminal_evidence_from_transition(
            transition=transition,
            pre_terminal_state_class="S8",
        )

        self.assertTrue(terminal.level_completed)
        self.assertEqual(terminal.trigger_action, 1)
        self.assertEqual(terminal.pre_terminal_state_class, "S8")

    def test_m117_template_extraction_retains_radius_but_removes_center(self) -> None:
        fixture = self._m106_live_graph_fixture()
        source_state = fixture["graph"].state("S0")
        assert source_state is not None
        execution = FrontierProbeExecution(
            probe_id="probe_prior",
            action_sequence=(4, 1),
            records=(),
            actions_executed=2,
            start_observation_hash="level0_before",
            final_observation_hash="level0_after",
            start_position=(32, 21),
            final_position=(42, 21),
            controlled_position_source="live_public_event_tracker",
            semantic_guards_preserved=True,
            stop_reason=None,
        )
        outcome = ProbeOutcomeClassifier().classify(
            execution=execution,
            known_options=(fixture["option_a"], fixture["option_b"]),
        )
        expansion = LiveOptionInducer().promote(
            expansion_index=0,
            source_state=source_state,
            source_position=(32, 21),
            outcome=outcome,
            execution=execution,
        )
        assert expansion is not None

        template = OptionTemplateExtractor().extract(
            options=(expansion.option,),
            source_level_index=0,
        )[0]

        self.assertEqual(template.predicate_family, "region_membership")
        self.assertEqual(template.radius, 2)
        self.assertNotIn("center", template.expression)
        self.assertNotIn("target_position", template.expression)
        self.assertFalse(template.contains_level_local_fact())

    def test_m117_instantiation_uses_fresh_level_context_not_prior_facts(self) -> None:
        fixture = self._m106_live_graph_fixture()
        source_state = fixture["graph"].state("S0")
        assert source_state is not None
        prior_execution = FrontierProbeExecution(
            probe_id="probe_prior",
            action_sequence=(4, 1),
            records=(),
            actions_executed=2,
            start_observation_hash="level0_before",
            final_observation_hash="level0_after",
            start_position=(32, 21),
            final_position=(42, 21),
            controlled_position_source="live_public_event_tracker",
            semantic_guards_preserved=True,
            stop_reason=None,
        )
        outcome = ProbeOutcomeClassifier().classify(
            execution=prior_execution,
            known_options=(fixture["option_a"], fixture["option_b"]),
        )
        expansion = LiveOptionInducer().promote(
            expansion_index=0,
            source_state=source_state,
            source_position=(32, 21),
            outcome=outcome,
            execution=prior_execution,
        )
        assert expansion is not None
        template = OptionTemplateExtractor().extract(
            options=(expansion.option,),
            source_level_index=0,
        )[0]
        context = LevelContextManager().create_context(
            level_index=1,
            observation_hash="level1_obs",
        )

        instance = LevelTemplateInstantiator().instantiate(
            template=template,
            level_context=context,
            observation_hash="level1_obs",
            controlled_position=(36, 43),
            controlled_position_source="live_public_event_tracker",
            available_actions=(1, 2, 3, 4),
        )

        self.assertEqual(instance.level_context_id, "level_1_context")
        self.assertEqual(instance.target_expression["center_source"], "fresh_level_context_probe_result")
        self.assertEqual(instance.target_expression["anchor_position"], [36, 43])
        self.assertNotEqual(instance.target_expression.get("anchor_position"), [42, 21])
        self.assertFalse(instance.forbidden_prior_fields_present)
        self.assertTrue(instance.new_level_plan_generated_from_current_state)

    def test_m117_stale_instance_is_rejected_and_fresh_instance_promotes(self) -> None:
        fixture = self._m106_live_graph_fixture()
        source_state = fixture["graph"].state("S0")
        assert source_state is not None
        prior_execution = FrontierProbeExecution(
            probe_id="probe_prior",
            action_sequence=(4, 1),
            records=(),
            actions_executed=2,
            start_observation_hash="level0_before",
            final_observation_hash="level0_after",
            start_position=(32, 21),
            final_position=(42, 21),
            controlled_position_source="live_public_event_tracker",
            semantic_guards_preserved=True,
            stop_reason=None,
        )
        outcome = ProbeOutcomeClassifier().classify(
            execution=prior_execution,
            known_options=(fixture["option_a"], fixture["option_b"]),
        )
        prior_expansion = LiveOptionInducer().promote(
            expansion_index=0,
            source_state=source_state,
            source_position=(32, 21),
            outcome=outcome,
            execution=prior_execution,
        )
        assert prior_expansion is not None
        template = OptionTemplateExtractor().extract(
            options=(prior_expansion.option,),
            source_level_index=0,
        )[0]
        context = LevelContextManager().create_context(
            level_index=1,
            observation_hash="level1_obs",
        )
        instantiator = LevelTemplateInstantiator()
        stale = instantiator.stale_instance_from_option(
            option=prior_expansion.option,
            template=template,
            level_context=context,
        )
        stale_validation = TemplateInstanceEvaluator().evaluate(
            instance=stale,
            execution=None,
        )
        fresh = instantiator.instantiate(
            template=template,
            level_context=context,
            observation_hash="level1_obs",
            controlled_position=(36, 43),
            controlled_position_source="live_public_event_tracker",
            available_actions=(1, 2, 3, 4),
        )
        fresh_execution = FrontierProbeExecution(
            probe_id="template_validation_000",
            action_sequence=fresh.validation_action_sequence,
            records=(),
            actions_executed=1,
            start_observation_hash="level1_obs",
            final_observation_hash="level1_after",
            start_position=(36, 43),
            final_position=(35, 61),
            controlled_position_source="live_public_event_tracker",
            semantic_guards_preserved=True,
            stop_reason=None,
        )
        fresh_validation = TemplateInstanceEvaluator().evaluate(
            instance=fresh,
            execution=fresh_execution,
        )
        level1_source = context.graph.state(context.initial_state_class_id)
        assert level1_source is not None
        promoted = promote_validated_template_instance(
            expansion_index=0,
            template=template,
            instance=fresh,
            validation=fresh_validation,
            execution=fresh_execution,
            source_state=level1_source,
        )

        self.assertEqual(stale_validation.result, "REJECTED_STALE_INSTANCE")
        self.assertEqual(fresh_validation.result, "VALIDATED_TEMPLATE_INSTANCE")
        self.assertIsNotNone(promoted)
        assert promoted is not None
        self.assertEqual(promoted.option.source, "cross_level_template_instance")
        self.assertNotEqual(
            promoted.option.selected_predicate_expression_hash,
            template.source_option_expression_hash,
        )

    def test_m117_promoted_instance_enters_candidate_generation(self) -> None:
        fixture = self._m106_live_graph_fixture()
        source_state = fixture["graph"].state("S0")
        assert source_state is not None
        prior_execution = FrontierProbeExecution(
            probe_id="probe_prior",
            action_sequence=(4, 1),
            records=(),
            actions_executed=2,
            start_observation_hash="level0_before",
            final_observation_hash="level0_after",
            start_position=(32, 21),
            final_position=(42, 21),
            controlled_position_source="live_public_event_tracker",
            semantic_guards_preserved=True,
            stop_reason=None,
        )
        outcome = ProbeOutcomeClassifier().classify(
            execution=prior_execution,
            known_options=(fixture["option_a"], fixture["option_b"]),
        )
        prior_expansion = LiveOptionInducer().promote(
            expansion_index=0,
            source_state=source_state,
            source_position=(32, 21),
            outcome=outcome,
            execution=prior_execution,
        )
        assert prior_expansion is not None
        template = OptionTemplateExtractor().extract(
            options=(prior_expansion.option,),
            source_level_index=0,
        )[0]
        context = LevelContextManager().create_context(
            level_index=1,
            observation_hash="level1_obs",
        )
        instance = LevelTemplateInstantiator().instantiate(
            template=template,
            level_context=context,
            observation_hash="level1_obs",
            controlled_position=(36, 43),
            controlled_position_source="live_public_event_tracker",
            available_actions=(1, 2, 3, 4),
        )
        execution = FrontierProbeExecution(
            probe_id="template_validation_000",
            action_sequence=instance.validation_action_sequence,
            records=(),
            actions_executed=1,
            start_observation_hash="level1_obs",
            final_observation_hash="level1_after",
            start_position=(36, 43),
            final_position=(35, 61),
            controlled_position_source="live_public_event_tracker",
            semantic_guards_preserved=True,
            stop_reason=None,
        )
        validation = TemplateInstanceEvaluator().evaluate(
            instance=instance,
            execution=execution,
        )
        level1_source = context.graph.state(context.initial_state_class_id)
        assert level1_source is not None
        promoted = promote_validated_template_instance(
            expansion_index=0,
            template=template,
            instance=instance,
            validation=validation,
            execution=execution,
            source_state=level1_source,
        )
        assert promoted is not None
        expanded_graph = apply_graph_expansion(
            graph=context.graph,
            expansion=promoted,
        )
        binding = LiveObservationStateClassifier().classify(
            observation_hash="level1_obs",
            graph=expanded_graph,
            options=(promoted.option,),
            controlled_position=(36, 43),
            controlled_position_source="live_public_event_tracker",
        )
        candidates = LiveCandidateGenerator().generate(
            graph=expanded_graph,
            binding=binding,
            options=(promoted.option,),
            action_plans={promoted.option.option_id: instance.validation_action_sequence},
        )

        self.assertTrue(any(
            promoted.option.option_id in candidate.option_ids
            for candidate in candidates
        ))
        self.assertEqual(
            CrossLevelTemplateRegistry(templates=(template,)).carry_into(
                level_context=context,
            )["template_count"],
            1,
        )

    def test_m117_status_and_audit_catch_counterfeits(self) -> None:
        common = dict(
            transition_handled=True,
            template_count=1,
            template_contains_level_local_fact=False,
            instance_count=1,
            instance_binding_from_prior_level_fact=False,
            stale_negative_control_ran=True,
            stale_instance_rejected=True,
            validation_probe_public=True,
            validated_instance_count=1,
            promoted_instance_count=1,
            promoted_instance_scored=True,
            template_contribution_detected=True,
            stale_spatial_instance_reused=False,
            prior_coordinate_leakage=False,
            prior_edge_leakage=False,
            source_probe_script_reused=False,
            trace_leakage=False,
            oracle_signal=False,
        )

        self.assertEqual(m117_status(**common), "complete_with_known_limits")
        self.assertEqual(
            m117_status(**{**common, "source_probe_script_reused": True}),
            "failed_source_probe_script_reused",
        )
        self.assertIn(
            "public_trace_segment",
            cross_level_transfer_findings({"source": "public_trace_segment"}),
        )

    def test_m12_terminal_evidence_creates_nonfinal_goal_hypothesis(self) -> None:
        evidence = (
            GoalEvidenceRecord(
                evidence_id="terminal_000",
                kind="TERMINAL_PUBLIC_EVIDENCE",
                source_artifact="terminal-evidence-ledger.json",
                public=True,
                supports=("level_transition_triggering",),
                contradicts=(),
                payload={"levels_completed_increases": True, "trigger_action": 1},
                provenance=({"kind": "public_terminal_transition"},),
            ),
        )

        hypotheses = GoalHypothesisGenerator().generate(evidence=evidence)
        terminal = next(
            item for item in hypotheses
            if item.family == "level_transition_triggering"
        )

        self.assertFalse(terminal.final_goal_claim)
        self.assertTrue(terminal.has_public_predicate())
        self.assertEqual(terminal.target_predicate["kind"], "public_counter_transition")
        self.assertIn("literal_action_is_goal", terminal.forbidden_interpretations)

    def test_m12_progress_features_support_but_do_not_become_goal_predicates(self) -> None:
        evidence = (
            GoalEvidenceRecord(
                evidence_id="progress_000",
                kind="PROGRESS_FEATURE_EVIDENCE",
                source_artifact="progress-hypotheses.json",
                public=True,
                supports=("decision_surface_expansion",),
                contradicts=(),
                payload={"active_progress_features": ["new_option_enabled"]},
                provenance=({"kind": "public_progress_feature_learning"},),
            ),
        )

        hypothesis = next(
            item for item in GoalHypothesisGenerator().generate(evidence=evidence)
            if item.family == "decision_surface_expansion"
        )

        self.assertIn("new_option_enabled", hypothesis.support_features)
        self.assertEqual(hypothesis.target_predicate["kind"], "state_class_property")
        self.assertNotEqual(hypothesis.target_predicate["kind"], "progress_feature")

    def test_m12_noise_hypothesis_is_suppressed(self) -> None:
        evidence = (
            GoalEvidenceRecord(
                evidence_id="noise_000",
                kind="NOISE_CONTROL_EVIDENCE",
                source_artifact="progress-hypotheses.json",
                public=True,
                supports=("volatile_visual_delta",),
                contradicts=("decision_surface_expansion",),
                payload={"volatile_visual_delta_suppressed": True},
                provenance=({"kind": "public_noise_control"},),
            ),
        )

        hypothesis = next(
            item for item in GoalHypothesisGenerator().generate(evidence=evidence)
            if item.family == "volatile_visual_delta"
        )

        self.assertEqual(hypothesis.status, "SUPPRESSED")
        self.assertFalse(hypothesis.testable)
        self.assertLessEqual(hypothesis.confidence_terms["noise_penalty"], 0)

    def test_m12_scoring_is_order_invariant_and_not_literal_action_script(self) -> None:
        evidence = (
            GoalEvidenceRecord(
                evidence_id="terminal_000",
                kind="TERMINAL_PUBLIC_EVIDENCE",
                source_artifact="terminal-evidence-ledger.json",
                public=True,
                supports=("level_transition_triggering",),
                contradicts=(),
                payload={"levels_completed_increases": True},
                provenance=({"kind": "public_terminal_transition"},),
            ),
            GoalEvidenceRecord(
                evidence_id="template_000",
                kind="TEMPLATE_TRANSFER_EVIDENCE",
                source_artifact="milestone-117-summary.json",
                public=True,
                supports=("transferable_template_progress",),
                contradicts=(),
                payload={"validated_template_instance_count": 1},
                provenance=({"kind": "public_cross_level_template_validation"},),
            ),
        )
        hypotheses = GoalHypothesisGenerator().generate(evidence=evidence)

        selected_a = next(
            item for item in GoalHypothesisScorer().score(hypotheses)
            if item.selected
        )
        selected_b = next(
            item for item in GoalHypothesisScorer().score(tuple(reversed(hypotheses)))
            if item.selected
        )
        counterfeit = GoalHypothesis(
            goal_hypothesis_id="bad",
            family="level_transition_triggering",
            claim_type="instrumental_goal_hypothesis",
            final_goal_claim=False,
            target_predicate={
                "kind": "public_counter_transition",
                "expression": {"trigger_action_is_goal": True},
            },
            support_features=("level_completion",),
            evidence_ids=("terminal_000",),
            confidence=0.1,
            confidence_terms={},
            support_count=1,
            contradiction_count=0,
            status="TESTABLE",
            testable=True,
            known_limits=(),
            forbidden_interpretations=(),
        )

        self.assertEqual(selected_a.goal_hypothesis_id, selected_b.goal_hypothesis_id)
        self.assertTrue(counterfeit.promotes_literal_action_script())

    def test_m12_goal_test_updates_confidence_and_changes_goal_aware_selection(self) -> None:
        evidence = (
            GoalEvidenceRecord(
                evidence_id="terminal_000",
                kind="TERMINAL_PUBLIC_EVIDENCE",
                source_artifact="terminal-evidence-ledger.json",
                public=True,
                supports=("level_transition_triggering",),
                contradicts=(),
                payload={"levels_completed_increases": True},
                provenance=({"kind": "public_terminal_transition"},),
            ),
        )
        selected = next(
            item for item in GoalHypothesisScorer().score(
                GoalHypothesisGenerator().generate(evidence=evidence)
            )
            if item.selected
        )
        test = GoalTestCandidateFactory().create(
            hypothesis=selected,
            public_actions=(4,),
            budget=12,
        )
        executed = execute_public_style_goal_test(
            test=test,
            observed_support=True,
            observed_outcome={"public_actions_executed": True},
        )
        outcome_evidence = GoalEvidenceRecord(
            evidence_id="goal_test_outcome",
            kind="TERMINAL_PUBLIC_EVIDENCE",
            source_artifact="goal-test-candidates.json",
            public=True,
            supports=(selected.family,),
            contradicts=(),
            payload={"result": executed.result},
            provenance=({"kind": "bounded_public_goal_test"},),
        )
        updated, update = GoalHypothesisUpdater().update(
            hypothesis=selected,
            test=executed,
            evidence=outcome_evidence,
        )
        selector = GoalAwareSelector(
            progress_model=LearnedProgressModel(ProgressHypothesisSet.from_m9_priors())
        )
        progress_only = selector.progress_only_score(
            candidate=ChainCandidate(
                candidate_id="heldout_terminal_like",
                candidate_type="goal_test_candidate",
                option_ids=("terminal_like_template_probe",),
                start_state_class_id="L1_S0",
                predicted_terminal_state_class_id=None,
                predicted_progress_features={
                    "level_completion": False,
                    "stable_semantic_delta": False,
                    "new_option_enabled": False,
                    "new_state_class": False,
                    "uncertainty_reduction": False,
                    "volatile_visual_delta": False,
                },
                cost_features={"predicted_action_cost": 4, "empirical_consistency": 1.0},
                risk_features={"guard_risk": 0},
                provenance=(),
            )
        )
        goal_aware = selector.goal_aware_score(
            candidate=progress_only,
            hypotheses=(updated,),
        )

        self.assertEqual(executed.test_mode, "TEST_HYPOTHESIS")
        self.assertTrue(executed.generated_before_execution)
        self.assertGreater(update.posterior_confidence, update.prior_confidence)
        self.assertGreater(goal_aware.score, progress_only.score)

    def test_m12_status_and_audit_catch_counterfeits(self) -> None:
        common = dict(
            evidence_count=4,
            hypothesis_count=4,
            all_hypotheses_have_predicates=True,
            fixed_hypothesis_selection=False,
            test_generated=True,
            test_updated_hypothesis=True,
            heldout_goal_generalization=True,
            progress_only_explains_selection=False,
            noise_promoted=False,
            candidate_memory_confused=False,
            terminal_action_script_promoted=False,
            unpublic_evidence=False,
            trace_leakage=False,
            oracle_signal=False,
        )

        self.assertEqual(m12_status(**common), "complete_with_known_limits")
        self.assertEqual(
            m12_status(**{**common, "terminal_action_script_promoted": True}),
            "failed_terminal_action_script_promoted",
        )
        self.assertIn(
            "trace_corridor",
            forbidden_goal_hypothesis_findings({"source": "trace_corridor"}),
        )


    def test_local_recovery_planner_finds_non_greedy_route_around_blocked_edge(
        self,
    ) -> None:
        planner = LocalRecoveryPlanner(budget=6)

        result = planner.plan(
            start=(0, 0),
            corridor=TraceCorridor.synthetic([
                (4, (0, 2), (("stable",),)),
            ]),
            action_deltas={1: (0, 1), 2: (1, 0), 3: (-1, 0)},
            blocked_edges={((0, 0), 1)},
            valid_positions=frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)}),
        )

        self.assertTrue(result.recovered)
        self.assertEqual(result.resume_index, 4)
        self.assertEqual(result.plan, (2, 1, 1, 3))
        self.assertTrue(result.non_greedy)

    def test_local_recovery_planner_reports_unreachable_without_looping(self) -> None:
        planner = LocalRecoveryPlanner(budget=3)

        result = planner.plan(
            start=(0, 0),
            corridor=TraceCorridor.synthetic([
                (4, (0, 2), (("stable",),)),
            ]),
            action_deltas={1: (0, 1)},
            blocked_edges={((0, 0), 1)},
            valid_positions=frozenset({(0, 0), (0, 1), (0, 2)}),
        )

        self.assertFalse(result.recovered)
        self.assertEqual(result.failure_reason, "TARGET_UNREACHABLE")
        self.assertEqual(result.repeated_blocked_action_count, 0)

    def test_control_belief_discovers_action_dependent_group(self) -> None:
        belief = ControlledEntityBelief(minimum_support=4)
        estimate = None
        for step, (action, delta) in enumerate((
            (1, (-1, 0)),
            (2, (1, 0)),
            (1, (-1, 0)),
            (2, (1, 0)),
        ), start=1):
            estimate = belief.observe(self._motion_event(step, action, delta, (7, 8)))

        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.object_ids, (7, 8))
        self.assertEqual(estimate.source, "local_action_evidence")
        self.assertEqual(estimate.action_model, {1: (-1, 0), 2: (1, 0)})
        self.assertEqual(estimate.actions_into_level, 4)

    def test_control_belief_rejects_action_independent_group(self) -> None:
        belief = ControlledEntityBelief(minimum_support=4)
        for step, action in enumerate((1, 2, 1, 2), start=1):
            estimate = belief.observe(self._motion_event(step, action, (0, 1), (7, 8)))

        self.assertIsNone(estimate)

    def test_control_belief_reacquires_from_transferred_action_model(self) -> None:
        belief = ControlledEntityBelief(minimum_support=4, prior_action_matches=2)
        for step, (action, delta) in enumerate((
            (1, (-1, 0)),
            (2, (1, 0)),
            (1, (-1, 0)),
            (2, (1, 0)),
        ), start=1):
            belief.observe(self._motion_event(step, action, delta, (7, 8)))
        belief.observe(self._motion_event(5, 1, (0, 0), (), levels_after=1))

        belief.observe(self._motion_event(6, 1, (-1, 0), (20, 21), levels_before=1, levels_after=1))
        estimate = belief.observe(
            self._motion_event(7, 2, (1, 0), (20, 21), levels_before=1, levels_after=1)
        )

        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.object_ids, (20, 21))
        self.assertEqual(estimate.source, "transferred_action_model")

    def test_interaction_detector_finds_object_in_intended_destination(self) -> None:
        parser = ObservationParser()
        tracker = ObjectTracker()
        before = observation([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 0, 0, 0],
        ])
        after = observation([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 0, 0, 0],
        ])
        tracker.update(parser.parse(before, frame_index=0))
        tracks = tracker.update(parser.parse(after, frame_index=1))
        event = EventDetector().detect(
            step=1,
            before=before,
            action=4,
            after=after,
            tracks=tracks,
        )
        control = self._control_estimate((1,), {4: (0, 1)})

        interaction = InteractionDetector().detect(
            event=event,
            tracks=tracks,
            control=control,
            expected_translation=(0, 1),
        )

        self.assertTrue(interaction.blocked)
        self.assertEqual(
            [
                (item.object_id, item.overlap_pixels, item.object_coverage)
                for item in interaction.contact_candidates
            ],
            [(2, 1, 1.0)],
        )

    def test_interaction_detector_rejects_scene_spanning_region(self) -> None:
        parser = ObservationParser()
        tracker = ObjectTracker()
        grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 1, 2, 0, 0],
            [0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ]
        before = observation(grid)
        after = observation(grid)
        tracker.update(parser.parse(before, frame_index=0))
        tracks = tracker.update(parser.parse(after, frame_index=1))
        event = EventDetector().detect(
            step=1,
            before=before,
            action=4,
            after=after,
            tracks=tracks,
        )

        interaction = InteractionDetector().detect(
            event=event,
            tracks=tracks,
            control=self._control_estimate((1,), {4: (0, 1)}),
            expected_translation=(0, 1),
        )

        self.assertEqual(interaction.contact_candidates, ())

    def test_conditional_effect_memory_keeps_contact_elevated_effect(self) -> None:
        memory = ConditionalEffectMemory(
            minimum_contact_support=2,
            minimum_baseline_opportunities=4,
            minimum_rate_excess=0.5,
        )
        contact = ContactDescriptor(2, 3, "shape-a")
        effect = EffectKey(99, 7, "form_change")
        for _ in range(4):
            memory.observe(EffectEvidence(0, (), frozenset({99}), frozenset()))
        for _ in range(2):
            memory.observe(EffectEvidence(0, (contact,), frozenset({99}), frozenset({effect})))

        summaries = memory.summaries()

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["contact_support"], 2)
        self.assertEqual(summaries[0]["baseline_rate"], 0.0)

    def test_conditional_effect_memory_rejects_autonomous_change(self) -> None:
        memory = ConditionalEffectMemory(
            minimum_contact_support=2,
            minimum_baseline_opportunities=4,
            minimum_rate_excess=0.25,
        )
        contact = ContactDescriptor(2, 3, "shape-a")
        effect = EffectKey(99, 7, "form_change")
        for _ in range(4):
            memory.observe(EffectEvidence(0, (), frozenset({99}), frozenset({effect})))
        for _ in range(2):
            memory.observe(EffectEvidence(0, (contact,), frozenset({99}), frozenset({effect})))

        self.assertEqual(memory.summaries(), [])
        self.assertEqual(memory.affordances(), ())

    def test_affordance_memory_groups_cochanging_targets_and_tracks_state_chain(
        self,
    ) -> None:
        memory = ConditionalEffectMemory(
            minimum_contact_support=3,
            minimum_baseline_opportunities=5,
            minimum_rate_excess=0.5,
        )
        contact = ContactDescriptor(2, 3, "contact-shape")
        first = EffectKey(38, 5, "rotation:1")
        second = EffectKey(39, 9, "rotation:1")
        for _ in range(5):
            memory.observe(EffectEvidence(
                1,
                (),
                frozenset({38, 39}),
                frozenset(),
            ))
        states = (
            ("a0", "a1", "b0", "b1"),
            ("a1", "a2", "b1", "b2"),
            ("a2", "a3", "b2", "b3"),
        )
        for step, (a_before, a_after, b_before, b_after) in enumerate(
            states,
            start=10,
        ):
            memory.observe(EffectEvidence(
                1,
                (contact,),
                frozenset({38, 39}),
                frozenset({first, second}),
                (
                    EffectTransition(first, a_before, a_after),
                    EffectTransition(second, b_before, b_after),
                ),
                step,
            ))

        affordances = memory.affordances()

        self.assertEqual(len(affordances), 1)
        affordance = affordances[0]
        self.assertEqual(affordance.target_object_ids, (38, 39))
        self.assertEqual(affordance.effect_kind, "rotation:1")
        self.assertEqual(affordance.state_model.observed_state_count, 4)
        self.assertTrue(affordance.state_model.deterministic)
        self.assertTrue(affordance.state_model.continuous)
        self.assertFalse(affordance.state_model.closed_cycle)
        self.assertEqual(
            affordance.state_model.current_state,
            ((38, "a3"), (39, "b3")),
        )

    def test_affordance_state_model_marks_only_observed_closed_cycle(self) -> None:
        memory = ConditionalEffectMemory(
            minimum_contact_support=4,
            minimum_baseline_opportunities=5,
            minimum_rate_excess=0.5,
        )
        contact = ContactDescriptor(2, 3, "contact-shape")
        effect = EffectKey(7, 5, "rotation:1")
        for _ in range(5):
            memory.observe(EffectEvidence(0, (), frozenset({7}), frozenset()))
        for step, (before, after) in enumerate((
            ("s0", "s1"),
            ("s1", "s2"),
            ("s2", "s3"),
            ("s3", "s0"),
        ), start=1):
            memory.observe(EffectEvidence(
                0,
                (contact,),
                frozenset({7}),
                frozenset({effect}),
                (EffectTransition(effect, before, after),),
                step,
            ))

        model = memory.affordances()[0].state_model

        self.assertTrue(model.closed_cycle)
        self.assertTrue(model.continuous)
        self.assertEqual(model.cycle_length, 4)

    def test_affordance_memory_retracts_relation_after_conflicting_baseline(
        self,
    ) -> None:
        memory = ConditionalEffectMemory(
            minimum_contact_support=2,
            minimum_baseline_opportunities=2,
            minimum_rate_excess=0.5,
        )
        contact = ContactDescriptor(2, 3, "contact-shape")
        effect = EffectKey(7, 5, "rotation:1")
        for _ in range(2):
            memory.observe(EffectEvidence(0, (), frozenset({7}), frozenset()))
        for _ in range(2):
            memory.observe(EffectEvidence(
                0,
                (contact,),
                frozenset({7}),
                frozenset({effect}),
            ))
        self.assertEqual(len(memory.affordances()), 1)

        for _ in range(3):
            memory.observe(EffectEvidence(
                0,
                (),
                frozenset({7}),
                frozenset({effect}),
            ))

        self.assertEqual(memory.affordances(), ())

    @staticmethod
    def _control_estimate(object_ids, action_model):
        from observation_option_agent import ControlEstimate

        return ControlEstimate(
            level=0,
            object_ids=object_ids,
            support=4,
            distinct_actions=len(action_model),
            consistency=1.0,
            confidence=1.0,
            source="test",
            action_model=action_model,
            discovered_step=1,
            actions_into_level=1,
        )

    @staticmethod
    def _option_candidate():
        return OptionCandidate(
            option_id="option_test",
            segment_range=(0, 2),
            initiation_condition={
                "type": "actor_at_or_near_position_with_semantic_signature",
                "actor_position": [0, 0],
                "semantic_signature": [],
            },
            termination_condition={
                "type": "actor_reaches_observable_position",
                "target_position": [0, 2],
                "tolerance": 0,
                "forbidden_form": "trace_index == segment_end",
            },
            expected_actor_delta={
                "from_position": [0, 0],
                "to_position": [0, 2],
                "delta": [0, 2],
            },
            expected_semantic_delta={
                "type": "none",
                "from_signature": [],
                "to_signature": [],
            },
            allowed_recovery_envelope={
                "max_actions": 12,
                "allowed_world_diff": "none_or_expected",
            },
            invalidating_conditions=(
                "semantic_mismatch",
                "target_unreachable",
                "unexpected_noncontrolled_change",
                "action_conflict",
            ),
            evidence=({"kind": "public_trace_segment", "start": 0, "end": 2},),
            trace_dependency_level=3,
            predicate_abstraction_level=1,
        )

    @staticmethod
    def _discovery_record(
        *,
        action_sequence=(3, 3, 3, 1),
        actor_delta=(-5, -15),
    ):
        start = (46, 36)
        return DiscoveryRecord(
            record_id="disc_000",
            source="self_exploration",
            self_discovery_start_source="public_reset_observation",
            observation_before_hash="before",
            action_sequence=tuple(action_sequence),
            observation_after_hash="after",
            start_position=start,
            end_position=(start[0] + actor_delta[0], start[1] + actor_delta[1]),
            actor_delta=actor_delta,
            action_model={1: (-5, 0), 3: (0, -5)},
            semantic_delta=(),
            contact_events=(),
            blocked_edges=(),
            evidence=(
                {"kind": "self_observed_transition"},
                {"kind": "self_observed_reachability"},
            ),
        )

    @staticmethod
    def _option_effect_observation(
        record_id,
        *,
        option=None,
        discovery=None,
        blocked_edges=frozenset({((46, 36), 3)}),
    ) -> OptionEffectObservation:
        discovery = discovery or ObservationOptionAgentTests._discovery_record()
        option = option or SelfOptionInducer().induce(discovery)
        execution = SelfOptionEvaluator().execute(
            assay_id=record_id,
            option=option,
            discovery=discovery,
            blocked_edges=blocked_edges,
        )
        return observation_from_self_execution(
            record_id=record_id,
            option=option,
            discovery=discovery,
            execution=execution,
            exact_start_observation_hash=discovery.observation_before_hash,
        )

    @staticmethod
    def _m8_state_class(
        state_class_id,
        *,
        applicable_options,
        termination_predicates,
        exact_hashes=(),
    ) -> OptionApplicabilityStateClass:
        return OptionApplicabilityStateClass(
            state_class_id=state_class_id,
            controlled_entity_region=f"{state_class_id}_controlled_region",
            semantic_guards=True,
            termination_predicates=dict(termination_predicates),
            applicable_options=tuple(applicable_options),
            non_controlled_signature_class=f"{state_class_id}_stable_semantics",
            exact_observation_hashes=tuple(exact_hashes),
        )

    @staticmethod
    def _m8_chain_fixture():
        discovery = ObservationOptionAgentTests._discovery_record()
        option_a = SelfOptionInducer().induce(discovery)
        option_b = replace(
            option_a,
            option_id="self_option_001",
            selected_predicate_id="self_option_001:pred_region",
        )
        state_0 = ObservationOptionAgentTests._m8_state_class(
            "S0",
            applicable_options=(option_a.option_id,),
            termination_predicates={
                option_a.selected_predicate_id or "": False,
                option_b.selected_predicate_id or "": False,
            },
            exact_hashes=("frame_start",),
        )
        state_1 = ObservationOptionAgentTests._m8_state_class(
            "S1",
            applicable_options=(option_b.option_id,),
            termination_predicates={
                option_a.selected_predicate_id or "": True,
                option_b.selected_predicate_id or "": False,
            },
            exact_hashes=("frame_after_a",),
        )
        state_2 = ObservationOptionAgentTests._m8_state_class(
            "S2",
            applicable_options=(),
            termination_predicates={
                option_a.selected_predicate_id or "": True,
                option_b.selected_predicate_id or "": True,
            },
            exact_hashes=("frame_after_b",),
        )
        train_a = replace(
            ObservationOptionAgentTests._option_effect_observation(
                "effect_train_a",
                option=option_a,
                discovery=discovery,
            ),
            start_signature_class=state_0.signature(),
            start_signature_class_hash=state_0.signature_hash,
        )
        heldout_a = replace(
            ObservationOptionAgentTests._option_effect_observation(
                "effect_test_a",
                option=option_a,
                discovery=discovery,
                blocked_edges=frozenset({(discovery.start_position, 1), ((46, 31), 3)}),
            ),
            start_signature_class=state_0.signature(),
            start_signature_class_hash=state_0.signature_hash,
        )
        train_b = replace(
            ObservationOptionAgentTests._option_effect_observation(
                "effect_train_b",
                option=option_b,
                discovery=discovery,
            ),
            start_signature_class=state_1.signature(),
            start_signature_class_hash=state_1.signature_hash,
        )
        heldout_b = replace(
            ObservationOptionAgentTests._option_effect_observation(
                "effect_test_b",
                option=option_b,
                discovery=discovery,
                blocked_edges=frozenset({(discovery.start_position, 1), ((46, 31), 3)}),
            ),
            start_signature_class=state_1.signature(),
            start_signature_class_hash=state_1.signature_hash,
        )
        model = OptionEffectModel.train((train_a, train_b))
        prediction_a = model.predict(
            prediction_id="pred_link_a",
            option=option_a,
            start_signature_class_hash=state_0.signature_hash,
        )
        prediction_b = model.predict(
            prediction_id="pred_link_b",
            option=option_b,
            start_signature_class_hash=state_1.signature_hash,
        )
        assert prediction_a is not None
        assert prediction_b is not None
        graph = OptionApplicabilityGraph(
            state_classes=(state_0, state_1, state_2),
            edges=(
                OptionApplicabilityEdge(
                    edge_id="edge_a_enables_b",
                    source_state_class_id=state_0.state_class_id,
                    option_id=option_a.option_id,
                    predicted_next_state_class_id=state_1.state_class_id,
                    enabled_options=(option_b.option_id,),
                    option_prediction=prediction_a,
                ),
                OptionApplicabilityEdge(
                    edge_id="edge_b_terminates_chain",
                    source_state_class_id=state_1.state_class_id,
                    option_id=option_b.option_id,
                    predicted_next_state_class_id=state_2.state_class_id,
                    enabled_options=(),
                    option_prediction=prediction_b,
                ),
            ),
        )
        chain_prediction = graph.predict_chain(
            prediction_id="pred_chain_000",
            start_state_class_id=state_0.state_class_id,
            options=(option_a, option_b),
        )
        assert chain_prediction is not None
        return {
            "option_a": option_a,
            "option_b": option_b,
            "state_0": state_0,
            "state_1": state_1,
            "state_2": state_2,
            "train_a": train_a,
            "train_b": train_b,
            "heldout_a": heldout_a,
            "heldout_b": heldout_b,
            "graph": graph,
            "prediction": chain_prediction,
        }

    @staticmethod
    def _m106_live_graph_fixture():
        fixture = ObservationOptionAgentTests._m8_chain_fixture()
        option_b = replace(
            fixture["option_b"],
            initiation_condition={
                **fixture["option_b"].initiation_condition,
                "actor_position": list(fixture["option_a"].termination_condition["center"]),
            },
            termination_condition={
                "type": "region_membership",
                "center": [36, 21],
                "radius": 2,
                "metric": "manhattan",
                "excludes_initiation": True,
            },
            expected_actor_delta={
                "from_position": list(fixture["option_a"].termination_condition["center"]),
                "to_position": [36, 21],
                "delta": [-5, 0],
                "observed_delta": [-5, 0],
            },
        )
        return {
            **fixture,
            "option_b": option_b,
        }

    @staticmethod
    def _m10_candidate(
        candidate_id,
        *,
        level_completion=False,
        stable_semantic_delta=False,
        new_option_enabled=False,
        new_state_class=False,
        uncertainty_reduction=False,
        volatile_visual_delta=False,
        action_cost=4,
    ) -> ChainCandidate:
        return ChainCandidate(
            candidate_id=candidate_id,
            candidate_type="option_chain",
            option_ids=("self_option_000",),
            start_state_class_id="S0",
            predicted_terminal_state_class_id="S1",
            predicted_progress_features={
                "level_completion": level_completion,
                "stable_semantic_delta": stable_semantic_delta,
                "new_option_enabled": new_option_enabled,
                "new_state_class": new_state_class,
                "uncertainty_reduction": uncertainty_reduction,
                "volatile_visual_delta": volatile_visual_delta,
            },
            cost_features={
                "predicted_action_cost": action_cost,
                "empirical_consistency": 1.0,
            },
            risk_features={"guard_risk": 0},
            provenance=({"kind": "m10_test_candidate"},),
        )

    @staticmethod
    def _record(
        *,
        action,
        contacts,
        transforms,
        level_delta=0,
        blocked=False,
    ):
        return {
            "action": action,
            "levels_before": 0,
            "levels_after": level_delta,
            "interaction": {
                "blocked": blocked,
                "contact_candidates": [
                    {
                        "value": value,
                        "object_area": area,
                        "shape_id": shape_id,
                    }
                    for value, area, shape_id in contacts
                ],
            },
            "object_deltas": [
                {
                    "shape_transform": transform,
                    "shape_changed": transform != "same",
                    "value_changed": False,
                    "area_delta": 0,
                    "translation": [0, 0],
                }
                for transform in transforms
            ],
        }

    @staticmethod
    def _effect_record(
        *,
        action,
        contacts,
        changed_object_id,
        transform,
        area_delta=0,
    ):
        record = ObservationOptionAgentTests._record(
            action=action,
            contacts=contacts,
            transforms=(),
        )
        record["interaction"]["changed_other_object_ids"] = [changed_object_id]
        record["object_deltas"] = [{
            "object_id": changed_object_id,
            "shape_transform": transform,
            "shape_changed": transform != "same",
            "value_changed": False,
            "area_delta": area_delta,
            "translation": [0, 0],
        }]
        return record

    @staticmethod
    def _probe_proposal(*, contacts, effect):
        from observation_option_agent import ProbeProposal

        return ProbeProposal(
            contact_signature=tuple(contacts),
            predicted_effect=effect,
            support=3,
            effect_rate=1.0,
            lift=2.0,
        )

    @staticmethod
    def _trace_step(
        before_grid,
        action,
        after_grid,
        *,
        levels_before=0,
        levels_after=0,
    ):
        return TraceStep(
            observation(before_grid, levels=levels_before),
            action,
            observation(after_grid, levels=levels_after),
        )

    @staticmethod
    def _event_record(
        *,
        step,
        action,
        levels_before=0,
        levels_after=0,
        contact_signature=(),
        disappeared=(),
        created=(),
        changed=(),
        full_reset=False,
        scene_change_fraction=0.01,
    ):
        return {
            "step": step,
            "action": action,
            "levels_before": levels_before,
            "levels_after": levels_after,
            "state_after": "WIN" if levels_after > levels_before else "NOT_FINISHED",
            "full_reset": full_reset,
            "scene_change_fraction": scene_change_fraction,
            "created_object_ids": list(created),
            "disappeared_object_ids": list(disappeared),
            "reappeared_object_ids": [],
            "object_deltas": [
                {
                    "object_id": object_id,
                    "translation": [0, 0],
                    "shape_changed": True,
                    "shape_transform": "unclassified_form_change",
                    "value_changed": False,
                    "area_delta": 1,
                }
                for object_id in changed
            ],
            "control": {"object_ids": [1]} if step else None,
            "interaction": {
                "blocked": False,
                "contact_candidates": [
                    {
                        "value": value,
                        "object_area": area,
                        "shape_id": shape_id,
                    }
                    for value, area, shape_id in contact_signature
                ],
                "changed_other_object_ids": list(changed),
                "created_other_object_ids": list(created),
                "disappeared_other_object_ids": list(disappeared),
            } if contact_signature else None,
            "affordance_changes": [],
        }

    @staticmethod
    def _motion_event(
        step,
        action,
        delta,
        object_ids,
        *,
        levels_before=0,
        levels_after=0,
    ):
        groups = (MotionGroup(delta, object_ids),) if object_ids else ()
        return Event(
            step=step,
            action=action,
            levels_before=levels_before,
            levels_after=levels_after,
            state_after="NOT_FINISHED",
            full_reset=False,
            changed_pixels=2,
            scene_change_fraction=0.1,
            object_deltas=(),
            created_object_ids=(),
            disappeared_object_ids=(),
            motion_groups=groups,
        )


class _FakeLiveFrame:
    def __init__(
        self,
        frame,
        *,
        levels_completed=0,
        state="NOT_FINISHED",
        full_reset=False,
        available_actions=(1, 2, 3, 4),
    ) -> None:
        self.frame = frame
        self.levels_completed = levels_completed
        self.state = state
        self.full_reset = full_reset
        self.available_actions = available_actions


class _FakeLiveEnv:
    def __init__(self, frames) -> None:
        self.frames = list(frames)
        self.index = 0

    def reset(self):
        self.index = 0
        return self.frames[0]

    def step(self, action, data=None, reasoning=None):
        self.index += 1
        return self.frames[self.index]


if __name__ == "__main__":
    unittest.main()
