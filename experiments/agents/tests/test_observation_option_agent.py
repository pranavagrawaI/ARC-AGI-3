from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.harness import Observation


AGENT_ROOT = Path(__file__).resolve().parents[1] / "0001-ls20-observation-option-agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from observation_option_agent import (
    ActionCouplingMemory,
    ControlledEntityBelief,
    ConditionalEffectMemory,
    ContactDescriptor,
    Event,
    EventDetector,
    EffectEvidence,
    EffectKey,
    EffectTransition,
    InteractionDetector,
    LiveContactProbeAgent,
    Milestone1Analyzer,
    ControlVerifier,
    D1InjectionPolicy,
    DriftClassifier,
    LocalRecoveryPlanner,
    SemanticSignature,
    StructuredRecoveryAgent,
    TraceAlignment,
    TraceAssistedReplayAgent,
    TraceCorridor,
    TraceReference,
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
    transition_symbols_from_record,
    iter_trace_steps,
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


if __name__ == "__main__":
    unittest.main()
