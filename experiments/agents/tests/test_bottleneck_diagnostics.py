from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
import csv

import numpy as np

from experiments.harness import Observation


AGENT_ROOT = Path(__file__).resolve().parents[1] / "0001-ls20-observation-option-agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from observation_option_agent.bottlenecks import (  # noqa: E402
    POST_FIELDS,
    R3A_FROZEN_FIELDS,
    R3A_COMPRESSED,
    R2A_STATIC,
    R2B_MEMORY,
    R4_CLICK_PROCEDURAL_LEARNED_ONLY,
    R4_RELATIONAL_GEOMETRY_ONLY,
    CellAttribution,
    ClickProceduralMemory,
    ClusterDecomposition,
    DecompositionHypothesis,
    HandleMemory,
    RepresentationFamily,
    ResidueAtom,
    ResidueCompetitionGroup,
    ResidueSourceHypothesis,
    RoleMemory,
    TraceSpec,
    TransitionRecord,
    add_contradiction_mode,
    annotate_labeler_v3,
    build_pre_features,
    build_post_labels_v2,
    causal_stage_decomposition_report,
    changed_region_factorization_report,
    component_signature,
    context_key,
    controlled_region_material_decomposition_report,
    controlled_entity_audit,
    delta_channel_assignment_v3_report,
    evaluate_suite,
    event_schema_metrics_report,
    feature_ablation_report,
    introspect_r3a_suite,
    load_transition_records,
    motion_target_granularity_report,
    motion_law_probe_report,
    mover_bootstrap_composition_v0_report,
    mover_bootstrap_report,
    mover_state_factorization_report,
    changed_cluster_merge_split_audit_v0_report,
    global_update_vs_container_audit_v0_report,
    global_update_layer_v0_report,
    global_phase_register_discovery_v0_report,
    global_phase_identifiability_v0_report,
    residue_source_attribution_v0_report,
    visibility_composition_probe_report,
    visibility_composition_precision_audit_suite,
    world_clock_transition_probe_report,
    ar25_overlap_composition_report,
    ar25_unknown_world_clock_atlas_report,
    ar25_latent_motion_history_filtered_taxonomy_report,
    ar25_latent_transition_taxonomy_report,
    independent_mover_dynamics_report,
    oracle_status,
    outcome_key,
    cd82_region_role_probe_report,
    cd82_region_role_integrated_report,
    cd82_procedural_state_probe_report,
    independent_coupled_mover_probe_report,
    outcome_labeler_v2_vs_v3_report,
    outcome_labeler_v1_vs_v2_report,
    parameter_model_probe_report,
    motion_probe_metrics,
    parameter_probe_metrics,
    prequential_parameter_metrics,
    prequential_metrics,
    r4_probe_comparison,
    r4_support_status,
    region_role_map_report,
    stage_report_v3,
    stage_report_v2,
    retrospective_metrics,
    t5_failure_anatomy_report,
    write_contradictions_csv,
    write_jsonl,
)
from observation_option_agent.observation import ObservationParser  # noqa: E402
from observation_option_agent.tracking import ObjectTracker  # noqa: E402


def obs(grid, *, levels=0, state="NOT_FINISHED", full_reset=False):
    pixels = np.asarray([grid], dtype=np.int16)
    pixels.setflags(write=False)
    return Observation(
        pixels=pixels,
        available_actions=(1, 2, 3, 4, 5, 6),
        levels_completed=levels,
        state=state,
        full_reset=full_reset,
    )


def synthetic_record(
    *,
    action=1,
    coarse="actor_move",
    exact=None,
    pre_context="same",
    step=1,
) -> TransitionRecord:
    before = obs([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    after = obs([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    exact = exact or {
        "actor_delta": [0, 1],
        "changed_cell_signature": {"count_bucket": "few", "bbox": (1, 1, 1, 2)},
        "hud_delta_signature": (),
        "resource_delta_signature": (),
        "status": "NOT_FINISHED",
    }
    return TransitionRecord(
        game="toy",
        trace_source="human",
        step=step,
        action=action,
        action_data=None,
        before=before,
        after=after,
        before_scene=None,  # type: ignore[arg-type]
        after_scene=None,  # type: ignore[arg-type]
        before_tracked=(),
        track_update=None,  # type: ignore[arg-type]
        event=None,  # type: ignore[arg-type]
        pre_features={
            "action_set": (1, 2, 3, 4),
            "action_kind": "move",
            "action_target_type": "near_actor",
            "actor_bbox_or_pos": pre_context,
            "actor_confidence": "medium",
            "actor_or_focus_region": "central_canvas",
            "actor_signature": pre_context,
            "clicked_region_signature": None,
            "component_signature_histogram": ((pre_context, 1),),
            "hud_or_resource_snapshot": pre_context,
            "level_snapshot": {"levels_completed": 0, "state": "NOT_FINISHED"},
            "nearby_component_signatures": (),
            "raw_observation_hash": pre_context,
            "recent_noop_streak": "zero",
            "target_relative_region": "near_actor",
            "visibility_or_unknown_mask": pre_context,
            "visible_state_summary": pre_context,
            "visible_independent_mover_signatures": (),
        },
        post_labels={
            "actor_delta": exact["actor_delta"],
            "blocked_or_rejected": coarse == "block_reject",
            "changed_cells": 2,
            "changed_cell_signature": exact["changed_cell_signature"],
            "coarse_outcome": coarse,
            "contacted_component_delta_class": "none",
            "exact_outcome": exact,
            "hud_delta_signature": exact["hud_delta_signature"],
            "remote_delta_present": False,
            "resource_delta_signature": exact["resource_delta_signature"],
            "terminal_status_delta": None,
        },
    )


class BottleneckDiagnosticTests(unittest.TestCase):
    def test_representation_field_audit_rejects_post_action_context_fields(self) -> None:
        family = RepresentationFamily(
            "bad",
            pre_action_fields=frozenset({"actor_signature", "remote_delta_present"}),
        )

        with self.assertRaises(AssertionError):
            family.assert_valid()

        self.assertIn("remote_delta_present", POST_FIELDS)

    def test_component_signature_does_not_include_raw_component_id(self) -> None:
        scene = ObservationParser().parse(
            obs([[0, 0, 0], [0, 2, 2], [0, 0, 0]]),
            frame_index=0,
        )

        signature = component_signature(scene.detections[0])
        encoded = json.dumps(signature, sort_keys=True)

        self.assertNotIn("object_id", encoded)
        self.assertIn("shape_bitmap", signature)
        self.assertIn("bbox_size", signature)

    def test_context_key_rejects_post_action_payload_fields_structurally(self) -> None:
        record = synthetic_record()
        bad = RepresentationFamily(
            "bad",
            pre_action_fields=frozenset({"actor_signature"}),
            memory_fields=frozenset({"terminal_delta"}),
        )

        with self.assertRaises(AssertionError):
            context_key(record, bad, memory=HandleMemory())

    def test_synthetic_contradiction_metrics_distinguish_coarse_and_exact(self) -> None:
        first = synthetic_record(exact={"actor_delta": [0, 1], "changed_cell_signature": {"a": 1}, "hud_delta_signature": (), "resource_delta_signature": (), "status": "NOT_FINISHED"})
        second = synthetic_record(exact={"actor_delta": [1, 0], "changed_cell_signature": {"a": 2}, "hud_delta_signature": (), "resource_delta_signature": (), "status": "NOT_FINISHED"}, step=2)

        coarse, _ = retrospective_metrics([first, second], R2A_STATIC, granularity="coarse")
        exact, _ = retrospective_metrics([first, second], R2A_STATIC, granularity="exact")

        self.assertEqual(coarse.contradiction_rate, 0.0)
        self.assertEqual(exact.contradiction_rate, 1.0)

    def test_prequential_memory_updates_after_current_step(self) -> None:
        scene = ObservationParser().parse(
            obs([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            frame_index=0,
        )
        tracked = ObjectTracker().update(scene, force_new=True).current
        record = synthetic_record(
            coarse="remote_delta",
            exact={
                "actor_delta": [0, 0],
                "changed_cell_signature": {"remote": 1},
                "hud_delta_signature": (),
                "resource_delta_signature": (),
                "status": "NOT_FINISHED",
            },
        )
        record = TransitionRecord(
            **{
                **record.__dict__,
                "before_tracked": tracked,
                "event": type(
                    "FakeEvent",
                    (),
                    {"object_deltas": (), "step": 1},
                )(),
            }
        )
        memory = HandleMemory()

        before_key = context_key(record, R2B_MEMORY, memory=memory)
        memory.update(record)
        after_key = context_key(record, R2B_MEMORY, memory=memory)

        self.assertNotEqual(before_key, after_key)
        self.assertNotIn("remote_delta", before_key)
        self.assertIn("remote_delta", after_key)

    def test_trace_adapter_separates_pre_features_from_post_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            start = {
                "type": "observation",
                "step": 0,
                "pixels": [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                "available_actions": [1, 2],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
                "full_reset": True,
            }
            transition = {
                "type": "transition",
                "step": 1,
                "action": 2,
                "action_data": None,
                "reasoning": None,
                "observation": {
                    "pixels": [[[0, 0, 0], [0, 0, 1], [0, 0, 0]]],
                    "available_actions": [1, 2],
                    "levels_completed": 0,
                    "state": "NOT_FINISHED",
                    "full_reset": False,
                },
            }
            path.write_text(json.dumps(start) + "\n" + json.dumps(transition) + "\n")

            records = load_transition_records(TraceSpec("toy", path, "human"))

        self.assertEqual(len(records), 1)
        self.assertIn("actor_signature", records[0].pre_features)
        self.assertIn("coarse_outcome", records[0].post_labels)
        self.assertNotIn("actor_delta", records[0].pre_features)

    def test_ls20_oracle_skips_incompatible_human_trace(self) -> None:
        status = oracle_status(TraceSpec("ls20", Path("missing.jsonl"), "human", False))

        self.assertEqual(status["oracle_status"], "skipped_incompatible_trace_source")

    def test_contradiction_records_preserve_mode_and_csv_identity(self) -> None:
        first = synthetic_record(coarse="actor_move")
        second = synthetic_record(coarse="no_op", step=2)
        _summary, contradictions = retrospective_metrics([first, second], R2A_STATIC, granularity="coarse")
        records = add_contradiction_mode(contradictions, mode="retrospective")

        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "contradictions.jsonl"
            csv_path = Path(tmp) / "contradictions.csv"
            write_jsonl(jsonl_path, records)
            write_contradictions_csv(csv_path, records)

            json_row = json.loads(jsonl_path.read_text().splitlines()[0])
            with csv_path.open() as stream:
                csv_row = next(csv.DictReader(stream))

        self.assertEqual(json_row["mode"], "retrospective")
        self.assertEqual(csv_row["mode"], "retrospective")
        self.assertEqual(json_row["family"], csv_row["family"])
        self.assertEqual(json_row["granularity"], csv_row["granularity"])

    def test_actor_confidence_marks_low_confidence_focus_unknown(self) -> None:
        grid = [[0 for _ in range(10)] for _ in range(10)]
        for row in range(4):
            for col in range(10):
                grid[row][col] = 1
        observation = obs(grid)
        scene = ObservationParser().parse(observation, frame_index=0)

        features = build_pre_features(
            before=observation,
            scene=scene,
            tracked=(),
            action=1,
        )

        self.assertEqual(features["actor_confidence"], "low")
        self.assertEqual(features["actor_or_focus_region"], "unknown")

    def test_feature_ablation_detects_synthetic_fragmentation_field(self) -> None:
        records = []
        for step in range(6):
            record = synthetic_record(step=step + 1)
            pre_features = dict(record.pre_features)
            pre_features["actor_signature"] = f"unique-{step}"
            records.append(TransitionRecord(**{**record.__dict__, "pre_features": pre_features}))

        report = feature_ablation_report(records)
        candidates = report["R2a_computable_static"]["fragmentation_candidates"]

        self.assertIn("actor_signature", {item["field"] for item in candidates})

    def test_r3_rejects_exact_fields_and_post_fields(self) -> None:
        with self.assertRaises(AssertionError):
            RepresentationFamily(
                "R3_bad_exact",
                frozenset({"actor_bbox_or_pos"}),
            ).assert_valid()

        with self.assertRaises(AssertionError):
            RepresentationFamily(
                "R3_bad_post",
                frozenset({"actor_or_focus_region", "remote_delta_present"}),
            ).assert_valid()

        R3A_COMPRESSED.assert_valid()

    def test_role_memory_updates_after_scoring_and_respects_cluster_cap(self) -> None:
        records = []
        for step, target in enumerate(("palette_or_tool", "canvas", "hud", "background"), start=1):
            record = synthetic_record(action=6, coarse="hud_delta", step=step)
            pre_features = {
                **record.pre_features,
                "action_kind": "click",
                "action_target_type": target,
                "target_relative_region": target,
                "clicked_region_signature": {"target": target},
            }
            records.append(TransitionRecord(**{**record.__dict__, "pre_features": pre_features}))
        memory = RoleMemory(max_clusters=2)

        role_before = memory.context_features(())["role_clusters"]
        for record in records:
            memory.update(record)
        role_after = memory.context_features(())["role_clusters"]

        self.assertEqual(role_before, [])
        self.assertLessEqual(len(role_after), 2)

    def test_cd82_style_click_trace_splits_action_six_by_target(self) -> None:
        palette = synthetic_record(action=6, coarse="hud_delta", step=1)
        canvas = synthetic_record(action=6, coarse="resource_delta", step=2)
        palette = TransitionRecord(
            **{
                **palette.__dict__,
                "pre_features": {
                    **palette.pre_features,
                    "action_kind": "click",
                    "action_target_type": "palette_or_tool",
                    "target_relative_region": "top",
                },
            }
        )
        canvas = TransitionRecord(
            **{
                **canvas.__dict__,
                "pre_features": {
                    **canvas.pre_features,
                    "action_kind": "click",
                    "action_target_type": "canvas",
                    "target_relative_region": "central_canvas",
                },
            }
        )

        summary, _ = retrospective_metrics([palette, canvas], R3A_COMPRESSED, granularity="coarse")

        self.assertEqual(summary.unique_contexts, 2)
        self.assertEqual(summary.contradiction_rate, 0.0)

    def test_ar25_style_translated_shapes_reuse_under_compressed_region(self) -> None:
        first = synthetic_record(step=1)
        second = synthetic_record(step=2)
        first_features = {
            **first.pre_features,
            "actor_bbox_or_pos": {"top": 4, "left": 4},
            "actor_or_focus_region": "central_canvas",
        }
        second_features = {
            **second.pre_features,
            "actor_bbox_or_pos": {"top": 9, "left": 9},
            "actor_or_focus_region": "central_canvas",
        }
        first = TransitionRecord(**{**first.__dict__, "pre_features": first_features})
        second = TransitionRecord(**{**second.__dict__, "pre_features": second_features})

        r2_summary, _ = retrospective_metrics([first, second], R2A_STATIC, granularity="coarse")
        r3_summary, _ = retrospective_metrics([first, second], R3A_COMPRESSED, granularity="coarse")

        self.assertEqual(r2_summary.reuse_coverage, 0.0)
        self.assertEqual(r3_summary.reuse_coverage, 1.0)

    def test_r3a_frozen_field_list_is_emitted_in_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            output = Path(tmp) / "out"
            start = {
                "type": "observation",
                "step": 0,
                "pixels": [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                "available_actions": [1, 2],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
                "full_reset": True,
            }
            transition = {
                "type": "transition",
                "step": 1,
                "action": 2,
                "action_data": None,
                "observation": {
                    "pixels": [[[0, 0, 0], [0, 0, 1], [0, 0, 0]]],
                    "available_actions": [1, 2],
                    "levels_completed": 0,
                    "state": "NOT_FINISHED",
                    "full_reset": False,
                },
            }
            trace.write_text(json.dumps(start) + "\n" + json.dumps(transition) + "\n")

            evaluate_suite((TraceSpec("toy", trace, "human"),), output_dir=output)
            audit = json.loads((output / "representation_field_audit.json").read_text())

        self.assertEqual(
            audit["frozen_baselines"][R3A_COMPRESSED.name]["pre_action_fields"],
            list(R3A_FROZEN_FIELDS),
        )

    def test_r4_rejects_post_fields_raw_ids_hashes_and_current_motion(self) -> None:
        for field in (
            "relative_group_motion_class",
            "raw_observation_hash",
            "object_id",
            "actor_bbox_or_pos",
        ):
            with self.assertRaises(AssertionError):
                RepresentationFamily("R4_bad", frozenset({field})).assert_valid()

        R4_RELATIONAL_GEOMETRY_ONLY.assert_valid()

    def test_parameter_metrics_detect_schema_stable_parameter_split(self) -> None:
        first = synthetic_record(
            coarse="actor_move",
            exact={
                "actor_delta": [0, 1],
                "changed_cell_signature": {"bbox": [1, 1, 1, 2]},
                "changed_value_signature": ((1, 0, "one"),),
                "hud_delta_signature": (),
                "resource_delta_signature": (),
                "status": "NOT_FINISHED",
                "status_delta": 0,
            },
        )
        second = synthetic_record(
            coarse="actor_move",
            exact={
                "actor_delta": [1, 0],
                "changed_cell_signature": {"bbox": [1, 1, 2, 1]},
                "changed_value_signature": ((1, 0, "one"),),
                "hud_delta_signature": (),
                "resource_delta_signature": (),
                "status": "NOT_FINISHED",
                "status_delta": 0,
            },
            step=2,
        )

        metrics = prequential_parameter_metrics([first, second], R3A_COMPRESSED)

        self.assertEqual(metrics["schema_contradiction_rate"], 0.0)
        self.assertEqual(metrics["parameter_contradiction_given_schema"], 1.0)

    def test_minimum_support_gate_can_mark_probe_insufficient(self) -> None:
        metrics = {
            "examples": 100,
            "schema_reused_contexts": 3,
            "schema_reused_transitions": 12,
            "schema_reuse_coverage": 0.12,
        }

        self.assertEqual(r4_support_status(metrics)["status"], "insufficient_support")

    def test_gain_per_reuse_lost_is_finite(self) -> None:
        baseline = {
            "parameter_contradiction_given_schema": 1.0,
            "schema_reuse_coverage": 0.5,
            "schema_one_off_ratio": 0.4,
        }
        candidate = {
            "examples": 100,
            "schema_reused_contexts": 20,
            "schema_reused_transitions": 40,
            "schema_reuse_coverage": 0.45,
            "schema_one_off_ratio": 0.42,
            "parameter_contradiction_given_schema": 0.7,
        }

        comparison = r4_probe_comparison(
            baseline,
            candidate,
            game="ar25",
            family="R4_relational_geometry_only",
        )

        self.assertGreater(comparison["gain_per_reuse_lost"], 0.0)
        self.assertEqual(comparison["support"]["status"], "eligible")

    def test_click_procedural_memory_updates_after_current_step(self) -> None:
        first = synthetic_record(action=6, coarse="hud_delta", step=1)
        first = TransitionRecord(
            **{
                **first.__dict__,
                "action_data": {"x": 1, "y": 1},
                "pre_features": {
                    **first.pre_features,
                    "click_target_type_visual": "palette_or_tool_visual",
                    "click_target_zone": "top",
                    "click_target_visual_signature": {"color": 2},
                },
            }
        )
        memory = ClickProceduralMemory()

        before = memory.context_features_for_record(first)
        memory.update(first)
        after = memory.context_features_for_record(first)

        self.assertIsNone(before["selected_tool_proxy_before_t"])
        self.assertIsNotNone(after["selected_tool_proxy_before_t"])

    def test_target_differs_from_initial_excludes_current_action_change(self) -> None:
        first = synthetic_record(action=6, coarse="resource_delta", step=1)
        first = TransitionRecord(
            **{
                **first.__dict__,
                "action_data": {"x": 1, "y": 1},
                "pre_features": {
                    **first.pre_features,
                    "click_target_type_visual": "canvas_visual",
                    "click_target_zone": "central_canvas",
                    "click_target_visual_signature": {"color": 1},
                },
            }
        )
        second = TransitionRecord(
            **{
                **first.__dict__,
                "step": 2,
                "before": obs([[0, 0, 0], [0, 2, 0], [0, 0, 0]]),
            }
        )
        memory = ClickProceduralMemory()

        first_context = memory.context_features_for_record(first)
        memory.update(first)
        second_context = memory.context_features_for_record(second)

        self.assertEqual(first_context["target_differs_from_initial_frame_before_t"], "same")
        self.assertEqual(second_context["target_differs_from_initial_frame_before_t"], "changed")

    def test_causal_stages_do_not_enter_context_keys(self) -> None:
        key = context_key(synthetic_record(), R3A_COMPRESSED)

        for stage in ("actor_move", "independent_tick", "contact_transform", "terminal"):
            self.assertNotIn(stage, key)

    def test_negative_control_logic_flags_fragmentation(self) -> None:
        baseline = {
            "parameter_contradiction_given_schema": 1.0,
            "schema_reuse_coverage": 0.8,
            "schema_one_off_ratio": 0.1,
        }
        candidate = {
            "examples": 100,
            "schema_reused_contexts": 20,
            "schema_reused_transitions": 40,
            "schema_reuse_coverage": 0.4,
            "schema_one_off_ratio": 0.5,
            "parameter_contradiction_given_schema": 0.5,
        }

        comparison = r4_probe_comparison(
            baseline,
            candidate,
            game="ls20",
            family="R4_relational_geometry_only",
        )

        self.assertTrue(comparison["is_negative_control"])
        self.assertTrue(comparison["suspicious_fragmentation_flag"])
        self.assertEqual(comparison["verdict"], "suspicious_fragmentation")

    def test_new_reports_cover_r3_and_r4_without_game_branches(self) -> None:
        records = [synthetic_record(step=1), synthetic_record(step=2)]

        schema = event_schema_metrics_report(records)
        causal = causal_stage_decomposition_report(records)

        self.assertIn(R3A_COMPRESSED.name, schema)
        self.assertIn(R4_CLICK_PROCEDURAL_LEARNED_ONLY.name, schema)
        self.assertIn("stage_metrics", causal)

    def test_labeler_v2_does_not_emit_low_confidence_actor_delta(self) -> None:
        before = obs([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        after = obs([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        before_scene = ObservationParser().parse(before, frame_index=0)
        after_scene = ObservationParser().parse(after, frame_index=1)
        tracker = ObjectTracker()
        tracker.update(before_scene, force_new=True)
        track_update = tracker.update(after_scene)
        event = type(
            "FakeEvent",
            (),
            {
                "action": 1,
                "object_deltas": tuple(
                    type(
                        "FakeDelta",
                        (),
                        {
                            "moved": True,
                            "translation": (0, 1),
                            "value_before": 1,
                            "value_after": 1,
                            "shape_changed": False,
                            "area_before": 1,
                            "area_after": 1,
                            "bbox_before": (1, 1, 1, 1),
                            "bbox_after": (1, 2, 1, 2),
                        },
                    )()
                    for _ in track_update.matches[:1]
                ),
            },
        )()
        labels = build_post_labels_v2(
            before=before,
            after=after,
            before_scene=before_scene,
            event=event,  # type: ignore[arg-type]
            pre_features={"actor_confidence": "low", "action_kind": "move"},
            action_data=None,
        )

        self.assertEqual(labels["actor_confidence"], "low")
        self.assertIsNone(labels["channels"]["actor_delta"]["delta"])

    def test_labeler_v2_assigns_click_edit_to_action_target_delta(self) -> None:
        before_grid = [[0 for _ in range(20)] for _ in range(20)]
        after_grid = [[0 for _ in range(20)] for _ in range(20)]
        before_grid[10][10] = 1
        after_grid[10][10] = 2
        before = obs(before_grid)
        after = obs(after_grid)
        before_scene = ObservationParser().parse(before, frame_index=0)
        event = type("FakeEvent", (), {"action": 6, "object_deltas": ()})()

        labels = build_post_labels_v2(
            before=before,
            after=after,
            before_scene=before_scene,
            event=event,  # type: ignore[arg-type]
            pre_features={"actor_confidence": "low", "action_kind": "click"},
            action_data={"x": 10, "y": 10},
        )

        self.assertGreater(labels["channels"]["action_target_delta"]["cluster_count"], 0)
        self.assertEqual(labels["channels"]["actor_delta"]["confidence"], "low")

    def test_labeler_v3_recovers_controlled_entity_from_action_conditioned_motion(self) -> None:
        def move_record(step: int) -> TransitionRecord:
            record = synthetic_record(action=4, step=step)
            delta = type(
                "FakeDelta",
                (),
                {
                    "object_id": 7,
                    "moved": True,
                    "translation": (0, 1),
                    "value_before": 2,
                    "value_after": 2,
                    "shape_changed": False,
                    "area_before": 1,
                    "area_after": 1,
                    "bbox_before": (1, 1, 1, 1),
                    "bbox_after": (1, 2, 1, 2),
                },
            )()
            cluster = {
                "cluster_id": 1,
                "bbox": (1, 2, 1, 2),
                "area_bucket": "one",
                "color_histogram_delta": ((0, 2, "one"),),
                "distance_to_actor": "unknown",
                "distance_to_action_target": "unknown",
                "overlaps_hud": False,
                "overlaps_known_component": True,
                "centroid_region": "central_canvas",
            }
            return TransitionRecord(
                **{
                    **record.__dict__,
                    "event": type("FakeEvent", (), {"action": 4, "object_deltas": (delta,)})(),
                    "pre_features": {**record.pre_features, "action_kind": "move"},
                    "post_labels": {
                        **record.post_labels,
                        "changed_cells": 1,
                        "labeler_v2": {
                            "changed_region_clusters": (cluster,),
                            "channels": {},
                            "active_channels": (),
                        },
                    },
                }
            )

        records = [move_record(1), move_record(2)]
        annotate_labeler_v3(records)

        audit = controlled_entity_audit(records)
        self.assertGreater(audit["controlled_delta_assignment_rate"], 0.0)
        self.assertIn(
            records[0].post_labels["labeler_v3"]["controlled_entity"]["type"],
            {"single", "coupled_set"},
        )

    def test_labeler_v3_assigns_click_edit_to_action_target_channel(self) -> None:
        before_grid = [[0 for _ in range(20)] for _ in range(20)]
        after_grid = [[0 for _ in range(20)] for _ in range(20)]
        before_grid[10][10] = 1
        after_grid[10][10] = 2
        record = synthetic_record(action=6, step=1)
        cluster = {
            "cluster_id": 1,
            "bbox": (10, 10, 10, 10),
            "area_bucket": "one",
            "color_histogram_delta": ((1, 2, "one"),),
            "distance_to_actor": "unknown",
            "distance_to_action_target": "zero",
            "overlaps_hud": False,
            "overlaps_known_component": True,
            "centroid_region": "central_canvas",
        }
        record = TransitionRecord(
            **{
                **record.__dict__,
                "before": obs(before_grid),
                "after": obs(after_grid),
                "action_data": {"x": 10, "y": 10},
                "event": type("FakeEvent", (), {"action": 6, "object_deltas": ()})(),
                "pre_features": {**record.pre_features, "action_kind": "click"},
                "post_labels": {
                    **record.post_labels,
                    "changed_cells": 1,
                    "labeler_v2": {
                        "changed_region_clusters": (cluster,),
                        "channels": {},
                        "active_channels": (),
                    },
                },
            }
        )

        annotate_labeler_v3([record])
        delta_report = delta_channel_assignment_v3_report([record])

        self.assertGreater(delta_report["click_target_assignment_rate"], 0.0)
        self.assertGreater(
            record.post_labels["labeler_v3"]["channels"]["action_target_delta"]["cluster_count"],
            0,
        )
        self.assertIn("canvas_edit_surface", json.dumps(region_role_map_report([record])))

    def test_labeler_v3_metrics_and_stage_report_run_on_synthetic_records(self) -> None:
        first = synthetic_record(step=1)
        second = synthetic_record(step=2)
        first = TransitionRecord(**{**first.__dict__, "post_labels": {**first.post_labels, "labeler_v3": {"coarse_outcome": "action_target_delta", "parameter_outcome": {"target": 1}, "active_channels": ("action_target_delta",), "channels": {"action_target_delta": {"cluster_count": 1, "bboxes": ((1, 1, 1, 1),)}}}}})
        second = TransitionRecord(**{**second.__dict__, "post_labels": {**second.post_labels, "labeler_v3": {"coarse_outcome": "action_target_delta", "parameter_outcome": {"target": 2}, "active_channels": ("action_target_delta",), "channels": {"action_target_delta": {"cluster_count": 1, "bboxes": ((2, 2, 2, 2),)}}}}})

        metrics = prequential_parameter_metrics([first, second], R3A_COMPRESSED, labeler="v3")
        comparison = outcome_labeler_v2_vs_v3_report([first, second])
        stage = stage_report_v3([first, second])

        self.assertEqual(metrics["parameter_contradiction_given_schema"], 1.0)
        self.assertIn(R3A_COMPRESSED.name, comparison)
        self.assertEqual(stage["top_localizing_channel"], "action_target_delta")

    def test_parameter_probe_local_geometry_can_beat_majority_baseline(self) -> None:
        def probe_record(step: int, row: int, parameter: str) -> TransitionRecord:
            grid = [[0 for _ in range(12)] for _ in range(12)]
            grid[row][5] = 2
            before = obs(grid)
            scene = ObservationParser().parse(before, frame_index=step)
            record = synthetic_record(step=step)
            return TransitionRecord(
                **{
                    **record.__dict__,
                    "game": "ls20",
                    "before": before,
                    "after": before,
                    "before_scene": scene,
                    "pre_features": {
                        **record.pre_features,
                        "action_kind": "move",
                        "actor_or_focus_region": "central_canvas",
                    },
                    "post_labels": {
                        **record.post_labels,
                        "labeler_v3": {
                            "coarse_outcome": "controlled_entity_delta",
                            "parameter_outcome": {"parameter": parameter},
                            "controlled_entity": {
                                "type": "single",
                                "confidence": "high",
                                "candidates": [
                                    {
                                        "score": 1.0,
                                        "translation": [0, 1],
                                        "bbox_before": (row, 5, row, 5),
                                        "bbox_after": (row, 6, row, 6),
                                    }
                                ],
                            },
                            "channels": {},
                            "active_channels": ("controlled_entity_delta",),
                        },
                    },
                }
            )

        records = [
            probe_record(1, 4, "central"),
            probe_record(2, 9, "edge"),
            probe_record(3, 4, "central"),
            probe_record(4, 9, "edge"),
        ]

        p0 = parameter_probe_metrics(records, model_kind="P0")
        p1 = parameter_probe_metrics(records, model_kind="P1")
        report = parameter_model_probe_report(records)

        self.assertLess(p1["parameter_error_given_schema"], p0["parameter_error_given_schema"])
        self.assertIn("P1_local_geometry_parameter_model", report["models"])

    def test_parameter_probe_skips_cd82(self) -> None:
        record = synthetic_record()
        record = TransitionRecord(**{**record.__dict__, "game": "cd82"})

        report = parameter_model_probe_report([record])

        self.assertEqual(report["status"], "skipped_not_parameter_probe_game")

    def test_motion_law_probe_skips_cd82(self) -> None:
        record = synthetic_record()
        record = TransitionRecord(**{**record.__dict__, "game": "cd82"})

        report = motion_law_probe_report([record])

        self.assertEqual(report["status"], "skipped_not_motion_probe_game")

    def test_motion_target_granularity_skips_cd82(self) -> None:
        record = synthetic_record()
        record = TransitionRecord(**{**record.__dict__, "game": "cd82"})

        report = motion_target_granularity_report([record])

        self.assertEqual(report["status"], "skipped_not_motion_granularity_game")

    def test_controlled_region_material_decomposition_skips_cd82(self) -> None:
        record = synthetic_record()
        record = TransitionRecord(**{**record.__dict__, "game": "cd82"})

        report = controlled_region_material_decomposition_report([record])

        self.assertEqual(report["status"], "skipped_not_controlled_region_material_game")

    def test_changed_region_factorization_skips_cd82(self) -> None:
        record = synthetic_record()
        record = TransitionRecord(**{**record.__dict__, "game": "cd82"})

        report = changed_region_factorization_report([record])

        self.assertEqual(report["status"], "skipped_not_changed_region_factorization_game")

    def test_t5_failure_anatomy_skips_cd82(self) -> None:
        record = synthetic_record()
        record = TransitionRecord(**{**record.__dict__, "game": "cd82"})

        report = t5_failure_anatomy_report([record])

        self.assertEqual(report["status"], "skipped_not_t5_failure_anatomy_game")

    def test_independent_coupled_mover_probe_skips_cd82(self) -> None:
        record = synthetic_record()
        record = TransitionRecord(**{**record.__dict__, "game": "cd82"})

        report = independent_coupled_mover_probe_report([record])

        self.assertEqual(report["status"], "skipped_not_independent_coupled_mover_game")

    def test_independent_mover_dynamics_skips_cd82(self) -> None:
        record = synthetic_record()
        record = TransitionRecord(**{**record.__dict__, "game": "cd82"})

        report = independent_mover_dynamics_report([record])

        self.assertEqual(report["status"], "skipped_not_independent_mover_game")

    def test_motion_probe_local_features_can_reduce_controlled_error(self) -> None:
        def probe_record(step: int, row: int, translation: list[int]) -> TransitionRecord:
            base = synthetic_record(action=4, step=step)
            grid = [[0 for _ in range(12)] for _ in range(12)]
            grid[row][5] = 1
            after_grid = [[0 for _ in range(12)] for _ in range(12)]
            after_col = 5 + int(translation[1])
            after_grid[row][after_col] = 1
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ls20",
                    "before": obs(grid),
                    "after": obs(after_grid),
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "coarse_outcome": "controlled_entity_delta",
                            "parameter_outcome": {"translation": translation},
                            "controlled_entity": {
                                "type": "single",
                                "confidence": "high",
                                "candidates": [
                                    {
                                        "object_id": step,
                                        "score": 1.0,
                                        "translation": translation,
                                        "bbox_before": (row, 5, row, 5),
                                        "bbox_after": (row, after_col, row, after_col),
                                    }
                                ],
                            },
                            "channels": {
                                "controlled_entity_delta": {
                                    "cluster_count": 1,
                                    "top_candidates": [
                                        {
                                            "score": 1.0,
                                            "translation": translation,
                                            "bbox_before": (row, 5, row, 5),
                                            "bbox_after": (row, after_col, row, after_col),
                                        }
                                    ],
                                }
                            },
                            "active_channels": ("controlled_entity_delta",),
                        },
                    },
                }
            )

        records = [
            probe_record(1, 4, [0, 1]),
            probe_record(2, 9, [0, 2]),
            probe_record(3, 4, [0, 1]),
            probe_record(4, 9, [0, 2]),
        ]

        phase_cluster = {"k": 0, "assignments": {}, "cluster_sizes": {}}
        m0 = motion_probe_metrics(records, head="controlled_entity_delta", model_kind="M0", phase_cluster=phase_cluster)
        m1 = motion_probe_metrics(records, head="controlled_entity_delta", model_kind="M1", phase_cluster=phase_cluster)

        self.assertLess(m1["parameter_error"], m0["parameter_error"])
        self.assertIn("feature_cardinality", m1)

    def test_motion_target_granularity_reports_fine_target_break(self) -> None:
        def probe_record(step: int, translation: list[int]) -> TransitionRecord:
            base = synthetic_record(action=4, step=step)
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ls20",
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "coarse_outcome": "controlled_entity_delta",
                            "parameter_outcome": {"translation": translation},
                            "controlled_entity": {
                                "type": "single",
                                "confidence": "high",
                                "candidates": [
                                    {
                                        "object_id": step,
                                        "score": 1.0,
                                        "translation": translation,
                                        "bbox_before": (1, 1, 1, 1),
                                        "bbox_after": (1 + translation[0], 1 + translation[1], 1 + translation[0], 1 + translation[1]),
                                    }
                                ],
                            },
                            "channels": {
                                "controlled_entity_delta": {
                                    "cluster_count": 1,
                                    "top_candidates": [{"translation": translation}],
                                }
                            },
                            "active_channels": ("controlled_entity_delta",),
                        },
                    },
                }
            )

        records = [probe_record(index + 1, [0, 1] if index < 3 else [0, 2]) for index in range(6)]

        report = motion_target_granularity_report(records)

        self.assertEqual(report["status"], "evaluated")
        self.assertIn("first_granularity_where_strict_M0_fails", report)
        self.assertNotIn("first_granularity_where_M0_fails", report)
        self.assertIn("T0_controlled_motion_happened", report["targets"])
        self.assertIn("T5_exact_controlled_region_parameter", report["targets"])
        target = report["targets"]["T0_controlled_motion_happened"]
        self.assertIn("strict_conjunctive_baseline", target)
        self.assertIn("M0_atom_vote_R3a_schema_action", target["models"])
        self.assertIn("atom_vote_vs_strict_baseline", target)
        self.assertIn("comparisons_to_strict_M0", target)

    def test_controlled_region_material_decomposition_splits_t5_subtargets(self) -> None:
        def probe_record(step: int, color_after: int) -> TransitionRecord:
            base = synthetic_record(action=4, step=step)
            grid = [[0 for _ in range(5)] for _ in range(5)]
            grid[2][2] = 1
            after_grid = [[0 for _ in range(5)] for _ in range(5)]
            after_grid[2][3] = color_after
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ls20",
                    "before": obs(grid),
                    "after": obs(after_grid),
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "coarse_outcome": "controlled_entity_delta",
                            "parameter_outcome": {"controlled_entity_delta": {"cluster_count": 1}},
                            "controlled_entity": {
                                "type": "single",
                                "confidence": "high",
                                "candidates": [
                                    {
                                        "object_id": 1,
                                        "score": 1.0,
                                        "translation": [0, 1],
                                        "bbox_before": (2, 2, 2, 2),
                                        "bbox_after": (2, 3, 2, 3),
                                    }
                                ],
                            },
                            "changed_region_clusters": (
                                {
                                    "cluster_id": 1,
                                    "bbox": (2, 2, 2, 3),
                                    "area_bucket": "few",
                                    "color_histogram_delta": ((1, color_after, "one"),),
                                },
                            ),
                            "changed_region_assignments": {
                                "controlled_entity_delta": [{"cluster_id": 1, "confidence": "high"}],
                            },
                            "channels": {
                                "controlled_entity_delta": {
                                    "cluster_count": 1,
                                    "area_buckets": ("few",),
                                    "bboxes": ((2, 2, 2, 3),),
                                    "color_histogram_delta": ((1, color_after, "one"),),
                                    "top_candidates": [
                                        {
                                            "translation": [0, 1],
                                            "bbox_before": (2, 2, 2, 2),
                                            "bbox_after": (2, 3, 2, 3),
                                        }
                                    ],
                                }
                            },
                            "active_channels": ("controlled_entity_delta",),
                        },
                    },
                }
            )

        records = [
            probe_record(1, 2),
            probe_record(2, 3),
            probe_record(3, 2),
            probe_record(4, 3),
        ]

        report = controlled_region_material_decomposition_report(records)

        self.assertEqual(report["status"], "evaluated")
        self.assertEqual(
            report["probe_type"],
            "diagnostic_t5_outcome_decomposition_not_new_representation",
        )
        self.assertIn("T5a_identity_correspondence", report["subtargets"])
        self.assertIn("T5b_footprint_changed_cells", report["subtargets"])
        self.assertIn("T5c_material_color_value_transition", report["subtargets"])
        self.assertIn("T5d_overlap_merge_split_behavior", report["subtargets"])
        self.assertIn("T5e_collateral_changed_regions", report["subtargets"])
        material = report["subtargets"]["T5c_material_color_value_transition"]
        self.assertIn("strict_conjunctive_baseline", material)
        self.assertGreater(material["strict_conjunctive_baseline"]["target_entropy"], 0.0)

    def test_changed_region_factorization_separates_old_and_new_footprints(self) -> None:
        def probe_record(step: int, old_color_after: int) -> TransitionRecord:
            base = synthetic_record(action=4, step=step)
            grid = [[0 for _ in range(6)] for _ in range(6)]
            grid[2][2] = 1
            after_grid = [[0 for _ in range(6)] for _ in range(6)]
            after_grid[2][2] = old_color_after
            after_grid[2][3] = 1
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ls20",
                    "before": obs(grid),
                    "after": obs(after_grid),
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "coarse_outcome": "controlled_entity_delta",
                            "parameter_outcome": {"controlled_entity_delta": {"cluster_count": 2}},
                            "controlled_entity": {
                                "type": "single",
                                "confidence": "high",
                                "candidates": [
                                    {
                                        "object_id": 1,
                                        "score": 1.0,
                                        "translation": [0, 1],
                                        "bbox_before": (2, 2, 2, 2),
                                        "bbox_after": (2, 3, 2, 3),
                                    }
                                ],
                            },
                            "changed_region_clusters": (
                                {
                                    "cluster_id": 1,
                                    "bbox": (2, 2, 2, 2),
                                    "area_bucket": "one",
                                    "color_histogram_delta": ((1, old_color_after, "one"),),
                                    "distance_to_actor": "zero",
                                    "overlaps_known_component": True,
                                    "overlaps_hud": False,
                                },
                                {
                                    "cluster_id": 2,
                                    "bbox": (2, 3, 2, 3),
                                    "area_bucket": "one",
                                    "color_histogram_delta": ((0, 1, "one"),),
                                    "distance_to_actor": "near",
                                    "overlaps_known_component": False,
                                    "overlaps_hud": False,
                                },
                            ),
                            "changed_region_assignments": {
                                "controlled_entity_delta": [
                                    {"cluster_id": 1, "confidence": "high"},
                                    {"cluster_id": 2, "confidence": "high"},
                                ],
                            },
                            "channels": {
                                "controlled_entity_delta": {
                                    "cluster_count": 2,
                                    "area_buckets": ("one", "one"),
                                    "bboxes": ((2, 2, 2, 2), (2, 3, 2, 3)),
                                    "color_histogram_delta": (
                                        (1, old_color_after, "one"),
                                        (0, 1, "one"),
                                    ),
                                    "top_candidates": [{"translation": [0, 1]}],
                                }
                            },
                            "active_channels": ("controlled_entity_delta",),
                        },
                    },
                }
            )

        report = changed_region_factorization_report(
            [
                probe_record(1, 0),
                probe_record(2, 2),
                probe_record(3, 0),
                probe_record(4, 2),
            ]
        )

        self.assertEqual(report["status"], "evaluated")
        self.assertIn("T5b_footprint", report["targets"])
        self.assertIn("old_v3_channel_target", report["targets"]["T5b_footprint"])
        self.assertIn("factorized_target", report["targets"]["T5b_footprint"])
        self.assertGreater(report["audit"]["role_distribution"]["controlled_footprint_old"], 0)
        self.assertGreater(report["audit"]["role_distribution"]["controlled_footprint_new"], 0)

    def test_t5_failure_anatomy_reports_slices_and_examples(self) -> None:
        def probe_record(step: int, old_color_after: int, extra_cluster: bool = False) -> TransitionRecord:
            base = synthetic_record(action=4, step=step)
            grid = [[0 for _ in range(8)] for _ in range(8)]
            grid[2][2] = 1
            after_grid = [[0 for _ in range(8)] for _ in range(8)]
            after_grid[2][2] = old_color_after
            after_grid[2][3] = 1
            clusters = [
                {
                    "cluster_id": 1,
                    "bbox": (2, 2, 2, 2),
                    "area_bucket": "one",
                    "color_histogram_delta": ((1, old_color_after, "one"),),
                    "distance_to_actor": "zero",
                    "overlaps_known_component": True,
                    "overlaps_hud": False,
                },
                {
                    "cluster_id": 2,
                    "bbox": (2, 3, 2, 3),
                    "area_bucket": "one",
                    "color_histogram_delta": ((0, 1, "one"),),
                    "distance_to_actor": "near",
                    "overlaps_known_component": False,
                    "overlaps_hud": False,
                },
            ]
            if extra_cluster:
                after_grid[6][6] = 5
                clusters.append(
                    {
                        "cluster_id": 3,
                        "bbox": (6, 6, 6, 6),
                        "area_bucket": "one",
                        "color_histogram_delta": ((0, 5, "one"),),
                        "distance_to_actor": "far",
                        "overlaps_known_component": False,
                        "overlaps_hud": False,
                    }
                )
            assignments = [{"cluster_id": 1, "confidence": "high"}, {"cluster_id": 2, "confidence": "high"}]
            remote_assignments = [{"cluster_id": 3, "confidence": "medium"}] if extra_cluster else []
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ls20",
                    "before": obs(grid),
                    "after": obs(after_grid),
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "coarse_outcome": "controlled_entity_delta",
                            "parameter_outcome": {"controlled_entity_delta": {"cluster_count": len(clusters)}},
                            "controlled_entity": {
                                "type": "single",
                                "confidence": "high",
                                "candidates": [
                                    {
                                        "object_id": 1,
                                        "score": 1.0,
                                        "translation": [0, 1],
                                        "bbox_before": (2, 2, 2, 2),
                                        "bbox_after": (2, 3, 2, 3),
                                    }
                                ],
                            },
                            "changed_region_clusters": tuple(clusters),
                            "changed_region_assignments": {
                                "controlled_entity_delta": assignments,
                                "remote_delta": remote_assignments,
                            },
                            "channels": {
                                "controlled_entity_delta": {
                                    "cluster_count": 2,
                                    "area_buckets": ("one", "one"),
                                    "bboxes": ((2, 2, 2, 2), (2, 3, 2, 3)),
                                    "color_histogram_delta": (
                                        (1, old_color_after, "one"),
                                        (0, 1, "one"),
                                    ),
                                    "top_candidates": [{"translation": [0, 1]}],
                                },
                                "remote_delta": {
                                    "cluster_count": 1 if extra_cluster else 0,
                                    "area_buckets": ("one",) if extra_cluster else (),
                                    "bboxes": ((6, 6, 6, 6),) if extra_cluster else (),
                                    "color_histogram_delta": ((0, 5, "one"),) if extra_cluster else (),
                                },
                            },
                            "active_channels": ("controlled_entity_delta", "remote_delta") if extra_cluster else ("controlled_entity_delta",),
                        },
                    },
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            report = t5_failure_anatomy_report(
                [
                    probe_record(1, 0),
                    probe_record(2, 2, extra_cluster=True),
                    probe_record(3, 0),
                    probe_record(4, 2, extra_cluster=True),
                ],
                prototype_dir=Path(tmp) / "prototypes",
            )

            self.assertEqual(report["status"], "evaluated")
            self.assertIn("error_by_action_schema", report)
            self.assertIn("error_by_factor_class", report)
            self.assertIn("factorizer_confidence_audit", report)
            self.assertIn("independent_mover_overlap", report)
            examples = report["top_failure_examples"]["T5c_material_value"]
            self.assertTrue(examples)
            self.assertTrue(Path(examples[0]["before_after_prototype_path"]).exists())
            self.assertIn("controlled_footprint_old", report["error_by_factor_class"])

    def test_independent_coupled_mover_probe_reports_two_heads(self) -> None:
        def probe_record(step: int, independent_col_after: int) -> TransitionRecord:
            base = synthetic_record(action=4, step=step)
            grid = [[0 for _ in range(8)] for _ in range(8)]
            grid[2][2] = 1
            grid[5][2] = 7
            after_grid = [[0 for _ in range(8)] for _ in range(8)]
            after_grid[2][3] = 1
            after_grid[5][independent_col_after] = 7
            controlled_delta = type(
                "FakeDelta",
                (),
                {
                    "object_id": 1,
                    "moved": True,
                    "translation": (0, 1),
                    "value_before": 1,
                    "value_after": 1,
                    "value_changed": False,
                    "shape_changed": False,
                    "area_before": 1,
                    "area_after": 1,
                    "area_delta": 0,
                    "bbox_before": (2, 2, 2, 2),
                    "bbox_after": (2, 3, 2, 3),
                    "shape_transform": "same",
                },
            )()
            independent_delta = type(
                "FakeDelta",
                (),
                {
                    "object_id": 2,
                    "moved": True,
                    "translation": (0, independent_col_after - 2),
                    "value_before": 7,
                    "value_after": 7,
                    "value_changed": False,
                    "shape_changed": False,
                    "area_before": 1,
                    "area_after": 1,
                    "area_delta": 0,
                    "bbox_before": (5, 2, 5, 2),
                    "bbox_after": (5, independent_col_after, 5, independent_col_after),
                    "shape_transform": "same",
                },
            )()
            motion_group = type("FakeMotionGroup", (), {"translation": (0, 1), "object_ids": (1, 2)})()
            event = type(
                "FakeEvent",
                (),
                {
                    "action": 4,
                    "object_deltas": (controlled_delta, independent_delta),
                    "created_object_ids": (),
                    "disappeared_object_ids": (),
                    "reappeared_object_ids": (),
                    "motion_groups": (motion_group,),
                },
            )()
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ar25",
                    "before": obs(grid),
                    "after": obs(after_grid),
                    "event": event,
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "coarse_outcome": "controlled_entity_delta+independent_motion_delta",
                            "parameter_outcome": {"controlled_entity_delta": {"cluster_count": 1}},
                            "controlled_entity": {
                                "type": "single",
                                "confidence": "high",
                                "candidates": [
                                    {
                                        "object_id": 1,
                                        "score": 1.0,
                                        "translation": [0, 1],
                                        "bbox_before": (2, 2, 2, 2),
                                        "bbox_after": (2, 3, 2, 3),
                                    }
                                ],
                            },
                            "changed_region_clusters": (
                                {
                                    "cluster_id": 1,
                                    "bbox": (2, 2, 2, 3),
                                    "area_bucket": "few",
                                    "color_histogram_delta": ((0, 1, "one"), (1, 0, "one")),
                                    "distance_to_actor": "zero",
                                    "overlaps_known_component": True,
                                    "overlaps_hud": False,
                                },
                                {
                                    "cluster_id": 2,
                                    "bbox": (5, 2, 5, independent_col_after),
                                    "area_bucket": "few",
                                    "color_histogram_delta": ((0, 7, "one"), (7, 0, "one")),
                                    "distance_to_actor": "far",
                                    "overlaps_known_component": True,
                                    "overlaps_hud": False,
                                },
                            ),
                            "changed_region_assignments": {
                                "controlled_entity_delta": [{"cluster_id": 1, "confidence": "high"}],
                                "independent_motion_delta": [{"cluster_id": 2, "confidence": "high"}],
                            },
                            "channels": {
                                "controlled_entity_delta": {
                                    "cluster_count": 1,
                                    "bboxes": ((2, 2, 2, 3),),
                                    "top_candidates": [{"translation": [0, 1]}],
                                },
                                "independent_motion_delta": {
                                    "cluster_count": 1,
                                    "bboxes": ((5, 2, 5, independent_col_after),),
                                },
                            },
                            "active_channels": ("controlled_entity_delta", "independent_motion_delta"),
                        },
                    },
                }
            )

        report = independent_coupled_mover_probe_report(
            [
                probe_record(1, 3),
                probe_record(2, 4),
                probe_record(3, 3),
                probe_record(4, 4),
            ]
        )

        self.assertEqual(report["status"], "evaluated")
        self.assertIn("independent_mover_fsa", report)
        self.assertIn("coupled_group_motion", report)
        self.assertIn("mover_presence", report["independent_mover_fsa"]["targets"])
        self.assertIn("single_vs_coupled_set", report["coupled_group_motion"]["targets"])
        self.assertIn("unexplained_extra_changed_region_reclassification", report)

    def test_mover_state_factorization_tracks_persistent_mover_state(self) -> None:
        def probe_record(step: int, independent_col_before: int) -> TransitionRecord:
            base = synthetic_record(action=4, step=step)
            independent_col_after = independent_col_before + 1
            grid = [[0 for _ in range(8)] for _ in range(8)]
            grid[2][2 + step] = 1
            grid[5][independent_col_before] = 7
            after_grid = [[0 for _ in range(8)] for _ in range(8)]
            after_grid[2][3 + step] = 1
            after_grid[5][independent_col_after] = 7
            controlled_delta = type(
                "FakeDelta",
                (),
                {
                    "object_id": 1,
                    "moved": True,
                    "translation": (0, 1),
                    "value_before": 1,
                    "value_after": 1,
                    "shape_changed": False,
                    "area_before": 1,
                    "area_after": 1,
                    "bbox_before": (2, 2 + step, 2, 2 + step),
                    "bbox_after": (2, 3 + step, 2, 3 + step),
                    "shape_transform": "same",
                },
            )()
            independent_delta = type(
                "FakeDelta",
                (),
                {
                    "object_id": 2,
                    "moved": True,
                    "translation": (0, 1),
                    "value_before": 7,
                    "value_after": 7,
                    "shape_changed": False,
                    "area_before": 1,
                    "area_after": 1,
                    "bbox_before": (5, independent_col_before, 5, independent_col_before),
                    "bbox_after": (5, independent_col_after, 5, independent_col_after),
                    "shape_transform": "same",
                },
            )()
            motion_group = type("FakeMotionGroup", (), {"translation": (0, 1), "object_ids": (1, 2)})()
            event = type(
                "FakeEvent",
                (),
                {
                    "action": 4,
                    "object_deltas": (controlled_delta, independent_delta),
                    "created_object_ids": (),
                    "disappeared_object_ids": (),
                    "reappeared_object_ids": (),
                    "motion_groups": (motion_group,),
                },
            )()
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ls20",
                    "before": obs(grid),
                    "after": obs(after_grid),
                    "event": event,
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "coarse_outcome": "controlled_entity_delta+independent_motion_delta",
                            "parameter_outcome": {"controlled_entity_delta": {"cluster_count": 1}},
                            "controlled_entity": {
                                "type": "single",
                                "confidence": "high",
                                "candidates": [
                                    {
                                        "object_id": 1,
                                        "score": 1.0,
                                        "translation": [0, 1],
                                        "bbox_before": controlled_delta.bbox_before,
                                        "bbox_after": controlled_delta.bbox_after,
                                    }
                                ],
                            },
                            "changed_region_clusters": (
                                {
                                    "cluster_id": 1,
                                    "bbox": (2, 2 + step, 2, 3 + step),
                                    "area_bucket": "few",
                                    "color_histogram_delta": ((0, 1, "one"),),
                                    "distance_to_actor": "zero",
                                    "overlaps_known_component": True,
                                    "overlaps_hud": False,
                                },
                                {
                                    "cluster_id": 2,
                                    "bbox": (5, independent_col_before, 5, independent_col_after),
                                    "area_bucket": "few",
                                    "color_histogram_delta": ((0, 7, "one"),),
                                    "distance_to_actor": "far",
                                    "overlaps_known_component": True,
                                    "overlaps_hud": False,
                                },
                            ),
                            "changed_region_assignments": {
                                "controlled_entity_delta": [{"cluster_id": 1, "confidence": "high"}],
                                "independent_motion_delta": [{"cluster_id": 2, "confidence": "high"}],
                            },
                            "channels": {
                                "controlled_entity_delta": {"cluster_count": 1},
                                "independent_motion_delta": {"cluster_count": 1},
                            },
                            "active_channels": ("controlled_entity_delta", "independent_motion_delta"),
                        },
                    },
                }
            )

        report = mover_state_factorization_report(
            [
                probe_record(1, 1),
                probe_record(2, 2),
                probe_record(3, 3),
            ]
        )

        self.assertEqual(report["status"], "evaluated")
        self.assertGreater(report["identity_linking"]["identity_link_success_rate"], 0.0)
        self.assertEqual(report["simple_kinematics"]["next_bbox_delta_error"], 0.0)
        self.assertIn("independent_vs_coupled", report)

    def test_mover_state_factorization_skips_cd82(self) -> None:
        record = TransitionRecord(**{**synthetic_record().__dict__, "game": "cd82"})
        report = mover_state_factorization_report([record])
        self.assertEqual(report["status"], "skipped_not_mover_state_factorization_game")

    def test_world_clock_transition_probe_classifies_hard_mover_cases(self) -> None:
        def fake_delta(object_id, before, after, translation):
            return type(
                "FakeDelta",
                (),
                {
                    "object_id": object_id,
                    "moved": True,
                    "translation": translation,
                    "value_before": 7,
                    "value_after": 7,
                    "shape_changed": False,
                    "area_before": 1,
                    "area_after": 1,
                    "bbox_before": before,
                    "bbox_after": after,
                    "shape_transform": "same",
                },
            )()

        def probe_record(step: int, deltas, *, groups=(), reappeared=(), disappeared=(), overlap=False):
            base = synthetic_record(action=4, step=step)
            clusters = []
            assignments = {"independent_motion_delta": [], "controlled_entity_delta": []}
            controlled = {
                "object_id": 1,
                "score": 1.0,
                "translation": [0, 1],
                "bbox_before": (3, 3, 3, 3),
                "bbox_after": (3, 4, 3, 4),
            }
            if overlap:
                clusters.append(
                    {
                        "cluster_id": 1,
                        "bbox": (3, 3, 3, 4),
                        "area_bucket": "few",
                        "color_histogram_delta": ((0, 1, "one"),),
                        "distance_to_actor": "zero",
                        "overlaps_known_component": True,
                        "overlaps_hud": False,
                    }
                )
                assignments["controlled_entity_delta"].append({"cluster_id": 1, "confidence": "high"})
            for index, delta in enumerate(deltas, start=2):
                bbox = tuple(delta.bbox_after)
                clusters.append(
                    {
                        "cluster_id": index,
                        "bbox": bbox,
                        "area_bucket": "one",
                        "color_histogram_delta": ((0, 7, "one"),),
                        "distance_to_actor": "far",
                        "overlaps_known_component": True,
                        "overlaps_hud": False,
                    }
                )
                assignments["independent_motion_delta"].append({"cluster_id": index, "confidence": "high"})
            event = type(
                "FakeEvent",
                (),
                {
                    "action": 4,
                    "object_deltas": tuple(deltas),
                    "created_object_ids": (),
                    "disappeared_object_ids": tuple(disappeared),
                    "reappeared_object_ids": tuple(reappeared),
                    "motion_groups": tuple(groups),
                },
            )()
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ar25",
                    "event": event,
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "coarse_outcome": "independent_motion_delta",
                            "parameter_outcome": {"independent_motion_delta": {"cluster_count": len(deltas)}},
                            "controlled_entity": {
                                "type": "single",
                                "confidence": "high",
                                "candidates": [controlled],
                            },
                            "changed_region_clusters": tuple(clusters),
                            "changed_region_assignments": assignments,
                            "channels": {
                                "independent_motion_delta": {"cluster_count": len(deltas)},
                                "controlled_entity_delta": {"cluster_count": int(overlap)},
                            },
                            "active_channels": ("independent_motion_delta",),
                        },
                    },
                }
            )

        group = type("FakeMotionGroup", (), {"translation": (0, 1), "object_ids": (2, 3)})()
        records = [
            probe_record(1, [fake_delta(2, (5, 1, 5, 1), (5, 2, 5, 2), (0, 1)), fake_delta(3, (6, 4, 6, 4), (6, 3, 6, 3), (0, -1))]),
            probe_record(2, [fake_delta(2, (5, 2, 5, 2), (5, 4, 5, 4), (0, 2)), fake_delta(3, (5, 5, 5, 5), (5, 3, 5, 3), (0, -2))]),
            probe_record(3, [fake_delta(2, (5, 4, 5, 4), (5, 5, 5, 5), (0, 1))], reappeared=(2,)),
            probe_record(4, [fake_delta(2, (5, 5, 5, 5), (5, 7, 5, 7), (0, 2)), fake_delta(3, (6, 3, 6, 3), (6, 5, 6, 5), (0, 2))], groups=(group,)),
            probe_record(5, [fake_delta(2, (5, 7, 5, 7), (3, 4, 3, 4), (-2, -3))], overlap=True),
        ]

        report = world_clock_transition_probe_report(records)
        distribution = report["failure_class_distribution"]

        self.assertEqual(report["status"], "evaluated")
        self.assertGreater(distribution.get("identity_crossing_failure", 0), 0)
        self.assertGreater(distribution.get("occlusion_composition_failure", 0), 0)
        self.assertGreater(distribution.get("coupled_group_role_failure", 0), 0)

    def test_world_clock_transition_probe_skips_cd82(self) -> None:
        record = TransitionRecord(**{**synthetic_record().__dict__, "game": "cd82"})
        report = world_clock_transition_probe_report([record])
        self.assertEqual(report["status"], "skipped_not_world_clock_transition_game")

    def test_visibility_composition_probe_splits_overlap_from_visible_translation(self) -> None:
        def fake_delta(step: int, before, after, translation):
            return type(
                "FakeDelta",
                (),
                {
                    "object_id": 2,
                    "moved": True,
                    "translation": translation,
                    "value_before": 7,
                    "value_after": 7,
                    "shape_changed": False,
                    "area_before": 1,
                    "area_after": 1,
                    "bbox_before": before,
                    "bbox_after": after,
                    "shape_transform": "same",
                },
            )()

        def composition_record(step: int, delta, *, controlled_after):
            before_grid = [[0 for _ in range(20)] for _ in range(20)]
            after_grid = [[0 for _ in range(20)] for _ in range(20)]
            before_grid[delta.bbox_before[0]][delta.bbox_before[1]] = 7
            after_grid[delta.bbox_after[0]][delta.bbox_after[1]] = 7
            before = obs(before_grid)
            after = obs(after_grid)
            base = synthetic_record(action=4, step=step)
            event = type(
                "FakeEvent",
                (),
                {
                    "action": 4,
                    "object_deltas": (delta,),
                    "created_object_ids": (),
                    "disappeared_object_ids": (),
                    "reappeared_object_ids": (),
                    "motion_groups": (),
                },
            )()
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ls20",
                    "before": before,
                    "after": after,
                    "before_scene": ObservationParser().parse(before, frame_index=step),
                    "event": event,
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "controlled_entity": {
                                "type": "single",
                                "confidence": "high",
                                "candidates": [
                                    {
                                        "object_id": 1,
                                        "score": 1.0,
                                        "translation": [0, 1],
                                        "bbox_before": (9, 9, 9, 9),
                                        "bbox_after": controlled_after,
                                    }
                                ],
                            },
                            "changed_region_clusters": (
                                {
                                    "cluster_id": 1,
                                    "bbox": delta.bbox_after,
                                    "area_bucket": "one",
                                    "color_histogram_delta": ((0, 7, "one"),),
                                    "distance_to_actor": "far",
                                    "overlaps_known_component": True,
                                    "overlaps_hud": False,
                                },
                            ),
                            "changed_region_assignments": {
                                "independent_motion_delta": [{"cluster_id": 1, "confidence": "high"}],
                            },
                            "channels": {"independent_motion_delta": {"cluster_count": 1}},
                            "active_channels": ("independent_motion_delta",),
                        },
                    },
                }
            )

        records = [
            composition_record(1, fake_delta(1, (10, 10, 10, 10), (10, 11, 10, 11), (0, 1)), controlled_after=(9, 10, 9, 10)),
            composition_record(2, fake_delta(2, (10, 11, 10, 11), (10, 12, 10, 12), (0, 1)), controlled_after=(10, 12, 10, 12)),
        ]

        report = visibility_composition_probe_report(records)

        self.assertEqual(report["status"], "evaluated")
        self.assertGreater(
            report["visible_delta_class_distribution"].get("translation_with_controlled_overlap", 0),
            0,
        )

    def test_visibility_precision_audit_accounts_for_trail_labels(self) -> None:
        def fake_delta(before, after, translation):
            return type(
                "FakeDelta",
                (),
                {
                    "object_id": 2,
                    "moved": True,
                    "translation": translation,
                    "value_before": 7,
                    "value_after": 7,
                    "shape_changed": False,
                    "area_before": 1,
                    "area_after": 1,
                    "bbox_before": before,
                    "bbox_after": after,
                    "shape_transform": "same",
                },
            )()

        def trail_record(game: str, step: int, delta):
            before_grid = [[0 for _ in range(20)] for _ in range(20)]
            after_grid = [[0 for _ in range(20)] for _ in range(20)]
            before_grid[delta.bbox_before[0]][delta.bbox_before[1]] = 7
            after_grid[delta.bbox_after[0]][delta.bbox_after[1]] = 7
            before = obs(before_grid)
            after = obs(after_grid)
            base = synthetic_record(action=4, step=step)
            event = type(
                "FakeEvent",
                (),
                {
                    "action": 4,
                    "object_deltas": (delta,),
                    "created_object_ids": (),
                    "disappeared_object_ids": (),
                    "reappeared_object_ids": (),
                    "motion_groups": (),
                },
            )()
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": game,
                    "before": before,
                    "after": after,
                    "before_scene": ObservationParser().parse(before, frame_index=step),
                    "event": event,
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "controlled_entity": {"type": "none", "confidence": "low", "candidates": []},
                            "changed_region_clusters": (
                                {
                                    "cluster_id": 1,
                                    "bbox": (delta.bbox_before[0], delta.bbox_before[1], delta.bbox_after[2], delta.bbox_after[3]),
                                    "area_bucket": "few",
                                    "color_histogram_delta": ((0, 7, "one"), (7, 0, "one")),
                                    "distance_to_actor": "far",
                                    "overlaps_known_component": True,
                                    "overlaps_hud": False,
                                },
                            ),
                            "changed_region_assignments": {
                                "independent_motion_delta": [{"cluster_id": 1, "confidence": "high"}],
                            },
                            "channels": {"independent_motion_delta": {"cluster_count": 1}},
                            "active_channels": ("independent_motion_delta",),
                        },
                    },
                }
            )

        records_by_game = {
            "ls20": [
                trail_record("ls20", 1, fake_delta((10, 10, 10, 10), (10, 11, 10, 11), (0, 1))),
                trail_record("ls20", 2, fake_delta((10, 11, 10, 11), (10, 12, 10, 12), (0, 1))),
            ],
            "ar25": [
                trail_record("ar25", 1, fake_delta((11, 10, 11, 10), (11, 11, 11, 11), (0, 1))),
                trail_record("ar25", 2, fake_delta((11, 11, 11, 11), (11, 12, 11, 12), (0, 1))),
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            report = visibility_composition_precision_audit_suite(records_by_game, output_dir=Path(tmp))
            composition = visibility_composition_probe_report(records_by_game["ls20"])
            split = composition["split_composition_label_accounting"]
            bootstrap = mover_bootstrap_report(records_by_game["ls20"])

            self.assertGreater(report["by_game"]["ls20"]["audited_case_count"], 0)
            self.assertIn("summary_label_accounting", report)
            self.assertTrue(report["summary_label_accounting"])
            self.assertTrue(report["by_game"]["ls20"]["cases"][0]["prototype_path"])
            self.assertGreater(composition["bootstrap_count"], 0)
            self.assertIn("mover_bootstrap_no_history", composition["visible_delta_class_distribution"])
            self.assertGreater(bootstrap["bootstrap_count"], 0)
            self.assertEqual(
                split["clean_old_new_exposure"] + split["merged_old_new_overlap_or_trail"],
                split["legacy_translation_with_trail_or_residue"],
            )
            self.assertTrue(split["accounting_matches_legacy"])

    def test_ar25_history_filtered_latent_readout_separates_cold_start(self) -> None:
        def fake_delta(object_id, before, after, translation):
            return type(
                "FakeDelta",
                (),
                {
                    "object_id": object_id,
                    "moved": True,
                    "translation": translation,
                    "value_before": 7,
                    "value_after": 7,
                    "shape_changed": False,
                    "area_before": 1,
                    "area_after": 1,
                    "bbox_before": before,
                    "bbox_after": after,
                    "shape_transform": "same",
                },
            )()

        def latent_record(step: int, delta):
            before_grid = [[0 for _ in range(24)] for _ in range(24)]
            after_grid = [[0 for _ in range(24)] for _ in range(24)]
            before_grid[delta.bbox_before[0]][delta.bbox_before[1]] = 7
            after_grid[delta.bbox_after[0]][delta.bbox_after[1]] = 7
            before = obs(before_grid)
            after = obs(after_grid)
            base = synthetic_record(action=4, step=step)
            event = type(
                "FakeEvent",
                (),
                {
                    "action": 4,
                    "object_deltas": (delta,),
                    "created_object_ids": (),
                    "disappeared_object_ids": (),
                    "reappeared_object_ids": (),
                    "motion_groups": (),
                },
            )()
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ar25",
                    "before": before,
                    "after": after,
                    "before_scene": ObservationParser().parse(before, frame_index=step),
                    "event": event,
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "controlled_entity": {"type": "none", "confidence": "low", "candidates": []},
                            "changed_region_clusters": (
                                {
                                    "cluster_id": 1,
                                    "bbox": (20, 20, 20, 20),
                                    "area_bucket": "one",
                                    "color_histogram_delta": ((3, 7, "one"),),
                                    "distance_to_actor": "far",
                                    "overlaps_known_component": False,
                                    "overlaps_hud": False,
                                },
                            ),
                            "changed_region_assignments": {
                                "independent_motion_delta": [{"cluster_id": 1, "confidence": "high"}],
                            },
                            "channels": {"independent_motion_delta": {"cluster_count": 1}},
                            "active_channels": ("independent_motion_delta",),
                        },
                    },
                }
            )

        records = [
            latent_record(1, fake_delta(2, (10, 10, 10, 10), (10, 11, 10, 11), (0, 1))),
            latent_record(2, fake_delta(2, (10, 11, 10, 11), (10, 12, 10, 12), (0, 1))),
            latent_record(3, fake_delta(2, (10, 12, 10, 12), (10, 13, 10, 13), (0, 1))),
            latent_record(4, fake_delta(2, (10, 13, 10, 13), (10, 14, 10, 14), (0, 1))),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            report = ar25_latent_motion_history_filtered_taxonomy_report(
                records,
                prototype_dir=Path(tmp),
            )
            composition = visibility_composition_probe_report(records)
            bootstrap = mover_bootstrap_report(records)

            self.assertEqual(report["cold_start_or_low_history_count"], 2)
            self.assertEqual(report["mover_bootstrap_no_history_count"], 1)
            self.assertEqual(report["true_latent_motion_unknown_count"], 3)
            self.assertEqual(composition["visible_delta_class_distribution"]["mover_bootstrap_no_history"], 1)
            self.assertEqual(composition["visible_delta_class_distribution"]["true_latent_motion_unknown"], 3)
            self.assertGreater(bootstrap["bootstrap_count"], 0)
            history_ge_2 = report["passes"]["history_ge_2"]
            history_ge_3 = report["passes"]["history_ge_3"]
            self.assertGreater(history_ge_2["eligible_count"], 0)
            self.assertGreater(history_ge_3["eligible_count"], 0)
            self.assertTrue(all(case["history_length"] >= 2 for case in history_ge_2["cases"]))
            self.assertTrue(all(case["history_length"] >= 3 for case in history_ge_3["cases"]))
            self.assertTrue(Path(history_ge_2["cases"][0]["prototype_path"]).exists())

    def test_ar25_overlap_composition_report_splits_merged_overlap(self) -> None:
        def fake_delta(object_id, before, after, translation):
            return type(
                "FakeDelta",
                (),
                {
                    "object_id": object_id,
                    "moved": True,
                    "translation": translation,
                    "value_before": 7,
                    "value_after": 7,
                    "shape_changed": False,
                    "area_before": 9,
                    "area_after": 9,
                    "bbox_before": before,
                    "bbox_after": after,
                    "shape_transform": "same",
                },
            )()

        def overlap_record(step: int, delta):
            before_grid = [[0 for _ in range(24)] for _ in range(24)]
            after_grid = [[0 for _ in range(24)] for _ in range(24)]
            before_grid[delta.bbox_before[0]][delta.bbox_before[1]] = 7
            after_grid[delta.bbox_after[0]][delta.bbox_after[1]] = 7
            before = obs(before_grid)
            after = obs(after_grid)
            base = synthetic_record(action=4, step=step)
            event = type(
                "FakeEvent",
                (),
                {
                    "action": 4,
                    "object_deltas": (delta,),
                    "created_object_ids": (),
                    "disappeared_object_ids": (),
                    "reappeared_object_ids": (),
                    "motion_groups": (),
                },
            )()
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ar25",
                    "before": before,
                    "after": after,
                    "before_scene": ObservationParser().parse(before, frame_index=step),
                    "event": event,
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "controlled_entity": {"type": "none", "confidence": "low", "candidates": []},
                            "changed_region_clusters": (
                                {
                                    "cluster_id": 1,
                                    "bbox": (10, 10, 10, 11),
                                    "area_bucket": "few",
                                    "color_histogram_delta": ((3, 7, "one"),),
                                    "distance_to_actor": "far",
                                    "overlaps_known_component": True,
                                    "overlaps_hud": False,
                                },
                            ),
                            "changed_region_assignments": {
                                "independent_motion_delta": [{"cluster_id": 1, "confidence": "high"}],
                            },
                            "channels": {"independent_motion_delta": {"cluster_count": 1}},
                            "active_channels": ("independent_motion_delta",),
                        },
                    },
                }
            )

        records = [
            overlap_record(1, fake_delta(2, (10, 10, 10, 10), (10, 11, 10, 11), (0, 1))),
            overlap_record(2, fake_delta(2, (10, 10, 10, 10), (10, 11, 10, 11), (0, 1))),
        ]

        report = ar25_overlap_composition_report(records)

        self.assertEqual(report["status"], "evaluated")
        self.assertGreater(report["case_count"], 0)
        self.assertIn("simple_overlap_same_object", report["classification_distribution"])

    def test_mover_bootstrap_composition_v0_scores_hypotheses_and_decomposition(self) -> None:
        def fake_delta(object_id, before, after, translation):
            return type(
                "FakeDelta",
                (),
                {
                    "object_id": object_id,
                    "moved": True,
                    "translation": translation,
                    "value_before": 7,
                    "value_after": 7,
                    "shape_changed": False,
                    "area_before": 9,
                    "area_after": 9,
                    "bbox_before": before,
                    "bbox_after": after,
                    "shape_transform": "same",
                },
            )()

        def record(step: int, delta):
            before_grid = [[0 for _ in range(24)] for _ in range(24)]
            after_grid = [[0 for _ in range(24)] for _ in range(24)]
            before_grid[delta.bbox_before[0]][delta.bbox_before[1]] = 7
            after_grid[delta.bbox_after[0]][delta.bbox_after[1]] = 7
            before = obs(before_grid)
            after = obs(after_grid)
            base = synthetic_record(action=4, step=step)
            event = type(
                "FakeEvent",
                (),
                {
                    "action": 4,
                    "object_deltas": (delta,),
                    "created_object_ids": (),
                    "disappeared_object_ids": (),
                    "reappeared_object_ids": (),
                    "motion_groups": (),
                },
            )()
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ar25",
                    "before": before,
                    "after": after,
                    "before_scene": ObservationParser().parse(before, frame_index=step),
                    "event": event,
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "controlled_entity": {"type": "none", "confidence": "low", "candidates": []},
                            "changed_region_clusters": (
                                {
                                    "cluster_id": 1,
                                    "bbox": (10, 10, 10, 11),
                                    "area_bucket": "few",
                                    "color_histogram_delta": ((3, 7, "one"),),
                                    "distance_to_actor": "far",
                                    "overlaps_known_component": True,
                                    "overlaps_hud": False,
                                },
                            ),
                            "changed_region_assignments": {
                                "independent_motion_delta": [{"cluster_id": 1, "confidence": "high"}],
                            },
                            "channels": {"independent_motion_delta": {"cluster_count": 1}},
                            "active_channels": ("independent_motion_delta",),
                        },
                    },
                }
            )

        records = [
            record(1, fake_delta(2, (10, 10, 10, 10), (10, 11, 10, 11), (0, 1))),
            record(2, fake_delta(2, (10, 10, 10, 10), (10, 11, 10, 11), (0, 1))),
        ]

        report = mover_bootstrap_composition_v0_report(records)

        self.assertEqual(report["status"], "evaluated")
        self.assertGreater(report["bootstrap_metrics"]["bootstrap_case_count"], 0)
        self.assertEqual(report["bootstrap_metrics"]["top3_previous_pose_coverage"], 1.0)
        self.assertGreater(report["merged_overlap_metrics"]["merged_overlap_case_count"], 0)
        self.assertEqual(report["merged_overlap_metrics"]["old_new_footprint_recovery_rate"], 1.0)
        self.assertTrue(report["bootstrap_hypotheses"])
        self.assertTrue(report["merged_overlap_decomposition_examples"])

    def test_changed_cluster_merge_split_audit_v0_splits_evidence_container(self) -> None:
        def fake_delta(object_id, before, after, translation):
            return type(
                "FakeDelta",
                (),
                {
                    "object_id": object_id,
                    "moved": True,
                    "translation": translation,
                    "value_before": 7,
                    "value_after": 7,
                    "shape_changed": False,
                    "area_before": 4,
                    "area_after": 4,
                    "bbox_before": before,
                    "bbox_after": after,
                    "shape_transform": "same",
                },
            )()

        def record(step: int, delta):
            before_grid = [[0 for _ in range(24)] for _ in range(24)]
            after_grid = [[0 for _ in range(24)] for _ in range(24)]
            before_grid[delta.bbox_before[0]][delta.bbox_before[1]] = 7
            before_grid[delta.bbox_before[0]][delta.bbox_before[1] + 1] = 7
            after_grid[delta.bbox_after[0]][delta.bbox_after[1]] = 7
            after_grid[delta.bbox_after[0]][delta.bbox_after[1] + 1] = 7
            before = obs(before_grid)
            after = obs(after_grid)
            base = synthetic_record(action=4, step=step)
            event = type(
                "FakeEvent",
                (),
                {
                    "action": 4,
                    "object_deltas": (delta,),
                    "created_object_ids": (),
                    "disappeared_object_ids": (),
                    "reappeared_object_ids": (),
                    "motion_groups": (),
                },
            )()
            clusters = tuple(
                {
                    "cluster_id": index + 1,
                    "bbox": bbox,
                    "area_bucket": "few",
                    "color_histogram_delta": ((0, 7, "one"),),
                    "distance_to_actor": "far",
                    "overlaps_known_component": True,
                    "overlaps_hud": False,
                }
                for index, bbox in enumerate(
                    (
                        (10, 10, 10, 12),
                        (1, 1, 1, 1),
                        (2, 2, 2, 2),
                        (3, 3, 3, 3),
                        (4, 4, 4, 4),
                        (5, 5, 5, 5),
                    )
                )
            )
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ar25",
                    "before": before,
                    "after": after,
                    "before_scene": ObservationParser().parse(before, frame_index=step),
                    "event": event,
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "controlled_entity": {"type": "none", "confidence": "low", "candidates": []},
                            "changed_region_clusters": clusters,
                            "changed_region_assignments": {
                                "independent_motion_delta": [{"cluster_id": 1, "confidence": "high"}],
                            },
                            "channels": {"independent_motion_delta": {"cluster_count": 1}},
                            "active_channels": ("independent_motion_delta",),
                        },
                    },
                }
            )

        records = [
            record(1, fake_delta(2, (10, 10, 10, 11), (10, 11, 10, 12), (0, 1))),
            record(2, fake_delta(2, (10, 10, 10, 11), (10, 11, 10, 12), (0, 1))),
        ]

        report = changed_cluster_merge_split_audit_v0_report(records)

        self.assertEqual(report["status"], "evaluated")
        self.assertGreater(report["case_count"], 0)
        self.assertGreater(report["metrics"]["causal_split_recovery_rate"], 0.0)
        self.assertEqual(report["metrics"]["old_new_footprint_recovery_rate"], 1.0)
        self.assertTrue(report["cluster_decomposition_hypotheses"])
        self.assertTrue(report["cell_attribution_examples"])
        self.assertIn(
            "cluster_merge_artifact",
            report["ar25_cluster_overmerge_taxonomy"]["overmerge_taxonomy_distribution"],
        )

    def test_residue_source_attribution_v0_buckets_high_residue_cases(self) -> None:
        def fake_delta(object_id, before, after, translation):
            return type(
                "FakeDelta",
                (),
                {
                    "object_id": object_id,
                    "moved": True,
                    "translation": translation,
                    "value_before": 7,
                    "value_after": 7,
                    "shape_changed": False,
                    "area_before": 2,
                    "area_after": 2,
                    "bbox_before": before,
                    "bbox_after": after,
                    "shape_transform": "same",
                },
            )()

        def record(step: int, delta):
            before_grid = [[0 for _ in range(24)] for _ in range(24)]
            after_grid = [[0 for _ in range(24)] for _ in range(24)]
            before_grid[delta.bbox_before[0]][delta.bbox_before[1]] = 7
            after_grid[delta.bbox_after[0]][delta.bbox_after[1]] = 7
            for col in range(2, 18):
                before_grid[1][col] = 3
                after_grid[1][col] = 4
            before = obs(before_grid)
            after = obs(after_grid)
            base = synthetic_record(action=4, step=step)
            event = type(
                "FakeEvent",
                (),
                {
                    "action": 4,
                    "object_deltas": (delta,),
                    "created_object_ids": (),
                    "disappeared_object_ids": (),
                    "reappeared_object_ids": (),
                    "motion_groups": (),
                },
            )()
            clusters = (
                {
                    "cluster_id": 1,
                    "bbox": (10, 10, 10, 11),
                    "area_bucket": "few",
                    "color_histogram_delta": ((0, 7, "one"),),
                    "distance_to_actor": "far",
                    "overlaps_known_component": True,
                    "overlaps_hud": False,
                },
                {
                    "cluster_id": 2,
                    "bbox": (1, 2, 1, 17),
                    "area_bucket": "some",
                    "color_histogram_delta": ((3, 4, "some"),),
                    "distance_to_actor": "far",
                    "overlaps_known_component": False,
                    "overlaps_hud": True,
                },
                {"cluster_id": 3, "bbox": (2, 2, 2, 2), "distance_to_actor": "far", "overlaps_hud": False},
                {"cluster_id": 4, "bbox": (3, 3, 3, 3), "distance_to_actor": "far", "overlaps_hud": False},
                {"cluster_id": 5, "bbox": (4, 4, 4, 4), "distance_to_actor": "far", "overlaps_hud": False},
                {"cluster_id": 6, "bbox": (5, 5, 5, 5), "distance_to_actor": "far", "overlaps_hud": False},
            )
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ar25",
                    "before": before,
                    "after": after,
                    "before_scene": ObservationParser().parse(before, frame_index=step),
                    "event": event,
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "controlled_entity": {"type": "none", "confidence": "low", "candidates": []},
                            "changed_region_clusters": clusters,
                            "changed_region_assignments": {
                                "independent_motion_delta": [{"cluster_id": 1, "confidence": "high"}],
                                "hud_status_delta": [{"cluster_id": 2, "confidence": "medium"}],
                            },
                            "channels": {
                                "independent_motion_delta": {"cluster_count": 1},
                                "hud_status_delta": {"cluster_count": 1},
                            },
                            "active_channels": ("independent_motion_delta", "hud_status_delta"),
                        },
                    },
                }
            )

        records = [
            record(1, fake_delta(2, (10, 10, 10, 10), (10, 11, 10, 11), (0, 1))),
            record(2, fake_delta(2, (10, 10, 10, 10), (10, 11, 10, 11), (0, 1))),
        ]

        report = residue_source_attribution_v0_report(records)

        self.assertEqual(report["status"], "evaluated")
        self.assertGreater(report["case_count"], 0)
        self.assertGreaterEqual(report["metrics"]["high_residue_taxonomy_coverage"], 0.8)
        self.assertTrue(report["residue_atoms"])
        self.assertTrue(report["residue_competition_groups"])
        self.assertTrue(report["residue_hypothesis_scores"])
        self.assertTrue(report["high_residue_case_gallery"])
        self.assertIn(
            report["metrics"]["dominant_blocker"],
            report["ar25_residue_taxonomy"]["top_hypothesis_distribution"],
        )

    def test_global_update_vs_container_audit_resolves_recurring_residue_tie(self) -> None:
        residue_cells = tuple((1, col) for col in range(2, 10))

        def record(step: int, action: int) -> TransitionRecord:
            base = synthetic_record(action=action, step=step)
            return TransitionRecord(**{**base.__dict__, "game": "ar25"})

        def hypothesis(label: str) -> ResidueSourceHypothesis:
            return ResidueSourceHypothesis(
                hypothesis_id=f"hyp:{label}",
                label=label,
                score=1.0,
                residue_cell_coverage=1.0,
                supporting_atom_ids=("atom:spatial",),
                reasons=("synthetic near-tie rival",),
                suggested_next_probe="synthetic",
            )

        def group(step: int) -> ResidueCompetitionGroup:
            transition_id = f"ar25:step:{step}:mover:{step}"
            atom = ResidueAtom(
                atom_id=f"{transition_id}:atom:spatial",
                atomizer="spatial_bridge_cut",
                cells=residue_cells,
                bbox=(1, 2, 1, 9),
                color_transition=None,
                source_hint="synthetic recurring residue",
            )
            return ResidueCompetitionGroup(
                transition_id=transition_id,
                cluster_id=f"{transition_id}:cluster",
                cluster_bbox=(0, 0, 12, 12),
                residue_cell_count=len(residue_cells),
                atoms=(atom,),
                hypotheses=(
                    hypothesis("container_overinclusive"),
                    hypothesis("global_animation_or_clock"),
                    hypothesis("true_unknown_effect"),
                ),
                top_hypothesis="container_overinclusive",
            )

        records = [record(1, 1), record(2, 2), record(3, 3)]
        report = global_update_vs_container_audit_v0_report(
            records,
            competition_groups=[group(1), group(2), group(3)],
            merge_split_metrics={"old_new_footprint_recovery_rate": 1.0},
        )

        self.assertEqual(report["status"], "evaluated")
        self.assertEqual(report["metrics"]["dominant_verdict"], "global_update_driven_container")
        self.assertEqual(report["metrics"]["near_tie_resolved_rate"], 1.0)
        self.assertGreater(report["metrics"]["global_mask_subtraction_mean_gain"], 0.9)
        self.assertEqual(report["metrics"]["bridge_cut_mean_gain"], 0.0)
        self.assertTrue(report["global_residue_templates"])
        self.assertTrue(report["global_mask_subtraction_examples"])
        self.assertTrue(report["bridge_cut_vs_global_subtraction"])
        self.assertIn(
            "global_update_driven_container",
            report["ar25_residue_verdict_taxonomy"]["verdict_distribution"],
        )

    def test_global_update_layer_v0_predicts_online_without_false_subtraction(self) -> None:
        residue_cells = tuple((1, col) for col in range(2, 10))

        def record(step: int, action: int) -> TransitionRecord:
            before_grid = [[0 for _ in range(12)] for _ in range(12)]
            after_grid = [[0 for _ in range(12)] for _ in range(12)]
            before_grid[5][5] = 7
            after_grid[5][6] = 7
            for row, col in residue_cells:
                before_grid[row][col] = 3
                after_grid[row][col] = 4
            before = obs(before_grid)
            after = obs(after_grid)
            base = synthetic_record(action=action, step=step)
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ar25",
                    "before": before,
                    "after": after,
                    "before_scene": ObservationParser().parse(before, frame_index=step),
                }
            )

        def decomposition(step: int) -> ClusterDecomposition:
            transition_id = f"ar25:step:{step}:mover:{step}"
            attributions = (
                CellAttribution((5, 5), 7, 0, ("old_footprint",), "H", 0.9),
                CellAttribution((5, 6), 0, 7, ("new_footprint",), "H", 0.9),
                *(
                    CellAttribution(cell, 3, 4, ("residue",), "H", 0.3)
                    for cell in residue_cells
                ),
            )
            hypothesis = DecompositionHypothesis(
                hypothesis_id="H",
                splitter="Synthetic",
                cell_attributions=attributions,
                score=1.0,
                coverage=0.2,
                residue_rate=0.8,
                ambiguity=0.0,
                complexity=1,
                reasons=("synthetic old/new plus recurring residue",),
            )
            return ClusterDecomposition(
                transition_id=transition_id,
                cluster_id=f"{transition_id}:cluster",
                bbox=(1, 2, 5, 6),
                hypotheses=(hypothesis,),
                best_hypothesis_id="H",
                residue_rate=0.8,
                ambiguity=0.0,
                overmerge_taxonomy_label="cluster_merge_artifact",
            )

        def group(step: int) -> ResidueCompetitionGroup:
            transition_id = f"ar25:step:{step}:mover:{step}"
            atom = ResidueAtom(
                atom_id=f"{transition_id}:atom:spatial",
                atomizer="spatial_bridge_cut",
                cells=residue_cells,
                bbox=(1, 2, 1, 9),
                color_transition=None,
                source_hint="synthetic recurring residue",
            )
            hypothesis = ResidueSourceHypothesis(
                hypothesis_id="hyp:global",
                label="global_animation_or_clock",
                score=1.0,
                residue_cell_coverage=1.0,
                supporting_atom_ids=(atom.atom_id,),
                reasons=("synthetic recurring residue",),
                suggested_next_probe="synthetic",
            )
            return ResidueCompetitionGroup(
                transition_id=transition_id,
                cluster_id=f"{transition_id}:cluster",
                cluster_bbox=(1, 2, 5, 6),
                residue_cell_count=len(residue_cells),
                atoms=(atom,),
                hypotheses=(hypothesis,),
                top_hypothesis="global_animation_or_clock",
            )

        records = [record(1, 1), record(2, 2), record(3, 3)]
        report = global_update_layer_v0_report(
            records,
            baseline_decompositions=[decomposition(1), decomposition(2), decomposition(3)],
            baseline_groups=[group(1), group(2), group(3)],
        )

        self.assertEqual(report["status"], "evaluated")
        self.assertTrue(report["online_global_update_predictions"])
        self.assertTrue(all(not row["used_future_evidence"] for row in report["online_global_update_predictions"]))
        self.assertGreater(report["metrics"]["online_global_update_prediction_match"], 0.0)
        self.assertEqual(report["metrics"]["online_false_subtraction_rate"], 0.0)
        self.assertTrue(report["delta_factorization_examples"])
        self.assertTrue(report["local_cluster_rebuild_before_after"])
        self.assertTrue(report["false_subtraction_audit"])

    def test_global_phase_register_discovery_v0_scores_prequential_hypotheses(self) -> None:
        residue_cells = tuple((1, col) for col in range(2, 10))

        def record(step: int, action: int) -> TransitionRecord:
            before_grid = [[0 for _ in range(12)] for _ in range(12)]
            after_grid = [[0 for _ in range(12)] for _ in range(12)]
            before_grid[5][5] = 7
            after_grid[5][6] = 7
            for row, col in residue_cells:
                before_grid[row][col] = 3
                after_grid[row][col] = 4
            before = obs(before_grid)
            after = obs(after_grid)
            base = synthetic_record(action=action, step=step)
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ar25",
                    "before": before,
                    "after": after,
                    "before_scene": ObservationParser().parse(before, frame_index=step),
                }
            )

        def decomposition(step: int) -> ClusterDecomposition:
            transition_id = f"ar25:step:{step}:mover:{step}"
            attributions = (
                CellAttribution((5, 5), 7, 0, ("old_footprint",), "H", 0.9),
                CellAttribution((5, 6), 0, 7, ("new_footprint",), "H", 0.9),
                *(
                    CellAttribution(cell, 3, 4, ("residue",), "H", 0.3)
                    for cell in residue_cells
                ),
            )
            hypothesis = DecompositionHypothesis(
                hypothesis_id="H",
                splitter="Synthetic",
                cell_attributions=attributions,
                score=1.0,
                coverage=0.2,
                residue_rate=0.8,
                ambiguity=0.0,
                complexity=1,
                reasons=("synthetic old/new plus recurring residue",),
            )
            return ClusterDecomposition(
                transition_id=transition_id,
                cluster_id=f"{transition_id}:cluster",
                bbox=(1, 2, 5, 6),
                hypotheses=(hypothesis,),
                best_hypothesis_id="H",
                residue_rate=0.8,
                ambiguity=0.0,
                overmerge_taxonomy_label="cluster_merge_artifact",
            )

        def group(step: int) -> ResidueCompetitionGroup:
            transition_id = f"ar25:step:{step}:mover:{step}"
            atom = ResidueAtom(
                atom_id=f"{transition_id}:atom:spatial",
                atomizer="spatial_bridge_cut",
                cells=residue_cells,
                bbox=(1, 2, 1, 9),
                color_transition=None,
                source_hint="synthetic recurring residue",
            )
            hypothesis = ResidueSourceHypothesis(
                hypothesis_id="hyp:global",
                label="global_animation_or_clock",
                score=1.0,
                residue_cell_coverage=1.0,
                supporting_atom_ids=(atom.atom_id,),
                reasons=("synthetic recurring residue",),
                suggested_next_probe="synthetic",
            )
            return ResidueCompetitionGroup(
                transition_id=transition_id,
                cluster_id=f"{transition_id}:cluster",
                cluster_bbox=(1, 2, 5, 6),
                residue_cell_count=len(residue_cells),
                atoms=(atom,),
                hypotheses=(hypothesis,),
                top_hypothesis="global_animation_or_clock",
            )

        records = [record(1, 1), record(2, 2), record(3, 3), record(4, 4)]
        decompositions = [decomposition(1), decomposition(2), decomposition(3), decomposition(4)]
        groups = [group(1), group(2), group(3), group(4)]
        layer_report = global_update_layer_v0_report(
            records,
            baseline_decompositions=decompositions,
            baseline_groups=groups,
        )

        report = global_phase_register_discovery_v0_report(
            records,
            baseline_decompositions=decompositions,
            baseline_groups=groups,
            global_layer_report=layer_report,
        )

        self.assertEqual(report["status"], "evaluated")
        self.assertTrue(report["phase_hypothesis_scores"])
        self.assertTrue(report["prequential_phase_predictions"])
        self.assertTrue(report["global_template_clusters"])
        self.assertTrue(report["alignment_match_comparison"])
        self.assertTrue(report["false_subtraction_locality_audit"])
        self.assertTrue(all(not row["used_future_evidence"] for row in report["prequential_phase_predictions"]))
        self.assertIn("best_prequential_exact_match", report["metrics"])
        self.assertIn("false_subtracted_mover_or_actor_rate", report["metrics"])

        identifiability = global_phase_identifiability_v0_report(
            records,
            baseline_decompositions=decompositions,
            baseline_groups=groups,
            phase_report=report,
            global_layer_report=layer_report,
        )

        self.assertEqual(identifiability["status"], "evaluated")
        self.assertTrue(identifiability["template_support_learning_curve"])
        self.assertTrue(identifiability["template_fragmentation_audit"])
        self.assertTrue(identifiability["current_frame_phase_readout_scores"])
        self.assertTrue(identifiability["higher_order_phase_predictions"])
        self.assertTrue(identifiability["compositional_global_atoms"])
        self.assertIn("oracle_template_transition_entropy", identifiability)
        self.assertIn("recommended_next_predictor_family", identifiability["phase_identifiability_verdict"])

    def test_ar25_unknown_world_clock_atlas_writes_prototypes(self) -> None:
        def fake_delta(object_id, before, after, translation):
            return type(
                "FakeDelta",
                (),
                {
                    "object_id": object_id,
                    "moved": True,
                    "translation": translation,
                    "value_before": 7,
                    "value_after": 7,
                    "shape_changed": False,
                    "area_before": 1,
                    "area_after": 1,
                    "bbox_before": before,
                    "bbox_after": after,
                    "shape_transform": "same",
                },
            )()

        def unknown_record(step: int, delta):
            before_grid = [[0 for _ in range(24)] for _ in range(24)]
            after_grid = [[0 for _ in range(24)] for _ in range(24)]
            before_grid[delta.bbox_before[0]][delta.bbox_before[1]] = 7
            after_grid[delta.bbox_after[0]][delta.bbox_after[1]] = 7
            before = obs(before_grid)
            after = obs(after_grid)
            base = synthetic_record(action=4, step=step)
            event = type(
                "FakeEvent",
                (),
                {
                    "action": 4,
                    "object_deltas": (delta,),
                    "created_object_ids": (),
                    "disappeared_object_ids": (),
                    "reappeared_object_ids": (),
                    "motion_groups": (),
                },
            )()
            return TransitionRecord(
                **{
                    **base.__dict__,
                    "game": "ar25",
                    "before": before,
                    "after": after,
                    "before_scene": ObservationParser().parse(before, frame_index=step),
                    "event": event,
                    "post_labels": {
                        **base.post_labels,
                        "labeler_v3": {
                            "controlled_entity": {"type": "none", "confidence": "low", "candidates": []},
                            "changed_region_clusters": (
                                {
                                    "cluster_id": 1,
                                    "bbox": delta.bbox_after,
                                    "area_bucket": "one",
                                    "color_histogram_delta": ((0, 7, "one"),),
                                    "distance_to_actor": "far",
                                    "overlaps_known_component": True,
                                    "overlaps_hud": False,
                                },
                            ),
                            "changed_region_assignments": {
                                "independent_motion_delta": [{"cluster_id": 1, "confidence": "high"}],
                            },
                            "channels": {"independent_motion_delta": {"cluster_count": 1}},
                            "active_channels": ("independent_motion_delta",),
                        },
                    },
                }
            )

        records = [
            unknown_record(1, fake_delta(2, (10, 10, 10, 10), (10, 11, 10, 11), (0, 1))),
            unknown_record(2, fake_delta(2, (10, 11, 10, 11), (11, 11, 11, 11), (1, 0))),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            report = ar25_unknown_world_clock_atlas_report(records, prototype_dir=Path(tmp))
            taxonomy = ar25_latent_transition_taxonomy_report(records, prototype_dir=Path(tmp) / "latent")

            self.assertEqual(report["status"], "evaluated")
            self.assertGreater(report["unknown_transition_count"], 0)
            self.assertTrue(Path(report["examples"][0]["prototype_path"]).exists())
            self.assertEqual(taxonomy["status"], "evaluated")
            self.assertIn("taxonomy_distribution", taxonomy)
            self.assertTrue(taxonomy["taxonomy_distribution"])

    def test_cd82_region_role_probe_reassigns_obvious_click_edit(self) -> None:
        record = synthetic_record(action=6, step=1)
        cluster = {
            "cluster_id": 1,
            "bbox": (18, 18, 18, 18),
            "area_bucket": "one",
            "color_histogram_delta": ((1, 2, "one"),),
            "distance_to_actor": "unknown",
            "distance_to_action_target": "zero",
            "overlaps_hud": True,
            "overlaps_known_component": True,
            "centroid_region": "hud",
        }
        record = TransitionRecord(
            **{
                **record.__dict__,
                "game": "cd82",
                "before": obs([[0 for _ in range(20)] for _ in range(20)]),
                "after": obs([[0 for _ in range(20)] for _ in range(20)]),
                "action_data": {"x": 18, "y": 18},
                "pre_features": {**record.pre_features, "action_kind": "click"},
                "post_labels": {
                    **record.post_labels,
                    "changed_cells": 1,
                    "labeler_v2": {"changed_region_clusters": (cluster,)},
                    "labeler_v3": {
                        "channels": {
                            "action_target_delta": {"cluster_count": 0},
                            "hud_status_delta": {"cluster_count": 1},
                        },
                        "changed_region_assignments": {
                            "action_target_delta": [],
                            "hud_status_delta": [{"cluster_id": 1, "confidence": "high", "reason": "region_role_hud_status"}],
                        },
                        "active_channels": ("hud_status_delta",),
                    },
                },
            }
        )

        report = cd82_region_role_probe_report([record])

        self.assertEqual(report["status"], "evaluated")
        self.assertGreater(
            report["probe"]["click_target_assignment_rate"],
            report["baseline_v3"]["click_target_assignment_rate"],
        )

    def test_cd82_region_role_integrated_labeler_reassigns_obvious_click_edit(self) -> None:
        record = synthetic_record(action=6, step=1)
        cluster = {
            "cluster_id": 1,
            "bbox": (18, 18, 18, 18),
            "area_bucket": "one",
            "color_histogram_delta": ((1, 2, "one"),),
            "distance_to_actor": "unknown",
            "distance_to_action_target": "zero",
            "overlaps_hud": True,
            "overlaps_known_component": True,
            "centroid_region": "hud",
        }
        record = TransitionRecord(
            **{
                **record.__dict__,
                "game": "cd82",
                "before": obs([[0 for _ in range(20)] for _ in range(20)]),
                "after": obs([[0 for _ in range(20)] for _ in range(20)]),
                "action_data": {"x": 18, "y": 18},
                "pre_features": {**record.pre_features, "action_kind": "click"},
                "post_labels": {
                    **record.post_labels,
                    "changed_cells": 1,
                    "labeler_v2": {"changed_region_clusters": (cluster,), "channels": {}},
                    "labeler_v3": {
                        "channels": {
                            "action_target_delta": {"cluster_count": 0},
                            "hud_status_delta": {"cluster_count": 1},
                        },
                        "changed_region_assignments": {
                            "action_target_delta": [],
                            "hud_status_delta": [{"cluster_id": 1, "confidence": "high", "reason": "region_role_hud_status"}],
                        },
                        "active_channels": ("hud_status_delta",),
                        "coarse_outcome": "hud_status_delta",
                        "parameter_outcome": {"hud_status_delta": {"cluster_count": 1}},
                    },
                },
            }
        )

        report = cd82_region_role_integrated_report([record])

        self.assertEqual(report["status"], "evaluated")
        self.assertGreater(
            report["integrated"]["click_target_assignment_rate"],
            report["baseline_v3"]["click_target_assignment_rate"],
        )

    def test_cd82_procedural_state_probe_skips_non_cd82(self) -> None:
        record = synthetic_record(action=6, step=1)

        report = cd82_procedural_state_probe_report([record])

        self.assertEqual(report["status"], "skipped_not_cd82")

    def test_labeler_v2_reports_are_emitted_by_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            output = Path(tmp) / "out"
            start = {
                "type": "observation",
                "step": 0,
                "pixels": [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                "available_actions": [6],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
                "full_reset": True,
            }
            transition = {
                "type": "transition",
                "step": 1,
                "action": 6,
                "action_data": {"x": 1, "y": 1},
                "observation": {
                    "pixels": [[[0, 0, 0], [0, 2, 0], [0, 0, 0]]],
                    "available_actions": [6],
                    "levels_completed": 0,
                    "state": "NOT_FINISHED",
                    "full_reset": False,
                },
            }
            trace.write_text(json.dumps(start) + "\n" + json.dumps(transition) + "\n")

            evaluate_suite((TraceSpec("toy", trace, "human"),), output_dir=output)

            self.assertTrue((output / "event_parameter_audit.json").exists())
            self.assertTrue((output / "actor_detection_audit.json").exists())
            self.assertTrue((output / "changed_region_assignment.json").exists())
            self.assertTrue((output / "outcome_labeler_v1_vs_v2.json").exists())
            self.assertTrue((output / "stage_report_v2.json").exists())
            self.assertTrue((output / "r3a_labeler_v2_ablation.json").exists())
            self.assertTrue((output / "controlled_entity_audit.json").exists())
            self.assertTrue((output / "region_role_map.json").exists())
            self.assertTrue((output / "delta_channel_assignment_v3.json").exists())
            self.assertTrue((output / "outcome_labeler_v2_vs_v3.json").exists())
            self.assertTrue((output / "stage_report_v3.json").exists())
            self.assertTrue((output / "outcome_labeler_v3_verdict.json").exists())
            self.assertTrue((output / "parameter_model_probe_report.json").exists())
            self.assertTrue((output / "cd82_region_role_probe_report.json").exists())
            self.assertTrue((output / "motion_law_probe_report.json").exists())
            self.assertTrue((output / "cd82_region_role_integrated_report.json").exists())
            self.assertTrue((output / "motion_target_granularity_report.json").exists())
            self.assertTrue((output / "independent_mover_dynamics_report.json").exists())
            self.assertTrue((output / "cd82_procedural_state_probe_report.json").exists())
            self.assertTrue((output / "controlled_region_material_decomposition_report.json").exists())
            self.assertTrue((output / "changed_region_factorization_report.json").exists())
            self.assertTrue((output / "t5_failure_anatomy_report.json").exists())
            self.assertTrue((output / "independent_coupled_mover_probe_report.json").exists())
            self.assertTrue((output / "mover_state_factorization_report.json").exists())
            self.assertTrue((output / "world_clock_transition_probe_report.json").exists())
            self.assertTrue((output / "visibility_composition_probe_report.json").exists())
            self.assertTrue((output / "mover_bootstrap_report.json").exists())
            self.assertTrue((output / "ar25_overlap_composition_report.json").exists())
            self.assertTrue((output / "mover_bootstrap_composition_v0_report.json").exists())
            self.assertTrue((output / "mover_bootstrap_hypotheses.jsonl").exists())
            self.assertTrue((output / "merged_overlap_decomposition_examples.jsonl").exists())
            self.assertTrue((output / "changed_cluster_merge_split_audit_v0_report.json").exists())
            self.assertTrue((output / "cluster_decomposition_hypotheses.jsonl").exists())
            self.assertTrue((output / "cell_attribution_examples.jsonl").exists())
            self.assertTrue((output / "decomposition_failure_gallery.jsonl").exists())
            self.assertTrue((output / "ar25_cluster_overmerge_taxonomy.json").exists())
            self.assertTrue((output / "decomposition_before_after_metrics.json").exists())
            self.assertTrue((output / "residue_source_attribution_v0_report.json").exists())
            self.assertTrue((output / "residue_atoms.jsonl").exists())
            self.assertTrue((output / "residue_competition_groups.jsonl").exists())
            self.assertTrue((output / "residue_hypothesis_scores.jsonl").exists())
            self.assertTrue((output / "high_residue_case_gallery.jsonl").exists())
            self.assertTrue((output / "ar25_residue_taxonomy.json").exists())
            self.assertTrue((output / "global_update_vs_container_audit_v0_report.json").exists())
            self.assertTrue((output / "global_residue_templates.jsonl").exists())
            self.assertTrue((output / "global_mask_subtraction_examples.jsonl").exists())
            self.assertTrue((output / "bridge_cut_vs_global_subtraction.jsonl").exists())
            self.assertTrue((output / "ar25_residue_verdict_taxonomy.json").exists())
            self.assertTrue((output / "global_update_layer_v0_report.json").exists())
            self.assertTrue((output / "online_global_update_predictions.jsonl").exists())
            self.assertTrue((output / "delta_factorization_examples.jsonl").exists())
            self.assertTrue((output / "local_cluster_rebuild_before_after.jsonl").exists())
            self.assertTrue((output / "ar25_composition_after_global_factoring.json").exists())
            self.assertTrue((output / "false_subtraction_audit.jsonl").exists())
            self.assertTrue((output / "global_phase_register_discovery_v0_report.json").exists())
            self.assertTrue((output / "phase_hypothesis_scores.jsonl").exists())
            self.assertTrue((output / "prequential_phase_predictions.jsonl").exists())
            self.assertTrue((output / "global_template_clusters.jsonl").exists())
            self.assertTrue((output / "alignment_match_comparison.jsonl").exists())
            self.assertTrue((output / "false_subtraction_locality_audit.jsonl").exists())
            self.assertTrue((output / "global_phase_identifiability_v0_report.json").exists())
            self.assertTrue((output / "template_support_learning_curve.jsonl").exists())
            self.assertTrue((output / "oracle_template_transition_entropy.json").exists())
            self.assertTrue((output / "template_fragmentation_audit.jsonl").exists())
            self.assertTrue((output / "current_frame_phase_readout_scores.jsonl").exists())
            self.assertTrue((output / "higher_order_phase_predictions.jsonl").exists())
            self.assertTrue((output / "compositional_global_atoms.jsonl").exists())
            self.assertTrue((output / "phase_identifiability_verdict.json").exists())
            self.assertTrue((output / "ar25_unknown_world_clock_atlas.json").exists())
            self.assertTrue((output / "visibility_composition_precision_audit.json").exists())
            self.assertTrue((output / "visibility_composition_precision_audit.md").exists())
            self.assertTrue((output / "ar25_latent_motion_unknown_readout.md").exists())
            self.assertTrue((output / "ar25_latent_transition_taxonomy_report.json").exists())
            self.assertTrue((output / "ar25_latent_motion_history_filtered_readout.md").exists())
            self.assertTrue((output / "ar25_latent_motion_history_filtered_taxonomy.json").exists())

    def test_r3a_schema_introspection_reports_are_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            output = Path(tmp) / "out"
            start = {
                "type": "observation",
                "step": 0,
                "pixels": [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                "available_actions": [1, 2],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
                "full_reset": True,
            }
            transition = {
                "type": "transition",
                "step": 1,
                "action": 2,
                "action_data": None,
                "observation": {
                    "pixels": [[[0, 0, 0], [0, 0, 1], [0, 0, 0]]],
                    "available_actions": [1, 2],
                    "levels_completed": 0,
                    "state": "NOT_FINISHED",
                    "full_reset": False,
                },
            }
            trace.write_text(json.dumps(start) + "\n" + json.dumps(transition) + "\n")

            summary = introspect_r3a_suite((TraceSpec("toy", trace, "human"),), output_dir=output)
            report_dir = output / "schema_introspection_report"

            self.assertEqual(summary["context_count"], 1)
            self.assertTrue((report_dir / "schema_contexts.jsonl").exists())
            self.assertTrue((report_dir / "schema_confusions.csv").exists())
            self.assertTrue((report_dir / "high_reuse_contexts.md").exists())
            self.assertTrue((report_dir / "high_entropy_contexts.md").exists())
            self.assertTrue((report_dir / "context_field_contributions.json").exists())
            self.assertTrue((report_dir / "field_contribution_examples.json").exists())
            self.assertTrue((report_dir / "vss_decomposition_report.json").exists())
            self.assertTrue((report_dir / "vss_robustness_report.json").exists())
            self.assertTrue((report_dir / "channel_parameter_heads_report.json").exists())
            self.assertTrue((report_dir / "parameter_entropy_context_contrast.json").exists())
            self.assertTrue((report_dir / "schema_atlas.md").exists())

    def test_labeler_v2_comparison_and_stage_report_run_on_synthetic_records(self) -> None:
        first = synthetic_record(step=1)
        second = synthetic_record(step=2)
        first = TransitionRecord(**{**first.__dict__, "post_labels": {**first.post_labels, "labeler_v2": {"coarse_outcome": "action_target_delta", "parameter_outcome": {"target": 1}, "active_channels": ("action_target_delta",), "channels": {"action_target_delta": {"cluster_count": 1, "bboxes": ((1, 1, 1, 1),)}}}}})
        second = TransitionRecord(**{**second.__dict__, "post_labels": {**second.post_labels, "labeler_v2": {"coarse_outcome": "action_target_delta", "parameter_outcome": {"target": 2}, "active_channels": ("action_target_delta",), "channels": {"action_target_delta": {"cluster_count": 1, "bboxes": ((2, 2, 2, 2),)}}}}})

        comparison = outcome_labeler_v1_vs_v2_report([first, second])
        stage = stage_report_v2([first, second])

        self.assertIn(R3A_COMPRESSED.name, comparison)
        self.assertEqual(stage["top_localizing_channel"], "action_target_delta")


if __name__ == "__main__":
    unittest.main()
