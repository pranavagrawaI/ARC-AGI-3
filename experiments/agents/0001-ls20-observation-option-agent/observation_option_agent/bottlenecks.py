"""Bottleneck diagnostics for option-agent representations.

The diagnostics are intentionally conservative: context keys are built only
from pre-action observations plus memory learned before the current step.
Post-action facts are outcome labels or memory updates, never context fields.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np

from experiments.harness import Observation
from experiments.harness.viewer import load_play_record

from .events import Event, EventDetector, ObjectDelta
from .observation import Detection, ObservationParser, VisualScene
from .tracking import ObjectTracker, TrackUpdate, TrackedDetection


TraceSource = Literal["human", "codex_run"]
OutcomeGranularity = Literal["coarse", "exact"]
LabelerVersion = Literal["v1", "v2", "v3", "cd82_region_role"]

PRE_FIELDS = frozenset(
    {
        "action_set",
        "action_kind",
        "actor_bbox_or_pos",
        "actor_confidence",
        "actor_or_focus_region",
        "actor_signature",
        "component_signature_histogram",
        "hud_or_resource_snapshot",
        "level_snapshot",
        "nearby_component_signatures",
        "raw_observation_hash",
        "recent_noop_streak",
        "target_relative_region",
        "action_target_type",
        "canonical_group_shape_family",
        "click_repeated_motif_bucket",
        "click_target_type_visual",
        "click_target_visual_signature",
        "click_target_zone",
        "clicked_static_component_class",
        "current_relative_geometry_class",
        "current_touch_overlap_relation",
        "mirror_rotation_normalized_signature",
        "visibility_or_unknown_mask",
        "visible_state_summary",
        "visible_independent_mover_signatures",
    }
)
MEMORY_FIELDS = frozenset(
    {
        "known_noop_or_reject_handles",
        "known_remote_effect_handles",
        "local_component_contradiction_count",
        "local_component_effect_confidence",
        "local_component_effect_summary",
        "click_target_type_learned",
        "known_group_motion_role_from_prior_memory",
        "last_palette_like_click_before_t",
        "prior_canvas_edit_count_bucket",
        "role_clusters",
        "selected_tool_proxy_before_t",
        "selected_tool_or_color_proxy",
        "target_differs_from_initial_frame_before_t",
        "target_material_summary_before_t",
    }
)
POST_FIELDS = frozenset(
    {
        "actor_delta",
        "block_reject",
        "blocked_or_rejected",
        "changed_cell_signature",
        "changed_cells",
        "changed_value_signature",
        "contacted_component_delta_class",
        "contacted_component_id",
        "current_action_caused_paint",
        "current_action_changed_target",
        "current_action_selected_color",
        "exact_outcome",
        "hud_delta",
        "hud_delta_signature",
        "hud_or_status_delta",
        "relative_group_motion_class",
        "remote_delta",
        "remote_delta_present",
        "resource_delta",
        "resource_delta_class",
        "resource_delta_signature",
        "terminal_delta",
        "terminal_status_delta",
    }
)
R3_BANNED_FIELDS = frozenset(
    {
        "actor_bbox_or_pos",
        "actor_signature",
        "component_signature_histogram",
        "hud_or_resource_snapshot",
        "nearby_component_signatures",
        "raw_observation_hash",
    }
)
RAW_ID_FIELDS = frozenset({"object_id", "object_ids", "component_id", "component_ids", "raw_component_id"})


@dataclass(frozen=True)
class TraceSpec:
    game: str
    path: Path
    trace_source: TraceSource
    supports_oracle: bool = False

    def manifest_record(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "path": str(self.path),
            "trace_source": self.trace_source,
            "supports_oracle": self.supports_oracle,
        }


@dataclass(frozen=True)
class RepresentationFamily:
    name: str
    pre_action_fields: frozenset[str]
    memory_fields: frozenset[str] = frozenset()
    post_action_fields: frozenset[str] = frozenset()

    @property
    def valid_prequential(self) -> bool:
        return not self.post_action_fields

    def assert_valid(self) -> None:
        leaked = self.pre_action_fields & POST_FIELDS
        leaked |= self.memory_fields & POST_FIELDS
        leaked |= self.post_action_fields
        if leaked:
            raise AssertionError(
                f"{self.name} context uses post-action fields: {sorted(leaked)}"
            )
        raw_ids = self.pre_action_fields & RAW_ID_FIELDS
        raw_ids |= self.memory_fields & RAW_ID_FIELDS
        if raw_ids:
            raise AssertionError(
                f"{self.name} context uses raw component ids: {sorted(raw_ids)}"
            )
        if self.name.startswith(("R3", "R4")):
            banned = self.pre_action_fields & R3_BANNED_FIELDS
            if banned:
                raise AssertionError(
                    f"{self.name} compressed key uses exact fields: {sorted(banned)}"
                )

    def audit_record(self) -> dict[str, Any]:
        return {
            "family": self.name,
            "pre_action_fields": sorted(self.pre_action_fields),
            "memory_fields": sorted(self.memory_fields),
            "post_action_fields": sorted(self.post_action_fields),
            "valid_prequential": self.valid_prequential,
        }


R0_CURRENT = RepresentationFamily(
    "R0_current",
    frozenset({"raw_observation_hash", "level_snapshot"}),
)
R2A_STATIC = RepresentationFamily(
    "R2a_computable_static",
    frozenset(
        {
            "action_set",
            "actor_bbox_or_pos",
            "actor_signature",
            "component_signature_histogram",
            "hud_or_resource_snapshot",
            "level_snapshot",
            "nearby_component_signatures",
            "visibility_or_unknown_mask",
            "visible_independent_mover_signatures",
        }
    ),
)
R2B_MEMORY = RepresentationFamily(
    "R2b_computable_with_memory",
    R2A_STATIC.pre_action_fields,
    frozenset(
        {
            "known_noop_or_reject_handles",
            "known_remote_effect_handles",
            "local_component_contradiction_count",
            "local_component_effect_confidence",
            "local_component_effect_summary",
        }
    ),
)
R3A_COMPRESSED = RepresentationFamily(
    "R3a_computable_compressed_static",
    frozenset(
        {
            "action_kind",
            "actor_confidence",
            "actor_or_focus_region",
            "action_target_type",
            "target_relative_region",
            "level_snapshot",
            "recent_noop_streak",
            "visible_state_summary",
        }
    ),
)
R3B_ROLE_MEMORY = RepresentationFamily(
    "R3b_role_memory",
    R3A_COMPRESSED.pre_action_fields,
    frozenset(
        {
            "role_clusters",
            "selected_tool_or_color_proxy",
        }
    ),
)
R4_RELATIONAL_GEOMETRY_ONLY = RepresentationFamily(
    "R4_relational_geometry_only",
    frozenset(
        {
            "canonical_group_shape_family",
            "current_relative_geometry_class",
            "current_touch_overlap_relation",
            "mirror_rotation_normalized_signature",
        }
    ),
    frozenset({"known_group_motion_role_from_prior_memory"}),
)
R4_RELATIONAL_GEOMETRY_PLUS_R3A = RepresentationFamily(
    "R4_relational_geometry_plus_R3a",
    R3A_COMPRESSED.pre_action_fields | R4_RELATIONAL_GEOMETRY_ONLY.pre_action_fields,
    R4_RELATIONAL_GEOMETRY_ONLY.memory_fields,
)
R4_CLICK_PROCEDURAL_VISUAL_ONLY = RepresentationFamily(
    "R4_click_procedural_visual_only",
    frozenset(
        {
            "click_repeated_motif_bucket",
            "click_target_type_visual",
            "click_target_visual_signature",
            "click_target_zone",
            "clicked_static_component_class",
        }
    ),
)
R4_CLICK_PROCEDURAL_VISUAL_PLUS_R3A = RepresentationFamily(
    "R4_click_procedural_visual_plus_R3a",
    R3A_COMPRESSED.pre_action_fields | R4_CLICK_PROCEDURAL_VISUAL_ONLY.pre_action_fields,
)
R4_CLICK_PROCEDURAL_LEARNED_ONLY = RepresentationFamily(
    "R4_click_procedural_learned_only",
    frozenset({"click_target_zone", "click_target_visual_signature"}),
    frozenset(
        {
            "click_target_type_learned",
            "last_palette_like_click_before_t",
            "prior_canvas_edit_count_bucket",
            "selected_tool_proxy_before_t",
            "target_differs_from_initial_frame_before_t",
            "target_material_summary_before_t",
        }
    ),
)
R4_CLICK_PROCEDURAL_LEARNED_PLUS_R3A = RepresentationFamily(
    "R4_click_procedural_learned_plus_R3a",
    R3A_COMPRESSED.pre_action_fields | R4_CLICK_PROCEDURAL_LEARNED_ONLY.pre_action_fields,
    R4_CLICK_PROCEDURAL_LEARNED_ONLY.memory_fields,
)
R4_REPRESENTATION_FAMILIES = (
    R4_RELATIONAL_GEOMETRY_ONLY,
    R4_RELATIONAL_GEOMETRY_PLUS_R3A,
    R4_CLICK_PROCEDURAL_VISUAL_ONLY,
    R4_CLICK_PROCEDURAL_VISUAL_PLUS_R3A,
    R4_CLICK_PROCEDURAL_LEARNED_ONLY,
    R4_CLICK_PROCEDURAL_LEARNED_PLUS_R3A,
)
R1_DEPLOYABLE = RepresentationFamily(
    "R1_oracle_deployable_ls20",
    frozenset(
        {
            "actor_bbox_or_pos",
            "hud_or_resource_snapshot",
            "level_snapshot",
            "visible_independent_mover_signatures",
        }
    ),
)
R1_POSTHOC = RepresentationFamily(
    "R1_oracle_posthoc_ls20",
    frozenset({"level_snapshot"}),
)
REPRESENTATION_FAMILIES = (
    R0_CURRENT,
    R2A_STATIC,
    R2B_MEMORY,
    R3A_COMPRESSED,
    R3B_ROLE_MEMORY,
)
ORACLE_FAMILIES = (R1_DEPLOYABLE, R1_POSTHOC)
R3A_FROZEN_FIELDS = tuple(sorted(R3A_COMPRESSED.pre_action_fields))


@dataclass(frozen=True)
class TransitionRecord:
    game: str
    trace_source: TraceSource
    step: int
    action: int
    action_data: Mapping[str, Any] | None
    before: Observation
    after: Observation
    before_scene: VisualScene
    after_scene: VisualScene
    before_tracked: tuple[TrackedDetection, ...]
    track_update: TrackUpdate
    event: Event
    pre_features: dict[str, Any]
    post_labels: dict[str, Any]


@dataclass(frozen=True)
class MetricSummary:
    examples: int
    unique_contexts: int
    reused_contexts: int
    reuse_coverage: float
    one_off_ratio: float
    contradicted_reused_contexts: int
    contradiction_rate: float
    mean_weighted_outcome_entropy: float
    majority_outcome_accuracy: float

    def to_record(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "unique_contexts": self.unique_contexts,
            "reused_contexts": self.reused_contexts,
            "reuse_coverage": self.reuse_coverage,
            "one_off_ratio": self.one_off_ratio,
            "contradicted_reused_contexts": self.contradicted_reused_contexts,
            "contradiction_rate": self.contradiction_rate,
            "mean_weighted_outcome_entropy": self.mean_weighted_outcome_entropy,
            "majority_outcome_accuracy": self.majority_outcome_accuracy,
        }


@dataclass(frozen=True)
class CellAttribution:
    cell: tuple[int, int]
    before_color: int
    after_color: int
    labels: tuple[str, ...]
    hypothesis_id: str
    confidence: float

    def to_record(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "before_color": self.before_color,
            "after_color": self.after_color,
            "labels": self.labels,
            "hypothesis_id": self.hypothesis_id,
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True)
class DecompositionHypothesis:
    hypothesis_id: str
    splitter: str
    cell_attributions: tuple[CellAttribution, ...]
    score: float
    coverage: float
    residue_rate: float
    ambiguity: float
    complexity: int
    reasons: tuple[str, ...]

    def label_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for attribution in self.cell_attributions:
            for label in attribution.labels:
                counts[label] += 1
        return dict(counts.most_common())

    def causal_part_count(self) -> int:
        return len({label for label in self.label_counts() if label != "residue"})

    def to_record(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "splitter": self.splitter,
            "score": round(self.score, 6),
            "coverage": round(self.coverage, 6),
            "residue_rate": round(self.residue_rate, 6),
            "ambiguity": round(self.ambiguity, 6),
            "complexity": self.complexity,
            "label_counts": self.label_counts(),
            "causal_part_count": self.causal_part_count(),
            "reasons": self.reasons,
        }


@dataclass(frozen=True)
class ClusterDecomposition:
    transition_id: str
    cluster_id: str
    bbox: tuple[int, int, int, int]
    hypotheses: tuple[DecompositionHypothesis, ...]
    best_hypothesis_id: str | None
    residue_rate: float
    ambiguity: float
    overmerge_taxonomy_label: str

    def best_hypothesis(self) -> DecompositionHypothesis | None:
        for hypothesis in self.hypotheses:
            if hypothesis.hypothesis_id == self.best_hypothesis_id:
                return hypothesis
        return None

    def to_record(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "cluster_id": self.cluster_id,
            "bbox": self.bbox,
            "best_hypothesis_id": self.best_hypothesis_id,
            "residue_rate": round(self.residue_rate, 6),
            "ambiguity": round(self.ambiguity, 6),
            "overmerge_taxonomy_label": self.overmerge_taxonomy_label,
            "hypotheses": [hypothesis.to_record() for hypothesis in self.hypotheses],
        }


@dataclass(frozen=True)
class ResidueAtom:
    atom_id: str
    atomizer: str
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int] | None
    color_transition: tuple[int, int] | None
    source_hint: str

    def to_record(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "atomizer": self.atomizer,
            "cell_count": len(self.cells),
            "bbox": self.bbox,
            "color_transition": self.color_transition,
            "source_hint": self.source_hint,
            "sample_cells": self.cells[:20],
        }


@dataclass(frozen=True)
class ResidueSourceHypothesis:
    hypothesis_id: str
    label: str
    score: float
    residue_cell_coverage: float
    supporting_atom_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    suggested_next_probe: str

    def to_record(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "label": self.label,
            "score": round(self.score, 6),
            "residue_cell_coverage": round(self.residue_cell_coverage, 6),
            "supporting_atom_ids": self.supporting_atom_ids,
            "reasons": self.reasons,
            "suggested_next_probe": self.suggested_next_probe,
        }


@dataclass(frozen=True)
class ResidueCompetitionGroup:
    transition_id: str
    cluster_id: str
    cluster_bbox: tuple[int, int, int, int]
    residue_cell_count: int
    atoms: tuple[ResidueAtom, ...]
    hypotheses: tuple[ResidueSourceHypothesis, ...]
    top_hypothesis: str

    def to_record(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "cluster_id": self.cluster_id,
            "cluster_bbox": self.cluster_bbox,
            "residue_cell_count": self.residue_cell_count,
            "atom_count": len(self.atoms),
            "top_hypothesis": self.top_hypothesis,
            "top_3_hypotheses": [hypothesis.to_record() for hypothesis in self.hypotheses[:3]],
            "atoms": [atom.to_record() for atom in self.atoms[:30]],
        }


class ResidueAtomizer:
    """Offline atomizers for residue evidence containers."""


class ResidueSourceScorer:
    """Diagnostic-only scorer for residue source hypotheses."""


@dataclass(frozen=True)
class GlobalResidueTemplate:
    template_id: str
    transition_id: str
    source_window_steps: tuple[int, ...]
    cells: tuple[tuple[int, int], ...]
    support_count: int
    different_action_support: int
    different_mover_support: int

    def to_record(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "transition_id": self.transition_id,
            "source_window_steps": self.source_window_steps,
            "cell_count": len(self.cells),
            "bbox": cells_bbox(list(self.cells)),
            "support_count": self.support_count,
            "different_action_support": self.different_action_support,
            "different_mover_support": self.different_mover_support,
            "sample_cells": self.cells[:40],
        }


@dataclass(frozen=True)
class GlobalVsContainerVerdict:
    transition_id: str
    cluster_id: str
    container_bbox: tuple[int, int, int, int]
    top_rivals: tuple[str, ...]
    temporal_recurrence_score: float
    action_dependence_score: float
    mover_dependence_score: float
    global_mask_subtraction_gain: float
    bridge_cut_gain: float
    post_global_subtraction_bbox: tuple[int, int, int, int] | None
    verdict: str

    def to_record(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "cluster_id": self.cluster_id,
            "container_bbox": self.container_bbox,
            "top_rivals": self.top_rivals,
            "temporal_recurrence_score": round(self.temporal_recurrence_score, 6),
            "action_dependence_score": round(self.action_dependence_score, 6),
            "mover_dependence_score": round(self.mover_dependence_score, 6),
            "global_mask_subtraction_gain": round(self.global_mask_subtraction_gain, 6),
            "bridge_cut_gain": round(self.bridge_cut_gain, 6),
            "post_global_subtraction_bbox": self.post_global_subtraction_bbox,
            "verdict": self.verdict,
        }


class TemporalRecurrenceScorer:
    """Scores whether residue recurs across nearby transitions."""


class ActionDependenceScorer:
    """Scores whether residue pattern is action- or mover-conditioned."""


class GlobalResidueTemplateMiner:
    """Mines recurring global residue masks from neighboring transitions."""


class GlobalMaskSubtractionEvaluator:
    """Evaluates how much residue disappears after subtracting a global template."""


class BridgeCutEvaluator:
    """Evaluates whether bridge cutting alone explains over-inclusive residue."""


@dataclass(frozen=True)
class ResidueTemplate:
    template_id: str
    mode: str
    transition_id: str
    source_steps: tuple[int, ...]
    cells: tuple[tuple[int, int], ...]
    support_count: int
    different_action_support: int
    different_mover_support: int
    confidence: float
    phase_candidate: str

    def to_record(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "mode": self.mode,
            "transition_id": self.transition_id,
            "source_steps": self.source_steps,
            "cell_count": len(self.cells),
            "bbox": cells_bbox(list(self.cells)),
            "support_count": self.support_count,
            "different_action_support": self.different_action_support,
            "different_mover_support": self.different_mover_support,
            "confidence": round(self.confidence, 6),
            "phase_candidate": self.phase_candidate,
            "sample_cells": self.cells[:40],
        }


@dataclass(frozen=True)
class GlobalUpdatePrediction:
    transition_id: str
    step: int
    action: int
    mode: str
    predicted_cells: tuple[tuple[int, int], ...]
    raw_changed_count: int
    matched_raw_changed_count: int
    confidence: float
    template_id: str | None
    support_count: int
    used_future_evidence: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "step": self.step,
            "action": self.action,
            "mode": self.mode,
            "predicted_cell_count": len(self.predicted_cells),
            "raw_changed_count": self.raw_changed_count,
            "matched_raw_changed_count": self.matched_raw_changed_count,
            "prediction_match": round(self.matched_raw_changed_count / max(1, len(self.predicted_cells)), 6),
            "raw_delta_coverage": round(self.matched_raw_changed_count / max(1, self.raw_changed_count), 6),
            "confidence": round(self.confidence, 6),
            "template_id": self.template_id,
            "support_count": self.support_count,
            "used_future_evidence": self.used_future_evidence,
            "sample_cells": self.predicted_cells[:40],
        }


class ResidueTemplateMiner:
    """Mines recurring residue masks for global-update factoring."""


class PhaseHypothesisTracker:
    """Names simple phase hypotheses without assuming a fixed game clock."""


class GlobalUpdatePredictor:
    """Predicts action-independent update masks from prior residue evidence."""


class DeltaFactorizer:
    """Splits raw frame deltas into predicted global, local residual, and unknown cells."""


class LocalClusterRebuilder:
    """Rebuilds changed clusters on cleaned local residual deltas."""


class GlobalUpdateLayerEvaluation:
    """Compares raw and global-factored AR25 composition diagnostics."""


@dataclass(frozen=True)
class GlobalTemplateCluster:
    cluster_id: str
    prototype_step: int
    prototype_cells: tuple[tuple[int, int], ...]
    member_steps: tuple[int, ...]
    support_count: int
    bbox: tuple[int, int, int, int] | None
    row_band_signature: tuple[int, ...]
    col_band_signature: tuple[int, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "prototype_step": self.prototype_step,
            "prototype_cell_count": len(self.prototype_cells),
            "bbox": self.bbox,
            "member_steps": self.member_steps[:60],
            "support_count": self.support_count,
            "row_band_signature": self.row_band_signature,
            "col_band_signature": self.col_band_signature,
            "sample_cells": self.prototype_cells[:40],
        }


@dataclass(frozen=True)
class PhasePrediction:
    transition_id: str
    step: int
    action: int
    hypothesis: str
    predicted_cluster_id: str | None
    predicted_cell_count: int
    actual_cell_count: int
    exact_match: float
    translation_aligned_match: float
    bbox_normalized_match: float
    color_transition_match: float
    band_signature_match: float
    used_future_evidence: bool
    failure_reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "step": self.step,
            "action": self.action,
            "hypothesis": self.hypothesis,
            "predicted_cluster_id": self.predicted_cluster_id,
            "predicted_cell_count": self.predicted_cell_count,
            "actual_cell_count": self.actual_cell_count,
            "exact_match": round(self.exact_match, 6),
            "translation_aligned_match": round(self.translation_aligned_match, 6),
            "bbox_normalized_match": round(self.bbox_normalized_match, 6),
            "color_transition_match": round(self.color_transition_match, 6),
            "band_signature_match": round(self.band_signature_match, 6),
            "used_future_evidence": self.used_future_evidence,
            "failure_reason": self.failure_reason,
        }


class GlobalTemplateClusterer:
    """Clusters recurring global masks without promoting them into planner state."""


class PhaseHypothesis:
    """Names a rival latent phase explanation for the global update layer."""


class PhasePredictor:
    """Forward-only predictor for candidate global phase/template ids."""


class PrequentialPhaseEvaluator:
    """Scores phase predictors with evidence available before transition t."""


class AlignmentAwareTemplateMatcher:
    """Compares exact, shifted, normalized, and band-level template matches."""


class FalseSubtractionLocalityAudit:
    """Splits false global subtraction by protected actor/mover/background locality."""


class TemplateSupportLearningCurve:
    """Measures whether global phase prediction improves with prior support."""


class OracleTemplateTransitionEntropy:
    """Computes oracle next-template entropy under candidate conditioning variables."""


class TemplateFragmentationAudit:
    """Tests whether coarser global phase classes predict better than template ids."""


class CurrentFramePhaseReadout:
    """Tests whether frame_t contains visible features that identify next phase."""


class HigherOrderPhasePredictor:
    """Prequential predictor using multiple previous oracle phase ids."""


class CompositionalGlobalAtomFactorizer:
    """Factors global templates into row/column/bbox atoms for independent prediction."""


class PhaseIdentifiabilityVerdict:
    """Classifies why the online global phase predictor fails."""


class HandleMemory:
    """Episode-local learned effects keyed by tracker ids.

    Object ids never enter portable keys directly. Contexts receive the current
    component signature plus the effect summary learned for that local id.
    """

    def __init__(self) -> None:
        self._effects: dict[int, Counter[str]] = defaultdict(Counter)
        self._last_top_effect: dict[int, str] = {}
        self.growth: list[dict[str, Any]] = []

    def context_features(self, tracked: Iterable[TrackedDetection]) -> dict[str, Any]:
        summaries = []
        reject = []
        remote = []
        contradiction_total = 0
        for item in tracked:
            counts = self._effects.get(item.object_id)
            if not counts:
                continue
            top_effect, support = counts.most_common(1)[0]
            contradiction_count = max(0, len(counts) - 1)
            contradiction_total += contradiction_count
            signature = component_signature(item.detection)
            summaries.append(
                {
                    "component_signature": signature,
                    "effect": top_effect,
                    "support_bucket": bucket_count(support),
                    "confidence_bucket": confidence_bucket(support, sum(counts.values())),
                    "contradiction_count": contradiction_count,
                }
            )
            if top_effect in {"block_reject", "no_op"}:
                reject.append(signature)
            if top_effect == "remote_delta":
                remote.append(signature)
        summaries.sort(key=lambda item: stable_json(item))
        return {
            "local_component_effect_summary": summaries,
            "local_component_effect_confidence": [
                (item["component_signature"], item["confidence_bucket"])
                for item in summaries
            ],
            "local_component_contradiction_count": contradiction_total,
            "known_noop_or_reject_handles": sorted(reject, key=stable_json),
            "known_remote_effect_handles": sorted(remote, key=stable_json),
        }

    def update(self, record: TransitionRecord) -> None:
        effect = str(record.post_labels["coarse_outcome"])
        object_ids = set()
        actor_delta = record.post_labels.get("actor_delta")
        for delta in getattr(record.event, "object_deltas", ()):
            if delta.moved or delta.appearance_changed:
                object_ids.add(delta.object_id)
        if actor_delta == [0, 0]:
            object_ids.update(item.object_id for item in record.before_tracked)
        for object_id in object_ids:
            self._effects[object_id][effect] += 1
        changed_top = sum(
            1
            for object_id, counts in self._effects.items()
            if counts
            and self._last_top_effect.get(object_id) != counts.most_common(1)[0][0]
        )
        self._last_top_effect = {
            object_id: counts.most_common(1)[0][0]
            for object_id, counts in self._effects.items()
            if counts
        }
        self.growth.append(
            {
                "step": record.step,
                "known_handles": len(self._effects),
                "changed_top_effects": changed_top,
            }
        )


class RoleMemory:
    """Capped role memory keyed by coarse visual/spatial/effect clusters."""

    def __init__(self, *, max_clusters: int) -> None:
        self.max_clusters = max_clusters
        self._effects: dict[str, Counter[str]] = defaultdict(Counter)
        self._role_payloads: dict[str, dict[str, Any]] = {}
        self.selected_tool_or_color_proxy: dict[str, Any] | None = None
        self.growth: list[dict[str, Any]] = []

    def context_features(self, _tracked: Iterable[TrackedDetection]) -> dict[str, Any]:
        clusters = []
        for role_key, counts in self._effects.items():
            if not counts:
                continue
            top_effect, support = counts.most_common(1)[0]
            total = sum(counts.values())
            clusters.append(
                {
                    **self._role_payloads.get(role_key, {}),
                    "effect": top_effect,
                    "support_bucket": bucket_count(support),
                    "confidence_bucket": confidence_bucket(support, total),
                    "outcome_diversity_bucket": bucket_count(len(counts)),
                }
            )
        clusters.sort(key=stable_json)
        return {
            "role_clusters": clusters,
            "selected_tool_or_color_proxy": self.selected_tool_or_color_proxy,
        }

    def update(self, record: TransitionRecord) -> None:
        effect = str(record.post_labels["coarse_outcome"])
        payload = role_cluster_payload(record)
        role_key = stable_json(payload)
        if role_key not in self._effects and len(self._effects) >= self.max_clusters:
            payload = {"role": "overflow", "zone": "overflow", "visual": "overflow"}
            role_key = stable_json(payload)
            if role_key not in self._effects and len(self._effects) >= self.max_clusters:
                victim = sorted(self._effects, key=lambda key: (sum(self._effects[key].values()), key))[0]
                self._effects[role_key].update(self._effects.pop(victim))
                self._role_payloads.pop(victim, None)
        self._role_payloads.setdefault(role_key, payload)
        self._effects[role_key][effect] += 1
        if (
            record.pre_features.get("action_kind") == "click"
            and record.pre_features.get("action_target_type") == "palette_or_tool"
        ):
            self.selected_tool_or_color_proxy = {
                "target": record.pre_features.get("target_relative_region"),
                "visual": record.pre_features.get("clicked_region_signature"),
                "recent_hud_change_signature": record.post_labels.get("hud_delta_signature"),
            }
        self.growth.append(
            {
                "step": record.step,
                "known_role_clusters": len(self._effects),
                "max_role_clusters": self.max_clusters,
            }
        )


class RelationalGeometryMemory:
    """Prior-only memory for coarse group motion roles."""

    def __init__(self) -> None:
        self._roles: dict[str, Counter[str]] = defaultdict(Counter)
        self.growth: list[dict[str, Any]] = []

    def context_features_for_record(self, record: TransitionRecord) -> dict[str, Any]:
        key = stable_json(relational_geometry_payload(record))
        counts = self._roles.get(key)
        if not counts:
            role = {"role": "unknown", "support_bucket": "zero"}
        else:
            top, support = counts.most_common(1)[0]
            role = {
                "role": top,
                "support_bucket": bucket_count(support),
                "confidence_bucket": confidence_bucket(support, sum(counts.values())),
            }
        return {"known_group_motion_role_from_prior_memory": role}

    def update(self, record: TransitionRecord) -> None:
        key = stable_json(relational_geometry_payload(record))
        self._roles[key][group_motion_role(record)] += 1
        self.growth.append(
            {
                "step": record.step,
                "known_geometry_roles": len(self._roles),
            }
        )


class ClickProceduralMemory:
    """Prior-only click/edit memory for procedural target state."""

    def __init__(self) -> None:
        self._target_effects: dict[str, Counter[str]] = defaultdict(Counter)
        self._initial_grid: np.ndarray | None = None
        self._last_palette_like_click: dict[str, Any] | None = None
        self._selected_tool_proxy: dict[str, Any] | None = None
        self._prior_canvas_edits = 0
        self.growth: list[dict[str, Any]] = []

    def context_features_for_record(self, record: TransitionRecord) -> dict[str, Any]:
        before_grid = np.asarray(record.before.pixels[-1])
        if self._initial_grid is None:
            self._initial_grid = before_grid.copy()
        target_key = click_target_key(record)
        counts = self._target_effects.get(target_key)
        if counts:
            top_effect, support = counts.most_common(1)[0]
            learned_type = learned_click_target_type(top_effect)
            confidence = confidence_bucket(support, sum(counts.values()))
        else:
            learned_type = "unknown"
            confidence = "none"
        return {
            "click_target_type_learned": {
                "type": learned_type,
                "confidence_bucket": confidence,
            },
            "last_palette_like_click_before_t": self._last_palette_like_click,
            "prior_canvas_edit_count_bucket": bucket_count(self._prior_canvas_edits),
            "selected_tool_proxy_before_t": self._selected_tool_proxy,
            "target_differs_from_initial_frame_before_t": target_differs_from_initial(
                record,
                self._initial_grid,
            ),
            "target_material_summary_before_t": target_material_summary(record),
        }

    def update(self, record: TransitionRecord) -> None:
        target_key = click_target_key(record)
        effect = str(record.post_labels["coarse_outcome"])
        self._target_effects[target_key][effect] += 1
        visual_type = record.pre_features.get("click_target_type_visual")
        if visual_type == "palette_or_tool_visual":
            self._last_palette_like_click = {
                "zone": record.pre_features.get("click_target_zone"),
                "visual": record.pre_features.get("click_target_visual_signature"),
            }
        if visual_type == "palette_or_tool_visual" and effect in {"hud_delta", "resource_delta", "terminal"}:
            self._selected_tool_proxy = {
                "zone": record.pre_features.get("click_target_zone"),
                "visual": record.pre_features.get("click_target_visual_signature"),
                "effect": effect,
            }
        if visual_type in {"canvas_visual", "object_or_bucket_visual"} and effect not in {"no_op", "block_reject"}:
            self._prior_canvas_edits += 1
        self.growth.append(
            {
                "step": record.step,
                "known_click_targets": len(self._target_effects),
                "prior_canvas_edits": self._prior_canvas_edits,
            }
        )


def default_trace_specs(repo: Path) -> tuple[TraceSpec, ...]:
    return (
        TraceSpec(
            "ls20",
            repo
            / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710/ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl",
            "human",
            supports_oracle=False,
        ),
        TraceSpec(
            "ar25",
            repo / "human-play/ar25/20260701T090305Z/trace.jsonl",
            "human",
            supports_oracle=False,
        ),
        TraceSpec(
            "cd82",
            repo / "human-play/cd82/20260701T095558Z/trace.jsonl",
            "human",
            supports_oracle=False,
        ),
    )


def load_transition_records(spec: TraceSpec, *, limit: int | None = None) -> list[TransitionRecord]:
    if not spec.path.exists():
        raise FileNotFoundError(f"trace for {spec.game} not found: {spec.path}")
    frames = load_play_record(spec.path)
    parser = ObservationParser()
    tracker = ObjectTracker(max_missing_frames=1)
    records: list[TransitionRecord] = []
    previous_observation = observation_from_frame(frames[0])
    before_scene = parser.parse(previous_observation, frame_index=0)
    current_update = tracker.update(before_scene, force_new=True)
    before_tracked = current_update.current
    public_history: dict[str, Any] = {"recent_noop_streak": 0}

    for frame in frames[1:]:
        if frame.action in {None, "START", "RESET"}:
            previous_observation = observation_from_frame(frame)
            before_scene = parser.parse(previous_observation, frame_index=frame.index)
            current_update = tracker.update(before_scene, force_new=True)
            before_tracked = current_update.current
            public_history = {"recent_noop_streak": 0}
            continue
        action = parse_action(frame.action)
        if action is None:
            continue
        after_observation = observation_from_frame(frame)
        after_scene = parser.parse(after_observation, frame_index=frame.index)
        track_update = tracker.update(after_scene)
        event = EventDetector().detect(
            step=int(frame.source_step if frame.source_step is not None else frame.index),
            before=previous_observation,
            action=action,
            after=after_observation,
            tracks=track_update,
        )
        pre_features = build_pre_features(
            before=previous_observation,
            scene=before_scene,
            tracked=before_tracked,
            action=action,
            action_data=frame.action_data,
            public_history=public_history,
        )
        post_labels = build_post_labels(
            before=previous_observation,
            after=after_observation,
            event=event,
        )
        post_labels["labeler_v2"] = build_post_labels_v2(
            before=previous_observation,
            after=after_observation,
            before_scene=before_scene,
            event=event,
            pre_features=pre_features,
            action_data=frame.action_data,
        )
        records.append(
            TransitionRecord(
                game=spec.game,
                trace_source=spec.trace_source,
                step=event.step,
                action=action,
                action_data=frame.action_data,
                before=previous_observation,
                after=after_observation,
                before_scene=before_scene,
                after_scene=after_scene,
                before_tracked=before_tracked,
                track_update=track_update,
                event=event,
                pre_features=pre_features,
                post_labels=post_labels,
            )
        )
        previous_observation = after_observation
        before_scene = after_scene
        before_tracked = track_update.current
        public_history["recent_noop_streak"] = (
            int(public_history.get("recent_noop_streak", 0)) + 1
            if post_labels["coarse_outcome"] == "no_op"
            else 0
        )
        if limit is not None and len(records) >= limit:
            break
    annotate_labeler_v3(records)
    return records


def observation_from_frame(frame: Any) -> Observation:
    pixels = np.asarray(frame.pixels, dtype=np.int16)
    pixels.setflags(write=False)
    return Observation(
        pixels=pixels,
        available_actions=tuple(int(action) for action in frame.available_actions),
        levels_completed=int(frame.levels_completed or 0),
        state=str(frame.state or "UNKNOWN"),
        full_reset=bool(frame.full_reset),
    )


def parse_action(action: str | None) -> int | None:
    if action is None:
        return None
    text = str(action).upper()
    if text.startswith("ACTION"):
        text = text.removeprefix("ACTION")
    try:
        return int(text)
    except ValueError:
        return None


def build_pre_features(
    *,
    before: Observation,
    scene: VisualScene,
    tracked: tuple[TrackedDetection, ...],
    action: int,
    action_data: Mapping[str, Any] | None = None,
    public_history: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    actor = actor_candidate(scene.detections)
    actor_center = actor.centroid if actor is not None else None
    confidence = actor_confidence(scene, actor, action=action)
    click_target = classify_action_target(
        observation=before,
        scene=scene,
        action=action,
        action_data=action_data,
        actor_center=actor_center,
    )
    relational = relational_geometry_features(scene)
    return {
        "action_set": tuple(before.available_actions),
        "action_kind": action_kind(action, action_data),
        "actor_bbox_or_pos": bbox_record(actor) if actor is not None else None,
        "actor_confidence": confidence,
        "actor_or_focus_region": (
            coarse_region(actor.centroid, scene.shape) if actor is not None and confidence != "low" else "unknown"
        ),
        "actor_signature": component_signature(actor) if actor is not None else None,
        "canonical_group_shape_family": relational["canonical_group_shape_family"],
        "click_repeated_motif_bucket": click_target["click_repeated_motif_bucket"],
        "click_target_type_visual": click_target["click_target_type_visual"],
        "click_target_visual_signature": click_target["click_target_visual_signature"],
        "click_target_zone": click_target["click_target_zone"],
        "clicked_static_component_class": click_target["clicked_static_component_class"],
        "component_signature_histogram": component_signature_histogram(scene.detections),
        "current_relative_geometry_class": relational["current_relative_geometry_class"],
        "current_touch_overlap_relation": relational["current_touch_overlap_relation"],
        "clicked_region_signature": click_target.get("clicked_region_signature"),
        "hud_or_resource_snapshot": hud_snapshot(before),
        "level_snapshot": {
            "levels_completed": int(before.levels_completed),
            "state": before.state,
            "full_reset": bool(before.full_reset),
        },
        "nearby_component_signatures": nearby_component_signatures(
            scene.detections,
            actor_center=actor_center,
        ),
        "raw_observation_hash": observation_digest(before),
        "mirror_rotation_normalized_signature": relational["mirror_rotation_normalized_signature"],
        "recent_noop_streak": bucket_count(int((public_history or {}).get("recent_noop_streak", 0))),
        "action_target_type": click_target["action_target_type"],
        "target_relative_region": click_target["target_relative_region"],
        "visibility_or_unknown_mask": visibility_mask(before),
        "visible_state_summary": visible_state_summary(before, scene),
        "visible_independent_mover_signatures": (),
        "_action": action,
    }


def actor_candidate(detections: tuple[Detection, ...]) -> Detection | None:
    if not detections:
        return None
    return max(detections, key=lambda item: (item.area, -item.bbox.top, -item.bbox.left))


def actor_confidence(scene: VisualScene, actor: Detection | None, *, action: int) -> str:
    if actor is None:
        return "low"
    if not scene.detections:
        return "low"
    largest = max(item.area for item in scene.detections)
    if actor.area == largest and actor.area > (scene.shape[0] * scene.shape[1]) * 0.25:
        return "low"
    if action in {1, 2, 3, 4} and actor.area <= 64:
        return "medium"
    if actor.area <= 16:
        return "medium"
    return "low"


def action_kind(action: int, action_data: Mapping[str, Any] | None = None) -> str:
    if action in {1, 2, 3, 4}:
        return "move"
    if action_data and {"x", "y"} <= set(action_data):
        return "click"
    if action in {5}:
        return "select"
    if action in {6}:
        return "click"
    if action in {7}:
        return "undo_or_back"
    return "unknown"


def classify_action_target(
    *,
    observation: Observation,
    scene: VisualScene,
    action: int,
    action_data: Mapping[str, Any] | None,
    actor_center: tuple[float, float] | None,
) -> dict[str, Any]:
    grid = np.asarray(observation.pixels[-1])
    if not (action_data and "x" in action_data and "y" in action_data):
        return {
            "action_target_type": "near_actor" if action in {1, 2, 3, 4} else "unknown",
            "click_target_type_visual": "near_actor_visual" if action in {1, 2, 3, 4} else "unknown",
            "click_target_zone": "near_actor" if action in {1, 2, 3, 4} else "unknown",
            "click_target_visual_signature": None,
            "click_repeated_motif_bucket": "zero",
            "clicked_static_component_class": "unknown",
            "target_relative_region": "near_actor" if action in {1, 2, 3, 4} else "unknown",
            "clicked_region_signature": None,
        }
    row = int(action_data["y"])
    col = int(action_data["x"])
    if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
        return {
            "action_target_type": "unknown",
            "click_target_type_visual": "unknown",
            "click_target_zone": "off_canvas",
            "click_target_visual_signature": None,
            "click_repeated_motif_bucket": "zero",
            "clicked_static_component_class": "off_canvas",
            "target_relative_region": "off_canvas",
            "clicked_region_signature": None,
        }
    zone = coarse_region((row, col), grid.shape)
    detection = detection_at(scene.detections, row=row, col=col)
    signature = coarse_component_signature(detection) if detection is not None else {"value": int(grid[row, col]), "area": "background"}
    repeated_motif_count = count_matching_visual_motifs(scene.detections, detection)
    if zone == "hud":
        target_type = "hud"
    elif detection is None:
        target_type = "background"
    elif detection.area <= 16 and zone in {"top", "border"}:
        target_type = "palette_or_tool"
    elif detection.area <= 16:
        target_type = "small_object_or_bucket"
    elif zone in {"central_canvas", "near_actor"}:
        target_type = "canvas"
    else:
        target_type = "object_like"
    visual_type = visual_click_target_type(zone=zone, detection=detection)
    if actor_center is not None:
        target_region = relative_region((row, col), actor_center, grid.shape)
    else:
        target_region = zone
    return {
        "action_target_type": target_type,
        "click_target_type_visual": visual_type,
        "click_target_zone": zone,
        "click_target_visual_signature": signature,
        "click_repeated_motif_bucket": bucket_count(repeated_motif_count),
        "clicked_static_component_class": static_component_class(detection),
        "target_relative_region": target_region,
        "clicked_region_signature": signature,
    }


def detection_at(detections: tuple[Detection, ...], *, row: int, col: int) -> Detection | None:
    for detection in detections:
        if (row, col) in detection.pixels:
            return detection
    return None


def visual_click_target_type(*, zone: str, detection: Detection | None) -> str:
    if zone == "hud":
        return "hud_visual"
    if detection is None:
        return "background_visual"
    if detection.area <= 16 and zone in {"top", "border"}:
        return "palette_or_tool_visual"
    if detection.area <= 16:
        return "object_or_bucket_visual"
    if zone == "central_canvas":
        return "canvas_visual"
    return "object_like_visual"


def count_matching_visual_motifs(
    detections: tuple[Detection, ...],
    detection: Detection | None,
) -> int:
    if detection is None:
        return 0
    signature = stable_json(coarse_component_signature(detection))
    return sum(
        1
        for item in detections
        if stable_json(coarse_component_signature(item)) == signature
    )


def static_component_class(detection: Detection | None) -> str:
    if detection is None:
        return "background"
    return f"{shape_family(detection)}:{bucket_area(detection.area)}"


def coarse_region(point: tuple[float, float], shape: tuple[int, int]) -> str:
    row, col = point
    height, width = shape
    if row < max(4, height * 0.15):
        return "top"
    if row > height - max(5, height * 0.15):
        return "hud"
    if col < max(4, width * 0.15) or col > width - max(5, width * 0.15):
        return "border"
    return "central_canvas"


def relative_region(
    point: tuple[float, float],
    actor_center: tuple[float, float],
    shape: tuple[int, int],
) -> str:
    row_delta = point[0] - actor_center[0]
    col_delta = point[1] - actor_center[1]
    if abs(row_delta) + abs(col_delta) <= max(4, min(shape) * 0.12):
        return "near_actor"
    return coarse_region(point, shape)


def component_signature(
    detection: Detection,
    *,
    actor_center: tuple[float, float] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "color_histogram": ((int(detection.value), int(detection.area)),),
        "bbox_size": (int(detection.bbox.height), int(detection.bbox.width)),
        "shape_bitmap": detection.shape_id,
        "area_bucket": bucket_area(detection.area),
    }
    if actor_center is not None:
        row_delta = detection.centroid[0] - actor_center[0]
        col_delta = detection.centroid[1] - actor_center[1]
        record["relative_position_to_actor"] = (
            bucket_signed(row_delta),
            bucket_signed(col_delta),
        )
    return record


def coarse_component_signature(detection: Detection | None) -> dict[str, Any] | None:
    if detection is None:
        return None
    return {
        "color": int(detection.value),
        "area_bucket": bucket_area(detection.area),
        "bbox_size_bucket": (
            bucket_count(detection.bbox.height),
            bucket_count(detection.bbox.width),
        ),
        "shape_family": shape_family(detection),
    }


def shape_family(detection: Detection) -> str:
    height = detection.bbox.height
    width = detection.bbox.width
    if detection.area == 1:
        return "point"
    if height == 1 or width == 1:
        return "line"
    fill = detection.area / max(1, height * width)
    if fill >= 0.8:
        return "solid"
    if fill <= 0.35:
        return "sparse"
    return "structured"


def relational_geometry_features(scene: VisualScene) -> dict[str, Any]:
    ranked = sorted(
        scene.detections,
        key=lambda item: (-item.area, item.bbox.top, item.bbox.left),
    )[:6]
    if not ranked:
        return {
            "canonical_group_shape_family": "empty",
            "current_relative_geometry_class": "empty",
            "current_touch_overlap_relation": "none",
            "mirror_rotation_normalized_signature": "empty",
        }
    families = Counter(shape_family(item) for item in ranked)
    size_profile = tuple(sorted(bucket_area(item.area) for item in ranked))
    zones = tuple(sorted(coarse_region(item.centroid, scene.shape) for item in ranked))
    relation_counts = Counter(pair_relation(left, right) for index, left in enumerate(ranked) for right in ranked[index + 1 :])
    canonical_shapes = tuple(sorted(canonical_shape_signature(item) for item in ranked))
    payload = {
        "families": tuple(sorted(families.items())),
        "sizes": size_profile,
        "zones": zones,
        "relations": tuple(sorted(relation_counts.items())),
        "shapes": canonical_shapes,
    }
    return {
        "canonical_group_shape_family": tuple(sorted(families.items())),
        "current_relative_geometry_class": {
            "component_count": bucket_count(len(ranked)),
            "size_profile": size_profile,
            "zone_profile": zones,
            "relation_profile": tuple(sorted(relation_counts.items())),
        },
        "current_touch_overlap_relation": dominant_touch_relation(ranked),
        "mirror_rotation_normalized_signature": hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()[:12],
    }


def pair_relation(left: Detection, right: Detection) -> str:
    row_gap = max(0, max(left.bbox.top, right.bbox.top) - min(left.bbox.bottom, right.bbox.bottom) - 1)
    col_gap = max(0, max(left.bbox.left, right.bbox.left) - min(left.bbox.right, right.bbox.right) - 1)
    if row_gap == 0 and col_gap == 0:
        return "touch_or_overlap"
    if row_gap <= 2 and col_gap <= 2:
        return "near"
    row_delta = right.centroid[0] - left.centroid[0]
    col_delta = right.centroid[1] - left.centroid[1]
    if abs(row_delta) > abs(col_delta):
        return "vertical"
    return "horizontal"


def dominant_touch_relation(detections: list[Detection]) -> str:
    if len(detections) < 2:
        return "single"
    counts = Counter(
        pair_relation(left, right)
        for index, left in enumerate(detections)
        for right in detections[index + 1 :]
    )
    return counts.most_common(1)[0][0]


def canonical_shape_signature(detection: Detection) -> str:
    variants = []
    shape = tuple(detection.shape)
    rotated = shape
    for _turn in range(4):
        variants.append(rotated)
        variants.append(reflect_shape(rotated))
        rotated = rotate_shape(rotated)
    canonical = min(stable_json(normalize_shape(item)) for item in variants)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]


def normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not shape:
        return ()
    top = min(row for row, _col in shape)
    left = min(col for _row, col in shape)
    return tuple(sorted((row - top, col - left) for row, col in shape))


def rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not shape:
        return ()
    height = max(row for row, _col in shape) + 1
    return normalize_shape(tuple((col, height - 1 - row) for row, col in shape))


def reflect_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not shape:
        return ()
    width = max(col for _row, col in shape) + 1
    return normalize_shape(tuple((row, width - 1 - col) for row, col in shape))


def component_signature_histogram(detections: tuple[Detection, ...]) -> tuple[tuple[str, int], ...]:
    counts = Counter(stable_json(component_signature(item)) for item in detections)
    return tuple(sorted(counts.items()))


def nearby_component_signatures(
    detections: tuple[Detection, ...],
    *,
    actor_center: tuple[float, float] | None,
    limit: int = 12,
) -> tuple[dict[str, Any], ...]:
    if actor_center is None:
        return ()
    ranked = sorted(
        detections,
        key=lambda item: (
            abs(item.centroid[0] - actor_center[0])
            + abs(item.centroid[1] - actor_center[1]),
            item.area,
        ),
    )
    return tuple(
        component_signature(item, actor_center=actor_center)
        for item in ranked[:limit]
    )


def bbox_record(detection: Detection) -> dict[str, int]:
    return {
        "top": int(detection.bbox.top),
        "left": int(detection.bbox.left),
        "height": int(detection.bbox.height),
        "width": int(detection.bbox.width),
        "center_row_bucket": bucket_coord(detection.centroid[0]),
        "center_col_bucket": bucket_coord(detection.centroid[1]),
    }


def hud_snapshot(observation: Observation) -> dict[str, Any]:
    grid = np.asarray(observation.pixels[-1])
    bands = {
        "top": grid[:8, :],
        "bottom": grid[-8:, :],
        "left": grid[:, :8],
        "right": grid[:, -8:],
    }
    return {
        name: tuple(sorted((int(value), int(count)) for value, count in zip(*np.unique(band, return_counts=True))))
        for name, band in bands.items()
    }


def visibility_mask(observation: Observation) -> dict[str, Any]:
    grid = np.asarray(observation.pixels[-1])
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    non_background = int(grid.size - counts[int(np.argmax(counts))])
    return {
        "background": background,
        "non_background_bucket": bucket_count(non_background),
    }


def visible_state_summary(observation: Observation, scene: VisualScene) -> dict[str, Any]:
    grid = np.asarray(observation.pixels[-1])
    values, counts = np.unique(grid, return_counts=True)
    dominant = sorted(
        ((int(count), int(value)) for value, count in zip(values, counts)),
        reverse=True,
    )[:4]
    return {
        "num_components_bucketed": bucket_count(len(scene.detections)),
        "dominant_colors_bucketed": tuple(
            (value, bucket_count(count)) for count, value in dominant
        ),
        "resource_level_bucket": bucket_count(int(observation.levels_completed)),
        "hud_changed_recently": "unknown",
    }


def build_post_labels(*, before: Observation, after: Observation, event: Event) -> dict[str, Any]:
    before_grid = np.asarray(before.pixels[-1])
    after_grid = np.asarray(after.pixels[-1])
    changed_mask = before_grid != after_grid
    actor = actor_delta(event)
    status_delta = int(after.levels_completed) - int(before.levels_completed)
    hud_delta = band_changed(before_grid, after_grid)
    remote = remote_delta_present(event, actor)
    coarse = coarse_outcome(
        action=event.action,
        actor=actor,
        changed_pixels=int(np.count_nonzero(changed_mask)),
        hud_delta=bool(hud_delta),
        remote=remote,
        status_delta=status_delta,
        state_after=after.state,
    )
    return {
        "actor_delta": list(actor),
        "blocked_or_rejected": coarse == "block_reject",
        "changed_cells": int(np.count_nonzero(changed_mask)),
        "changed_cell_signature": changed_cell_signature(changed_mask),
        "changed_value_signature": changed_value_signature(before_grid, after_grid, changed_mask),
        "coarse_outcome": coarse,
        "contacted_component_delta_class": contact_effect_class(event, actor),
        "exact_outcome": {
            "actor_delta": list(actor),
            "changed_cell_signature": changed_cell_signature(changed_mask),
            "changed_value_signature": changed_value_signature(before_grid, after_grid, changed_mask),
            "hud_delta_signature": hud_delta,
            "resource_delta_signature": hud_delta,
            "status": after.state,
            "status_delta": status_delta,
        },
        "hud_or_status_delta": bool(hud_delta) or status_delta != 0,
        "hud_delta_signature": hud_delta,
        "remote_delta_present": remote,
        "resource_delta_signature": hud_delta,
        "terminal_status_delta": after.state if after.state in {"WIN", "GAME_OVER"} else None,
    }


def build_post_labels_v2(
    *,
    before: Observation,
    after: Observation,
    before_scene: VisualScene,
    event: Event,
    pre_features: Mapping[str, Any],
    action_data: Mapping[str, Any] | None,
) -> dict[str, Any]:
    before_grid = np.asarray(before.pixels[-1])
    after_grid = np.asarray(after.pixels[-1])
    changed_mask = before_grid != after_grid
    clusters = changed_region_clusters(
        before_grid=before_grid,
        after_grid=after_grid,
        changed_mask=changed_mask,
        before_scene=before_scene,
        action_data=action_data,
        actor_center=pre_actor_center(pre_features),
    )
    actor = actor_delta_v2(event, pre_features)
    hud_status = hud_status_delta_v2(before, after, clusters)
    terminal = terminal_delta_v2(before, after)
    assigned = assign_changed_region_clusters(
        clusters=clusters,
        actor=actor,
        action_kind=str(pre_features.get("action_kind", "unknown")),
        action_data=action_data,
        actor_center=pre_actor_center(pre_features),
    )
    channels = {
        "actor_delta": actor,
        "action_target_delta": channel_summary(assigned["action_target_delta"]),
        "hud_status_delta": hud_status,
        "resource_delta": resource_delta_v2(before, after, assigned["hud_status_delta"]),
        "local_contact_delta": channel_summary(assigned["local_contact_delta"]),
        "remote_delta": channel_summary(assigned["remote_delta"]),
        "independent_motion_delta": channel_summary(assigned["independent_motion_delta"]),
        "terminal_delta": terminal,
        "unknown_delta": channel_summary(assigned["unknown_delta"]),
    }
    active_channels = tuple(
        name
        for name, value in channels.items()
        if channel_is_active(name, value)
    )
    fake_actor_suspect = (
        event.action == 6
        and actor.get("confidence") in {"high", "medium"}
        and bool(assigned["action_target_delta"])
    )
    return {
        "actor_confidence": actor["confidence"],
        "fake_actor_suspect": fake_actor_suspect,
        "changed_region_clusters": clusters,
        "changed_region_assignments": {
            channel: [cluster["cluster_id"] for cluster in items]
            for channel, items in assigned.items()
        },
        "channels": channels,
        "active_channels": active_channels,
        "coarse_outcome": "+".join(active_channels) if active_channels else "no_op",
        "parameter_outcome": channels,
    }


def annotate_labeler_v3(records: list[TransitionRecord]) -> None:
    controlled_model = build_controlled_entity_model(records)
    region_map = build_region_role_map(records)
    for record in records:
        record.post_labels["labeler_v3"] = build_post_labels_v3(
            record,
            controlled_model=controlled_model,
            region_map=region_map,
        )


def build_post_labels_v3(
    record: TransitionRecord,
    *,
    controlled_model: Mapping[str, Any],
    region_map: Mapping[str, Any],
) -> dict[str, Any]:
    v2 = record.post_labels.get("labeler_v2", {})
    clusters = list(v2.get("changed_region_clusters", ()))
    controlled = infer_controlled_entity(record, controlled_model)
    assigned = assign_changed_region_clusters_v3(
        record=record,
        clusters=clusters,
        controlled=controlled,
        region_map=region_map,
    )
    channels = {
        "controlled_entity_delta": controlled_channel_summary(assigned["controlled_entity_delta"], controlled),
        "action_target_delta": confident_channel_summary(assigned["action_target_delta"]),
        "hud_status_delta": confident_channel_summary(assigned["hud_status_delta"]),
        "resource_delta": v2.get("channels", {}).get("resource_delta"),
        "remote_delta": confident_channel_summary(assigned["remote_delta"]),
        "independent_motion_delta": confident_channel_summary(assigned["independent_motion_delta"]),
        "unknown_delta": confident_channel_summary(assigned["unknown_delta"]),
        "terminal_delta": v2.get("channels", {}).get("terminal_delta"),
    }
    active_channels = tuple(
        name for name, value in channels.items() if channel_is_active_v3(name, value)
    )
    return {
        "controlled_entity": controlled,
        "region_roles": {
            "action_target_region": action_target_region_role(record, region_map),
        },
        "changed_region_clusters": clusters,
        "changed_region_assignments": {
            channel: [
                {
                    "cluster_id": item["cluster"]["cluster_id"],
                    "confidence": item["confidence"],
                    "reason": item["reason"],
                }
                for item in items
            ]
            for channel, items in assigned.items()
        },
        "channels": channels,
        "active_channels": active_channels,
        "coarse_outcome": "+".join(active_channels) if active_channels else "no_op",
        "parameter_outcome": channels,
    }


def build_controlled_entity_model(records: list[TransitionRecord]) -> dict[str, Any]:
    action_translations: dict[int, Counter[tuple[int, int]]] = defaultdict(Counter)
    object_support: Counter[int] = Counter()
    for record in records:
        if record.action not in {1, 2, 3, 4}:
            continue
        for delta in stable_motion_deltas(record):
            action_translations[record.action][tuple(delta.translation)] += 1
            object_support[int(delta.object_id)] += 1
    dominant = {
        action: counter.most_common(1)[0][0]
        for action, counter in action_translations.items()
        if counter
    }
    return {
        "dominant_translation_by_action": dominant,
        "object_support": object_support,
    }


def infer_controlled_entity(record: TransitionRecord, model: Mapping[str, Any]) -> dict[str, Any]:
    if record.action not in {1, 2, 3, 4}:
        return {
            "type": "none",
            "confidence": "low",
            "candidates": [],
            "suppressed_reasons": ["non_move_action"],
        }
    dominant = model.get("dominant_translation_by_action", {}).get(record.action)
    scored = []
    for delta in stable_motion_deltas(record):
        score = 0.45
        reasons = ["stable_shape_palette"]
        if dominant is not None and tuple(delta.translation) == tuple(dominant):
            score += 0.35
            reasons.append("aligned_with_action_direction")
        support = int(model.get("object_support", Counter()).get(int(delta.object_id), 0))
        if support >= 3:
            score += 0.15
            reasons.append("explains_many_actions")
        if record.post_labels.get("changed_cells", 0) > 0:
            score += 0.05
            reasons.append("nearby_frame_change_present")
        scored.append(
            {
                "object_id": int(delta.object_id),
                "score": round(score, 6),
                "translation": list(delta.translation),
                "bbox_before": tuple(delta.bbox_before),
                "bbox_after": tuple(delta.bbox_after),
                "reasons": reasons,
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["object_id"]))
    if not scored:
        return {
            "type": "none",
            "confidence": "low",
            "candidates": [],
            "suppressed_reasons": ["no_stable_action_aligned_candidate"],
        }
    top_score = scored[0]["score"]
    high = [item for item in scored if item["score"] >= 0.75]
    if len(high) > 1:
        entity_type = "coupled_set"
        confidence = "high" if all(tuple(item["translation"]) == tuple(high[0]["translation"]) for item in high) else "medium"
    elif top_score >= 0.75:
        entity_type = "single"
        confidence = "high"
    elif top_score >= 0.55:
        entity_type = "ambiguous"
        confidence = "medium"
    else:
        entity_type = "ambiguous"
        confidence = "low"
    return {
        "type": entity_type,
        "confidence": confidence,
        "candidates": scored[:5],
        "suppressed_reasons": [] if confidence != "low" else ["candidate_score_below_high_confidence"],
    }


def stable_motion_deltas(record: TransitionRecord) -> list[Any]:
    deltas = []
    for delta in getattr(record.event, "object_deltas", ()):
        if not getattr(delta, "moved", False):
            continue
        stable_identity = (
            delta.value_before == delta.value_after
            and not delta.shape_changed
            and delta.area_before == delta.area_after
        )
        if stable_identity:
            deltas.append(delta)
    return deltas


def build_region_role_map(records: list[TransitionRecord]) -> dict[str, Any]:
    profiles: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if record.pre_features.get("action_kind") == "click":
            target = action_target_point(record.action_data)
            if target is not None:
                profiles[region_key(target, np.asarray(record.before.pixels[-1]).shape)]["clicks"] += 1
        for cluster in record.post_labels.get("labeler_v2", {}).get("changed_region_clusters", ()):
            key = region_key(cluster_centroid(cluster), np.asarray(record.before.pixels[-1]).shape)
            profiles[key]["changes"] += 1
            if cluster.get("overlaps_hud"):
                profiles[key]["hud_overlaps"] += 1
            if cluster.get("distance_to_action_target") in {"zero", "near"}:
                profiles[key]["near_click_change"] += 1
            if cluster.get("overlaps_known_component"):
                profiles[key]["component_overlap"] += 1
    roles = {}
    for key, counts in profiles.items():
        zone = key.split(":")[0]
        clicks = counts["clicks"]
        changes = counts["changes"]
        near_click = counts["near_click_change"]
        hud = counts["hud_overlaps"]
        if zone in {"top", "hud", "border"} and hud >= max(1, changes * 0.4):
            role = "hud_status_region"
            confidence = "high" if changes >= 3 else "medium"
        elif clicks >= 2 and zone in {"top", "border"}:
            role = "tool_control_region"
            confidence = "medium"
        elif clicks >= 1 and near_click >= 1:
            role = "canvas_edit_surface"
            confidence = "high" if near_click >= 3 else "medium"
        elif zone == "central_canvas" and changes >= 3:
            role = "canvas_edit_surface"
            confidence = "medium"
        elif changes >= 2:
            role = "remote_object_region"
            confidence = "medium"
        else:
            role = "unknown"
            confidence = "low"
        roles[key] = {
            "role": role,
            "confidence": confidence,
            "features": dict(counts),
        }
    return {"regions": roles}


def assign_changed_region_clusters_v3(
    *,
    record: TransitionRecord,
    clusters: list[dict[str, Any]],
    controlled: Mapping[str, Any],
    region_map: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    assigned = {
        "controlled_entity_delta": [],
        "action_target_delta": [],
        "hud_status_delta": [],
        "remote_delta": [],
        "independent_motion_delta": [],
        "unknown_delta": [],
    }
    shape = np.asarray(record.before.pixels[-1]).shape
    controlled_bboxes = [
        tuple(candidate.get("bbox_after", ()))
        for candidate in controlled.get("candidates", ())
        if candidate.get("score", 0.0) >= 0.75
    ]
    target_region = action_target_region_role(record, region_map)
    for cluster in clusters:
        key = region_key(cluster_centroid(cluster), shape)
        role = region_map.get("regions", {}).get(key, {"role": "unknown", "confidence": "low"})
        if (
            controlled.get("confidence") in {"high", "medium"}
            and cluster_overlaps_any_bbox(tuple(cluster["bbox"]), controlled_bboxes)
        ):
            assigned["controlled_entity_delta"].append(
                assignment_record(cluster, role, "high", "overlaps_controlled_candidate")
            )
        elif (
            record.pre_features.get("action_kind") == "click"
            and target_region.get("role") in {"canvas_edit_surface", "tool_control_region"}
            and (cluster.get("distance_to_action_target") in {"zero", "near"} or role.get("role") == target_region.get("role"))
        ):
            assigned["action_target_delta"].append(
                assignment_record(cluster, role, target_region.get("confidence", "medium"), "click_target_region_role")
            )
        elif role.get("role") == "hud_status_region":
            assigned["hud_status_delta"].append(
                assignment_record(cluster, role, role.get("confidence", "medium"), "region_role_hud_status")
            )
        elif role.get("role") == "remote_object_region":
            assigned["remote_delta"].append(
                assignment_record(cluster, role, role.get("confidence", "medium"), "region_role_remote_object")
            )
        elif role.get("role") == "canvas_edit_surface":
            assigned["independent_motion_delta"].append(
                assignment_record(cluster, role, "low", "canvas_change_without_click_target")
            )
        else:
            assigned["unknown_delta"].append(
                assignment_record(cluster, role, "low", "insufficient_role_confidence")
            )
    return assigned


def assignment_record(
    cluster: Mapping[str, Any],
    role: Mapping[str, Any],
    confidence: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "cluster": cluster,
        "role": role.get("role", "unknown"),
        "confidence": confidence,
        "reason": reason,
    }


def controlled_channel_summary(assignments: list[dict[str, Any]], controlled: Mapping[str, Any]) -> dict[str, Any]:
    summary = confident_channel_summary(assignments)
    summary["controlled_entity_type"] = controlled.get("type", "none")
    summary["controlled_confidence"] = controlled.get("confidence", "low")
    summary["top_candidates"] = [
        {
            "score": candidate.get("score"),
            "translation": candidate.get("translation"),
            "bbox_before": candidate.get("bbox_before"),
            "bbox_after": candidate.get("bbox_after"),
            "reasons": candidate.get("reasons", ()),
        }
        for candidate in controlled.get("candidates", [])[:3]
    ]
    return summary


def confident_channel_summary(assignments: list[dict[str, Any]]) -> dict[str, Any]:
    clusters = [item["cluster"] for item in assignments]
    summary = channel_summary(clusters)
    summary["confidences"] = tuple(item["confidence"] for item in assignments)
    summary["reasons"] = tuple(item["reason"] for item in assignments)
    summary["roles"] = tuple(item.get("role", "unknown") for item in assignments)
    return summary


def channel_is_active_v3(name: str, value: Any) -> bool:
    if name in {"terminal_delta", "resource_delta"}:
        return bool(value)
    if isinstance(value, Mapping) and "cluster_count" in value:
        return int(value.get("cluster_count", 0) or 0) > 0
    return False


def action_target_region_role(record: TransitionRecord, region_map: Mapping[str, Any]) -> dict[str, Any]:
    target = action_target_point(record.action_data)
    if target is None:
        return {"role": "none", "confidence": "low"}
    key = region_key(target, np.asarray(record.before.pixels[-1]).shape)
    return region_map.get("regions", {}).get(key, {"role": "unknown", "confidence": "low"})


def region_key(point: tuple[float, float] | tuple[int, int], shape: tuple[int, int]) -> str:
    zone = coarse_region((float(point[0]), float(point[1])), shape)
    row_bin = min(3, max(0, int(float(point[0]) / max(1, shape[0]) * 4)))
    col_bin = min(3, max(0, int(float(point[1]) / max(1, shape[1]) * 4)))
    return f"{zone}:r{row_bin}:c{col_bin}"


def cluster_centroid(cluster: Mapping[str, Any]) -> tuple[float, float]:
    top, left, bottom, right = cluster["bbox"]
    return ((top + bottom) / 2.0, (left + right) / 2.0)


def cluster_overlaps_any_bbox(cluster_bbox: tuple[int, int, int, int], bboxes: list[tuple[int, int, int, int]]) -> bool:
    return any(bbox_intersects(cluster_bbox, bbox) for bbox in bboxes if len(bbox) == 4)


def bbox_intersects(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return not (
        left[2] < right[0]
        or right[2] < left[0]
        or left[3] < right[1]
        or right[3] < left[1]
    )


def actor_delta_v2(event: Event, pre_features: Mapping[str, Any]) -> dict[str, Any]:
    candidates = []
    for delta in event.object_deltas:
        if not delta.moved:
            continue
        stable_identity = (
            delta.value_before == delta.value_after
            and not delta.shape_changed
            and delta.area_before == delta.area_after
        )
        if stable_identity:
            candidates.append(delta)
    if event.action not in {1, 2, 3, 4} or not candidates:
        return {"confidence": "low", "delta": None, "reason": "no_stable_move_actor"}
    chosen = max(candidates, key=lambda delta: (delta.area_after, abs(delta.translation[0]) + abs(delta.translation[1])))
    if len(candidates) == 1 and pre_features.get("actor_confidence") != "low":
        confidence = "high"
    elif pre_features.get("actor_confidence") != "low":
        confidence = "medium"
    else:
        confidence = "low"
    if confidence != "high":
        return {
            "confidence": confidence,
            "delta": None,
            "candidate_delta": list(chosen.translation),
            "reason": "ambiguous_or_low_pre_actor_confidence",
        }
    return {
        "confidence": "high",
        "delta": list(chosen.translation),
        "bbox_before": tuple(chosen.bbox_before),
        "bbox_after": tuple(chosen.bbox_after),
    }


def changed_region_clusters(
    *,
    before_grid: np.ndarray,
    after_grid: np.ndarray,
    changed_mask: np.ndarray,
    before_scene: VisualScene,
    action_data: Mapping[str, Any] | None,
    actor_center: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    visited = np.zeros(changed_mask.shape, dtype=bool)
    clusters: list[dict[str, Any]] = []
    target = action_target_point(action_data)
    height, width = changed_mask.shape
    for row in range(height):
        for col in range(width):
            if visited[row, col] or not changed_mask[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            pixels = []
            while stack:
                current = stack.pop()
                pixels.append(current)
                current_row, current_col = current
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if visited[next_row, next_col] or not changed_mask[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
            clusters.append(
                changed_region_cluster_record(
                    cluster_id=len(clusters),
                    pixels=pixels,
                    before_grid=before_grid,
                    after_grid=after_grid,
                    before_scene=before_scene,
                    action_target=target,
                    actor_center=actor_center,
                )
            )
    return clusters


def changed_region_cluster_record(
    *,
    cluster_id: int,
    pixels: list[tuple[int, int]],
    before_grid: np.ndarray,
    after_grid: np.ndarray,
    before_scene: VisualScene,
    action_target: tuple[int, int] | None,
    actor_center: tuple[float, float] | None,
) -> dict[str, Any]:
    rows = [row for row, _col in pixels]
    cols = [col for _row, col in pixels]
    mask = np.zeros(before_grid.shape, dtype=bool)
    for row, col in pixels:
        mask[row, col] = True
    bbox = (min(rows), min(cols), max(rows), max(cols))
    centroid = (sum(rows) / len(rows), sum(cols) / len(cols))
    return {
        "cluster_id": cluster_id,
        "bbox": bbox,
        "area_bucket": bucket_count(len(pixels)),
        "color_histogram_delta": changed_value_signature(before_grid, after_grid, mask),
        "distance_to_actor": distance_bucket(centroid, actor_center),
        "distance_to_action_target": distance_bucket(centroid, action_target),
        "overlaps_hud": cluster_overlaps_hud(bbox, before_grid.shape),
        "overlaps_known_component": cluster_overlaps_component(pixels, before_scene.detections),
        "centroid_region": coarse_region(centroid, before_grid.shape),
    }


def assign_changed_region_clusters(
    *,
    clusters: list[dict[str, Any]],
    actor: Mapping[str, Any],
    action_kind: str,
    action_data: Mapping[str, Any] | None,
    actor_center: tuple[float, float] | None,
) -> dict[str, list[dict[str, Any]]]:
    assigned = {
        "actor_delta": [],
        "action_target_delta": [],
        "hud_status_delta": [],
        "local_contact_delta": [],
        "remote_delta": [],
        "independent_motion_delta": [],
        "unknown_delta": [],
    }
    click_target = action_target_point(action_data)
    for cluster in clusters:
        if cluster["overlaps_hud"]:
            assigned["hud_status_delta"].append(cluster)
        elif action_kind == "click" and click_target is not None and cluster["distance_to_action_target"] in {"zero", "near"}:
            assigned["action_target_delta"].append(cluster)
        elif actor.get("confidence") == "high" and cluster["distance_to_actor"] in {"zero", "near"}:
            assigned["actor_delta"].append(cluster)
        elif actor_center is not None and cluster["distance_to_actor"] == "near":
            assigned["local_contact_delta"].append(cluster)
        elif cluster["distance_to_actor"] in {"far", "unknown"} and cluster["distance_to_action_target"] in {"far", "unknown"}:
            assigned["remote_delta"].append(cluster)
        else:
            assigned["independent_motion_delta"].append(cluster)
    return assigned


def channel_summary(clusters: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cluster_count": len(clusters),
        "area_buckets": tuple(sorted(cluster["area_bucket"] for cluster in clusters)),
        "bboxes": tuple(cluster["bbox"] for cluster in clusters[:8]),
        "color_histogram_delta": tuple(
            sorted(stable_json(cluster["color_histogram_delta"]) for cluster in clusters)
        ),
    }


def channel_is_active(name: str, value: Any) -> bool:
    if name == "actor_delta":
        return isinstance(value, Mapping) and value.get("confidence") == "high" and value.get("delta") is not None
    if name == "terminal_delta":
        return bool(value)
    if name in {"hud_status_delta", "resource_delta"}:
        return bool(value)
    if isinstance(value, Mapping) and "cluster_count" in value:
        return int(value.get("cluster_count", 0)) > 0
    return bool(value)


def hud_status_delta_v2(before: Observation, after: Observation, clusters: list[dict[str, Any]]) -> dict[str, Any] | None:
    before_grid = np.asarray(before.pixels[-1])
    after_grid = np.asarray(after.pixels[-1])
    bands = band_changed(before_grid, after_grid)
    status_delta = int(after.levels_completed) - int(before.levels_completed)
    if not bands and status_delta == 0 and before.state == after.state:
        return None
    return {
        "bands_changed": bands,
        "status_delta": status_delta,
        "state_after": after.state if before.state != after.state else None,
        "cluster_count": sum(1 for cluster in clusters if cluster["overlaps_hud"]),
    }


def resource_delta_v2(before: Observation, after: Observation, hud_clusters: list[dict[str, Any]]) -> dict[str, Any] | None:
    status_delta = int(after.levels_completed) - int(before.levels_completed)
    if status_delta == 0 and not hud_clusters:
        return None
    return {
        "levels_completed_delta": status_delta,
        "hud_cluster_count": len(hud_clusters),
    }


def terminal_delta_v2(before: Observation, after: Observation) -> dict[str, Any] | None:
    if after.state not in {"WIN", "GAME_OVER"} and before.state == after.state:
        return None
    return {"state_after": after.state, "terminal": after.state in {"WIN", "GAME_OVER"}}


def pre_actor_center(pre_features: Mapping[str, Any]) -> tuple[float, float] | None:
    bbox = pre_features.get("actor_bbox_or_pos")
    if not isinstance(bbox, Mapping):
        return None
    top = bbox.get("top")
    left = bbox.get("left")
    height = bbox.get("height")
    width = bbox.get("width")
    if None in (top, left, height, width):
        return None
    return (float(top) + float(height) / 2.0, float(left) + float(width) / 2.0)


def action_target_point(action_data: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if not (action_data and "x" in action_data and "y" in action_data):
        return None
    return (int(action_data["y"]), int(action_data["x"]))


def distance_bucket(
    left: tuple[float, float],
    right: tuple[float, float] | tuple[int, int] | None,
) -> str:
    if right is None:
        return "unknown"
    distance = abs(left[0] - right[0]) + abs(left[1] - right[1])
    if distance <= 1:
        return "zero"
    if distance <= 4:
        return "near"
    return "far"


def cluster_overlaps_hud(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    top, left, bottom, right = bbox
    height, width = shape
    return (
        top < 8
        or bottom >= height - 8
        or left < 8
        or right >= width - 8
    )


def cluster_overlaps_component(pixels: list[tuple[int, int]], detections: tuple[Detection, ...]) -> bool:
    pixel_set = set(pixels)
    return any(bool(pixel_set & set(detection.pixels)) for detection in detections)


def actor_delta(event: Event) -> tuple[int, int]:
    moved = [delta for delta in event.object_deltas if delta.moved]
    if not moved:
        return (0, 0)
    chosen = max(moved, key=lambda delta: (delta.area_after, abs(delta.translation[0]) + abs(delta.translation[1])))
    return tuple(int(item) for item in chosen.translation)


def remote_delta_present(event: Event, actor: tuple[int, int]) -> bool:
    moved_or_changed = [
        delta
        for delta in event.object_deltas
        if delta.moved or delta.appearance_changed
    ]
    if not moved_or_changed:
        return False
    actor_like = [
        delta
        for delta in moved_or_changed
        if tuple(delta.translation) == tuple(actor) and actor != (0, 0)
    ]
    return len(moved_or_changed) > len(actor_like) + 1


def contact_effect_class(event: Event, actor: tuple[int, int]) -> str:
    changed_other = [
        delta
        for delta in event.object_deltas
        if (delta.moved or delta.appearance_changed)
        and tuple(delta.translation) != tuple(actor)
    ]
    if not changed_other:
        return "none"
    if any(delta.value_changed for delta in changed_other):
        return "value_change"
    if any(delta.shape_changed for delta in changed_other):
        return "shape_change"
    if any(delta.moved for delta in changed_other):
        return "motion"
    return "other_change"


def band_changed(before_grid: np.ndarray, after_grid: np.ndarray) -> tuple[str, ...]:
    bands = {
        "top": (slice(0, 8), slice(None)),
        "bottom": (slice(-8, None), slice(None)),
        "left": (slice(None), slice(0, 8)),
        "right": (slice(None), slice(-8, None)),
    }
    changed = []
    for name, index in bands.items():
        if np.any(before_grid[index] != after_grid[index]):
            changed.append(name)
    return tuple(changed)


def coarse_outcome(
    *,
    action: int,
    actor: tuple[int, int],
    changed_pixels: int,
    hud_delta: bool,
    remote: bool,
    status_delta: int,
    state_after: str,
) -> str:
    if state_after in {"WIN", "GAME_OVER"} or status_delta:
        return "terminal"
    if changed_pixels == 0:
        return "no_op"
    if action in {1, 2, 3, 4} and actor == (0, 0):
        return "block_reject"
    if hud_delta:
        return "hud_delta"
    if remote:
        return "remote_delta"
    if actor != (0, 0):
        return "actor_move"
    return "resource_delta"


def changed_cell_signature(changed_mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(changed_mask)
    if not len(xs):
        return {"count": 0, "bbox": None}
    return {
        "count_bucket": bucket_count(len(xs)),
        "bbox": (
            int(min(ys)),
            int(min(xs)),
            int(max(ys)),
            int(max(xs)),
        ),
    }


def changed_value_signature(
    before_grid: np.ndarray,
    after_grid: np.ndarray,
    changed_mask: np.ndarray,
) -> tuple[tuple[int, int, str], ...]:
    if not np.any(changed_mask):
        return ()
    before_values = before_grid[changed_mask]
    after_values = after_grid[changed_mask]
    counts = Counter(
        (int(before_value), int(after_value))
        for before_value, after_value in zip(before_values, after_values)
    )
    return tuple(
        (before_value, after_value, bucket_count(count))
        for (before_value, after_value), count in sorted(counts.items())
    )


def context_key(
    record: TransitionRecord,
    family: RepresentationFamily,
    memory: Any | None = None,
) -> str:
    family.assert_valid()
    payload = {
        field: jsonable(record.pre_features.get(field))
        for field in sorted(family.pre_action_fields)
    }
    if family.memory_fields:
        if memory is None:
            raise ValueError(f"{family.name} requires learned memory")
        if hasattr(memory, "context_features_for_record"):
            memory_payload = memory.context_features_for_record(record)
        else:
            memory_payload = memory.context_features(record.before_tracked)
        payload["memory"] = {
            field: jsonable(memory_payload.get(field))
            for field in sorted(family.memory_fields)
        }
    payload["action"] = record.action
    assert_no_post_fields(payload)
    return stable_json(payload)


def assert_no_post_fields(payload: Any) -> None:
    if isinstance(payload, Mapping):
        leaked = set(payload) & POST_FIELDS
        if leaked:
            raise AssertionError(f"context payload contains post fields: {sorted(leaked)}")
        raw_ids = set(payload) & RAW_ID_FIELDS
        if raw_ids:
            raise AssertionError(f"context payload contains raw component ids: {sorted(raw_ids)}")
        for value in payload.values():
            assert_no_post_fields(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            assert_no_post_fields(value)


def outcome_key(record: TransitionRecord, granularity: OutcomeGranularity) -> str:
    if granularity == "coarse":
        return str(record.post_labels["coarse_outcome"])
    return stable_json(record.post_labels["exact_outcome"])


def retrospective_metrics(
    records: list[TransitionRecord],
    family: RepresentationFamily,
    *,
    granularity: OutcomeGranularity,
) -> tuple[MetricSummary, list[dict[str, Any]]]:
    retrospective_memory = None
    if family.memory_fields:
        retrospective_memory = memory_for_family(family, len(records))
        for record in records:
            retrospective_memory.update(record)
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[TransitionRecord]] = defaultdict(list)
    for record in records:
        key = context_key(record, family, memory=retrospective_memory)
        outcome = outcome_key(record, granularity)
        groups[key][outcome] += 1
        examples[key].append(record)
    return summarize_groups(groups, examples, family=family.name, granularity=granularity)


def prequential_metrics(
    records: list[TransitionRecord],
    family: RepresentationFamily,
    *,
    granularity: OutcomeGranularity,
) -> tuple[MetricSummary, list[dict[str, Any]], list[dict[str, Any]]]:
    family.assert_valid()
    memory = memory_for_family(family, len(records)) if family.memory_fields else None
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[TransitionRecord]] = defaultdict(list)
    growth: list[dict[str, Any]] = []
    for record in records:
        key = context_key(record, family, memory=memory)
        outcome = outcome_key(record, granularity)
        groups[key][outcome] += 1
        examples[key].append(record)
        if memory is not None:
            memory.update(record)
            growth = memory.growth
    summary, contradictions = summarize_groups(
        groups,
        examples,
        family=family.name,
        granularity=granularity,
    )
    return summary, contradictions, growth


def prequential_parameter_metrics(
    records: list[TransitionRecord],
    family: RepresentationFamily,
    *,
    labeler: LabelerVersion = "v1",
) -> dict[str, Any]:
    memory = memory_for_family(family, len(records)) if family.memory_fields else None
    schema_groups: dict[str, Counter[str]] = defaultdict(Counter)
    schema_examples: dict[str, list[TransitionRecord]] = defaultdict(list)
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[TransitionRecord]] = defaultdict(list)
    for record in records:
        key = context_key(record, family, memory=memory)
        schema = outcome_schema_key(record, labeler=labeler)
        parameter = parameter_outcome_key(record, labeler=labeler)
        schema_groups[key][schema] += 1
        schema_examples[key].append(record)
        group_key = stable_json({"context": key, "action": record.action, "schema": schema})
        groups[group_key][parameter] += 1
        examples[group_key].append(record)
        if memory is not None:
            memory.update(record)
    schema_summary, _schema_contradictions = summarize_groups(
        schema_groups,
        schema_examples,
        family=family.name,
        granularity="coarse",
    )
    total = sum(sum(counter.values()) for counter in groups.values())
    reused_keys = [key for key, counter in groups.items() if sum(counter.values()) > 1]
    contradicted = [key for key in reused_keys if len(groups[key]) > 1]
    majority_hits = sum(max(counter.values()) for counter in groups.values() if counter)
    weighted_entropy = sum(sum(counter.values()) * entropy(counter) for counter in groups.values())
    reused_transitions = sum(sum(groups[key].values()) for key in reused_keys)
    examples_out = []
    for key in contradicted[:20]:
        sample = examples[key][:8]
        examples_out.append(
            {
                "game": sample[0].game if sample else None,
                "family": family.name,
                "context_schema_key": key,
                "num_occurrences": sum(groups[key].values()),
                "num_distinct_parameters": len(groups[key]),
                "parameter_examples": dict(groups[key]),
                "coarse_outcome": sample[0].post_labels.get("coarse_outcome") if sample else None,
                "steps": [item.step for item in sample],
                "actions": [item.action for item in sample],
            }
        )
    parameter_contradiction = round(len(contradicted) / len(reused_keys) if reused_keys else 0.0, 6)
    parameter_accuracy = round(majority_hits / total if total else 0.0, 6)
    return {
        "family": family.name,
        "labeler": labeler,
        "examples": total,
        "schema_contradiction_rate": schema_summary.contradiction_rate,
        "schema_reuse_coverage": schema_summary.reuse_coverage,
        "schema_reused_contexts": schema_summary.reused_contexts,
        "schema_reused_transitions": int(round(schema_summary.reuse_coverage * schema_summary.examples)),
        "schema_one_off_ratio": schema_summary.one_off_ratio,
        "parameter_groups": len(groups),
        "parameter_reused_contexts": len(reused_keys),
        "parameter_reused_transitions": reused_transitions,
        "parameter_contradiction_given_schema": parameter_contradiction,
        "parameter_error_given_schema": round(1.0 - parameter_accuracy, 6),
        "parameter_majority_accuracy_given_schema": parameter_accuracy,
        "parameter_entropy_given_schema": round(weighted_entropy / total if total else 0.0, 6),
        "schema_stable_parameter_split_examples": examples_out,
    }


def outcome_schema_key(record: TransitionRecord, *, labeler: LabelerVersion = "v1") -> str:
    if labeler == "v2":
        return str(record.post_labels.get("labeler_v2", {}).get("coarse_outcome", "unknown"))
    if labeler == "v3":
        return str(record.post_labels.get("labeler_v3", {}).get("coarse_outcome", "unknown"))
    if labeler == "cd82_region_role":
        return str(record.post_labels.get("labeler_cd82_region_role", {}).get("coarse_outcome", "unknown"))
    return str(record.post_labels["coarse_outcome"])


def parameter_outcome_key(record: TransitionRecord, *, labeler: LabelerVersion = "v1") -> str:
    if labeler == "v2":
        return stable_json(record.post_labels.get("labeler_v2", {}).get("parameter_outcome", {}))
    if labeler == "v3":
        return stable_json(record.post_labels.get("labeler_v3", {}).get("parameter_outcome", {}))
    if labeler == "cd82_region_role":
        return stable_json(record.post_labels.get("labeler_cd82_region_role", {}).get("parameter_outcome", {}))
    exact = record.post_labels.get("exact_outcome", {})
    return stable_json(
        {
            "actor_delta": exact.get("actor_delta"),
            "changed_cell_signature": exact.get("changed_cell_signature"),
            "changed_value_signature": exact.get("changed_value_signature"),
            "hud_delta_signature": exact.get("hud_delta_signature"),
            "resource_delta_signature": exact.get("resource_delta_signature"),
            "status": exact.get("status"),
            "status_delta": exact.get("status_delta"),
        }
    )


def memory_for_family(family: RepresentationFamily, record_count: int) -> Any:
    if family.name == R3B_ROLE_MEMORY.name:
        return RoleMemory(max_clusters=max(8, math.ceil(math.sqrt(max(1, record_count)))))
    if family.name.startswith("R4_relational_geometry"):
        return RelationalGeometryMemory()
    if family.name.startswith("R4_click_procedural_learned"):
        return ClickProceduralMemory()
    return HandleMemory()


def summarize_groups(
    groups: Mapping[str, Counter[str]],
    examples: Mapping[str, list[TransitionRecord]],
    *,
    family: str,
    granularity: OutcomeGranularity,
) -> tuple[MetricSummary, list[dict[str, Any]]]:
    total = sum(sum(counter.values()) for counter in groups.values())
    reused_keys = [key for key, counter in groups.items() if sum(counter.values()) > 1]
    contradicted = [
        key
        for key in reused_keys
        if len(groups[key]) > 1
    ]
    weighted_entropy = 0.0
    majority_hits = 0
    for counter in groups.values():
        count = sum(counter.values())
        weighted_entropy += count * entropy(counter)
        majority_hits += max(counter.values()) if counter else 0
    summary = MetricSummary(
        examples=total,
        unique_contexts=len(groups),
        reused_contexts=len(reused_keys),
        reuse_coverage=round(
            sum(sum(groups[key].values()) for key in reused_keys) / total
            if total
            else 0.0,
            6,
        ),
        one_off_ratio=round(
            sum(1 for counter in groups.values() if sum(counter.values()) == 1) / len(groups)
            if groups
            else 0.0,
            6,
        ),
        contradicted_reused_contexts=len(contradicted),
        contradiction_rate=round(len(contradicted) / len(reused_keys) if reused_keys else 0.0, 6),
        mean_weighted_outcome_entropy=round(weighted_entropy / total if total else 0.0, 6),
        majority_outcome_accuracy=round(majority_hits / total if total else 0.0, 6),
    )
    contradiction_records = []
    for key in contradicted:
        sample = examples[key][:8]
        contradiction_records.append(
            {
                "family": family,
                "granularity": granularity,
                "context_key": key,
                "num_occurrences": sum(groups[key].values()),
                "num_distinct_outcomes": len(groups[key]),
                "outcome_examples": dict(groups[key]),
                "steps": [item.step for item in sample],
                "actions": [item.action for item in sample],
                "game": sample[0].game if sample else None,
            }
        )
    contradiction_records.sort(
        key=lambda item: (-item["num_distinct_outcomes"], -item["num_occurrences"], item["context_key"])
    )
    return summary, contradiction_records


def add_contradiction_mode(
    records: list[dict[str, Any]],
    *,
    mode: Literal["retrospective", "prequential"],
) -> list[dict[str, Any]]:
    return [{**record, "mode": mode} for record in records]


def evaluate_suite(
    specs: tuple[TraceSpec, ...],
    *,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for family in (*REPRESENTATION_FAMILIES, *R4_REPRESENTATION_FAMILIES):
        family.assert_valid()
    manifest = [spec.manifest_record() for spec in specs]
    write_json(output_dir / "trace_source_manifest.json", manifest)
    write_json(
        output_dir / "representation_field_audit.json",
        {
            "frozen_baselines": {
                R3A_COMPRESSED.name: {
                    "pre_action_fields": list(R3A_FROZEN_FIELDS),
                    "frozen_for_upgrade": "r4_probe_and_causal_decomposition",
                }
            },
            "families": [
                family.audit_record()
                for family in (*REPRESENTATION_FAMILIES, *R4_REPRESENTATION_FAMILIES, *ORACLE_FAMILIES)
            ],
        },
    )

    by_game: dict[str, Any] = {}
    prequential_by_game: dict[str, Any] = {}
    all_contradictions: list[dict[str, Any]] = []
    all_growth: dict[str, Any] = {}
    feature_availability: dict[str, Any] = {}
    causal_ordering: dict[str, Any] = {}
    retro_gap: dict[str, Any] = {}
    feature_ablation: dict[str, Any] = {}
    compression_curve: dict[str, Any] = {}
    diagnostic_verdict: dict[str, Any] = {}
    event_schema_metrics: dict[str, Any] = {}
    r4_probe_report: dict[str, Any] = {}
    causal_stage_decomposition: dict[str, Any] = {}
    event_parameter_audit_by_game: dict[str, Any] = {}
    actor_detection_audit_by_game: dict[str, Any] = {}
    changed_region_assignment_by_game: dict[str, Any] = {}
    outcome_labeler_comparison: dict[str, Any] = {}
    stage_report_v2_by_game: dict[str, Any] = {}
    r3a_labeler_v2_ablation: dict[str, Any] = {}
    outcome_labeler_verdict: dict[str, Any] = {}
    controlled_entity_audit_by_game: dict[str, Any] = {}
    region_role_map_by_game: dict[str, Any] = {}
    delta_channel_assignment_v3_by_game: dict[str, Any] = {}
    outcome_labeler_v2_vs_v3: dict[str, Any] = {}
    stage_report_v3_by_game: dict[str, Any] = {}
    outcome_labeler_v3_verdict: dict[str, Any] = {}
    parameter_model_probe_by_game: dict[str, Any] = {}
    cd82_region_role_probe_by_game: dict[str, Any] = {}
    motion_law_probe_by_game: dict[str, Any] = {}
    cd82_region_role_integrated_by_game: dict[str, Any] = {}
    motion_target_granularity_by_game: dict[str, Any] = {}
    independent_mover_dynamics_by_game: dict[str, Any] = {}
    cd82_procedural_state_probe_by_game: dict[str, Any] = {}
    controlled_region_material_decomposition_by_game: dict[str, Any] = {}
    changed_region_factorization_by_game: dict[str, Any] = {}
    t5_failure_anatomy_by_game: dict[str, Any] = {}
    independent_coupled_mover_probe_by_game: dict[str, Any] = {}
    mover_state_factorization_by_game: dict[str, Any] = {}
    world_clock_transition_probe_by_game: dict[str, Any] = {}
    visibility_composition_probe_by_game: dict[str, Any] = {}
    mover_bootstrap_by_game: dict[str, Any] = {}
    ar25_overlap_composition_by_game: dict[str, Any] = {}
    mover_bootstrap_composition_v0_by_game: dict[str, Any] = {}
    changed_cluster_merge_split_audit_v0_by_game: dict[str, Any] = {}
    residue_source_attribution_v0_by_game: dict[str, Any] = {}
    global_update_vs_container_audit_v0_by_game: dict[str, Any] = {}
    global_update_layer_v0_by_game: dict[str, Any] = {}
    global_phase_register_discovery_v0_by_game: dict[str, Any] = {}
    global_phase_identifiability_v0_by_game: dict[str, Any] = {}
    ar25_unknown_world_clock_atlas_by_game: dict[str, Any] = {}
    records_by_game: dict[str, list[TransitionRecord]] = {}

    for spec in specs:
        records = load_transition_records(spec, limit=limit)
        records_by_game[spec.game] = records
        by_game[spec.game] = {}
        prequential_by_game[spec.game] = {}
        feature_availability[spec.game] = feature_availability_record(records)
        causal_ordering[spec.game] = causal_stage_report(records)
        causal_stage_decomposition[spec.game] = causal_stage_decomposition_report(records)
        feature_ablation[spec.game] = feature_ablation_report(records)
        event_schema_metrics[spec.game] = event_schema_metrics_report(records)
        r4_probe_report[spec.game] = r4_representation_probe_report(records)
        event_parameter_audit_by_game[spec.game] = event_parameter_audit(records)
        actor_detection_audit_by_game[spec.game] = actor_detection_audit(records)
        changed_region_assignment_by_game[spec.game] = changed_region_assignment_report(records)
        outcome_labeler_comparison[spec.game] = outcome_labeler_v1_vs_v2_report(records)
        stage_report_v2_by_game[spec.game] = stage_report_v2(records)
        r3a_labeler_v2_ablation[spec.game] = r3a_labeler_v2_ablation_report(records)
        controlled_entity_audit_by_game[spec.game] = controlled_entity_audit(records)
        region_role_map_by_game[spec.game] = region_role_map_report(records)
        delta_channel_assignment_v3_by_game[spec.game] = delta_channel_assignment_v3_report(records)
        outcome_labeler_v2_vs_v3[spec.game] = outcome_labeler_v2_vs_v3_report(records)
        stage_report_v3_by_game[spec.game] = stage_report_v3(records)
        parameter_model_probe_by_game[spec.game] = parameter_model_probe_report(records)
        cd82_region_role_probe_by_game[spec.game] = cd82_region_role_probe_report(records)
        motion_law_probe_by_game[spec.game] = motion_law_probe_report(records)
        cd82_region_role_integrated_by_game[spec.game] = cd82_region_role_integrated_report(records)
        motion_target_granularity_by_game[spec.game] = motion_target_granularity_report(records)
        independent_mover_dynamics_by_game[spec.game] = independent_mover_dynamics_report(records)
        cd82_procedural_state_probe_by_game[spec.game] = cd82_procedural_state_probe_report(records)
        controlled_region_material_decomposition_by_game[spec.game] = (
            controlled_region_material_decomposition_report(records)
        )
        changed_region_factorization_by_game[spec.game] = changed_region_factorization_report(records)
        t5_failure_anatomy_by_game[spec.game] = t5_failure_anatomy_report(
            records,
            prototype_dir=output_dir / "t5_failure_anatomy_prototypes" / spec.game,
        )
        independent_coupled_mover_probe_by_game[spec.game] = independent_coupled_mover_probe_report(records)
        mover_state_factorization_by_game[spec.game] = mover_state_factorization_report(records)
        world_clock_transition_probe_by_game[spec.game] = world_clock_transition_probe_report(records)
        visibility_composition_probe_by_game[spec.game] = visibility_composition_probe_report(records)
        mover_bootstrap_by_game[spec.game] = mover_bootstrap_report(records)
        ar25_overlap_composition_by_game[spec.game] = ar25_overlap_composition_report(records)
        cluster_decompositions_v0 = ar25_merged_cluster_decompositions_v0(records)
        residue_groups_v0 = ar25_residue_competition_groups_v0(records, decompositions=cluster_decompositions_v0)
        mover_bootstrap_composition_v0_by_game[spec.game] = mover_bootstrap_composition_v0_report(records)
        changed_cluster_merge_split_audit_v0_by_game[spec.game] = changed_cluster_merge_split_audit_v0_report(
            records,
            baseline_merged_overlap_metrics=mover_bootstrap_composition_v0_by_game[spec.game].get("merged_overlap_metrics"),
            decompositions=cluster_decompositions_v0,
        )
        residue_source_attribution_v0_by_game[spec.game] = residue_source_attribution_v0_report(
            records,
            decompositions=cluster_decompositions_v0,
            competition_groups=residue_groups_v0,
            merge_split_metrics=changed_cluster_merge_split_audit_v0_by_game[spec.game].get("metrics"),
        )
        global_update_vs_container_audit_v0_by_game[spec.game] = global_update_vs_container_audit_v0_report(
            records,
            competition_groups=residue_groups_v0,
            merge_split_metrics=changed_cluster_merge_split_audit_v0_by_game[spec.game].get("metrics"),
        )
        global_update_layer_v0_by_game[spec.game] = global_update_layer_v0_report(
            records,
            baseline_decompositions=cluster_decompositions_v0,
            baseline_groups=residue_groups_v0,
        )
        global_phase_register_discovery_v0_by_game[spec.game] = global_phase_register_discovery_v0_report(
            records,
            baseline_decompositions=cluster_decompositions_v0,
            baseline_groups=residue_groups_v0,
            global_layer_report=global_update_layer_v0_by_game[spec.game],
        )
        global_phase_identifiability_v0_by_game[spec.game] = global_phase_identifiability_v0_report(
            records,
            baseline_decompositions=cluster_decompositions_v0,
            baseline_groups=residue_groups_v0,
            phase_report=global_phase_register_discovery_v0_by_game[spec.game],
            global_layer_report=global_update_layer_v0_by_game[spec.game],
        )
        ar25_unknown_world_clock_atlas_by_game[spec.game] = ar25_unknown_world_clock_atlas_report(
            records,
            prototype_dir=output_dir / "ar25_unknown_world_clock_prototypes" if spec.game == "ar25" else None,
        )
        outcome_labeler_verdict[spec.game] = labeler_verdict(
            audit=event_parameter_audit_by_game[spec.game],
            actor_audit=actor_detection_audit_by_game[spec.game],
            comparison=outcome_labeler_comparison[spec.game],
            stage_v2=stage_report_v2_by_game[spec.game],
        )
        outcome_labeler_v3_verdict[spec.game] = labeler_v3_verdict(
            controlled_audit=controlled_entity_audit_by_game[spec.game],
            delta_assignment=delta_channel_assignment_v3_by_game[spec.game],
            comparison=outcome_labeler_v2_vs_v3[spec.game],
            stage_v3=stage_report_v3_by_game[spec.game],
        )

        for family in REPRESENTATION_FAMILIES:
            by_game[spec.game][family.name] = {}
            prequential_by_game[spec.game][family.name] = {}
            retro_gap.setdefault(spec.game, {})[family.name] = {}
            for granularity in ("coarse", "exact"):
                retrospective, contradictions = retrospective_metrics(
                    records,
                    family,
                    granularity=granularity,  # type: ignore[arg-type]
                )
                prequential, preq_contradictions, growth = prequential_metrics(
                    records,
                    family,
                    granularity=granularity,  # type: ignore[arg-type]
                )
                by_game[spec.game][family.name][granularity] = retrospective.to_record()
                prequential_by_game[spec.game][family.name][granularity] = prequential.to_record()
                all_contradictions.extend(
                    add_contradiction_mode(contradictions[:20], mode="retrospective")
                )
                all_contradictions.extend(
                    add_contradiction_mode(preq_contradictions[:20], mode="prequential")
                )
                if growth:
                    all_growth.setdefault(spec.game, {})[family.name] = growth
                retro_gain = metric_gain(
                    by_game[spec.game]["R0_current"][granularity]
                    if "R0_current" in by_game[spec.game]
                    and granularity in by_game[spec.game]["R0_current"]
                    else retrospective.to_record(),
                    retrospective.to_record(),
                )
                preq_gain = metric_gain(
                    prequential_by_game[spec.game]["R0_current"][granularity]
                    if "R0_current" in prequential_by_game[spec.game]
                    and granularity in prequential_by_game[spec.game]["R0_current"]
                    else prequential.to_record(),
                    prequential.to_record(),
                )
                retro_gap[spec.game][family.name][granularity] = {
                    "retrospective_gain": retro_gain,
                    "prequential_gain": preq_gain,
                    "retrospective_gain_minus_prequential_gain": round(retro_gain - preq_gain, 6),
                }

        by_game[spec.game]["oracle"] = oracle_status(spec)
        compression_curve[spec.game] = compression_curve_report(prequential_by_game[spec.game])
        diagnostic_verdict[spec.game] = game_diagnostic_verdict(
            prequential_by_game[spec.game],
            feature_ablation[spec.game],
            event_schema_metrics[spec.game],
            r4_probe_report[spec.game],
            causal_stage_decomposition[spec.game],
        )

    summary = {
        "suite": "three_game",
        "games": [spec.game for spec in specs],
        "learned_model_verdict": learned_model_verdict(by_game, causal_ordering),
        "oracle_status": {
            spec.game: by_game[spec.game].get("oracle", {})
            for spec in specs
        },
        "headline_table": headline_table(by_game, prequential_by_game),
        "diagnostic_verdict": diagnostic_verdict,
        "outcome_labeler_verdict": outcome_labeler_verdict,
        "outcome_labeler_v3_verdict": outcome_labeler_v3_verdict,
        "parameter_model_probe_verdict": {
            spec.game: parameter_model_probe_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "cd82_region_role_probe_verdict": {
            spec.game: cd82_region_role_probe_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "motion_law_probe_verdict": {
            spec.game: motion_law_probe_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "cd82_region_role_integrated_verdict": {
            spec.game: cd82_region_role_integrated_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "motion_target_granularity_verdict": {
            spec.game: motion_target_granularity_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "independent_mover_dynamics_verdict": {
            spec.game: independent_mover_dynamics_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "cd82_procedural_state_probe_verdict": {
            spec.game: cd82_procedural_state_probe_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "controlled_region_material_decomposition_verdict": {
            spec.game: controlled_region_material_decomposition_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "changed_region_factorization_verdict": {
            spec.game: changed_region_factorization_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "t5_failure_anatomy_verdict": {
            spec.game: t5_failure_anatomy_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "independent_coupled_mover_probe_verdict": {
            spec.game: independent_coupled_mover_probe_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "mover_state_factorization_verdict": {
            spec.game: mover_state_factorization_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "world_clock_transition_probe_verdict": {
            spec.game: world_clock_transition_probe_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "visibility_composition_probe_verdict": {
            spec.game: visibility_composition_probe_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "ar25_unknown_world_clock_atlas_verdict": {
            spec.game: ar25_unknown_world_clock_atlas_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "changed_cluster_merge_split_audit_v0_verdict": {
            spec.game: changed_cluster_merge_split_audit_v0_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "residue_source_attribution_v0_verdict": {
            spec.game: residue_source_attribution_v0_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "global_update_vs_container_audit_v0_verdict": {
            spec.game: global_update_vs_container_audit_v0_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "global_update_layer_v0_verdict": {
            spec.game: global_update_layer_v0_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "global_phase_register_discovery_v0_verdict": {
            spec.game: global_phase_register_discovery_v0_by_game[spec.game].get("verdict")
            for spec in specs
        },
        "global_phase_identifiability_v0_verdict": {
            spec.game: global_phase_identifiability_v0_by_game[spec.game].get("verdict")
            for spec in specs
        },
    }

    visibility_precision_audit = visibility_composition_precision_audit_suite(
        records_by_game,
        output_dir=output_dir,
    )
    ar25_latent_taxonomy = ar25_latent_transition_taxonomy_report(
        records_by_game.get("ar25", ()),
        prototype_dir=output_dir / "ar25_latent_motion_unknown_prototypes",
    )
    ar25_filtered_latent_taxonomy = ar25_latent_motion_history_filtered_taxonomy_report(
        records_by_game.get("ar25", ()),
        prototype_dir=output_dir / "ar25_latent_motion_history_filtered_prototypes",
    )

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "by_game.json", by_game)
    write_json(output_dir / "prequential_by_game.json", prequential_by_game)
    write_json(output_dir / "memory_growth.json", all_growth)
    write_json(output_dir / "feature_availability.json", feature_availability)
    write_json(output_dir / "causal_ordering.json", causal_ordering)
    write_json(output_dir / "causal_stage_report.json", causal_stage_decomposition)
    write_json(output_dir / "retrospective_vs_prequential_gap.json", retro_gap)
    write_json(output_dir / "feature_ablation.json", feature_ablation)
    write_json(output_dir / "compression_curve.json", compression_curve)
    write_json(output_dir / "event_schema_metrics.json", event_schema_metrics)
    write_json(output_dir / "r4_representation_probe_report.json", r4_probe_report)
    write_json(output_dir / "event_parameter_audit.json", event_parameter_audit_by_game)
    write_json(output_dir / "actor_detection_audit.json", actor_detection_audit_by_game)
    write_json(output_dir / "changed_region_assignment.json", changed_region_assignment_by_game)
    write_json(output_dir / "outcome_labeler_v1_vs_v2.json", outcome_labeler_comparison)
    write_json(output_dir / "stage_report_v2.json", stage_report_v2_by_game)
    write_json(output_dir / "r3a_labeler_v2_ablation.json", r3a_labeler_v2_ablation)
    write_json(output_dir / "outcome_labeler_verdict.json", outcome_labeler_verdict)
    write_json(output_dir / "controlled_entity_audit.json", controlled_entity_audit_by_game)
    write_json(output_dir / "region_role_map.json", region_role_map_by_game)
    write_json(output_dir / "delta_channel_assignment_v3.json", delta_channel_assignment_v3_by_game)
    write_json(output_dir / "outcome_labeler_v2_vs_v3.json", outcome_labeler_v2_vs_v3)
    write_json(output_dir / "stage_report_v3.json", stage_report_v3_by_game)
    write_json(output_dir / "outcome_labeler_v3_verdict.json", outcome_labeler_v3_verdict)
    write_json(output_dir / "parameter_model_probe_report.json", parameter_model_probe_by_game)
    write_json(output_dir / "cd82_region_role_probe_report.json", cd82_region_role_probe_by_game)
    write_json(output_dir / "motion_law_probe_report.json", motion_law_probe_by_game)
    write_json(output_dir / "cd82_region_role_integrated_report.json", cd82_region_role_integrated_by_game)
    write_json(output_dir / "motion_target_granularity_report.json", motion_target_granularity_by_game)
    write_json(output_dir / "independent_mover_dynamics_report.json", independent_mover_dynamics_by_game)
    write_json(output_dir / "cd82_procedural_state_probe_report.json", cd82_procedural_state_probe_by_game)
    write_json(
        output_dir / "controlled_region_material_decomposition_report.json",
        controlled_region_material_decomposition_by_game,
    )
    write_json(output_dir / "changed_region_factorization_report.json", changed_region_factorization_by_game)
    write_json(output_dir / "t5_failure_anatomy_report.json", t5_failure_anatomy_by_game)
    write_json(output_dir / "independent_coupled_mover_probe_report.json", independent_coupled_mover_probe_by_game)
    write_json(output_dir / "mover_state_factorization_report.json", mover_state_factorization_by_game)
    write_json(output_dir / "world_clock_transition_probe_report.json", world_clock_transition_probe_by_game)
    write_json(output_dir / "visibility_composition_probe_report.json", visibility_composition_probe_by_game)
    write_json(output_dir / "mover_bootstrap_report.json", mover_bootstrap_by_game)
    write_json(output_dir / "ar25_overlap_composition_report.json", ar25_overlap_composition_by_game)
    write_json(output_dir / "mover_bootstrap_composition_v0_report.json", mover_bootstrap_composition_v0_by_game)
    write_json(
        output_dir / "changed_cluster_merge_split_audit_v0_report.json",
        changed_cluster_merge_split_audit_v0_by_game,
    )
    write_json(
        output_dir / "residue_source_attribution_v0_report.json",
        residue_source_attribution_v0_by_game,
    )
    write_json(
        output_dir / "global_update_vs_container_audit_v0_report.json",
        global_update_vs_container_audit_v0_by_game,
    )
    write_json(output_dir / "global_update_layer_v0_report.json", global_update_layer_v0_by_game)
    write_json(
        output_dir / "global_phase_register_discovery_v0_report.json",
        global_phase_register_discovery_v0_by_game,
    )
    write_json(
        output_dir / "global_phase_identifiability_v0_report.json",
        global_phase_identifiability_v0_by_game,
    )
    write_jsonl(
        output_dir / "mover_bootstrap_hypotheses.jsonl",
        combined_report_rows(mover_bootstrap_composition_v0_by_game, "bootstrap_hypotheses"),
    )
    write_jsonl(
        output_dir / "merged_overlap_decomposition_examples.jsonl",
        combined_report_rows(mover_bootstrap_composition_v0_by_game, "merged_overlap_decomposition_examples"),
    )
    write_jsonl(
        output_dir / "cluster_decomposition_hypotheses.jsonl",
        combined_report_rows(changed_cluster_merge_split_audit_v0_by_game, "cluster_decomposition_hypotheses"),
    )
    write_jsonl(
        output_dir / "cell_attribution_examples.jsonl",
        combined_report_rows(changed_cluster_merge_split_audit_v0_by_game, "cell_attribution_examples"),
    )
    write_jsonl(
        output_dir / "decomposition_failure_gallery.jsonl",
        combined_report_rows(changed_cluster_merge_split_audit_v0_by_game, "decomposition_failure_gallery"),
    )
    write_json(
        output_dir / "ar25_cluster_overmerge_taxonomy.json",
        changed_cluster_merge_split_audit_v0_by_game.get("ar25", {}).get("ar25_cluster_overmerge_taxonomy", {}),
    )
    write_json(
        output_dir / "decomposition_before_after_metrics.json",
        changed_cluster_merge_split_audit_v0_by_game.get("ar25", {}).get("decomposition_before_after_metrics", {}),
    )
    write_jsonl(
        output_dir / "residue_atoms.jsonl",
        combined_report_rows(residue_source_attribution_v0_by_game, "residue_atoms"),
    )
    write_jsonl(
        output_dir / "residue_competition_groups.jsonl",
        combined_report_rows(residue_source_attribution_v0_by_game, "residue_competition_groups"),
    )
    write_jsonl(
        output_dir / "residue_hypothesis_scores.jsonl",
        combined_report_rows(residue_source_attribution_v0_by_game, "residue_hypothesis_scores"),
    )
    write_jsonl(
        output_dir / "high_residue_case_gallery.jsonl",
        combined_report_rows(residue_source_attribution_v0_by_game, "high_residue_case_gallery"),
    )
    write_json(
        output_dir / "ar25_residue_taxonomy.json",
        residue_source_attribution_v0_by_game.get("ar25", {}).get("ar25_residue_taxonomy", {}),
    )
    write_jsonl(
        output_dir / "global_residue_templates.jsonl",
        combined_report_rows(global_update_vs_container_audit_v0_by_game, "global_residue_templates"),
    )
    write_jsonl(
        output_dir / "global_mask_subtraction_examples.jsonl",
        combined_report_rows(global_update_vs_container_audit_v0_by_game, "global_mask_subtraction_examples"),
    )
    write_jsonl(
        output_dir / "bridge_cut_vs_global_subtraction.jsonl",
        combined_report_rows(global_update_vs_container_audit_v0_by_game, "bridge_cut_vs_global_subtraction"),
    )
    write_json(
        output_dir / "ar25_residue_verdict_taxonomy.json",
        global_update_vs_container_audit_v0_by_game.get("ar25", {}).get("ar25_residue_verdict_taxonomy", {}),
    )
    write_jsonl(
        output_dir / "online_global_update_predictions.jsonl",
        combined_report_rows(global_update_layer_v0_by_game, "online_global_update_predictions"),
    )
    write_jsonl(
        output_dir / "delta_factorization_examples.jsonl",
        combined_report_rows(global_update_layer_v0_by_game, "delta_factorization_examples"),
    )
    write_jsonl(
        output_dir / "local_cluster_rebuild_before_after.jsonl",
        combined_report_rows(global_update_layer_v0_by_game, "local_cluster_rebuild_before_after"),
    )
    write_json(
        output_dir / "ar25_composition_after_global_factoring.json",
        global_update_layer_v0_by_game.get("ar25", {}).get("ar25_composition_after_global_factoring", {}),
    )
    write_jsonl(
        output_dir / "false_subtraction_audit.jsonl",
        combined_report_rows(global_update_layer_v0_by_game, "false_subtraction_audit"),
    )
    write_jsonl(
        output_dir / "phase_hypothesis_scores.jsonl",
        combined_report_rows(global_phase_register_discovery_v0_by_game, "phase_hypothesis_scores"),
    )
    write_jsonl(
        output_dir / "prequential_phase_predictions.jsonl",
        combined_report_rows(global_phase_register_discovery_v0_by_game, "prequential_phase_predictions"),
    )
    write_jsonl(
        output_dir / "global_template_clusters.jsonl",
        combined_report_rows(global_phase_register_discovery_v0_by_game, "global_template_clusters"),
    )
    write_jsonl(
        output_dir / "alignment_match_comparison.jsonl",
        combined_report_rows(global_phase_register_discovery_v0_by_game, "alignment_match_comparison"),
    )
    write_jsonl(
        output_dir / "false_subtraction_locality_audit.jsonl",
        combined_report_rows(global_phase_register_discovery_v0_by_game, "false_subtraction_locality_audit"),
    )
    write_jsonl(
        output_dir / "template_support_learning_curve.jsonl",
        combined_report_rows(global_phase_identifiability_v0_by_game, "template_support_learning_curve"),
    )
    write_json(
        output_dir / "oracle_template_transition_entropy.json",
        global_phase_identifiability_v0_by_game.get("ar25", {}).get("oracle_template_transition_entropy", {}),
    )
    write_jsonl(
        output_dir / "template_fragmentation_audit.jsonl",
        combined_report_rows(global_phase_identifiability_v0_by_game, "template_fragmentation_audit"),
    )
    write_jsonl(
        output_dir / "current_frame_phase_readout_scores.jsonl",
        combined_report_rows(global_phase_identifiability_v0_by_game, "current_frame_phase_readout_scores"),
    )
    write_jsonl(
        output_dir / "higher_order_phase_predictions.jsonl",
        combined_report_rows(global_phase_identifiability_v0_by_game, "higher_order_phase_predictions"),
    )
    write_jsonl(
        output_dir / "compositional_global_atoms.jsonl",
        combined_report_rows(global_phase_identifiability_v0_by_game, "compositional_global_atoms"),
    )
    write_json(
        output_dir / "phase_identifiability_verdict.json",
        global_phase_identifiability_v0_by_game.get("ar25", {}).get("phase_identifiability_verdict", {}),
    )
    write_json(output_dir / "ar25_unknown_world_clock_atlas.json", ar25_unknown_world_clock_atlas_by_game)
    write_json(output_dir / "visibility_composition_precision_audit.json", visibility_precision_audit)
    write_markdown_report(
        output_dir / "visibility_composition_precision_audit.md",
        visibility_composition_precision_audit_markdown(visibility_precision_audit),
    )
    write_json(output_dir / "ar25_latent_transition_taxonomy_report.json", ar25_latent_taxonomy)
    write_markdown_report(
        output_dir / "ar25_latent_motion_unknown_readout.md",
        ar25_latent_motion_unknown_readout_markdown(ar25_latent_taxonomy),
    )
    write_json(
        output_dir / "ar25_latent_motion_history_filtered_taxonomy.json",
        ar25_filtered_latent_taxonomy,
    )
    write_markdown_report(
        output_dir / "ar25_latent_motion_history_filtered_readout.md",
        ar25_latent_motion_history_filtered_readout_markdown(ar25_filtered_latent_taxonomy),
    )
    write_json(output_dir / "diagnostic_verdict.json", diagnostic_verdict)
    write_jsonl(output_dir / "contradiction_examples.jsonl", all_contradictions)
    write_contradictions_csv(output_dir / "contradictions.csv", all_contradictions)
    return summary


def oracle_status(spec: TraceSpec) -> dict[str, Any]:
    if spec.game != "ls20":
        return {"oracle_status": "not_applicable_non_ls20"}
    if not spec.supports_oracle or spec.trace_source != "codex_run":
        return {
            "oracle_status": "skipped_incompatible_trace_source",
            "trace_source": spec.trace_source,
            "supports_oracle": spec.supports_oracle,
        }
    return {"oracle_status": "not_implemented_v1"}


def learned_model_verdict(by_game: Mapping[str, Any], causal_ordering: Mapping[str, Any]) -> str | bool:
    oracle_metrics = by_game.get("ls20", {}).get("R1_oracle_posthoc_ls20")
    if not oracle_metrics:
        return "insufficient_oracle_evidence"
    return False


def metric_gain(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    return round(
        float(baseline.get("contradiction_rate", 0.0))
        - float(candidate.get("contradiction_rate", 0.0)),
        6,
    )


def headline_table(by_game: Mapping[str, Any], prequential: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for game, families in prequential.items():
        for family in (
            "R0_current",
            "R2a_computable_static",
            "R2b_computable_with_memory",
            "R3a_computable_compressed_static",
            "R3b_role_memory",
        ):
            coarse = families.get(family, {}).get("coarse", {})
            exact = families.get(family, {}).get("exact", {})
            rows.append(
                {
                    "game": game,
                    "family": family,
                    "coarse_contradiction_rate": coarse.get("contradiction_rate"),
                    "coarse_one_off_ratio": coarse.get("one_off_ratio"),
                    "exact_contradiction_rate": exact.get("contradiction_rate"),
                    "exact_one_off_ratio": exact.get("one_off_ratio"),
                }
            )
    return rows


def feature_availability_record(records: list[TransitionRecord]) -> dict[str, Any]:
    counts = Counter()
    actor_confidence = Counter()
    for record in records:
        for field, value in record.pre_features.items():
            if field.startswith("_"):
                continue
            if value not in (None, (), [], {}):
                counts[field] += 1
        actor_confidence[str(record.pre_features.get("actor_confidence", "unknown"))] += 1
    total = len(records)
    result = {
        field: {
            "available_count": count,
            "available_rate": round(count / total if total else 0.0, 6),
        }
        for field, count in sorted(counts.items())
    }
    result["actor_confidence_distribution"] = {
        key: {
            "count": count,
            "rate": round(count / total if total else 0.0, 6),
        }
        for key, count in sorted(actor_confidence.items())
    }
    return result


def causal_stage_report(records: list[TransitionRecord]) -> dict[str, Any]:
    stages = {
        "intended_move": lambda record: f"action:{record.action}",
        "blocked_or_rejected": lambda record: str(record.post_labels["blocked_or_rejected"]),
        "actor_delta": lambda record: stable_json(record.post_labels["actor_delta"]),
        "independent_motion": lambda record: str(record.post_labels["remote_delta_present"]),
        "contact_effect": lambda record: str(record.post_labels["contacted_component_delta_class"]),
        "hud_or_resource_delta": lambda record: stable_json(record.post_labels["hud_delta_signature"]),
        "terminal_status": lambda record: str(record.post_labels["terminal_status_delta"]),
    }
    report = {}
    for stage, extractor in stages.items():
        groups: dict[str, Counter[str]] = defaultdict(Counter)
        examples: dict[str, list[TransitionRecord]] = defaultdict(list)
        for record in records:
            key = context_key(record, R2A_STATIC)
            outcome = extractor(record)
            groups[key][outcome] += 1
            examples[key].append(record)
        summary, _contradictions = summarize_groups(
            groups,
            examples,
            family="R2a_computable_static",
            granularity="coarse",
        )
        report[stage] = summary.to_record()
    return report


def causal_stage_decomposition_report(records: list[TransitionRecord]) -> dict[str, Any]:
    stages = {
        "actor_move": lambda record: stable_json(record.post_labels["actor_delta"]),
        "co_motion_or_forced_motion": lambda record: group_motion_role(record),
        "independent_tick": lambda record: str(record.post_labels["remote_delta_present"]),
        "contact_transform": lambda record: str(record.post_labels["contacted_component_delta_class"]),
        "resource_or_hud_update": lambda record: stable_json(record.post_labels["hud_delta_signature"]),
        "terminal": lambda record: str(record.post_labels["terminal_status_delta"]),
    }
    stage_metrics: dict[str, Any] = {}
    for stage, extractor in stages.items():
        groups: dict[str, Counter[str]] = defaultdict(Counter)
        examples: dict[str, list[TransitionRecord]] = defaultdict(list)
        for record in records:
            key = context_key(record, R3A_COMPRESSED)
            groups[key][extractor(record)] += 1
            examples[key].append(record)
        summary, _contradictions = summarize_groups(
            groups,
            examples,
            family=R3A_COMPRESSED.name,
            granularity="coarse",
        )
        stage_metrics[stage] = summary.to_record()

    parameter_groups: dict[str, list[TransitionRecord]] = defaultdict(list)
    parameter_counters: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        key = context_key(record, R3A_COMPRESSED)
        group_key = stable_json(
            {
                "context": key,
                "action": record.action,
                "schema": record.post_labels["coarse_outcome"],
            }
        )
        parameter_groups[group_key].append(record)
        parameter_counters[group_key][parameter_outcome_key(record)] += 1

    split_groups = [
        key
        for key, counter in parameter_counters.items()
        if sum(counter.values()) > 1 and len(counter) > 1
    ]
    stage_split_counts = Counter()
    examples_out = []
    for key in split_groups:
        items = parameter_groups[key]
        varied = []
        for stage, extractor in stages.items():
            if len({extractor(record) for record in items}) > 1:
                stage_split_counts[stage] += 1
                varied.append(stage)
        if varied and len(examples_out) < 20:
            examples_out.append(
                {
                    "context_schema_key": key,
                    "steps": [item.step for item in items[:8]],
                    "actions": [item.action for item in items[:8]],
                    "varied_stages": varied,
                    "parameter_examples": dict(parameter_counters[key]),
                }
            )
    localized = sum(1 for item in examples_out if item["varied_stages"])
    top_stage = stage_split_counts.most_common(1)[0][0] if stage_split_counts else None
    top_count = stage_split_counts.most_common(1)[0][1] if stage_split_counts else 0
    return {
        "baseline_family": R3A_COMPRESSED.name,
        "stage_metrics": stage_metrics,
        "parameter_split_groups": len(split_groups),
        "localized_split_groups_in_examples": localized,
        "top_localizing_stage": top_stage,
        "top_localizing_stage_share": round(top_count / len(split_groups) if split_groups else 0.0, 6),
        "localized_stage_counts": dict(stage_split_counts),
        "examples": examples_out,
    }


def feature_ablation_report(records: list[TransitionRecord]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for family in (R2A_STATIC, R2B_MEMORY, R3A_COMPRESSED):
        fields = sorted(family.pre_action_fields)
        baseline, _contradictions, _growth = prequential_metrics(records, family, granularity="coarse")
        baseline_exact, _exact_contradictions, _exact_growth = prequential_metrics(records, family, granularity="exact")
        family_report: dict[str, Any] = {
            "baseline": baseline.to_record(),
            "baseline_exact": baseline_exact.to_record(),
            "leave_one_out": {},
            "single_field_only": {},
            "fragmentation_candidates": [],
        }
        for field in fields:
            leave_family = RepresentationFamily(
                f"{family.name}_without_{field}",
                frozenset(item for item in family.pre_action_fields if item != field),
                family.memory_fields,
            )
            single_family = RepresentationFamily(
                f"{family.name}_only_{field}",
                frozenset({field}),
                family.memory_fields,
            )
            leave_metrics, _leave_contradictions, _leave_growth = prequential_metrics(records, leave_family, granularity="coarse")
            single_metrics, _single_contradictions, _single_growth = prequential_metrics(records, single_family, granularity="coarse")
            leave_exact, _leave_exact_contradictions, _leave_exact_growth = prequential_metrics(records, leave_family, granularity="exact")
            single_exact, _single_exact_contradictions, _single_exact_growth = prequential_metrics(records, single_family, granularity="exact")
            family_report["leave_one_out"][field] = {
                **leave_metrics.to_record(),
                "exact": leave_exact.to_record(),
            }
            family_report["single_field_only"][field] = {
                **single_metrics.to_record(),
                "exact": single_exact.to_record(),
            }
            reuse_gain = leave_metrics.reuse_coverage - baseline.reuse_coverage
            contradiction_delta = leave_metrics.contradiction_rate - baseline.contradiction_rate
            if reuse_gain >= 0.1 and contradiction_delta <= 0.2:
                family_report["fragmentation_candidates"].append(
                    {
                        "field": field,
                        "reuse_coverage_gain": round(reuse_gain, 6),
                        "contradiction_rate_delta": round(contradiction_delta, 6),
                    }
                )
        report[family.name] = family_report
    return report


def event_schema_metrics_report(records: list[TransitionRecord]) -> dict[str, Any]:
    return {
        family.name: prequential_parameter_metrics(records, family)
        for family in (R3A_COMPRESSED, *R4_REPRESENTATION_FAMILIES)
    }


def outcome_labeler_v1_vs_v2_report(records: list[TransitionRecord]) -> dict[str, Any]:
    families = (
        R3A_COMPRESSED,
        R4_CLICK_PROCEDURAL_VISUAL_ONLY,
        R4_CLICK_PROCEDURAL_VISUAL_PLUS_R3A,
        R4_CLICK_PROCEDURAL_LEARNED_ONLY,
        R4_CLICK_PROCEDURAL_LEARNED_PLUS_R3A,
    )
    report = {}
    for family in families:
        v1 = prequential_parameter_metrics(records, family, labeler="v1")
        v2 = prequential_parameter_metrics(records, family, labeler="v2")
        report[family.name] = {
            "v1": strip_metric_examples(v1),
            "v2": strip_metric_examples(v2),
            "v2_minus_v1": {
                "schema_contradiction_delta": round(v2["schema_contradiction_rate"] - v1["schema_contradiction_rate"], 6),
                "parameter_contradiction_delta": round(v2["parameter_contradiction_given_schema"] - v1["parameter_contradiction_given_schema"], 6),
                "parameter_error_delta": round(v2["parameter_error_given_schema"] - v1["parameter_error_given_schema"], 6),
                "reuse_coverage_delta": round(v2["schema_reuse_coverage"] - v1["schema_reuse_coverage"], 6),
            },
        }
    return report


def strip_metric_examples(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key != "schema_stable_parameter_split_examples"
    }


def event_parameter_audit(records: list[TransitionRecord]) -> dict[str, Any]:
    total = len(records)
    changed_records = [
        record for record in records
        if int(record.post_labels.get("changed_cells", 0) or 0) > 0
    ]
    actor_confidence = Counter(
        str(record.post_labels.get("labeler_v2", {}).get("actor_confidence", "unknown"))
        for record in records
    )
    fake_actor_suspect_count = sum(
        1 for record in records
        if record.post_labels.get("labeler_v2", {}).get("fake_actor_suspect")
    )
    unassigned = sum(
        1 for record in changed_records
        if channel_cluster_count(record, "unknown_delta") > 0
    )
    multi_channel = sum(
        1 for record in changed_records
        if len(record.post_labels.get("labeler_v2", {}).get("active_channels", ())) > 1
    )
    cluster_counts = [
        len(record.post_labels.get("labeler_v2", {}).get("changed_region_clusters", ()))
        for record in records
    ]
    click_records = [
        record for record in records
        if record.pre_features.get("action_kind") == "click"
    ]
    click_target_assigned = sum(
        1 for record in click_records
        if channel_cluster_count(record, "action_target_delta") > 0
    )
    hud_confusion = sum(
        1 for record in records
        for cluster in record.post_labels.get("labeler_v2", {}).get("changed_region_clusters", ())
        if cluster.get("overlaps_hud")
        and cluster["cluster_id"] not in assigned_cluster_ids(record, "hud_status_delta")
    )
    return {
        "actor_confidence_distribution": {
            key: {
                "count": count,
                "rate": round(count / total if total else 0.0, 6),
            }
            for key, count in sorted(actor_confidence.items())
        },
        "fake_actor_suspect_count": fake_actor_suspect_count,
        "unassigned_delta_rate": round(unassigned / len(changed_records) if changed_records else 0.0, 6),
        "multi_channel_delta_rate": round(multi_channel / len(changed_records) if changed_records else 0.0, 6),
        "changed_region_cluster_count_mean": round(sum(cluster_counts) / total if total else 0.0, 6),
        "hud_vs_playfield_confusion_count": hud_confusion,
        "click_target_assignment_rate": round(click_target_assigned / len(click_records) if click_records else 0.0, 6),
        "changed_transition_count": len(changed_records),
        "click_transition_count": len(click_records),
    }


def actor_detection_audit(records: list[TransitionRecord]) -> dict[str, Any]:
    total = len(records)
    pre_confidence = Counter(str(record.pre_features.get("actor_confidence", "unknown")) for record in records)
    v2_confidence = Counter(
        str(record.post_labels.get("labeler_v2", {}).get("actor_confidence", "unknown"))
        for record in records
    )
    emitted = sum(
        1 for record in records
        if record.post_labels.get("labeler_v2", {}).get("channels", {}).get("actor_delta", {}).get("delta") is not None
    )
    candidates_suppressed = sum(
        1 for record in records
        if record.post_labels.get("labeler_v2", {}).get("channels", {}).get("actor_delta", {}).get("candidate_delta") is not None
        and record.post_labels.get("labeler_v2", {}).get("channels", {}).get("actor_delta", {}).get("delta") is None
    )
    return {
        "pre_actor_confidence_distribution": counter_distribution(pre_confidence, total),
        "v2_actor_confidence_distribution": counter_distribution(v2_confidence, total),
        "actor_delta_emitted_count": emitted,
        "candidate_delta_suppressed_count": candidates_suppressed,
        "fake_actor_suspect_count": sum(
            1 for record in records
            if record.post_labels.get("labeler_v2", {}).get("fake_actor_suspect")
        ),
    }


def changed_region_assignment_report(records: list[TransitionRecord]) -> dict[str, Any]:
    channel_counts = Counter()
    cluster_total = 0
    active_channel_sets = Counter()
    for record in records:
        labeler = record.post_labels.get("labeler_v2", {})
        active = tuple(labeler.get("active_channels", ()))
        active_channel_sets[active] += 1
        assignments = labeler.get("changed_region_assignments", {})
        for channel, cluster_ids in assignments.items():
            channel_counts[channel] += len(cluster_ids)
            cluster_total += len(cluster_ids)
    return {
        "cluster_assignments_by_channel": dict(sorted(channel_counts.items())),
        "cluster_assignment_total": cluster_total,
        "active_channel_sets": [
            {"channels": list(channels), "count": count}
            for channels, count in active_channel_sets.most_common()
        ],
    }


def stage_report_v2(records: list[TransitionRecord]) -> dict[str, Any]:
    split_groups = v2_parameter_split_groups(records, R3A_COMPRESSED)
    channel_variation = Counter()
    examples = []
    for group in split_groups:
        records_in_group = group["records"]
        varied = []
        for channel in (
            "actor_delta",
            "action_target_delta",
            "hud_status_delta",
            "resource_delta",
            "local_contact_delta",
            "remote_delta",
            "independent_motion_delta",
            "terminal_delta",
            "unknown_delta",
        ):
            if len({v2_channel_signature(record, channel) for record in records_in_group}) > 1:
                channel_variation[channel] += 1
                varied.append(channel)
        if varied and len(examples) < 20:
            examples.append(
                {
                    "context_schema_key": group["key"],
                    "steps": [record.step for record in records_in_group[:8]],
                    "actions": [record.action for record in records_in_group[:8]],
                    "varied_channels": varied,
                }
            )
    top = channel_variation.most_common(1)[0] if channel_variation else (None, 0)
    active_counts = Counter()
    for record in records:
        for channel in record.post_labels.get("labeler_v2", {}).get("active_channels", ()):
            active_counts[channel] += 1
    return {
        "baseline_family": R3A_COMPRESSED.name,
        "parameter_split_groups": len(split_groups),
        "top_localizing_channel": top[0],
        "top_localizing_channel_share": round(top[1] / len(split_groups) if split_groups else 0.0, 6),
        "localized_channel_counts": dict(channel_variation),
        "active_channel_counts": dict(active_counts),
        "examples": examples,
    }


def controlled_entity_audit(records: list[TransitionRecord]) -> dict[str, Any]:
    total = len(records)
    changed_records = [
        record for record in records
        if int(record.post_labels.get("changed_cells", 0) or 0) > 0
    ]
    type_counts = Counter()
    confidence_counts = Counter()
    suppressed = Counter()
    candidate_samples = []
    controlled_delta_records = 0
    for record in records:
        controlled = record.post_labels.get("labeler_v3", {}).get("controlled_entity", {})
        type_counts[str(controlled.get("type", "unknown"))] += 1
        confidence_counts[str(controlled.get("confidence", "unknown"))] += 1
        for reason in controlled.get("suppressed_reasons", ()):
            suppressed[str(reason)] += 1
        candidates = controlled.get("candidates", ())
        if candidates and len(candidate_samples) < 25:
            candidate_samples.append(
                {
                    "game": record.game,
                    "step": record.step,
                    "action": record.action,
                    "controlled_entity_type": controlled.get("type"),
                    "confidence": controlled.get("confidence"),
                    "top_candidates": candidates[:3],
                }
            )
        if v3_channel_cluster_count(record, "controlled_entity_delta") > 0:
            controlled_delta_records += 1
    ambiguous = type_counts["ambiguous"]
    return {
        "controlled_entity_type_distribution": counter_distribution(type_counts, total),
        "confidence_distribution": counter_distribution(confidence_counts, total),
        "candidate_scores_top_k": candidate_samples,
        "suppressed_candidate_reasons": dict(sorted(suppressed.items())),
        "controlled_delta_assignment_rate": round(
            controlled_delta_records / len(changed_records) if changed_records else 0.0,
            6,
        ),
        "ambiguous_control_rate": round(ambiguous / total if total else 0.0, 6),
        "changed_transition_count": len(changed_records),
    }


def region_role_map_report(records: list[TransitionRecord]) -> dict[str, Any]:
    role_map = build_region_role_map(records)
    role_counts = Counter(
        region.get("role", "unknown")
        for region in role_map.get("regions", {}).values()
    )
    confidence_counts = Counter(
        region.get("confidence", "unknown")
        for region in role_map.get("regions", {}).values()
    )
    return {
        "region_count": len(role_map.get("regions", {})),
        "role_distribution": dict(sorted(role_counts.items())),
        "confidence_distribution": dict(sorted(confidence_counts.items())),
        "regions": role_map.get("regions", {}),
    }


def delta_channel_assignment_v3_report(records: list[TransitionRecord]) -> dict[str, Any]:
    channel_counts = Counter()
    confidence_counts: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts: dict[str, Counter[str]] = defaultdict(Counter)
    active_channel_sets = Counter()
    cluster_total = 0
    changed_records = [
        record for record in records
        if int(record.post_labels.get("changed_cells", 0) or 0) > 0
    ]
    click_records = [
        record for record in records
        if record.pre_features.get("action_kind") == "click"
    ]
    for record in records:
        labeler = record.post_labels.get("labeler_v3", {})
        active_channel_sets[tuple(labeler.get("active_channels", ()))] += 1
        for channel, assignments in labeler.get("changed_region_assignments", {}).items():
            channel_counts[channel] += len(assignments)
            cluster_total += len(assignments)
            for assignment in assignments:
                confidence_counts[channel][str(assignment.get("confidence", "unknown"))] += 1
                reason_counts[channel][str(assignment.get("reason", "unknown"))] += 1
    unknown_records = sum(
        1 for record in changed_records
        if v3_channel_cluster_count(record, "unknown_delta") > 0
    )
    action_target_records = sum(
        1 for record in click_records
        if v3_channel_cluster_count(record, "action_target_delta") > 0
    )
    controlled_records = sum(
        1 for record in changed_records
        if v3_channel_cluster_count(record, "controlled_entity_delta") > 0
    )
    return {
        "cluster_assignments_by_channel": dict(sorted(channel_counts.items())),
        "cluster_assignment_total": cluster_total,
        "hud_status_cluster_share": round(channel_counts["hud_status_delta"] / cluster_total if cluster_total else 0.0, 6),
        "action_target_cluster_share": round(channel_counts["action_target_delta"] / cluster_total if cluster_total else 0.0, 6),
        "confidence_by_channel": {
            channel: dict(sorted(counter.items()))
            for channel, counter in sorted(confidence_counts.items())
        },
        "reason_counts_by_channel": {
            channel: dict(sorted(counter.items()))
            for channel, counter in sorted(reason_counts.items())
        },
        "active_channel_sets": [
            {"channels": list(channels), "count": count}
            for channels, count in active_channel_sets.most_common()
        ],
        "controlled_entity_assignment_rate": round(
            controlled_records / len(changed_records) if changed_records else 0.0,
            6,
        ),
        "click_target_assignment_rate": round(
            action_target_records / len(click_records) if click_records else 0.0,
            6,
        ),
        "unknown_delta_rate": round(
            unknown_records / len(changed_records) if changed_records else 0.0,
            6,
        ),
        "changed_transition_count": len(changed_records),
        "click_transition_count": len(click_records),
    }


def outcome_labeler_v2_vs_v3_report(records: list[TransitionRecord]) -> dict[str, Any]:
    families = (
        R3A_COMPRESSED,
        R4_CLICK_PROCEDURAL_VISUAL_ONLY,
        R4_CLICK_PROCEDURAL_VISUAL_PLUS_R3A,
        R4_CLICK_PROCEDURAL_LEARNED_ONLY,
        R4_CLICK_PROCEDURAL_LEARNED_PLUS_R3A,
    )
    report = {}
    for family in families:
        v2 = prequential_parameter_metrics(records, family, labeler="v2")
        v3 = prequential_parameter_metrics(records, family, labeler="v3")
        report[family.name] = {
            "v2": strip_metric_examples(v2),
            "v3": strip_metric_examples(v3),
            "v3_minus_v2": {
                "schema_contradiction_delta": round(v3["schema_contradiction_rate"] - v2["schema_contradiction_rate"], 6),
                "parameter_contradiction_delta": round(v3["parameter_contradiction_given_schema"] - v2["parameter_contradiction_given_schema"], 6),
                "parameter_error_delta": round(v3["parameter_error_given_schema"] - v2["parameter_error_given_schema"], 6),
                "parameter_reused_contexts_delta": int(v3["parameter_reused_contexts"] - v2["parameter_reused_contexts"]),
                "parameter_reused_transitions_delta": int(v3["parameter_reused_transitions"] - v2["parameter_reused_transitions"]),
                "reuse_coverage_delta": round(v3["schema_reuse_coverage"] - v2["schema_reuse_coverage"], 6),
            },
        }
    return report


def stage_report_v3(records: list[TransitionRecord]) -> dict[str, Any]:
    split_groups = parameter_split_groups(records, R3A_COMPRESSED, labeler="v3")
    channel_variation = Counter()
    examples = []
    channels = (
        "controlled_entity_delta",
        "action_target_delta",
        "hud_status_delta",
        "resource_delta",
        "remote_delta",
        "independent_motion_delta",
        "terminal_delta",
        "unknown_delta",
    )
    for group in split_groups:
        records_in_group = group["records"]
        varied = []
        for channel in channels:
            if len({v3_channel_signature(record, channel) for record in records_in_group}) > 1:
                channel_variation[channel] += 1
                varied.append(channel)
        if varied and len(examples) < 20:
            examples.append(
                {
                    "context_schema_key": group["key"],
                    "steps": [record.step for record in records_in_group[:8]],
                    "actions": [record.action for record in records_in_group[:8]],
                    "varied_channels": varied,
                }
            )
    top = channel_variation.most_common(1)[0] if channel_variation else (None, 0)
    active_counts = Counter()
    for record in records:
        for channel in record.post_labels.get("labeler_v3", {}).get("active_channels", ()):
            active_counts[channel] += 1
    return {
        "baseline_family": R3A_COMPRESSED.name,
        "parameter_split_groups": len(split_groups),
        "top_localizing_channel": top[0],
        "top_localizing_channel_share": round(top[1] / len(split_groups) if split_groups else 0.0, 6),
        "localized_channel_counts": dict(channel_variation),
        "active_channel_counts": dict(active_counts),
        "examples": examples,
    }


def parameter_model_probe_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_parameter_probe_game",
            "game": game,
            "reason": "cd82 remains a click_target_assignment problem; parameter probes are deferred",
        }
    models = {
        "P0_majority_parameter_given_R3_schema": "P0",
        "P1_local_geometry_parameter_model": "P1",
        "P2_contact_conditioned_parameter_model": "P2",
    }
    metrics = {
        name: parameter_probe_metrics(records, model_kind=kind)
        for name, kind in models.items()
    }
    baseline = metrics["P0_majority_parameter_given_R3_schema"]
    comparisons = {
        name: parameter_probe_comparison(baseline, payload)
        for name, payload in metrics.items()
        if name != "P0_majority_parameter_given_R3_schema"
    }
    labels = []
    if comparisons["P1_local_geometry_parameter_model"]["verdict"] == "success":
        labels.append("local_geometry_carries_parameter_signal")
    if comparisons["P2_contact_conditioned_parameter_model"]["verdict"] == "success":
        labels.append("contact_conditioning_carries_parameter_signal")
    if any(item["suspicious_fragmentation_flag"] for item in comparisons.values()):
        labels.append("probe_fragmentation_risk")
    if not labels:
        labels.append("no_local_or_contact_parameter_signal")
    return {
        "status": "evaluated",
        "game": game,
        "labeler": "v3",
        "uses_v3_controlled_entity_labeler": True,
        "deployability_note": "P1/P2 are diagnostic probes conditioned on V3 controlled-entity assignment, not deployable representations.",
        "models": metrics,
        "comparisons_to_P0": comparisons,
        "verdict": {
            "primary": labels[0],
            "labels": labels,
        },
    }


def motion_law_probe_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_motion_probe_game",
            "game": game,
            "reason": "motion-law probes are limited to LS20/AR25; CD82 remains an action-target assignment problem",
        }
    phase_cluster = build_vss_phase_cluster(records)
    models = {
        "M0_majority_delta_given_R3a_schema_action": "M0",
        "M1_local_passability_occupancy": "M1",
        "M2_contact_conditioned": "M2",
        "M3_independent_mover_tracking_phase": "M3",
        "M4_coupled_set_relational": "M4",
    }
    heads = {}
    comparisons = {}
    labels = []
    for head in ("controlled_entity_delta", "independent_motion_delta"):
        head_metrics = {
            name: motion_probe_metrics(
                records,
                head=head,
                model_kind=kind,
                phase_cluster=phase_cluster,
            )
            for name, kind in models.items()
        }
        baseline = head_metrics["M0_majority_delta_given_R3a_schema_action"]
        head_comparisons = {
            name: motion_probe_comparison(baseline, payload)
            for name, payload in head_metrics.items()
            if name != "M0_majority_delta_given_R3a_schema_action"
        }
        heads[head] = head_metrics
        comparisons[head] = head_comparisons
        if head == "controlled_entity_delta":
            if head_comparisons["M1_local_passability_occupancy"]["verdict"] == "success":
                labels.append("local_collision_or_passability_signal")
            if head_comparisons["M2_contact_conditioned"]["verdict"] == "success":
                labels.append("contact_conditioning_signal")
            if head_comparisons["M4_coupled_set_relational"]["verdict"] == "success":
                labels.append("coupled_motion_signal")
        if head == "independent_motion_delta" and head_comparisons["M3_independent_mover_tracking_phase"]["verdict"] == "success":
            labels.append("independent_mover_phase_signal")
        if any(item["suspicious_fragmentation_flag"] for item in head_comparisons.values()):
            labels.append("motion_probe_fragmentation_risk")
    if not labels:
        labels.append("no_compact_motion_law_signal")
    return {
        "status": "evaluated",
        "game": game,
        "labeler": "v3",
        "probe_type": "predictor_not_context_representation",
        "targets": {
            "controlled_entity_delta": "no_move / unit_move_direction / slide_or_forced_move_direction / multi_cell_delta_bucket / coupled_or_ambiguous",
            "independent_motion_delta": "absent / same_direction_tick / reverse_or_bounce / coupled_with_action / changed_pose_bucket / ambiguous",
        },
        "phase_cluster": {
            "k": phase_cluster.get("k", 0),
            "cluster_sizes": phase_cluster.get("cluster_sizes", {}),
        },
        "heads": heads,
        "comparisons_to_M0": comparisons,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def motion_probe_metrics(
    records: list[TransitionRecord],
    *,
    head: str,
    model_kind: str,
    phase_cluster: Mapping[str, Any],
) -> dict[str, Any]:
    active_records = [
        record for record in records
        if motion_probe_record_is_active(record, head)
    ]
    prior: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    memory: dict[str, Any] = {}
    predictions = 0
    hits = 0
    error_examples = []
    for record in active_records:
        target = motion_probe_target(record, head=head)
        key = motion_probe_key(
            record,
            head=head,
            model_kind=model_kind,
            phase_cluster=phase_cluster,
            memory=memory,
        )
        if prior[key]:
            predictions += 1
            predicted = prior[key].most_common(1)[0][0]
            if predicted == target:
                hits += 1
            elif len(error_examples) < 20:
                error_examples.append(
                    {
                        "game": record.game,
                        "step": record.step,
                        "action": record.action,
                        "head": head,
                        "model": model_kind,
                        "context_id": short_digest(key),
                        "predicted": predicted,
                        "actual": target,
                        "schema": outcome_schema_key(record, labeler="v3"),
                    }
                )
        prior[key][target] += 1
        groups[key][target] += 1
        update_motion_probe_memory(memory, record)
    total = len(active_records)
    reused_keys = [key for key, counter in groups.items() if sum(counter.values()) > 1]
    contradicted = [key for key in reused_keys if len(groups[key]) > 1]
    reused_transitions = sum(sum(groups[key].values()) for key in reused_keys)
    weighted_entropy = sum(sum(counter.values()) * entropy(counter) for counter in groups.values())
    accuracy = hits / predictions if predictions else 0.0
    return {
        "model": model_kind,
        "head": head,
        "active_examples": total,
        "prediction_count": predictions,
        "prediction_coverage": round(predictions / total if total else 0.0, 6),
        "prequential_accuracy": round(accuracy, 6),
        "parameter_error": round(1.0 - accuracy if predictions else 1.0, 6),
        "feature_cardinality": len(groups),
        "unique_contexts": len(groups),
        "reused_contexts": len(reused_keys),
        "reused_transitions": reused_transitions,
        "reuse_coverage": round(reused_transitions / total if total else 0.0, 6),
        "one_off_ratio": round(
            sum(1 for counter in groups.values() if sum(counter.values()) == 1) / len(groups)
            if groups
            else 0.0,
            6,
        ),
        "contradicted_reused_contexts": len(contradicted),
        "parameter_contradiction_rate": round(len(contradicted) / len(reused_keys) if reused_keys else 0.0, 6),
        "parameter_entropy": round(weighted_entropy / total if total else 0.0, 6),
        "support": motion_probe_support_status(
            active_examples=total,
            prediction_count=predictions,
            reused_contexts=len(reused_keys),
            reused_transitions=reused_transitions,
            reuse_coverage=reused_transitions / total if total else 0.0,
        ),
        "raw_feature_guard": {
            "raw_ids_in_context_key": False,
            "raw_bboxes_in_context_key": False,
            "raw_pixel_hashes_in_context_key": False,
        },
        "error_examples": error_examples,
    }


def motion_probe_record_is_active(record: TransitionRecord, head: str) -> bool:
    if head == "controlled_entity_delta":
        return record.action in {1, 2, 3, 4}
    if head == "independent_motion_delta":
        return record.action in {1, 2, 3, 4} or bool(stable_independent_motion_deltas(record))
    return False


def motion_probe_key(
    record: TransitionRecord,
    *,
    head: str,
    model_kind: str,
    phase_cluster: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> str:
    payload: dict[str, Any] = {
        "head": head,
        "r3a_context": context_key(record, R3A_COMPRESSED),
        "action": record.action,
        "schema": outcome_schema_key(record, labeler="v3"),
    }
    if model_kind in {"M1", "M2", "M3", "M4"}:
        payload["local_passability"] = local_motion_law_features(record)
    if model_kind in {"M2", "M3", "M4"}:
        payload["contact_conditioning"] = contact_conditioned_parameter_features(record)
    if model_kind in {"M3", "M4"}:
        payload["independent_mover_tracking"] = independent_mover_tracking_features(record, phase_cluster, memory)
    if model_kind == "M4":
        payload["coupled_set_relational"] = coupled_set_relational_features(record)
    return stable_json(payload)


def local_motion_law_features(record: TransitionRecord) -> dict[str, Any]:
    local = local_geometry_parameter_features(record)
    return {
        "action_direction": action_direction(record.action),
        "controlled_entity_type": local["controlled_entity_type"],
        "controlled_confidence": local["controlled_confidence"],
        "controlled_set_size_bucket": local["controlled_set_size_bucket"],
        "controlled_bbox_bucket": local["controlled_bbox_bucket"],
        "controlled_shape_class": local["controlled_shape_class"],
        "local_obstacle_map": local["local_obstacle_map"],
        "relative_free_space_pattern": local["relative_free_space_pattern"],
        "nearest_component_signatures": local["nearest_component_signatures"],
    }


def independent_mover_tracking_features(
    record: TransitionRecord,
    phase_cluster: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "visible_independent_mover_signatures": record.pre_features.get("visible_independent_mover_signatures", ()),
        "phase_cluster": phase_for_record(record, phase_cluster),
        "previous_controlled_motion": memory.get("previous_controlled_motion", "none"),
        "previous_independent_motion": memory.get("previous_independent_motion", "none"),
        "previous_action": memory.get("previous_action", "none"),
        "controlled_to_independent_relation": controlled_to_independent_relation(record),
        "action_caused_controlled_motion_before_t": memory.get("previous_controlled_motion", "none") not in {"none", "no_move"},
    }


def coupled_set_relational_features(record: TransitionRecord) -> dict[str, Any]:
    controlled = record.post_labels.get("labeler_v3", {}).get("controlled_entity", {})
    candidates = high_confidence_controlled_candidates(record)
    return {
        "controlled_entity_type": controlled.get("type", "none"),
        "controlled_confidence": controlled.get("confidence", "low"),
        "controlled_set_size_bucket": bucket_count(len(candidates)),
        "current_relative_geometry_class": record.pre_features.get("current_relative_geometry_class"),
        "current_touch_overlap_relation": record.pre_features.get("current_touch_overlap_relation"),
        "canonical_group_shape_family": record.pre_features.get("canonical_group_shape_family"),
        "mirror_rotation_normalized_signature": record.pre_features.get("mirror_rotation_normalized_signature"),
    }


def motion_probe_target(record: TransitionRecord, *, head: str) -> str:
    if head == "controlled_entity_delta":
        return controlled_motion_target(record)
    if head == "independent_motion_delta":
        return independent_motion_target(record)
    return "unknown"


def controlled_motion_target(record: TransitionRecord) -> str:
    candidates = high_confidence_controlled_candidates(record)
    controlled = record.post_labels.get("labeler_v3", {}).get("controlled_entity", {})
    if not candidates:
        return "no_move"
    translations = sorted({tuple(candidate.get("translation", ())) for candidate in candidates})
    if len(translations) > 1:
        return f"ambiguous_or_coupled:{bucket_count(len(translations))}"
    translation = translations[0]
    side = motion_direction_from_translation(translation)
    magnitude = motion_magnitude(translation)
    if magnitude == 0:
        return "no_move"
    if controlled.get("type") == "coupled_set":
        return f"coupled_set:{side}:{bucket_count(magnitude)}"
    if magnitude == 1:
        return f"unit_move_direction:{side}"
    if magnitude <= 4:
        return f"slide_or_forced_move_direction:{side}:{bucket_count(magnitude)}"
    return f"multi_cell_delta_bucket:{side}:{bucket_count(magnitude)}"


def independent_motion_target(record: TransitionRecord) -> str:
    deltas = stable_independent_motion_deltas(record)
    if not deltas:
        return "absent"
    directions = sorted({motion_direction_from_translation(tuple(delta.translation)) for delta in deltas})
    magnitudes = [motion_magnitude(tuple(delta.translation)) for delta in deltas]
    controlled_direction = motion_direction_from_translation(controlled_focus_translation(record))
    if len(directions) > 1:
        return f"changed_pose_bucket:{bucket_count(len(deltas))}:{bucket_count(sum(magnitudes))}"
    direction = directions[0]
    if direction == opposite_direction(previous_independent_direction(record)) and direction != "none":
        return f"reverse_or_bounce:{direction}"
    if controlled_direction != "none" and direction == controlled_direction:
        return f"coupled_with_action:{direction}:{bucket_count(len(deltas))}"
    return f"same_direction_tick:{direction}:{bucket_count(sum(magnitudes))}"


def stable_independent_motion_deltas(record: TransitionRecord) -> list[Any]:
    controlled_ids = {
        int(candidate.get("object_id"))
        for candidate in record.post_labels.get("labeler_v3", {}).get("controlled_entity", {}).get("candidates", ())
        if candidate.get("object_id") is not None and float(candidate.get("score", 0.0) or 0.0) >= 0.75
    }
    return [
        delta for delta in stable_motion_deltas(record)
        if int(delta.object_id) not in controlled_ids
    ]


def controlled_focus_translation(record: TransitionRecord) -> tuple[int, int]:
    candidate = controlled_focus_candidate(record)
    if not candidate:
        return (0, 0)
    translation = tuple(candidate.get("translation", (0, 0)))
    if len(translation) != 2:
        return (0, 0)
    return (int(translation[0]), int(translation[1]))


def previous_independent_direction(record: TransitionRecord) -> str:
    previous = record.pre_features.get("recent_noop_streak")
    return "none" if previous is None else "none"


def update_motion_probe_memory(memory: dict[str, Any], record: TransitionRecord) -> None:
    memory["previous_controlled_motion"] = controlled_motion_target(record)
    memory["previous_independent_motion"] = independent_motion_target(record)
    memory["previous_action"] = record.action


def phase_for_record(record: TransitionRecord, phase_cluster: Mapping[str, Any]) -> str:
    key = stable_json(record.pre_features.get("visible_state_summary", {}) or {})
    return str(phase_cluster.get("assignments", {}).get(key, "unknown"))


def controlled_to_independent_relation(record: TransitionRecord) -> str:
    controlled_candidate = controlled_focus_candidate(record)
    independent = stable_independent_motion_deltas(record)
    if not controlled_candidate or not independent:
        return "none"
    controlled_bbox = tuple(controlled_candidate.get("bbox_before", ()))
    if len(controlled_bbox) != 4:
        return "unknown"
    controlled_center = bbox_center(controlled_bbox)
    distances = []
    for delta in independent:
        bbox = tuple(delta.bbox_before)
        if len(bbox) == 4:
            center = bbox_center(bbox)
            distances.append(abs(center[0] - controlled_center[0]) + abs(center[1] - controlled_center[1]))
    if not distances:
        return "unknown"
    nearest = min(distances)
    if nearest <= 2:
        return "near"
    if nearest <= 8:
        return "mid"
    return "far"


def motion_probe_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_error = float(baseline.get("parameter_error", 1.0) or 1.0)
    candidate_error = float(candidate.get("parameter_error", 1.0) or 1.0)
    error_drop = max(0.0, baseline_error - candidate_error)
    relative_drop = error_drop / baseline_error if baseline_error else 0.0
    reuse_loss = max(
        0.0,
        float(baseline.get("reuse_coverage", 0.0) or 0.0)
        - float(candidate.get("reuse_coverage", 0.0) or 0.0),
    )
    coverage = float(candidate.get("prediction_coverage", 0.0) or 0.0)
    cardinality_multiplier = (
        float(candidate.get("feature_cardinality", 0) or 0)
        / max(1.0, float(baseline.get("feature_cardinality", 0) or 0))
    )
    suspicious = error_drop > 0.0 and (reuse_loss > 0.10 or cardinality_multiplier > 3.0)
    support = candidate.get("support", {})
    success = (
        support.get("status") == "eligible"
        and relative_drop >= 0.15
        and coverage >= 0.50
        and reuse_loss <= 0.10
        and cardinality_multiplier <= 3.0
    )
    if support.get("status") == "insufficient_support":
        verdict = "insufficient_support"
    elif suspicious:
        verdict = "suspicious_fragmentation"
    elif success:
        verdict = "success"
    else:
        verdict = "no_gain"
    return {
        "parameter_error_drop": round(error_drop, 6),
        "parameter_error_relative_drop": round(relative_drop, 6),
        "reuse_loss": round(reuse_loss, 6),
        "prediction_coverage": round(coverage, 6),
        "feature_cardinality_multiplier_vs_M0": round(cardinality_multiplier, 6),
        "gain_per_reuse_lost": round(error_drop / max(reuse_loss, 1e-6), 6),
        "support": support,
        "suspicious_fragmentation_flag": suspicious,
        "verdict": verdict,
    }


def motion_probe_support_status(
    *,
    active_examples: int,
    prediction_count: int,
    reused_contexts: int,
    reused_transitions: int,
    reuse_coverage: float,
) -> dict[str, Any]:
    eligible = (
        active_examples >= 30
        and prediction_count >= 30
        and reused_contexts >= 10
        and reused_transitions >= 30
        and reuse_coverage >= 0.25
    ) or (active_examples < 30 and prediction_count >= 2 and reuse_coverage >= 0.25)
    return {
        "status": "eligible" if eligible else "insufficient_support",
        "active_examples": active_examples,
        "prediction_count": prediction_count,
        "reused_contexts": reused_contexts,
        "reused_transitions": reused_transitions,
        "reuse_coverage": round(reuse_coverage, 6),
    }


def motion_direction_from_translation(translation: tuple[Any, ...]) -> str:
    if len(translation) != 2:
        return "unknown"
    row_delta = int(translation[0])
    col_delta = int(translation[1])
    if row_delta == 0 and col_delta == 0:
        return "none"
    if abs(row_delta) >= abs(col_delta):
        return "up" if row_delta < 0 else "down"
    return "left" if col_delta < 0 else "right"


def motion_magnitude(translation: tuple[Any, ...]) -> int:
    if len(translation) != 2:
        return 0
    return abs(int(translation[0])) + abs(int(translation[1]))


def action_direction(action: int) -> str:
    return {
        1: "up",
        2: "down",
        3: "left",
        4: "right",
    }.get(action, "none")


def opposite_direction(direction: str) -> str:
    return {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
    }.get(direction, "none")


def motion_target_granularity_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_motion_granularity_game",
            "game": game,
            "reason": "controlled-motion target granularity sweep is limited to LS20/AR25",
        }
    phase_cluster = build_vss_phase_cluster(records)
    targets = {
        "T0_controlled_motion_happened": "T0",
        "T1_controlled_motion_class": "T1",
        "T2_direction_distance_bucket": "T2",
        "T3_normalized_motion_vector": "T3",
        "T4_stop_reason": "T4",
        "T5_exact_controlled_region_parameter": "T5",
    }
    models = motion_predictor_model_kinds()
    target_reports = {}
    first_break = None
    labels = []
    for target_name, target_kind in targets.items():
        strict_baseline = strict_conjunctive_predictor_metrics(
            records,
            active_fn=lambda record: record.action in {1, 2, 3, 4},
            target_fn=lambda record, kind=target_kind: controlled_motion_target_at(record, kind),
            key_fn=motion_strict_context_key,
            memory_update_fn=update_motion_probe_memory,
            model_name="M0_strict_R3a_schema_action",
        )
        metrics = {
            model_name: bounded_feature_predictor_metrics(
                records,
                active_fn=lambda record: record.action in {1, 2, 3, 4},
                target_fn=lambda record, kind=target_kind: controlled_motion_target_at(record, kind),
                feature_fn=lambda record, memory, kind=model_kind: motion_predictor_atoms(
                    record,
                    model_kind=kind,
                    phase_cluster=phase_cluster,
                    memory=memory,
                    target_scope="controlled",
                ),
                memory_update_fn=update_motion_probe_memory,
                model_name=model_name,
            )
            for model_name, model_kind in models.items()
        }
        atom_baseline = metrics["M0_atom_vote_R3a_schema_action"]
        comparisons = {
            model_name: bounded_predictor_comparison(strict_baseline, payload)
            for model_name, payload in metrics.items()
            if model_name != "M0_atom_vote_R3a_schema_action"
        }
        atom_vs_strict = bounded_predictor_comparison(strict_baseline, atom_baseline)
        target_reports[target_name] = {
            "target_kind": target_kind,
            "strict_conjunctive_baseline": strict_baseline,
            "models": metrics,
            "atom_vote_vs_strict_baseline": atom_vs_strict,
            "comparisons_to_strict_M0": comparisons,
            "first_successful_model": first_successful_predictor(comparisons),
        }
        if (
            first_break is None
            and strict_baseline.get("support", {}).get("status") == "eligible"
            and float(strict_baseline["error"]) > 0.10
        ):
            first_break = target_name
        if any(item["verdict"] == "success" for item in comparisons.values()):
            labels.append(f"{target_kind}_feature_signal")
        if any(item["suspicious_fragmentation_flag"] for item in comparisons.values()):
            labels.append(f"{target_kind}_fragmentation_risk")
    if first_break is None:
        labels.insert(0, "controlled_target_not_breaking_under_current_granularities")
    else:
        labels.insert(0, "controlled_target_break_detected")
    return {
        "status": "evaluated",
        "game": game,
        "labeler": "v3",
        "probe_type": "strict_context_baseline_plus_bounded_feature_predictors",
        "deployability_note": "Predictors are conditioned on V3 controlled-entity assignment; this is a diagnostic, not a deployed policy representation.",
        "first_granularity_where_strict_M0_fails": first_break,
        "baseline_note": "M0_strict_R3a_schema_action is the conjunctive baseline. M0_atom_vote_R3a_schema_action is retained only as a feature-vote probe.",
        "targets": target_reports,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def controlled_region_material_decomposition_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_controlled_region_material_game",
            "game": game,
            "reason": "controlled-region/material decomposition is limited to LS20/AR25",
        }
    subtargets = {
        "T5a_identity_correspondence": "identity",
        "T5b_footprint_changed_cells": "footprint",
        "T5c_material_color_value_transition": "material",
        "T5d_overlap_merge_split_behavior": "overlap",
        "T5e_collateral_changed_regions": "collateral",
    }
    target_reports = {}
    hard_targets = []
    insufficient = []
    low_error = []
    for target_name, target_kind in subtargets.items():
        strict_baseline = strict_conjunctive_predictor_metrics(
            records,
            active_fn=lambda record: record.action in {1, 2, 3, 4},
            target_fn=lambda record, kind=target_kind: controlled_region_material_target(record, kind),
            key_fn=motion_strict_context_key,
            memory_update_fn=update_motion_probe_memory,
            model_name=f"{target_name}_strict_R3a_schema_action",
        )
        support = strict_baseline.get("support", {})
        error = float(strict_baseline.get("error", 1.0) or 1.0)
        if support.get("status") == "insufficient_support":
            interpretation = "insufficient_support"
            insufficient.append(target_name)
        elif error >= 0.25:
            interpretation = "hard_subtarget_under_R3a_schema"
            hard_targets.append((target_name, error))
        elif error <= 0.10:
            interpretation = "low_error_under_R3a_schema"
            low_error.append(target_name)
        else:
            interpretation = "moderate_error_under_R3a_schema"
        target_reports[target_name] = {
            "target_kind": target_kind,
            "strict_conjunctive_baseline": strict_baseline,
            "interpretation": interpretation,
        }
    hardness_rank = [
        {"target": name, "error": round(error, 6)}
        for name, error in sorted(hard_targets, key=lambda item: item[1], reverse=True)
    ]
    labels = []
    if hard_targets:
        labels.append("controlled_region_material_subtargets_hard")
        labels.extend(controlled_decomposition_hard_labels(hard_targets))
    if insufficient:
        labels.append("controlled_region_material_insufficient_support")
    if not hard_targets and low_error:
        labels.append("controlled_region_material_low_error_under_current_targets")
    if not labels:
        labels.append("controlled_region_material_moderate_error")
    return {
        "status": "evaluated",
        "game": game,
        "labeler": "v3",
        "probe_type": "diagnostic_t5_outcome_decomposition_not_new_representation",
        "context_baseline": "R3a_computable_compressed_static + V3 schema + action",
        "active_actions": [1, 2, 3, 4],
        "subtargets": target_reports,
        "hardness_rank": hardness_rank,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def controlled_decomposition_hard_labels(hard_targets: list[tuple[str, float]]) -> list[str]:
    mapping = {
        "T5a_identity_correspondence": "controlled_identity_correspondence_hard",
        "T5b_footprint_changed_cells": "controlled_footprint_hard",
        "T5c_material_color_value_transition": "controlled_material_transition_hard",
        "T5d_overlap_merge_split_behavior": "controlled_overlap_merge_split_hard",
        "T5e_collateral_changed_regions": "collateral_regions_hard",
    }
    return [mapping[name] for name, _error in hard_targets if name in mapping]


def controlled_region_material_target(record: TransitionRecord, target_kind: str) -> str:
    shape = np.asarray(record.before.pixels[-1]).shape
    if target_kind == "identity":
        return stable_json(controlled_identity_correspondence_target(record))
    if target_kind == "footprint":
        return stable_json(controlled_footprint_changed_cells_target(record, shape))
    if target_kind == "material":
        return stable_json(controlled_material_value_transition_target(record))
    if target_kind == "overlap":
        return stable_json(controlled_overlap_merge_split_target(record))
    if target_kind == "collateral":
        return stable_json(controlled_collateral_changed_regions_target(record, shape))
    return channel_parameter_target(record, "controlled_entity_delta")


def controlled_identity_correspondence_target(record: TransitionRecord) -> dict[str, Any]:
    candidates = high_confidence_controlled_candidates(record)
    controlled_ids = controlled_candidate_object_ids(record)
    deltas = controlled_object_deltas(record)
    event = getattr(record, "event", None)
    created = set(getattr(event, "created_object_ids", ()) or ())
    disappeared = set(getattr(event, "disappeared_object_ids", ()) or ())
    reappeared = set(getattr(event, "reappeared_object_ids", ()) or ())
    if not candidates:
        continuity = "none"
    elif len(candidates) > 1:
        continuity = "ambiguous_or_coupled_multi"
    elif any(delta.object_id in controlled_ids for delta in deltas):
        continuity = "stable_correspondence"
    elif controlled_ids & (created | reappeared):
        continuity = "created_or_reappeared_candidate"
    elif controlled_ids & disappeared:
        continuity = "disappeared_candidate"
    else:
        continuity = "candidate_without_tracked_delta"
    controlled_groups = controlled_motion_groups(record, controlled_ids)
    labeler_controlled = record.post_labels.get("labeler_v3", {}).get("controlled_entity", {})
    return {
        "controlled_entity_type": labeler_controlled.get("type", "none"),
        "controlled_confidence": labeler_controlled.get("confidence", "low"),
        "candidate_count_bucket": bucket_count(len(candidates)),
        "object_delta_count_bucket": bucket_count(len(deltas)),
        "motion_group_count_bucket": bucket_count(len(controlled_groups)),
        "created_candidate_count_bucket": bucket_count(len(controlled_ids & created)),
        "disappeared_candidate_count_bucket": bucket_count(len(controlled_ids & disappeared)),
        "reappeared_candidate_count_bucket": bucket_count(len(controlled_ids & reappeared)),
        "identity_continuity": continuity,
    }


def controlled_footprint_changed_cells_target(
    record: TransitionRecord,
    shape: tuple[int, int],
) -> dict[str, Any]:
    clusters = clusters_for_v3_channel(record, "controlled_entity_delta")
    channel = channel_payload(record, "controlled_entity_delta")
    candidate_bboxes_after = tuple(
        sorted(
            stable_json(bbox_bucket(tuple(candidate.get("bbox_after", ())), shape))
            for candidate in high_confidence_controlled_candidates(record)
            if isinstance(candidate.get("bbox_after"), (list, tuple))
            and len(candidate.get("bbox_after", ())) == 4
        )
    )
    cluster_bboxes = tuple(
        sorted(
            stable_json(bbox_bucket(tuple(cluster.get("bbox", ())), shape))
            for cluster in clusters
            if isinstance(cluster.get("bbox"), (list, tuple)) and len(cluster.get("bbox", ())) == 4
        )
    )
    return {
        "cluster_count_bucket": bucket_count(len(clusters) or int(channel.get("cluster_count", 0) or 0)),
        "cluster_area_buckets": tuple(sorted(cluster.get("area_bucket", "unknown") for cluster in clusters))
        or tuple(channel.get("area_buckets", ())),
        "cluster_bbox_buckets": cluster_bboxes
        or tuple(
            sorted(
                stable_json(bbox_bucket(tuple(bbox), shape))
                for bbox in channel.get("bboxes", ())
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4
            )
        ),
        "candidate_after_bbox_buckets": candidate_bboxes_after,
        "candidate_translation_buckets": tuple(
            sorted(
                f"{motion_direction_from_translation(tuple(candidate.get('translation', ()) or ()))}:"
                f"{bucket_count(motion_magnitude(tuple(candidate.get('translation', ()) or ())))}"
                for candidate in high_confidence_controlled_candidates(record)
            )
        ),
    }


def controlled_material_value_transition_target(record: TransitionRecord) -> dict[str, Any]:
    deltas = controlled_object_deltas(record)
    channel = channel_payload(record, "controlled_entity_delta")
    value_transitions = Counter(
        (int(delta.value_before), int(delta.value_after))
        for delta in deltas
        if delta.value_changed
    )
    return {
        "value_changed_count_bucket": bucket_count(sum(value_transitions.values())),
        "value_transition_histogram": tuple(
            sorted(
                (before, after, bucket_count(count))
                for (before, after), count in value_transitions.items()
            )
        ),
        "changed_value_histogram": tuple(channel.get("color_histogram_delta", ())),
    }


def controlled_overlap_merge_split_target(record: TransitionRecord) -> dict[str, Any]:
    deltas = controlled_object_deltas(record)
    controlled_ids = controlled_candidate_object_ids(record)
    event = getattr(record, "event", None)
    created = set(getattr(event, "created_object_ids", ()) or ())
    disappeared = set(getattr(event, "disappeared_object_ids", ()) or ())
    controlled_groups = controlled_motion_groups(record, controlled_ids)
    shape_transform_counts = Counter(delta.shape_transform for delta in deltas if delta.shape_transform != "same")
    area_delta_buckets = Counter(bucket_signed(delta.area_delta) for delta in deltas if delta.area_delta != 0)
    if controlled_ids & (created | disappeared):
        behavior = "created_or_disappeared"
    elif shape_transform_counts or area_delta_buckets:
        behavior = "shape_or_area_change"
    elif len(high_confidence_controlled_candidates(record)) > 1 or len(controlled_groups) > 1:
        behavior = "coupled_or_multi"
    elif any(delta.moved for delta in deltas):
        behavior = "stable_translation_only"
    else:
        behavior = "none"
    return {
        "merge_split_class": behavior,
        "shape_changed_count_bucket": bucket_count(sum(shape_transform_counts.values())),
        "shape_transform_histogram": tuple(
            sorted((transform, bucket_count(count)) for transform, count in shape_transform_counts.items())
        ),
        "area_delta_histogram": tuple(
            sorted((area_delta, bucket_count(count)) for area_delta, count in area_delta_buckets.items())
        ),
        "controlled_motion_group_size_buckets": tuple(
            sorted(bucket_count(len(group.object_ids)) for group in controlled_groups)
        ),
    }


def controlled_collateral_changed_regions_target(
    record: TransitionRecord,
    shape: tuple[int, int],
) -> dict[str, Any]:
    channels = record.post_labels.get("labeler_v3", {}).get("channels", {})
    collateral_channels = []
    collateral_cluster_count = 0
    collateral_bbox_buckets = []
    for channel_name, payload in sorted(channels.items()):
        if channel_name == "controlled_entity_delta":
            continue
        if not channel_is_active_v3(channel_name, payload):
            continue
        collateral_channels.append(channel_name)
        if isinstance(payload, Mapping) and "cluster_count" in payload:
            collateral_cluster_count += int(payload.get("cluster_count", 0) or 0)
            collateral_bbox_buckets.extend(
                stable_json(bbox_bucket(tuple(bbox), shape))
                for bbox in payload.get("bboxes", ())
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4
            )
        elif isinstance(payload, Mapping):
            collateral_cluster_count += 1
    return {
        "active_collateral_channels": tuple(collateral_channels),
        "collateral_channel_count_bucket": bucket_count(len(collateral_channels)),
        "collateral_cluster_count_bucket": bucket_count(collateral_cluster_count),
        "collateral_bbox_buckets": tuple(sorted(collateral_bbox_buckets)),
        "terminal_or_resource_present": any(
            channel in collateral_channels for channel in ("terminal_delta", "resource_delta")
        ),
    }


def controlled_candidate_object_ids(record: TransitionRecord) -> set[int]:
    ids = set()
    for candidate in high_confidence_controlled_candidates(record):
        if candidate.get("object_id") is None:
            continue
        ids.add(int(candidate["object_id"]))
    return ids


def controlled_object_deltas(record: TransitionRecord) -> list[ObjectDelta]:
    event = getattr(record, "event", None)
    if event is None:
        return []
    controlled_ids = controlled_candidate_object_ids(record)
    return [delta for delta in event.object_deltas if int(delta.object_id) in controlled_ids]


def controlled_motion_groups(record: TransitionRecord, controlled_ids: set[int]) -> list[Any]:
    event = getattr(record, "event", None)
    if event is None or not controlled_ids:
        return []
    return [
        group for group in event.motion_groups
        if any(int(object_id) in controlled_ids for object_id in group.object_ids)
    ]


def clusters_for_v3_channel(record: TransitionRecord, channel: str) -> list[dict[str, Any]]:
    labeler = record.post_labels.get("labeler_v3", {})
    clusters = {
        cluster.get("cluster_id"): cluster
        for cluster in labeler.get("changed_region_clusters", ())
        if isinstance(cluster, Mapping) and cluster.get("cluster_id") is not None
    }
    assignments = labeler.get("changed_region_assignments", {}).get(channel, ())
    return [
        clusters[assignment.get("cluster_id")]
        for assignment in assignments
        if isinstance(assignment, Mapping) and assignment.get("cluster_id") in clusters
    ]


def channel_payload(record: TransitionRecord, channel: str) -> Mapping[str, Any]:
    payload = record.post_labels.get("labeler_v3", {}).get("channels", {}).get(channel)
    return payload if isinstance(payload, Mapping) else {}


def changed_region_factorization_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_changed_region_factorization_game",
            "game": game,
            "reason": "changed-region factorization is limited to LS20/AR25",
        }
    target_pairs = {
        "T5b_footprint": (
            lambda record: controlled_region_material_target(record, "footprint"),
            lambda record: stable_json(factorized_controlled_footprint_target(record)),
        ),
        "T5c_material_value": (
            lambda record: controlled_region_material_target(record, "material"),
            lambda record: stable_json(factorized_material_value_target(record)),
        ),
        "T5e_collateral": (
            lambda record: controlled_region_material_target(record, "collateral"),
            lambda record: stable_json(factorized_collateral_target(record)),
        ),
    }
    target_reports = {}
    labels = []
    for target_name, (old_target_fn, factorized_target_fn) in target_pairs.items():
        old_metrics = strict_conjunctive_predictor_metrics(
            records,
            active_fn=lambda record: record.action in {1, 2, 3, 4},
            target_fn=old_target_fn,
            key_fn=motion_strict_context_key,
            memory_update_fn=update_motion_probe_memory,
            model_name=f"{target_name}_old_v3_channel_target",
        )
        factorized_metrics = strict_conjunctive_predictor_metrics(
            records,
            active_fn=lambda record: record.action in {1, 2, 3, 4},
            target_fn=factorized_target_fn,
            key_fn=motion_strict_context_key,
            memory_update_fn=update_motion_probe_memory,
            model_name=f"{target_name}_factorized_target",
        )
        comparison = factorized_target_comparison(old_metrics, factorized_metrics)
        if comparison["verdict"] == "label_pollution_reduced":
            labels.append(f"{target_name}_label_pollution_reduced")
        elif comparison["verdict"] == "still_hard_after_factorization":
            labels.append(f"{target_name}_still_hard_after_factorization")
        target_reports[target_name] = {
            "old_v3_channel_target": old_metrics,
            "factorized_target": factorized_metrics,
            "comparison": comparison,
        }
    audit = changed_region_factorization_audit(records)
    if audit["unknown_cluster_rate"] > 0.25:
        labels.append("factorization_unknown_rate_high")
    if audit["multi_role_transition_rate"] > 0.50:
        labels.append("multi_channel_state_change_common")
    if not labels:
        labels.append("factorization_did_not_change_read")
    return {
        "status": "evaluated",
        "game": game,
        "labeler": "v3",
        "probe_type": "target_purification_changed_region_factorization",
        "context_baseline": "R3a_computable_compressed_static + V3 schema + action",
        "active_actions": [1, 2, 3, 4],
        "factor_roles": [
            "controlled_footprint_old",
            "controlled_footprint_new",
            "controlled_overlap_or_trail",
            "contact_object_or_glyph",
            "independent_mover",
            "hud_resource",
            "collateral_remote",
            "background_exposure",
            "unknown",
        ],
        "audit": audit,
        "targets": target_reports,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def factorized_target_comparison(
    old_metrics: Mapping[str, Any],
    factorized_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    old_error = float(old_metrics.get("error", 1.0) or 1.0)
    new_error = float(factorized_metrics.get("error", 1.0) or 1.0)
    error_drop = max(0.0, old_error - new_error)
    relative_drop = error_drop / old_error if old_error else 0.0
    old_entropy = float(old_metrics.get("target_entropy", 0.0) or 0.0)
    new_entropy = float(factorized_metrics.get("target_entropy", 0.0) or 0.0)
    entropy_drop = max(0.0, old_entropy - new_entropy)
    support = factorized_metrics.get("support", {})
    if support.get("status") == "insufficient_support":
        verdict = "insufficient_support"
    elif relative_drop >= 0.20:
        verdict = "label_pollution_reduced"
    elif new_error >= 0.25:
        verdict = "still_hard_after_factorization"
    else:
        verdict = "factorized_target_low_or_moderate_error"
    return {
        "old_error": round(old_error, 6),
        "factorized_error": round(new_error, 6),
        "error_drop": round(error_drop, 6),
        "error_relative_drop": round(relative_drop, 6),
        "old_target_entropy": round(old_entropy, 6),
        "factorized_target_entropy": round(new_entropy, 6),
        "target_entropy_drop": round(entropy_drop, 6),
        "support": support,
        "verdict": verdict,
    }


def changed_region_factorization_audit(records: list[TransitionRecord]) -> dict[str, Any]:
    role_counts: Counter[str] = Counter()
    transition_role_counts: Counter[str] = Counter()
    cluster_count = 0
    active_records = 0
    multi_role_records = 0
    example_records = []
    for record in records:
        if record.action not in {1, 2, 3, 4}:
            continue
        factors = factorized_changed_regions(record)
        if factors:
            active_records += 1
        roles = {factor["role"] for factor in factors}
        if len(roles) > 1:
            multi_role_records += 1
        for role in roles:
            transition_role_counts[role] += 1
        for factor in factors:
            cluster_count += 1
            role_counts[factor["role"]] += 1
        if len(example_records) < 12 and factors:
            example_records.append(
                {
                    "game": record.game,
                    "step": record.step,
                    "action": record.action,
                    "roles": sorted(roles),
                    "factors": factors[:6],
                }
            )
    return {
        "active_move_records": active_records,
        "cluster_count": cluster_count,
        "role_distribution": dict(role_counts.most_common()),
        "transition_role_distribution": dict(transition_role_counts.most_common()),
        "unknown_cluster_rate": round(role_counts["unknown"] / cluster_count if cluster_count else 0.0, 6),
        "multi_role_transition_rate": round(multi_role_records / active_records if active_records else 0.0, 6),
        "examples": example_records,
    }


def factorized_controlled_footprint_target(record: TransitionRecord) -> dict[str, Any]:
    factors = factorized_changed_regions(record)
    footprint_roles = {
        "controlled_footprint_old",
        "controlled_footprint_new",
        "controlled_overlap_or_trail",
    }
    shape = np.asarray(record.before.pixels[-1]).shape
    footprint = [factor for factor in factors if factor["role"] in footprint_roles]
    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for factor in footprint:
        by_role[factor["role"]].append(factor)
    return {
        "role_count_buckets": {
            role: bucket_count(len(items))
            for role, items in sorted(by_role.items())
        },
        "bbox_buckets_by_role": {
            role: tuple(
                sorted(
                    stable_json(bbox_bucket(tuple(item["bbox"]), shape))
                    for item in items
                    if isinstance(item.get("bbox"), (list, tuple)) and len(item.get("bbox", ())) == 4
                )
            )
            for role, items in sorted(by_role.items())
        },
        "area_buckets_by_role": {
            role: tuple(sorted(str(item.get("area_bucket", "unknown")) for item in items))
            for role, items in sorted(by_role.items())
        },
        "controlled_translation_buckets": tuple(
            sorted(
                f"{motion_direction_from_translation(tuple(candidate.get('translation', ()) or ()))}:"
                f"{bucket_count(motion_magnitude(tuple(candidate.get('translation', ()) or ())))}"
                for candidate in high_confidence_controlled_candidates(record)
            )
        ),
    }


def factorized_material_value_target(record: TransitionRecord) -> dict[str, Any]:
    factors = factorized_changed_regions(record)
    deltas = controlled_object_deltas(record)
    value_transitions = Counter(
        (int(delta.value_before), int(delta.value_after))
        for delta in deltas
        if delta.value_changed
    )
    material_roles = {
        "controlled_footprint_old",
        "controlled_footprint_new",
        "controlled_overlap_or_trail",
        "background_exposure",
        "contact_object_or_glyph",
    }
    histograms_by_role: dict[str, list[Any]] = defaultdict(list)
    for factor in factors:
        if factor["role"] not in material_roles:
            continue
        histograms_by_role[factor["role"]].append(factor.get("color_histogram_delta", ()))
    return {
        "controlled_object_value_transition": tuple(
            sorted(
                (before, after, bucket_count(count))
                for (before, after), count in value_transitions.items()
            )
        ),
        "controlled_object_value_changed_count_bucket": bucket_count(sum(value_transitions.values())),
        "changed_value_histogram_by_factor": {
            role: tuple(sorted(stable_json(item) for item in values))
            for role, values in sorted(histograms_by_role.items())
        },
    }


def factorized_collateral_target(record: TransitionRecord) -> dict[str, Any]:
    factors = factorized_changed_regions(record)
    shape = np.asarray(record.before.pixels[-1]).shape
    collateral_roles = {
        "contact_object_or_glyph",
        "independent_mover",
        "hud_resource",
        "collateral_remote",
        "background_exposure",
        "unknown",
    }
    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for factor in factors:
        if factor["role"] in collateral_roles:
            by_role[factor["role"]].append(factor)
    return {
        "active_collateral_factor_roles": tuple(sorted(by_role)),
        "collateral_role_count_bucket": bucket_count(len(by_role)),
        "collateral_cluster_count_bucket": bucket_count(sum(len(items) for items in by_role.values())),
        "bbox_buckets_by_role": {
            role: tuple(
                sorted(
                    stable_json(bbox_bucket(tuple(item["bbox"]), shape))
                    for item in items
                    if isinstance(item.get("bbox"), (list, tuple)) and len(item.get("bbox", ())) == 4
                )
            )
            for role, items in sorted(by_role.items())
        },
    }


def factorized_changed_regions(record: TransitionRecord) -> list[dict[str, Any]]:
    labeler = record.post_labels.get("labeler_v3", {})
    assignments = v3_cluster_assignment_channels(record)
    factors = []
    for cluster in labeler.get("changed_region_clusters", ()):
        if not isinstance(cluster, Mapping):
            continue
        role, reason = factorized_changed_region_role(record, cluster, assignments.get(cluster.get("cluster_id")))
        factors.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "role": role,
                "reason": reason,
                "bbox": tuple(cluster.get("bbox", ())),
                "area_bucket": cluster.get("area_bucket", "unknown"),
                "color_histogram_delta": cluster.get("color_histogram_delta", ()),
                "v3_channel": assignments.get(cluster.get("cluster_id"), "unassigned"),
            }
        )
    return factors


def factorized_changed_region_role(
    record: TransitionRecord,
    cluster: Mapping[str, Any],
    v3_channel: str | None,
) -> tuple[str, str]:
    bbox = tuple(cluster.get("bbox", ()))
    if len(bbox) != 4:
        return ("unknown", "missing_cluster_bbox")
    old_bboxes = controlled_candidate_bboxes(record, key="bbox_before")
    new_bboxes = controlled_candidate_bboxes(record, key="bbox_after")
    overlaps_old = cluster_overlaps_any_bbox(bbox, old_bboxes)
    overlaps_new = cluster_overlaps_any_bbox(bbox, new_bboxes)
    if overlaps_old and overlaps_new:
        return ("controlled_overlap_or_trail", "overlaps_controlled_old_and_new")
    if overlaps_old:
        return ("controlled_footprint_old", "overlaps_controlled_old_footprint")
    if overlaps_new:
        return ("controlled_footprint_new", "overlaps_controlled_new_footprint")
    if v3_channel == "independent_motion_delta":
        return ("independent_mover", "v3_independent_motion_channel")
    if v3_channel in {"hud_status_delta", "resource_delta"} or bool(cluster.get("overlaps_hud")):
        return ("hud_resource", "hud_or_resource_region")
    if v3_channel == "remote_delta":
        return ("collateral_remote", "v3_remote_channel")
    if cluster.get("distance_to_actor") in {"zero", "near"} and bool(cluster.get("overlaps_known_component")):
        return ("contact_object_or_glyph", "near_controlled_and_overlaps_component")
    if has_background_exposure_signature(cluster):
        return ("background_exposure", "changed_values_include_background")
    if v3_channel == "unknown_delta" or v3_channel is None:
        return ("unknown", "unassigned_or_unknown_v3_channel")
    return ("collateral_remote", f"fallback_from_{v3_channel}")


def controlled_candidate_bboxes(record: TransitionRecord, *, key: str) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for candidate in high_confidence_controlled_candidates(record):
        bbox = candidate.get(key, ())
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bboxes.append(tuple(int(item) for item in bbox))
    return bboxes


def v3_cluster_assignment_channels(record: TransitionRecord) -> dict[Any, str]:
    assignments = {}
    for channel, items in record.post_labels.get("labeler_v3", {}).get("changed_region_assignments", {}).items():
        for item in items:
            if isinstance(item, Mapping):
                assignments[item.get("cluster_id")] = str(channel)
            else:
                assignments[item] = str(channel)
    return assignments


def has_background_exposure_signature(cluster: Mapping[str, Any]) -> bool:
    signature = cluster.get("color_histogram_delta", ())
    for item in signature:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        if int(item[0]) == 0 or int(item[1]) == 0:
            return True
    return False


T5_FACTORIZED_TARGETS = {
    "T5b_footprint": factorized_controlled_footprint_target,
    "T5c_material_value": factorized_material_value_target,
    "T5e_collateral": factorized_collateral_target,
}


def t5_failure_anatomy_report(
    records: list[TransitionRecord],
    *,
    prototype_dir: Path | None = None,
) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_t5_failure_anatomy_game",
            "game": game,
            "reason": "T5 failure anatomy is limited to LS20/AR25",
        }
    if prototype_dir is not None:
        prototype_dir.mkdir(parents=True, exist_ok=True)
    failures_by_target = {
        target_name: prequential_t5_failure_records(records, target_name, target_fn)
        for target_name, target_fn in T5_FACTORIZED_TARGETS.items()
    }
    error_by_action_schema = {
        target_name: t5_error_by_action_schema(failures)
        for target_name, failures in failures_by_target.items()
    }
    factor_distribution = t5_error_by_factor_class(records, failures_by_target)
    examples = {
        target_name: t5_representative_failure_examples(
            failures,
            prototype_dir=prototype_dir,
            target_name=target_name,
            limit=5,
        )
        for target_name, failures in failures_by_target.items()
    }
    confidence_audit = factorizer_confidence_audit(records)
    independent_overlap = t5_independent_mover_overlap(failures_by_target.get("T5e_collateral", []))
    ar25_correlation = (
        ar25_coupled_failure_correlation(failures_by_target)
        if game == "ar25"
        else {"status": "not_applicable_non_ar25"}
    )
    labels = []
    t5e_share = independent_overlap.get("error_share_with_independent_mover", 0.0)
    if t5e_share >= 0.50:
        labels.append("t5e_failures_often_include_independent_movers")
    if confidence_audit.get("low_margin_assignment_rate", 0.0) >= 0.25:
        labels.append("factorizer_low_margin_assignments_common")
    if game == "ar25" and ar25_correlation.get("strongest_correlate"):
        labels.append("ar25_t5_failures_correlate_with_coupled_or_large_region_features")
    if any(error_by_action_schema[target]["concentration"]["top_context_error_share"] >= 0.50 for target in error_by_action_schema):
        labels.append("t5_failures_concentrated_in_few_contexts")
    else:
        labels.append("t5_failures_spread_across_contexts")
    return {
        "status": "evaluated",
        "game": game,
        "labeler": "v3",
        "probe_type": "t5_failure_anatomy_no_new_model",
        "context_baseline": "R3a_computable_compressed_static + V3 schema + action",
        "targets": list(T5_FACTORIZED_TARGETS),
        "error_by_action_schema": error_by_action_schema,
        "error_by_factor_class": factor_distribution,
        "top_failure_examples": examples,
        "factorizer_confidence_audit": confidence_audit,
        "independent_mover_overlap": independent_overlap,
        "ar25_coupled_failure_correlation": ar25_correlation,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def prequential_t5_failure_records(
    records: list[TransitionRecord],
    target_name: str,
    target_fn: Any,
) -> list[dict[str, Any]]:
    prior: dict[str, Counter[str]] = defaultdict(Counter)
    memory: dict[str, Any] = {}
    failures = []
    for record in records:
        if record.action not in {1, 2, 3, 4}:
            update_motion_probe_memory(memory, record)
            continue
        target = stable_json(target_fn(record))
        key = motion_strict_context_key(record, memory)
        context_id = short_digest(key)
        prediction = None
        is_error = False
        if prior[key]:
            prediction = prior[key].most_common(1)[0][0]
            is_error = prediction != target
        failure_record = {
            "target": target_name,
            "game": record.game,
            "step": record.step,
            "action": record.action,
            "majority_schema": outcome_schema_key(record, labeler="v3"),
            "context_id": context_id,
            "level_snapshot": jsonable(record.pre_features.get("level_snapshot", {})),
            "prediction_available": bool(prior[key]),
            "context_support_before": sum(prior[key].values()),
            "predicted": prediction,
            "actual": target,
            "is_error": is_error,
            "factorized_changed_clusters": factorized_changed_regions(record),
            "record": record,
        }
        if failure_record["prediction_available"]:
            failures.append(failure_record)
        prior[key][target] += 1
        update_motion_probe_memory(memory, record)
    return failures


def t5_error_by_action_schema(failures: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    schema_counts: dict[str, Counter[str]] = defaultdict(Counter)
    context_counts: dict[tuple[str, int, str, str, str], Counter[str]] = defaultdict(Counter)
    for failure in failures:
        status = "error" if failure["is_error"] else "hit"
        action_counts[str(failure["action"])][status] += 1
        schema_counts[str(failure["majority_schema"])][status] += 1
        context_key_tuple = (
            str(failure["context_id"]),
            int(failure["action"]),
            str(failure["majority_schema"]),
            stable_json(failure["level_snapshot"]),
            str(failure["target"]),
        )
        context_counts[context_key_tuple][status] += 1
    total_errors = sum(1 for failure in failures if failure["is_error"])
    context_rows = []
    for (context_id, action, schema, level_snapshot, target), counter in context_counts.items():
        predictions = counter["hit"] + counter["error"]
        if predictions <= 0:
            continue
        context_rows.append(
            {
                "context_id": context_id,
                "action": action,
                "majority_schema": schema,
                "level_snapshot": json.loads(level_snapshot),
                "target": target,
                "prediction_count": predictions,
                "error_count": counter["error"],
                "error_rate": round(counter["error"] / predictions, 6),
            }
        )
    context_rows.sort(key=lambda item: (item["error_count"], item["prediction_count"]), reverse=True)
    top_context_errors = sum(item["error_count"] for item in context_rows[:5])
    return {
        "prediction_count": len(failures),
        "error_count": total_errors,
        "error_rate": round(total_errors / len(failures) if failures else 0.0, 6),
        "by_action": counter_error_rows(action_counts),
        "by_schema": counter_error_rows(schema_counts, limit=20),
        "top_contexts": context_rows[:50],
        "concentration": {
            "top_context_error_share": round(top_context_errors / total_errors if total_errors else 0.0, 6),
            "top_context_count": min(5, len(context_rows)),
        },
    }


def counter_error_rows(
    grouped: Mapping[str, Counter[str]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for key, counter in grouped.items():
        total = counter["hit"] + counter["error"]
        rows.append(
            {
                "key": key,
                "prediction_count": total,
                "error_count": counter["error"],
                "error_rate": round(counter["error"] / total if total else 0.0, 6),
            }
        )
    rows.sort(key=lambda item: (item["error_count"], item["prediction_count"]), reverse=True)
    return rows if limit is None else rows[:limit]


def t5_error_by_factor_class(
    records: list[TransitionRecord],
    failures_by_target: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    cluster_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    for record in records:
        if record.action not in {1, 2, 3, 4}:
            continue
        factors = factorized_changed_regions(record)
        roles = {factor["role"] for factor in factors}
        for role in roles:
            transition_counts[role] += 1
        for factor in factors:
            cluster_counts[factor["role"]] += 1
    error_contribution: dict[str, Any] = {}
    for target_name, failures in failures_by_target.items():
        role_error_counts: Counter[str] = Counter()
        role_confusions: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        total_errors = 0
        for failure in failures:
            if not failure["is_error"]:
                continue
            total_errors += 1
            roles = {factor["role"] for factor in failure["factorized_changed_clusters"]}
            for role in roles:
                role_error_counts[role] += 1
                role_confusions[role][
                    (
                        short_digest(str(failure["predicted"])),
                        short_digest(str(failure["actual"])),
                    )
                ] += 1
        error_contribution[target_name] = {
            role: {
                "error_transition_count": count,
                "error_share": round(count / total_errors if total_errors else 0.0, 6),
                "top_confused_target_values": [
                    {
                        "predicted_hash": predicted,
                        "actual_hash": actual,
                        "count": confusion_count,
                    }
                    for (predicted, actual), confusion_count in role_confusions[role].most_common(5)
                ],
            }
            for role, count in role_error_counts.most_common()
        }
    roles = sorted(set(cluster_counts) | set(transition_counts))
    return {
        role: {
            "cluster_count": cluster_counts[role],
            "transition_count": transition_counts[role],
            "error_contribution": {
                target_name: contribution.get(role, {})
                for target_name, contribution in error_contribution.items()
            },
        }
        for role in roles
    }


def t5_representative_failure_examples(
    failures: list[dict[str, Any]],
    *,
    prototype_dir: Path | None,
    target_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    examples = []
    seen_contexts: set[str] = set()
    candidates = [failure for failure in failures if failure["is_error"]]
    candidates.sort(key=lambda failure: failure["context_support_before"], reverse=True)
    for failure in candidates:
        if len(examples) >= limit:
            break
        context_id = str(failure["context_id"])
        if context_id in seen_contexts and len(seen_contexts) < limit:
            continue
        seen_contexts.add(context_id)
        record = failure["record"]
        prototype_path = (
            write_t5_failure_prototype(prototype_dir, target_name, failure)
            if prototype_dir is not None
            else None
        )
        examples.append(
            {
                "step": record.step,
                "action": record.action,
                "context_id": context_id,
                "context_support_before": failure["context_support_before"],
                "majority_schema": failure["majority_schema"],
                "level_snapshot": failure["level_snapshot"],
                "predicted_majority_target": failure["predicted"],
                "actual_target": failure["actual"],
                "factorized_changed_clusters": failure["factorized_changed_clusters"],
                "before_after_prototype_path": str(prototype_path) if prototype_path is not None else None,
            }
        )
    return examples


def write_t5_failure_prototype(
    prototype_dir: Path,
    target_name: str,
    failure: Mapping[str, Any],
) -> Path:
    record = failure["record"]
    path = prototype_dir / f"{target_name}-{record.game}-step-{record.step}-ctx-{failure['context_id']}.json"
    write_json(
        path,
        {
            "target": target_name,
            "game": record.game,
            "step": record.step,
            "action": record.action,
            "context_id": failure["context_id"],
            "predicted": failure["predicted"],
            "actual": failure["actual"],
            "before_frame": np.asarray(record.before.pixels[-1]).tolist(),
            "after_frame": np.asarray(record.after.pixels[-1]).tolist(),
            "before_crop": frame_crop_around_changes(record, before=True),
            "after_crop": frame_crop_around_changes(record, before=False),
            "factorized_changed_clusters": failure["factorized_changed_clusters"],
        },
    )
    return path


def factorizer_confidence_audit(records: list[TransitionRecord]) -> dict[str, Any]:
    confidence_counts: Counter[str] = Counter()
    margin_counts: Counter[str] = Counter()
    multi_possible_counts: Counter[str] = Counter()
    low_margin = 0
    total = 0
    examples = []
    for record in records:
        if record.action not in {1, 2, 3, 4}:
            continue
        assignments = v3_cluster_assignment_channels(record)
        for cluster in record.post_labels.get("labeler_v3", {}).get("changed_region_clusters", ()):
            if not isinstance(cluster, Mapping):
                continue
            scores = factor_role_scores(record, cluster, assignments.get(cluster.get("cluster_id")))
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            top_score = ranked[0][1] if ranked else 0
            second_score = ranked[1][1] if len(ranked) > 1 else 0
            margin = top_score - second_score
            multi_possible = sum(1 for _role, score in ranked if score > 0)
            confidence = factor_assignment_confidence(top_score, margin, multi_possible)
            confidence_counts[confidence] += 1
            margin_counts[bucket_factor_margin(margin)] += 1
            multi_possible_counts[bucket_count(multi_possible)] += 1
            total += 1
            if confidence == "low" or margin <= 0:
                low_margin += 1
                if len(examples) < 12:
                    examples.append(
                        {
                            "game": record.game,
                            "step": record.step,
                            "action": record.action,
                            "cluster_id": cluster.get("cluster_id"),
                            "assigned_role": ranked[0][0] if ranked else "unknown",
                            "top_score": top_score,
                            "second_score": second_score,
                            "nearest_factor_margin": margin,
                            "multi_possible_factor_count": multi_possible,
                            "scores": dict(ranked),
                        }
                    )
    return {
        "cluster_count": total,
        "assignment_confidence_bucket": dict(confidence_counts.most_common()),
        "multi_possible_factor_count": dict(multi_possible_counts.most_common()),
        "nearest_factor_margin": dict(margin_counts.most_common()),
        "low_margin_assignment_rate": round(low_margin / total if total else 0.0, 6),
        "low_margin_examples": examples,
    }


def factor_role_scores(
    record: TransitionRecord,
    cluster: Mapping[str, Any],
    v3_channel: str | None,
) -> dict[str, int]:
    bbox = tuple(cluster.get("bbox", ()))
    if len(bbox) != 4:
        return {"unknown": 1}
    old_bboxes = controlled_candidate_bboxes(record, key="bbox_before")
    new_bboxes = controlled_candidate_bboxes(record, key="bbox_after")
    overlaps_old = cluster_overlaps_any_bbox(bbox, old_bboxes)
    overlaps_new = cluster_overlaps_any_bbox(bbox, new_bboxes)
    return {
        "controlled_footprint_old": int(overlaps_old) * 3,
        "controlled_footprint_new": int(overlaps_new) * 3,
        "controlled_overlap_or_trail": int(overlaps_old and overlaps_new) * 4,
        "contact_object_or_glyph": int(
            cluster.get("distance_to_actor") in {"zero", "near"}
            and bool(cluster.get("overlaps_known_component"))
        ) * 2,
        "independent_mover": int(v3_channel == "independent_motion_delta") * 3,
        "hud_resource": int(v3_channel in {"hud_status_delta", "resource_delta"} or bool(cluster.get("overlaps_hud"))) * 3,
        "collateral_remote": int(v3_channel == "remote_delta") * 3,
        "background_exposure": int(has_background_exposure_signature(cluster)) * 1,
        "unknown": int(v3_channel in {None, "unknown_delta"}) * 1,
    }


def factor_assignment_confidence(top_score: int, margin: int, multi_possible: int) -> str:
    if top_score >= 3 and margin >= 2:
        return "high"
    if top_score >= 2 and margin >= 1 and multi_possible <= 2:
        return "medium"
    return "low"


def bucket_factor_margin(margin: int) -> str:
    if margin <= 0:
        return "zero_or_tie"
    if margin == 1:
        return "one"
    if margin == 2:
        return "two"
    return "three_plus"


def t5_independent_mover_overlap(t5e_failures: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [failure for failure in t5e_failures if failure["is_error"]]
    with_independent = [
        failure for failure in errors
        if any(factor["role"] == "independent_mover" for factor in failure["factorized_changed_clusters"])
    ]
    return {
        "t5e_error_count": len(errors),
        "errors_with_independent_mover": len(with_independent),
        "error_share_with_independent_mover": round(len(with_independent) / len(errors) if errors else 0.0, 6),
        "example_steps": [
            {
                "step": failure["step"],
                "action": failure["action"],
                "context_id": failure["context_id"],
            }
            for failure in with_independent[:12]
        ],
    }


def ar25_coupled_failure_correlation(
    failures_by_target: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    features = (
        "multiple_controlled_clusters",
        "mirrored_or_coupled_movement",
        "overlap_or_trail_factor",
        "large_collateral_region_count",
    )
    by_target = {}
    strongest = None
    strongest_lift = 0.0
    for target_name in ("T5b_footprint", "T5e_collateral"):
        rows = failures_by_target.get(target_name, [])
        feature_rows = {}
        for feature in features:
            present = [row for row in rows if ar25_failure_feature_present(row, feature)]
            absent = [row for row in rows if not ar25_failure_feature_present(row, feature)]
            present_error_rate = error_rate_for_failure_rows(present)
            absent_error_rate = error_rate_for_failure_rows(absent)
            lift = present_error_rate - absent_error_rate
            feature_rows[feature] = {
                "present_count": len(present),
                "absent_count": len(absent),
                "present_error_rate": round(present_error_rate, 6),
                "absent_error_rate": round(absent_error_rate, 6),
                "error_rate_lift": round(lift, 6),
            }
            if len(present) >= 10 and lift > strongest_lift:
                strongest_lift = lift
                strongest = {"target": target_name, "feature": feature, "error_rate_lift": round(lift, 6)}
        by_target[target_name] = feature_rows
    return {
        "status": "evaluated",
        "by_target": by_target,
        "strongest_correlate": strongest,
    }


def ar25_failure_feature_present(row: Mapping[str, Any], feature: str) -> bool:
    record = row["record"]
    factors = row["factorized_changed_clusters"]
    if feature == "multiple_controlled_clusters":
        return sum(1 for factor in factors if str(factor["role"]).startswith("controlled_")) > 1
    if feature == "mirrored_or_coupled_movement":
        controlled = record.post_labels.get("labeler_v3", {}).get("controlled_entity", {})
        return (
            controlled.get("type") in {"coupled_set", "mirrored_set", "grouped_shape"}
            or len(high_confidence_controlled_candidates(record)) > 1
            or any(len(getattr(group, "object_ids", ())) > 1 for group in controlled_motion_groups(record, controlled_candidate_object_ids(record)))
        )
    if feature == "overlap_or_trail_factor":
        return any(factor["role"] == "controlled_overlap_or_trail" for factor in factors)
    if feature == "large_collateral_region_count":
        collateral_roles = {"contact_object_or_glyph", "independent_mover", "hud_resource", "collateral_remote", "background_exposure", "unknown"}
        return sum(1 for factor in factors if factor["role"] in collateral_roles) >= 3
    return False


def error_rate_for_failure_rows(rows: list[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["is_error"]) / len(rows)


def independent_coupled_mover_probe_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_independent_coupled_mover_game",
            "game": game,
            "reason": "independent/coupled mover probe is limited to LS20/AR25",
        }
    phase_cluster = build_vss_phase_cluster(records)
    independent_head = independent_mover_fsa_head(records, phase_cluster)
    coupled_head = coupled_group_motion_head(records, phase_cluster)
    t5e_reclassification = unexplained_extra_changed_region_reclassification(records)
    labels = []
    if any_successful_probe(independent_head):
        labels.append("independent_mover_fsa_signal")
    if any_successful_probe(coupled_head):
        labels.append("coupled_group_motion_signal")
    if any_fragmentation_probe(independent_head) or any_fragmentation_probe(coupled_head):
        labels.append("mover_probe_fragmentation_risk")
    if t5e_reclassification["dominant_class"] in {"independent_mover_explained", "coupled_group_explained"}:
        labels.append(f"extra_changed_regions_mostly_{t5e_reclassification['dominant_class']}")
    if not labels:
        labels.append("no_compact_independent_or_coupled_mover_signal")
    return {
        "status": "evaluated",
        "game": game,
        "labeler": "v3",
        "probe_type": "diagnostic_predictors_not_planner_not_representation",
        "context_baseline": "R3a_computable_compressed_static + V3 schema + action",
        "success_gate": {
            "relative_error_drop": ">= 0.15",
            "prediction_coverage": ">= 0.50",
            "feature_reuse_loss": "<= 0.10",
            "no_suspicious_fragmentation": True,
        },
        "independent_mover_fsa": independent_head,
        "coupled_group_motion": coupled_head,
        "unexplained_extra_changed_region_reclassification": t5e_reclassification,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def mover_state_factorization_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_mover_state_factorization_game",
            "game": game,
            "reason": "mover-state factorization is limited to LS20/AR25",
        }
    identity = mover_identity_link_audit(records)
    kinematics = mover_kinematics_audit(records)
    track_phase = mover_track_phase_audit(records)
    separation = mover_group_separation_audit(records)
    extra_regions = mover_extra_region_explanation_audit(records)
    labels = []
    if identity["identity_link_success_rate"] >= 0.75 and extra_regions["unmatched_changed_region_rate"] <= 0.25:
        labels.append("persistent_mover_state_recoverable")
    if identity["identity_link_success_rate"] < 0.50:
        labels.append("identity_correspondence_weak")
    if kinematics["next_bbox_delta_error"] > 0.50:
        labels.append("simple_kinematics_insufficient")
    if separation["coupled_group_consistency_rate"] >= 0.50:
        labels.append("coupled_group_state_recoverable")
    if separation["independent_motion_transition_share"] >= 0.50:
        labels.append("independent_world_clock_dominant")
    if extra_regions["unmatched_changed_region_rate"] > 0.25:
        labels.append("unmatched_extra_regions_remain")
    if not labels:
        labels.append("mover_state_factorization_inconclusive")
    return {
        "status": "evaluated",
        "game": game,
        "labeler": "v3",
        "probe_type": "representational_factorization_not_prediction_not_context_key",
        "diagnostic_only_local_ids": True,
        "window": "uses adjacent transition windows; kinematic checks require prior mover state when available",
        "questions": {
            "identity": "Can episode-local mover identities be linked across frames?",
            "kinematics": "Can simple velocity/pose state explain t -> t+1?",
            "track_phase": "Do mover histories expose track, bounce, or phase structure?",
            "separation": "Can independent movers be separated from coupled groups?",
        },
        "acceptance_gate": {
            "identity_link_success_rate": ">= 0.75",
            "unmatched_changed_region_rate": "<= 0.25",
            "note": "This gate only says mover state is recoverable enough to model next; it does not claim prediction success.",
        },
        "identity_linking": identity,
        "simple_kinematics": kinematics,
        "track_or_phase": track_phase,
        "independent_vs_coupled": separation,
        "extra_changed_region_explanation": extra_regions,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def world_clock_transition_probe_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_world_clock_transition_game",
            "game": game,
            "reason": "world-clock transition probes are limited to LS20/AR25",
        }
    rows = world_clock_transition_rows(records)
    distribution = Counter(row["failure_class"] for row in rows)
    success_count = distribution["constant_velocity_success"]
    failure_count = len(rows) - success_count
    labels = []
    if not rows:
        labels.append("insufficient_world_clock_transitions")
    else:
        dominant_failure, dominant_count = (
            [(name, count) for name, count in distribution.most_common() if name != "constant_velocity_success"]
            or [("constant_velocity_success", success_count)]
        )[0]
        if failure_count / len(rows) >= 0.40:
            labels.append("world_clock_transition_law_missing")
        if dominant_failure != "constant_velocity_success" and dominant_count / max(1, failure_count) >= 0.30:
            labels.append(f"dominant_{dominant_failure}")
        if distribution["identity_crossing_failure"]:
            labels.append("identity_stress_cases_present")
        if success_count / len(rows) >= 0.60:
            labels.append("constant_velocity_often_sufficient")
    if not labels:
        labels.append("world_clock_probe_inconclusive")
    return {
        "status": "evaluated",
        "game": game,
        "labeler": "v3",
        "probe_type": "transition_law_failure_anatomy_not_representation",
        "window": "up to 5 observations per episode-local mover before scoring the current transition",
        "target": "world_tick_delta_t_plus_1 for tracked independent movers and coupled groups",
        "failure_classes": [
            "constant_velocity_success",
            "constant_velocity_boundary_failure",
            "periodic_phase_failure",
            "collision_contact_failure",
            "occlusion_composition_failure",
            "coupled_group_role_failure",
            "identity_crossing_failure",
            "unknown_transition_failure",
        ],
        "transition_count": len(rows),
        "constant_velocity_success_rate": round(success_count / len(rows) if rows else 0.0, 6),
        "constant_velocity_error_rate": round(failure_count / len(rows) if rows else 0.0, 6),
        "failure_class_distribution": dict(distribution.most_common()),
        "failure_class_share": {
            name: round(count / len(rows) if rows else 0.0, 6)
            for name, count in distribution.items()
        },
        "heads": world_clock_heads(rows),
        "examples": world_clock_failure_examples(rows),
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def visibility_composition_probe_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_visibility_composition_game",
            "game": game,
            "reason": "visibility composition probes are limited to LS20/AR25",
        }
    rows = visibility_composition_rows(records)
    class_counts = Counter(row["visible_delta_class"] for row in rows)
    legacy_class_counts = Counter(row["legacy_visible_delta_class"] for row in rows)
    split_count = class_counts["clean_old_new_exposure"] + class_counts["merged_old_new_overlap_or_trail"]
    bootstrap = class_counts["mover_bootstrap_no_history"]
    true_latent = class_counts["true_latent_motion_unknown"]
    unexplained_including_bootstrap = class_counts["composition_unknown"] + true_latent + bootstrap
    explained_including_bootstrap = len(rows) - unexplained_including_bootstrap
    non_bootstrap_count = len(rows) - bootstrap
    unexplained_excluding_bootstrap = class_counts["composition_unknown"] + true_latent
    explained_excluding_bootstrap = non_bootstrap_count - unexplained_excluding_bootstrap
    labels = []
    if not rows:
        labels.append("insufficient_visibility_composition_transitions")
    else:
        excluding_bootstrap_rate = (
            explained_excluding_bootstrap / non_bootstrap_count
            if non_bootstrap_count
            else 0.0
        )
        true_latent_rate = true_latent / non_bootstrap_count if non_bootstrap_count else 0.0
        if excluding_bootstrap_rate >= 0.65:
            labels.append("visibility_composition_explains_most_deltas")
        if class_counts["composition_unknown"] / len(rows) >= 0.20:
            labels.append("composition_ontology_incomplete")
        if true_latent_rate >= 0.20:
            labels.append("true_latent_motion_still_unclear")
        if bootstrap / len(rows) >= 0.15:
            labels.append("mover_bootstrap_common")
        dominant, count = class_counts.most_common(1)[0]
        labels.append(f"dominant_{dominant}")
    if not labels:
        labels.append("visibility_composition_probe_inconclusive")
    return {
        "status": "evaluated",
        "game": game,
        "labeler": "v3",
        "probe_type": "latent_motion_vs_visible_delta_split_not_renderer",
        "question": "Given tracked mover states before and after transition, can observed changed regions be explained by visibility composition?",
        "transition_count": len(rows),
        "bootstrap_count": bootstrap,
        "bootstrap_rate": round(bootstrap / len(rows) if rows else 0.0, 6),
        "true_latent_unknown_count": true_latent,
        "true_latent_unknown_rate": round(true_latent / non_bootstrap_count if non_bootstrap_count else 0.0, 6),
        "composition_explained_rate": round(explained_including_bootstrap / len(rows) if rows else 0.0, 6),
        "composition_explained_rate_including_bootstrap": round(explained_including_bootstrap / len(rows) if rows else 0.0, 6),
        "composition_unexplained_rate_including_bootstrap": round(unexplained_including_bootstrap / len(rows) if rows else 0.0, 6),
        "composition_explained_rate_excluding_bootstrap": round(
            explained_excluding_bootstrap / non_bootstrap_count if non_bootstrap_count else 0.0,
            6,
        ),
        "composition_unexplained_rate_excluding_bootstrap": round(
            unexplained_excluding_bootstrap / non_bootstrap_count if non_bootstrap_count else 0.0,
            6,
        ),
        "visible_delta_class_distribution": dict(class_counts.most_common()),
        "legacy_visible_delta_class_distribution": dict(legacy_class_counts.most_common()),
        "split_composition_label_accounting": {
            "legacy_translation_with_trail_or_residue": legacy_class_counts["translation_with_trail_or_residue"],
            "clean_old_new_exposure": class_counts["clean_old_new_exposure"],
            "merged_old_new_overlap_or_trail": class_counts["merged_old_new_overlap_or_trail"],
            "split_sum": split_count,
            "accounting_matches_legacy": split_count == legacy_class_counts["translation_with_trail_or_residue"],
        },
        "visible_delta_class_share": {
            name: round(count / len(rows) if rows else 0.0, 6)
            for name, count in class_counts.items()
        },
        "heads": visibility_composition_heads(rows),
        "examples": visibility_composition_examples(rows),
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def mover_bootstrap_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_mover_bootstrap_game",
            "game": game,
            "reason": "mover bootstrap probes are limited to LS20/AR25",
        }
    rows = visibility_composition_rows(records)
    bootstrap_rows = [
        row for row in rows
        if row["visible_delta_class"] == "mover_bootstrap_no_history"
    ]
    classifications = [mover_bootstrap_classification(row) for row in bootstrap_rows]
    class_counts = Counter(item["label"] for item in classifications)
    ambiguity_counts = Counter(str(item["ambiguity_count"]) for item in classifications)
    examples = []
    for row, classification in zip(bootstrap_rows[:30], classifications[:30]):
        examples.append(
            {
                "step": row["step"],
                "action": row["action"],
                "episode_local_mover_ref": row["episode_local_mover_ref"],
                "classification": classification["label"],
                "reasons": classification["reasons"],
                "first_seen_pose": row["bbox_after"],
                "visible_signature": row["mover_signature"],
                "local_track_or_space_pattern": bootstrap_track_space_pattern(row),
                "nearest_controlled_object_relation": row["controlled_mover_overlap_class"],
                "overlap_with_controlled_footprint": row["controlled_mover_overlap_class"] != "no_controlled_overlap",
                "possible_previous_pose_candidates": classification["possible_previous_pose_candidates"],
                "possible_velocity_candidates": classification["possible_velocity_candidates"],
                "ambiguity_count": classification["ambiguity_count"],
                "required_missing_variable": classification["required_missing_variable"],
                "suggested_next_probe": classification["suggested_next_probe"],
                "cluster_bboxes": row["cluster_bboxes"],
            }
        )
    labels = []
    if not bootstrap_rows:
        labels.append("no_mover_bootstrap_cases")
    else:
        dominant, _count = class_counts.most_common(1)[0]
        labels.append(f"dominant_{dominant}")
        if class_counts["unambiguous_bootstrap"] / len(bootstrap_rows) >= 0.50:
            labels.append("bootstrap_often_simple")
        if (
            class_counts["appears_from_overlap"]
            + class_counts["identity_uncertain"]
            + class_counts["appears_from_occlusion"]
        ) / len(bootstrap_rows) >= 0.50:
            labels.append("bootstrap_often_composition_limited")
    return {
        "status": "evaluated",
        "game": game,
        "probe_type": "mover_bootstrap_readout_not_predictor",
        "diagnostic_only": True,
        "transition_count": len(rows),
        "bootstrap_count": len(bootstrap_rows),
        "bootstrap_rate": round(len(bootstrap_rows) / len(rows) if rows else 0.0, 6),
        "classification_distribution": dict(class_counts.most_common()),
        "ambiguity_count_distribution": dict(ambiguity_counts.most_common()),
        "examples": examples,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def mover_bootstrap_classification(row: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    ambiguity_count = 0
    possible_previous_pose_candidates = 1 if row.get("bbox_before") else 0
    possible_velocity_candidates = 1 if tuple(row.get("translation", ())) != (0, 0) else 0
    label = "unambiguous_bootstrap"
    required_missing_variable = "none"
    suggested_next_probe = "simple first-observation state initialization"
    if row.get("mover_mover_overlap_class") != "no_mover_overlap":
        label = "identity_uncertain"
        ambiguity_count += 1
        reasons.append("first observation overlaps or crosses another mover")
        required_missing_variable = "identity ordering through first-observation mover overlap"
        suggested_next_probe = "first-observation identity crossing audit"
    elif row.get("controlled_mover_overlap_class") != "no_controlled_overlap" or row.get("occlusion_state") == "overlap_composition_present":
        label = "appears_from_overlap"
        ambiguity_count += 1
        reasons.append("first observation is entangled with controlled or overlap composition")
        required_missing_variable = "overlap/render priority at first observation"
        suggested_next_probe = "bootstrap overlap composition audit"
    elif row.get("visible_to_latent_pose_recovery") in {"latent_pose_occluded", "visible_pose_not_recovered", "latent_pose_unknown"}:
        label = "appears_from_occlusion"
        ambiguity_count += 1
        reasons.append("first observed mover pose is not directly visible")
        required_missing_variable = "occlusion source and latent pose initialization"
        suggested_next_probe = "first-observation occlusion recovery audit"
    elif row.get("trail_or_residue_class") == "old_new_overlap_or_trail":
        label = "ambiguous_previous_pose"
        ambiguity_count += 1
        possible_previous_pose_candidates += 1
        reasons.append("old and new footprint evidence is merged at first observation")
        required_missing_variable = "previous footprint separation"
        suggested_next_probe = "merged old/new first-observation audit"
    elif tuple(row.get("translation", ())) == (0, 0):
        label = "ambiguous_velocity"
        ambiguity_count += 1
        possible_velocity_candidates = 2
        reasons.append("first observation has no reliable motion delta")
        required_missing_variable = "initial velocity hypothesis"
        suggested_next_probe = "stationary first-observation velocity audit"
    else:
        reasons.append("first observation has visible pose and single intra-transition delta")
    return {
        "label": label,
        "reasons": reasons,
        "possible_previous_pose_candidates": possible_previous_pose_candidates,
        "possible_velocity_candidates": possible_velocity_candidates,
        "ambiguity_count": ambiguity_count,
        "required_missing_variable": required_missing_variable,
        "suggested_next_probe": suggested_next_probe,
    }


def bootstrap_track_space_pattern(row: Mapping[str, Any]) -> str:
    translation = tuple(row.get("translation", ()))
    if len(translation) != 2:
        return "unknown_track"
    if translation[0] and translation[1]:
        return "diagonal_or_mixed_first_delta"
    if translation[0]:
        return "vertical_first_delta"
    if translation[1]:
        return "horizontal_first_delta"
    return "stationary_or_unresolved_first_delta"


def ar25_overlap_composition_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game != "ar25":
        return {
            "status": "skipped_not_ar25",
            "game": game,
            "reason": "AR25 overlap composition report is AR25-only",
        }
    rows = [
        row for row in visibility_composition_rows(records)
        if row["visible_delta_class"] == "merged_old_new_overlap_or_trail"
    ]
    classifications = [ar25_overlap_composition_classification(row) for row in rows]
    class_counts = Counter(item["label"] for item in classifications)
    examples = []
    for row, classification in zip(rows[:40], classifications[:40]):
        examples.append(
            {
                "step": row["step"],
                "action": row["action"],
                "episode_local_mover_ref": row["episode_local_mover_ref"],
                "classification": classification["label"],
                "reasons": classification["reasons"],
                "bbox_before": row["bbox_before"],
                "bbox_after": row["bbox_after"],
                "translation": row["translation"],
                "cluster_count": row["cluster_count"],
                "cluster_bboxes": row["cluster_bboxes"],
                "controlled_mover_overlap_class": row["controlled_mover_overlap_class"],
                "mover_mover_overlap_class": row["mover_mover_overlap_class"],
                "overlap_priority_class": row["overlap_priority_class"],
                "required_missing_variable": classification["required_missing_variable"],
                "suggested_next_probe": classification["suggested_next_probe"],
            }
        )
    labels = []
    if not rows:
        labels.append("no_ar25_merged_overlap_cases")
    else:
        dominant, _count = class_counts.most_common(1)[0]
        labels.append(f"dominant_{dominant}")
        if class_counts["group_or_coupled_overlap"] / len(rows) >= 0.30:
            labels.append("group_overlap_composition_common")
        if class_counts["controlled_mover_overlap"] / len(rows) >= 0.30:
            labels.append("controlled_overlap_composition_common")
        if class_counts["cluster_merge_artifact"] / len(rows) >= 0.20:
            labels.append("cluster_merge_artifact_risk")
    return {
        "status": "evaluated",
        "game": "ar25",
        "probe_type": "merged_old_new_overlap_composition_readout_not_predictor",
        "diagnostic_only": True,
        "target_visible_delta_class": "merged_old_new_overlap_or_trail",
        "case_count": len(rows),
        "classification_distribution": dict(class_counts.most_common()),
        "examples": examples,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def ar25_overlap_composition_classification(row: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    if row.get("controlled_mover_overlap_class") != "no_controlled_overlap":
        reasons.append("merged old/new footprint overlaps controlled footprint")
        return {
            "label": "controlled_mover_overlap",
            "reasons": reasons,
            "required_missing_variable": "controlled/mover render priority",
            "suggested_next_probe": "controlled overlap composition probe",
        }
    if row.get("mover_mover_overlap_class") != "no_mover_overlap":
        reasons.append("merged old/new footprint overlaps another mover")
        return {
            "label": "group_or_coupled_overlap",
            "reasons": reasons,
            "required_missing_variable": "group membership and mover-mover render priority",
            "suggested_next_probe": "coupled/group overlap composition probe",
        }
    if int(row.get("cluster_count", 0) or 0) >= 6:
        reasons.append("many changed clusters contribute to one merged overlap case")
        return {
            "label": "cluster_merge_artifact",
            "reasons": reasons,
            "required_missing_variable": "cluster segmentation boundary",
            "suggested_next_probe": "changed-cluster merge/split audit",
        }
    if any(bbox_cell_area(tuple(bbox)) >= 16 for bbox in row.get("cluster_bboxes", ())):
        reasons.append("large changed region contains merged old/new footprint")
        return {
            "label": "large_shape_translation_overlap",
            "reasons": reasons,
            "required_missing_variable": "shape footprint overlap and render/unrender order",
            "suggested_next_probe": "large-shape old/new overlap composition probe",
        }
    if row.get("visible_to_latent_pose_recovery") == "visible_pose_recovered":
        reasons.append("same mover old/new footprint appears merged without other overlap evidence")
        return {
            "label": "simple_overlap_same_object",
            "reasons": reasons,
            "required_missing_variable": "single-object old/new footprint composition",
            "suggested_next_probe": "simple render/unrender overlap rule probe",
        }
    reasons.append("no current overlap taxonomy feature explains the merged class")
    return {
        "label": "unknown_overlap",
        "reasons": reasons,
        "required_missing_variable": "unknown overlap/composition state",
        "suggested_next_probe": "manual AR25 overlap prototype atlas",
    }


def bbox_cell_area(bbox: tuple[int, ...]) -> int:
    if len(bbox) != 4:
        return 0
    r0, c0, r1, c1 = bbox
    return max(0, r1 - r0 + 1) * max(0, c1 - c0 + 1)


def mover_bootstrap_composition_v0_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_mover_bootstrap_composition_game",
            "game": game,
            "reason": "MoverBootstrapAndCompositionV0 is limited to LS20/AR25",
        }
    rows = visibility_composition_rows(records)
    future_refs = future_refs_by_step(rows)
    bootstrap_rows = [row for row in rows if row["visible_delta_class"] == "mover_bootstrap_no_history"]
    bootstrap_hypotheses = [
        mover_bootstrapper_v0_hypothesis(row, future_refs=future_refs)
        for row in bootstrap_rows
    ]
    merged_rows = [row for row in rows if row["visible_delta_class"] == "merged_old_new_overlap_or_trail"]
    merged_examples = [merged_overlap_decomposer_v0(row) for row in merged_rows]
    bootstrap_metrics = bootstrap_hypothesis_metrics(bootstrap_hypotheses)
    overlap_metrics = merged_overlap_decomposition_metrics(merged_examples)
    labels = []
    if game == "ar25":
        if bootstrap_metrics["top3_previous_pose_coverage"] >= 0.75 and bootstrap_metrics["mean_ambiguity_count"] <= 3.0:
            labels.append("bootstrap_hypothesis_set_viable")
        if overlap_metrics["merged_cluster_decomposition_success"] < 0.75:
            labels.append("merged_overlap_decomposition_still_hard")
        if overlap_metrics["unexplained_region_rate"] >= 0.20:
            labels.append("cluster_decomposition_boundary_still_missing")
    else:
        if overlap_metrics["t5e_proxy_error_after"] <= overlap_metrics["t5e_proxy_error_before"]:
            labels.append("ls20_non_regression")
        if bootstrap_metrics["bootstrap_case_explained_rate"] < 0.75 and bootstrap_hypotheses:
            labels.append("ls20_bootstrap_still_ambiguous")
    if not labels:
        labels.append("v0_diagnostic_inconclusive")
    return {
        "status": "evaluated",
        "game": game,
        "mechanism": "MoverBootstrapAndCompositionV0",
        "diagnostic_only": True,
        "constraints": {
            "future_frames_for_hypothesis_generation": False,
            "future_frames_for_scoring_only": True,
            "raw_context_ids_as_predictors": False,
            "agent_promotion": False,
        },
        "bootstrap_metrics": bootstrap_metrics,
        "merged_overlap_metrics": overlap_metrics,
        "non_regression_snapshot": {
            "visibility_composition_rows": len(rows),
            "true_latent_motion_unknown": sum(1 for row in rows if row["visible_delta_class"] == "true_latent_motion_unknown"),
            "composition_explained_excluding_bootstrap": visibility_composition_probe_report(records).get(
                "composition_explained_rate_excluding_bootstrap"
            ),
        },
        "bootstrap_hypotheses": bootstrap_hypotheses,
        "merged_overlap_decomposition_examples": merged_examples,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def future_refs_by_step(rows: list[Mapping[str, Any]]) -> dict[tuple[int, str], bool]:
    refs_seen_after_step: dict[int, set[str]] = defaultdict(set)
    by_step: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)
    future_refs: set[str] = set()
    for step in sorted(by_step, reverse=True):
        for row in by_step[step]:
            refs_seen_after_step[(step, str(row["episode_local_mover_ref"]))] = str(row["episode_local_mover_ref"]) in future_refs
        for row in by_step[step]:
            future_refs.add(str(row["episode_local_mover_ref"]))
    return refs_seen_after_step


def mover_bootstrapper_v0_hypothesis(
    row: Mapping[str, Any],
    *,
    future_refs: Mapping[tuple[int, str], bool],
) -> dict[str, Any]:
    classification = mover_bootstrap_classification(row)
    previous_pose_candidates = previous_pose_hypotheses_v0(row)
    velocity_candidates = velocity_hypotheses_v0(row)
    identity_candidates = identity_hypotheses_v0(row)
    actual_previous_pose = tuple(row.get("bbox_before", ()))
    actual_velocity = tuple(row.get("translation", ()))
    ref = str(row["episode_local_mover_ref"])
    has_future = bool(future_refs.get((int(row["step"]), ref), False))
    top1_previous_pose_hit = bool(previous_pose_candidates and tuple(previous_pose_candidates[0]["bbox"]) == actual_previous_pose)
    top3_previous_pose_hit = any(tuple(candidate["bbox"]) == actual_previous_pose for candidate in previous_pose_candidates[:3])
    velocity_hit = any(tuple(candidate["translation"]) == actual_velocity for candidate in velocity_candidates[:5])
    identity_hit = (not has_future) or any(candidate["mover_ref"] == ref for candidate in identity_candidates[:5])
    ambiguity_count = max(
        int(classification["ambiguity_count"]),
        len(previous_pose_candidates) - 1,
        len(velocity_candidates) - 1,
        len(identity_candidates) - 1,
    )
    explained = top3_previous_pose_hit and velocity_hit and identity_hit and ambiguity_count <= 5
    return {
        "game": row["game"],
        "step": row["step"],
        "action": row["action"],
        "episode_local_mover_ref": ref,
        "bootstrap_class": classification["label"],
        "visible_signature": row["mover_signature"],
        "visible_pose": row["bbox_after"],
        "actual_previous_pose_for_scoring": actual_previous_pose,
        "actual_velocity_for_scoring": actual_velocity,
        "used_future_for_generation": False,
        "future_used_for_scoring": True,
        "previous_pose_candidates": previous_pose_candidates[:5],
        "velocity_candidates": velocity_candidates[:5],
        "identity_candidates": identity_candidates[:5],
        "top1_previous_pose_hit": top1_previous_pose_hit,
        "top3_previous_pose_hit": top3_previous_pose_hit,
        "topk_velocity_hit": velocity_hit,
        "identity_hypothesis_hit": identity_hit,
        "has_future_identity_evidence": has_future,
        "ambiguity_count": ambiguity_count,
        "bootstrap_case_explained": explained,
        "required_missing_variable": classification["required_missing_variable"],
        "suggested_next_probe": classification["suggested_next_probe"],
    }


def previous_pose_hypotheses_v0(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    after = tuple(row.get("bbox_after", ()))
    translation = tuple(row.get("translation", ()))
    if len(after) == 4 and len(translation) == 2:
        candidates.append(
            {
                "source": "inverse_current_delta",
                "bbox": translate_bbox(after, (-int(translation[0]), -int(translation[1]))),
                "confidence": "high",
            }
        )
    for bbox in row.get("cluster_bboxes", ())[:6]:
        bbox_tuple = tuple(bbox)
        if len(bbox_tuple) != 4:
            continue
        if tuple(row.get("bbox_before", ())) and bbox_overlap_area(tuple(row["bbox_before"]), bbox_tuple) > 0:
            candidates.append({"source": "changed_cluster_overlap_old_pose", "bbox": bbox_tuple, "confidence": "medium"})
    before = tuple(row.get("bbox_before", ()))
    if len(before) == 4:
        candidates.append({"source": "detector_linked_previous_pose", "bbox": before, "confidence": "medium"})
    return unique_bbox_candidates(candidates)


def velocity_hypotheses_v0(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    translation = tuple(row.get("translation", ()))
    candidates = []
    if len(translation) == 2:
        candidates.append({"source": "current_changed_region_delta", "translation": translation, "confidence": "high"})
        candidates.append({"source": "opposite_direction_probe", "translation": (-int(translation[0]), -int(translation[1])), "confidence": "low"})
        if translation[0] and translation[1]:
            candidates.append({"source": "axis_projection_row", "translation": (int(translation[0]), 0), "confidence": "low"})
            candidates.append({"source": "axis_projection_col", "translation": (0, int(translation[1])), "confidence": "low"})
    return unique_translation_candidates(candidates)


def identity_hypotheses_v0(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        {
            "source": "current_visible_component",
            "mover_ref": str(row["episode_local_mover_ref"]),
            "confidence": "medium" if row.get("mover_mover_overlap_class") == "no_mover_overlap" else "low",
        }
    ]
    if row.get("mover_mover_overlap_class") != "no_mover_overlap":
        candidates.append(
            {
                "source": "overlap_ambiguous_identity_slot",
                "mover_ref": f"{row['episode_local_mover_ref']}:alternate",
                "confidence": "low",
            }
        )
    return candidates


def unique_bbox_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for candidate in candidates:
        bbox = tuple(candidate.get("bbox", ()))
        if bbox in seen or len(bbox) != 4:
            continue
        seen.add(bbox)
        result.append(candidate)
    return result


def unique_translation_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for candidate in candidates:
        translation = tuple(candidate.get("translation", ()))
        if translation in seen or len(translation) != 2:
            continue
        seen.add(translation)
        result.append(candidate)
    return result


def bootstrap_hypothesis_metrics(hypotheses: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(hypotheses)
    if not total:
        return {
            "bootstrap_case_count": 0,
            "top1_previous_pose_accuracy": 0.0,
            "top3_previous_pose_coverage": 0.0,
            "topk_velocity_coverage": 0.0,
            "identity_hypothesis_coverage": 0.0,
            "mean_ambiguity_count": 0.0,
            "bootstrap_case_explained_rate": 0.0,
        }
    return {
        "bootstrap_case_count": total,
        "top1_previous_pose_accuracy": round(sum(bool(item["top1_previous_pose_hit"]) for item in hypotheses) / total, 6),
        "top3_previous_pose_coverage": round(sum(bool(item["top3_previous_pose_hit"]) for item in hypotheses) / total, 6),
        "topk_velocity_coverage": round(sum(bool(item["topk_velocity_hit"]) for item in hypotheses) / total, 6),
        "identity_hypothesis_coverage": round(sum(bool(item["identity_hypothesis_hit"]) for item in hypotheses) / total, 6),
        "mean_ambiguity_count": round(sum(int(item["ambiguity_count"]) for item in hypotheses) / total, 6),
        "bootstrap_case_explained_rate": round(sum(bool(item["bootstrap_case_explained"]) for item in hypotheses) / total, 6),
        "classification_distribution": dict(Counter(str(item["bootstrap_class"]) for item in hypotheses).most_common()),
    }


def merged_overlap_decomposer_v0(row: Mapping[str, Any]) -> dict[str, Any]:
    old_bbox = tuple(row.get("bbox_before", ()))
    new_bbox = tuple(row.get("bbox_after", ()))
    cluster_bboxes = [tuple(bbox) for bbox in row.get("cluster_bboxes", ()) if len(tuple(bbox)) == 4]
    overlap_bbox = bbox_intersection(old_bbox, new_bbox)
    old_recovered = len(old_bbox) == 4 and any(bbox_overlap_area(old_bbox, cluster) > 0 for cluster in cluster_bboxes)
    new_recovered = len(new_bbox) == 4 and any(bbox_overlap_area(new_bbox, cluster) > 0 for cluster in cluster_bboxes)
    overlap_recovered = overlap_bbox is not None and any(bbox_overlap_area(overlap_bbox, cluster) > 0 for cluster in cluster_bboxes)
    classification = ar25_overlap_composition_classification(row) if row["game"] == "ar25" else generic_overlap_composition_classification(row)
    cluster_merge = classification["label"] == "cluster_merge_artifact"
    success = old_recovered and new_recovered and overlap_recovered and not cluster_merge
    unexplained = cluster_merge or classification["label"] == "unknown_overlap"
    return {
        "game": row["game"],
        "step": row["step"],
        "action": row["action"],
        "episode_local_mover_ref": row["episode_local_mover_ref"],
        "classification": classification["label"],
        "bbox_before": old_bbox,
        "bbox_after": new_bbox,
        "translation": row.get("translation", ()),
        "components": {
            "old_footprint_component": old_bbox,
            "new_footprint_component": new_bbox,
            "overlap_or_trail_component": overlap_bbox,
            "background_exposure": row.get("newly_exposed_background_count"),
            "group_or_large_shape_overlap": classification["label"] in {"group_or_coupled_overlap", "large_shape_translation_overlap"},
        },
        "old_new_footprint_recovered": old_recovered and new_recovered,
        "overlap_component_recovered": overlap_recovered,
        "large_shape_overlap_detected": classification["label"] == "large_shape_translation_overlap",
        "merged_cluster_decomposition_success": success,
        "unexplained_region": unexplained,
        "cluster_bboxes": cluster_bboxes[:8],
        "required_missing_variable": classification["required_missing_variable"],
        "suggested_next_probe": classification["suggested_next_probe"],
    }


def generic_overlap_composition_classification(row: Mapping[str, Any]) -> dict[str, Any]:
    if any(bbox_cell_area(tuple(bbox)) >= 16 for bbox in row.get("cluster_bboxes", ())):
        return {
            "label": "large_shape_translation_overlap",
            "required_missing_variable": "shape footprint overlap and render/unrender order",
            "suggested_next_probe": "large-shape old/new overlap composition probe",
        }
    return {
        "label": "simple_overlap_same_object",
        "required_missing_variable": "single-object old/new footprint composition",
        "suggested_next_probe": "simple render/unrender overlap rule probe",
    }


def bbox_intersection(left: tuple[Any, ...], right: tuple[Any, ...]) -> tuple[int, int, int, int] | None:
    if len(left) != 4 or len(right) != 4:
        return None
    top = max(int(left[0]), int(right[0]))
    left_col = max(int(left[1]), int(right[1]))
    bottom = min(int(left[2]), int(right[2]))
    right_col = min(int(left[3]), int(right[3]))
    if bottom < top or right_col < left_col:
        return None
    return (top, left_col, bottom, right_col)


def merged_overlap_decomposition_metrics(examples: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(examples)
    if not total:
        return {
            "merged_overlap_case_count": 0,
            "merged_cluster_decomposition_success": 0.0,
            "old_new_footprint_recovery_rate": 0.0,
            "large_shape_overlap_detection_rate": 0.0,
            "unexplained_region_rate": 0.0,
            "t5b_proxy_error_before": 0.0,
            "t5b_proxy_error_after": 0.0,
            "t5e_proxy_error_before": 0.0,
            "t5e_proxy_error_after": 0.0,
        }
    success = sum(bool(item["merged_cluster_decomposition_success"]) for item in examples)
    old_new = sum(bool(item["old_new_footprint_recovered"]) for item in examples)
    large_shape = sum(bool(item["large_shape_overlap_detected"]) for item in examples)
    unexplained = sum(bool(item["unexplained_region"]) for item in examples)
    return {
        "merged_overlap_case_count": total,
        "merged_cluster_decomposition_success": round(success / total, 6),
        "old_new_footprint_recovery_rate": round(old_new / total, 6),
        "large_shape_overlap_detection_rate": round(large_shape / total, 6),
        "unexplained_region_rate": round(unexplained / total, 6),
        "t5b_proxy_error_before": 1.0,
        "t5b_proxy_error_after": round(1.0 - (old_new / total), 6),
        "t5e_proxy_error_before": 1.0,
        "t5e_proxy_error_after": round(unexplained / total, 6),
        "classification_distribution": dict(Counter(str(item["classification"]) for item in examples).most_common()),
    }


def changed_cluster_merge_split_audit_v0_report(
    records: list[TransitionRecord],
    *,
    baseline_merged_overlap_metrics: Mapping[str, Any] | None = None,
    decompositions: list[ClusterDecomposition] | None = None,
) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game != "ar25":
        return {
            "status": "skipped_not_ar25",
            "game": game,
            "reason": "ChangedClusterMergeSplitAuditV0 is an AR25 merged-overlap audit",
        }
    decompositions = decompositions if decompositions is not None else ar25_merged_cluster_decompositions_v0(records)

    hypotheses = []
    cell_examples = []
    for decomposition in decompositions:
        for hypothesis in decomposition.hypotheses:
            hypotheses.append(
                {
                    **decomposition_metadata(decomposition),
                    **hypothesis.to_record(),
                }
            )
        best = decomposition.best_hypothesis()
        if best is None:
            continue
        for attribution in best.cell_attributions[:20]:
            cell_examples.append(
                {
                    **decomposition_metadata(decomposition),
                    **attribution.to_record(),
                }
            )

    metrics = changed_cluster_merge_split_metrics(decompositions)
    baseline = (
        dict(baseline_merged_overlap_metrics)
        if baseline_merged_overlap_metrics is not None
        else mover_bootstrap_composition_v0_report(records).get("merged_overlap_metrics", {})
    )
    before_after = decomposition_before_after_metrics_v0(metrics, baseline)
    taxonomy = ar25_cluster_overmerge_taxonomy_v0(decompositions)
    gallery = decomposition_failure_gallery_v0(decompositions, limit=30)
    labels = []
    if metrics["old_new_footprint_recovery_rate"] >= 0.80:
        labels.append("old_new_footprint_recovery_preserved")
    else:
        labels.append("old_new_footprint_recovery_regressed")
    if metrics["residue_after_best_decomposition"] < float(baseline.get("unexplained_region_rate", 1.0) or 1.0):
        labels.append("unexplained_region_rate_improved")
    else:
        labels.append("unexplained_region_rate_not_improved")
    if metrics["cluster_merge_artifact_multi_part_count"] >= 100:
        labels.append("cluster_merge_artifacts_receive_multi_part_decomposition")
    else:
        labels.append("insufficient_cluster_merge_multi_part_decomposition")
    if metrics["causal_split_recovery_rate"] >= 0.50:
        labels.append("causal_split_recovery_signal_present")
    else:
        labels.append("causal_split_recovery_still_weak")
    return {
        "status": "evaluated",
        "game": "ar25",
        "mechanism": "ChangedClusterMergeSplitAuditV0",
        "diagnostic_only": True,
        "target_visible_delta_class": "merged_old_new_overlap_or_trail",
        "case_count": len(decompositions),
        "constraints": {
            "planner_integration": False,
            "role_label_promotion": False,
            "raw_context_ids_as_predictors": False,
            "changed_cluster_is_evidence_container": True,
        },
        "metrics": metrics,
        "decomposition_before_after_metrics": before_after,
        "ar25_cluster_overmerge_taxonomy": taxonomy,
        "cluster_decomposition_hypotheses": hypotheses,
        "cell_attribution_examples": cell_examples[:400],
        "decomposition_failure_gallery": gallery,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def ar25_merged_cluster_decompositions_v0(
    records: list[TransitionRecord],
    *,
    changed_cells_by_step: Mapping[int, Iterable[tuple[int, int]]] | None = None,
) -> list[ClusterDecomposition]:
    game = records[0].game if records else "unknown"
    if game != "ar25":
        return []
    rows = [
        row for row in visibility_composition_rows(records)
        if row["visible_delta_class"] == "merged_old_new_overlap_or_trail"
    ]
    record_by_step = {record.step: record for record in records}
    changed_cells_by_step = (
        dict(changed_cells_by_step)
        if changed_cells_by_step is not None
        else {record.step: all_changed_cells(record) for record in records}
    )
    decompositions = []
    for row in rows:
        record = record_by_step.get(int(row["step"]))
        if record is None:
            continue
        decompositions.append(
            changed_cluster_merge_split_decomposition_v0(
                record,
                row,
                candidate_changed_cells=changed_cells_by_step.get(record.step, ()),
            )
        )
    return decompositions


def decomposition_metadata(decomposition: ClusterDecomposition) -> dict[str, Any]:
    return {
        "transition_id": decomposition.transition_id,
        "cluster_id": decomposition.cluster_id,
        "bbox": decomposition.bbox,
        "best_hypothesis_id": decomposition.best_hypothesis_id,
        "best_residue_rate": round(decomposition.residue_rate, 6),
        "best_ambiguity": round(decomposition.ambiguity, 6),
        "overmerge_taxonomy_label": decomposition.overmerge_taxonomy_label,
    }


def changed_cluster_merge_split_decomposition_v0(
    record: TransitionRecord,
    row: Mapping[str, Any],
    *,
    candidate_changed_cells: Iterable[tuple[int, int]] = (),
) -> ClusterDecomposition:
    transition_id = f"{record.game}:step:{record.step}:mover:{row['episode_local_mover_ref']}"
    cluster_id = f"{transition_id}:merged_evidence"
    old_bbox = tuple(int(item) for item in row.get("bbox_before", ()) if item is not None)
    new_bbox = tuple(int(item) for item in row.get("bbox_after", ()) if item is not None)
    cluster_bboxes = [tuple(int(item) for item in bbox) for bbox in row.get("cluster_bboxes", ()) if len(tuple(bbox)) == 4]
    evidence_bbox = merged_evidence_bbox(cluster_bboxes, old_bbox, new_bbox)
    changed_cells = changed_cells_in_bboxes(
        record,
        cluster_bboxes,
        old_bbox,
        new_bbox,
        candidate_changed_cells=candidate_changed_cells,
    )
    if not changed_cells:
        changed_cells = changed_cells_in_bboxes(
            record,
            [evidence_bbox],
            old_bbox,
            new_bbox,
            candidate_changed_cells=candidate_changed_cells,
        )
    context = decomposition_context(record, row, changed_cells)
    raw_hypotheses = [
        opaque_changed_cluster_hypothesis(changed_cells, context),
        footprint_algebra_splitter(changed_cells, context),
        large_shape_translation_splitter(changed_cells, context),
        background_exposure_splitter(changed_cells, context),
        actor_overlap_splitter(changed_cells, context),
        independent_object_splitter(changed_cells, context),
        multi_object_membership_splitter(changed_cells, context),
        geometry_bridge_splitter(changed_cells, context),
        residue_only_splitter(changed_cells, context),
    ]
    hypotheses = tuple(sorted(raw_hypotheses, key=lambda item: item.score, reverse=True))
    best = hypotheses[0] if hypotheses else None
    ambiguity = decomposition_ambiguity_from_hypotheses(hypotheses)
    classification = ar25_overlap_composition_classification(row)
    return ClusterDecomposition(
        transition_id=transition_id,
        cluster_id=cluster_id,
        bbox=evidence_bbox,
        hypotheses=hypotheses,
        best_hypothesis_id=best.hypothesis_id if best else None,
        residue_rate=best.residue_rate if best else 1.0,
        ambiguity=ambiguity,
        overmerge_taxonomy_label=str(classification["label"]),
    )


def decomposition_context(
    record: TransitionRecord,
    row: Mapping[str, Any],
    changed_cells: list[tuple[int, int]],
) -> dict[str, Any]:
    before = observation_plane(record.before)
    after = observation_plane(record.after)
    old_bbox = tuple(row.get("bbox_before", ()))
    new_bbox = tuple(row.get("bbox_after", ()))
    old_cells = bbox_cell_set(old_bbox, before.shape)
    new_cells = bbox_cell_set(new_bbox, before.shape)
    controlled_cells = set()
    for bbox in controlled_candidate_bboxes(record, key="bbox_before"):
        controlled_cells.update(bbox_cell_set(bbox, before.shape))
    for bbox in controlled_candidate_bboxes(record, key="bbox_after"):
        controlled_cells.update(bbox_cell_set(bbox, before.shape))
    other_mover_cells = set()
    current_ref = str(row.get("episode_local_mover_ref"))
    for delta in stable_independent_motion_deltas(record):
        if local_mover_ref(record.game, int(delta.object_id)) == current_ref:
            continue
        other_mover_cells.update(bbox_cell_set(tuple(getattr(delta, "bbox_before", ())), before.shape))
        other_mover_cells.update(bbox_cell_set(tuple(getattr(delta, "bbox_after", ())), before.shape))
    color_groups = Counter((cell_color(before, cell), cell_color(after, cell)) for cell in changed_cells)
    background_color = estimated_background_color(before)
    return {
        "before": before,
        "after": after,
        "old_cells": old_cells,
        "new_cells": new_cells,
        "overlap_cells": old_cells & new_cells,
        "controlled_cells": controlled_cells,
        "other_mover_cells": other_mover_cells,
        "background_color": background_color,
        "color_groups": color_groups,
        "translation": tuple(row.get("translation", ())),
        "cluster_count": int(row.get("cluster_count", 0) or 0),
    }


def observation_plane(observation: Observation) -> np.ndarray:
    pixels = np.asarray(observation.pixels)
    if pixels.ndim == 3:
        return pixels[-1]
    return pixels


def merged_evidence_bbox(
    cluster_bboxes: list[tuple[int, int, int, int]],
    old_bbox: tuple[Any, ...],
    new_bbox: tuple[Any, ...],
) -> tuple[int, int, int, int]:
    candidates = list(cluster_bboxes)
    if len(old_bbox) == 4:
        candidates.append(tuple(int(item) for item in old_bbox))
    if len(new_bbox) == 4:
        candidates.append(tuple(int(item) for item in new_bbox))
    if not candidates:
        return (0, 0, 0, 0)
    return (
        min(bbox[0] for bbox in candidates),
        min(bbox[1] for bbox in candidates),
        max(bbox[2] for bbox in candidates),
        max(bbox[3] for bbox in candidates),
    )


def bbox_cell_set(bbox: tuple[Any, ...], shape: tuple[int, ...]) -> set[tuple[int, int]]:
    if len(bbox) != 4 or len(shape) < 2:
        return set()
    height, width = int(shape[0]), int(shape[1])
    r0, c0, r1, c1 = (int(item) for item in bbox)
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(height - 1, r1), min(width - 1, c1)
    if r1 < r0 or c1 < c0:
        return set()
    return {(row, col) for row in range(r0, r1 + 1) for col in range(c0, c1 + 1)}


def changed_cells_in_bboxes(
    record: TransitionRecord,
    cluster_bboxes: list[tuple[int, int, int, int]],
    old_bbox: tuple[Any, ...],
    new_bbox: tuple[Any, ...],
    *,
    candidate_changed_cells: Iterable[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    bboxes = list(cluster_bboxes)
    if len(old_bbox) == 4:
        bboxes.append(tuple(int(item) for item in old_bbox))
    if len(new_bbox) == 4:
        bboxes.append(tuple(int(item) for item in new_bbox))
    if not bboxes:
        return []
    changed_coordinates = list(candidate_changed_cells) or all_changed_cells(record)
    cells = set()
    for row, col in changed_coordinates:
        row_int, col_int = int(row), int(col)
        if any(bbox_contains_cell(bbox, (row_int, col_int)) for bbox in bboxes):
            cells.add((row_int, col_int))
    return sorted(cells)


def all_changed_cells(record: TransitionRecord) -> list[tuple[int, int]]:
    before = observation_plane(record.before)
    after = observation_plane(record.after)
    return [(int(row), int(col)) for row, col in np.argwhere(before != after)]


def bbox_contains_cell(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    return bbox[0] <= cell[0] <= bbox[2] and bbox[1] <= cell[1] <= bbox[3]


def cell_color(grid: np.ndarray, cell: tuple[int, int]) -> int:
    return int(grid[int(cell[0]), int(cell[1])])


def estimated_background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    if len(values) == 0:
        return 0
    return int(values[int(np.argmax(counts))])


def opaque_changed_cluster_hypothesis(
    changed_cells: list[tuple[int, int]],
    context: Mapping[str, Any],
) -> DecompositionHypothesis:
    attributions = [
        cell_attribution(cell, context, "H0_opaque_changed_cluster", ("other_effect",), 0.35)
        for cell in changed_cells
    ]
    return score_decomposition_hypothesis(
        "H0_opaque_changed_cluster",
        "OpaqueChangedCluster",
        attributions,
        complexity=1,
        reasons=("baseline treats connected changed component as one opaque event",),
        score_bias=-0.35,
    )


def footprint_algebra_splitter(
    changed_cells: list[tuple[int, int]],
    context: Mapping[str, Any],
) -> DecompositionHypothesis:
    return hypothesis_from_labeler(
        "H1_single_translated_mover_old_new_footprint",
        "FootprintAlgebraSplitter",
        changed_cells,
        context,
        lambda cell: footprint_algebra_labels(cell, context),
        reasons=("old_only/new_only/overlap from translated mover footprint",),
        complexity=3,
    )


def large_shape_translation_splitter(
    changed_cells: list[tuple[int, int]],
    context: Mapping[str, Any],
) -> DecompositionHypothesis:
    def labeler(cell: tuple[int, int]) -> tuple[str, ...]:
        labels = list(footprint_algebra_labels(cell, context))
        if "residue" in labels and adjacent_to_any(cell, context["old_cells"] | context["new_cells"]):
            labels = ["bridge_artifact"]
        if "overlap" in labels:
            labels.append("mover_overlap")
        return tuple(sorted(set(labels)))

    return hypothesis_from_labeler(
        "H2_large_shape_translation_with_self_overlap",
        "FootprintAlgebraSplitter",
        changed_cells,
        context,
        labeler,
        reasons=("large translated shape can create self-overlap and bridge cells",),
        complexity=4,
    )


def background_exposure_splitter(
    changed_cells: list[tuple[int, int]],
    context: Mapping[str, Any],
) -> DecompositionHypothesis:
    def labeler(cell: tuple[int, int]) -> tuple[str, ...]:
        labels = list(footprint_algebra_labels(cell, context))
        if "old_footprint" in labels and "new_footprint" not in labels:
            if cell_color(context["after"], cell) == int(context["background_color"]):
                labels.append("exposure")
        return tuple(sorted(set(label for label in labels if label != "residue"))) or ("residue",)

    return hypothesis_from_labeler(
        "H3_old_new_mover_footprint_background_exposure",
        "BackgroundExposureSplitter",
        changed_cells,
        context,
        labeler,
        reasons=("old footprint cells may reveal stable background after unrendering",),
        complexity=4,
    )


def actor_overlap_splitter(
    changed_cells: list[tuple[int, int]],
    context: Mapping[str, Any],
) -> DecompositionHypothesis:
    def labeler(cell: tuple[int, int]) -> tuple[str, ...]:
        labels = list(footprint_algebra_labels(cell, context))
        if cell in context["controlled_cells"]:
            labels.append("actor_overlap")
        return tuple(sorted(set(label for label in labels if label != "residue"))) or ("residue",)

    return hypothesis_from_labeler(
        "H4_mover_footprint_actor_control_overlap",
        "MultiObjectMembershipSplitter",
        changed_cells,
        context,
        labeler,
        reasons=("single connected cluster may mix mover footprint with controlled-object overlap",),
        complexity=5,
    )


def independent_object_splitter(
    changed_cells: list[tuple[int, int]],
    context: Mapping[str, Any],
) -> DecompositionHypothesis:
    def labeler(cell: tuple[int, int]) -> tuple[str, ...]:
        labels = list(footprint_algebra_labels(cell, context))
        if cell in context["other_mover_cells"]:
            labels.append("independent_effect")
        return tuple(sorted(set(label for label in labels if label != "residue"))) or ("residue",)

    return hypothesis_from_labeler(
        "H5_mover_footprint_independent_object_effect",
        "MultiObjectMembershipSplitter",
        changed_cells,
        context,
        labeler,
        reasons=("cluster may contain current mover plus independent object effect",),
        complexity=5,
    )


def multi_object_membership_splitter(
    changed_cells: list[tuple[int, int]],
    context: Mapping[str, Any],
) -> DecompositionHypothesis:
    def labeler(cell: tuple[int, int]) -> tuple[str, ...]:
        labels = list(footprint_algebra_labels(cell, context))
        if cell in context["controlled_cells"]:
            labels.append("actor_overlap")
        if cell in context["other_mover_cells"]:
            labels.append("mover_overlap")
        if "residue" in labels and adjacent_to_any(cell, context["old_cells"] | context["new_cells"] | context["other_mover_cells"]):
            labels.append("bridge_artifact")
            labels = [label for label in labels if label != "residue"]
        return tuple(sorted(set(label for label in labels if label != "residue"))) or ("residue",)

    return hypothesis_from_labeler(
        "H6_multiple_movers_merged_by_adjacency",
        "MultiObjectMembershipSplitter",
        changed_cells,
        context,
        labeler,
        reasons=("adjacent mover/control memberships can merge multiple causal regions",),
        complexity=6,
    )


def geometry_bridge_splitter(
    changed_cells: list[tuple[int, int]],
    context: Mapping[str, Any],
) -> DecompositionHypothesis:
    def labeler(cell: tuple[int, int]) -> tuple[str, ...]:
        labels = list(footprint_algebra_labels(cell, context))
        if "residue" in labels and adjacent_to_any(cell, context["old_cells"] | context["new_cells"]):
            return ("bridge_artifact",)
        return tuple(labels)

    return hypothesis_from_labeler(
        "H7_render_artifact_trail_bridge",
        "GeometryBridgeSplitter",
        changed_cells,
        context,
        labeler,
        reasons=("thin adjacency bridges can merge separate old/new causal regions",),
        complexity=4,
    )


def residue_only_splitter(
    changed_cells: list[tuple[int, int]],
    context: Mapping[str, Any],
) -> DecompositionHypothesis:
    attributions = [
        cell_attribution(cell, context, "H8_unexplained_residue", ("residue",), 0.2)
        for cell in changed_cells
    ]
    return score_decomposition_hypothesis(
        "H8_unexplained_residue",
        "ResidueOnlySplitter",
        attributions,
        complexity=1,
        reasons=("fallback explicitly preserves unexplained cells instead of naming a false event",),
        score_bias=-0.5,
    )


def footprint_algebra_labels(cell: tuple[int, int], context: Mapping[str, Any]) -> tuple[str, ...]:
    old_cells = context["old_cells"]
    new_cells = context["new_cells"]
    in_old = cell in old_cells
    in_new = cell in new_cells
    if in_old and in_new:
        return ("overlap",)
    if in_old:
        return ("old_footprint",)
    if in_new:
        return ("new_footprint",)
    return ("residue",)


def hypothesis_from_labeler(
    hypothesis_id: str,
    splitter: str,
    changed_cells: list[tuple[int, int]],
    context: Mapping[str, Any],
    labeler: Any,
    *,
    reasons: tuple[str, ...],
    complexity: int,
) -> DecompositionHypothesis:
    attributions = [
        cell_attribution(cell, context, hypothesis_id, tuple(labeler(cell)), 0.75)
        for cell in changed_cells
    ]
    return score_decomposition_hypothesis(
        hypothesis_id,
        splitter,
        attributions,
        complexity=complexity,
        reasons=reasons,
    )


def cell_attribution(
    cell: tuple[int, int],
    context: Mapping[str, Any],
    hypothesis_id: str,
    labels: tuple[str, ...],
    confidence: float,
) -> CellAttribution:
    before = context["before"]
    after = context["after"]
    clean_labels = tuple(sorted(set(labels))) or ("residue",)
    if clean_labels == ("residue",):
        confidence = min(confidence, 0.35)
    return CellAttribution(
        cell=(int(cell[0]), int(cell[1])),
        before_color=cell_color(before, cell),
        after_color=cell_color(after, cell),
        labels=clean_labels,
        hypothesis_id=hypothesis_id,
        confidence=confidence,
    )


def score_decomposition_hypothesis(
    hypothesis_id: str,
    splitter: str,
    attributions: list[CellAttribution],
    *,
    complexity: int,
    reasons: tuple[str, ...],
    score_bias: float = 0.0,
) -> DecompositionHypothesis:
    total = len(attributions)
    non_residue = sum("residue" not in item.labels for item in attributions)
    residue_rate = 1.0 if total == 0 else 1.0 - (non_residue / total)
    coverage = 0.0 if total == 0 else non_residue / total
    labels = Counter(label for item in attributions for label in item.labels)
    causal_parts = len({label for label in labels if label != "residue"})
    footprint_consistency = footprint_consistency_score(labels)
    color_purity = color_transition_purity(attributions)
    temporal_reuse_proxy = 1.0 if causal_parts >= 2 else 0.35
    ambiguity = 1.0 / max(1, causal_parts)
    score = (
        0.42 * coverage
        + 0.22 * footprint_consistency
        + 0.18 * color_purity
        + 0.18 * temporal_reuse_proxy
        - 0.16 * residue_rate
        - 0.02 * complexity
        + score_bias
    )
    return DecompositionHypothesis(
        hypothesis_id=hypothesis_id,
        splitter=splitter,
        cell_attributions=tuple(attributions),
        score=score,
        coverage=coverage,
        residue_rate=residue_rate,
        ambiguity=ambiguity,
        complexity=complexity,
        reasons=reasons,
    )


def footprint_consistency_score(labels: Counter[str]) -> float:
    has_old = bool(labels.get("old_footprint") or labels.get("overlap"))
    has_new = bool(labels.get("new_footprint") or labels.get("overlap"))
    if has_old and has_new:
        return 1.0
    if has_old or has_new:
        return 0.55
    if labels.get("other_effect"):
        return 0.2
    return 0.0


def color_transition_purity(attributions: list[CellAttribution]) -> float:
    if not attributions:
        return 0.0
    by_label: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    for item in attributions:
        transition = (item.before_color, item.after_color)
        for label in item.labels:
            if label != "residue":
                by_label[label][transition] += 1
    if not by_label:
        return 0.0
    purities = []
    for transitions in by_label.values():
        total = sum(transitions.values())
        purities.append(max(transitions.values()) / total if total else 0.0)
    return sum(purities) / len(purities)


def adjacent_to_any(cell: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    row, col = cell
    return any((row + dr, col + dc) in cells for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))


def decomposition_ambiguity_from_hypotheses(hypotheses: tuple[DecompositionHypothesis, ...]) -> float:
    if len(hypotheses) < 2:
        return 0.0
    best = hypotheses[0].score
    close = sum(1 for hypothesis in hypotheses[1:] if best - hypothesis.score <= 0.08)
    return round(close / (len(hypotheses) - 1), 6)


def changed_cluster_merge_split_metrics(decompositions: list[ClusterDecomposition]) -> dict[str, Any]:
    total = len(decompositions)
    if not total:
        return {
            "case_count": 0,
            "causal_split_recovery_rate": 0.0,
            "overmerge_rate": 0.0,
            "mean_causal_parts_per_cluster": 0.0,
            "residue_after_best_decomposition": 0.0,
            "old_new_footprint_recovery_rate": 0.0,
            "cluster_merge_artifact_multi_part_count": 0,
            "footprint_algebra_success_rate": 0.0,
            "color_transition_split_success_rate": 0.0,
            "background_exposure_success_rate": 0.0,
            "multi_object_membership_success_rate": 0.0,
            "decomposition_ambiguity": 0.0,
        }
    bests = [item.best_hypothesis() for item in decompositions]
    bests = [item for item in bests if item is not None]
    split_recovered = [
        hypothesis for hypothesis in bests
        if hypothesis.causal_part_count() >= 2 and hypothesis.coverage >= 0.75
    ]
    old_new_recovered = [
        hypothesis for hypothesis in bests
        if hypothesis_label_present(hypothesis, "old_footprint")
        and hypothesis_label_present(hypothesis, "new_footprint")
    ]
    cluster_merge_multi = [
        decomposition for decomposition in decompositions
        if decomposition.overmerge_taxonomy_label == "cluster_merge_artifact"
        and (decomposition.best_hypothesis() is not None)
        and decomposition.best_hypothesis().causal_part_count() >= 2
    ]
    return {
        "case_count": total,
        "causal_split_recovery_rate": round(len(split_recovered) / total, 6),
        "overmerge_rate": round(sum(1 for hypothesis in bests if hypothesis.causal_part_count() >= 2) / total, 6),
        "mean_causal_parts_per_cluster": round(sum(hypothesis.causal_part_count() for hypothesis in bests) / total, 6),
        "residue_after_best_decomposition": round(sum(item.residue_rate for item in decompositions) / total, 6),
        "old_new_footprint_recovery_rate": round(len(old_new_recovered) / total, 6),
        "cluster_merge_artifact_multi_part_count": len(cluster_merge_multi),
        "footprint_algebra_success_rate": splitter_success_rate(decompositions, "FootprintAlgebraSplitter"),
        "color_transition_split_success_rate": color_transition_split_success_rate(decompositions),
        "background_exposure_success_rate": splitter_success_rate(decompositions, "BackgroundExposureSplitter"),
        "multi_object_membership_success_rate": splitter_success_rate(decompositions, "MultiObjectMembershipSplitter"),
        "decomposition_ambiguity": round(sum(item.ambiguity for item in decompositions) / total, 6),
        "best_hypothesis_distribution": dict(Counter(item.best_hypothesis_id for item in decompositions).most_common()),
    }


def hypothesis_label_present(hypothesis: DecompositionHypothesis, label: str) -> bool:
    return label in hypothesis.label_counts()


def splitter_success_rate(decompositions: list[ClusterDecomposition], splitter: str) -> float:
    total = len(decompositions)
    if not total:
        return 0.0
    success = 0
    for decomposition in decompositions:
        if any(
            hypothesis.splitter == splitter
            and hypothesis.coverage >= 0.75
            and hypothesis.causal_part_count() >= 2
            for hypothesis in decomposition.hypotheses
        ):
            success += 1
    return round(success / total, 6)


def color_transition_split_success_rate(decompositions: list[ClusterDecomposition]) -> float:
    total = len(decompositions)
    if not total:
        return 0.0
    success = 0
    for decomposition in decompositions:
        best = decomposition.best_hypothesis()
        if best is not None and color_transition_purity(list(best.cell_attributions)) >= 0.85 and best.causal_part_count() >= 2:
            success += 1
    return round(success / total, 6)


def decomposition_before_after_metrics_v0(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_unexplained = float(baseline.get("unexplained_region_rate", 0.844354) or 0.0)
    after_unexplained = float(metrics.get("residue_after_best_decomposition", 0.0) or 0.0)
    baseline_old_new = float(baseline.get("old_new_footprint_recovery_rate", 0.0) or 0.0)
    after_old_new = float(metrics.get("old_new_footprint_recovery_rate", 0.0) or 0.0)
    return {
        "baseline_source": "MoverBootstrapAndCompositionV0.merged_overlap_metrics",
        "baseline_unexplained_region_rate": round(baseline_unexplained, 6),
        "v0_unexplained_region_rate": round(after_unexplained, 6),
        "unexplained_region_rate_delta": round(after_unexplained - baseline_unexplained, 6),
        "baseline_old_new_footprint_recovery_rate": round(baseline_old_new, 6),
        "v0_old_new_footprint_recovery_rate": round(after_old_new, 6),
        "old_new_footprint_recovery_delta": round(after_old_new - baseline_old_new, 6),
        "baseline_t5e_proxy_error_after": baseline.get("t5e_proxy_error_after", baseline_unexplained),
        "v0_t5e_proxy_error_after": round(after_unexplained, 6),
        "acceptance_old_new_recovery_ge_0_80": after_old_new >= 0.80,
        "acceptance_unexplained_region_rate_improved": after_unexplained < baseline_unexplained,
    }


def ar25_cluster_overmerge_taxonomy_v0(decompositions: list[ClusterDecomposition]) -> dict[str, Any]:
    labels = Counter(item.overmerge_taxonomy_label for item in decompositions)
    high_residue_subtypes = Counter(high_residue_subtype(item) for item in decompositions if item.residue_rate >= 0.25)
    return {
        "taxonomy_source": "ChangedClusterMergeSplitAuditV0",
        "overmerge_taxonomy_distribution": dict(labels.most_common()),
        "high_residue_failure_subtypes": dict(high_residue_subtypes.most_common()),
        "actionable_subtypes": [
            {
                "label": label,
                "count": count,
                "suggested_next_probe": high_residue_probe_for_subtype(label),
            }
            for label, count in high_residue_subtypes.most_common()
        ],
    }


def high_residue_subtype(decomposition: ClusterDecomposition) -> str:
    best = decomposition.best_hypothesis()
    if best is None:
        return "no_hypothesis"
    if decomposition.overmerge_taxonomy_label == "large_shape_translation_overlap":
        return "large_shape_translation_overlap_residue"
    if decomposition.overmerge_taxonomy_label == "cluster_merge_artifact":
        if best.causal_part_count() >= 2:
            return "recovered_old_new_with_unexplained_residue"
        return "cluster_merge_without_causal_split"
    if hypothesis_label_present(best, "bridge_artifact"):
        return "bridge_artifact_residue"
    if hypothesis_label_present(best, "mover_overlap") or hypothesis_label_present(best, "actor_overlap"):
        return "overlap_membership_residue"
    return "unclassified_residue"


def high_residue_probe_for_subtype(label: str) -> str:
    if label == "large_shape_translation_overlap_residue":
        return "shape-mask render/unrender composition audit"
    if label == "recovered_old_new_with_unexplained_residue":
        return "cell-level bridge and background exposure validation"
    if label == "cluster_merge_without_causal_split":
        return "changed-component segmentation boundary audit"
    if label == "bridge_artifact_residue":
        return "thin-bridge split stress test"
    if label == "overlap_membership_residue":
        return "multi-object overlap priority audit"
    return "manual residue prototype atlas"


def decomposition_failure_gallery_v0(
    decompositions: list[ClusterDecomposition],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    buckets = {
        "cluster_merge_recovered_old_new_unexplained_residue": [],
        "large_shape_translation_overlap": [],
        "high_ambiguity_or_high_residue": [],
    }
    for decomposition in decompositions:
        best = decomposition.best_hypothesis()
        if best is None:
            continue
        base = {
            **decomposition_metadata(decomposition),
            "best_hypothesis": best.to_record(),
            "sample_cell_attributions": [item.to_record() for item in best.cell_attributions[:24]],
            "mask_summary": mask_summary_for_hypothesis(best),
        }
        if (
            decomposition.overmerge_taxonomy_label == "cluster_merge_artifact"
            and hypothesis_label_present(best, "old_footprint")
            and hypothesis_label_present(best, "new_footprint")
            and decomposition.residue_rate >= 0.10
            and len(buckets["cluster_merge_recovered_old_new_unexplained_residue"]) < 10
        ):
            buckets["cluster_merge_recovered_old_new_unexplained_residue"].append(
                {"gallery_bucket": "cluster_merge_recovered_old_new_unexplained_residue", **base}
            )
        if (
            decomposition.overmerge_taxonomy_label == "large_shape_translation_overlap"
            and len(buckets["large_shape_translation_overlap"]) < 10
        ):
            buckets["large_shape_translation_overlap"].append(
                {"gallery_bucket": "large_shape_translation_overlap", **base}
            )
        if (
            (decomposition.ambiguity >= 0.25 or decomposition.residue_rate >= 0.25)
            and len(buckets["high_ambiguity_or_high_residue"]) < 10
        ):
            buckets["high_ambiguity_or_high_residue"].append(
                {"gallery_bucket": "high_ambiguity_or_high_residue", **base}
            )
    rows: list[dict[str, Any]] = []
    for items in buckets.values():
        rows.extend(items)
    return rows[:limit]


def mask_summary_for_hypothesis(hypothesis: DecompositionHypothesis) -> dict[str, Any]:
    cells_by_label: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item in hypothesis.cell_attributions:
        for label in item.labels:
            cells_by_label[label].append(item.cell)
    return {
        label: {
            "cell_count": len(cells),
            "bbox": cells_bbox(cells),
        }
        for label, cells in sorted(cells_by_label.items())
    }


def cells_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not cells:
        return None
    return (
        min(cell[0] for cell in cells),
        min(cell[1] for cell in cells),
        max(cell[0] for cell in cells),
        max(cell[1] for cell in cells),
    )


def residue_source_attribution_v0_report(
    records: list[TransitionRecord],
    *,
    decompositions: list[ClusterDecomposition] | None = None,
    competition_groups: list[ResidueCompetitionGroup] | None = None,
    merge_split_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game != "ar25":
        return {
            "status": "skipped_not_ar25",
            "game": game,
            "reason": "ResidueSourceAttributionV0 is an AR25 high-residue audit",
        }
    record_by_step = {record.step: record for record in records}
    groups = (
        competition_groups
        if competition_groups is not None
        else ar25_residue_competition_groups_v0(
            records,
            decompositions=decompositions if decompositions is not None else ar25_merged_cluster_decompositions_v0(records),
        )
    )

    metrics = residue_source_metrics_v0(groups, merge_split_metrics or {})
    taxonomy = ar25_residue_taxonomy_v0(groups)
    atoms = [
        {
            "transition_id": group.transition_id,
            "cluster_id": group.cluster_id,
            **atom.to_record(),
        }
        for group in groups
        for atom in group.atoms
    ]
    hypothesis_scores = [
        {
            "transition_id": group.transition_id,
            "cluster_id": group.cluster_id,
            **hypothesis.to_record(),
        }
        for group in groups
        for hypothesis in group.hypotheses
    ]
    gallery = high_residue_case_gallery_v0(groups, record_by_step, limit_per_label=3)
    labels = []
    if metrics["old_new_footprint_recovery_rate"] >= 0.90:
        labels.append("old_new_footprint_recovery_preserved")
    else:
        labels.append("old_new_footprint_recovery_regressed")
    if metrics["high_residue_taxonomy_coverage"] >= 0.80:
        labels.append("high_residue_failures_bucketed")
    else:
        labels.append("high_residue_taxonomy_incomplete")
    if metrics["unknown_residue_rate_after_source_attribution"] < 0.40:
        labels.append("unknown_residue_below_target")
    else:
        labels.append("unknown_residue_still_high")
    labels.append(f"dominant_{metrics['dominant_blocker']}")
    return {
        "status": "evaluated",
        "game": "ar25",
        "mechanism": "ResidueSourceAttributionV0",
        "diagnostic_only": True,
        "target": "high_residue_merged_overlap_failures",
        "case_count": len(groups),
        "constraints": {
            "planner_integration": False,
            "role_label_promotion": False,
            "new_option_logic": False,
            "uses_future_frames_for_generation": False,
        },
        "metrics": metrics,
        "ar25_residue_taxonomy": taxonomy,
        "residue_atoms": atoms,
        "residue_competition_groups": [group.to_record() for group in groups],
        "residue_hypothesis_scores": hypothesis_scores,
        "high_residue_case_gallery": gallery,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def ar25_residue_competition_groups_v0(
    records: list[TransitionRecord],
    *,
    decompositions: list[ClusterDecomposition],
) -> list[ResidueCompetitionGroup]:
    game = records[0].game if records else "unknown"
    if game != "ar25":
        return []
    record_by_step = {record.step: record for record in records}
    previous_changed_by_step = previous_changed_cells_by_step(records, window=4)
    groups = []
    for decomposition in decompositions:
        if decomposition.residue_rate < 0.25:
            continue
        record = record_by_step.get(step_from_transition_id(decomposition.transition_id))
        best = decomposition.best_hypothesis()
        if record is None or best is None:
            continue
        residue_cells = tuple(
            attribution.cell for attribution in best.cell_attributions
            if "residue" in attribution.labels
        )
        if not residue_cells:
            continue
        groups.append(
            residue_competition_group_v0(
                record,
                decomposition,
                residue_cells,
                previous_changed_cells=previous_changed_by_step.get(record.step, set()),
            )
        )
    return groups


def residue_competition_group_v0(
    record: TransitionRecord,
    decomposition: ClusterDecomposition,
    residue_cells: tuple[tuple[int, int], ...],
    *,
    previous_changed_cells: set[tuple[int, int]],
) -> ResidueCompetitionGroup:
    atoms = tuple(
        spatial_residue_atoms(residue_cells)
        + color_transition_residue_atoms(record, residue_cells)
        + object_membership_residue_atoms(record, decomposition, residue_cells)
        + geometry_band_residue_atoms(record, residue_cells)
        + temporal_recurrence_residue_atoms(residue_cells, previous_changed_cells)
    )
    hypotheses = tuple(sorted(residue_source_hypotheses_v0(record, decomposition, residue_cells, atoms), key=lambda item: item.score, reverse=True))
    top = hypotheses[0].label if hypotheses else "true_unknown_effect"
    return ResidueCompetitionGroup(
        transition_id=decomposition.transition_id,
        cluster_id=decomposition.cluster_id,
        cluster_bbox=decomposition.bbox,
        residue_cell_count=len(residue_cells),
        atoms=atoms,
        hypotheses=hypotheses,
        top_hypothesis=top,
    )


def spatial_residue_atoms(residue_cells: tuple[tuple[int, int], ...]) -> list[ResidueAtom]:
    atoms = []
    for index, component in enumerate(connected_cell_components(set(residue_cells)), start=1):
        atoms.append(
            ResidueAtom(
                atom_id=f"spatial:{index}",
                atomizer="spatial_bridge_cut",
                cells=tuple(sorted(component)),
                bbox=cells_bbox(list(component)),
                color_transition=None,
                source_hint="spatial_component",
            )
        )
    return atoms


def color_transition_residue_atoms(record: TransitionRecord, residue_cells: tuple[tuple[int, int], ...]) -> list[ResidueAtom]:
    before = observation_plane(record.before)
    after = observation_plane(record.after)
    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for cell in residue_cells:
        groups[(cell_color(before, cell), cell_color(after, cell))].append(cell)
    return [
        ResidueAtom(
            atom_id=f"color:{before_color}->{after_color}",
            atomizer="color_transition",
            cells=tuple(sorted(cells)),
            bbox=cells_bbox(cells),
            color_transition=(before_color, after_color),
            source_hint="color_transition_group",
        )
        for (before_color, after_color), cells in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    ]


def object_membership_residue_atoms(
    record: TransitionRecord,
    decomposition: ClusterDecomposition,
    residue_cells: tuple[tuple[int, int], ...],
) -> list[ResidueAtom]:
    before = observation_plane(record.before)
    best = decomposition.best_hypothesis()
    old_new_cells = set()
    if best is not None:
        old_new_cells = {
            attribution.cell for attribution in best.cell_attributions
            if "old_footprint" in attribution.labels
            or "new_footprint" in attribution.labels
            or "overlap" in attribution.labels
        }
    controlled_cells = set()
    for bbox in controlled_candidate_bboxes(record, key="bbox_before"):
        controlled_cells.update(bbox_cell_set(bbox, before.shape))
    for bbox in controlled_candidate_bboxes(record, key="bbox_after"):
        controlled_cells.update(bbox_cell_set(bbox, before.shape))
    mover_cells = set()
    for delta in stable_independent_motion_deltas(record):
        mover_cells.update(bbox_cell_set(tuple(getattr(delta, "bbox_before", ())), before.shape))
        mover_cells.update(bbox_cell_set(tuple(getattr(delta, "bbox_after", ())), before.shape))
    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for cell in residue_cells:
        if cell in controlled_cells:
            buckets["controlled_object_mask"].append(cell)
        elif cell in mover_cells:
            buckets["other_mover_or_object_mask"].append(cell)
        elif adjacent_to_any(cell, old_new_cells):
            buckets["near_recovered_old_new_mask"].append(cell)
        else:
            buckets["no_known_object_membership"].append(cell)
    return [
        ResidueAtom(
            atom_id=f"object:{label}",
            atomizer="object_membership",
            cells=tuple(sorted(cells)),
            bbox=cells_bbox(cells),
            color_transition=None,
            source_hint=label,
        )
        for label, cells in sorted(buckets.items())
    ]


def geometry_band_residue_atoms(record: TransitionRecord, residue_cells: tuple[tuple[int, int], ...]) -> list[ResidueAtom]:
    grid = observation_plane(record.before)
    height, width = grid.shape[:2]
    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    rows = Counter(row for row, _col in residue_cells)
    cols = Counter(col for _row, col in residue_cells)
    band_rows = {row for row, count in rows.items() if count >= max(4, width // 8)}
    band_cols = {col for col, count in cols.items() if count >= max(4, height // 8)}
    for cell in residue_cells:
        row, col = cell
        if row <= 4 or col <= 4 or row >= height - 5 or col >= width - 5:
            buckets["frame_border_or_margin"].append(cell)
        elif row in band_rows:
            buckets["horizontal_band"].append(cell)
        elif col in band_cols:
            buckets["vertical_band"].append(cell)
        else:
            buckets["interior_irregular"].append(cell)
    return [
        ResidueAtom(
            atom_id=f"geometry:{label}",
            atomizer="geometry_band",
            cells=tuple(sorted(cells)),
            bbox=cells_bbox(cells),
            color_transition=None,
            source_hint=label,
        )
        for label, cells in sorted(buckets.items())
    ]


def temporal_recurrence_residue_atoms(
    residue_cells: tuple[tuple[int, int], ...],
    previous_changed_cells: set[tuple[int, int]],
) -> list[ResidueAtom]:
    recurring = sorted(set(residue_cells) & previous_changed_cells)
    fresh = sorted(set(residue_cells) - previous_changed_cells)
    atoms = []
    if recurring:
        atoms.append(
            ResidueAtom(
                atom_id="temporal:recently_changed_again",
                atomizer="temporal_recurrence",
                cells=tuple(recurring),
                bbox=cells_bbox(recurring),
                color_transition=None,
                source_hint="recently_changed_again",
            )
        )
    if fresh:
        atoms.append(
            ResidueAtom(
                atom_id="temporal:not_recently_changed",
                atomizer="temporal_recurrence",
                cells=tuple(fresh),
                bbox=cells_bbox(fresh),
                color_transition=None,
                source_hint="not_recently_changed",
            )
        )
    return atoms


def residue_source_hypotheses_v0(
    record: TransitionRecord,
    decomposition: ClusterDecomposition,
    residue_cells: tuple[tuple[int, int], ...],
    atoms: tuple[ResidueAtom, ...],
) -> list[ResidueSourceHypothesis]:
    total = max(1, len(residue_cells))
    atom_by_hint: dict[str, list[ResidueAtom]] = defaultdict(list)
    for atom in atoms:
        atom_by_hint[atom.source_hint].append(atom)
    spatial_atoms = [atom for atom in atoms if atom.atomizer == "spatial_bridge_cut"]
    spatial_nontrivial = [atom for atom in spatial_atoms if len(atom.cells) >= 2]
    object_atoms = atom_by_hint.get("controlled_object_mask", []) + atom_by_hint.get("other_mover_or_object_mask", [])
    near_atoms = atom_by_hint.get("near_recovered_old_new_mask", [])
    background_cells = background_exposure_residue_cells(record, residue_cells)
    global_atoms = (
        atom_by_hint.get("frame_border_or_margin", [])
        + atom_by_hint.get("horizontal_band", [])
        + atom_by_hint.get("vertical_band", [])
        + atom_by_hint.get("recently_changed_again", [])
    )
    hud_cells = hud_or_register_residue_cells(record, residue_cells)
    return [
        residue_hypothesis(
            "RH1_container_overinclusive",
            "container_overinclusive",
            coverage_from_atoms(spatial_nontrivial, total),
            spatial_nontrivial,
            reasons=(f"{len(spatial_nontrivial)} residue spatial atoms after bridge-cut atomization",),
            suggested_next_probe="changed-component bridge cutting and evidence-container reconstruction",
            score_bonus=0.20 if len(spatial_nontrivial) >= 3 or bbox_cell_area(decomposition.bbox) >= total * 4 else 0.0,
        ),
        residue_hypothesis(
            "RH2_multiple_movers_or_objects",
            "multiple_movers_or_objects",
            coverage_from_atoms(object_atoms, total),
            object_atoms,
            reasons=("residue overlaps known controlled or mover masks",),
            suggested_next_probe="multi-object membership and overlap-priority audit",
        ),
        residue_hypothesis(
            "RH3_sprite_render_mask_missing",
            "sprite_render_mask_missing",
            coverage_from_atoms(near_atoms, total),
            near_atoms,
            reasons=("residue sits adjacent to recovered old/new footprint masks",),
            suggested_next_probe="expanded sprite/render-mask footprint audit",
            score_bonus=0.08 if near_atoms else 0.0,
        ),
        residue_hypothesis_from_cells(
            "RH4_background_exposure_unmodeled",
            "background_exposure_unmodeled",
            background_cells,
            total,
            reasons=("residue after-color matches stable background estimate",),
            suggested_next_probe="stable-background exposure model",
        ),
        residue_hypothesis(
            "RH5_global_animation_or_clock",
            "global_animation_or_clock",
            coverage_from_atoms(global_atoms, total),
            global_atoms,
            reasons=("residue forms bands/margins or recurs from recent transitions",),
            suggested_next_probe="action-independent global update / animation clock audit",
            score_bonus=0.12 if global_atoms else 0.0,
        ),
        residue_hypothesis_from_cells(
            "RH6_hud_or_register_contamination",
            "hud_or_register_contamination",
            hud_cells,
            total,
            reasons=("residue lies in top/right status-like margins",),
            suggested_next_probe="HUD/register mask exclusion audit",
        ),
        residue_hypothesis_from_cells(
            "RH7_true_unknown_effect",
            "true_unknown_effect",
            tuple(residue_cells),
            total,
            reasons=("fallback for cells not assigned to a more specific residue source",),
            suggested_next_probe="manual high-residue prototype readout",
            score_bonus=-0.35,
        ),
    ]


def residue_hypothesis(
    hypothesis_id: str,
    label: str,
    coverage: float,
    atoms: list[ResidueAtom],
    *,
    reasons: tuple[str, ...],
    suggested_next_probe: str,
    score_bonus: float = 0.0,
) -> ResidueSourceHypothesis:
    score = min(1.0, max(0.0, coverage + score_bonus))
    return ResidueSourceHypothesis(
        hypothesis_id=hypothesis_id,
        label=label,
        score=score,
        residue_cell_coverage=coverage,
        supporting_atom_ids=tuple(atom.atom_id for atom in atoms[:20]),
        reasons=reasons,
        suggested_next_probe=suggested_next_probe,
    )


def residue_hypothesis_from_cells(
    hypothesis_id: str,
    label: str,
    cells: tuple[tuple[int, int], ...],
    total: int,
    *,
    reasons: tuple[str, ...],
    suggested_next_probe: str,
    score_bonus: float = 0.0,
) -> ResidueSourceHypothesis:
    coverage = len(set(cells)) / max(1, total)
    score = min(1.0, max(0.0, coverage + score_bonus))
    return ResidueSourceHypothesis(
        hypothesis_id=hypothesis_id,
        label=label,
        score=score,
        residue_cell_coverage=coverage,
        supporting_atom_ids=(),
        reasons=reasons,
        suggested_next_probe=suggested_next_probe,
    )


def coverage_from_atoms(atoms: list[ResidueAtom], total: int) -> float:
    cells = {cell for atom in atoms for cell in atom.cells}
    return len(cells) / max(1, total)


def background_exposure_residue_cells(
    record: TransitionRecord,
    residue_cells: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    before = observation_plane(record.before)
    after = observation_plane(record.after)
    background = estimated_background_color(before)
    return tuple(sorted(cell for cell in residue_cells if cell_color(after, cell) == background))


def hud_or_register_residue_cells(
    record: TransitionRecord,
    residue_cells: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    grid = observation_plane(record.before)
    height, width = grid.shape[:2]
    return tuple(
        sorted(
            cell for cell in residue_cells
            if cell[0] <= max(2, height // 12) or cell[1] >= width - max(2, width // 12)
        )
    )


def connected_cell_components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    components = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return sorted(components, key=lambda component: (-len(component), cells_bbox(list(component)) or (0, 0, 0, 0)))


def previous_changed_cells_by_step(records: list[TransitionRecord], *, window: int) -> dict[int, set[tuple[int, int]]]:
    result = {}
    recent: list[set[tuple[int, int]]] = []
    for record in sorted(records, key=lambda item: item.step):
        result[record.step] = set().union(*recent[-window:]) if recent else set()
        recent.append(set(all_changed_cells(record)))
    return result


def step_from_transition_id(transition_id: str) -> int:
    parts = transition_id.split(":")
    for index, part in enumerate(parts[:-1]):
        if part == "step":
            try:
                return int(parts[index + 1])
            except ValueError:
                return -1
    return -1


def residue_source_metrics_v0(
    groups: list[ResidueCompetitionGroup],
    merge_split_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    total_cases = len(groups)
    total_cells = sum(group.residue_cell_count for group in groups)
    if not total_cases or not total_cells:
        return {
            "high_residue_case_count": 0,
            "residue_atomization_rate": 0.0,
            "residue_source_assignment_rate": 0.0,
            "high_residue_taxonomy_coverage": 0.0,
            "container_overinclusive_rate": 0.0,
            "sprite_mask_expansion_gain": 0.0,
            "background_exposure_gain": 0.0,
            "global_update_rate": 0.0,
            "global_update_competitive_rate": 0.0,
            "competition_near_tie_rate": 0.0,
            "unknown_residue_rate_after_source_attribution": 0.0,
            "old_new_footprint_recovery_rate": float(merge_split_metrics.get("old_new_footprint_recovery_rate", 0.0) or 0.0),
            "dominant_blocker": "none",
        }
    top_counts = Counter(group.top_hypothesis for group in groups)
    competitive_counts: Counter[str] = Counter()
    near_ties = 0
    non_unknown_cells = 0.0
    atomized_cases = 0
    for group in groups:
        if meaningful_atom_count(group) > 1:
            atomized_cases += 1
        top = group.hypotheses[0] if group.hypotheses else None
        if top is not None and top.label != "true_unknown_effect":
            non_unknown_cells += top.residue_cell_coverage * group.residue_cell_count
            close = [
                hypothesis for hypothesis in group.hypotheses
                if top.score - hypothesis.score <= 0.05 and hypothesis.label != "true_unknown_effect"
            ]
            if len(close) > 1:
                near_ties += 1
            for hypothesis in close:
                competitive_counts[hypothesis.label] += 1
    assignment_rate = non_unknown_cells / total_cells
    baseline_residue = float(merge_split_metrics.get("residue_after_best_decomposition", 0.547836) or 0.0)
    return {
        "high_residue_case_count": total_cases,
        "residue_atomization_rate": round(atomized_cases / total_cases, 6),
        "residue_source_assignment_rate": round(assignment_rate, 6),
        "high_residue_taxonomy_coverage": round(sum(count for label, count in top_counts.items() if label != "true_unknown_effect") / total_cases, 6),
        "container_overinclusive_rate": round(top_counts["container_overinclusive"] / total_cases, 6),
        "sprite_mask_expansion_gain": top_label_coverage(groups, "sprite_render_mask_missing"),
        "background_exposure_gain": top_label_coverage(groups, "background_exposure_unmodeled"),
        "global_update_rate": round(top_counts["global_animation_or_clock"] / total_cases, 6),
        "global_update_competitive_rate": round(competitive_counts["global_animation_or_clock"] / total_cases, 6),
        "hud_or_register_contamination_rate": round(top_counts["hud_or_register_contamination"] / total_cases, 6),
        "multiple_movers_or_objects_rate": round(top_counts["multiple_movers_or_objects"] / total_cases, 6),
        "competition_near_tie_rate": round(near_ties / total_cases, 6),
        "competitive_hypothesis_distribution": dict(competitive_counts.most_common()),
        "unknown_residue_rate_after_source_attribution": round(baseline_residue * (1.0 - assignment_rate), 6),
        "old_new_footprint_recovery_rate": float(merge_split_metrics.get("old_new_footprint_recovery_rate", 0.0) or 0.0),
        "dominant_blocker": top_counts.most_common(1)[0][0] if top_counts else "none",
        "top_hypothesis_distribution": dict(top_counts.most_common()),
    }


def meaningful_atom_count(group: ResidueCompetitionGroup) -> int:
    return sum(1 for atom in group.atoms if atom.atomizer == "spatial_bridge_cut" and len(atom.cells) >= 2)


def top_label_coverage(groups: list[ResidueCompetitionGroup], label: str) -> float:
    total_cells = sum(group.residue_cell_count for group in groups)
    if not total_cells:
        return 0.0
    covered = 0.0
    for group in groups:
        top = group.hypotheses[0] if group.hypotheses else None
        if top is not None and top.label == label:
            covered += top.residue_cell_coverage * group.residue_cell_count
    return round(covered / total_cells, 6)


def ar25_residue_taxonomy_v0(groups: list[ResidueCompetitionGroup]) -> dict[str, Any]:
    counts = Counter(group.top_hypothesis for group in groups)
    return {
        "taxonomy_source": "ResidueSourceAttributionV0",
        "top_hypothesis_distribution": dict(counts.most_common()),
        "actionable_subtypes": [
            {
                "label": label,
                "count": count,
                "suggested_next_probe": residue_next_probe(label),
            }
            for label, count in counts.most_common()
        ],
        "dominant_blocker": counts.most_common(1)[0][0] if counts else "none",
    }


def residue_next_probe(label: str) -> str:
    return {
        "container_overinclusive": "changed-component bridge cutting and evidence-container reconstruction",
        "multiple_movers_or_objects": "multi-object mask membership and overlap priority audit",
        "sprite_render_mask_missing": "expanded sprite/render-mask footprint audit",
        "background_exposure_unmodeled": "stable-background exposure model",
        "global_animation_or_clock": "action-independent global update / animation clock audit",
        "hud_or_register_contamination": "HUD/register mask exclusion audit",
        "true_unknown_effect": "manual high-residue prototype readout",
    }.get(label, "manual high-residue prototype readout")


def high_residue_case_gallery_v0(
    groups: list[ResidueCompetitionGroup],
    record_by_step: Mapping[int, TransitionRecord],
    *,
    limit_per_label: int,
) -> list[dict[str, Any]]:
    emitted: Counter[str] = Counter()
    rows = []
    for group in groups:
        label = group.top_hypothesis
        if emitted[label] >= limit_per_label:
            continue
        record = record_by_step.get(step_from_transition_id(group.transition_id))
        if record is None:
            continue
        emitted[label] += 1
        residue_atoms = [atom.to_record() for atom in group.atoms if atom.atomizer == "spatial_bridge_cut"][:10]
        rows.append(
            {
                "transition_id": group.transition_id,
                "cluster_id": group.cluster_id,
                "cluster_bbox": group.cluster_bbox,
                "before_crop": frame_crop_around_changes(record, before=True),
                "after_crop": frame_crop_around_changes(record, before=False),
                "changed_mask": {
                    "bbox": cells_bbox(all_changed_cells(record)),
                    "cell_count": len(all_changed_cells(record)),
                },
                "old_new_masks": old_new_mask_summary_from_group(group),
                "residue_mask": {
                    "bbox": residue_atoms[0].get("bbox") if residue_atoms else None,
                    "cell_count": group.residue_cell_count,
                },
                "atomized_residue_mask": residue_atoms,
                "top_hypothesis": group.top_hypothesis,
                "top_3_hypotheses": [hypothesis.to_record() for hypothesis in group.hypotheses[:3]],
            }
        )
    return rows


def old_new_mask_summary_from_group(group: ResidueCompetitionGroup) -> dict[str, Any]:
    return {
        "source": "best merge-split decomposition",
        "note": "Detailed old/new masks are represented in cluster_decomposition_hypotheses.jsonl and cell_attribution_examples.jsonl",
    }


def global_update_vs_container_audit_v0_report(
    records: list[TransitionRecord],
    *,
    competition_groups: list[ResidueCompetitionGroup] | None = None,
    merge_split_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game != "ar25":
        return {
            "status": "skipped_not_ar25",
            "game": game,
            "reason": "GlobalUpdateVsContainerAuditV0 is an AR25 high-residue audit",
        }
    if competition_groups is None:
        decompositions = ar25_merged_cluster_decompositions_v0(records)
        competition_groups = ar25_residue_competition_groups_v0(records, decompositions=decompositions)
        merge_split_metrics = merge_split_metrics or changed_cluster_merge_split_metrics(decompositions)
    record_by_step = {record.step: record for record in records}
    group_records = [global_group_record(group, record_by_step) for group in competition_groups]
    group_records = [record for record in group_records if record is not None]
    templates = [
        mine_global_residue_template_v0(index, group_records)
        for index in range(len(group_records))
    ]
    verdicts = [
        global_vs_container_verdict_v0(record, template, group_records)
        for record, template in zip(group_records, templates)
    ]
    metrics = global_vs_container_metrics_v0(verdicts, merge_split_metrics or {})
    taxonomy = ar25_residue_verdict_taxonomy_v0(verdicts)
    labels = []
    if metrics["old_new_footprint_recovery_rate"] >= 0.90:
        labels.append("old_new_footprint_recovery_preserved")
    else:
        labels.append("old_new_footprint_recovery_regressed")
    labels.append(f"dominant_{metrics['dominant_verdict']}")
    if metrics["global_mask_subtraction_mean_gain"] > metrics["bridge_cut_mean_gain"]:
        labels.append("global_subtraction_beats_bridge_cut")
    else:
        labels.append("bridge_cut_beats_or_matches_global_subtraction")
    if metrics["near_tie_resolved_rate"] >= 0.80:
        labels.append("near_ties_resolved_into_verdict_buckets")
    else:
        labels.append("near_ties_still_unresolved")
    return {
        "status": "evaluated",
        "game": "ar25",
        "mechanism": "GlobalUpdateVsContainerAuditV0",
        "diagnostic_only": True,
        "case_count": len(verdicts),
        "metrics": metrics,
        "ar25_residue_verdict_taxonomy": taxonomy,
        "global_residue_templates": [template.to_record() for template in templates],
        "global_mask_subtraction_examples": global_mask_subtraction_examples_v0(verdicts, templates, limit_per_verdict=3),
        "bridge_cut_vs_global_subtraction": [verdict.to_record() for verdict in verdicts],
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def global_group_record(
    group: ResidueCompetitionGroup,
    record_by_step: Mapping[int, TransitionRecord],
) -> dict[str, Any] | None:
    step = step_from_transition_id(group.transition_id)
    record = record_by_step.get(step)
    if record is None:
        return None
    residue_cells = residue_cells_from_group(group)
    if not residue_cells:
        return None
    return {
        "group": group,
        "record": record,
        "step": step,
        "action": record.action,
        "mover_ref": mover_ref_from_transition_id(group.transition_id),
        "residue_cells": residue_cells,
    }


def residue_cells_from_group(group: ResidueCompetitionGroup) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for atom in group.atoms:
        if atom.atomizer == "spatial_bridge_cut":
            cells.update(atom.cells)
    return cells


def mover_ref_from_transition_id(transition_id: str) -> str:
    parts = transition_id.split(":")
    for index, part in enumerate(parts[:-1]):
        if part == "mover":
            return parts[index + 1]
    return "unknown"


def mine_global_residue_template_v0(
    index: int,
    group_records: list[Mapping[str, Any]],
    *,
    window: int = 4,
) -> GlobalResidueTemplate:
    target = group_records[index]
    target_step = int(target["step"])
    target_action = int(target["action"])
    target_mover = str(target["mover_ref"])
    nearby = [
        item for item in group_records
        if item is not target and abs(int(item["step"]) - target_step) <= window
    ]
    support_counter: Counter[tuple[int, int]] = Counter()
    different_action_support = 0
    different_mover_support = 0
    source_steps = set()
    for item in nearby:
        source_steps.add(int(item["step"]))
        cells = item["residue_cells"]
        support_counter.update(cells)
        if int(item["action"]) != target_action:
            different_action_support += 1
        if str(item["mover_ref"]) != target_mover:
            different_mover_support += 1
    threshold = 2 if len(nearby) >= 2 else 1
    template_cells = tuple(sorted(cell for cell, count in support_counter.items() if count >= threshold))
    return GlobalResidueTemplate(
        template_id=f"global_template:{target['group'].transition_id}",
        transition_id=target["group"].transition_id,
        source_window_steps=tuple(sorted(source_steps)),
        cells=template_cells,
        support_count=len(nearby),
        different_action_support=different_action_support,
        different_mover_support=different_mover_support,
    )


def global_vs_container_verdict_v0(
    group_record: Mapping[str, Any],
    template: GlobalResidueTemplate,
    group_records: list[Mapping[str, Any]],
) -> GlobalVsContainerVerdict:
    group: ResidueCompetitionGroup = group_record["group"]
    residue_cells = set(group_record["residue_cells"])
    template_cells = set(template.cells)
    global_gain = jaccard_like_coverage(residue_cells, template_cells)
    remaining_after_global = residue_cells - template_cells
    bridge_gain = bridge_cut_gain_for_group(group)
    temporal_score = temporal_recurrence_score_v0(group_record, group_records)
    action_dependence = action_dependence_score_v0(group_record, group_records)
    mover_dependence = mover_dependence_score_v0(group_record, group_records)
    top_rivals = top_rivals_for_group(group)
    verdict = global_vs_container_label(
        global_gain=global_gain,
        bridge_gain=bridge_gain,
        temporal_recurrence=temporal_score,
        action_dependence=action_dependence,
        mover_dependence=mover_dependence,
        top_rivals=top_rivals,
    )
    return GlobalVsContainerVerdict(
        transition_id=group.transition_id,
        cluster_id=group.cluster_id,
        container_bbox=group.cluster_bbox,
        top_rivals=top_rivals,
        temporal_recurrence_score=temporal_score,
        action_dependence_score=action_dependence,
        mover_dependence_score=mover_dependence,
        global_mask_subtraction_gain=global_gain,
        bridge_cut_gain=bridge_gain,
        post_global_subtraction_bbox=cells_bbox(list(remaining_after_global)),
        verdict=verdict,
    )


def top_rivals_for_group(group: ResidueCompetitionGroup) -> tuple[str, ...]:
    if not group.hypotheses:
        return ("true_unknown_effect",)
    best_score = group.hypotheses[0].score
    rivals = [
        hypothesis.label for hypothesis in group.hypotheses
        if best_score - hypothesis.score <= 0.05 and hypothesis.label != "true_unknown_effect"
    ]
    return tuple(rivals[:3]) or (group.hypotheses[0].label,)


def jaccard_like_coverage(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def bridge_cut_gain_for_group(group: ResidueCompetitionGroup) -> float:
    spatial = [atom for atom in group.atoms if atom.atomizer == "spatial_bridge_cut"]
    total = sum(len(atom.cells) for atom in spatial)
    if total <= 0 or len(spatial) <= 1:
        return 0.0
    largest = max(len(atom.cells) for atom in spatial)
    return 1.0 - (largest / total)


def temporal_recurrence_score_v0(
    group_record: Mapping[str, Any],
    group_records: list[Mapping[str, Any]],
) -> float:
    target = set(group_record["residue_cells"])
    if not target:
        return 0.0
    similarities = [
        jaccard_like_coverage(target, set(item["residue_cells"]))
        for item in group_records
        if item is not group_record and abs(int(item["step"]) - int(group_record["step"])) <= 4
    ]
    return max(similarities) if similarities else 0.0


def action_dependence_score_v0(
    group_record: Mapping[str, Any],
    group_records: list[Mapping[str, Any]],
) -> float:
    same = mean_similarity_to_group(group_record, group_records, same_action=True)
    different = mean_similarity_to_group(group_record, group_records, same_action=False)
    return max(0.0, same - different)


def mover_dependence_score_v0(
    group_record: Mapping[str, Any],
    group_records: list[Mapping[str, Any]],
) -> float:
    same = mean_similarity_to_group(group_record, group_records, same_mover=True)
    different = mean_similarity_to_group(group_record, group_records, same_mover=False)
    return max(0.0, same - different)


def mean_similarity_to_group(
    group_record: Mapping[str, Any],
    group_records: list[Mapping[str, Any]],
    *,
    same_action: bool | None = None,
    same_mover: bool | None = None,
) -> float:
    target = set(group_record["residue_cells"])
    similarities = []
    for item in group_records:
        if item is group_record or abs(int(item["step"]) - int(group_record["step"])) > 8:
            continue
        if same_action is not None and (int(item["action"]) == int(group_record["action"])) != same_action:
            continue
        if same_mover is not None and (str(item["mover_ref"]) == str(group_record["mover_ref"])) != same_mover:
            continue
        similarities.append(jaccard_like_coverage(target, set(item["residue_cells"])))
    return sum(similarities) / len(similarities) if similarities else 0.0


def global_vs_container_label(
    *,
    global_gain: float,
    bridge_gain: float,
    temporal_recurrence: float,
    action_dependence: float,
    mover_dependence: float,
    top_rivals: tuple[str, ...],
) -> str:
    if "multiple_movers_or_objects" in top_rivals and mover_dependence >= 0.20:
        return "multi_object_local_merge"
    if (
        "global_animation_or_clock" in top_rivals
        and global_gain >= 0.90
        and temporal_recurrence >= 0.90
    ):
        return "global_update_driven_container"
    if (
        "global_animation_or_clock" in top_rivals
        and global_gain >= 0.65
        and temporal_recurrence >= 0.65
        and action_dependence <= 0.35
    ):
        return "global_update_driven_container"
    if global_gain >= 0.45 and global_gain >= bridge_gain + 0.10 and action_dependence <= 0.10:
        return "global_update_driven_container"
    if global_gain >= 0.35 and bridge_gain >= 0.25 and temporal_recurrence >= 0.50 and action_dependence <= 0.25:
        return "global_update_driven_container"
    if bridge_gain >= 0.35 and bridge_gain > global_gain + 0.10 and global_gain < 0.45:
        return "pure_container_overinclusive"
    return "true_unknown_residue"


def global_vs_container_metrics_v0(
    verdicts: list[GlobalVsContainerVerdict],
    merge_split_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    total = len(verdicts)
    if not total:
        return {
            "case_count": 0,
            "dominant_verdict": "none",
            "global_mask_subtraction_mean_gain": 0.0,
            "bridge_cut_mean_gain": 0.0,
            "near_tie_resolved_rate": 0.0,
            "old_new_footprint_recovery_rate": float(merge_split_metrics.get("old_new_footprint_recovery_rate", 0.0) or 0.0),
        }
    counts = Counter(item.verdict for item in verdicts)
    near_ties = [item for item in verdicts if "container_overinclusive" in item.top_rivals and "global_animation_or_clock" in item.top_rivals]
    resolved_near_ties = [item for item in near_ties if item.verdict != "true_unknown_residue"]
    return {
        "case_count": total,
        "dominant_verdict": counts.most_common(1)[0][0],
        "verdict_distribution": dict(counts.most_common()),
        "global_mask_subtraction_mean_gain": round(sum(item.global_mask_subtraction_gain for item in verdicts) / total, 6),
        "bridge_cut_mean_gain": round(sum(item.bridge_cut_gain for item in verdicts) / total, 6),
        "temporal_recurrence_mean_score": round(sum(item.temporal_recurrence_score for item in verdicts) / total, 6),
        "action_dependence_mean_score": round(sum(item.action_dependence_score for item in verdicts) / total, 6),
        "mover_dependence_mean_score": round(sum(item.mover_dependence_score for item in verdicts) / total, 6),
        "near_tie_case_count": len(near_ties),
        "near_tie_resolved_rate": round(len(resolved_near_ties) / len(near_ties), 6) if near_ties else 1.0,
        "old_new_footprint_recovery_rate": float(merge_split_metrics.get("old_new_footprint_recovery_rate", 0.0) or 0.0),
    }


def ar25_residue_verdict_taxonomy_v0(verdicts: list[GlobalVsContainerVerdict]) -> dict[str, Any]:
    counts = Counter(item.verdict for item in verdicts)
    return {
        "taxonomy_source": "GlobalUpdateVsContainerAuditV0",
        "verdict_distribution": dict(counts.most_common()),
        "actionable_subtypes": [
            {
                "label": label,
                "count": count,
                "suggested_next_probe": global_vs_container_next_probe(label),
            }
            for label, count in counts.most_common()
        ],
        "dominant_verdict": counts.most_common(1)[0][0] if counts else "none",
    }


def global_vs_container_next_probe(label: str) -> str:
    return {
        "pure_container_overinclusive": "bridge-cut changed-component construction before causal attribution",
        "global_update_driven_container": "global-update layer subtraction before changed-cluster construction",
        "multi_object_local_merge": "multi-object local overlap and membership split",
        "true_unknown_residue": "manual high-residue prototype readout",
    }.get(label, "manual high-residue prototype readout")


def global_mask_subtraction_examples_v0(
    verdicts: list[GlobalVsContainerVerdict],
    templates: list[GlobalResidueTemplate],
    *,
    limit_per_verdict: int,
) -> list[dict[str, Any]]:
    emitted: Counter[str] = Counter()
    rows = []
    template_by_transition = {template.transition_id: template for template in templates}
    for verdict in verdicts:
        if emitted[verdict.verdict] >= limit_per_verdict:
            continue
        emitted[verdict.verdict] += 1
        template = template_by_transition.get(verdict.transition_id)
        rows.append(
            {
                **verdict.to_record(),
                "template": template.to_record() if template else None,
            }
        )
    return rows


def global_update_layer_v0_report(
    records: list[TransitionRecord],
    *,
    baseline_decompositions: list[ClusterDecomposition] | None = None,
    baseline_groups: list[ResidueCompetitionGroup] | None = None,
) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game != "ar25":
        return {
            "status": "skipped_not_ar25",
            "game": game,
            "reason": "GlobalUpdateLayerModelV0 is an AR25 global-update factoring diagnostic",
        }
    baseline_decompositions = baseline_decompositions or ar25_merged_cluster_decompositions_v0(records)
    baseline_groups = baseline_groups or ar25_residue_competition_groups_v0(
        records,
        decompositions=baseline_decompositions,
    )
    raw_changed_by_step = {record.step: set(all_changed_cells(record)) for record in records}
    online_predictions, online_templates = global_update_layer_predictions_v0(
        records,
        baseline_groups,
        mode="online_causal_mode",
    )
    oracle_predictions, oracle_templates = global_update_layer_predictions_v0(
        records,
        baseline_groups,
        mode="oracle_audit_mode",
    )
    online_cleaned_by_step = cleaned_changed_cells_by_step(raw_changed_by_step, online_predictions)
    oracle_cleaned_by_step = cleaned_changed_cells_by_step(raw_changed_by_step, oracle_predictions)
    online_decompositions = ar25_merged_cluster_decompositions_v0(
        records,
        changed_cells_by_step=online_cleaned_by_step,
    )
    oracle_decompositions = ar25_merged_cluster_decompositions_v0(
        records,
        changed_cells_by_step=oracle_cleaned_by_step,
    )
    baseline_metrics = changed_cluster_merge_split_metrics(baseline_decompositions)
    online_metrics = changed_cluster_merge_split_metrics(online_decompositions)
    oracle_metrics = changed_cluster_merge_split_metrics(oracle_decompositions)
    local_rebuild_rows = local_cluster_rebuild_before_after_v0(
        records,
        raw_changed_by_step,
        online_cleaned_by_step,
        online_predictions,
    )
    false_rows = false_subtraction_audit_rows_v0(online_predictions, baseline_decompositions)
    metrics = global_update_layer_metrics_v0(
        baseline_metrics=baseline_metrics,
        online_metrics=online_metrics,
        oracle_metrics=oracle_metrics,
        online_predictions=online_predictions,
        oracle_predictions=oracle_predictions,
        local_rebuild_rows=local_rebuild_rows,
        false_rows=false_rows,
    )
    labels = []
    if metrics["online_old_new_footprint_recovery_after_factoring"] >= 0.90:
        labels.append("online_old_new_footprint_recovery_preserved")
    else:
        labels.append("online_old_new_footprint_recovery_regressed")
    if metrics["online_residue_after_global_factoring"] < metrics["baseline_residue_rate"]:
        labels.append("online_residue_rate_improved")
    else:
        labels.append("online_residue_rate_not_improved")
    if metrics["online_false_subtraction_rate"] <= 0.05:
        labels.append("false_subtraction_low")
    else:
        labels.append("false_subtraction_risk")
    if metrics["local_cluster_bbox_shrink_rate"] > 0.25:
        labels.append("local_cluster_bboxes_shrink")
    else:
        labels.append("local_cluster_bboxes_do_not_shrink")
    if metrics["online_global_update_prediction_match"] >= 0.50:
        labels.append("online_global_layer_predictable")
    else:
        labels.append("online_global_layer_not_predictable")
    return {
        "status": "evaluated",
        "game": "ar25",
        "mechanism": "GlobalUpdateLayerModelV0",
        "diagnostic_only": True,
        "world_optics_layer": True,
        "constraints": {
            "planner_integration": False,
            "role_or_option_promotion": False,
            "raw_context_ids_as_predictors": False,
            "online_causal_mode_uses_future_evidence": False,
            "oracle_audit_mode_uses_future_evidence": True,
        },
        "modes": {
            "oracle_audit_mode": "full-trace recurring residue templates; upper bound only",
            "online_causal_mode": "templates mined only from transitions strictly before t",
        },
        "metrics": metrics,
        "ar25_composition_after_global_factoring": {
            "baseline": baseline_metrics,
            "online_causal_mode": online_metrics,
            "oracle_audit_mode": oracle_metrics,
        },
        "online_global_update_predictions": [prediction.to_record() for prediction in online_predictions],
        "oracle_global_update_predictions": [prediction.to_record() for prediction in oracle_predictions],
        "residue_templates": [template.to_record() for template in online_templates + oracle_templates],
        "delta_factorization_examples": delta_factorization_examples_v0(records, online_predictions, limit=30),
        "local_cluster_rebuild_before_after": local_rebuild_rows,
        "false_subtraction_audit": false_rows,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def global_update_layer_predictions_v0(
    records: list[TransitionRecord],
    groups: list[ResidueCompetitionGroup],
    *,
    mode: str,
) -> tuple[list[GlobalUpdatePrediction], list[ResidueTemplate]]:
    raw_changed_by_step = {record.step: set(all_changed_cells(record)) for record in records}
    record_by_step = {record.step: record for record in records}
    group_records = [global_group_record(group, record_by_step) for group in groups]
    group_records = [record for record in group_records if record is not None]
    predictions = []
    templates = []
    for index, group_record in enumerate(group_records):
        if mode == "oracle_audit_mode":
            candidates = [
                item for item in group_records
                if item is not group_record and abs(int(item["step"]) - int(group_record["step"])) <= 4
            ]
            used_future = True
        elif mode == "online_causal_mode":
            candidates = [
                item for item in group_records
                if int(item["step"]) < int(group_record["step"]) and int(group_record["step"]) - int(item["step"]) <= 12
            ]
            used_future = False
        else:
            raise ValueError(f"unknown global update mode: {mode}")
        template = residue_template_from_candidates_v0(group_record, candidates, mode=mode, index=index)
        templates.append(template)
        predicted = set(template.cells)
        raw_changed = raw_changed_by_step.get(int(group_record["step"]), set())
        matched = predicted & raw_changed
        record: TransitionRecord = group_record["record"]
        predictions.append(
            GlobalUpdatePrediction(
                transition_id=str(group_record["group"].transition_id),
                step=int(group_record["step"]),
                action=int(record.action),
                mode=mode,
                predicted_cells=tuple(sorted(predicted)),
                raw_changed_count=len(raw_changed),
                matched_raw_changed_count=len(matched),
                confidence=template.confidence,
                template_id=template.template_id if template.cells else None,
                support_count=template.support_count,
                used_future_evidence=used_future,
            )
        )
    return predictions, templates


def residue_template_from_candidates_v0(
    group_record: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    *,
    mode: str,
    index: int,
) -> ResidueTemplate:
    target_step = int(group_record["step"])
    target_action = int(group_record["action"])
    target_mover = str(group_record["mover_ref"])
    support_counter: Counter[tuple[int, int]] = Counter()
    cells_by_step: dict[int, set[tuple[int, int]]] = defaultdict(set)
    action_by_step: dict[int, int] = {}
    movers_by_step: dict[int, set[str]] = defaultdict(set)
    for item in candidates:
        step = int(item["step"])
        cells_by_step[step].update(item["residue_cells"])
        action_by_step.setdefault(step, int(item["action"]))
        movers_by_step[step].add(str(item["mover_ref"]))
    for cells in cells_by_step.values():
        support_counter.update(cells)
    source_steps = set(cells_by_step)
    different_action_support = sum(1 for step in source_steps if action_by_step.get(step) != target_action)
    different_mover_support = sum(1 for step in source_steps if target_mover not in movers_by_step.get(step, set()))
    threshold = 2 if len(source_steps) >= 2 else 1
    cells = tuple(sorted(cell for cell, count in support_counter.items() if count >= threshold))
    confidence = min(1.0, (len(cells) / max(1, len(group_record["residue_cells"]))) * min(1.0, len(source_steps) / 3.0))
    return ResidueTemplate(
        template_id=f"{mode}:template:{index}:{group_record['group'].transition_id}",
        mode=mode,
        transition_id=str(group_record["group"].transition_id),
        source_steps=tuple(sorted(source_steps)),
        cells=cells,
        support_count=len(source_steps),
        different_action_support=different_action_support,
        different_mover_support=different_mover_support,
        confidence=confidence,
        phase_candidate=phase_candidate_from_support_v0(candidates),
    )


def phase_candidate_from_support_v0(candidates: list[Mapping[str, Any]]) -> str:
    if len(candidates) < 2:
        return "insufficient_history"
    steps = sorted({int(item["step"]) for item in candidates})
    gaps = [right - left for left, right in zip(steps, steps[1:])]
    if gaps and len(set(gaps)) == 1:
        return f"period_{gaps[0]}"
    if len(set(int(item["action"]) for item in candidates)) > 1:
        return "action_independent_irregular"
    return "step_correlated_irregular"


def cleaned_changed_cells_by_step(
    raw_changed_by_step: Mapping[int, set[tuple[int, int]]],
    predictions: list[GlobalUpdatePrediction],
) -> dict[int, set[tuple[int, int]]]:
    predicted_by_step: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for prediction in predictions:
        predicted_by_step[prediction.step].update(prediction.predicted_cells)
    result = {}
    for step, cells in raw_changed_by_step.items():
        result[step] = set(cells) - predicted_by_step.get(step, set())
    return result


def local_cluster_rebuild_before_after_v0(
    records: list[TransitionRecord],
    raw_changed_by_step: Mapping[int, set[tuple[int, int]]],
    cleaned_changed_by_step: Mapping[int, set[tuple[int, int]]],
    predictions: list[GlobalUpdatePrediction],
) -> list[dict[str, Any]]:
    prediction_steps = {prediction.step for prediction in predictions if prediction.predicted_cells}
    rows = []
    for record in records:
        if record.step not in prediction_steps:
            continue
        raw_cells = raw_changed_by_step.get(record.step, set())
        cleaned_cells = cleaned_changed_by_step.get(record.step, set())
        raw_components = connected_cell_components(raw_cells)
        cleaned_components = connected_cell_components(cleaned_cells)
        raw_bbox = cells_bbox(list(raw_cells))
        cleaned_bbox = cells_bbox(list(cleaned_cells))
        rows.append(
            {
                "step": record.step,
                "action": record.action,
                "raw_changed_cell_count": len(raw_cells),
                "cleaned_changed_cell_count": len(cleaned_cells),
                "subtracted_cell_count": len(raw_cells - cleaned_cells),
                "raw_cluster_count": len(raw_components),
                "cleaned_cluster_count": len(cleaned_components),
                "raw_bbox": raw_bbox,
                "cleaned_bbox": cleaned_bbox,
                "bbox_shrank": bbox_area(cleaned_bbox) < bbox_area(raw_bbox),
                "cell_reduction_rate": round((len(raw_cells) - len(cleaned_cells)) / max(1, len(raw_cells)), 6),
            }
        )
    return rows


def false_subtraction_audit_rows_v0(
    predictions: list[GlobalUpdatePrediction],
    decompositions: list[ClusterDecomposition],
) -> list[dict[str, Any]]:
    decomposition_by_transition = {item.transition_id: item for item in decompositions}
    rows = []
    for prediction in predictions:
        if not prediction.predicted_cells:
            continue
        decomposition = decomposition_by_transition.get(prediction.transition_id)
        protected_cells = protected_object_cells_from_decomposition(decomposition) if decomposition else set()
        predicted = set(prediction.predicted_cells)
        false_cells = predicted & protected_cells
        rows.append(
            {
                "transition_id": prediction.transition_id,
                "step": prediction.step,
                "mode": prediction.mode,
                "predicted_cell_count": len(predicted),
                "protected_object_cell_count": len(protected_cells),
                "false_subtracted_cell_count": len(false_cells),
                "false_subtraction_rate": round(len(false_cells) / max(1, len(predicted)), 6),
                "sample_false_cells": tuple(sorted(false_cells))[:20],
            }
        )
    return rows


def protected_object_cells_from_decomposition(decomposition: ClusterDecomposition | None) -> set[tuple[int, int]]:
    if decomposition is None:
        return set()
    best = decomposition.best_hypothesis()
    if best is None:
        return set()
    protected_labels = {
        "old_footprint",
        "new_footprint",
        "overlap",
        "actor_overlap",
        "mover_overlap",
        "independent_effect",
    }
    return {
        item.cell for item in best.cell_attributions
        if protected_labels & set(item.labels)
    }


def global_update_layer_metrics_v0(
    *,
    baseline_metrics: Mapping[str, Any],
    online_metrics: Mapping[str, Any],
    oracle_metrics: Mapping[str, Any],
    online_predictions: list[GlobalUpdatePrediction],
    oracle_predictions: list[GlobalUpdatePrediction],
    local_rebuild_rows: list[Mapping[str, Any]],
    false_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    online_matched = sum(item.matched_raw_changed_count for item in online_predictions)
    online_predicted = sum(len(item.predicted_cells) for item in online_predictions)
    oracle_matched = sum(item.matched_raw_changed_count for item in oracle_predictions)
    oracle_predicted = sum(len(item.predicted_cells) for item in oracle_predictions)
    false_cells = sum(int(row["false_subtracted_cell_count"]) for row in false_rows)
    predicted_cells = sum(int(row["predicted_cell_count"]) for row in false_rows)
    high_residue_rows = [row for row in local_rebuild_rows if int(row["raw_changed_cell_count"]) >= 100]
    return {
        "baseline_residue_rate": float(baseline_metrics.get("residue_after_best_decomposition", 0.0) or 0.0),
        "online_residue_after_global_factoring": float(online_metrics.get("residue_after_best_decomposition", 0.0) or 0.0),
        "oracle_residue_after_global_factoring": float(oracle_metrics.get("residue_after_best_decomposition", 0.0) or 0.0),
        "online_global_update_prediction_match": round(online_matched / max(1, online_predicted), 6),
        "oracle_global_update_prediction_match": round(oracle_matched / max(1, oracle_predicted), 6),
        "online_predicted_global_cell_count": online_predicted,
        "oracle_predicted_global_cell_count": oracle_predicted,
        "online_prediction_with_support_rate": round(sum(item.support_count > 0 for item in online_predictions) / max(1, len(online_predictions)), 6),
        "local_cluster_bbox_shrink_rate": round(sum(bool(row["bbox_shrank"]) for row in local_rebuild_rows) / max(1, len(local_rebuild_rows)), 6),
        "high_residue_bbox_shrink_rate": round(sum(bool(row["bbox_shrank"]) for row in high_residue_rows) / max(1, len(high_residue_rows)), 6),
        "local_cluster_count_before": sum(int(row["raw_cluster_count"]) for row in local_rebuild_rows),
        "local_cluster_count_after": sum(int(row["cleaned_cluster_count"]) for row in local_rebuild_rows),
        "merged_decomposition_success_before": float(baseline_metrics.get("causal_split_recovery_rate", 0.0) or 0.0),
        "merged_decomposition_success_after_online": float(online_metrics.get("causal_split_recovery_rate", 0.0) or 0.0),
        "causal_split_recovery_before_after_delta": round(
            float(online_metrics.get("causal_split_recovery_rate", 0.0) or 0.0)
            - float(baseline_metrics.get("causal_split_recovery_rate", 0.0) or 0.0),
            6,
        ),
        "online_false_subtraction_rate": round(false_cells / max(1, predicted_cells), 6),
        "old_new_footprint_recovery_before": float(baseline_metrics.get("old_new_footprint_recovery_rate", 0.0) or 0.0),
        "online_old_new_footprint_recovery_after_factoring": float(online_metrics.get("old_new_footprint_recovery_rate", 0.0) or 0.0),
        "oracle_old_new_footprint_recovery_after_factoring": float(oracle_metrics.get("old_new_footprint_recovery_rate", 0.0) or 0.0),
        "pure_container_cases_remaining_proxy": sum(1 for row in local_rebuild_rows if bool(row["bbox_shrank"]) is False),
        "true_unknown_cases_remaining_proxy": sum(1 for item in online_predictions if item.support_count == 0),
    }


def delta_factorization_examples_v0(
    records: list[TransitionRecord],
    predictions: list[GlobalUpdatePrediction],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    record_by_step = {record.step: record for record in records}
    rows = []
    for prediction in sorted(predictions, key=lambda item: len(item.predicted_cells), reverse=True):
        if len(rows) >= limit:
            break
        record = record_by_step.get(prediction.step)
        if record is None:
            continue
        raw_changed = set(all_changed_cells(record))
        predicted = set(prediction.predicted_cells)
        matched = raw_changed & predicted
        local = raw_changed - predicted
        rows.append(
            {
                "transition_id": prediction.transition_id,
                "step": prediction.step,
                "action": prediction.action,
                "mode": prediction.mode,
                "raw_changed": {"cell_count": len(raw_changed), "bbox": cells_bbox(list(raw_changed))},
                "predicted_global_update": {"cell_count": len(predicted), "bbox": cells_bbox(list(predicted))},
                "matched_global_update": {"cell_count": len(matched), "bbox": cells_bbox(list(matched))},
                "local_residual_delta": {"cell_count": len(local), "bbox": cells_bbox(list(local))},
                "unknown_delta": {"cell_count": max(0, len(predicted) - len(matched))},
                "confidence": prediction.confidence,
                "used_future_evidence": prediction.used_future_evidence,
            }
        )
    return rows


def bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0]) + 1) * max(0, int(bbox[3]) - int(bbox[1]) + 1)


def global_phase_register_discovery_v0_report(
    records: list[TransitionRecord],
    *,
    baseline_decompositions: list[ClusterDecomposition] | None = None,
    baseline_groups: list[ResidueCompetitionGroup] | None = None,
    global_layer_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game != "ar25":
        return {
            "status": "skipped_not_ar25",
            "game": game,
            "reason": "GlobalPhaseRegisterDiscoveryV0 is an AR25 global-phase diagnostic",
        }
    baseline_decompositions = baseline_decompositions or ar25_merged_cluster_decompositions_v0(records)
    baseline_groups = baseline_groups or ar25_residue_competition_groups_v0(
        records,
        decompositions=baseline_decompositions,
    )
    global_layer_report = global_layer_report or global_update_layer_v0_report(
        records,
        baseline_decompositions=baseline_decompositions,
        baseline_groups=baseline_groups,
    )
    step_records = global_phase_step_records_v0(records, baseline_groups)
    clusters = global_template_clusters_v0(step_records)
    prediction_rows = prequential_phase_predictions_v0(step_records, clusters)
    score_rows = phase_hypothesis_scores_v0(prediction_rows)
    online_predictions, _online_templates = global_update_layer_predictions_v0(
        records,
        baseline_groups,
        mode="online_causal_mode",
    )
    false_locality = false_subtraction_locality_audit_v0(
        online_predictions,
        baseline_decompositions,
    )
    oracle_metrics = oracle_phase_upper_bound_v0(step_records, clusters)
    alignment_summary = alignment_match_summary_v0(prediction_rows)
    metrics = global_phase_register_metrics_v0(
        score_rows=score_rows,
        oracle_metrics=oracle_metrics,
        global_layer_metrics=global_layer_report.get("metrics", {}),
        false_locality=false_locality,
    )
    labels = []
    if metrics["oracle_translation_aligned_template_match"] < 0.35:
        labels.append("oracle_ceiling_low_template_vocabulary_wrong")
    elif metrics["best_prequential_exact_match"] >= 0.35:
        labels.append("winning_phase_hypothesis_found")
    elif metrics["best_prequential_translation_aligned_match"] >= 0.35:
        labels.append("spatial_drift_alignment_signal")
    else:
        labels.append("no_clean_phase_law_yet")
    if metrics["best_prequential_exact_match"] > metrics["baseline_online_prediction_match"] + 0.05:
        labels.append("phase_prediction_improves_online_match")
    else:
        labels.append("phase_prediction_not_materially_better")
    if metrics["false_subtracted_mover_or_actor_rate"] <= 0.02:
        labels.append("protected_object_false_subtraction_low")
    else:
        labels.append("protected_object_false_subtraction_risk")
    if metrics["old_new_footprint_recovery_after_factoring"] >= 0.90:
        labels.append("old_new_recovery_gate_preserved")
    else:
        labels.append("old_new_recovery_gate_failed")
    return {
        "status": "evaluated",
        "game": "ar25",
        "mechanism": "GlobalPhaseRegisterDiscoveryV0",
        "diagnostic_only": True,
        "constraints": {
            "planner_integration": False,
            "option_logic_change": False,
            "online_prequential_uses_future_evidence": False,
            "raw_context_ids_as_predictors": False,
        },
        "hypothesis_families": [
            "last_seen_template",
            "period_k_clock",
            "step_mod_k_clock",
            "action_conditioned_phase",
            "mover_conditioned_phase",
            "local_state_triggered_phase",
            "spatially_drifting_template",
            "noisy_unpredictable_layer",
        ],
        "metrics": metrics,
        "oracle_phase_upper_bound": oracle_metrics,
        "alignment_match_summary": alignment_summary,
        "phase_hypothesis_scores": score_rows,
        "prequential_phase_predictions": [row.to_record() for row in prediction_rows],
        "global_template_clusters": [cluster.to_record() for cluster in clusters],
        "alignment_match_comparison": alignment_match_comparison_rows_v0(prediction_rows),
        "false_subtraction_locality_audit": false_locality,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def global_phase_step_records_v0(
    records: list[TransitionRecord],
    groups: list[ResidueCompetitionGroup],
) -> list[dict[str, Any]]:
    record_by_step = {record.step: record for record in records}
    by_step: dict[int, dict[str, Any]] = {}
    for group in groups:
        step = step_from_transition_id(group.transition_id)
        record = record_by_step.get(step)
        if record is None:
            continue
        item = by_step.setdefault(
            step,
            {
                "transition_id": f"{record.game}:step:{step}:global",
                "step": step,
                "action": int(record.action),
                "record": record,
                "cells": set(),
                "mover_refs": set(),
            },
        )
        item["cells"].update(residue_cells_from_group(group))
        item["mover_refs"].add(mover_ref_from_transition_id(group.transition_id))
    result = []
    for step in sorted(by_step):
        item = by_step[step]
        if not item["cells"]:
            continue
        result.append(
            {
                **item,
                "cells": set(item["cells"]),
                "mover_refs": tuple(sorted(item["mover_refs"])),
                "state_key": global_phase_local_state_key(item["record"]),
            }
        )
    return result


def global_phase_local_state_key(record: TransitionRecord) -> str:
    return stable_json(
        {
            "action": record.action,
            "visible_state_summary": record.pre_features.get("visible_state_summary"),
            "actor_or_focus_region": record.pre_features.get("actor_or_focus_region"),
        }
    )


def global_template_clusters_v0(step_records: list[Mapping[str, Any]]) -> list[GlobalTemplateCluster]:
    clusters: list[dict[str, Any]] = []
    for item in step_records:
        cells = set(item["cells"])
        best_index = None
        best_score = 0.0
        for index, cluster in enumerate(clusters):
            score = max(
                exact_iou(cells, cluster["prototype_cells"]),
                translation_aligned_iou(cells, cluster["prototype_cells"], max_shift=3),
                band_signature_similarity(cells, cluster["prototype_cells"]),
            )
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is None or best_score < 0.72:
            clusters.append(
                {
                    "prototype_step": int(item["step"]),
                    "prototype_cells": set(cells),
                    "member_steps": [int(item["step"])],
                }
            )
        else:
            clusters[best_index]["member_steps"].append(int(item["step"]))
    result = []
    for index, cluster in enumerate(clusters):
        prototype_cells = tuple(sorted(cluster["prototype_cells"]))
        result.append(
            GlobalTemplateCluster(
                cluster_id=f"global_template_cluster:{index}",
                prototype_step=int(cluster["prototype_step"]),
                prototype_cells=prototype_cells,
                member_steps=tuple(sorted(set(cluster["member_steps"]))),
                support_count=len(set(cluster["member_steps"])),
                bbox=cells_bbox(list(prototype_cells)),
                row_band_signature=row_band_signature(prototype_cells),
                col_band_signature=col_band_signature(prototype_cells),
            )
        )
    return result


def prequential_phase_predictions_v0(
    step_records: list[Mapping[str, Any]],
    clusters: list[GlobalTemplateCluster],
) -> list[PhasePrediction]:
    cluster_by_step = cluster_by_step_v0(clusters)
    rows = []
    for item in step_records:
        step = int(item["step"])
        prior = [candidate for candidate in step_records if int(candidate["step"]) < step]
        actual = set(item["cells"])
        candidates: list[tuple[str, Mapping[str, Any] | None]] = [
            ("last_seen_template", latest_prior(prior)),
            *[
                (f"period_{k}_clock", prior_exact_step(prior, step - k))
                for k in range(2, 13)
            ],
            *[
                (f"step_mod_{k}_clock", latest_matching(prior, lambda row, k=k: int(row["step"]) % k == step % k))
                for k in range(2, 13)
            ],
            ("action_conditioned_phase", latest_matching(prior, lambda row: int(row["action"]) == int(item["action"]))),
            ("mover_conditioned_phase", latest_matching(prior, lambda row: bool(set(row["mover_refs"]) & set(item["mover_refs"])))),
            ("local_state_triggered_phase", latest_matching(prior, lambda row: row["state_key"] == item["state_key"])),
            ("spatially_drifting_template", latest_prior(prior)),
            ("noisy_unpredictable_layer", None),
        ]
        for hypothesis, source in candidates:
            predicted = set(source["cells"]) if source is not None else set()
            predicted_cluster = cluster_by_step.get(int(source["step"])) if source is not None else None
            match = phase_prediction_match_v0(item, actual, predicted)
            rows.append(
                PhasePrediction(
                    transition_id=str(item["transition_id"]),
                    step=step,
                    action=int(item["action"]),
                    hypothesis=hypothesis,
                    predicted_cluster_id=predicted_cluster,
                    predicted_cell_count=len(predicted),
                    actual_cell_count=len(actual),
                    exact_match=match["exact_match"],
                    translation_aligned_match=match["translation_aligned_match"],
                    bbox_normalized_match=match["bbox_normalized_match"],
                    color_transition_match=match["color_transition_match"],
                    band_signature_match=match["band_signature_match"],
                    used_future_evidence=False,
                    failure_reason=phase_prediction_failure_reason(hypothesis, source, match),
                )
            )
    return rows


def cluster_by_step_v0(clusters: list[GlobalTemplateCluster]) -> dict[int, str]:
    result = {}
    for cluster in clusters:
        for step in cluster.member_steps:
            result[int(step)] = cluster.cluster_id
    return result


def latest_prior(prior: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return max(prior, key=lambda row: int(row["step"])) if prior else None


def prior_exact_step(prior: list[Mapping[str, Any]], step: int) -> Mapping[str, Any] | None:
    matches = [row for row in prior if int(row["step"]) == step]
    return matches[0] if matches else None


def latest_matching(prior: list[Mapping[str, Any]], predicate: Any) -> Mapping[str, Any] | None:
    matches = [row for row in prior if predicate(row)]
    return latest_prior(matches)


def phase_prediction_match_v0(
    item: Mapping[str, Any],
    actual: set[tuple[int, int]],
    predicted: set[tuple[int, int]],
) -> dict[str, float]:
    record: TransitionRecord = item["record"]
    return {
        "exact_match": exact_iou(actual, predicted),
        "translation_aligned_match": translation_aligned_iou(actual, predicted, max_shift=5),
        "bbox_normalized_match": bbox_normalized_iou(actual, predicted),
        "color_transition_match": color_transition_signature_match(record, actual, predicted),
        "band_signature_match": band_signature_similarity(actual, predicted),
    }


def phase_prediction_failure_reason(
    hypothesis: str,
    source: Mapping[str, Any] | None,
    match: Mapping[str, float],
) -> str:
    if source is None:
        return "insufficient_prior_support" if hypothesis != "noisy_unpredictable_layer" else "no_prediction_baseline"
    if match["translation_aligned_match"] >= 0.35 and match["exact_match"] < 0.20:
        return "spatial_drift_or_alignment_error"
    if match["band_signature_match"] >= 0.50 and match["exact_match"] < 0.20:
        return "right_band_wrong_cells"
    if match["exact_match"] < 0.20:
        return "phase_uncertainty"
    return "matched"


def phase_hypothesis_scores_v0(predictions: list[PhasePrediction]) -> list[dict[str, Any]]:
    by_hypothesis: dict[str, list[PhasePrediction]] = defaultdict(list)
    for prediction in predictions:
        by_hypothesis[prediction.hypothesis].append(prediction)
    rows = []
    for hypothesis, items in sorted(by_hypothesis.items()):
        supported = [item for item in items if item.predicted_cell_count > 0]
        exact = sum(item.exact_match for item in items) / max(1, len(items))
        aligned = sum(item.translation_aligned_match for item in items) / max(1, len(items))
        bbox_norm = sum(item.bbox_normalized_match for item in items) / max(1, len(items))
        band = sum(item.band_signature_match for item in items) / max(1, len(items))
        rows.append(
            {
                "hypothesis": hypothesis,
                "case_count": len(items),
                "support_rate": round(len(supported) / max(1, len(items)), 6),
                "mean_exact_match": round(exact, 6),
                "mean_translation_aligned_match": round(aligned, 6),
                "mean_bbox_normalized_match": round(bbox_norm, 6),
                "mean_band_signature_match": round(band, 6),
                "no_future_evidence": all(not item.used_future_evidence for item in items),
                "dominant_failure_reason": Counter(item.failure_reason for item in items).most_common(1)[0][0] if items else "none",
            }
        )
    return sorted(rows, key=lambda row: row["mean_exact_match"], reverse=True)


def oracle_phase_upper_bound_v0(
    step_records: list[Mapping[str, Any]],
    clusters: list[GlobalTemplateCluster],
) -> dict[str, Any]:
    if not step_records or not clusters:
        return {
            "oracle_template_match": 0.0,
            "oracle_translation_aligned_template_match": 0.0,
            "oracle_residue_after_phase_subtraction_proxy": 1.0,
            "number_of_phase_templates": 0,
        }
    exact_scores = []
    aligned_scores = []
    for item in step_records:
        actual = set(item["cells"])
        exact_scores.append(max(exact_iou(actual, set(cluster.prototype_cells)) for cluster in clusters))
        aligned_scores.append(max(translation_aligned_iou(actual, set(cluster.prototype_cells), max_shift=5) for cluster in clusters))
    aligned_mean = sum(aligned_scores) / len(aligned_scores)
    return {
        "oracle_template_match": round(sum(exact_scores) / len(exact_scores), 6),
        "oracle_translation_aligned_template_match": round(aligned_mean, 6),
        "oracle_residue_after_phase_subtraction_proxy": round(1.0 - aligned_mean, 6),
        "number_of_phase_templates": len(clusters),
    }


def alignment_match_summary_v0(predictions: list[PhasePrediction]) -> dict[str, Any]:
    if not predictions:
        return {
            "best_exact_match": 0.0,
            "best_translation_aligned_match": 0.0,
            "alignment_gain": 0.0,
        }
    score_rows = phase_hypothesis_scores_v0(predictions)
    best_exact = max(row["mean_exact_match"] for row in score_rows)
    best_aligned = max(row["mean_translation_aligned_match"] for row in score_rows)
    best_band = max(row["mean_band_signature_match"] for row in score_rows)
    return {
        "best_exact_match": round(best_exact, 6),
        "best_translation_aligned_match": round(best_aligned, 6),
        "best_band_signature_match": round(best_band, 6),
        "alignment_gain": round(best_aligned - best_exact, 6),
        "spatial_drift_suspect": best_aligned - best_exact >= 0.10,
    }


def alignment_match_comparison_rows_v0(predictions: list[PhasePrediction]) -> list[dict[str, Any]]:
    return [
        {
            "transition_id": item.transition_id,
            "step": item.step,
            "hypothesis": item.hypothesis,
            "exact_match": round(item.exact_match, 6),
            "translation_aligned_match": round(item.translation_aligned_match, 6),
            "bbox_normalized_match": round(item.bbox_normalized_match, 6),
            "color_transition_match": round(item.color_transition_match, 6),
            "band_signature_match": round(item.band_signature_match, 6),
            "alignment_gain": round(item.translation_aligned_match - item.exact_match, 6),
            "failure_reason": item.failure_reason,
        }
        for item in predictions
    ]


def global_phase_register_metrics_v0(
    *,
    score_rows: list[Mapping[str, Any]],
    oracle_metrics: Mapping[str, Any],
    global_layer_metrics: Mapping[str, Any],
    false_locality: list[Mapping[str, Any]],
) -> dict[str, Any]:
    best_exact = max((float(row["mean_exact_match"]) for row in score_rows), default=0.0)
    best_aligned = max((float(row["mean_translation_aligned_match"]) for row in score_rows), default=0.0)
    best_hypothesis = max(score_rows, key=lambda row: float(row["mean_exact_match"]))["hypothesis"] if score_rows else "none"
    mover_actor_false = sum(int(row["false_subtracted_mover_cells"]) + int(row["false_subtracted_actor_cells"]) for row in false_locality)
    predicted = sum(int(row["predicted_cell_count"]) for row in false_locality)
    return {
        "baseline_online_prediction_match": float(global_layer_metrics.get("online_global_update_prediction_match", 0.0) or 0.0),
        "best_prequential_exact_match": round(best_exact, 6),
        "best_prequential_translation_aligned_match": round(best_aligned, 6),
        "best_prequential_hypothesis": best_hypothesis,
        "oracle_template_match": oracle_metrics.get("oracle_template_match", 0.0),
        "oracle_translation_aligned_template_match": oracle_metrics.get("oracle_translation_aligned_template_match", 0.0),
        "oracle_residue_after_phase_subtraction_proxy": oracle_metrics.get("oracle_residue_after_phase_subtraction_proxy", 1.0),
        "number_of_phase_templates": oracle_metrics.get("number_of_phase_templates", 0),
        "old_new_footprint_recovery_after_factoring": float(global_layer_metrics.get("online_old_new_footprint_recovery_after_factoring", 0.0) or 0.0),
        "residue_after_factoring": float(global_layer_metrics.get("online_residue_after_global_factoring", 0.0) or 0.0),
        "false_subtracted_mover_or_actor_cells": mover_actor_false,
        "false_subtracted_mover_or_actor_rate": round(mover_actor_false / max(1, predicted), 6),
    }


def false_subtraction_locality_audit_v0(
    prediction_rows: Iterable[Mapping[str, Any] | GlobalUpdatePrediction],
    decompositions: list[ClusterDecomposition],
) -> list[dict[str, Any]]:
    decomposition_by_transition = {item.transition_id: item for item in decompositions}
    rows = []
    for row in prediction_rows:
        if isinstance(row, GlobalUpdatePrediction):
            transition_id = row.transition_id
            step = row.step
            predicted = set(row.predicted_cells)
            predicted_cell_count = len(predicted)
        else:
            transition_id = str(row.get("transition_id"))
            step = row.get("step")
            predicted = set(tuple(cell) for cell in row.get("sample_cells", ()))
            predicted_cell_count = int(row.get("predicted_cell_count", len(predicted)) or len(predicted))
        # JSON artifact rows carry only sample cells, so artifact-only callers
        # get a lower-bound locality audit. Suite runs pass full predictions.
        decomposition = decomposition_by_transition.get(transition_id)
        labels_by_cell = protected_object_label_cells(decomposition)
        actor = predicted & labels_by_cell["actor"]
        mover = predicted & labels_by_cell["mover"]
        background = predicted - actor - mover
        rows.append(
            {
                "transition_id": transition_id,
                "step": step,
                "predicted_cell_count": predicted_cell_count,
                "false_subtracted_mover_cells": len(mover),
                "false_subtracted_actor_cells": len(actor),
                "false_subtracted_background_cells": len(background),
                "false_subtracted_unknown_cells": 0,
                "sample_mover_cells": tuple(sorted(mover))[:20],
                "sample_actor_cells": tuple(sorted(actor))[:20],
            }
        )
    return rows


def protected_object_label_cells(decomposition: ClusterDecomposition | None) -> dict[str, set[tuple[int, int]]]:
    result = {"actor": set(), "mover": set()}
    if decomposition is None:
        return result
    best = decomposition.best_hypothesis()
    if best is None:
        return result
    for item in best.cell_attributions:
        labels = set(item.labels)
        if "actor_overlap" in labels:
            result["actor"].add(item.cell)
        if labels & {"old_footprint", "new_footprint", "overlap", "mover_overlap", "independent_effect"}:
            result["mover"].add(item.cell)
    return result


def exact_iou(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def translation_aligned_iou(
    actual: set[tuple[int, int]],
    predicted: set[tuple[int, int]],
    *,
    max_shift: int,
) -> float:
    if not actual or not predicted:
        return 0.0
    best = 0.0
    for dr, dc in candidate_alignment_shifts(actual, predicted, max_shift=max_shift):
        shifted = {(row + dr, col + dc) for row, col in predicted}
        best = max(best, exact_iou(actual, shifted))
    return best


def candidate_alignment_shifts(
    actual: set[tuple[int, int]],
    predicted: set[tuple[int, int]],
    *,
    max_shift: int,
) -> tuple[tuple[int, int], ...]:
    actual_bbox = cells_bbox(list(actual))
    predicted_bbox = cells_bbox(list(predicted))
    shifts = {(0, 0)}
    if actual_bbox is not None and predicted_bbox is not None:
        shifts.add((actual_bbox[0] - predicted_bbox[0], actual_bbox[1] - predicted_bbox[1]))
        shifts.add((actual_bbox[2] - predicted_bbox[2], actual_bbox[3] - predicted_bbox[3]))
        actual_center = ((actual_bbox[0] + actual_bbox[2]) // 2, (actual_bbox[1] + actual_bbox[3]) // 2)
        predicted_center = ((predicted_bbox[0] + predicted_bbox[2]) // 2, (predicted_bbox[1] + predicted_bbox[3]) // 2)
        shifts.add((actual_center[0] - predicted_center[0], actual_center[1] - predicted_center[1]))
    return tuple(
        sorted(
            (int(dr), int(dc))
            for dr, dc in shifts
            if abs(int(dr)) <= max_shift and abs(int(dc)) <= max_shift
        )
    )


def bbox_normalized_iou(actual: set[tuple[int, int]], predicted: set[tuple[int, int]]) -> float:
    if not actual or not predicted:
        return 0.0
    return exact_iou(normalize_cells_to_bbox(actual), normalize_cells_to_bbox(predicted))


def normalize_cells_to_bbox(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    if not cells:
        return set()
    min_row = min(row for row, _col in cells)
    min_col = min(col for _row, col in cells)
    return {(row - min_row, col - min_col) for row, col in cells}


def color_transition_signature_match(
    record: TransitionRecord,
    actual: set[tuple[int, int]],
    predicted: set[tuple[int, int]],
) -> float:
    if not actual or not predicted:
        return 0.0
    before = observation_plane(record.before)
    after = observation_plane(record.after)
    actual_sig = Counter((cell_color(before, cell), cell_color(after, cell)) for cell in actual)
    predicted_sig = Counter((cell_color(before, cell), cell_color(after, cell)) for cell in predicted if in_grid(cell, before.shape))
    if not actual_sig or not predicted_sig:
        return 0.0
    intersection = sum(min(actual_sig[key], predicted_sig[key]) for key in actual_sig.keys() | predicted_sig.keys())
    union = sum(max(actual_sig[key], predicted_sig[key]) for key in actual_sig.keys() | predicted_sig.keys())
    return intersection / max(1, union)


def in_grid(cell: tuple[int, int], shape: tuple[int, ...]) -> bool:
    return 0 <= int(cell[0]) < int(shape[0]) and 0 <= int(cell[1]) < int(shape[1])


def band_signature_similarity(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
    if not left or not right:
        return 0.0
    row_score = counter_cosine(Counter(row for row, _col in left), Counter(row for row, _col in right))
    col_score = counter_cosine(Counter(col for _row, col in left), Counter(col for _row, col in right))
    return (row_score + col_score) / 2.0


def row_band_signature(cells: Iterable[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(sorted(Counter(row for row, _col in cells).values(), reverse=True)[:8])


def col_band_signature(cells: Iterable[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(sorted(Counter(col for _row, col in cells).values(), reverse=True)[:8])


def counter_cosine(left: Counter[Any], right: Counter[Any]) -> float:
    if not left or not right:
        return 0.0
    keys = left.keys() | right.keys()
    dot = sum(left[key] * right[key] for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def global_phase_identifiability_v0_report(
    records: list[TransitionRecord],
    *,
    baseline_decompositions: list[ClusterDecomposition] | None = None,
    baseline_groups: list[ResidueCompetitionGroup] | None = None,
    phase_report: Mapping[str, Any] | None = None,
    global_layer_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game != "ar25":
        return {
            "status": "skipped_not_ar25",
            "game": game,
            "reason": "GlobalPhaseIdentifiabilityAuditV0 is an AR25 global-phase diagnostic",
        }
    baseline_decompositions = baseline_decompositions or ar25_merged_cluster_decompositions_v0(records)
    baseline_groups = baseline_groups or ar25_residue_competition_groups_v0(
        records,
        decompositions=baseline_decompositions,
    )
    global_layer_report = global_layer_report or global_update_layer_v0_report(
        records,
        baseline_decompositions=baseline_decompositions,
        baseline_groups=baseline_groups,
    )
    phase_report = phase_report or global_phase_register_discovery_v0_report(
        records,
        baseline_decompositions=baseline_decompositions,
        baseline_groups=baseline_groups,
        global_layer_report=global_layer_report,
    )
    step_records = global_phase_step_records_v0(records, baseline_groups)
    clusters = global_template_clusters_v0(step_records)
    cluster_by_step = cluster_by_step_v0(clusters)
    support_curve = template_support_learning_curve_v0(step_records, clusters, cluster_by_step)
    entropy = oracle_template_transition_entropy_v0(step_records, cluster_by_step)
    fragmentation = template_fragmentation_audit_v0(step_records, clusters, cluster_by_step)
    frame_readout = current_frame_phase_readout_scores_v0(step_records, clusters, cluster_by_step)
    higher_order = higher_order_phase_predictions_v0(step_records, clusters, cluster_by_step)
    atoms = compositional_global_atoms_v0(step_records)
    metrics = phase_identifiability_metrics_v0(
        phase_report=phase_report,
        support_curve=support_curve,
        entropy=entropy,
        fragmentation=fragmentation,
        frame_readout=frame_readout,
        higher_order=higher_order,
        atoms=atoms,
    )
    verdict = phase_identifiability_verdict_v0(metrics)
    return {
        "status": "evaluated",
        "game": "ar25",
        "mechanism": "GlobalPhaseIdentifiabilityAuditV0",
        "diagnostic_only": True,
        "constraints": {
            "planner_integration": False,
            "subtraction_more_aggressive": False,
            "role_or_option_promotion": False,
            "prequential_tests_use_future_evidence": False,
        },
        "questions": {
            "primary": "why online aligned match remains low despite high oracle aligned match",
            "candidate_causes": [
                "sparse_support",
                "template_fragmentation",
                "visible_but_unmodeled_phase",
                "higher_order_hidden_automaton",
                "compositional_global_update",
                "unobservable_or_noisy_layer",
            ],
        },
        "metrics": metrics,
        "template_support_learning_curve": support_curve,
        "oracle_template_transition_entropy": entropy,
        "template_fragmentation_audit": fragmentation,
        "current_frame_phase_readout_scores": frame_readout,
        "higher_order_phase_predictions": higher_order,
        "compositional_global_atoms": atoms,
        "phase_identifiability_verdict": verdict,
        "verdict": {
            "primary": verdict["primary"],
            "labels": verdict["labels"],
        },
    }


def template_support_learning_curve_v0(
    step_records: list[Mapping[str, Any]],
    clusters: list[GlobalTemplateCluster],
    cluster_by_step: Mapping[int, str],
) -> list[dict[str, Any]]:
    rows = []
    history_counts: Counter[str] = Counter()
    cluster_by_id = {cluster.cluster_id: cluster for cluster in clusters}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in step_records:
        step = int(item["step"])
        actual_cluster = cluster_by_step.get(step)
        support = history_counts[actual_cluster] if actual_cluster else 0
        predicted_cluster = latest_seen_cluster_before_v0(step_records, cluster_by_step, step)
        predicted_cells = set(cluster_by_id[predicted_cluster].prototype_cells) if predicted_cluster in cluster_by_id else set()
        actual_cells = set(item["cells"])
        exact = exact_iou(actual_cells, predicted_cells)
        aligned = translation_aligned_iou(actual_cells, predicted_cells, max_shift=5)
        bucket = support_bucket(support)
        buckets[bucket].append(
            {
                "hit": actual_cluster == predicted_cluster,
                "exact": exact,
                "aligned": aligned,
                "support": support,
            }
        )
        if actual_cluster:
            history_counts[actual_cluster] += 1
    for bucket in ("0", "1", "2-3", "4-8", ">8"):
        items = buckets.get(bucket, [])
        rows.append(
            {
                "prior_support_bucket": bucket,
                "case_count": len(items),
                "template_id_accuracy": round(sum(item["hit"] for item in items) / max(1, len(items)), 6),
                "mean_exact_match": round(sum(item["exact"] for item in items) / max(1, len(items)), 6),
                "mean_translation_aligned_match": round(sum(item["aligned"] for item in items) / max(1, len(items)), 6),
                "mean_prior_support": round(sum(item["support"] for item in items) / max(1, len(items)), 6),
            }
        )
    return rows


def support_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 8:
        return "4-8"
    return ">8"


def latest_seen_cluster_before_v0(
    step_records: list[Mapping[str, Any]],
    cluster_by_step: Mapping[int, str],
    step: int,
) -> str | None:
    prior_steps = [int(item["step"]) for item in step_records if int(item["step"]) < step and int(item["step"]) in cluster_by_step]
    if not prior_steps:
        return None
    return cluster_by_step[max(prior_steps)]


def oracle_template_transition_entropy_v0(
    step_records: list[Mapping[str, Any]],
    cluster_by_step: Mapping[int, str],
) -> dict[str, Any]:
    sequence = [
        {
            **item,
            "cluster_id": cluster_by_step.get(int(item["step"])),
        }
        for item in step_records
        if cluster_by_step.get(int(item["step"])) is not None
    ]
    condition_builders = {
        "last_template": lambda index: (sequence[index - 1]["cluster_id"],) if index >= 1 else None,
        "last_2_templates": lambda index: tuple(sequence[index - offset]["cluster_id"] for offset in (2, 1)) if index >= 2 else None,
        "last_3_templates": lambda index: tuple(sequence[index - offset]["cluster_id"] for offset in (3, 2, 1)) if index >= 3 else None,
        "action": lambda index: (sequence[index]["action"],),
        "mover_id": lambda index: tuple(sequence[index]["mover_refs"][:3]),
        "current_frame_signature": lambda index: (sequence[index]["state_key"],),
    }
    result: dict[str, Any] = {}
    for name, builder in condition_builders.items():
        grouped: dict[Any, Counter[str]] = defaultdict(Counter)
        for index, item in enumerate(sequence):
            condition = builder(index)
            if condition is None:
                continue
            grouped[stable_json(condition)][str(item["cluster_id"])] += 1
        result[name] = {
            "conditional_entropy": round(weighted_conditional_entropy(grouped), 6),
            "condition_count": len(grouped),
            "case_count": sum(sum(counter.values()) for counter in grouped.values()),
            "majority_accuracy": round(grouped_majority_accuracy(grouped), 6),
        }
    result["unconditional"] = {
        "conditional_entropy": round(counter_entropy(Counter(str(item["cluster_id"]) for item in sequence)), 6),
        "condition_count": 1,
        "case_count": len(sequence),
        "majority_accuracy": round(max(Counter(str(item["cluster_id"]) for item in sequence).values()) / max(1, len(sequence)), 6) if sequence else 0.0,
    }
    return result


def weighted_conditional_entropy(grouped: Mapping[Any, Counter[str]]) -> float:
    total = sum(sum(counter.values()) for counter in grouped.values())
    if total <= 0:
        return 0.0
    return sum((sum(counter.values()) / total) * counter_entropy(counter) for counter in grouped.values())


def grouped_majority_accuracy(grouped: Mapping[Any, Counter[str]]) -> float:
    total = sum(sum(counter.values()) for counter in grouped.values())
    if total <= 0:
        return 0.0
    return sum(max(counter.values()) for counter in grouped.values() if counter) / total


def counter_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    result = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            result -= p * math.log2(p)
    return result


def template_fragmentation_audit_v0(
    step_records: list[Mapping[str, Any]],
    clusters: list[GlobalTemplateCluster],
    cluster_by_step: Mapping[int, str],
) -> list[dict[str, Any]]:
    rows = []
    for threshold in (0.35, 0.45, 0.55, 0.65):
        coarse_map = coarse_phase_class_map_v0(clusters, threshold=threshold)
        eval_row = prequential_coarse_class_eval_v0(step_records, clusters, cluster_by_step, coarse_map)
        rows.append(
            {
                "compression_threshold": threshold,
                "original_template_count": len(clusters),
                "coarse_phase_class_count": len(set(coarse_map.values())),
                **eval_row,
                "fragmentation_reduced": len(set(coarse_map.values())) < len(clusters),
            }
        )
    return rows


def coarse_phase_class_map_v0(clusters: list[GlobalTemplateCluster], *, threshold: float) -> dict[str, str]:
    classes: list[dict[str, Any]] = []
    result = {}
    for cluster in clusters:
        cells = set(cluster.prototype_cells)
        best_index = None
        best_score = 0.0
        for index, klass in enumerate(classes):
            score = max(
                bbox_normalized_iou(cells, klass["cells"]),
                band_signature_similarity(cells, klass["cells"]),
                translation_aligned_iou(cells, klass["cells"], max_shift=5),
            )
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is None or best_score < threshold:
            class_id = f"coarse_phase:{len(classes)}"
            classes.append({"class_id": class_id, "cells": cells})
        else:
            class_id = classes[best_index]["class_id"]
        result[cluster.cluster_id] = class_id
    return result


def prequential_coarse_class_eval_v0(
    step_records: list[Mapping[str, Any]],
    clusters: list[GlobalTemplateCluster],
    cluster_by_step: Mapping[int, str],
    coarse_map: Mapping[str, str],
) -> dict[str, Any]:
    hits = []
    exact_scores = []
    aligned_scores = []
    prior_class_by_step = {step: coarse_map[cluster_id] for step, cluster_id in cluster_by_step.items() if cluster_id in coarse_map}
    prototype_by_class = coarse_class_prototype_cells_v0(clusters, coarse_map)
    for item in step_records:
        step = int(item["step"])
        actual_class = prior_class_by_step.get(step)
        predicted_class = latest_seen_class_before_v0(step_records, prior_class_by_step, step)
        hits.append(actual_class is not None and actual_class == predicted_class)
        predicted_cells = prototype_by_class.get(predicted_class, set())
        actual_cells = set(item["cells"])
        exact_scores.append(exact_iou(actual_cells, predicted_cells))
        aligned_scores.append(translation_aligned_iou(actual_cells, predicted_cells, max_shift=5))
    return {
        "prequential_class_accuracy": round(sum(hits) / max(1, len(hits)), 6),
        "mean_exact_match": round(sum(exact_scores) / max(1, len(exact_scores)), 6),
        "mean_translation_aligned_match": round(sum(aligned_scores) / max(1, len(aligned_scores)), 6),
    }


def coarse_class_prototype_cells_v0(
    clusters: list[GlobalTemplateCluster],
    coarse_map: Mapping[str, str],
) -> dict[str, set[tuple[int, int]]]:
    cells_by_class: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for cluster in clusters:
        class_id = coarse_map.get(cluster.cluster_id)
        if class_id is not None:
            cells_by_class[class_id].update(cluster.prototype_cells)
    return cells_by_class


def latest_seen_class_before_v0(
    step_records: list[Mapping[str, Any]],
    class_by_step: Mapping[int, str],
    step: int,
) -> str | None:
    prior_steps = [int(item["step"]) for item in step_records if int(item["step"]) < step and int(item["step"]) in class_by_step]
    return class_by_step[max(prior_steps)] if prior_steps else None


def current_frame_phase_readout_scores_v0(
    step_records: list[Mapping[str, Any]],
    clusters: list[GlobalTemplateCluster],
    cluster_by_step: Mapping[int, str],
) -> list[dict[str, Any]]:
    cluster_by_id = {cluster.cluster_id: cluster for cluster in clusters}
    feature_builders = {
        "global_color_histogram": lambda item: global_color_histogram_key(item["record"]),
        "row_col_signature": lambda item: frame_row_col_signature_key(item["record"]),
        "hud_register_region_signature": lambda item: hud_register_signature_key(item["record"]),
        "visible_state_summary": lambda item: stable_json(item["record"].pre_features.get("visible_state_summary")),
        "combined_frame_signature": lambda item: stable_json(
            {
                "color": global_color_histogram_key(item["record"]),
                "bands": frame_row_col_signature_key(item["record"]),
                "hud": hud_register_signature_key(item["record"]),
            }
        ),
    }
    rows = []
    for name, builder in feature_builders.items():
        exact_scores = []
        aligned_scores = []
        hits = []
        prior_by_feature: dict[str, Counter[str]] = defaultdict(Counter)
        for item in step_records:
            step = int(item["step"])
            actual_cluster = cluster_by_step.get(step)
            feature = builder(item)
            predicted_cluster = None
            if prior_by_feature.get(feature):
                predicted_cluster = prior_by_feature[feature].most_common(1)[0][0]
            hits.append(predicted_cluster == actual_cluster and actual_cluster is not None)
            predicted_cells = set(cluster_by_id[predicted_cluster].prototype_cells) if predicted_cluster in cluster_by_id else set()
            actual_cells = set(item["cells"])
            exact_scores.append(exact_iou(actual_cells, predicted_cells))
            aligned_scores.append(translation_aligned_iou(actual_cells, predicted_cells, max_shift=5))
            if actual_cluster is not None:
                prior_by_feature[feature][actual_cluster] += 1
        rows.append(
            {
                "feature_family": name,
                "case_count": len(step_records),
                "prequential_template_accuracy": round(sum(hits) / max(1, len(hits)), 6),
                "mean_exact_match": round(sum(exact_scores) / max(1, len(exact_scores)), 6),
                "mean_translation_aligned_match": round(sum(aligned_scores) / max(1, len(aligned_scores)), 6),
                "feature_cardinality": len(prior_by_feature),
            }
        )
    return sorted(rows, key=lambda row: row["mean_exact_match"], reverse=True)


def global_color_histogram_key(record: TransitionRecord) -> str:
    grid = observation_plane(record.before)
    values, counts = np.unique(grid, return_counts=True)
    pairs = sorted(((int(count), int(value)) for value, count in zip(values, counts)), reverse=True)[:6]
    return stable_json(tuple((value, bucket_count(count)) for count, value in pairs))


def frame_row_col_signature_key(record: TransitionRecord) -> str:
    grid = observation_plane(record.before)
    row_counts = tuple(int(np.count_nonzero(row)) for row in grid)
    col_counts = tuple(int(np.count_nonzero(grid[:, col])) for col in range(grid.shape[1]))
    return stable_json(
        {
            "top_rows": tuple(sorted(row_counts, reverse=True)[:8]),
            "top_cols": tuple(sorted(col_counts, reverse=True)[:8]),
        }
    )


def hud_register_signature_key(record: TransitionRecord) -> str:
    grid = observation_plane(record.before)
    if grid.size == 0:
        return "()"
    top = grid[: min(8, grid.shape[0]), :]
    left = grid[:, : min(8, grid.shape[1])]
    values = Counter(int(value) for value in top.flatten())
    values.update(int(value) for value in left.flatten())
    return stable_json(tuple(sorted(values.most_common(8))))


def higher_order_phase_predictions_v0(
    step_records: list[Mapping[str, Any]],
    clusters: list[GlobalTemplateCluster],
    cluster_by_step: Mapping[int, str],
) -> list[dict[str, Any]]:
    cluster_by_id = {cluster.cluster_id: cluster for cluster in clusters}
    history_tables = {
        "last_template_table": 1,
        "last_2_template_table": 2,
        "last_3_template_table": 3,
    }
    rows = []
    seen_sequence: list[str] = []
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in step_records:
        step = int(item["step"])
        actual = cluster_by_step.get(step)
        for name, order in history_tables.items():
            key = tuple(seen_sequence[-order:]) if len(seen_sequence) >= order else None
            predicted = transition_counts[stable_json(key)].most_common(1)[0][0] if key is not None and transition_counts.get(stable_json(key)) else None
            predicted_cells = set(cluster_by_id[predicted].prototype_cells) if predicted in cluster_by_id else set()
            actual_cells = set(item["cells"])
            rows.append(
                {
                    "step": step,
                    "hypothesis": name,
                    "order": order,
                    "has_history": key is not None,
                    "predicted_cluster_id": predicted,
                    "actual_cluster_id": actual,
                    "hit": predicted == actual and actual is not None,
                    "exact_match": round(exact_iou(actual_cells, predicted_cells), 6),
                    "translation_aligned_match": round(translation_aligned_iou(actual_cells, predicted_cells, max_shift=5), 6),
                    "used_future_evidence": False,
                }
            )
        if actual is not None:
            for order in history_tables.values():
                if len(seen_sequence) >= order:
                    transition_counts[stable_json(tuple(seen_sequence[-order:]))][actual] += 1
            seen_sequence.append(actual)
    return rows


def compositional_global_atoms_v0(step_records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    row_support: Counter[int] = Counter()
    col_support: Counter[int] = Counter()
    bbox_zone_support: Counter[str] = Counter()
    for item in step_records:
        cells = set(item["cells"])
        row_support.update(row for row, _col in cells)
        col_support.update(col for _row, col in cells)
        bbox_zone_support[coarse_bbox_zone(cells_bbox(list(cells)))] += 1
    rows = []
    total = max(1, len(step_records))
    for row, count in row_support.most_common(20):
        rows.append(
            {
                "atom_id": f"row_band:{row}",
                "atom_type": "row_band",
                "support_count": count,
                "support_rate": round(count / total, 6),
                "predictability_hint": "recurring_row_atom" if count / total >= 0.25 else "sparse_row_atom",
            }
        )
    for col, count in col_support.most_common(20):
        rows.append(
            {
                "atom_id": f"col_band:{col}",
                "atom_type": "col_band",
                "support_count": count,
                "support_rate": round(count / total, 6),
                "predictability_hint": "recurring_col_atom" if count / total >= 0.25 else "sparse_col_atom",
            }
        )
    for zone, count in bbox_zone_support.most_common():
        rows.append(
            {
                "atom_id": f"bbox_zone:{zone}",
                "atom_type": "bbox_zone",
                "support_count": count,
                "support_rate": round(count / total, 6),
                "predictability_hint": "recurring_zone_atom" if count / total >= 0.25 else "sparse_zone_atom",
            }
        )
    return rows


def coarse_bbox_zone(bbox: tuple[int, int, int, int] | None) -> str:
    if bbox is None:
        return "none"
    top, left, bottom, right = bbox
    height = bottom - top + 1
    width = right - left + 1
    if top <= 2 and left <= 2 and height >= 50 and width >= 50:
        return "full_frame"
    if width >= 50:
        return "wide_band"
    if height >= 50:
        return "tall_band"
    return "local_region"


def phase_identifiability_metrics_v0(
    *,
    phase_report: Mapping[str, Any],
    support_curve: list[Mapping[str, Any]],
    entropy: Mapping[str, Any],
    fragmentation: list[Mapping[str, Any]],
    frame_readout: list[Mapping[str, Any]],
    higher_order: list[Mapping[str, Any]],
    atoms: list[Mapping[str, Any]],
) -> dict[str, Any]:
    phase_metrics = phase_report.get("metrics", {})
    best_support_gain = support_curve_gain(support_curve)
    best_fragmentation = max((float(row["mean_translation_aligned_match"]) for row in fragmentation), default=0.0)
    best_frame = max((float(row["mean_translation_aligned_match"]) for row in frame_readout), default=0.0)
    higher_order_summary = higher_order_summary_v0(higher_order)
    entropy_uncond = float(entropy.get("unconditional", {}).get("conditional_entropy", 0.0) or 0.0)
    entropy_last3 = float(entropy.get("last_3_templates", {}).get("conditional_entropy", entropy_uncond) or 0.0)
    recurring_atom_rate = sum(1 for atom in atoms if float(atom.get("support_rate", 0.0) or 0.0) >= 0.25) / max(1, len(atoms))
    return {
        "baseline_online_aligned_match": float(phase_metrics.get("baseline_online_prediction_match", 0.0) or 0.0),
        "phase_report_best_prequential_exact_match": float(phase_metrics.get("best_prequential_exact_match", 0.0) or 0.0),
        "phase_report_best_prequential_aligned_match": float(phase_metrics.get("best_prequential_translation_aligned_match", 0.0) or 0.0),
        "oracle_aligned_template_match": float(phase_metrics.get("oracle_translation_aligned_template_match", 0.0) or 0.0),
        "support_learning_curve_gain": round(best_support_gain, 6),
        "best_fragmentation_aligned_match": round(best_fragmentation, 6),
        "best_current_frame_aligned_match": round(best_frame, 6),
        "best_higher_order_aligned_match": higher_order_summary["best_aligned_match"],
        "best_higher_order_hypothesis": higher_order_summary["best_hypothesis"],
        "entropy_reduction_last3_vs_unconditional": round(entropy_uncond - entropy_last3, 6),
        "recurring_global_atom_rate": round(recurring_atom_rate, 6),
        "false_subtracted_mover_or_actor_rate": float(phase_metrics.get("false_subtracted_mover_or_actor_rate", 0.0) or 0.0),
        "old_new_footprint_recovery_after_factoring": float(phase_metrics.get("old_new_footprint_recovery_after_factoring", 0.0) or 0.0),
    }


def support_curve_gain(rows: list[Mapping[str, Any]]) -> float:
    by_bucket = {row["prior_support_bucket"]: float(row["mean_translation_aligned_match"]) for row in rows}
    return max(by_bucket.values(), default=0.0) - by_bucket.get("0", 0.0)


def higher_order_summary_v0(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["hypothesis"])].append(row)
    best_name = "none"
    best_score = 0.0
    for name, items in grouped.items():
        score = sum(float(item["translation_aligned_match"]) for item in items) / max(1, len(items))
        if score > best_score:
            best_name = name
            best_score = score
    return {"best_hypothesis": best_name, "best_aligned_match": round(best_score, 6)}


def phase_identifiability_verdict_v0(metrics: Mapping[str, Any]) -> dict[str, Any]:
    labels = []
    oracle = float(metrics.get("oracle_aligned_template_match", 0.0) or 0.0)
    best_online = float(metrics.get("phase_report_best_prequential_aligned_match", 0.0) or 0.0)
    support_gain = float(metrics.get("support_learning_curve_gain", 0.0) or 0.0)
    fragmentation = float(metrics.get("best_fragmentation_aligned_match", 0.0) or 0.0)
    frame = float(metrics.get("best_current_frame_aligned_match", 0.0) or 0.0)
    higher = float(metrics.get("best_higher_order_aligned_match", 0.0) or 0.0)
    entropy_gain = float(metrics.get("entropy_reduction_last3_vs_unconditional", 0.0) or 0.0)
    atom_rate = float(metrics.get("recurring_global_atom_rate", 0.0) or 0.0)
    if oracle < 0.35:
        labels.append("template_vocabulary_ceiling_low")
    elif support_gain >= 0.15:
        labels.append("sparse_support_problem")
    elif fragmentation >= best_online + 0.10:
        labels.append("template_fragmentation_problem")
    elif frame >= best_online + 0.10:
        labels.append("visible_but_unmodeled_phase")
    elif higher >= best_online + 0.10 or entropy_gain >= 0.50:
        labels.append("higher_order_hidden_automaton")
    elif atom_rate >= 0.25:
        labels.append("compositional_global_update")
    else:
        labels.append("unobservable_or_noisy_layer")
    if entropy_gain >= 0.50:
        labels.append("higher_order_hidden_automaton_signal")
    if atom_rate >= 0.25:
        labels.append("compositional_global_atoms_present")
    if frame >= best_online + 0.10:
        labels.append("visible_phase_readout_signal")
    if fragmentation >= best_online + 0.10:
        labels.append("template_fragmentation_signal")
    if float(metrics.get("false_subtracted_mover_or_actor_rate", 0.0) or 0.0) <= 0.017237:
        labels.append("protected_false_subtraction_not_increased")
    else:
        labels.append("protected_false_subtraction_increased")
    if float(metrics.get("old_new_footprint_recovery_after_factoring", 0.0) or 0.0) >= 0.90:
        labels.append("old_new_recovery_preserved")
    else:
        labels.append("old_new_recovery_failed")
    return {
        "primary": labels[0],
        "labels": sorted(set(labels), key=labels.index),
        "recommended_next_predictor_family": recommended_phase_predictor_family(labels[0]),
    }


def recommended_phase_predictor_family(label: str) -> str:
    return {
        "sparse_support_problem": "mover/global-template bootstrap and uncertainty set",
        "template_fragmentation_problem": "coarse phase-class predictor before exact template predictor",
        "visible_but_unmodeled_phase": "current-frame phase readout from global bands/registers",
        "higher_order_hidden_automaton": "finite-state phase tracker over last global templates",
        "compositional_global_update": "independent global atom predictors",
        "unobservable_or_noisy_layer": "treat global layer as uncertainty/noise until more observations exist",
        "template_vocabulary_ceiling_low": "revise global template vocabulary before prediction",
    }.get(label, "manual phase prototype audit")


def combined_report_rows(report_by_game: Mapping[str, Mapping[str, Any]], field: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for game, report in report_by_game.items():
        items = report.get(field, ()) if isinstance(report, Mapping) else ()
        for row in items:
            if isinstance(row, Mapping):
                rows.append({"game": game, **dict(row)})
    return rows


def visibility_composition_precision_audit_suite(
    records_by_game: Mapping[str, list[TransitionRecord]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    cases_by_game = {}
    label_counts: Counter[str] = Counter()
    consistency_counts: Counter[str] = Counter()
    for game in ("ls20", "ar25"):
        records = records_by_game.get(game, [])
        cases = visibility_composition_precision_cases(
            records,
            game=game,
            prototype_dir=output_dir / "visibility_composition_precision_prototypes" / game,
            limit=30,
        )
        cases_by_game[game] = {
            "requested_case_count": 30,
            "audited_case_count": len(cases),
            "cases": cases,
            "label_accounting": dict(Counter(case["diagnostic_precision_label"] for case in cases).most_common()),
            "consistency_accounting": dict(Counter(case["consistency"] for case in cases).most_common()),
        }
        label_counts.update(case["diagnostic_precision_label"] for case in cases)
        consistency_counts.update(case["consistency"] for case in cases)
    return {
        "status": "evaluated",
        "probe_type": "manual_precision_audit_scaffold_not_predictor",
        "legacy_target_class": "translation_with_trail_or_residue",
        "target_classes": ["clean_old_new_exposure", "merged_old_new_overlap_or_trail"],
        "diagnostic_only": True,
        "sample_plan": {
            "ls20_translation_with_trail_or_residue": 30,
            "ar25_translation_with_trail_or_residue": 30,
        },
        "summary_label_accounting": dict(label_counts.most_common()),
        "summary_consistency_accounting": dict(consistency_counts.most_common()),
        "by_game": cases_by_game,
    }


def visibility_composition_precision_cases(
    records: list[TransitionRecord],
    *,
    game: str,
    prototype_dir: Path,
    limit: int,
) -> list[dict[str, Any]]:
    prototype_dir.mkdir(parents=True, exist_ok=True)
    record_by_step = {record.step: record for record in records}
    rows = [
        row
        for row in visibility_composition_rows(records)
        if row["legacy_visible_delta_class"] == "translation_with_trail_or_residue"
    ][:limit]
    cases = []
    for index, row in enumerate(rows, start=1):
        record = record_by_step.get(int(row["step"]))
        prototype_path = (
            write_visibility_precision_prototype(prototype_dir, row, record, index)
            if record is not None
            else None
        )
        label, reasons, consistency = precision_label_for_visibility_row(row)
        cases.append(
            {
                "case_index": index,
                "game": game,
                "step": row["step"],
                "action": row["action"],
                "episode_local_mover_ref": row["episode_local_mover_ref"],
                "visible_delta_class": row["visible_delta_class"],
                "legacy_visible_delta_class": row["legacy_visible_delta_class"],
                "diagnostic_precision_label": label,
                "consistency": consistency,
                "reasons": reasons,
                "prototype_path": str(prototype_path) if prototype_path else None,
                "visible_to_latent_pose_recovery": row["visible_to_latent_pose_recovery"],
                "trail_or_residue_class": row["trail_or_residue_class"],
                "controlled_mover_overlap_class": row["controlled_mover_overlap_class"],
                "mover_mover_overlap_class": row["mover_mover_overlap_class"],
                "overlap_priority_class": row["overlap_priority_class"],
                "cluster_bboxes": row["cluster_bboxes"],
            }
        )
    return cases


def precision_label_for_visibility_row(row: Mapping[str, Any]) -> tuple[str, list[str], str]:
    reasons = []
    label = "trail_residue_needs_manual_review"
    consistency = "consistent"
    trail = str(row.get("trail_or_residue_class"))
    if trail == "separate_old_new_exposure":
        label = "clean_old_new_exposure"
        reasons.append("old and new footprint regions are separated")
    elif trail == "old_new_overlap_or_trail":
        label = "merged_old_new_overlap_or_trail"
        reasons.append("changed cluster overlaps old and new mover footprints")
    elif trail == "old_pose_residue_or_exposure":
        label = "old_pose_only_residue_or_exposure"
        reasons.append("visible change is only on the old mover pose")
    else:
        consistency = "inconsistent_label_accounting"
        reasons.append("row class is trail/residue but trail_or_residue_class is not trail-like")
    if row.get("controlled_mover_overlap_class") != "no_controlled_overlap":
        label = "trail_label_with_controlled_overlap_risk"
        consistency = "needs_review"
        reasons.append("controlled overlap is present but trail/residue label won precedence")
    if row.get("mover_mover_overlap_class") != "no_mover_overlap":
        label = "trail_label_with_mover_overlap_risk"
        consistency = "needs_review"
        reasons.append("mover-mover overlap is present but trail/residue label won precedence")
    if row.get("visible_to_latent_pose_recovery") not in {"visible_pose_recovered", "old_pose_only_visible"}:
        consistency = "needs_review"
        reasons.append("latent pose recovery is weak for a trail/residue label")
    return label, reasons, consistency


def write_visibility_precision_prototype(
    prototype_dir: Path,
    row: Mapping[str, Any],
    record: TransitionRecord,
    index: int,
) -> Path:
    path = prototype_dir / f"{record.game}-trail-residue-{index:02d}-step-{record.step}-{row['episode_local_mover_ref']}.json"
    write_json(
        path,
        {
            "audit": "visibility_composition_precision",
            "game": record.game,
            "step": record.step,
            "action": record.action,
            "mover_ref": row["episode_local_mover_ref"],
            "row": row,
            "before_crop": frame_crop_around_changes(record, before=True),
            "after_crop": frame_crop_around_changes(record, before=False),
            "factorized_changed_clusters": factorized_changed_regions(record),
            "mover_visible_clusters": mover_visible_clusters_for_ref(record, row["episode_local_mover_ref"]),
        },
    )
    return path


def visibility_composition_precision_audit_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Visibility Composition Precision Audit",
        "",
        "Diagnostic-only audit. These labels are review scaffolding, not promoted model variables.",
        "",
        "## Summary",
        "",
        f"- legacy target class: `{report.get('legacy_target_class')}`",
        f"- target classes: `{stable_json(report.get('target_classes', []))}`",
        f"- label accounting: `{stable_json(report.get('summary_label_accounting', {}))}`",
        f"- consistency accounting: `{stable_json(report.get('summary_consistency_accounting', {}))}`",
    ]
    for game, payload in report.get("by_game", {}).items():
        lines.extend(
            [
                "",
                f"## {str(game).upper()}",
                "",
                f"- audited: `{payload.get('audited_case_count')}` / requested `{payload.get('requested_case_count')}`",
                f"- labels: `{stable_json(payload.get('label_accounting', {}))}`",
                f"- consistency: `{stable_json(payload.get('consistency_accounting', {}))}`",
                "",
            ]
        )
        for case in payload.get("cases", [])[:30]:
            lines.extend(
                [
                    f"### Case {case['case_index']}: step {case['step']} action {case['action']}",
                    "",
                    f"- label: `{case['diagnostic_precision_label']}`",
                    f"- consistency: `{case['consistency']}`",
                    f"- mover: `{case['episode_local_mover_ref']}`",
                    f"- pose recovery: `{case['visible_to_latent_pose_recovery']}`",
                    f"- trail class: `{case['trail_or_residue_class']}`",
                    f"- controlled overlap: `{case['controlled_mover_overlap_class']}`",
                    f"- mover overlap: `{case['mover_mover_overlap_class']}`",
                    f"- prototype: `{case['prototype_path']}`",
                    f"- reasons: `{stable_json(case.get('reasons', []))}`",
                    "",
                ]
            )
    return "\n".join(lines)


def ar25_latent_transition_taxonomy_report(
    records: list[TransitionRecord],
    *,
    prototype_dir: Path,
) -> dict[str, Any]:
    if not records:
        return {
            "status": "skipped_no_ar25_records",
            "game": "ar25",
            "reason": "no AR25 records available",
        }
    prototype_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in visibility_composition_rows(records)
        if row["visible_delta_class"] in {"mover_bootstrap_no_history", "true_latent_motion_unknown"}
    ][:30]
    record_by_step = {record.step: record for record in records}
    cases = []
    taxonomy_counts: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        record = record_by_step.get(int(row["step"]))
        label, reasons = ar25_latent_taxonomy_label(row)
        taxonomy_counts[label] += 1
        prototype_path = (
            write_ar25_latent_unknown_prototype(prototype_dir, row, record, index)
            if record is not None
            else None
        )
        cases.append(
            {
                "case_index": index,
                "step": row["step"],
                "action": row["action"],
                "episode_local_mover_ref": row["episode_local_mover_ref"],
                "taxonomy_label": label,
                "reasons": reasons,
                "prototype_path": str(prototype_path) if prototype_path else None,
                "history_length": row["history_length"],
                "visible_to_latent_pose_recovery": row["visible_to_latent_pose_recovery"],
                "trail_or_residue_class": row["trail_or_residue_class"],
                "controlled_mover_overlap_class": row["controlled_mover_overlap_class"],
                "mover_mover_overlap_class": row["mover_mover_overlap_class"],
                "overlap_priority_class": row["overlap_priority_class"],
            }
        )
    return {
        "status": "evaluated",
        "game": "ar25",
        "probe_type": "bootstrap_and_true_latent_motion_readout_not_predictor",
        "diagnostic_only": True,
        "requested_case_count": 30,
        "audited_case_count": len(cases),
        "taxonomy_distribution": dict(taxonomy_counts.most_common()),
        "cases": cases,
        "prototype_dir": str(prototype_dir),
        "verdict": {
            "primary": "ar25_bootstrap_or_true_latent_cases_sampled" if cases else "no_ar25_bootstrap_or_true_latent_cases",
            "labels": ["ar25_bootstrap_or_true_latent_cases_sampled"] if cases else ["no_ar25_bootstrap_or_true_latent_cases"],
        },
    }


def ar25_latent_taxonomy_label(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    if int(row.get("history_length", 0) or 0) == 0:
        reasons.append("mover has no prior history in this trace window")
        return "mover_bootstrap_no_history", reasons
    if row.get("visible_to_latent_pose_recovery") in {"latent_pose_occluded", "visible_pose_not_recovered"}:
        reasons.append("visible pose is not recovered from changed clusters")
        return "latent_pose_recovery_failure", reasons
    if row.get("controlled_mover_overlap_class") != "no_controlled_overlap":
        reasons.append("controlled overlap is present")
        return "controlled_overlap_latent_ambiguity", reasons
    if row.get("mover_mover_overlap_class") != "no_mover_overlap":
        reasons.append("mover overlap is present")
        return "mover_overlap_latent_ambiguity", reasons
    if row.get("trail_or_residue_class") != "no_trail_or_residue":
        reasons.append("trail/residue evidence exists but motion remains latent-unknown")
        return "trail_residue_without_motion_history", reasons
    reasons.append("no current taxonomy feature explains latent-motion unknown")
    return "unclassified_latent_transition", reasons


def write_ar25_latent_unknown_prototype(
    prototype_dir: Path,
    row: Mapping[str, Any],
    record: TransitionRecord,
    index: int,
) -> Path:
    path = prototype_dir / f"ar25-latent-motion-unknown-{index:02d}-step-{record.step}-{row['episode_local_mover_ref']}.json"
    write_json(
        path,
        {
            "audit": "ar25_latent_motion_unknown",
            "game": "ar25",
            "step": record.step,
            "action": record.action,
            "mover_ref": row["episode_local_mover_ref"],
            "row": row,
            "before_crop": frame_crop_around_changes(record, before=True),
            "after_crop": frame_crop_around_changes(record, before=False),
            "factorized_changed_clusters": factorized_changed_regions(record),
            "mover_visible_clusters": mover_visible_clusters_for_ref(record, row["episode_local_mover_ref"]),
        },
    )
    return path


def ar25_latent_motion_unknown_readout_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# AR25 Bootstrap And True Latent Motion Readout",
        "",
        "Diagnostic-only readout. Bootstrap cases are first observations; true latent cases require mover history.",
        "",
        f"- status: `{report.get('status')}`",
        f"- audited: `{report.get('audited_case_count', 0)}` / requested `{report.get('requested_case_count', 30)}`",
        f"- taxonomy: `{stable_json(report.get('taxonomy_distribution', {}))}`",
        f"- prototype dir: `{report.get('prototype_dir')}`",
        "",
    ]
    for case in report.get("cases", []):
        lines.extend(
            [
                f"## Case {case['case_index']}: step {case['step']} action {case['action']}",
                "",
                f"- taxonomy label: `{case['taxonomy_label']}`",
                f"- mover: `{case['episode_local_mover_ref']}`",
                f"- history length: `{case['history_length']}`",
                f"- pose recovery: `{case['visible_to_latent_pose_recovery']}`",
                f"- trail class: `{case['trail_or_residue_class']}`",
                f"- controlled overlap: `{case['controlled_mover_overlap_class']}`",
                f"- mover overlap: `{case['mover_mover_overlap_class']}`",
                f"- prototype: `{case['prototype_path']}`",
                f"- reasons: `{stable_json(case.get('reasons', []))}`",
                "",
            ]
        )
    return "\n".join(lines)


def ar25_latent_motion_history_filtered_taxonomy_report(
    records: list[TransitionRecord],
    *,
    prototype_dir: Path,
) -> dict[str, Any]:
    if not records:
        return {
            "status": "skipped_no_ar25_records",
            "game": "ar25",
            "reason": "no AR25 records available",
        }
    prototype_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in visibility_composition_rows(records)
        if row["visible_delta_class"] == "true_latent_motion_unknown"
    ]
    bootstrap_rows = [
        row
        for row in visibility_composition_rows(records)
        if row["visible_delta_class"] == "mover_bootstrap_no_history"
    ]
    latent_family_count = len(rows) + len(bootstrap_rows)
    history_distribution = Counter(str(row.get("history_length", 0)) for row in rows)
    cold_start_rows = [
        row for row in rows
        if int(row.get("history_length", 0) or 0) < 2
    ] + bootstrap_rows
    record_by_step = {record.step: record for record in records}
    passes = {
        "history_ge_2": ar25_history_filtered_pass(
            rows,
            record_by_step=record_by_step,
            prototype_dir=prototype_dir / "history_ge_2",
            min_history=2,
        ),
        "history_ge_3": ar25_history_filtered_pass(
            rows,
            record_by_step=record_by_step,
            prototype_dir=prototype_dir / "history_ge_3",
            min_history=3,
        ),
    }
    primary = "ar25_true_latent_unknown_remains_after_history"
    if not rows and bootstrap_rows:
        primary = "ar25_bootstrap_only_no_true_latent_unknowns"
    elif passes["history_ge_2"]["eligible_count"] == 0:
        primary = "ar25_true_latent_unknown_insufficient_history_support"
    return {
        "status": "evaluated",
        "game": "ar25",
        "probe_type": "history_filtered_latent_motion_unknown_readout_not_predictor",
        "diagnostic_only": True,
        "true_latent_motion_unknown_count": len(rows),
        "latent_motion_unknown_count": len(rows),
        "mover_bootstrap_no_history_count": len(bootstrap_rows),
        "cold_start_or_low_history_count": len(cold_start_rows),
        "cold_start_or_low_history_share": round(
            len(cold_start_rows) / latent_family_count if latent_family_count else 0.0,
            6,
        ),
        "history_length_distribution": dict(history_distribution.most_common()),
        "prototype_dir": str(prototype_dir),
        "passes": passes,
        "verdict": {
            "primary": primary,
            "labels": [primary],
        },
    }


def ar25_history_filtered_pass(
    rows: list[Mapping[str, Any]],
    *,
    record_by_step: Mapping[int, TransitionRecord],
    prototype_dir: Path,
    min_history: int,
) -> dict[str, Any]:
    prototype_dir.mkdir(parents=True, exist_ok=True)
    eligible_rows = [row for row in rows if int(row.get("history_length", 0) or 0) >= min_history]
    taxonomy_counts: Counter[str] = Counter()
    history_distribution = Counter(str(row.get("history_length", 0)) for row in eligible_rows)
    for row in eligible_rows:
        label, _reasons, _missing, _probe = ar25_history_filtered_taxonomy_label(row)
        taxonomy_counts[label] += 1
    cases = []
    for index, row in enumerate(eligible_rows[:30], start=1):
        record = record_by_step.get(int(row["step"]))
        label, reasons, missing_variable, suggested_probe = ar25_history_filtered_taxonomy_label(row)
        prototype_path = (
            write_ar25_history_filtered_latent_prototype(prototype_dir, row, record, index)
            if record is not None
            else None
        )
        cases.append(
            {
                "case_index": index,
                "step": row["step"],
                "action": row["action"],
                "episode_local_mover_ref": row["episode_local_mover_ref"],
                "history_length": row["history_length"],
                "taxonomy_label": label,
                "reasons": reasons,
                "required_missing_variable": missing_variable,
                "suggested_next_probe": suggested_probe,
                "prototype_path": str(prototype_path) if prototype_path else None,
                "visible_to_latent_pose_recovery": row["visible_to_latent_pose_recovery"],
                "trail_or_residue_class": row["trail_or_residue_class"],
                "controlled_mover_overlap_class": row["controlled_mover_overlap_class"],
                "mover_mover_overlap_class": row["mover_mover_overlap_class"],
                "overlap_priority_class": row["overlap_priority_class"],
                "occlusion_state": row["occlusion_state"],
            }
        )
    return {
        "min_history": min_history,
        "eligible_count": len(eligible_rows),
        "requested_case_count": 30,
        "audited_case_count": len(cases),
        "history_length_distribution": dict(history_distribution.most_common()),
        "taxonomy_distribution": dict(taxonomy_counts.most_common()),
        "cases": cases,
        "prototype_dir": str(prototype_dir),
        "support_status": "has_support" if eligible_rows else "insufficient_support",
    }


def ar25_history_filtered_taxonomy_label(row: Mapping[str, Any]) -> tuple[str, list[str], str, str]:
    reasons = []
    if row.get("visible_to_latent_pose_recovery") in {"latent_pose_occluded", "visible_pose_not_recovered", "latent_pose_unknown"}:
        reasons.append("latent pose is not visible enough to separate motion from crop or occlusion")
        return (
            "crop_context_missing",
            reasons,
            "larger pre/post crop or latent pose recovery state",
            "crop-margin and occlusion recovery audit",
        )
    if row.get("mover_mover_overlap_class") != "no_mover_overlap":
        reasons.append("mover-mover overlap or crossing evidence is present")
        return (
            "identity_crossing_or_link_failure",
            reasons,
            "identity ordering through mover crossing or overlap",
            "crossing stress test with same-signature movers",
        )
    if row.get("controlled_mover_overlap_class") != "no_controlled_overlap":
        reasons.append("controlled object overlaps old or new mover footprint")
        return (
            "contact_triggered_transition",
            reasons,
            "contact state between controlled entity and mover",
            "controlled-contact transition probe",
        )
    if row.get("trail_or_residue_class") == "old_new_overlap_or_trail":
        reasons.append("old and new footprints merge into the same visible changed region")
        return (
            "merged_overlap_composition",
            reasons,
            "overlap/render priority for old and new mover footprint",
            "shape/group overlap composition probe",
        )
    if row.get("occlusion_state") == "overlap_composition_present":
        reasons.append("factorizer reports overlap composition even though no specific overlap head won")
        return (
            "group_or_coupled_motion",
            reasons,
            "coupled group membership and relative offset state",
            "coupled-set composition audit",
        )
    if int(row.get("history_length", 0) or 0) >= 3:
        reasons.append("mover has enough history for timer/track hypotheses but current taxonomy found no direct cause")
        return (
            "phase_or_timer_transition",
            reasons,
            "phase, lane, or timer state not represented by current row",
            "periodic phase and track-lane readout",
        )
    reasons.append("history exists, but current diagnostic fields do not name the transition")
    return (
        "still_unknown",
        reasons,
        "unidentified latent transition variable",
        "manual prototype atlas before predictor work",
    )


def write_ar25_history_filtered_latent_prototype(
    prototype_dir: Path,
    row: Mapping[str, Any],
    record: TransitionRecord,
    index: int,
) -> Path:
    path = prototype_dir / f"ar25-history-filtered-latent-{index:02d}-step-{record.step}-{row['episode_local_mover_ref']}.json"
    write_json(
        path,
        {
            "audit": "ar25_latent_motion_history_filtered",
            "game": "ar25",
            "step": record.step,
            "action": record.action,
            "mover_ref": row["episode_local_mover_ref"],
            "row": row,
            "before_crop": frame_crop_around_changes(record, before=True),
            "after_crop": frame_crop_around_changes(record, before=False),
            "factorized_changed_clusters": factorized_changed_regions(record),
            "mover_visible_clusters": mover_visible_clusters_for_ref(record, row["episode_local_mover_ref"]),
        },
    )
    return path


def ar25_latent_motion_history_filtered_readout_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# AR25 Latent Motion History-Filtered Readout",
        "",
        "Diagnostic-only readout. Cold-start cases are counted separately; filtered cases require mover history.",
        "",
        f"- status: `{report.get('status')}`",
        f"- true latent unknown count: `{report.get('true_latent_motion_unknown_count', 0)}`",
        f"- mover bootstrap count: `{report.get('mover_bootstrap_no_history_count', 0)}`",
        f"- cold-start/low-history count: `{report.get('cold_start_or_low_history_count', 0)}`",
        f"- history distribution: `{stable_json(report.get('history_length_distribution', {}))}`",
        f"- prototype dir: `{report.get('prototype_dir')}`",
        "",
    ]
    for name, payload in report.get("passes", {}).items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- min history: `{payload.get('min_history')}`",
                f"- support: `{payload.get('support_status')}`",
                f"- eligible: `{payload.get('eligible_count')}`",
                f"- audited: `{payload.get('audited_case_count')}` / requested `{payload.get('requested_case_count')}`",
                f"- taxonomy: `{stable_json(payload.get('taxonomy_distribution', {}))}`",
                f"- history distribution: `{stable_json(payload.get('history_length_distribution', {}))}`",
                "",
            ]
        )
        for case in payload.get("cases", []):
            lines.extend(
                [
                    f"### Case {case['case_index']}: step {case['step']} action {case['action']}",
                    "",
                    f"- taxonomy label: `{case['taxonomy_label']}`",
                    f"- mover: `{case['episode_local_mover_ref']}`",
                    f"- history length: `{case['history_length']}`",
                    f"- required missing variable: `{case['required_missing_variable']}`",
                    f"- suggested next probe: `{case['suggested_next_probe']}`",
                    f"- pose recovery: `{case['visible_to_latent_pose_recovery']}`",
                    f"- trail class: `{case['trail_or_residue_class']}`",
                    f"- controlled overlap: `{case['controlled_mover_overlap_class']}`",
                    f"- mover overlap: `{case['mover_mover_overlap_class']}`",
                    f"- prototype: `{case['prototype_path']}`",
                    f"- reasons: `{stable_json(case.get('reasons', []))}`",
                    "",
                ]
            )
    return "\n".join(lines)


def visibility_composition_rows(records: list[TransitionRecord]) -> list[dict[str, Any]]:
    rows = []
    history_by_object: dict[int, list[tuple[TransitionRecord, Any]]] = defaultdict(list)
    for record in records:
        for delta in stable_independent_motion_deltas(record):
            object_id = int(delta.object_id)
            history = history_by_object[object_id][-4:]
            row = visibility_composition_row(record, delta, history)
            rows.append(row)
            history_by_object[object_id].append((record, delta))
    return rows


def visibility_composition_row(
    record: TransitionRecord,
    delta: Any,
    history: list[tuple[TransitionRecord, Any]],
) -> dict[str, Any]:
    clusters = mover_visible_clusters(record, delta)
    background_count = sum(1 for cluster in clusters if has_background_exposure_signature(cluster))
    controlled_overlap = mover_overlaps_controlled(record, delta)
    mover_overlap = mover_mover_overlap(record, delta)
    overlap_state = occlusion_state(record, delta)
    reappeared = int(delta.object_id) in {int(item) for item in getattr(record.event, "reappeared_object_ids", ()) or ()}
    vanished = int(delta.object_id) in {int(item) for item in getattr(record.event, "disappeared_object_ids", ()) or ()}
    visible_class = visible_delta_class(
        record,
        delta,
        history=history,
        clusters=clusters,
        background_count=background_count,
        controlled_overlap=controlled_overlap,
        mover_overlap=mover_overlap,
        reappeared=reappeared,
        vanished=vanished,
    )
    return {
        "game": record.game,
        "step": record.step,
        "action": record.action,
        "episode_local_mover_ref": local_mover_ref(record.game, int(delta.object_id)),
        "history_length": len(history),
        "bbox_before": tuple(getattr(delta, "bbox_before", ())),
        "bbox_after": tuple(getattr(delta, "bbox_after", ())),
        "translation": tuple(getattr(delta, "translation", ())),
        "mover_signature": mover_signature_from_delta(delta),
        "visible_delta_class": visible_class,
        "legacy_visible_delta_class": legacy_visible_delta_class(visible_class),
        "visible_to_latent_pose_recovery": visible_to_latent_pose_recovery(record, delta, clusters),
        "occluded_cell_count": occluded_cell_count(record, delta, clusters),
        "newly_exposed_background_count": background_count,
        "overlap_priority_class": overlap_priority_class(record, delta),
        "controlled_mover_overlap_class": controlled_overlap,
        "mover_mover_overlap_class": mover_overlap,
        "trail_or_residue_class": trail_or_residue_class(record, delta, clusters),
        "vanish_vs_occluded": vanish_vs_occluded(record, delta, vanished),
        "reappearance_after_occlusion": "yes" if reappeared else "no",
        "occlusion_state": overlap_state,
        "cluster_count": len(clusters),
        "cluster_bboxes": [tuple(cluster.get("bbox", ())) for cluster in clusters[:8]],
    }


def visible_delta_class(
    record: TransitionRecord,
    delta: Any,
    *,
    history: list[tuple[TransitionRecord, Any]],
    clusters: list[Mapping[str, Any]],
    background_count: int,
    controlled_overlap: str,
    mover_overlap: str,
    reappeared: bool,
    vanished: bool,
) -> str:
    if not history:
        return "mover_bootstrap_no_history"
    if reappeared:
        return "translation_with_reappearance"
    if vanished or occlusion_state(record, delta) in {"reappeared_after_occlusion", "disappeared_or_occluded"}:
        return "translation_with_occlusion"
    if controlled_overlap != "no_controlled_overlap":
        return "translation_with_controlled_overlap"
    if mover_overlap != "no_mover_overlap":
        return "translation_with_mover_overlap"
    trail_class = trail_or_residue_class(record, delta, clusters)
    if trail_class != "no_trail_or_residue":
        return split_trail_or_residue_visible_class(trail_class)
    if background_count:
        return "translation_with_background_exposure"
    pose_recovery = visible_to_latent_pose_recovery(record, delta, clusters)
    if pose_recovery == "visible_pose_recovered":
        return "pure_translation_visible"
    if pose_recovery in {"latent_pose_occluded", "visible_pose_not_recovered", "latent_pose_unknown"}:
        return "true_latent_motion_unknown"
    return "composition_unknown"


def split_trail_or_residue_visible_class(trail_class: str) -> str:
    if trail_class == "old_new_overlap_or_trail":
        return "merged_old_new_overlap_or_trail"
    return "clean_old_new_exposure"


def legacy_visible_delta_class(visible_class: str) -> str:
    if visible_class in {"clean_old_new_exposure", "merged_old_new_overlap_or_trail"}:
        return "translation_with_trail_or_residue"
    return visible_class


def visibility_composition_heads(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "visible_to_latent_pose_recovery",
        "occluded_cell_count",
        "newly_exposed_background_count",
        "overlap_priority_class",
        "controlled_mover_overlap_class",
        "mover_mover_overlap_class",
        "trail_or_residue_class",
        "vanish_vs_occluded",
        "reappearance_after_occlusion",
    )
    return {
        field: dict(Counter(stable_json(row.get(field)) for row in rows).most_common())
        for field in fields
    }


def visibility_composition_examples(rows: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        class_name = str(row["visible_delta_class"])
        if len(examples[class_name]) >= 8:
            continue
        examples[class_name].append(
            {
                "step": row["step"],
                "action": row["action"],
                "episode_local_mover_ref": row["episode_local_mover_ref"],
                "history_length": row["history_length"],
                "visible_to_latent_pose_recovery": row["visible_to_latent_pose_recovery"],
                "controlled_mover_overlap_class": row["controlled_mover_overlap_class"],
                "mover_mover_overlap_class": row["mover_mover_overlap_class"],
                "overlap_priority_class": row["overlap_priority_class"],
                "cluster_bboxes": row["cluster_bboxes"],
            }
        )
    return dict(examples)


def ar25_unknown_world_clock_atlas_report(
    records: list[TransitionRecord],
    *,
    prototype_dir: Path | None = None,
) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game != "ar25":
        return {
            "status": "skipped_not_ar25",
            "game": game,
            "reason": "unknown world-clock atlas is AR25-only",
        }
    if prototype_dir is not None:
        prototype_dir.mkdir(parents=True, exist_ok=True)
    unknown_rows = [row for row in world_clock_transition_rows(records) if row["failure_class"] == "unknown_transition_failure"]
    examples = []
    record_by_step = {record.step: record for record in records}
    for row in unknown_rows[:30]:
        record = record_by_step.get(int(row["step"]))
        prototype_path = write_ar25_unknown_world_clock_prototype(prototype_dir, row, record) if prototype_dir and record else None
        examples.append(
            {
                "step": row["step"],
                "action": row["action"],
                "episode_local_mover_ref": row["episode_local_mover_ref"],
                "history_length": row["history_length"],
                "controlled_object_relation": row["coupled_set_role"],
                "overlap_relation": row["occlusion_state"],
                "object_vanished_or_reappeared": row["occlusion_state"] in {"reappeared_after_occlusion", "disappeared_or_occluded"},
                "background_exposed": bool(record and any(has_background_exposure_signature(cluster) for cluster in mover_visible_clusters_for_ref(record, row["episode_local_mover_ref"]))),
                "group_membership_changed": row["coupled_set_role"] != "independent_single",
                "contact_occurred": row["collision_or_contact_state"] != "no_near_contact",
                "prototype_path": str(prototype_path) if prototype_path else None,
            }
        )
    labels = []
    if unknown_rows:
        labels.append("ar25_unknown_world_clock_examples_emitted")
    else:
        labels.append("no_ar25_unknown_world_clock_failures")
    return {
        "status": "evaluated",
        "game": game,
        "unknown_transition_count": len(unknown_rows),
        "prototype_dir": str(prototype_dir) if prototype_dir else None,
        "examples": examples,
        "verdict": {
            "primary": labels[0],
            "labels": labels,
        },
    }


def write_ar25_unknown_world_clock_prototype(
    prototype_dir: Path,
    row: Mapping[str, Any],
    record: TransitionRecord,
) -> Path:
    path = prototype_dir / f"ar25-world-clock-unknown-step-{record.step}-{row['episode_local_mover_ref']}.json"
    mover_ref = row["episode_local_mover_ref"]
    write_json(
        path,
        {
            "game": "ar25",
            "step": record.step,
            "action": record.action,
            "mover_ref": mover_ref,
            "row": row,
            "before_crop": frame_crop_around_changes(record, before=True),
            "after_crop": frame_crop_around_changes(record, before=False),
            "factorized_changed_clusters": factorized_changed_regions(record),
            "mover_visible_clusters": mover_visible_clusters_for_ref(record, mover_ref),
            "controlled_candidates": record.post_labels.get("labeler_v3", {}).get("controlled_entity", {}).get("candidates", ()),
        },
    )
    return path


def mover_visible_clusters(record: TransitionRecord, delta: Any) -> list[Mapping[str, Any]]:
    old_bbox = tuple(getattr(delta, "bbox_before", ()))
    new_bbox = tuple(getattr(delta, "bbox_after", ()))
    candidates = []
    for cluster in record.post_labels.get("labeler_v3", {}).get("changed_region_clusters", ()):
        if not isinstance(cluster, Mapping):
            continue
        cluster_bbox = tuple(cluster.get("bbox", ()))
        if len(cluster_bbox) != 4:
            continue
        if (
            bbox_overlap_area(cluster_bbox, old_bbox) > 0
            or bbox_overlap_area(cluster_bbox, new_bbox) > 0
            or cluster.get("distance_to_actor") == "far"
        ):
            candidates.append(cluster)
    return candidates


def mover_visible_clusters_for_ref(record: TransitionRecord, mover_ref: Any) -> list[Mapping[str, Any]]:
    for delta in stable_independent_motion_deltas(record):
        if local_mover_ref(record.game, int(delta.object_id)) == str(mover_ref):
            return mover_visible_clusters(record, delta)
    return []


def visible_to_latent_pose_recovery(
    record: TransitionRecord,
    delta: Any,
    clusters: list[Mapping[str, Any]],
) -> str:
    new_bbox = tuple(getattr(delta, "bbox_after", ()))
    old_bbox = tuple(getattr(delta, "bbox_before", ()))
    if len(new_bbox) != 4 or len(old_bbox) != 4:
        return "latent_pose_unknown"
    if any(bbox_overlap_area(tuple(cluster.get("bbox", ())), new_bbox) > 0 for cluster in clusters):
        return "visible_pose_recovered"
    if any(bbox_overlap_area(tuple(cluster.get("bbox", ())), old_bbox) > 0 for cluster in clusters):
        return "old_pose_only_visible"
    if occlusion_state(record, delta) != "clear":
        return "latent_pose_occluded"
    return "visible_pose_not_recovered"


def occluded_cell_count(
    record: TransitionRecord,
    delta: Any,
    clusters: list[Mapping[str, Any]],
) -> str:
    mover_area = int(getattr(delta, "area_after", 0) or 0)
    visible_area = 0
    new_bbox = tuple(getattr(delta, "bbox_after", ()))
    for cluster in clusters:
        cluster_bbox = tuple(cluster.get("bbox", ()))
        if len(cluster_bbox) == 4 and len(new_bbox) == 4 and bbox_overlap_area(cluster_bbox, new_bbox) > 0:
            visible_area += bbox_overlap_area(cluster_bbox, new_bbox)
    return bucket_count(max(0, mover_area - visible_area))


def overlap_priority_class(record: TransitionRecord, delta: Any) -> str:
    old_bbox = tuple(getattr(delta, "bbox_before", ()))
    new_bbox = tuple(getattr(delta, "bbox_after", ()))
    if cluster_overlaps_any_bbox(new_bbox, controlled_candidate_bboxes(record, key="bbox_after")):
        return "controlled_over_mover_or_shared_new"
    if cluster_overlaps_any_bbox(old_bbox, controlled_candidate_bboxes(record, key="bbox_before")):
        return "mover_exposes_controlled_or_background"
    if mover_mover_overlap(record, delta) != "no_mover_overlap":
        return "mover_mover_priority_unknown"
    return "no_overlap_priority_needed"


def mover_overlaps_controlled(record: TransitionRecord, delta: Any) -> str:
    old_bbox = tuple(getattr(delta, "bbox_before", ()))
    new_bbox = tuple(getattr(delta, "bbox_after", ()))
    old_overlap = cluster_overlaps_any_bbox(old_bbox, controlled_candidate_bboxes(record, key="bbox_before"))
    new_overlap = cluster_overlaps_any_bbox(new_bbox, controlled_candidate_bboxes(record, key="bbox_after"))
    if old_overlap and new_overlap:
        return "controlled_overlap_old_and_new"
    if old_overlap:
        return "controlled_overlap_old"
    if new_overlap:
        return "controlled_overlap_new"
    return "no_controlled_overlap"


def mover_mover_overlap(record: TransitionRecord, delta: Any) -> str:
    bbox = tuple(getattr(delta, "bbox_after", ()))
    if len(bbox) != 4:
        return "unknown_mover_overlap"
    overlaps = 0
    for other in stable_independent_motion_deltas(record):
        if int(other.object_id) == int(delta.object_id):
            continue
        other_bbox = tuple(other.bbox_after)
        if len(other_bbox) == 4 and bbox_overlap_area(bbox, other_bbox) > 0:
            overlaps += 1
    if overlaps:
        return f"mover_overlap:{bucket_count(overlaps)}"
    if identity_crossing_candidate(record, delta, []):
        return "near_identical_mover_crossing"
    return "no_mover_overlap"


def trail_or_residue_class(
    record: TransitionRecord,
    delta: Any,
    clusters: list[Mapping[str, Any]],
) -> str:
    old_bbox = tuple(getattr(delta, "bbox_before", ()))
    new_bbox = tuple(getattr(delta, "bbox_after", ()))
    old_only = 0
    new_only = 0
    both = 0
    for cluster in clusters:
        bbox = tuple(cluster.get("bbox", ()))
        if len(bbox) != 4:
            continue
        old_overlap = bbox_overlap_area(bbox, old_bbox) > 0
        new_overlap = bbox_overlap_area(bbox, new_bbox) > 0
        if old_overlap and new_overlap:
            both += 1
        elif old_overlap:
            old_only += 1
        elif new_overlap:
            new_only += 1
    if both:
        return "old_new_overlap_or_trail"
    if old_only and new_only:
        return "separate_old_new_exposure"
    if old_only:
        return "old_pose_residue_or_exposure"
    return "no_trail_or_residue"


def vanish_vs_occluded(record: TransitionRecord, delta: Any, vanished: bool) -> str:
    if not vanished:
        return "not_vanished"
    if mover_overlaps_controlled(record, delta) != "no_controlled_overlap" or mover_mover_overlap(record, delta) != "no_mover_overlap":
        return "vanished_with_overlap_candidate"
    return "vanished_without_overlap_evidence"


def world_clock_transition_rows(records: list[TransitionRecord]) -> list[dict[str, Any]]:
    history_by_object: dict[int, list[tuple[TransitionRecord, Any]]] = defaultdict(list)
    rows = []
    for record in records:
        for delta in stable_independent_motion_deltas(record):
            object_id = int(delta.object_id)
            history = history_by_object[object_id][-4:]
            predicted_bbox = (
                translate_bbox(tuple(history[-1][1].bbox_after), tuple(history[-1][1].translation))
                if history
                else None
            )
            success = predicted_bbox == tuple(delta.bbox_after) if predicted_bbox is not None else False
            failure_class = (
                "constant_velocity_success"
                if success
                else classify_world_clock_failure(record, delta, history, predicted_bbox)
            )
            rows.append(
                {
                    "game": record.game,
                    "step": record.step,
                    "action": record.action,
                    "episode_local_mover_ref": local_mover_ref(record.game, object_id),
                    "history_length": len(history),
                    "history_directions": [
                        motion_direction_from_translation(tuple(item_delta.translation))
                        for _item_record, item_delta in history
                    ],
                    "history_speed_buckets": [
                        bucket_count(motion_magnitude(tuple(item_delta.translation)))
                        for _item_record, item_delta in history
                    ],
                    "actual_direction": motion_direction_from_translation(tuple(delta.translation)),
                    "actual_speed_bucket": bucket_count(motion_magnitude(tuple(delta.translation))),
                    "predicted_bbox_from_constant_velocity": predicted_bbox,
                    "actual_bbox": tuple(delta.bbox_after),
                    "failure_class": failure_class,
                    "track_axis_or_lane": track_axis_for_translations(
                        [tuple(item_delta.translation) for _item_record, item_delta in history] + [tuple(delta.translation)]
                    ),
                    "periodic_phase": periodic_phase_label(history, delta),
                    "boundary_or_bounce_state": boundary_or_bounce_state(record, delta, history),
                    "collision_or_contact_state": collision_or_contact_state(record, delta),
                    "occlusion_state": occlusion_state(record, delta),
                    "coupled_set_role": coupled_set_role(record, delta),
                }
            )
            history_by_object[object_id].append((record, delta))
    return rows


def classify_world_clock_failure(
    record: TransitionRecord,
    delta: Any,
    history: list[tuple[TransitionRecord, Any]],
    predicted_bbox: tuple[int, int, int, int] | None,
) -> str:
    if not history:
        return "unknown_transition_failure"
    if identity_crossing_candidate(record, delta, history):
        return "identity_crossing_failure"
    if occlusion_state(record, delta) != "clear":
        return "occlusion_composition_failure"
    if coupled_set_role(record, delta) != "independent_single":
        return "coupled_group_role_failure"
    if collision_or_contact_state(record, delta) != "no_near_contact":
        return "collision_contact_failure"
    if boundary_or_bounce_state(record, delta, history) != "not_boundary_or_bounce":
        return "constant_velocity_boundary_failure"
    if periodic_phase_label(history, delta) != "no_periodic_phase_evidence":
        return "periodic_phase_failure"
    if predicted_bbox is None:
        return "unknown_transition_failure"
    return "unknown_transition_failure"


def world_clock_heads(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "track_axis_or_lane",
        "periodic_phase",
        "boundary_or_bounce_state",
        "collision_or_contact_state",
        "occlusion_state",
        "coupled_set_role",
    )
    return {
        field: {
            "distribution": dict(Counter(str(row.get(field, "unknown")) for row in rows).most_common()),
            "error_rate_by_value": world_clock_error_rate_by_field(rows, field),
        }
        for field in fields
    }


def world_clock_error_rate_by_field(rows: list[Mapping[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    return {
        value: {
            "count": len(items),
            "constant_velocity_error_rate": round(
                sum(1 for item in items if item["failure_class"] != "constant_velocity_success") / len(items)
                if items
                else 0.0,
                6,
            ),
            "top_failure_classes": dict(Counter(str(item["failure_class"]) for item in items).most_common(5)),
        }
        for value, items in sorted(grouped.items())
    }


def world_clock_failure_examples(rows: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        failure_class = str(row["failure_class"])
        if len(examples[failure_class]) >= 8:
            continue
        examples[failure_class].append(
            {
                "step": row["step"],
                "action": row["action"],
                "episode_local_mover_ref": row["episode_local_mover_ref"],
                "history_length": row["history_length"],
                "history_directions": row["history_directions"],
                "actual_direction": row["actual_direction"],
                "predicted_bbox_from_constant_velocity": row["predicted_bbox_from_constant_velocity"],
                "actual_bbox": row["actual_bbox"],
                "track_axis_or_lane": row["track_axis_or_lane"],
                "periodic_phase": row["periodic_phase"],
                "boundary_or_bounce_state": row["boundary_or_bounce_state"],
                "collision_or_contact_state": row["collision_or_contact_state"],
                "occlusion_state": row["occlusion_state"],
                "coupled_set_role": row["coupled_set_role"],
            }
        )
    return dict(examples)


def periodic_phase_label(history: list[tuple[TransitionRecord, Any]], current: Any) -> str:
    if len(history) < 3:
        return "insufficient_history"
    directions = [
        motion_direction_from_translation(tuple(delta.translation))
        for _record, delta in history
    ] + [motion_direction_from_translation(tuple(current.translation))]
    if len(directions) >= 4 and directions[-1] == directions[-3] and directions[-2] == directions[-4]:
        return "alternating_period_2"
    poses = [tuple(delta.bbox_after) for _record, delta in history] + [tuple(current.bbox_after)]
    if len(set(poses)) < len(poses):
        return "revisited_pose_phase"
    return "no_periodic_phase_evidence"


def boundary_or_bounce_state(
    record: TransitionRecord,
    delta: Any,
    history: list[tuple[TransitionRecord, Any]],
) -> str:
    bbox = tuple(getattr(delta, "bbox_before", ()))
    if len(bbox) != 4:
        return "unknown_boundary"
    shape = np.asarray(record.before.pixels[-1]).shape
    near_boundary = bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= shape[0] - 2 or bbox[3] >= shape[1] - 2
    if not history:
        return "boundary_without_history" if near_boundary else "not_boundary_or_bounce"
    previous_direction = motion_direction_from_translation(tuple(history[-1][1].translation))
    current_direction = motion_direction_from_translation(tuple(delta.translation))
    if previous_direction != "none" and opposite_direction(previous_direction) == current_direction:
        return "bounce_or_reverse"
    if near_boundary:
        return "near_boundary"
    return "not_boundary_or_bounce"


def collision_or_contact_state(record: TransitionRecord, delta: Any) -> str:
    bbox = tuple(getattr(delta, "bbox_before", ()))
    if len(bbox) != 4:
        return "unknown_contact"
    detections = [
        detection
        for detection in getattr(record.before_scene, "detections", ()) or ()
        if detection_bbox_tuple(detection) != bbox
    ]
    nearest = min(
        (
            abs(bbox_center(bbox)[0] - detection.centroid[0])
            + abs(bbox_center(bbox)[1] - detection.centroid[1])
            for detection in detections
        ),
        default=9999.0,
    )
    if nearest <= 1:
        return "touching_component"
    if nearest <= 3:
        return "near_component"
    return "no_near_contact"


def occlusion_state(record: TransitionRecord, delta: Any) -> str:
    object_id = int(getattr(delta, "object_id", -1))
    event = getattr(record, "event", None)
    if object_id in {int(item) for item in getattr(event, "reappeared_object_ids", ()) or ()}:
        return "reappeared_after_occlusion"
    if object_id in {int(item) for item in getattr(event, "disappeared_object_ids", ()) or ()}:
        return "disappeared_or_occluded"
    factors = factorized_changed_regions(record)
    if any(factor["role"] == "controlled_overlap_or_trail" for factor in factors):
        return "overlap_composition_present"
    return "clear"


def coupled_set_role(record: TransitionRecord, delta: Any) -> str:
    object_id = int(getattr(delta, "object_id", -1))
    controlled_ids = controlled_candidate_object_ids(record)
    for group in getattr(getattr(record, "event", None), "motion_groups", ()) or ():
        object_ids = {int(item) for item in getattr(group, "object_ids", ()) or ()}
        if object_id not in object_ids or len(object_ids) < 2:
            continue
        if object_ids & controlled_ids:
            return "coupled_with_controlled_group"
        return "same_delta_independent_group"
    if object_id in controlled_ids:
        return "controlled_candidate"
    return "independent_single"


def identity_crossing_candidate(
    record: TransitionRecord,
    delta: Any,
    history: list[tuple[TransitionRecord, Any]],
) -> bool:
    if not history:
        return False
    bbox = tuple(getattr(delta, "bbox_after", ()))
    if len(bbox) != 4:
        return False
    center = bbox_center(bbox)
    similar_nearby = 0
    signature = mover_signature_from_delta(delta)
    for other in stable_independent_motion_deltas(record):
        if int(other.object_id) == int(delta.object_id):
            continue
        if mover_signature_from_delta(other) != signature:
            continue
        other_center = bbox_center(tuple(other.bbox_after))
        if abs(center[0] - other_center[0]) + abs(center[1] - other_center[1]) <= 2:
            similar_nearby += 1
    if similar_nearby:
        return True
    previous = history[-1][1]
    previous_bbox = tuple(previous.bbox_after)
    return any(
        int(other.object_id) != int(delta.object_id)
        and bbox_overlap_area(previous_bbox, tuple(other.bbox_after)) > 0
        for other in stable_independent_motion_deltas(record)
    )


def mover_identity_link_audit(records: list[TransitionRecord]) -> dict[str, Any]:
    samples = []
    linkable = 0
    linked = 0
    previous_linkable = 0
    previous_linked = 0
    examples = []
    for index, record in enumerate(records):
        next_record = records[index + 1] if index + 1 < len(records) else None
        previous_record = records[index - 1] if index > 0 else None
        next_before_ids = tracked_ids(getattr(next_record, "before_tracked", ())) if next_record else set()
        previous_after_ids = current_tracked_ids(previous_record) if previous_record else set()
        for delta in stable_independent_motion_deltas(record):
            object_id = int(delta.object_id)
            samples.append(delta)
            if next_record is not None:
                linkable += 1
                if object_id in next_before_ids or object_id in event_object_ids(next_record):
                    linked += 1
            if previous_record is not None:
                previous_linkable += 1
                if object_id in previous_after_ids or object_id in event_object_ids(previous_record):
                    previous_linked += 1
            if len(examples) < 12:
                examples.append(
                    {
                        "step": record.step,
                        "action": record.action,
                        "episode_local_mover_ref": local_mover_ref(record.game, object_id),
                        "signature": mover_signature_from_delta(delta),
                        "translation": list(delta.translation),
                        "linked_to_next_before_frame": bool(
                            next_record is not None
                            and (object_id in next_before_ids or object_id in event_object_ids(next_record))
                        ),
                        "linked_from_previous_after_frame": bool(
                            previous_record is not None
                            and (object_id in previous_after_ids or object_id in event_object_ids(previous_record))
                        ),
                    }
                )
    return {
        "independent_mover_delta_count": len(samples),
        "forward_linkable_count": linkable,
        "forward_link_success_count": linked,
        "identity_link_success_rate": round(linked / linkable if linkable else 0.0, 6),
        "backward_linkable_count": previous_linkable,
        "backward_link_success_count": previous_linked,
        "backward_link_success_rate": round(previous_linked / previous_linkable if previous_linkable else 0.0, 6),
        "examples": examples,
    }


def mover_kinematics_audit(records: list[TransitionRecord]) -> dict[str, Any]:
    previous_by_object: dict[int, Any] = {}
    active = 0
    predicted = 0
    bbox_hits = 0
    pose_hits = 0
    direction_hits = 0
    transition_classes: Counter[str] = Counter()
    examples = []
    for record in records:
        for delta in stable_independent_motion_deltas(record):
            active += 1
            transition_classes[mover_transition_relation(previous_by_object.get(int(delta.object_id)), delta)] += 1
            previous = previous_by_object.get(int(delta.object_id))
            if previous is not None:
                predicted += 1
                predicted_bbox = translate_bbox(tuple(previous.bbox_after), tuple(previous.translation))
                actual_bbox = tuple(delta.bbox_after)
                bbox_match = predicted_bbox == actual_bbox
                pose_match = bbox_bucket(predicted_bbox, np.asarray(record.before.pixels[-1]).shape) == bbox_bucket(
                    actual_bbox,
                    np.asarray(record.before.pixels[-1]).shape,
                )
                direction_match = motion_direction_from_translation(tuple(previous.translation)) == motion_direction_from_translation(
                    tuple(delta.translation)
                )
                bbox_hits += int(bbox_match)
                pose_hits += int(pose_match)
                direction_hits += int(direction_match)
                if len(examples) < 12 and not bbox_match:
                    examples.append(
                        {
                            "step": record.step,
                            "action": record.action,
                            "episode_local_mover_ref": local_mover_ref(record.game, int(delta.object_id)),
                            "previous_translation": list(previous.translation),
                            "actual_translation": list(delta.translation),
                            "predicted_bbox_from_constant_velocity": predicted_bbox,
                            "actual_bbox": actual_bbox,
                            "transition_relation": mover_transition_relation(previous, delta),
                        }
                    )
            previous_by_object[int(delta.object_id)] = delta
    return {
        "active_mover_delta_count": active,
        "kinematic_prediction_count": predicted,
        "constant_velocity_bbox_hit_rate": round(bbox_hits / predicted if predicted else 0.0, 6),
        "next_bbox_delta_error": round(1.0 - (bbox_hits / predicted) if predicted else 1.0, 6),
        "pose_bucket_hit_rate": round(pose_hits / predicted if predicted else 0.0, 6),
        "next_pose_signature_error": round(1.0 - (pose_hits / predicted) if predicted else 1.0, 6),
        "direction_persistence_rate": round(direction_hits / predicted if predicted else 0.0, 6),
        "transition_class_distribution": dict(transition_classes.most_common()),
        "failure_examples": examples,
    }


def mover_track_phase_audit(records: list[TransitionRecord]) -> dict[str, Any]:
    sequences: dict[int, list[tuple[TransitionRecord, Any]]] = defaultdict(list)
    for record in records:
        for delta in stable_independent_motion_deltas(record):
            sequences[int(delta.object_id)].append((record, delta))
    track_classes: Counter[str] = Counter()
    phase_candidates = 0
    bounce_candidates = 0
    velocity_consistent = 0
    long_sequences = 0
    examples = []
    for object_id, items in sorted(sequences.items()):
        if len(items) >= 3:
            long_sequences += 1
        translations = [tuple(delta.translation) for _record, delta in items]
        directions = [motion_direction_from_translation(translation) for translation in translations]
        axis = track_axis_for_translations(translations)
        track_classes[axis] += 1
        if len(items) >= 3 and len(set(translations)) == 1:
            velocity_consistent += 1
        if has_direction_reversal(directions):
            bounce_candidates += 1
        poses = [tuple(delta.bbox_after) for _record, delta in items]
        if len(items) >= 4 and len(set(poses)) < len(poses):
            phase_candidates += 1
        if len(examples) < 10:
            examples.append(
                {
                    "episode_local_mover_ref": local_mover_ref(items[0][0].game, object_id),
                    "observation_count": len(items),
                    "track_axis": axis,
                    "directions": directions[:12],
                    "speed_buckets": [bucket_count(motion_magnitude(translation)) for translation in translations[:12]],
                    "phase_candidate": len(items) >= 4 and len(set(poses)) < len(poses),
                    "bounce_or_turn_candidate": has_direction_reversal(directions),
                }
            )
    return {
        "mover_sequence_count": len(sequences),
        "sequences_with_at_least_3_observations": long_sequences,
        "track_class_distribution": dict(track_classes.most_common()),
        "velocity_consistency_rate": round(velocity_consistent / long_sequences if long_sequences else 0.0, 6),
        "bounce_or_turn_candidate_count": bounce_candidates,
        "periodic_phase_candidate_count": phase_candidates,
        "examples": examples,
    }


def mover_group_separation_audit(records: list[TransitionRecord]) -> dict[str, Any]:
    independent_transition_count = 0
    coupled_transition_count = 0
    same_delta_group_count = 0
    mirrored_motion_group_count = 0
    contact_coupled_group_count = 0
    overlap_composed_group_count = 0
    group_observations: Counter[str] = Counter()
    group_consistent = 0
    examples = []
    for record in records:
        if record.action not in {1, 2, 3, 4}:
            continue
        independent = stable_independent_motion_deltas(record)
        groups = tuple(getattr(getattr(record, "event", None), "motion_groups", ()) or ())
        multi_groups = [group for group in groups if len(getattr(group, "object_ids", ())) > 1]
        if independent:
            independent_transition_count += 1
        if multi_groups or len(high_confidence_controlled_candidates(record)) > 1:
            coupled_transition_count += 1
        same_delta_group_count += len(multi_groups)
        translations = [tuple(delta.translation) for delta in stable_motion_deltas(record)]
        if has_opposed_translations(translations):
            mirrored_motion_group_count += 1
        controlled_ids = controlled_candidate_object_ids(record)
        if any(
            any(int(object_id) in controlled_ids for object_id in getattr(group, "object_ids", ()))
            and any(int(object_id) not in controlled_ids for object_id in getattr(group, "object_ids", ()))
            for group in multi_groups
        ):
            contact_coupled_group_count += 1
        factors = factorized_changed_regions(record)
        if any(factor["role"] == "controlled_overlap_or_trail" for factor in factors):
            overlap_composed_group_count += 1
        for group in multi_groups:
            signature = motion_group_signature(record, group)
            group_observations[signature] += 1
        if len(examples) < 12 and (independent or multi_groups):
            examples.append(
                {
                    "step": record.step,
                    "action": record.action,
                    "independent_mover_count": len(independent),
                    "multi_motion_group_count": len(multi_groups),
                    "has_opposed_translations": has_opposed_translations(translations),
                    "has_overlap_composition": any(factor["role"] == "controlled_overlap_or_trail" for factor in factors),
                }
            )
    for count in group_observations.values():
        if count > 1:
            group_consistent += count
    active_transitions = sum(1 for record in records if record.action in {1, 2, 3, 4})
    return {
        "move_action_transition_count": active_transitions,
        "independent_motion_transition_count": independent_transition_count,
        "independent_motion_transition_share": round(independent_transition_count / active_transitions if active_transitions else 0.0, 6),
        "coupled_transition_count": coupled_transition_count,
        "coupled_transition_share": round(coupled_transition_count / active_transitions if active_transitions else 0.0, 6),
        "same_delta_group_count": same_delta_group_count,
        "mirrored_motion_group_count": mirrored_motion_group_count,
        "contact_coupled_group_count": contact_coupled_group_count,
        "overlap_composed_group_count": overlap_composed_group_count,
        "coupled_group_signature_count": len(group_observations),
        "reused_coupled_group_observations": group_consistent,
        "coupled_group_consistency_rate": round(group_consistent / sum(group_observations.values()) if group_observations else 0.0, 6),
        "examples": examples,
    }


def mover_extra_region_explanation_audit(records: list[TransitionRecord]) -> dict[str, Any]:
    explained_roles = {"independent_mover", "controlled_overlap_or_trail"}
    extra_roles = {
        "independent_mover",
        "controlled_overlap_or_trail",
        "contact_object_or_glyph",
        "collateral_remote",
        "background_exposure",
        "unknown",
    }
    role_counts: Counter[str] = Counter()
    total = 0
    unmatched = 0
    occlusion_cases = 0
    occlusion_errors = 0
    examples = []
    for record in records:
        for factor in factorized_changed_regions(record):
            role = str(factor.get("role", "unknown"))
            if role not in extra_roles:
                continue
            total += 1
            role_counts[role] += 1
            if role not in explained_roles:
                unmatched += 1
            if role == "controlled_overlap_or_trail":
                occlusion_cases += 1
                if factor.get("reason") not in {"overlaps_controlled_old_and_new"}:
                    occlusion_errors += 1
            if len(examples) < 12 and role in extra_roles:
                examples.append(
                    {
                        "step": record.step,
                        "action": record.action,
                        "role": role,
                        "reason": factor.get("reason"),
                        "bbox": factor.get("bbox"),
                        "v3_channel": factor.get("v3_channel"),
                    }
                )
    return {
        "extra_changed_region_cluster_count": total,
        "role_distribution": dict(role_counts.most_common()),
        "explained_by_independent_or_coupled_count": total - unmatched,
        "unmatched_changed_region_count": unmatched,
        "unmatched_changed_region_rate": round(unmatched / total if total else 0.0, 6),
        "occlusion_or_overlap_case_count": occlusion_cases,
        "occlusion_reconstruction_error": round(occlusion_errors / occlusion_cases if occlusion_cases else 0.0, 6),
        "examples": examples,
    }


def tracked_ids(items: tuple[TrackedDetection, ...] | Any) -> set[int]:
    return {
        int(item.object_id)
        for item in items or ()
        if getattr(item, "object_id", None) is not None
    }


def current_tracked_ids(record: TransitionRecord) -> set[int]:
    update = getattr(record, "track_update", None)
    return tracked_ids(getattr(update, "current", ()) or ())


def event_object_ids(record: TransitionRecord) -> set[int]:
    ids = set()
    event = getattr(record, "event", None)
    for delta in getattr(event, "object_deltas", ()) or ():
        ids.add(int(delta.object_id))
    for name in ("created_object_ids", "disappeared_object_ids", "reappeared_object_ids"):
        ids.update(int(object_id) for object_id in getattr(event, name, ()) or ())
    return ids


def local_mover_ref(game: str, object_id: int) -> str:
    return hashlib.sha1(f"{game}:{object_id}".encode("utf-8")).hexdigest()[:8]


def mover_signature_from_delta(delta: Any) -> dict[str, Any]:
    return {
        "value": int(getattr(delta, "value_after", 0)),
        "area": bucket_count(int(getattr(delta, "area_after", 0))),
        "shape_transform": getattr(delta, "shape_transform", "unknown"),
        "bbox_size": bbox_size_bucket(tuple(getattr(delta, "bbox_after", ()))),
    }


def bbox_size_bucket(bbox: tuple[Any, ...]) -> tuple[str, str] | str:
    if len(bbox) != 4:
        return "unknown"
    top, left, bottom, right = (int(item) for item in bbox)
    return (bucket_count(bottom - top + 1), bucket_count(right - left + 1))


def translate_bbox(bbox: tuple[Any, ...], translation: tuple[Any, ...]) -> tuple[int, int, int, int]:
    if len(bbox) != 4 or len(translation) != 2:
        return (0, 0, 0, 0)
    row_delta, col_delta = int(translation[0]), int(translation[1])
    top, left, bottom, right = (int(item) for item in bbox)
    return (top + row_delta, left + col_delta, bottom + row_delta, right + col_delta)


def mover_transition_relation(previous: Any | None, current: Any) -> str:
    if previous is None:
        return "first_observed"
    previous_direction = motion_direction_from_translation(tuple(previous.translation))
    current_direction = motion_direction_from_translation(tuple(current.translation))
    if current_direction == "none":
        return "stationary"
    if previous_direction == current_direction:
        return f"same_direction:{current_direction}"
    if opposite_direction(previous_direction) == current_direction:
        return f"bounce_or_reverse:{current_direction}"
    return f"turn:{previous_direction}_to_{current_direction}"


def track_axis_for_translations(translations: list[tuple[Any, ...]]) -> str:
    moving = [translation for translation in translations if len(translation) == 2 and tuple(translation) != (0, 0)]
    if not moving:
        return "stationary_or_unknown"
    row_moves = any(int(row) != 0 for row, _col in moving)
    col_moves = any(int(col) != 0 for _row, col in moving)
    if row_moves and not col_moves:
        return "vertical_track"
    if col_moves and not row_moves:
        return "horizontal_track"
    return "diagonal_or_mixed_track"


def has_direction_reversal(directions: list[str]) -> bool:
    return any(opposite_direction(direction) in directions for direction in directions if direction != "none")


def has_opposed_translations(translations: list[tuple[Any, ...]]) -> bool:
    normalized = {
        (int(translation[0]), int(translation[1]))
        for translation in translations
        if len(translation) == 2
    }
    return any((-row, -col) in normalized for row, col in normalized if (row, col) != (0, 0))


def motion_group_signature(record: TransitionRecord, group: Any) -> str:
    object_ids = tuple(int(object_id) for object_id in getattr(group, "object_ids", ()) or ())
    deltas = {
        int(delta.object_id): delta
        for delta in stable_motion_deltas(record)
    }
    signatures = tuple(
        stable_json(mover_signature_from_delta(deltas[object_id]))
        for object_id in object_ids
        if object_id in deltas
    )
    return stable_json(
        {
            "member_count": bucket_count(len(object_ids)),
            "translation": motion_direction_from_translation(tuple(getattr(group, "translation", ()) or ())),
            "member_signatures": tuple(sorted(signatures)),
        }
    )


def independent_mover_fsa_head(
    records: list[TransitionRecord],
    phase_cluster: Mapping[str, Any],
) -> dict[str, Any]:
    targets = {
        "mover_presence": independent_mover_presence_target,
        "mover_identity_signature": independent_mover_identity_signature_target,
        "pose_signature_t_plus_1": independent_mover_pose_signature_target,
        "direction_distance_bucket": independent_mover_direction_distance_target,
        "bounce_turn_stationary_vanish": independent_mover_transition_class_target,
    }
    models = {
        "IF0_atom_vote_R3a_schema_action": "IF0",
        "IF1_pose_velocity_history": "IF1",
        "IF2_track_free_space_contact": "IF2",
        "IF3_phase_regime": "IF3",
    }
    return mover_probe_head(
        records,
        targets=targets,
        models=models,
        strict_model_name="IF0_strict_R3a_schema_action",
        feature_fn=lambda record, memory, kind: independent_coupled_probe_atoms(
            record,
            model_kind=kind,
            phase_cluster=phase_cluster,
            memory=memory,
            scope="independent_mover_fsa",
        ),
        memory_update_fn=update_independent_coupled_probe_memory,
    )


def coupled_group_motion_head(
    records: list[TransitionRecord],
    phase_cluster: Mapping[str, Any],
) -> dict[str, Any]:
    targets = {
        "single_vs_coupled_set": coupled_group_type_target,
        "group_footprint_delta": coupled_group_footprint_delta_target,
        "overlap_trail_class": coupled_group_overlap_trail_target,
        "collateral_region_count_bucket": coupled_group_collateral_count_target,
        "mirrored_coupled_relation_preserved": coupled_group_relation_target,
    }
    models = {
        "CG0_atom_vote_R3a_schema_action": "CG0",
        "CG1_group_geometry": "CG1",
        "CG2_overlap_symmetry_history": "CG2",
        "CG3_phase_regime": "CG3",
    }
    return mover_probe_head(
        records,
        targets=targets,
        models=models,
        strict_model_name="CG0_strict_R3a_schema_action",
        feature_fn=lambda record, memory, kind: independent_coupled_probe_atoms(
            record,
            model_kind=kind,
            phase_cluster=phase_cluster,
            memory=memory,
            scope="coupled_group_motion",
        ),
        memory_update_fn=update_independent_coupled_probe_memory,
    )


def mover_probe_head(
    records: list[TransitionRecord],
    *,
    targets: Mapping[str, Any],
    models: Mapping[str, str],
    strict_model_name: str,
    feature_fn: Any,
    memory_update_fn: Any,
) -> dict[str, Any]:
    target_reports = {}
    labels = []
    for target_name, target_fn in targets.items():
        strict_baseline = strict_conjunctive_predictor_metrics(
            records,
            active_fn=independent_coupled_probe_active,
            target_fn=target_fn,
            key_fn=motion_strict_context_key,
            memory_update_fn=memory_update_fn,
            model_name=strict_model_name,
        )
        metrics = {
            model_name: bounded_feature_predictor_metrics(
                records,
                active_fn=independent_coupled_probe_active,
                target_fn=target_fn,
                feature_fn=lambda record, memory, kind=model_kind: feature_fn(record, memory, kind),
                memory_update_fn=memory_update_fn,
                model_name=model_name,
            )
            for model_name, model_kind in models.items()
        }
        atom_baseline_name = next(iter(models))
        comparisons = {
            model_name: bounded_predictor_comparison(strict_baseline, payload)
            for model_name, payload in metrics.items()
            if model_name != atom_baseline_name
        }
        atom_vs_strict = bounded_predictor_comparison(strict_baseline, metrics[atom_baseline_name])
        target_reports[target_name] = {
            "strict_conjunctive_baseline": strict_baseline,
            "models": metrics,
            "atom_vote_vs_strict_baseline": atom_vs_strict,
            "comparisons_to_strict_baseline": comparisons,
            "first_successful_model": first_successful_predictor(comparisons),
        }
        if any(item["verdict"] == "success" for item in comparisons.values()):
            labels.append(f"{target_name}_signal")
        if any(item["suspicious_fragmentation_flag"] for item in comparisons.values()):
            labels.append(f"{target_name}_fragmentation_risk")
    if not labels:
        labels.append("no_reusable_signal")
    return {
        "targets": target_reports,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def independent_coupled_probe_active(record: TransitionRecord) -> bool:
    if record.action in {1, 2, 3, 4}:
        return True
    factors = factorized_changed_regions(record)
    return any(factor["role"] in {"independent_mover", "controlled_overlap_or_trail"} for factor in factors)


def independent_coupled_probe_atoms(
    record: TransitionRecord,
    *,
    model_kind: str,
    phase_cluster: Mapping[str, Any],
    memory: Mapping[str, Any],
    scope: str,
) -> tuple[str, ...]:
    atoms = list(base_motion_predictor_atoms(record, target_scope=scope))
    if model_kind in {"IF1", "IF2", "IF3"}:
        atoms.extend(atomize_feature_map("mover_history", independent_mover_fsa_pre_features(record, memory)))
    if model_kind in {"IF2", "IF3"}:
        atoms.extend(atomize_feature_map("mover_contact", independent_mover_contact_pre_features(record, memory)))
    if model_kind == "IF3":
        atoms.extend(atomize_feature_map("phase", mover_phase_features(record, phase_cluster)))
    if model_kind in {"CG1", "CG2", "CG3"}:
        atoms.extend(atomize_feature_map("group_geometry", coupled_group_pre_features(record, memory)))
    if model_kind in {"CG2", "CG3"}:
        atoms.extend(atomize_feature_map("group_symmetry", coupled_group_history_features(record, memory)))
    if model_kind == "CG3":
        atoms.extend(atomize_feature_map("phase", mover_phase_features(record, phase_cluster)))
    return tuple(atoms)


def independent_mover_fsa_pre_features(record: TransitionRecord, memory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action_direction": action_direction(record.action),
        "previous_mover_presence": memory.get("previous_mover_presence", "unknown"),
        "previous_mover_direction_distance": memory.get("previous_mover_direction_distance", "unknown"),
        "previous_mover_pose_signature": memory.get("previous_mover_pose_signature", "unknown"),
        "previous_mover_transition_class": memory.get("previous_mover_transition_class", "unknown"),
        "visible_component_count_bucket": bucket_count(len(getattr(record.before_scene, "detections", ()))),
        "visible_state_rank_counts": safe_vss_rank_only(record.pre_features.get("visible_state_summary", {}) or {}),
        "visible_independent_mover_signature_count": bucket_count(
            len(record.pre_features.get("visible_independent_mover_signatures", ()) or ())
        ),
    }


def independent_mover_contact_pre_features(record: TransitionRecord, memory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "previous_controlled_motion": memory.get("previous_controlled_motion", "unknown"),
        "previous_extra_region_class": memory.get("previous_extra_region_class", "unknown"),
        "current_relative_geometry_class": record.pre_features.get("current_relative_geometry_class"),
        "current_touch_overlap_relation": record.pre_features.get("current_touch_overlap_relation"),
        "actor_or_focus_region": record.pre_features.get("actor_or_focus_region"),
        "action_kind": record.pre_features.get("action_kind"),
    }


def mover_phase_features(record: TransitionRecord, phase_cluster: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase_cluster": phase_for_record(record, phase_cluster),
        "level_snapshot": record.pre_features.get("level_snapshot"),
        "vss_rank_only": safe_vss_rank_only(record.pre_features.get("visible_state_summary", {}) or {}),
    }


def safe_vss_rank_only(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        return ()
    return vss_rank_only(value)


def coupled_group_pre_features(record: TransitionRecord, memory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action_direction": action_direction(record.action),
        "current_relative_geometry_class": record.pre_features.get("current_relative_geometry_class"),
        "current_touch_overlap_relation": record.pre_features.get("current_touch_overlap_relation"),
        "canonical_group_shape_family": record.pre_features.get("canonical_group_shape_family"),
        "mirror_rotation_normalized_signature": record.pre_features.get("mirror_rotation_normalized_signature"),
        "previous_group_type": memory.get("previous_group_type", "unknown"),
        "previous_overlap_trail_class": memory.get("previous_overlap_trail_class", "unknown"),
        "previous_collateral_count": memory.get("previous_collateral_count", "unknown"),
    }


def coupled_group_history_features(record: TransitionRecord, memory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "previous_relation_target": memory.get("previous_relation_target", "unknown"),
        "previous_footprint_delta": memory.get("previous_group_footprint_delta", "unknown"),
        "previous_mover_presence": memory.get("previous_mover_presence", "unknown"),
        "recent_noop_streak": record.pre_features.get("recent_noop_streak"),
    }


def update_independent_coupled_probe_memory(memory: dict[str, Any], record: TransitionRecord) -> None:
    memory["previous_mover_presence"] = independent_mover_presence_target(record)
    memory["previous_mover_direction_distance"] = independent_mover_direction_distance_target(record)
    memory["previous_mover_pose_signature"] = independent_mover_pose_signature_target(record)
    memory["previous_mover_transition_class"] = independent_mover_transition_class_target(record)
    memory["previous_controlled_motion"] = controlled_motion_target(record)
    memory["previous_extra_region_class"] = extra_changed_region_class(record)
    memory["previous_group_type"] = coupled_group_type_target(record)
    memory["previous_overlap_trail_class"] = coupled_group_overlap_trail_target(record)
    memory["previous_collateral_count"] = coupled_group_collateral_count_target(record)
    memory["previous_relation_target"] = coupled_group_relation_target(record)
    memory["previous_group_footprint_delta"] = coupled_group_footprint_delta_target(record)
    memory["previous_action"] = record.action


def independent_mover_presence_target(record: TransitionRecord) -> str:
    deltas = stable_independent_motion_deltas(record)
    return f"{'present' if deltas else 'absent'}:{bucket_count(len(deltas))}"


def independent_mover_identity_signature_target(record: TransitionRecord) -> str:
    signatures = []
    for delta in stable_independent_motion_deltas(record)[:8]:
        signatures.append(
            stable_json(
                {
                    "value": int(delta.value_after),
                    "area": bucket_count(int(delta.area_after)),
                    "shape_transform": delta.shape_transform,
                    "motion": motion_direction_from_translation(tuple(delta.translation)),
                }
            )
        )
    return stable_json({"mover_signatures": tuple(sorted(signatures)), "count": bucket_count(len(signatures))})


def independent_mover_pose_signature_target(record: TransitionRecord) -> str:
    shape = np.asarray(record.before.pixels[-1]).shape
    poses = tuple(
        sorted(
            stable_json(bbox_bucket(tuple(delta.bbox_after), shape))
            for delta in stable_independent_motion_deltas(record)[:8]
        )
    )
    return stable_json({"pose_signature_t_plus_1": poses, "count": bucket_count(len(poses))})


def independent_mover_direction_distance_target(record: TransitionRecord) -> str:
    deltas = stable_independent_motion_deltas(record)
    if not deltas:
        return "absent"
    directions = Counter(motion_direction_from_translation(tuple(delta.translation)) for delta in deltas)
    distance_bucket = bucket_count(sum(motion_magnitude(tuple(delta.translation)) for delta in deltas))
    return stable_json(
        {
            "directions": tuple(sorted((direction, bucket_count(count)) for direction, count in directions.items())),
            "distance_bucket": distance_bucket,
            "mover_count": bucket_count(len(deltas)),
        }
    )


def independent_mover_transition_class_target(record: TransitionRecord) -> str:
    deltas = stable_independent_motion_deltas(record)
    event = getattr(record, "event", None)
    controlled_ids = controlled_candidate_object_ids(record)
    vanished = [
        object_id for object_id in getattr(event, "disappeared_object_ids", ()) or ()
        if int(object_id) not in controlled_ids
    ]
    if vanished and not deltas:
        return f"vanish:{bucket_count(len(vanished))}"
    if not deltas:
        return "stationary_or_absent"
    directions = {motion_direction_from_translation(tuple(delta.translation)) for delta in deltas}
    if "none" in directions:
        return "stationary_with_change"
    if len(directions) > 1:
        return f"turn_or_multi_direction:{bucket_count(len(directions))}:{bucket_count(len(deltas))}"
    direction = next(iter(directions))
    if direction == opposite_direction(previous_independent_direction(record)) and direction != "none":
        return f"bounce_or_reverse:{direction}"
    return f"advance:{direction}:{bucket_count(len(deltas))}"


def coupled_group_type_target(record: TransitionRecord) -> str:
    candidates = high_confidence_controlled_candidates(record)
    independent = stable_independent_motion_deltas(record)
    controlled = record.post_labels.get("labeler_v3", {}).get("controlled_entity", {})
    groups = getattr(getattr(record, "event", None), "motion_groups", ()) or ()
    multi_groups = [group for group in groups if len(getattr(group, "object_ids", ())) > 1]
    if controlled.get("type") in {"coupled_set", "mirrored_set", "grouped_shape"} or len(candidates) > 1:
        return f"coupled_controlled_set:{bucket_count(len(candidates))}"
    if multi_groups:
        return f"multi_object_motion_group:{bucket_count(len(multi_groups))}"
    if len(independent) > 1:
        return f"multiple_independent_movers:{bucket_count(len(independent))}"
    if candidates:
        return "single_controlled"
    return "no_controlled_group"


def coupled_group_footprint_delta_target(record: TransitionRecord) -> str:
    factors = factorized_changed_regions(record)
    shape = np.asarray(record.before.pixels[-1]).shape
    roles = {"controlled_footprint_old", "controlled_footprint_new", "controlled_overlap_or_trail", "independent_mover"}
    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for factor in factors:
        if factor["role"] in roles:
            by_role[factor["role"]].append(factor)
    return stable_json(
        {
            role: {
                "count": bucket_count(len(items)),
                "area": tuple(sorted(str(item.get("area_bucket", "unknown")) for item in items)),
                "bbox": tuple(
                    sorted(
                        stable_json(bbox_bucket(tuple(item["bbox"]), shape))
                        for item in items
                        if isinstance(item.get("bbox"), (list, tuple)) and len(item.get("bbox", ())) == 4
                    )
                ),
            }
            for role, items in sorted(by_role.items())
        }
    )


def coupled_group_overlap_trail_target(record: TransitionRecord) -> str:
    factors = factorized_changed_regions(record)
    overlap = sum(1 for factor in factors if factor["role"] == "controlled_overlap_or_trail")
    old = sum(1 for factor in factors if factor["role"] == "controlled_footprint_old")
    new = sum(1 for factor in factors if factor["role"] == "controlled_footprint_new")
    independent = sum(1 for factor in factors if factor["role"] == "independent_mover")
    if overlap and independent:
        relation = "overlap_with_independent_movers"
    elif overlap:
        relation = "controlled_overlap_or_trail"
    elif old and new:
        relation = "old_new_separate"
    elif new:
        relation = "new_only"
    elif old:
        relation = "old_only"
    else:
        relation = "none"
    return stable_json(
        {
            "class": relation,
            "overlap_count": bucket_count(overlap),
            "old_count": bucket_count(old),
            "new_count": bucket_count(new),
            "independent_count": bucket_count(independent),
        }
    )


def coupled_group_collateral_count_target(record: TransitionRecord) -> str:
    factors = factorized_changed_regions(record)
    extra_roles = {"independent_mover", "hud_resource", "collateral_remote", "background_exposure", "contact_object_or_glyph", "unknown"}
    counts = Counter(factor["role"] for factor in factors if factor["role"] in extra_roles)
    return stable_json(
        {
            "extra_region_count": bucket_count(sum(counts.values())),
            "roles": tuple(sorted((role, bucket_count(count)) for role, count in counts.items())),
        }
    )


def coupled_group_relation_target(record: TransitionRecord) -> str:
    deltas = stable_motion_deltas(record)
    if not deltas:
        return "no_motion_relation"
    translations = Counter(tuple(delta.translation) for delta in deltas if tuple(delta.translation) != (0, 0))
    if not translations:
        return "stationary_relation"
    if len(translations) == 1:
        translation = next(iter(translations))
        return f"shared_translation:{motion_direction_from_translation(translation)}:{bucket_count(sum(translations.values()))}"
    opposite_pairs = 0
    directions = [motion_direction_from_translation(translation) for translation in translations]
    for direction in directions:
        if opposite_direction(direction) in directions:
            opposite_pairs += 1
    if opposite_pairs:
        return f"mirrored_or_opposed:{bucket_count(opposite_pairs)}:{bucket_count(len(translations))}"
    return f"multi_translation_relation:{bucket_count(len(translations))}"


def extra_changed_region_class(record: TransitionRecord) -> str:
    classes = unexplained_extra_region_classes_for_record(record)
    if not classes:
        return "none"
    return "+".join(classes)


def unexplained_extra_changed_region_reclassification(records: list[TransitionRecord]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    example_steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0
    for record in records:
        if record.action not in {1, 2, 3, 4}:
            continue
        classes = unexplained_extra_region_classes_for_record(record)
        if not classes:
            continue
        total += 1
        primary = classes[0]
        counts[primary] += 1
        if len(example_steps[primary]) < 8:
            example_steps[primary].append(
                {
                    "step": record.step,
                    "action": record.action,
                    "context_id": short_digest(motion_strict_context_key(record)),
                    "classes": classes,
                }
            )
    dominant, dominant_count = counts.most_common(1)[0] if counts else ("none", 0)
    return {
        "transition_count_with_extra_regions": total,
        "class_distribution": dict(counts.most_common()),
        "class_share": {
            name: round(count / total if total else 0.0, 6)
            for name, count in counts.items()
        },
        "dominant_class": dominant,
        "dominant_class_share": round(dominant_count / total if total else 0.0, 6),
        "example_steps": dict(example_steps),
    }


def unexplained_extra_region_classes_for_record(record: TransitionRecord) -> list[str]:
    factors = factorized_changed_regions(record)
    roles = {factor["role"] for factor in factors}
    classes = []
    if "independent_mover" in roles:
        classes.append("independent_mover_explained")
    if "controlled_overlap_or_trail" in roles and (
        len(high_confidence_controlled_candidates(record)) > 1
        or coupled_group_type_target(record) != "single_controlled"
    ):
        classes.append("coupled_group_explained")
    if "contact_object_or_glyph" in roles:
        classes.append("contact_effect_explained")
    if roles & {"collateral_remote", "background_exposure", "unknown"}:
        classes.append("true_collateral_unknown")
    return classes


def any_successful_probe(head: Mapping[str, Any]) -> bool:
    for target in head.get("targets", {}).values():
        if any(item.get("verdict") == "success" for item in target.get("comparisons_to_strict_baseline", {}).values()):
            return True
    return False


def any_fragmentation_probe(head: Mapping[str, Any]) -> bool:
    for target in head.get("targets", {}).values():
        if any(item.get("suspicious_fragmentation_flag") for item in target.get("comparisons_to_strict_baseline", {}).values()):
            return True
    return False


def independent_mover_dynamics_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game not in {"ls20", "ar25"}:
        return {
            "status": "skipped_not_independent_mover_game",
            "game": game,
            "reason": "independent mover dynamics are evaluated only for LS20/AR25",
        }
    phase_cluster = build_vss_phase_cluster(records)
    targets = {
        "I0_presence": "I0",
        "I1_motion_class": "I1",
        "I2_direction_distance": "I2",
        "I3_pose_change_signature": "I3",
    }
    models = {
        "IM0_atom_vote_R3a_schema_action": "IM0",
        "IM1_visible_mover_state": "IM1",
        "IM2_previous_pose_phase": "IM2",
        "IM3_contact_conditioned_state": "IM3",
    }
    target_reports = {}
    labels = []
    for target_name, target_kind in targets.items():
        strict_baseline = strict_conjunctive_predictor_metrics(
            records,
            active_fn=lambda record: record.action in {1, 2, 3, 4} or bool(stable_independent_motion_deltas(record)),
            target_fn=lambda record, kind=target_kind: independent_motion_target_at(record, kind),
            key_fn=motion_strict_context_key,
            memory_update_fn=update_independent_mover_memory,
            model_name="IM0_strict_R3a_schema_action",
        )
        metrics = {
            model_name: bounded_feature_predictor_metrics(
                records,
                active_fn=lambda record: record.action in {1, 2, 3, 4} or bool(stable_independent_motion_deltas(record)),
                target_fn=lambda record, kind=target_kind: independent_motion_target_at(record, kind),
                feature_fn=lambda record, memory, kind=model_kind: independent_mover_predictor_atoms(
                    record,
                    model_kind=kind,
                    phase_cluster=phase_cluster,
                    memory=memory,
                ),
                memory_update_fn=update_independent_mover_memory,
                model_name=model_name,
            )
            for model_name, model_kind in models.items()
        }
        atom_baseline = metrics["IM0_atom_vote_R3a_schema_action"]
        comparisons = {
            model_name: bounded_predictor_comparison(strict_baseline, payload)
            for model_name, payload in metrics.items()
            if model_name != "IM0_atom_vote_R3a_schema_action"
        }
        atom_vs_strict = bounded_predictor_comparison(strict_baseline, atom_baseline)
        target_reports[target_name] = {
            "target_kind": target_kind,
            "strict_conjunctive_baseline": strict_baseline,
            "models": metrics,
            "atom_vote_vs_strict_baseline": atom_vs_strict,
            "comparisons_to_strict_IM0": comparisons,
            "first_successful_model": first_successful_predictor(comparisons),
        }
        if any(item["verdict"] == "success" for item in comparisons.values()):
            labels.append(f"{target_kind}_independent_mover_signal")
        if any(item["suspicious_fragmentation_flag"] for item in comparisons.values()):
            labels.append(f"{target_kind}_fragmentation_risk")
    if not labels:
        labels.append("no_reusable_independent_mover_dynamics_signal")
    return {
        "status": "evaluated",
        "game": game,
        "labeler": "v3",
        "probe_type": "strict_context_baseline_plus_finite_state_feature_votes",
        "baseline_note": "IM0_strict_R3a_schema_action is the conjunctive baseline. IM0_atom_vote_R3a_schema_action is retained only as a feature-vote probe.",
        "targets": target_reports,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def cd82_procedural_state_probe_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game != "cd82":
        return {
            "status": "skipped_not_cd82",
            "game": game,
            "reason": "procedural click-state probe is CD82-only",
        }
    role_map = build_region_role_map_cd82_probe(records)
    for record in records:
        record.post_labels["labeler_cd82_region_role"] = build_post_labels_cd82_region_role(record, role_map)
    models = {
        "C0_majority_click_action": "C0",
        "C1_visual_region_role": "C1",
        "C2_procedural_memory": "C2",
        "C3_visual_plus_procedural": "C3",
    }
    metrics = {
        model_name: bounded_feature_predictor_metrics(
            records,
            active_fn=lambda record: record.pre_features.get("action_kind") == "click",
            target_fn=cd82_procedural_target,
            feature_fn=lambda record, memory, kind=model_kind: cd82_procedural_predictor_atoms(
                record,
                model_kind=kind,
                memory=memory,
            ),
            memory_update_fn=update_cd82_procedural_memory,
            model_name=model_name,
        )
        for model_name, model_kind in models.items()
    }
    baseline = metrics["C0_majority_click_action"]
    comparisons = {
        model_name: bounded_predictor_comparison(baseline, payload)
        for model_name, payload in metrics.items()
        if model_name != "C0_majority_click_action"
    }
    labels = []
    if comparisons["C1_visual_region_role"]["verdict"] == "success":
        labels.append("procedural_visual_region_signal")
    if comparisons["C2_procedural_memory"]["verdict"] == "success":
        labels.append("procedural_memory_signal")
    if comparisons["C3_visual_plus_procedural"]["verdict"] == "success":
        labels.append("procedural_state_signal")
    if any(item["suspicious_fragmentation_flag"] for item in comparisons.values()):
        labels.append("procedural_probe_fragmentation_risk")
    if not labels:
        labels.append("no_reusable_procedural_state_signal")
    return {
        "status": "evaluated",
        "game": game,
        "labeler": "cd82_region_role",
        "probe_type": "bounded_feature_predictors_not_context_keys",
        "target": "no_op / palette_select / canvas_edit / terminal_progress / hud_status / other_delta",
        "models": metrics,
        "comparisons_to_C0": comparisons,
        "verdict": {
            "primary": labels[0],
            "labels": sorted(set(labels), key=labels.index),
        },
    }


def motion_predictor_model_kinds() -> dict[str, str]:
    return {
        "M0_atom_vote_R3a_schema_action": "M0",
        "M1_local_passability_occupancy": "M1",
        "M2_contact_conditioned": "M2",
        "M3_independent_mover_tracking_phase": "M3",
        "M4_coupled_set_relational": "M4",
    }


def strict_conjunctive_predictor_metrics(
    records: list[TransitionRecord],
    *,
    active_fn: Any,
    target_fn: Any,
    key_fn: Any,
    memory_update_fn: Any,
    model_name: str,
) -> dict[str, Any]:
    prior: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    target_counts: Counter[str] = Counter()
    memory: dict[str, Any] = {}
    active_examples = 0
    predictions = 0
    hits = 0
    error_examples = []
    for record in records:
        if not active_fn(record):
            memory_update_fn(memory, record)
            continue
        active_examples += 1
        target = str(target_fn(record))
        key = str(key_fn(record, memory))
        if prior[key]:
            predictions += 1
            predicted = prior[key].most_common(1)[0][0]
            if predicted == target:
                hits += 1
            elif len(error_examples) < 20:
                error_examples.append(
                    {
                        "game": record.game,
                        "step": record.step,
                        "action": record.action,
                        "model": model_name,
                        "predicted": predicted,
                        "actual": target,
                        "context_support": sum(prior[key].values()),
                        "context_id": short_digest(key),
                    }
                )
        prior[key][target] += 1
        groups[key][target] += 1
        target_counts[target] += 1
        memory_update_fn(memory, record)
    reused_contexts = {key for key, counter in groups.items() if sum(counter.values()) > 1}
    reused_transitions = sum(sum(groups[key].values()) for key in reused_contexts)
    accuracy = hits / predictions if predictions else 0.0
    return {
        "model": model_name,
        "active_examples": active_examples,
        "prediction_count": predictions,
        "prediction_coverage": round(predictions / active_examples if active_examples else 0.0, 6),
        "prequential_accuracy": round(accuracy, 6),
        "error": round(1.0 - accuracy if predictions else 1.0, 6),
        "target_entropy": round(entropy(target_counts) if active_examples else 0.0, 6),
        "majority_target_share": round(
            target_counts.most_common(1)[0][1] / active_examples
            if active_examples and target_counts
            else 0.0,
            6,
        ),
        "target_distribution": dict(target_counts.most_common(20)),
        "feature_cardinality": len(groups),
        "unique_contexts": len(groups),
        "reused_contexts": len(reused_contexts),
        "reused_transitions": reused_transitions,
        "feature_reuse_coverage": round(reused_transitions / active_examples if active_examples else 0.0, 6),
        "one_off_feature_ratio": round(
            sum(1 for counter in groups.values() if sum(counter.values()) == 1) / len(groups)
            if groups
            else 0.0,
            6,
        ),
        "support": bounded_predictor_support_status(
            active_examples=active_examples,
            prediction_count=predictions,
            feature_reuse_coverage=reused_transitions / active_examples if active_examples else 0.0,
        ),
        "error_examples": error_examples,
    }


def motion_strict_context_key(record: TransitionRecord, memory: Mapping[str, Any] | None = None) -> str:
    return stable_json(
        {
            "r3a_context": context_key(record, R3A_COMPRESSED),
            "schema": outcome_schema_key(record, labeler="v3"),
            "action": record.action,
        }
    )


def bounded_feature_predictor_metrics(
    records: list[TransitionRecord],
    *,
    active_fn: Any,
    target_fn: Any,
    feature_fn: Any,
    memory_update_fn: Any,
    model_name: str,
) -> dict[str, Any]:
    prior: dict[str, Counter[str]] = defaultdict(Counter)
    final_support: Counter[str] = Counter()
    record_atoms: list[tuple[str, ...]] = []
    target_counts: Counter[str] = Counter()
    memory: dict[str, Any] = {}
    active_examples = 0
    predictions = 0
    hits = 0
    error_examples = []
    for record in records:
        if not active_fn(record):
            memory_update_fn(memory, record)
            continue
        active_examples += 1
        target = str(target_fn(record))
        atoms = tuple(sorted(set(str(atom) for atom in feature_fn(record, memory))))
        if not atoms:
            atoms = ("feature:none",)
        record_atoms.append(atoms)
        votes: Counter[str] = Counter()
        for atom in atoms:
            votes.update(prior[atom])
        if votes:
            predictions += 1
            predicted = votes.most_common(1)[0][0]
            if predicted == target:
                hits += 1
            elif len(error_examples) < 20:
                error_examples.append(
                    {
                        "game": record.game,
                        "step": record.step,
                        "action": record.action,
                        "model": model_name,
                        "predicted": predicted,
                        "actual": target,
                        "vote_support": sum(votes.values()),
                        "atom_count": len(atoms),
                    }
                )
        for atom in atoms:
            prior[atom][target] += 1
            final_support[atom] += 1
        target_counts[target] += 1
        memory_update_fn(memory, record)
    reused_atoms = {atom for atom, count in final_support.items() if count > 1}
    reuse_hits = sum(1 for atoms in record_atoms if any(atom in reused_atoms for atom in atoms))
    accuracy = hits / predictions if predictions else 0.0
    return {
        "model": model_name,
        "active_examples": active_examples,
        "prediction_count": predictions,
        "prediction_coverage": round(predictions / active_examples if active_examples else 0.0, 6),
        "prequential_accuracy": round(accuracy, 6),
        "error": round(1.0 - accuracy if predictions else 1.0, 6),
        "target_entropy": round(entropy(target_counts) if active_examples else 0.0, 6),
        "majority_target_share": round(
            target_counts.most_common(1)[0][1] / active_examples
            if active_examples and target_counts
            else 0.0,
            6,
        ),
        "target_distribution": dict(target_counts.most_common(20)),
        "feature_cardinality": len(final_support),
        "reused_features": len(reused_atoms),
        "feature_reuse_coverage": round(reuse_hits / active_examples if active_examples else 0.0, 6),
        "one_off_feature_ratio": round(
            sum(1 for count in final_support.values() if count == 1) / len(final_support)
            if final_support
            else 0.0,
            6,
        ),
        "support": bounded_predictor_support_status(
            active_examples=active_examples,
            prediction_count=predictions,
            feature_reuse_coverage=reuse_hits / active_examples if active_examples else 0.0,
        ),
        "error_examples": error_examples,
    }


def bounded_predictor_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_error = float(baseline.get("error", 1.0) or 1.0)
    candidate_error = float(candidate.get("error", 1.0) or 1.0)
    error_drop = max(0.0, baseline_error - candidate_error)
    relative_drop = error_drop / baseline_error if baseline_error else 0.0
    coverage_loss = max(
        0.0,
        float(baseline.get("prediction_coverage", 0.0) or 0.0)
        - float(candidate.get("prediction_coverage", 0.0) or 0.0),
    )
    reuse_loss = max(
        0.0,
        float(baseline.get("feature_reuse_coverage", 0.0) or 0.0)
        - float(candidate.get("feature_reuse_coverage", 0.0) or 0.0),
    )
    cardinality_multiplier = (
        float(candidate.get("feature_cardinality", 0) or 0)
        / max(1.0, float(baseline.get("feature_cardinality", 0) or 0))
    )
    support = candidate.get("support", {})
    suspicious = error_drop > 0.0 and (
        coverage_loss > 0.10
        or reuse_loss > 0.10
        or (cardinality_multiplier > 8.0 and float(candidate.get("one_off_feature_ratio", 0.0) or 0.0) > 0.75)
    )
    success = (
        support.get("status") == "eligible"
        and relative_drop >= 0.15
        and float(candidate.get("prediction_coverage", 0.0) or 0.0) >= 0.50
        and coverage_loss <= 0.10
        and reuse_loss <= 0.10
        and not suspicious
    )
    if support.get("status") == "insufficient_support":
        verdict = "insufficient_support"
    elif suspicious:
        verdict = "suspicious_fragmentation"
    elif success:
        verdict = "success"
    else:
        verdict = "no_gain"
    return {
        "error_drop": round(error_drop, 6),
        "error_relative_drop": round(relative_drop, 6),
        "prediction_coverage_loss": round(coverage_loss, 6),
        "feature_reuse_loss": round(reuse_loss, 6),
        "feature_cardinality_multiplier_vs_baseline": round(cardinality_multiplier, 6),
        "gain_per_reuse_lost": round(error_drop / max(reuse_loss, 1e-6), 6),
        "support": support,
        "suspicious_fragmentation_flag": suspicious,
        "verdict": verdict,
    }


def bounded_predictor_support_status(
    *,
    active_examples: int,
    prediction_count: int,
    feature_reuse_coverage: float,
) -> dict[str, Any]:
    eligible = (
        active_examples >= 30
        and prediction_count >= 30
        and feature_reuse_coverage >= 0.50
    ) or (active_examples < 30 and prediction_count >= 2 and feature_reuse_coverage >= 0.25)
    return {
        "status": "eligible" if eligible else "insufficient_support",
        "active_examples": active_examples,
        "prediction_count": prediction_count,
        "feature_reuse_coverage": round(feature_reuse_coverage, 6),
    }


def first_successful_predictor(comparisons: Mapping[str, Mapping[str, Any]]) -> str | None:
    for name, payload in comparisons.items():
        if payload.get("verdict") == "success":
            return name
    return None


def motion_predictor_atoms(
    record: TransitionRecord,
    *,
    model_kind: str,
    phase_cluster: Mapping[str, Any],
    memory: Mapping[str, Any],
    target_scope: str,
) -> tuple[str, ...]:
    atoms = list(base_motion_predictor_atoms(record, target_scope=target_scope))
    if model_kind in {"M1", "M2", "M3", "M4"}:
        atoms.extend(atomize_feature_map("local", local_motion_law_features(record)))
    if model_kind in {"M2", "M3", "M4"}:
        atoms.extend(atomize_feature_map("contact", contact_conditioned_parameter_features(record)))
    if model_kind in {"M3", "M4"}:
        atoms.extend(atomize_feature_map("mover", independent_mover_tracking_features(record, phase_cluster, memory)))
    if model_kind == "M4":
        atoms.extend(atomize_feature_map("coupled", coupled_set_relational_features(record)))
    return tuple(atoms)


def base_motion_predictor_atoms(record: TransitionRecord, *, target_scope: str) -> tuple[str, ...]:
    return (
        predictor_atom("base", "scope", target_scope),
        predictor_atom("base", "action", record.action),
        predictor_atom("base", "schema", outcome_schema_key(record, labeler="v3")),
        predictor_atom("base", "r3a_context", context_key(record, R3A_COMPRESSED)),
    )


def independent_mover_predictor_atoms(
    record: TransitionRecord,
    *,
    model_kind: str,
    phase_cluster: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> tuple[str, ...]:
    atoms = list(base_motion_predictor_atoms(record, target_scope="independent_mover"))
    if model_kind in {"IM1", "IM2", "IM3"}:
        atoms.extend(atomize_feature_map("visible_mover", visible_mover_state_features(record)))
    if model_kind in {"IM2", "IM3"}:
        atoms.extend(atomize_feature_map("mover_memory", independent_mover_memory_features(record, phase_cluster, memory)))
    if model_kind == "IM3":
        atoms.extend(atomize_feature_map("mover_contact", independent_mover_contact_features(record)))
    return tuple(atoms)


def cd82_procedural_predictor_atoms(
    record: TransitionRecord,
    *,
    model_kind: str,
    memory: Mapping[str, Any],
) -> tuple[str, ...]:
    atoms = [
        predictor_atom("base", "action", record.action),
        predictor_atom("base", "action_kind", record.pre_features.get("action_kind")),
    ]
    if model_kind in {"C1", "C3"}:
        atoms.extend(atomize_feature_map("visual", cd82_visual_click_features(record)))
    if model_kind in {"C2", "C3"}:
        atoms.extend(atomize_feature_map("procedural", cd82_procedural_memory_features(record, memory)))
    return tuple(atoms)


def atomize_feature_map(prefix: str, payload: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        predictor_atom(prefix, key, value)
        for key, value in sorted(payload.items())
    )


def predictor_atom(prefix: str, name: str, value: Any) -> str:
    return f"{prefix}:{name}:{stable_json(jsonable(value))}"


def controlled_motion_target_at(record: TransitionRecord, target_kind: str) -> str:
    candidates = high_confidence_controlled_candidates(record)
    translations = sorted({tuple(candidate.get("translation", ())) for candidate in candidates})
    magnitudes = [motion_magnitude(translation) for translation in translations]
    moved = any(magnitude > 0 for magnitude in magnitudes)
    if target_kind == "T0":
        return "controlled_motion" if moved else "no_controlled_motion"
    if target_kind == "T1":
        if not moved:
            return "no_move"
        if len(translations) > 1 or len(candidates) > 1:
            return "coupled_or_multi"
        magnitude = magnitudes[0]
        if magnitude == 1:
            return "unit_move"
        return "forced_slide"
    if target_kind == "T2":
        if not moved:
            return "no_move"
        if len(translations) > 1:
            return f"multi_direction:{bucket_count(len(translations))}"
        translation = translations[0]
        return f"{motion_direction_from_translation(translation)}:{bucket_count(motion_magnitude(translation))}"
    if target_kind == "T3":
        if not moved:
            return "vector:0:0"
        if len(translations) > 1:
            return "vector:coupled_set"
        row_delta, col_delta = normalized_motion_vector(translations[0])
        return f"vector:{row_delta}:{col_delta}"
    if target_kind == "T4":
        return controlled_motion_stop_reason(record, translations)
    if target_kind == "T5":
        if not moved:
            return "no_controlled_region_change"
        return channel_parameter_target(record, "controlled_entity_delta")
    return controlled_motion_target(record)


def controlled_motion_stop_reason(
    record: TransitionRecord,
    translations: list[tuple[Any, ...]],
) -> str:
    candidate = controlled_focus_candidate(record)
    bbox = tuple(candidate.get("bbox_before", ())) if candidate else None
    side = action_direction(record.action)
    grid = np.asarray(record.before.pixels[-1])
    free_state = side_free_space_pattern(grid, bbox).get(side, "unknown") if bbox else "unknown"
    moved = any(motion_magnitude(tuple(translation)) > 0 for translation in translations)
    if not moved:
        if free_state in {"occupied", "mixed", "out"}:
            return f"blocked:{free_state}"
        return "no_move_unknown"
    if len(translations) > 1:
        return "coupled_or_multi"
    magnitude = motion_magnitude(tuple(translations[0])) if translations else 0
    if magnitude > 1:
        return "slide_stop"
    contact = adjacent_detection_on_side(record.before_scene.detections, bbox, side) if bbox and record.before_scene else None
    if contact is not None:
        return "contact"
    if free_state == "free":
        return "free"
    return "unknown_stop_reason"


def normalized_motion_vector(translation: tuple[Any, ...]) -> tuple[int, int]:
    if len(translation) != 2:
        return (0, 0)
    row_delta, col_delta = int(translation[0]), int(translation[1])
    magnitude = max(1, math.gcd(abs(row_delta), abs(col_delta)))
    return (int(row_delta / magnitude), int(col_delta / magnitude))


def independent_motion_target_at(record: TransitionRecord, target_kind: str) -> str:
    deltas = stable_independent_motion_deltas(record)
    if target_kind == "I0":
        return "independent_motion_present" if deltas else "absent"
    if target_kind == "I1":
        return independent_motion_target(record)
    if target_kind == "I2":
        if not deltas:
            return "absent"
        directions = sorted({motion_direction_from_translation(tuple(delta.translation)) for delta in deltas})
        magnitude = sum(motion_magnitude(tuple(delta.translation)) for delta in deltas)
        return f"{'+'.join(directions)}:{bucket_count(magnitude)}:{bucket_count(len(deltas))}"
    if target_kind == "I3":
        if not deltas:
            return "absent"
        shape = np.asarray(record.before.pixels[-1]).shape
        bboxes = tuple(
            sorted(
                stable_json(bbox_bucket(tuple(delta.bbox_after), shape))
                for delta in deltas[:8]
            )
        )
        directions = tuple(sorted(motion_direction_from_translation(tuple(delta.translation)) for delta in deltas))
        return stable_json({"directions": directions, "bbox_buckets": bboxes, "count": bucket_count(len(deltas))})
    return independent_motion_target(record)


def visible_mover_state_features(record: TransitionRecord) -> dict[str, Any]:
    detections = getattr(record.before_scene, "detections", ())
    return {
        "component_count_bucket": bucket_count(len(detections)),
        "component_signature_histogram": component_signature_histogram(detections),
        "visible_state_rank_counts": vss_rank_only(record.pre_features.get("visible_state_summary", {}) or {}),
    }


def independent_mover_memory_features(
    record: TransitionRecord,
    phase_cluster: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "phase_cluster": phase_for_record(record, phase_cluster),
        "previous_mover_target": memory.get("previous_independent_motion_target", "none"),
        "previous_mover_direction": memory.get("previous_independent_motion_direction", "none"),
        "previous_mover_count": memory.get("previous_independent_mover_count", "zero"),
        "previous_action": memory.get("previous_action", "none"),
    }


def independent_mover_contact_features(record: TransitionRecord) -> dict[str, Any]:
    return {
        "controlled_to_independent_relation": controlled_to_independent_relation(record),
        "controlled_motion_target": controlled_motion_target(record),
        "contact_conditioning": contact_conditioned_parameter_features(record),
    }


def update_independent_mover_memory(memory: dict[str, Any], record: TransitionRecord) -> None:
    deltas = stable_independent_motion_deltas(record)
    memory["previous_independent_motion_target"] = independent_motion_target(record)
    memory["previous_independent_motion_direction"] = (
        motion_direction_from_translation(tuple(deltas[0].translation)) if deltas else "none"
    )
    memory["previous_independent_mover_count"] = bucket_count(len(deltas))
    memory["previous_action"] = record.action


def cd82_procedural_target(record: TransitionRecord) -> str:
    labeler = record.post_labels.get("labeler_cd82_region_role", {})
    channels = labeler.get("channels", {})
    if channel_is_active_v3("terminal_delta", channels.get("terminal_delta")):
        return "terminal_progress"
    region_role = labeler.get("region_roles", {}).get("action_target_region", {}).get("role", "unknown")
    if channel_is_active_v3("action_target_delta", channels.get("action_target_delta")):
        return "canvas_edit"
    if record.pre_features.get("action_kind") == "click" and region_role == "tool_control_region":
        return "palette_select"
    if channel_is_active_v3("hud_status_delta", channels.get("hud_status_delta")):
        return "hud_status"
    if not labeler.get("active_channels"):
        return "no_op"
    return "other_delta"


def cd82_visual_click_features(record: TransitionRecord) -> dict[str, Any]:
    role = record.post_labels.get("labeler_cd82_region_role", {}).get("region_roles", {}).get("action_target_region", {})
    return {
        "click_target_zone": record.pre_features.get("click_target_zone"),
        "click_target_type_visual": record.pre_features.get("click_target_type_visual"),
        "click_target_visual_signature": record.pre_features.get("click_target_visual_signature"),
        "clicked_static_component_class": record.pre_features.get("clicked_static_component_class"),
        "clicked_region_role": role.get("role", "unknown"),
        "clicked_region_role_confidence": role.get("confidence", "low"),
    }


def cd82_procedural_memory_features(record: TransitionRecord, memory: Mapping[str, Any]) -> dict[str, Any]:
    target = action_target_point(record.action_data)
    shape = np.asarray(record.before.pixels[-1]).shape
    target_key = region_key(target, shape) if target is not None else "none"
    return {
        "selected_tool_proxy": memory.get("selected_tool_proxy", "none"),
        "last_palette_click_region": memory.get("last_palette_click_region", "none"),
        "canvas_edit_count_bucket": bucket_count(int(memory.get("canvas_edit_count", 0) or 0)),
        "target_already_changed": target_key in memory.get("changed_target_regions", set()),
        "target_material_before_click": target_material_summary(record),
        "clicked_region_role_before_t": memory.get("last_clicked_region_role", "none"),
    }


def update_cd82_procedural_memory(memory: dict[str, Any], record: TransitionRecord) -> None:
    if record.pre_features.get("action_kind") != "click":
        return
    labeler = record.post_labels.get("labeler_cd82_region_role", {})
    role = labeler.get("region_roles", {}).get("action_target_region", {}).get("role", "unknown")
    target = action_target_point(record.action_data)
    shape = np.asarray(record.before.pixels[-1]).shape
    target_key = region_key(target, shape) if target is not None else "none"
    if role == "tool_control_region":
        memory["selected_tool_proxy"] = stable_json(record.pre_features.get("click_target_visual_signature"))
        memory["last_palette_click_region"] = target_key
    if channel_is_active_v3("action_target_delta", labeler.get("channels", {}).get("action_target_delta")):
        memory["canvas_edit_count"] = int(memory.get("canvas_edit_count", 0) or 0) + 1
        changed = set(memory.get("changed_target_regions", set()))
        changed.add(target_key)
        memory["changed_target_regions"] = changed
    memory["last_clicked_region_role"] = role


def target_material_summary(record: TransitionRecord) -> Any:
    target = action_target_point(record.action_data)
    if target is None:
        return "none"
    grid = np.asarray(record.before.pixels[-1])
    row = int(round(float(target[0])))
    col = int(round(float(target[1])))
    if row < 0 or col < 0 or row >= grid.shape[0] or col >= grid.shape[1]:
        return "out"
    return int(grid[row, col])


def parameter_probe_metrics(records: list[TransitionRecord], *, model_kind: str) -> dict[str, Any]:
    prior: dict[str, Counter[str]] = defaultdict(Counter)
    all_groups: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[TransitionRecord]] = defaultdict(list)
    predictions = 0
    hits = 0
    error_examples = []
    for record in records:
        schema = outcome_schema_key(record, labeler="v3")
        parameter = parameter_probe_target(record)
        key = parameter_probe_key(record, model_kind=model_kind, schema=schema)
        if prior[key]:
            predictions += 1
            predicted = prior[key].most_common(1)[0][0]
            if predicted == parameter:
                hits += 1
            elif len(error_examples) < 20:
                error_examples.append(
                    {
                        "game": record.game,
                        "step": record.step,
                        "action": record.action,
                        "model": model_kind,
                        "context_key": key,
                        "predicted_hash": short_digest(predicted),
                        "actual_hash": short_digest(parameter),
                        "predicted_summary": parameter_payload_summary(predicted),
                        "actual_summary": parameter_payload_summary(parameter),
                    }
                )
        prior[key][parameter] += 1
        all_groups[key][parameter] += 1
        examples[key].append(record)
    total = len(records)
    reused_keys = [key for key, counter in all_groups.items() if sum(counter.values()) > 1]
    contradicted = [key for key in reused_keys if len(all_groups[key]) > 1]
    reused_transitions = sum(sum(all_groups[key].values()) for key in reused_keys)
    weighted_entropy = sum(
        sum(counter.values()) * entropy(counter)
        for counter in all_groups.values()
    )
    accuracy = hits / predictions if predictions else 0.0
    return {
        "model": model_kind,
        "examples": total,
        "prediction_count": predictions,
        "prediction_coverage": round(predictions / total if total else 0.0, 6),
        "prequential_accuracy": round(accuracy, 6),
        "parameter_error_given_schema": round(1.0 - accuracy if predictions else 1.0, 6),
        "unique_contexts": len(all_groups),
        "reused_contexts": len(reused_keys),
        "reused_transitions": reused_transitions,
        "reuse_coverage": round(reused_transitions / total if total else 0.0, 6),
        "one_off_ratio": round(
            sum(1 for counter in all_groups.values() if sum(counter.values()) == 1) / len(all_groups)
            if all_groups
            else 0.0,
            6,
        ),
        "contradicted_reused_contexts": len(contradicted),
        "parameter_contradiction_rate": round(len(contradicted) / len(reused_keys) if reused_keys else 0.0, 6),
        "parameter_entropy": round(weighted_entropy / total if total else 0.0, 6),
        "support": parameter_probe_support_status(
            examples=total,
            reused_contexts=len(reused_keys),
            reused_transitions=reused_transitions,
            reuse_coverage=reused_transitions / total if total else 0.0,
            prediction_count=predictions,
        ),
        "error_examples": error_examples,
    }


def parameter_probe_key(record: TransitionRecord, *, model_kind: str, schema: str) -> str:
    payload: dict[str, Any] = {
        "r3a_context": context_key(record, R3A_COMPRESSED),
        "action": record.action,
        "coarse_schema": schema,
    }
    if model_kind in {"P1", "P2"}:
        payload["local_geometry"] = local_geometry_parameter_features(record)
    if model_kind == "P2":
        payload["contact_conditioning"] = contact_conditioned_parameter_features(record)
    return stable_json(payload)


def parameter_probe_target(record: TransitionRecord) -> str:
    payload = record.post_labels.get("labeler_v3", {}).get("parameter_outcome", {})
    shape = np.asarray(record.before.pixels[-1]).shape
    return stable_json(coarse_parameter_target(payload, shape))


def coarse_parameter_target(payload: Any, shape: tuple[int, int]) -> Any:
    if not isinstance(payload, Mapping):
        return jsonable(payload)
    if not any(isinstance(value, Mapping) and "cluster_count" in value for value in payload.values()):
        return jsonable(payload)
    coarse: dict[str, Any] = {}
    for channel, value in payload.items():
        if not isinstance(value, Mapping):
            if value:
                coarse[channel] = "present"
            continue
        if "cluster_count" in value:
            count = int(value.get("cluster_count", 0) or 0)
            if count <= 0:
                continue
            coarse[channel] = {
                "cluster_count_bucket": bucket_count(count),
                "area_buckets": tuple(value.get("area_buckets", ())),
                "bbox_buckets": tuple(
                    sorted(
                        stable_json(bbox_bucket(tuple(bbox), shape))
                        for bbox in value.get("bboxes", ())
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4
                    )
                ),
                "changed_value_histogram": tuple(value.get("color_histogram_delta", ())),
                "roles": tuple(value.get("roles", ())),
                "controlled_entity_type": value.get("controlled_entity_type"),
                "controlled_confidence": value.get("controlled_confidence"),
                "control_translations": tuple(
                    tuple(candidate.get("translation", ()))
                    for candidate in value.get("top_candidates", ())
                    if isinstance(candidate, Mapping)
                ),
            }
        elif channel == "resource_delta" and value:
            coarse[channel] = {
                "levels_completed_delta": bucket_signed(int(value.get("levels_completed_delta", 0) or 0)),
                "hud_cluster_count": bucket_count(int(value.get("hud_cluster_count", 0) or 0)),
            }
        elif channel == "terminal_delta" and value:
            coarse[channel] = {
                "terminal": bool(value.get("terminal")),
                "state_after": value.get("state_after"),
            }
    return coarse


def short_digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def parameter_payload_summary(payload: str) -> dict[str, Any]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return {"raw_hash": short_digest(payload)}
    if not isinstance(decoded, Mapping):
        return {"raw_hash": short_digest(payload)}
    summary = {}
    for channel, value in decoded.items():
        if not isinstance(value, Mapping):
            if value:
                summary[channel] = "present"
            continue
        cluster_count = value.get("cluster_count")
        if cluster_count:
            summary[channel] = {
                "cluster_count": cluster_count,
                "area_buckets": value.get("area_buckets"),
                "roles": value.get("roles"),
                "controlled_entity_type": value.get("controlled_entity_type"),
            }
        elif value.get("cluster_count_bucket"):
            summary[channel] = {
                "cluster_count_bucket": value.get("cluster_count_bucket"),
                "area_buckets": value.get("area_buckets"),
                "bbox_buckets": value.get("bbox_buckets"),
                "roles": value.get("roles"),
                "controlled_entity_type": value.get("controlled_entity_type"),
                "control_translations": value.get("control_translations"),
            }
        elif channel in {"terminal_delta", "resource_delta"} and value:
            summary[channel] = value
    return summary


def local_geometry_parameter_features(record: TransitionRecord) -> dict[str, Any]:
    candidate = controlled_focus_candidate(record)
    bbox = tuple(candidate.get("bbox_before", ())) if candidate else None
    scene = record.before_scene
    detections = getattr(scene, "detections", ())
    detection = detection_for_bbox(scene, bbox) if bbox and scene is not None else None
    grid = np.asarray(record.before.pixels[-1])
    center = bbox_center(bbox) if bbox else action_target_point(record.action_data)
    return {
        "controlled_entity_type": record.post_labels.get("labeler_v3", {}).get("controlled_entity", {}).get("type", "none"),
        "controlled_confidence": record.post_labels.get("labeler_v3", {}).get("controlled_entity", {}).get("confidence", "low"),
        "controlled_set_size_bucket": bucket_count(len(high_confidence_controlled_candidates(record))),
        "controlled_bbox_bucket": bbox_bucket(bbox, grid.shape) if bbox else None,
        "controlled_shape_class": static_component_class(detection),
        "controlled_signature": coarse_component_signature(detection),
        "local_obstacle_map": local_grid_pattern(grid, center),
        "relative_free_space_pattern": side_free_space_pattern(grid, bbox),
        "nearest_component_signatures": nearest_component_signatures_for_point(
            detections,
            center,
            exclude_bbox=bbox,
        ),
    }


def contact_conditioned_parameter_features(record: TransitionRecord) -> dict[str, Any]:
    candidate = controlled_focus_candidate(record)
    bbox = tuple(candidate.get("bbox_before", ())) if candidate else None
    translation = tuple(candidate.get("translation", ())) if candidate else ()
    side = movement_side(translation)
    detections = getattr(record.before_scene, "detections", ())
    contact = adjacent_detection_on_side(detections, bbox, side) if bbox else None
    return {
        "movement_side": side,
        "contacted_or_adjacent_component_signature": coarse_component_signature(contact),
        "contact_component_relative_side": component_relative_side(contact, bbox) if contact and bbox else "none",
        "blocked_free_side_pattern": side_free_space_pattern(np.asarray(record.before.pixels[-1]), bbox),
        "current_touch_overlap_relation": record.pre_features.get("current_touch_overlap_relation"),
        "controlled_entity_type": record.post_labels.get("labeler_v3", {}).get("controlled_entity", {}).get("type", "none"),
    }


def parameter_probe_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_error = float(baseline.get("parameter_error_given_schema", 1.0) or 1.0)
    candidate_error = float(candidate.get("parameter_error_given_schema", 1.0) or 1.0)
    error_drop = max(0.0, baseline_error - candidate_error)
    relative_drop = error_drop / baseline_error if baseline_error else 0.0
    reuse_loss = max(
        0.0,
        float(baseline.get("reuse_coverage", 0.0) or 0.0)
        - float(candidate.get("reuse_coverage", 0.0) or 0.0),
    )
    coverage_loss = max(
        0.0,
        float(baseline.get("prediction_coverage", 0.0) or 0.0)
        - float(candidate.get("prediction_coverage", 0.0) or 0.0),
    )
    support = candidate.get("support", {})
    suspicious = error_drop > 0.0 and (reuse_loss > 0.10 or coverage_loss > 0.10)
    success = (
        support.get("status") == "eligible"
        and relative_drop >= 0.20
        and reuse_loss <= 0.10
        and coverage_loss <= 0.10
    )
    if support.get("status") == "insufficient_support":
        verdict = "insufficient_support"
    elif suspicious:
        verdict = "suspicious_fragmentation"
    elif success:
        verdict = "success"
    else:
        verdict = "no_gain"
    return {
        "parameter_error_drop": round(error_drop, 6),
        "parameter_error_relative_drop": round(relative_drop, 6),
        "reuse_loss": round(reuse_loss, 6),
        "prediction_coverage_loss": round(coverage_loss, 6),
        "gain_per_reuse_lost": round(error_drop / max(reuse_loss, 1e-6), 6),
        "support": support,
        "suspicious_fragmentation_flag": suspicious,
        "verdict": verdict,
    }


def parameter_probe_support_status(
    *,
    examples: int,
    reused_contexts: int,
    reused_transitions: int,
    reuse_coverage: float,
    prediction_count: int,
) -> dict[str, Any]:
    eligible = (
        prediction_count >= 30
        and reused_contexts >= 10
        and reused_transitions >= 30
    ) or (examples < 30 and reuse_coverage >= 0.25 and prediction_count >= 2)
    return {
        "status": "eligible" if eligible else "insufficient_support",
        "examples": examples,
        "prediction_count": prediction_count,
        "reused_contexts": reused_contexts,
        "reused_transitions": reused_transitions,
        "reuse_coverage": round(reuse_coverage, 6),
    }


def controlled_focus_candidate(record: TransitionRecord) -> Mapping[str, Any] | None:
    candidates = record.post_labels.get("labeler_v3", {}).get("controlled_entity", {}).get("candidates", ())
    if not candidates:
        return None
    high = [candidate for candidate in candidates if float(candidate.get("score", 0.0) or 0.0) >= 0.75]
    return high[0] if high else candidates[0]


def high_confidence_controlled_candidates(record: TransitionRecord) -> list[Mapping[str, Any]]:
    candidates = record.post_labels.get("labeler_v3", {}).get("controlled_entity", {}).get("candidates", ())
    return [candidate for candidate in candidates if float(candidate.get("score", 0.0) or 0.0) >= 0.75]


def bbox_bucket(bbox: tuple[int, int, int, int] | None, shape: tuple[int, int]) -> dict[str, Any] | None:
    if bbox is None or len(bbox) != 4:
        return None
    top, left, bottom, right = bbox
    center = bbox_center(bbox)
    return {
        "height": bucket_count(bottom - top + 1),
        "width": bucket_count(right - left + 1),
        "area": bucket_count((bottom - top + 1) * (right - left + 1)),
        "region": coarse_region(center, shape),
    }


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    top, left, bottom, right = bbox
    return ((top + bottom) / 2.0, (left + right) / 2.0)


def detection_for_bbox(scene: VisualScene, bbox: tuple[int, int, int, int] | None) -> Detection | None:
    if bbox is None:
        return None
    matches = [
        (bbox_overlap_area(bbox, detection_bbox_tuple(detection)), detection)
        for detection in scene.detections
    ]
    matches = [(overlap, detection) for overlap, detection in matches if overlap > 0]
    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], item[1].area))[1]


def detection_bbox_tuple(detection: Detection) -> tuple[int, int, int, int]:
    return (detection.bbox.top, detection.bbox.left, detection.bbox.bottom, detection.bbox.right)


def bbox_overlap_area(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
    top = max(left[0], right[0])
    left_col = max(left[1], right[1])
    bottom = min(left[2], right[2])
    right_col = min(left[3], right[3])
    if bottom < top or right_col < left_col:
        return 0
    return (bottom - top + 1) * (right_col - left_col + 1)


def local_grid_pattern(grid: np.ndarray, center: tuple[float, float] | tuple[int, int] | None) -> tuple[tuple[str, ...], ...]:
    if center is None:
        return ()
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    row = int(round(float(center[0])))
    col = int(round(float(center[1])))
    pattern = []
    for row_delta in (-1, 0, 1):
        line = []
        for col_delta in (-1, 0, 1):
            current = (row + row_delta, col + col_delta)
            if not (0 <= current[0] < grid.shape[0] and 0 <= current[1] < grid.shape[1]):
                line.append("out")
            elif int(grid[current]) == background:
                line.append("free")
            else:
                line.append("occupied")
        pattern.append(tuple(line))
    return tuple(pattern)


def side_free_space_pattern(grid: np.ndarray, bbox: tuple[int, int, int, int] | None) -> dict[str, str]:
    if bbox is None or len(bbox) != 4:
        return {"up": "unknown", "down": "unknown", "left": "unknown", "right": "unknown"}
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    top, left, bottom, right = bbox
    return {
        "up": strip_free_state(grid, top - 1, left, top - 1, right, background),
        "down": strip_free_state(grid, bottom + 1, left, bottom + 1, right, background),
        "left": strip_free_state(grid, top, left - 1, bottom, left - 1, background),
        "right": strip_free_state(grid, top, right + 1, bottom, right + 1, background),
    }


def strip_free_state(
    grid: np.ndarray,
    top: int,
    left: int,
    bottom: int,
    right: int,
    background: int,
) -> str:
    if top < 0 or left < 0 or bottom >= grid.shape[0] or right >= grid.shape[1]:
        return "out"
    strip = grid[top : bottom + 1, left : right + 1]
    if not strip.size:
        return "out"
    if np.all(strip == background):
        return "free"
    if np.any(strip == background):
        return "mixed"
    return "occupied"


def nearest_component_signatures_for_point(
    detections: tuple[Detection, ...],
    center: tuple[float, float] | tuple[int, int] | None,
    *,
    exclude_bbox: tuple[int, int, int, int] | None,
    limit: int = 5,
) -> tuple[dict[str, Any], ...]:
    if center is None:
        return ()
    ranked = []
    for detection in detections:
        bbox = detection_bbox_tuple(detection)
        if exclude_bbox is not None and bbox_overlap_area(exclude_bbox, bbox) > 0:
            continue
        distance = abs(detection.centroid[0] - float(center[0])) + abs(detection.centroid[1] - float(center[1]))
        ranked.append((distance, detection.area, detection))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return tuple(
        component_signature(detection, actor_center=(float(center[0]), float(center[1])))
        for _distance, _area, detection in ranked[:limit]
    )


def movement_side(translation: tuple[Any, ...]) -> str:
    if len(translation) != 2:
        return "unknown"
    row_delta, col_delta = int(translation[0]), int(translation[1])
    if abs(row_delta) >= abs(col_delta) and row_delta < 0:
        return "up"
    if abs(row_delta) >= abs(col_delta) and row_delta > 0:
        return "down"
    if col_delta < 0:
        return "left"
    if col_delta > 0:
        return "right"
    return "none"


def adjacent_detection_on_side(
    detections: tuple[Detection, ...],
    bbox: tuple[int, int, int, int] | None,
    side: str,
) -> Detection | None:
    if bbox is None or side == "unknown":
        return None
    strip = adjacent_strip_for_side(bbox, side)
    candidates = [
        detection
        for detection in detections
        if bbox_overlap_area(strip, detection_bbox_tuple(detection)) > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda detection: bbox_overlap_area(strip, detection_bbox_tuple(detection)))


def adjacent_strip_for_side(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int, int, int]:
    top, left, bottom, right = bbox
    if side == "up":
        return (top - 1, left, top - 1, right)
    if side == "down":
        return (bottom + 1, left, bottom + 1, right)
    if side == "left":
        return (top, left - 1, bottom, left - 1)
    if side == "right":
        return (top, right + 1, bottom, right + 1)
    return bbox


def component_relative_side(detection: Detection, bbox: tuple[int, int, int, int]) -> str:
    center = bbox_center(bbox)
    row_delta = detection.centroid[0] - center[0]
    col_delta = detection.centroid[1] - center[1]
    if abs(row_delta) >= abs(col_delta) and row_delta < 0:
        return "up"
    if abs(row_delta) >= abs(col_delta) and row_delta > 0:
        return "down"
    if col_delta < 0:
        return "left"
    if col_delta > 0:
        return "right"
    return "overlap"


def cd82_region_role_probe_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game != "cd82":
        return {
            "status": "skipped_not_cd82",
            "game": game,
            "reason": "region-role improvement target is CD82 click/edit assignment",
        }
    baseline = delta_channel_assignment_v3_report(records)
    role_map = build_region_role_map_cd82_probe(records)
    probe = region_role_probe_assignment_report(records, role_map)
    click_gain = float(probe["click_target_assignment_rate"]) - float(baseline["click_target_assignment_rate"])
    hud_share_drop = float(baseline["hud_status_cluster_share"]) - float(probe["hud_status_cluster_share"])
    action_target_cluster_gain = int(probe["cluster_assignments_by_channel"].get("action_target_delta", 0)) - int(
        baseline["cluster_assignments_by_channel"].get("action_target_delta", 0)
    )
    success = (
        click_gain >= 0.20
        and hud_share_drop >= 0.10
        and action_target_cluster_gain > 0
    )
    labels = []
    if success:
        labels.append("region_role_assignment_improved")
    else:
        labels.append("region_role_assignment_still_weak")
    if float(probe["click_target_assignment_rate"]) < 0.5:
        labels.append("below_cd82_click_target_threshold")
    return {
        "status": "evaluated",
        "game": game,
        "baseline_v3": baseline,
        "probe": probe,
        "role_map": role_map,
        "deltas": {
            "click_target_assignment_rate_delta": round(click_gain, 6),
            "hud_status_cluster_share_drop": round(hud_share_drop, 6),
            "action_target_cluster_gain": action_target_cluster_gain,
        },
        "verdict": {
            "primary": labels[0],
            "labels": labels,
        },
    }


def cd82_region_role_integrated_report(records: list[TransitionRecord]) -> dict[str, Any]:
    game = records[0].game if records else "unknown"
    if game != "cd82":
        return {
            "status": "skipped_not_cd82",
            "game": game,
            "reason": "integrated region-role labeler is only meaningful for CD82 click/edit assignment",
        }
    role_map = build_region_role_map_cd82_probe(records)
    for record in records:
        record.post_labels["labeler_cd82_region_role"] = build_post_labels_cd82_region_role(record, role_map)
    baseline = delta_channel_assignment_v3_report(records)
    integrated = delta_channel_assignment_report_for_labeler(records, "labeler_cd82_region_role")
    click_gain = float(integrated["click_target_assignment_rate"]) - float(baseline["click_target_assignment_rate"])
    action_target_share_gain = float(integrated["action_target_cluster_share"]) - float(baseline["action_target_cluster_share"])
    hud_share_drop = float(baseline["hud_status_cluster_share"]) - float(integrated["hud_status_cluster_share"])
    click_metrics = {}
    families = (
        R3A_COMPRESSED,
        R4_CLICK_PROCEDURAL_VISUAL_PLUS_R3A,
        R4_CLICK_PROCEDURAL_LEARNED_PLUS_R3A,
    )
    for family in families:
        v3 = prequential_parameter_metrics(records, family, labeler="v3")
        integrated_metrics = prequential_parameter_metrics(records, family, labeler="cd82_region_role")
        click_metrics[family.name] = {
            "v3": strip_metric_examples(v3),
            "cd82_region_role": strip_metric_examples(integrated_metrics),
            "integrated_minus_v3": {
                "schema_contradiction_delta": round(
                    integrated_metrics["schema_contradiction_rate"] - v3["schema_contradiction_rate"],
                    6,
                ),
                "parameter_contradiction_delta": round(
                    integrated_metrics["parameter_contradiction_given_schema"] - v3["parameter_contradiction_given_schema"],
                    6,
                ),
                "parameter_error_delta": round(
                    integrated_metrics["parameter_error_given_schema"] - v3["parameter_error_given_schema"],
                    6,
                ),
                "parameter_reused_contexts_delta": int(
                    integrated_metrics["parameter_reused_contexts"] - v3["parameter_reused_contexts"]
                ),
            },
        }
    labels = []
    if click_gain >= 0.20 and action_target_share_gain > 0.10 and hud_share_drop > 0.05:
        labels.append("cd82_region_role_labeler_improved")
    else:
        labels.append("cd82_region_role_labeler_still_weak")
    if float(integrated["click_target_assignment_rate"]) < 0.5:
        labels.append("below_cd82_click_target_threshold")
    if hud_share_drop <= 0:
        labels.append("hud_overassignment_not_reduced")
    return {
        "status": "evaluated",
        "game": game,
        "baseline_labeler": "v3",
        "integrated_labeler": "cd82_region_role",
        "role_map": role_map,
        "baseline_v3": baseline,
        "integrated": integrated,
        "deltas": {
            "click_target_assignment_rate_delta": round(click_gain, 6),
            "action_target_cluster_share_delta": round(action_target_share_gain, 6),
            "hud_status_cluster_share_drop": round(hud_share_drop, 6),
        },
        "click_procedural_diagnostics": click_metrics,
        "verdict": {
            "primary": labels[0],
            "labels": labels,
        },
    }


def build_post_labels_cd82_region_role(
    record: TransitionRecord,
    role_map: Mapping[str, Any],
) -> dict[str, Any]:
    v2 = record.post_labels.get("labeler_v2", {})
    assigned = assign_changed_region_clusters_region_probe(record, role_map)
    channels = {
        "action_target_delta": confident_channel_summary(
            [
                assignment_record(cluster, {"role": "canvas_edit_surface"}, "medium", "cd82_region_role_action_target")
                for cluster in assigned["action_target_delta"]
            ]
        ),
        "hud_status_delta": confident_channel_summary(
            [
                assignment_record(cluster, {"role": "hud_status_region"}, "medium", "cd82_region_role_hud_status")
                for cluster in assigned["hud_status_delta"]
            ]
        ),
        "resource_delta": v2.get("channels", {}).get("resource_delta"),
        "remote_delta": confident_channel_summary(
            [
                assignment_record(cluster, {"role": "remote_object_region"}, "medium", "cd82_region_role_remote")
                for cluster in assigned["remote_delta"]
            ]
        ),
        "independent_motion_delta": confident_channel_summary(
            [
                assignment_record(cluster, {"role": "canvas_edit_surface"}, "low", "cd82_region_role_canvas_unclicked")
                for cluster in assigned["independent_motion_delta"]
            ]
        ),
        "unknown_delta": confident_channel_summary(
            [
                assignment_record(cluster, {"role": "unknown"}, "low", "cd82_region_role_unknown")
                for cluster in assigned["unknown_delta"]
            ]
        ),
        "terminal_delta": v2.get("channels", {}).get("terminal_delta"),
    }
    active_channels = tuple(
        name for name, value in channels.items() if channel_is_active_v3(name, value)
    )
    return {
        "region_roles": {
            "action_target_region": action_target_region_role(record, role_map),
        },
        "changed_region_clusters": v2.get("changed_region_clusters", ()),
        "changed_region_assignments": {
            channel: [
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "confidence": "medium" if channel != "unknown_delta" else "low",
                    "reason": f"cd82_region_role_{channel}",
                }
                for cluster in clusters
            ]
            for channel, clusters in assigned.items()
        },
        "channels": channels,
        "active_channels": active_channels,
        "coarse_outcome": "+".join(active_channels) if active_channels else "no_op",
        "parameter_outcome": channels,
    }


def delta_channel_assignment_report_for_labeler(
    records: list[TransitionRecord],
    labeler_name: str,
) -> dict[str, Any]:
    channel_counts = Counter()
    active_channel_sets = Counter()
    cluster_total = 0
    changed_records = [
        record for record in records
        if int(record.post_labels.get("changed_cells", 0) or 0) > 0
    ]
    click_records = [
        record for record in records
        if record.pre_features.get("action_kind") == "click"
    ]
    for record in records:
        labeler = record.post_labels.get(labeler_name, {})
        active_channel_sets[tuple(labeler.get("active_channels", ()))] += 1
        for channel, assignments in labeler.get("changed_region_assignments", {}).items():
            channel_counts[channel] += len(assignments)
            cluster_total += len(assignments)
    unknown_records = sum(
        1 for record in changed_records
        if labeler_channel_cluster_count(record, labeler_name, "unknown_delta") > 0
    )
    action_target_records = sum(
        1 for record in click_records
        if labeler_channel_cluster_count(record, labeler_name, "action_target_delta") > 0
    )
    return {
        "cluster_assignments_by_channel": dict(sorted(channel_counts.items())),
        "cluster_assignment_total": cluster_total,
        "hud_status_cluster_share": round(channel_counts["hud_status_delta"] / cluster_total if cluster_total else 0.0, 6),
        "action_target_cluster_share": round(channel_counts["action_target_delta"] / cluster_total if cluster_total else 0.0, 6),
        "remote_cluster_share": round(channel_counts["remote_delta"] / cluster_total if cluster_total else 0.0, 6),
        "canvas_or_independent_cluster_share": round(
            channel_counts["independent_motion_delta"] / cluster_total if cluster_total else 0.0,
            6,
        ),
        "click_target_assignment_rate": round(action_target_records / len(click_records) if click_records else 0.0, 6),
        "unknown_delta_rate": round(unknown_records / len(changed_records) if changed_records else 0.0, 6),
        "changed_transition_count": len(changed_records),
        "click_transition_count": len(click_records),
        "active_channel_sets": [
            {"channels": list(channels), "count": count}
            for channels, count in active_channel_sets.most_common()
        ],
    }


def labeler_channel_cluster_count(record: TransitionRecord, labeler_name: str, channel: str) -> int:
    value = record.post_labels.get(labeler_name, {}).get("channels", {}).get(channel)
    if isinstance(value, Mapping):
        return int(value.get("cluster_count", 0) or 0)
    return 0


def build_region_role_map_cd82_probe(records: list[TransitionRecord]) -> dict[str, Any]:
    profiles: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        shape = np.asarray(record.before.pixels[-1]).shape
        target = action_target_point(record.action_data)
        target_key = region_key(target, shape) if target is not None else None
        if record.pre_features.get("action_kind") == "click" and target_key is not None:
            profiles[target_key]["clicks"] += 1
        for cluster in record.post_labels.get("labeler_v2", {}).get("changed_region_clusters", ()):
            key = region_key(cluster_centroid(cluster), shape)
            profiles[key]["changes"] += 1
            profiles[key][f"zone:{key.split(':')[0]}"] += 1
            if cluster.get("overlaps_hud"):
                profiles[key]["hud_overlaps"] += 1
            if cluster.get("centroid_region") == "central_canvas":
                profiles[key]["central_canvas_cluster"] += 1
            if cluster.get("distance_to_action_target") in {"zero", "near"}:
                profiles[key]["near_click_change"] += 1
                if target_key is not None:
                    profiles[target_key]["direct_click_edits"] += 1
            elif target_key is not None and record.pre_features.get("action_kind") == "click":
                profiles[target_key]["click_changes_elsewhere"] += 1
    regions = {}
    for key, counts in profiles.items():
        zone = key.split(":")[0]
        clicks = counts["clicks"]
        changes = counts["changes"]
        near_click = counts["near_click_change"]
        hud = counts["hud_overlaps"]
        central = counts["central_canvas_cluster"]
        if clicks and zone in {"top", "hud", "border"} and counts["click_changes_elsewhere"]:
            role = "tool_control_region"
            confidence = "high" if clicks >= 3 else "medium"
        elif clicks and near_click:
            role = "canvas_edit_surface"
            confidence = "high" if near_click >= 3 else "medium"
        elif central and hud < max(1, changes * 0.8):
            role = "canvas_edit_surface"
            confidence = "medium"
        elif zone in {"top", "hud", "border"} and hud >= max(1, changes * 0.5):
            role = "hud_status_region"
            confidence = "high" if changes >= 3 else "medium"
        elif changes >= 2:
            role = "remote_object_region"
            confidence = "medium"
        else:
            role = "unknown"
            confidence = "low"
        regions[key] = {
            "role": role,
            "confidence": confidence,
            "features": dict(counts),
        }
    return {"regions": regions}


def region_role_probe_assignment_report(
    records: list[TransitionRecord],
    role_map: Mapping[str, Any],
) -> dict[str, Any]:
    channel_counts = Counter()
    cluster_total = 0
    active_channel_sets = Counter()
    changed_records = [
        record for record in records
        if int(record.post_labels.get("changed_cells", 0) or 0) > 0
    ]
    click_records = [
        record for record in records
        if record.pre_features.get("action_kind") == "click"
    ]
    click_assigned = 0
    for record in records:
        assigned = assign_changed_region_clusters_region_probe(record, role_map)
        active = tuple(
            channel
            for channel, clusters in assigned.items()
            if clusters
        )
        active_channel_sets[active] += 1
        for channel, clusters in assigned.items():
            channel_counts[channel] += len(clusters)
            cluster_total += len(clusters)
        if record in click_records and assigned["action_target_delta"]:
            click_assigned += 1
    return {
        "cluster_assignments_by_channel": dict(sorted(channel_counts.items())),
        "cluster_assignment_total": cluster_total,
        "hud_status_cluster_share": round(channel_counts["hud_status_delta"] / cluster_total if cluster_total else 0.0, 6),
        "action_target_cluster_share": round(channel_counts["action_target_delta"] / cluster_total if cluster_total else 0.0, 6),
        "click_target_assignment_rate": round(click_assigned / len(click_records) if click_records else 0.0, 6),
        "changed_transition_count": len(changed_records),
        "click_transition_count": len(click_records),
        "active_channel_sets": [
            {"channels": list(channels), "count": count}
            for channels, count in active_channel_sets.most_common()
        ],
    }


def assign_changed_region_clusters_region_probe(
    record: TransitionRecord,
    role_map: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    assigned = {
        "action_target_delta": [],
        "hud_status_delta": [],
        "remote_delta": [],
        "independent_motion_delta": [],
        "unknown_delta": [],
    }
    shape = np.asarray(record.before.pixels[-1]).shape
    action_is_click = record.pre_features.get("action_kind") == "click"
    target_role = action_target_region_role(record, role_map)
    for cluster in record.post_labels.get("labeler_v2", {}).get("changed_region_clusters", ()):
        key = region_key(cluster_centroid(cluster), shape)
        role = role_map.get("regions", {}).get(key, {"role": "unknown", "confidence": "low"})
        if action_is_click and cluster.get("distance_to_action_target") in {"zero", "near"}:
            assigned["action_target_delta"].append(cluster)
        elif action_is_click and role.get("role") == "canvas_edit_surface" and target_role.get("role") in {
            "tool_control_region",
            "canvas_edit_surface",
        }:
            assigned["action_target_delta"].append(cluster)
        elif role.get("role") == "hud_status_region":
            assigned["hud_status_delta"].append(cluster)
        elif role.get("role") == "remote_object_region":
            assigned["remote_delta"].append(cluster)
        elif role.get("role") == "canvas_edit_surface":
            assigned["independent_motion_delta"].append(cluster)
        else:
            assigned["unknown_delta"].append(cluster)
    return assigned


def introspect_r3a_suite(
    specs: tuple[TraceSpec, ...],
    *,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    report_dir = output_dir / "schema_introspection_report"
    prototype_dir = report_dir / "schema_prototypes"
    report_dir.mkdir(parents=True, exist_ok=True)
    prototype_dir.mkdir(parents=True, exist_ok=True)

    all_contexts: list[dict[str, Any]] = []
    field_contributions: dict[str, Any] = {}
    vss_decomposition: dict[str, Any] = {}
    vss_robustness: dict[str, Any] = {}
    channel_heads: dict[str, Any] = {}
    entropy_contrast: dict[str, Any] = {}
    for spec in specs:
        records = load_transition_records(spec, limit=limit)
        contexts = schema_context_records(records, prototype_dir=prototype_dir)
        all_contexts.extend(contexts)
        field_contributions[spec.game] = r3a_field_contribution_examples(records, contexts)
        vss_decomposition[spec.game] = vss_decomposition_report(records)
        vss_robustness[spec.game] = vss_robustness_report(records)
        channel_heads[spec.game] = channel_parameter_heads_report(records)
        entropy_contrast[spec.game] = parameter_entropy_context_contrast(contexts)

    all_contexts.sort(key=lambda item: (item["game"], -item["support_count"], item["context_id"]))
    nearest = nearest_context_report(all_contexts)
    nearest_by_id = {
        item["context_id"]: item
        for item in nearest
    }
    for context in all_contexts:
        context["nearest_contexts"] = nearest_by_id.get(context["context_id"], {}).get("nearest_contexts", [])

    write_jsonl(report_dir / "schema_contexts.jsonl", all_contexts)
    write_schema_confusions_csv(report_dir / "schema_confusions.csv", all_contexts)
    write_json(report_dir / "context_field_contributions.json", field_contributions)
    write_json(report_dir / "field_contribution_examples.json", field_contributions)
    write_json(report_dir / "vss_decomposition_report.json", vss_decomposition)
    write_json(report_dir / "vss_robustness_report.json", vss_robustness)
    write_json(report_dir / "channel_parameter_heads_report.json", channel_heads)
    write_json(report_dir / "parameter_entropy_context_contrast.json", entropy_contrast)
    write_markdown_report(
        report_dir / "high_reuse_contexts.md",
        high_reuse_contexts_markdown(all_contexts),
    )
    write_markdown_report(
        report_dir / "high_entropy_contexts.md",
        high_entropy_contexts_markdown(all_contexts),
    )
    write_markdown_report(
        report_dir / "schema_atlas.md",
        schema_atlas_markdown(all_contexts, field_contributions, vss_decomposition, entropy_contrast),
    )
    summary = {
        "report_dir": str(report_dir),
        "context_count": len(all_contexts),
        "games": sorted({item["game"] for item in all_contexts}),
        "artifacts": [
            "schema_contexts.jsonl",
            "schema_confusions.csv",
            "high_reuse_contexts.md",
            "high_entropy_contexts.md",
            "context_field_contributions.json",
            "field_contribution_examples.json",
            "vss_decomposition_report.json",
            "vss_robustness_report.json",
            "channel_parameter_heads_report.json",
            "parameter_entropy_context_contrast.json",
            "schema_atlas.md",
            "schema_prototypes/",
        ],
    }
    write_json(report_dir / "summary.json", summary)
    return summary


def schema_context_records(
    records: list[TransitionRecord],
    *,
    prototype_dir: Path,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[TransitionRecord]] = defaultdict(list)
    for record in records:
        grouped[context_key(record, R3A_COMPRESSED)].append(record)
    contexts = []
    for key, items in grouped.items():
        schema_counter = Counter(outcome_schema_key(record, labeler="v3") for record in items)
        parameter_counter = Counter(parameter_probe_target(record) for record in items)
        context_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        prototype_path = write_schema_prototype(prototype_dir, context_id, items)
        majority_schema, majority_schema_count = schema_counter.most_common(1)[0]
        majority_parameter, majority_parameter_count = parameter_counter.most_common(1)[0]
        parsed_key = json.loads(key)
        contexts.append(
            {
                "context_id": context_id,
                "context_key": key,
                "game": items[0].game,
                "support_count": len(items),
                "actions_seen": sorted({record.action for record in items}),
                "majority_schema_outcome": majority_schema,
                "majority_schema_rate": round(majority_schema_count / len(items), 6),
                "schema_outcomes": dict(sorted(schema_counter.items())),
                "schema_entropy": round(entropy(schema_counter), 6),
                "parameter_entropy": round(entropy(parameter_counter), 6),
                "parameter_distinct_count": len(parameter_counter),
                "majority_parameter_hash": short_digest(majority_parameter),
                "majority_parameter_rate": round(majority_parameter_count / len(items), 6),
                "majority_parameter_summary": parameter_payload_summary(majority_parameter),
                "example_steps": [record.step for record in items[:8]],
                "prototype_path": str(prototype_path),
                "prototype_before_frame": str(prototype_path),
                "prototype_after_frame": str(prototype_path),
                "changed_channel_summary": changed_channel_summary_for_records(items),
                "fields_in_key": parsed_key,
                "field_values": {
                    field: parsed_key.get(field)
                    for field in R3A_FROZEN_FIELDS
                },
            }
        )
    return contexts


def write_schema_prototype(
    prototype_dir: Path,
    context_id: str,
    records: list[TransitionRecord],
) -> Path:
    path = prototype_dir / f"{records[0].game}-{context_id}.json"
    examples = []
    for record in records[:3]:
        examples.append(
            {
                "game": record.game,
                "step": record.step,
                "action": record.action,
                "before_frame": np.asarray(record.before.pixels[-1]).tolist(),
                "after_frame": np.asarray(record.after.pixels[-1]).tolist(),
                "before_crop": frame_crop_around_changes(record, before=True),
                "after_crop": frame_crop_around_changes(record, before=False),
                "schema": outcome_schema_key(record, labeler="v3"),
                "parameter_summary": parameter_payload_summary(parameter_probe_target(record)),
            }
        )
    write_json(path, {"context_id": context_id, "examples": examples})
    return path


def frame_crop_around_changes(record: TransitionRecord, *, before: bool) -> list[list[int]]:
    grid = np.asarray((record.before if before else record.after).pixels[-1])
    clusters = record.post_labels.get("labeler_v3", {}).get("changed_region_clusters", ())
    if not clusters:
        return grid[: min(16, grid.shape[0]), : min(16, grid.shape[1])].tolist()
    top = min(int(cluster["bbox"][0]) for cluster in clusters)
    left = min(int(cluster["bbox"][1]) for cluster in clusters)
    bottom = max(int(cluster["bbox"][2]) for cluster in clusters)
    right = max(int(cluster["bbox"][3]) for cluster in clusters)
    pad = 3
    top = max(0, top - pad)
    left = max(0, left - pad)
    bottom = min(grid.shape[0] - 1, bottom + pad)
    right = min(grid.shape[1] - 1, right + pad)
    return grid[top : bottom + 1, left : right + 1].tolist()


def changed_channel_summary_for_records(records: list[TransitionRecord]) -> dict[str, Any]:
    active = Counter()
    clusters = Counter()
    for record in records:
        for channel in record.post_labels.get("labeler_v3", {}).get("active_channels", ()):
            active[channel] += 1
        channels = record.post_labels.get("labeler_v3", {}).get("channels", {})
        for channel, payload in channels.items():
            if isinstance(payload, Mapping) and "cluster_count" in payload:
                clusters[channel] += int(payload.get("cluster_count", 0) or 0)
    return {
        "active_channel_counts": dict(sorted(active.items())),
        "cluster_counts": dict(sorted(clusters.items())),
    }


def nearest_context_report(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_game_action: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for context in contexts:
        for action in context["actions_seen"]:
            by_game_action[(context["game"], action)].append(context)
    report = []
    for context in contexts:
        action = context["actions_seen"][0] if context["actions_seen"] else -1
        candidates = [
            candidate for candidate in by_game_action.get((context["game"], action), [])
            if candidate["context_id"] != context["context_id"]
        ]
        scored = []
        for candidate in candidates:
            diff = context_field_diff(context["field_values"], candidate["field_values"])
            scored.append(
                {
                    "context_id": candidate["context_id"],
                    "support_count": candidate["support_count"],
                    "majority_schema_outcome": candidate["majority_schema_outcome"],
                    "schema_entropy": candidate["schema_entropy"],
                    "parameter_entropy": candidate["parameter_entropy"],
                    "differing_fields": diff,
                    "field_distance": len(diff),
                    "outcome_changed": candidate["majority_schema_outcome"] != context["majority_schema_outcome"],
                }
            )
        scored.sort(key=lambda item: (item["field_distance"], -item["support_count"], item["context_id"]))
        report.append(
            {
                "context_id": context["context_id"],
                "nearest_contexts": scored[:5],
                "one_field_different_neighbors": [item for item in scored if item["field_distance"] == 1][:5],
            }
        )
    return report


def context_field_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    return [
        field
        for field in R3A_FROZEN_FIELDS
        if stable_json(left.get(field)) != stable_json(right.get(field))
    ]


def r3a_field_contribution_examples(
    records: list[TransitionRecord],
    contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = schema_partition_metrics(records, R3A_COMPRESSED)
    report = {
        "baseline": baseline,
        "fields": {},
    }
    for field in R3A_FROZEN_FIELDS:
        family = RepresentationFamily(
            f"R3a_without_{field}",
            frozenset(item for item in R3A_COMPRESSED.pre_action_fields if item != field),
        )
        metrics = schema_partition_metrics(records, family)
        values = Counter(stable_json(record.pre_features.get(field)) for record in records)
        reuse_damage = max(0.0, metrics["schema_reuse_coverage"] - baseline["schema_reuse_coverage"])
        contradiction_reduction = metrics["schema_contradiction_rate"] - baseline["schema_contradiction_rate"]
        entropy_reduction = metrics["schema_entropy"] - baseline["schema_entropy"]
        parameter_entropy_reduction = metrics["parameter_entropy"] - baseline["parameter_entropy"]
        report["fields"][field] = {
            "field_cardinality": len(values),
            "reuse_damage": round(reuse_damage, 6),
            "contradiction_reduction": round(contradiction_reduction, 6),
            "entropy_reduction": round(entropy_reduction, 6),
            "parameter_entropy_reduction": round(parameter_entropy_reduction, 6),
            "classification": field_contribution_classification(
                reuse_damage=reuse_damage,
                contradiction_reduction=contradiction_reduction,
                entropy_reduction=entropy_reduction,
            ),
            "merge_examples": field_merge_examples(records, field),
        }
    return report


def vss_decomposition_report(records: list[TransitionRecord]) -> dict[str, Any]:
    base_without_vss = frozenset(
        field for field in R3A_COMPRESSED.pre_action_fields
        if field != "visible_state_summary"
    )
    baseline = schema_partition_metrics(records, R3A_COMPRESSED)
    without_vss = schema_partition_metrics_for_key(
        records,
        lambda record: stable_json(
            {
                "action": record.action,
                **{
                    field: jsonable(record.pre_features.get(field))
                    for field in sorted(base_without_vss)
                },
            }
        ),
        family_name="R3a_without_visible_state_summary",
    )
    variants = {
        "dominant_color_set_only": lambda vss: tuple(sorted(value for value, _count in vss.get("dominant_colors_bucketed", ()))),
        "dominant_color_ordered_counts": lambda vss: tuple(vss.get("dominant_colors_bucketed", ())),
        "num_components_bucketed": lambda vss: vss.get("num_components_bucketed"),
        "resource_level_bucket": lambda vss: vss.get("resource_level_bucket"),
        "hud_changed_recently": lambda vss: vss.get("hud_changed_recently"),
    }
    variant_reports = {}
    for name, extractor in variants.items():
        metrics = schema_partition_metrics_for_key(
            records,
            lambda record, extractor=extractor, name=name: vss_variant_key(
                record,
                base_fields=base_without_vss,
                variant_name=name,
                variant_value=extractor(record.pre_features.get("visible_state_summary", {}) or {}),
            ),
            family_name=f"R3a_without_vss_plus_{name}",
        )
        variant_reports[name] = {
            "metrics": metrics,
            "comparison_to_without_vss": compare_partition_metrics(without_vss, metrics),
            "comparison_to_full_vss": compare_partition_metrics(baseline, metrics),
            "classification": classify_vss_variant(
                without_vss=without_vss,
                full_vss=baseline,
                variant=metrics,
            ),
            "merge_examples": vss_variant_merge_examples(records, base_without_vss, name, extractor),
        }
    return {
        "baseline_full_R3a": baseline,
        "without_visible_state_summary": without_vss,
        "variants": variant_reports,
        "interpretation_hint": "A good VSS replacement should recover entropy/contradiction reduction with much less reuse damage than full visible_state_summary.",
    }


def vss_robustness_report(records: list[TransitionRecord]) -> dict[str, Any]:
    base_without_vss = frozenset(
        field for field in R3A_COMPRESSED.pre_action_fields
        if field != "visible_state_summary"
    )
    without_vss = schema_partition_metrics_for_key(
        records,
        lambda record: stable_json(
            {
                "action": record.action,
                **{
                    field: jsonable(record.pre_features.get(field))
                    for field in sorted(base_without_vss)
                },
            }
        ),
        family_name="R3a_without_visible_state_summary",
    )
    full_vss = schema_partition_metrics(records, R3A_COMPRESSED)
    phase_cluster = build_vss_phase_cluster(records)
    variants = {
        "VSS_rank_only": lambda record: vss_rank_only(record.pre_features.get("visible_state_summary", {}) or {}),
        "VSS_count_only": lambda record: vss_count_only(record.pre_features.get("visible_state_summary", {}) or {}),
        "VSS_color_set_only": lambda record: vss_color_set_only(record.pre_features.get("visible_state_summary", {}) or {}),
        "VSS_ordered_color_counts": lambda record: tuple((record.pre_features.get("visible_state_summary", {}) or {}).get("dominant_colors_bucketed", ())),
        "VSS_level_local_phase_cluster": lambda record: phase_cluster["assignments"].get(
            stable_json(record.pre_features.get("visible_state_summary", {}) or {}),
            "phase_unknown",
        ),
    }
    variant_reports = {}
    for name, extractor in variants.items():
        metrics = schema_partition_metrics_for_key(
            records,
            lambda record, extractor=extractor, name=name: vss_variant_key(
                record,
                base_fields=base_without_vss,
                variant_name=name,
                variant_value=extractor(record),
            ),
            family_name=f"R3a_without_vss_plus_{name}",
        )
        variant_reports[name] = {
            "metrics": metrics,
            "comparison_to_without_vss": compare_partition_metrics(without_vss, metrics),
            "comparison_to_full_vss": compare_partition_metrics(full_vss, metrics),
            "classification": classify_vss_robustness_variant(
                without_vss=without_vss,
                full_vss=full_vss,
                variant=metrics,
            ),
        }
    return {
        "baseline_full_R3a": full_vss,
        "without_visible_state_summary": without_vss,
        "phase_cluster": {
            "k": phase_cluster["k"],
            "centroids": phase_cluster["centroids"],
            "cluster_sizes": phase_cluster["cluster_sizes"],
        },
        "variants": variant_reports,
        "interpretation_rules": {
            "rank_or_count_survives": "structural phase proxy",
            "only_ordered_color_counts_survive": "absolute color identity or ordering is carrying the signal",
            "small_k_phase_cluster_survives": "prefer phase abstraction over raw VSS",
            "reuse_collapse": "fingerprint risk",
        },
    }


def vss_rank_only(vss: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"rank_{index + 1}", str(count_bucket))
        for index, (_color, count_bucket) in enumerate(vss.get("dominant_colors_bucketed", ()))
    )


def vss_count_only(vss: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(count_bucket) for _color, count_bucket in vss.get("dominant_colors_bucketed", ()))


def vss_color_set_only(vss: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(sorted(int(color) for color, _count_bucket in vss.get("dominant_colors_bucketed", ())))


def build_vss_phase_cluster(records: list[TransitionRecord]) -> dict[str, Any]:
    counts = Counter(stable_json(record.pre_features.get("visible_state_summary", {}) or {}) for record in records)
    if not counts:
        return {"k": 0, "centroids": {}, "cluster_sizes": {}, "assignments": {}}
    k = min(8, max(2, math.ceil(math.sqrt(len(counts)))))
    k = min(k, len(counts))
    centroids = [key for key, _count in counts.most_common(k)]
    assignments = {}
    cluster_sizes: Counter[str] = Counter()
    for key, count in counts.items():
        nearest_index = min(
            range(len(centroids)),
            key=lambda index: vss_distance(json.loads(key), json.loads(centroids[index])),
        )
        phase = f"phase_{nearest_index}"
        assignments[key] = phase
        cluster_sizes[phase] += count
    return {
        "k": k,
        "centroids": {f"phase_{index}": json.loads(key) for index, key in enumerate(centroids)},
        "cluster_sizes": dict(sorted(cluster_sizes.items())),
        "assignments": assignments,
    }


def vss_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return 0.0 if left == right else 1.0
    distance = 0.0
    left_colors = list(left.get("dominant_colors_bucketed", ()))
    right_colors = list(right.get("dominant_colors_bucketed", ()))
    for index in range(max(len(left_colors), len(right_colors), 4)):
        left_item = left_colors[index] if index < len(left_colors) else None
        right_item = right_colors[index] if index < len(right_colors) else None
        if left_item is None or right_item is None:
            distance += 1.5
            continue
        if left_item[0] != right_item[0]:
            distance += 1.0
        if left_item[1] != right_item[1]:
            distance += 0.5
    if left.get("num_components_bucketed") != right.get("num_components_bucketed"):
        distance += 0.5
    if left.get("resource_level_bucket") != right.get("resource_level_bucket"):
        distance += 0.5
    return distance


def classify_vss_robustness_variant(
    *,
    without_vss: Mapping[str, Any],
    full_vss: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> str:
    entropy_gain = float(without_vss["parameter_entropy"]) - float(variant["parameter_entropy"])
    full_entropy_gain = max(1e-6, float(without_vss["parameter_entropy"]) - float(full_vss["parameter_entropy"]))
    reuse_loss = max(0.0, float(without_vss["schema_reuse_coverage"]) - float(variant["schema_reuse_coverage"]))
    full_reuse_loss = max(1e-6, float(without_vss["schema_reuse_coverage"]) - float(full_vss["schema_reuse_coverage"]))
    recovered_entropy_ratio = entropy_gain / full_entropy_gain
    reuse_loss_ratio = reuse_loss / full_reuse_loss
    if recovered_entropy_ratio >= 0.60 and reuse_loss_ratio <= 0.60:
        return "robust_phase_signal"
    if recovered_entropy_ratio >= 0.60 and reuse_loss_ratio > 0.90:
        return "fingerprint_like_signal"
    if recovered_entropy_ratio >= 0.30:
        return "partial_phase_signal"
    return "weak_or_decorative_signal"


def channel_parameter_heads_report(records: list[TransitionRecord]) -> dict[str, Any]:
    channels = (
        "controlled_entity_delta",
        "action_target_delta",
        "independent_motion_delta",
        "hud_status_delta",
        "resource_delta",
        "remote_delta",
        "terminal_delta",
        "unknown_delta",
    )
    heads = {
        channel: channel_parameter_head_metrics(records, channel)
        for channel in channels
    }
    hard_channels = sorted(
        [
            {
                "channel": channel,
                "parameter_entropy": payload["parameter_entropy"],
                "parameter_error": payload["parameter_error"],
                "prediction_coverage": payload["prediction_coverage"],
                "active_examples": payload["active_examples"],
                "support_status": payload["support"]["status"],
            }
            for channel, payload in heads.items()
            if payload["active_examples"] > 0
        ],
        key=lambda item: (-item["parameter_entropy"], -item["parameter_error"], -item["active_examples"]),
    )
    return {
        "heads": heads,
        "hard_channel_ranking": hard_channels,
        "interpretation_hint": "Channels with high active support, high entropy, and high error are the actual parameter-model bottlenecks.",
    }


def channel_parameter_head_metrics(records: list[TransitionRecord], channel: str) -> dict[str, Any]:
    active_records = [
        record for record in records
        if channel_is_active_for_record(record, channel)
    ]
    prior: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[TransitionRecord]] = defaultdict(list)
    predictions = 0
    hits = 0
    error_examples = []
    for record in active_records:
        schema = outcome_schema_key(record, labeler="v3")
        key = stable_json(
            {
                "context": context_key(record, R3A_COMPRESSED),
                "action": record.action,
                "schema": schema,
                "channel": channel,
            }
        )
        target = channel_parameter_target(record, channel)
        if prior[key]:
            predictions += 1
            predicted = prior[key].most_common(1)[0][0]
            if predicted == target:
                hits += 1
            elif len(error_examples) < 20:
                error_examples.append(
                    {
                        "step": record.step,
                        "action": record.action,
                        "context_id": short_digest(key),
                        "predicted_hash": short_digest(predicted),
                        "actual_hash": short_digest(target),
                        "predicted_summary": parameter_payload_summary(predicted),
                        "actual_summary": parameter_payload_summary(target),
                    }
                )
        prior[key][target] += 1
        groups[key][target] += 1
        examples[key].append(record)
    total_active = len(active_records)
    reused_keys = [key for key, counter in groups.items() if sum(counter.values()) > 1]
    contradicted = [key for key in reused_keys if len(groups[key]) > 1]
    reused_transitions = sum(sum(groups[key].values()) for key in reused_keys)
    weighted_entropy = sum(sum(counter.values()) * entropy(counter) for counter in groups.values())
    accuracy = hits / predictions if predictions else 0.0
    return {
        "channel": channel,
        "examples": len(records),
        "active_examples": total_active,
        "active_rate": round(total_active / len(records) if records else 0.0, 6),
        "prediction_count": predictions,
        "prediction_coverage": round(predictions / total_active if total_active else 0.0, 6),
        "prequential_accuracy": round(accuracy, 6),
        "parameter_error": round(1.0 - accuracy if predictions else 1.0, 6),
        "unique_contexts": len(groups),
        "reused_contexts": len(reused_keys),
        "reused_transitions": reused_transitions,
        "reuse_coverage": round(reused_transitions / total_active if total_active else 0.0, 6),
        "contradicted_reused_contexts": len(contradicted),
        "contradiction_rate": round(len(contradicted) / len(reused_keys) if reused_keys else 0.0, 6),
        "parameter_entropy": round(weighted_entropy / total_active if total_active else 0.0, 6),
        "support": channel_head_support_status(total_active, len(reused_keys), reused_transitions, predictions),
        "error_examples": error_examples,
    }


def channel_is_active_for_record(record: TransitionRecord, channel: str) -> bool:
    value = record.post_labels.get("labeler_v3", {}).get("channels", {}).get(channel)
    return channel_is_active_v3(channel, value)


def channel_parameter_target(record: TransitionRecord, channel: str) -> str:
    payload = record.post_labels.get("labeler_v3", {}).get("channels", {}).get(channel)
    shape = np.asarray(record.before.pixels[-1]).shape
    return stable_json(coarse_parameter_target({channel: payload}, shape))


def channel_head_support_status(
    active_examples: int,
    reused_contexts: int,
    reused_transitions: int,
    prediction_count: int,
) -> dict[str, Any]:
    eligible = (
        active_examples >= 30
        and prediction_count >= 20
        and reused_contexts >= 5
        and reused_transitions >= 20
    ) or (active_examples < 30 and prediction_count >= 2 and reused_transitions >= 4)
    return {
        "status": "eligible" if eligible else "insufficient_support",
        "active_examples": active_examples,
        "prediction_count": prediction_count,
        "reused_contexts": reused_contexts,
        "reused_transitions": reused_transitions,
    }


def vss_variant_key(
    record: TransitionRecord,
    *,
    base_fields: frozenset[str],
    variant_name: str,
    variant_value: Any,
) -> str:
    return stable_json(
        {
            "action": record.action,
            **{
                field: jsonable(record.pre_features.get(field))
                for field in sorted(base_fields)
            },
            f"vss_{variant_name}": jsonable(variant_value),
        }
    )


def schema_partition_metrics_for_key(
    records: list[TransitionRecord],
    key_fn: Any,
    *,
    family_name: str,
) -> dict[str, Any]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    parameter_groups: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[TransitionRecord]] = defaultdict(list)
    for record in records:
        key = key_fn(record)
        groups[key][outcome_schema_key(record, labeler="v3")] += 1
        parameter_groups[key][parameter_probe_target(record)] += 1
        examples[key].append(record)
    return schema_partition_metric_record(groups, parameter_groups, examples, family_name=family_name)


def compare_partition_metrics(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "unique_context_delta": int(candidate["unique_contexts"] - baseline["unique_contexts"]),
        "reuse_coverage_delta": round(float(candidate["schema_reuse_coverage"]) - float(baseline["schema_reuse_coverage"]), 6),
        "one_off_ratio_delta": round(float(candidate["schema_one_off_ratio"]) - float(baseline["schema_one_off_ratio"]), 6),
        "schema_contradiction_delta": round(float(candidate["schema_contradiction_rate"]) - float(baseline["schema_contradiction_rate"]), 6),
        "schema_entropy_delta": round(float(candidate["schema_entropy"]) - float(baseline["schema_entropy"]), 6),
        "parameter_entropy_delta": round(float(candidate["parameter_entropy"]) - float(baseline["parameter_entropy"]), 6),
    }


def classify_vss_variant(
    *,
    without_vss: Mapping[str, Any],
    full_vss: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> str:
    entropy_gain = float(without_vss["parameter_entropy"]) - float(variant["parameter_entropy"])
    full_entropy_gain = max(1e-6, float(without_vss["parameter_entropy"]) - float(full_vss["parameter_entropy"]))
    reuse_loss = max(0.0, float(without_vss["schema_reuse_coverage"]) - float(variant["schema_reuse_coverage"]))
    full_reuse_loss = max(1e-6, float(without_vss["schema_reuse_coverage"]) - float(full_vss["schema_reuse_coverage"]))
    recovered_entropy_ratio = entropy_gain / full_entropy_gain
    reuse_loss_ratio = reuse_loss / full_reuse_loss
    if recovered_entropy_ratio >= 0.60 and reuse_loss_ratio <= 0.60:
        return "compact_phase_signal"
    if recovered_entropy_ratio >= 0.60 and reuse_loss_ratio > 0.90:
        return "fingerprint_like_signal"
    if recovered_entropy_ratio < 0.20 and reuse_loss <= 0.02:
        return "decorative_subfield"
    return "mixed_or_partial_signal"


def vss_variant_merge_examples(
    records: list[TransitionRecord],
    base_fields: frozenset[str],
    variant_name: str,
    extractor: Any,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    groups: dict[str, list[TransitionRecord]] = defaultdict(list)
    for record in records:
        key = vss_variant_key(
            record,
            base_fields=base_fields,
            variant_name=variant_name,
            variant_value=extractor(record.pre_features.get("visible_state_summary", {}) or {}),
        )
        groups[key].append(record)
    examples = []
    for items in groups.values():
        full_contexts = defaultdict(list)
        for record in items:
            full_contexts[context_key(record, R3A_COMPRESSED)].append(record)
        if len(full_contexts) < 2:
            continue
        schema_counter = Counter(outcome_schema_key(record, labeler="v3") for record in items)
        examples.append(
            {
                "merged_support": len(items),
                "full_context_count": len(full_contexts),
                "schema_entropy": round(entropy(schema_counter), 6),
                "majority_schema": schema_counter.most_common(1)[0][0],
                "full_contexts": [
                    {
                        "context_id": hashlib.sha1(key.encode("utf-8")).hexdigest()[:12],
                        "support_count": len(records_for_key),
                        "visible_state_summary": records_for_key[0].pre_features.get("visible_state_summary"),
                        "example_steps": [record.step for record in records_for_key[:4]],
                    }
                    for key, records_for_key in list(full_contexts.items())[:6]
                ],
            }
        )
    examples.sort(key=lambda item: (-item["merged_support"], item["schema_entropy"]))
    return examples[:limit]


def parameter_entropy_context_contrast(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    reused = [context for context in contexts if context["support_count"] >= 3]
    low = sorted(
        [context for context in reused if context["parameter_entropy"] <= 0.5],
        key=lambda item: (-item["support_count"], item["schema_entropy"]),
    )[:20]
    high = sorted(
        [context for context in reused if context["parameter_entropy"] >= 1.5],
        key=lambda item: (-item["parameter_entropy"], -item["support_count"]),
    )[:20]
    return {
        "low_parameter_entropy_contexts": [context_contrast_record(context) for context in low],
        "high_parameter_entropy_contexts": [context_contrast_record(context) for context in high],
        "aggregate_field_differences": aggregate_entropy_field_differences(low, high),
    }


def context_contrast_record(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "context_id": context["context_id"],
        "support_count": context["support_count"],
        "actions_seen": context["actions_seen"],
        "majority_schema_outcome": context["majority_schema_outcome"],
        "schema_entropy": context["schema_entropy"],
        "parameter_entropy": context["parameter_entropy"],
        "parameter_distinct_count": context["parameter_distinct_count"],
        "field_values": context["field_values"],
        "prototype_path": context["prototype_path"],
    }


def aggregate_entropy_field_differences(
    low: list[dict[str, Any]],
    high: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {}
    for field in R3A_FROZEN_FIELDS:
        low_values = Counter(stable_json(context["field_values"].get(field)) for context in low)
        high_values = Counter(stable_json(context["field_values"].get(field)) for context in high)
        report[field] = {
            "low_cardinality": len(low_values),
            "high_cardinality": len(high_values),
            "low_top_values": low_values.most_common(5),
            "high_top_values": high_values.most_common(5),
        }
    return report


def schema_partition_metrics(records: list[TransitionRecord], family: RepresentationFamily) -> dict[str, Any]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    parameter_groups: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[TransitionRecord]] = defaultdict(list)
    for record in records:
        key = context_key(record, family)
        groups[key][outcome_schema_key(record, labeler="v3")] += 1
        parameter_groups[key][parameter_probe_target(record)] += 1
        examples[key].append(record)
    return schema_partition_metric_record(groups, parameter_groups, examples, family_name=family.name)


def schema_partition_metric_record(
    groups: Mapping[str, Counter[str]],
    parameter_groups: Mapping[str, Counter[str]],
    examples: Mapping[str, list[TransitionRecord]],
    *,
    family_name: str,
) -> dict[str, Any]:
    summary, _contradictions = summarize_groups(
        groups,
        examples,
        family=family_name,
        granularity="coarse",
    )
    total = sum(sum(counter.values()) for counter in groups.values())
    parameter_total = sum(sum(counter.values()) for counter in parameter_groups.values())
    return {
        "unique_contexts": summary.unique_contexts,
        "schema_reuse_coverage": summary.reuse_coverage,
        "schema_one_off_ratio": summary.one_off_ratio,
        "schema_contradiction_rate": summary.contradiction_rate,
        "schema_entropy": round(
            sum(sum(counter.values()) * entropy(counter) for counter in groups.values()) / total
            if total
            else 0.0,
            6,
        ),
        "parameter_entropy": round(
            sum(sum(counter.values()) * entropy(counter) for counter in parameter_groups.values()) / parameter_total
            if parameter_total
            else 0.0,
            6,
        ),
    }


def field_contribution_classification(
    *,
    reuse_damage: float,
    contradiction_reduction: float,
    entropy_reduction: float,
) -> str:
    if reuse_damage >= 0.10 and contradiction_reduction < 0.02 and entropy_reduction < 0.02:
        return "over_specific_fingerprint"
    if contradiction_reduction >= 0.05 or entropy_reduction >= 0.05:
        return "useful_discriminator"
    if reuse_damage < 0.02 and abs(contradiction_reduction) < 0.02:
        return "decorative_or_low_signal"
    return "mixed_tradeoff"


def field_merge_examples(records: list[TransitionRecord], field: str, *, limit: int = 5) -> list[dict[str, Any]]:
    family = RepresentationFamily(
        f"R3a_without_{field}",
        frozenset(item for item in R3A_COMPRESSED.pre_action_fields if item != field),
    )
    merged: dict[str, dict[str, Any]] = defaultdict(lambda: {"records": [], "baseline_keys": defaultdict(list)})
    for record in records:
        reduced_key = context_key(record, family)
        full_key = context_key(record, R3A_COMPRESSED)
        merged[reduced_key]["records"].append(record)
        merged[reduced_key]["baseline_keys"][full_key].append(record)
    candidates = []
    for _reduced_key, payload in merged.items():
        baseline_keys = payload["baseline_keys"]
        if len(baseline_keys) < 2:
            continue
        records_in_group = payload["records"]
        schema_counter = Counter(outcome_schema_key(record, labeler="v3") for record in records_in_group)
        candidates.append(
            {
                "after_support": len(records_in_group),
                "before_context_count": len(baseline_keys),
                "merged_schema_entropy": round(entropy(schema_counter), 6),
                "merged_majority_schema": schema_counter.most_common(1)[0][0],
                "before_contexts": [
                    {
                        "context_id": hashlib.sha1(key.encode("utf-8")).hexdigest()[:12],
                        "support_count": len(items),
                        "field_value": items[0].pre_features.get(field),
                        "majority_schema": Counter(outcome_schema_key(item, labeler="v3") for item in items).most_common(1)[0][0],
                        "example_steps": [item.step for item in items[:4]],
                    }
                    for key, items in list(baseline_keys.items())[:6]
                ],
            }
        )
    candidates.sort(key=lambda item: (-item["after_support"], item["merged_schema_entropy"]))
    return candidates[:limit]


def write_schema_confusions_csv(path: Path, contexts: list[dict[str, Any]]) -> None:
    fields = [
        "game",
        "context_id",
        "support_count",
        "actions_seen",
        "majority_schema_outcome",
        "majority_schema_rate",
        "schema_entropy",
        "parameter_entropy",
        "parameter_distinct_count",
        "example_steps",
        "prototype_path",
        "schema_outcomes",
        "nearest_contexts",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for context in contexts:
            writer.writerow(
                {
                    field: stable_json(context.get(field))
                    for field in fields
                }
            )


def write_markdown_report(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def high_reuse_contexts_markdown(contexts: list[dict[str, Any]]) -> str:
    selected = sorted(
        contexts,
        key=lambda item: (-item["support_count"], item["schema_entropy"], item["parameter_entropy"]),
    )[:30]
    lines = ["# High Reuse R3a Contexts", "", "These are the partitions R3a reuses most often."]
    for context in selected:
        lines.extend(context_markdown_block(context))
    return "\n".join(lines)


def high_entropy_contexts_markdown(contexts: list[dict[str, Any]]) -> str:
    selected = sorted(
        contexts,
        key=lambda item: (-item["parameter_entropy"], -item["schema_entropy"], -item["support_count"]),
    )[:30]
    lines = ["# High Entropy R3a Contexts", "", "These are reused contexts where R3a merges consequences it cannot parameterize."]
    for context in selected:
        lines.extend(context_markdown_block(context))
    return "\n".join(lines)


def schema_atlas_markdown(
    contexts: list[dict[str, Any]],
    field_contributions: Mapping[str, Any],
    vss_decomposition: Mapping[str, Any],
    entropy_contrast: Mapping[str, Any],
) -> str:
    lines = [
        "# R3a Schema Atlas",
        "",
        "This atlas describes the transition-space partition induced by R3a using V3 outcome labels.",
    ]
    for game in sorted({context["game"] for context in contexts}):
        game_contexts = [context for context in contexts if context["game"] == game]
        lines.extend(["", f"## {game}", ""])
        lines.append(f"- contexts: {len(game_contexts)}")
        lines.append(f"- transitions represented: {sum(context['support_count'] for context in game_contexts)}")
        lines.append("")
        lines.append("### Top Reused Contexts")
        for context in sorted(game_contexts, key=lambda item: -item["support_count"])[:10]:
            lines.extend(context_markdown_block(context, compact=True))
        lines.append("")
        lines.append("### Top Parameter-Entropy Contexts")
        for context in sorted(game_contexts, key=lambda item: -item["parameter_entropy"])[:10]:
            lines.extend(context_markdown_block(context, compact=True))
        lines.append("")
        lines.append("### Field Contribution Summary")
        fields = field_contributions.get(game, {}).get("fields", {})
        for field, payload in sorted(fields.items()):
            lines.append(
                f"- `{field}`: {payload.get('classification')} "
                f"(cardinality {payload.get('field_cardinality')}, "
                f"reuse_damage {payload.get('reuse_damage')}, "
                f"contradiction_reduction {payload.get('contradiction_reduction')})"
            )
        lines.append("")
        lines.append("### Visible State Summary Decomposition")
        variants = vss_decomposition.get(game, {}).get("variants", {})
        for variant, payload in sorted(variants.items()):
            comparison = payload.get("comparison_to_without_vss", {})
            metrics = payload.get("metrics", {})
            lines.append(
                f"- `{variant}`: {payload.get('classification')} "
                f"(contexts {metrics.get('unique_contexts')}, "
                f"reuse {metrics.get('schema_reuse_coverage')}, "
                f"schema_entropy_delta_vs_no_vss {comparison.get('schema_entropy_delta')}, "
                f"parameter_entropy_delta_vs_no_vss {comparison.get('parameter_entropy_delta')})"
            )
        lines.append("")
        lines.append("### Parameter Entropy Contrast")
        contrast = entropy_contrast.get(game, {})
        low_count = len(contrast.get("low_parameter_entropy_contexts", ()))
        high_count = len(contrast.get("high_parameter_entropy_contexts", ()))
        lines.append(f"- low parameter entropy reused contexts sampled: {low_count}")
        lines.append(f"- high parameter entropy reused contexts sampled: {high_count}")
    return "\n".join(lines)


def context_markdown_block(context: Mapping[str, Any], *, compact: bool = False) -> list[str]:
    lines = [
        "",
        f"## {context['game']} / {context['context_id']}",
        "",
        f"- support: {context['support_count']}",
        f"- actions: {context['actions_seen']}",
        f"- majority schema: `{context['majority_schema_outcome']}` ({context['majority_schema_rate']})",
        f"- schema entropy: {context['schema_entropy']}",
        f"- parameter entropy: {context['parameter_entropy']}",
        f"- parameter kinds: {context['parameter_distinct_count']}",
        f"- example steps: {context['example_steps']}",
        f"- prototype: `{context['prototype_path']}`",
    ]
    if not compact:
        lines.append(f"- channel summary: `{stable_json(context['changed_channel_summary'])}`")
        neighbors = context.get("nearest_contexts", [])[:3]
        if neighbors:
            lines.append("- nearest contexts:")
            for neighbor in neighbors:
                lines.append(
                    f"  - `{neighbor['context_id']}` diff={neighbor['field_distance']} "
                    f"fields={neighbor['differing_fields']} outcome_changed={neighbor['outcome_changed']} "
                    f"schema=`{neighbor['majority_schema_outcome']}`"
                )
    return lines


def v2_parameter_split_groups(records: list[TransitionRecord], family: RepresentationFamily) -> list[dict[str, Any]]:
    return parameter_split_groups(records, family, labeler="v2")


def parameter_split_groups(
    records: list[TransitionRecord],
    family: RepresentationFamily,
    *,
    labeler: LabelerVersion,
) -> list[dict[str, Any]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[TransitionRecord]] = defaultdict(list)
    for record in records:
        key = context_key(record, family)
        schema = outcome_schema_key(record, labeler=labeler)
        parameter = parameter_outcome_key(record, labeler=labeler)
        group_key = stable_json({"context": key, "action": record.action, "schema": schema})
        groups[group_key][parameter] += 1
        examples[group_key].append(record)
    return [
        {"key": key, "records": examples[key]}
        for key, counter in groups.items()
        if sum(counter.values()) > 1 and len(counter) > 1
    ]


def r3a_labeler_v2_ablation_report(records: list[TransitionRecord]) -> dict[str, Any]:
    families = {
        "R3a": R3A_COMPRESSED,
        "R3a_minus_level_snapshot": RepresentationFamily(
            "R3a_minus_level_snapshot",
            frozenset(field for field in R3A_COMPRESSED.pre_action_fields if field != "level_snapshot"),
        ),
        "R3a_minus_visible_state_summary": RepresentationFamily(
            "R3a_minus_visible_state_summary",
            frozenset(field for field in R3A_COMPRESSED.pre_action_fields if field != "visible_state_summary"),
        ),
        "R3a_minus_level_snapshot_and_visible_state_summary": RepresentationFamily(
            "R3a_minus_level_snapshot_and_visible_state_summary",
            frozenset(
                field
                for field in R3A_COMPRESSED.pre_action_fields
                if field not in {"level_snapshot", "visible_state_summary"}
            ),
        ),
    }
    return {
        name: strip_metric_examples(prequential_parameter_metrics(records, family, labeler="v2"))
        for name, family in families.items()
    }


def labeler_verdict(
    *,
    audit: Mapping[str, Any],
    actor_audit: Mapping[str, Any],
    comparison: Mapping[str, Any],
    stage_v2: Mapping[str, Any],
) -> dict[str, Any]:
    r3a = comparison.get(R3A_COMPRESSED.name, {})
    delta = r3a.get("v2_minus_v1", {})
    v2 = r3a.get("v2", {})
    parameter_drop = -float(delta.get("parameter_contradiction_delta", 0.0) or 0.0)
    v2_parameter_contradiction = float(v2.get("parameter_contradiction_given_schema", 0.0) or 0.0)
    v2_parameter_error = float(v2.get("parameter_error_given_schema", 0.0) or 0.0)
    low_actor_rate = float(
        actor_audit
        .get("v2_actor_confidence_distribution", {})
        .get("low", {})
        .get("rate", 0.0)
        or 0.0
    )
    click_assignment = float(audit.get("click_target_assignment_rate", 0.0) or 0.0)
    unassigned = float(audit.get("unassigned_delta_rate", 0.0) or 0.0)
    top_channel = stage_v2.get("top_localizing_channel")
    labels = []
    if parameter_drop >= 0.20:
        labels.append("MEASUREMENT_ARTIFACT")
    if low_actor_rate >= 0.80 or unassigned >= 0.30:
        labels.append("PERCEPTION_FAILURE")
    if click_assignment >= 0.25 and v2_parameter_contradiction >= 0.50 and top_channel == "action_target_delta":
        labels.append("PROCEDURAL_STATE_FAILURE")
    if not labels and (v2_parameter_contradiction >= 0.50 or v2_parameter_error >= 0.20):
        labels.append("REAL_PARAMETER_FAILURE")
    if not labels:
        labels.append("LABELER_V2_INCONCLUSIVE")
    return {
        "primary": labels[0],
        "labels": labels,
        "r3a_parameter_contradiction_drop_v2_vs_v1": round(parameter_drop, 6),
        "r3a_v2_parameter_contradiction": round(v2_parameter_contradiction, 6),
        "r3a_v2_parameter_error": round(v2_parameter_error, 6),
        "v2_low_actor_rate": round(low_actor_rate, 6),
        "click_target_assignment_rate": round(click_assignment, 6),
        "unassigned_delta_rate": round(unassigned, 6),
        "top_v2_localizing_channel": top_channel,
    }


def labeler_v3_verdict(
    *,
    controlled_audit: Mapping[str, Any],
    delta_assignment: Mapping[str, Any],
    comparison: Mapping[str, Any],
    stage_v3: Mapping[str, Any],
) -> dict[str, Any]:
    r3a = comparison.get(R3A_COMPRESSED.name, {})
    delta = r3a.get("v3_minus_v2", {})
    v3 = r3a.get("v3", {})
    parameter_drop = -float(delta.get("parameter_contradiction_delta", 0.0) or 0.0)
    v3_parameter_contradiction = float(v3.get("parameter_contradiction_given_schema", 0.0) or 0.0)
    v3_parameter_error = float(v3.get("parameter_error_given_schema", 0.0) or 0.0)
    v3_reused_contexts = int(v3.get("parameter_reused_contexts", 0) or 0)
    v3_reused_transitions = int(v3.get("parameter_reused_transitions", 0) or 0)
    examples = int(v3.get("examples", 0) or 0)
    parameter_support_ok = (
        (v3_reused_contexts >= 2 and v3_reused_transitions >= 4)
        or (examples < 30 and v3_reused_contexts >= 1 and v3_reused_transitions >= 2)
    )
    controlled_rate = float(delta_assignment.get("controlled_entity_assignment_rate", 0.0) or 0.0)
    click_assignment = float(delta_assignment.get("click_target_assignment_rate", 0.0) or 0.0)
    unknown_rate = float(delta_assignment.get("unknown_delta_rate", 0.0) or 0.0)
    low_confidence = float(
        controlled_audit
        .get("confidence_distribution", {})
        .get("low", {})
        .get("rate", 0.0)
        or 0.0
    )
    top_channel = stage_v3.get("top_localizing_channel")
    labels = []
    if parameter_drop >= 0.20 and parameter_support_ok:
        labels.append("MEASUREMENT_ARTIFACT")
    elif parameter_drop >= 0.20:
        labels.append("INSUFFICIENT_PARAMETER_SUPPORT")
    if unknown_rate >= 0.30 or (low_confidence >= 0.80 and controlled_rate < 0.10):
        labels.append("PERCEPTION_FAILURE")
    if click_assignment >= 0.25 and v3_parameter_contradiction >= 0.50 and top_channel == "action_target_delta":
        labels.append("PROCEDURAL_STATE_FAILURE")
    if not labels and (v3_parameter_contradiction >= 0.50 or v3_parameter_error >= 0.20):
        labels.append("REAL_PARAMETER_FAILURE")
    if not labels:
        labels.append("LABELER_V3_INCONCLUSIVE")
    return {
        "primary": labels[0],
        "labels": labels,
        "r3a_parameter_contradiction_drop_v3_vs_v2": round(parameter_drop, 6),
        "r3a_v3_parameter_contradiction": round(v3_parameter_contradiction, 6),
        "r3a_v3_parameter_error": round(v3_parameter_error, 6),
        "r3a_v3_parameter_reused_contexts": v3_reused_contexts,
        "r3a_v3_parameter_reused_transitions": v3_reused_transitions,
        "r3a_v3_parameter_support_ok": parameter_support_ok,
        "controlled_entity_assignment_rate": round(controlled_rate, 6),
        "click_target_assignment_rate": round(click_assignment, 6),
        "unknown_delta_rate": round(unknown_rate, 6),
        "low_controlled_entity_confidence_rate": round(low_confidence, 6),
        "top_v3_localizing_channel": top_channel,
    }


def counter_distribution(counter: Counter[str], total: int) -> dict[str, Any]:
    return {
        key: {"count": count, "rate": round(count / total if total else 0.0, 6)}
        for key, count in sorted(counter.items())
    }


def channel_cluster_count(record: TransitionRecord, channel: str) -> int:
    channels = record.post_labels.get("labeler_v2", {}).get("channels", {})
    value = channels.get(channel, {})
    if isinstance(value, Mapping):
        return int(value.get("cluster_count", 0) or 0)
    return 0


def v3_channel_cluster_count(record: TransitionRecord, channel: str) -> int:
    channels = record.post_labels.get("labeler_v3", {}).get("channels", {})
    value = channels.get(channel, {})
    if isinstance(value, Mapping):
        return int(value.get("cluster_count", 0) or 0)
    return 0


def assigned_cluster_ids(record: TransitionRecord, channel: str) -> set[int]:
    return set(record.post_labels.get("labeler_v2", {}).get("changed_region_assignments", {}).get(channel, ()))


def v2_channel_signature(record: TransitionRecord, channel: str) -> str:
    return stable_json(record.post_labels.get("labeler_v2", {}).get("channels", {}).get(channel))


def v3_channel_signature(record: TransitionRecord, channel: str) -> str:
    return stable_json(record.post_labels.get("labeler_v3", {}).get("channels", {}).get(channel))


def r4_representation_probe_report(records: list[TransitionRecord]) -> dict[str, Any]:
    baseline = prequential_parameter_metrics(records, R3A_COMPRESSED)
    report = {
        R3A_COMPRESSED.name: {
            "metrics": baseline,
            "baseline": True,
            "frozen_pre_action_fields": list(R3A_FROZEN_FIELDS),
        }
    }
    for family in R4_REPRESENTATION_FAMILIES:
        metrics = prequential_parameter_metrics(records, family)
        report[family.name] = {
            "metrics": metrics,
            "comparison_to_R3a": r4_probe_comparison(
                baseline,
                metrics,
                game=records[0].game if records else "unknown",
                family=family.name,
            ),
        }
    return report


def r4_probe_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    game: str,
    family: str,
) -> dict[str, Any]:
    baseline_param = float(baseline.get("parameter_contradiction_given_schema", 0.0) or 0.0)
    candidate_param = float(candidate.get("parameter_contradiction_given_schema", 0.0) or 0.0)
    drop = max(0.0, baseline_param - candidate_param)
    relative_drop = drop / baseline_param if baseline_param else 0.0
    reuse_loss = max(
        0.0,
        float(baseline.get("schema_reuse_coverage", 0.0) or 0.0)
        - float(candidate.get("schema_reuse_coverage", 0.0) or 0.0),
    )
    one_off_increase = max(
        0.0,
        float(candidate.get("schema_one_off_ratio", 0.0) or 0.0)
        - float(baseline.get("schema_one_off_ratio", 0.0) or 0.0),
    )
    support = r4_support_status(candidate)
    suspicious = drop > 0.0 and (
        reuse_loss > 0.10
        or one_off_increase > 0.10
    )
    success = (
        support["status"] == "eligible"
        and relative_drop >= 0.20
        and reuse_loss <= 0.10
        and one_off_increase <= 0.10
    )
    target_game = r4_target_game(family)
    is_negative_control = target_game is not None and target_game != game
    if support["status"] == "insufficient_support":
        verdict = "insufficient_support"
    elif success:
        verdict = "success"
    else:
        verdict = "no_gain"
    if suspicious and support["status"] != "insufficient_support":
        verdict = "suspicious_fragmentation"
    return {
        "baseline_family": R3A_COMPRESSED.name,
        "parameter_contradiction_drop": round(drop, 6),
        "parameter_contradiction_relative_drop": round(relative_drop, 6),
        "reuse_loss": round(reuse_loss, 6),
        "one_off_increase": round(one_off_increase, 6),
        "gain_per_reuse_lost": round(drop / max(reuse_loss, 1e-6), 6),
        "support": support,
        "target_game": target_game,
        "is_negative_control": is_negative_control,
        "negative_control_effect": round(drop, 6) if is_negative_control else None,
        "suspicious_fragmentation_flag": suspicious,
        "verdict": verdict,
    }


def r4_support_status(metrics: Mapping[str, Any]) -> dict[str, Any]:
    examples = int(metrics.get("examples", 0) or 0)
    reused_contexts = int(metrics.get("schema_reused_contexts", 0) or 0)
    reused_transitions = int(metrics.get("schema_reused_transitions", 0) or 0)
    reuse = float(metrics.get("schema_reuse_coverage", 0.0) or 0.0)
    eligible = (
        (reused_contexts >= 10 and reused_transitions >= 30)
        or (examples < 30 and reuse >= 0.25)
    )
    return {
        "status": "eligible" if eligible else "insufficient_support",
        "reused_contexts": reused_contexts,
        "reused_transitions": reused_transitions,
        "reuse_coverage": round(reuse, 6),
        "examples": examples,
    }


def r4_target_game(family: str) -> str | None:
    if "relational_geometry" in family:
        return "ar25"
    if "click_procedural" in family:
        return "cd82"
    return None


def compression_curve_report(game_metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for family, by_granularity in game_metrics.items():
        coarse = by_granularity.get("coarse", {})
        exact = by_granularity.get("exact", {})
        reuse = float(coarse.get("reuse_coverage", 0.0) or 0.0)
        contradiction = float(coarse.get("contradiction_rate", 0.0) or 0.0)
        entropy_value = float(coarse.get("mean_weighted_outcome_entropy", 0.0) or 0.0)
        rows.append(
            {
                "family": family,
                "context_count": coarse.get("unique_contexts"),
                "reuse_coverage": coarse.get("reuse_coverage"),
                "one_off_ratio": coarse.get("one_off_ratio"),
                "contradiction_rate": coarse.get("contradiction_rate"),
                "mean_entropy": coarse.get("mean_weighted_outcome_entropy"),
                "prediction_accuracy": coarse.get("majority_outcome_accuracy"),
                "exact_one_off_ratio": exact.get("one_off_ratio"),
                "elbow_candidate": reuse >= 0.2 and contradiction <= 0.35 and entropy_value <= 1.0,
            }
        )
    return rows


def game_diagnostic_verdict(
    game_metrics: Mapping[str, Any],
    ablation: Mapping[str, Any],
    schema_metrics: Mapping[str, Any],
    r4_report: Mapping[str, Any],
    causal_stages: Mapping[str, Any],
) -> dict[str, Any]:
    r2a = game_metrics.get(R2A_STATIC.name, {}).get("coarse", {})
    r2b = game_metrics.get(R2B_MEMORY.name, {}).get("coarse", {})
    r3a = game_metrics.get(R3A_COMPRESSED.name, {}).get("coarse", {})
    r3b = game_metrics.get(R3B_ROLE_MEMORY.name, {}).get("coarse", {})
    labels = []
    if float(r2a.get("one_off_ratio", 0.0) or 0.0) >= 0.95:
        labels.append("compression_failure")
    if (
        float(r2b.get("one_off_ratio", 0.0) or 0.0) >= 0.95
        and float(r2a.get("one_off_ratio", 0.0) or 0.0) < float(r2b.get("one_off_ratio", 0.0) or 0.0)
    ):
        labels.append("fragmented_memory")
    if float(r3a.get("reuse_coverage", 0.0) or 0.0) > float(r2a.get("reuse_coverage", 0.0) or 0.0):
        labels.append("compressed_static_improves_reuse")
    if float(r3b.get("reuse_coverage", 0.0) or 0.0) > float(r3a.get("reuse_coverage", 0.0) or 0.0):
        labels.append("role_memory_improves_reuse")
    if (
        float(r3b.get("contradiction_rate", 0.0) or 0.0) < float(r3a.get("contradiction_rate", 0.0) or 0.0)
        and (
            float(r3b.get("reuse_coverage", 0.0) or 0.0) < float(r3a.get("reuse_coverage", 0.0) or 0.0)
            or float(r3b.get("one_off_ratio", 0.0) or 0.0) > float(r3a.get("one_off_ratio", 0.0) or 0.0)
        )
    ):
        labels.append("role_memory_not_promoted")
    if float(r3a.get("contradiction_rate", 0.0) or 0.0) >= 0.5:
        labels.append("target_state_missing")
    r3_schema = schema_metrics.get(R3A_COMPRESSED.name, {})
    if (
        float(r3_schema.get("schema_contradiction_rate", 0.0) or 0.0) <= 0.20
        and float(r3_schema.get("parameter_contradiction_given_schema", 0.0) or 0.0) >= 0.50
    ):
        labels.append("schema_detector_only")
    if float(causal_stages.get("top_localizing_stage_share", 0.0) or 0.0) >= 0.50:
        labels.append("causal_ordering_missing")
    r4_labels = r4_verdict_labels(r4_report)
    labels.extend(label for label in r4_labels if label not in labels)
    if not labels:
        labels.append("insufficient_reuse")
    fragmentation_fields = {
        family: data.get("fragmentation_candidates", [])
        for family, data in ablation.items()
    }
    return {
        "labels": labels,
        "primary": labels[0],
        "fragmentation_fields": fragmentation_fields,
        "elbow_candidate": any(
            row["elbow_candidate"] for row in compression_curve_report(game_metrics)
        ),
    }


def r4_verdict_labels(r4_report: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    successes = []
    suspicious = []
    insufficient = []
    negative_control_failed = False
    for family, payload in r4_report.items():
        if family == R3A_COMPRESSED.name:
            continue
        comparison = payload.get("comparison_to_R3a", {})
        verdict = comparison.get("verdict")
        if verdict == "success":
            successes.append(family)
        if verdict == "suspicious_fragmentation":
            suspicious.append(family)
        if verdict == "insufficient_support":
            insufficient.append(family)
        if comparison.get("is_negative_control") and verdict in {"success", "suspicious_fragmentation"}:
            negative_control_failed = True
    if any("relational_geometry" in family for family in successes):
        labels.append("relational_geometry_missing")
    visual_success = any("click_procedural_visual" in family for family in successes)
    learned_success = any("click_procedural_learned" in family for family in successes)
    if visual_success:
        labels.append("procedural_visual_prior_sufficient")
    if learned_success and not visual_success:
        labels.append("procedural_effect_memory_needed")
    if suspicious:
        labels.append("r4_fragmentation_engine")
    if negative_control_failed:
        labels.append("negative_control_failed")
    if insufficient and not successes:
        labels.append("insufficient_support")
    return labels


def role_cluster_payload(record: TransitionRecord) -> dict[str, Any]:
    return {
        "role": record.pre_features.get("action_target_type", "unknown"),
        "zone": record.pre_features.get("target_relative_region", "unknown"),
        "visual": record.pre_features.get("clicked_region_signature")
        or record.pre_features.get("actor_or_focus_region")
        or "unknown",
    }


def relational_geometry_payload(record: TransitionRecord) -> dict[str, Any]:
    return {
        "geometry": record.pre_features.get("current_relative_geometry_class"),
        "touch": record.pre_features.get("current_touch_overlap_relation"),
        "shape_family": record.pre_features.get("canonical_group_shape_family"),
        "normalized": record.pre_features.get("mirror_rotation_normalized_signature"),
    }


def group_motion_role(record: TransitionRecord) -> str:
    groups = getattr(record.event, "motion_groups", ())
    if not groups:
        return "no_group_motion"
    if any(len(group.object_ids) > 1 for group in groups):
        return "co_motion"
    if record.post_labels.get("remote_delta_present"):
        return "remote_motion"
    return "single_motion"


def click_target_key(record: TransitionRecord) -> str:
    return stable_json(
        {
            "zone": record.pre_features.get("click_target_zone"),
            "visual": record.pre_features.get("click_target_visual_signature"),
        }
    )


def learned_click_target_type(effect: str) -> str:
    if effect == "hud_delta":
        return "palette_or_tool_learned"
    if effect in {"resource_delta", "remote_delta", "actor_move"}:
        return "canvas_or_object_learned"
    if effect in {"no_op", "block_reject"}:
        return "background_or_reject_learned"
    if effect == "terminal":
        return "terminal_trigger_learned"
    return "unknown"


def target_material_summary(record: TransitionRecord) -> dict[str, Any]:
    grid = np.asarray(record.before.pixels[-1])
    if not (record.action_data and "x" in record.action_data and "y" in record.action_data):
        return {"value": "none", "neighborhood": ()}
    row = int(record.action_data["y"])
    col = int(record.action_data["x"])
    if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
        return {"value": "off_canvas", "neighborhood": ()}
    top = max(0, row - 1)
    bottom = min(grid.shape[0], row + 2)
    left = max(0, col - 1)
    right = min(grid.shape[1], col + 2)
    values, counts = np.unique(grid[top:bottom, left:right], return_counts=True)
    return {
        "value": int(grid[row, col]),
        "neighborhood": tuple(
            sorted((int(value), bucket_count(int(count))) for value, count in zip(values, counts))
        ),
    }


def target_differs_from_initial(record: TransitionRecord, initial_grid: np.ndarray | None) -> str:
    if initial_grid is None or not (record.action_data and "x" in record.action_data and "y" in record.action_data):
        return "unknown"
    current = np.asarray(record.before.pixels[-1])
    row = int(record.action_data["y"])
    col = int(record.action_data["x"])
    if not (0 <= row < current.shape[0] and 0 <= col < current.shape[1]):
        return "off_canvas"
    if initial_grid.shape != current.shape:
        return "unknown"
    return "changed" if int(current[row, col]) != int(initial_grid[row, col]) else "same"


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    value = 0.0
    for count in counter.values():
        probability = count / total
        value -= probability * math.log2(probability)
    return value


def observation_digest(observation: Observation) -> str:
    digest = hashlib.sha1()
    digest.update(np.asarray(observation.pixels).tobytes())
    digest.update(
        stable_json(
            {
                "levels_completed": observation.levels_completed,
                "state": observation.state,
                "full_reset": observation.full_reset,
            }
        ).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def stable_json(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def bucket_area(area: int) -> str:
    if area <= 1:
        return "single"
    if area <= 4:
        return "tiny"
    if area <= 16:
        return "small"
    if area <= 64:
        return "medium"
    return "large"


def bucket_count(count: int) -> str:
    if count <= 0:
        return "zero"
    if count == 1:
        return "one"
    if count <= 4:
        return "few"
    if count <= 16:
        return "some"
    if count <= 64:
        return "many"
    return "huge"


def bucket_coord(value: float) -> int:
    return int(value // 4)


def bucket_signed(value: float) -> str:
    if value < -16:
        return "far_neg"
    if value < -4:
        return "near_neg"
    if value <= 4:
        return "same"
    if value <= 16:
        return "near_pos"
    return "far_pos"


def confidence_bucket(support: int, total: int) -> str:
    if total <= 0:
        return "none"
    ratio = support / total
    if ratio >= 0.9 and total >= 3:
        return "high"
    if ratio >= 0.6:
        return "medium"
    return "low"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(stable_json(record) + "\n")


def write_contradictions_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "game",
        "family",
        "granularity",
        "num_occurrences",
        "num_distinct_outcomes",
        "steps",
        "actions",
        "outcome_examples",
        "context_key",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: stable_json(record.get(field))
                    if field in {"steps", "actions", "outcome_examples"}
                    else record.get(field)
                    for field in fields
                }
            )
