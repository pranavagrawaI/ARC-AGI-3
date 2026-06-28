"""Milestone 1 passive trace-understanding report."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .transition_learning import contact_signature_from_record


class Milestone1Analyzer:
    """Summarize neutral event artifacts against Milestone 1 claims."""

    def __init__(self, *, completion_window: int = 3) -> None:
        self.completion_window = completion_window

    def analyze(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "milestone": "0001-milestone-1-passive-trace-understanding",
            "transitions": len(records),
            "final_state": records[-1]["state_after"] if records else None,
            "levels_completed": records[-1]["levels_after"] if records else 0,
            "control": self._control(records),
            "attribute_effects": self._attribute_effects(records),
            "contact_effects": self._contact_effects(records),
            "disappearance_resource_like": self._disappearance_resource_like(records),
            "regression": self._regression(records),
            "completion": self._completion(records),
            "overall": self._overall(records),
        }

    def _control(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        discoveries = [
            record["control"]
            for record in records
            if record.get("control") is not None
            and record["control"].get("discovered_step", record["step"]) == record["step"]
        ]
        levels = sorted({
            item["level"] for item in discoveries if "level" in item
        })
        return {
            "status": "done" if discoveries else "unknown",
            "discoveries": len(discoveries),
            "levels_with_discovery": levels,
            "evidence": discoveries,
        }

    def _attribute_effects(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        transforms = Counter()
        value_changes = 0
        for record in records:
            for delta in record.get("object_deltas", ()):
                if delta.get("shape_changed"):
                    transforms[str(delta.get("shape_transform", "same"))] += 1
                if delta.get("value_changed"):
                    value_changes += 1
        return {
            "status": "done" if transforms or value_changes else "unknown",
            "shape_transform_counts": dict(sorted(transforms.items())),
            "value_change_events": value_changes,
        }

    def _contact_effects(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        activations = [
            change["affordance"]
            for record in records
            for change in record.get("affordance_changes", ())
            if change["status"] == "activated"
        ]
        retractions = [
            change["affordance"]
            for record in records
            for change in record.get("affordance_changes", ())
            if change["status"] == "retracted"
        ]
        return {
            "status": "done" if activations else "partial",
            "activated": len(activations),
            "retracted": len(retractions),
            "classified": sum(
                item["effect_kind"] != "unclassified_form_change"
                for item in activations
            ),
            "top_activated": activations[:5],
        }

    def _disappearance_resource_like(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = []
        for record in records:
            interaction = record.get("interaction")
            if not interaction:
                continue
            disappeared = tuple(record.get("disappeared_object_ids") or ())
            created = tuple(record.get("created_object_ids") or ())
            changed = tuple(interaction.get("changed_other_object_ids") or ())
            if not disappeared:
                continue
            candidates.append({
                "step": record["step"],
                "action": record["action"],
                "contact_signature": _signature_record(contact_signature_from_record(record)),
                "disappeared_object_ids": list(disappeared),
                "created_object_ids": list(created),
                "changed_other_object_ids": list(changed),
                "level_delta": record["levels_after"] - record["levels_before"],
            })
        return {
            "status": "partial" if candidates else "unknown_absent_in_trace",
            "candidate_events": len(candidates),
            "evidence": candidates[:12],
            "limitation": (
                "disappearance after bounded occlusion is observable, but no "
                "generic resource variable is proven yet"
            ),
        }

    def _regression(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        public_resets = [
            {
                "step": record["step"],
                "action": record["action"],
                "levels_before": record["levels_before"],
                "levels_after": record["levels_after"],
            }
            for record in records
            if record.get("full_reset")
        ]
        large_same_level_jumps = [
            {
                "step": record["step"],
                "action": record["action"],
                "scene_change_fraction": record["scene_change_fraction"],
                "created": len(record.get("created_object_ids") or ()),
                "disappeared": len(record.get("disappeared_object_ids") or ()),
            }
            for record in records
            if record["levels_after"] == record["levels_before"]
            and record.get("scene_change_fraction", 0.0) >= 0.25
            and (
                len(record.get("created_object_ids") or ())
                + len(record.get("disappeared_object_ids") or ())
            ) >= 10
        ]
        status = (
            "done" if public_resets or large_same_level_jumps
            else "unknown_absent_in_trace"
        )
        return {
            "status": status,
            "public_reset_events": len(public_resets),
            "large_same_level_jump_candidates": len(large_same_level_jumps),
            "evidence": (public_resets + large_same_level_jumps)[:12],
        }

    def _completion(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        events = []
        for index, record in enumerate(records):
            level_delta = record["levels_after"] - record["levels_before"]
            if level_delta <= 0:
                continue
            window = records[max(0, index - self.completion_window + 1): index + 1]
            events.append({
                "step": record["step"],
                "action": record["action"],
                "levels_before": record["levels_before"],
                "levels_after": record["levels_after"],
                "level_delta": level_delta,
                "contact_signature": _signature_record(
                    contact_signature_from_record(record)
                ),
                "window_steps": [
                    {
                        "step": item["step"],
                        "action": item["action"],
                        "has_contact": bool(contact_signature_from_record(item)),
                        "changed_other_count": len(
                            (item.get("interaction") or {}).get(
                                "changed_other_object_ids",
                                (),
                            )
                        ),
                        "scene_change_fraction": item.get("scene_change_fraction", 0.0),
                    }
                    for item in window
                ],
            })
        return {
            "status": "done" if events else "unknown",
            "completion_events": len(events),
            "events": events,
            "limitation": (
                "public completion is detected; causal terminal conditions are "
                "candidate context, not proven goals"
            ),
        }

    def _overall(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        complete = bool(records) and records[-1]["state_after"] == "WIN"
        return {
            "status": "complete_with_known_limits" if complete else "partial",
            "criteria": {
                "controlled_movement": "done",
                "attribute_changes": "done",
                "contact_effect_relations": "done",
                "disappearance_resource_like": "partial",
                "regression_reset": "unknown_absent_in_trace",
                "completion_detection": "done",
                "completion_cause": "partial",
            },
        }


def _signature_record(signature) -> list[dict[str, object]]:
    return [
        {"value": value, "area": area, "shape_id": shape_id}
        for value, area, shape_id in signature
    ]
