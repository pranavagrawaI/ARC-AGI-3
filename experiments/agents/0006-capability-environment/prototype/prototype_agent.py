"""THROWAWAY: crude 0006 package for one paired potency probe."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

AGENTS_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = AGENTS_ROOT / "0002-llm-repl-agent"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from experiments.harness import Decision, Observation, Transition
from llm_repl_agent.agent import (
    DecisionFormatError,
    LlmReplAgent,
    _json_object,
    _message_text,
    parse_decision_message,
)


PROTOTYPE_ADDENDUM = """

CRUDE 0006 POTENCY PROTOTYPE — deliberately imperfect, not the final contract.
A host-maintained capability_packet accompanies each observation. It preserves your latest
working hypotheses and gives an exact mechanical summary of the last visible frame change.
Before every action, use that packet to choose a revealing probe rather than merely continuing.

Your action JSON must use this expanded reason object:
{
  "action": <legal integer>,
  "data": <object or null>,
  "reason": {
    "hypotheses": [
      {"claim": <brief testable string>, "status": "active"|"uncertain"|"rejected"}
    ],
    "contradiction_check": <what the latest reveal supports, contradicts, or leaves unresolved>,
    "prediction": <concrete visible consequence expected from this action>,
    "probe_goal": <which uncertainty this action discriminates>
  }
}
Replace the full hypothesis list on each action; keep at most five. The host preserves this
state across context eviction but does not decide, rank, or revise it for you.
"""

_REQUIRED_REASON_FIELDS = (
    "hypotheses",
    "contradiction_check",
    "prediction",
    "probe_goal",
)
_VALID_STATUSES = {"active", "uncertain", "rejected"}


class PrototypeCapabilityAgent(LlmReplAgent):
    """Minimal package: exact delta, persistent hypotheses, and a pre-action gate."""

    def __init__(self, *args: Any, system_prompt: str, **kwargs: Any) -> None:
        super().__init__(
            *args,
            system_prompt=system_prompt.rstrip() + PROTOTYPE_ADDENDUM,
            **kwargs,
        )
        self._working_state: dict[str, Any] = {
            "hypotheses": [],
            "contradiction_check": "No transition observed yet.",
            "prediction": "",
            "probe_goal": "Establish the first action-effect relation.",
        }
        self._last_reveal: dict[str, Any] | None = None
        self._state_path = Path(self.workspace.root) / "prototype_working_state.json"
        self._persist_working_state()

    def _observation_message(
        self,
        current: Any,
        *,
        model_round: int,
        python_calls_this_action: int,
    ) -> dict[str, Any]:
        message = super()._observation_message(
            current,
            model_round=model_round,
            python_calls_this_action=python_calls_this_action,
        )
        packet = {
            "prototype": "crude-0006-potency-v0",
            "working_state": deepcopy(self._working_state),
            "last_reveal": deepcopy(self._last_reveal),
            "instruction": (
                "Reconcile last_reveal with working_state, then choose the cheapest action "
                "that discriminates a live uncertainty. Return the required structured reason."
            ),
        }
        content = list(message["content"])
        content.insert(1, {
            "type": "text",
            "text": "capability_packet:\n" + json.dumps(packet, indent=2, sort_keys=True),
        })
        return {**message, "content": content}

    def _decision_from_response(
        self,
        response: Any,
        *,
        observation: Observation,
        repairs: int,
    ) -> Decision:
        height, width = observation.pixels.shape[-2:]
        action, data, _ = parse_decision_message(
            response.message,
            available_actions=observation.available_actions,
            frame_width=width,
            frame_height=height,
        )
        payload = _json_object(_message_text(response.message.get("content")))
        reason = payload.get("reason")
        state = self._validate_reason(reason)
        self._working_state = state
        self._persist_working_state()
        return Decision(
            action=action,
            data=data,
            reasoning={
                "policy": "crude_0006_potency_prototype",
                **deepcopy(state),
                "decision_repairs": repairs,
                "response_id": response.response_id,
            },
        )

    def observe(self, transition: Transition) -> None:
        prediction = str(self._working_state.get("prediction", ""))
        before = self.store.colors.apply(transition.before.pixels)
        after = self.store.colors.apply(transition.after.pixels)
        self._last_reveal = {
            "action": transition.decision.action,
            "action_data": transition.decision.data,
            "prior_prediction": prediction,
            "exact_frame_change": _frame_change(before, after),
            "public_state": {
                "levels_completed_before": transition.before.levels_completed,
                "levels_completed_after": transition.after.levels_completed,
                "state_before": transition.before.state,
                "state_after": transition.after.state,
                "full_reset_after": transition.after.full_reset,
            },
        }
        super().observe(transition)

    def _validate_reason(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise DecisionFormatError("reason must be the required object")
        missing = [field for field in _REQUIRED_REASON_FIELDS if field not in value]
        if missing:
            raise DecisionFormatError(f"reason is missing fields: {', '.join(missing)}")
        extra = sorted(set(value) - set(_REQUIRED_REASON_FIELDS))
        if extra:
            raise DecisionFormatError(f"reason has unexpected fields: {', '.join(extra)}")
        hypotheses = value["hypotheses"]
        if not isinstance(hypotheses, list) or len(hypotheses) > 5:
            raise DecisionFormatError("reason.hypotheses must be a list of at most five items")
        checked: list[dict[str, str]] = []
        for index, hypothesis in enumerate(hypotheses):
            if not isinstance(hypothesis, Mapping) or set(hypothesis) != {"claim", "status"}:
                raise DecisionFormatError(
                    f"reason.hypotheses[{index}] must contain exactly claim and status"
                )
            claim = hypothesis["claim"]
            status = hypothesis["status"]
            if not isinstance(claim, str) or not claim.strip():
                raise DecisionFormatError(f"reason.hypotheses[{index}].claim must be nonempty")
            if status not in _VALID_STATUSES:
                raise DecisionFormatError(f"reason.hypotheses[{index}].status is invalid")
            checked.append({"claim": claim.strip()[:500], "status": str(status)})
        result: dict[str, Any] = {"hypotheses": checked}
        for field in _REQUIRED_REASON_FIELDS[1:]:
            item = value[field]
            if not isinstance(item, str) or not item.strip():
                raise DecisionFormatError(f"reason.{field} must be a nonempty string")
            result[field] = item.strip()[:1000]
        return result

    def _persist_working_state(self) -> None:
        self._state_path.write_text(
            json.dumps(self._working_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _frame_change(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    if before.shape != after.shape:
        return {
            "comparable": False,
            "before_shape": list(before.shape),
            "after_shape": list(after.shape),
        }
    changed = np.argwhere(before != after)
    substitutions: dict[tuple[int, int], int] = {}
    for layer, y, x in changed:
        key = (int(before[layer, y, x]), int(after[layer, y, x]))
        substitutions[key] = substitutions.get(key, 0) + 1
    result: dict[str, Any] = {
        "comparable": True,
        "changed_cell_count": int(len(changed)),
        "substitutions": [
            {"before_color_id": key[0], "after_color_id": key[1], "count": count}
            for key, count in sorted(substitutions.items())
        ],
    }
    if len(changed):
        result["changed_bbox_inclusive"] = {
            "layer_min": int(changed[:, 0].min()),
            "layer_max": int(changed[:, 0].max()),
            "x_min": int(changed[:, 2].min()),
            "x_max": int(changed[:, 2].max()),
            "y_min": int(changed[:, 1].min()),
            "y_max": int(changed[:, 1].max()),
        }
    else:
        result["changed_bbox_inclusive"] = None
    return result
