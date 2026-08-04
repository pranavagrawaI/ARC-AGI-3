"""Treatment-only agent selected by the shared factory for ``learning_loop``."""

from __future__ import annotations

from typing import Any

from experiments.harness import Decision, Observation, Transition
from llm_repl_agent.agent import (
    DecisionFormatError,
    _json_object,
    _message_text,
    parse_decision_message,
)

from .agent import SharedArmAgent
from .learning_loop import DecisionContractError, LearningLoopRuntime


class LearningLoopAgent(SharedArmAgent):
    """Shared control plus the minimum model-owned empirical-loop capability."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if "learning_loop" not in self.capability_names:
            raise ValueError("LearningLoopAgent requires the learning_loop capability")
        self.learning_loop = LearningLoopRuntime(self.workspace.root)

    def _extend_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = super()._extend_payload(payload)
        payload["capability_packet"] = self.learning_loop.packet(self._history)
        return payload

    def _decision_from_response(
        self,
        response: Any,
        *,
        observation: Observation,
        repairs: int,
    ) -> Decision:
        height, width = observation.pixels.shape[-2:]
        action, data, _reason = parse_decision_message(
            response.message,
            available_actions=observation.available_actions,
            frame_width=width,
            frame_height=height,
        )
        payload = _json_object(_message_text(response.message.get("content")))
        try:
            commitment = self.learning_loop.accept(
                payload.get("reason"),
                history=self._history,
                frame_shape=(height, width),
                levels_completed=observation.levels_completed,
            )
        except DecisionContractError as error:
            raise DecisionFormatError(str(error)) from error
        return Decision(
            action=action,
            data=data,
            reasoning={
                "policy": "0006_learning_loop_v1",
                "reason": commitment["test"],
                "commitment": commitment,
                "decision_repairs": repairs,
                "response_id": response.response_id,
            },
        )

    def observe(self, transition: Transition) -> None:
        super().observe(transition)
        assert self._current is not None
        result = self.learning_loop.observe(
            self._history[-1],
            final_grid=self._current.pixels[-1],
        )
        self.process_log.write(
            "prediction_result",
            action_step=self.actions,
            **result,
        )

    def decision_contract(self) -> dict[str, Any]:
        return {
            "parser": "p9.learning_loop.validate_commitment",
            "fields": ["action", "data", "reason"],
            "reason_fields": [
                "working_state",
                "test",
                "prediction",
                "objective",
                "history_steps",
                "reconciliation",
            ],
            "legality_checked_against": "observation.available_actions",
            "prediction_checked_against": [
                "changed_pixels",
                "levels_completed",
                "state",
                "final_layer_cell",
            ],
            "objective_progress_checked_against": "levels_completed",
            "objective_progress_rule": "levels_completed > count at claim creation",
            "maximum_objective_horizon": 8,
        }
