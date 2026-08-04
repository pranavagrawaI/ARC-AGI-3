"""Minimum model-owned empirical state and public prediction checker for 0006.

This module supplies mechanics, not cognition. The model owns the JSON state, selects
history, authors predictions, and reconciles contradictions. The host validates,
preserves, retrieves, and compares only public values.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "0006-learning-loop-v1"
STATE_FILE = "learning-loop-state.json"
MAX_WORKING_STATE_CHARS = 4_000
MAX_TEST_CHARS = 500
MAX_RECONCILIATION_CHARS = 1_000
MAX_OBJECTIVE_CLAIM_CHARS = 500
MAX_OBJECTIVE_HORIZON = 8
MAX_PREDICTIONS = 5
MAX_HISTORY_STEPS = 4
_NUMERIC_OBSERVABLES = {"changed_pixels", "levels_completed"}
_COMPARISON_OPERATORS = {"==", "!=", "<", "<=", ">", ">="}


class DecisionContractError(ValueError):
    """The model's treatment commitment is malformed or not publicly checkable."""


def validate_commitment(
    value: Any,
    *,
    available_history_steps: Sequence[int],
    reconciliation_required: bool,
    objective_active: bool = False,
    frame_shape: tuple[int, int] | None = None,
    objective_level_baseline: int = 0,
) -> dict[str, Any]:
    """Validate the treatment envelope without interpreting model-authored state."""

    required = {
        "working_state",
        "test",
        "prediction",
        "objective",
        "history_steps",
        "reconciliation",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise DecisionContractError(
            "reason must contain exactly working_state, test, prediction, objective, "
            "history_steps, and reconciliation"
        )

    working_state = value["working_state"]
    if not isinstance(working_state, Mapping):
        raise DecisionContractError("reason.working_state must be a JSON object")
    try:
        encoded_state = json.dumps(
            working_state,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise DecisionContractError("reason.working_state must be valid JSON") from error
    if len(encoded_state) > MAX_WORKING_STATE_CHARS:
        raise DecisionContractError(
            f"reason.working_state exceeds {MAX_WORKING_STATE_CHARS} encoded characters"
        )
    normalized_state = json.loads(encoded_state)

    test = _bounded_text(value["test"], "reason.test", MAX_TEST_CHARS)

    prediction = value["prediction"]
    if not isinstance(prediction, list) or not 1 <= len(prediction) <= MAX_PREDICTIONS:
        raise DecisionContractError(
            f"reason.prediction must contain 1 to {MAX_PREDICTIONS} assertions"
        )
    checked_prediction = [
        _validate_assertion(item, index=index, frame_shape=frame_shape)
        for index, item in enumerate(prediction)
    ]

    objective_raw = value["objective"]
    if objective_active:
        if objective_raw is not None:
            raise DecisionContractError(
                "reason.objective must be null while capability_packet.active_objective exists"
            )
        checked_objective = None
    else:
        checked_objective = _validate_objective(
            objective_raw,
            level_baseline=objective_level_baseline,
        )

    history_steps = value["history_steps"]
    if not isinstance(history_steps, list) or len(history_steps) > MAX_HISTORY_STEPS:
        raise DecisionContractError(
            f"reason.history_steps must be a list of at most {MAX_HISTORY_STEPS} steps"
        )
    if any(not _is_int(step) for step in history_steps):
        raise DecisionContractError("reason.history_steps must contain integers")
    if len(set(history_steps)) != len(history_steps):
        raise DecisionContractError("reason.history_steps must not contain duplicates")
    available = set(available_history_steps)
    missing = [step for step in history_steps if step not in available]
    if missing:
        raise DecisionContractError(
            f"reason.history_steps names unavailable steps: {missing}; "
            f"available steps are {sorted(available)}"
        )

    reconciliation = value["reconciliation"]
    if reconciliation_required:
        reconciliation = _bounded_text(
            reconciliation,
            "reason.reconciliation",
            MAX_RECONCILIATION_CHARS,
        )
    elif reconciliation is not None:
        raise DecisionContractError(
            "reason.reconciliation must be null unless the preceding prediction mismatched"
        )

    return {
        "working_state": normalized_state,
        "test": test,
        "prediction": checked_prediction,
        "objective": checked_objective,
        "history_steps": list(history_steps),
        "reconciliation": reconciliation,
    }


def evaluate_prediction(
    prediction: Sequence[Mapping[str, Any]],
    *,
    transition: Mapping[str, Any],
    final_grid: np.ndarray,
) -> dict[str, Any]:
    """Compare a pre-action commitment with public transition values."""

    grid = np.asarray(final_grid)
    if grid.ndim != 2:
        raise ValueError("final_grid must be the public final layer")
    checks: list[dict[str, Any]] = []
    for assertion in prediction:
        observable = assertion["observable"]
        if observable == "changed_pixels":
            observed = transition.get("changed_pixels")
        elif observable == "levels_completed":
            observed = transition.get("levels_after")
        elif observable == "state":
            observed = transition.get("state_after")
        elif observable == "cell":
            x, y = int(assertion["x"]), int(assertion["y"])
            observed = int(grid[y, x]) if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] else None
        else:  # Validation closes this path; keep evidence explicit if called directly.
            observed = None
        matched = (
            None
            if observed is None
            else _compare(observed, assertion["op"], assertion["value"])
        )
        checks.append({**deepcopy(dict(assertion)), "observed": observed, "matched": matched})

    outcomes = [check["matched"] for check in checks]
    outcome = "mismatch" if False in outcomes else "match" if all(outcomes) else "inconclusive"
    return {"schema": SCHEMA, "outcome": outcome, "checks": checks}


class LearningLoopRuntime:
    """Exact state echo, selected-history retrieval, and prediction bookkeeping."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.working_state: dict[str, Any] = {}
        self.selected_history_steps: tuple[int, ...] = ()
        self.last_prediction_result: dict[str, Any] | None = None
        self.active_objective: dict[str, Any] | None = None
        self.last_objective_result: dict[str, Any] | None = None
        self.pending_commitment: dict[str, Any] | None = None
        self._pending_raw_reason: Any = None
        self._persist()

    @property
    def reconciliation_required(self) -> bool:
        local_mismatch = bool(
            self.last_prediction_result
            and self.last_prediction_result.get("outcome") == "mismatch"
        )
        objective_falsified = bool(
            self.last_objective_result
            and self.last_objective_result.get("outcome") == "falsified"
        )
        return local_mismatch or objective_falsified

    def packet(self, history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        by_step = {int(item["step"]): item for item in history}
        selected = [deepcopy(dict(by_step[step])) for step in self.selected_history_steps if step in by_step]
        return {
            "schema": SCHEMA,
            "working_state": deepcopy(self.working_state),
            "last_prediction_result": deepcopy(self.last_prediction_result),
            "active_objective": deepcopy(self.active_objective),
            "last_objective_result": deepcopy(self.last_objective_result),
            "available_history_steps": sorted(by_step),
            "selected_history": selected,
            "limits": {
                "working_state_chars": MAX_WORKING_STATE_CHARS,
                "prediction_assertions": MAX_PREDICTIONS,
                "history_steps": MAX_HISTORY_STEPS,
            },
        }

    def accept(
        self,
        raw_reason: Any,
        *,
        history: Sequence[dict[str, Any]],
        frame_shape: tuple[int, int],
        levels_completed: int,
    ) -> dict[str, Any]:
        if self.pending_commitment is not None:
            if raw_reason == self._pending_raw_reason:
                return deepcopy(self.pending_commitment)
            raise DecisionContractError(
                "a different commitment cannot replace the pending decision before observation"
            )
        commitment = validate_commitment(
            raw_reason,
            available_history_steps=tuple(int(item["step"]) for item in history),
            reconciliation_required=self.reconciliation_required,
            objective_active=self.active_objective is not None,
            frame_shape=frame_shape,
            objective_level_baseline=levels_completed,
        )
        self.working_state = deepcopy(commitment["working_state"])
        self.selected_history_steps = tuple(commitment["history_steps"])
        self.pending_commitment = deepcopy(commitment)
        self._pending_raw_reason = deepcopy(raw_reason)
        if self.active_objective is None:
            objective = deepcopy(commitment["objective"])
            assert objective is not None
            objective["remaining_steps"] = objective["horizon"]
            self.active_objective = objective
            self.last_objective_result = None
        self._persist()
        return commitment

    def observe(
        self,
        transition: Mapping[str, Any],
        *,
        final_grid: np.ndarray,
    ) -> dict[str, Any]:
        if self.pending_commitment is None:
            raise RuntimeError("a transition arrived without a learning-loop commitment")
        result = evaluate_prediction(
            self.pending_commitment["prediction"],
            transition=transition,
            final_grid=final_grid,
        )
        assert self.active_objective is not None
        objective_check = evaluate_prediction(
            [self.active_objective["progress"]],
            transition=transition,
            final_grid=final_grid,
        )["checks"][0]
        remaining_steps = int(self.active_objective["remaining_steps"]) - 1
        if objective_check["matched"] is True:
            objective_outcome = "supported"
        elif remaining_steps == 0:
            objective_outcome = "falsified"
        else:
            objective_outcome = "pending"
        objective_result = {
            "schema": SCHEMA,
            "claim": self.active_objective["claim"],
            "outcome": objective_outcome,
            "remaining_steps": remaining_steps,
            "check": objective_check,
        }
        if objective_outcome == "pending":
            self.active_objective["remaining_steps"] = remaining_steps
        else:
            self.active_objective = None
        self.last_objective_result = objective_result
        result["objective_result"] = deepcopy(objective_result)
        self.last_prediction_result = result
        self.pending_commitment = None
        self._pending_raw_reason = None
        self._persist()
        return result

    def _persist(self) -> None:
        record = {
            "schema": SCHEMA,
            "working_state": self.working_state,
            "last_prediction_result": self.last_prediction_result,
            "active_objective": self.active_objective,
            "last_objective_result": self.last_objective_result,
            "selected_history_steps": list(self.selected_history_steps),
        }
        (self.workspace / STATE_FILE).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _validate_objective(value: Any, *, level_baseline: int) -> dict[str, Any]:
    label = "reason.objective"
    if not isinstance(value, Mapping) or set(value) != {"claim", "progress", "horizon"}:
        raise DecisionContractError(
            f"{label} must contain exactly claim, progress, and horizon"
        )
    claim = _bounded_text(value["claim"], f"{label}.claim", MAX_OBJECTIVE_CLAIM_CHARS)
    progress = _validate_assertion(value["progress"], index=0, frame_shape=None)
    if progress["observable"] != "levels_completed":
        raise DecisionContractError(
            f"{label}.progress must predict levels_completed, not a local visual effect"
        )
    if progress["op"] != ">" or progress["value"] != level_baseline:
        raise DecisionContractError(
            f"{label}.progress must predict an increase from {level_baseline}: "
            "use observable='levels_completed', op='>', "
            f"value={level_baseline}"
        )
    horizon = value["horizon"]
    if not _is_int(horizon) or not 1 <= int(horizon) <= MAX_OBJECTIVE_HORIZON:
        raise DecisionContractError(
            f"{label}.horizon must be an integer from 1 to {MAX_OBJECTIVE_HORIZON}"
        )
    return {"claim": claim, "progress": progress, "horizon": int(horizon)}


def _validate_assertion(
    value: Any,
    *,
    index: int,
    frame_shape: tuple[int, int] | None,
) -> dict[str, Any]:
    label = f"reason.prediction[{index}]"
    if not isinstance(value, Mapping):
        raise DecisionContractError(f"{label} must be an object")
    observable = value.get("observable")
    if observable in _NUMERIC_OBSERVABLES:
        required = {"observable", "op", "value"}
        if set(value) != required or value.get("op") not in _COMPARISON_OPERATORS:
            raise DecisionContractError(f"{label} has an invalid numeric assertion shape")
        if not _is_int(value.get("value")) or int(value["value"]) < 0:
            raise DecisionContractError(f"{label}.value must be a nonnegative integer")
        return {"observable": observable, "op": value["op"], "value": int(value["value"])}
    if observable == "state":
        if set(value) != {"observable", "op", "value"} or value.get("op") not in {"==", "!="}:
            raise DecisionContractError(f"{label} has an invalid state assertion shape")
        if not isinstance(value.get("value"), str):
            raise DecisionContractError(f"{label}.value must be a string")
        return {"observable": observable, "op": value["op"], "value": value["value"]}
    if observable == "cell":
        if set(value) != {"observable", "x", "y", "op", "value"} or value.get("op") not in {"==", "!="}:
            raise DecisionContractError(f"{label} has an invalid cell assertion shape")
        if any(not _is_int(value.get(axis)) for axis in ("x", "y", "value")):
            raise DecisionContractError(f"{label} cell coordinates and value must be integers")
        x, y = int(value["x"]), int(value["y"])
        if x < 0 or y < 0:
            raise DecisionContractError(f"{label} cell coordinates must be nonnegative")
        if frame_shape is not None and (y >= frame_shape[0] or x >= frame_shape[1]):
            raise DecisionContractError(
                f"{label} cell ({x}, {y}) is outside width={frame_shape[1]}, height={frame_shape[0]}"
            )
        return {
            "observable": observable,
            "x": x,
            "y": y,
            "op": value["op"],
            "value": int(value["value"]),
        }
    raise DecisionContractError(f"{label}.observable is not publicly checkable")


def _bounded_text(value: Any, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DecisionContractError(f"{label} must be a nonempty string")
    cleaned = value.strip()
    if len(cleaned) > limit:
        raise DecisionContractError(f"{label} exceeds {limit} characters")
    return cleaned


def _compare(observed: Any, operator: str, expected: Any) -> bool:
    if operator == "==":
        return observed == expected
    if operator == "!=":
        return observed != expected
    if operator == "<":
        return observed < expected
    if operator == "<=":
        return observed <= expected
    if operator == ">":
        return observed > expected
    if operator == ">=":
        return observed >= expected
    raise ValueError(f"unknown comparison operator: {operator}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
