"""Fresh-process evaluator for Muse-authored executable commitments.

This module is a trusted entry point.  It installs OS-enforced restrictions
before loading candidate source, then emits one bounded JSON result on stdout.
Candidate output is never part of that protocol.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import resource
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from nooa.runtime.sandbox.config import LandlockRule, ResolvedSpec
from nooa.runtime.sandbox.guards import install_guards, probe_capabilities

UNKNOWN = -1
ALLOWED_OUTCOMES = frozenset({"ongoing", "level_complete", "game_over"})
PROTOCOL = "wm1-evaluator-v1"
MOORE_NAMES = ("State", "init_state", "transition", "render", "outcome")


class EvaluationFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class BoundedText:
    """Discard candidate output while failing deterministically at the ceiling."""

    def __init__(self, limit: int = 16_384) -> None:
        self.limit = limit
        self.count = 0

    def write(self, value: str) -> int:
        self.count += len(value.encode("utf-8", errors="replace"))
        if self.count > self.limit:
            raise EvaluationFailure("OUTPUT_LIMIT", "candidate output exceeded limit")
        return len(value)

    def flush(self) -> None:
        return None


@dataclass
class ReplayState:
    attempt_id: str
    state: Any


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    bundle = Path(sys.argv[1]).resolve()
    try:
        evidence = json.loads((bundle / "evidence.json").read_text(encoding="utf-8"))
        _install_containment(bundle)
        module, contract_kind = _load_candidate(bundle / "source.py")
        evaluation = (
            _evaluate_partial_predictor(bundle, module, evidence)
            if contract_kind == "partial_predictor"
            else _evaluate_moore_machine(bundle, module, evidence)
        )
        result = {
            "protocol": PROTOCOL,
            "operational": {
                "status": "valid",
                "failure_code": None,
                "contract_kind": contract_kind,
            },
            **evaluation,
        }
    except MemoryError:
        result = {
            "protocol": PROTOCOL,
            "operational": {
                "status": "invalid",
                "failure_code": "MEMORY_LIMIT",
                "message": "candidate exceeded memory limit",
            },
        }
    except EvaluationFailure as error:
        result = {
            "protocol": PROTOCOL,
            "operational": {
                "status": "invalid",
                "failure_code": error.code,
                "message": str(error)[:240],
            },
        }
    except BaseException as error:  # noqa: BLE001 - emit bounded failure, never a traceback
        result = {
            "protocol": PROTOCOL,
            "operational": {
                "status": "invalid",
                "failure_code": "EVALUATOR_FAILURE",
                "message": type(error).__name__,
            },
        }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if len(payload) > 256_000:
        payload = json.dumps(
            {
                "protocol": PROTOCOL,
                "operational": {
                    "status": "invalid",
                    "failure_code": "RESULT_LIMIT",
                    "message": "structured result exceeded limit",
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    os.write(1, payload.encode("utf-8") + b"\n")
    return 0


def _install_containment(bundle: Path) -> None:
    capabilities = probe_capabilities()
    if not (capabilities.filesystem and capabilities.network and capabilities.rlimit):
        raise EvaluationFailure(
            "ISOLATION_UNAVAILABLE", "required Linux isolation is unavailable"
        )

    read_roots = {str(bundle), sys.prefix, sys.base_prefix, "/usr", "/lib", "/lib64"}
    rules = tuple(
        LandlockRule(path, write=False, required=path == str(bundle))
        for path in sorted(read_roots)
        if Path(path).exists()
    )
    rules = (*rules, LandlockRule(str(bundle / "tmp"), write=True, required=True))
    install_guards(
        ResolvedSpec(
            filesystem=True,
            landlock_rules=rules,
            block_network=True,
            max_memory_mb=192,
            # Fresh NumPy imports can consume more than two CPU seconds on a
            # cold filesystem.  The parent still enforces the tighter wall
            # bound; this ceiling exists to stop compute after parent death.
            max_cpu_seconds=5,
        )
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    # One process already exists.  A limit of one denies fork/exec while not
    # confusing runtimes which count the calling process against the limit.
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))


def _load_candidate(path: Path) -> tuple[Any, str]:
    spec = importlib.util.spec_from_file_location("wm0_candidate", path)
    if spec is None or spec.loader is None:
        raise EvaluationFailure("LOAD_FAILED", "candidate module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    output = BoundedText()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        spec.loader.exec_module(module)
    if all(hasattr(module, name) for name in MOORE_NAMES):
        for name in ("init_state", "transition", "render", "outcome"):
            if not callable(getattr(module, name)):
                raise EvaluationFailure(
                    "MISSING_CONTRACT", f"required name is not callable: {name}"
                )
        return module, "moore_machine"
    if callable(getattr(module, "predict", None)):
        return module, "partial_predictor"
    raise EvaluationFailure(
        "MISSING_CONTRACT",
        "define predict(grid, action), or the complete Moore-machine contract",
    )


def _grid(bundle: Path, digest: str) -> np.ndarray:
    path = bundle / "blobs" / f"grid-{digest}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        grid = np.asarray(value, dtype=np.int16)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise EvaluationFailure("INVALID_EVIDENCE", f"unreadable grid {digest}") from error
    if grid.ndim != 2 or not grid.size:
        raise EvaluationFailure("INVALID_EVIDENCE", f"invalid grid {digest}")
    return grid


def _render(module: Any, state: Any, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(module.render(state))
    if value.shape != shape:
        raise EvaluationFailure(
            "CONTRACT_RENDER_SHAPE",
            f"render shape {value.shape!r} differs from evidence shape {shape!r}",
        )
    if not np.issubdtype(value.dtype, np.integer):
        raise EvaluationFailure("CONTRACT_RENDER_DTYPE", "render must be integer-valued")
    value = value.astype(np.int16, copy=False)
    if np.any((value < UNKNOWN) | (value > 15)):
        raise EvaluationFailure("CONTRACT_RENDER_RANGE", "render values must be -1..15")
    return value


def _prediction(
    module: Any, grid: np.ndarray, action: dict[str, Any]
) -> tuple[np.ndarray, bool]:
    """Normalize one quick-theory prediction; None is complete abstention."""
    raw = module.predict(grid.copy(), action)
    if raw is None:
        return np.full(grid.shape, UNKNOWN, dtype=np.int16), True
    value = np.asarray(raw)
    if value.shape != grid.shape:
        raise EvaluationFailure(
            "CONTRACT_PREDICT_SHAPE",
            f"predict shape {value.shape!r} differs from evidence shape {grid.shape!r}",
        )
    if not np.issubdtype(value.dtype, np.integer):
        raise EvaluationFailure("CONTRACT_PREDICT_DTYPE", "predict must be integer-valued")
    value = value.astype(np.int16, copy=False)
    if np.any((value < UNKNOWN) | (value > 15)):
        raise EvaluationFailure("CONTRACT_PREDICT_RANGE", "predict values must be -1..15")
    return value, False


def _score_grid(predicted: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    known = predicted != UNKNOWN
    known_count = int(known.sum())
    total = int(predicted.size)
    matched = int(((predicted == observed) & known).sum())
    exact = bool(known_count == total and np.array_equal(predicted, observed))
    return {
        "exact": exact,
        "accepted": bool(known_count and matched == known_count),
        "known_cell_accuracy": (matched / known_count if known_count else None),
        "strict_cell_accuracy": int((predicted == observed).sum()) / total,
        "prediction_coverage": known_count / total,
        "matched_known_cells": matched,
        "known_cells": known_count,
        "total_cells": total,
    }


def _render_candidates(
    module: Any, state: Any, shape: tuple[int, ...]
) -> list[np.ndarray]:
    canonical = _render(module, state, shape)
    function = getattr(module, "observation_variants", None)
    if function is None:
        return [canonical]
    if not callable(function):
        raise EvaluationFailure(
            "CONTRACT_VARIANTS", "observation_variants must be callable when present"
        )
    raw = function(state)
    if raw is None:
        return [canonical]
    if not isinstance(raw, (list, tuple)) or len(raw) > 5:
        raise EvaluationFailure(
            "CONTRACT_VARIANTS", "observation_variants must return at most five grids"
        )
    variants = [canonical]
    unknown_mask = canonical == UNKNOWN
    for item in raw:
        variant = np.asarray(item)
        if variant.shape != shape or not np.issubdtype(variant.dtype, np.integer):
            raise EvaluationFailure(
                "CONTRACT_VARIANTS", "observation variant has invalid shape or dtype"
            )
        variant = variant.astype(np.int16, copy=False)
        if np.any((variant < UNKNOWN) | (variant > 15)):
            raise EvaluationFailure(
                "CONTRACT_VARIANTS", "observation variant values must be -1..15"
            )
        if not np.array_equal(variant == UNKNOWN, unknown_mask):
            raise EvaluationFailure(
                "CONTRACT_VARIANTS", "observation variants must preserve UNKNOWN mask"
            )
        variants.append(variant)
    return variants


def _best_score(candidates: list[np.ndarray], observed: np.ndarray) -> tuple[dict[str, Any], int]:
    scores = [_score_grid(candidate, observed) for candidate in candidates]
    index = max(
        range(len(scores)),
        key=lambda item: (
            scores[item]["accepted"],
            scores[item]["known_cell_accuracy"] or 0.0,
            scores[item]["strict_cell_accuracy"],
            -item,
        ),
    )
    return scores[index], index


def _delta(before: np.ndarray, after: np.ndarray, *, limit: int = 128) -> dict[str, Any]:
    locations = np.argwhere(before != after)
    cells = [
        [int(row), int(column), int(before[row, column]), int(after[row, column])]
        for row, column in locations[:limit]
    ]
    return {
        "cells": cells,
        "cell_count": len(locations),
        "truncated": len(locations) > limit,
    }


def _action(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": value["label"],
        "x": value.get("x"),
        "y": value.get("y"),
    }


def _evaluate_partial_predictor(
    bundle: Path, module: Any, evidence: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate independent, explicitly partial next-grid commitments."""
    output = BoundedText()
    evaluated = 0
    claimed = 0
    matched = 0
    changing = 0
    no_op = 0
    known_cells = 0
    matched_known_cells = 0
    total_cells = 0
    execution_errors: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    per_attempt: dict[str, dict[str, int]] = {}
    per_level: dict[str, dict[str, int]] = {}
    per_action: dict[str, dict[str, int | str]] = {}
    transition_results: list[dict[str, Any]] = []

    # Loading proves the callable exists. Replay, rather than an invented dummy
    # action, supplies the first meaningful contract execution.
    os.write(2, b"WM1_PHASE empirical\n")
    for transition in evidence.get("transitions", []):
        if transition["boundary"] != "none":
            continue
        before = _grid(bundle, transition["before_grid_sha256"])
        after = _grid(bundle, transition["after_grid_sha256"])
        action = _action(transition["action"])
        action_label = action["action"]
        attempt_id = transition["attempt_before"]
        evaluated += 1
        changing += int(not np.array_equal(before, after))
        no_op += int(np.array_equal(before, after))
        attempt_partition = per_attempt.setdefault(
            attempt_id, {"evaluated": 0, "claimed": 0, "matched": 0, "errors": 0}
        )
        level_partition = per_level.setdefault(
            str(transition["levels_completed_before"]),
            {"evaluated": 0, "claimed": 0, "matched": 0, "errors": 0},
        )
        action_partition = per_action.setdefault(
            action_label,
            {"evaluated": 0, "claimed": 0, "matched": 0, "errors": 0},
        )
        for partition in (attempt_partition, level_partition, action_partition):
            partition["evaluated"] += 1
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                predicted, whole_action_abstention = _prediction(
                    module, before, action
                )
        except EvaluationFailure:
            raise
        except BaseException as error:  # noqa: BLE001 - candidate failures are evidence
            execution_errors.append(
                {
                    "source_event_id": transition["source_event_id"],
                    "error_type": type(error).__name__,
                    "message": str(error)[:240],
                }
            )
            for partition in (attempt_partition, level_partition, action_partition):
                partition["errors"] += 1
            if first_divergence is None:
                first_divergence = {
                    "source_event_id": transition["source_event_id"],
                    "action": transition["action"],
                    "kind": "execution_error",
                }
            continue

        score = _score_grid(predicted, after)
        claim = (
            "whole_action_abstention"
            if whole_action_abstention
            else "no_cell_claim"
            if not score["known_cells"]
            else "full_cell_claim"
            if score["known_cells"] == score["total_cells"]
            else "partial_cell_claim"
        )
        transition_results.append(
            {
                "source_event_id": transition["source_event_id"],
                "action": action_label,
                "claim": claim,
                "known_cells": score["known_cells"],
                "total_cells": score["total_cells"],
                "matched": score["accepted"] if score["known_cells"] else None,
            }
        )
        known_cells += score["known_cells"]
        matched_known_cells += score["matched_known_cells"]
        total_cells += score["total_cells"]
        if not score["known_cells"]:
            continue
        claimed += 1
        for partition in (attempt_partition, level_partition, action_partition):
            partition["claimed"] += 1
        if score["accepted"]:
            matched += 1
            for partition in (attempt_partition, level_partition, action_partition):
                partition["matched"] += 1
        elif first_divergence is None:
            observed_delta = _delta(before, after)
            predicted_delta = _delta(before, predicted)
            first_divergence = {
                "source_event_id": transition["source_event_id"],
                "action": transition["action"],
                "kind": "grid_mismatch",
                "observed_delta_cells": [cell[:2] for cell in observed_delta["cells"][:32]],
                "predicted_delta_cells": [cell[:2] for cell in predicted_delta["cells"][:32]],
                "observed_delta": observed_delta,
                "predicted_delta": predicted_delta,
                **score,
            }

    for partition in per_action.values():
        partition["status"] = (
            "contradicted"
            if partition["errors"] or partition["claimed"] != partition["matched"]
            else "supported"
            if partition["claimed"]
            else "inconclusive"
        )
    dynamics_status = (
        "contradicted"
        if execution_errors or claimed != matched
        else "supported"
        if claimed
        else "inconclusive"
    )
    return {
        "status": dynamics_status,
        "contract_kind": "partial_predictor",
        "initial_render": {"status": "inconclusive", "evaluated": 0, "matched": 0},
        "dynamics": {
            "status": dynamics_status,
            "evaluated": evaluated,
            "claimed": claimed,
            "matched": matched,
            "changing": changing,
            "no_op": no_op,
            "known_cell_accuracy": (
                matched_known_cells / known_cells if known_cells else None
            ),
            "prediction_coverage": known_cells / total_cells if total_cells else None,
            "observation_variant_matches": 0,
            "per_attempt": per_attempt,
            "per_level": per_level,
            "per_action": per_action,
            "transition_results": transition_results,
            "execution_errors": execution_errors,
        },
        "outcome": {
            "status": "inconclusive",
            "evaluated": 0,
            "matched": 0,
            "terminal_evidence": 0,
            "false_positives": 0,
            "false_negatives": 0,
        },
        "first_divergence": first_divergence,
    }


def _evaluate_moore_machine(
    bundle: Path, module: Any, evidence: dict[str, Any]
) -> dict[str, Any]:
    records = json.loads(evidence["occurrence_records_json"])
    observations: dict[str, dict[str, Any]] = {}
    for record in records:
        if "observation" in record:
            observation = record["observation"]
            observations[observation["observation_id"]] = observation
        if "after_observation" in record:
            observation = record["after_observation"]
            observations[observation["observation_id"]] = observation

    if not observations:
        raise EvaluationFailure("INVALID_EVIDENCE", "snapshot has no observations")

    first = next(iter(observations.values()))
    smoke_grid = _grid(bundle, first["grid_sha256"])
    output = BoundedText()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        smoke_state = module.init_state(smoke_grid.copy(), int(first["levels_completed"]))
        _render_candidates(module, smoke_state, smoke_grid.shape)
        smoke_outcome = module.outcome(smoke_state)
    if smoke_outcome not in ALLOWED_OUTCOMES:
        raise EvaluationFailure("CONTRACT_OUTCOME", "outcome returned an unsupported value")

    # The parent uses this trusted phase boundary to distinguish a candidate
    # that never passed bounded contract smoke from one that failed only while
    # replaying a particular observation.  Candidate stderr is redirected.
    os.write(2, b"WM1_PHASE empirical\n")

    initial_evaluated = 0
    initial_matched = 0
    initial_known_cells = 0
    initial_total_cells = 0
    dynamics_evaluated = 0
    dynamics_matched = 0
    dynamics_changing = 0
    dynamics_noop = 0
    dynamics_known_cells = 0
    dynamics_matched_known_cells = 0
    dynamics_total_cells = 0
    variant_matches = 0
    terminal_evidence = 0
    outcome_evaluated = 0
    outcome_matched = 0
    false_positive = 0
    false_negative = 0
    execution_errors: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    replay: ReplayState | None = None
    per_attempt: dict[str, dict[str, int]] = {}
    per_level: dict[str, dict[str, int]] = {}
    transition_results: list[dict[str, Any]] = []

    transitions = evidence.get("transitions", [])
    for transition in transitions:
        # Reset boundaries connect different replay segments; they are public
        # lifecycle evidence, not a learned ordinary dynamic.
        if transition["boundary"] != "none":
            replay = None
            continue

        before = _grid(bundle, transition["before_grid_sha256"])
        after = _grid(bundle, transition["after_grid_sha256"])
        attempt_id = transition["attempt_before"]
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                if replay is None or replay.attempt_id != attempt_id:
                    state = module.init_state(
                        before.copy(), int(transition["levels_completed_before"])
                    )
                    initial_candidates = _render_candidates(module, state, before.shape)
                    initial_score, _ = _best_score(initial_candidates, before)
                    initial_evaluated += 1
                    initial_known_cells += initial_score["known_cells"]
                    initial_total_cells += initial_score["total_cells"]
                    if initial_score["accepted"]:
                        initial_matched += 1
                    replay = ReplayState(attempt_id, state)

                predicted_state = module.transition(replay.state, _action(transition["action"]))
                predicted_candidates = _render_candidates(
                    module, predicted_state, after.shape
                )
                predicted_outcome = module.outcome(predicted_state)
            if predicted_outcome not in ALLOWED_OUTCOMES:
                raise EvaluationFailure(
                    "CONTRACT_OUTCOME", "outcome returned an unsupported value"
                )
            replay.state = predicted_state
        except EvaluationFailure:
            raise
        except BaseException as error:  # noqa: BLE001 - candidate failures are evidence
            execution_errors.append(
                {
                    "source_event_id": transition["source_event_id"],
                    "error_type": type(error).__name__,
                    "message": str(error)[:240],
                }
            )
            if first_divergence is None:
                first_divergence = {
                    "source_event_id": transition["source_event_id"],
                    "action": transition["action"],
                    "kind": "execution_error",
                }
            replay = None
            continue

        score, variant_index = _best_score(predicted_candidates, after)
        predicted = predicted_candidates[variant_index]
        transition_results.append(
            {
                "source_event_id": transition["source_event_id"],
                "action": transition["action"]["label"],
                "claim": (
                    "no_cell_claim"
                    if not score["known_cells"]
                    else "full_cell_claim"
                    if score["known_cells"] == score["total_cells"]
                    else "partial_cell_claim"
                ),
                "known_cells": score["known_cells"],
                "total_cells": score["total_cells"],
                "matched": score["accepted"] if score["known_cells"] else None,
            }
        )
        dynamics_evaluated += 1
        dynamics_changing += int(not np.array_equal(before, after))
        dynamics_noop += int(np.array_equal(before, after))
        dynamics_known_cells += score["known_cells"]
        dynamics_matched_known_cells += score["matched_known_cells"]
        dynamics_total_cells += score["total_cells"]
        variant_matches += int(variant_index > 0 and score["accepted"])
        attempt_partition = per_attempt.setdefault(
            attempt_id, {"evaluated": 0, "matched": 0, "errors": 0}
        )
        level_partition = per_level.setdefault(
            str(transition["levels_completed_before"]),
            {"evaluated": 0, "matched": 0, "errors": 0},
        )
        attempt_partition["evaluated"] += 1
        level_partition["evaluated"] += 1
        if score["accepted"]:
            dynamics_matched += 1
            attempt_partition["matched"] += 1
            level_partition["matched"] += 1
        elif first_divergence is None:
            observed_delta = _delta(before, after)
            predicted_delta = _delta(before, predicted)
            first_divergence = {
                "source_event_id": transition["source_event_id"],
                "action": transition["action"],
                "kind": "grid_mismatch",
                "observed_delta_cells": [cell[:2] for cell in observed_delta["cells"][:32]],
                "predicted_delta_cells": [cell[:2] for cell in predicted_delta["cells"][:32]],
                "observed_delta": observed_delta,
                "predicted_delta": predicted_delta,
                "accepted_variant_index": variant_index if variant_index else None,
                **score,
            }

        expected_outcome = transition["outcome"]
        outcome_evaluated += 1
        terminal_evidence += int(expected_outcome != "ongoing")
        if predicted_outcome == expected_outcome:
            outcome_matched += 1
        else:
            false_positive += int(expected_outcome == "ongoing" and predicted_outcome != "ongoing")
            false_negative += int(expected_outcome != "ongoing" and predicted_outcome == "ongoing")
            if first_divergence is None:
                first_divergence = {
                    "source_event_id": transition["source_event_id"],
                    "action": transition["action"],
                    "kind": "outcome_mismatch",
                    "observed_outcome": expected_outcome,
                    "predicted_outcome": predicted_outcome,
                }

    initial_status = (
        "supported"
        if initial_evaluated and initial_evaluated == initial_matched and initial_known_cells
        else "contradicted"
        if initial_evaluated and initial_known_cells
        else "inconclusive"
    )
    dynamics_status = (
        "contradicted"
        if execution_errors or (
            dynamics_evaluated
            and dynamics_known_cells
            and dynamics_matched != dynamics_evaluated
        )
        else "supported"
        if dynamics_evaluated and dynamics_known_cells
        else "inconclusive"
    )
    outcome_status = (
        "contradicted"
        if outcome_evaluated and outcome_matched != outcome_evaluated
        else "supported"
        if terminal_evidence and outcome_evaluated
        else "inconclusive"
    )
    contradicted = "contradicted" in {
        initial_status,
        dynamics_status,
        outcome_status,
    }
    all_supported = all(
        status == "supported"
        for status in (initial_status, dynamics_status, outcome_status)
    )
    return {
        "status": "contradicted" if contradicted else "supported" if all_supported else "inconclusive",
        "contract_kind": "moore_machine",
        "initial_render": {
            "status": initial_status,
            "evaluated": initial_evaluated,
            "matched": initial_matched,
            "prediction_coverage": (
                initial_known_cells / initial_total_cells if initial_total_cells else None
            ),
        },
        "dynamics": {
            "status": dynamics_status,
            "evaluated": dynamics_evaluated,
            "matched": dynamics_matched,
            "changing": dynamics_changing,
            "no_op": dynamics_noop,
            "known_cell_accuracy": (
                dynamics_matched_known_cells / dynamics_known_cells
                if dynamics_known_cells
                else None
            ),
            "prediction_coverage": (
                dynamics_known_cells / dynamics_total_cells
                if dynamics_total_cells
                else None
            ),
            "observation_variant_matches": variant_matches,
            "per_attempt": per_attempt,
            "per_level": per_level,
            "transition_results": transition_results,
            "execution_errors": execution_errors,
        },
        "outcome": {
            "status": outcome_status,
            "evaluated": outcome_evaluated,
            "matched": outcome_matched,
            "terminal_evidence": terminal_evidence,
            "false_positives": false_positive,
            "false_negatives": false_negative,
        },
        "first_divergence": first_divergence,
    }


if __name__ == "__main__":
    raise SystemExit(main())
