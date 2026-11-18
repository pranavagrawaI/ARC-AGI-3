"""Host-authored JSON probe copied into an isolated candidate evaluation bundle."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import host_planner_contract
import numpy as np
import wmlib

RESULT_PREFIX = "THEORY_EVALUATION_JSON="
obligations: list[dict] = []
runtime_errors: list[str] = []
unsupported = False


def obligation(obligation_id, kind, status, description, **diagnostics):
    global unsupported
    obligations.append(
        {
            "obligation_id": obligation_id,
            "kind": kind,
            "status": status,
            "description": description,
            "diagnostics": diagnostics,
        }
    )
    unsupported = unsupported or status == "unsupported"


def grid_hash(grid):
    value = np.asarray(grid, dtype=np.int16)
    return hashlib.sha256(value.tobytes()).hexdigest()[:16]


def transition_id(context, item):
    return (
        f"transition:{context}:L{item['level']}:T{item['turn']}:"
        f"{item.get('action')}:{item.get('row')}:{item.get('col')}:"
        f"{grid_hash(item['prev'])}:{grid_hash(item['next'])}"
    )


def render_result(state, expected, previous=None):
    try:
        predicted = np.asarray(world_model.render(state))
    except Exception as error:  # noqa: BLE001 - candidate failure is typed evidence
        message = f"render: {type(error).__name__}: {error}"
        runtime_errors.append(message)
        return "error", {"error": message}
    expected = np.asarray(expected)
    diagnostics = {
        "evaluated": True,
        "strict_exact": False,
        "prediction_coverage": 0.0,
        "first_difference": None,
    }
    if predicted.shape != expected.shape:
        diagnostics["shape"] = [list(predicted.shape), list(expected.shape)]
        return "failed", diagnostics
    known = predicted != -1
    diagnostics["prediction_coverage"] = float(np.count_nonzero(known) / known.size)
    difference = np.argwhere(predicted != expected)
    if difference.size:
        diagnostics["first_difference"] = [int(value) for value in difference[0]]
    diagnostics["strict_exact"] = not bool(difference.size)
    if previous is not None:
        previous = np.asarray(previous)
        diagnostics["evaluated"] = not (
            np.array_equal(previous, expected)
            and np.array_equal(predicted, expected)
        )
    if not bool(np.all(known)):
        return "unsupported", diagnostics
    variants = getattr(world_model, "observation_variants", None)
    if difference.size and callable(variants):
        try:
            if variants(state):
                diagnostics["observation_variants_used"] = True
                return "unsupported", diagnostics
        except Exception as error:  # noqa: BLE001 - candidate failure is typed evidence
            message = f"observation_variants: {type(error).__name__}: {error}"
            runtime_errors.append(message)
            return "error", {**diagnostics, "error": message}
    return ("passed" if diagnostics["strict_exact"] else "failed"), diagnostics


def safe_outcome(state, expected):
    try:
        actual = wmlib.outcome(world_model, state)
    except Exception as error:  # noqa: BLE001 - candidate failure is typed evidence
        message = f"outcome: {type(error).__name__}: {error}"
        runtime_errors.append(message)
        return "error", {"error": message, "expected": expected}
    return (
        "passed" if actual == expected else "failed",
        {"expected": expected, "actual": actual},
    )


def contexts():
    result = [("live", ".")]
    result.extend(
        (
            f"attempt-{attempt.get('level')}-{attempt.get('attempt')}",
            attempt["root"],
        )
        for attempt in wmlib.attempts()
    )
    return result


def evaluate_context(context, root):
    frames = wmlib.frames(root)
    by_level = {}
    for item in wmlib.transitions(root):
        by_level.setdefault(int(item["level"]), []).append(item)
    for level in sorted({key[0] for key in frames}):
        level_frames = {
            turn: grid for (item_level, turn), grid in frames.items() if item_level == level
        }
        first_turn = min(level_frames)
        initial_id = (
            f"initial:{context}:L{level}:T{first_turn}:"
            f"{grid_hash(level_frames[first_turn])}"
        )
        try:
            state = world_model.init_state(level_frames[first_turn], level)
        except Exception as error:  # noqa: BLE001 - candidate failure is typed evidence
            message = f"init_state L{level}: {type(error).__name__}: {error}"
            runtime_errors.append(message)
            obligation(
                initial_id,
                "initial_render",
                "error",
                f"{context} level {level} initial render",
                error=message,
            )
            state = None
        if state is not None:
            status, diagnostics = render_result(state, level_frames[first_turn])
            obligation(
                initial_id,
                "initial_render",
                status,
                f"{context} level {level} initial render",
                **diagnostics,
            )
        for item in sorted(by_level.get(level, []), key=lambda row: int(row["turn"])):
            decision_turn = int(item["turn"]) - 1
            outcome_id = f"outcome:{context}:L{level}:T{decision_turn}:ongoing"
            if state is None:
                obligation(
                    outcome_id,
                    "outcome",
                    "error",
                    f"{context} L{level} T{decision_turn} remains ongoing",
                    error="state unavailable",
                )
            else:
                status, diagnostics = safe_outcome(state, "ongoing")
                obligation(
                    outcome_id,
                    "outcome",
                    status,
                    f"{context} L{level} T{decision_turn} remains ongoing",
                    **diagnostics,
                )
            key = transition_id(context, item)
            if state is None:
                obligation(key, "transition", "error", key, error="state unavailable")
                continue
            try:
                state = world_model.transition(state, wmlib._act(item))
            except Exception as error:  # noqa: BLE001 - candidate failure is typed evidence
                message = f"transition {key}: {type(error).__name__}: {error}"
                runtime_errors.append(message)
                obligation(key, "transition", "error", key, error=message)
                state = None
                continue
            status, diagnostics = render_result(state, item["next"], item["prev"])
            obligation(key, "transition", status, key, **diagnostics)
        if context == "live" and state is not None:
            final_turn = max(level_frames)
            outcome_id = f"outcome:{context}:L{level}:T{final_turn}:ongoing"
            status, diagnostics = safe_outcome(state, "ongoing")
            obligation(
                outcome_id,
                "outcome",
                status,
                f"{context} L{level} T{final_turn} remains ongoing",
                **diagnostics,
            )


def evaluate_terminals():
    frames = wmlib.frames()
    by_level = {}
    for (level, turn), grid in frames.items():
        by_level.setdefault(level, {})[turn] = grid
    transitions = wmlib.transitions()
    for level, event in sorted(wmlib.terminal_events().items()):
        state = None
        try:
            level_frames = by_level[level]
            state = world_model.init_state(level_frames[min(level_frames)], level)
            for item in transitions:
                if item["level"] == level:
                    state = world_model.transition(state, wmlib._act(item))
            state = world_model.transition(state, event["action"])
        except Exception as error:  # noqa: BLE001 - candidate failure is typed evidence
            message = f"terminal L{level}: {type(error).__name__}: {error}"
            runtime_errors.append(message)
            state = None
        outcome_id = f"outcome:terminal:L{level}:level_complete"
        render_id = f"terminal-render:L{level}:{grid_hash(event['terminal_grid'])}"
        if state is None:
            obligation(
                outcome_id,
                "outcome",
                "error",
                f"host-confirmed terminal L{level}",
                error="state unavailable",
            )
            obligation(
                render_id,
                "terminal",
                "error",
                f"host-confirmed terminal render L{level}",
                error="state unavailable",
            )
            continue
        status, diagnostics = safe_outcome(state, "level_complete")
        obligation(
            outcome_id,
            "outcome",
            status,
            f"host-confirmed terminal L{level}",
            **diagnostics,
        )
        status, diagnostics = render_result(state, event["terminal_grid"])
        obligation(
            render_id,
            "terminal",
            status,
            f"host-confirmed terminal render L{level}",
            **diagnostics,
        )


def evaluate_planner_contract():
    report = host_planner_contract.wmgate_report(
        world_model, wmlib, source=MODEL_SOURCE, budget_s=0.1
    )
    error = report.get("error")
    obligation(
        "planner:contract",
        "planner_contract",
        "failed" if error else "passed",
        "planner entry points execute under the host contract audit",
        error=error,
        goal_status=report.get("goal_status"),
    )


try:
    MODEL_SOURCE = Path("world_model.py").read_text(encoding="utf-8")
    import world_model
except Exception as error:  # noqa: BLE001 - candidate load failure is conclusive
    print(
        RESULT_PREFIX
        + json.dumps(
            {
                "run_status": "candidate_load_failed",
                "error": f"{type(error).__name__}: {error}",
                "obligations": [],
            },
            sort_keys=True,
        )
    )
    sys.exit(0)

try:
    for context, root in contexts():
        evaluate_context(context, root)
    evaluate_terminals()
    evaluate_planner_contract()
except Exception as error:  # noqa: BLE001 - uncaught probe failure is infrastructure
    print(
        RESULT_PREFIX
        + json.dumps(
            {
                "run_status": "evaluator_unavailable",
                "error": f"{type(error).__name__}: {error}",
                "obligations": obligations,
            },
            sort_keys=True,
        )
    )
    sys.exit(0)

run_status = (
    "candidate_runtime_failed"
    if runtime_errors
    else "inconclusive_not_publishable"
    if unsupported
    else "complete"
)
print(
    RESULT_PREFIX
    + json.dumps(
        {
            "run_status": run_status,
            "error": "; ".join(runtime_errors) or None,
            "obligations": obligations,
        },
        sort_keys=True,
    )
)
