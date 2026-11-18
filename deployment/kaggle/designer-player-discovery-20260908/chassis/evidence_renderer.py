"""Measured-evidence and execution-generated discovery-example rendering.

The renderer deliberately keeps host observations separate from agent beliefs.
It is small enough to run in the deployed worker and has no model dependency.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Callable


def measured_evidence(agent) -> dict[str, Any]:
    """Return facts measured by the host for the current observation.

    ``None`` for the previous transition means that no transition has been
    executed; it is never rendered as a measured no-op.
    """
    state = agent.latest_state()
    previous = agent.last_transition(include_grids=True)
    return {
        "kind": "measured_evidence",
        "current": {
            key: state.get(key)
            for key in (
                "turn", "step", "level", "state", "levels_completed",
                "available_actions", "grid_rows",
            )
            if key in state
        },
        "grid_summary": agent.grid_summary(),
        "previous_transition": previous,
        "previous_transition_status": "measured" if previous is not None else "none_executed",
        "available_actions": list(state.get("available_actions", [])),
        "meaning": (
            "Measured host evidence only. Object identity, cause, objective, and "
            "hypotheses remain agent-authored."
        ),
    }


def render_measured_evidence(agent) -> str:
    """Render a bounded, explicit evidence block for the LLM prompt."""
    evidence = measured_evidence(agent)
    previous = evidence["previous_transition"]
    lines = [
        "<measured_evidence>",
        "Host-measured facts; do not treat interpretations as facts.",
        f"current={json.dumps(evidence['current'], sort_keys=True)}",
        f"grid_summary={json.dumps(evidence['grid_summary'], sort_keys=True)}",
        f"available_actions={json.dumps(evidence['available_actions'])}",
    ]
    if previous is None:
        lines.append("previous_transition=none_executed (no action transition exists yet)")
    else:
        lines.append(f"previous_transition={json.dumps(previous, sort_keys=True)}")
    lines.append(evidence["meaning"])
    lines.append("</measured_evidence>")
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _capture(name: str, fn: Callable, *args, **kwargs) -> dict[str, Any]:
    value = fn(*args, **kwargs)
    return {"name": name, "value": _jsonable(value), "return_type": type(value).__name__}


def run_two_decision_fixture(agent, *, initial_grid: list[list[int]] | None = None) -> dict[str, Any]:
    """Exercise the real compact tools across two decisions and a host transition.

    The first decision observes and submits one action. Only after submission does
    this fixture write the successor transition, mirroring the deployed protocol.
    The second decision measures that transition and records a narrow update.
    """
    initial = initial_grid or [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    successor = [row[:] for row in initial]
    successor[0][1] = successor[1][1]
    successor[1][1] = 0
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    state0 = {
        "turn": 0, "step": 0, "level": 0, "state": "NOT_FINISHED",
        "levels_completed": 0, "available_actions": actions,
        "grid": initial, "grid_rows": ["".join(format(v, "x") for v in row) for row in initial],
    }
    agent._states_path.write_text(json.dumps(state0) + "\n")
    first = [
        _capture("model_grid", agent.model_grid),
        _capture("grid_summary", agent.grid_summary),
        _capture("changed_region", agent.changed_region),
        _capture("preflight_model", agent.preflight_model, ["UP"]),
        _capture(
            "set_experiment", agent.set_experiment,
            "Does UP translate the marked region?",
            ["translation", "no_effect"], "UP",
            {"moved": "supports translation in this tested situation",
             "unchanged": "does not support translation here",
             "inconclusive": "keep both provisional"},
        ),
    ]
    try:
        submit = agent.submit_actions(["UP"], rationale="predict: test one-cell translation")
    except BaseException as exc:
        # CodeAct uses a control-flow BaseException to end a decision after a
        # successful submit.  The fixture must cross that same boundary rather
        # than replacing it with a direct actions-file write.
        if type(exc).__name__ != "_ReturnResultSignal":
            raise
        submit = getattr(exc, "result", {"kind": "WAIT"})
    first.append({"name": "submit_actions", "value": submit, "return_type": type(submit).__name__})

    transition = {
        "step": 1, "turn": 0, "action": "UP",
        "before": {"grid": initial},
        "after": {"grid": successor, "state": "NOT_FINISHED", "levels_completed": 0},
    }
    (agent.ipc_dir / "transitions.jsonl").write_text(json.dumps(transition) + "\n")
    state1 = dict(state0, turn=1, step=1, grid=successor,
                  grid_rows=["".join(format(v, "x") for v in row) for row in successor])
    with agent._states_path.open("a") as stream:
        stream.write(json.dumps(state1) + "\n")
    second = [
        _capture("last_transition", agent.last_transition, True),
        _capture("changed_region", agent.changed_region),
        _capture("grid_summary", agent.grid_summary),
        _capture("model_grid", agent.model_grid),
        _capture(
            "set_experiment", agent.set_experiment,
            "Did the measured change support the translation hypothesis?",
            ["translation", "other_change"], "DOWN",
            {"shape_preserved": "supports translation in this tested situation",
             "shape_changed": "revise the interpretation",
             "inconclusive": "retain the unresolved question"},
        ),
    ]
    signatures = {
        name: str(inspect.signature(getattr(agent, name)))
        for name in ("model_grid", "grid_summary", "changed_region", "last_transition", "submit_actions")
    }
    return {
        "schema": "discovery-fixture-v1",
        "protocol": "submit then host transition then measure",
        "api_signatures": signatures,
        "first_decision": first,
        "host_transition": transition,
        "second_decision": second,
        "negative_boundary": (
            "The measured movement supports translation in this tested situation; "
            "it does not establish other controls, collision behavior, or the objective."
        ),
    }


def render_discovery_demonstration(trace: dict[str, Any]) -> str:
    """Render a demonstration from captured tool outputs, never a hand transcript."""
    first = {item["name"]: item["value"] for item in trace["first_decision"]}
    second = {item["name"]: item["value"] for item in trace["second_decision"]}
    return "\n".join([
        "<discovery_demonstration>",
        "This example was generated by the passing two-decision integration fixture.",
        "Decision 1: inspect the actual grid and compact measurements, state two provisional hypotheses, then submit one action.",
        f"api_signatures={json.dumps(trace['api_signatures'], sort_keys=True)}",
        f"model_grid_return_type={next(item['return_type'] for item in trace['first_decision'] if item['name'] == 'model_grid')}",
        f"grid_summary={json.dumps(first['grid_summary'], sort_keys=True)}",
        f"changed_region_before_action={json.dumps(first['changed_region'], sort_keys=True)}",
        f"preflight={json.dumps(first['preflight_model'], sort_keys=True)}",
        f"action_result={first['submit_actions']!r}",
        "Host transition: the action is executed before the successor observation is delivered.",
        f"transition={json.dumps(trace['host_transition'], sort_keys=True)}",
        "Decision 2: use the measured before/after evidence, then update only the supported claim.",
        f"last_transition={json.dumps(second['last_transition'], sort_keys=True)}",
        f"changed_region_after_action={json.dumps(second['changed_region'], sort_keys=True)}",
        f"grid_summary_after={json.dumps(second['grid_summary'], sort_keys=True)}",
        f"negative_boundary={trace['negative_boundary']}",
        "Do not copy the conclusion when the measured result changes; reuse the procedure.",
        "</discovery_demonstration>",
    ])


def write_fixture_artifacts(agent, output_dir: str | Path) -> tuple[Path, Path]:
    """Run the fixture and write raw JSON plus the prompt-ready demonstration."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace = run_two_decision_fixture(agent)
    raw = output / "fixture-trace.json"
    raw.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    demo = output / "discovery-demo.md"
    demo.write_text(render_discovery_demonstration(trace) + "\n")
    return raw, demo
