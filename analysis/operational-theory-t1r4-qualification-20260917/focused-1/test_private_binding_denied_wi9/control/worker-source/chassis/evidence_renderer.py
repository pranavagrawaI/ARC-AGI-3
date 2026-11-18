"""Measured-evidence and execution-generated discovery-example rendering.

The renderer deliberately keeps host observations separate from agent beliefs.
It is small enough to run in the deployed worker and has no model dependency.
"""

from __future__ import annotations

import inspect
import hashlib
import json
import contextlib
import io
from pathlib import Path
from typing import Any, Callable


def measured_evidence(agent) -> dict[str, Any]:
    """Return facts measured by the host for the current observation.

    ``None`` for the previous transition means that no transition has been
    executed; it is never rendered as a measured no-op.
    """
    state = agent.latest_state()
    # The exact grids remain available through model_grid()/last_transition(True).
    # The always-on capsule carries only deterministic, bounded measurements.
    previous = agent.last_transition(include_grids=False)
    packet = agent.decision_packet()
    evidence = {
        "schema": "measured-evidence-v2",
        "kind": "measured_evidence",
        "current": {
            key: state.get(key)
            for key in (
                "turn", "step", "level", "state", "levels_completed",
                "available_actions",
            )
            if key in state
        },
        "grid_summary": agent.grid_summary(),
        "previous_transition": previous,
        "previous_transition_status": "measured" if previous is not None else "none_executed",
        "available_actions": list(state.get("available_actions", [])),
        "unresolved_question": packet.get("experiment"),
        "meaning": (
            "Measured host evidence only. Object identity, cause, objective, and "
            "hypotheses remain agent-authored."
        ),
    }
    # Persist one capsule per distinct rendered fact set. This is evidence, not
    # conversational memory, and lets later analysis reproduce the exact suffix.
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
    evidence["sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    path = agent.ipc_dir / "evidence_capsules.jsonl"
    prior = None
    if path.exists():
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if lines:
            prior = json.loads(lines[-1]).get("sha256")
    if evidence["sha256"] != prior:
        with path.open("a") as stream:
            stream.write(json.dumps(evidence, sort_keys=True, allow_nan=False) + "\n")
    return evidence


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
        f"unresolved_question={json.dumps(evidence['unresolved_question'], sort_keys=True)}",
    ]
    if previous is None:
        lines.append("previous_transition=none_executed (no action transition exists yet)")
    else:
        lines.append(f"previous_transition={json.dumps(previous, sort_keys=True)}")
    lines.append(evidence["meaning"])
    lines.append("</measured_evidence>")
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value") and type(value).__module__ == "enum":
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _terminal_result(exc: BaseException) -> dict[str, Any]:
    result = getattr(exc, "result", None)
    kind = getattr(result, "kind", None)
    return {
        "kind": getattr(kind, "value", kind) or "WAIT",
        "explanation": getattr(result, "explanation", str(result) if result else ""),
    }


def _execute_cell(agent, source: str, captured_names: tuple[str, ...]) -> dict[str, Any]:
    """Execute and retain the exact source used by the fixture CodeAct cell."""
    namespace: dict[str, Any] = {"self": agent}
    output = io.StringIO()
    terminal = None
    with contextlib.redirect_stdout(output):
        try:
            exec(compile(source, "<fixture-codeact-cell>", "exec"), namespace)
        except BaseException as exc:
            if type(exc).__name__ != "_ReturnResultSignal":
                raise
            terminal = _terminal_result(exc)
    return {
        "source": source,
        "stdout": output.getvalue(),
        "captured": {name: _jsonable(namespace.get(name)) for name in captured_names},
        "terminal": terminal,
    }


_TRANSFER_CASES = {
    "translated_variant": {
        "initial": [[0, 0, 0], [0, 0, 0], [2, 0, 0]],
        "successor": [[0, 0, 0], [2, 0, 0], [0, 0, 0]],
        "probe_action": "UP", "expected_bbox": [1, 0, 2, 0],
        "alternative_bbox": [2, 0, 2, 1],
        "next_actions": {"translation_observed": "DOWN", "unexpected_region": "LEFT",
                         "shape_changed": "DOWN", "no_effect": "LEFT",
                         "alternative_region": "RIGHT"},
    },
    "changed_action_mapping": {
        "initial": [[0, 0, 0], [0, 3, 0], [0, 0, 0]],
        "successor": [[0, 3, 0], [0, 0, 0], [0, 0, 0]],
        "probe_action": "RIGHT", "expected_bbox": [1, 1, 1, 2],
        "alternative_bbox": [1, 0, 1, 1],
        "next_actions": {"translation_observed": "DOWN", "unexpected_region": "LEFT",
                         "shape_changed": "DOWN", "no_effect": "LEFT",
                         "alternative_region": "RIGHT"},
    },
    "reshaped_target": {
        "initial": [[0, 4], [0, 0]],
        "successor": [[0, 4, 0], [0, 4, 0]],
        "probe_action": "UP", "expected_bbox": [0, 1, 1, 1],
        "alternative_bbox": [0, 0, 0, 1],
        "next_actions": {"translation_observed": "DOWN", "unexpected_region": "LEFT",
                         "shape_changed": "DOWN", "no_effect": "LEFT",
                         "alternative_region": "RIGHT"},
    },
    "salient_noop": {
        "initial": [[0, 5, 0], [0, 0, 0]],
        "successor": [[0, 5, 0], [0, 0, 0]],
        "probe_action": "UP", "expected_bbox": [0, 1, 1, 1],
        "alternative_bbox": [0, 0, 0, 1],
        "next_actions": {"translation_observed": "DOWN", "unexpected_region": "LEFT",
                         "shape_changed": "DOWN", "no_effect": "LEFT",
                         "alternative_region": "RIGHT"},
    },
    "alternative_region": {
        "initial": [[6, 0, 0], [0, 0, 0], [0, 0, 0]],
        "successor": [[0, 6, 0], [0, 0, 0], [0, 0, 0]],
        "probe_action": "UP", "expected_bbox": [0, 0, 1, 0],
        "alternative_bbox": [0, 0, 0, 1],
        "next_actions": {"translation_observed": "DOWN", "unexpected_region": "LEFT",
                         "shape_changed": "DOWN", "no_effect": "LEFT",
                         "alternative_region": "RIGHT"},
    },
}


def run_two_decision_fixture(agent, *, case: str = "translated_variant") -> dict[str, Any]:
    """Exercise the real compact tools across two decisions and a host transition.

    The first decision observes and submits one action. Only after submission does
    this fixture write the successor transition, mirroring the deployed protocol.
    The second decision measures that transition and records a narrow update.
    """
    if case not in _TRANSFER_CASES:
        raise ValueError(f"unknown transfer case: {case}")
    scenario = _TRANSFER_CASES[case]
    initial = scenario["initial"]
    successor = scenario["successor"]
    probe_action = scenario["probe_action"]
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    state0 = {
        "turn": 0, "step": 0, "level": 0, "state": "NOT_FINISHED",
        "levels_completed": 0, "available_actions": actions,
        "grid": initial, "grid_rows": ["".join(format(v, "x") for v in row) for row in initial],
    }
    agent._states_path.write_text(json.dumps(state0) + "\n")
    first_source = "\n".join([
        "grid = self.model_grid()",
        "summary = self.grid_summary()",
        "question = 'Which observable effect follows the probe action?'",
        "hypotheses = ['some_grid_change', 'no_grid_change']",
        f"action = {probe_action!r}",
        "outcomes = {'target': 'any_change', 'metric': 'grid_changed', "
        "'predictions': {'some_grid_change': True, 'no_grid_change': False}, "
        "'updates': {'some_grid_change': 'support an effect here', "
        "'no_grid_change': 'reject an effect here'}, "
        "'otherwise': 'revise the measurement'}",
        "experiment = self.set_experiment(question, hypotheses, action, outcomes)",
        "print({'summary': summary, 'experiment': experiment})",
        "self.submit_actions([action], rationale='predict: discriminate two observable regions')",
    ])
    first = _execute_cell(agent, first_source, ("summary", "question", "hypotheses", "action", "outcomes", "experiment"))

    transition = {
        "step": 1, "turn": 0, "action": probe_action,
        "before": {"grid": initial},
        "after": {"grid": successor, "state": "NOT_FINISHED", "levels_completed": 0},
    }
    (agent.ipc_dir / "transitions.jsonl").write_text(json.dumps(transition) + "\n")
    state1 = dict(state0, turn=1, step=1, grid=successor,
                  grid_rows=["".join(format(v, "x") for v in row) for row in successor])
    with agent._states_path.open("a") as stream:
        stream.write(json.dumps(state1) + "\n")
    second_source = "\n".join([
        "transition = self.last_transition(True)",
        "region = self.changed_region()",
        f"expected_bbox = {scenario['expected_bbox']!r}",
        f"alternative_bbox = {scenario['alternative_bbox']!r}",
        "if transition['shape_changed']:",
        "    update = 'shape_changed'",
        "elif transition['changed_cells'] == 0:",
        "    update = 'no_effect'",
        "elif transition['changed_bbox'] == expected_bbox:",
        "    update = 'translation_observed'",
        "elif transition['changed_bbox'] == alternative_bbox:",
        "    update = 'alternative_region'",
        "else:",
        "    update = 'unexpected_region'",
        f"next_actions = {scenario['next_actions']!r}",
        "next_action = next_actions[update]",
        "question = 'Which remaining mapping explains the next measured region?'",
        "hypotheses = ['first_mapping', 'alternative_mapping']",
        "outcomes = {'target': 'region', 'metric': 'changed_bbox', "
        "'predictions': {'first_mapping': expected_bbox, 'alternative_mapping': alternative_bbox}, "
        "'updates': {'first_mapping': 'support the first regional mapping', "
        "'alternative_mapping': 'support the alternative regional mapping'}, "
        "'otherwise': 'revise the measurement'}",
        "experiment = self.set_experiment(question, hypotheses, next_action, outcomes)",
        "print({'transition': transition, 'update': update, 'next_action': next_action})",
        "self.submit_actions([next_action], rationale='predict: discriminate the remaining observable mappings')",
    ])
    second = _execute_cell(agent, second_source, ("transition", "region", "update", "next_action", "experiment"))
    signatures = {
        name: str(inspect.signature(getattr(agent, name)))
        for name in ("model_grid", "grid_summary", "changed_region", "last_transition", "set_experiment", "submit_actions")
    }
    return {
        "schema": "discovery-fixture-v2",
        "case": case,
        "protocol": "submit then host transition then measure",
        "api_signatures": signatures,
        "code_cells": [first, second],
        "host_transition": transition,
        "second_action": second["captured"]["next_action"],
        "negative_boundary": (
            "The measured movement supports translation in this tested situation; "
            "it does not establish other controls, collision behavior, or the objective."
        ),
    }


def render_discovery_demonstration(trace: dict[str, Any]) -> str:
    """Render a demonstration from captured tool outputs, never a hand transcript."""
    first, second = trace["code_cells"]
    return "\n".join([
        "<discovery_demonstration>",
        "This example was generated by the passing two-decision integration fixture.",
        "Decision 1: inspect the actual grid and compact measurements, state two provisional hypotheses, then submit one action.",
        f"api_signatures={json.dumps(trace['api_signatures'], sort_keys=True)}",
        "```python",
        first["source"],
        "```",
        "```output",
        first["stdout"].rstrip(),
        json.dumps(first["terminal"], sort_keys=True),
        "```",
        "Host transition: the action is executed before the successor observation is delivered.",
        f"transition={json.dumps(trace['host_transition'], sort_keys=True)}",
        "Decision 2: use the measured before/after evidence, then update only the supported claim.",
        "```python",
        second["source"],
        "```",
        "```output",
        second["stdout"].rstrip(),
        json.dumps(second["terminal"], sort_keys=True),
        "```",
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
