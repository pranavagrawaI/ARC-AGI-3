"""Frozen failure taxonomy over the unchanged 0002 substrate.

Nothing here changes runtime behavior. These classifiers name the behavior the
existing tool, decision parser, and transport already produce, so a reader of
``process.jsonl`` can reconstruct which failure occurred, whether the model saw
it, and whether the host retried it.

The classification inputs are the exact records already written by
``llm_repl_agent``: the parsed ``python_tool.output`` object, the message of a
``DecisionFormatError``, and the ``retryable`` flag on ``ModelRequestError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


TOOL_FAILURE_KINDS = (
    "none",
    "tool_argument_validation_failure",
    "python_execution_failure",
    "execution_timeout_restart",
    "worker_crash_restart",
)

DECISION_FAILURE_KINDS = (
    "decision_not_json",
    "decision_action_not_integer",
    "decision_illegal_action",
    "decision_missing_coordinates",
    "decision_coordinates_out_of_bounds",
    "decision_data_not_object",
    "decision_tool_after_final_round",
)

TRANSPORT_FAILURE_KINDS = (
    "transport_retryable",
    "transport_fatal",
)


@dataclass(frozen=True)
class ToolOutcome:
    """What one python-tool result means for the model and for the workspace."""

    kind: str
    truncated: bool
    workspace_restarted: bool
    in_memory_state_lost: bool


def classify_tool_result(output: Mapping[str, Any]) -> ToolOutcome:
    """Classify a parsed ``PythonResult.model_text`` object.

    The five kinds partition the reachable outputs of ``PythonWorkspace.execute``
    plus the two argument-validation paths in ``LlmReplAgent._execute_tool_calls``:
    a timeout and a worker exit both set ``workspace_restarted``, an interpreter
    failure carries a traceback, and an argument failure carries neither.
    """

    ok = bool(output.get("ok"))
    truncated = bool(output.get("truncated"))
    restarted = bool(output.get("workspace_restarted"))
    timed_out = bool(output.get("timed_out"))
    has_traceback = isinstance(output.get("traceback"), str)
    if ok:
        kind = "none"
    elif timed_out:
        kind = "execution_timeout_restart"
    elif restarted:
        kind = "worker_crash_restart"
    elif has_traceback:
        kind = "python_execution_failure"
    else:
        kind = "tool_argument_validation_failure"
    return ToolOutcome(
        kind=kind,
        truncated=truncated,
        workspace_restarted=restarted,
        in_memory_state_lost=restarted,
    )


def classify_decision_error(message: str) -> str:
    """Name the parser rejection behind a ``DecisionFormatError``.

    The patterns track the frozen messages raised by
    ``llm_repl_agent.agent.parse_decision_message`` and by the final-round guard
    in ``LlmReplAgent._required_action_response``.
    """

    lowered = message.lower()
    if "python is unavailable" in lowered:
        return "decision_tool_after_final_round"
    if "is not legal" in lowered:
        return "decision_illegal_action"
    if "outside" in lowered and "width=" in lowered:
        return "decision_coordinates_out_of_bounds"
    if "requires integer x and y" in lowered:
        return "decision_missing_coordinates"
    if "action must be an integer" in lowered:
        return "decision_action_not_integer"
    if "data must be an object" in lowered:
        return "decision_data_not_object"
    return "decision_not_json"


def classify_transport_error(*, retryable: bool) -> str:
    return "transport_retryable" if retryable else "transport_fatal"


FRICTION_KINDS = (
    "constructed_observation_path_not_found",
    "path_not_found",
    "module_unavailable",
    "observation_schema_key_guessed",
    "stale_name",
    "execution_timeout",
    "tool_argument_validation",
    "python_error",
)

# Which side a failure is attributable to. "interface" means the model could not
# have known better from what the harness told it; "model" means the harness
# stated what was needed. Attribution is a heuristic over error text, so a third
# value exists rather than forcing every failure into one of the two.
FRICTION_ATTRIBUTION = {
    "constructed_observation_path_not_found": "interface",
    "module_unavailable": "interface",
    "observation_schema_key_guessed": "interface",
    "tool_argument_validation": "model",
    "execution_timeout": "model",
    "path_not_found": "unclear",
    "stale_name": "unclear",
    "python_error": "model",
}


@dataclass(frozen=True)
class Friction:
    kind: str
    attributable_to: str
    detail: str


def classify_friction(code: str | None, output: Mapping[str, Any]) -> Friction | None:
    """Name why a failed tool call failed, and whose fault the harness can claim it is.

    Reads only the code the model sent and the result it received, so it works over
    any run directory that has a ``process.jsonl`` — including the recorded 0002
    runs that predate this module.
    """

    if output.get("ok"):
        return None
    text = f"{output.get('traceback') or ''}{output.get('stderr') or ''}"
    last = next(
        (line.strip() for line in reversed(text.splitlines()) if line.strip()), ""
    )
    if output.get("timed_out"):
        kind = "execution_timeout"
    elif "FileNotFoundError" in text and "observations/step-" in text:
        kind = "constructed_observation_path_not_found"
    elif "FileNotFoundError" in text:
        kind = "path_not_found"
    elif "ModuleNotFoundError" in text or "ImportError" in text:
        kind = "module_unavailable"
    elif "KeyError" in text:
        kind = "observation_schema_key_guessed"
    elif "NameError" in text:
        kind = "stale_name"
    elif not output.get("traceback"):
        kind = "tool_argument_validation"
    else:
        kind = "python_error"
    return Friction(
        kind=kind,
        attributable_to=FRICTION_ATTRIBUTION[kind],
        detail=last[:200],
    )


def failure_taxonomy() -> dict[str, Any]:
    """The model-visible/host-visible split each arm commits to."""

    return {
        "schema": "p9-failure-taxonomy-v1",
        "tool": {
            "kinds": list(TOOL_FAILURE_KINDS),
            "returned_to_model": True,
            "host_retries": False,
            "discriminating_fields": [
                "ok",
                "timed_out",
                "workspace_restarted",
                "traceback",
                "truncated",
            ],
        },
        "decision": {
            "kinds": list(DECISION_FAILURE_KINDS),
            "returned_to_model": True,
            "host_retries": False,
            "discriminating_fields": ["error"],
        },
        "transport": {
            "kinds": list(TRANSPORT_FAILURE_KINDS),
            "returned_to_model": False,
            "host_retries": "transport_retryable only",
            "discriminating_fields": ["retryable", "http_status"],
        },
    }
