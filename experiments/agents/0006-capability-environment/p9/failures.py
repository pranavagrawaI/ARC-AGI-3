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
