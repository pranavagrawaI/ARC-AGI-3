from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from real_trace import RealTraceCase
from sandbox_eval import evaluate_candidate, execute_inspection

SYSTEM = """You are the model constructor, not the game-playing policy.

Objective: Construct the smallest model that supports the next useful decision, while preserving what is known and exposing what remains unresolved.

You must discover the work item from the visible evidence. The host is a verifier, not an oracle. A mismatch does not identify its semantic cause.

Operation repertoire: ground, contrast, close, compose, test. These are repeatable operation types, not lifecycle stages. Lifecycle values are OPEN, WORKING, CANDIDATE_READY, NEEDS_EVIDENCE, and BUDGET_EXHAUSTED.

A candidate is construction-complete only when executable visible-prefix checks pass with informative joint commitments for every evaluator-owned target. This is not prospective support. Do not claim unseen evidence.

Distinguish needs_computation, needs_observation, and does_not_block_decision. Unknown is not unchanged. Empty indirection and all-unknown predictions receive no credit.

Candidate Python ABI:
- initialize(observation) -> JSON state
- predict(state, action) -> nonempty list of branches
- each branch has state, commitments, and declared_dependencies
- assimilate(predicted_state, observation, verdict) -> JSON state
Assimilation consumes an already-predicted successor. It must not recompute the transition. Candidate ancestry and joint commitments are host-enforced.

Use execute_python to inspect only the visible case when useful. Finish with submit_candidate or request_evidence. Never infer that a sealed suffix exists."""


def construction_messages(
    case: RealTraceCase, feedback: dict[str, Any] | None
) -> list[dict[str, str]]:
    packet = case.constructor_view()
    user = {
        "visible_case": packet,
        "required_submission": {
            "work_item": {
                "decision": "decision blocked by the current model",
                "obstacle": "candidate explanation, not host-established diagnosis",
                "operation": "ground | contrast | close | compose | test",
                "claim": "narrow model component being constructed",
                "evidence_refs": "visible step numbers only",
                "dependencies": "declared quantities and their suppliers",
                "completion_checks": "visible executable checks",
                "remaining_scope": "unsupported conditions",
                "reasoning_need": (
                    "needs_computation | needs_observation | does_not_block_decision"
                ),
            },
            "candidate_source": "complete Python source implementing the ABI",
        },
    }
    if feedback is not None:
        user["candidate_feedback"] = feedback
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(user, separators=(",", ":"))},
    ]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Run bounded Python against /evidence/case.json, which contains only constructor-visible evidence.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_candidate",
            "description": "Submit one selected work item and complete candidate source for visible-prefix checks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "work_item": {"type": "object"},
                    "candidate_source": {"type": "string"},
                },
                "required": ["work_item", "candidate_source"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_evidence",
            "description": "Stop with an explicit observation request when computation cannot resolve a decision-blocking obstacle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "obstacle": {"type": "string"},
                    "needed_observation": {"type": "string"},
                    "decision_relevance": {"type": "string"},
                },
                "required": ["obstacle", "needed_observation", "decision_relevance"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(frozen=True)
class ConstructionLimits:
    max_model_requests: int = 6
    max_inspections: int = 4
    max_submissions: int = 2
    max_requested_tokens: int = 8192


@dataclass
class ConstructionResult:
    status: str
    candidate_source: str | None = None
    work_item: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    evidence_request: dict[str, Any] | None = None
    model_requests: int = 0
    inspections: int = 0
    submissions: int = 0
    requested_tokens: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)


def _validate_work_item(value: Any, frontier: int) -> dict[str, Any]:
    required = {
        "decision",
        "obstacle",
        "operation",
        "claim",
        "evidence_refs",
        "dependencies",
        "completion_checks",
        "remaining_scope",
        "reasoning_need",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("work_item must contain the exact required fields")
    if value["operation"] not in {"ground", "contrast", "close", "compose", "test"}:
        raise ValueError("work_item operation is not in the construction repertoire")
    if value["reasoning_need"] not in {
        "needs_computation",
        "needs_observation",
        "does_not_block_decision",
    }:
        raise ValueError("work_item reasoning_need is invalid")
    for key in ("decision", "obstacle", "claim", "remaining_scope"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"work_item {key} must be nonempty text")
    for key in ("evidence_refs", "completion_checks"):
        values = value[key]
        if not isinstance(values, list) or not values or any(
            not isinstance(item, int) or not 1 <= item <= frontier for item in values
        ):
            raise ValueError(f"work_item {key} must cite visible step numbers")
    dependencies = value["dependencies"]
    if not isinstance(dependencies, list) or not dependencies or any(
        not isinstance(item, str) or not item.strip() for item in dependencies
    ):
        raise ValueError("work_item dependencies must be nonempty names")
    return value


def _tool_result(call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, separators=(",", ":"), sort_keys=True),
    }


def run_construction(
    case: RealTraceCase,
    *,
    transport,
    limits: ConstructionLimits | None = None,
    event_sink=None,
) -> ConstructionResult:
    limits = limits or ConstructionLimits()
    messages: list[dict[str, Any]] = list(construction_messages(case, feedback=None))
    result = ConstructionResult(status="WORKING")
    remaining_tokens = limits.max_requested_tokens

    while result.model_requests < limits.max_model_requests and remaining_tokens > 0:
        request_tokens = min(3072 if result.model_requests == 0 else 2048, remaining_tokens)
        body = {
            "model": "qwen/qwen3.8-27b",
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(TOOLS),
            "tool_choice": "auto",
            "max_tokens": request_tokens,
            "reasoning": {"enabled": False},
        }
        result.model_requests += 1
        result.requested_tokens += request_tokens
        remaining_tokens -= request_tokens
        response = transport(body)
        usage = response.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens"):
            result.usage[key] = result.usage.get(key, 0) + int(usage.get(key) or 0)
        result.transcript.append({"request": body, "response": response})
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0].get("message"), dict):
            result.status = "INVALID_RESPONSE"
            return result
        assistant = choices[0]["message"]
        messages.append(assistant)
        calls = assistant.get("tool_calls") or []
        if not calls:
            messages.append(
                {"role": "user", "content": "Use a construction tool; prose alone cannot complete this work item."}
            )
            continue
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name")
            call_id = call.get("id") or "missing-call-id"
            # Charge recognized attempts before parsing or executing untrusted input.
            exhausted = False
            if name == "execute_python":
                exhausted = result.inspections >= limits.max_inspections
                if not exhausted:
                    result.inspections += 1
            elif name == "submit_candidate":
                exhausted = result.submissions >= limits.max_submissions
                if not exhausted:
                    result.submissions += 1
            try:
                if exhausted:
                    raise ValueError("tool attempt budget exhausted")
                arguments = json.loads(function.get("arguments") or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
                if name == "execute_python":
                    if exhausted:
                        payload = {"status": "budget_exhausted", "kind": "inspection"}
                    else:
                        code = arguments.get("code")
                        if not isinstance(code, str):
                            raise ValueError("execute_python code must be text")
                        payload = execute_inspection(code, case)
                elif name == "submit_candidate":
                    if exhausted:
                        payload = {"status": "budget_exhausted", "kind": "candidate_submission"}
                    else:
                        work_item = _validate_work_item(arguments.get("work_item"), case.frontier)
                        source = arguments.get("candidate_source")
                        if not isinstance(source, str) or not source.strip():
                            raise ValueError("candidate_source must be nonempty text")
                        if len(source.encode()) > 200_000:
                            raise ValueError("candidate_source exceeds 200,000 bytes")
                        evaluation = evaluate_candidate(source, case, include_suffix=False)
                        payload = {"status": "evaluated", "visible_evaluation": evaluation}
                        if (
                            evaluation["visible"]["status"] == "passed"
                            and not evaluation["coverage"]["missing_required"]
                        ):
                            result.status = "CANDIDATE_READY"
                            result.candidate_source = source
                            result.work_item = work_item
                            result.evaluation = evaluation
                elif name == "request_evidence":
                    expected = {"obstacle", "needed_observation", "decision_relevance"}
                    if set(arguments) != expected or any(
                        not isinstance(arguments[key], str) or not arguments[key].strip()
                        for key in expected
                    ):
                        raise ValueError("evidence request is malformed")
                    payload = {"status": "accepted", "request": arguments}
                    result.status = "NEEDS_EVIDENCE"
                    result.evidence_request = arguments
                else:
                    raise ValueError(f"unknown construction tool {name!r}")
            except Exception as error:  # fail closed and let the model repair its request
                payload = {"status": "rejected", "error": f"{type(error).__name__}: {error}"}
            rendered_result = _tool_result(call_id, payload)
            messages.append(rendered_result)
            result.tool_events.append(
                {
                    "call": call,
                    "rendered_result_message": rendered_result,
                }
            )
            if event_sink is not None:
                event_sink(copy.deepcopy(result.tool_events[-1]))
            if result.status in {"CANDIDATE_READY", "NEEDS_EVIDENCE"}:
                return result

    result.status = "BUDGET_EXHAUSTED"
    return result
