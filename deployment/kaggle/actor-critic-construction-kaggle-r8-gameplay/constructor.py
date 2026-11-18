from __future__ import annotations

import copy
import json
import time
from request_contract import validate_request
from pathlib import Path

ROUTING = {}  # logical relay marker, never a gateway provider claim
from dataclasses import dataclass, field
from typing import Any

from real_trace import RealTraceCase
from integration_boundary import evaluate_candidate, execute_inspection

from constructor_contract import SYSTEM, runtime_contract  # noqa: E402


def construction_messages(
    case: RealTraceCase, feedback: dict[str, Any] | None
) -> list[dict[str, str]]:
    packet = case.compact_view()
    user = {
        "visible_case": packet,
        "required_submission": {
            "work_item": {
                "decision": "decision blocked by the current model",
                "obstacle": "candidate explanation, not host-established diagnosis",
                "operation": "ground | contrast | close | compose | test",
                "claim": "narrow model component being constructed",
                "evidence_refs": "nonempty list of integers: visible step numbers only",
                "dependencies": "nonempty list of strings naming declared quantities and their suppliers",
                "completion_checks": "nonempty list of integers: visible step numbers to check",
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
    max_model_requests: int = 16
    max_inspections: int = 10
    max_submissions: int = 4
    max_completion_tokens: int = 16384
    max_prompt_tokens: int = 300000
    wall_seconds: int = 900
    reserve_requests: int = 2
    reserve_completion_tokens: int = 4096
    max_dollars: float = 0.50


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
    progress_sink=None,
    result=None,
) -> ConstructionResult:
    limits = limits or ConstructionLimits()
    messages: list[dict[str, Any]] = list(construction_messages(case, feedback=None))
    messages.append({"role": "user", "content": json.dumps({"runtime_contract": runtime_contract(limits)}, separators=(",", ":"))})
    result = result or ConstructionResult(status="WORKING")
    def checkpoint():
        if progress_sink is not None:
            progress_sink(copy.deepcopy(result))
    started = time.monotonic()
    commit_phase = False
    seen_call_ids: set[str] = set()

    prose_failures = 0
    while result.model_requests < limits.max_model_requests:
        remaining_tokens = limits.max_completion_tokens - result.usage.get('completion_tokens', 0)
        remaining_requests = limits.max_model_requests - result.model_requests
        seconds = limits.wall_seconds - (time.monotonic() - started)
        if remaining_tokens <= 0 or seconds <= 20:
            break
        commit_phase = commit_phase or remaining_requests <= limits.reserve_requests or remaining_tokens <= limits.reserve_completion_tokens + 4096 or result.inspections >= limits.max_inspections or seconds <= 180
        request_tokens = min(4096, remaining_tokens if commit_phase else remaining_tokens - limits.reserve_completion_tokens)
        if commit_phase and remaining_requests >= 2 and remaining_tokens >= 4096:
            request_tokens = min(request_tokens, remaining_tokens - 2048)
        tools = [tool for tool in TOOLS if tool['function']['name'] != 'execute_python' or not commit_phase]
        tools = [tool for tool in tools if tool['function']['name'] != 'submit_candidate' or result.submissions < limits.max_submissions]
        telemetry = {
            'phase': 'commit' if commit_phase else 'explore',
            'seconds_remaining': round(seconds, 3), 'requests_remaining': remaining_requests,
            'completion_tokens_remaining': remaining_tokens,
            'prompt_tokens_remaining': limits.max_prompt_tokens - result.usage.get('prompt_tokens', 0),
            'inspections_remaining': 0 if commit_phase else limits.max_inspections-result.inspections,
            'submissions_remaining': limits.max_submissions-result.submissions,
            'reserved_requests': limits.reserve_requests, 'reserved_completion_tokens': limits.reserve_completion_tokens,
            'current_completion_reservation': request_tokens,
            'dollars_remaining_conservative': round(limits.max_dollars - result.usage.get('prompt_tokens',0)*1e-6 - result.usage.get('completion_tokens',0)*1e-5, 6),
        }
        body = {
            'model': 'qwen/qwen3.8-27b',
            'messages': copy.deepcopy(messages) + [{'role': 'user', 'content': json.dumps({'budget': telemetry}, separators=(',', ':'))}],
            'tools': copy.deepcopy(tools), 'tool_choice': 'required',
            'provider': copy.deepcopy(ROUTING),
            'max_tokens': request_tokens, 'reasoning': {'enabled': False},
        }
        # UTF-8 bytes plus 4096 framing tokens is a deliberately loose token
        # upper bound for this text-only Qwen request, checked against usage.
        prompt_reservation = len(json.dumps(body, ensure_ascii=False).encode()) + 4096
        if prompt_reservation + result.usage.get('prompt_tokens', 0) > limits.max_prompt_tokens:
            break
        worst_cost = (prompt_reservation + result.usage.get('prompt_tokens',0))*1e-6 + (request_tokens + result.usage.get('completion_tokens',0))*1e-5
        if worst_cost > limits.max_dollars:
            break
        validate_request(body)
        result.model_requests += 1
        result.requested_tokens += request_tokens
        checkpoint()
        response = transport(body)
        result.transcript.append({'request': body, 'response': response})
        checkpoint()
        usage = response.get('usage')
        if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ('prompt_tokens','completion_tokens')) or usage['completion_tokens'] > request_tokens or usage['prompt_tokens'] > prompt_reservation:
            result.status = 'INVALID_USAGE'
            return result
        for key in ('prompt_tokens', 'completion_tokens'):
            result.usage[key] = result.usage.get(key, 0) + usage[key]
        checkpoint()
        if (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0):
            result.status = 'INVALID_REASONING'
            return result
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0].get("message"), dict):
            result.status = "INVALID_RESPONSE"
            return result
        assistant = choices[0]["message"]
        if any(assistant.get(k) for k in ('reasoning','reasoning_content','reasoning_details')):
            result.status = 'INVALID_REASONING'
            return result
        # Raw response remains lossless in transcript; outbound messages use only
        # reviewed request fields. Provider response tool indices are not inputs.
        outbound = {'role': 'assistant', 'content': assistant.get('content')}
        if assistant.get('tool_calls'):
            outbound['tool_calls'] = [
                {'id': c['id'], 'type': c['type'],
                 'function': {k: c['function'][k] for k in ('name', 'arguments')}}
                for c in assistant['tool_calls']]
        messages.append(outbound)
        calls = assistant.get("tool_calls") or []
        if not calls:
            prose_failures += 1
            if prose_failures >= 2:
                result.status = 'INVALID_RESPONSE'
                return result
            commit_phase = True
            messages.append({'role': 'user', 'content': 'A tool call is required: submit_candidate or request_evidence.'})
            continue
        response_ids = [call.get("id") for call in calls]
        if any(not isinstance(value, str) or not value for value in response_ids):
            raise ValueError("tool call id is missing")
        if len(response_ids) != len(set(response_ids)) or any(value in seen_call_ids for value in response_ids):
            raise ValueError("duplicate tool call id")
        # Provider call IDs are opaque. Preserve response-list order for dispatch;
        # enforce only nonempty identity and uniqueness, never an invented ordinal.
        seen_call_ids.update(response_ids)
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name")
            call_id = call.get("id") or "missing-call-id"
            # Charge recognized attempts before parsing or executing untrusted input.
            exhausted = False
            if name == "execute_python":
                exhausted = commit_phase or result.inspections >= limits.max_inspections
                if not exhausted:
                    result.inspections += 1
            elif name == "submit_candidate":
                exhausted = result.submissions >= limits.max_submissions
                if not exhausted:
                    result.submissions += 1
            tool_started = time.monotonic()
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
                        payload = {"status": "evaluated", "visible_evaluation": {key: copy.deepcopy(value) for key, value in evaluation.items() if key != "suffix"}}
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
                    "wall_seconds": round(time.monotonic() - tool_started, 6),
                }
            )
            if event_sink is not None:
                checkpoint()
                event_sink(copy.deepcopy(result.tool_events[-1]))
            checkpoint()
            if result.status in {"CANDIDATE_READY", "NEEDS_EVIDENCE"}:
                return result

    result.status = "BUDGET_EXHAUSTED"
    return result
