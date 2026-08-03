"""Bounded, action-withholding supervision from Luna to Gemma.

Luna selects the immediate problem. Gemma still selects the environment action.
This module owns the supervisor crossing, information boundary, and handoff
schema; it does not interpret game semantics.
"""

from __future__ import annotations

from copy import deepcopy
from base64 import b64decode
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Callable

FIELDS = (
    "LOCAL_GOAL",
    "RELEVANT_EVIDENCE",
    "SUCCESS_TEST",
    "IF_TEST_FAILS",
)
GEMMA_RESPONSE_FIELDS = (
    "assignment_sha256",
    "disposition",
    "prediction",
    "contradiction_evidence",
)
MAX_TEXT_CHARS = 500
MAX_EVIDENCE_ITEMS = 3
MAX_RECENT_TRANSITIONS = 6
LUNA_MODEL = "openai/gpt-5.6-luna"
LUNA_REASONING_EFFORT = "medium"
LUNA_MAX_OUTPUT_TOKENS = 2_000
LUNA_PROVIDER_ROUTE = {
    "order": ["openai"],
    "only": ["openai"],
    "allow_fallbacks": False,
    "require_parameters": False,
}

STATE_CONTINUITY = {
    "PLAYER_BINDING": (
        "The verbatim accepted value. The controller treats it as opaque persistent "
        "text, so changing it invokes the binding-continuity rules."
    ),
    "CURRENT_APPROX_POSITION": (
        "A separate position estimate that Gemma may update independently without "
        "rewriting PLAYER_BINDING."
    ),
}


class AssignmentRejected(ValueError):
    """Luna's response does not satisfy the supervision boundary."""


@dataclass(frozen=True)
class SupervisorStateView:
    """Typed projection of accepted controller state exposed to Luna."""

    player_binding: str
    current_approx_position: str
    local_objective: str
    last_action: int | None
    last_outcome: str | None

    def as_record(self) -> dict[str, object]:
        return {
            "PLAYER_BINDING": self.player_binding,
            "CURRENT_APPROX_POSITION": self.current_approx_position,
            "LOCAL_OBJECTIVE": self.local_objective,
            "LAST_ACTION": self.last_action,
            "LAST_OUTCOME": self.last_outcome,
        }


def _bounded_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssignmentRejected(f"{field} must be a non-empty string")
    text = value.strip()
    if len(text) > MAX_TEXT_CHARS:
        raise AssignmentRejected(f"{field} exceeds {MAX_TEXT_CHARS} characters")
    return text


@dataclass(frozen=True)
class SupervisorAssignment:
    local_goal: str
    relevant_evidence: tuple[str, ...]
    success_test: str
    if_test_fails: str

    def as_record(self) -> dict[str, object]:
        return {
            "LOCAL_GOAL": self.local_goal,
            "RELEVANT_EVIDENCE": list(self.relevant_evidence),
            "SUCCESS_TEST": self.success_test,
            "IF_TEST_FAILS": self.if_test_fails,
        }

    @property
    def sha256(self) -> str:
        return sha256(
            json.dumps(
                self.as_record(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class GemmaSupervisionResponse:
    assignment_sha256: str
    disposition: str
    prediction: str | None
    contradiction_evidence: str | None


def parse_assignment(raw: object) -> SupervisorAssignment:
    """Accept exactly one bounded assignment with no action-bearing field."""

    if not isinstance(raw, Mapping) or set(raw) != set(FIELDS):
        raise AssignmentRejected(f"assignment must contain exactly {list(FIELDS)}")
    evidence = raw["RELEVANT_EVIDENCE"]
    if (
        not isinstance(evidence, list)
        or not 1 <= len(evidence) <= MAX_EVIDENCE_ITEMS
    ):
        raise AssignmentRejected(
            f"RELEVANT_EVIDENCE must contain 1 to {MAX_EVIDENCE_ITEMS} items"
        )
    return SupervisorAssignment(
        local_goal=_bounded_text(raw["LOCAL_GOAL"], field="LOCAL_GOAL"),
        relevant_evidence=tuple(
            _bounded_text(item, field="RELEVANT_EVIDENCE item") for item in evidence
        ),
        success_test=_bounded_text(raw["SUCCESS_TEST"], field="SUCCESS_TEST"),
        if_test_fails=_bounded_text(raw["IF_TEST_FAILS"], field="IF_TEST_FAILS"),
    )


def build_supervisor_view(
    *,
    rules: str,
    current_observation: Mapping[str, object],
    available_actions: Sequence[int],
    recent_transitions: Sequence[Mapping[str, object]],
    current_state: SupervisorStateView,
    previous_assignment: SupervisorAssignment | Mapping[str, object] | None,
    gemma_contradiction: str | None = None,
) -> dict[str, object]:
    """Give Luna admissible evidence and bounded continuity, never hidden thought."""

    prior = (
        previous_assignment.as_record()
        if isinstance(previous_assignment, SupervisorAssignment)
        else deepcopy(dict(previous_assignment))
        if isinstance(previous_assignment, Mapping)
        else None
    )
    return {
        "schema": "0006-nooa-supervisor-view-v1",
        "role": (
            "Select one immediate objective for Gemma. Do not provide an action, "
            "action number, route, multi-step plan, cardinal or relative movement "
            "direction, adjacent target cell, or prose that uniquely determines one "
            "legal action under the supplied rules. State the desired observable "
            "change without selecting how Gemma achieves it. Return exactly the four "
            f"fields {list(FIELDS)}. SUCCESS_TEST must be observable after one action."
        ),
        "rules": rules,
        "current_observation": deepcopy(dict(current_observation)),
        "available_actions": list(available_actions),
        "recent_transitions": deepcopy(
            [dict(item) for item in recent_transitions[-MAX_RECENT_TRANSITIONS:]]
        ),
        "current_state": current_state.as_record(),
        "state_continuity": deepcopy(STATE_CONTINUITY),
        "previous_assignment": prior,
        "gemma_contradiction": gemma_contradiction,
    }


def parse_gemma_supervision_response(
    raw_test: object, *, assignment: SupervisorAssignment
) -> GemmaSupervisionResponse:
    """Bind Gemma's action-point response to the assignment it received."""

    if not isinstance(raw_test, str):
        raise AssignmentRejected("supervised test must be a JSON string")
    try:
        raw = json.loads(raw_test)
    except json.JSONDecodeError as error:
        raise AssignmentRejected("supervised test is not valid JSON") from error
    if not isinstance(raw, Mapping) or set(raw) != set(GEMMA_RESPONSE_FIELDS):
        raise AssignmentRejected(
            f"supervised test must contain exactly {list(GEMMA_RESPONSE_FIELDS)}"
        )
    assignment_sha256 = raw["assignment_sha256"]
    if assignment_sha256 != assignment.sha256:
        raise AssignmentRejected("Gemma response names the wrong supervisor assignment")
    disposition = raw["disposition"]
    if disposition not in {"FOLLOW", "CONTRADICT"}:
        raise AssignmentRejected("disposition must be FOLLOW or CONTRADICT")
    prediction = raw["prediction"]
    evidence = raw["contradiction_evidence"]
    if disposition == "FOLLOW":
        prediction = _bounded_text(prediction, field="prediction")
        if evidence is not None:
            raise AssignmentRejected("FOLLOW requires null contradiction_evidence")
    if disposition == "CONTRADICT":
        if prediction is not None:
            raise AssignmentRejected("CONTRADICT requires null prediction")
        evidence = _bounded_text(evidence, field="contradiction_evidence")
    return GemmaSupervisionResponse(
        assignment_sha256=assignment_sha256,
        disposition=disposition,
        prediction=prediction,
        contradiction_evidence=evidence,
    )


def build_luna_request(
    view: Mapping[str, object], *, frame_data_url: str
) -> dict[str, object]:
    """Build the exact transport-neutral Responses request for Luna."""

    if not frame_data_url.startswith("data:image/png;base64,"):
        raise ValueError("supervisor frame must be a base64 PNG data URL")
    properties: dict[str, object] = {
        "LOCAL_GOAL": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS},
        "RELEVANT_EVIDENCE": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_EVIDENCE_ITEMS,
            "items": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS},
        },
        "SUCCESS_TEST": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS},
        "IF_TEST_FAILS": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS},
    }
    return {
        "model": LUNA_MODEL,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You supervise one Gemma action point. Select the immediate "
                            "problem, not the action. Follow the action-withholding "
                            "boundary in the supplied view role and use only supplied "
                            "admissible evidence."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            dict(view), sort_keys=True, separators=(",", ":")
                        ),
                    },
                    {"type": "input_image", "image_url": frame_data_url},
                ],
            },
        ],
        "reasoning": {"effort": LUNA_REASONING_EFFORT},
        "max_output_tokens": LUNA_MAX_OUTPUT_TOKENS,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "supervisor_assignment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": properties,
                    "required": list(FIELDS),
                },
            }
        },
        "provider": deepcopy(LUNA_PROVIDER_ROUTE),
    }


def parse_luna_response(response: object) -> SupervisorAssignment:
    """Extract one structured assignment from a Responses API record."""

    if not isinstance(response, Mapping):
        raise AssignmentRejected("Luna response must be an object")
    texts: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        texts.append(text)
    if len(texts) != 1:
        raise AssignmentRejected(
            f"Luna response must contain exactly one output_text, found {len(texts)}"
        )
    try:
        raw = json.loads(texts[0])
    except json.JSONDecodeError as error:
        raise AssignmentRejected("Luna output_text is not JSON") from error
    return parse_assignment(raw)


def gemma_supervision_context(
    assignment: SupervisorAssignment,
) -> dict[str, object]:
    """The authority split Gemma receives with the current assignment."""

    return {
        "schema": "0006-nooa-supervision-handoff-v1",
        "instruction": (
            "Work only on this action point. Choose one legal action intended to "
            "satisfy SUCCESS_TEST and predict its immediate observable effect. "
            "Do not replace LOCAL_GOAL. Apply state_continuity below: update "
            "CURRENT_APPROX_POSITION independently when the position estimate changes, "
            "without rewriting the opaque PLAYER_BINDING merely to track movement. If "
            "admissible evidence contradicts the assignment, return a precise "
            "CONTRADICTION with the evidence instead."
        ),
        "state_continuity": deepcopy(STATE_CONTINUITY),
        "authority": {
            "LOCAL_GOAL": "binding for this action point",
            "SUCCESS_TEST": "binding evaluation criterion for this action point",
            "RELEVANT_EVIDENCE": "defeasible advice; current admissible evidence wins",
            "IF_TEST_FAILS": "defeasible advice for the next state update",
        },
        "response_contract": (
            "The competence test field must be a compact JSON string with exactly "
            f"{list(GEMMA_RESPONSE_FIELDS)}. assignment_sha256 must equal the value "
            "below. disposition is FOLLOW or CONTRADICT. prediction is the immediate "
            "observable effect. FOLLOW requires a prediction and null "
            "contradiction_evidence. CONTRADICT requires null prediction plus precise "
            "admissible evidence and releases no action."
        ),
        "assignment_sha256": assignment.sha256,
        "assignment": assignment.as_record(),
    }


class SupervisorSession:
    """One Luna crossing per action point with bounded cross-turn continuity."""

    def __init__(
        self,
        call: Callable[
            [dict[str, object]], tuple[object, dict[str, object] | None]
        ],
    ) -> None:
        self._call = call
        self.previous_assignment: SupervisorAssignment | None = None
        self.calls = 0
        self.last_record: dict[str, object] | None = None

    def assign(self, view: dict[str, object]) -> SupervisorAssignment:
        raw, crossing = self._call(deepcopy(view))
        assignment = parse_assignment(raw)
        self.last_record = deepcopy(crossing)
        self.previous_assignment = assignment
        self.calls += 1
        return assignment


class LunaSupervisorClient:
    """Call Luna through an injected transport and retain credential-free audit."""

    def __init__(
        self,
        *,
        transport: Callable[[dict[str, object]], Mapping[str, object]],
        frame_data_url: Callable[[], str],
        audit_path: Path | None = None,
    ) -> None:
        self._transport = transport
        self._frame_data_url = frame_data_url
        self._audit_path = audit_path
        self.records: list[dict[str, object]] = []

    def _persist(self, record: dict[str, object]) -> None:
        if self._audit_path is None:
            return
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def __call__(
        self, view: dict[str, object]
    ) -> tuple[dict[str, object], dict[str, object]]:
        frame = self._frame_data_url()
        request = build_luna_request(view, frame_data_url=frame)
        request_sha256 = sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        frame_bytes = b64decode(frame.split(",", 1)[1], validate=True)
        started = {
            "schema": "0006-nooa-supervisor-crossing-v1",
            "phase": "started",
            "model": LUNA_MODEL,
            "request_sha256": request_sha256,
            "frame_sha256": sha256(frame_bytes).hexdigest(),
            "view": deepcopy(view),
        }
        self._persist(started)
        response_audit: dict[str, object] = {}
        try:
            response = self._transport(request)
            response_sha256 = sha256(
                json.dumps(
                    dict(response), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            generation_id = response.get("id")
            response_audit = {
                "generation_id": generation_id,
                "response_sha256": response_sha256,
                "usage": deepcopy(response.get("usage")),
            }
            assignment = parse_luna_response(response)
            if not isinstance(generation_id, str) or not generation_id:
                raise AssignmentRejected("Luna response has no generation id")
        except BaseException as error:
            failed = {
                **started,
                "phase": "finished",
                "disposition": "failure",
                **response_audit,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            self.records.append(failed)
            self._persist(failed)
            raise
        finished = {
            **started,
            "phase": "finished",
            "disposition": "success",
            **response_audit,
            "assignment": assignment.as_record(),
        }
        self.records.append(finished)
        self._persist(finished)
        return assignment.as_record(), finished
