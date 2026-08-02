"""0006 host treatment over the frozen 0002-NOOA process bridge.

This module does not modify the frozen worker, its transport, or the eventual
``Decision`` shape.  It adds one host-owned admission boundary: the worker's
otherwise non-authoritative ``reason`` must carry a typed competence commitment.
Only after the host accepts that commitment does it release the inherited
environment action.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from arcengine import GameAction

from experiments.harness import Decision, Observation, Transition

_FROZEN_CONTROL_ROOT = Path(__file__).resolve().parents[2] / "0002-nooa"
if str(_FROZEN_CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(_FROZEN_CONTROL_ROOT))

from host.runner import (
    NooaProcessAgent,
    NooaWorkerError,
    _append_jsonl,
    _host_observation_record,
    _redact_images,
    _workspace_commitment,
)

_TREATMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_TREATMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TREATMENT_ROOT))

from competence import (
    SCHEMA,
    CompetenceController,
    ContractError,
    parse_model_reason,
)

# The producer is experiments/harness/runner.py, which builds the observation as
# np.array(frame.frame) over the game's nested Python ints, giving platform
# int64. The uint8 in historical artifacts belongs to a different producer: the
# rendered current_frame.png, which is RGB and genuinely 8-bit. Binding the
# projection to that value rejected every real observation.
EXPECTED_PROJECTION_DTYPE = "<i8"

_REASON_FIELDS = (
    "working_state",
    "test",
    "claim_ids",
    "ledger",
    "assertions",
    "evidence_requests",
    "reconciliation",
)


def attach_competence_context(
    frozen_observation: Mapping[str, object],
    packet: Mapping[str, object],
) -> dict[str, object]:
    """Add the fixed treatment viewport without granting a new host affordance."""

    model_observation = deepcopy(dict(frozen_observation))
    model_observation["competence"] = {
        "schema": SCHEMA,
        "instruction": (
            "For the final return_result, reason must be compact JSON and contain "
            "exactly the listed competence fields, except that reconciliation may be "
            "omitted when there is no unresolved mismatch. "
            "It must not contain action or data. "
            "The host alone checks the successor and admits the environment action. "
            "Do not manually quote or escape reason. In execute_python, first define "
            "model-authored action and data variables plus commitment as a Python dict, "
            "then execute the "
            "provided reason_serialization.python_pattern. Use Python None for JSON null."
        ),
        "reason_json_fields": list(_REASON_FIELDS),
        "reason_authoring_schema": {
            "working_state": {"type": "object", "maxEncodedCharacters": 4_000},
            "test": {"type": "string", "minLength": 1, "maxLength": 500},
            "claim_ids": {
                "type": "array",
                "maxItems": 4,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "maxLength": 80},
                "constraint": "each ID must name an active claim after ledger operations",
            },
            "ledger": {
                "type": "array",
                "items": {
                    "create_or_revise": {
                        "exactKeys": ["op", "claim_id", "statement"],
                        "op": ["create", "revise"],
                        "claim_id": {"type": "string", "minLength": 1, "maxLength": 80},
                        "statement": {"type": "string", "minLength": 1, "maxLength": 1_000},
                    },
                    "retire": {
                        "exactKeys": ["op", "claim_id"],
                        "op": ["retire"],
                        "claim_id": {"type": "string", "minLength": 1, "maxLength": 80},
                    },
                },
            },
            "assertions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "numeric": {
                        "exactKeys": ["observable", "op", "value"],
                        "observable": ["changed_cells", "levels_completed"],
                        "op": ["==", "!=", "<", "<=", ">", ">="],
                        "value": {"type": "integer", "minimum": 0},
                    },
                    "state": {
                        "exactKeys": ["observable", "op", "value"],
                        "observable": ["state"],
                        "op": ["==", "!="],
                        "value": {"type": "string", "minLength": 1, "maxLength": 500},
                    },
                },
            },
            "evidence_requests": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "exactKeys": ["operator", "action_id"],
                    "operator": ["transition"],
                    "action_id": {
                        "type": "integer",
                        "constraint": "must name an already recorded transition",
                    },
                },
            },
            "reconciliation": {
                "when_no_unresolved_mismatches": (
                    "omit this field entirely, or set it to null; both are accepted"
                ),
                "when_unresolved_mismatches": {
                    "type": "array",
                    "itemCount": "exactly one per unresolved mismatch",
                    "items": {
                        "exactKeys": ["mismatch_id", "disposition", "note"],
                        "mismatch_id": {"type": "string", "minLength": 1, "maxLength": 100},
                        "disposition": ["ABANDON", "NARROW", "RETAIN"],
                        "note": {"type": "string", "minLength": 1, "maxLength": 1_000},
                    },
                },
            },
        },
        "reason_serialization": {
            "method": "python-json-dumps",
            "python_pattern": (
                "import json\n"
                "if not isinstance(commitment, dict):\n"
                "    raise TypeError('commitment must be a Python dict')\n"
                "reason = json.dumps(commitment, sort_keys=True, separators=(',', ':'))\n"
                "return_result({'action': action, 'data': data, 'reason': reason})"
            ),
        },
        "context": deepcopy(dict(packet)),
    }
    return model_observation


def _projection_identity(projection: Any) -> dict[str, Any]:
    pixels = projection.pixels
    return {
        "digest": projection.digest,
        "dtype": pixels.dtype.str,
        "shape": list(pixels.shape),
    }


def _require_expected_projection_dtype(observation: Observation) -> None:
    actual = observation.pixels.dtype.str
    if actual != EXPECTED_PROJECTION_DTYPE:
        raise NooaWorkerError(
            "0006-NOOA observation dtype differs from the approved ls20 contract: "
            f"expected {EXPECTED_PROJECTION_DTYPE}, received {actual}"
        )


def _bound_model_audit(audit: object) -> dict[str, Any]:
    bound = _redact_images(audit if isinstance(audit, dict) else {})
    model_calls = bound.get("model_calls") if isinstance(bound, dict) else None
    if isinstance(model_calls, list):
        for call in model_calls:
            canonical_request = (
                call.get("canonical_request") if isinstance(call, dict) else None
            )
            if isinstance(canonical_request, dict):
                call["canonical_audit_request_sha256"] = sha256(
                    json.dumps(
                        canonical_request,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
    return bound


class NooaCompetenceProcessAgent(NooaProcessAgent):
    """Frozen 0002 transport plus the 0006 competence admission protocol."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.competence = CompetenceController()
        self.competence_events_path = self.run_root / "competence.jsonl"
        self._competence_event_offset = 0
        self._competence_step: int | None = None

    def act(self, observation: Observation) -> Decision:
        """Decide one action, recording why if the decision cannot be made.

        The shared harness maps a NooaWorkerError to the stop reason
        ``agent_error`` and discards the exception, so a live failure otherwise
        reaches the summary with no cause attached. The harness is frozen
        substrate shared with the control arm, so the record is written here.
        """

        try:
            return self._act(observation)
        except BaseException as error:
            self._write_process(
                "agent_error",
                action_point_id=self._step,
                error={"type": type(error).__name__, "message": str(error)},
            )
            raise

    def _act(self, observation: Observation) -> Decision:
        if self._closed:
            raise NooaWorkerError("worker is closed")
        if self._pending_decision is not None:
            raise NooaWorkerError("observe must close the previous action before act")
        if self._step >= self.paired_run_spec.max_actions:
            raise NooaWorkerError(
                "environment-action budget exhausted "
                f"({self._step}/{self.paired_run_spec.max_actions})"
            )
        _require_expected_projection_dtype(observation)

        host_public = {"step": self._step, **_host_observation_record(observation)}
        if self._competence_step is None:
            current_evidence = self.competence.receive_observation(host_public)
            self._competence_step = self._step
        elif self._competence_step == self._step:
            current_evidence = self.competence.require_current_observation(host_public)
        else:
            raise NooaWorkerError("competence controller and frozen host disagree on action step")
        competence_packet = self.competence.context_packet()
        self._flush_competence_events()

        host_observation_path = self.host_observations / f"step-{self._step:05d}.json"
        host_observation_bytes = json.dumps(
            host_public,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        host_observation_path.write_bytes(host_observation_bytes)
        host_observation_commitment = sha256(host_observation_bytes).hexdigest()
        if self._current_projection is None:
            self._current_projection = self.observation_store.write_observation(
                observation,
                step=self._step,
            )
        current = self._current_projection
        from p9.annotated_frame import annotate, annotation_descriptor

        self.model_frame_path.write_bytes(annotate(current.pixels))
        height, width = observation.pixels.shape[-2:]
        coordinate_actions = [
            action
            for action in observation.available_actions
            if GameAction.from_id(action).is_complex()
        ]
        representations = json.loads(json.dumps(current.representations))
        representations["rendered_image"]["annotation"] = annotation_descriptor()
        frozen_observation = {
            "step": self._step,
            "current_metadata": current.metadata,
            "representations": representations,
            "recent_transitions": self._history[
                -self.paired_run_spec.max_recent_transitions:
            ],
            "full_history_file": "public_history.jsonl",
            "current_observation_file": "current_observation.json",
            "observations_glob": "observations/step-*.json",
        }
        request: dict[str, Any] = {
            "protocol": "0002-nooa-ipc-v1",
            "run_id": self.run_id,
            "action_point_id": self._step,
            "observation": attach_competence_context(
                frozen_observation,
                competence_packet,
            ),
            "legal_actions": list(observation.available_actions),
            "coordinate_actions": coordinate_actions,
            "frame_width": int(width),
            "frame_height": int(height),
            "paired_run_spec": self.paired_run_spec.as_record(),
        }
        if self.fixture_policy is not None:
            request["fixture_python_cells"] = self.fixture_policy(self._step, observation)
        _append_jsonl(self.requests_path, request)
        self._write_process(
            "competence_context_assembled",
            action_point_id=self._step,
            current_evidence_id=current_evidence.evidence_id,
            competence_packet_sha256=sha256(
                json.dumps(competence_packet, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            projection=_projection_identity(current),
        )
        self._write_process(
            "action_request",
            action_point_id=self._step,
            host_observation={
                "path": host_observation_path.relative_to(self.run_root).as_posix(),
                "sha256": host_observation_commitment,
            },
            request_sha256=sha256(
                json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            request=request,
        )
        response = self._wait_for_response(self._step)
        audit = response.get("audit")
        worker_error = response.get("error")
        bound_audit = _bound_model_audit(audit)
        self._write_process(
            "model_crossing",
            action_point_id=self._step,
            audit=bound_audit,
            worker_decision=response.get("decision"),
            worker_error=_redact_images(worker_error),
        )
        self._validate_worker_outcome(audit, worker_error)
        payload = response.get("decision")
        if not isinstance(payload, dict):
            raise NooaWorkerError("worker response has no decision object")
        action = payload.get("action")
        data = payload.get("data")
        reason = payload.get("reason")
        if type(action) is not int or action not in observation.available_actions:
            raise NooaWorkerError(f"worker returned illegal action: {action!r}")
        if data is not None and not isinstance(data, dict):
            raise NooaWorkerError("worker decision data must be an object or null")
        if not isinstance(reason, str):
            raise NooaWorkerError("worker decision reason must be a string")
        if action in coordinate_actions:
            if (
                not isinstance(data, dict)
                or any(
                    not isinstance(data.get(axis), int)
                    or isinstance(data.get(axis), bool)
                    for axis in ("x", "y")
                )
            ):
                raise NooaWorkerError(
                    f"action {action} requires integer x and y data"
                )
            if not (0 <= data["x"] < width and 0 <= data["y"] < height):
                raise NooaWorkerError(
                    f"action {action} coordinates {(data['x'], data['y'])} are outside "
                    f"width={width}, height={height}"
                )
        try:
            raw_commitment = parse_model_reason(reason, action=action, data=data)
            commitment = self.competence.admit(
                raw_commitment,
                legal_actions=observation.available_actions,
            )
        except ContractError as error:
            self._flush_competence_events()
            self._write_process(
                "competence_rejected",
                action_point_id=self._step,
                error={"type": type(error).__name__, "message": str(error)},
            )
            raise NooaWorkerError(f"competence commitment rejected: {error}") from error

        self._flush_competence_events()

        decision = Decision(
            action=action,
            data=data,
            reasoning={"reason": reason, "competence": commitment},
        )
        workspace_commitment = _workspace_commitment(self.workspace)
        self._write_process(
            "decision_validated",
            action_point_id=self._step,
            workspace_commitment=workspace_commitment,
            decision={
                "action": decision.action,
                "data": dict(decision.data) if decision.data else None,
                "reasoning": dict(decision.reasoning) if decision.reasoning else None,
            },
        )
        self._write_process(
            "competence_action_admitted",
            action_point_id=self._step,
            commitment=commitment,
        )
        self._pending_decision = decision
        self._accepted_workspace_commitment = workspace_commitment
        return decision

    def observe(self, transition: Transition) -> None:
        """Bind and evaluate the actual successor before any later model decision."""

        _require_expected_projection_dtype(transition.before)
        _require_expected_projection_dtype(transition.after)
        successor_step = self._step + 1
        super().observe(transition)
        host_public = {
            "step": successor_step,
            **_host_observation_record(transition.after),
        }
        successor_evidence = self.competence.receive_observation(host_public)
        self._competence_step = successor_step
        if self._current_projection is None:
            raise NooaWorkerError("frozen host did not preserve the successor projection")
        host_observation_path = (
            self.host_observations / f"step-{successor_step:05d}.json"
        )
        host_observation_bytes = json.dumps(
            host_public,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        host_observation_path.write_bytes(host_observation_bytes)
        self._write_process(
            "competence_successor_observed",
            action_point_id=successor_step - 1,
            successor_step=successor_step,
            evidence_id=successor_evidence.evidence_id,
            host_observation={
                "path": host_observation_path.relative_to(self.run_root).as_posix(),
                "sha256": sha256(host_observation_bytes).hexdigest(),
            },
            projection=_projection_identity(self._current_projection),
        )
        self._flush_competence_events()

    def _flush_competence_events(self) -> None:
        """Durably record every controller event before an action is released."""

        for event in self.competence.events[self._competence_event_offset :]:
            _append_jsonl(self.competence_events_path, event)
        self._competence_event_offset = len(self.competence.events)

    @staticmethod
    def _validate_worker_outcome(audit: object, worker_error: object) -> None:
        model_calls = audit.get("model_calls") if isinstance(audit, dict) else None
        budget_exhausted = (
            isinstance(worker_error, dict)
            and worker_error.get("kind") == "budget_exhausted"
        )
        if (
            not isinstance(model_calls, list)
            or (not model_calls and not budget_exhausted)
            or any(
                not isinstance(call, dict)
                or not isinstance(call.get("messages"), list)
                or not call["messages"]
                or not isinstance(call.get("tools"), list)
                for call in model_calls
            )
        ):
            raise NooaWorkerError("worker response has no valid model-call audit")
        if (
            isinstance(worker_error, dict)
            and worker_error.get("kind") == "provider_transport"
        ):
            raise NooaWorkerError(
                "provider transport "
                f"{worker_error.get('disposition')} after "
                f"{worker_error.get('attempts')} attempt(s)"
            )
        if (
            isinstance(worker_error, dict)
            and worker_error.get("kind") == "decision_failure"
        ):
            raise NooaWorkerError(
                "worker decision "
                f"{worker_error.get('type')}: {worker_error.get('message')}"
            )
        if budget_exhausted:
            raise NooaWorkerError(
                f"{worker_error.get('resource')} budget exhausted "
                f"({worker_error.get('used')}/{worker_error.get('limit')})"
            )
