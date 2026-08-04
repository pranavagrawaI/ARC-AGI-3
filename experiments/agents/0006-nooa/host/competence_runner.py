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
    NOTES_FILE,
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
# A sibling module, imported bare. ``host.*`` is the *frozen 0002* package on
# this sys.path, not this directory, so ``from host.commitment_controller``
# would silently reach into the control arm's tree.
from commitment_controller import (
    FIELDS,
    MAY_BE_UNKNOWN,
    OUTCOMES,
    UNKNOWN,
    CommitmentController,
    Rejected,
)
from supervision import (
    AssignmentRejected,
    LUNA_MODEL,
    SupervisorAssignment,
    SupervisorStateView,
    SupervisorSession,
    build_supervisor_view,
    gemma_supervision_context,
    parse_gemma_supervision_response,
)

# A correctable rejection may be answered this many times within one action
# point. It is a 0006 bound, not a frozen 0002 one, so it is pinned by the
# approval's source hash rather than by the paired run spec. Each attempt also
# spends the spec's model-call budget, which remains the ceiling on total cost.
MAX_COMMITMENT_REPAIRS = 1


def admit_transactionally(
    controller: CommitmentController | None,
    competence: CompetenceController,
    raw_commitment: dict[str, Any],
    *,
    action: int,
    legal_actions: tuple[int, ...],
) -> tuple[dict[str, object], CommitmentController | None]:
    """Validate both gates before publishing controller state.

    The controller is copied because its successful admission mutates history
    and measurements. A later competence-contract rejection must leave the
    previously accepted block untouched.
    """

    candidate = deepcopy(controller)
    if candidate is not None:
        candidate.admit_working_state(
            raw_commitment.get("working_state"),
            action=action,
            legal_actions=legal_actions,
        )
    admitted = competence.admit(raw_commitment, legal_actions=legal_actions)
    return admitted, candidate


class ModelCorrectableError(NooaWorkerError):
    """A rejection the model could answer by re-authoring its own commitment.

    Most failures here are preconditions: nothing the model writes can satisfy
    them, because the observation arrived with the wrong dtype, a budget is
    spent, or the worker is gone. Those must fail fast.

    This class marks the other kind. The value that failed was authored by the
    model, so a later crossing that reports the error is answerable — run
    20260802T120006Z lost a complete and correct commitment to one surplus brace
    that the model had every means to fix.

    Both kinds still stop the run; repair is not built. The split exists so that
    the summary says which failures were the model's to correct, and so the
    repair crossing has one class to catch rather than a list of message
    prefixes.
    """


class SupervisorContradictionError(ModelCorrectableError):
    """Gemma rejected the current assignment with admissible evidence."""

    def __init__(self, evidence: str) -> None:
        super().__init__(f"Gemma contradicted the supervisor assignment: {evidence}")
        self.evidence = evidence

# arc_agi declares the frame dtype at remote_wrapper.py:242,
# np.array(frame_layer, dtype=np.int8), and experiments/harness/runner.py:53
# passes it through with np.array(frame.frame, copy=True). Live ls20 observations
# are therefore int8.
#
# Two earlier values were inferred from proxies rather than from this producer,
# and each rejected every real observation: |u1 came from uint8 in the artifacts,
# which is the RGB render (arc_agi/rendering.py:91), and <i8 came from numpy's
# promotion of nested Python ints, which is not what the game hands over.
EXPECTED_PROJECTION_DTYPE = "|i1"

_REASON_FIELDS = (
    "working_state",
    "test",
    "claim_ids",
    "ledger",
    "assertions",
    "evidence_requests",
    "reconciliation",
)


def rejection_notice(message: str, *, attempts_remaining: int) -> dict[str, object]:
    """State the host's own rejection without authoring a replacement.

    The notice carries the contract error and points at the model's workspace,
    where the ``reason`` string it already serialized is still in scope. It
    supplies no field values, no corrected payload, and no suggestion about what
    to claim or predict: the model re-authors, or the action is refused.
    """

    if message.startswith("deliberation rounds exhausted"):
        # There was no commitment to refuse, so the ordinary notice would be
        # false. Still no field values and no action: the model is told what it
        # spent and asked to commit, not told what to commit to.
        instruction = (
            "You used every deliberation round for this action point without "
            "calling return_result, so no environment action was released and "
            "the rounds are spent. The Python workspace is unchanged, so every "
            "variable and file from those rounds is still in scope; do not "
            "recompute what you already worked out. Commit this round: author "
            "the commitment and call return_result. Inspecting further without "
            "committing spends the run and changes nothing. The host will not "
            "choose an action for you."
        )
    else:
        instruction = (
            "Your previous commitment was refused for the stated reason and no "
            "environment action was released. The Python workspace is unchanged, "
            "so any variable your last cell defined is still in scope. If you "
            "already serialized a commitment there, pass that exact string as "
            "reason rather than retyping it. Otherwise author a new commitment. "
            "The host will not repair or complete a commitment for you."
        )
    return {
        "schema": "0006-nooa-rejection-notice-v1",
        "rejected_reason": message,
        "attempts_remaining": attempts_remaining,
        "instruction": instruction,
    }


def _is_round_exhaustion(error_type: object, message: str) -> bool:
    """Whether the worker gave up because the model never committed.

    The frozen worker reports this as a generic GenerationError, so it has to be
    told apart from a real decision failure by its cause. Matched on the
    iteration bound rather than the round count so the test does not silently
    stop matching when max_model_rounds_per_action changes.
    """

    return error_type == "GenerationError" and "max_iterations" in message


def _controller_working_state_schema(
    controller: CommitmentController,
) -> dict[str, object]:
    """Publish the eight fields, their legal values, and the current block.

    The host states the shape and the continuity rules. Every value in the block
    is authored by the model; nothing here names an object, a coordinate, or an
    action.
    """

    return {
        "type": "object",
        "maxEncodedCharacters": 4_000,
        "exactKeys": list(FIELDS),
        "purpose": (
            "your whole persistent abstraction. It is checked as a continuation "
            "of the previous action's block, not for truth. Keep the same eight "
            "lines in NOTES.md so it also survives between rounds of one action."
        ),
        "PLAYER_BINDING": {
            "type": "string",
            "mayBe": UNKNOWN,
            "purpose": (
                "which object you control and its current attributes. Once it is "
                "not 'unknown' it is settled and rule R1 applies."
            ),
        },
        "LOCAL_OBJECTIVE": {"type": "string", "purpose": "one near-term target"},
        "ROUTE1": {
            "type": "string",
            "format": "comma-separated action ids, or 'unknown'",
            "purpose": "the remaining actions you intend to take",
        },
        "CURRENT_APPROX_POSITION": {"type": "string", "mayBe": UNKNOWN},
        "LAST_ACTION": {"type": "integer-or-null"},
        "LAST_OUTCOME": {
            "enum": list(OUTCOMES),
            "purpose": (
                "your classification of the previous transition. The host does "
                "not label it for you; request the transition operator for the "
                "changed_cells count and name the outcome yourself."
            ),
        },
        "NEXT_STEP": {
            "type": "integer",
            "constraint": "must equal decision.action and be currently legal",
        },
        "CONTRADICTION": {
            "type": "string-or-null",
            "absence": "use JSON null exactly",
            "purpose": (
                "JSON null, or one non-empty sentence naming the transition that "
                "broke a settled binding. Never use an empty string or the text "
                "'none' or 'null'. Evidence licenses exactly one reinterpretation "
                "and must return to JSON null on the next action."
            ),
        },
        "rules": {
            "R1": "a settled PLAYER_BINDING changes only with a CONTRADICTION in "
            "the same block, or when LAST_OUTCOME is special_event",
            "R2": "after 'moved', neither PLAYER_BINDING nor LOCAL_OBJECTIVE may "
            "change unless you name a CONTRADICTION",
            "R3": "after 'blocked', edit ROUTE1; binding and objective hold unless "
            "you name a CONTRADICTION",
            "R4": f"{list(MAY_BE_UNKNOWN)} may stay '{UNKNOWN}'; unknown is not a "
            "contradiction and does not license rebinding",
            "R5": "only special_event licenses focused semantic inspection",
            "R7": "NEXT_STEP is exactly one legal action and equals decision.action",
            "R8": "set CONTRADICTION to JSON null on the action after evidence",
            "R9": "at most two consecutive 'uncertain' outcomes",
            "not_enforced": "replacing ROUTE1 is recorded but never rejected; "
            "re-planning after you learn the map is ordinary play",
        },
        "current_block": controller.state.render(),
    }


def attach_competence_context(
    frozen_observation: Mapping[str, object],
    packet: Mapping[str, object],
    *,
    rejection: Mapping[str, object] | None = None,
    commitment_controller: CommitmentController | None = None,
    supervisor_assignment: SupervisorAssignment | None = None,
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
            "test records your intent and is never checked; assertions carry the "
            "prediction the host evaluates against the successor, so at least "
            "one is required and an empty list releases no action. "
            "The host alone checks the successor and admits the environment action. "
            "Do not manually quote or escape reason. Use Python None for JSON null. "
            "Return by exactly one route: inside execute_python, define the "
            "model-authored action and data variables and commitment as a Python "
            "dict, then run the provided reason_serialization.python_pattern, whose "
            "final statement calls return_result with the reason variable. Do not "
            "call return_result as a separate tool: that requires retyping the "
            "serialized string by hand, and a single mistyped character discards "
            "the whole commitment and releases no action. Passing the variable "
            "cannot be mistyped. Do not spend turns deciding between the two "
            "routes; the inline call is the only supported one."
        ),
        "reason_json_fields": list(_REASON_FIELDS),
        "reason_authoring_schema": {
            "working_state": {"type": "object", "maxEncodedCharacters": 4_000},
            "test": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
                "purpose": (
                    "your intent for this action, in prose, for the record. The "
                    "host does not evaluate it and nothing is checked against it."
                ),
            },
            "claim_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "maxLength": 80},
                "constraint": "each ID must name an active claim after ledger operations",
                "purpose": (
                    "the claims you are working from on this action, which is "
                    "what carries your understanding across actions. On the "
                    "first action no claim exists yet, so create one in ledger "
                    "and name it here. At least one is required; an empty list "
                    "is refused and releases no action."
                ),
            },
            "ledger": {
                "type": "array",
                "purpose": (
                    "how you change the claims themselves: create a claim you "
                    "did not hold before, revise one the successor changed your "
                    "mind about, retire one you no longer hold. Operations are "
                    "applied before claim_ids is resolved, so a claim created "
                    "here can be named there on the same action."
                ),
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
                "purpose": (
                    "the part of your prediction the host actually checks. After "
                    "the action the host evaluates each assertion against the "
                    "observed successor and reports match or mismatch back to "
                    "you, and a mismatch must be reconciled before you act "
                    "again. Stating your prediction only in test leaves nothing "
                    "to check and teaches you nothing. At least one assertion is "
                    "required; an empty list is refused and releases no action."
                ),
                "constraint": (
                    "at least one assertion must be one the successor could "
                    "falsify, and the host refuses the commitment otherwise. An "
                    "assertion about state or levels_completed is at risk only "
                    "if it contradicts the value in current_metadata: asserting "
                    "the value you were just given predicts nothing, because it "
                    "is what you would observe if the action did nothing. "
                    "changed_cells has no current value, so any count you name "
                    "is a real prediction, including zero; only a comparison "
                    "true of every possible count, such as >= 0, is refused."
                ),
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
    if supervisor_assignment is not None:
        model_observation["supervision"] = gemma_supervision_context(
            supervisor_assignment
        )
    if commitment_controller is not None:
        model_observation["competence"]["reason_authoring_schema"]["working_state"] = (
            _controller_working_state_schema(commitment_controller)
        )
    if rejection is not None:
        model_observation["competence"]["rejection"] = deepcopy(dict(rejection))
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

    def __init__(
        self,
        *args: Any,
        max_commitment_repairs: int = MAX_COMMITMENT_REPAIRS,
        commitment_controller: bool = False,
        supervisor: SupervisorSession | None = None,
        supervisor_rules: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.competence = CompetenceController()
        # Off unless the rules_controller knowledge condition asked for it, so
        # every other condition and the control arm are byte-identical to before.
        self.commitment = CommitmentController() if commitment_controller else None
        if supervisor is not None and (
            self.commitment is None or not isinstance(supervisor_rules, str)
        ):
            raise ValueError(
                "supervision requires the commitment controller and RULES.md text"
            )
        self.supervisor = supervisor
        self.supervisor_rules = supervisor_rules
        self.competence_events_path = self.run_root / "competence.jsonl"
        self._competence_event_offset = 0
        self._competence_step: int | None = None
        if max_commitment_repairs < 0:
            raise ValueError("max_commitment_repairs must not be negative")
        self.max_commitment_repairs = max_commitment_repairs
        # The frozen worker refuses a repeated (run_id, action_point_id) at
        # launcher.py:367 and validates nothing else about the id, so a repair is
        # simply the next one. Crossings therefore outnumber environment steps.
        self._crossing_id = 0

    def _notes_diverged_from_seed(self) -> bool:
        """Whether the model has rewritten NOTES.md away from the seeded block.

        In all three earlier recorded runs it never did: NOTES.md sat at its
        seed for the whole run while the model re-derived everything each round.
        The host only observes this; it neither reads the contents nor acts on
        them, exactly as the workspace documentation promises the model.
        """

        seed = self.paired_run_spec.usability_probe_instruction
        notes = self.workspace / NOTES_FILE
        if seed is None or not notes.is_file():
            return False
        return notes.read_text(encoding="utf-8", errors="replace") != seed

    def _next_crossing_id(self) -> int:
        crossing_id = self._crossing_id
        self._crossing_id += 1
        return crossing_id

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
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                    "correctable": isinstance(error, ModelCorrectableError),
                },
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
        def request_supervision(
            gemma_contradiction: str | None = None,
        ) -> SupervisorAssignment:
            if self.supervisor is None:
                raise NooaWorkerError("supervision was requested while disabled")
            if self.commitment is None:
                raise NooaWorkerError("supervision requires the commitment controller")
            state = self.commitment.state
            supervisor_view = build_supervisor_view(
                rules=self.supervisor_rules or "",
                current_observation=host_public,
                available_actions=observation.available_actions,
                recent_transitions=self._history,
                current_state=SupervisorStateView(
                    player_binding=state.PLAYER_BINDING,
                    current_approx_position=state.CURRENT_APPROX_POSITION,
                    local_objective=state.LOCAL_OBJECTIVE,
                    last_action=state.LAST_ACTION,
                    last_outcome=state.LAST_OUTCOME,
                ),
                previous_assignment=self.supervisor.previous_assignment,
                gemma_contradiction=gemma_contradiction,
            )
            assignment = self.supervisor.assign(supervisor_view)
            self._write_process(
                "supervisor_assignment",
                action_point_id=self._step,
                step=self._step,
                supervisor_model=LUNA_MODEL,
                view_sha256=sha256(
                    json.dumps(
                        supervisor_view, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                assignment_token=assignment.token,
                assignment_sha256=assignment.sha256,
                assignment=assignment.as_record(),
                crossing=self.supervisor.last_record,
            )
            return assignment

        supervisor_assignment: SupervisorAssignment | None = None
        if self.supervisor is not None:
            supervisor_assignment = request_supervision()
        def build_request(
            crossing_id: int, rejection: Mapping[str, Any] | None
        ) -> dict[str, Any]:
            built: dict[str, Any] = {
                "protocol": "0002-nooa-ipc-v1",
                "run_id": self.run_id,
                "action_point_id": crossing_id,
                "observation": attach_competence_context(
                    frozen_observation,
                    competence_packet,
                    rejection=rejection,
                    commitment_controller=self.commitment,
                    supervisor_assignment=supervisor_assignment,
                ),
                "legal_actions": list(observation.available_actions),
                "coordinate_actions": coordinate_actions,
                "frame_width": int(width),
                "frame_height": int(height),
                "paired_run_spec": self.paired_run_spec.as_record(),
            }
            if self.fixture_policy is not None:
                built["fixture_python_cells"] = self.fixture_policy(
                    self._step, observation
                )
            return built

        self._write_process(
            "competence_context_assembled",
            action_point_id=self._step,
            step=self._step,
            current_evidence_id=current_evidence.evidence_id,
            competence_packet_sha256=sha256(
                json.dumps(competence_packet, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            projection=_projection_identity(current),
        )
        def admit_from(response: Mapping[str, Any]) -> tuple[int, Any, str, dict[str, Any]]:
            self._validate_worker_outcome(
                response.get("audit"), response.get("error")
            )
            payload = response.get("decision")
            if not isinstance(payload, dict):
                raise NooaWorkerError("worker response has no decision object")
            action = payload.get("action")
            data = payload.get("data")
            reason = payload.get("reason")
            if type(action) is not int or action not in observation.available_actions:
                raise ModelCorrectableError(f"worker returned illegal action: {action!r}")
            if data is not None and not isinstance(data, dict):
                raise ModelCorrectableError(
                    "worker decision data must be an object or null"
                )
            if not isinstance(reason, str):
                raise ModelCorrectableError("worker decision reason must be a string")
            if action in coordinate_actions:
                if (
                    not isinstance(data, dict)
                    or any(
                        not isinstance(data.get(axis), int)
                        or isinstance(data.get(axis), bool)
                        for axis in ("x", "y")
                    )
                ):
                    raise ModelCorrectableError(
                        f"action {action} requires integer x and y data"
                    )
                if not (0 <= data["x"] < width and 0 <= data["y"] < height):
                    raise ModelCorrectableError(
                        f"action {action} coordinates {(data['x'], data['y'])} are outside "
                        f"width={width}, height={height}"
                    )
            try:
                raw_commitment = parse_model_reason(reason, action=action, data=data)
            except ContractError as error:
                raise ModelCorrectableError(
                    f"competence commitment rejected: {error}"
                ) from error
            if supervisor_assignment is not None:
                try:
                    supervised_response = parse_gemma_supervision_response(
                        raw_commitment.get("test"),
                        assignment=supervisor_assignment,
                    )
                except AssignmentRejected as error:
                    raise ModelCorrectableError(
                        f"supervision response rejected: {error}"
                    ) from error
                self._write_process(
                    "supervision_response",
                    action_point_id=self._step,
                    step=self._step,
                    assignment_sha256=supervised_response.assignment_sha256,
                    disposition=supervised_response.disposition,
                    prediction=supervised_response.prediction,
                    contradiction_evidence=supervised_response.contradiction_evidence,
                )
                if supervised_response.disposition == "CONTRADICT":
                    raise SupervisorContradictionError(
                        supervised_response.contradiction_evidence or ""
                    )
            try:
                admitted, candidate = admit_transactionally(
                    self.commitment,
                    self.competence,
                    raw_commitment,
                    action=action,
                    legal_actions=observation.available_actions,
                )
            except Rejected as error:
                raise ModelCorrectableError(
                    f"commitment controller rejected {error}"
                ) from error
            except ContractError as error:
                raise ModelCorrectableError(
                    f"competence commitment rejected: {error}"
                ) from error
            self.commitment = candidate
            return action, data, reason, admitted

        rejection: dict[str, Any] | None = None
        for attempt in range(self.max_commitment_repairs + 1):
            crossing_id = self._next_crossing_id()
            request = build_request(crossing_id, rejection)
            _append_jsonl(self.requests_path, request)
            self._write_process(
                "action_request",
                action_point_id=crossing_id,
                step=self._step,
                host_observation={
                    "path": host_observation_path.relative_to(self.run_root).as_posix(),
                    "sha256": host_observation_commitment,
                },
                request_sha256=sha256(
                    json.dumps(request, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest(),
                request=request,
            )
            response = self._wait_for_response(crossing_id)
            self._write_process(
                "model_crossing",
                action_point_id=crossing_id,
                step=self._step,
                audit=_bound_model_audit(response.get("audit")),
                worker_decision=response.get("decision"),
                worker_error=_redact_images(response.get("error")),
            )
            try:
                action, data, reason, commitment = admit_from(response)
                break
            except ModelCorrectableError as error:
                attempts_remaining = self.max_commitment_repairs - attempt
                self._flush_competence_events()
                self._write_process(
                    "competence_rejected",
                    action_point_id=crossing_id,
                    step=self._step,
                    error={
                        "type": type(error).__name__,
                        "message": str(error),
                        "correctable": True,
                    },
                    attempts_remaining=attempts_remaining,
                )
                if attempts_remaining <= 0:
                    raise
                if isinstance(error, SupervisorContradictionError):
                    supervisor_assignment = request_supervision(error.evidence)
                rejection = rejection_notice(
                    str(error), attempts_remaining=attempts_remaining
                )

        self._flush_competence_events()

        decision = Decision(
            action=action,
            data=data,
            reasoning={"reason": reason, "competence": commitment},
        )
        workspace_commitment = _workspace_commitment(self.workspace)
        self._write_process(
            "decision_validated",
            action_point_id=crossing_id,
            step=self._step,
            workspace_commitment=workspace_commitment,
            decision={
                "action": decision.action,
                "data": dict(decision.data) if decision.data else None,
                "reasoning": dict(decision.reasoning) if decision.reasoning else None,
            },
        )
        self._write_process(
            "competence_action_admitted",
            action_point_id=crossing_id,
            step=self._step,
            commitment=commitment,
        )
        if self.commitment is not None:
            # Recorded per action so a null result stays interpretable. Without
            # notes_written a null has two readings that cannot be told apart:
            # the rules did not help, or the round-level half of the protocol was
            # never obeyed. The frozen worker echoes NOTES.md every round but has
            # no hook to reject one, so obedience there can only be observed.
            self._write_process(
                "commitment_controller_state",
                action_point_id=crossing_id,
                step=self._step,
                state=self.commitment.state.render(),
                metrics=self.commitment.metrics(),
                notes_written=self._notes_diverged_from_seed(),
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
            message = str(worker_error.get("message", ""))
            if _is_round_exhaustion(worker_error.get("type"), message):
                # Thinking too long is not a reason to end a run. The model spent
                # its deliberation rounds without calling return_result, which is
                # a failure to author a commitment, not a failure of the host or
                # the provider. It is answered the way every other model-authored
                # failure is: a repair crossing that names the problem and asks
                # again. The host still supplies no action and no field value, so
                # the run stays inside the decision boundary in CONTRACT.md.
                #
                # This is what killed the recorded gemma+rules treatment run at
                # action 0 after eight rounds of execute_python.
                raise ModelCorrectableError(
                    f"deliberation rounds exhausted without a committed action: {message}"
                )
            raise NooaWorkerError(
                "worker decision "
                f"{worker_error.get('type')}: {worker_error.get('message')}"
            )
        if budget_exhausted:
            raise NooaWorkerError(
                f"{worker_error.get('resource')} budget exhausted "
                f"({worker_error.get('used')}/{worker_error.get('limit')})"
            )
