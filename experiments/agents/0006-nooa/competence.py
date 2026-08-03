"""0006-owned competence mechanisms, expressed as ordinary Python objects.

The controller is deliberately narrower than Tycho.  It does not name objects,
rank evidence, propose a world model, plan, or infer a goal.  It gives one
NOOA action method a typed boundary around four earned functions:

* immutable public observation occurrences and exact evidence operators;
* a model-authored, append-only claim ledger;
* action commitments, successor checks, and mismatch dispositions; and
* deterministic context assembly over only model-selected evidence.

The NOOA adapter owns the LLM method.  This module owns exact state,
validation, and audit so each part is directly testable without a provider.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256

SCHEMA = "0006-nooa-competence-v1"
_ASSERTION_OPERATORS = frozenset({"==", "!=", "<", "<=", ">", ">="})
_DISPOSITIONS = frozenset({"ABANDON", "NARROW", "RETAIN"})
_MAX_CONTEXT_BYTES = 12_000
_MAX_RETAINED_MISMATCHES = 4
_MAX_WORKING_CLAIMS = 4


class ContractError(ValueError):
    """A model-authored competence record does not meet the executable contract."""


def parse_model_reason(reason: object, *, action: int, data: object) -> dict[str, object]:
    """Decode the 0006 commitment carried by NOOA's otherwise-neutral reason field.

    The frozen control keeps its final action transport intact.  The treatment
    replaces only the non-authoritative free text with this typed JSON object;
    action and data remain bound by the inherited legality parser.
    """
    if not isinstance(reason, str):
        raise ContractError("final action reason must be a JSON object string")
    try:
        payload = json.loads(reason)
    except json.JSONDecodeError as error:
        raise ContractError("final action reason must be a JSON object") from error
    if not isinstance(payload, dict):
        raise ContractError("final action reason must be a JSON object")
    if {"action", "data"} & set(payload):
        raise ContractError("final action reason must not override action or data")
    return {"action": action, "data": deepcopy(data), **payload}


def _canonical_json(value: object) -> str:
    """JCS-compatible JSON for this contract's finite, integer-only logical domain."""

    def validate(node: object) -> None:
        if node is None or isinstance(node, (bool, int, str)):
            return
        if isinstance(node, float):
            raise ContractError("logical records do not permit floating-point values")
        if isinstance(node, list):
            for item in node:
                validate(item)
            return
        if isinstance(node, Mapping):
            if not all(isinstance(key, str) for key in node):
                raise ContractError("logical record object keys must be strings")
            for item in node.values():
                validate(item)
            return
        raise ContractError("record must be JSON-serializable")

    validate(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ContractError("record must be JSON-serializable") from error


def _digest(value: object, *, domain: str) -> str:
    return sha256(
        _canonical_json({"domain": domain, "value": value}).encode("utf-8")
    ).hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _text(value: object, *, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{field} must be a string")
    result = value.strip()
    if not result:
        raise ContractError(f"{field} must not be empty")
    if len(result) > limit:
        raise ContractError(f"{field} exceeds {limit} characters")
    return result


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} must be an object")
    return value


def _compare(observed: object, operator: str, expected: object) -> bool | None:
    if observed is None:
        return None
    if operator == "==":
        return observed == expected
    if operator == "!=":
        return observed != expected
    if operator == "<":
        return observed < expected
    if operator == "<=":
        return observed <= expected
    if operator == ">":
        return observed > expected
    if operator == ">=":
        return observed >= expected
    raise ContractError(f"unsupported assertion operator {operator!r}")


@dataclass(frozen=True)
class ObservationOccurrence:
    """A host-owned copy of one public observation, identified by exact bytes."""

    step: int
    evidence_id: str
    public: dict[str, object]


@dataclass(frozen=True)
class TransitionOccurrence:
    """The one adjacent action-conditioned relation between two observations."""

    action_id: int
    transition_id: str
    record: dict[str, object]


class EvidenceArchive:
    """Append-only archive of public occurrences; it makes no semantic claims."""

    def __init__(self) -> None:
        self._by_step: dict[int, ObservationOccurrence] = {}

    def append(self, public: Mapping[str, object]) -> ObservationOccurrence:
        raw_step = public.get("step")
        if not _is_int(raw_step) or int(raw_step) < 0:
            raise ContractError("public observation requires a nonnegative integer step")
        step = int(raw_step)
        if self._by_step and step <= max(self._by_step):
            raise ContractError("public observation steps must strictly increase")
        copied = deepcopy(dict(public))
        record = {
            "schema": SCHEMA,
            "kind": "observation_occurrence",
            "step": step,
            "public": copied,
        }
        evidence_id = "obs-" + _digest(record, domain="observation_occurrence")
        occurrence = ObservationOccurrence(step=step, evidence_id=evidence_id, public=copied)
        self._by_step[step] = occurrence
        return occurrence

    def at_step(self, step: object) -> ObservationOccurrence:
        if not _is_int(step) or int(step) not in self._by_step:
            raise ContractError(f"no recorded public observation for step {step!r}")
        return self._by_step[int(step)]

    def at_evidence_id(self, evidence_id: object) -> ObservationOccurrence:
        if not isinstance(evidence_id, str):
            raise ContractError("observation evidence ID must be a string")
        for occurrence in self._by_step.values():
            if occurrence.evidence_id == evidence_id:
                return occurrence
        raise ContractError(f"no recorded public observation for {evidence_id!r}")


class TransitionArchive:
    """Append-only action-bound transitions; no inferred effect taxonomy enters here."""

    def __init__(self) -> None:
        self._by_action: dict[int, TransitionOccurrence] = {}

    def append(
        self,
        commitment: Mapping[str, object],
        *,
        before: ObservationOccurrence,
        after: ObservationOccurrence,
    ) -> TransitionOccurrence:
        action_id = commitment.get("action_id")
        action = commitment.get("action")
        if not _is_int(action_id) or not _is_int(action):
            raise ContractError("committed transition requires integer action identity and action")
        if int(action_id) in self._by_action:
            raise ContractError(f"transition already exists for action {action_id}")
        if before.step + 1 != after.step:
            raise ContractError("a transition occurrence requires adjacent observations")
        source = {
            "schema": SCHEMA,
            "kind": "transition_occurrence",
            "action_id": int(action_id),
            "action": int(action),
            "data": deepcopy(commitment.get("data")),
            "before_evidence_id": before.evidence_id,
            "after_evidence_id": after.evidence_id,
        }
        occurrence = TransitionOccurrence(
            action_id=int(action_id),
            transition_id="trn-" + _digest(source, domain="transition_occurrence"),
            record=source,
        )
        self._by_action[occurrence.action_id] = occurrence
        return occurrence

    def for_action(self, action_id: object) -> TransitionOccurrence:
        if not _is_int(action_id) or int(action_id) not in self._by_action:
            raise ContractError(f"no recorded transition for action {action_id!r}")
        return self._by_action[int(action_id)]


class EvidenceOperators:
    """One exact, action-bound transition measurement with citable persistence."""

    def __init__(self, archive: EvidenceArchive, transitions: TransitionArchive) -> None:
        self._archive = archive
        self._transitions = transitions
        self._derived: dict[str, dict[str, object]] = {}

    def preview(self, request: object) -> dict[str, object]:
        value = _mapping(request, field="evidence request")
        if set(value) != {"operator", "action_id"}:
            raise ContractError("evidence request must contain operator and action_id")
        if value.get("operator") != "transition":
            raise ContractError("only the neutral transition evidence operator is available")
        transition = self._transitions.for_action(value.get("action_id"))
        before = self._archive.at_evidence_id(transition.record.get("before_evidence_id"))
        after = self._archive.at_evidence_id(transition.record.get("after_evidence_id"))
        output = _transition_output(before.public, after.public)
        derived = {
            "schema": SCHEMA,
            "kind": "derived_evidence",
            "operator": "transition",
            "source_transition": {
                **deepcopy(transition.record),
                "transition_id": transition.transition_id,
            },
            "output": output,
        }
        return {
            **derived,
            "evidence_id": "derived-" + _digest(derived, domain="derived_evidence"),
        }

    def persist(
        self, records: Sequence[Mapping[str, object]]
    ) -> list[tuple[dict[str, object], bool]]:
        persisted: list[tuple[dict[str, object], bool]] = []
        for record in records:
            evidence_id = record.get("evidence_id")
            if not isinstance(evidence_id, str):
                raise ContractError("derived evidence requires a content-addressed ID")
            created = evidence_id not in self._derived
            existing = self._derived.setdefault(evidence_id, deepcopy(dict(record)))
            if existing != record:
                raise ContractError("derived evidence identity has divergent content")
            persisted.append((deepcopy(existing), created))
        return persisted


def _layers(value: Mapping[str, object]) -> list[list[list[int]]] | None:
    """Decode the public image as one or more rectangular named-by-index layers."""

    raw = value.get("pixels")
    if not isinstance(raw, list) or not raw:
        return None
    if isinstance(raw[0], list) and raw[0] and _is_int(raw[0][0]):
        raw = [raw]
    if not all(isinstance(layer, list) and layer for layer in raw):
        return None
    result: list[list[list[int]]] = []
    shape: tuple[int, int] | None = None
    for layer in raw:
        if not all(isinstance(row, list) and row and all(_is_int(cell) for cell in row) for row in layer):
            return None
        current_shape = (len(layer), len(layer[0]))
        if any(len(row) != current_shape[1] for row in layer):
            return None
        if shape is None:
            shape = current_shape
        elif shape != current_shape:
            return None
        result.append([[int(cell) for cell in row] for row in layer])
    return result


def _changed_cells_by_layer(
    before: Mapping[str, object], after: Mapping[str, object]
) -> list[int] | None:
    old, new = _layers(before), _layers(after)
    if old is None or new is None or len(old) != len(new):
        return None
    return [
        sum(old_cell != new_cell for old_row, new_row in zip(old_layer, new_layer) for old_cell, new_cell in zip(old_row, new_row))
        for old_layer, new_layer in zip(old, new)
    ]


def _transition_output(before: Mapping[str, object], after: Mapping[str, object]) -> dict[str, object]:
    by_layer = _changed_cells_by_layer(before, after)
    return {
        "changed_cells": sum(by_layer) if by_layer is not None else None,
        "changed_cells_by_layer": by_layer,
        "state_before": before.get("state"),
        "state_after": after.get("state"),
        "levels_completed_before": before.get("levels_completed"),
        "levels_completed_after": after.get("levels_completed"),
    }




class EpistemicLedger:
    """Host-validated identity/history for claims whose meaning remains model-owned."""

    def __init__(self) -> None:
        self._claims: dict[str, dict[str, object]] = {}
        self.events: list[dict[str, object]] = []

    def apply(self, operations: Sequence[object], *, action_id: int) -> dict[str, dict[str, object]]:
        proposed = deepcopy(self._claims)
        proposed_events: list[dict[str, object]] = []
        for index, raw_operation in enumerate(operations):
            operation = _mapping(raw_operation, field=f"ledger[{index}]")
            op = operation.get("op")
            claim_id = _text(operation.get("claim_id"), field=f"ledger[{index}].claim_id", limit=80)
            if op == "create":
                if set(operation) != {"op", "claim_id", "statement"}:
                    raise ContractError("ledger create must contain op, claim_id, and statement")
                if claim_id in proposed:
                    raise ContractError(f"ledger claim already exists: {claim_id}")
                proposed[claim_id] = {
                    "claim_id": claim_id,
                    "statement": _text(operation.get("statement"), field="ledger statement", limit=1_000),
                    "status": "active",
                    "revision": 0,
                }
            elif op == "revise":
                if set(operation) != {"op", "claim_id", "statement"}:
                    raise ContractError("ledger revise must contain op, claim_id, and statement")
                if claim_id not in proposed:
                    raise ContractError(f"cannot revise unknown claim: {claim_id}")
                proposed[claim_id] = {
                    **proposed[claim_id],
                    "statement": _text(operation.get("statement"), field="ledger statement", limit=1_000),
                    "revision": int(proposed[claim_id]["revision"]) + 1,
                }
            elif op == "retire":
                if set(operation) != {"op", "claim_id"}:
                    raise ContractError("ledger retire must contain only op and claim_id")
                if claim_id not in proposed:
                    raise ContractError(f"cannot retire unknown claim: {claim_id}")
                proposed[claim_id] = {**proposed[claim_id], "status": "retired"}
            else:
                raise ContractError("ledger op must be create, revise, or retire")
            proposed_events.append(
                {"schema": SCHEMA, "type": "ledger", "action_id": action_id, "operation": deepcopy(dict(operation))}
            )
        self._claims = proposed
        self.events.extend(proposed_events)
        return deepcopy(proposed)

    def active(self) -> list[dict[str, object]]:
        return [
            deepcopy(self._claims[claim_id])
            for claim_id in sorted(self._claims)
            if self._claims[claim_id]["status"] == "active"
        ]


def _assertion(raw: object, *, index: int) -> dict[str, object]:
    value = _mapping(raw, field=f"assertions[{index}]")
    observable = value.get("observable")
    operator = value.get("op")
    if operator not in _ASSERTION_OPERATORS:
        raise ContractError(f"assertions[{index}].op is invalid")
    if observable in {"changed_cells", "levels_completed"}:
        if set(value) != {"observable", "op", "value"} or not _is_int(value.get("value")):
            raise ContractError(f"assertions[{index}] has an invalid numeric shape")
        if int(value["value"]) < 0:
            raise ContractError(f"assertions[{index}].value must be nonnegative")
        return {"observable": observable, "op": str(operator), "value": int(value["value"])}
    if observable == "state":
        if set(value) != {"observable", "op", "value"} or not isinstance(value.get("value"), str):
            raise ContractError(f"assertions[{index}] has an invalid state shape")
        if operator not in {"==", "!="}:
            raise ContractError(f"assertions[{index}].state supports only == or !=")
        return {
            "observable": "state",
            "op": str(operator),
            "value": _text(value["value"], field=f"assertions[{index}].value", limit=500),
        }
    raise ContractError(f"assertions[{index}].observable is not permitted")


class CompetenceController:
    """The 0006 treatment state machine behind a one-action NOOA method."""

    def __init__(self) -> None:
        self.archive = EvidenceArchive()
        self.transitions = TransitionArchive()
        self.operators = EvidenceOperators(self.archive, self.transitions)
        self.ledger = EpistemicLedger()
        self._current: ObservationOccurrence | None = None
        self._pending: dict[str, object] | None = None
        self._working_state: dict[str, object] = {}
        self._working_claim_ids: list[str] = []
        self._last_prediction_result: dict[str, object] | None = None
        self._unresolved: list[dict[str, object]] = []
        self._retained: list[dict[str, object]] = []
        self._requested_evidence: list[dict[str, object]] = []
        self._continuity: list[dict[str, object]] = []
        self._next_action_id = 1
        self.events: list[dict[str, object]] = []

    def receive_observation(self, public: Mapping[str, object]) -> ObservationOccurrence:
        """Archive one public successor and mechanically evaluate a pending commitment."""
        occurrence = self.archive.append(public)
        if self._pending is not None:
            before = self.archive.at_step(self._pending["before_step"])
            transition = self.transitions.append(
                self._pending,
                before=before,
                after=occurrence,
            )
            result = self._evaluate_pending(before, occurrence)
            self._last_prediction_result = result
            self._continuity.append(
                {
                    "action_id": self._pending["action_id"],
                    "before_evidence_id": before.evidence_id,
                    "after_evidence_id": occurrence.evidence_id,
                    "outcome": result["outcome"],
                }
            )
            self._continuity = self._continuity[-4:]
            self.events.append(
                {
                    "schema": SCHEMA,
                    "type": "transition_occurrence",
                    "transition_id": transition.transition_id,
                    "record": deepcopy(transition.record),
                }
            )
            self.events.append({"schema": SCHEMA, "type": "prediction_result", **deepcopy(result)})
            self._pending = None
        self._current = occurrence
        self.events.append(
            {
                "schema": SCHEMA,
                "type": "observation",
                "record": {
                    "schema": SCHEMA,
                    "kind": "observation_occurrence",
                    "evidence_id": occurrence.evidence_id,
                    "step": occurrence.step,
                    "public": deepcopy(occurrence.public),
                },
            }
        )
        return occurrence

    def require_current_observation(
        self, public: Mapping[str, object]
    ) -> ObservationOccurrence:
        """Reject a later action point that is not the already observed successor."""

        current = self._current
        if current is None:
            raise ContractError("no current public observation")
        if _canonical_json(current.public) != _canonical_json(dict(public)):
            raise ContractError("action point differs from the recorded successor observation")
        return current

    def admit(self, raw: object, *, legal_actions: Sequence[int]) -> dict[str, object]:
        """Validate a model decision, update its ledger, and bind one action to evidence."""
        if self._current is None:
            raise ContractError("cannot admit an action before the first public observation")
        if self._pending is not None:
            raise ContractError("the previous committed action has no successor observation")
        decision = _mapping(raw, field="decision")
        required = {
            "action",
            "data",
            "working_state",
            "test",
            "claim_ids",
            "ledger",
            "assertions",
            "evidence_requests",
            "reconciliation",
        }
        provided = set(decision)
        if not self._unresolved and "reconciliation" not in provided:
            # With nothing unresolved, null is the only legal value and the host
            # already knows it. Accepting omission invents no semantic value; it
            # removes a syntax tax that costs live runs.
            provided |= {"reconciliation"}
        if provided != required:
            raise ContractError(f"decision must contain exactly {sorted(required)}")
        action = decision.get("action")
        if not _is_int(action) or int(action) not in set(legal_actions):
            raise ContractError("decision.action is not in the current legal action set")
        data = deepcopy(decision.get("data"))
        _canonical_json(data)
        working_state = _mapping(decision.get("working_state"), field="decision.working_state")
        copied_state = deepcopy(dict(working_state))
        if len(_canonical_json(copied_state)) > 4_000:
            raise ContractError("decision.working_state exceeds 4000 encoded characters")
        test = _text(decision.get("test"), field="decision.test", limit=500)
        raw_assertions = decision.get("assertions")
        if not isinstance(raw_assertions, list) or not 1 <= len(raw_assertions) <= 5:
            raise ContractError("decision.assertions must contain one to five assertions")
        assertions = [_assertion(item, index=index) for index, item in enumerate(raw_assertions)]
        claim_ids = self._claim_ids(decision.get("claim_ids"))
        ledger_ops = decision.get("ledger")
        if not isinstance(ledger_ops, list):
            raise ContractError("decision.ledger must be a list")
        raw_requests = decision.get("evidence_requests")
        if not isinstance(raw_requests, list) or len(raw_requests) > 4:
            raise ContractError("decision.evidence_requests must be a list of at most four requests")
        reconciliations = self._validate_reconciliation(decision.get("reconciliation"))
        projected = self._project_ledger(ledger_ops)
        for claim_id in claim_ids:
            if claim_id not in projected or projected[claim_id]["status"] != "active":
                raise ContractError(f"decision.claim_ids names no active claim: {claim_id}")
        requested = [self.operators.preview(request) for request in raw_requests]
        action_id = self._next_action_id
        working_claims = self._working_claims(projected, claim_ids)
        retained = self._next_retained(reconciliations)
        prospective_packet = self._packet(
            working_state=copied_state,
            working_claims=working_claims,
            requested_evidence=requested,
            unresolved_mismatches=[],
            retained_mismatches=retained,
        )
        self._require_context_capacity(prospective_packet)
        ledger_event_offset = len(self.ledger.events)
        self.ledger.apply(ledger_ops, action_id=action_id)
        self._next_action_id += 1
        self.events.extend(deepcopy(self.ledger.events[ledger_event_offset:]))
        if reconciliations is not None:
            self.events.append(
                {"schema": SCHEMA, "type": "reconciliation", "items": deepcopy(reconciliations)}
            )
            self._unresolved = []
        self._retained = retained
        self._working_state = copied_state
        self._working_claim_ids = claim_ids
        persisted_evidence = self.operators.persist(requested)
        self._requested_evidence = [record for record, _created in persisted_evidence]
        self.events.extend(
            {
                "schema": SCHEMA,
                "type": "derived_evidence",
                "record": deepcopy(record),
            }
            for record, created in persisted_evidence
            if created
        )
        commitment = {
            "schema": SCHEMA,
            "action_id": action_id,
            "action": int(action),
            "data": data,
            "before_step": self._current.step,
            "before_evidence_id": self._current.evidence_id,
            "working_state": deepcopy(copied_state),
            "test": test,
            "claim_ids": claim_ids,
            "assertions": assertions,
            "reconciliation": reconciliations,
        }
        self._pending = deepcopy(commitment)
        self.events.append({"schema": SCHEMA, "type": "commitment", **deepcopy(commitment)})
        return commitment

    def context_packet(self) -> dict[str, object]:
        """Return the fixed-order, model-visible treatment packet with no relevance ranking."""
        current = self._current
        if current is None:
            raise ContractError("no current public observation")
        packet = self._packet(
            working_state=self._working_state,
            working_claims=self._working_claims(self.ledger._claims, self._working_claim_ids),
            requested_evidence=self._requested_evidence,
            unresolved_mismatches=self._unresolved,
            retained_mismatches=self._retained,
        )
        self._require_context_capacity(packet)
        self.events.append(
            {
                "schema": SCHEMA,
                "type": "context_assembled",
                "packet_sha256": _digest(packet, domain="context_packet"),
            }
        )
        return packet

    def _packet(
        self,
        *,
        working_state: Mapping[str, object],
        working_claims: Sequence[Mapping[str, object]],
        requested_evidence: Sequence[Mapping[str, object]],
        unresolved_mismatches: Sequence[Mapping[str, object]],
        retained_mismatches: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        current = self._current
        if current is None:
            raise ContractError("no current public observation")
        return {
            "schema": SCHEMA,
            "current_observation": {"step": current.step, "evidence_id": current.evidence_id},
            "working_state": deepcopy(dict(working_state)),
            "working_claims": [deepcopy(dict(claim)) for claim in working_claims],
            "last_prediction_result": deepcopy(self._last_prediction_result),
            "unresolved_mismatches": [deepcopy(dict(item)) for item in unresolved_mismatches],
            "retained_mismatches": [deepcopy(dict(item)) for item in retained_mismatches],
            "recent_continuity": deepcopy(self._continuity),
            "requested_evidence": [deepcopy(dict(record)) for record in requested_evidence],
        }

    def _claim_ids(self, value: object) -> list[str]:
        # At least one, for the same reason assertions require at least one. A
        # commitment carrying no claim leaves the next action nothing to
        # continue, revise, or retire, so the continuity this treatment exists
        # to observe has no subject.
        if not isinstance(value, list) or not 1 <= len(value) <= _MAX_WORKING_CLAIMS:
            raise ContractError(
                f"decision.claim_ids must name one to {_MAX_WORKING_CLAIMS} active claims"
            )
        result = [_text(item, field="claim id", limit=80) for item in value]
        if len(result) != len(set(result)):
            raise ContractError("decision.claim_ids must not contain duplicates")
        return result

    def _project_ledger(self, operations: list[object]) -> dict[str, dict[str, object]]:
        snapshot = deepcopy(self.ledger._claims)
        sandbox = EpistemicLedger()
        sandbox._claims = snapshot
        sandbox.apply(operations, action_id=self._next_action_id)
        return sandbox._claims

    def _working_claims(
        self,
        claims: Mapping[str, Mapping[str, object]],
        claim_ids: Sequence[str],
    ) -> list[dict[str, object]]:
        selected: list[dict[str, object]] = []
        for claim_id in claim_ids:
            claim = claims.get(claim_id)
            if claim is None or claim.get("status") != "active":
                raise ContractError(f"decision.claim_ids names no active claim: {claim_id}")
            selected.append(deepcopy(dict(claim)))
        return selected

    def _next_retained(
        self, reconciliations: Sequence[Mapping[str, object]] | None
    ) -> list[dict[str, object]]:
        retained = deepcopy(self._retained)
        if reconciliations is None:
            return retained
        by_id = {str(item["mismatch_id"]): item for item in self._unresolved}
        for reconciliation in reconciliations:
            if reconciliation["disposition"] == "RETAIN":
                mismatch = deepcopy(dict(by_id[str(reconciliation["mismatch_id"])]))
                retained.append({**mismatch, "retained_note": reconciliation["note"]})
        return retained[-_MAX_RETAINED_MISMATCHES:]

    @staticmethod
    def _require_context_capacity(packet: Mapping[str, object]) -> None:
        bytes_used = len(_canonical_json(packet).encode("utf-8"))
        if bytes_used > _MAX_CONTEXT_BYTES:
            raise ContractError(
                f"assembled context exceeds {_MAX_CONTEXT_BYTES} bytes ({bytes_used})"
            )

    def _validate_reconciliation(self, value: object) -> list[dict[str, object]] | None:
        if not self._unresolved:
            if value is not None:
                raise ContractError("reconciliation is allowed only for an unresolved mismatch")
            return None
        if not isinstance(value, list) or len(value) != len(self._unresolved):
            raise ContractError("reconciliation must account for every unresolved mismatch")
        outstanding = {str(item["mismatch_id"]) for item in self._unresolved}
        resolved: list[dict[str, object]] = []
        for index, raw in enumerate(value):
            entry = _mapping(raw, field=f"reconciliation[{index}]")
            if set(entry) != {"mismatch_id", "disposition", "note"}:
                raise ContractError("reconciliation entries require mismatch_id, disposition, and note")
            mismatch_id = _text(entry.get("mismatch_id"), field="reconciliation.mismatch_id", limit=100)
            if mismatch_id not in outstanding:
                raise ContractError("reconciliation names an unknown or duplicate mismatch")
            disposition = entry.get("disposition")
            if disposition not in _DISPOSITIONS:
                raise ContractError(f"reconciliation.disposition must be one of {sorted(_DISPOSITIONS)}")
            resolved.append(
                {
                    "mismatch_id": mismatch_id,
                    "disposition": disposition,
                    "note": _text(entry.get("note"), field="reconciliation.note", limit=1_000),
                }
            )
            outstanding.remove(mismatch_id)
        if outstanding:
            raise ContractError("reconciliation omitted an unresolved mismatch")
        return resolved

    def _evaluate_pending(
        self,
        before: ObservationOccurrence,
        after: ObservationOccurrence,
    ) -> dict[str, object]:
        assert self._pending is not None
        transition = _transition_output(before.public, after.public)
        checks: list[dict[str, object]] = []
        mismatches: list[dict[str, object]] = []
        for index, assertion in enumerate(self._pending["assertions"]):
            assert isinstance(assertion, Mapping)
            observed = _assertion_value(assertion, transition=transition, after=after.public)
            matched = _compare(observed, str(assertion["op"]), assertion["value"])
            check = {**dict(assertion), "observed": observed, "matched": matched}
            checks.append(check)
            if matched is False:
                mismatch = {
                    "mismatch_id": f"mismatch-{self._pending['action_id']}-{index}",
                    "action_id": self._pending["action_id"],
                    "assertion": deepcopy(dict(assertion)),
                    "observed": observed,
                }
                mismatches.append(mismatch)
        self._unresolved.extend(mismatches)
        outcomes = [check["matched"] for check in checks]
        outcome = "mismatch" if False in outcomes else "match" if all(outcomes) else "inconclusive"
        return {
            "schema": SCHEMA,
            "action_id": self._pending["action_id"],
            "before_evidence_id": before.evidence_id,
            "after_evidence_id": after.evidence_id,
            "checks": checks,
            "outcome": outcome,
        }


def _assertion_value(
    assertion: Mapping[str, object],
    *,
    transition: Mapping[str, object],
    after: Mapping[str, object],
) -> object:
    observable = assertion["observable"]
    if observable == "changed_cells":
        return transition["changed_cells"]
    if observable == "levels_completed":
        return transition["levels_completed_after"]
    if observable == "state":
        return transition["state_after"]
    raise ContractError(f"unsupported assertion observable {observable!r}")
