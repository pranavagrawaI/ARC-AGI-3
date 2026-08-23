"""Durable authorization, routing, and accounting for compatibility inference."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CallState = Literal[
    "planned",
    "dispatched",
    "received",
    "route_validated",
    "route_violated",
    "accounted",
    "cancelled",
    "failed",
]
RunState = Literal["armed", "stopped", "completed"]


class PricingSchedule(BaseModel):
    """Frozen USD prices per million tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    currency: Literal["USD"] = "USD"
    input_usd_per_million: Decimal = Field(ge=0)
    cached_input_usd_per_million: Decimal = Field(ge=0)
    output_usd_per_million: Decimal = Field(ge=0)
    source: str = Field(min_length=1)
    effective_at: datetime

    @model_validator(mode="after")
    def cached_input_is_not_more_expensive(self) -> PricingSchedule:
        if self.cached_input_usd_per_million > self.input_usd_per_million:
            raise ValueError("cached-input price exceeds ordinary input price")
        if self.effective_at.tzinfo is None:
            raise ValueError("pricing effective_at must include a timezone")
        return self

    def maximum_call_cost(
        self, *, max_input_tokens: int, max_output_tokens: int
    ) -> Decimal:
        return _token_cost(
            input_tokens=max_input_tokens,
            cached_input_tokens=0,
            output_tokens=max_output_tokens,
            pricing=self,
        )

    def actual_call_cost(self, usage: dict[str, Any]) -> Decimal:
        input_tokens = _usage_integer(usage, "prompt_tokens", "input_tokens")
        output_tokens = _usage_integer(
            usage, "completion_tokens", "output_tokens"
        )
        cached_tokens = _cached_tokens(usage)
        if cached_tokens > input_tokens:
            raise ValueError("cached input tokens exceed total input tokens")
        return _token_cost(
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
            pricing=self,
        )


class CompatibilityAuthorization(BaseModel):
    """Human authority bound to one exact experiment-manifest value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    authorization_id: str = Field(min_length=1)
    experiment_manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    allowed_provider: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    expected_reported_model: str = Field(min_length=1)
    allowed_reasoning_effort: str = Field(min_length=1)
    allowed_temperature: float | None = None
    maximum_calls: int = Field(ge=1, le=30)
    maximum_spend_usd: Decimal = Field(gt=0)
    max_input_tokens_per_call: int = Field(ge=1)
    max_output_tokens_per_call: int = Field(ge=1)
    expires_at: datetime
    pricing: PricingSchedule

    @model_validator(mode="after")
    def route_and_pricing_match(self) -> CompatibilityAuthorization:
        if self.pricing.provider != self.allowed_provider:
            raise ValueError("pricing provider does not match authorized provider")
        if self.pricing.model != self.requested_model:
            raise ValueError("pricing model does not match requested model")
        if self.expires_at.tzinfo is None:
            raise ValueError("authorization expiry must include a timezone")
        return self

    @property
    def maximum_call_cost_usd(self) -> Decimal:
        return self.pricing.maximum_call_cost(
            max_input_tokens=self.max_input_tokens_per_call,
            max_output_tokens=self.max_output_tokens_per_call,
        )

    def require_current(self, *, now: datetime | None = None) -> None:
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            raise ValueError("current time must include a timezone")
        if current >= self.expires_at:
            raise RuntimeError("compatibility authorization has expired")


class CallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    call_index: int
    gate_id: str
    arm: Literal["native", "codeact"]
    request_sha256: str
    reservation_usd: Decimal
    state: CallState
    request: dict[str, Any]
    response: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    observed_provider: str | None = None
    observed_model: str | None = None
    observed_reasoning_effort: str | None = None
    actual_cost_usd: Decimal | None = None
    failure: str | None = None


class GateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate_id: str
    state: Literal["started", "completed"]
    passed: bool | None = None
    report_sha256: str | None = None


class AuthorizationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authorization_id: str
    authorization_sha256: str
    experiment_manifest_sha256: str
    run_state: RunState
    stop_reason: str | None = None
    calls: dict[str, CallRecord] = Field(default_factory=dict)
    gates: dict[str, GateRecord] = Field(default_factory=dict)
    ladder_disposition: str | None = None

    @property
    def active_call(self) -> CallRecord | None:
        active = [
            call
            for call in self.calls.values()
            if call.state not in {"accounted", "cancelled", "failed"}
        ]
        if len(active) > 1:
            raise ValueError("journal contains concurrent provider calls")
        return active[0] if active else None

    @property
    def committed_calls(self) -> int:
        return sum(call.state != "cancelled" for call in self.calls.values())

    @property
    def committed_spend_usd(self) -> Decimal:
        return sum(
            (
                call.actual_cost_usd
                if call.actual_cost_usd is not None
                else Decimal(0)
                if call.state == "cancelled"
                else call.reservation_usd
            )
            for call in self.calls.values()
        )

    @property
    def active_gate(self) -> GateRecord | None:
        active = [gate for gate in self.gates.values() if gate.state == "started"]
        if len(active) > 1:
            raise ValueError("journal contains concurrent ladder gates")
        return active[0] if active else None


class DurableAuthorizationLedger:
    """Append-first call transitions whose state is reconstructed from disk."""

    def __init__(
        self,
        path: Path,
        *,
        authorization: CompatibilityAuthorization,
        authorization_sha256: str,
    ):
        self.path = path
        self.authorization = authorization
        self.authorization_sha256 = authorization_sha256
        self.state = self._replay()

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        authorization: CompatibilityAuthorization,
        authorization_sha256: str,
    ) -> DurableAuthorizationLedger:
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ledger = cls.__new__(cls)
        ledger.path = path
        ledger.authorization = authorization
        ledger.authorization_sha256 = authorization_sha256
        ledger.state = AuthorizationState(
            authorization_id=authorization.authorization_id,
            authorization_sha256=authorization_sha256,
            experiment_manifest_sha256=authorization.experiment_manifest_sha256,
            run_state="armed",
        )
        ledger._append(
            "RunArmed",
            {
                "authorization_id": authorization.authorization_id,
                "authorization_sha256": authorization_sha256,
                "experiment_manifest_sha256": (
                    authorization.experiment_manifest_sha256
                ),
            },
        )
        return ledger

    def recover(self) -> AuthorizationState:
        """Resolve only transitions whose consequence is knowable from the journal."""
        call = self.state.active_call
        if call is not None:
            if call.state == "planned":
                self.cancel_planned(call.call_id, reason="recovery_before_dispatch")
            elif call.state == "dispatched":
                self.stop("unknown_dispatched_call")
                return self.state
            elif call.state in {"received", "route_validated", "route_violated"}:
                self._finish_received_call(call)
        latest = max(self.state.calls.values(), key=lambda row: row.call_index, default=None)
        if self.state.run_state == "armed" and latest is not None:
            if latest.state == "failed":
                self.stop(f"provider_failure:{latest.call_id}")
                return self.state
            if latest.state == "accounted" and not _route_matches(
                latest, self.authorization
            ):
                self.stop(f"route_violation:{latest.call_id}")
                return self.state
        if self.state.run_state == "armed" and self.state.active_gate is not None:
            self.stop(f"interrupted_gate:{self.state.active_gate.gate_id}")
        elif (
            self.state.run_state == "armed"
            and self.state.gates
            and self.state.ladder_disposition is None
        ):
            self.stop("interrupted_ladder")
        return self.state

    def start_gate(self, gate_id: str) -> None:
        self._require_armed()
        if self.state.active_call is not None or self.state.active_gate is not None:
            raise RuntimeError("cannot start a gate while other work is active")
        if gate_id in self.state.gates:
            raise RuntimeError("compatibility gate was already started")
        self._append("GateStarted", {"gate_id": gate_id})

    def complete_gate(self, gate_id: str, *, passed: bool, report: object) -> None:
        self._require_armed()
        if self.state.active_call is not None:
            raise RuntimeError("cannot complete a gate with an active provider call")
        gate = self.state.active_gate
        if gate is None or gate.gate_id != gate_id:
            raise RuntimeError("gate completion does not match the active gate")
        self._append(
            "GateCompleted",
            {
                "gate_id": gate_id,
                "passed": passed,
                "report_sha256": sha256_json(report),
            },
        )

    def finish_ladder(self, disposition: str, *, report: object) -> None:
        self._require_armed()
        if self.state.active_call is not None or self.state.active_gate is not None:
            raise RuntimeError("cannot finish the ladder while work is active")
        self._append(
            "LadderCompleted",
            {"disposition": disposition, "report_sha256": sha256_json(report)},
        )
        self.complete()

    def plan_call(
        self,
        *,
        gate_id: str,
        arm: Literal["native", "codeact"],
        request: dict[str, Any],
    ) -> CallRecord:
        self._require_armed()
        self.authorization.require_current()
        if self.state.active_call is not None:
            raise RuntimeError("another provider call is already active")
        if self.state.committed_calls >= self.authorization.maximum_calls:
            self.stop("call_cap")
            raise RuntimeError("global provider-call cap reached")
        reservation = self.authorization.maximum_call_cost_usd
        if (
            self.state.committed_spend_usd + reservation
            > self.authorization.maximum_spend_usd
        ):
            self.stop("cost_reservation")
            raise RuntimeError("global worst-case cost reservation refused")
        call_index = len(self.state.calls) + 1
        call_id = f"call-{call_index:04d}"
        payload = {
            "call_id": call_id,
            "call_index": call_index,
            "gate_id": gate_id,
            "arm": arm,
            "request_sha256": sha256_json(request),
            "reservation_usd": str(reservation),
            "request": request,
        }
        self._append("CallPlanned", payload)
        return self.state.calls[call_id]

    def dispatch(self, call_id: str) -> None:
        self._transition(call_id, {"planned"}, "CallDispatched", {})

    def receive(
        self,
        call_id: str,
        *,
        response: dict[str, Any],
        usage: dict[str, Any],
        observed_provider: str | None,
        observed_model: str | None,
        observed_reasoning_effort: str | None = None,
    ) -> None:
        self._transition(
            call_id,
            {"dispatched"},
            "CallReceived",
            {
                "response": response,
                "usage": usage,
                "observed_provider": observed_provider,
                "observed_model": observed_model,
                "observed_reasoning_effort": observed_reasoning_effort,
            },
        )

    def reconcile(self, call_id: str) -> Decimal:
        call = self._call(call_id)
        if call.state not in {"received", "route_validated", "route_violated"}:
            raise RuntimeError(f"cannot reconcile call in state {call.state}")
        return self._finish_received_call(call)

    def fail(
        self, call_id: str, *, reason: str, response: dict[str, Any] | None = None
    ) -> None:
        self._transition(
            call_id,
            {"dispatched"},
            "CallFailed",
            {"failure": reason, "response": response},
        )
        self.stop(f"provider_failure:{call_id}")

    def cancel_planned(self, call_id: str, *, reason: str) -> None:
        self._transition(
            call_id,
            {"planned"},
            "CallCancelled",
            {"failure": reason},
        )

    def stop(self, reason: str) -> None:
        if self.state.run_state == "completed":
            raise RuntimeError("completed authorization cannot be stopped")
        if self.state.run_state == "stopped":
            return
        self._append("RunStopped", {"reason": reason})

    def complete(self) -> None:
        self._require_armed()
        if self.state.active_call is not None or self.state.active_gate is not None:
            raise RuntimeError("cannot complete while work is active")
        self._append("RunCompleted", {})

    def _finish_received_call(self, call: CallRecord) -> Decimal:
        if call.usage is None:
            self.stop(f"missing_usage:{call.call_id}")
            raise RuntimeError("received call lacks usage accounting")
        if call.state == "received":
            route_ok = _route_matches(call, self.authorization)
            event = "RouteValidated" if route_ok else "RouteViolated"
            self._transition(call.call_id, {"received"}, event, {})
            call = self._call(call.call_id)
        try:
            _validate_usage_caps(call.usage, self.authorization)
            actual = self.authorization.pricing.actual_call_cost(call.usage)
        except ValueError as error:
            self.stop(f"accounting_failure:{call.call_id}:{error}")
            raise
        self._transition(
            call.call_id,
            {"route_validated", "route_violated"},
            "CallAccounted",
            {"actual_cost_usd": str(actual)},
        )
        if call.state == "route_violated":
            self.stop(f"route_violation:{call.call_id}")
        return actual

    def _transition(
        self,
        call_id: str,
        allowed: set[CallState],
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        call = self._call(call_id)
        if call.state not in allowed:
            raise RuntimeError(
                f"{event_type} is invalid from call state {call.state}"
            )
        self._append(event_type, {"call_id": call_id, **payload})

    def _call(self, call_id: str) -> CallRecord:
        try:
            return self.state.calls[call_id]
        except KeyError as error:
            raise KeyError(f"unknown provider call {call_id}") from error

    def _require_armed(self) -> None:
        if self.state.run_state != "armed":
            raise RuntimeError(f"authorization run is {self.state.run_state}")

    def _append(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "event_seq": _event_count(self.path) + 1,
            "event_type": event_type,
            "recorded_at": datetime.now(UTC).isoformat(),
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.state = self._replay()

    def _replay(self) -> AuthorizationState:
        events = _read_events(self.path)
        if not events:
            raise ValueError("authorization journal is empty")
        state: AuthorizationState | None = None
        for expected_seq, event in enumerate(events, start=1):
            if event.get("event_seq") != expected_seq:
                raise ValueError("authorization journal event sequence is invalid")
            state = _apply_event(state, event)
        assert state is not None
        _validate_journal_identity(
            state,
            authorization=self.authorization,
            authorization_sha256=self.authorization_sha256,
        )
        return state


def authorization_sha256(authorization: CompatibilityAuthorization) -> str:
    return sha256_json(authorization.model_dump(mode="json"))


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _apply_event(
    state: AuthorizationState | None, event: dict[str, Any]
) -> AuthorizationState:
    event_type = event.get("event_type")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise TypeError("authorization journal payload must be an object")
    if event_type == "RunArmed":
        if state is not None:
            raise ValueError("authorization journal contains duplicate RunArmed")
        return AuthorizationState(
            authorization_id=payload["authorization_id"],
            authorization_sha256=payload["authorization_sha256"],
            experiment_manifest_sha256=payload["experiment_manifest_sha256"],
            run_state="armed",
        )
    if state is None:
        raise ValueError("authorization journal must begin with RunArmed")
    if event_type == "CallPlanned":
        _replay_plan(state, payload)
    elif event_type in {
        "GateStarted",
        "GateCompleted",
        "LadderCompleted",
    }:
        _replay_ladder_transition(state, event_type, payload)
    elif event_type in {
        "CallDispatched",
        "CallReceived",
        "RouteValidated",
        "RouteViolated",
        "CallAccounted",
        "CallCancelled",
        "CallFailed",
    }:
        _replay_call_transition(state, event_type, payload)
    elif event_type == "RunStopped":
        if state.run_state != "armed":
            raise ValueError("RunStopped requires an armed run")
        state.run_state = "stopped"
        state.stop_reason = str(payload["reason"])
    elif event_type == "RunCompleted":
        if (
            state.run_state != "armed"
            or state.active_call is not None
            or state.active_gate is not None
        ):
            raise ValueError("RunCompleted requires an idle armed run")
        state.run_state = "completed"
    else:
        raise ValueError(f"unknown authorization journal event {event_type!r}")
    return state


def _replay_plan(state: AuthorizationState, payload: dict[str, Any]) -> None:
    if state.run_state != "armed" or state.active_call is not None:
        raise ValueError("CallPlanned requires an idle armed run")
    call = CallRecord(
        call_id=payload["call_id"],
        call_index=payload["call_index"],
        gate_id=payload["gate_id"],
        arm=payload["arm"],
        request_sha256=payload["request_sha256"],
        reservation_usd=Decimal(payload["reservation_usd"]),
        state="planned",
        request=payload["request"],
    )
    if call.call_id in state.calls:
        raise ValueError("authorization journal reuses a call id")
    if call.call_index != len(state.calls) + 1:
        raise ValueError("authorization journal call index is invalid")
    state.calls[call.call_id] = call


def _replay_call_transition(
    state: AuthorizationState, event_type: str, payload: dict[str, Any]
) -> None:
    call_id = payload.get("call_id")
    if not isinstance(call_id, str) or call_id not in state.calls:
        raise ValueError("call transition names an unknown call")
    call = state.calls[call_id]
    transitions: dict[str, tuple[set[CallState], CallState]] = {
        "CallDispatched": ({"planned"}, "dispatched"),
        "CallReceived": ({"dispatched"}, "received"),
        "RouteValidated": ({"received"}, "route_validated"),
        "RouteViolated": ({"received"}, "route_violated"),
        "CallAccounted": ({"route_validated", "route_violated"}, "accounted"),
        "CallCancelled": ({"planned"}, "cancelled"),
        "CallFailed": ({"dispatched"}, "failed"),
    }
    allowed, target = transitions[event_type]
    if call.state not in allowed:
        raise ValueError(f"{event_type} is invalid from call state {call.state}")
    call.state = target
    if event_type == "CallReceived":
        call.response = payload["response"]
        call.usage = payload["usage"]
        call.observed_provider = payload.get("observed_provider")
        call.observed_model = payload.get("observed_model")
        call.observed_reasoning_effort = payload.get("observed_reasoning_effort")
    elif event_type == "CallAccounted":
        call.actual_cost_usd = Decimal(payload["actual_cost_usd"])
    elif event_type in {"CallCancelled", "CallFailed"}:
        call.failure = str(payload["failure"])
        if event_type == "CallFailed":
            call.response = payload.get("response")


def _replay_ladder_transition(
    state: AuthorizationState, event_type: str, payload: dict[str, Any]
) -> None:
    if state.run_state != "armed":
        raise ValueError(f"{event_type} requires an armed run")
    if event_type == "GateStarted":
        gate_id = str(payload["gate_id"])
        if state.active_gate is not None or gate_id in state.gates:
            raise ValueError("GateStarted requires a new idle gate")
        state.gates[gate_id] = GateRecord(gate_id=gate_id, state="started")
        return
    if event_type == "GateCompleted":
        gate = state.active_gate
        if gate is None or gate.gate_id != payload.get("gate_id"):
            raise ValueError("GateCompleted does not match the active gate")
        gate.state = "completed"
        gate.passed = bool(payload["passed"])
        gate.report_sha256 = str(payload["report_sha256"])
        return
    if state.active_gate is not None:
        raise ValueError("LadderCompleted requires no active gate")
    if state.ladder_disposition is not None:
        raise ValueError("authorization journal repeats LadderCompleted")
    state.ladder_disposition = str(payload["disposition"])


def _validate_journal_identity(
    state: AuthorizationState,
    *,
    authorization: CompatibilityAuthorization,
    authorization_sha256: str,
) -> None:
    if state.authorization_id != authorization.authorization_id:
        raise ValueError("journal authorization id does not match")
    if state.authorization_sha256 != authorization_sha256:
        raise ValueError("journal authorization commitment does not match")
    if (
        state.experiment_manifest_sha256
        != authorization.experiment_manifest_sha256
    ):
        raise ValueError("journal experiment commitment does not match")


def _validate_usage_caps(
    usage: dict[str, Any], authorization: CompatibilityAuthorization
) -> None:
    input_tokens = _usage_integer(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_integer(usage, "completion_tokens", "output_tokens")
    if input_tokens > authorization.max_input_tokens_per_call:
        raise ValueError("reported input tokens exceed authorized per-call cap")
    if output_tokens > authorization.max_output_tokens_per_call:
        raise ValueError("reported output tokens exceed authorized per-call cap")


def _route_matches(
    call: CallRecord, authorization: CompatibilityAuthorization
) -> bool:
    return (
        call.observed_provider == authorization.allowed_provider
        and call.observed_model == authorization.expected_reported_model
        and (
            call.observed_reasoning_effort is None
            or call.observed_reasoning_effort
            == authorization.allowed_reasoning_effort
        )
    )


def _usage_integer(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    raise ValueError(f"provider usage lacks nonnegative {names[0]}")


def _cached_tokens(usage: dict[str, Any]) -> int:
    for key in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict):
            value = details.get("cached_tokens")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return 0


def _token_cost(
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    pricing: PricingSchedule,
) -> Decimal:
    million = Decimal(1_000_000)
    uncached = input_tokens - cached_input_tokens
    return (
        Decimal(uncached) * pricing.input_usd_per_million
        + Decimal(cached_input_tokens) * pricing.cached_input_usd_per_million
        + Decimal(output_tokens) * pricing.output_usd_per_million
    ) / million


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"authorization journal line {line_number} is corrupt"
            ) from error
        if not isinstance(event, dict):
            raise TypeError("authorization journal event must be an object")
        events.append(event)
    return events


def _event_count(path: Path) -> int:
    return len(_read_events(path))
