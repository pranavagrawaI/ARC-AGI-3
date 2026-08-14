"""Frozen live-run authority, one-action control, and mechanical usage accounting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .canonical import append_jsonl, content_sha256, write_atomic_json


class LiveRunTerminated(RuntimeError):
    """The scored trajectory is closed and cannot be resumed."""


@dataclass
class UsageLedger:
    max_calls: int = 200
    max_reported_tokens: int = 5_000_000
    max_output_tokens_per_generation: int = 4_096
    calls: int = 0
    reported_tokens: int = 0
    invalid_reason: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining_calls(self) -> int:
        return self.max_calls - self.calls

    def _consume_call(self) -> None:
        self.calls += 1
        if self.calls > self.max_calls:
            self.invalid_reason = "MODEL_CALL_CAP_EXCEEDED"
            raise RuntimeError(self.invalid_reason)

    def record(self, usage: dict[str, Any], *, retry: bool = False) -> None:
        self._consume_call()
        input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        if type(input_tokens) is not int or input_tokens < 0 or type(output_tokens) is not int or output_tokens < 0:
            self.invalid_reason = "MISSING_REQUIRED_PROVIDER_USAGE"
            raise ValueError(self.invalid_reason)
        counted = input_tokens + output_tokens
        record = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "counted_tokens": counted,
            "retry": retry,
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "cache_read_tokens": usage.get("cache_read_tokens"),
            "cache_write_tokens": usage.get("cache_write_tokens"),
        }
        self.records.append(record)
        self.reported_tokens += counted
        if output_tokens > self.max_output_tokens_per_generation:
            self.invalid_reason = "OUTPUT_TOKEN_CAP_EXCEEDED"
            raise RuntimeError(self.invalid_reason)
        if self.reported_tokens > self.max_reported_tokens:
            self.invalid_reason = "REPORTED_TOKEN_CAP_EXCEEDED"
            raise RuntimeError(self.invalid_reason)

    def record_transport_failure(self, *, retry: bool, disposition: str) -> None:
        """Count a pre-output provider attempt while recording that it reported no usage."""
        if disposition not in {"retryable", "fatal", "uncertain"}:
            raise ValueError("unknown transport disposition")
        self._consume_call()
        self.records.append(
            {
                "input_tokens": None,
                "output_tokens": None,
                "counted_tokens": 0,
                "retry": retry,
                "transport_disposition": disposition,
                "reasoning_tokens": None,
                "cache_read_tokens": None,
                "cache_write_tokens": None,
            }
        )


@dataclass
class TransportRetryGuard:
    original_request_sha256: str
    used: bool = False

    def authorize(self, *, request_sha256: str, output_started: bool) -> None:
        if output_started:
            raise RuntimeError("transport retry forbidden after response output")
        if self.used:
            raise RuntimeError("transport retry budget exhausted")
        if request_sha256 != self.original_request_sha256:
            raise RuntimeError("transport retry request is not byte-identical")
        self.used = True


class ActiveTimeBudget:
    """Count active execution time while excluding explicit human-review intervals."""

    def __init__(self, *, max_seconds: float = 5_400.0, clock=time.monotonic) -> None:
        self.max_seconds = max_seconds
        self.clock = clock
        self._started = clock()
        self._paused_at: float | None = None
        self._excluded = 0.0

    def pause_for_review(self) -> None:
        if self._paused_at is not None:
            raise RuntimeError("human-review timer is already paused")
        self._paused_at = self.clock()

    def resume_after_review(self) -> None:
        if self._paused_at is None:
            raise RuntimeError("human-review timer is not paused")
        self._excluded += self.clock() - self._paused_at
        self._paused_at = None

    def active_seconds(self) -> float:
        end = self._paused_at if self._paused_at is not None else self.clock()
        return end - self._started - self._excluded

    def check(self) -> None:
        if self.active_seconds() > self.max_seconds:
            raise RuntimeError("ACTIVE_WALL_TIME_CAP_EXCEEDED")


class LiveRunGuard:
    """No review can reopen this object's scored action authority."""

    def __init__(self, run_root: Path, *, card_digest: str, approval_digest: str | None) -> None:
        if approval_digest is None:
            raise PermissionError("the frozen card has no separate live-run approval")
        self.run_root = run_root
        self.run_root.mkdir(parents=True, exist_ok=False)
        self.run_id = uuid4().hex
        self.card_digest = card_digest
        self.approval_digest = approval_digest
        self.state = "ACTIVE"
        self.action_authority_open = True
        self.pending_commitment_id: str | None = None
        self.consumed_commitments: set[str] = set()
        self.actions = 0
        write_atomic_json(
            self.run_root / "run.json",
            {
                "schema": "0158-live-run-v1",
                "run_id": self.run_id,
                "card_digest": card_digest,
                "approval_digest": approval_digest,
                "state": self.state,
            },
        )

    def authorize_one_action(self, commitment_id: str) -> str:
        if self.state != "ACTIVE" or not self.action_authority_open:
            raise LiveRunTerminated("scored action authority is closed")
        if self.pending_commitment_id is not None:
            raise RuntimeError("one commitment is already awaiting its successor")
        if commitment_id in self.consumed_commitments:
            raise RuntimeError("commitment has already authorized an action")
        if self.actions >= 44:
            raise RuntimeError("contestant action cap reached")
        self.pending_commitment_id = commitment_id
        token = "action-token-" + content_sha256({"run_id": self.run_id, "commitment_id": commitment_id})
        append_jsonl(self.run_root / "authority.jsonl", {"event": "authorized", "commitment_id": commitment_id, "token": token})
        return token

    def record_successor(self, commitment_id: str) -> None:
        if commitment_id != self.pending_commitment_id:
            raise RuntimeError("successor does not close the pending commitment")
        self.pending_commitment_id = None
        self.consumed_commitments.add(commitment_id)
        self.actions += 1
        append_jsonl(self.run_root / "authority.jsonl", {"event": "consumed", "commitment_id": commitment_id})

    def consequential_failure(self, packet_id: str, *, levels_completed: int) -> None:
        self.state = "PAUSED_FOR_REVIEW"
        self.action_authority_open = False
        self.pending_commitment_id = None
        write_atomic_json(
            self.run_root / "scorecard.json",
            {
                "schema": "0158-scorecard-v1",
                "run_id": self.run_id,
                "state": self.state,
                "terminal": True,
                "levels_completed": levels_completed,
                "review_packet_id": packet_id,
                "resumable": False,
            },
        )
        raise LiveRunTerminated("consequential failure closed the scored trajectory")

    def uncertain_action_failure(
        self,
        *,
        reason: str,
        levels_completed: int,
        packet_id: str | None = None,
    ) -> None:
        """Terminalize when an authorized action has no confirmed successor."""
        if self.state != "ACTIVE" or self.pending_commitment_id is None:
            raise RuntimeError("uncertain-action stop requires a pending active commitment")
        pending = self.pending_commitment_id
        self.state = "TERMINATED_UNCERTAIN_ACTION"
        self.action_authority_open = False
        self.pending_commitment_id = None
        write_atomic_json(
            self.run_root / "scorecard.json",
            {
                "schema": "0158-scorecard-v1",
                "run_id": self.run_id,
                "state": self.state,
                "terminal": True,
                "levels_completed": levels_completed,
                "success": False,
                "stop_reason": reason,
                "unconfirmed_commitment_id": pending,
                "review_packet_id": packet_id,
                "resumable": False,
            },
        )
        raise LiveRunTerminated("uncertain environment action closed the scored trajectory")

    def resume(self) -> None:
        raise LiveRunTerminated("a reviewed scored run can never resume; create a new run ID")

    def complete(self, *, levels_completed: int) -> None:
        if levels_completed < 1:
            raise ValueError("gameplay success requires levels_completed >= 1")
        if self.state != "ACTIVE" or self.pending_commitment_id is not None:
            raise RuntimeError("live run cannot complete from the current authority state")
        self.state = "COMPLETED"
        self.action_authority_open = False
        write_atomic_json(
            self.run_root / "scorecard.json",
            {
                "schema": "0158-scorecard-v1",
                "run_id": self.run_id,
                "state": self.state,
                "terminal": True,
                "levels_completed": levels_completed,
                "success": True,
                "resumable": False,
            },
        )

    def stop_at_g2(self, *, levels_completed: int) -> None:
        if self.state != "ACTIVE" or self.pending_commitment_id is not None:
            raise RuntimeError("G2 stop requires a closed action and active authority")
        self.state = "STOPPED_AT_G2"
        self.action_authority_open = False
        write_atomic_json(
            self.run_root / "scorecard.json",
            {
                "schema": "0158-scorecard-v1",
                "run_id": self.run_id,
                "state": self.state,
                "terminal": True,
                "levels_completed": levels_completed,
                "success": levels_completed >= 1,
                "stop_reason": "g2_transition_eligibility",
                "review_required": True,
                "resumable": False,
            },
        )

    def stop_without_success(self, *, reason: str, levels_completed: int) -> None:
        if self.state != "ACTIVE" or self.pending_commitment_id is not None:
            raise RuntimeError("unsuccessful stop requires closed active authority")
        if levels_completed >= 1:
            raise ValueError("successful level completion must use complete()")
        self.state = "STOPPED"
        self.action_authority_open = False
        write_atomic_json(
            self.run_root / "scorecard.json",
            {
                "schema": "0158-scorecard-v1",
                "run_id": self.run_id,
                "state": self.state,
                "terminal": True,
                "levels_completed": levels_completed,
                "success": False,
                "stop_reason": reason,
                "resumable": False,
            },
        )


class OfflineQualificationGuard:
    """Pause/resume exists only to qualify mechanics without a scored trajectory."""

    def __init__(self) -> None:
        self.paused = False

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        if not self.paused:
            raise RuntimeError("offline qualification is not paused")
        self.paused = False
