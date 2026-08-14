from __future__ import annotations

import json
from pathlib import Path

import pytest
from host.failure import FailurePacketWriter, verify_packet
from host.live import (
    ActiveTimeBudget,
    LiveRunGuard,
    LiveRunTerminated,
    OfflineQualificationGuard,
    TransportRetryGuard,
    UsageLedger,
)
from host.review import apply_review


def frame(evidence_id: str, value: int) -> dict:
    return {"evidence_id": evidence_id, "kind": "frame", "grid": [[value, 0], [0, 0]]}


def causal() -> dict:
    return {
        "observation": "exact mismatch",
        "proximate_failure": "contradicted",
        "live_causal_hypotheses": ["candidate wrong", "effect wrong"],
        "discriminating_probe": "test one alternative",
        "conclusion": "pending review",
        "system_decision": "pause",
        "failure_class": "gemma_semantic",
    }


def test_failure_packet_is_atomic_exact_and_review_is_outside_immutable_checksums(tmp_path: Path) -> None:
    packet = FailurePacketWriter(tmp_path / "packets").create(
        before=frame("before", 1), after=frame("after", 2),
        grounding_claim={"claim_id": "c"}, probe_commitment={"commitment_id": "p"},
        verifier_verdict={"status": "contradicted"}, prompt={"redacted": True},
        callable_surface={"names": ["frames"]}, state_before={"gate": "G0"},
        state_after={"gate": "INVALIDATED"}, trace=[{"kind": "x"}],
        promotion_history=[], causal_record=causal(),
    )
    required = {
        "before.txt", "before.png", "before.json", "after.txt", "after.png", "after.json",
        "diff.txt", "grounding-claim.json", "probe-commitment.json", "verifier-verdict.json",
        "prompt.json", "callable-surface.json", "state-before.json", "state-after.json",
        "trace.jsonl", "causal-files.json", "promotion-history.jsonl", "review.yaml", "SHA256SUMS",
    }
    assert required <= {path.name for path in packet.iterdir()}
    verified = verify_packet(packet)
    assert "review.yaml" not in verified
    (packet / "review.yaml").write_text(json.dumps({"schema": "0158-review-v1", "status": "pending"}))
    verify_packet(packet)
    (packet / "diff.txt").write_text("tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_packet(packet)


def test_live_review_creates_only_a_next_run_decision_and_cannot_resume_source(tmp_path: Path) -> None:
    packet = FailurePacketWriter(tmp_path / "run/failure-packets").create(
        before=frame("before", 1), after=frame("after", 2), grounding_claim=None,
        probe_commitment=None, verifier_verdict=None, prompt={}, callable_surface={},
        state_before={}, state_after={}, trace=[], promotion_history=[], causal_record=causal(),
    )
    review = apply_review(
        packet,
        {
            "schema": "0158-review-v1", "reviewer": "Pranav", "decision": "revise", "rationale": "change the candidate test",
            "next_run_changes": {"candidate_revision": "v2"},
        },
        run_root=tmp_path / "run",
        mode="live",
    )
    assert review["resume_scored_run"] is False
    next_run = json.loads((packet / "next-run.json").read_text())
    assert next_run["requires_new_run_id"] is True and next_run["resume_source_run"] is False

    guard = LiveRunGuard(tmp_path / "scored", card_digest="card", approval_digest="approval")
    token = guard.authorize_one_action("commitment-a")
    assert token.startswith("action-token-")
    guard.record_successor("commitment-a")
    with pytest.raises(LiveRunTerminated):
        guard.consequential_failure(packet.name, levels_completed=0)
    with pytest.raises(LiveRunTerminated):
        guard.resume()
    scorecard = json.loads((tmp_path / "scored/scorecard.json").read_text())
    assert scorecard["terminal"] is True and scorecard["resumable"] is False


def test_offline_pause_resume_is_available_but_live_requires_separate_approval(tmp_path: Path) -> None:
    offline = OfflineQualificationGuard()
    offline.pause(); offline.resume()
    assert offline.paused is False
    with pytest.raises(PermissionError, match="approval"):
        LiveRunGuard(tmp_path / "unapproved", card_digest="card", approval_digest=None)


def test_usage_and_transport_budgets_are_mechanical() -> None:
    ledger = UsageLedger(max_calls=2, max_reported_tokens=100)
    ledger.record({"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 99})
    ledger.record({"input_tokens": 20, "output_tokens": 6, "cache_read_tokens": 500}, retry=True)
    assert ledger.reported_tokens == 41
    assert ledger.records[0]["reasoning_tokens"] == 99
    with pytest.raises(RuntimeError, match="MODEL_CALL_CAP"):
        ledger.record({"input_tokens": 1, "output_tokens": 1})

    attempts = UsageLedger(max_calls=2)
    attempts.record_transport_failure(retry=False, disposition="retryable")
    attempts.record({"input_tokens": 3, "output_tokens": 2}, retry=True)
    assert attempts.calls == 2 and attempts.reported_tokens == 5

    missing = UsageLedger()
    with pytest.raises(ValueError, match="MISSING_REQUIRED"):
        missing.record({"total_tokens": 10})

    retry = TransportRetryGuard(original_request_sha256="same")
    retry.authorize(request_sha256="same", output_started=False)
    with pytest.raises(RuntimeError, match="exhausted"):
        retry.authorize(request_sha256="same", output_started=False)
    with pytest.raises(RuntimeError, match="after response output"):
        TransportRetryGuard(original_request_sha256="same").authorize(request_sha256="same", output_started=True)
    with pytest.raises(RuntimeError, match="byte-identical"):
        TransportRetryGuard(original_request_sha256="same").authorize(request_sha256="different", output_started=False)


def test_active_time_excludes_review_and_success_requires_level_completion(tmp_path: Path) -> None:
    now = [0.0]
    budget = ActiveTimeBudget(max_seconds=90.0, clock=lambda: now[0])
    now[0] = 30.0; budget.pause_for_review()
    now[0] = 1_030.0
    assert budget.active_seconds() == 30.0
    budget.resume_after_review(); now[0] = 1_090.0; budget.check()
    now[0] = 1_091.0
    with pytest.raises(RuntimeError, match="ACTIVE_WALL"):
        budget.check()

    guard = LiveRunGuard(tmp_path / "success", card_digest="card", approval_digest="approval")
    with pytest.raises(ValueError, match="levels_completed"):
        guard.complete(levels_completed=0)
    guard.complete(levels_completed=1)
    scorecard = json.loads((tmp_path / "success/scorecard.json").read_text())
    assert scorecard["success"] is True and scorecard["levels_completed"] == 1


def test_g2_stop_is_terminal_review_only_and_never_resumable(tmp_path: Path) -> None:
    guard = LiveRunGuard(tmp_path / "g2", card_digest="card", approval_digest="approval")
    guard.authorize_one_action("commitment")
    guard.record_successor("commitment")
    guard.stop_at_g2(levels_completed=0)
    scorecard = json.loads((tmp_path / "g2/scorecard.json").read_text())
    assert scorecard["state"] == "STOPPED_AT_G2"
    assert scorecard["review_required"] is True
    assert scorecard["resumable"] is False
    with pytest.raises(LiveRunTerminated):
        guard.resume()


def test_uncertain_action_terminalizes_pending_authority_without_consuming_it(tmp_path: Path) -> None:
    guard = LiveRunGuard(tmp_path / "uncertain", card_digest="card", approval_digest="approval")
    guard.authorize_one_action("commitment-uncertain")
    with pytest.raises(LiveRunTerminated, match="uncertain environment action"):
        guard.uncertain_action_failure(
            reason="environment_read_timeout",
            levels_completed=0,
            packet_id="packet-a",
        )
    scorecard = json.loads((tmp_path / "uncertain/scorecard.json").read_text())
    assert scorecard["state"] == "TERMINATED_UNCERTAIN_ACTION"
    assert scorecard["unconfirmed_commitment_id"] == "commitment-uncertain"
    assert scorecard["resumable"] is False
