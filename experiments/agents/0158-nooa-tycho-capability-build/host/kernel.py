"""Environment-agnostic analyze -> commit -> one action -> verdict kernel."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .canonical import append_jsonl, read_jsonl, write_atomic_json
from .evidence import EvidenceStore
from .failure import FailurePacketWriter
from .gates import GateTracker
from .ipc import ArtifactBridge, ArtifactGenerationError, ArtifactWorkerError
from .protocol import (
    GroundingClaim,
    GroundingVerdict,
    ProbeCommitment,
    ProtocolError,
    Unchanged,
    grounding_claim_from_draft,
    predicted_effect_signature,
    probe_commitment_from_draft,
)
from .verifier import account_visible_change, verify_effect


@dataclass(frozen=True)
class ActionEnvelope:
    commitment_id: str
    action: int
    data: dict[str, Any] | None


@dataclass(frozen=True)
class _SupportedPrediction:
    claim: GroundingClaim
    commitment: ProbeCommitment


class QualificationPaused(RuntimeError):
    def __init__(self, packet: Path, failure_class: str) -> None:
        self.packet = packet
        self.failure_class = failure_class
        super().__init__(f"qualification paused for review: {failure_class}; packet={packet.name}")


def _callable_surface(phase: str) -> dict[str, Any]:
    names = [
        "inspect_evidence",
        "return_result",
    ]
    if phase == "claim":
        names.extend(
            ["account_prior", "make_mask", "provisional_heuristic", "submit_grounding_claim"]
        )
    elif phase == "commitment":
        names.extend(
            [
                "action_prediction",
                "cell_after",
                "commit_contrastive",
                "predict_cell_values",
                "predict_translate",
                "predict_unchanged",
            ]
        )
    else:
        raise ValueError("callable surface phase must be claim or commitment")
    return {
        "names": sorted(names),
        "packages": ["numpy", "PIL"],
        "persistent_between_artifacts": False,
    }


class GroundingKernel:
    """The only component that can turn one accepted commitment into an action."""

    def __init__(
        self,
        *,
        run_root: Path,
        evidence: EvidenceStore,
        bridge: ArtifactBridge,
        level: int = 0,
        autonomous_continuation: bool = False,
        model_frame_window: int | None = None,
    ) -> None:
        self.run_root = run_root
        self.evidence = evidence
        self.bridge = bridge
        self.gates = GateTracker(level=level)
        self.autonomous_continuation = autonomous_continuation
        self.model_frame_window = model_frame_window
        self.failure_writer = FailurePacketWriter(run_root / "failure-packets")
        self.current_frame: dict[str, Any] | None = None
        self.current_claim: GroundingClaim | None = None
        self.pending_commitment: ProbeCommitment | None = None
        self.last_verdict: GroundingVerdict | None = None
        self._supported_predictions: list[_SupportedPrediction] = []
        self._credited_action_effects: set[tuple[Any, ...]] = set()
        self._action_issued = False

    def start(
        self,
        grid: Iterable[Iterable[int]],
        *,
        attempt_id: str,
        available_actions: Iterable[int],
    ) -> dict[str, Any]:
        if self.current_frame is not None:
            raise RuntimeError("kernel has already started")
        self.current_frame = self.evidence.record_frame(
            grid,
            level=self.gates.level,
            turn=0,
            attempt_id=attempt_id,
            available_actions=available_actions,
        )
        return self.current_frame

    def analyze(self) -> GroundingClaim:
        frame = self._require_frame()
        if self.pending_commitment is not None:
            raise RuntimeError("cannot analyze while an action is pending")
        parent = self.current_claim
        probe_results = [
            item
            for item in self.evidence.probe_results()
            if item["after_id"] == frame["evidence_id"]
        ]
        latest_probe_result = probe_results[-1] if probe_results else None
        try:
            artifact = self.bridge.request(
                phase="claim",
                observation_id=frame["evidence_id"],
                evidence=self.evidence.model_view(frame_window=self.model_frame_window),
                legal_actions=list(frame["available_actions"]),
                accepted_claim=parent.to_record() if parent else None,
            )
            claim = grounding_claim_from_draft(
                artifact,
                parent=parent,
                latest_probe_result=latest_probe_result,
            )
            if claim.observation_id != frame["evidence_id"]:
                raise ProtocolError("claim is stale: observation_id is not current")
            referenced = True
            for evidence_id in claim.evidence_refs:
                try:
                    self.evidence.get(evidence_id)
                except KeyError:
                    referenced = False
                    break
            heuristic_refs_exist = True
            for heuristic in claim.heuristic_register:
                for evidence_id in heuristic.evidence_refs:
                    try:
                        self.evidence.get(evidence_id)
                    except KeyError:
                        heuristic_refs_exist = False
                        break
                if not heuristic_refs_exist:
                    break
            if frame["evidence_id"] not in claim.evidence_refs:
                raise ProtocolError("claim must cite the current observation")
            self.gates.accept_claim(
                claim,
                observation_exists=True,
                evidence_refs_exist=referenced and heuristic_refs_exist,
            )
        except ArtifactGenerationError as error:
            self._pause_artifact_failure(frame, phase="claim", error=error, failure_class="model_generation")
        except ArtifactWorkerError as error:
            self._pause_artifact_failure(frame, phase="claim", error=error, failure_class="substrate")
        except (ProtocolError, ValueError) as error:
            self._pause_artifact_failure(frame, phase="claim", error=error, failure_class="model_semantic")
        self.current_claim = claim
        return claim

    def commit(self, *, coordinate_actions: Iterable[int] = ()) -> ProbeCommitment:
        frame = self._require_frame()
        claim = self.current_claim
        if claim is None or claim.observation_id != frame["evidence_id"]:
            raise RuntimeError("a current accepted claim is required before commitment")
        if self.pending_commitment is not None or self._action_issued:
            raise RuntimeError("one commitment/action is already pending")
        try:
            artifact = self.bridge.request(
                phase="commitment",
                observation_id=frame["evidence_id"],
                evidence=self.evidence.model_view(frame_window=self.model_frame_window),
                legal_actions=list(frame["available_actions"]),
                accepted_claim=claim.to_record(),
            )
            commitment = probe_commitment_from_draft(artifact, claim=claim)
            self._validate_action(commitment, frame, set(coordinate_actions))
        except ArtifactGenerationError as error:
            self._pause_artifact_failure(frame, phase="commitment", error=error, failure_class="model_generation")
        except ArtifactWorkerError as error:
            self._pause_artifact_failure(frame, phase="commitment", error=error, failure_class="substrate")
        except (ProtocolError, ValueError) as error:
            self._pause_artifact_failure(frame, phase="commitment", error=error, failure_class="model_semantic")
        self.pending_commitment = commitment
        return commitment

    def issue_one_action(self) -> ActionEnvelope:
        commitment = self.pending_commitment
        if commitment is None:
            raise RuntimeError("no accepted commitment is pending")
        if self._action_issued:
            raise RuntimeError("the pending commitment has already issued its one action")
        self._action_issued = True
        return ActionEnvelope(
            commitment_id=commitment.commitment_id,
            action=commitment.action,
            data=commitment.data,
        )

    def observe_successor(
        self,
        grid: Iterable[Iterable[int]],
        *,
        available_actions: Iterable[int],
    ) -> GroundingVerdict:
        before = self._require_frame()
        claim = self.current_claim
        commitment = self.pending_commitment
        if not self._action_issued or claim is None or commitment is None:
            raise RuntimeError("a successor requires one issued committed action")
        after = self.evidence.record_frame(
            grid,
            level=before["level"],
            turn=before["turn"] + 1,
            attempt_id=before["attempt_id"],
            available_actions=available_actions,
        )
        transition = self.evidence.record_transition(
            before["evidence_id"],
            after["evidence_id"],
            action=commitment.action,
            data=commitment.data,
            commitment_id=commitment.commitment_id,
        )
        verdict = verify_effect(claim, commitment, before["grid"], after["grid"])
        verdict = self._resolve_action_specificity(claim, commitment, verdict)
        change_accounting = account_visible_change(
            claim,
            before["grid"],
            after["grid"],
            transition_id=transition["evidence_id"],
        )
        probe_result = self.evidence.record_probe_result(
            before_id=before["evidence_id"],
            after_id=after["evidence_id"],
            grounding_claim=claim.to_record(),
            probe_commitment=commitment.to_record(),
            verifier_verdict=verdict.to_record(),
            change_accounting=change_accounting,
        )
        state_before = self.gates.to_record()
        gate = self.gates.apply_verdict(verdict)
        self.current_frame = after
        self.pending_commitment = None
        self._action_issued = False
        self.last_verdict = verdict
        if gate in {"INVALIDATED", "STOPPED"} and self.autonomous_continuation:
            state_at_stop = self.gates.to_record()
            self.gates.begin_new_grounding_cycle(reason=verdict.status)
            self.current_claim = None
            append_jsonl(
                self.run_root / "autonomous-cycle-resets.jsonl",
                {
                    "schema": "0158-autonomous-cycle-reset-v1",
                    "probe_result_id": probe_result["evidence_id"],
                    "reason": verdict.status,
                    "state_before": state_before,
                    "state_at_stop": state_at_stop,
                    "state_after": self.gates.to_record(),
                    "human_mediation": False,
                    "new_api_exposed": False,
                },
            )
        elif gate in {"INVALIDATED", "STOPPED"}:
            self._write_failure(before, after, claim, commitment, verdict, state_before)
        elif gate == "G2":
            stop_record = {
                "schema": "0158-g2-stop-v1",
                "gate": "G2",
                "transition_capability_eligible": True,
                "new_api_exposed": False,
                "review_required": not self.autonomous_continuation,
                "state": self.gates.to_record(),
            }
            if self.autonomous_continuation:
                append_jsonl(
                    self.run_root / "autonomous-g2-cycles.jsonl",
                    {**stop_record, "human_mediation": False},
                )
                self.gates.begin_new_grounding_cycle(reason="g2_recorded_without_unlock")
                self.current_claim = None
            else:
                write_atomic_json(
                    self.run_root / "review-stops" / "g2-transition-eligibility.json",
                    stop_record,
                )
        return verdict

    def record_separate_event(
        self,
        event_kind: str,
        grids: Iterable[Iterable[Iterable[int]]],
    ) -> dict[str, Any]:
        frame = self._require_frame()
        commitment = self.pending_commitment
        return self.evidence.record_event(
            event_kind,
            level=frame["level"],
            attempt_id=frame["attempt_id"],
            before_id=frame["evidence_id"],
            grids=grids,
            action=commitment.action if commitment else None,
            data=commitment.data if commitment else None,
        )

    def observe_terminal_successor(self, grid: Iterable[Iterable[int]]) -> dict[str, Any]:
        """Close the winning action without inserting a terminal into ordinary transitions."""
        frame = self._require_frame()
        commitment = self.pending_commitment
        if not self._action_issued or commitment is None:
            raise RuntimeError("terminal successor requires one issued committed action")
        event = self.evidence.record_event(
            "terminal",
            level=frame["level"],
            attempt_id=frame["attempt_id"],
            before_id=frame["evidence_id"],
            grids=[grid],
            action=commitment.action,
            data=commitment.data,
        )
        self.pending_commitment = None
        self._action_issued = False
        write_atomic_json(
            self.run_root / "terminal-action.json",
            {
                "schema": "0158-terminal-action-v1",
                "event_id": event["evidence_id"],
                "commitment_id": commitment.commitment_id,
                "ordinary_transition_recorded": False,
            },
        )
        return event

    def observe_death_successor(self, grid: Iterable[Iterable[int]]) -> None:
        """Record death separately, close the action, and stop for review."""
        frame = self._require_frame()
        commitment = self.pending_commitment
        claim = self.current_claim
        if not self._action_issued or commitment is None or claim is None:
            raise RuntimeError("death successor requires one issued committed action")
        event = self.evidence.record_event(
            "death",
            level=frame["level"],
            attempt_id=frame["attempt_id"],
            before_id=frame["evidence_id"],
            grids=[grid],
            action=commitment.action,
            data=commitment.data,
        )
        after = {**event, "grid": event["grids"][-1]}
        state_before = self.gates.to_record()
        self.gates.stop_for_failure(failure_class="observed_death")
        self.pending_commitment = None
        self._action_issued = False
        packet = self.failure_writer.create(
            before=frame,
            after=after,
            grounding_claim=claim.to_record(),
            probe_commitment=commitment.to_record(),
            verifier_verdict=None,
            prompt={"model_calls": (self.bridge.latest_response or {}).get("audit", {}).get("model_calls", [])},
            callable_surface=_callable_surface("commitment"),
            state_before=state_before,
            state_after=self.gates.to_record(),
            trace=read_jsonl(self.bridge.trace_path),
            promotion_history=read_jsonl(self.run_root / "promotion-history.jsonl"),
            causal_record={
                "observation": "the committed action produced an observed death event",
                "proximate_failure": "observed_death",
                "live_causal_hypotheses": ["the action-conditioned candidate or action choice was unsafe"],
                "discriminating_probe": "review the immutable pre-action and death channels before another run",
                "conclusion": "observed_death",
                "system_decision": "pause_and_close_action_authority",
                "failure_class": "model_semantic",
            },
        )
        raise QualificationPaused(packet, "model_semantic")

    def pause_external_failure(self, error: BaseException, *, phase: str) -> None:
        """Close a live run when the environment boundary fails or becomes uncertain."""
        frame = self._require_frame()
        claim = self.current_claim
        commitment = self.pending_commitment
        state_before = self.gates.to_record()
        self.gates.stop_for_failure(failure_class="substrate")
        self.pending_commitment = None
        self._action_issued = False
        packet = self.failure_writer.create(
            before=frame,
            after=frame,
            grounding_claim=claim.to_record() if claim else None,
            probe_commitment=commitment.to_record() if commitment else None,
            verifier_verdict=None,
            prompt={"model_calls": (self.bridge.latest_response or {}).get("audit", {}).get("model_calls", [])},
            callable_surface={
                **_callable_surface("commitment"),
                "external_failure_phase": phase,
            },
            state_before=state_before,
            state_after=self.gates.to_record(),
            trace=read_jsonl(self.bridge.trace_path),
            promotion_history=read_jsonl(self.run_root / "promotion-history.jsonl"),
            causal_record={
                "observation": f"the external {phase} boundary failed or had uncertain completion",
                "proximate_failure": type(error).__name__,
                "live_causal_hypotheses": ["ARC transport or environment substrate failed"],
                "discriminating_probe": "reproduce the boundary without reopening this scored run",
                "conclusion": "substrate_failure",
                "system_decision": "pause_and_close_action_authority",
                "failure_class": "substrate",
            },
        )
        raise QualificationPaused(packet, "substrate") from error

    def begin_new_level(
        self,
        grid: Iterable[Iterable[int]],
        *,
        level: int,
        attempt_id: str,
        available_actions: Iterable[int],
    ) -> dict[str, Any]:
        if self.pending_commitment is not None or self._action_issued:
            raise RuntimeError("level boundary cannot retain a pending action")
        self.gates.reset_for_level(level)
        self.current_claim = None
        self.current_frame = self.evidence.record_frame(
            grid,
            level=level,
            turn=0,
            attempt_id=attempt_id,
            available_actions=available_actions,
        )
        return self.current_frame

    def _write_failure(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        claim: GroundingClaim,
        commitment: ProbeCommitment,
        verdict: GroundingVerdict,
        state_before: dict[str, Any],
    ) -> Path:
        latest = self.bridge.latest_response or {}
        checks = ",".join(verdict.checks)
        return self.failure_writer.create(
            before=before,
            after=after,
            grounding_claim=claim.to_record(),
            probe_commitment=commitment.to_record(),
            verifier_verdict=verdict.to_record(),
            prompt={"model_calls": latest.get("audit", {}).get("model_calls", [])},
            callable_surface=_callable_surface("commitment"),
            state_before=state_before,
            state_after=self.gates.to_record(),
            trace=read_jsonl(self.bridge.trace_path),
            promotion_history=read_jsonl(self.run_root / "promotion-history.jsonl"),
            causal_record={
                "observation": f"verifier status={verdict.status}; checks={checks}",
                "proximate_failure": verdict.status,
                "live_causal_hypotheses": [
                    "the selected region does not preserve the proposed identity",
                    "the prospective effect is wrong",
                    "the public successor is insufficient to discriminate the candidates",
                ],
                "discriminating_probe": "review one alternative candidate only if the gate permits it",
                "conclusion": "undetermined_pending_review",
                "system_decision": "pause_and_close_action_authority",
                "failure_class": "model_semantic" if verdict.status != "inconclusive" else "evidence_inconclusive",
            },
        )

    def _pause_artifact_failure(
        self,
        frame: dict[str, Any],
        *,
        phase: str,
        error: BaseException,
        failure_class: str,
    ) -> None:
        state_before = self.gates.to_record()
        self.gates.stop_for_failure(failure_class=failure_class)
        latest = self.bridge.latest_response or {}
        packet = self.failure_writer.create(
            before=frame,
            after=frame,
            grounding_claim=self.current_claim.to_record() if self.current_claim else None,
            probe_commitment=None,
            verifier_verdict=None,
            prompt={"model_calls": latest.get("audit", {}).get("model_calls", [])},
            callable_surface=_callable_surface(phase),
            state_before=state_before,
            state_after=self.gates.to_record(),
            trace=read_jsonl(self.bridge.trace_path),
            promotion_history=read_jsonl(self.run_root / "promotion-history.jsonl"),
            causal_record={
                "observation": f"{phase} artifact failed validation or transport",
                "proximate_failure": f"{type(error).__name__}: {error}",
                "live_causal_hypotheses": (
                    ["worker transport or isolation substrate failed"]
                    if failure_class == "substrate"
                    else (
                        ["model exhausted its output budget or omitted the required execution tool call"]
                        if failure_class == "model_generation"
                        else ["model artifact violated the frozen schema or temporal binding"]
                    )
                ),
                "discriminating_probe": (
                    "inspect provider completion status and output item types before any new run"
                    if failure_class == "model_generation"
                    else "reproduce with the deterministic offline fixture before any new run"
                ),
                "conclusion": "undetermined_pending_review",
                "system_decision": "pause_and_close_action_authority",
                "failure_class": failure_class,
            },
        )
        raise QualificationPaused(packet, failure_class) from error

    @staticmethod
    def _validate_action(commitment: ProbeCommitment, frame: dict[str, Any], coordinate_actions: set[int]) -> None:
        GroundingKernel._validate_action_prediction(
            commitment.action,
            commitment.data,
            frame,
            coordinate_actions,
        )
        for prediction in commitment.alternative_predictions:
            GroundingKernel._validate_action_prediction(
                prediction.action,
                prediction.data,
                frame,
                coordinate_actions,
            )

    @staticmethod
    def _validate_action_prediction(
        action: int,
        data: dict[str, Any] | None,
        frame: dict[str, Any],
        coordinate_actions: set[int],
    ) -> None:
        if action not in frame["available_actions"]:
            raise ValueError("commitment action is not currently legal")
        if action in coordinate_actions:
            if data is None or set(data) != {"x", "y"} or any(type(data[axis]) is not int for axis in ("x", "y")):
                raise ValueError("coordinate action requires exact integer x and y data")
            height = len(frame["grid"])
            width = len(frame["grid"][0])
            if not (0 <= data["x"] < width and 0 <= data["y"] < height):
                raise ValueError("coordinate action lies outside the public frame")
        elif data is not None:
            raise ValueError("non-coordinate action data must be null")

    def _resolve_action_specificity(
        self,
        claim: GroundingClaim,
        commitment: ProbeCommitment,
        verdict: GroundingVerdict,
    ) -> GroundingVerdict:
        """Close only reciprocal, prospectively recorded action-effect contrasts."""
        if verdict.status != "supported":
            return verdict

        selected_signature = predicted_effect_signature(
            commitment.predicted_effect,
            primary_region=claim.primary_region,
        )
        current_alternatives = {
            prediction.action: predicted_effect_signature(
                prediction.predicted_effect,
                primary_region=claim.primary_region,
            )
            for prediction in commitment.alternative_predictions
        }
        matches: list[_SupportedPrediction] = []
        for prior in self._supported_predictions:
            if prior.claim.lineage_id != claim.lineage_id:
                continue
            prior_commitment = prior.commitment
            if prior_commitment.action == commitment.action:
                continue
            prior_signature = predicted_effect_signature(
                prior_commitment.predicted_effect,
                primary_region=prior.claim.primary_region,
            )
            if prior_signature == selected_signature:
                continue
            prior_alternatives = {
                prediction.action: predicted_effect_signature(
                    prediction.predicted_effect,
                    primary_region=prior.claim.primary_region,
                )
                for prediction in prior_commitment.alternative_predictions
            }
            if prior_alternatives.get(commitment.action) != selected_signature:
                continue
            if current_alternatives.get(prior_commitment.action) != prior_signature:
                continue
            matches.append(prior)

        current = _SupportedPrediction(claim=claim, commitment=commitment)
        self._supported_predictions.append(current)
        if not matches:
            return replace(
                verdict,
                checks=(*verdict.checks, "ACTION_CONTRAST_PENDING"),
                action_specificity="inconclusive",
                contrast_commitment_ids=(),
                promotion_credit=0,
                promotable=False,
            )

        credit_candidates = [current, *matches]
        new_credit_keys: set[tuple[Any, ...]] = set()
        for candidate in credit_candidates:
            if isinstance(candidate.commitment.predicted_effect, Unchanged):
                continue
            signature = predicted_effect_signature(
                candidate.commitment.predicted_effect,
                primary_region=candidate.claim.primary_region,
            )
            key = (
                candidate.claim.lineage_id,
                candidate.commitment.action,
                signature,
            )
            if key not in self._credited_action_effects:
                new_credit_keys.add(key)
        self._credited_action_effects.update(new_credit_keys)
        promotion_credit = len(new_credit_keys)
        return replace(
            verdict,
            checks=(*verdict.checks, "ACTION_CONTRAST_PROSPECTIVELY_CLOSED"),
            action_specificity="supported",
            contrast_commitment_ids=tuple(
                sorted(item.commitment.commitment_id for item in matches)
            ),
            promotion_credit=promotion_credit,
            promotable=promotion_credit > 0,
        )

    def _require_frame(self) -> dict[str, Any]:
        if self.current_frame is None:
            raise RuntimeError("kernel has no current frame")
        return self.current_frame
