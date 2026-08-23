"""Representation-neutral frozen tasks for native and CodeAct comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from tycho_nooa.evidence import EvidenceRef
from tycho_nooa.models import ActionDecision

TaskKind = Literal["retrieve", "discriminate", "recover_invariant", "propose_action"]
Scope = Literal["transition", "level", "cross_level", "trajectory"]


class CognitiveTask(BaseModel):
    """The treatment-visible part of one frozen benchmark packet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    evidence_frontier: EvidenceRef
    task_kind: TaskKind
    instruction: str = Field(min_length=1)
    inputs: dict[str, str | int | float | bool | list[str]] = Field(
        default_factory=dict
    )


class ExpectedAnswer(BaseModel):
    """Evaluator-only facts; runners must never include these in model context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    required_evidence_refs: tuple[EvidenceRef, ...] = ()
    accepted_verdicts: tuple[str, ...] = ()
    accepted_invariant_ids: tuple[str, ...] = ()
    accepted_scopes: tuple[Scope, ...] = ()
    native_action: ActionDecision | None = None


class CognitiveParityPacket(BaseModel):
    """One immutable checkpoint task plus its separately held answer key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    checkpoint_root: str
    task: CognitiveTask
    expected: ExpectedAnswer

    @model_validator(mode="after")
    def expected_fields_match_task_kind(self) -> CognitiveParityPacket:
        _require_discrimination_key(self.task.task_kind, self.expected)
        _require_invariant_key(self.task.task_kind, self.expected)
        _confine_native_action(self.task.task_kind, self.expected)
        return self

    def treatment_payload(self) -> dict[str, object]:
        """Return exactly what either reasoning treatment may see."""
        return self.task.model_dump(mode="json")


class RetrievalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_kind: Literal["retrieve"] = "retrieve"
    evidence_refs: tuple[EvidenceRef, ...]
    conclusion: str


class HypothesisAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_kind: Literal["discriminate"] = "discriminate"
    supported: str
    falsified: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    conclusion: str


class InvariantAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_kind: Literal["recover_invariant"] = "recover_invariant"
    invariant_id: str
    invariant: str
    evidence_refs: tuple[EvidenceRef, ...]
    scope: Scope


class ActionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_kind: Literal["propose_action"] = "propose_action"
    decision: ActionDecision
    evidence_refs: tuple[EvidenceRef, ...]
    expected_effect: str
    information_goal: str | None = None


CognitiveAnswer = Annotated[
    RetrievalAnswer | HypothesisAnswer | InvariantAnswer | ActionProposal,
    Field(discriminator="task_kind"),
]
COGNITIVE_ANSWER_ADAPTER = TypeAdapter(CognitiveAnswer)


def _require_discrimination_key(kind: TaskKind, expected: ExpectedAnswer) -> None:
    if kind == "discriminate" and not expected.accepted_verdicts:
        raise ValueError("discrimination packets require accepted_verdicts")


def _require_invariant_key(kind: TaskKind, expected: ExpectedAnswer) -> None:
    if kind == "recover_invariant" and not expected.accepted_invariant_ids:
        raise ValueError("invariant packets require accepted_invariant_ids")


def _confine_native_action(kind: TaskKind, expected: ExpectedAnswer) -> None:
    if kind != "propose_action" and expected.native_action is not None:
        raise ValueError("native_action is only meaningful for action parity")


class ParityEvaluation(BaseModel):
    """Factored scores; action disagreement is never mislabeled inaccuracy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str
    task_kind: TaskKind
    evidence_precision: float = Field(ge=0, le=1)
    evidence_recall: float = Field(ge=0, le=1)
    verdict_correct: bool | None = None
    invariant_correct: bool | None = None
    scope_correct: bool | None = None
    action_legal: bool | None = None
    native_action_agreement: bool | None = None
    decision_quality_scored: Literal[False] = False


def load_packet(path: Path, *, repository_root: Path) -> CognitiveParityPacket:
    """Load a packet only while its frozen checkpoint still matches its hash."""
    packet = CognitiveParityPacket.model_validate_json(path.read_text())
    checkpoint = _checkpoint_path(packet, repository_root)
    actual = f"sha256:{tree_sha256(checkpoint)}"
    if actual != packet.task.checkpoint_sha256:
        raise ValueError(
            f"checkpoint commitment mismatch: expected "
            f"{packet.task.checkpoint_sha256}, got {actual}"
        )
    return packet


def evaluate_answer(
    packet: CognitiveParityPacket, raw_answer: object
) -> ParityEvaluation:
    answer = COGNITIVE_ANSWER_ADAPTER.validate_python(raw_answer)
    if answer.task_kind != packet.task.task_kind:
        raise ValueError("answer task_kind does not match its packet")
    precision, recall = _evidence_scores(
        answer.evidence_refs, packet.expected.required_evidence_refs
    )
    facts: dict[str, object] = {
        "packet_id": packet.task.packet_id,
        "task_kind": packet.task.task_kind,
        "evidence_precision": precision,
        "evidence_recall": recall,
    }
    if isinstance(answer, HypothesisAnswer):
        facts["verdict_correct"] = answer.supported in packet.expected.accepted_verdicts
    if isinstance(answer, InvariantAnswer):
        facts["invariant_correct"] = (
            answer.invariant_id in packet.expected.accepted_invariant_ids
        )
        facts["scope_correct"] = answer.scope in packet.expected.accepted_scopes
    if isinstance(answer, ActionProposal):
        facts["action_legal"] = True
        native = packet.expected.native_action
        facts["native_action_agreement"] = (
            None if native is None else answer.decision == native
        )
    return ParityEvaluation.model_validate(facts)


def tree_sha256(root: Path) -> str:
    """Commit relative paths and bytes, excluding no files or metadata."""
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _checkpoint_path(
    packet: CognitiveParityPacket, repository_root: Path
) -> Path:
    root = repository_root.resolve()
    candidate = (root / packet.checkpoint_root).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("checkpoint_root escapes repository_root")
    return candidate


def _evidence_scores(
    proposed: tuple[EvidenceRef, ...], required: tuple[EvidenceRef, ...]
) -> tuple[float, float]:
    proposed_set = {(ref.level, ref.turn) for ref in proposed}
    required_set = {(ref.level, ref.turn) for ref in required}
    correct = len(proposed_set & required_set)
    precision = correct / len(proposed_set) if proposed_set else float(not required_set)
    recall = correct / len(required_set) if required_set else 1.0
    return precision, recall


def write_packet(path: Path, packet: CognitiveParityPacket) -> None:
    """Write one reviewable packet without blending task and answer files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    value = packet.model_dump(mode="json")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
