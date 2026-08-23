"""Eligibility gates that precede any Native-versus-CodeAct reasoning claim."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tycho_nooa.cognitive_parity import RetrievalAnswer, tree_sha256
from tycho_nooa.evidence import EvidenceQuery, EvidenceRef, WorkspaceEvidenceCapability

CompatibilityGateKind = Literal[
    "termination",
    "metadata_query",
    "representation_rle",
    "representation_raw",
    "easy_positive",
]


class CompatibilityTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_id: str = Field(min_length=1)
    gate_kind: CompatibilityGateKind
    task_kind: Literal["retrieve"] = "retrieve"
    instruction: str = Field(min_length=1)
    inputs: dict[str, str | int | bool] = Field(default_factory=dict)


class CompatibilityEvidencePolicy(BaseModel):
    """The exact query shape available in one eligibility gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    kind: Literal["frames", "transitions"] | None = None
    level: int | None = Field(default=None, ge=0)
    limit: int | None = Field(default=None, ge=1, le=32)
    projection: Literal["full", "metadata"] | None = None
    grid_encoding: Literal["raw", "rle"] | None = None
    max_queries: int = Field(default=1, ge=0, le=3)

    @model_validator(mode="after")
    def enabled_policy_is_complete(self) -> CompatibilityEvidencePolicy:
        if self.enabled:
            _validate_enabled_policy(self)
        else:
            _validate_disabled_policy(self)
        return self

    def query_value(self) -> dict[str, str | int]:
        if not self.enabled:
            raise RuntimeError("evidence is disabled for this compatibility gate")
        assert self.kind is not None
        assert self.level is not None
        assert self.limit is not None
        assert self.projection is not None
        assert self.grid_encoding is not None
        return {
            "kind": self.kind,
            "level": self.level,
            "limit": self.limit,
            "projection": self.projection,
            "grid_encoding": self.grid_encoding,
        }


class CompatibilityExpected(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conclusion: str
    evidence_refs: tuple[EvidenceRef, ...] = ()


class CompatibilityGate(BaseModel):
    """One treatment-visible eligibility task plus its hidden exact key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    checkpoint_root: str | None = None
    checkpoint_sha256: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    task: CompatibilityTask
    evidence: CompatibilityEvidencePolicy
    expected: CompatibilityExpected

    @model_validator(mode="after")
    def checkpoint_matches_authority(self) -> CompatibilityGate:
        checkpoint_values = (self.checkpoint_root, self.checkpoint_sha256)
        if self.evidence.enabled and any(value is None for value in checkpoint_values):
            raise ValueError("evidence-enabled gate requires a frozen checkpoint")
        if not self.evidence.enabled and any(
            value is not None for value in checkpoint_values
        ):
            raise ValueError("return-only gate cannot carry an evidence checkpoint")
        if (self.task.gate_kind == "termination") == self.evidence.enabled:
            raise ValueError("only the termination gate may disable evidence")
        return self

    def treatment_payload(self) -> dict[str, Any]:
        return {
            "task": self.task.model_dump(mode="json"),
            "evidence_contract": self.evidence.model_dump(mode="json"),
        }


class CompatibilityEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_id: str
    valid_typed_return: Literal[True] = True
    conclusion_correct: bool
    evidence_refs_correct: bool
    passed: bool


class CompatibilityEvidenceCapability:
    """Permit only the single predeclared query shape for one arm and gate."""

    def __init__(
        self,
        capability: WorkspaceEvidenceCapability,
        policy: CompatibilityEvidencePolicy,
    ):
        if not policy.enabled:
            raise ValueError("disabled evidence policy has no capability adapter")
        self._capability = capability
        self._policy = policy
        self._queries = 0

    @classmethod
    def from_gate(
        cls, gate: CompatibilityGate, *, repository_root: Path
    ) -> CompatibilityEvidenceCapability | None:
        if not gate.evidence.enabled:
            return None
        assert gate.checkpoint_root is not None
        return cls(
            WorkspaceEvidenceCapability.from_root(repository_root / gate.checkpoint_root),
            gate.evidence,
        )

    @property
    def queries(self) -> int:
        return self._queries

    def query(self, raw_query: dict[str, Any] | EvidenceQuery) -> dict[str, Any]:
        self._queries += 1
        if self._queries > self._policy.max_queries:
            raise ValueError("compatibility gate evidence-query cap exceeded")
        query = (
            raw_query.model_dump(exclude_none=True)
            if isinstance(raw_query, EvidenceQuery)
            else raw_query
        )
        expected = self._policy.query_value()
        if query != expected:
            raise ValueError(
                f"compatibility gate requires exact evidence query {expected}"
            )
        return self._capability.query(query)


def load_compatibility_gate(
    path: Path, *, repository_root: Path
) -> CompatibilityGate:
    gate = CompatibilityGate.model_validate_json(path.read_text())
    if not gate.evidence.enabled:
        return gate
    assert gate.checkpoint_root is not None
    assert gate.checkpoint_sha256 is not None
    checkpoint = _checkpoint_path(gate.checkpoint_root, repository_root)
    actual = f"sha256:{tree_sha256(checkpoint)}"
    if actual != gate.checkpoint_sha256:
        raise ValueError("compatibility checkpoint commitment mismatch")
    return gate


def evaluate_compatibility_answer(
    gate: CompatibilityGate, answer: RetrievalAnswer
) -> CompatibilityEvaluation:
    conclusion_correct = answer.conclusion == gate.expected.conclusion
    refs = {(ref.level, ref.turn) for ref in answer.evidence_refs}
    expected_refs = {(ref.level, ref.turn) for ref in gate.expected.evidence_refs}
    evidence_refs_correct = refs == expected_refs
    return CompatibilityEvaluation(
        gate_id=gate.task.gate_id,
        conclusion_correct=conclusion_correct,
        evidence_refs_correct=evidence_refs_correct,
        passed=conclusion_correct and evidence_refs_correct,
    )


def _checkpoint_path(value: str, repository_root: Path) -> Path:
    root = repository_root.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("checkpoint_root escapes repository root")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _validate_enabled_policy(policy: CompatibilityEvidencePolicy) -> None:
    configured = (
        policy.kind,
        policy.level,
        policy.limit,
        policy.projection,
        policy.grid_encoding,
    )
    if any(value is None for value in configured):
        raise ValueError("enabled evidence policy must fix the complete query shape")
    if policy.max_queries < 1:
        raise ValueError("enabled evidence policy requires a positive query cap")


def _validate_disabled_policy(policy: CompatibilityEvidencePolicy) -> None:
    configured = (
        policy.kind,
        policy.level,
        policy.limit,
        policy.projection,
        policy.grid_encoding,
    )
    if any(value is not None for value in configured):
        raise ValueError("disabled evidence policy cannot configure a query")
    if policy.max_queries != 0:
        raise ValueError("disabled evidence policy must allow zero queries")
