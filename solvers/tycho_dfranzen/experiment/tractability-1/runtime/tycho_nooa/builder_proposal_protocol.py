"""Authority-free Builder proposal values and host-owned evaluation ledger.

The Builder may construct these frozen values, but only the host-owned
``ProposalLedger`` can validate lineage and invoke the committed evaluator.
Neither operation has a reference to canonical theory or transaction state.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import unicodedata
import uuid
from base64 import b64encode
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Protocol

from tycho_nooa.theory_evaluation import (
    EvaluationRunStatus,
    EvidenceFile,
    EvidenceFrontier,
    ObligationKind,
    ObligationResult,
    ObligationStatus,
    TheoryEvaluation,
)

_DOMAIN = b"tycho-builder-proposal-v1"
_SHA_PREFIX = "sha256:"
_MAX_SOURCE_BYTES = 2_000_000
_MAX_PROMPT_PREVIEW_BYTES = 16_000

type JsonScalar = None | bool | int | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type FrozenJson = JsonScalar | tuple[FrozenJson, ...] | _FrozenJsonObject


class ProposalValidationError(ValueError):
    """A Builder message failed a host-enforced protocol invariant."""


@dataclass(frozen=True)
class _FrozenJsonObject(Mapping[str, object]):
    """Unambiguous immutable representation of a JSON object."""

    entries: tuple[tuple[str, object], ...]

    def __getitem__(self, key: str) -> object:
        return dict(self.entries)[key]

    def __iter__(self):
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


class ProposalEvaluator(Protocol):
    evaluator_sha256: str

    def evaluate(
        self, model: bytes, frontier: EvidenceFrontier
    ) -> TheoryEvaluation: ...


class BuilderEvaluationCapability:
    """Validate one data-only realm request before it reaches the host ledger."""

    def __init__(self, ledger: ProposalLedger) -> None:
        self._ledger = ledger

    def __call__(self, arguments: dict[str, JsonValue]) -> JsonValue:
        expected = {
            "candidate_source",
            "declared_parent_source_sha256",
            "parent_source",
            "provenance",
        }
        if set(arguments) != expected:
            raise ProposalValidationError(
                "builder.evaluate requires the exact candidate submission fields"
            )
        candidate = arguments["candidate_source"]
        parent = arguments["parent_source"]
        parent_sha = arguments["declared_parent_source_sha256"]
        provenance = arguments["provenance"]
        if not isinstance(candidate, str) or not isinstance(parent, str):
            raise ProposalValidationError("candidate and parent source must be text")
        if not isinstance(parent_sha, str) or not isinstance(provenance, dict):
            raise ProposalValidationError("submission lineage or provenance is malformed")
        submission = CandidateSubmission.replace(
            case=self._ledger.case,
            parent_source=parent.encode("utf-8"),
            candidate_source=candidate.encode("utf-8"),
            declared_parent_source_sha256=parent_sha,
            provenance=provenance,
        )
        return self._ledger.evaluate(submission).builder_view()


def canonical_json(value: object) -> bytes:
    """Serialize the protocol's narrow, unambiguous JSON profile."""
    normalized = _validated_json(value, path="$", freeze=False)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def object_sha256(type_name: str, payload: object) -> str:
    """Hash one canonical value with protocol and type domain separation."""
    if not type_name.isascii() or not type_name:
        raise ValueError("protocol type name must be non-empty ASCII")
    body = b"\0".join((_DOMAIN, type_name.encode("ascii"), canonical_json(payload)))
    return _SHA_PREFIX + hashlib.sha256(body).hexdigest()


def canonicalize_source(source: bytes) -> bytes:
    """Remove transport variance without rewriting Python source semantics."""
    if source.startswith(b"\xef\xbb\xbf"):
        raise ValueError("candidate source must not contain a byte-order mark")
    if b"\0" in source:
        raise ValueError("candidate source must not contain NUL bytes")
    if len(source) > _MAX_SOURCE_BYTES:
        raise ValueError(f"candidate source exceeds {_MAX_SOURCE_BYTES} bytes")
    try:
        source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("candidate source must be valid UTF-8") from error
    canonical = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return canonical if canonical.endswith(b"\n") else canonical + b"\n"


@dataclass(frozen=True)
class BlobReference:
    media_type: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_bytes(cls, content: bytes, *, media_type: str) -> BlobReference:
        return cls(
            media_type=media_type,
            size_bytes=len(content),
            sha256=_digest(content),
        )

    def verify(self, content: bytes) -> None:
        if len(content) != self.size_bytes:
            raise ProposalValidationError("blob size does not match its reference")
        if _digest(content) != self.sha256:
            raise ProposalValidationError("blob digest does not match its reference")

    def payload(self) -> dict[str, JsonValue]:
        return {
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class EvidenceSnapshot:
    identifier: str
    path: str
    provenance: str
    blob: BlobReference
    content: bytes

    @classmethod
    def from_frontier_file(cls, path: str, content: bytes) -> EvidenceSnapshot:
        _validate_manifest_path(path)
        detached = bytes(content)
        return cls(
            identifier=f"evidence:{path}",
            path=path,
            provenance="host_observation",
            blob=BlobReference.from_bytes(detached, media_type="application/octet-stream"),
            content=detached,
        )

    def payload(self) -> dict[str, JsonValue]:
        return {
            "blob": self.blob.payload(),
            "identifier": self.identifier,
            "path": self.path,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class ObligationDefinition:
    obligation_id: str
    kind: ObligationKind
    description: str
    protected: bool = True

    @classmethod
    def from_result(cls, result: ObligationResult) -> ObligationDefinition:
        return cls(result.obligation_id, result.kind, result.description)

    def payload(self) -> dict[str, JsonValue]:
        return {
            "description": self.description,
            "kind": self.kind.value,
            "obligation_id": self.obligation_id,
            "protected": self.protected,
        }


@dataclass(frozen=True)
class FrozenBuildCase:
    """Realm-owned immutable snapshot of every Builder-visible semantic input."""

    evidence: tuple[EvidenceSnapshot, ...]
    incumbent_source: bytes
    incumbent_blob: BlobReference
    incumbent_evaluation: TheoryEvaluation
    obligations: tuple[ObligationDefinition, ...]
    obligation_universe_sha256: str
    evidence_frontier_sha256: str
    evaluator_sha256: str
    model_abi: str
    trusted_dependencies: tuple[str, ...]
    metadata: FrozenJson
    case_sha256: str = ""
    schema: str = "tycho.frozen-build-case.v1"

    @classmethod
    def freeze(
        cls,
        *,
        frontier: EvidenceFrontier,
        incumbent_source: bytes,
        incumbent_evaluation: TheoryEvaluation,
        evaluator_sha256: str,
        metadata: Mapping[str, object] | None = None,
        model_abi: str = "tycho.world-model.v1",
        trusted_dependencies: Sequence[str] = (),
    ) -> FrozenBuildCase:
        source = canonicalize_source(bytes(incumbent_source))
        source_blob = BlobReference.from_bytes(source, media_type="text/x-python")
        expected_model_sha = _bare_digest(source_blob.sha256)
        if incumbent_evaluation.model_sha256 != expected_model_sha:
            raise ProposalValidationError(
                "incumbent evaluation does not commit to canonical incumbent source"
            )
        if incumbent_evaluation.evidence_frontier_sha256 != frontier.sha256:
            raise ProposalValidationError(
                "incumbent evaluation does not commit to the frozen frontier"
            )
        if incumbent_evaluation.evaluator_sha256 != evaluator_sha256:
            raise ProposalValidationError(
                "incumbent evaluation does not commit to the requested evaluator"
            )
        detached_evaluation = _detached_evaluation(incumbent_evaluation)
        evidence = tuple(
            EvidenceSnapshot.from_frontier_file(item.path, item.content)
            for item in frontier.files
        )
        obligations = tuple(
            ObligationDefinition.from_result(item)
            for item in detached_evaluation.obligations
        )
        _validate_unique_obligations(obligations)
        obligation_sha = object_sha256(
            "ObligationUniverse",
            [item.payload() for item in obligations],
        )
        frozen_metadata = _validated_json(
            dict(metadata or {}), path="$.metadata", freeze=True
        )
        dependencies = tuple(trusted_dependencies)
        preliminary = cls(
            evidence=evidence,
            incumbent_source=source,
            incumbent_blob=source_blob,
            incumbent_evaluation=detached_evaluation,
            obligations=obligations,
            obligation_universe_sha256=obligation_sha,
            evidence_frontier_sha256=frontier.sha256,
            evaluator_sha256=evaluator_sha256,
            model_abi=model_abi,
            trusted_dependencies=dependencies,
            metadata=frozen_metadata,
        )
        return cls(**{**preliminary.__dict__, "case_sha256": object_sha256(
            "FrozenBuildCase", preliminary.payload()
        )})

    @property
    def incumbent_source_sha256(self) -> str:
        return self.incumbent_blob.sha256

    def payload(self) -> dict[str, JsonValue]:
        return {
            "evidence": [item.payload() for item in self.evidence],
            "evidence_frontier_sha256": self.evidence_frontier_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "incumbent": {
                "evaluation": _evaluation_payload(self.incumbent_evaluation),
                "model_abi": self.model_abi,
                "source": self.incumbent_blob.payload(),
                "trusted_dependencies": list(self.trusted_dependencies),
            },
            "metadata": _thaw_json(self.metadata),
            "obligation_universe_sha256": self.obligation_universe_sha256,
            "obligations": [item.payload() for item in self.obligations],
            "schema": self.schema,
        }

    def builder_view(self) -> dict[str, JsonValue]:
        """Return the lossless JSON-only value copied into a sterile realm."""
        evidence: list[JsonValue] = []
        for item in self.evidence:
            try:
                content_utf8: str | None = item.content.decode("utf-8")
            except UnicodeDecodeError:
                content_utf8 = None
            evidence.append(
                {
                    **item.payload(),
                    "content_base64": b64encode(item.content).decode("ascii"),
                    "content_utf8": content_utf8,
                }
            )
        return {
            "case_sha256": self.case_sha256,
            "evidence": evidence,
            "incumbent": {
                "source": self.incumbent_source.decode("utf-8"),
                "source_sha256": self.incumbent_source_sha256,
            },
            "metadata": _thaw_json(self.metadata),
            "obligation_universe_sha256": self.obligation_universe_sha256,
            "obligations": [
                {
                    **definition.payload(),
                    "incumbent_result": _obligation_result_payload(result),
                }
                for definition, result in zip(
                    self.obligations,
                    self.incumbent_evaluation.obligations,
                    strict=True,
                )
            ],
            "schema": self.schema,
        }

    def prompt_preview(self) -> dict[str, JsonValue]:
        """Bounded common provider context; lossless bytes remain inside Python."""
        source = self.incumbent_source.decode("utf-8")
        preview: dict[str, JsonValue] = {
            "case_sha256": self.case_sha256,
            "evidence_count": len(self.evidence),
            "evidence_frontier_sha256": self.evidence_frontier_sha256,
            "evidence_preview": [
                item.payload() for item in self.evidence[:24]
            ],
            "incumbent": {
                "source_preview": source[:6_000],
                "source_sha256": self.incumbent_source_sha256,
                "source_size_bytes": len(self.incumbent_source),
                "source_truncated": len(source) > 6_000,
            },
            "obligation_count": len(self.obligations),
            "obligation_universe_sha256": self.obligation_universe_sha256,
            "obligations_preview": [
                {
                    **definition.payload(),
                    "description": definition.description[:160],
                    "incumbent_status": result.status.value,
                }
                for definition, result in zip(
                    self.obligations[:48],
                    self.incumbent_evaluation.obligations[:48],
                    strict=True,
                )
            ],
            "preview_contract": (
                "Use Python case for complete evidence and incumbent bytes; this "
                "provider preview is intentionally bounded."
            ),
            "schema": self.schema,
        }
        if len(canonical_json(preview)) > _MAX_PROMPT_PREVIEW_BYTES:
            raise ProposalValidationError("bounded Builder prompt preview exceeds its cap")
        return preview


@dataclass(frozen=True)
class ByteEdit:
    start_byte: int
    end_byte: int
    before_slice_sha256: str
    replacement_blob: BlobReference
    replacement: bytes

    def payload(self) -> dict[str, JsonValue]:
        return {
            "before_slice_sha256": self.before_slice_sha256,
            "end_byte": self.end_byte,
            "replacement_blob": self.replacement_blob.payload(),
            "start_byte": self.start_byte,
        }


@dataclass(frozen=True)
class CandidateSubmission:
    case_sha256: str
    declared_parent_source_sha256: str
    parent_bound_patch: tuple[ByteEdit, ...]
    candidate_source: bytes
    candidate_blob: BlobReference
    provenance: FrozenJson
    submission_sha256: str
    schema: str = "tycho.candidate-submission.v1"

    @classmethod
    def replace(
        cls,
        *,
        case: FrozenBuildCase,
        parent_source: bytes,
        candidate_source: bytes,
        declared_parent_source_sha256: str | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> CandidateSubmission:
        parent = canonicalize_source(bytes(parent_source))
        candidate = canonicalize_source(bytes(candidate_source))
        replacement_blob = BlobReference.from_bytes(
            candidate, media_type="text/x-python"
        )
        edit = ByteEdit(
            start_byte=0,
            end_byte=len(parent),
            before_slice_sha256=_digest(parent),
            replacement_blob=replacement_blob,
            replacement=candidate,
        )
        frozen_provenance = _validated_json(
            dict(provenance or {}), path="$.provenance", freeze=True
        )
        preliminary = cls(
            case_sha256=case.case_sha256,
            declared_parent_source_sha256=(
                declared_parent_source_sha256 or _digest(parent)
            ),
            parent_bound_patch=(edit,),
            candidate_source=candidate,
            candidate_blob=replacement_blob,
            provenance=frozen_provenance,
            submission_sha256="",
        )
        return cls(**{**preliminary.__dict__, "submission_sha256": object_sha256(
            "CandidateSubmission", preliminary.payload()
        )})

    def payload(self) -> dict[str, JsonValue]:
        return {
            "candidate_source": self.candidate_blob.payload(),
            "case_sha256": self.case_sha256,
            "declared_parent_source_sha256": self.declared_parent_source_sha256,
            "parent_bound_patch": [item.payload() for item in self.parent_bound_patch],
            "provenance": _thaw_json(self.provenance),
            "schema": self.schema,
        }


@dataclass(frozen=True)
class EvaluationSemanticResult:
    case_sha256: str
    submission_sha256: str
    candidate_source_sha256: str
    parent_source_sha256: str
    evaluator_sha256: str
    obligation_universe_sha256: str
    evaluation: TheoryEvaluation
    obligation_outcome_sha256: str
    semantic_result_sha256: str
    schema: str = "tycho.evaluation-semantic-result.v1"

    def payload(self) -> dict[str, JsonValue]:
        return {
            "candidate_source_sha256": self.candidate_source_sha256,
            "case_sha256": self.case_sha256,
            "evaluation": _semantic_evaluation_payload(self.evaluation),
            "evaluator_sha256": self.evaluator_sha256,
            "obligation_outcome_sha256": self.obligation_outcome_sha256,
            "obligation_universe_sha256": self.obligation_universe_sha256,
            "parent_source_sha256": self.parent_source_sha256,
            "schema": self.schema,
            "submission_sha256": self.submission_sha256,
        }


@dataclass(frozen=True)
class EvaluationReceipt:
    semantic: EvaluationSemanticResult
    attempt: int
    evaluation_wall_seconds: str
    evaluator_cost: str
    determinism_fault: bool = False

    @property
    def candidate_source_sha256(self) -> str:
        return self.semantic.candidate_source_sha256

    def builder_view(self) -> dict[str, JsonValue]:
        return {
            "candidate_source_sha256": self.candidate_source_sha256,
            "determinism_fault": self.determinism_fault,
            "evaluation_wall_seconds": self.evaluation_wall_seconds,
            "evaluator_cost": self.evaluator_cost,
            "evaluation": _evaluation_payload(self.semantic.evaluation),
            "obligation_outcome_sha256": (
                self.semantic.obligation_outcome_sha256
            ),
            "parent_source_sha256": self.semantic.parent_source_sha256,
            "semantic_result_sha256": self.semantic.semantic_result_sha256,
            "submission_sha256": self.semantic.submission_sha256,
        }


@dataclass(frozen=True)
class BuilderReport:
    case_sha256: str
    nominated_candidate_source_sha256: str | None
    judgment: str
    evidence_references: tuple[str, ...]
    open_uncertainties: tuple[str, ...]
    report_sha256: str
    schema: str = "tycho.builder-report.v1"

    @classmethod
    def create(
        cls,
        *,
        case: FrozenBuildCase,
        judgment: str,
        nominated_candidate_source_sha256: str | None = None,
        evidence_references: Sequence[str] = (),
        open_uncertainties: Sequence[str] = (),
    ) -> BuilderReport:
        preliminary = cls(
            case_sha256=case.case_sha256,
            nominated_candidate_source_sha256=nominated_candidate_source_sha256,
            judgment=judgment,
            evidence_references=tuple(evidence_references),
            open_uncertainties=tuple(open_uncertainties),
            report_sha256="",
        )
        return cls(**{**preliminary.__dict__, "report_sha256": object_sha256(
            "BuilderReport", preliminary.payload()
        )})

    def payload(self) -> dict[str, JsonValue]:
        return {
            "case_sha256": self.case_sha256,
            "evidence_references": list(self.evidence_references),
            "judgment": self.judgment,
            "nominated_candidate_source_sha256": (
                self.nominated_candidate_source_sha256
            ),
            "open_uncertainties": list(self.open_uncertainties),
            "schema": self.schema,
        }


class ProposalLedger:
    """Append-only host ledger for pure candidate evaluation."""

    def __init__(
        self,
        case: FrozenBuildCase,
        evaluator: ProposalEvaluator,
        store_root: Path,
        *,
        max_evaluations: int = 8,
    ) -> None:
        if max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        if evaluator.evaluator_sha256 != case.evaluator_sha256:
            raise ProposalValidationError("ledger evaluator does not match frozen case")
        self.case = case
        self.evaluator = evaluator
        self.store_root = store_root.resolve()
        self.max_evaluations = max_evaluations
        self._receipts: list[EvaluationReceipt] = []
        self._candidate_bytes: dict[str, bytes] = {
            case.incumbent_source_sha256: case.incumbent_source
        }
        self._evaluated_candidates: set[str] = set()
        self._parents: dict[str, str] = {}
        self._submissions: dict[str, CandidateSubmission] = {}
        self._deterministic_results: dict[str, str] = {}
        self._determinism_faults: set[str] = set()
        self._provider_turn_evaluations: int | None = None
        _write_immutable(
            self.store_root
            / "blobs"
            / f"{_bare_digest(case.incumbent_source_sha256)}.py",
            case.incumbent_source,
        )

    @classmethod
    def resume(
        cls,
        case: FrozenBuildCase,
        evaluator: ProposalEvaluator,
        store_root: Path,
        *,
        max_evaluations: int = 8,
    ) -> ProposalLedger:
        """Reconstruct the append-only receipt DAG without rerunning evaluation."""
        ledger = cls(
            case,
            evaluator,
            store_root,
            max_evaluations=max_evaluations,
        )
        observations: list[dict[str, JsonValue]] = []
        for path in (ledger.store_root / "observations").glob("*.json"):
            loaded = json.loads(path.read_bytes())
            if not isinstance(loaded, dict):
                raise ProposalValidationError("stored observation is malformed")
            observations.append(loaded)
        observations.sort(key=lambda item: int(item["attempt"]))
        for observation in observations:
            semantic_sha = str(observation["semantic_result_sha256"])
            receipt_path = (
                ledger.store_root
                / "receipts"
                / f"{_bare_digest(semantic_sha)}.json"
            )
            raw = json.loads(receipt_path.read_bytes())
            semantic_evaluation_raw = raw["evaluation"]
            evaluation_raw = observation["evaluation_observation"]
            results = tuple(
                ObligationResult(
                    obligation_id=str(item["obligation_id"]),
                    kind=ObligationKind(str(item["kind"])),
                    status=ObligationStatus(str(item["status"])),
                    description=str(item["description"]),
                    diagnostics=tuple(
                        (
                            str(key),
                            _validated_json(
                                value,
                                path=f"$.resume.diagnostics.{key}",
                                freeze=True,
                            ),
                        )
                        for key, value in dict(item["diagnostics"]).items()
                    ),
                )
                for item in evaluation_raw["obligations"]
            )
            evaluation = TheoryEvaluation(
                model_sha256=str(evaluation_raw["model_sha256"]),
                evidence_frontier_sha256=str(
                    evaluation_raw["evidence_frontier_sha256"]
                ),
                evaluator_sha256=str(evaluation_raw["evaluator_sha256"]),
                run_status=EvaluationRunStatus(str(evaluation_raw["run_status"])),
                obligations=results,
                error=(
                    None
                    if evaluation_raw["error"] is None
                    else str(evaluation_raw["error"])
                ),
            )
            if [
                _semantic_obligation_payload(item) for item in evaluation.obligations
            ] != semantic_evaluation_raw["obligations"]:
                raise ProposalValidationError(
                    "stored evaluation observation changes semantic outcomes"
                )
            semantic = EvaluationSemanticResult(
                case_sha256=str(raw["case_sha256"]),
                submission_sha256=str(raw["submission_sha256"]),
                candidate_source_sha256=str(raw["candidate_source_sha256"]),
                parent_source_sha256=str(raw["parent_source_sha256"]),
                evaluator_sha256=str(raw["evaluator_sha256"]),
                obligation_universe_sha256=str(raw["obligation_universe_sha256"]),
                evaluation=evaluation,
                obligation_outcome_sha256=str(raw["obligation_outcome_sha256"]),
                semantic_result_sha256=semantic_sha,
            )
            receipt = EvaluationReceipt(
                semantic=semantic,
                attempt=int(observation["attempt"]),
                evaluation_wall_seconds=str(observation["evaluation_wall_seconds"]),
                evaluator_cost=str(observation["evaluator_cost"]),
                determinism_fault=bool(observation["determinism_fault"]),
            )
            candidate_path = (
                ledger.store_root
                / "blobs"
                / f"{_bare_digest(semantic.candidate_source_sha256)}.py"
            )
            candidate = candidate_path.read_bytes()
            submission_path = (
                ledger.store_root
                / "submissions"
                / f"{_bare_digest(semantic.submission_sha256)}.json"
            )
            submission_raw = json.loads(submission_path.read_bytes())
            parent = ledger._candidate_bytes[semantic.parent_source_sha256]
            submission = CandidateSubmission.replace(
                case=case,
                parent_source=parent,
                candidate_source=candidate,
                declared_parent_source_sha256=semantic.parent_source_sha256,
                provenance=submission_raw["provenance"],
            )
            if submission.submission_sha256 != semantic.submission_sha256:
                raise ProposalValidationError("stored submission commitment is invalid")
            ledger._candidate_bytes[semantic.candidate_source_sha256] = candidate
            ledger._evaluated_candidates.add(semantic.candidate_source_sha256)
            ledger._parents[semantic.candidate_source_sha256] = (
                semantic.parent_source_sha256
            )
            ledger._submissions[submission.submission_sha256] = submission
            previous = ledger._deterministic_results.setdefault(
                semantic.submission_sha256, semantic.semantic_result_sha256
            )
            if receipt.determinism_fault or previous != semantic.semantic_result_sha256:
                ledger._determinism_faults.add(semantic.submission_sha256)
            ledger._receipts.append(receipt)
        ledger.validate_integrity()
        return ledger

    def begin_provider_turn(self) -> None:
        """Open the same one-submission evaluator budget for either adapter."""
        self._provider_turn_evaluations = 0

    @property
    def receipts(self) -> tuple[EvaluationReceipt, ...]:
        return tuple(self._receipts)

    @property
    def determinism_faults(self) -> tuple[str, ...]:
        return tuple(sorted(self._determinism_faults))

    def evaluate(self, submission: CandidateSubmission) -> EvaluationReceipt:
        if len(self._receipts) >= self.max_evaluations:
            raise ProposalValidationError("Builder evaluation budget is exhausted")
        if (
            self._provider_turn_evaluations is not None
            and self._provider_turn_evaluations >= 1
        ):
            raise ProposalValidationError(
                "only one candidate may be evaluated per provider turn"
            )
        self._validate_submission(submission)
        parent = self._candidate_bytes.get(submission.declared_parent_source_sha256)
        if parent is None:
            raise ProposalValidationError("declared parent is unknown to this session")
        if (
            submission.declared_parent_source_sha256
            != self.case.incumbent_source_sha256
            and submission.declared_parent_source_sha256
            not in self._evaluated_candidates
        ):
            raise ProposalValidationError("declared parent has no prior evaluation receipt")
        candidate = self._apply_patch(parent, submission.parent_bound_patch)
        if candidate != submission.candidate_source:
            raise ProposalValidationError(
                "host-reconstructed candidate differs from submitted candidate source"
            )
        submission.candidate_blob.verify(candidate)
        candidate_sha = submission.candidate_blob.sha256
        self._validate_acyclic_lineage(
            candidate_sha,
            submission.declared_parent_source_sha256,
        )

        started = time.monotonic()
        evaluation = self.evaluator.evaluate(candidate, self._frontier())
        wall_seconds = _canonical_decimal(time.monotonic() - started)
        evaluator_cost = _canonical_decimal(
            getattr(self.evaluator, "last_cost", "0")
        )
        self._validate_evaluation(evaluation, candidate)
        evaluation = _detached_evaluation(evaluation)
        outcome_payload = [
            _semantic_obligation_payload(item) for item in evaluation.obligations
        ]
        outcome_sha = object_sha256("ObligationOutcome", outcome_payload)
        preliminary = EvaluationSemanticResult(
            case_sha256=self.case.case_sha256,
            submission_sha256=submission.submission_sha256,
            candidate_source_sha256=submission.candidate_blob.sha256,
            parent_source_sha256=submission.declared_parent_source_sha256,
            evaluator_sha256=self.case.evaluator_sha256,
            obligation_universe_sha256=self.case.obligation_universe_sha256,
            evaluation=evaluation,
            obligation_outcome_sha256=outcome_sha,
            semantic_result_sha256="",
        )
        semantic = EvaluationSemanticResult(
            **{
                **preliminary.__dict__,
                "semantic_result_sha256": object_sha256(
                    "EvaluationSemanticResult", preliminary.payload()
                ),
            }
        )
        previous = self._deterministic_results.setdefault(
            submission.submission_sha256, semantic.semantic_result_sha256
        )
        fault = previous != semantic.semantic_result_sha256
        if fault:
            self._determinism_faults.add(submission.submission_sha256)
        receipt = EvaluationReceipt(
            semantic=semantic,
            attempt=len(self._receipts) + 1,
            evaluation_wall_seconds=wall_seconds,
            evaluator_cost=evaluator_cost,
            determinism_fault=fault,
        )
        self._persist(submission, receipt, candidate)
        self._receipts.append(receipt)
        self._candidate_bytes[candidate_sha] = candidate
        self._evaluated_candidates.add(candidate_sha)
        self._parents.setdefault(
            candidate_sha, submission.declared_parent_source_sha256
        )
        self._submissions[submission.submission_sha256] = submission
        if self._provider_turn_evaluations is not None:
            self._provider_turn_evaluations += 1
        return receipt

    def candidate_bytes(self, source_sha256: str) -> bytes:
        try:
            content = self._candidate_bytes[source_sha256]
        except KeyError as error:
            raise ProposalValidationError("candidate source is unknown to this ledger") from error
        if _digest(content) != source_sha256:
            raise RuntimeError("candidate bytes no longer match ledger commitment")
        return content

    def eligible_receipts(self) -> tuple[EvaluationReceipt, ...]:
        seen: set[str] = set()
        eligible: list[EvaluationReceipt] = []
        for receipt in self._receipts:
            submission_sha = receipt.semantic.submission_sha256
            source_sha = receipt.candidate_source_sha256
            if submission_sha in self._determinism_faults or source_sha in seen:
                continue
            seen.add(source_sha)
            eligible.append(receipt)
        return tuple(eligible)

    def receipt_for_source(self, source_sha256: str) -> EvaluationReceipt | None:
        return next(
            (
                item
                for item in reversed(self.eligible_receipts())
                if item.candidate_source_sha256 == source_sha256
            ),
            None,
        )

    def validate_integrity(self) -> None:
        """Recompute every authority-bearing commitment before publication."""
        if object_sha256("FrozenBuildCase", self.case.payload()) != self.case.case_sha256:
            raise ProposalValidationError("frozen case commitment is invalid")
        obligation_sha = object_sha256(
            "ObligationUniverse",
            [item.payload() for item in self.case.obligations],
        )
        if obligation_sha != self.case.obligation_universe_sha256:
            raise ProposalValidationError("obligation-universe commitment is invalid")
        for receipt in self._receipts:
            semantic = receipt.semantic
            submission = self._submissions.get(semantic.submission_sha256)
            if submission is None or (
                object_sha256("CandidateSubmission", submission.payload())
                != semantic.submission_sha256
            ):
                raise ProposalValidationError("submission commitment is invalid")
            expected_outcome = object_sha256(
                "ObligationOutcome",
                [
                    _semantic_obligation_payload(item)
                    for item in semantic.evaluation.obligations
                ],
            )
            if expected_outcome != semantic.obligation_outcome_sha256:
                raise ProposalValidationError("receipt outcome commitment is invalid")
            if (
                object_sha256("EvaluationSemanticResult", semantic.payload())
                != semantic.semantic_result_sha256
            ):
                raise ProposalValidationError("semantic receipt commitment is invalid")
            candidate = self.candidate_bytes(semantic.candidate_source_sha256)
            if hashlib.sha256(candidate).hexdigest() != semantic.evaluation.model_sha256:
                raise ProposalValidationError("receipt evaluation names another candidate")
            if self._parents.get(semantic.candidate_source_sha256) != (
                semantic.parent_source_sha256
            ):
                raise ProposalValidationError("receipt lineage commitment is invalid")

    def _validate_submission(self, submission: CandidateSubmission) -> None:
        if submission.case_sha256 != self.case.case_sha256:
            raise ProposalValidationError("submission belongs to another frozen case")
        expected = object_sha256("CandidateSubmission", submission.payload())
        if expected != submission.submission_sha256:
            raise ProposalValidationError("submission hash does not match its payload")
        if not submission.parent_bound_patch:
            raise ProposalValidationError("submission has no parent-bound patch")

    @staticmethod
    def _apply_patch(parent: bytes, edits: tuple[ByteEdit, ...]) -> bytes:
        previous_end = 0
        for edit in edits:
            if edit.start_byte < previous_end or edit.end_byte < edit.start_byte:
                raise ProposalValidationError("parent-bound edits overlap or are unordered")
            if edit.end_byte > len(parent):
                raise ProposalValidationError("parent-bound edit exceeds parent source")
            before = parent[edit.start_byte : edit.end_byte]
            if _digest(before) != edit.before_slice_sha256:
                raise ProposalValidationError("parent preimage does not match declared parent")
            edit.replacement_blob.verify(edit.replacement)
            previous_end = edit.end_byte
        candidate = bytearray(parent)
        for edit in reversed(edits):
            candidate[edit.start_byte : edit.end_byte] = edit.replacement
        return canonicalize_source(bytes(candidate))

    def _validate_evaluation(self, evaluation: TheoryEvaluation, candidate: bytes) -> None:
        if evaluation.model_sha256 != hashlib.sha256(candidate).hexdigest():
            raise ProposalValidationError("evaluator returned a different candidate digest")
        if evaluation.evidence_frontier_sha256 != self.case.evidence_frontier_sha256:
            raise ProposalValidationError("evaluator returned a different evidence frontier")
        if evaluation.evaluator_sha256 != self.case.evaluator_sha256:
            raise ProposalValidationError("evaluator identity changed during the session")
        definitions = tuple(
            (item.obligation_id, item.kind, item.description)
            for item in self.case.obligations
        )
        results = tuple(
            (item.obligation_id, item.kind, item.description)
            for item in evaluation.obligations
        )
        if definitions != results and evaluation.run_status is EvaluationRunStatus.COMPLETE:
            raise ProposalValidationError("evaluator changed the obligation universe")

    def _validate_acyclic_lineage(self, candidate_sha: str, parent_sha: str) -> None:
        existing_parent = self._parents.get(candidate_sha)
        if existing_parent is not None and existing_parent != parent_sha:
            raise ProposalValidationError(
                "candidate source already has a different declared parent"
            )
        ancestor = parent_sha
        visited: set[str] = set()
        while True:
            if ancestor == candidate_sha:
                raise ProposalValidationError("candidate lineage contains a cycle")
            if ancestor in visited:
                raise RuntimeError("host ledger contains a lineage cycle")
            visited.add(ancestor)
            if ancestor == self.case.incumbent_source_sha256:
                return
            try:
                ancestor = self._parents[ancestor]
            except KeyError as error:
                raise ProposalValidationError(
                    "candidate lineage does not terminate at the case incumbent"
                ) from error

    def _frontier(self) -> EvidenceFrontier:
        files = tuple(_evidence_file(item) for item in self.case.evidence)
        return EvidenceFrontier(self.case.evidence_frontier_sha256, files)

    def _persist(
        self,
        submission: CandidateSubmission,
        receipt: EvaluationReceipt,
        candidate: bytes,
    ) -> None:
        digest = _bare_digest(submission.candidate_blob.sha256)
        _write_immutable(self.store_root / "blobs" / f"{digest}.py", candidate)
        semantic_payload: dict[str, JsonValue] = {
            **receipt.semantic.payload(),
            "semantic_result_sha256": receipt.semantic.semantic_result_sha256,
        }
        receipt_name = _bare_digest(receipt.semantic.semantic_result_sha256)
        _write_immutable(
            self.store_root / "receipts" / f"{receipt_name}.json",
            canonical_json(semantic_payload) + b"\n",
        )
        submission_payload: dict[str, JsonValue] = {
            **submission.payload(),
            "submission_sha256": submission.submission_sha256,
        }
        submission_name = _bare_digest(submission.submission_sha256)
        _write_immutable(
            self.store_root / "submissions" / f"{submission_name}.json",
            canonical_json(submission_payload) + b"\n",
        )
        observation_payload: dict[str, JsonValue] = {
            "attempt": receipt.attempt,
            "determinism_fault": receipt.determinism_fault,
            "evaluation_wall_seconds": receipt.evaluation_wall_seconds,
            "evaluator_cost": receipt.evaluator_cost,
            "evaluation_observation": _evaluation_payload(
                receipt.semantic.evaluation
            ),
            "semantic_result_sha256": receipt.semantic.semantic_result_sha256,
            "submission_sha256": submission.submission_sha256,
        }
        _write_immutable(
            self.store_root
            / "observations"
            / f"{submission_name}-{receipt.attempt:06d}.json",
            canonical_json(observation_payload) + b"\n",
        )


def _validated_json(value: object, *, path: str, freeze: bool) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise TypeError(f"{path}: floating-point values are forbidden")
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            raise ValueError(f"{path}: protocol strings must use Unicode NFC")
        return value
    if isinstance(value, Mapping):
        normalized: list[tuple[str, object]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: object keys must be strings")
            if unicodedata.normalize("NFC", key) != key:
                raise ValueError(f"{path}: protocol keys must use Unicode NFC")
            normalized.append(
                (key, _validated_json(item, path=f"{path}.{key}", freeze=freeze))
            )
        normalized.sort(key=lambda pair: pair[0])
        if freeze:
            return _FrozenJsonObject(tuple(normalized))
        return {key: item for key, item in normalized}
    if isinstance(value, (list, tuple)):
        items = tuple(
            _validated_json(item, path=f"{path}[{index}]", freeze=freeze)
            for index, item in enumerate(value)
        )
        return items if freeze else list(items)
    raise TypeError(f"{path}: unsupported protocol value {type(value).__name__}")


def _thaw_json(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"invalid frozen JSON value: {type(value).__name__}")


def _evaluation_payload(evaluation: TheoryEvaluation) -> dict[str, JsonValue]:
    return {
        "error": evaluation.error,
        "evaluator_sha256": evaluation.evaluator_sha256,
        "evidence_frontier_sha256": evaluation.evidence_frontier_sha256,
        "model_sha256": evaluation.model_sha256,
        "obligations": [
            _obligation_result_payload(item) for item in evaluation.obligations
        ],
        "run_status": evaluation.run_status.value,
    }


def _semantic_evaluation_payload(
    evaluation: TheoryEvaluation,
) -> dict[str, JsonValue]:
    """Stable semantic result; observation-only diagnostics are deliberately absent."""
    return {
        "evaluator_sha256": evaluation.evaluator_sha256,
        "evidence_frontier_sha256": evaluation.evidence_frontier_sha256,
        "model_sha256": evaluation.model_sha256,
        "obligations": [
            _semantic_obligation_payload(item) for item in evaluation.obligations
        ],
        "run_status": evaluation.run_status.value,
    }


def _semantic_obligation_payload(result: ObligationResult) -> dict[str, JsonValue]:
    return {
        "kind": result.kind.value,
        "obligation_id": result.obligation_id,
        "status": result.status.value,
    }


def _obligation_result_payload(result: ObligationResult) -> dict[str, JsonValue]:
    diagnostics = {
        str(key): _validated_json(value, path=f"$.diagnostics.{key}", freeze=False)
        for key, value in result.diagnostics
    }
    return {
        "description": result.description,
        "diagnostics": diagnostics,
        "kind": result.kind.value,
        "obligation_id": result.obligation_id,
        "status": result.status.value,
    }


def _evidence_file(snapshot: EvidenceSnapshot) -> EvidenceFile:
    snapshot.blob.verify(snapshot.content)
    return EvidenceFile(
        path=snapshot.path,
        sha256=_bare_digest(snapshot.blob.sha256),
        content=snapshot.content,
    )


def _detached_evaluation(evaluation: TheoryEvaluation) -> TheoryEvaluation:
    """Copy evaluator output into the protocol's recursively immutable data profile."""
    return TheoryEvaluation(
        model_sha256=str(evaluation.model_sha256),
        evidence_frontier_sha256=str(evaluation.evidence_frontier_sha256),
        evaluator_sha256=str(evaluation.evaluator_sha256),
        run_status=evaluation.run_status,
        obligations=tuple(
            ObligationResult(
                obligation_id=str(item.obligation_id),
                kind=item.kind,
                status=item.status,
                description=str(item.description),
                diagnostics=tuple(
                    (
                        str(key),
                        _validated_json(
                            value,
                            path=f"$.obligations.{item.obligation_id}.{key}",
                            freeze=True,
                        ),
                    )
                    for key, value in item.diagnostics
                ),
            )
            for item in evaluation.obligations
        ),
        error=None if evaluation.error is None else str(evaluation.error),
    )


def _validate_unique_obligations(
    obligations: tuple[ObligationDefinition, ...],
) -> None:
    identifiers = [item.obligation_id for item in obligations]
    if len(set(identifiers)) != len(identifiers):
        raise ProposalValidationError("obligation universe contains duplicate IDs")


def _validate_manifest_path(path: str) -> None:
    parsed = PurePosixPath(path)
    if (
        not path
        or parsed.is_absolute()
        or "\\" in path
        or ":" in parsed.parts[0]
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != path
    ):
        raise ProposalValidationError(f"unsafe evidence manifest path: {path!r}")


def _digest(content: bytes) -> str:
    return _SHA_PREFIX + hashlib.sha256(content).hexdigest()


def _canonical_decimal(value: object) -> str:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ProposalValidationError("cost or timing metadata is not decimal") from error
    if not decimal.is_finite() or decimal < 0:
        raise ProposalValidationError("cost or timing metadata must be finite and nonnegative")
    normalized = format(decimal, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _bare_digest(value: str) -> str:
    if not value.startswith(_SHA_PREFIX) or len(value) != len(_SHA_PREFIX) + 64:
        raise ProposalValidationError(f"invalid SHA-256 commitment: {value!r}")
    digest = value.removeprefix(_SHA_PREFIX)
    if any(character not in "0123456789abcdef" for character in digest):
        raise ProposalValidationError(f"invalid SHA-256 commitment: {value!r}")
    return digest


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable proposal path changed content: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError(f"immutable proposal path changed content: {path}")
    finally:
        temporary.unlink(missing_ok=True)
