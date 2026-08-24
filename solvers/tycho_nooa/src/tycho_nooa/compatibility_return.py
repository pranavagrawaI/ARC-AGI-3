"""Minimal terminal contracts for each compatibility-ladder burden."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from tycho_nooa.cognitive_parity import RetrievalAnswer
from tycho_nooa.evidence import EvidenceRef


class _CompatibilityReturn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TerminationReturn(_CompatibilityReturn):
    ok: bool


class CapabilityRetrievalReturn(_CompatibilityReturn):
    evidence_ref: str = Field(pattern=r"^L\d+:T\d+$")


class IdentifierBindingReturn(_CompatibilityReturn):
    evidence_ref: str = Field(pattern=r"^L\d+:T\d+$")


class RepresentationReturn(_CompatibilityReturn):
    decoded_value: int


class CompositionReturn(_CompatibilityReturn):
    conclusion: str
    evidence_ref: str = Field(pattern=r"^L\d+:T\d+$")


type CompatibilityReturn = (
    TerminationReturn
    | CapabilityRetrievalReturn
    | IdentifierBindingReturn
    | RepresentationReturn
    | CompositionReturn
)

RETURN_MODELS: dict[str, type[_CompatibilityReturn]] = {
    "termination": TerminationReturn,
    "capability_retrieval": CapabilityRetrievalReturn,
    "identifier_binding": IdentifierBindingReturn,
    "representation_rle": RepresentationReturn,
    "composition": CompositionReturn,
}


def return_model_for_gate(gate_kind: str) -> type[_CompatibilityReturn]:
    try:
        return RETURN_MODELS[gate_kind]
    except KeyError as error:
        raise ValueError(f"unsupported compatibility gate kind: {gate_kind}") from error


def compatibility_return_schemas_sha256() -> str:
    """Commit every gate-specific model-facing terminal schema as one contract."""
    schemas = {
        gate_kind: model.model_json_schema()
        for gate_kind, model in RETURN_MODELS.items()
    }
    encoded = json.dumps(schemas, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class RetrievalReturnArguments(_CompatibilityReturn):
    """Frozen V9 contract retained so old experiment packages remain readable."""

    evidence_refs: tuple[EvidenceRef, ...]
    conclusion: str

    def answer(self) -> RetrievalAnswer:
        return RetrievalAnswer(
            evidence_refs=self.evidence_refs,
            conclusion=self.conclusion,
        )


def retrieval_return_schema_sha256() -> str:
    encoded = json.dumps(
        RetrievalReturnArguments.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
