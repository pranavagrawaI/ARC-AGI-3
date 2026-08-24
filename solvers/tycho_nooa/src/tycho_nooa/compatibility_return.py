"""Gate-local terminal contract for the compatibility ladder."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict

from tycho_nooa.cognitive_parity import RetrievalAnswer
from tycho_nooa.evidence import EvidenceRef


class RetrievalReturnArguments(BaseModel):
    """A retrieval terminal contract without a union or wrapper object."""

    model_config = ConfigDict(extra="forbid")

    evidence_refs: tuple[EvidenceRef, ...]
    conclusion: str

    def answer(self) -> RetrievalAnswer:
        return RetrievalAnswer(
            evidence_refs=self.evidence_refs,
            conclusion=self.conclusion,
        )


def retrieval_return_schema_sha256() -> str:
    """Commit the flat retrieval-only compatibility terminal schema."""
    encoded = json.dumps(
        RetrievalReturnArguments.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
