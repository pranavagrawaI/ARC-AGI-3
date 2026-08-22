"""Crash-recoverable publication transactions for Builder-authored theory edits."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from tycho_nooa.theory_evaluation import (
    EvidenceFrontier,
    ObligationKind,
    ObligationStatus,
    PublicationDecision,
    PublicationDisposition,
    TheoryEvaluation,
    compare,
)


class TheoryEvaluator(Protocol):
    evaluator_sha256: str

    def evaluate(
        self, model: bytes, frontier: EvidenceFrontier
    ) -> TheoryEvaluation: ...


@dataclass(frozen=True)
class TheoryEditResult:
    tool_result: str
    feedback: str
    decision: PublicationDecision
    record_id: str | None

    def render(self) -> str:
        return f"{self.tool_result}\n\n{self.feedback}" if self.tool_result else self.feedback


class TheoryTransaction:
    """Keep one monotonically advancing incumbent inside a Builder pass."""

    def __init__(
        self,
        *,
        workspace: Path,
        evaluator: TheoryEvaluator,
        frontier: EvidenceFrontier,
        cause: str,
        focus: str | None,
        store_root: Path,
        canonical_bytes: bytes,
        canonical_evaluation: TheoryEvaluation,
    ) -> None:
        self.workspace = workspace
        self.evaluator = evaluator
        self.frontier = frontier
        self.cause = cause
        self.focus = focus
        self.store_root = store_root
        self.transaction_id = uuid.uuid4().hex
        self._canonical_bytes = canonical_bytes
        self._canonical_evaluation = canonical_evaluation
        self._working_bytes = canonical_bytes
        self._working_evaluation = canonical_evaluation
        self._closed = False

    @classmethod
    def begin(
        cls,
        *,
        workspace: Path,
        evaluator: TheoryEvaluator,
        frontier: EvidenceFrontier,
        cause: str,
        focus: str | None = None,
        store_root: Path | None = None,
    ) -> TheoryTransaction:
        workspace = workspace.resolve()
        store = (
            store_root.resolve()
            if store_root is not None
            else workspace.parent / ".theory-transactions" / workspace.name
        )
        store.mkdir(parents=True, exist_ok=True)
        cls._recover_incomplete(workspace, store)
        canonical = (workspace / "world_model.py").read_bytes()
        evaluation = evaluator.evaluate(canonical, frontier)
        transaction = cls(
            workspace=workspace,
            evaluator=evaluator,
            frontier=frontier,
            cause=cause,
            focus=focus,
            store_root=store,
            canonical_bytes=canonical,
            canonical_evaluation=evaluation,
        )
        canonical_sha = transaction._write_blob(canonical)
        transaction._write_active(
            {
                "schema_version": 1,
                "transaction_id": transaction.transaction_id,
                "workspace": str(workspace),
                "canonical_sha256": canonical_sha,
                "frontier_sha256": frontier.sha256,
                "cause": cause,
                "focus": focus,
            }
        )
        transaction._journal(
            "transaction_began",
            canonical_sha256=canonical_sha,
            frontier_sha256=frontier.sha256,
            cause=cause,
            focus=focus,
        )
        return transaction

    def apply_edit(
        self,
        operation: str,
        mutate: Callable[[], str],
    ) -> TheoryEditResult:
        self._require_open()
        parent_bytes = self._working_bytes
        parent_evaluation = self._working_evaluation
        parent_sha = hashlib.sha256(parent_bytes).hexdigest()
        self._journal("edit_intent", operation=operation, parent_sha256=parent_sha)
        try:
            tool_result = mutate()
        except BaseException:
            self._restore(parent_bytes)
            self._journal("edit_failed_and_restored", operation=operation)
            raise
        candidate = (self.workspace / "world_model.py").read_bytes()
        return self._assess(
            candidate,
            operation=operation,
            tool_result=tool_result,
            parent_bytes=parent_bytes,
            parent_evaluation=parent_evaluation,
        )

    def checkout(self, record_id: str) -> TheoryEditResult:
        self._require_open()
        candidate = self.candidate_bytes(record_id)
        return self._assess(
            candidate,
            operation=f"checkout:{record_id}",
            tool_result=f"checked immutable candidate {record_id}",
            parent_bytes=self._working_bytes,
            parent_evaluation=self._working_evaluation,
            source_record_id=record_id,
        )

    def finish(self) -> PublicationDecision:
        self._require_open()
        decision = compare(self._canonical_evaluation, self._working_evaluation)
        if decision.disposition is PublicationDisposition.PUBLISHABLE:
            disposition = "published"
        else:
            disposition = "canonical_restored"
            self._restore(self._canonical_bytes)
            self._working_bytes = self._canonical_bytes
            self._working_evaluation = self._canonical_evaluation
        canonical = {
            "schema_version": 1,
            "transaction_id": self.transaction_id,
            "parent_id": _theory_id(self._canonical_evaluation),
            "child_id": _theory_id(self._working_evaluation),
            "parent_model_sha256": self._canonical_evaluation.model_sha256,
            "model_sha256": self._working_evaluation.model_sha256,
            "evidence_frontier_sha256": self.frontier.sha256,
            "evaluator_sha256": self._working_evaluation.evaluator_sha256,
            "cause": self.cause,
            "focus": self.focus,
            "disposition": disposition,
            "decision": _jsonable(decision),
            "parent_evaluation": _jsonable(self._canonical_evaluation),
            "child_evaluation": _jsonable(self._working_evaluation),
            "canonical_path": str(self.workspace / "world_model.py"),
        }
        _write_json_atomic(self.store_root / "canonical.json", canonical)
        if disposition == "published":
            self._invalidate_stale_plan()
        self._journal("transaction_finished", **canonical)
        self._close()
        return decision

    def abort(self, reason: str) -> None:
        self._require_open()
        self._restore(self._canonical_bytes)
        self._journal(
            "transaction_aborted",
            reason=reason,
            restored_sha256=self._canonical_evaluation.model_sha256,
        )
        self._close()

    def candidate_bytes(self, record_id: str) -> bytes:
        if re.fullmatch(r"[0-9a-f]{16}-[0-9a-f]{16}", record_id) is None:
            raise ValueError(f"invalid candidate record ID: {record_id!r}")
        record_path = self.store_root / "records" / f"{record_id}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("record_id") != record_id:
            raise RuntimeError(f"candidate record ID mismatch: {record_id}")
        record_body = {
            key: value
            for key, value in record.items()
            if key not in {"record_id", "record_sha256"}
        }
        record_sha = hashlib.sha256(
            json.dumps(record_body, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        if record.get("record_sha256") != record_sha:
            raise RuntimeError(f"candidate record commitment mismatch: {record_id}")
        model_sha = str(record["model_sha256"])
        expected_id = f"{model_sha[:16]}-{record_sha[:16]}"
        if record_id != expected_id:
            raise RuntimeError(f"candidate record name commitment mismatch: {record_id}")
        blob = self.store_root / "blobs" / f"{model_sha}.py"
        content = blob.read_bytes()
        if hashlib.sha256(content).hexdigest() != model_sha:
            raise RuntimeError(f"candidate blob commitment mismatch: {record_id}")
        source_frontier = str(record.get("evidence_frontier_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", source_frontier) is None:
            raise RuntimeError(f"candidate frontier commitment is invalid: {record_id}")
        return content

    def _assess(
        self,
        candidate: bytes,
        *,
        operation: str,
        tool_result: str,
        parent_bytes: bytes,
        parent_evaluation: TheoryEvaluation,
        source_record_id: str | None = None,
    ) -> TheoryEditResult:
        try:
            candidate_sha = self._write_blob(candidate)
            self._journal(
                "candidate_staged",
                operation=operation,
                model_sha256=candidate_sha,
                source_record_id=source_record_id,
            )
            candidate_evaluation = self.evaluator.evaluate(candidate, self.frontier)
            decision = compare(parent_evaluation, candidate_evaluation)
            accepted = decision.disposition is PublicationDisposition.PUBLISHABLE
            record_id = self._record_candidate(
                operation=operation,
                source_record_id=source_record_id,
                parent_evaluation=parent_evaluation,
                candidate_evaluation=candidate_evaluation,
                decision=decision,
            )
            self._journal(
                "candidate_assessed",
                operation=operation,
                record_id=record_id,
                disposition=decision.disposition.value,
                restored=not accepted,
            )
        except BaseException as error:
            self._restore(parent_bytes)
            self._working_bytes = parent_bytes
            self._working_evaluation = parent_evaluation
            with suppress(Exception):  # Preserve the assessment failure as primary.
                self._journal(
                    "candidate_assessment_failed_and_restored",
                    operation=operation,
                    error=f"{type(error).__name__}: {error}",
                )
            raise
        if accepted:
            self._restore(candidate)
            self._working_bytes = candidate
            self._working_evaluation = candidate_evaluation
        else:
            self._restore(parent_bytes)
        return TheoryEditResult(
            tool_result=tool_result,
            feedback=self._feedback(
                parent_evaluation,
                candidate_evaluation,
                decision,
                record_id,
            ),
            decision=decision,
            record_id=record_id,
        )

    def _record_candidate(
        self,
        *,
        operation: str,
        source_record_id: str | None,
        parent_evaluation: TheoryEvaluation,
        candidate_evaluation: TheoryEvaluation,
        decision: PublicationDecision,
    ) -> str:
        context = {
            "transaction_id": self.transaction_id,
            "operation": operation,
            "source_record_id": source_record_id,
            "parent_model_sha256": parent_evaluation.model_sha256,
            "model_sha256": candidate_evaluation.model_sha256,
            "evidence_frontier_sha256": self.frontier.sha256,
            "decision": _jsonable(decision),
        }
        record_body = {
            "schema_version": 1,
            **context,
            "evaluation": _jsonable(candidate_evaluation),
            "authority": "noncanonical_candidate",
        }
        record_sha = hashlib.sha256(
            json.dumps(record_body, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        record_id = f"{candidate_evaluation.model_sha256[:16]}-{record_sha[:16]}"
        record = {
            "record_id": record_id,
            "record_sha256": record_sha,
            **record_body,
        }
        _write_immutable(
            self.store_root / "records" / f"{record_id}.json",
            (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
        )
        return record_id

    def _feedback(
        self,
        parent: TheoryEvaluation,
        candidate: TheoryEvaluation,
        decision: PublicationDecision,
        record_id: str,
    ) -> str:
        accepted = decision.disposition is PublicationDisposition.PUBLISHABLE
        lines = [
            "[theory transaction]",
            (
                "disposition: accepted_as_working_incumbent"
                if accepted
                else "disposition: rejected_and_restored"
            ),
            f"parent:    {_exact_score(parent)} exact",
            f"candidate: {_exact_score(candidate)} exact",
        ]
        if decision.lost:
            lines.append("lost protected obligations:")
            lines.extend(f"- {item}" for item in decision.lost)
        if decision.gained:
            lines.append("newly satisfied obligations:")
            lines.extend(f"- {item}" for item in decision.gained)
        lines.append(f"retained candidate: {record_id}")
        if not accepted:
            lines.append(
                "world_model.py restored to working incumbent sha256:"
                f"{parent.model_sha256}"
            )
        return "\n".join(lines)

    def _write_blob(self, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        _write_immutable(self.store_root / "blobs" / f"{digest}.py", content)
        return digest

    def _restore(self, content: bytes) -> None:
        path = self.workspace / "world_model.py"
        temporary = path.with_name(f".{path.name}.restore-{self.transaction_id}")
        temporary.write_bytes(content)
        temporary.replace(path)

    def _invalidate_stale_plan(self) -> None:
        plan = self.workspace / "notes" / "validated_plan.json"
        plan.unlink(missing_ok=True)

    def _write_active(self, value: object) -> None:
        _write_json_atomic(self.store_root / "active.json", value)

    def _journal(self, event: str, **fields: object) -> None:
        path = self.store_root / "ledger.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "event": event,
            "transaction_id": self.transaction_id,
            **fields,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _close(self) -> None:
        (self.store_root / "active.json").unlink(missing_ok=True)
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("theory transaction is already closed")

    @staticmethod
    def _recover_incomplete(workspace: Path, store: Path) -> None:
        active = store / "active.json"
        if not active.is_file():
            return
        record = json.loads(active.read_text(encoding="utf-8"))
        if Path(record["workspace"]).resolve() != workspace:
            raise RuntimeError("active theory transaction belongs to another workspace")
        digest = str(record["canonical_sha256"])
        canonical_path = store / "canonical.json"
        if canonical_path.is_file():
            canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
            if (
                canonical.get("transaction_id") == record.get("transaction_id")
                and canonical.get("disposition") == "published"
            ):
                digest = str(canonical["model_sha256"])
        blob = store / "blobs" / f"{digest}.py"
        content = blob.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError("cannot recover: canonical theory blob is corrupt")
        target = workspace / "world_model.py"
        temporary = target.with_name(f".{target.name}.recovery")
        temporary.write_bytes(content)
        temporary.replace(target)
        if digest != str(record["canonical_sha256"]):
            (workspace / "notes" / "validated_plan.json").unlink(missing_ok=True)
        active.unlink()


def _exact_score(evaluation: TheoryEvaluation) -> str:
    transitions = [
        item
        for item in evaluation.obligations
        if item.kind is ObligationKind.TRANSITION
    ]
    evaluated = [
        item
        for item in transitions
        if dict(item.diagnostics).get("evaluated", True) is not False
    ]
    passed = sum(item.status is ObligationStatus.PASSED for item in evaluated)
    return f"{passed}/{len(evaluated)}"


def _theory_id(evaluation: TheoryEvaluation) -> str:
    return (
        f"theory-{evaluation.model_sha256[:16]}-"
        f"{evaluation.evidence_frontier_sha256[:16]}"
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable theory path changed content: {path}")
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
                raise RuntimeError(f"immutable theory path changed content: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
