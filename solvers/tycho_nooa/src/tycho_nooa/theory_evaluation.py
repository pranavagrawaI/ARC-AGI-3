"""Typed, pointwise publication judgments over one frozen evidence frontier."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tycho.workspace import wmlib_template
from tycho.workspace.sandbox import PythonSandbox

from tycho_nooa import probe_gate

_RESULT_PREFIX = "THEORY_EVALUATION_JSON="


@dataclass(frozen=True)
class EvidenceFile:
    path: str
    sha256: str
    content: bytes


@dataclass(frozen=True)
class EvidenceFrontier:
    """Immutable bytes and commitment for every verifier-visible public artifact."""

    sha256: str
    files: tuple[EvidenceFile, ...]

    @classmethod
    def freeze(cls, workspace: Path) -> EvidenceFrontier:
        files = tuple(
            EvidenceFile(
                path=path.relative_to(workspace).as_posix(),
                sha256=hashlib.sha256(content).hexdigest(),
                content=content,
            )
            for path in _evidence_paths(workspace)
            if (content := path.read_bytes()) is not None
        )
        manifest = [{"path": item.path, "sha256": item.sha256} for item in files]
        commitment = hashlib.sha256(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        return cls(commitment, files)


def _evidence_paths(workspace: Path) -> Iterable[Path]:
    paths: set[Path] = set()
    attempts = workspace / "attempts"
    if attempts.is_dir():
        paths.update(path for path in attempts.rglob("*") if path.is_file())
    paths.update(
        path
        for level in workspace.glob("level_*")
        if level.is_dir()
        for path in level.rglob("*")
        if path.is_file()
    )
    marker = workspace / ".current_frame.json"
    if marker.is_file():
        paths.add(marker)
    return sorted(paths, key=lambda path: path.relative_to(workspace).as_posix())


class EvaluationRunStatus(StrEnum):
    """Why an evaluation can or cannot support a publication judgment."""

    COMPLETE = "complete"
    CANDIDATE_LOAD_FAILED = "candidate_load_failed"
    CANDIDATE_RUNTIME_FAILED = "candidate_runtime_failed"
    EVALUATOR_UNAVAILABLE = "evaluator_unavailable"
    INCONCLUSIVE_NOT_PUBLISHABLE = "inconclusive_not_publishable"


class ObligationKind(StrEnum):
    TRANSITION = "transition"
    INITIAL_RENDER = "initial_render"
    OUTCOME = "outcome"
    TERMINAL = "terminal"
    PLANNER_CONTRACT = "planner_contract"


class ObligationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class PublicationDisposition(StrEnum):
    PUBLISHABLE = "publishable"
    REJECTED_REGRESSION = "rejected_regression"
    REJECTED_INVALID = "rejected_invalid"
    WITHHELD_INCONCLUSIVE = "withheld_inconclusive"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class ObligationResult:
    """One independently named fact the candidate must preserve or improve."""

    obligation_id: str
    kind: ObligationKind
    status: ObligationStatus
    description: str
    diagnostics: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class TheoryEvaluation:
    """Complete typed result for one model on one committed evaluator/frontier."""

    model_sha256: str
    evidence_frontier_sha256: str
    evaluator_sha256: str
    run_status: EvaluationRunStatus
    obligations: tuple[ObligationResult, ...] = ()
    error: str | None = None

    def obligation_map(self) -> dict[str, ObligationResult]:
        result = {item.obligation_id: item for item in self.obligations}
        if len(result) != len(self.obligations):
            raise ValueError("evaluation contains duplicate obligation IDs")
        return result


@dataclass(frozen=True)
class PublicationDecision:
    disposition: PublicationDisposition
    gained: tuple[str, ...] = ()
    lost: tuple[str, ...] = ()
    inconclusive: tuple[str, ...] = ()
    reason: str = ""


class SandboxTheoryEvaluator:
    """Run the host probe against copies of candidate code and frozen evidence."""

    def __init__(
        self,
        sandbox: PythonSandbox | None = None,
        *,
        allow_unsafe_host: bool = False,
    ) -> None:
        self._sandbox = sandbox or PythonSandbox()
        self._allow_unsafe_host = allow_unsafe_host
        self._probe_source = (
            Path(__file__).with_name("templates")
            / "theory_evaluation_probe.py.tmpl"
        ).read_bytes()
        self._wmlib_source = Path(wmlib_template.__file__).read_bytes()
        self._planner_source = Path(probe_gate.__file__).read_bytes()
        self.evaluator_sha256 = hashlib.sha256(
            b"\0".join(
                (self._probe_source, self._wmlib_source, self._planner_source)
            )
        ).hexdigest()

    def evaluate(
        self,
        model: bytes,
        frontier: EvidenceFrontier,
    ) -> TheoryEvaluation:
        model_sha = hashlib.sha256(model).hexdigest()
        if self._sandbox.runtime == "host" and not self._allow_unsafe_host:
            return self._unavailable(
                model_sha,
                frontier,
                "publication evaluation requires an isolated container; "
                "host runtime is trusted-development only",
            )
        result, error = self._run_probe(model, frontier)
        if error is not None:
            return self._unavailable(model_sha, frontier, error)
        execution_error = self._execution_error(result)
        if execution_error is not None:
            return self._unavailable(model_sha, frontier, execution_error)
        payload = self._payload(result.stdout)
        if payload is None:
            return self._unavailable(
                model_sha, frontier, "typed evaluator returned no result envelope"
            )
        return self._typed_evaluation(model_sha, frontier, payload)

    def _run_probe(
        self, model: bytes, frontier: EvidenceFrontier
    ) -> tuple[object, str | None]:
        try:
            with tempfile.TemporaryDirectory(prefix="tycho-theory-eval-") as root:
                bundle = Path(root)
                self._write_bundle(bundle, model, frontier)
                result = self._sandbox.run_script(
                    bundle,
                    bundle / "theory_probe.py",
                    timeout=30,
                )
        except Exception as error:  # noqa: BLE001 - fail closed on judge failure
            return object(), f"{type(error).__name__}: {error}"
        return result, None

    @staticmethod
    def _execution_error(result: object) -> str | None:
        if result.timed_out or result.returncode != 0:
            if result.timed_out:
                return "typed evaluator timed out"
            return f"typed evaluator exited {result.returncode}: {result.stderr}"
        return None

    def _unavailable(
        self, model_sha: str, frontier: EvidenceFrontier, error: str
    ) -> TheoryEvaluation:
        return TheoryEvaluation(
            model_sha,
            frontier.sha256,
            self.evaluator_sha256,
            EvaluationRunStatus.EVALUATOR_UNAVAILABLE,
            error=error,
        )

    def _typed_evaluation(
        self,
        model_sha: str,
        frontier: EvidenceFrontier,
        payload: dict[str, object],
    ) -> TheoryEvaluation:
        return TheoryEvaluation(
            model_sha256=model_sha,
            evidence_frontier_sha256=frontier.sha256,
            evaluator_sha256=self.evaluator_sha256,
            run_status=EvaluationRunStatus(payload["run_status"]),
            obligations=tuple(
                ObligationResult(
                    obligation_id=item["obligation_id"],
                    kind=ObligationKind(item["kind"]),
                    status=ObligationStatus(item["status"]),
                    description=item["description"],
                    diagnostics=tuple(
                        sorted((item.get("diagnostics") or {}).items())
                    ),
                )
                for item in payload.get("obligations", [])
            ),
            error=payload.get("error"),
        )

    def _write_bundle(
        self,
        bundle: Path,
        model: bytes,
        frontier: EvidenceFrontier,
    ) -> None:
        for name, content in (
            ("world_model.py", model),
            ("wmlib.py", self._wmlib_source),
            ("host_planner_contract.py", self._planner_source),
            ("theory_probe.py", self._probe_source),
        ):
            (bundle / name).write_bytes(content)
        for item in frontier.files:
            path = bundle / item.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item.content)

    @staticmethod
    def _payload(stdout: str) -> dict[str, object] | None:
        matches = [
            line.removeprefix(_RESULT_PREFIX)
            for line in stdout.splitlines()
            if line.startswith(_RESULT_PREFIX)
        ]
        if not matches:
            return None
        try:
            value = json.loads(matches[-1])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


def compare(
    parent: TheoryEvaluation,
    candidate: TheoryEvaluation,
) -> PublicationDecision:
    """Order two evaluations pointwise without letting one obligation buy another."""
    if precondition := _comparison_precondition(parent, candidate):
        return precondition

    parent_by_id = parent.obligation_map()
    candidate_by_id = candidate.obligation_map()
    if universe_change := _obligation_universe_change(parent_by_id, candidate_by_id):
        return universe_change

    gained, lost, inconclusive = _obligation_deltas(parent_by_id, candidate_by_id)
    if lost:
        return PublicationDecision(
            PublicationDisposition.REJECTED_REGRESSION,
            gained=gained,
            lost=lost,
            inconclusive=inconclusive,
            reason="candidate lost protected obligations",
        )
    if inconclusive:
        return PublicationDecision(
            PublicationDisposition.WITHHELD_INCONCLUSIVE,
            gained=gained,
            inconclusive=inconclusive,
            reason="candidate has unordered obligation results",
        )
    if any(_is_publication_gain(candidate_by_id[item]) for item in gained):
        return PublicationDecision(
            PublicationDisposition.PUBLISHABLE,
            gained=gained,
            reason="candidate preserves every obligation and adds a named gain",
        )
    return PublicationDecision(
        PublicationDisposition.UNCHANGED,
        gained=gained,
        reason=(
            "candidate has no publication-driving transition or "
            "host-confirmed outcome gain"
            if gained
            else "candidate is evaluation-equivalent on the frozen frontier"
        ),
    )


def _comparison_precondition(
    parent: TheoryEvaluation,
    candidate: TheoryEvaluation,
) -> PublicationDecision | None:
    if candidate.run_status in {
        EvaluationRunStatus.CANDIDATE_LOAD_FAILED,
        EvaluationRunStatus.CANDIDATE_RUNTIME_FAILED,
    }:
        return PublicationDecision(
            PublicationDisposition.REJECTED_INVALID,
            reason=candidate.error or candidate.run_status.value,
        )
    if candidate.run_status is not EvaluationRunStatus.COMPLETE:
        return PublicationDecision(
            PublicationDisposition.WITHHELD_INCONCLUSIVE,
            reason=candidate.error or candidate.run_status.value,
        )
    if parent.run_status is not EvaluationRunStatus.COMPLETE:
        return PublicationDecision(
            PublicationDisposition.WITHHELD_INCONCLUSIVE,
            reason="parent evaluation is not complete",
        )
    if (
        parent.evidence_frontier_sha256 != candidate.evidence_frontier_sha256
        or parent.evaluator_sha256 != candidate.evaluator_sha256
    ):
        return PublicationDecision(
            PublicationDisposition.WITHHELD_INCONCLUSIVE,
            reason="parent and candidate were not evaluated by the same committed judge",
        )
    return None


def _obligation_universe_change(
    parent_by_id: dict[str, ObligationResult],
    candidate_by_id: dict[str, ObligationResult],
) -> PublicationDecision | None:
    if parent_by_id.keys() != candidate_by_id.keys():
        missing = sorted(parent_by_id.keys() - candidate_by_id.keys())
        added = sorted(candidate_by_id.keys() - parent_by_id.keys())
        detail = tuple(f"missing:{item}" for item in missing) + tuple(
            f"unexpected:{item}" for item in added
        )
        return PublicationDecision(
            PublicationDisposition.WITHHELD_INCONCLUSIVE,
            inconclusive=detail,
            reason="obligation universe changed on a frozen frontier",
        )
    return None


def _obligation_deltas(
    parent_by_id: dict[str, ObligationResult],
    candidate_by_id: dict[str, ObligationResult],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    gained: list[str] = []
    lost: list[str] = []
    inconclusive: list[str] = []
    for obligation_id in sorted(parent_by_id):
        before = parent_by_id[obligation_id]
        after = candidate_by_id[obligation_id]
        if before.kind is not after.kind:
            inconclusive.append(obligation_id)
            continue
        if after.status in {ObligationStatus.UNSUPPORTED, ObligationStatus.ERROR}:
            inconclusive.append(obligation_id)
        if (
            before.status is ObligationStatus.PASSED
            and after.status is not ObligationStatus.PASSED
        ):
            lost.append(obligation_id)
        elif (
            before.status is not ObligationStatus.PASSED
            and after.status is ObligationStatus.PASSED
        ):
            gained.append(obligation_id)
    return tuple(gained), tuple(lost), tuple(inconclusive)


def _is_publication_gain(obligation: ObligationResult) -> bool:
    transition_gain = (
        obligation.kind is ObligationKind.TRANSITION
        and dict(obligation.diagnostics).get("evaluated", True) is not False
    )
    return transition_gain or (
        obligation.kind is ObligationKind.OUTCOME
        and obligation.obligation_id.startswith("outcome:terminal:")
    )
