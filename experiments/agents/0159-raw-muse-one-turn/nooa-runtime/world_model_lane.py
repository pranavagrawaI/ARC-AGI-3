"""Brokered durable executable commitments for the 0159 Muse worker."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

CandidateStatus = Literal[
    "rejected_operationally",
    "active_contradicted",
    "active_inconclusive",
    "active_supported",
]
EVALUATOR_PROTOCOL = "wm1-evaluator-v1"

QUICK_THEORY_SOURCE = '''import numpy as np

UNKNOWN = -1

def predict(grid: np.ndarray, action: dict) -> np.ndarray | None:
    """Return a partial next grid, or None to abstain for this action."""
    return None
'''

SEED_SOURCE = '''from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    grid: np.ndarray

def init_state(grid0: np.ndarray, level: int) -> State:
    return State(grid0.copy())

def transition(state: State, action: dict) -> State:
    return State(state.grid.copy())

def render(state: State) -> np.ndarray:
    return state.grid

def outcome(state: State) -> str:
    return "ongoing"
'''

ALLOWED_IMPORTS = frozenset({"dataclasses", "numpy", "math", "copy", "typing", "collections"})
MAX_SOURCE_BYTES = 96_000
PROHIBITED_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "input",
        "help",
        "sys",
        "os",
        "socket",
        "subprocess",
    }
)
PROHIBITED_ATTRIBUTES = frozenset(
    {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__globals__",
        "__code__",
        "__dict__",
        "__mro__",
        "__loader__",
        "__spec__",
        "fromfile",
        "tofile",
        "load",
        "save",
        "savez",
        "memmap",
        "dump",
        "sys",
        "os",
        "modules",
        "environ",
        "fork",
        "spawn",
        "system",
        "popen",
        "ctypeslib",
        "load_library",
        "DataSource",
    }
)

_E0_PATH = Path(__file__).resolve().parent.parent / "e0_evidence.py"
_E0_SPEC = importlib.util.spec_from_file_location("wm0_broker_e0", _E0_PATH)
if _E0_SPEC is None or _E0_SPEC.loader is None:
    raise RuntimeError(f"cannot load E0 evidence adapter {_E0_PATH}")
E0 = importlib.util.module_from_spec(_E0_SPEC)
sys.modules[_E0_SPEC.name] = E0
_E0_SPEC.loader.exec_module(E0)


class LaneIntegrityError(RuntimeError):
    pass


class LaneInfrastructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class EditProvenance:
    author_event_id: str

    def __post_init__(self) -> None:
        if not self.author_event_id or any(character.isspace() for character in self.author_event_id):
            raise ValueError("author_event_id must be a non-empty identifier")


@dataclass(frozen=True)
class ModelEdit:
    kind: Literal["whole_file", "function"]
    source: str
    function_name: str | None = None

    @classmethod
    def whole_file(cls, source: str) -> ModelEdit:
        return cls("whole_file", source)

    @classmethod
    def function(cls, name: str, source: str) -> ModelEdit:
        return cls("function", source, name)


@dataclass(frozen=True)
class LaneState:
    evidence_snapshot_id: str
    current_model_version_id: str | None
    resolved_prediction_scores: tuple[str, ...] = ()
    still_pending_prediction_receipts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelEvaluation:
    evaluation_id: str
    candidate_id: str
    candidate_status: CandidateStatus
    active_model_version_id: str | None
    previous_model_version_id: str | None
    source_sha256: str
    evidence_snapshot_id: str
    operational: dict[str, Any]
    initial_render: dict[str, Any] | None
    dynamics: dict[str, Any] | None
    outcome: dict[str, Any] | None
    first_divergence: dict[str, Any] | None


class ExecutableTheoryLane:
    """Deep broker: evidence installation and one atomic model-edit operation."""

    def __init__(self, root: Path, *, evaluator_timeout_seconds: float = 8.0) -> None:
        self.root = Path(root)
        self.candidates = self.root / "candidates"
        self.versions = self.root / "versions"
        self.evidence = self.root / "evidence"
        self.bundles = self.root / "evaluation-bundles"
        self.current_path = self.root / "current.json"
        self.counters_path = self.root / "counters.json"
        self.evaluator_timeout_seconds = evaluator_timeout_seconds
        for path in (self.candidates, self.versions, self.evidence, self.bundles):
            path.mkdir(parents=True, exist_ok=True)
        if not self.counters_path.exists():
            _atomic_json(self.counters_path, {"candidate": 0, "version": 0, "evaluation": 0})

    def advance(self, snapshot: Path | str) -> LaneState:
        source = Path(snapshot)
        try:
            verified = E0.load_snapshot(source)
        except Exception as error:
            raise LaneIntegrityError(str(error)) from error
        value = json.loads(source.read_text(encoding="utf-8"))
        snapshot_id = verified.snapshot_id
        self._validate_frontier(value)
        current = self._frontier()
        if current is not None and current["snapshot_id"] == snapshot_id:
            return LaneState(snapshot_id, self._current_version_id())
        destination = self.evidence / f"{snapshot_id}.json"
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise LaneIntegrityError("installed snapshot id has divergent content")
        else:
            shutil.copyfile(source, destination)
            destination.chmod(0o444)
        _atomic_json(
            self.evidence / "current.json",
            {
                "snapshot_id": snapshot_id,
                "snapshot_path": str(source.resolve()),
                "through_host_sequence": value["through_host_sequence"],
                "source_event_ids": value["source_event_ids"],
            },
        )
        return LaneState(snapshot_id, self._current_version_id())

    def read_world_model(self) -> dict[str, Any]:
        frontier = self._frontier()
        current = self._current_record()
        if current is None:
            return {
                "source": QUICK_THEORY_SOURCE,
                "source_kind": "quick_theory_template",
                "contract_kind": None,
                "quick_theory_template": QUICK_THEORY_SOURCE,
                "moore_machine_template": SEED_SOURCE,
                "current_model_version_id": None,
                "source_sha256": _digest_bytes(SEED_SOURCE.encode()),
                "installed_evidence_snapshot_id": frontier["snapshot_id"] if frontier else None,
                "evaluation_evidence_snapshot_id": None,
                "evaluation_stale": False,
                "latest_evaluation": None,
                "unresolved_shadow": False,
            }
        source = (self.versions / current["model_version_id"] / "source.py").read_text(
            encoding="utf-8"
        )
        evaluation = json.loads(
            (self.versions / current["model_version_id"] / "evaluation.json").read_text(
                encoding="utf-8"
            )
        )
        return {
            "source": source,
            "source_kind": "active_commitment",
            "contract_kind": current.get("contract_kind"),
            "quick_theory_template": QUICK_THEORY_SOURCE,
            "moore_machine_template": SEED_SOURCE,
            "current_model_version_id": current["model_version_id"],
            "source_sha256": current["source_sha256"],
            "installed_evidence_snapshot_id": frontier["snapshot_id"] if frontier else None,
            "evaluation_evidence_snapshot_id": current["evidence_snapshot_id"],
            "evaluation_stale": bool(
                frontier and frontier["snapshot_id"] != current["evidence_snapshot_id"]
            ),
            "latest_evaluation": evaluation,
            "unresolved_shadow": False,
        }

    def apply(self, edit: ModelEdit, provenance: EditProvenance) -> ModelEvaluation:
        frontier = self._frontier()
        if frontier is None:
            raise LaneIntegrityError("install evidence before applying a model edit")
        previous = self._current_version_id()
        source = self._construct_source(edit)
        candidate_id, evaluation_id = self._allocate_candidate_and_evaluation()
        candidate_dir = self.candidates / candidate_id
        candidate_dir.mkdir()
        _atomic_text(candidate_dir / "source.py", source)
        (candidate_dir / "source.py").chmod(0o444)
        source_sha = _digest_bytes(source.encode("utf-8"))
        _atomic_json(
            candidate_dir / "provenance.json",
            {
                "candidate_id": candidate_id,
                "author_event_id": provenance.author_event_id,
                "edit_kind": edit.kind,
                "function_name": edit.function_name,
                "source_sha256": source_sha,
                "evidence_snapshot_id": frontier["snapshot_id"],
                "previous_model_version_id": previous,
            },
        )
        (candidate_dir / "provenance.json").chmod(0o444)

        static_failure = self._static_validate(source)
        if static_failure is not None:
            result = self._rejection(
                evaluation_id,
                candidate_id,
                previous,
                source_sha,
                frontier["snapshot_id"],
                static_failure,
            )
            _atomic_json(candidate_dir / "evaluation.json", _plain_evaluation(result))
            (candidate_dir / "evaluation.json").chmod(0o444)
            return result

        isolated = self._evaluate_isolated(candidate_dir, frontier)
        operational = isolated["operational"]
        if operational["status"] != "valid":
            result = self._rejection(
                evaluation_id,
                candidate_id,
                previous,
                source_sha,
                frontier["snapshot_id"],
                operational,
            )
            _atomic_json(candidate_dir / "evaluation.json", _plain_evaluation(result))
            (candidate_dir / "evaluation.json").chmod(0o444)
            return result

        self._externalize_divergence_blobs(candidate_dir, isolated)
        empirical = isolated["status"]
        status: CandidateStatus = {
            "contradicted": "active_contradicted",
            "inconclusive": "active_inconclusive",
            "supported": "active_supported",
        }[empirical]
        version_id = self._allocate_version()
        result = ModelEvaluation(
            evaluation_id,
            candidate_id,
            status,
            version_id,
            previous,
            source_sha,
            frontier["snapshot_id"],
            operational,
            isolated.get("initial_render"),
            isolated.get("dynamics"),
            isolated.get("outcome"),
            isolated.get("first_divergence"),
        )
        _atomic_json(candidate_dir / "evaluation.json", _plain_evaluation(result))
        (candidate_dir / "evaluation.json").chmod(0o444)
        self._activate(version_id, source, result, provenance)
        return result

    def _validate_frontier(self, incoming: dict[str, Any]) -> None:
        current = self._frontier()
        if current is None:
            return
        old_events = current["source_event_ids"]
        new_events = incoming["source_event_ids"]
        if incoming["snapshot_id"] == current["snapshot_id"]:
            return
        if len(new_events) < len(old_events):
            raise LaneIntegrityError("cannot install an older evidence frontier")
        if len(new_events) == len(old_events):
            raise LaneIntegrityError("divergent snapshot at installed frontier")
        if new_events[: len(old_events)] != old_events:
            raise LaneIntegrityError("evidence frontier is not append-only")

    def _frontier(self) -> dict[str, Any] | None:
        path = self.evidence / "current.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _current_record(self) -> dict[str, Any] | None:
        return json.loads(self.current_path.read_text(encoding="utf-8")) if self.current_path.exists() else None

    def _current_version_id(self) -> str | None:
        current = self._current_record()
        return current["model_version_id"] if current else None

    def _construct_source(self, edit: ModelEdit) -> str:
        if edit.kind == "whole_file":
            return edit.source
        if edit.kind != "function" or not edit.function_name:
            raise ValueError("invalid model edit")
        current = self._current_record()
        if current is None:
            base = QUICK_THEORY_SOURCE if edit.function_name == "predict" else SEED_SOURCE
        else:
            base = self.read_world_model()["source"]
        try:
            tree = ast.parse(base)
        except SyntaxError:
            # The unified apply transaction will preserve this as a candidate.
            return edit.source
        target = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == edit.function_name
            ),
            None,
        )
        if target is None or target.end_lineno is None:
            return edit.source
        lines = base.splitlines(keepends=True)
        replacement = edit.source.rstrip() + "\n"
        return "".join(lines[: target.lineno - 1]) + replacement + "".join(lines[target.end_lineno :])

    def _static_validate(self, source: str) -> dict[str, Any] | None:
        if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
            return {
                "status": "invalid",
                "failure_code": "SOURCE_LIMIT",
                "message": f"candidate source exceeds {MAX_SOURCE_BYTES} bytes",
            }
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            return {
                "status": "invalid",
                "failure_code": "SYNTAX_ERROR",
                "line": error.lineno,
                "column": error.offset,
                "message": error.msg,
            }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                        return self._prohibited("PROHIBITED_IMPORT", node, alias.name)
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if node.level or root not in ALLOWED_IMPORTS:
                    return self._prohibited("PROHIBITED_IMPORT", node, node.module or "relative import")
                if any(
                    alias.name.startswith("_")
                    or alias.name in {"sys", "os", "socket", "subprocess"}
                    for alias in node.names
                ):
                    return self._prohibited(
                        "PROHIBITED_IMPORT", node, "module-internal import"
                    )
            elif isinstance(node, ast.Name) and node.id in PROHIBITED_NAMES:
                return self._prohibited("PROHIBITED_OPERATION", node, node.id)
            elif isinstance(node, ast.Attribute) and (
                node.attr.startswith("_") or node.attr in PROHIBITED_ATTRIBUTES
            ):
                return self._prohibited("PROHIBITED_OPERATION", node, node.attr)
        return None

    @staticmethod
    def _prohibited(code: str, node: ast.AST, name: str) -> dict[str, Any]:
        return {
            "status": "invalid",
            "failure_code": code,
            "line": getattr(node, "lineno", None),
            "column": getattr(node, "col_offset", None),
            "message": f"prohibited name: {name}",
        }

    def _evaluate_isolated(self, candidate_dir: Path, frontier: dict[str, Any]) -> dict[str, Any]:
        snapshot_path = Path(frontier["snapshot_path"])
        evidence = json.loads(snapshot_path.read_text(encoding="utf-8"))
        bundle = Path(tempfile.mkdtemp(prefix="bundle-", dir=self.bundles))
        try:
            shutil.copyfile(candidate_dir / "source.py", bundle / "source.py")
            _atomic_json(bundle / "evidence.json", evidence)
            blobs = bundle / "blobs"
            blobs.mkdir()
            (bundle / "tmp").mkdir()
            evidence_root = snapshot_path.parent.parent
            digests: set[str] = set()
            for transition in evidence.get("transitions", []):
                digests.add(transition["before_grid_sha256"])
                digests.add(transition["after_grid_sha256"])
            for record in json.loads(evidence["occurrence_records_json"]):
                observation = record.get("observation") or record.get("after_observation")
                if observation:
                    digests.add(observation["grid_sha256"])
            for digest in digests:
                source = evidence_root / "blobs" / f"grid-{digest}.json"
                if not source.is_file():
                    raise LaneIntegrityError(f"missing evidence grid blob {digest}")
                shutil.copyfile(source, blobs / source.name)
            for path in bundle.rglob("*"):
                if path.is_file():
                    path.chmod(0o444)
            command = [sys.executable, "-I", "-B", str(Path(__file__).with_name("world_model_evaluator.py")), str(bundle)]
            environment = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "TMPDIR": "/tmp"}
            try:
                completed = subprocess.run(
                    command,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=self.evaluator_timeout_seconds,
                    check=False,
                    close_fds=True,
                )
            except subprocess.TimeoutExpired as error:
                empirical_started = b"WM1_PHASE empirical" in (error.stderr or b"")
                if empirical_started:
                    return self._empirical_resource_failure("WALL_TIME_LIMIT")
                return {
                    "operational": {
                        "status": "invalid",
                        "failure_code": "WALL_TIME_LIMIT",
                        "message": "candidate evaluation exceeded wall-time limit",
                    }
                }
            if len(completed.stdout) > 256_000 or len(completed.stderr) > 64_000:
                return {
                    "operational": {
                        "status": "invalid",
                        "failure_code": "OUTPUT_LIMIT",
                        "message": "evaluator output exceeded limit",
                    }
                }
            try:
                result = json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                empirical_started = b"WM1_PHASE empirical" in completed.stderr
                if completed.returncode < 0:
                    if empirical_started:
                        return self._empirical_resource_failure("PROCESS_TERMINATED")
                    return {
                        "operational": {
                            "status": "invalid",
                            "failure_code": "PROCESS_TERMINATED",
                            "message": "candidate process terminated before contract evaluation",
                        }
                    }
                raise LaneInfrastructureError(
                    f"isolated evaluator returned no structured result (exit {completed.returncode})"
                ) from error
            if result.get("protocol") != EVALUATOR_PROTOCOL:
                raise LaneInfrastructureError("isolated evaluator protocol mismatch")
            return result
        finally:
            shutil.rmtree(bundle, ignore_errors=True)

    @staticmethod
    def _empirical_resource_failure(code: str) -> dict[str, Any]:
        error = {
            "source_event_id": None,
            "error_type": code,
            "message": "candidate exceeded a resource limit during empirical replay",
        }
        return {
            "protocol": EVALUATOR_PROTOCOL,
            "operational": {"status": "valid", "failure_code": None},
            "status": "contradicted",
            "initial_render": {
                "status": "inconclusive",
                "evaluated": 0,
                "matched": 0,
            },
            "dynamics": {
                "status": "contradicted",
                "evaluated": 0,
                "matched": 0,
                "changing": 0,
                "no_op": 0,
                "execution_errors": [error],
            },
            "outcome": {
                "status": "inconclusive",
                "evaluated": 0,
                "matched": 0,
                "terminal_evidence": 0,
                "false_positives": 0,
                "false_negatives": 0,
            },
            "first_divergence": {
                "source_event_id": None,
                "kind": "execution_error",
                "error_type": code,
            },
        }

    @staticmethod
    def _externalize_divergence_blobs(
        candidate_dir: Path, isolated: dict[str, Any]
    ) -> None:
        divergence = isolated.get("first_divergence")
        if not isinstance(divergence, dict):
            return
        blob_dir = candidate_dir / "evaluation-blobs"
        for name in ("observed_delta", "predicted_delta"):
            value = divergence.pop(name, None)
            if value is None:
                continue
            payload = (_canonical(value) + "\n").encode("utf-8")
            digest = _digest_bytes(payload)
            path = blob_dir / f"delta-{digest}.json"
            if not path.exists():
                _atomic_json(path, value)
                path.chmod(0o444)
            divergence[f"{name}_blob"] = {
                "sha256": digest,
                "candidate_relative_path": str(path.relative_to(candidate_dir)),
                "cell_count": value.get("cell_count"),
                "truncated": value.get("truncated"),
            }

    @staticmethod
    def _rejection(
        evaluation_id: str,
        candidate_id: str,
        previous: str | None,
        source_sha: str,
        snapshot_id: str,
        operational: dict[str, Any],
    ) -> ModelEvaluation:
        return ModelEvaluation(
            evaluation_id,
            candidate_id,
            "rejected_operationally",
            previous,
            previous,
            source_sha,
            snapshot_id,
            operational,
            None,
            None,
            None,
            None,
        )

    def _activate(
        self,
        version_id: str,
        source: str,
        evaluation: ModelEvaluation,
        provenance: EditProvenance,
    ) -> None:
        staging = self.versions / f".{version_id}.staging-{os.getpid()}"
        final = self.versions / version_id
        staging.mkdir()
        _atomic_text(staging / "source.py", source)
        _atomic_json(staging / "evaluation.json", _plain_evaluation(evaluation))
        record = {
            "model_version_id": version_id,
            "candidate_id": evaluation.candidate_id,
            "author_event_id": provenance.author_event_id,
            "evidence_snapshot_id": evaluation.evidence_snapshot_id,
            "source_sha256": evaluation.source_sha256,
            "previous_model_version_id": evaluation.previous_model_version_id,
            "evaluation_id": evaluation.evaluation_id,
            "candidate_status": evaluation.candidate_status,
            "contract_kind": evaluation.operational.get("contract_kind"),
        }
        _atomic_json(staging / "record.json", record)
        for path in staging.iterdir():
            path.chmod(0o444)
        os.replace(staging, final)
        _atomic_json(self.current_path, record)

    def _allocate_candidate_and_evaluation(self) -> tuple[str, str]:
        counters = json.loads(self.counters_path.read_text(encoding="utf-8"))
        candidate = f"candidate-{counters['candidate']:08d}"
        evaluation = f"evaluation-{counters['evaluation']:08d}"
        counters["candidate"] += 1
        counters["evaluation"] += 1
        _atomic_json(self.counters_path, counters)
        return candidate, evaluation

    def _allocate_version(self) -> str:
        counters = json.loads(self.counters_path.read_text(encoding="utf-8"))
        version = f"model-version-{counters['version']:08d}"
        counters["version"] += 1
        _atomic_json(self.counters_path, counters)
        return version


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(_canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _plain_evaluation(value: ModelEvaluation) -> dict[str, Any]:
    return asdict(value)
