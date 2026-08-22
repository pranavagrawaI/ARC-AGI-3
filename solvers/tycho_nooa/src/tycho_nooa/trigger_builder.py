"""Host-side trigger, construction, delivery, and uptake instrumentation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from nooa.unifiedllm import UnifiedLLM
from pydantic import ValidationError
from tycho.agent.dispatcher import TriggerDispatcher
from tycho.workspace.agent_tools import ToolExecutor, canonical

from tycho_nooa.builder import (
    BuilderConfig,
    BuilderRequest,
    BuilderResult,
    FocusedBuilder,
)
from tycho_nooa.deadlines import DecisionDeadline
from tycho_nooa.decision_checkpoints import DecisionCheckpointRecorder
from tycho_nooa.llm_trace import LLMTrace
from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.theory_evaluation import (
    EvidenceFrontier,
    PublicationDecision,
)
from tycho_nooa.theory_transaction import (
    TheoryEvaluator,
    TheoryTransaction,
)
from tycho_nooa.wm_probe import AST_UNCHANGED_NOTICE


@dataclass(frozen=True)
class TriggerDecision:
    fired: bool
    cause: str | None
    reason: str | None


@dataclass(frozen=True)
class BuilderExposure:
    result: BuilderResult
    recommended_action: ActionDecision | None
    recommendation_parse_error: str | None = None
    publication_decision: PublicationDecision | None = None
    publication_feedback: str | None = None

    @property
    def delivered_report(self) -> str:
        if self.publication_feedback is None:
            return self.result.report
        return f"{self.result.report}\n\n{self.publication_feedback}"


class BuilderDeadlineError(RuntimeError):
    """A bounded builder request did not produce a completed advisory result."""


class BuilderDeadlineUnavailable(BuilderDeadlineError):
    """The actor spent the builder allocation before making its request."""


class BuilderDeadlineExceeded(BuilderDeadlineError):
    """The builder was cancelled and its partial workspace transaction rolled back."""


@dataclass(frozen=True)
class RecommendationParse:
    action: ActionDecision | None
    error: str | None


@dataclass(frozen=True)
class RetainedCandidate:
    """One noncanonical model body preserved across transaction rollback."""

    record_id: str
    model_sha256: str
    model_path: str
    record_path: str
    contemporaneous_score: dict[str, Any]


def _snapshot_workspace(workspace_dir: Path) -> dict[str, bytes]:
    """Capture the builder transaction without duplicating volatile bytecode."""
    return {
        path.relative_to(workspace_dir).as_posix(): path.read_bytes()
        for path in workspace_dir.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.relative_to(workspace_dir).parts
    }


def _restore_workspace(
    workspace_dir: Path,
    before: dict[str, bytes],
) -> list[str]:
    """Roll back every builder-visible file and report exactly what changed."""
    current = _snapshot_workspace(workspace_dir)
    changed = sorted(
        path
        for path in before.keys() | current.keys()
        if before.get(path) != current.get(path)
    )
    for relative in current.keys() - before.keys():
        (workspace_dir / relative).unlink()
    for relative, content in before.items():
        path = workspace_dir / relative
        if not path.is_file() or path.read_bytes() != content:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    return changed


def _rollback_with_candidate_retention(
    workspace_dir: Path,
    before: dict[str, bytes],
    partial_tool_calls: list[dict[str, Any]],
    *,
    turn: GameTurn,
    reason: str,
    focus: str | None,
) -> tuple[list[str], RetainedCandidate | None, str | None]:
    """Restore canonical state, then publish a noncanonical copy of changed theory.

    Staging happens before rollback because the candidate is otherwise destroyed.
    Publication happens after rollback so the candidate path cannot accidentally
    become canonical or be removed as part of the transaction.
    """
    staged = None
    retention_error = None
    try:
        staged = _stage_candidate(
            workspace_dir,
            before,
            partial_tool_calls,
            turn=turn,
            reason=reason,
            focus=focus,
        )
    except Exception as error:  # noqa: BLE001 - rollback must still complete
        retention_error = f"{type(error).__name__}: {error}"
    changed = _restore_workspace(workspace_dir, before)
    retained = None
    if staged is not None and retention_error is None:
        try:
            retained = _publish_candidate(workspace_dir, staged)
        except Exception as error:  # noqa: BLE001 - canonical rollback already won
            retention_error = f"{type(error).__name__}: {error}"
    return changed, retained, retention_error


def _stage_candidate(
    workspace_dir: Path,
    before: dict[str, bytes],
    partial_tool_calls: list[dict[str, Any]],
    *,
    turn: GameTurn,
    reason: str,
    focus: str | None,
) -> tuple[bytes, dict[str, Any]] | None:
    model_path = workspace_dir / "world_model.py"
    if not model_path.is_file():
        return None
    body = model_path.read_bytes()
    canonical_before = before.get("world_model.py")
    if canonical_before == body:
        return None

    model_sha = hashlib.sha256(body).hexdigest()
    frontier = _evidence_frontier(workspace_dir)
    frontier_sha = hashlib.sha256(
        json.dumps(frontier, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    canonical_before_sha = (
        hashlib.sha256(canonical_before).hexdigest()
        if canonical_before is not None
        else None
    )
    score = _contemporaneous_score(partial_tool_calls)
    source_context = {
        "source_event": "builder_deadline_exceeded",
        "level": turn.level,
        "turn_in_level": turn.turn_in_level,
        "builder_reason": reason,
        "builder_focus": focus,
        "model_sha256": model_sha,
        "canonical_before_sha256": canonical_before_sha,
        "evidence_frontier_sha256": frontier_sha,
        "contemporaneous_score": score,
    }
    source_context_sha = hashlib.sha256(
        json.dumps(source_context, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    record_id = f"{model_sha[:16]}-{source_context_sha[:16]}"
    metadata = {
        "schema_version": 1,
        "record_id": record_id,
        "authority": "retained_noncanonical_candidate",
        "publication_allowed": False,
        "planner_eligible": False,
        **source_context,
        "source_context_sha256": source_context_sha,
        "evidence_frontier": frontier,
    }
    return body, metadata


def _publish_candidate(
    workspace_dir: Path,
    staged: tuple[bytes, dict[str, Any]],
) -> RetainedCandidate:
    body, metadata = staged
    root = workspace_dir / "candidates" / "world_model"
    model_path = root / "blobs" / f"{metadata['model_sha256']}.py"
    record_path = root / "records" / f"{metadata['record_id']}.json"
    index_path = root / "index.json"

    _write_immutable(model_path, body)
    rendered = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    _write_immutable(record_path, rendered)
    index = _read_candidate_index(index_path)
    relative_record = record_path.relative_to(workspace_dir).as_posix()
    if not any(row.get("record_path") == relative_record for row in index["records"]):
        index["records"].append(
            {
                "record_id": metadata["record_id"],
                "model_sha256": metadata["model_sha256"],
                "model_path": model_path.relative_to(workspace_dir).as_posix(),
                "record_path": relative_record,
                "level": metadata["level"],
                "turn_in_level": metadata["turn_in_level"],
                "score": metadata["contemporaneous_score"],
                "publication_allowed": False,
                "planner_eligible": False,
            }
        )
        _write_json_atomic(index_path, index)

    return RetainedCandidate(
        record_id=metadata["record_id"],
        model_sha256=metadata["model_sha256"],
        model_path=model_path.relative_to(workspace_dir).as_posix(),
        record_path=relative_record,
        contemporaneous_score=metadata["contemporaneous_score"],
    )


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable candidate path changed content: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _read_candidate_index(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "records": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("records"), list):
        raise RuntimeError(f"invalid retained-candidate index: {path}")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _evidence_frontier(workspace_dir: Path) -> list[dict[str, Any]]:
    evidence = []
    # Include archived attempts as well as the current attempt. A candidate's
    # score is only interpretable against the full transition history visible
    # to the builder when it made the edit.
    for path in sorted(workspace_dir.rglob("turn_*.json")):
        if not path.is_file():
            continue
        evidence.append(
            {
                "path": path.relative_to(workspace_dir).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return evidence


_EDIT_TOOLS = frozenset({"write_file", "edit_file", "edit_function"})
_MODEL_FILES = frozenset({"world_model.py", "worldmodel.py"})

_FEEDBACK_MARKER = "[auto world-model feedback]"

_SIMULATION_ACCURACY = re.compile(
    r"(?:accepted|exact)_transition_accuracy \(simulation_accuracy\) = ([0-9.]+)"
)
_TRANSITION_REPLAY = re.compile(
    r"transition_replay = matched (\d+) / evaluated (\d+) / observed (\d+)"
)
_FIRST_DIVERGENCE = re.compile(r"^  first divergence: (.+)$", re.MULTILINE)
_LEVEL_START_ACCURACY = re.compile(r"accepted_level_start_accuracy = ([0-9.]+)")
_MODEL_LOAD_FAILURE = re.compile(r"^world_model\.py does not load: (.+)$", re.MULTILINE)
_VERIFY_CRASH = re.compile(r"^verify crashed: (.+)$", re.MULTILINE)

# The obligations the verifier prints as named statuses. A publication gate has to
# order candidates on all of them at once -- a body can hold its transition score
# while breaking level-start reconstruction, a confirmed terminal, or the planner
# contract -- so they are read together rather than per branch.
_STATUS_FIELDS = (
    "CURRENT_LEVEL_INITIAL_RENDER_STATUS",
    "ALL_LEVELS_INITIAL_RENDER_STATUS",
    "CURRENT_LEVEL_DYNAMICS_STATUS",
    "ALL_LEVELS_DYNAMICS_STATUS",
    "CURRENT_LEVEL_OUTCOME_STATUS",
    "CURRENT_LEVEL_TERMINAL_RENDER_STATUS",
    "PLANNER_STATUS",
)


def _verifier_status(result: str, field: str) -> str | None:
    match = re.search(rf"^  {re.escape(field)}: (.+)$", result, re.MULTILINE)
    return match.group(1) if match is not None else None


def _verifier_reading(result: str) -> dict[str, Any]:
    """Every typed obligation the host verifier printed for one graded body."""
    accuracy = _SIMULATION_ACCURACY.search(result)
    replay = _TRANSITION_REPLAY.search(result)
    divergence = _FIRST_DIVERGENCE.search(result)
    level_start = _LEVEL_START_ACCURACY.search(result)
    reading: dict[str, Any] = {
        "accepted_transition_accuracy": (
            float(accuracy.group(1)) if accuracy is not None else None
        ),
        "matched_transitions": int(replay.group(1)) if replay is not None else None,
        "evaluated_transitions": int(replay.group(2)) if replay is not None else None,
        "observed_transitions": int(replay.group(3)) if replay is not None else None,
        "first_divergence": divergence.group(1) if divergence is not None else None,
        "accepted_level_start_accuracy": (
            float(level_start.group(1)) if level_start is not None else None
        ),
    }
    reading.update(
        {field.lower(): _verifier_status(result, field) for field in _STATUS_FIELDS}
    )
    return reading


def _load_status(result: str) -> tuple[str, str | None]:
    """Whether the graded body was executable at all, and why not when it was not."""
    load_failure = _MODEL_LOAD_FAILURE.search(result)
    if load_failure is not None:
        return "load_failed", load_failure.group(1)
    crash = _VERIFY_CRASH.search(result)
    if crash is not None:
        return "verify_crashed", crash.group(1)
    return "loaded", None


def _contemporaneous_score(
    partial_tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score the body this pass is about to leave behind, from the evidence it saw.

    The score belongs to the last edit the verifier actually graded, which is not
    always the last edit made: the host skips the verifier when an edit leaves the
    syntax tree intact, so a pass that ends on a comment or formatting change would
    otherwise retain a candidate with no score at all. Every skipped edit between
    the graded one and the retained body is AST-preserving by construction, so that
    grade still describes this body -- and the count is reported, because the bytes
    the record commits to are not the bytes that were graded.
    """
    edits = [row for row in partial_tool_calls if _is_model_changing(row)]
    if not edits:
        return {
            "status": "unavailable",
            "reason": "no completed model-changing tool result",
        }
    call, ast_preserving_edits = _last_graded_edit(edits)
    if call is None:
        return {
            "status": "unavailable",
            "reason": (
                "every model-changing edit left the syntax tree unchanged, so this "
                "pass never graded the body it is about to retain"
            ),
            "ast_preserving_edits_since_grade": ast_preserving_edits,
        }
    provenance = _score_provenance(call, ast_preserving_edits)
    result = str(call.get("result", ""))
    if _FEEDBACK_MARKER not in result:
        return {
            "status": "unavailable",
            "reason": "completed model-changing tool result had no verifier metrics",
            **provenance,
        }
    load_status, load_error = _load_status(result)
    reading = _verifier_reading(result)
    if load_status != "loaded":
        return {
            "status": "not_evaluable",
            "reason": f"{load_status}: {load_error}",
            "source": "host auto world-model feedback",
            "load_status": load_status,
            **provenance,
            **reading,
        }
    if reading["evaluated_transitions"] is None and (
        reading["accepted_transition_accuracy"] is None
    ):
        return {
            "status": "assessed_without_transition_score",
            "reason": (
                "host verifier ran but no graded transition score was available"
            ),
            "source": "host auto world-model feedback",
            "load_status": load_status,
            **provenance,
            **reading,
        }
    return {
        "status": "measured",
        "source": "host auto world-model feedback",
        "load_status": load_status,
        **provenance,
        **reading,
    }


def _score_provenance(
    call: dict[str, Any], ast_preserving_edits: int
) -> dict[str, Any]:
    provenance: dict[str, Any] = {"tool_call_index": call.get("call_index")}
    if ast_preserving_edits:
        provenance["ast_preserving_edits_since_grade"] = ast_preserving_edits
    return provenance


def _is_model_changing(row: dict[str, Any]) -> bool:
    if row.get("status") != "complete":
        return False
    if canonical(str(row.get("name", ""))) not in _EDIT_TOOLS:
        return False
    path = str((row.get("arguments") or {}).get("path", ""))
    return path.replace("\\", "/").rsplit("/", 1)[-1].lower() in _MODEL_FILES


def _last_graded_edit(
    edits: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, int]:
    """The most recent edit the verifier graded, and how many it skipped after it."""
    skipped = 0
    for row in reversed(edits):
        if AST_UNCHANGED_NOTICE in str(row.get("result", "")):
            skipped += 1
            continue
        return row, skipped
    return None, skipped


class TychoTransitionTrigger:
    """Adapt Tycho's exact TriggerDispatcher/wm_signal path to one transition."""

    def __init__(self) -> None:
        self._dispatcher = TriggerDispatcher()

    def evaluate(self, level: int, executor: ToolExecutor) -> TriggerDecision:
        fired = self._dispatcher.should_fire(level, executor)
        if not fired:
            return TriggerDecision(fired=False, cause=None, reason=None)
        reason = self._dispatcher.trigger_reason(level)
        cause = getattr(self._dispatcher, "_last_cause", None)
        self._dispatcher.note_fired(level)
        return TriggerDecision(fired=True, cause=cause, reason=reason)

    def evaluate_level_start(self, level: int) -> TriggerDecision:
        """LEVEL_START fires unconditionally (no divergence probe), mirroring
        Tycho's own `elif ... turn_in_level == 0 and level > 0` branch, which
        never calls should_fire() for this cause."""
        reason = self._dispatcher.level_start_reason(level)
        self._dispatcher.note_fired(level)
        return TriggerDecision(fired=True, cause="level_start", reason=reason)


class TriggerBuilderLoop:
    """Run a fresh builder after divergent transitions, before the actor acts."""

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        executor: ToolExecutor,
        trace_path: Path | None,
        builder_tool_trace_path: Path | None,
        builder_config: BuilderConfig | None = None,
        llm_trace: LLMTrace | None = None,
        decision_checkpoints: DecisionCheckpointRecorder | None = None,
        builder_executor: ToolExecutor | None = None,
        theory_evaluator: TheoryEvaluator | None = None,
    ) -> None:
        self._llm = llm
        self._executor = executor
        self._builder_executor = builder_executor or executor
        self._theory_evaluator = theory_evaluator
        self._trace_path = trace_path
        self._builder_tool_trace_path = builder_tool_trace_path
        self._builder_config = builder_config
        self._llm_trace = llm_trace
        self._decision_checkpoints = decision_checkpoints
        self._trigger = TychoTransitionTrigger()
        self.builder: FocusedBuilder | None = None
        # -1 never equals a real level, so a genuine level-start check at level 0
        # (guarded out below regardless) never depends on this sentinel.
        self._last_level = -1
        self._pending_outcome_evidence: str | None = None
        self._pending_outcome_level: int | None = None
        self._resolved_outcome_levels: set[int] = set()

    async def before_actor(
        self,
        turn: GameTurn,
        *,
        deadline: DecisionDeadline | None = None,
    ) -> BuilderExposure | None:
        """Evaluate the recorded transition into this turn, then build if signaled.

        arc-host's `full_reset` resets `turn_in_level` to 0 WITHOUT changing
        `level` (runner.py), so `turn_in_level == 0` alone cannot distinguish a
        genuine level advance from a same-level restart — Tycho's own three
        `turn_in_level = 0` call sites have the identical ambiguity, disambiguated
        there only by if/elif branch priority. `_last_level`, updated
        unconditionally before either branch below, plays that same role here:
        only a level that actually differs from the immediately preceding turn's
        level counts as a level-start.
        """
        triggered = self._evaluate_trigger(turn)
        if triggered is None:
            return None
        decision, mechanism = triggered

        self._write_trace(
            {
                "event": "mechanical_trigger",
                "mechanism": mechanism,
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "fired": decision.fired,
                "cause": decision.cause,
                "reason": decision.reason,
            }
        )
        if not decision.fired:
            return None

        self._capture_outcome_evidence(turn, decision)
        try:
            return await self._invoke_builder_before_actor(
                turn,
                decision.reason or "",
                deadline,
            )
        except BuilderDeadlineError:
            # Automatic construction is advisory. Its trace remains evidence, but
            # exhausting its allocation must not consume the actor's reserved turn.
            return None

    async def actor_requested(
        self,
        turn: GameTurn,
        reason: str,
        *,
        deadline: DecisionDeadline | None = None,
        focus: str | None = None,
    ) -> BuilderExposure:
        """Run the same fresh builder because the actor identified a need."""
        self._write_trace(
            {
                "event": "actor_request",
                "mechanism": "tycho orchestrator invoke_builder",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "reason": reason,
                "focus": focus,
            }
        )
        return await self._invoke_builder_before_actor(
            turn,
            reason,
            deadline,
            focus=focus,
        )

    async def _invoke_builder_before_actor(
        self,
        turn: GameTurn,
        reason: str,
        deadline: DecisionDeadline | None,
        *,
        focus: str | None = None,
    ) -> BuilderExposure:
        if deadline is None:
            return await self._invoke_builder(turn, reason, focus=focus)

        builder_seconds = deadline.builder_seconds_remaining(
            objective=focus == "objective"
        )
        trace = {
            "level": turn.level,
            "turn_in_level": turn.turn_in_level,
            "builder_budget_seconds": builder_seconds,
            "builder_focus": focus,
            **deadline.trace_fields(),
        }
        if builder_seconds <= 0:
            self._write_trace({"event": "builder_deadline_skipped", **trace})
            raise BuilderDeadlineUnavailable(
                "builder request not started: no allocated builder time remained"
            )
        workspace_before = _snapshot_workspace(self._executor.ws.dir)
        builder_deadline_at = monotonic() + builder_seconds
        try:
            async with asyncio.timeout(builder_seconds):
                return await self._invoke_builder(
                    turn,
                    reason,
                    focus=focus,
                    deadline_at=builder_deadline_at,
                )
        except TimeoutError as error:
            partial_tool_calls = (
                self.builder.completed_tool_calls() if self.builder is not None else []
            )
            if self._theory_evaluator is None:
                (
                    rolled_back_paths,
                    retained_candidate,
                    candidate_retention_error,
                ) = _rollback_with_candidate_retention(
                    self._executor.ws.dir,
                    workspace_before,
                    partial_tool_calls,
                    turn=turn,
                    reason=reason,
                    focus=focus,
                )
            else:
                # TheoryTransaction already restored canonical executable theory.
                # Advisory notes and reports intentionally survive the deadline.
                rolled_back_paths = []
                retained_candidate = None
                candidate_retention_error = None
            self._write_trace(
                {
                    "event": "builder_deadline_exceeded",
                    "partial_tool_calls": len(partial_tool_calls),
                    "rolled_back_paths": rolled_back_paths,
                    "retained_candidate": (
                        {
                            "record_id": retained_candidate.record_id,
                            "model_sha256": retained_candidate.model_sha256,
                            "model_path": retained_candidate.model_path,
                            "record_path": retained_candidate.record_path,
                            "contemporaneous_score": (
                                retained_candidate.contemporaneous_score
                            ),
                            "publication_allowed": False,
                            "planner_eligible": False,
                        }
                        if retained_candidate is not None
                        else None
                    ),
                    "candidate_retention_error": candidate_retention_error,
                    **trace,
                }
            )
            noun = "call" if len(partial_tool_calls) == 1 else "calls"
            retention = (
                f"; retained noncanonical candidate {retained_candidate.record_id}"
                if retained_candidate is not None
                else ""
            )
            raise BuilderDeadlineExceeded(
                "builder deadline exceeded after "
                f"{len(partial_tool_calls)} completed tool {noun}; partial workspace "
                f"changes were rolled back and no builder report is valid{retention}"
            ) from error

    async def after_level_complete(
        self, turn: GameTurn, outcome_evidence: str
    ) -> BuilderExposure | None:
        """Repair the outcome contract while the winning transition is current.

        A level may complete on the final permitted action, in which case no
        next-level actor turn exists to fire the ordinary level-start builder.
        The host's action-result callback is therefore the only reliable place
        to make this mandatory checkpoint independent of another decision.

        Builder failure remains recorded and the evidence stays pending for a
        later level-start retry; it must not erase an already observed win.
        """
        self._pending_outcome_evidence = outcome_evidence
        self._pending_outcome_level = turn.level
        self._write_trace(
            {
                "event": "mechanical_trigger",
                "mechanism": "arc-host action_result (mandatory outcome checkpoint)",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "fired": True,
                "cause": "level_complete",
                "reason": "repair host-confirmed completed-level outcome before continuing",
            }
        )
        try:
            return await self._invoke_builder(
                turn,
                "repair host-confirmed completed-level outcome before continuing",
            )
        except Exception:  # noqa: BLE001 - preserve the completed game outcome
            return None

    async def _invoke_builder(
        self,
        turn: GameTurn,
        reason: str,
        *,
        focus: str | None = None,
        deadline_at: float | None = None,
    ) -> BuilderExposure:
        transaction = (
            TheoryTransaction.begin(
                workspace=self._builder_executor.ws.dir,
                evaluator=self._theory_evaluator,
                frontier=EvidenceFrontier.freeze(self._builder_executor.ws.dir),
                cause=reason,
                focus=focus,
            )
            if self._theory_evaluator is not None
            else None
        )
        self.builder = self._new_builder(transaction)
        try:
            result = await self.builder.build(
                BuilderRequest(
                    turn=turn,
                    reason=reason,
                    outcome_evidence=self._pending_outcome_evidence,
                    focus=focus,
                    deadline_at=deadline_at,
                )
            )
        except BaseException as error:
            if transaction is not None:
                transaction.abort(f"{type(error).__name__}: {error}")
            if isinstance(error, Exception):
                self._record_builder_error(turn, error)
            raise
        if result.outcome_checkpoint_resolved:
            if self._pending_outcome_level is not None:
                self._resolved_outcome_levels.add(self._pending_outcome_level)
            self._pending_outcome_evidence = None
            self._pending_outcome_level = None
        recommendation = _recommended_action(result.report, turn.available_actions)
        publication_decision = transaction.finish() if transaction is not None else None
        publication_feedback = (
            _publication_feedback(publication_decision)
            if publication_decision is not None
            else None
        )
        self._record_builder_result(turn, result, recommendation)
        return BuilderExposure(
            result=result,
            recommended_action=recommendation.action,
            recommendation_parse_error=recommendation.error,
            publication_decision=publication_decision,
            publication_feedback=publication_feedback,
        )

    def _evaluate_trigger(self, turn: GameTurn) -> tuple[TriggerDecision, str] | None:
        last_level = self._last_level
        self._last_level = turn.level
        if turn.turn_in_level == 0 and turn.level > 0 and turn.level != last_level:
            return (
                self._trigger.evaluate_level_start(turn.level),
                "tycho TriggerDispatcher.level_start_reason (unconditional)",
            )
        if turn.turn_in_level <= 0 or not turn.previous_action:
            return None
        mechanism = "tycho TriggerDispatcher.should_fire -> wm_signal"
        try:
            return self._trigger.evaluate(turn.level, self._executor), mechanism
        except Exception as error:  # noqa: BLE001 - an unanswerable probe is no signal
            # wm_signal runs verify.py in Tycho's sandbox. An unavailable probe is
            # absence of signal, not a reason to end the episode; the builder itself
            # still fails loudly because unverifiable model edits are not evidence.
            self._write_trace(
                {
                    "event": "trigger_error",
                    "mechanism": mechanism,
                    "level": turn.level,
                    "turn_in_level": turn.turn_in_level,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            return None

    def _capture_outcome_evidence(
        self, turn: GameTurn, decision: TriggerDecision
    ) -> None:
        if decision.cause != "level_start":
            return
        completed_level = turn.level - 1
        if completed_level in self._resolved_outcome_levels:
            return
        evidence = _level_complete_evidence(self._executor.ws.dir, completed_level)
        if evidence is not None:
            self._pending_outcome_evidence = evidence
            self._pending_outcome_level = completed_level

    def _new_builder(
        self, transaction: TheoryTransaction | None = None
    ) -> FocusedBuilder:
        # A new Agent per fire is the NOOA equivalent of Tycho's local builder
        # transcript: same LLM/workspace/verifier, but no actor-history leakage.
        builder = FocusedBuilder(
            llm=self._llm,
            executor=self._builder_executor,
            trace_path=self._builder_tool_trace_path,
            config=self._builder_config,
            theory_transaction=transaction,
        )
        # Each fresh agent has a fresh event stream; attach the run-wide trace now.
        if self._llm_trace is not None:
            self._llm_trace.attach(builder.event_manager, source="builder")
        if (
            self._decision_checkpoints is not None
            and builder.event_manager is not None
        ):
            self._decision_checkpoints.attach(builder.event_manager, source="builder")
        return builder

    def _record_builder_error(self, turn: GameTurn, error: Exception) -> None:
        assert self.builder is not None
        self._write_trace(
            {
                "event": "builder_error",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "builder_agent_id": self.builder.agent_id,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )

    def _record_builder_result(
        self,
        turn: GameTurn,
        result: BuilderResult,
        recommendation: RecommendationParse,
    ) -> None:
        assert self.builder is not None
        self._write_trace(
            {
                "event": "builder_result",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "builder_agent_id": self.builder.agent_id,
                "report_fresh": result.report_fresh,
                "report_forced": result.report_forced,
                "world_model_checkpoint_forced": result.world_model_checkpoint_forced,
                "outcome_checkpoint_required": result.outcome_checkpoint_required,
                "outcome_checkpoint_resolved": result.outcome_checkpoint_resolved,
                "recommended_action": (
                    recommendation.action.model_dump()
                    if recommendation.action is not None
                    else None
                ),
                "recommendation_parse_error": recommendation.error,
                "artifact_hashes": result.artifact_hashes,
                "tool_calls": [
                    {"name": call["name"], "status": call["status"]}
                    for call in result.tool_calls
                ],
            }
        )

    def after_actor(
        self,
        turn: GameTurn,
        exposure: BuilderExposure | None,
        decision: ActionDecision,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """Record syntactic follow/test evidence without guessing at model intent."""
        if exposure is None:
            return
        tested_by = [
            call["name"]
            for call in tool_calls
            if _tests_constructed_result(call["name"], call["arguments"])
        ]
        recommended = exposure.recommended_action
        self._write_trace(
            {
                "event": "actor_uptake",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "actor_action": decision.model_dump(),
                "recommended_action": (
                    recommended.model_dump() if recommended is not None else None
                ),
                "followed_recommended_action": (
                    decision == recommended if recommended is not None else None
                ),
                "tested_report": bool(tested_by),
                "test_evidence": tested_by,
            }
        )

    def after_actor_error(
        self,
        turn: GameTurn,
        exposure: BuilderExposure | None,
        error: Exception,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """Make failed post-report actor turns visible rather than silently incomplete."""
        if exposure is None:
            return
        self._write_trace(
            {
                "event": "actor_uptake",
                "level": turn.level,
                "turn_in_level": turn.turn_in_level,
                "actor_action": None,
                "recommended_action": (
                    exposure.recommended_action.model_dump()
                    if exposure.recommended_action is not None
                    else None
                ),
                "followed_recommended_action": None,
                "tested_report": any(
                    _tests_constructed_result(call["name"], call["arguments"])
                    for call in tool_calls
                ),
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )

    def _write_trace(self, record: dict[str, Any]) -> None:
        if self._trace_path is None:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps(record, separators=(",", ":")) + "\n")


def _publication_feedback(decision: PublicationDecision) -> str:
    lines = [
        "[host theory publication decision — authoritative]",
        f"disposition: {decision.disposition.value}",
    ]
    if decision.gained:
        lines.append("gained obligations:")
        lines.extend(f"- {item}" for item in decision.gained)
    if decision.lost:
        lines.append("lost obligations:")
        lines.extend(f"- {item}" for item in decision.lost)
    if decision.inconclusive:
        lines.append("inconclusive obligations:")
        lines.extend(f"- {item}" for item in decision.inconclusive)
    lines.append(f"reason: {decision.reason}")
    return "\n".join(lines)


def _level_complete_evidence(workspace_dir: Path, level: int) -> str | None:
    path = workspace_dir / f"level_{level}" / "level_complete.json"
    if not path.is_file():
        return None
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if evidence.get("source") != "arc-host action_result":
        return None
    if evidence.get("outcome") != "level_complete":
        return None
    return json.dumps(evidence, separators=(",", ":"), sort_keys=True)


_RECOMMENDED_ACTION = re.compile(
    r"^recommended_action:\s*([A-Za-z0-9_]+)"
    r"(?:\(\s*(?:row\s*=\s*)?(\d+)\s*,\s*(?:col\s*=\s*)?(\d+)\s*\))?",
    re.MULTILINE,
)


def _recommended_action(
    report: str, available_actions: list[str]
) -> RecommendationParse:
    match = _RECOMMENDED_ACTION.search(report)
    if match is None or match.group(1).lower() == "none":
        return RecommendationParse(action=None, error=None)
    action = match.group(1).upper()
    if action not in available_actions:
        return RecommendationParse(
            action=None,
            error=(
                f"recommended_action {action!r} is not available; "
                f"available actions: {available_actions}"
            ),
        )
    try:
        decision = ActionDecision(
            action=action,
            row=int(match.group(2)) if match.group(2) is not None else None,
            col=int(match.group(3)) if match.group(3) is not None else None,
        )
    except ValidationError as error:
        return RecommendationParse(action=None, error=str(error))
    return RecommendationParse(action=decision, error=None)


def _tests_constructed_result(name: str, arguments: dict[str, Any]) -> bool:
    canonical_name = canonical(name)
    if canonical_name == "read_file":
        return str(arguments.get("path", "")) in {
            "notes/world_model_report.md",
            "notes/world_model.md",
            "world_model.py",
        }
    if canonical_name != "run_python":
        return False
    code = str(arguments.get("code", "")).lower()
    return any(
        marker in code for marker in ("world_model", "wmlib", "verify.py", "plan.py")
    )
