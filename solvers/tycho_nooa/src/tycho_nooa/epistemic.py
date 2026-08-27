"""Typed, model-authored objective hypotheses over host-registered evidence."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal

from nooa.unifiedllm import Tool
from pydantic import BaseModel, ConfigDict, Field

from tycho_nooa.models import GameTurn

TYPED_GOAL_PROMPT_BRIEF = """Typed objective discipline (named local treatment):
- When public evidence suggests an objective but does not prove it, propose a falsifiable candidate goal with `propose_goal`; cite the recorded `level_N/turn_NNN.txt` or `.json` evidence and name an observation that would falsify it.
- Keep candidate goals separate from canonical `outcome()`. Selection means worth testing, not true.
- A proposal creates a mandatory review loop. Immediately call `request_builder` with `focus="objective"`, the same authoritative evidence references, and its `candidate_goal_id`; do not take an environment action first.
- After the structured objective review returns, call `update_goal` to select, reject, or retain the proposal with a note that answers the review. An environment action is unavailable until this review is acknowledged.
- Later decisions inline the live ledger: any required next step, plus each unresolved or selected candidate with its last review. Selection means worth testing, not a write to `outcome()`.
- Use `request_builder` for every focused construction request in this treatment. The legacy untyped builder interface is intentionally unavailable.
"""


class CandidateGoalStatus(str, Enum):
    """Lifecycle states that do not imply executable-model truth."""

    PROPOSED = "proposed"
    SELECTED = "selected"
    REJECTED = "rejected"


class ObjectiveAssessment(str, Enum):
    """Builder evidence about a candidate, never an automatic actor decision."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


class CandidateGoal(BaseModel):
    """One falsifiable objective hypothesis, separate from ``outcome()``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    predicate: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    falsifier: str = Field(min_length=1)
    status: CandidateGoalStatus = CandidateGoalStatus.PROPOSED
    note: str = ""
    created_level: int = Field(ge=0)
    created_turn: int = Field(ge=0)
    updated_level: int = Field(ge=0)
    updated_turn: int = Field(ge=0)
    revision: int = Field(ge=1)
    resolved_review_id: str | None = None


class ObjectiveBuilderReview(BaseModel):
    """One durable, evidence-linked result from objective-focused construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    candidate_goal_id: str
    candidate_goal_revision: int = Field(ge=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    assessment: ObjectiveAssessment
    reason: str = ""
    recommended_route: str = ""
    recommended_action: str = ""
    report: str
    created_level: int = Field(ge=0)
    created_turn: int = Field(ge=0)


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProposeGoalArguments(_Arguments):
    predicate: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    falsifier: str = Field(min_length=1)


class UpdateGoalArguments(_Arguments):
    goal_id: str
    status: CandidateGoalStatus
    evidence_refs: list[str] = Field(default_factory=list)
    note: str = ""


class ListCandidateGoalsArguments(_Arguments):
    pass


class RequestBuilderArguments(_Arguments):
    focus: Literal["dynamics", "objective", "verification", "planning"]
    evidence_refs: list[str] = Field(min_length=1)
    candidate_goal_id: str | None = None


class EvidenceCommitment(BaseModel):
    """Content identity for one host-registered public evidence file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @classmethod
    def from_path(cls, path: Path) -> EvidenceCommitment:
        payload = path.read_bytes()
        return cls(
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )


class GoalLedger:
    """Durable candidate-goal state behind a small typed interface.

    The host registers evidence paths; model-facing methods may only cite those
    paths. Updating this ledger never writes the executable world model.
    """

    DIRECTORY = "epistemic"
    SNAPSHOT = "candidate-goals.json"
    EVENTS = "candidate-goal-events.jsonl"
    REVIEWS = "objective-builder-reviews.jsonl"
    EVIDENCE_INDEX = "authoritative-evidence.json"

    def __init__(self, workspace_dir: Path) -> None:
        self._workspace_dir = workspace_dir.resolve()
        self._directory = self._workspace_dir / self.DIRECTORY
        self._snapshot_path = self._directory / self.SNAPSHOT
        self._events_path = self._directory / self.EVENTS
        self._reviews_path = self._directory / self.REVIEWS
        self._evidence_path = self._directory / self.EVIDENCE_INDEX
        self._goals = self._load_goals()
        self._reviews = self._load_reviews()
        self._evidence_refs = self._load_evidence_refs()

    def register_authoritative_evidence(self, relative_path: str) -> None:
        """Bind a host-owned workspace artifact to its current content."""
        normalized = self._normalized_workspace_file(relative_path)
        if not normalized.parts or not normalized.parts[0].startswith("level_"):
            raise ValueError(
                f"{relative_path!r} is not authoritative public evidence"
            )
        ref = normalized.as_posix()
        self._evidence_refs[ref] = EvidenceCommitment.from_path(
            self._workspace_dir / normalized
        )
        self._write_json(
            self._evidence_path,
            {
                "schema_version": 1,
                "files": {
                    path: commitment.model_dump(mode="json")
                    for path, commitment in sorted(self._evidence_refs.items())
                },
            },
        )

    def propose_goal(
        self,
        *,
        predicate: str,
        evidence_refs: list[str],
        falsifier: str,
        level: int,
        turn_in_level: int,
    ) -> CandidateGoal:
        refs = self.validate_evidence_refs(evidence_refs)
        goal = CandidateGoal(
            id=f"goal-{len(self._goals) + 1:04d}",
            predicate=predicate,
            evidence_refs=refs,
            falsifier=falsifier,
            created_level=level,
            created_turn=turn_in_level,
            updated_level=level,
            updated_turn=turn_in_level,
            revision=1,
        )
        self._goals[goal.id] = goal
        self._persist("goal_proposed", goal)
        return goal

    def update_goal(
        self,
        goal_id: str,
        *,
        status: CandidateGoalStatus,
        evidence_refs: list[str],
        note: str,
        level: int,
        turn_in_level: int,
        resolved_review_id: str | None = None,
    ) -> CandidateGoal:
        current = self.get(goal_id)
        if status is CandidateGoalStatus.SELECTED:
            selected = [
                goal.id
                for goal in self._goals.values()
                if goal.status is CandidateGoalStatus.SELECTED and goal.id != goal_id
            ]
            if selected:
                raise ValueError(
                    f"candidate goal {selected[0]} is already selected; revise it first"
                )
        combined = list(dict.fromkeys([*current.evidence_refs, *evidence_refs]))
        refs = self.validate_evidence_refs(combined)
        if not refs:
            raise ValueError("candidate goals require authoritative public evidence")
        updated = current.model_copy(
            update={
                "evidence_refs": refs,
                "status": status,
                "note": note,
                "updated_level": level,
                "updated_turn": turn_in_level,
                "revision": current.revision + 1,
                "resolved_review_id": resolved_review_id,
            }
        )
        self._goals[goal_id] = updated
        self._persist("goal_updated", updated)
        return updated

    def get(self, goal_id: str) -> CandidateGoal:
        try:
            return self._goals[goal_id]
        except KeyError as error:
            raise ValueError(f"unknown candidate goal {goal_id!r}") from error

    def list_goals(self) -> tuple[CandidateGoal, ...]:
        return tuple(self._goals.values())

    def record_builder_review(
        self,
        *,
        candidate_goal_id: str,
        evidence_refs: list[str],
        report: str,
        level: int,
        turn_in_level: int,
        unavailable: bool = False,
    ) -> ObjectiveBuilderReview:
        candidate = self.get(candidate_goal_id)
        refs = self.validate_evidence_refs(evidence_refs)
        review = ObjectiveBuilderReview(
            id=f"review-{len(self._reviews) + 1:04d}",
            candidate_goal_id=candidate.id,
            candidate_goal_revision=candidate.revision,
            evidence_refs=refs,
            assessment=(
                ObjectiveAssessment.UNAVAILABLE
                if unavailable
                else _objective_assessment(report)
            ),
            reason=_report_value(report, "candidate_goal_reason"),
            recommended_route=_report_value(report, "candidate_goal_route"),
            recommended_action=_report_value(report, "recommended_action"),
            report=report,
            created_level=level,
            created_turn=turn_in_level,
        )
        self._reviews.append(review)
        self._directory.mkdir(parents=True, exist_ok=True)
        with self._reviews_path.open("a", encoding="utf-8") as handle:
            handle.write(review.model_dump_json() + "\n")
        return review

    def latest_review(self, goal_id: str) -> ObjectiveBuilderReview | None:
        self.get(goal_id)
        return next(
            (
                review
                for review in reversed(self._reviews)
                if review.candidate_goal_id == goal_id
            ),
            None,
        )

    def pending_review(self, goal_id: str) -> ObjectiveBuilderReview | None:
        goal = self.get(goal_id)
        review = self.latest_review(goal_id)
        if review is None or review.id == goal.resolved_review_id:
            return None
        return review

    def validate_evidence_refs(self, refs: list[str]) -> tuple[str, ...]:
        if not refs:
            raise ValueError("candidate goals require authoritative public evidence")
        normalized: list[str] = []
        for ref in refs:
            path = self._normalized_workspace_file(ref)
            canonical = path.as_posix()
            commitment = self._evidence_refs.get(canonical)
            if commitment is None:
                raise ValueError(f"{ref!r} is not authoritative public evidence")
            observed = EvidenceCommitment.from_path(self._workspace_dir / path)
            if observed != commitment:
                raise ValueError(
                    f"authoritative public evidence changed after registration: {ref!r}"
                )
            normalized.append(canonical)
        return tuple(dict.fromkeys(normalized))

    def _normalized_workspace_file(self, relative_path: str) -> Path:
        candidate = (self._workspace_dir / relative_path).resolve()
        try:
            relative = candidate.relative_to(self._workspace_dir)
        except ValueError as error:
            raise ValueError(
                f"{relative_path!r} is not authoritative public evidence"
            ) from error
        if not candidate.is_file():
            raise ValueError(f"{relative_path!r} is not authoritative public evidence")
        return relative

    def _load_goals(self) -> dict[str, CandidateGoal]:
        if not self._snapshot_path.is_file():
            return {}
        values = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
        goals = [CandidateGoal.model_validate(value) for value in values]
        return {goal.id: goal for goal in goals}

    def _load_evidence_refs(self) -> dict[str, EvidenceCommitment]:
        if not self._evidence_path.is_file():
            return {}
        payload = json.loads(self._evidence_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raise TypeError(
                "legacy path-only evidence index has no content identity; "
                "re-register evidence from host-authored files"
            )
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or not isinstance(payload.get("files"), dict)
        ):
            raise ValueError(f"invalid authoritative evidence index: {self._evidence_path}")
        return {
            reference: EvidenceCommitment.model_validate(commitment)
            for reference, commitment in payload["files"].items()
        }

    def _load_reviews(self) -> list[ObjectiveBuilderReview]:
        if not self._reviews_path.is_file():
            return []
        return [
            ObjectiveBuilderReview.model_validate_json(line)
            for line in self._reviews_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _persist(self, event: str, goal: CandidateGoal) -> None:
        self._write_json(
            self._snapshot_path,
            [item.model_dump(mode="json") for item in self._goals.values()],
        )
        self._directory.mkdir(parents=True, exist_ok=True)
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"event": event, "goal": goal.model_dump(mode="json")},
                    separators=(",", ":"),
                )
                + "\n"
            )

    def _write_json(self, path: Path, value: object) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


class GoalToolSurface:
    """Model-visible typed methods over candidate goals, not the workspace."""

    TOOL_NAMES: ClassVar[frozenset[str]] = frozenset({
        "propose_goal",
        "update_goal",
        "list_candidate_goals",
        "request_builder",
    })

    def __init__(self, ledger: GoalLedger) -> None:
        self._ledger = ledger
        self._turn: GameTurn | None = None

    def begin_turn(self, turn: GameTurn) -> None:
        self._turn = turn

    def native_tools(self) -> list[Tool]:
        return [
            Tool(
                name="propose_goal",
                description=(
                    "Record a falsifiable candidate objective supported by host-owned "
                    "public evidence. This does not change world_model.py or outcome()."
                ),
                callable=self.propose_goal,
                parameters_model=ProposeGoalArguments,
            ),
            Tool(
                name="update_goal",
                description=(
                    "Select or reject a candidate objective after citing new public "
                    "evidence. Selection still does not make it canonical."
                ),
                callable=self.update_goal,
                parameters_model=UpdateGoalArguments,
            ),
            Tool(
                name="list_candidate_goals",
                description="Inspect all durable candidate objectives and their status.",
                callable=self.list_candidate_goals,
                parameters_model=ListCandidateGoalsArguments,
            ),
            Tool(
                name="request_builder",
                description=(
                    "Ask a fresh advisory builder to address one named epistemic gap "
                    "using validated public evidence references and, optionally, one "
                    "candidate goal. The builder cannot act in the environment."
                ),
                callable=self.request_builder,
                parameters_model=RequestBuilderArguments,
            ),
        ]

    def dispatch(self, tool: Tool, arguments: dict[str, Any]) -> str:
        if tool.name == "request_builder":
            raise RuntimeError("request_builder is implemented asynchronously by the agent")
        model = tool.parameters_model
        assert model is not None
        validated = model.model_validate(arguments)
        result = tool.callable(**validated.model_dump(mode="python"))
        return str(result)

    def propose_goal(
        self,
        predicate: str,
        evidence_refs: list[str],
        falsifier: str,
    ) -> str:
        turn = self._require_turn()
        goal = self._ledger.propose_goal(
            predicate=predicate,
            evidence_refs=evidence_refs,
            falsifier=falsifier,
            level=turn.level,
            turn_in_level=turn.turn_in_level,
        )
        return goal.model_dump_json()

    def update_goal(
        self,
        goal_id: str,
        status: CandidateGoalStatus,
        evidence_refs: list[str],
        note: str,
    ) -> str:
        turn = self._require_turn()
        review = self._ledger.pending_review(goal_id)
        if review is None:
            raise ValueError(
                f"candidate goal {goal_id} must receive request_builder objective "
                "evidence before update_goal"
            )
        goal = self._ledger.update_goal(
            goal_id,
            status=status,
            evidence_refs=evidence_refs,
            note=note,
            level=turn.level,
            turn_in_level=turn.turn_in_level,
            resolved_review_id=review.id,
        )
        return goal.model_dump_json()

    def list_candidate_goals(self) -> str:
        return json.dumps(
            [goal.model_dump(mode="json") for goal in self._ledger.list_goals()],
            separators=(",", ":"),
        )

    def request_builder(self, **_: Any) -> str:
        raise RuntimeError("request_builder is implemented asynchronously by the agent")

    def builder_request(self, arguments: RequestBuilderArguments) -> tuple[str, str]:
        refs = self._ledger.validate_evidence_refs(arguments.evidence_refs)
        if arguments.focus == "objective" and arguments.candidate_goal_id is None:
            raise ValueError(
                "objective-focused request_builder requires candidate_goal_id"
            )
        candidate: CandidateGoal | None = None
        if arguments.candidate_goal_id is not None:
            candidate = self._ledger.get(arguments.candidate_goal_id)
            pending = self._ledger.pending_review(candidate.id)
            if pending is not None:
                raise ValueError(
                    f"candidate goal {candidate.id} has unresolved review {pending.id}; "
                    "call update_goal before requesting another builder"
                )
        payload = {
            "focus": arguments.focus,
            "evidence_refs": refs,
            "candidate_goal": (
                candidate.model_dump(mode="json") if candidate is not None else None
            ),
            "constraint": (
                "Candidate goals are model-authored hypotheses. Independently inspect "
                "the cited evidence before changing canonical outcome()."
            ),
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True), arguments.focus

    def record_builder_review(
        self,
        arguments: RequestBuilderArguments,
        *,
        report: str,
        unavailable: bool = False,
    ) -> ObjectiveBuilderReview:
        turn = self._require_turn()
        if arguments.focus != "objective" or arguments.candidate_goal_id is None:
            raise ValueError("only an objective candidate request produces a goal review")
        return self._ledger.record_builder_review(
            candidate_goal_id=arguments.candidate_goal_id,
            evidence_refs=arguments.evidence_refs,
            report=report,
            level=turn.level,
            turn_in_level=turn.turn_in_level,
            unavailable=unavailable,
        )

    def required_before_action(self) -> str | None:
        """Return the typed lifecycle edge the actor must close before commitment."""
        for goal in self._ledger.list_goals():
            pending = self._ledger.pending_review(goal.id)
            if goal.status is CandidateGoalStatus.REJECTED and pending is None:
                continue
            review = self._ledger.latest_review(goal.id)
            try:
                self._ledger.validate_evidence_refs(list(goal.evidence_refs))
                if review is not None:
                    self._ledger.validate_evidence_refs(list(review.evidence_refs))
            except ValueError as error:
                return f"candidate goal {goal.id} has stale evidence: {error}"
            if pending is not None:
                return (
                    f"candidate goal {goal.id} has objective review {pending.id}; "
                    "call update_goal to select, reject, or retain it before take_action"
                )
            if (
                goal.status is CandidateGoalStatus.PROPOSED
                and self._ledger.latest_review(goal.id) is None
            ):
                return (
                    f"candidate goal {goal.id} has not been independently reviewed; "
                    "call request_builder with focus='objective' before take_action"
                )
        return None

    def observation_advisory(self) -> list[str]:
        """Compact live ledger for the next Actor observation; never canonical."""
        goals = self._ledger.list_goals()
        required = self.required_before_action()
        visible = [
            goal
            for goal in goals
            if goal.status is not CandidateGoalStatus.REJECTED
            or self._ledger.pending_review(goal.id) is not None
        ]
        if required is None and not visible:
            return []
        lines = [
            "",
            "typed candidate goals (advisory; not canonical outcome()):",
        ]
        if required is not None:
            lines.append(f"required_next_step: {required}")
        for goal in visible:
            review = self._ledger.latest_review(goal.id)
            parts = [
                goal.id,
                f"status={goal.status.value}",
                f"predicate={goal.predicate}",
            ]
            if review is not None:
                parts.extend(
                    [
                        f"assessment={review.assessment.value}",
                        f"recommended_action={review.recommended_action or 'none'}",
                        f"route={review.recommended_route or 'none'}",
                    ]
                )
            lines.append("- " + "; ".join(parts))
        return lines

    def _require_turn(self) -> GameTurn:
        if self._turn is None:
            raise RuntimeError("candidate-goal tools require an active game turn")
        return self._turn


def _objective_assessment(report: str) -> ObjectiveAssessment:
    value = _report_value(report, "candidate_goal_assessment").lower()
    try:
        return ObjectiveAssessment(value)
    except ValueError:
        return ObjectiveAssessment.INCONCLUSIVE


def _report_value(report: str, field: str) -> str:
    prefix = f"{field}:"
    for line in report.splitlines():
        if line.strip().lower().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""
