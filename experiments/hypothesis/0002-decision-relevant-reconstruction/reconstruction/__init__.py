"""Offline executable-reconstruction tracer bullet.

The runtime is deliberately small: it accepts only the closed fixture grammar, keeps
Oracle Coach semantics labelled, and reveals the held-out successor only after a plan
certificate exists. It makes no provider call and cannot release an environment action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
PUBLIC_FIELDS = ("available_actions", "full_reset", "levels_completed", "state", "win_levels")


class FixtureError(ValueError):
    pass


def _plain(value: Any) -> Any:
    if hasattr(value, "to_record"):
        return value.to_record()
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def load_fixture(path: Path | str) -> dict[str, Any]:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    if fixture.get("schema") != "0002-operational-fixture-v1":
        raise FixtureError("unsupported fixture schema")
    if fixture.get("authority", {}).get("condition") != "oracle_coach":
        raise FixtureError("the operational fixture must remain Oracle Coach labelled")
    return fixture


@dataclass(frozen=True)
class ObservationOccurrence:
    step: int
    public: dict[str, Any]
    frame: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TransitionOccurrence:
    after_step: int
    before_step: int
    action: int


@dataclass(frozen=True)
class ClosedOccurrences:
    schema: str
    authority_label: str
    source_trace_id: str
    source_sha256: str
    frontier: int
    observations: tuple[ObservationOccurrence, ...]
    transitions: tuple[TransitionOccurrence, ...]

    def observation(self, step: int) -> ObservationOccurrence:
        try:
            return next(row for row in self.observations if row.step == step)
        except StopIteration as exc:
            raise FixtureError(f"observation step {step} is outside the closed frontier") from exc


def _action_id(raw: Any) -> int:
    text = str(raw)
    if not text.startswith("ACTION") or not text[6:].isdigit():
        raise FixtureError(f"non-movement action {text!r} in transition prefix")
    return int(text[6:])


def _bbox(values: Any) -> tuple[int, int, int, int]:
    if not isinstance(values, list) or len(values) != 4:
        raise FixtureError(f"bbox must contain four integers: {values!r}")
    return (int(values[0]), int(values[1]), int(values[2]), int(values[3]))


def _delta(values: Any) -> tuple[int, int]:
    if not isinstance(values, list) or len(values) != 2:
        raise FixtureError(f"delta must contain two integers: {values!r}")
    return (int(values[0]), int(values[1]))


def _source_path(fixture: dict[str, Any]) -> Path:
    path = REPO / fixture["source"]["path"]
    if not path.is_file():
        raise FixtureError(f"source trace not found: {path}")
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != fixture["source"]["sha256"]:
        raise FixtureError(f"source digest mismatch: expected {fixture['source']['sha256']}, got {digest}")
    return path


def _trace_rows_through(
    fixture: dict[str, Any], last_step_inclusive: int
) -> list[dict[str, Any]]:
    path = _source_path(fixture)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line)["data"])
            if len(rows) - 1 == last_step_inclusive:
                break
    if len(rows) <= last_step_inclusive:
        raise FixtureError(f"source trace has no step {last_step_inclusive}")
    return rows


def load_occurrences(fixture: dict[str, Any]) -> ClosedOccurrences:
    source = fixture["source"]
    observations_spec = source["observation_prefix"]
    transitions_spec = source["transition_prefix"]
    first = int(observations_spec["first_step"])
    frontier = int(observations_spec["last_step_inclusive"])
    held_out = int(source["held_out_transition"]["after_step"])
    if held_out <= frontier:
        raise FixtureError("held-out successor is not beyond the evidence frontier")
    rows = _trace_rows_through(fixture, frontier)

    observations = []
    for step in range(first, frontier + 1):
        row = rows[step]
        public = {field: row[field] for field in PUBLIC_FIELDS if field in row}
        frame = tuple(tuple(int(cell) for cell in line) for line in row["frame"][-1])
        observations.append(ObservationOccurrence(step=step, public=public, frame=frame))

    transitions = []
    for after_step in range(
        int(transitions_spec["first_after_step"]),
        int(transitions_spec["last_after_step_inclusive"]) + 1,
    ):
        transitions.append(
            TransitionOccurrence(
                after_step=after_step,
                before_step=after_step - 1,
                action=_action_id(rows[after_step]["action_input"]["id"]),
            )
        )
    return ClosedOccurrences(
        schema="0002-closed-occurrences-v1",
        authority_label="oracle_coach",
        source_trace_id=source["trace_id"],
        source_sha256=source["sha256"],
        frontier=frontier,
        observations=tuple(observations),
        transitions=tuple(transitions),
    )


@dataclass(frozen=True)
class PatchCandidate:
    candidate_id: str
    step: int
    bbox: tuple[int, int, int, int]
    values: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CandidateScene:
    schema: str
    authority_label: str
    representation_version: str
    evidence_frontier: int
    representation_manifest: tuple[dict[str, Any], ...]
    candidates: tuple[PatchCandidate, ...]

    def to_bytes(self) -> bytes:
        return canonical_bytes(self)


class CandidateWorld:
    """Enumerate every lattice patch; choose no role and apply no relevance ranking."""

    def measure(
        self, evidence_frontier: ClosedOccurrences, representation: dict[str, Any]
    ) -> CandidateScene:
        parameters = {row["name"]: row for row in representation["parameters"]}
        for row in parameters.values():
            if not row.get("frozen"):
                raise FixtureError(f"representation parameter {row['name']} is not frozen")
            if row["selected"] not in row["domain"]:
                raise FixtureError(f"representation parameter {row['name']} is outside its domain")
        size = int(parameters["patch_size"]["selected"])
        row_offset = int(parameters["row_offset"]["selected"])
        column_offset = int(parameters["column_offset"]["selected"])
        candidates: list[PatchCandidate] = []
        version = str(representation["version"])
        for occurrence in evidence_frontier.observations:
            rows, columns = len(occurrence.frame), len(occurrence.frame[0])
            for row in range(row_offset, rows - size + 1, size):
                for column in range(column_offset, columns - size + 1, size):
                    bbox = (row, row + size, column, column + size)
                    values = tuple(
                        tuple(occurrence.frame[y][column : column + size])
                        for y in range(row, row + size)
                    )
                    candidates.append(
                        PatchCandidate(
                            candidate_id=(
                                f"{version}:frontier-{evidence_frontier.frontier}:"
                                f"step-{occurrence.step}:patch-{row}-{column}"
                            ),
                            step=occurrence.step,
                            bbox=bbox,
                            values=values,
                        )
                    )
        return CandidateScene(
            schema="0002-candidate-scene-v1",
            authority_label="oracle_coach",
            representation_version=version,
            evidence_frontier=evidence_frontier.frontier,
            representation_manifest=tuple(representation["parameters"]),
            candidates=tuple(candidates),
        )


@dataclass(frozen=True)
class ReconstructedState:
    evidence_step: int
    position: tuple[int, int]
    variables: dict[str, int]
    public: dict[str, Any]


@dataclass(frozen=True)
class ConjectureResult:
    conjecture_id: str
    conjecture_class: str
    status: str
    supports: tuple[int, ...] = ()
    contradictions: tuple[int, ...] = ()
    uncovered: tuple[int, ...] = ()


@dataclass(frozen=True)
class Discriminator:
    discriminator_id: str
    action: int
    distinguishes: tuple[str, str]
    predicted_outcomes: dict[str, Any]


@dataclass(frozen=True)
class ReplayRow:
    after_step: int
    action: int
    before_position: tuple[int, int]
    predicted_position: tuple[int, int]
    actual_position: tuple[int, int]
    result: str


@dataclass(frozen=True)
class ConjectureReport:
    schema: str
    authority_label: str
    evidence_frontier: int
    results: tuple[ConjectureResult, ...]
    replay_rows: tuple[ReplayRow, ...]
    state_trace: tuple[ReconstructedState, ...]
    final_state: ReconstructedState
    discriminators: tuple[Discriminator, ...]
    selected_transition_ids: tuple[str, ...]
    grounding_ids: tuple[str, ...]
    grounded_candidate_ids: tuple[str, ...]
    objective_id: str
    action_deltas: dict[int, tuple[int, int]]
    cycle: dict[str, Any]
    objective: dict[str, Any]

    def by_id(self, conjecture_id: str) -> ConjectureResult:
        try:
            return next(row for row in self.results if row.conjecture_id == conjecture_id)
        except StopIteration as exc:
            raise KeyError(conjecture_id) from exc


class Theory:
    def evaluate(
        self,
        proposal: dict[str, Any],
        evidence_frontier: ClosedOccurrences,
        scene: CandidateScene,
    ) -> ConjectureReport:
        grounding_rows = proposal["groundings"]
        groundings = {row["id"]: row for row in grounding_rows}
        controlled_rows = [row for row in grounding_rows if row.get("kind") == "tracked_candidate"]
        if len(controlled_rows) != 1:
            raise FixtureError("v0 requires exactly one tracked_candidate grounding")
        controlled = controlled_rows[0]
        grounding_by_symbol = {row["symbol"]: row for row in grounding_rows}
        occurrence_boxes = {
            int(step): _bbox(bbox)
            for step, bbox in controlled["occurrences"].items()
        }
        candidate_by_location = {
            (candidate.step, candidate.bbox): candidate for candidate in scene.candidates
        }
        grounded_candidate_ids: list[str] = []
        for step, bbox in sorted(occurrence_boxes.items()):
            candidate = candidate_by_location.get((step, bbox))
            if candidate is None:
                raise FixtureError(
                    f"grounding {controlled['id']} cites absent candidate at step {step}: {bbox}"
                )
            grounded_candidate_ids.append(candidate.candidate_id)
        for grounding in proposal["groundings"]:
            if grounding["kind"] == "region":
                bbox = _bbox(grounding["bbox"])
                candidate = candidate_by_location.get(
                    (evidence_frontier.observations[0].step, bbox)
                )
                if candidate is None:
                    raise FixtureError(f"grounding {grounding['id']} cites absent candidate {bbox}")
                grounded_candidate_ids.append(candidate.candidate_id)

        transition_rows = proposal["transitions"]
        translations = [row for row in transition_rows if row.get("kind") == "action_translation"]
        cycles = [row for row in transition_rows if row.get("kind") == "advance_on_entry"]
        if len(translations) != 1 or len(cycles) != 1:
            raise FixtureError("v0 requires one action_translation and one advance_on_entry")
        translation = translations[0]
        action_deltas = {
            int(action): _delta(delta)
            for action, delta in translation["action_deltas"].items()
        }
        cycle = cycles[0]
        cycle_region = _bbox(grounding_by_symbol[cycle["region"]]["bbox"])
        start_step = min(occurrence_boxes)
        start_bbox = occurrence_boxes[start_step]
        position = (start_bbox[0], start_bbox[2])
        variable = str(cycle["variable"])
        variables = {variable: int(cycle["initial"])}
        state_trace = [
            ReconstructedState(
                start_step,
                position,
                dict(variables),
                dict(evidence_frontier.observation(start_step).public),
            )
        ]
        supports: list[int] = []
        contradictions: list[int] = []
        entry_steps: list[int] = []
        replay_rows: list[ReplayRow] = []

        for occurrence in evidence_frontier.transitions:
            previous_bbox = occurrence_boxes[occurrence.before_step]
            actual_bbox = occurrence_boxes[occurrence.after_step]
            delta = action_deltas[occurrence.action]
            predicted = (previous_bbox[0] + delta[0], previous_bbox[2] + delta[1])
            actual = (actual_bbox[0], actual_bbox[2])
            previous_position = (previous_bbox[0], previous_bbox[2])
            matches = predicted == actual
            (supports if matches else contradictions).append(occurrence.after_step)
            replay_rows.append(
                ReplayRow(
                    after_step=occurrence.after_step,
                    action=occurrence.action,
                    before_position=previous_position,
                    predicted_position=predicted,
                    actual_position=actual,
                    result="support" if matches else "contradiction",
                )
            )
            position = actual
            region_position = (cycle_region[0], cycle_region[2])
            if position == region_position and previous_position != region_position:
                variables[variable] = (
                    variables[variable] + int(cycle["delta"])
                ) % int(cycle["modulus"])
                entry_steps.append(occurrence.after_step)
            state_trace.append(
                ReconstructedState(
                    occurrence.after_step,
                    position,
                    dict(variables),
                    dict(evidence_frontier.observation(occurrence.after_step).public),
                )
            )

        results = [
            ConjectureResult(
                conjecture_id=str(translation["id"]),
                conjecture_class="transition",
                status="working" if not contradictions else "contradicted",
                supports=tuple(supports),
                contradictions=tuple(contradictions),
            ),
            ConjectureResult(
                conjecture_id=str(cycle["id"]),
                conjecture_class="transition",
                status="unresolved",
                uncovered=tuple(entry_steps),
            ),
        ]
        competitor = proposal["competitor"]
        results.append(
            ConjectureResult(
                conjecture_id=str(competitor["id"]),
                conjecture_class="transition",
                status="unresolved",
                uncovered=tuple(entry_steps),
            )
        )
        objectives = [row for row in proposal["objectives"] if row.get("kind") == "entry_with_equality"]
        if len(objectives) != 1:
            raise FixtureError("v0 requires exactly one entry_with_equality objective")
        objective = objectives[0]
        results.append(
            ConjectureResult(
                conjecture_id=str(objective["id"]),
                conjecture_class="objective",
                status="unresolved",
                uncovered=(evidence_frontier.frontier,),
            )
        )
        final_state = state_trace[-1]
        objective_bbox = _bbox(grounding_by_symbol[objective["region"]]["bbox"])
        target_position = (objective_bbox[0], objective_bbox[2])
        needed_delta = (
            target_position[0] - final_state.position[0],
            target_position[1] - final_state.position[1],
        )
        discriminator_actions = [action for action, delta in action_deltas.items() if delta == needed_delta]
        if len(discriminator_actions) != 1:
            raise FixtureError("v0 discriminator requires one action reaching the objective region")
        competitor_value = (
            int(competitor["initial"]) + len(entry_steps) * int(competitor["delta"])
        ) % int(competitor["modulus"])
        effect_delta = int(objective["public_effect"]["delta"])
        intended_outcome = effect_delta if variables[str(objective["left"])] == int(objective["right"]) else 0
        competitor_outcome = effect_delta if competitor_value == int(objective["right"]) else 0
        discriminators = ()
        if intended_outcome != competitor_outcome:
            discriminators = (
                Discriminator(
                    discriminator_id="D-cycle-delta-at-objective-entry",
                    action=discriminator_actions[0],
                    distinguishes=(str(cycle["id"]), str(competitor["id"])),
                    predicted_outcomes={
                        str(cycle["id"]): {"public.levels_completed_delta": intended_outcome},
                        str(competitor["id"]): {"public.levels_completed_delta": competitor_outcome},
                    },
                ),
            )
        return ConjectureReport(
            schema="0002-conjecture-report-v1",
            authority_label="oracle_coach",
            evidence_frontier=evidence_frontier.frontier,
            results=tuple(results),
            replay_rows=tuple(replay_rows),
            state_trace=tuple(state_trace),
            final_state=final_state,
            discriminators=discriminators,
            selected_transition_ids=(str(translation["id"]), str(cycle["id"])),
            grounding_ids=tuple(groundings),
            grounded_candidate_ids=tuple(grounded_candidate_ids),
            objective_id=str(objective["id"]),
            action_deltas=action_deltas,
            cycle=cycle,
            objective=objective,
        )


@dataclass(frozen=True)
class PlanCertificate:
    schema: str
    authority_label: str
    objective_conjecture: str
    working_theory_version: str
    initial_reconstructed_state: ReconstructedState
    actions: tuple[int, ...]
    relied_upon_groundings: tuple[str, ...]
    relied_upon_transitions: tuple[str, ...]
    relied_upon_candidate_ids: tuple[str, ...]
    predicted_states: tuple[ReconstructedState, ...]
    prediction: dict[str, Any]
    unverified_assumptions: tuple[str, ...]
    risk_disposition: str
    search_expansions: int
    frozen_projection: dict[str, Any]


class Planner:
    def compare(
        self,
        state: ReconstructedState,
        theory: ConjectureReport,
        proposal: dict[str, Any],
        budget: dict[str, Any],
        frozen_projection: dict[str, Any],
    ) -> PlanCertificate:
        contradicted = tuple(
            conjecture_id
            for conjecture_id in theory.selected_transition_ids
            if theory.by_id(conjecture_id).contradictions
        )
        if contradicted:
            raise FixtureError(
                "selected transition has replay contradictions: " + ", ".join(contradicted)
            )
        groundings = {row["id"]: row for row in proposal["groundings"]}
        objective = theory.objective
        grounding_by_symbol = {row["symbol"]: row for row in groundings.values()}
        objective_region = _bbox(grounding_by_symbol[objective["region"]]["bbox"])
        target_position = (objective_region[0], objective_region[2])
        max_depth = int(budget["max_depth"])
        max_expansions = int(budget["max_expansions"])
        if max_depth < 1 or max_expansions < 1:
            raise FixtureError("planner budget cannot examine one successor")
        available_actions = state.public.get("available_actions")
        if (
            not isinstance(available_actions, list)
            or not available_actions
            or any(not isinstance(action, int) for action in available_actions)
        ):
            raise FixtureError("reconstructed state has no valid public available_actions")
        legal_actions = set(available_actions)

        expansions = 0
        chosen_action: int | None = None
        chosen_state: ReconstructedState | None = None
        for action in sorted(action for action in theory.action_deltas if action in legal_actions):
            if expansions >= max_expansions:
                raise FixtureError("planner exceeded expansion budget")
            expansions += 1
            delta = theory.action_deltas[action]
            position = (state.position[0] + delta[0], state.position[1] + delta[1])
            variables = dict(state.variables)
            public: dict[str, Any] = {}
            candidate = ReconstructedState(state.evidence_step + 1, position, variables, public)
            if (
                position == target_position
                and variables[str(objective["left"])] == int(objective["right"])
            ):
                effect = objective["public_effect"]
                if effect["field"] != "levels_completed" or not isinstance(effect["delta"], int):
                    raise FixtureError("v0 objective public effect must be an integer levels_completed delta")
                candidate.public["levels_completed"] = (
                    int(state.public["levels_completed"]) + int(effect["delta"])
                )
                chosen_action = action
                chosen_state = candidate
                break
        if chosen_action is None or chosen_state is None:
            raise FixtureError("planner returned UNKNOWN within the frozen budget")
        projection_paths = _validate_projection(
            frozen_projection,
            expected_candidate_ids=theory.grounded_candidate_ids,
        )
        prediction: dict[str, Any] = {}
        for path in projection_paths:
            prefix, field = path.split(".", 1)
            if prefix != "public" or field not in chosen_state.public:
                raise FixtureError(f"projection path cannot be generated from predicted state: {path}")
            prediction[path] = chosen_state.public[field]
        return PlanCertificate(
            schema="0002-plan-certificate-v1",
            authority_label="oracle_coach",
            objective_conjecture=theory.objective_id,
            working_theory_version=str(budget["working_theory_id"]),
            initial_reconstructed_state=state,
            actions=(chosen_action,),
            relied_upon_groundings=theory.grounding_ids,
            relied_upon_transitions=theory.selected_transition_ids,
            relied_upon_candidate_ids=theory.grounded_candidate_ids,
            predicted_states=(chosen_state,),
            prediction=prediction,
            unverified_assumptions=(
                f"{theory.cycle['id']} and {proposal['competitor']['id']} "
                "are observationally indistinguishable on the prefix",
                f"{theory.objective_id} has no supporting completion event inside the prefix",
            ),
            risk_disposition=str(budget["risk_disposition"]),
            search_expansions=expansions,
            frozen_projection=frozen_projection,
        )


@dataclass(frozen=True)
class ExecutionComparison:
    schema: str
    authority_label: str
    scope: str
    selected_action: int
    historical_action: int
    prediction: dict[str, Any]
    actual: dict[str, Any]
    field_matches: dict[str, bool]
    exact_match: bool


def _validate_projection(
    projection: dict[str, Any], *, expected_candidate_ids: tuple[str, ...]
) -> tuple[str, ...]:
    if projection.get("schema") != "0002-heldout-projection-v1":
        raise FixtureError("unsupported frozen projection schema")
    if projection.get("comparison") != "exact_equality":
        raise FixtureError("v0 frozen projection comparison must be exact_equality")
    candidate_ids = projection.get("candidate_ids")
    if candidate_ids != list(expected_candidate_ids) or not candidate_ids:
        raise FixtureError("frozen projection candidate IDs do not match grounded plan dependencies")
    fields = projection.get("fields")
    if not isinstance(fields, list) or not fields:
        raise FixtureError("frozen projection must contain at least one field")
    if any(not isinstance(field, dict) or set(field) != {"path"} for field in fields):
        raise FixtureError("each frozen projection field must contain only a path")
    paths = tuple(str(field["path"]) for field in fields)
    if len(set(paths)) != len(paths):
        raise FixtureError("frozen projection paths must be unique")
    return paths


def compare_held_out(
    fixture: dict[str, Any], certificate: PlanCertificate
) -> ExecutionComparison:
    held_out = fixture["source"]["held_out_transition"]
    historical_action = int(held_out["historical_action"])
    if certificate.frozen_projection.get("historical_action") != historical_action:
        raise FixtureError("frozen projection historical action does not match fixture")
    paths = _validate_projection(
        certificate.frozen_projection,
        expected_candidate_ids=certificate.relied_upon_candidate_ids,
    )
    if set(certificate.prediction) != set(paths):
        raise FixtureError("certificate prediction does not exactly cover frozen projection paths")
    selected_action = certificate.actions[0]
    if selected_action != historical_action:
        return ExecutionComparison(
            schema="0002-execution-comparison-v1",
            authority_label="oracle_coach",
            scope="decision_divergence_unscored",
            selected_action=selected_action,
            historical_action=historical_action,
            prediction={},
            actual={},
            field_matches={},
            exact_match=False,
        )

    after_step = int(held_out["after_step"])
    rows = _trace_rows_through(fixture, after_step)
    successor = rows[after_step]
    prediction = dict(certificate.prediction)
    actual: dict[str, Any] = {}
    for path in paths:
        prefix, field = path.split(".", 1)
        if prefix != "public" or field not in PUBLIC_FIELDS:
            raise FixtureError(f"projection path is not a closed public field: {path}")
        actual[path] = successor[field]
    matches = {path: actual[path] == expected for path, expected in prediction.items()}
    return ExecutionComparison(
        schema="0002-execution-comparison-v1",
        authority_label="oracle_coach",
        scope="historical_action_prediction_fit",
        selected_action=selected_action,
        historical_action=historical_action,
        prediction=prediction,
        actual=actual,
        field_matches=matches,
        exact_match=all(matches.values()),
    )


def _compiled_conjectures_artifact(
    fixture: dict[str, Any], report: ConjectureReport
) -> dict[str, Any]:
    proposal = fixture["oracle_proposal"]
    return {
        "schema": "0002-compiled-conjectures-v1",
        "authority_label": "oracle_coach",
        "evidence_frontier": report.evidence_frontier,
        "groundings": proposal["groundings"],
        "transitions": proposal["transitions"],
        "competitors": [proposal["competitor"]],
        "objectives": proposal["objectives"],
        "compiled_action_deltas": report.action_deltas,
        "selected_transition_ids": report.selected_transition_ids,
        "objective_id": report.objective_id,
    }


def _held_out_successor_artifact(fixture: dict[str, Any]) -> dict[str, Any]:
    held_out = fixture["source"]["held_out_transition"]
    after_step = int(held_out["after_step"])
    successor = _trace_rows_through(fixture, after_step)[after_step]
    return {
        "schema": "0002-held-out-successor-v1",
        "authority_label": "oracle_coach",
        "after_step": after_step,
        "historical_action": int(held_out["historical_action"]),
        "public": {field: successor[field] for field in PUBLIC_FIELDS if field in successor},
        "frame": successor["frame"][-1],
    }


def _record_adapter():
    """Load 0007 recording through one explicit sibling-experiment seam."""

    import sys

    root = REPO / "experiments/agents/0007-guided-control"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from bench.record import RunRecorder, RunStatus  # pyright: ignore[reportMissingImports]

    return RunRecorder, RunStatus


def run_operational(fixture_path: Path | str, output: Path | str) -> Path:
    fixture_path = Path(fixture_path)
    output = Path(output)
    fixture = load_fixture(fixture_path)
    card = EXPERIMENT / "EXPERIMENT-CARD.md"
    if len(card.read_bytes()) > 1024:
        raise FixtureError("experiment card exceeds 1,024 UTF-8 bytes")
    if output.exists():
        raise FileExistsError(output)

    started = perf_counter()
    occurrences = load_occurrences(fixture)
    scene = CandidateWorld().measure(occurrences, fixture["representation"])
    replay_started = perf_counter()
    report = Theory().evaluate(fixture["oracle_proposal"], occurrences, scene)
    replay_seconds = perf_counter() - replay_started
    if not report.discriminators:
        raise FixtureError("working theory and competitor have no explicit discriminator")
    certificate = Planner().compare(
        report.final_state,
        report,
        fixture["oracle_proposal"],
        fixture["planning"],
        fixture["frozen_projection"],
    )
    compiled_conjectures = _compiled_conjectures_artifact(fixture, report)

    RunRecorder, RunStatus = _record_adapter()
    recorder = RunRecorder.open(
        output.parent,
        output.name,
        condition="0002-operational-oracle-coach",
        game="ls20",
        models={},
        budgets={
            "max_actions": 0,
            "max_provider_calls": 0,
            "max_search_expansions": fixture["planning"]["max_expansions"],
        },
        extra={
            "authority_label": "oracle_coach",
            "fixture": str(fixture_path.relative_to(REPO)),
            "fixture_sha256": sha256(fixture_path.read_bytes()).hexdigest(),
            "experiment_card": str(card.relative_to(REPO)),
            "experiment_card_sha256": sha256(card.read_bytes()).hexdigest(),
            "producer_source_sha256": {
                str(path.relative_to(EXPERIMENT)): sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__).resolve(), EXPERIMENT / "run_tracer.py")
            },
        },
    )

    def record_artifact(name: str, artifact: Any) -> None:
        _write_json(recorder.root / name, artifact)
        recorder.event(
            name.removesuffix(".json"),
            {
                "authority_label": "oracle_coach",
                "path": name,
                "sha256": sha256((recorder.root / name).read_bytes()).hexdigest(),
            },
            actor="machinery",
            output=_plain(artifact),
        )

    with recorder:
        for name, artifact in {
            "closed_occurrences.json": occurrences,
            "candidate_scene.json": scene,
            "compiled_conjectures.json": compiled_conjectures,
            "replay_report.json": report,
            "plan_certificate.json": certificate,
        }.items():
            record_artifact(name, artifact)

        comparison = compare_held_out(fixture, certificate)
        held_out_successor = _held_out_successor_artifact(fixture)
        record_artifact("held_out_successor.json", held_out_successor)
        record_artifact("comparison.json", comparison)

        unresolved = sum(row.status == "unresolved" for row in report.results)
        summary = {
            "verdict": "PASS" if comparison.exact_match else "FAIL",
            "authority_label": "oracle_coach",
            "provider_calls": 0,
            "environment_actions": 0,
            "candidate_count": len(scene.candidates),
            "candidate_scene_bytes": len(scene.to_bytes()),
            "model_tokens": 0,
            "model_seconds": 0,
            "replay_seconds": replay_seconds,
            "search_expansions": certificate.search_expansions,
            "unresolved_cases": unresolved,
            "elapsed_seconds": perf_counter() - started,
        }
        recorder.finish(RunStatus.COMPLETE if comparison.exact_match else RunStatus.ABORTED, summary)
    return recorder.root
