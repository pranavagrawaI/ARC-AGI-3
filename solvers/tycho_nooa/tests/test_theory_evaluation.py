from __future__ import annotations

from pathlib import Path

from tycho.workspace.sandbox import PythonSandbox

from tycho_nooa.theory_evaluation import (
    EvaluationRunStatus,
    EvidenceFrontier,
    ObligationKind,
    ObligationResult,
    ObligationStatus,
    PublicationDisposition,
    SandboxTheoryEvaluator,
    TheoryEvaluation,
    compare,
)
from tycho_nooa.trusted_runtime import TrustedModule, TrustedRuntime, current_runtime


def transition(name: str, status: ObligationStatus) -> ObligationResult:
    return ObligationResult(
        obligation_id=name,
        kind=ObligationKind.TRANSITION,
        status=status,
        description=name,
    )


def evaluation(
    model_hash: str,
    *obligations: ObligationResult,
    status: EvaluationRunStatus = EvaluationRunStatus.COMPLETE,
) -> TheoryEvaluation:
    return TheoryEvaluation(
        model_sha256=model_hash,
        evidence_frontier_sha256="frontier",
        evaluator_sha256="evaluator",
        run_status=status,
        obligations=tuple(obligations),
    )


def test_strict_improvement_is_publishable() -> None:
    parent = evaluation(
        "parent",
        transition("T1", ObligationStatus.PASSED),
        transition("T2", ObligationStatus.PASSED),
        transition("T3", ObligationStatus.FAILED),
    )
    candidate = evaluation(
        "candidate",
        transition("T1", ObligationStatus.PASSED),
        transition("T2", ObligationStatus.PASSED),
        transition("T3", ObligationStatus.PASSED),
    )

    decision = compare(parent, candidate)

    assert decision.disposition is PublicationDisposition.PUBLISHABLE
    assert decision.gained == ("T3",)
    assert decision.lost == ()


def test_residual_substitution_is_rejected_even_at_equal_aggregate_score() -> None:
    parent = evaluation(
        "parent",
        transition("T1", ObligationStatus.PASSED),
        transition("T2", ObligationStatus.PASSED),
        transition("T3", ObligationStatus.FAILED),
    )
    candidate = evaluation(
        "candidate",
        transition("T1", ObligationStatus.FAILED),
        transition("T2", ObligationStatus.PASSED),
        transition("T3", ObligationStatus.PASSED),
    )

    decision = compare(parent, candidate)

    assert decision.disposition is PublicationDisposition.REJECTED_REGRESSION
    assert decision.gained == ("T3",)
    assert decision.lost == ("T1",)


def test_aggregate_dynamics_gain_cannot_pay_for_terminal_regression() -> None:
    terminal = ObligationResult(
        obligation_id="terminal:L0",
        kind=ObligationKind.TERMINAL,
        status=ObligationStatus.PASSED,
        description="host-confirmed terminal L0",
    )
    parent = evaluation(
        "parent",
        transition("T1", ObligationStatus.PASSED),
        transition("T2", ObligationStatus.PASSED),
        transition("T3", ObligationStatus.FAILED),
        terminal,
    )
    candidate = evaluation(
        "candidate",
        transition("T1", ObligationStatus.PASSED),
        transition("T2", ObligationStatus.PASSED),
        transition("T3", ObligationStatus.PASSED),
        ObligationResult(
            obligation_id="terminal:L0",
            kind=ObligationKind.TERMINAL,
            status=ObligationStatus.FAILED,
            description="host-confirmed terminal L0",
        ),
    )

    decision = compare(parent, candidate)

    assert decision.disposition is PublicationDisposition.REJECTED_REGRESSION
    assert decision.lost == ("terminal:L0",)


def test_load_failure_is_rejected() -> None:
    parent = evaluation("parent", transition("T1", ObligationStatus.PASSED))
    candidate = evaluation(
        "candidate",
        status=EvaluationRunStatus.CANDIDATE_LOAD_FAILED,
    )

    assert compare(parent, candidate).disposition is (
        PublicationDisposition.REJECTED_INVALID
    )


def test_evaluator_unavailable_withholds_publication() -> None:
    parent = evaluation("parent", transition("T1", ObligationStatus.PASSED))
    candidate = evaluation(
        "candidate",
        status=EvaluationRunStatus.EVALUATOR_UNAVAILABLE,
    )

    assert compare(parent, candidate).disposition is (
        PublicationDisposition.WITHHELD_INCONCLUSIVE
    )


def test_equal_frontier_behavior_is_not_republished() -> None:
    obligation = transition("T1", ObligationStatus.PASSED)
    parent = evaluation("parent", obligation)
    candidate = evaluation("different-bytes", obligation)

    assert compare(parent, candidate).disposition is PublicationDisposition.UNCHANGED


def test_protected_but_non_driving_repair_does_not_publish_by_itself() -> None:
    parent = evaluation(
        "parent",
        ObligationResult(
            obligation_id="planner:contract",
            kind=ObligationKind.PLANNER_CONTRACT,
            status=ObligationStatus.FAILED,
            description="planner contract",
        ),
    )
    candidate = evaluation(
        "candidate",
        ObligationResult(
            obligation_id="planner:contract",
            kind=ObligationKind.PLANNER_CONTRACT,
            status=ObligationStatus.PASSED,
            description="planner contract",
        ),
    )

    decision = compare(parent, candidate)

    assert decision.disposition is PublicationDisposition.UNCHANGED
    assert decision.gained == ("planner:contract",)


def test_unevaluated_noop_transition_is_not_a_publication_gain() -> None:
    parent = evaluation("parent", transition("T-noop", ObligationStatus.FAILED))
    candidate = evaluation(
        "candidate",
        ObligationResult(
            obligation_id="T-noop",
            kind=ObligationKind.TRANSITION,
            status=ObligationStatus.PASSED,
            description="joint noop",
            diagnostics=(("evaluated", False),),
        ),
    )

    assert compare(parent, candidate).disposition is PublicationDisposition.UNCHANGED


def test_host_confirmed_outcome_false_negative_is_a_publication_gain() -> None:
    parent = evaluation(
        "parent",
        ObligationResult(
            obligation_id="outcome:terminal:L0:level_complete",
            kind=ObligationKind.OUTCOME,
            status=ObligationStatus.FAILED,
            description="host-confirmed terminal L0",
        ),
    )
    candidate = evaluation(
        "candidate",
        ObligationResult(
            obligation_id="outcome:terminal:L0:level_complete",
            kind=ObligationKind.OUTCOME,
            status=ObligationStatus.PASSED,
            description="host-confirmed terminal L0",
        ),
    )

    assert compare(parent, candidate).disposition is (
        PublicationDisposition.PUBLISHABLE
    )


def test_unsupported_exact_regime_withholds_publication() -> None:
    parent = evaluation("parent", transition("T1", ObligationStatus.PASSED))
    candidate = evaluation(
        "candidate",
        transition("T1", ObligationStatus.UNSUPPORTED),
        status=EvaluationRunStatus.INCONCLUSIVE_NOT_PUBLISHABLE,
    )

    assert compare(parent, candidate).disposition is (
        PublicationDisposition.WITHHELD_INCONCLUSIVE
    )


def test_frontier_freezes_host_evidence_bytes(tmp_path: Path) -> None:
    workspace = observed_right_transition(tmp_path)

    frontier = EvidenceFrontier.freeze(workspace)
    original = frontier.files[0].content
    (workspace / frontier.files[0].path).write_text("mutated after freeze")

    assert frontier.files[0].content == original
    assert "world_model.py" not in {item.path for item in frontier.files}


def test_host_probe_returns_named_exact_obligations_without_scraping_prose(
    tmp_path: Path,
) -> None:
    workspace = observed_right_transition(tmp_path)
    evaluator = SandboxTheoryEvaluator(
        PythonSandbox(runtime="host"), allow_unsafe_host=True
    )
    exact_model = b"""import numpy as np
def init_state(grid, level): return np.asarray(grid).copy()
def transition(state, action):
    result = state.copy()
    if action[\"action\"] == \"RIGHT\": result[0, 0] = 1
    return result
def render(state): return state
def outcome(state): return \"ongoing\"
def actions(state): return [{\"action\": \"RIGHT\", \"row\": None, \"col\": None}]
"""

    result = evaluator.evaluate(exact_model, EvidenceFrontier.freeze(workspace))

    transitions = [
        item for item in result.obligations if item.kind is ObligationKind.TRANSITION
    ]
    assert result.run_status is EvaluationRunStatus.COMPLETE
    assert len(transitions) == 1
    assert transitions[0].status is ObligationStatus.PASSED
    assert "L0:T1:RIGHT" in transitions[0].obligation_id


def test_evaluator_supplies_the_declared_symmetry_runtime(tmp_path: Path) -> None:
    workspace = observed_right_transition(tmp_path)
    evaluator = SandboxTheoryEvaluator(
        PythonSandbox(runtime="host"), allow_unsafe_host=True
    )
    exact_model = b"""import numpy as np
import symmetry
def init_state(grid, level): return np.asarray(grid).copy()
def transition(state, action):
    result = state.copy()
    if symmetry.canon({(0, 0): 1}) and action["action"] == "RIGHT": result[0, 0] = 1
    return result
def render(state): return state
def outcome(state): return "ongoing"
def actions(state): return [{"action": "RIGHT", "row": None, "col": None}]
"""

    result = evaluator.evaluate(exact_model, EvidenceFrontier.freeze(workspace))

    assert result.run_status is EvaluationRunStatus.COMPLETE


def test_evaluator_uses_the_runtime_version_it_was_given(tmp_path: Path) -> None:
    workspace = observed_right_transition(tmp_path)
    current = current_runtime()
    historical_modules = tuple(
        TrustedModule.from_source(
            module.path,
            module.import_name,
            module.provenance,
            module.content + b"\nRUNTIME_TEST_MARKER = True\n",
            module.availability,
        )
        if module.path == "symmetry.py"
        else module
        for module in current.modules
    )
    historical = TrustedRuntime.create(historical_modules)
    evaluator = SandboxTheoryEvaluator(
        PythonSandbox(runtime="host"),
        allow_unsafe_host=True,
        trusted_runtime=historical,
    )
    exact_model = b"""import numpy as np
import symmetry
assert symmetry.RUNTIME_TEST_MARKER
def init_state(grid, level): return np.asarray(grid).copy()
def transition(state, action):
    result = state.copy()
    if action["action"] == "RIGHT": result[0, 0] = 1
    return result
def render(state): return state
def outcome(state): return "ongoing"
def actions(state): return [{"action": "RIGHT", "row": None, "col": None}]
"""

    result = evaluator.evaluate(exact_model, EvidenceFrontier.freeze(workspace))

    assert result.run_status is EvaluationRunStatus.COMPLETE


def test_unknown_or_observation_variants_are_inconclusive_not_publishable(
    tmp_path: Path,
) -> None:
    workspace = observed_right_transition(tmp_path)
    evaluator = SandboxTheoryEvaluator(
        PythonSandbox(runtime="host"), allow_unsafe_host=True
    )
    masked_model = b"""import numpy as np
def init_state(grid, level): return np.asarray(grid).copy()
def transition(state, action): return state
def render(state): return np.full_like(state, -1)
def outcome(state): return \"ongoing\"
def actions(state): return []
"""

    result = evaluator.evaluate(masked_model, EvidenceFrontier.freeze(workspace))

    assert result.run_status is EvaluationRunStatus.INCONCLUSIVE_NOT_PUBLISHABLE
    assert any(
        item.status is ObligationStatus.UNSUPPORTED for item in result.obligations
    )


def observed_right_transition(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    level = workspace / "level_0"
    level.mkdir(parents=True)
    (level / "turn_000.txt").write_text("0 0\n0 0\n")
    (level / "turn_000.json").write_text('{"action": null}\n')
    (level / "turn_001.txt").write_text("1 0\n0 0\n")
    (level / "turn_001.json").write_text(
        '{"action": "RIGHT", "row": null, "col": null}\n'
    )
    (workspace / ".current_frame.json").write_text('{"level": 0, "turn": 1}\n')
    (workspace / "world_model.py").write_text("builder-authored, not evidence")
    return workspace
