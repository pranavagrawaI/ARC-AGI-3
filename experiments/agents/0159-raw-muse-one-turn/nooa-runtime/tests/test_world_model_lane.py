from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from world_model_lane import (
    EditProvenance,
    ExecutableTheoryLane,
    LaneIntegrityError,
    ModelEdit,
)

EXPERIMENT = Path(__file__).resolve().parents[2]
E0_PATH = EXPERIMENT / "e0_evidence.py"
E0_SPEC = importlib.util.spec_from_file_location("wm0_test_e0", E0_PATH)
assert E0_SPEC is not None and E0_SPEC.loader is not None
E0 = importlib.util.module_from_spec(E0_SPEC)
sys.modules[E0_SPEC.name] = E0
E0_SPEC.loader.exec_module(E0)


def observation(
    grid: tuple[tuple[int, ...], ...],
    *,
    state: str = "NOT_FINISHED",
    levels_completed: int = 0,
) -> object:
    return E0.PublicObservationInput(
        frames=(grid,),
        public_state=state,
        levels_completed=levels_completed,
        full_reset=False,
        legal_actions=("UP", "DOWN", "LEFT", "RIGHT"),
    )


def evidence_snapshot(
    tmp_path: Path,
    *,
    after: tuple[tuple[int, ...], ...] | None = None,
    terminal: bool = False,
) -> Path:
    root = tmp_path / "host-evidence"
    ledger = E0.HostEvidenceLedger(root, run_id="test-run")
    ledger.record_initial_reset(observation(((0, 0), (0, 0))))
    if after is not None:
        result = ledger.record_action_result(
            global_turn=0,
            action=E0.Action("RIGHT"),
            successor=observation(
                after,
                state="WIN" if terminal else "NOT_FINISHED",
                levels_completed=1 if terminal else 0,
            ),
        )
        snapshot = result.snapshot
    else:
        snapshot = ledger.snapshot()
    return root / "snapshots" / f"{snapshot.snapshot_id}.json"


IDENTITY_SOURCE = '''from dataclasses import dataclass
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


SUPPORTED_SOURCE = '''from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    grid: np.ndarray
    terminal: bool = False

def init_state(grid0: np.ndarray, level: int) -> State:
    return State(grid0.copy())

def transition(state: State, action: dict) -> State:
    grid = state.grid.copy()
    if action["action"] == "RIGHT":
        grid[0, 1] = 1
    return State(grid, terminal=True)

def render(state: State) -> np.ndarray:
    return state.grid

def outcome(state: State) -> str:
    return "level_complete" if state.terminal else "ongoing"
'''


PARTIAL_RIGHT_SOURCE = '''import numpy as np

UNKNOWN = -1

def predict(grid: np.ndarray, action: dict) -> np.ndarray | None:
    if action["action"] != "RIGHT":
        return None
    result = np.full_like(grid, UNKNOWN)
    result[0, 1] = 1
    return result
'''


def provenance(index: int = 1) -> EditProvenance:
    return EditProvenance(author_event_id=f"worker-turn-0:edit-{index:04d}")


def test_lane_starts_with_a_template_but_no_active_model(tmp_path: Path) -> None:
    lane = ExecutableTheoryLane(tmp_path / "lane")
    state = lane.advance(evidence_snapshot(tmp_path))

    view = lane.read_world_model()

    assert state.current_model_version_id is None
    assert view["current_model_version_id"] is None
    assert view["installed_evidence_snapshot_id"] == state.evidence_snapshot_id
    assert "def predict" in view["source"]
    assert "def init_state" in view["moore_machine_template"]
    assert view["source_kind"] == "quick_theory_template"
    assert view["contract_kind"] is None


def test_partial_predictor_is_a_first_class_supported_commitment(tmp_path: Path) -> None:
    lane = ExecutableTheoryLane(tmp_path / "lane")
    lane.advance(evidence_snapshot(tmp_path, after=((0, 1), (0, 0))))

    evaluation = lane.apply(ModelEdit.whole_file(PARTIAL_RIGHT_SOURCE), provenance())

    assert evaluation.candidate_status == "active_supported"
    assert evaluation.operational["contract_kind"] == "partial_predictor"
    assert evaluation.initial_render["status"] == "inconclusive"
    assert evaluation.outcome["status"] == "inconclusive"
    assert evaluation.dynamics["status"] == "supported"
    assert evaluation.dynamics["claimed"] == 1
    assert evaluation.dynamics["per_action"]["RIGHT"]["status"] == "supported"
    assert evaluation.dynamics["transition_results"] == [
        {
            "source_event_id": "test-run:event:00000001",
            "action": "RIGHT",
            "claim": "partial_cell_claim",
            "known_cells": 1,
            "total_cells": 4,
            "matched": True,
        }
    ]
    assert lane.read_world_model()["contract_kind"] == "partial_predictor"


def test_partial_predictor_can_abstain_without_making_a_false_claim(tmp_path: Path) -> None:
    lane = ExecutableTheoryLane(tmp_path / "lane")
    lane.advance(evidence_snapshot(tmp_path, after=((0, 1), (0, 0))))
    source = PARTIAL_RIGHT_SOURCE.replace('"RIGHT"', '"LEFT"')

    evaluation = lane.apply(ModelEdit.whole_file(source), provenance())

    assert evaluation.candidate_status == "active_inconclusive"
    assert evaluation.dynamics["evaluated"] == 1
    assert evaluation.dynamics["claimed"] == 0
    assert evaluation.dynamics["per_action"]["RIGHT"]["status"] == "inconclusive"
    assert evaluation.dynamics["transition_results"][0]["claim"] == (
        "whole_action_abstention"
    )
    assert evaluation.dynamics["transition_results"][0]["matched"] is None


def test_partial_predictor_contradiction_is_activated_not_silently_rejected(
    tmp_path: Path,
) -> None:
    lane = ExecutableTheoryLane(tmp_path / "lane")
    lane.advance(evidence_snapshot(tmp_path, after=((0, 1), (0, 0))))
    source = PARTIAL_RIGHT_SOURCE.replace("result[0, 1] = 1", "result[0, 1] = 2")

    evaluation = lane.apply(ModelEdit.whole_file(source), provenance())

    assert evaluation.candidate_status == "active_contradicted"
    assert evaluation.dynamics["claimed"] == 1
    assert evaluation.dynamics["matched"] == 0
    assert lane.read_world_model()["current_model_version_id"] == (
        evaluation.active_model_version_id
    )


def test_evidence_advance_is_idempotent_append_only_and_rejects_old_frontier(
    tmp_path: Path,
) -> None:
    root = tmp_path / "host-evidence"
    ledger = E0.HostEvidenceLedger(root, run_id="frontier-test")
    ledger.record_initial_reset(observation(((0, 0), (0, 0))))
    first = ledger.snapshot()
    first_path = root / "snapshots" / f"{first.snapshot_id}.json"
    result = ledger.record_action_result(
        global_turn=0,
        action=E0.Action("RIGHT"),
        successor=observation(((0, 1), (0, 0))),
    )
    second_path = root / "snapshots" / f"{result.snapshot.snapshot_id}.json"
    lane = ExecutableTheoryLane(tmp_path / "lane")

    assert lane.advance(first_path).evidence_snapshot_id == first.snapshot_id
    assert lane.advance(first_path).evidence_snapshot_id == first.snapshot_id
    assert lane.advance(second_path).evidence_snapshot_id == result.snapshot.snapshot_id
    with pytest.raises(LaneIntegrityError, match="older"):
        lane.advance(first_path)


def test_evidence_advance_rejects_different_verified_snapshot_at_same_frontier(
    tmp_path: Path,
) -> None:
    paths = []
    for name in ("one", "two"):
        root = tmp_path / name / "host-evidence"
        ledger = E0.HostEvidenceLedger(root, run_id=f"run-{name}")
        ledger.record_initial_reset(observation(((0, 0), (0, 0))))
        snapshot = ledger.snapshot()
        paths.append(root / "snapshots" / f"{snapshot.snapshot_id}.json")
    lane = ExecutableTheoryLane(tmp_path / "lane")

    lane.advance(paths[0])
    with pytest.raises(LaneIntegrityError, match="divergent"):
        lane.advance(paths[1])


def test_operational_rejection_preserves_candidate_and_prior_current_version(
    tmp_path: Path,
) -> None:
    lane = ExecutableTheoryLane(tmp_path / "lane")
    lane.advance(evidence_snapshot(tmp_path))
    accepted = lane.apply(ModelEdit.whole_file(IDENTITY_SOURCE), provenance())

    rejected = lane.apply(ModelEdit.whole_file("def broken(:\n"), provenance(2))

    assert accepted.candidate_status == "active_inconclusive"
    assert rejected.candidate_status == "rejected_operationally"
    assert rejected.active_model_version_id == accepted.active_model_version_id
    assert lane.read_world_model()["current_model_version_id"] == (
        accepted.active_model_version_id
    )
    assert (tmp_path / "lane" / "candidates" / rejected.candidate_id / "source.py").is_file()


def test_operationally_valid_contradicted_candidate_becomes_current(
    tmp_path: Path,
) -> None:
    lane = ExecutableTheoryLane(tmp_path / "lane")
    lane.advance(evidence_snapshot(tmp_path, after=((0, 1), (0, 0))))

    evaluation = lane.apply(ModelEdit.whole_file(IDENTITY_SOURCE), provenance())

    assert evaluation.candidate_status == "active_contradicted"
    assert evaluation.active_model_version_id is not None
    assert evaluation.dynamics is not None
    assert evaluation.dynamics["evaluated"] == 1
    assert evaluation.dynamics["matched"] == 0
    assert evaluation.dynamics["transition_results"][0]["claim"] == "full_cell_claim"
    assert evaluation.dynamics["transition_results"][0]["matched"] is False
    observed_blob = evaluation.first_divergence["observed_delta_blob"]
    blob_path = (
        tmp_path
        / "lane"
        / "candidates"
        / evaluation.candidate_id
        / observed_blob["candidate_relative_path"]
    )
    assert blob_path.is_file()
    assert __import__("hashlib").sha256(blob_path.read_bytes()).hexdigest() == (
        observed_blob["sha256"]
    )
    assert lane.read_world_model()["current_model_version_id"] == (
        evaluation.active_model_version_id
    )


def test_unknown_cells_abstain_without_hiding_known_cell_support(tmp_path: Path) -> None:
    lane = ExecutableTheoryLane(tmp_path / "lane")
    lane.advance(evidence_snapshot(tmp_path, after=((0, 1), (0, 0))))
    source = IDENTITY_SOURCE.replace(
        "def render(state: State) -> np.ndarray:\n    return state.grid",
        "def render(state: State) -> np.ndarray:\n"
        "    grid = state.grid.copy()\n"
        "    grid[0, 1] = -1\n"
        "    return grid",
    )

    evaluation = lane.apply(ModelEdit.whole_file(source), provenance())

    assert evaluation.candidate_status == "active_inconclusive"
    assert evaluation.dynamics["status"] == "supported"
    assert evaluation.dynamics["known_cell_accuracy"] == 1.0
    assert evaluation.dynamics["prediction_coverage"] == 0.75


def test_game_over_action_is_replayed_before_attempt_closes(tmp_path: Path) -> None:
    root = tmp_path / "host-evidence"
    ledger = E0.HostEvidenceLedger(root, run_id="game-over-test")
    ledger.record_initial_reset(observation(((0, 0), (0, 0))))
    result = ledger.record_action_result(
        global_turn=0,
        action=E0.Action("RIGHT"),
        successor=observation(
            ((0, 1), (0, 0)), state="GAME_OVER", levels_completed=0
        ),
    )
    snapshot_path = root / "snapshots" / f"{result.snapshot.snapshot_id}.json"
    source = SUPPORTED_SOURCE.replace("level_complete", "game_over")
    lane = ExecutableTheoryLane(tmp_path / "lane")
    lane.advance(snapshot_path)

    evaluation = lane.apply(ModelEdit.whole_file(source), provenance())

    assert evaluation.dynamics["evaluated"] == 1
    assert evaluation.outcome["terminal_evidence"] == 1
    assert evaluation.outcome["status"] == "supported"


def test_supported_candidate_reports_dynamics_and_outcome_separately(
    tmp_path: Path,
) -> None:
    lane = ExecutableTheoryLane(tmp_path / "lane")
    lane.advance(
        evidence_snapshot(
            tmp_path,
            after=((0, 1), (0, 0)),
            terminal=True,
        )
    )

    evaluation = lane.apply(ModelEdit.whole_file(SUPPORTED_SOURCE), provenance())

    assert evaluation.candidate_status == "active_supported"
    assert evaluation.dynamics["status"] == "supported"
    assert evaluation.outcome["status"] == "supported"
    assert evaluation.outcome["terminal_evidence"] == 1


def test_function_edit_uses_the_same_atomic_apply_transaction(tmp_path: Path) -> None:
    lane = ExecutableTheoryLane(tmp_path / "lane")
    lane.advance(evidence_snapshot(tmp_path, after=((0, 1), (0, 0))))
    first = lane.apply(ModelEdit.whole_file(IDENTITY_SOURCE), provenance())

    second = lane.apply(
        ModelEdit.function(
            "transition",
            '''def transition(state: State, action: dict) -> State:
    grid = state.grid.copy()
    grid[0, 1] = 1
    return State(grid)
''',
        ),
        provenance(2),
    )

    assert second.previous_model_version_id == first.active_model_version_id
    assert second.active_model_version_id != first.active_model_version_id
    assert second.dynamics["matched"] == 1
    assert "grid[0, 1] = 1" in lane.read_world_model()["source"]


def test_source_cannot_import_authority_bearing_modules(tmp_path: Path) -> None:
    lane = ExecutableTheoryLane(tmp_path / "lane")
    lane.advance(evidence_snapshot(tmp_path))
    source = IDENTITY_SOURCE + "\nimport os\n"

    evaluation = lane.apply(ModelEdit.whole_file(source), provenance())

    assert evaluation.candidate_status == "rejected_operationally"
    assert evaluation.operational["failure_code"] == "PROHIBITED_IMPORT"
    assert evaluation.active_model_version_id is None


@pytest.mark.parametrize(
    ("probe", "payload"),
    [
        ("provider_credential", "\nimport os\nSECRET = os.environ.get('ARC_LLM_API_KEY')\n"),
        ("worker_process_state", "\nopen('/proc/self/status').read()\n"),
        ("inherited_ipc", "\nimport os\nos.read(3, 1)\n"),
        ("outside_bundle", "\nopen('/etc/passwd').read()\n"),
        ("overwrite_bundle", "\nopen('source.py', 'w').write('changed')\n"),
        ("network", "\nimport socket\nsocket.create_connection(('127.0.0.1', 1))\n"),
        ("spawn", "\nimport subprocess\nsubprocess.run(['true'])\n"),
        ("cpu", "\nwhile True:\n    pass\n"),
        ("memory", "\nPAYLOAD = bytearray(10_000_000_000)\n"),
        ("output", "\nprint('x' * 20_000)\n"),
        ("oversized_file", "\nopen('/tmp/large', 'wb').write(b'x' * 1_000_000)\n"),
        (
            "dynamic_reflection",
            "\nimport dataclasses\ndataclasses.sys.modules['os'].write(1, b'escape')\n",
        ),
        ("path_escape", "\nopen('../../../../etc/passwd').read()\n"),
    ],
)
def test_containment_probes_reject_before_activation(
    tmp_path: Path, probe: str, payload: str
) -> None:
    lane = ExecutableTheoryLane(tmp_path / "lane")
    lane.advance(evidence_snapshot(tmp_path))

    evaluation = lane.apply(
        ModelEdit.whole_file(IDENTITY_SOURCE + payload), provenance()
    )

    assert evaluation.candidate_status == "rejected_operationally", probe
    assert evaluation.active_model_version_id is None
    assert evaluation.operational["failure_code"]
    assert not (tmp_path / "large").exists()


def test_transition_only_hang_is_bounded_empirical_contradiction(tmp_path: Path) -> None:
    lane = ExecutableTheoryLane(tmp_path / "lane", evaluator_timeout_seconds=8.0)
    lane.advance(evidence_snapshot(tmp_path, after=((0, 1), (0, 0))))
    source = IDENTITY_SOURCE.replace(
        "def transition(state: State, action: dict) -> State:\n"
        "    return State(state.grid.copy())",
        "def transition(state: State, action: dict) -> State:\n"
        "    while True:\n"
        "        pass",
    )

    evaluation = lane.apply(ModelEdit.whole_file(source), provenance())

    assert evaluation.candidate_status == "active_contradicted"
    assert evaluation.active_model_version_id is not None
    assert evaluation.operational["status"] == "valid"
    assert evaluation.dynamics["status"] == "contradicted"
    assert evaluation.dynamics["execution_errors"][0]["error_type"] in {
        "PROCESS_TERMINATED",
        "WALL_TIME_LIMIT",
    }
