from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from theory_guidance import (
    COMPLETE_PRACTICE_EXAMPLES,
    GUIDE_SCHEMA,
    MOORE_PROMOTION_SOURCE,
    NARROW_PARTIAL_SOURCE,
    render_practice_guide,
)
from world_model_lane import EditProvenance, ExecutableTheoryLane, ModelEdit

EXPERIMENT = Path(__file__).resolve().parents[2]
E0_PATH = EXPERIMENT / "e0_evidence.py"
E0_SPEC = importlib.util.spec_from_file_location("wm2_guide_test_e0", E0_PATH)
assert E0_SPEC is not None and E0_SPEC.loader is not None
E0 = importlib.util.module_from_spec(E0_SPEC)
sys.modules[E0_SPEC.name] = E0
E0_SPEC.loader.exec_module(E0)


def test_practice_guide_teaches_use_recovery_promotion_and_deferral() -> None:
    rendered = render_practice_guide()

    assert len(rendered.encode("utf-8")) <= 3_072
    assert "preserve a rule" in rendered
    assert "claim only" in rendered
    assert "first divergence" in rendered
    assert "Moore machine" in rendered
    assert "return `None`" in rendered
    assert "You may defer" in rendered
    assert "greater coverage is not necessarily better" in rendered
    assert "LS20" not in rendered
    assert "[0, 0]" not in rendered


def test_complete_examples_include_executable_and_nonuse_lifecycles() -> None:
    assert GUIDE_SCHEMA == "0159-theory-practice-guide-v1"
    assert set(COMPLETE_PRACTICE_EXAMPLES) == {
        "narrow_partial_predictor",
        "abstention",
        "contradiction_and_revision",
        "moore_promotion",
        "legitimate_nonuse",
    }
    assert "def predict" in COMPLETE_PRACTICE_EXAMPLES["narrow_partial_predictor"]
    assert "[0, 0]" not in COMPLETE_PRACTICE_EXAMPLES["narrow_partial_predictor"]
    assert "operationally valid" in COMPLETE_PRACTICE_EXAMPLES[
        "contradiction_and_revision"
    ]
    assert "competing hypotheses" in COMPLETE_PRACTICE_EXAMPLES[
        "legitimate_nonuse"
    ]
    assert "state.phase >= 2" in COMPLETE_PRACTICE_EXAMPLES["moore_promotion"]


def test_executable_guide_examples_pass_the_real_broker_and_evaluator(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    ledger = E0.HostEvidenceLedger(root, run_id="guide-example")
    observation = E0.PublicObservationInput(
        frames=(((0, 0), (0, 0)),),
        public_state="NOT_FINISHED",
        levels_completed=0,
        full_reset=False,
        legal_actions=("UP", "DOWN", "LEFT", "RIGHT"),
    )
    ledger.record_initial_reset(observation)
    snapshot = ledger.record_action_result(
        global_turn=0,
        action=E0.Action("LEFT"),
        successor=observation,
    ).snapshot
    snapshot_path = root / "snapshots" / f"{snapshot.snapshot_id}.json"
    lane = ExecutableTheoryLane(tmp_path / "lane")
    lane.advance(snapshot_path)

    partial = lane.apply(
        ModelEdit.whole_file(NARROW_PARTIAL_SOURCE),
        EditProvenance("guide-example:partial"),
    )
    moore = lane.apply(
        ModelEdit.whole_file(MOORE_PROMOTION_SOURCE),
        EditProvenance("guide-example:moore"),
    )

    assert partial.operational["status"] == "valid"
    assert partial.operational["contract_kind"] == "partial_predictor"
    assert partial.candidate_status == "active_supported"
    assert moore.operational["status"] == "valid"
    assert moore.operational["contract_kind"] == "moore_machine"
    assert moore.candidate_status == "active_inconclusive"
