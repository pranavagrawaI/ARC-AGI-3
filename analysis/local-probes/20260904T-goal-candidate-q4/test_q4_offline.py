#!/usr/bin/env python3
"""Offline checks frozen before the paid q4 run. No model calls."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROBE = Path(__file__).resolve().parent
NOOA = Path("/home/pranav/projects/arc-agi-3-reference/nooa/examples/arc_agi_3")
sys.path.insert(0, str(PROBE))
sys.path.insert(0, str(NOOA))
import harness as H  # noqa: E402


def test_dynamics_only_path_is_not_a_discriminator():
    path = [{"expect": "move"}] * 19
    assert not H.path_has_terminal_expect(path)


def test_terminal_class_path_is_a_discriminator():
    path = [{"expect": "move"}] * 3 + [{"expect": "level_up"}]
    assert H.path_has_terminal_expect(path)


def test_truncated_before_terminal_action_is_not_scored():
    path = [{"expect": "move"}, {"expect": "level_up"}]
    results = [{"outcome": "move"}]  # batch truncated after action 0
    assert H.score_terminal_expects(path, results) is None


def test_missed_terminal_expect_contradicts():
    ipc = Path(tempfile.mkdtemp())
    H.open_goal_candidate(
        ipc, level=1, opened_at_step=32, deadline_actions=32, transferred_from=0
    )
    rec = H.load_goal_candidate(ipc)
    assert rec["status"] == "proposed"
    note = H.update_goal_candidate_after_batch(
        ipc,
        actions=["UP"] * 4,
        predicted_path=[{"expect": "move"}] * 3 + [{"expect": "level_up"}],
        results=[{"outcome": "move"}] * 4,
        level_action_count=4,
    )
    rec = H.load_goal_candidate(ipc)
    assert rec["status"] == "contradicted", rec
    assert "CONTRADICTED" in note
    assert rec["score"]["terminal_hit"] is False


def test_matched_terminal_expect_promotes():
    ipc = Path(tempfile.mkdtemp())
    H.open_goal_candidate(
        ipc, level=1, opened_at_step=32, deadline_actions=32, transferred_from=0
    )
    H.update_goal_candidate_after_batch(
        ipc,
        actions=["UP"],
        predicted_path=[{"expect": "level_up"}],
        results=[{"outcome": "level_up"}],
        level_action_count=1,
    )
    rec = H.load_goal_candidate(ipc)
    assert rec["status"] == "promoted", rec


def test_deadline_expires_without_discriminator():
    ipc = Path(tempfile.mkdtemp())
    H.open_goal_candidate(
        ipc, level=1, opened_at_step=32, deadline_actions=16, transferred_from=0
    )
    H.update_goal_candidate_after_batch(
        ipc,
        actions=["LEFT"],
        predicted_path=[{"expect": "move"}],
        results=[{"outcome": "move"}],
        level_action_count=16,
    )
    rec = H.load_goal_candidate(ipc)
    assert rec["status"] == "expired", rec
    assert H.CANDIDATE_FALLBACK in rec["fallback"]
    log = (ipc / "goal_candidate.jsonl").read_text()
    assert "expired" in log


def test_all_move_batch_does_not_bind_discriminator():
    ipc = Path(tempfile.mkdtemp())
    H.open_goal_candidate(
        ipc, level=1, opened_at_step=32, deadline_actions=32, transferred_from=0
    )
    H.update_goal_candidate_after_batch(
        ipc,
        actions=["UP"] * 19,
        predicted_path=[{"expect": "move"}] * 19,
        results=[{"outcome": "move"}] * 19,
        level_action_count=19,
    )
    rec = H.load_goal_candidate(ipc)
    assert rec["discriminator"] is None
    assert rec["status"] == "proposed"


def test_deadline_formula():
    assert max(H.CANDIDATE_DEADLINE_FLOOR, 32) == 32
    assert max(H.CANDIDATE_DEADLINE_FLOOR, 5) == 16


def test_long_level_up_plan_is_not_a_discriminator():
    """q3c t16: 11 actions ending in level_up. A plan, not a probe."""
    actions = ["UP"] * 11
    path = [{"expect": "move"}] * 10 + [{"expect": "level_up"}]
    assert not H.is_discriminator_batch(actions, path)
    ipc = Path(tempfile.mkdtemp())
    H.open_goal_candidate(
        ipc, level=1, opened_at_step=32, deadline_actions=32, transferred_from=0
    )
    H.update_goal_candidate_after_batch(
        ipc,
        actions=actions,
        predicted_path=path,
        results=[{"outcome": "move"}] * 11,
        level_action_count=11,
    )
    rec = H.load_goal_candidate(ipc)
    assert rec["discriminator"] is None
    assert rec["status"] == "proposed"


def test_short_level_up_probe_is_a_discriminator():
    actions = ["DOWN"]
    path = [{"expect": "level_up"}]
    assert H.is_discriminator_batch(actions, path)


def test_q3c_incumbent_fails_p6_and_p7():
    """The scorer must not pass the run we are contrasting against."""
    import score_q4 as S

    run = Path(
        "/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/"
        "tall-humerus/analysis/local-probes/20260904T-host-control-loop-q3/"
        "results/nemo_solver/20260904_140403_ls20_mdfiles_q3c"
    )
    s = S.score(run)
    assert s.get("P6_discriminator_before_deadline") == 0.0, s.get("P6_detail")
    assert s.get("P7_revision_after_miss_or_expiry") == 0.0, s.get("P7_detail")
    assert s.get("P1_search_batches") >= 0.5
    assert s.get("P2_scribe") == 1.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            failed += 1
            print("FAIL", fn.__name__, type(e).__name__, e)
    sys.exit(1 if failed else 0)
