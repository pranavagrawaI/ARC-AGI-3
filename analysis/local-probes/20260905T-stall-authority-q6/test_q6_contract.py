"""Scripted real submission methods, real host evidence, independent scoring."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "patched-source/after"))
import solver_agent as chassis  # noqa: E402
from host_contract import HostEvidence  # noqa: E402
from stall_contract import StallReview, append, read_rows, detect_stall  # noqa: E402

spec = importlib.util.spec_from_file_location("score_q6", HERE / "score_q6.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


class Actor(chassis.ArcSolverBase):
    """Run production submission methods without starting model/queue services."""

    def __init__(self, run: Path, treatment: bool):
        self.ipc_dir = run / "ipc"
        self.ipc_dir.mkdir(parents=True)
        self._states_path = self.ipc_dir / "states.jsonl"
        self._actions_path = self.ipc_dir / "actions.jsonl"
        self._actions_path.touch()
        self._observations_path = self.ipc_dir / "observations.json"
        self._rejections_path = self.ipc_dir / "rejections.jsonl"
        self._stall = StallReview(self.ipc_dir, treatment)
        self._effort_ladder = []
        self._max_actions_per_turn = 20
        self.knowledge_dir = run / "team_nemo/shared/knowledge"
        self.knowledge_dir.mkdir(parents=True)


class Session:
    def __init__(self, root, treatment=True):
        self.root = root
        self.actor = Actor(root, treatment)
        self.host = HostEvidence(self.actor.ipc_dir)
        self.state = {
            "grid": [[0] * 4 for _ in range(4)],
            "levels_completed": 0,
            "state": "NOT_FINISHED",
            "available_actions": ["UP", "DOWN", "LEFT", "RIGHT"],
        }
        self.host.publish(self.state, 0)
        (root / "launch.json").write_text(
            json.dumps(
                {
                    "arm": "treatment" if treatment else "control",
                    "qualification": True,
                    "game_id": "ls20",
                    "platform": "scripted",
                    "contract_sha256": "test",
                }
            )
        )
        append(self.actor.ipc_dir / "integrity.jsonl", {"event": "GUARD_INSTALLED"})

    def submit(self, actions, rationale="go up to x"):
        try:
            result = self.actor.submit_actions(
                actions, rationale, predicted_path=[{"expect": "move"}] * len(actions)
            )
        except chassis._ReturnResultSignal:
            return None
        return result

    def execute(self, inert=False):
        batch = read_rows(self.actor._actions_path)[-1]
        self.host.begin(batch)
        for i, action in enumerate(batch["actions"]):
            grid = [r[:] for r in self.state["grid"]]
            if not inert:
                step = self.host.step
                grid[(step // 4) % 4][step % 4] = (
                    grid[(step // 4) % 4][step % 4] + 1
                ) % 16
            after = {**self.state, "grid": grid}
            stop = self.host.observe(
                self.state, after, action, len(batch["actions"]) - i - 1
            )
            self.state = after
            if stop:
                break
        self.host.publish(self.state, batch["turn"] + 1)

    def seed(self):
        for _ in range(6):
            assert self.submit(["UP"] * 5) is None
            self.execute()


def test_control_logs_same_detection_without_review_or_suspension(tmp_path):
    s = Session(tmp_path, False)
    s.seed()
    assert s.submit(["UP"]) is None
    rows = read_rows(s.actor.ipc_dir / "authority.jsonl")
    assert [r["event"] for r in rows] == ["STALL_DETECTED"]
    assert rows[0]["step"] == 30
    assert s.actor._stall.opened is None


def test_treatment_single_action_and_four_event_trace_are_scored_from_frames(tmp_path):
    s = Session(tmp_path)
    s.seed()
    assert s.submit(["UP"] * 5).startswith("REJECTED:")
    assert len(read_rows(s.actor._actions_path)) == 6
    with pytest.raises(chassis._ReturnResultSignal):
        s.actor.resolve_stall(
            "continue",
            ["old"],
            ["old"],
            "DOWN",
            "test direction",
            abandon_if="no_impact",
        )
    assert read_rows(s.actor._actions_path)[-1]["actions"] == ["DOWN"]
    s.execute(inert=True)
    assert s.submit(["UP"]).startswith("REJECTED:")
    assert "recorded" in s.actor.finish_stall(
        ["new"], "inert action falsified the old basis"
    )
    events = read_rows(s.actor.ipc_dir / "authority.jsonl")
    assert [r["event"] for r in events] == [
        "STALL_DETECTED",
        "AUTHORITY_REVIEWED",
        "CONSEQUENCE_OBSERVED",
        "AUTHORITY_RESOLVED",
    ]
    result = scorer.score_run(tmp_path)
    assert result["status"] == "SCORED", result
    assert result["P12"] == 1
    assert result["P11_zero_latency"] is True
    assert result["P9"] is None and result["events"][0]["censored"]
    # Verdict tampering cannot turn the measured changed basis into retained.
    for row in events:
        row.update(
            actual="move",
            abandon_if_satisfied=False,
            resolution="retained",
            stall_escaped=True,
        )
    (s.actor.ipc_dir / "authority.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in events)
    )
    result = scorer.score_run(tmp_path)
    assert result["P12"] == 1 and result["P9"] is None


def test_rejected_resolution_keeps_stall_open_and_aborts_at_eight(tmp_path):
    s = Session(tmp_path)
    s.seed()
    s.submit(["UP"])
    for _ in range(8):
        assert s.actor.resolve_stall("probe", ["x"], ["x"], "UP", "testing").startswith(
            "REJECTED:"
        )
    assert s.actor._stall.opened is not None
    assert len(read_rows(s.actor._actions_path)) == 6
    assert (s.actor.ipc_dir / "ABORT").exists()


def test_long_bases_and_probe_predictions_are_preserved(tmp_path):
    s = Session(tmp_path)
    s.seed()
    s.submit(["UP"])
    old = ["x" * 600 + "A"]
    new = ["x" * 600 + "B"]
    with pytest.raises(chassis._ReturnResultSignal):
        s.actor.resolve_stall("switch", old, new, "DOWN", "switch")
    row = read_rows(s.actor.ipc_dir / "authority.jsonl")[-1]
    assert row["previous_basis"] == old and row["active_basis"] == new
    assert row["previous_basis"] != row["active_basis"]


def test_probe_requires_distinct_predictions_and_respects_existing_reentry_gate(
    tmp_path,
):
    s = Session(tmp_path)
    s.seed()
    s.submit(["UP"])
    assert s.actor.resolve_stall(
        "probe", ["x"], ["y"], "UP", "probe", predictions={"a": "same", "b": "same"}
    ).startswith("REJECTED:")
    with pytest.raises(chassis._ReturnResultSignal):
        s.actor.resolve_stall(
            "probe",
            ["x"],
            ["y"],
            "DOWN",
            "probe",
            predictions={"a": "move", "b": "no_impact"},
        )
    s.execute(True)
    s.actor.finish_stall(["y"], "still testing")
    # The stalled policy review does not erase known no-impact evidence.
    assert "recorded" in s.submit(
        ["DOWN"], rationale="entirely different unrelated rationale"
    )


def test_scorer_refuses_missing_frames_and_forged_detection(tmp_path):
    s = Session(tmp_path, False)
    s.seed()
    s.submit(["UP"])
    s.execute()
    assert scorer.score_run(tmp_path)["status"] == "SCORED"
    path = s.actor.ipc_dir / "authority.jsonl"
    events = read_rows(path)
    events[0]["step"] = 29
    path.write_text("".join(json.dumps(r) + "\n" for r in events))
    assert scorer.score_run(tmp_path)["status"] == "INVALID"
    (s.actor.ipc_dir / "transitions.jsonl").unlink()
    assert scorer.score_run(tmp_path)["status"] == "INVALID"


def test_detector_matches_historical_replay_at_every_operating_point():
    replay_spec = importlib.util.spec_from_file_location(
        "replay", HERE.parents[1] / "discrimination-replay/replay.py"
    )
    replay = importlib.util.module_from_spec(replay_spec)
    sys.modules["replay"] = replay
    replay_spec.loader.exec_module(replay)
    run = replay.Run(
        HERE,
        [replay.Turn(i * 6, i, ["UP"], "go up to x") for i in range(10)],
        [
            replay.State(i * 6, i, 0, 0, ["UP", "DOWN"], "NOT_FINISHED")
            for i in range(10)
        ],
        {},
        {},
        [],
        [],
        [],
    )
    for window in (4, 6, 8):
        for floor in (15, 25, 40):
            for similarity in (0.45, 0.55, 0.65):
                got = []
                for i, turn in enumerate(run.turns):
                    history = [
                        {
                            "rationale": r.rationale,
                            "actions": r.actions,
                            "levels_completed": 0,
                        }
                        for r in run.turns[:i]
                    ]
                    state = {
                        "turn": i,
                        "step": turn.step,
                        "levels_completed": 0,
                        "available_actions": ["UP", "DOWN"],
                    }
                    if detect_stall(
                        history,
                        state,
                        turn.rationale,
                        window=window,
                        min_actions=floor,
                        similarity=similarity,
                    ):
                        got.append(turn.step)
                expected = [
                    r.step
                    for r in replay.trigger_stall_repetition(
                        run, window=window, min_actions=floor, similarity=similarity
                    )
                ]
                assert got == expected


def test_terminal_consequence_is_host_recorded_without_another_actor_call(tmp_path):
    s = Session(tmp_path)
    s.seed()
    s.submit(["UP"])
    with pytest.raises(chassis._ReturnResultSignal):
        s.actor.resolve_stall(
            "continue", ["old"], ["old"], "DOWN", "test", abandon_if="terminal"
        )
    batch = read_rows(s.actor._actions_path)[-1]
    s.host.begin(batch)
    after = {**s.state, "state": "WIN", "levels_completed": 1}
    assert s.host.observe(s.state, after, "DOWN", 0)
    s.host.publish(after, 7)
    assert (
        read_rows(s.actor.ipc_dir / "authority.jsonl")[-1]["event"]
        == "CONSEQUENCE_OBSERVED"
    )
    score = scorer.score_run(tmp_path)
    assert score["status"] == "SCORED", score
    assert score["events"][0]["authority_censored"]
    assert score["P12"] == 0  # No unobserved authoring is credited as a change.


def test_missing_attempt_and_nonzero_latency_cannot_score(tmp_path):
    s = Session(tmp_path)
    s.seed()
    s.submit(["UP"])
    with pytest.raises(chassis._ReturnResultSignal):
        s.actor.resolve_stall("switch", ["old"], ["new"], "DOWN", "switch")
    s.execute()
    s.actor.finish_stall(["new"], "retained")
    path = s.actor.ipc_dir / "authority.jsonl"
    events = read_rows(path)
    review = next(r for r in events if r["event"] == "AUTHORITY_REVIEWED")
    review["step"] += 1
    path.write_text("".join(json.dumps(r) + "\n" for r in events))
    assert scorer.score_run(tmp_path)["status"] == "INVALID"
    review["step"] -= 1
    path.write_text("".join(json.dumps(r) + "\n" for r in events))
    attempts = read_rows(s.actor.ipc_dir / "policy_attempts.jsonl")
    (s.actor.ipc_dir / "policy_attempts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in attempts[:-1])
    )
    assert scorer.score_run(tmp_path)["status"] == "INVALID"


def test_scribe_receives_winning_frames_before_next_level_submission(tmp_path):
    s = Session(tmp_path, False)
    assert s.submit(["UP"]) is None
    s.host.begin(read_rows(s.actor._actions_path)[-1])
    after = {**s.state, "grid": [[1] * 4 for _ in range(4)], "levels_completed": 1}
    assert s.host.observe(s.state, after, "UP", 0)
    s.host.publish(after, 1)
    evidence = s.actor.level_terminal(0)
    assert evidence["pre_win_grid_rows"] == ["0000"] * 4
    assert evidence["terminal_grid_rows"] == ["1111"] * 4
    assert evidence["winning_action"] == "UP"
    assert "COMPLETED level 0" in s.submit(["DOWN"])
    assert "COMPLETED level 0" in s.submit(["DOWN"])
    assert s.submit(["DOWN"]) is None
    note = (s.actor.knowledge_dir / "level_0_insights.md").read_text()
    assert "winning action: 'UP'" in note and "None" not in note


def test_game_scoring_retains_episode_denominators(tmp_path):
    for index in range(2):
        session = Session(tmp_path / f"episode-{index}", False)
        session.seed()
        session.submit(["UP"])
        session.execute()
    score = scorer.score_game(tmp_path)
    assert score["status"] == "SCORED", score
    assert score["session_count"] == 2
    assert score["steps"] == 62 and score["stall_events"] == 2
    assert score["P9"] is None  # Neither session supplies K follow-up actions.
    assert score["holds"]["counts"]["executed"] == 62


def test_arm_rejects_duplicate_sessions_and_different_games():
    def record(name, arm, game="ls20"):
        return {
            "run": name,
            "run_id": name,
            "session_id": name + "-session",
            "status": "SCORED",
            "qualification": False,
            "arm": arm,
            "game_id": game,
            "platform": "kaggle",
            "model_config": {"ARC_LLM_MODEL": "fixed"},
            "contract_sha256": "fixed",
            "evaluable_events": 3,
            "P9": 0 if arm == "control" else 1,
            "P10": None,
            "P12": None,
            "P11_zero_latency": True,
            "holds": {"status": "PASS"},
        }

    c = record("/tmp/c", "control")
    t = record("/tmp/t", "treatment")
    assert scorer.score_arm([c] * 3 + [t] * 3)["status"] == "INVALID"
    runs = [record(f"/tmp/c{i}", "control") for i in range(3)] + [
        record(f"/tmp/t{i}", "treatment", "cd82") for i in range(3)
    ]
    assert scorer.score_arm(runs)["status"] == "INVALID"
    for run in runs:
        run["game_id"] = "ls20"
    assert (
        scorer.score_arm(runs)["status"] == "PASS"
    )  # No continue dispositions is N/A.
    original_id = runs[-1]["run_id"]
    runs[-1]["run_id"] = runs[0]["run_id"]
    assert scorer.score_arm(runs)["status"] == "INVALID"
    runs[-1]["run_id"] = original_id
    runs[-1]["session_id"] = runs[0]["session_id"]
    assert scorer.score_arm(runs)["status"] == "INVALID"
    runs[-1]["session_id"] = original_id + "-session"
    runs[-1]["P11_zero_latency"] = False
    assert scorer.score_arm(runs)["status"] == "INVALID"
