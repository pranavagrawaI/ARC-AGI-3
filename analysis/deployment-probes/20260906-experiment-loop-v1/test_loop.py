import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "analysis/local-probes/20260905T-stall-authority-q6"),
]
from test_q6_contract import Actor, Session, chassis  # noqa: E402
from experiment_agent import ExperimentAgent  # noqa: E402
from state_delivery import StateDelivery  # noqa: E402


class LoopActor(ExperimentAgent):
    def __init__(self, root):
        Actor.__init__(self, root, False)
        self._experiment = None


def test_stale_wakeup_delivers_current_and_suppresses_duplicate():
    delivery = StateDelivery()
    first = {"turn": 0, "step": 0, "grid_rows": ["00"]}
    initial = delivery.prepare({"user_messages": ["start"]}, first)
    assert json.loads(initial["game_states"][0]) == first
    next_state = {**first, "turn": 1, "step": 1, "grid_rows": ["01"]}
    current = delivery.prepare({"game_states": [json.dumps(first)]}, next_state)
    assert json.loads(current["game_states"][0]) == next_state
    assert delivery.prepare({"game_states": [json.dumps(first)]}, next_state) is None
    assert delivery.prepare({"user_messages": ["new request"]}, next_state)[
        "user_messages"
    ] == ["new request"]


def test_wrong_prediction_requires_honest_revision_before_next_action(tmp_path):
    s = Session(tmp_path, False)
    # Use the real host and existing initialized actor attributes.
    actor = LoopActor.__new__(LoopActor)
    actor.__dict__.update(s.actor.__dict__)
    actor._experiment = None
    s.actor = actor
    with pytest.raises(chassis._ReturnResultSignal):
        actor.experiment(
            "top-right moves",
            "top-right stays",
            "UP",
            [0, 0, 3, 3],
            True,
            "UP separates motion from scenery",
        )
    assert actor.experiment_result()["status"] == "awaiting successor observation"
    assert actor.close_experiment("supported", "same", "repeat").startswith("REJECTED")
    s.execute()  # Real fixture changes (0,0), outside the predicted region.
    result = actor.experiment_result()
    assert result["prediction_matched"] is False
    assert result["changed_inside"] == [] and result["changed_outside_count"] == 1
    assert actor.experiment("a", "b", "DOWN", [0, 0, 0, 0], True, "test").startswith(
        "REJECTED"
    )
    assert actor.close_experiment("supported", "top-right moves", "repeat").startswith(
        "REJECTED"
    )
    assert actor.close_experiment("rejected", "top-right moves", "repeat").startswith(
        "REJECTED"
    )
    assert actor.close_experiment(
        "rejected", "top-right is static", "test the changed cell"
    ).startswith("Experiment closed")
    assert actor.submit_actions(["DOWN"], "predict: move").startswith("REJECTED")
    with pytest.raises(chassis._ReturnResultSignal):
        actor.experiment(
            "next pixel moves",
            "next pixel stays",
            "DOWN",
            [0, 0, 1, 1],
            True,
            "test where the next change occurs",
        )
    s.execute()
    assert actor.experiment_result()["prediction_matched"]
    assert actor.close_experiment(
        "supported", "next pixel moves", "test another direction"
    ).startswith("Experiment closed")
    events = [
        json.loads(line)
        for line in (actor.ipc_dir / "experiments.jsonl").read_text().splitlines()
    ]
    assert len([e for e in events if e["event"] == "CLOSED"]) == 2


def test_real_headless_closes_two_loops_after_current_observations(
    tmp_path, monkeypatch
):
    import asyncio
    import argparse
    import importlib.util
    from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
    from host_contract import HostEvidence

    spec = importlib.util.spec_from_file_location("loop_headless", HERE / "headless.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def reply(code, identity):
        return LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[
                ToolCall(
                    id=identity,
                    name="execute_python",
                    arguments=json.dumps({"code": code}),
                )
            ],
            finish_reason="tool_calls",
            assistant_message={"role": "assistant", "content": ""},
        )

    fake = FakeLLMClient(
        scripted_responses=[
            reply(
                "self.experiment('corner moves','corner stays','UP',[0,0,3,3],True,'distinguish controllable corner from static scenery')",
                "first",
            ),
            reply(
                "print(self.latest_state()['step']); print(self.experiment_result()); self.close_experiment('rejected','corner static; next cell controllable','test next cell'); self.experiment('next cell changes','next cell static','DOWN',[0,0,1,1],True,'test next cell')",
                "second",
            ),
            reply(
                "print(self.latest_state()['step']); print(self.experiment_result()); print(self.close_experiment('supported','next cell responds','test another direction'))",
                "third",
            ),
        ]
    )
    monkeypatch.setattr(module, "build_llm", lambda: fake)
    monkeypatch.setattr(module, "MdArcSolverAgent", ExperimentAgent)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            run_dir=str(tmp_path),
            reflect_every=8,
            visual="off",
            png_scale=8,
            skill="interactive-game-solver",
            max_actions_per_turn=20,
        ),
    )

    async def drive():
        task = asyncio.create_task(module.run())
        host = HostEvidence(tmp_path / "ipc")

        async def until(predicate):
            async with asyncio.timeout(15):
                while not predicate():
                    if task.done():
                        task.result()
                    await asyncio.sleep(0.01)

        state = {
            "grid": [[0] * 4 for _ in range(4)],
            "state": "NOT_FINISHED",
            "levels_completed": 0,
            "available_actions": ["UP", "DOWN", "LEFT", "RIGHT"],
        }
        actions = tmp_path / "ipc/actions.jsonl"

        def batches():
            return (
                [json.loads(line) for line in actions.read_text().splitlines()]
                if actions.exists()
                else []
            )

        try:
            await until(lambda: (tmp_path / "ipc/agent_ready").exists())
            host.publish(state, 0)
            for index in range(2):
                await until(lambda: len(batches()) > index)
                batch = batches()[index]
                host.begin(batch)
                grid = [row[:] for row in state["grid"]]
                grid[0][index] = 1
                after = {**state, "grid": grid}
                host.observe(state, after, batch["actions"][0], 0)
                host.publish(after, index + 1)
                state = after
            path = tmp_path / "ipc/experiments.jsonl"
            await until(
                lambda: sum(
                    json.loads(line)["event"] == "CLOSED"
                    for line in path.read_text().splitlines()
                )
                == 2
            )
            assert [row["turn"] for row in batches()] == [0, 1]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(drive())
