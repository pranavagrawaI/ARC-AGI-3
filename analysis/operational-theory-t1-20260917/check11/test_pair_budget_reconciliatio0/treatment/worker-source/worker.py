"""Sterile Actor process: no network, credentials, future evidence or host store."""

import asyncio
import json
import resource
import sys
from pathlib import Path

# Protocol uses inherited pipes; arbitrary Actor prints go to diagnostic stderr.
READER, WRITER = sys.stdin, sys.stdout
sys.stdout = sys.stderr


def rpc(operation, payload=None):
    WRITER.write(
        json.dumps({"op": operation, "payload": payload}, allow_nan=False) + "\n"
    )
    WRITER.flush()
    response = json.loads(READER.readline())
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["value"]


import bootstrap  # noqa: E402,F401
from bootstrap import RUNTIME  # noqa: E402
from actor import MdArcSolverAgent, OperationalActor  # noqa: E402
from client import ActorClient  # noqa: E402
from experiment import COMMON_TASK, install_evidence, protected_snapshot  # noqa: E402


from nooa import hidden  # noqa: E402


@hidden
class RemoteStore:
    def call(self, operation, payload=None):
        return rpc("theory", {"operation": operation, "payload": payload})

    @property
    def phase(self):
        return self.call("read")["phase"]

    @property
    def pending(self):
        return self.call("read")["pending"]

    @property
    def current(self):
        return self.call("read")["theory"]

    def feedback(self, message):
        return rpc("feedback", message)

    def register(self, filename, source):
        return rpc("artifact", {"filename": filename, "source": source})


async def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
    initial = rpc("initial")
    arm, rows = initial["arm"], initial["rows"]
    run = Path("/work")
    latest = install_evidence(run, rows)
    checks = protected_snapshot(run)

    def model(body):
        if checks != protected_snapshot(run) or (run / "ipc/ABORT").exists():
            raise RuntimeError("worker evidence/isolation failure")
        return rpc("model", body)

    kwargs = dict(
        llm=ActorClient(model),
        run_dir=run,
        game_id="the game",
        alias="the game",
        skill_path=RUNTIME / "chassis/skills/interactive-game-solver/SKILL.md",
        enable_summarizer=False,
        max_actions_per_turn=20,
    )
    agent = (
        OperationalActor(**kwargs, operational_store=RemoteStore())
        if arm == "treatment"
        else MdArcSolverAgent(**kwargs)
    )
    for turn in range(2):
        guidance = (
            COMMON_TASK
            if turn == 0
            else (
                "NEW RECORDED OBSERVATION transition:2. Historical action DOWN; this "
                "is a recorded continuation, not the outcome of your proposed action. "
                "Reconcile it in your existing knowledge and prepare the next decision."
            )
        )
        await agent.handle(
            {"game_states": [json.dumps(latest)], "user_messages": [guidance]}
        )
        if checks != protected_snapshot(run) or (run / "ipc/ABORT").exists():
            raise RuntimeError("worker evidence/isolation failure")
        actions = [
            json.loads(s) for s in (run / "ipc/actions.jsonl").read_text().splitlines()
        ]
        if len(actions) != turn + 1:
            rpc("incomplete", {"reason": "handle ended without one proposal"})
            break
        next_state = rpc("proposed", actions[-1])
        if turn == 0:
            rows = next_state["rows"]
            latest = install_evidence(run, rows)
            checks = protected_snapshot(run)
    rpc("done")


if __name__ == "__main__":
    asyncio.run(main())
