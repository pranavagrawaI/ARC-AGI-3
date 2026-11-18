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
from experiment import COMMON_TASK, state, install_evidence  # noqa: E402


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

    def write_helper(self, filename, source):
        return rpc("write_helper", {"filename": filename, "source": source})

    def register(self, filename, source):
        return rpc("artifact", {"filename": filename, "source": source})


async def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
    initial = rpc("initial")
    arm, rows = initial["arm"], initial["rows"]
    run = Path("/work")
    # Explicitly non-authoritative scratch projection; all evidence getters read /evidence.
    install_evidence(run, rows)
    latest = state(rows[-1]["after"], rows[-1]["step"], rows[-1]["step"])

    def model(body):
        if (run / "ipc/ABORT").exists():
            raise RuntimeError("worker evidence/isolation failure")
        return rpc("model", body)

    client = ActorClient(model)
    from prefill import install

    install(client.ids)
    from cell_state import install as install_cell_state, CONTRACT

    install_cell_state(client.ids)
    kwargs = dict(
        llm=client,
        run_dir=run,
        game_id="the game",
        alias="the game",
        skill_path=RUNTIME / "chassis/skills/interactive-game-solver/SKILL.md",
        enable_summarizer=False,
        max_actions_per_turn=20,
    )
    from evidence_api import install_tail

    install_tail(rpc)
    agent = (
        OperationalActor(**kwargs, operational_store=RemoteStore())
        if arm == "treatment"
        else MdArcSolverAgent(**kwargs)
    )
    from evidence_api import attach

    attach(agent)
    from nooa.context_blocks import Context

    if arm == "control":
        agent.context["python_state_contract"] = Context(CONTRACT, prefix=True)
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
        if (run / "ipc/ABORT").exists():
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
            latest = state(rows[-1]["after"], rows[-1]["step"], rows[-1]["step"])
    # Join broker-using producers before telling the host to stop servicing RPC.
    await agent.queue_manager.shutdown()
    rpc("done")


if __name__ == "__main__":
    asyncio.run(main())
