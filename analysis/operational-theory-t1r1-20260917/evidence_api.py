"""Shared read-only evidence backing for unchanged incumbent getters.

The broker projects only admitted rows into /evidence, outside all writable mounts.
Writable IPC retains actions/diagnostics; scratch filenames are never authoritative.
"""

from pathlib import Path
from nooa import hidden


@hidden
class EvidenceIPC(type(Path())):
    def __truediv__(self, key):
        if str(key) in ("states.jsonl", "transitions.jsonl", "observations.json"):
            return Path("/evidence") / key
        return super().__truediv__(key)


def attach(agent):
    import solver_agent

    # Extend the incumbent read jail to read-only mounts, before producers run.
    solver_agent._JAIL_ROOTS.append(Path("/evidence"))
    if Path("/helpers").exists():
        solver_agent._JAIL_ROOTS.append(Path("/helpers"))
    agent.ipc_dir = EvidenceIPC(agent.ipc_dir)
    agent._states_path = Path("/evidence/states.jsonl")
    agent._observations_path = Path("/evidence/observations.json")
    # Prediction/evidence consumers retain the same IPC API, with authoritative reads.
    agent._model_feedback.ipc = agent.ipc_dir


def install_tail():
    # Redirect the producer at creation, before its first read; scratch is never queued.
    import solver_agent

    original = solver_agent._tail_from_start

    def authoritative_tail(path, poll_interval=0.5):
        if Path(path).name == "states.jsonl":
            path = "/evidence/states.jsonl"
        return original(path, poll_interval)

    solver_agent._tail_from_start = authoritative_tail
