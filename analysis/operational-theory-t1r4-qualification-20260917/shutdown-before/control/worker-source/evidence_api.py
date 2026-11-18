"""All authoritative reads use the custody broker; no backing files are mounted."""

import asyncio
import io
from pathlib import Path
from nooa import hidden

_READER = None


@hidden
class EvidencePath(type(Path())):
    def read_text(self, encoding=None, errors=None):
        value = _READER("evidence", self.name)
        if value is None:
            raise FileNotFoundError(str(self))
        return value

    def read_bytes(self):
        return self.read_text().encode()

    def exists(self):
        return _READER("evidence", self.name) is not None

    def open(self, mode="r", *args, **kwargs):
        if mode not in ("r", "rt", "rb"):
            raise PermissionError("authoritative evidence is read-only")
        return (
            io.BytesIO(self.read_bytes())
            if "b" in mode
            else io.StringIO(self.read_text())
        )


@hidden
class EvidenceIPC(type(Path())):
    def __truediv__(self, key):
        if str(key) in (
            "states.jsonl",
            "transitions.jsonl",
            "observations.json",
            "catalog.json",
        ):
            return EvidencePath("/evidence") / key
        return super().__truediv__(key)


def attach(agent):
    import solver_agent

    solver_agent._JAIL_ROOTS.append(Path("/evidence"))
    if Path("/helpers").exists():
        solver_agent._JAIL_ROOTS.append(Path("/helpers"))
    agent.ipc_dir = EvidenceIPC(agent.ipc_dir)
    agent._states_path = EvidencePath("/evidence/states.jsonl")
    agent._observations_path = EvidencePath("/evidence/observations.json")
    agent._model_feedback.ipc = agent.ipc_dir


def install_tail(reader):
    global _READER
    _READER = reader
    import solver_agent

    original = solver_agent._tail_from_start
    original_open = solver_agent._REAL_OPEN

    def verified_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, Path)) and str(Path(file).absolute()).startswith(
            "/evidence/"
        ):
            return EvidencePath(file).open(mode, *args, **kwargs)
        return original_open(file, mode, *args, **kwargs)

    solver_agent._REAL_OPEN = verified_open

    async def authoritative_tail(path, poll_interval=0.5):
        if Path(path).name != "states.jsonl":
            async for line in original(path, poll_interval):
                yield line
            return
        position = 0
        while True:
            lines = EvidencePath("/evidence/states.jsonl").read_text().splitlines()
            for line in lines[position:]:
                yield line
            position = len(lines)
            await asyncio.sleep(poll_interval)

    solver_agent._tail_from_start = authoritative_tail
