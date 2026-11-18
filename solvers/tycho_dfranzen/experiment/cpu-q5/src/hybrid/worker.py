"""Canonical protocol-v3 worker with explicitly selected candidate configuration."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from tycho_nooa.worker import WorkerSession, _write
from tycho_nooa.protocol import (HOST_PREFIX, StartGameMessage, TurnMessage,
    ActionResultMessage, FinishMessage, CloseMessage, parse_host_message)
from hybrid import policy


async def serve():
    # The generated worker.py imports this factory. No process-global patching
    # of the live canonical package is performed by this candidate.
    session = WorkerSession()
    while line := await asyncio.to_thread(sys.stdin.readline):
        if not line.startswith(HOST_PREFIX):
            continue
        sequence = None
        try:
            message = parse_host_message(line[len(HOST_PREFIX):])
            sequence = getattr(message, "sequence", None)
            if isinstance(message, CloseMessage):
                return
            if isinstance(message, StartGameMessage):
                policy.OBSERVATION = message.initial_observation
                payload = session.start_game(message)
                workspace = session.agent.workspace_dir
                segmentation = Path(__file__).resolve().parents[1]/"inference/utils/segmentation.py"
                (workspace/"dfranzen_helpers.py").write_text(
                    "# Modified for hybrid: numeric color labels and a numpy-compatible wrapper.\n"
                    + segmentation.read_text()
                    + '\n\ndef segment(grid):\n'
                    + '    """4-connected components; colors use Tycho numeric hex labels."""\n'
                    + '    layer = grid.tolist() if hasattr(grid, "tolist") else grid\n'
                    + '    return segment_layer(layer, "0123456789abcdef")\n')
                (workspace/"notes/hybrid_tools.md").write_text(
                    "run_python preloads dfr. Use dfr.segment(grid) for donor connected "
                    "components, adjacency, containment, shape hashes and rotation signatures. "
                    "Colors are numeric hex labels 0-f. Components are computations over "
                    "pixels, not claims about which objects or rules matter. numpy and scipy "
                    "are installed. Reuse your own Python functions by writing workspace "
                    "modules and importing them on later calls.\n")
                payload["metadata"]["decision_envelope"].update(
                    max_output_tokens_per_call=12*1024,
                    max_output_tokens_per_call_after_initial=12*1024)
                payload["metadata"]["hybrid_helper"] = "dfr.segment(grid); notes/hybrid_tools.md"
                kind = "ready"
            elif isinstance(message, TurnMessage):
                policy.OBSERVATION = message.observation
                payload = await session.decide(message)
                kind = "decision"
            elif isinstance(message, ActionResultMessage):
                policy.PREVIOUS = message.before
                policy.OBSERVATION = message.after
                policy.DEATH = message.after.terminal_frame if message.after.state == "GAME_OVER" else None
                payload = await session.observe(message)
                kind = "observed"
            elif isinstance(message, FinishMessage):
                payload, kind = session.finish(message), "finished"
            else:
                raise ValueError("unsupported protocol message")
            _write(sys.stdout, kind, hybrid=dict(policy.LEDGER), **payload)
        except Exception as error:
            _write(sys.stdout, "error", sequence=sequence,
                   error_type=type(error).__name__, message=str(error),
                   hybrid=dict(policy.LEDGER))


def main():
    asyncio.run(serve())
