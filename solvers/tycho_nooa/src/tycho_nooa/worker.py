"""Serve one persistent Tycho-NOOA agent over the ARC-host protocol."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from tycho_nooa.agent import TychoNooaAgent
from tycho_nooa.llm import build_llm
from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.protocol import (
    HOST_PREFIX,
    PROTOCOL_VERSION,
    WORKER_PREFIX,
    ActionResultMessage,
    CloseMessage,
    FinishMessage,
    StartGameMessage,
    TurnMessage,
    parse_host_message,
)


class WorkerSession:
    """Protocol state for exactly one process-long game and agent."""

    def __init__(self) -> None:
        self.agent: TychoNooaAgent | None = None
        self.client_kind: str | None = None
        self.next_sequence = 0
        self.pending: tuple[int, ActionDecision] | None = None

    def start_game(self, message: StartGameMessage) -> dict[str, Any]:
        if self.agent is not None:
            raise RuntimeError("a game is already active in this worker")
        llm, self.client_kind = build_llm()
        self.agent = TychoNooaAgent(
            llm=llm,
            game_id=message.game_id,
            workspace_root=Path(message.workspace_root),
            available_actions=message.initial_observation.available_actions,
            tool_trace_path=Path(message.workspace_root).parent / "tool-trace.jsonl",
        )
        return {
            "worker_id": str(os.getpid()),
            "metadata": {
                "agent": "tycho_nooa.TychoNooaAgent",
                "client_kind": self.client_kind,
                "workspace": str(self.agent.workspace_dir),
            },
        }

    async def decide(self, message: TurnMessage) -> dict[str, Any]:
        agent = self._agent()
        if self.pending is not None:
            raise RuntimeError("action_result is required before the next turn")
        if message.sequence != self.next_sequence:
            raise RuntimeError(
                f"expected turn sequence {self.next_sequence}, got {message.sequence}"
            )
        previous = message.previous_decision
        observation = message.observation
        decision = await agent.act(
            GameTurn(
                grid=observation.grid,
                available_actions=observation.available_actions,
                level=message.level,
                turn_in_level=message.turn_in_level,
                levels_completed=observation.levels_completed,
                state=observation.state,
                previous_action=previous.action if previous else "",
                previous_row=previous.row if previous else None,
                previous_col=previous.col if previous else None,
            )
        )
        self.pending = (message.sequence, decision)
        return {"sequence": message.sequence, "decision": decision.model_dump()}

    def observe(self, message: ActionResultMessage) -> dict[str, Any]:
        agent = self._agent()
        if self.pending is None:
            raise RuntimeError("action_result arrived without a pending decision")
        sequence, decision = self.pending
        if message.sequence != sequence or message.decision != decision:
            raise RuntimeError("action_result does not match the pending decision")
        agent.observe(message)
        self.pending = None
        self.next_sequence += 1
        return {"sequence": message.sequence}

    def finish(self, message: FinishMessage) -> dict[str, Any]:
        self._agent()
        if self.pending is not None:
            raise RuntimeError("cannot finish before the pending action_result")
        if message.actions != self.next_sequence:
            raise RuntimeError(
                f"finish reports {message.actions} actions; worker observed {self.next_sequence}"
            )
        return {}

    def _agent(self) -> TychoNooaAgent:
        if self.agent is None:
            raise RuntimeError("start_game is required before this message")
        return self.agent


async def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    session = WorkerSession()
    while line := await asyncio.to_thread(stdin.readline):
        if not line.startswith(HOST_PREFIX):
            continue
        sequence: int | None = None
        try:
            message = parse_host_message(line[len(HOST_PREFIX) :])
            sequence = getattr(message, "sequence", None)
            if isinstance(message, CloseMessage):
                return
            if isinstance(message, StartGameMessage):
                _write(stdout, "ready", **session.start_game(message))
            elif isinstance(message, TurnMessage):
                _write(stdout, "decision", **await session.decide(message))
            elif isinstance(message, ActionResultMessage):
                _write(stdout, "observed", **session.observe(message))
            elif isinstance(message, FinishMessage):
                _write(stdout, "finished", **session.finish(message))
        except Exception as error:  # noqa: BLE001 - protocol errors must be returned
            _write(
                stdout,
                "error",
                sequence=sequence,
                error_type=type(error).__name__,
                message=str(error),
            )


def _write(stdout: TextIO, message_type: str, **fields: Any) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "type": message_type,
        **fields,
    }
    stdout.write(WORKER_PREFIX + json.dumps(payload, separators=(",", ":")) + "\n")
    stdout.flush()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
