"""Serve one persistent Tycho-NOOA agent over the ARC-host protocol."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from tycho_nooa.action_batch import MAX_ACTIONS_PER_BATCH
from tycho_nooa.agent import TychoNooaAgent
from tycho_nooa.llm import (
    build_llm,
    generation_action_demand_disable_thinking,
    generation_action_demand_max_tokens,
    generation_force_action_tool,
    generation_max_actor_calls,
    generation_max_actor_calls_after_initial,
    generation_max_tokens,
    generation_max_tokens_after_initial,
    generation_retries,
    generation_timeout_seconds,
)
from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.protocol import (
    HOST_PREFIX,
    PROTOCOL_VERSION,
    WORKER_PREFIX,
    ActionResultMessage,
    CloseMessage,
    FinishMessage,
    ObservationPayload,
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
        self.pending: tuple[TurnMessage, ActionDecision] | None = None
        self.previous_result: ActionResultMessage | None = None
        self.initial_observation: ObservationPayload | None = None
        self.finished = False

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
        self.initial_observation = message.initial_observation
        return {
            "worker_id": str(os.getpid()),
            "metadata": {
                "agent": "tycho_nooa.TychoNooaAgent",
                "client_kind": self.client_kind,
                "decision_envelope": {
                    "force_action_tool": generation_force_action_tool(),
                    "max_actor_calls": generation_max_actor_calls(),
                    "max_actor_calls_after_initial": (
                        generation_max_actor_calls_after_initial()
                    ),
                    "max_output_tokens_per_call": generation_max_tokens(),
                    "max_output_tokens_per_call_after_initial": (
                        generation_max_tokens_after_initial()
                    ),
                    "action_demand_max_output_tokens": (
                        generation_action_demand_max_tokens()
                    ),
                    "action_demand_disable_thinking": (
                        generation_action_demand_disable_thinking()
                    ),
                    "model_retries": generation_retries(),
                    "model_timeout_seconds": generation_timeout_seconds(),
                    "max_actions_per_batch": MAX_ACTIONS_PER_BATCH,
                },
                "workspace": str(self.agent.workspace_dir),
                "mode": self.agent.mode_name,
                "treatment": self.agent.treatment_name,
                "level_boundary_policy": self.agent.level_boundary_policy,
                "sandbox_runtime": os.environ.get("TYCHO_SANDBOX_RUNTIME", "auto"),
            },
        }

    async def decide(self, message: TurnMessage) -> dict[str, Any]:
        agent = self._active_agent()
        if self.pending is not None:
            raise RuntimeError("action_result is required before the next turn")
        if message.sequence != self.next_sequence:
            raise RuntimeError(
                f"expected turn sequence {self.next_sequence}, got {message.sequence}"
            )
        if self.previous_result is None:
            if self.initial_observation is None:
                raise RuntimeError("turn arrived before the initial observation")
            if (
                message.observation != self.initial_observation
                or message.previous_decision is not None
            ):
                raise RuntimeError("first turn does not match the initial observation")
        else:
            crossed_boundary = (
                self.previous_result.after.full_reset
                or self.previous_result.after.levels_completed
                > self.previous_result.before.levels_completed
            )
            expected_previous = (
                None if crossed_boundary else self.previous_result.decision
            )
            if (
                message.observation != self.previous_result.after
                or message.previous_decision != expected_previous
            ):
                raise RuntimeError("turn does not continue the previous action result")
        previous = message.previous_decision
        observation = message.observation
        decision = await agent.act(
            GameTurn(
                grid=observation.grid,
                available_actions=observation.available_actions,
                sequence=message.sequence,
                action_limit=message.action_limit,
                actions_used=message.actions_used,
                actions_remaining=message.actions_remaining,
                final_permitted_action=message.final_permitted_action,
                level=message.level,
                turn_in_level=message.turn_in_level,
                levels_completed=observation.levels_completed,
                state=observation.state,
                previous_action=previous.action if previous else "",
                previous_row=previous.row if previous else None,
                previous_col=previous.col if previous else None,
            ),
            decision_timeout_seconds=message.decision_timeout_seconds,
        )
        self.pending = (message, decision)
        return {"sequence": message.sequence, "decision": decision.model_dump()}

    async def observe(self, message: ActionResultMessage) -> dict[str, Any]:
        agent = self._active_agent()
        if self.pending is None:
            raise RuntimeError("action_result arrived without a pending decision")
        turn, decision = self.pending
        if (
            message.sequence != turn.sequence
            or message.level != turn.level
            or message.turn_in_level != turn.turn_in_level
            or message.before != turn.observation
            or message.decision != decision
        ):
            raise RuntimeError(
                "action_result does not match the pending turn and decision"
            )
        await agent.observe(message)
        self.pending = None
        self.previous_result = message
        self.next_sequence += 1
        return {"sequence": message.sequence}

    def finish(self, message: FinishMessage) -> dict[str, Any]:
        self._active_agent()
        if self.pending is not None:
            raise RuntimeError("cannot finish before the pending action_result")
        if message.actions != self.next_sequence:
            raise RuntimeError(
                f"finish reports {message.actions} actions; worker observed {self.next_sequence}"
            )
        final_observation = (
            self.previous_result.after
            if self.previous_result is not None
            else self.initial_observation
        )
        if final_observation is None:
            raise RuntimeError("finish arrived before the initial observation")
        if message.final_observation != final_observation:
            raise RuntimeError("finish observation does not match the last accepted successor")
        if message.levels_completed != final_observation.levels_completed:
            raise RuntimeError("finish level count does not match the final observation")
        self.finished = True
        return {}

    def _active_agent(self) -> TychoNooaAgent:
        agent = self._agent()
        if self.finished:
            raise RuntimeError("the game has already finished")
        return agent

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
                _write(stdout, "observed", **await session.observe(message))
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
