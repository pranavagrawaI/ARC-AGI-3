"""Plain JSONL evidence for one host-controlled episode."""

from __future__ import annotations

import json
from pathlib import Path

from arc_host.types import (
    ActionDecision,
    EpisodeResult,
    GameStart,
    JsonValue,
    SolverPort,
    Transition,
    Turn,
    WorkerReady,
)


class RecordingSolver:
    """Record the host-visible seam without changing the wrapped solver."""

    def __init__(self, solver: SolverPort, path: Path):
        self.solver = solver
        self.path = path
        self.initial_levels_completed: int | None = None
        self.actions_recorded = 0
        self.latest_levels_completed: int | None = None

    def start_game(self, start: GameStart) -> WorkerReady:
        self._append({"type": "start_game", **start.to_json()})
        self.initial_levels_completed = start.initial_observation.levels_completed
        self.latest_levels_completed = start.initial_observation.levels_completed
        ready = self.solver.start_game(start)
        self._append(
            {
                "type": "ready",
                "worker_id": ready.worker_id,
                "metadata": ready.metadata,
            }
        )
        return ready

    def decide(self, turn: Turn) -> ActionDecision:
        self._append({"type": "turn", **turn.to_json()})
        decision = self.solver.decide(turn)
        self._append(
            {
                "type": "decision",
                "sequence": turn.sequence,
                "decision": decision.to_json(),
            }
        )
        return decision

    def observe(self, transition: Transition) -> None:
        self._append({"type": "action_result", **transition.to_json()})
        self.actions_recorded += 1
        self.latest_levels_completed = transition.after.levels_completed
        self.solver.observe(transition)

    def finish(self, result: EpisodeResult) -> None:
        self._append(
            {
                "type": "finish",
                "stop_reason": result.stop_reason,
                "actions": result.actions,
                "levels_completed": result.levels_completed,
                "final_observation": result.final.to_json(),
            }
        )
        self.solver.finish(result)

    def close(self) -> None:
        self.solver.close()

    def record_error(self, error: Exception) -> None:
        """Persist a host-visible failure even when no EpisodeResult can be built."""
        self._append(
            {
                "type": "error",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "actions": self.actions_recorded,
                "levels_completed": self.latest_levels_completed,
            }
        )

    def _append(self, record: dict[str, JsonValue]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            handle.flush()
