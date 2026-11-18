"""v11 Actor plus frozen-model query tools. Critic path is unchanged."""

from __future__ import annotations

import json
from pathlib import Path

from construction_service import construct_from_run
from frozen_client import FrozenModelClient
from solver_agent import MdArcSolverAgent

HEX = "0123456789abcdef"


def observation_from_actor_state(state: dict | None) -> dict | None:
    rows = (state or {}).get("grid_rows") or []
    if not rows:
        return None
    try:
        return {"grid": [[HEX.index(char) for char in row] for row in rows]}
    except ValueError:
        return None


def last_executed_row(ipc_dir: Path) -> dict | None:
    path = Path(ipc_dir) / "transitions.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[-1] if rows else None


class ConstructionAwareAgent(MdArcSolverAgent):
    def request_construction(self) -> str:
        """Host-owned construction on this run's executed prefix. Not a game action.

        One episode per game. Failure or NEEDS_EVIDENCE leaves play intact.
        """
        record = construct_from_run(self.run_dir, self.ipc_dir)
        return json.dumps(record, sort_keys=True)

    def query_frozen_predict(self, action: str) -> str:
        """Sealed predict on the host freeze. Does not submit a game action."""
        client = getattr(self, "_frozen_client", None)
        if client is None:
            client = FrozenModelClient(self.run_dir)
            self._frozen_client = client
        state = self._latest_state() or {}
        return client.query_frozen_predict(
            action,
            observation_from_actor_state(state),
            step=int(state.get("step") or 0),
            last_executed=last_executed_row(self.ipc_dir),
        )

    def assimilate_frozen(self) -> str:
        """Sealed assimilate bound to the last executed transition. Not a game action."""
        client = getattr(self, "_frozen_client", None)
        if client is None:
            return "NO_FROZEN_MODEL"
        row = last_executed_row(self.ipc_dir)
        if row is None:
            return "REJECT_UNKNOWN_TRANSITION"
        after = row.get("after") if isinstance(row.get("after"), dict) else {}
        grid = after.get("grid")
        if grid is None:
            observation = observation_from_actor_state(self._latest_state())
            if observation is None:
                return "NO_FROZEN_STATE"
        else:
            observation = {"grid": grid}
        return client.assimilate_frozen(
            observation,
            executed_action=row.get("action") if isinstance(row.get("action"), str) else None,
            executed_step=row.get("step") if isinstance(row.get("step"), int) else None,
        )
