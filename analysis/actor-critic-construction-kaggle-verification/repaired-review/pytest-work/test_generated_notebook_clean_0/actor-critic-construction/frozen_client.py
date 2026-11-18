"""Actor-facing sealed query of a host-owned freeze. No game-action authority."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

LIVE = __import__("pathlib").Path(__file__).resolve().parent


class FrozenModelClient:
    """Read-only consumer of FROZEN.json through CandidateRPC."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self._state: dict[str, Any] | None = None
        self._pending: dict[str, Any] | None = None
        self._pending_key: tuple[str, int] | None = None
        self._levels_completed: int | None = None

    def drop(self) -> None:
        self._state = None
        self._pending = None
        self._pending_key = None

    def _frozen_source(self) -> str:
        path = self.run_dir / "FROZEN.json"
        if not path.is_file():
            raise FileNotFoundError("NO_FROZEN_MODEL")
        record = json.loads(path.read_text())
        source = record.get("candidate_source")
        digest = record.get("source_digest")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("FROZEN.json missing candidate_source")
        expected = hashlib.sha256(source.encode()).hexdigest()
        if not isinstance(digest, str) or digest != expected:
            raise ValueError("TAMPERED_FREEZE")
        return source

    def _rpc(self, source: str):
        if str(LIVE) not in sys.path:
            sys.path.insert(0, str(LIVE))
        from integration_boundary import CandidateRPC, SealedBoundary

        return CandidateRPC(SealedBoundary(), source)

    def note_executed(self, row: dict[str, Any] | None) -> None:
        """Drop belief only after an executed RESET or level change."""
        if not row:
            return
        action = row.get("action")
        after = row.get("after") if isinstance(row.get("after"), dict) else {}
        levels = after.get("levels_completed")
        if action == "RESET" or (
            isinstance(levels, int)
            and self._levels_completed is not None
            and levels != self._levels_completed
        ):
            self.drop()
        if isinstance(levels, int):
            self._levels_completed = levels

    def query_frozen_predict(
        self,
        action: str,
        observation: dict[str, Any] | None = None,
        *,
        step: int | None = None,
        last_executed: dict[str, Any] | None = None,
    ) -> str:
        self.note_executed(last_executed)
        try:
            source = self._frozen_source()
        except FileNotFoundError:
            return "NO_FROZEN_MODEL"
        except ValueError as error:
            return str(error)
        if self._pending is not None:
            return "PREDICT_WITHOUT_ASSIMILATE"
        rpc = self._rpc(source)
        if self._state is None:
            if observation is None or "grid" not in observation:
                return "NO_FROZEN_STATE"
            self._state = rpc.initialize({"grid": observation["grid"]})
        produced = rpc.predict(self._state, action)
        if not isinstance(produced, list) or len(produced) != 1:
            return "PREDICT_BRANCH_COUNT_UNSUPPORTED"
        if step is None:
            return "NO_FROZEN_STATE"
        self._pending = produced[0]["state"]
        self._pending_key = (action, step)
        return json.dumps({"status": "predicted", "commitments": produced[0].get("commitments")}, sort_keys=True)

    def assimilate_frozen(
        self,
        observation: dict[str, Any],
        *,
        executed_action: str | None,
        executed_step: int | None,
    ) -> str:
        try:
            self._frozen_source()
        except FileNotFoundError:
            return "NO_FROZEN_MODEL"
        except ValueError as error:
            return str(error)
        if self._pending_key is None or self._pending is None:
            return "AMBIGUOUS_PENDING"
        if not isinstance(executed_action, str) or not isinstance(executed_step, int):
            return "REJECT_UNKNOWN_TRANSITION"
        pending_action, pending_step = self._pending_key
        if (executed_action, executed_step) != (pending_action, pending_step + 1):
            return "REJECT_MISMATCH"
        source = self._frozen_source()
        rpc = self._rpc(source)
        self._state = rpc.assimilate(self._pending, {"grid": observation["grid"]}, {"status": "unscored"})
        self._pending = None
        self._pending_key = None
        return json.dumps({"status": "assimilated", "verdict": "unscored"}, sort_keys=True)
