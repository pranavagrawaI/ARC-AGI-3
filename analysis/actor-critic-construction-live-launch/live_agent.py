"""Actor overlay plus factual construction API docs in the incumbent prompt channel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from construction_agent import ConstructionAwareAgent
from solver_agent import MdArcSolverAgent

CONSTRUCTION_API = """
CONSTRUCTION API (host-owned freeze; not a game action; not markdown notes):
- self.request_construction() — run one construction episode on this game's already-executed ipc/transitions.jsonl prefix. Returns JSON with construction_status. Failure leaves play intact. Do not call it to submit UP/LEFT/RIGHT/DOWN/CLICK/RESET.
- self.query_frozen_predict(action) — sealed predict on the host FROZEN.json for one action string. Does not submit that action.
- self.assimilate_frozen() — sealed assimilate bound to the last executed transitions.jsonl row. Not a game action.
These methods exist on this agent. Call them from execute_python.
BOUNDED INTEGRATION EXERCISE: after obtaining a nonempty prefix using actions you select, exercise request_construction once. If frozen, query a prediction for your next chosen action, submit that single action, and assimilate its observed successor on the following turn. This is an instructed integration smoke test, not a test of autonomous construction timing. Keep all game-action selection with submit_actions and do not invent observations or force critic activation.
SCRIBE notes are not the world model. A freeze the host did not write is not usable. If request_construction does not freeze, keep playing with submit_actions.
""".strip()


def _append_host_event(run_dir: Path, row: dict) -> None:
    path = Path(run_dir) / "host-events.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


class LiveConstructionAgent(ConstructionAwareAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        existing = str(self.context.get("knowledge_api") or "")
        if "self.request_construction()" not in existing:
            self.context["knowledge_api"] = existing + "\n" + CONSTRUCTION_API

    def _source_digest(self) -> str | None:
        path = Path(self.run_dir) / "FROZEN.json"
        if not path.is_file():
            return None
        record = json.loads(path.read_text())
        source = record.get("candidate_source")
        digest = record.get("source_digest")
        if not isinstance(source, str) or not source.strip():
            return None
        expected = hashlib.sha256(source.encode()).hexdigest()
        if digest != expected:
            return None
        return expected

    def _record(self, event: str, status: str, **fields: object) -> None:
        row = {"event": event, "status": status, **fields}
        _append_host_event(self.run_dir, row)

    def request_construction(self) -> str:
        raw = super().request_construction()
        digest = self._source_digest()
        last = None
        try:
            from construction_agent import last_executed_row

            last = last_executed_row(self.ipc_dir)
        except Exception:
            last = None
        frozen = False
        try:
            frozen = json.loads(raw).get("frozen") is True and digest is not None
        except json.JSONDecodeError:
            frozen = False
        status = "COMPLETED" if frozen else "REJECTED"
        self._record(
            "request_construction",
            status,
            action=None if last is None else last.get("action"),
            step=None if last is None else last.get("step"),
            source_digest=digest,
            tool_result=raw,
        )
        return raw

    def query_frozen_predict(self, action: str) -> str:
        raw = super().query_frozen_predict(action)
        digest = self._source_digest()
        state = self._latest_state() or {}
        step = int(state.get("step") or 0)
        completed = False
        try:
            payload = json.loads(raw)
            completed = isinstance(payload, dict) and payload.get("status") == "predicted"
        except json.JSONDecodeError:
            completed = False
        self._record(
            "query_frozen_predict",
            "COMPLETED" if completed and digest else "REJECTED",
            action=action,
            step=step,
            source_digest=digest,
            tool_result=raw,
        )
        return raw

    def assimilate_frozen(self) -> str:
        raw = super().assimilate_frozen()
        digest = self._source_digest()
        last = None
        try:
            from construction_agent import last_executed_row

            last = last_executed_row(self.ipc_dir)
        except Exception:
            last = None
        completed = False
        try:
            payload = json.loads(raw)
            completed = isinstance(payload, dict) and payload.get("status") == "assimilated"
        except json.JSONDecodeError:
            completed = False
        self._record(
            "assimilate_frozen",
            "COMPLETED" if completed and digest else "REJECTED",
            action=None if last is None else last.get("action"),
            step=None if last is None else last.get("step"),
            source_digest=digest,
            tool_result=raw,
        )
        return raw


LiveConstructionAgent.__doc__ = (MdArcSolverAgent.__doc__ or "") + "\n\n" + CONSTRUCTION_API
