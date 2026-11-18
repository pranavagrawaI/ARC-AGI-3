#!/usr/bin/env python3
"""Protocol-v3 worker that runs upstream Tycho's own game loop against the official host.

The official ARC-AGI-3 ``MyAgent`` seam owns the environment on Kaggle. Upstream
Tycho owns a loop that wants to own it too: ``tycho.harness.harness.run_env``
calls ``arc.make(...)``, then ``env.reset()`` and ``env.step(...)``.

Rather than re-derive Tycho's agent-facing contract here — reset, choose_action,
note_external_reset, note_final_observation, is_done, the GAME_OVER RESET, level
bookkeeping — this worker supplies a *proxy Arcade* whose ``step`` is backed by
the host over protocol v3, and then runs Tycho's unmodified ``run_env`` on a
second thread. Every line of agent policy stays upstream's.

The two loops have opposite control flow: protocol v3 is host-driven (host sends
``turn``, worker answers ``decision``), while ``run_env`` is worker-driven. Two
one-slot queues invert it. ``run_env`` blocks in ``env.step`` until the host
reports the successor; the protocol thread blocks for a choice until Tycho has
one.

Nothing in this file decides an action.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, TextIO

from arcengine import FrameData, GameAction, GameState

from tycho_nooa.protocol import (
    ARC_ACTION_NAME_TO_ID,
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

# Tycho's per-level budget is ACTION_BUDGET_MULTIPLIER x the human baseline. The
# official host owns the real budget and stops us first, so the worker is given a
# generous synthetic baseline instead of reading the mounted environment files.
# This affects only Tycho's internal budget and its own RunSummary scoring; the
# measured outcome comes from the official recordings, identically for both arms.
DEFAULT_SYNTHETIC_BASELINE = 512
DEFAULT_SYNTHETIC_LEVELS = 32
BASELINE_ENV = "TYCHO_SYNTHETIC_BASELINE"
LEVELS_ENV = "TYCHO_SYNTHETIC_LEVELS"


class WorkerError(RuntimeError):
    """Protocol or bridge failure returned to LocalSolver as type=error."""


def _write(stdout: TextIO, kind: str, **fields: Any) -> None:
    payload = {"protocol_version": PROTOCOL_VERSION, "type": kind, **fields}
    stdout.write(WORKER_PREFIX + json.dumps(payload, separators=(",", ":")) + "\n")
    stdout.flush()


# --------------------------------------------------------------------------
# Observation projection
# --------------------------------------------------------------------------


def raw_frame_sequence(observation: ObservationPayload) -> list[list[list[int]]]:
    """Rebuild the host's raw frame sequence from the evidence envelope.

    ``project_arc_observation`` splits one ARC frame list into a decision frame, an
    optional terminal frame, and the remaining animation frames, recording each
    source index. That split is invertible, and Tycho reads the whole sequence
    (``_grid_to_first`` for terminal states, animation evidence for its summaries),
    so it is inverted rather than flattened to the settled grid.
    """
    provenance = observation.provenance
    sequence: list[list[list[int]] | None] = [None] * provenance.raw_frame_count
    sequence[provenance.decision_frame_index] = observation.decision_frame
    if provenance.terminal_frame_index is not None:
        sequence[provenance.terminal_frame_index] = observation.terminal_frame
    for event in observation.event_frames:
        sequence[event.source_index] = event.frame
    missing = [index for index, grid in enumerate(sequence) if grid is None]
    if missing:
        raise WorkerError(f"evidence envelope has no frame at indices {missing}")
    return [grid for grid in sequence if grid is not None]


def available_action_ids(observation: ObservationPayload) -> list[int]:
    ids = []
    for name in observation.available_actions:
        try:
            ids.append(ARC_ACTION_NAME_TO_ID[name])
        except KeyError as error:
            raise WorkerError(f"unsupported action label {name!r}") from error
    return ids


def frame_data(observation: ObservationPayload, game_id: str) -> FrameData:
    """Project one host observation onto the arcengine FrameData Tycho expects."""
    return FrameData(
        game_id=game_id,
        frame=raw_frame_sequence(observation),
        state=GameState[observation.state],
        levels_completed=observation.levels_completed,
        full_reset=observation.full_reset,
        available_actions=available_action_ids(observation),
    )


def decision_payload(action: GameAction, data: dict | None) -> dict[str, Any]:
    """Translate one Tycho ActionChoice into the seam's ActionDecision fields."""
    name = {
        GameAction.RESET: "RESET",
        GameAction.ACTION1: "UP",
        GameAction.ACTION2: "DOWN",
        GameAction.ACTION3: "LEFT",
        GameAction.ACTION4: "RIGHT",
        GameAction.ACTION5: "SPACEBAR",
        GameAction.ACTION6: "CLICK",
        GameAction.ACTION7: "UNDO",
    }.get(action)
    if name is None:
        raise WorkerError(f"Tycho chose an action the seam cannot carry: {action!r}")
    if name != "CLICK":
        return {"action": name, "row": None, "col": None}
    if not data or "x" not in data or "y" not in data:
        raise WorkerError("Tycho chose ACTION6 without x/y coordinates")
    return {"action": "CLICK", "row": int(data["y"]), "col": int(data["x"])}


# --------------------------------------------------------------------------
# The bridge: host-driven protocol <-> worker-driven run_env
# --------------------------------------------------------------------------


class Bridge:
    """One-slot handoff between the protocol thread and Tycho's run_env thread."""

    def __init__(self, initial: FrameData) -> None:
        self.initial = initial
        self._choices: queue.Queue[tuple[GameAction, dict | None]] = queue.Queue(maxsize=1)
        self._successors: queue.Queue[FrameData] = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.stop_reason: str | None = None
        self.run_error: str | None = None

    # --- called from Tycho's thread -------------------------------------
    def offer_choice(self, action: GameAction, data: dict | None) -> FrameData:
        self._choices.put((action, data))
        successor = self._successors.get()
        if successor is None:  # pragma: no cover - only on shutdown
            raise WorkerError("host closed before reporting the action result")
        return successor

    # --- called from the protocol thread --------------------------------
    def take_choice(self, timeout: float) -> tuple[GameAction, dict | None]:
        try:
            return self._choices.get(timeout=timeout)
        except queue.Empty as error:
            if self.stopped.is_set():
                raise WorkerError(
                    "Tycho's run_env stopped before this turn: "
                    f"{self.stop_reason or self.run_error or 'unknown'}"
                ) from error
            raise TimeoutError("timed out waiting for Tycho to choose an action") from error

    def report_successor(self, successor: FrameData) -> None:
        self._successors.put(successor)

    def release(self) -> None:
        try:
            self._successors.put_nowait(None)  # type: ignore[arg-type]
        except queue.Full:
            pass


class ProxyEnv:
    """The arcengine-facing surface ``run_env`` uses, backed by the official host."""

    def __init__(self, bridge: Bridge) -> None:
        self._bridge = bridge

    def reset(self) -> FrameData:
        # The host already issued the bootstrap RESET; its initial observation is
        # the post-reset frame. Returning it here keeps run_env's uncharged
        # initialization reset uncharged on the official scorecard too.
        return self._bridge.initial

    def step(self, action: GameAction, data: dict | None = None, reasoning: Any = None) -> FrameData:
        return self._bridge.offer_choice(action, data)


class ProxyArcade:
    def __init__(self, bridge: Bridge) -> None:
        self._bridge = bridge

    def make(self, game_id: str, seed: int = 0, scorecard_id: str | None = None) -> ProxyEnv:
        return ProxyEnv(self._bridge)


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------


class TychoSession:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.bridge: Bridge | None = None
        self.thread: threading.Thread | None = None
        self.record_path = run_dir / "tycho-run-record.json"
        self.finished = False
        self._agent_name = ""
        self.game_id = ""

    def start(self, message: StartGameMessage) -> dict[str, Any]:
        if self.bridge is not None:
            raise WorkerError("a game is already active in this worker")
        # Imported here so an import failure is reported through the protocol as a
        # worker error rather than killing the process before `ready`.
        from tycho.agent.agent import build_agent
        from tycho.harness.harness import run_env

        workspace = Path(message.workspace_root)
        workspace.mkdir(parents=True, exist_ok=True)
        # Tycho creates ws_root/<game> itself on reset; create it up front so the
        # probe's per-game session marker exists even if the run dies early.
        (workspace / message.game_id).mkdir(parents=True, exist_ok=True)
        self.game_id = message.game_id
        initial = frame_data(message.initial_observation, message.game_id)
        bridge = Bridge(initial)
        self.bridge = bridge
        agent = build_agent()
        self._agent_name = getattr(agent, "name", type(agent).__name__)
        baselines = self._baselines()

        def play() -> None:
            try:
                record = run_env(
                    agent,
                    ProxyArcade(bridge),
                    message.game_id,
                    baselines,
                    seed=message.seed,
                    keep_trace=True,
                    ws_root=str(workspace),
                )
                bridge.stop_reason = getattr(record, "stop_reason", None) or "run_env_returned"
                self._write_record(record)
            except BaseException as error:  # noqa: BLE001 - surfaced through the protocol
                bridge.run_error = f"{type(error).__name__}: {error}"
                traceback.print_exc(file=sys.stderr)
            finally:
                bridge.stopped.set()

        self.thread = threading.Thread(target=play, name="tycho-run-env", daemon=True)
        self.thread.start()
        return {
            "worker_id": str(os.getpid()),
            "metadata": {
                "agent": f"tycho.{self._agent_name}",
                "join": "s0-tycho-upstream-join",
                "upstream": "arc-agi-3-tycho 0.1.0 (wheelhouse, byte-identical to Initial release)",
                "mode": os.environ.get("TYCHO_MODE", "single"),
                "sandbox_runtime": os.environ.get("TYCHO_SANDBOX_RUNTIME", ""),
                "synthetic_baseline": {"per_level": baselines[0], "levels": len(baselines)},
                "workspace": str(workspace),
            },
        }

    def decide(self, message: TurnMessage) -> dict[str, Any]:
        if self.finished:
            raise WorkerError("the game has already finished")
        bridge = self._bridge()
        action, data = bridge.take_choice(message.decision_timeout_seconds or 1800.0)
        return {"sequence": message.sequence, "decision": decision_payload(action, data)}

    def observe(self, message: ActionResultMessage) -> dict[str, Any]:
        bridge = self._bridge()
        bridge.report_successor(frame_data(message.after, self.game_id))
        return {"sequence": message.sequence}

    def finish(self, _message: FinishMessage) -> dict[str, Any]:
        self.finished = True
        bridge = self.bridge
        if bridge is not None:
            bridge.release()
        thread = self.thread
        if thread is not None:
            thread.join(timeout=30)
        return {}

    def close(self) -> None:
        bridge = self.bridge
        if bridge is not None:
            bridge.release()
        thread = self.thread
        if thread is not None:
            thread.join(timeout=10)
        self.bridge = None
        self.thread = None

    def _bridge(self) -> Bridge:
        if self.bridge is None:
            raise WorkerError("start_game is required before this message")
        return self.bridge

    @staticmethod
    def _baselines() -> list[int]:
        baseline = int(os.environ.get(BASELINE_ENV, DEFAULT_SYNTHETIC_BASELINE))
        levels = int(os.environ.get(LEVELS_ENV, DEFAULT_SYNTHETIC_LEVELS))
        if baseline <= 0 or levels <= 0:
            raise WorkerError("synthetic baseline and level count must be positive")
        return [baseline] * levels

    def _write_record(self, record: Any) -> None:
        try:
            from dataclasses import asdict, is_dataclass

            value = asdict(record) if is_dataclass(record) else {"repr": repr(record)}
            self.record_path.write_text(json.dumps(value, indent=2, default=str) + "\n")
        except Exception:  # noqa: BLE001 - bookkeeping must not fail the run
            traceback.print_exc(file=sys.stderr)


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    session: TychoSession | None = None
    try:
        while line := stdin.readline():
            if not line.startswith(HOST_PREFIX):
                continue
            sequence: int | None = None
            try:
                message = parse_host_message(line[len(HOST_PREFIX) :])
                sequence = getattr(message, "sequence", None)
                if isinstance(message, CloseMessage):
                    return
                if isinstance(message, StartGameMessage):
                    session = TychoSession(Path.cwd())
                    _write(stdout, "ready", **session.start(message))
                elif session is None:
                    raise WorkerError("start_game is required before this message")
                elif isinstance(message, TurnMessage):
                    _write(stdout, "decision", **session.decide(message))
                elif isinstance(message, ActionResultMessage):
                    _write(stdout, "observed", **session.observe(message))
                elif isinstance(message, FinishMessage):
                    _write(stdout, "finished", **session.finish(message))
            except Exception as error:  # noqa: BLE001 - protocol errors must be returned
                payload: dict[str, Any] = {"message": f"{type(error).__name__}: {error}"}
                if sequence is not None:
                    payload["sequence"] = sequence
                traceback.print_exc(file=sys.stderr)
                _write(stdout, "error", **payload)
                return
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    serve()
