"""TAAF owns games; canonical Tycho protocol workers own reasoning decisions."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from urllib.request import urlopen
from dataclasses import dataclass, field
from pathlib import Path

import arcengine
from taaf.solver import Solver
from inference.agent.tool_agent import _priority_gate, _PRIORITY_UNTRIMMED_BASE
from inference.agent.priority_scheduler import PrioritySnapshot
from hybrid_local_solver import worker_environment, _identity_for_game, _require_hardened_host_supervisor
from hybrid.transport import HybridLocalSolver
from tycho_nooa.models import GameTurn
from tycho_nooa.protocol import (PROTOCOL_VERSION, ActionResultMessage,
    ARC_ACTION_NAME_TO_ID, project_arc_observation)


def observation(game, *, previous_levels=None, full_reset=False):
    raw = game.current_state.raw
    frames = [frame.tolist() if hasattr(frame, "tolist") else frame for frame in raw.frame]
    return project_arc_observation(frames=frames,
        available_action_ids=list(raw.available_actions), state=raw.state.name,
        levels_completed=int(raw.levels_completed), full_reset=full_reset,
        previous_levels_completed=previous_levels)


@dataclass
class HybridSolver(Solver):
    label: str = "tycho-dfranzen-orchestrator-control-v3"
    concurrency: int = 25
    max_runtime_s_per_game: float = 7254.0
    max_actions_per_game: int | None = None
    analyzer_timeout: float = 900.0
    save_request_logs: bool = True
    worker_python: str = sys.executable
    runtime_root: str = ""
    _stop: threading.Event | None = field(default=None, init=False, repr=False)
    _workers: dict = field(default_factory=dict, init=False, repr=False)
    _lock: object = field(default=None, init=False, repr=False)

    def _setup(self):
        self._stop = threading.Event()
        self._lock = threading.Lock()
        if not self.runtime_root:
            self.runtime_root = str(Path(__file__).resolve().parents[1])
        if not (Path(self.runtime_root)/"worker_entry.py").is_file():
            raise RuntimeError("hybrid worker entry missing")

    def _teardown(self):
        self._stop.set()
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.close()

    async def _run_games(self, games):
        slots = asyncio.Semaphore(max(1, self.concurrency))
        async def play(index, game):
            async with slots:
                try:
                    await asyncio.to_thread(self._play, game, index)
                except Exception as error:
                    # A persistence/cleanup failure belongs to this game. Do
                    # not cancel healthy siblings and turn it into an arm loss.
                    game.game_run.state = "crashed"
                    game.game_run.solver_note = f"{type(error).__name__}: {error}"
                    game.finish_game()
                    print(f"hybrid host failure {game.game_id}: {type(error).__name__}",file=sys.stderr)
        tasks = [asyncio.create_task(play(i, game)) for i, game in enumerate(games)]
        try:
            await asyncio.gather(*(asyncio.shield(task) for task in tasks))
        except asyncio.CancelledError:
            self._stop.set()
            with self._lock:
                workers = list(self._workers.values())
            # Close pipes to retire blocked model calls, then wait for host game
            # transitions to settle. No thread is abandoned mutating an engine.
            await asyncio.to_thread(lambda: [worker.close() for worker in workers])
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def _play(self, game, index):
        root = Path(self.job_dir or ".") / "hybrid-games"
        started = time.monotonic()
        deadline = started + self.max_runtime_s_per_game
        if self.soft_end_time is not None:
            deadline = min(deadline, started + max(0, self.soft_end_time.timestamp()-time.time()))
        gate = _priority_gate()
        admitted = False
        worker = None
        previous = None
        actions = 0
        engine_actions = 0
        turn_in_level = 0
        credited_tokens = 0
        level_actions = 0
        level_tokens = 0
        old_evictions = 0
        failure = None
        stop_reason = None
        try:
            if gate is not None:
                gate.configure_clock(started, deadline)
                gate.acquire(_PRIORITY_UNTRIMMED_BASE-index)
                admitted = True
            if self._stop.is_set() or time.monotonic() >= deadline:
                return
            current = observation(game)
            # The donor may release its benchmark while SGLang is still loading.
            # Bound initial readiness separately from model-call retries, and let
            # cancellation interrupt the wait. Offline candidates need no keepalive RESET.
            readiness = float(os.environ.get("ARC3_HTTP_RETRY_INITIAL_SECONDS", "0"))
            base_url = os.environ.get("ARC_LLM_BASE_URL", "").rstrip("/")
            if readiness > 0 and base_url and not os.environ.get("TYCHO_NOOA_SCRIPTED_TOOL_CALLS"):
                ready_deadline = min(deadline, time.monotonic()+readiness)
                health_url = base_url.removesuffix("/v1")+"/health"
                while not self._stop.is_set():
                    remaining = ready_deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("model server initial readiness deadline")
                    try:
                        with urlopen(health_url, timeout=min(2, remaining)) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    self._stop.wait(min(1, remaining))
                if self._stop.is_set():
                    return
            environment = dict(os.environ)
            if environment.get("TYCHO_KAGGLE_HARDENED_HOST") == "1":
                _require_hardened_host_supervisor(environment)
                uid = _identity_for_game(f"{game.game_id}-{index}")
                environment.update(TYCHO_KAGGLE_GAME_UID=str(uid), TYCHO_KAGGLE_GAME_GID=str(uid))
            environment = worker_environment(environment)
            worker = HybridLocalSolver(game_id=game.game_id, workspace_root=root,
                initial_observation=current,
                command=[self.worker_python, str(Path(self.runtime_root)/"worker_entry.py")],
                timeout_seconds=min(self.analyzer_timeout, max(1, deadline-time.monotonic())),
                environment=environment, cancellation=self._stop)
            with self._lock:
                self._workers[index] = worker
            receipts = worker.session_root/"host-actions.jsonl"
            while (not self._stop.is_set() and time.monotonic() < deadline
                   and not game.current_state.won
                   and (self.max_actions_per_game is None or actions < self.max_actions_per_game)):
                old_level = current.levels_completed
                expected_previous = None if previous is None or previous.after.levels_completed > previous.before.levels_completed or previous.after.full_reset else previous.decision
                horizon = {}
                if self.max_actions_per_game is not None:
                    horizon = dict(action_limit=self.max_actions_per_game, actions_used=actions,
                        actions_remaining=self.max_actions_per_game-actions,
                        final_permitted_action=actions+1 == self.max_actions_per_game)
                turn = GameTurn(grid=current.grid, available_actions=current.available_actions,
                    sequence=actions, level=old_level, turn_in_level=turn_in_level,
                    levels_completed=old_level, state=current.state,
                    previous_action=expected_previous.action if expected_previous else "",
                    previous_row=expected_previous.row if expected_previous else None,
                    previous_col=expected_previous.col if expected_previous else None, **horizon)
                worker.timeout_seconds = min(self.analyzer_timeout, max(0.1, deadline-time.monotonic()))
                decision = worker.decide(turn)
                if self._stop.is_set() or time.monotonic() >= deadline:
                    break
                action_id = arcengine.GameAction.from_id(ARC_ACTION_NAME_TO_ID[decision.action])
                data = {} if decision.action != "CLICK" else {"x": decision.col, "y": decision.row}
                tokens = int((worker.stats or {}).get("completion_tokens", 0))
                delta = max(0, tokens-credited_tokens)
                game.execute_action(arcengine.ActionInput(id=action_id, data=data), generated_tokens=delta)
                engine_actions += 1
                credited_tokens = tokens
                # ONLY_RESET_LEVELS is retained from donor: RESET is a current-level
                # reset, not a rewind of the whole game. Do not label it full_reset.
                after = observation(game, previous_levels=old_level)
                result = ActionResultMessage(protocol_version=PROTOCOL_VERSION,
                    type="action_result", sequence=actions, level=old_level,
                    turn_in_level=turn_in_level, before=current, decision=decision, after=after)
                with receipts.open("a") as stream:
                    stream.write(result.model_dump_json()+"\n")
                worker.timeout_seconds = min(self.analyzer_timeout, max(0.1, deadline-time.monotonic()))
                worker.observe(result)
                previous, current = result, after
                actions += 1
                level_actions += 1
                level_tokens += delta
                turn_in_level += 1
                if current.levels_completed > old_level:
                    turn_in_level, level_actions, level_tokens = 0, 0, 0
                evictions = int((worker.stats or {}).get("evictions", 0))
                if gate is not None and evictions > old_evictions:
                    snapshot = PrioritySnapshot(level=current.levels_completed+1,
                        actions=level_actions, tokens=level_tokens,
                        total_levels=game.number_of_levels)
                    priority = gate._snapshot_priority(snapshot, time.monotonic())
                    gate.handover(priority, snapshot)
                old_evictions = evictions
            if worker._pending is None and not self._stop.is_set() and time.monotonic() < deadline:
                worker.timeout_seconds = min(10, self.analyzer_timeout, max(0.1, deadline-time.monotonic()))
                worker.finish(current, actions)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            if isinstance(error, TimeoutError) and time.monotonic() >= deadline:
                stop_reason = "game_deadline"
                game.game_run.solver_note = failure
            elif not self._stop.is_set():
                game.game_run.state = "crashed"
                game.game_run.solver_note = f"{type(error).__name__}: {error}"
        finally:
            stop_reason = stop_reason or (
                "cancelled" if self._stop.is_set() else
                "error" if failure else
                "won" if game.current_state.won else
                "game_deadline" if time.monotonic() >= deadline else
                "action_cap" if self.max_actions_per_game is not None and actions >= self.max_actions_per_game else
                "finished")
            try:
                if worker is not None:
                    try:
                        (worker.session_root/"compute.json").write_text(json.dumps(worker.stats or {}, indent=2)+"\n")
                    finally:
                        worker.close()
                    (worker.session_root/"execution-receipt.json").write_text(json.dumps({
                        "elapsed_seconds": time.monotonic()-started,
                        "engine_actions": engine_actions,
                        "failure": failure,
                        "stop_reason": stop_reason,
                        "worker_returncode": worker._process.returncode,
                        "reported_usage_scope": "completed responses reported by worker only",
                        "unresolved_provider_usage": bool(failure or self._stop.is_set()),
                        "provider_cancellation_confirmed": False,
                    }, indent=2)+"\n")
            finally:
                # Artifact failures must not strand workers, games or slots.
                with self._lock:
                    self._workers.pop(index, None)
                try:
                    if self._stop.is_set() and game.game_run.state == "playing":
                        game.game_run.state = "cancelled"
                    if game.game_run.state in {"playing", "won", "crashed", "cancelled"}:
                        total = int((worker.stats or {}).get("completion_tokens", 0)) if worker else 0
                        game.finish_game(generated_tokens=max(0, total-credited_tokens))
                finally:
                    if admitted:
                        gate.release()
