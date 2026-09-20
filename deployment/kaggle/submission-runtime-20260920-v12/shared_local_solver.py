"""Synchronous Kaggle adapter for one persistent Tycho-NOOA worker per game."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import IO, Any

# LiteLLM reads this switch while Solver dependencies are imported, before any
# worker is created. Kaggle evaluation is offline, so select bundled metadata
# before the first tycho_nooa import instead of incurring a failed network fetch.
LOCAL_MODEL_COST_MAP_ENV = "LITELLM_LOCAL_MODEL_COST_MAP"
os.environ.setdefault(LOCAL_MODEL_COST_MAP_ENV, "True")

from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.protocol import (
    HOST_PREFIX,
    PROTOCOL_VERSION,
    WORKER_PREFIX,
    ActionResultMessage,
    FinishMessage,
    ObservationPayload,
    StartGameMessage,
    TurnMessage,
)

DEFAULT_TIMEOUT_SECONDS = 3_600.0
DEFAULT_CLOSE_TIMEOUT_SECONDS = 10.0
WORKER_PYTHON_ENV = "TYCHO_NOOA_WORKER_PYTHON"
WORKER_TIMEOUT_ENV = "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS"
HARDENED_HOST_ENV = "TYCHO_KAGGLE_HARDENED_HOST"
GAME_UID_ENV = "TYCHO_KAGGLE_GAME_UID"
GAME_GID_ENV = "TYCHO_KAGGLE_GAME_GID"
SANDBOX_RUNTIME_ENV = "TYCHO_SANDBOX_RUNTIME"
_FIRST_GAME_IDENTITY = 60_001
_LAST_GAME_IDENTITY = 60_999

# The official Swarm starts every game in a separate thread. Before a scored
# route establishes safe serving concurrency, exactly one game may ask the
# shared model server to generate at a time.
_MODEL_GENERATION_ADMISSION = threading.BoundedSemaphore(1)
_MODEL_GENERATION_RETIREMENT_FAILED = False
_IDENTITY_LOCK = threading.Lock()
_GAME_IDENTITIES: dict[str, int] = {}


class GameBudgetExhausted(Exception):
    """Normal game stop acknowledged at an idle model admission boundary."""

WORKER_ENVIRONMENT_KEYS = frozenset(
    {
        "ARC_GAME_MODEL_ADMISSION_DEADLINE",
        "ARC_LLM_API_KEY",
        "ARC_LLM_BASE_URL",
        "ARC_LLM_CONTEXT_WINDOW",
        "ARC_LLM_FORCE_ACTION_TOOL",
        "ARC_LLM_MAX_ACTOR_CALLS",
        "ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL",
        "ARC_LLM_MAX_TOKENS",
        "ARC_LLM_MAX_TOKENS_AFTER_INITIAL",
        "ARC_LLM_MODEL",
        "ARC_LLM_PROVIDER_ENDPOINT",
        "ARC_LLM_RETRIES",
        "ARC_LLM_TIMEOUT_SECONDS",
        "ARC_LLM_USE_RESPONSES",
        "ARC_LLM_EXTRA_BODY",
        "ARC_STALL_REVIEW",
        "ARC_TRANSCRIPT_KEEP_TURNS",
        "ARC_SOLVER_ARM",
        "ARC_DISCOVERY_ARM",
        "ARC_CONTINGENT_ARM",
        "ARC_PROCEDURAL_DIAGNOSTIC",
        "ARC_PROCEDURAL_CASE",
        "ARC_DISABLE_DOTENV",
        "ARC_Q6_RUN_ID",
        "HOME",
        HARDENED_HOST_ENV,
        "LANG",
        "LC_ALL",
        LOCAL_MODEL_COST_MAP_ENV,
        "NOOA_AGENT_COMMAND",
        "NOOA_AGENT_READY_TIMEOUT_SECONDS",
        "NOOA_JOIN_ROOT",
        "NOOA_LLM_CONFIG",
        "NOOA_STUB_BATCH",
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "TMPDIR",
        "TYCHO_MODE",
        "TYCHO_NOOA_TREATMENT",
        "TYCHO_NOOA_CAPTURE_RESPONSE_TEXT",
        "TYCHO_NOOA_REQUEST_REASONING",
        "TYCHO_SANDBOX_RUNTIME",
        GAME_GID_ENV,
        GAME_UID_ENV,
        "TZ",
    }
)


class LocalSolverError(RuntimeError):
    """The local worker violated or could not complete its protocol."""


def worker_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """Forward solver configuration without the ARC environment credential."""
    values = dict(os.environ) if source is None else source
    return {key: values[key] for key in WORKER_ENVIRONMENT_KEYS if key in values}


def create_solver(
    *,
    game_id: str,
    workspace_root: Path,
    initial_observation: ObservationPayload,
) -> LocalSolver:
    """Create the default competition worker used by ``MyAgent``."""
    worker_python = os.environ.get(WORKER_PYTHON_ENV, sys.executable)
    timeout = _positive_timeout(os.environ.get(WORKER_TIMEOUT_ENV))
    environment = dict(os.environ)
    # Kaggle evaluation has no internet. Force LiteLLM's bundled cost/context
    # metadata instead of paying one failed remote-fetch timeout per worker.
    environment.setdefault(LOCAL_MODEL_COST_MAP_ENV, "True")
    if environment.get(HARDENED_HOST_ENV) == "1":
        _require_hardened_host_supervisor(environment)
        identity = _identity_for_game(game_id)
        environment[GAME_UID_ENV] = str(identity)
        environment[GAME_GID_ENV] = str(identity)
    return LocalSolver(
        game_id=game_id,
        workspace_root=workspace_root,
        initial_observation=initial_observation,
        command=[worker_python, "-m", "tycho_nooa.worker"],
        timeout_seconds=timeout,
        environment=environment,
    )


class LocalSolver:
    """Translate the MyAgent seam to one process-long protocol-v3 worker."""

    def __init__(
        self,
        *,
        game_id: str,
        workspace_root: Path,
        initial_observation: ObservationPayload,
        command: list[str],
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        close_timeout_seconds: float = DEFAULT_CLOSE_TIMEOUT_SECONDS,
        environment: dict[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("worker command must be non-empty")
        if timeout_seconds <= 0 or close_timeout_seconds <= 0:
            raise ValueError("worker timeouts must be positive")
        self.timeout_seconds = timeout_seconds
        self._game_work_deadline = float((environment or {}).get('ARC_GAME_WORK_DEADLINE', 'inf'))
        self.close_timeout_seconds = close_timeout_seconds
        self._current_observation = initial_observation
        self._next_sequence = 0
        self._pending: tuple[GameTurn, ActionDecision] | None = None
        self._previous_result: ActionResultMessage | None = None
        self._finished = False
        self._closed = False

        workspace_root.mkdir(parents=True, exist_ok=True)
        session_root = Path(
            tempfile.mkdtemp(
                prefix=f"{_safe_name(game_id)}-",
                dir=workspace_root,
            )
        )
        self.session_root = session_root
        self.workspace = session_root / "workspace"
        self.workspace.mkdir()
        self.stdout_log = session_root / "worker-stdout.log"
        self.stderr_log = session_root / "worker-stderr.log"
        self.stdout_log.touch(mode=0o600)
        self._stderr: IO[str] = self.stderr_log.open("w", encoding="utf-8")
        os.chmod(self.stderr_log, 0o600)
        child_environment = worker_environment(environment)
        _prepare_hardened_session(
            session_root,
            self.workspace,
            child_environment,
        )
        try:
            self._process = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                bufsize=1,
                env=child_environment,
                cwd=session_root,
            )
        except BaseException:
            self._stderr.close()
            raise
        self._stdout_lines: queue.Queue[str | None] = queue.Queue()
        self._stdout_reader = threading.Thread(
            target=self._read_stdout,
            name=f"local-solver-stdout-{_safe_name(game_id)}",
            daemon=True,
        )
        self._stdout_reader.start()
        try:
            self._send_model(
                StartGameMessage(
                    protocol_version=PROTOCOL_VERSION,
                    type="start_game",
                    game_id=game_id,
                    # The pinned official launcher calls Arcade.make without a
                    # seed, whose public default is zero.
                    seed=0,
                    workspace_root=str(self.workspace),
                    initial_observation=initial_observation,
                )
            )
            ready = self._receive("ready")
            metadata = ready.get("metadata")
            if not isinstance(metadata, dict):
                raise LocalSolverError("worker ready message has invalid metadata")
            self.worker_metadata = metadata
            metadata_path = self.session_root / "worker-metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            metadata_path.chmod(0o600)
        except BaseException:
            self.close()
            raise

    def decide(self, turn: GameTurn) -> ActionDecision:
        global _MODEL_GENERATION_RETIREMENT_FAILED
        self._require_active()
        # A later decision must not receive a fresh full-game watchdog.
        self.timeout_seconds = min(self.timeout_seconds,
            max(0.001, self._game_work_deadline - time.monotonic()))
        if self._pending is not None:
            raise LocalSolverError("action_result is required before the next turn")
        sequence = turn.sequence
        if sequence is None or sequence != self._next_sequence:
            raise LocalSolverError(
                f"expected turn sequence {self._next_sequence}, got {sequence}"
            )
        self._validate_turn_continuity(turn)
        message = TurnMessage(
            protocol_version=PROTOCOL_VERSION,
            type="turn",
            sequence=sequence,
            action_limit=turn.action_limit,
            actions_used=turn.actions_used,
            actions_remaining=turn.actions_remaining,
            final_permitted_action=turn.final_permitted_action,
            level=turn.level,
            turn_in_level=turn.turn_in_level,
            decision_timeout_seconds=self.timeout_seconds,
            observation=self._current_observation,
            previous_decision=self._expected_previous_decision(),
        )
        with _MODEL_GENERATION_ADMISSION:
            if _MODEL_GENERATION_RETIREMENT_FAILED:
                raise LocalSolverError('previous worker retirement unresolved; admission stopped')
            try:
                self._send_model(message)
                payload = self._receive("decision", sequence=sequence)
                if payload.get('budget_exhausted') is True:
                    raise GameBudgetExhausted('model_admission_budget')
                try:
                    decision = ActionDecision.model_validate(payload.get("decision"))
                except Exception as error:
                    raise LocalSolverError("worker emitted an invalid decision") from error
            except GameBudgetExhausted:
                raise
            except BaseException as error:
                # The protocol worker may still be inside this turn. Do not send
                # finish behind it or release admission while it can issue calls.
                # This also covers an outer SIGALRM/KeyboardInterrupt, not only
                # the transport's own timeout exception.
                try:
                    self.close(abort=True)
                except Exception as cleanup_error:
                    _MODEL_GENERATION_RETIREMENT_FAILED = True
                    error.add_note(f'Worker retirement failed: {cleanup_error!r}; admission stopped')
                raise
        self._pending = (turn, decision)
        return decision

    def observe(self, result: ActionResultMessage) -> None:
        self._require_active()
        if self._pending is None:
            raise LocalSolverError("action_result arrived without a pending decision")
        turn, decision = self._pending
        if (
            result.sequence != turn.sequence
            or result.level != turn.level
            or result.turn_in_level != turn.turn_in_level
            or result.decision != decision
            or result.before != self._current_observation
        ):
            raise LocalSolverError("action_result does not match the pending turn")
        self._send_model(result)
        self._receive("observed", sequence=result.sequence)
        self._current_observation = result.after
        self._previous_result = result
        self._pending = None
        self._next_sequence += 1

    def finish(self, final_observation: ObservationPayload, actions: int) -> None:
        self._require_active()
        if self._pending is not None:
            raise LocalSolverError("cannot finish before the pending action_result")
        if final_observation != self._current_observation:
            raise LocalSolverError("finish observation is not the last accepted successor")
        if actions < self._next_sequence:
            raise LocalSolverError(
                f"official agent reports {actions} actions after "
                f"{self._next_sequence} Solver decisions"
            )
        self._send_model(
            FinishMessage(
                protocol_version=PROTOCOL_VERSION,
                type="finish",
                stop_reason="competition_agent_cleanup",
                # Agent.action_counter includes bootstrap RESET actions. The
                # worker protocol counts only decisions it emitted and observed.
                actions=self._next_sequence,
                levels_completed=final_observation.levels_completed,
                final_observation=final_observation,
            )
        )
        self._receive("finished")
        self._finished = True

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process.poll() is None:
            try:
                if abort:
                    process.terminate()
                else:
                    self._send_payload({"type": "close"})
                process.wait(timeout=self.close_timeout_seconds)
            except (BrokenPipeError, OSError, LocalSolverError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=self.close_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=self.close_timeout_seconds)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        self._stderr.close()

    def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            self._stdout_lines.put(None)
            return
        for line in stdout:
            self._stdout_lines.put(line)
        self._stdout_lines.put(None)

    def _validate_turn_continuity(self, turn: GameTurn) -> None:
        observation = self._current_observation
        if (
            turn.grid != observation.grid
            or turn.available_actions != observation.available_actions
            or turn.levels_completed != observation.levels_completed
            or turn.state != observation.state
        ):
            raise LocalSolverError("turn does not match the last public observation")
        previous = self._expected_previous_decision()
        expected = (
            ("", None, None)
            if previous is None
            else (previous.action, previous.row, previous.col)
        )
        actual = (turn.previous_action, turn.previous_row, turn.previous_col)
        if actual != expected:
            raise LocalSolverError("turn previous action does not match accepted history")

    def _expected_previous_decision(self) -> ActionDecision | None:
        result = self._previous_result
        if result is None:
            return None
        crossed_boundary = (
            result.after.full_reset
            or result.after.levels_completed > result.before.levels_completed
        )
        return None if crossed_boundary else result.decision

    def _send_model(self, message: Any) -> None:
        self._send_payload(message.model_dump(mode="json", exclude={"protocol_version"}))

    def _send_payload(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process.poll() is not None or process.stdin is None:
            raise LocalSolverError(f"worker exited with code {process.poll()}")
        message = {"protocol_version": PROTOCOL_VERSION, **payload}
        process.stdin.write(
            HOST_PREFIX + json.dumps(message, separators=(",", ":")) + "\n"
        )
        process.stdin.flush()

    def _receive(
        self, expected_type: str, *, sequence: int | None = None
    ) -> dict[str, Any]:
        while True:
            try:
                line = self._stdout_lines.get(timeout=self.timeout_seconds)
            except queue.Empty as error:
                raise TimeoutError(
                    f"worker timed out waiting for {expected_type}"
                ) from error
            if line is None:
                raise LocalSolverError(
                    f"worker exited with code {self._process.poll()}"
                )
            if not line.startswith(WORKER_PREFIX):
                with self.stdout_log.open("a", encoding="utf-8") as log:
                    log.write(line)
                continue
            try:
                payload = json.loads(line[len(WORKER_PREFIX) :])
            except json.JSONDecodeError as error:
                raise LocalSolverError("worker emitted malformed JSON") from error
            if not isinstance(payload, dict):
                raise LocalSolverError("worker response must be a JSON object")
            if payload.get("protocol_version") != PROTOCOL_VERSION:
                raise LocalSolverError("worker protocol version mismatch")
            if payload.get("type") == "error":
                raise LocalSolverError(str(payload.get("message", "worker error")))
            if payload.get("type") != expected_type:
                raise LocalSolverError(
                    f"expected worker message {expected_type!r}, "
                    f"got {payload.get('type')!r}"
                )
            if sequence is not None and payload.get("sequence") != sequence:
                raise LocalSolverError(
                    f"expected sequence {sequence}, got {payload.get('sequence')!r}"
                )
            return payload

    def _require_active(self) -> None:
        if self._closed:
            raise LocalSolverError("local Solver is closed")
        if self._finished:
            raise LocalSolverError("local Solver has already finished")


def _positive_timeout(raw: str | None) -> float:
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{WORKER_TIMEOUT_ENV} must be positive") from error
    if value <= 0:
        raise ValueError(f"{WORKER_TIMEOUT_ENV} must be positive")
    return value


def _require_hardened_host_supervisor(environment: dict[str, str]) -> None:
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise RuntimeError(
            f"{HARDENED_HOST_ENV}=1 requires the Kaggle root supervisor"
        )
    if environment.get(SANDBOX_RUNTIME_ENV) != "host":
        raise RuntimeError(
            f"{HARDENED_HOST_ENV}=1 requires {SANDBOX_RUNTIME_ENV}=host"
        )


def _identity_for_game(game_id: str) -> int:
    with _IDENTITY_LOCK:
        existing = _GAME_IDENTITIES.get(game_id)
        if existing is not None:
            return existing
        identity = _FIRST_GAME_IDENTITY + len(_GAME_IDENTITIES)
        if identity > _LAST_GAME_IDENTITY:
            raise RuntimeError("hardened-host per-game identity pool exhausted")
        _GAME_IDENTITIES[game_id] = identity
        return identity


def _prepare_hardened_session(
    session_root: Path,
    workspace: Path,
    environment: dict[str, str],
) -> None:
    if environment.get(HARDENED_HOST_ENV) != "1":
        return
    _require_hardened_host_supervisor(environment)
    try:
        uid = int(environment[GAME_UID_ENV])
        gid = int(environment[GAME_GID_ENV])
    except (KeyError, ValueError) as error:
        raise RuntimeError("hardened-host worker identity is missing or invalid") from error
    if uid <= 0 or gid <= 0:
        raise RuntimeError("hardened-host worker identity must be non-root")
    os.chown(session_root, 0, gid)
    # The game group may traverse to its workspace but cannot list or read the
    # root-owned worker logs and protocol artifacts beside it.
    os.chmod(session_root, 0o710)
    os.chown(workspace, 0, gid)
    # Root may create the Tycho GameWorkspace beneath this setgid directory;
    # the game group may traverse to it but cannot publish sibling paths here.
    os.chmod(workspace, 0o2710)


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return (name or "game")[:80]
