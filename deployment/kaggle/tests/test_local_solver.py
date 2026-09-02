from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.protocol import (
    PROTOCOL_VERSION,
    ActionResultMessage,
    ObservationPayload,
    project_arc_observation,
)

from runtime.local_solver import (
    GAME_GID_ENV,
    GAME_UID_ENV,
    HARDENED_HOST_ENV,
    LOCAL_MODEL_COST_MAP_ENV,
    SANDBOX_RUNTIME_ENV,
    LocalSolver,
    LocalSolverError,
    _identity_for_game,
    _prepare_hardened_session,
    _require_hardened_host_supervisor,
    create_solver,
    worker_environment,
)

FAKE_WORKER = r'''
import json
import os
import sys
import time
from pathlib import Path

HOST_PREFIX = "ARC_HOST\t"
WORKER_PREFIX = "ARC_AGENT\t"
behavior = sys.argv[1]
events = Path(sys.argv[2])
snapshot_path = sys.argv[3] if len(sys.argv) > 3 else ""

def record(value):
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, separators=(",", ":")) + "\n")

def reply(kind, **fields):
    payload = {"protocol_version": 3, "type": kind, **fields}
    print(WORKER_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)

for line in sys.stdin:
    if not line.startswith(HOST_PREFIX):
        continue
    message = json.loads(line[len(HOST_PREFIX):])
    kind = message["type"]
    record({"event": kind, "message": message, "pid": os.getpid()})
    if kind == "start_game":
        if snapshot_path:
            Path(snapshot_path).write_text(json.dumps(dict(os.environ), sort_keys=True))
        reply("ready", worker_id=str(os.getpid()), metadata={"fake": True})
    elif kind == "turn":
        if behavior == "exit":
            raise SystemExit(23)
        if behavior == "timeout":
            time.sleep(10)
        if behavior == "malformed":
            print(WORKER_PREFIX + "{not-json", flush=True)
            continue
        record({"event": "generation_begin", "pid": os.getpid(), "at": time.monotonic()})
        if behavior != "fast":
            time.sleep(0.15)
        record({"event": "generation_end", "pid": os.getpid(), "at": time.monotonic()})
        sequence = message["sequence"] + (1 if behavior == "wrong_sequence" else 0)
        reply(
            "decision",
            sequence=sequence,
            decision={"action": "UP", "row": None, "col": None},
        )
    elif kind == "action_result":
        reply("observed", sequence=message["sequence"])
    elif kind == "finish":
        reply("finished")
    elif kind == "close":
        raise SystemExit(0)
'''


def observation(value: int = 1, *, levels: int = 0) -> ObservationPayload:
    return project_arc_observation(
        frames=[[value]],
        available_action_ids=[1],
        state="NOT_FINISHED",
        levels_completed=levels,
        full_reset=False,
    )


def turn(value: int = 1) -> GameTurn:
    return GameTurn(
        grid=[[value]],
        available_actions=["RESET", "UP"],
        sequence=0,
        level=0,
        turn_in_level=0,
        levels_completed=0,
        state="NOT_FINISHED",
    )


def make_solver(
    tmp_path: Path,
    *,
    behavior: str = "normal",
    timeout_seconds: float = 1,
    events: Path | None = None,
    extra_environment: dict[str, str] | None = None,
) -> LocalSolver:
    tmp_path.mkdir(parents=True, exist_ok=True)
    worker = tmp_path / f"fake-worker-{behavior}-{id(events)}.py"
    worker.write_text(FAKE_WORKER)
    event_path = events or tmp_path / f"events-{behavior}-{id(worker)}.jsonl"
    extra_environment = dict(extra_environment or {})
    snapshot = str(extra_environment.pop("FAKE_ENV_SNAPSHOT", ""))
    environment = {
        "PATH": "/usr/bin",
        **extra_environment,
    }
    return LocalSolver(
        game_id=f"game-{behavior}-{id(worker)}",
        workspace_root=tmp_path / "workspaces",
        initial_observation=observation(),
        command=[sys.executable, str(worker), behavior, str(event_path), snapshot],
        timeout_seconds=timeout_seconds,
        close_timeout_seconds=0.2,
        environment=environment,
    )


def read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_worker_environment_forwards_only_declared_solver_inputs() -> None:
    result = worker_environment(
        {
            "ARC_API_KEY": "environment-secret",
            "ARC_LLM_API_KEY": "model-secret",
            "ARC_LLM_BASE_URL": "http://127.0.0.1:30000/v1",
            "ARC_LLM_FORCE_ACTION_TOOL": "1",
            "ARC_LLM_MODEL": "served-model",
            "ARC_LLM_MAX_ACTOR_CALLS": "4",
            "ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL": "1",
            "ARC_LLM_MAX_TOKENS": "1024",
            "ARC_LLM_MAX_TOKENS_AFTER_INITIAL": "256",
            "ARC_LLM_PROVIDER_ENDPOINT": "local",
            "ARC_LLM_RETRIES": "0",
            "ARC_LLM_TIMEOUT_SECONDS": "60",
            LOCAL_MODEL_COST_MAP_ENV: "True",
            "TYCHO_MODE": "incumbent",
            "TYCHO_NOOA_TREATMENT": "decision_discipline",
            "PATH": "/bin",
            "UNRELATED_SECRET": "do-not-forward",
        }
    )

    assert result == {
        "ARC_LLM_API_KEY": "model-secret",
        "ARC_LLM_BASE_URL": "http://127.0.0.1:30000/v1",
        "ARC_LLM_FORCE_ACTION_TOOL": "1",
        "ARC_LLM_MODEL": "served-model",
        "ARC_LLM_MAX_ACTOR_CALLS": "4",
        "ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL": "1",
        "ARC_LLM_MAX_TOKENS": "1024",
        "ARC_LLM_MAX_TOKENS_AFTER_INITIAL": "256",
        "ARC_LLM_PROVIDER_ENDPOINT": "local",
        "ARC_LLM_RETRIES": "0",
        "ARC_LLM_TIMEOUT_SECONDS": "60",
        LOCAL_MODEL_COST_MAP_ENV: "True",
        "TYCHO_MODE": "incumbent",
        "TYCHO_NOOA_TREATMENT": "decision_discipline",
        "PATH": "/bin",
    }


def test_local_cost_map_is_selected_before_solver_dependencies_import() -> None:
    environment = dict(os.environ)
    environment.pop(LOCAL_MODEL_COST_MAP_ENV, None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import runtime.local_solver as adapter; "
                "print(os.environ[adapter.LOCAL_MODEL_COST_MAP_ENV])"
            ),
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"
    assert "Failed to fetch remote model cost map" not in result.stderr


def test_worker_environment_forwards_only_explicit_hardening_contract() -> None:
    result = worker_environment(
        {
            HARDENED_HOST_ENV: "1",
            GAME_UID_ENV: "60001",
            GAME_GID_ENV: "60001",
            SANDBOX_RUNTIME_ENV: "host",
            "KAGGLE_API_V1_TOKEN": "must-not-cross",
            "KAGGLE_USER_SECRETS_TOKEN": "must-not-cross",
        }
    )

    assert result == {
        HARDENED_HOST_ENV: "1",
        GAME_UID_ENV: "60001",
        GAME_GID_ENV: "60001",
        SANDBOX_RUNTIME_ENV: "host",
    }


def test_hardened_host_requires_root_and_explicit_host_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {HARDENED_HOST_ENV: "1", SANDBOX_RUNTIME_ENV: "host"}
    monkeypatch.setattr("runtime.local_solver.os.geteuid", lambda: 1000)
    with pytest.raises(RuntimeError, match="root supervisor"):
        _require_hardened_host_supervisor(environment)

    monkeypatch.setattr("runtime.local_solver.os.geteuid", lambda: 0)
    with pytest.raises(RuntimeError, match="requires TYCHO_SANDBOX_RUNTIME=host"):
        _require_hardened_host_supervisor(
            {HARDENED_HOST_ENV: "1", SANDBOX_RUNTIME_ENV: "docker"}
        )


def test_hardened_identities_are_stable_and_distinct_under_concurrency() -> None:
    game_ids = [f"identity-test-{index}" for index in range(20)]
    observed: dict[str, list[int]] = {game_id: [] for game_id in game_ids}

    def allocate(game_id: str) -> None:
        observed[game_id].append(_identity_for_game(game_id))

    threads = [
        threading.Thread(target=allocate, args=(game_id,))
        for game_id in game_ids
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(len(set(values)) == 1 for values in observed.values())
    assert len({values[0] for values in observed.values()}) == len(game_ids)


def test_create_solver_binds_one_identity_into_the_worker_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def capture_solver(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setenv(HARDENED_HOST_ENV, "1")
    monkeypatch.setenv(SANDBOX_RUNTIME_ENV, "host")
    monkeypatch.setattr("runtime.local_solver.os.geteuid", lambda: 0)
    monkeypatch.setattr("runtime.local_solver.LocalSolver", capture_solver)

    created = create_solver(
        game_id="create-solver-identity-test",
        workspace_root=tmp_path,
        initial_observation=observation(),
    )

    assert created is not None
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment[HARDENED_HOST_ENV] == "1"
    assert environment[SANDBOX_RUNTIME_ENV] == "host"
    assert int(environment[GAME_UID_ENV]) > 0
    assert environment[GAME_GID_ENV] == environment[GAME_UID_ENV]
    assert environment[LOCAL_MODEL_COST_MAP_ENV] == "True"


def test_hardened_session_exposes_only_workspace_traversal_to_game_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    workspace = session / "workspace"
    workspace.mkdir(parents=True)
    modes: dict[Path, int] = {}
    monkeypatch.setattr("runtime.local_solver.os.geteuid", lambda: 0)
    monkeypatch.setattr("runtime.local_solver.os.chown", lambda *_args: None)
    monkeypatch.setattr(
        "runtime.local_solver.os.chmod",
        lambda path, mode: modes.__setitem__(Path(path), mode),
    )

    _prepare_hardened_session(
        session,
        workspace,
        {
            HARDENED_HOST_ENV: "1",
            GAME_UID_ENV: "60001",
            GAME_GID_ENV: "60001",
            SANDBOX_RUNTIME_ENV: "host",
        },
    )

    assert modes == {session: 0o710, workspace: 0o2710}


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() != 0,
    reason="real Unix ownership requires a root-capable integration environment",
)
def test_root_supervisor_applies_real_session_ownership_and_log_modes(
    tmp_path: Path,
) -> None:
    solver = make_solver(
        tmp_path,
        extra_environment={
            HARDENED_HOST_ENV: "1",
            GAME_UID_ENV: "60001",
            GAME_GID_ENV: "60001",
            SANDBOX_RUNTIME_ENV: "host",
        },
    )
    try:
        assert solver.session_root.stat().st_gid == 60001
        assert solver.session_root.stat().st_mode & 0o777 == 0o710
        assert solver.workspace.stat().st_gid == 60001
        assert solver.workspace.stat().st_mode & 0o7777 == 0o2710
        assert solver.stdout_log.stat().st_mode & 0o777 == 0o600
        assert solver.stderr_log.stat().st_mode & 0o777 == 0o600
    finally:
        solver.close()


def test_protocol_v3_lifecycle_through_my_agent_facing_interface(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.jsonl"
    solver = make_solver(tmp_path, events=events)
    assert solver.worker_metadata == {"fake": True}
    assert json.loads((solver.session_root / "worker-metadata.json").read_text()) == {
        "fake": True
    }
    before = observation()
    after = observation(2)

    decision = solver.decide(turn())
    assert decision == ActionDecision(action="UP")
    solver.observe(
        ActionResultMessage(
            protocol_version=PROTOCOL_VERSION,
            type="action_result",
            sequence=0,
            level=0,
            turn_in_level=0,
            decision=decision,
            before=before,
            after=after,
        )
    )
    # The official Agent counter includes its bootstrap RESET; the worker
    # protocol must still receive its one observed Solver decision.
    solver.finish(after, actions=2)
    solver.close()
    solver.close()

    messages = [
        event["message"]
        for event in read_events(events)
        if event["event"] in {"start_game", "turn", "action_result", "finish", "close"}
    ]
    assert [message["type"] for message in messages] == [
        "start_game",
        "turn",
        "action_result",
        "finish",
        "close",
    ]
    assert all(message["protocol_version"] == PROTOCOL_VERSION for message in messages)
    assert messages[0]["seed"] == 0
    assert messages[-2]["actions"] == 1
    assert messages[-2]["stop_reason"] == "competition_agent_cleanup"


@pytest.mark.parametrize("action_limit", (32, 80))
def test_protocol_turn_preserves_the_exact_action_horizon(
    tmp_path: Path,
    action_limit: int,
) -> None:
    events = tmp_path / "events.jsonl"
    solver = make_solver(tmp_path, behavior="fast", events=events)
    current = observation()
    previous = ""
    try:
        for actions_used in range(action_limit):
            actions_remaining = action_limit - actions_used
            turn = GameTurn(
                grid=current.grid,
                available_actions=current.available_actions,
                sequence=actions_used,
                action_limit=action_limit,
                actions_used=actions_used,
                actions_remaining=actions_remaining,
                final_permitted_action=actions_remaining == 1,
                level=0,
                turn_in_level=actions_used,
                levels_completed=0,
                state="NOT_FINISHED",
                previous_action=previous,
            )
            decision = solver.decide(turn)
            successor = observation((actions_used + 1) % 10)
            solver.observe(
                ActionResultMessage(
                    protocol_version=PROTOCOL_VERSION,
                    type="action_result",
                    sequence=actions_used,
                    level=0,
                    turn_in_level=actions_used,
                    decision=decision,
                    before=current,
                    after=successor,
                )
            )
            current = successor
            previous = decision.action
    finally:
        solver.close()

    messages = [
        event["message"]
        for event in read_events(events)
        if event["event"] == "turn"
    ]
    assert len(messages) == action_limit
    for actions_used, actions_remaining, final in (
        (action_limit - 2, 2, False),
        (action_limit - 1, 1, True),
    ):
        message = messages[actions_used]
        assert message["sequence"] == actions_used
        assert message["action_limit"] == action_limit
        assert message["actions_used"] == actions_used
        assert message["actions_remaining"] == actions_remaining
        assert message["final_permitted_action"] is final


def test_decide_requires_observe_before_another_turn(tmp_path: Path) -> None:
    solver = make_solver(tmp_path)
    try:
        solver.decide(turn())
        with pytest.raises(LocalSolverError, match="action_result is required"):
            solver.decide(turn())
    finally:
        solver.close()


@pytest.mark.parametrize(
    ("behavior", "message"),
    [
        ("malformed", "malformed JSON"),
        ("wrong_sequence", "expected sequence 0"),
        ("exit", "worker exited"),
        ("timeout", "timed out"),
    ],
)
def test_worker_failures_are_not_converted_to_actions(
    tmp_path: Path, behavior: str, message: str
) -> None:
    solver = make_solver(
        tmp_path,
        behavior=behavior,
        timeout_seconds=0.05 if behavior == "timeout" else 1,
    )
    try:
        with pytest.raises((LocalSolverError, TimeoutError), match=message):
            solver.decide(turn())
    finally:
        solver.close()


def test_two_games_share_one_generation_admission_slot(tmp_path: Path) -> None:
    events = tmp_path / "shared-events.jsonl"
    first = make_solver(tmp_path / "first", events=events)
    second = make_solver(tmp_path / "second", events=events)
    errors: list[BaseException] = []

    def decide(solver: LocalSolver) -> None:
        try:
            solver.decide(turn())
        except BaseException as error:  # noqa: BLE001 - preserve thread failure
            errors.append(error)

    threads = [
        threading.Thread(target=decide, args=(first,)),
        threading.Thread(target=decide, args=(second,)),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        assert not errors
        assert all(not thread.is_alive() for thread in threads)
    finally:
        first.close()
        second.close()

    generation_events = [
        event
        for event in read_events(events)
        if event["event"] in {"generation_begin", "generation_end"}
    ]
    assert [event["event"] for event in generation_events] == [
        "generation_begin",
        "generation_end",
        "generation_begin",
        "generation_end",
    ]


def test_worker_never_receives_arc_environment_secret(tmp_path: Path) -> None:
    snapshot = tmp_path / "environment.json"
    solver = make_solver(
        tmp_path,
        extra_environment={
            "ARC_API_KEY": "environment-secret",
            "ARC_LLM_API_KEY": "model-secret",
            "UNRELATED_SECRET": "do-not-forward",
            "FAKE_ENV_SNAPSHOT": str(snapshot),
        },
    )
    solver.close()

    environment = json.loads(snapshot.read_text())
    assert environment["ARC_LLM_API_KEY"] == "model-secret"
    assert "ARC_API_KEY" not in environment
    assert "UNRELATED_SECRET" not in environment
