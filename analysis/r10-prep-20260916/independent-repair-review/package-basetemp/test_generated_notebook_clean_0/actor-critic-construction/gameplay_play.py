#!/usr/bin/env python3
"""Official Agent.main gameplay driver: paired arms on one pinned public game through one local SGLang relay.

Arm A: incumbent v11 Actor (nooa-join/headless.py), construction disabled.
Arm B: the same Actor under the R6/R7 construction overlay (package headless.py with
constructor transport). The arms share the served model, the relay, the game bytes,
the budgets and every environment binding except the worker command; ARM-DIFF.json
records that and the driver refuses to play if anything else differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from game_preflight import DEFAULT_ENVIRONMENTS
import game_registry

HERE = Path(__file__).resolve().parent
DRIVER = "official Agent.main (pinned ARC-AGI-3-Agents agents/agent.py) via MyAgent adapter; OFFLINE Arcade; paired arms"
LOCAL_MODEL = "/kaggle/input/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot"
DEFAULT_COMPETITION = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
DEFAULT_WORKING = Path("/kaggle/working")

# Frozen package parameters (the card's Cap). GAMES is the pinned game set
# (every entry bound to the public manifest in game_registry); PLAY_GAME is the
# single game this identity plays, set per package by the builder.
GAMES = ("ls20", "s5i5", "cd82")
PLAY_GAME = 'cd82'  # BUILD-PARAMETER
GAME_ID = game_registry.game_id(PLAY_GAME)
REQUEST_BODY_DIR = "raw-requests"  # full request bodies, one file per call, kept out of requests.jsonl
ARMS = ("A", "B")
ADMISSION_SECONDS = 600
MAX_ACTIONS = 200
LLM_TIMEOUT_SECONDS = 90
ACTOR_MAX_TOKENS = 16384
CRITIC_MAX_TOKENS = 192
ARM_CALL_CAP = {"A": 129, "B": 129}  # symmetric by ruling: any per-arm purse difference invalidates the pairing
TOTAL_CALL_CAP = 258  # + 1 runtime-owned warmup = 259 server chat requests
HTTP_POST_SECONDS = {"actor": 90, "summarizer": 90, "critic": 90, "constructor": 30}
CONSTRUCTOR_WALL = 240
CONSTRUCTOR_HARD = 245
CONSTRUCTOR_MAX_REQUESTS = 6
REQUEST_RESERVE_SECONDS = 15
CLEANUP_RESERVE_SECONDS = 30
STOP_REASONS = {"win", "step_budget", "wall_budget", "model_admission_budget", "call_cap"}

OFFICIAL_REVISION = "135f20aaf44f13341ebc425666bf03b5cac58d3c"
OFFICIAL_HASHES = {
    ".env.example": "ffa415e8a6b5e8e29b69fc351faba6161e588f550daaca950a7fee1687a0c3af",
    ".git/logs/HEAD": "edf83ad6d2902dc27ab28542e7fba3c3af3d8506f3f3b54524f1ea42386d513e",
    "agents/agent.py": "85655f7a11a105e50bba76efd00bebf8056d02ddbae827a8f8a5b46e849fc087",
    "agents/recorder.py": "0a08d89f4067a760012767c05d4406bd2bf409f426e29a1193106abfcbb696c8",
    "agents/swarm.py": "d9dc48f710f1b90a6552db0921293c7e89c8a925ed00a3faefa07ae19998ad39",
    "agents/templates/random_agent.py": "e5e3610fd56a4819572cc0f0d768cff426ac8b935b549735bb396aefc810705e",
    "agents/tracing.py": "951ca56508c524504e116303f7c64f4eb5cf723c72cab892d4d1a3292b1cc51f",
    "main.py": "864254c750bbbd12a211f2d8aa1b1025d0609283f07dea4ede83722f2435301b",
}
# Environment keys that may legitimately differ between arms (labels, per-arm
# roots and absolute deadlines). Everything else must be byte-equal.
ARM_VARIABLE_KEYS = {
    "NOOA_AGENT_COMMAND", "ARC_CONSTRUCTION_ARM", "TYCHO_KAGGLE_WORKSPACE_ROOT", "RECORDINGS_DIR",
    "ARC_GAME_MODEL_ADMISSION_DEADLINE", "ARC_GAME_WORK_DEADLINE",
}


def append(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_gpu(out: Path) -> dict:
    sys.path.insert(0, str(HERE))
    import accelerator

    def execute(command, timeout=10, output_cap=8192, env=None):
        import subprocess

        completed = subprocess.run(command, capture_output=True, timeout=timeout, env=env)
        stdout = completed.stdout.decode("utf-8", "replace")[:output_cap]
        stderr = completed.stderr.decode("utf-8", "replace")[:output_cap]
        return {"returncode": completed.returncode, "stdout": stdout, "stderr": stderr, "status": "completed"}

    discovery = accelerator.discover(execute)
    record = {"discovery": discovery, "status": "discovered"}
    (out / "GPU.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    try:
        device = accelerator.parse_device(discovery)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        (out / "GPU.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        raise
    record.update(status="passed", device=device)
    (out / "GPU.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return device


def require_seal(out: Path) -> None:
    from containment_preflight import qualify
    qualify(out / "SEAL.json")


def install_framework(competition: Path, working: Path, receipt: Path) -> Path:
    """Copy the pinned official ARC-AGI-3-Agents checkout and install the join adapter."""
    source = competition / "ARC-AGI-3-Agents"
    framework = working / "ARC-AGI-3-Agents"
    if framework.exists():
        raise RuntimeError("framework destination exists; use a fresh execution")
    shutil.copytree(source, framework, symlinks=True)
    for name, digest in OFFICIAL_HASHES.items():
        path = framework / name
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise RuntimeError("official framework drift: " + name)
    if OFFICIAL_REVISION not in (framework / ".git/logs/HEAD").read_text():
        raise RuntimeError("official source revision is not the pinned competition revision")
    join = HERE / "nooa-join"
    (framework / "runtime").mkdir(exist_ok=True)
    (framework / "runtime/__init__.py").write_text("")
    shutil.copyfile(join / "adapter.py", framework / "agents/templates/my_agent.py")
    shutil.copyfile(join / "shared_local_solver.py", framework / "runtime/local_solver.py")
    shutil.copyfile(join / "local_solver.py", framework / "local_solver.py")
    (framework / "agents/__init__.py").write_text(
        "from .agent import Agent, Playback\nfrom .swarm import Swarm\n"
        "from .templates.my_agent import MyAgent\nAVAILABLE_AGENTS = {\"myagent\": MyAgent}\n")
    (framework / ".env").write_text("")
    receipt.write_text(json.dumps({
        "framework": str(framework), "revision": OFFICIAL_REVISION,
        "official": {name: sha256(framework / name) for name in OFFICIAL_HASHES},
        "installed": {
            "agents/templates/my_agent.py": sha256(framework / "agents/templates/my_agent.py"),
            "runtime/local_solver.py": sha256(framework / "runtime/local_solver.py"),
            "local_solver.py": sha256(framework / "local_solver.py"),
        },
    }, indent=2) + "\n")
    return framework


def slim_request(request: dict) -> dict:
    """Request receipt without the message bodies: identity, bounds, tools and a digest of messages.

    Full bodies are written per call under raw-requests/; the jsonl stays under the retrieval bound."""
    messages = json.dumps(request.get("messages"), sort_keys=True).encode()
    slim = {k: v for k, v in request.items() if k != "messages"}
    slim["messages_sha256"] = hashlib.sha256(messages).hexdigest()
    slim["messages_bytes"] = len(messages)
    slim["messages_count"] = len(request.get("messages") or [])
    return slim


def session_directories(workspace: Path) -> list[dict]:
    """Name every solver session directory under an arm workspace so its ipc files are addressable."""
    from audit import session_freeze_record

    rows = []
    for path in sorted(p for p in workspace.iterdir() if p.is_dir() and p.name != "recordings"):
        ipc = path / "ipc"
        rows.append({
            "session_dir": path.name,
            "ipc_files": sorted(p.name for p in ipc.iterdir()) if ipc.is_dir() else [],
            "files": sorted(p.name for p in path.iterdir() if p.is_file()),
            "frozen": session_freeze_record(path) is not None,
            "child_freeze_artifacts": [str(p.relative_to(workspace)) for p in (path / "constructor-job").rglob("FROZEN.json")],
            "host_events": [str(p.relative_to(workspace)) for p in path.rglob("host-events.jsonl")],
            "constructor_results": [str(p.relative_to(workspace)) for p in path.rglob("RESULT.json")],
            "constructor_files": sorted(str(p.relative_to(workspace)) for p in (path / "constructor-job").rglob("*") if p.is_file()),
        })
    return rows


def reconcile_official(recordings: Path, card, game_id: str, agent) -> dict:
    """Bind the driver's progress row to the official recorder file and scorecard run."""
    files = sorted(recordings.glob("*.recording.jsonl")) if recordings.is_dir() else []
    rows = [json.loads(line) for path in files for line in path.read_text().splitlines() if line.strip()]
    frames = [r["data"] for r in rows if isinstance(r.get("data"), dict) and "frame" in r["data"] and "state" in r["data"]]
    run = card.get(game_id) if card is not None else {}
    record = {
        "recording_files": [p.name for p in files],
        "recorded_frames": len(frames),
        "agent_frames": len(agent.frames), "agent_actions": agent.action_counter,
        "recorded_final_state": frames[-1]["state"] if frames else None,
        "recorded_final_levels": frames[-1].get("levels_completed") if frames else None,
        "scorecard_run": {k: run.get(k) for k in ("id", "state", "score", "actions", "levels_completed") if isinstance(run, dict)} if run else None,
    }
    final_matches = not frames or (
        frames[-1]["state"] == agent.frames[-1].state.value
        and frames[-1].get("levels_completed") == agent.frames[-1].levels_completed)
    record["reconciled"] = bool(
        len(files) == 1 and len(frames) == agent.action_counter and final_matches
        and (not run or run.get("state") == agent.frames[-1].state.value)
    )
    return record


def admission_budget(timeout_s: float, arm_count: int, request_timeout_s: float) -> float:
    """Rolling fair share (v10 play.py): reserve final output, per-arm cleanup and one call."""
    allowance = (timeout_s - 180) / arm_count - CLEANUP_RESERVE_SECONDS
    budget = allowance - request_timeout_s - REQUEST_RESERVE_SECONDS
    if budget < ADMISSION_SECONDS:
        raise RuntimeError("remaining launcher budget cannot fund the precommitted admission window")
    return float(ADMISSION_SECONDS)


def base_environment(relay: str, launcher_python: str) -> dict[str, str]:
    """Bindings shared by both arms; the worker sees only WORKER_ENVIRONMENT_KEYS of these."""
    return {
        "ARC_DISABLE_DOTENV": "1",
        "ARC_LLM_MODEL": "qwen/qwen3.8-27b",
        "ARC_LLM_API_KEY": "local-sglang",
        "ARC_LLM_BASE_URL": relay,
        "ARC_LLM_PROVIDER_ENDPOINT": "local",
        "ARC_LLM_USE_RESPONSES": "0",
        "ARC_LLM_EXTRA_BODY": '{"reasoning":{"enabled":false}}',
        "ARC_LLM_MAX_TOKENS": str(ACTOR_MAX_TOKENS),
        "ARC_LLM_TIMEOUT_SECONDS": str(LLM_TIMEOUT_SECONDS),
        "ARC_LLM_RETRIES": "0",
        "ARC_STALL_REVIEW": "0",
        "ARC_SOLVER_ARM": "loop",
        "ARC_DISCOVERY_ARM": "a",
        "TYCHO_KAGGLE_MAX_ACTIONS": str(MAX_ACTIONS),
        "TYCHO_GAMEPLAY_ADMISSION_SECONDS": str(ADMISSION_SECONDS),
        "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS": str(ADMISSION_SECONDS + LLM_TIMEOUT_SECONDS + REQUEST_RESERVE_SECONDS),
        "TYCHO_NOOA_WORKER_PYTHON": launcher_python,
        "TYCHO_SANDBOX_RUNTIME": "host",
        "NOOA_JOIN_ROOT": str(HERE / "nooa-join"),
        "NOOA_AGENT_READY_TIMEOUT_SECONDS": "30",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "MPLBACKEND": "agg",
        "PYTHONUNBUFFERED": "1",
    }


def worker_command(arm: str, relay: str, launcher_python: str) -> list[str]:
    if arm == "A":
        return [launcher_python, str(HERE / "nooa-join" / "headless.py"), "--run-dir", "."]
    return [
        launcher_python, str(HERE / "headless.py"),
        "--constructor-transport", "http",
        "--constructor-url", relay.rstrip("/") + "/chat/completions",
        "--constructor-wall-seconds", str(CONSTRUCTOR_WALL),
        "--constructor-hard-seconds", str(CONSTRUCTOR_HARD),
        "--constructor-max-requests", str(CONSTRUCTOR_MAX_REQUESTS),
        "--run-dir", ".",
    ]


def arm_environment(arm: str, base: dict[str, str], workspace: Path, relay: str, launcher_python: str) -> dict[str, str]:
    return {
        **base,
        "ARC_CONSTRUCTION_ARM": "0" if arm == "A" else "1",
        "TYCHO_KAGGLE_WORKSPACE_ROOT": str(workspace),
        "RECORDINGS_DIR": str(workspace / "recordings"),
        "NOOA_AGENT_COMMAND": json.dumps(worker_command(arm, relay, launcher_python)),
    }


def arm_diff(environments: dict[str, dict[str, str]], relay: str, launcher_python: str) -> dict:
    """Prove the arms differ only in the worker command and per-arm labels/roots."""
    a, b = environments["A"], environments["B"]
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    forbidden = sorted(set(differing) - ARM_VARIABLE_KEYS)
    join = HERE / "nooa-join"
    record = {
        "schema": "paired-arm-diff-v1",
        "games": list(GAMES),
        "differing_keys": differing,
        "forbidden_differences": forbidden,
        "worker_command": {arm: worker_command(arm, relay, launcher_python) for arm in ARMS},
        "worker_entry_sha256": {
            "A": sha256(join / "headless.py"),
            "B": sha256(HERE / "headless.py"),
        },
        "shared_agent_sha256": {
            "nooa-join/adapter.py": sha256(join / "adapter.py"),
            "nooa-join/shared_local_solver.py": sha256(join / "shared_local_solver.py"),
            "nooa-join/chassis/solver_agent.py": sha256(join / "chassis" / "solver_agent.py"),
            "live_agent.py": sha256(HERE / "live_agent.py"),
            "constructor_contract.py": sha256(HERE / "constructor_contract.py"),
        },
        "shared_budgets": {
            "admission_seconds": ADMISSION_SECONDS, "max_actions": MAX_ACTIONS,
            "llm_timeout_seconds": LLM_TIMEOUT_SECONDS, "actor_max_tokens": ACTOR_MAX_TOKENS,
            "critic_max_tokens": CRITIC_MAX_TOKENS, "arm_call_cap": ARM_CALL_CAP,
        },
        "treatment": "arm B worker is the package headless.py (LiveConstructionAgent overlay over the same MdArcSolverAgent) with constructor transport; arm A worker is the unmodified v11 nooa-join/headless.py",
        "equal": not forbidden,
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--startup-argv-json", required=True)
    parser.add_argument("--environments", type=Path, default=DEFAULT_ENVIRONMENTS)
    parser.add_argument("--competition-root", type=Path, default=DEFAULT_COMPETITION)
    parser.add_argument("--working", type=Path, default=DEFAULT_WORKING)
    args = parser.parse_args()
    if os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
        raise SystemExit("gameplay driver refuses competition rerun mode")
    if args.model != LOCAL_MODEL:
        raise SystemExit("local model identity drift")
    started = time.monotonic()
    global_deadline = started + args.timeout_s
    out = args.workspace_root / "raw"
    out.mkdir(parents=True, exist_ok=False)
    if PLAY_GAME not in GAMES:
        raise SystemExit("PLAY_GAME is not in the frozen game set")
    game_identity = game_registry.qualify(out / "GAME-PREFLIGHT.json", PLAY_GAME, args.environments)
    require_seal(out)
    inspect_gpu(out)
    startup_argv = json.loads(args.startup_argv_json)
    from local_wire import validate_identity
    validate_identity(args.model, args.model, startup_argv)
    (out / "MODEL-IDENTITY.json").write_text(json.dumps({"model": args.model, "mounted_weights": args.model, "startup_argv": startup_argv}))
    sys.path.insert(0, str(HERE))
    from probe_lifecycle import AdmissionWindow, shutdown_bounded

    state = {"calls": 0, "arm": None, "arm_calls": {arm: 0 for arm in ARMS}, "roles": {}, "stop": {}, "fatal": False}
    window = AdmissionWindow(int(args.timeout_s))
    lock = threading.Lock()

    def abort(arm: str | None, reason: str) -> None:
        if arm is not None:
            state["stop"].setdefault(arm, reason)
            for ipc in (args.workspace_root / f"arm-{arm}").rglob("ipc"):
                (ipc / "ABORT").write_text(reason)
        else:
            state["fatal"] = True
            window.close(reason)

    class Relay(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def do_POST(self) -> None:
            try:
                with window.admit():
                    self._forward()
            except PermissionError as error:
                self.send_error(429, str(error))

        def _forward(self) -> None:
            from local_wire import project_local, request_purpose, usage_receipt, valid_response

            data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(data)
            purpose = request_purpose(payload)
            with lock:
                arm = state["arm"]
                if arm is None or arm in state["stop"]:
                    self.send_error(429, "no arm admitting")
                    return
                if state["calls"] >= TOTAL_CALL_CAP or state["arm_calls"][arm] >= ARM_CALL_CAP[arm]:
                    append(out / "admission-stops.jsonl", {"arm": arm, "reason": "call_cap", "calls": state["calls"], "arm_calls": state["arm_calls"][arm]})
                    abort(arm, "call_cap")
                    self.send_error(429, "call cap")
                    return
                if arm == "A" and purpose == "constructor":
                    append(out / "local-rejections.jsonl", {"arm": arm, "error": "constructor request in baseline arm", "sent": False})
                    abort(None, "constructor request in baseline arm")
                    self.send_error(400)
                    return
                try:
                    body = project_local(payload, local_model=LOCAL_MODEL, mounted_weights=LOCAL_MODEL, startup_argv=startup_argv)
                except ValueError as error:
                    append(out / "local-rejections.jsonl", {"arm": arm, "error": str(error), "sent": False})
                    abort(None, str(error))
                    self.send_error(400)
                    return
                state["calls"] += 1
                state["arm_calls"][arm] += 1
                identity = {"call": state["calls"], "purpose": purpose, "arm": arm, "time": time.time()}
                (out / REQUEST_BODY_DIR).mkdir(exist_ok=True)
                (out / REQUEST_BODY_DIR / f"{state['calls']:03d}.logical.json").write_bytes(data)
                (out / REQUEST_BODY_DIR / f"{state['calls']:03d}.wire.json").write_text(json.dumps(body))
                append(out / "logical-requests.jsonl", {**identity, "request": slim_request(payload)})
                append(out / "requests.jsonl", {**identity, "request": slim_request(body)})
                request = urllib.request.Request(
                    args.base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                begun = time.monotonic()
                try:
                    with urllib.request.urlopen(request, timeout=HTTP_POST_SECONDS[purpose]) as response:
                        raw = response.read()
                    parsed = json.loads(raw)
                    (out / "raw-http").mkdir(exist_ok=True)
                    (out / "raw-http" / f"{state['calls']:02d}.body").write_bytes(raw)
                    append(out / "responses.jsonl", {**identity, "response": parsed})
                    if not valid_response(parsed):
                        abort(None, "local model identity/envelope mismatch")
                        self.send_error(502)
                        return
                    if usage_receipt(parsed) is None:
                        abort(None, "missing usage receipt; no forwarding retry")
                        self.send_error(502)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    try:
                        self.wfile.write(raw)
                    except (BrokenPipeError, ConnectionResetError):
                        append(out / "delivery.jsonl", {**identity, "status": "completed_response_not_delivered"})
                        abort(None, "completed response not delivered")
                except Exception as error:
                    with (out / "errors.jsonl").open("a") as handle:
                        handle.write(json.dumps({**identity, "type": type(error).__name__, "message": str(error)}) + "\n")
                    abort(None, "upstream failure; no forwarding retry")
                    self.send_error(502)
                finally:
                    append(out / "latency.jsonl", {**identity, "elapsed_seconds": time.monotonic() - begun})

    server = HTTPServer(("127.0.0.1", 0), Relay)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    progress: list[dict] = []
    error = None
    try:
        server_thread.start()
        relay = f"http://127.0.0.1:{server.server_port}/v1"
        launcher_python = sys.executable
        framework = install_framework(args.competition_root, args.working, out / "FRAMEWORK.json")
        base = base_environment(relay, launcher_python)
        environments = {arm: arm_environment(arm, base, args.workspace_root / f"arm-{arm}", relay, launcher_python) for arm in ARMS}
        diff = arm_diff(environments, relay, launcher_python)
        (out / "ARM-DIFF.json").write_text(json.dumps(diff, indent=2) + "\n")
        if not diff["equal"]:
            raise RuntimeError("arms differ beyond the worker command: " + repr(diff["forbidden_differences"]))
        (out / "environment-steps.jsonl").write_text("")
        sys.path.insert(0, str(framework))
        os.environ.update(base)
        from agents.templates.my_agent import MyAgent
        from runtime.local_solver import GameBudgetExhausted
        from arc_agi import Arcade, OperationMode

        for index, arm in enumerate(ARMS):
            workspace = args.workspace_root / f"arm-{arm}"
            workspace.mkdir(parents=True, exist_ok=False)
            arm_started = time.monotonic()
            remaining = global_deadline - arm_started
            budget = admission_budget(remaining, len(ARMS) - index, LLM_TIMEOUT_SECONDS)
            admission_deadline = arm_started + budget
            worker_timeout = budget + LLM_TIMEOUT_SECONDS + REQUEST_RESERVE_SECONDS
            for key in ARM_VARIABLE_KEYS:
                os.environ.pop(key, None)
            os.environ.update(environments[arm])
            os.environ["ARC_GAME_MODEL_ADMISSION_DEADLINE"] = str(admission_deadline)
            os.environ["ARC_GAME_WORK_DEADLINE"] = str(arm_started + worker_timeout)
            append(out / "budget-allocations.jsonl", {
                "arm": arm, "game_id": GAME_ID, "remaining_seconds": remaining, "admission_seconds": budget,
                "admission_deadline": admission_deadline, "request_reserve_seconds": LLM_TIMEOUT_SECONDS,
                "tool_margin_seconds": REQUEST_RESERVE_SECONDS, "cleanup_reserve_seconds": CLEANUP_RESERVE_SECONDS,
                "worker_watchdog_seconds": worker_timeout, "max_actions": MAX_ACTIONS, "call_cap": ARM_CALL_CAP[arm],
            })
            arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=game_identity["selected_directory"], recordings_dir=str(workspace / "recordings"))
            if [info.game_id for info in arcade.available_environments] != [GAME_ID]:
                raise RuntimeError("Arcade discovery identity mismatch")
            scorecard = arcade.open_scorecard(tags=["actor-critic-construction-r8-gameplay", f"arm-{arm}"])
            with lock:
                state["arm"] = arm
            env = arcade.make(GAME_ID, scorecard_id=scorecard)
            if env is None:
                raise RuntimeError("exact public game initialization failed")
            agent = MyAgent(card_id=scorecard, game_id=GAME_ID, agent_name="myagent", ROOT_URL="http://127.0.0.1",
                            record=True, arc_env=env, tags=[f"arm-{arm}"])
            failure = None
            try:
                agent.main()
            except GameBudgetExhausted:
                agent._game_stop_reason = "model_admission_budget"
            except BaseException as exc:
                solver = getattr(agent, "_solver", None)
                retired = solver is not None and getattr(solver, "_closed", False) and solver._process.poll() is not None
                if state["stop"].get(arm) == "call_cap" and retired:
                    # The relay refused further admission and the worker abort-retired
                    # on ABORT; do not send a fictitious finish behind a closed transport.
                    agent._game_stop_reason = "call_cap"
                    agent._finished = True
                    append(out / "cleanup.jsonl", {"arm": arm, "status": "abort_retired_after_call_cap", "worker_returncode": solver._process.returncode, "finish_sent": False})
                else:
                    failure = exc
                    raise
            finally:
                with lock:
                    state["arm"] = None
                card = None
                try:
                    card = arcade.get_scorecard(scorecard)
                    agent.cleanup(scorecard=card)
                except BaseException as cleanup_error:
                    if failure is None:
                        raise
                    failure.add_note(f"Additional cleanup failure: {cleanup_error!r}")
                try:
                    arcade.close_scorecard(scorecard)
                except BaseException as close_error:
                    if failure is None:
                        raise
                    failure.add_note(f"Additional scorecard failure: {close_error!r}")
            from audit import classify
            facts = classify(workspace)
            arm_rows = [r for r in _rows(out / "requests.jsonl") if r.get("arm") == arm]
            facts["critic_exercised"] = any(r.get("purpose") == "critic" for r in arm_rows)
            facts["roles"] = {p: sum(1 for r in arm_rows if r.get("purpose") == p) for p in ("actor", "summarizer", "constructor", "critic")}
            official = reconcile_official(workspace / "recordings", card, GAME_ID, agent)
            sessions = session_directories(workspace)
            item = {
                "arm": arm, "game_id": GAME_ID, "actions": agent.action_counter, "official": official,
                "workspace": f"arm-{arm}", "sessions": sessions,
                "solver_actions": agent._solver_decisions_emitted,
                "levels_completed": agent.frames[-1].levels_completed,
                "state": agent.frames[-1].state.value, "stop_reason": agent._game_stop_reason,
                "game_elapsed_seconds": time.monotonic() - arm_started, "elapsed_seconds": time.monotonic() - started,
                "calls": state["arm_calls"][arm],
                "roles": facts["roles"], "critic_exercised": facts["critic_exercised"],
                "construction": {k: facts[k] for k in ("frozen", "requested_construction", "queried_frozen_predict", "assimilated_frozen", "action_step_digest_bound", "executed_transitions")},
            }
            progress.append(item)
            append(out / "game-progress.jsonl", item)
            for frame in agent.frames[1:]:
                append(out / "environment-steps.jsonl", {"arm": arm, "levels_completed": frame.levels_completed})
            if not official["reconciled"]:
                raise RuntimeError(f"arm {arm} official recording/scorecard disagree with game progress: {official}")
            if item["stop_reason"] not in STOP_REASONS:
                raise RuntimeError(f"arm {arm} ended without a recognized stop reason: {item['stop_reason']!r}")
            if arm == "A" and (facts["frozen"] or facts["requested_construction"]):
                raise RuntimeError("construction activated in the baseline arm")
        if time.monotonic() > global_deadline:
            raise RuntimeError("global gameplay deadline exceeded")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        window.close("cleanup")
        idle = window.wait_idle(max(HTTP_POST_SECONDS.values()) + 5)
        retired = shutdown_bounded(server, 2)
        if server_thread.is_alive():
            server_thread.join(2)
        server.server_close()
        from local_accounting import accounting
        from local_wire import LOCAL_MODEL as WIRE_MODEL
        assert WIRE_MODEL == LOCAL_MODEL
        record = accounting(out)
        roles = {arm: {} for arm in ARMS}
        for receipt in record["receipts"]:
            row_arm = next((r["arm"] for r in _rows(out / "requests.jsonl") if r["call"] == receipt["call"]), None)
            roles[row_arm][receipt["purpose"]] = roles[row_arm].get(receipt["purpose"], 0) + 1
        success = error is None and not state["fatal"] and idle and retired and record["complete"] and len(progress) == len(ARMS)
        summary = {
            "status": "passed" if success else "failed", "error": error, "driver": DRIVER, "games": progress,
            "arms": list(ARMS), "game_set": list(GAMES), "play_game": PLAY_GAME, "game_id": GAME_ID, "stop": state["stop"], "fatal": state["fatal"],
            "calls": state["calls"], "arm_calls": state["arm_calls"], "arm_roles": roles,
            "relay_idle": idle, "relay_retired": retired,
            "local_requests": record,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        (out / "RESULTS.json").write_text(json.dumps(summary, indent=2) + "\n")
        if error and not progress:
            (out / "STARTUP-FAILURE.json").write_text(json.dumps(summary, indent=2) + "\n")
        if not idle or not retired:
            append(out / "cleanup.jsonl", {"status": "unresolved_final_cleanup"})
    return 0 if summary["status"] == "passed" else 1


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


if __name__ == "__main__":
    raise SystemExit(main())
