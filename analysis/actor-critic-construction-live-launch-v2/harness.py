#!/usr/bin/env python3
"""Bounded OpenRouter LS20 join. Coordinator launches with --execute. Overlay is frozen."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import signal
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

LIVE = Path(__file__).resolve().parent
ROOT = LIVE.parents[1]
V11 = ROOT / "deployment/kaggle/submission-runtime-20260912-v11"
OVERLAY = LIVE.parent / "actor-critic-construction-integrated"
CARD = ROOT / "analysis/actor-critic-construction-openrouter-20260914T154902Z"
DEFAULT_OUT = CARD / "raw"
sys.path.insert(0, str(LIVE))
sys.path.insert(0, str(V11))

from audit import classify  # noqa: E402
from request_wire import (  # noqa: E402
    MODEL,
    actual_usd,
    conservative_reserve_usd,
    project_outbound,
    request_purpose,
    usage_receipt,
)

MAX_CALLS = 24
ACTOR_SUMMARIZER_CAP = 20
MAX_SPEND = 2.0
MAX_ACTIONS = 8
WALL_SECONDS = 900
HTTP_POST_SECONDS = 30
CONSTRUCTOR_WALL = 45
CONSTRUCTOR_HARD = 50
GAME = "ls20-9607627b"


def append(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def load_key() -> str:
    for path in (ROOT / ".env",):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if key.strip() == "OPENROUTER_API_KEY":
                    return value.strip().strip("\"'")
    raise RuntimeError("OPENROUTER_API_KEY missing from project credential file")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_sources(out: Path) -> dict:
    files = {
        "card": CARD / "EXPERIMENT-CARD.md",
        "harness": LIVE / "harness.py",
        "headless": LIVE / "headless.py",
        "live_agent": LIVE / "live_agent.py",
        "request_wire": LIVE / "request_wire.py",
        "persist": LIVE / "persist.py",
        "persist_job": LIVE / "persist_job.py",
        "audit": LIVE / "audit.py",
        "overlay_service": OVERLAY / "construction_service.py",
        "overlay_job": OVERLAY / "construction_job.py",
        "overlay_headless": OVERLAY / "headless.py",
        "overlay_agent": OVERLAY / "construction_agent.py",
        "v11_adapter": V11 / "adapter.py",
    }
    bound = {
        "complete": False,
        "note": "Partial harness bind only. Coordinator supplies the complete external manifest and approval covering constructor donors, remaining overlay/v11, and environment.",
        "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in files.items()},
    }
    (out / "SOURCE-BIND.json").write_text(json.dumps(bound, indent=2, sort_keys=True) + "\n")
    return bound


def convert_frame():
    source = ROOT / "analysis/deployment-probes/20260904T-nooa-join-public3-depth/kaggle-output/ARC-AGI-3-Agents/agents/agent.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != "85655f7a11a105e50bba76efd00bebf8056d02ddbae827a8f8a5b46e849fc087":
        raise RuntimeError("official Agent source digest drifted")
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "Agent")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_convert_raw_frame_data")
    from arcengine import FrameData, FrameDataRaw

    namespace = {"FrameData": FrameData, "FrameDataRaw": FrameDataRaw}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return lambda raw: namespace["_convert_raw_frame_data"](None, raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fixture-upstream")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if not (CARD / "EXPERIMENT-CARD.md").is_file():
        raise SystemExit("missing coordinator card parent")
    if args.execute and args.fixture_upstream:
        raise SystemExit("refuse combining --execute with --fixture-upstream")
    if out.exists():
        raise SystemExit(f"refuse reused raw: {out}")
    if not args.execute and not args.fixture_upstream:
        print(json.dumps({
            "source_verified": False,
            "source_bind": "not_run",
            "paid_calls": 0,
            "out": str(out),
            "card": str(CARD),
            "caps": {
                "calls": MAX_CALLS,
                "actor_summarizer_cap": ACTOR_SUMMARIZER_CAP,
                "spend_usd": MAX_SPEND,
                "actions": MAX_ACTIONS,
                "wall_seconds": WALL_SECONDS,
                "http_post_seconds": HTTP_POST_SECONDS,
                "constructor_wall_seconds": CONSTRUCTOR_WALL,
                "constructor_hard_seconds": CONSTRUCTOR_HARD,
            },
            "note": "HttpTransport overlay POST timeout remains 30s (initial attempt cap). 50s is process hard kill, not HTTP.",
            "driver": "test_join_qualification.build_agent: real local worker/game, not Agent.main",
        }))
        return 0
    out.mkdir(parents=False)
    (out / "mpl").mkdir()
    bind_sources(out)
    (out / "STARTED.json").write_text(json.dumps({
        "started": time.time(),
        "card": str(CARD),
        "paid": bool(args.execute),
        "driver": "test_join_qualification.build_agent: real local worker/game, not Agent.main",
        "source_bind": "partial_harness_only",
    }, indent=2) + "\n")
    if args.execute:
        key = load_key()
        upstream = "https://openrouter.ai/api/v1/chat/completions"
        for name in list(os.environ):
            if "KEY" in name or "TOKEN" in name:
                del os.environ[name]
    else:
        key = "fixture"
        upstream = args.fixture_upstream
    from probe_lifecycle import AdmissionWindow, shutdown_bounded

    state = {
        "calls": 0,
        "actor_summarizer": 0,
        "reserved": 0.0,
        "spent_usd": 0.0,
        "stop": None,
        "fatal": False,
        "constructor_undelivered": False,
        "qualified": False,
    }
    window = AdmissionWindow(WALL_SECONDS)
    lock = threading.Lock()

    def abort(reason: str) -> None:
        window.close(reason)
        state["stop"] = reason
        for ipc in out.rglob("ipc"):
            (ipc / "ABORT").write_text(reason)

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
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                abort("input cap")
                self.send_error(429)
                return
            data = self.rfile.read(length)
            with lock:
                payload = json.loads(data)
                if classify(out).get("success"):
                    state["qualified"] = True
                    append(out / "admission-stops.jsonl", {"reason": "QUALIFICATION_COMPLETE", "paid_request_admitted": False})
                    abort("QUALIFICATION_COMPLETE")
                    self.send_error(409, "QUALIFICATION_COMPLETE")
                    return
                purpose = request_purpose(payload)
                try:
                    body = project_outbound(payload)
                except ValueError as error:
                    state["fatal"] = True
                    abort(str(error))
                    self.send_error(500)
                    return
                reserve = conservative_reserve_usd(body)
                actor_like = purpose in {"actor", "summarizer"}
                if state["calls"] >= MAX_CALLS or state["spent_usd"] + reserve > MAX_SPEND or state["stop"]:
                    abort("call/spend cap")
                    self.send_error(429)
                    return
                if actor_like and state["actor_summarizer"] >= ACTOR_SUMMARIZER_CAP:
                    abort("actor/summarizer reserve for constructor+critic")
                    self.send_error(429)
                    return
                state["calls"] += 1
                state["reserved"] += reserve
                if actor_like:
                    state["actor_summarizer"] += 1
                identity = {
                    "call": state["calls"],
                    "purpose": purpose,
                    "reservation": reserve,
                    "time": time.time(),
                    "wire_projection": {"stripped": ["parallel_tool_calls", "chat_template_kwargs"], "provider": body["provider"]},
                }
                append(out / "requests.jsonl", {**identity, "request": body})
                raw = None
                try:
                    request = urllib.request.Request(
                        upstream,
                        data=json.dumps(body).encode(),
                        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=HTTP_POST_SECONDS) as response:
                        raw = response.read()
                    parsed = json.loads(raw)
                    if parsed.get("model") != MODEL or parsed.get("provider") != "CoreWeave":
                        append(out / "responses.jsonl", {**identity, "response": parsed, "identity_status": "mismatch"})
                        abort("provider/model identity mismatch")
                        self.send_error(502)
                        return
                    receipt = usage_receipt(parsed)
                    if receipt is None:
                        append(out / "responses.jsonl", {**identity, "response": parsed, "usage_status": "missing"})
                        abort("missing usage receipt; no forwarding retry")
                        self.send_error(502)
                        return
                    spent = actual_usd(receipt)
                    state["spent_usd"] += spent
                    state["reserved"] = state["spent_usd"]
                    identity = {**identity, "usage": receipt, "actual_usd": spent}
                    append(out / "responses.jsonl", {**identity, "response": parsed})
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(raw)
                    except (BrokenPipeError, ConnectionResetError):
                        append(out / "delivery.jsonl", {**identity, "status": "completed_response_not_delivered"})
                        state["fatal"] = True
                        abort("completed response not delivered")
                        if purpose == "constructor":
                            state["constructor_undelivered"] = True
                except Exception as error:
                    body_text = error.read().decode(errors="replace")[:8000] if hasattr(error, "read") else None
                    append(out / "errors.jsonl", {**identity, "type": type(error).__name__, "message": str(error), "body": body_text})
                    abort("upstream failure; no forwarding retry")
                    try:
                        self.send_error(502)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

    server = HTTPServer(("127.0.0.1", 0), Relay)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    relay_url = f"http://127.0.0.1:{server.server_port}/v1"
    constructor_url = relay_url + "/chat/completions"
    os.environ.update({
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(out / "mpl"),
        "ARC_DISABLE_DOTENV": "1",
        "ARC_LLM_MODEL": MODEL,
        "ARC_LLM_API_KEY": "local-relay",
        "ARC_LLM_BASE_URL": relay_url,
        "ARC_LLM_EXTRA_BODY": '{"reasoning":{"enabled":false}}',
        "ARC_STALL_REVIEW": "0",
        "ARC_LLM_MAX_TOKENS": "16384",
        "ARC_LLM_TIMEOUT_SECONDS": "300",
        "ARC_SOLVER_ARM": "loop",
        "ARC_DISCOVERY_ARM": "a",
        "TYCHO_KAGGLE_MAX_ACTIONS": str(MAX_ACTIONS),
        "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS": "900",
        "NOOA_AGENT_READY_TIMEOUT_SECONDS": "30",
        "NOOA_JOIN_ROOT": str(V11),
        "NOOA_AGENT_COMMAND": json.dumps([
            sys.executable, str(LIVE / "headless.py"),
            "--constructor-transport", "http",
            "--constructor-url", constructor_url,
            "--constructor-wall-seconds", str(CONSTRUCTOR_WALL),
            "--constructor-hard-seconds", str(CONSTRUCTOR_HARD),
            "--run-dir", ".",
        ]),
    })
    sys.path[:0] = [str(ROOT / "deployment/kaggle/tests"), str(ROOT / "deployment/kaggle"), str(V11)]
    spec = importlib.util.spec_from_file_location("runtime.local_solver", V11 / "shared_local_solver.py")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules["runtime.local_solver"] = runtime
    spec.loader.exec_module(runtime)
    import pytest
    import test_my_agent
    import test_join_qualification as qual
    from arc_agi import Arcade, OperationMode

    test_my_agent.AGENT_PATH = V11 / "adapter.py"
    convert = convert_frame()
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files"), recordings_dir=str(out / "recordings"))
    env = arcade.make(GAME)
    current = convert(env.reset())
    patch = pytest.MonkeyPatch()
    agent = qual.build_agent(patch, out, current)
    error = None
    start = time.monotonic()

    def timeout(*_: object) -> None:
        abort("900s wall")
        raise TimeoutError("900s wall")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(WALL_SECONDS)
    steps = []
    try:
        for _ in range(MAX_ACTIONS):
            if agent.is_done(agent.frames, current):
                break
            if classify(out).get("success"):
                break
            action = agent.choose_action(agent.frames, current)
            agent.action_counter += 1
            current = convert(env.step(action))
            agent.frames.append(current)
            steps.append({"action": getattr(action, "name", str(action)), "levels_completed": current.levels_completed})
            append(out / "environment-steps.jsonl", steps[-1])
            if classify(out).get("success"):
                break
    except Exception as exc:
        if state["qualified"] and classify(out).get("success"):
            solver = agent._solver
            retired = (
                solver is not None and solver._closed
                and solver._process.poll() is not None
                and not runtime._MODEL_GENERATION_RETIREMENT_FAILED
                and solver._pending is None and agent._pending_decision is None
            )
            if retired:
                assert solver is not None
                # decide() already abort-retired this worker. Do not send a
                # fictitious finish behind a closed protocol transport.
                agent._finished = True
                append(out / "cleanup.jsonl", {"status": "abort_retired_after_qualification", "worker_returncode": solver._process.returncode, "finish_sent": False})
            else:
                error = "qualification observed but worker retirement unresolved"
        else:
            error = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        window.close("cleanup")
        if error:
            abort(error)
        try:
            agent.cleanup(scorecard=None)
        except Exception as exc:
            error = error or f"cleanup: {exc}"
        patch.undo()
        if not window.wait_idle(HTTP_POST_SECONDS + 5):
            error = error or "provider transport did not retire within cleanup bound"
            state["fatal"] = True
        if not shutdown_bounded(server, 2):
            append(out / "cleanup.jsonl", {"status": "unresolved"})
            error = error or "relay shutdown unresolved"
    (out / "environment-steps.json").write_text(json.dumps(steps, indent=2) + "\n")
    audit = classify(out)
    if state["constructor_undelivered"]:
        audit["verdict"] = "CONSTRUCTOR_RESPONSE_NOT_DELIVERED"
        audit["success"] = False
    summary = {
        "error": error,
        "stop": state["stop"],
        "qualification_complete": state["qualified"],
        "fatal": state["fatal"],
        "calls": state["calls"],
        "actor_summarizer": state["actor_summarizer"],
        "reserved_usd": state["reserved"],
        "spent_usd": state["spent_usd"],
        "actions": len(steps),
        "wall_seconds": time.monotonic() - start,
        "audit": audit,
        "http_post_seconds": HTTP_POST_SECONDS,
        "driver": "test_join_qualification.build_agent: real local worker/game, not Agent.main",
        "source_bind": "partial_harness_only",
    }
    (out / "RESULTS.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("error", "stop", "calls", "actions", "audit")}))
    if state["fatal"] or error or not audit.get("success"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

