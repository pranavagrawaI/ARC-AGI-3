#!/usr/bin/env python3
"""Official-launcher probe: local SGLang LS20 actor/critic/constructor. Not Agent.main."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = "test_join_qualification.build_agent: real local worker/game, not Agent.main"
LOCAL_MODEL = "/kaggle/input/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot"
MAX_CALLS = 24
ACTOR_SUMMARIZER_CAP = 20
MAX_ACTIONS = 8
WALL_SECONDS = 900
HTTP_POST_SECONDS = 30
CONSTRUCTOR_WALL = 45
CONSTRUCTOR_HARD = 50
GAME = "ls20"


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


def convert_frame(root: Path):
    source = root / "analysis/deployment-probes/20260904T-nooa-join-public3-depth/kaggle-output/ARC-AGI-3-Agents/agents/agent.py"
    if not source.is_file():
        # kernel package may ship the hashed convert helper only
        helper = HERE / "convert_frame.py"
        spec = importlib.util.spec_from_file_location("convert_frame", helper)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.convert
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
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--startup-argv-json", required=True)
    parser.add_argument("--environments", type=Path, default=Path("/kaggle/input/arc-prize-2026-arc-agi-3/environment_files"))
    args = parser.parse_args()
    if args.model != LOCAL_MODEL:
        raise SystemExit("local model identity drift")
    out = args.workspace_root / "raw"
    out.mkdir(parents=True, exist_ok=False)
    require_seal(out)
    inspect_gpu(out)
    startup_argv = json.loads(args.startup_argv_json)
    from local_wire import validate_identity
    validate_identity(args.model, args.model, startup_argv)
    (out / "MODEL-IDENTITY.json").write_text(json.dumps({"model": args.model, "mounted_weights": args.model, "startup_argv": startup_argv}))
    sys.path.insert(0, str(HERE))
    from audit import classify
    from probe_lifecycle import AdmissionWindow, shutdown_bounded

    state = {"calls": 0, "actor_summarizer": 0, "stop": None, "fatal": False, "qualified": False, "constructor_undelivered": False}
    window = AdmissionWindow(min(WALL_SECONDS, int(args.timeout_s)))
    import threading

    lock = threading.Lock()

    def abort(reason: str) -> None:
        window.close(reason)
        state["stop"] = state["stop"] or reason
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
            from local_wire import project_local, request_purpose, usage_receipt, valid_response

            data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(data)
            purpose = request_purpose(payload)
            actor_like = purpose in {"actor", "summarizer"}
            with lock:
                if classify(out).get("success"):
                    state["qualified"] = True
                    append(out / "admission-stops.jsonl", {"reason": "QUALIFICATION_COMPLETE", "model_request_admitted": False})
                    abort("QUALIFICATION_COMPLETE")
                    self.send_error(409, "QUALIFICATION_COMPLETE")
                    return
                if state["calls"] + 1 >= MAX_CALLS or state["stop"]:
                    # The supervisor owns exactly one separately reconciled warmup.
                    # Keep total server chat calls within 24, not 24 + warmup.
                    abort("call cap")
                    self.send_error(429)
                    return
                if actor_like and state["actor_summarizer"] >= ACTOR_SUMMARIZER_CAP:
                    abort("actor/summarizer reserve")
                    self.send_error(429)
                    return
                try:
                    body = project_local(payload, local_model=LOCAL_MODEL, mounted_weights=LOCAL_MODEL, startup_argv=startup_argv)
                except ValueError as error:
                    state["fatal"] = True
                    abort(str(error))
                    append(out / "local-rejections.jsonl", {"error": str(error), "sent": False})
                    self.send_error(400)
                    return
                state["calls"] += 1
                if actor_like:
                    state["actor_summarizer"] += 1
                identity = {"call": state["calls"], "purpose": purpose, "time": time.time()}
                append(out / "logical-requests.jsonl", {**identity, "request": payload})
                append(out / "requests.jsonl", {**identity, "request": body})
                request = urllib.request.Request(
                    args.base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                started = time.monotonic()
                try:
                    with urllib.request.urlopen(request, timeout=HTTP_POST_SECONDS) as response:
                        raw = response.read()
                    parsed = json.loads(raw)
                    (out / "raw-http").mkdir(exist_ok=True)
                    (out / "raw-http" / f"{state['calls']:02d}.body").write_bytes(raw)
                    append(out / "responses.jsonl", {**identity, "response": parsed})
                    if not valid_response(parsed):
                        abort("local model identity/envelope mismatch")
                        self.send_error(502)
                        return
                    receipt = usage_receipt(parsed)
                    if receipt is None:
                        abort("missing usage receipt; no forwarding retry")
                        self.send_error(502)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    try:
                        self.wfile.write(raw)
                    except (BrokenPipeError, ConnectionResetError):
                        state["fatal"] = True
                        state["constructor_undelivered"] = purpose == "constructor"
                        append(out / "delivery.jsonl", {**identity, "status": "completed_response_not_delivered"})
                        abort("completed response not delivered")
                except Exception as error:
                    abort("upstream failure; no forwarding retry")
                    with (out / "errors.jsonl").open("a") as handle:
                        handle.write(json.dumps({**identity, "type": type(error).__name__, "message": str(error)}) + "\n")
                    self.send_error(502)
                finally:
                    append(out / "latency.jsonl", {**identity, "elapsed_seconds": time.monotonic() - started})

    server = HTTPServer(("127.0.0.1", 0), Relay)
    try:
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        relay = f"http://127.0.0.1:{server.server_port}/v1"
        os.environ.update({
            "ARC_DISABLE_DOTENV": "1",
            "ARC_LLM_MODEL": "qwen/qwen3.8-27b",
            "ARC_LLM_API_KEY": "local-sglang",
            "ARC_LLM_BASE_URL": relay,
            "ARC_LLM_EXTRA_BODY": '{"reasoning":{"enabled":false}}',
            "ARC_STALL_REVIEW": "0",
            "ARC_LLM_MAX_TOKENS": "16384",
            "ARC_SOLVER_ARM": "loop",
            "ARC_DISCOVERY_ARM": "a",
            "TYCHO_KAGGLE_MAX_ACTIONS": str(MAX_ACTIONS),
            "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS": "900",
            "NOOA_AGENT_READY_TIMEOUT_SECONDS": "30",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        })
        # Game loop is coordinator-launched remotely; this probe records identity and refuses Agent.main labeling.
        result = {
            "schema": "actor-critic-construction-kaggle-probe-v1",
            "status": "started",
            "driver": DRIVER,
            "model": LOCAL_MODEL,
            "calls": state["calls"],
            "gpu_wall_seconds": WALL_SECONDS,
            "openrouter": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        v11 = HERE / "nooa-join"
        os.environ["NOOA_JOIN_ROOT"] = str(v11)
        os.environ["NOOA_AGENT_COMMAND"] = json.dumps([
            sys.executable, str(HERE / "headless.py"),
            "--constructor-transport", "http",
            "--constructor-url", relay + "/chat/completions",
            "--constructor-wall-seconds", str(CONSTRUCTOR_WALL),
            "--constructor-hard-seconds", str(CONSTRUCTOR_HARD),
            "--run-dir", ".",
        ])
        sys.path[:0] = [str(HERE), str(v11), str(v11 / "chassis"), str(Path("/kaggle/working"))]
        spec = importlib.util.spec_from_file_location("runtime.local_solver", v11 / "shared_local_solver.py")
        runtime = importlib.util.module_from_spec(spec)
        sys.modules["runtime.local_solver"] = runtime
        spec.loader.exec_module(runtime)
        from driver_patch import DriverPatch
        import test_join_qualification as qual
        import test_my_agent
        from arc_agi import Arcade, OperationMode

        test_my_agent.AGENT_PATH = v11 / "adapter.py"
        convert = convert_frame(Path("/kaggle/working"))
        arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(args.environments), recordings_dir=str(out / "recordings"))
        env = arcade.make("ls20")
        current = convert(env.reset())
        patch = DriverPatch()
        agent = qual.build_agent(patch, out, current)
        error = None
        steps = []
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("900s wall")))
        signal.alarm(WALL_SECONDS)
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
                with (out / "environment-steps.jsonl").open("a") as handle:
                    handle.write(json.dumps(steps[-1]) + "\n")
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
            server_thread.join(2)
        audit = classify(out)
        success = not error and not state["fatal"] and audit.get("success")
        from local_accounting import accounting
        summary = {"status": "passed" if success else "failed", "error": error,
            "stop": state["stop"], "qualification_complete": state["qualified"],
            "fatal": state["fatal"], "actor_summarizer": state["actor_summarizer"],
            "calls": state["calls"], "actions": len(steps), "audit": audit, "driver": DRIVER,
            "local_requests": accounting(out)}
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        (out / "RESULTS.json").write_text(json.dumps(summary, indent=2) + "\n")
        return 0 if success else 1
    except BaseException as exc:
        from local_accounting import accounting
        failure = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "driver": DRIVER, "local_requests": accounting(out)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(failure, indent=2) + "\n")
        (out / "STARTUP-FAILURE.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        window.close("final cleanup")
        idle = window.wait_idle(HTTP_POST_SECONDS + 5)
        retired = shutdown_bounded(server, 2)
        if 'server_thread' in locals():
            server_thread.join(2)
        server.server_close()
        if not idle or not retired:
            append(out / "cleanup.jsonl", {"status": "unresolved_final_cleanup"})
            failure = {"status": "failed", "error": "relay final cleanup unresolved", "driver": DRIVER}
            args.output.write_text(json.dumps(failure, indent=2) + "\n")
            (out / "RESULTS.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise RuntimeError("relay final cleanup unresolved")



if __name__ == "__main__":
    raise SystemExit(main())
