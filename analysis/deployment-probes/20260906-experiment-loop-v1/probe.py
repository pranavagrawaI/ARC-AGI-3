"""Bounded live diagnostic runs; relay holds the real credential outside the actor."""

import ast
import hashlib
import json
import os
import signal
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "deployment/kaggle/q6-nooa-candidate"
for p in (ROOT / ".env", Path("/home/pranav/projects/arc-agi-3/.env")):
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ["ARC_LLM_API_KEY"]
# Never pass the upstream credential into the actor worker.
for name in list(os.environ):
    if "KEY" in name and ("OPENROUTER" in name or name.startswith("ARC_LLM")):
        del os.environ[name]
os.environ.update(
    LITELLM_LOCAL_MODEL_COST_MAP="True",
    MPLCONFIGDIR=str(OUT / "mpl"),
    ARC_LLM_MODEL="qwen/qwen3.8-27b",
    ARC_LLM_API_KEY="local-relay",
    ARC_LLM_EXTRA_BODY='{"reasoning":{"enabled":false}}',
    TYCHO_KAGGLE_MAX_ACTIONS="40",
    TYCHO_NOOA_WORKER_TIMEOUT_SECONDS="150",
    NOOA_AGENT_READY_TIMEOUT_SECONDS="30",
)
os.environ["NOOA_AGENT_COMMAND"] = json.dumps(
    [sys.executable, str(OUT / "headless.py"), "--run-dir", "."]
)
state = {"reserved": 0.0, "calls": 0, "arm": "", "stop": None}
lock = threading.Lock()


def append(name, row):
    with (OUT / name).open("a") as f:
        f.write(json.dumps(row) + "\n")


def abort(reason):
    state["stop"] = reason
    for ipc in (OUT / state["arm"]).rglob("ipc"):
        (ipc / "ABORT").write_text(reason)


class Relay(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        with lock:
            reserve = len(data) * 0.00000042 + 4096 * 0.000003 + 0.01
            if (
                state["calls"] >= 12
                or state["reserved"] + reserve > 3.0
                or state["run_reserved"] + reserve > 0.75
                or len(data) > 80000
                or state["stop"]
            ):
                abort("diagnostic call/input/budget cap")
                self.send_error(429, "diagnostic cap")
                return
            payload = json.loads(data)
            payload.update(
                model="qwen/qwen3.8-27b",
                max_tokens=4096,
                stream=False,
                reasoning={"enabled": False},
            )
            payload["provider"] = {
                "max_price": {"prompt": 0.42, "completion": 3, "request": 0},
                "allow_fallbacks": False,
            }
            state["calls"] += 1
            state["reserved"] += reserve
            state["run_reserved"] += reserve
            identity = {
                "arm": state["arm"],
                "call": state["calls"],
                "time": time.time(),
            }
            append(
                "requests.jsonl",
                {**identity, "request": payload, "reserved_dollars": reserve},
            )
            try:
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={
                        "Authorization": "Bearer " + KEY,
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=120) as response:
                    raw = response.read()
                parsed = json.loads(raw)
                append("responses.jsonl", {**identity, "response": parsed})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)
            except Exception as error:
                append("responses.jsonl", {**identity, "error": str(error)})
                abort("upstream request failed; no retry")
                self.send_error(502, "upstream failure")


def main():
    assert not (OUT / "requests.jsonl").exists(), (
        "Use a new card/run directory for a retry"
    )
    server = HTTPServer(("127.0.0.1", 0), Relay)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ["ARC_LLM_BASE_URL"] = f"http://127.0.0.1:{server.server_port}/v1"
    sys.path[:0] = [
        str(ROOT / "deployment/kaggle/tests"),
        str(ROOT / "deployment/kaggle"),
    ]
    import pytest
    import test_nooa_join as join
    import test_join_qualification as qual
    from arc_agi import Arcade, OperationMode
    from arcengine import FrameData, FrameDataRaw

    join.CANDIDATE = CANDIDATE
    source = (
        ROOT
        / "analysis/deployment-probes/20260904T-nooa-join-public3-depth/kaggle-output/ARC-AGI-3-Agents/agents/agent.py"
    )
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "85655f7a11a105e50bba76efd00bebf8056d02ddbae827a8f8a5b46e849fc087"
    )
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Agent"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_convert_raw_frame_data"
    )
    ns = {"FrameData": FrameData, "FrameDataRaw": FrameDataRaw}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns)

    def convert(raw):
        return ns["_convert_raw_frame_data"](None, raw)

    results = []
    for arm in ("control-a", "workflow-a", "workflow-b", "control-b"):
        path = OUT / arm
        path.mkdir(exist_ok=True)
        assert not (path / "workspaces").exists(), "Do not reuse a run"
        state.update(arm=arm, calls=0, run_reserved=0.0, stop=None)
        variant = "1" if arm.startswith("workflow") else "0"
        actor_script = OUT / f"actor-{variant}.py"
        os.environ["NOOA_AGENT_COMMAND"] = json.dumps(
            [sys.executable, str(actor_script), "--run-dir", "."]
        )
        os.environ["ARC_STALL_REVIEW"] = "0"
        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(ROOT / "environment_files"),
            recordings_dir=str(path / "recordings"),
        )
        env = arcade.make("ls20")
        current = convert(env.reset())
        patch = pytest.MonkeyPatch()
        agent = qual.build_agent(patch, path, current)
        start = time.monotonic()
        error = None

        def timed_out(*_):
            raise TimeoutError("600s diagnostic wall cap")

        signal.signal(signal.SIGALRM, timed_out)
        signal.alarm(600)
        try:
            for _ in range(82):
                if agent.is_done(agent.frames, current):
                    break
                action = agent.choose_action(agent.frames, current)
                agent.action_counter += 1
                current = convert(env.step(action))
                agent.frames.append(current)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            signal.alarm(0)
            try:
                agent.cleanup(scorecard=None)
            except Exception as exc:
                error = error or f"cleanup: {exc}"
            patch.undo()
        result = {
            "arm": arm,
            "decisions": agent._solver_decisions_emitted,
            "levels": current.levels_completed,
            "calls": state["calls"],
            "wall_seconds": time.monotonic() - start,
            "error": error,
            "stop": state["stop"],
            "total_reserved_dollars": state["reserved"],
        }
        results.append(result)
        (OUT / "RESULTS.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    server.shutdown()


if __name__ == "__main__":
    main()
