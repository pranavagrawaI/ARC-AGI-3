"""Local end-to-end qualification of the oracle-hint overlay: real taaf Benchmark + HarnessSolver + ToolAgent + sandbox +
offline engine, with a loopback fake model that replays a sibling run's solving actions as python tool calls.

  ONLY_RESET_LEVELS=true PYTHONPATH=<dfranzen tufa src>:<dfranzen ARC3-Inference> solvers/tycho_nooa/.venv/bin/python -B test_e2e.py

Checks (each prints PASS/FAIL; exit 1 on any FAIL):
  1. hinted attempt starts at level L, the system prompt of every request carries the hint, the sibling's actions
     complete L and the attempt ends 'won' at L+1 with exactly those actions;
  2. unhinted attempt: no request carries the hint header; same solve;
  3. RESET as the first action keeps level L (level-only reset) and the solve still lands;
  4. an attempt that never acts ends at the time cap on level L, not won.
"""
import asyncio, json, os, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
ENV_DIR = Path.home() / "projects/arc-agi-3-reference/competition-env/environment_files"
SCRIPTS = {}   # attempt name -> list of model actions; consumed one per turn
CALLS = []


def model_action(a):
    eng = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE",
           "ACTION7": "UNDO", "RESET": "RESET"}
    if a["id"] == "ACTION6":
        return {"action": "MOUSE", "row": a["data"]["y"], "col": a["data"]["x"]}
    return {"action": eng[a["id"]]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"data":[{"id":"flashnext"}]}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body.get("messages", [])
        system = str(msgs[0].get("content")) if msgs else ""
        name = next((n for n in SCRIPTS if f"TAG={n}" in system), None)
        if name is None:  # the one unhinted attempt carries no TAG
            name = "tn36-L1-plain"
        CALLS.append({"name": name, "system": system, "n_msgs": len(msgs)})
        msg = {"role": "assistant", "content": "", "reasoning_content": "fixture"}
        if msgs and msgs[-1].get("role") == "tool" or name is None or not SCRIPTS.get(name):
            msg["content"] = "Waiting for the next state."
        else:
            act = SCRIPTS[name].pop(0)
            msg["tool_calls"] = [{"id": f"c{len(CALLS)}", "type": "function",
                                  "function": {"name": "python", "arguments": json.dumps({"code": f"action([{act!r}])"})}}]
        reply = {"id": f"f{len(CALLS)}", "object": "chat.completion", "created": 0, "model": "flashnext",
                 "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}
        wire = json.dumps(reply).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(wire))); self.end_headers(); self.wfile.write(wire)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{server.server_port}/v1"
os.environ.update({
    "LOCAL_ANALYZER_BASE_URL": URL, "OPENAI_BASE_URL": URL, "LOCAL_ANALYZER_PROVIDER": "vllm", "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": "flashnext", "INFERENCE_ANALYZER_MODEL": "flashnext", "OPENAI_API_KEY": "x",
    "LOCAL_ANALYZER_API_KEY": "x", "ARC3_MEMORY_SECTIONS": "off", "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "10", "ARC3_ANIMATION": "1", "EXPOSE_UNDO": "on", "ARC3_GUARDS_FROM_LEVEL": "2",
    "ARC3_PERSISTENT_FUNCTIONS": "1", "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game", "ARC3_MAX_ACTIVE_STREAMS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30", "ARC3_HTTP_RETRY_INITIAL_SECONDS": "5", "TAAF_MINIMAL_DIAGNOSTICS": "1",
    "MPLBACKEND": "Agg",
})
EVENTS_PATH = Path(tempfile.mkdtemp()) / "events.jsonl"
os.environ["ORACLE_EVENTS"] = str(EVENTS_PATH)
assert os.environ.get("ONLY_RESET_LEVELS") == "true", "run with ONLY_RESET_LEVELS=true, as the notebook sets it"

import overlay  # noqa: E402
from taaf.benchmark import Benchmark  # noqa: E402
from inference.framework.solver import HarnessSolver  # noqa: E402

starts = {(s["game"], s["level"]): s for s in json.loads((HERE / "starts.json").read_text())}
S = starts[("tn36", 1)]
seg_run = S["sibling_check"][0]["run"]
sys.path.insert(0, str(HERE.parent / "metric-validation-20261008"))
from validate import RUNS, P  # noqa: E402
b = json.loads((RUNS[seg_run] / "benchmark.json").read_text())
r = next(x for x in b["game_runs"] if x["game_id"] == S["gid"])
h, apl = [x["action"] for x in P(r["history"])], P(r["actions_per_level"])
SEG = h[sum(apl[:S["level"]]):sum(apl[:S["level"] + 1])]

cases = [
    dict(name="tn36-L1-hinted", hint="TAG=tn36-L1-hinted. Fixture hint text.", script=[model_action(a) for a in SEG], expect="won"),
    dict(name="tn36-L1-plain", hint="", script=[model_action(a) for a in SEG], expect="won"),
    dict(name="tn36-L1-reset", hint="TAG=tn36-L1-reset.", script=[{"action": "RESET"}] + [model_action(a) for a in SEG], expect="won"),
    dict(name="tn36-L1-idle", hint="TAG=tn36-L1-idle.", script=[], expect="capped"),
]
ok = True
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    attempts = [dict(name=c["name"], gid=S["gid"], level=S["level"], prefix=S["prefix"], hint=c["hint"]) for c in cases]
    for c in cases:
        SCRIPTS[c["name"]] = list(c["script"])
    bm = Benchmark(games=[], job_dir=root, solver=HarnessSolver(model="flashnext", max_runtime_s_per_game=45,
                                                               analyzer_timeout=30, concurrency=4, save_request_logs=True))
    overlay.install(bm, attempts, ENV_DIR)
    asyncio.run(bm.run())
    runs = {r.game_id: r for r in bm.game_runs}
    for c in cases:
        run = runs[c["name"]]
        acts = [x.action.id.name for x in run.history]
        mine = [x for x in CALLS if x["name"] == c["name"]]
        hinted = [x for x in mine if overlay.HINT_HEADER.strip() in x["system"]]
        res = dict(state=run.state, levels=run.levels_completed, actions=len(acts), requests=len(mine), hinted_requests=len(hinted),
                   note=run.solver_note)
        if c["expect"] == "won":
            n_exp = len(SEG) + (1 if c["name"].endswith("reset") else 0)
            good = run.state == "won" and run.levels_completed == S["level"] + 1 and len(acts) == n_exp
        else:
            good = run.state != "won" and run.levels_completed <= S["level"]
        good &= (len(hinted) == len(mine) > 0) if c["hint"] else (len(hinted) == 0 and len(mine) > 0)
        ok &= good
        print("PASS" if good else "FAIL", c["name"], res)
    ev = [json.loads(l) for l in EVENTS_PATH.read_text().splitlines()]
    kinds = {k: sorted(e["attempt"] for e in ev if e["kind"] == k) for k in ("landed", "admitted", "solved", "prefix_failed")}
    good = (len(kinds["landed"]) == 4 and len(kinds["admitted"]) == 4 and kinds["prefix_failed"] == []
            and kinds["solved"] == sorted(c["name"] for c in cases if c["expect"] == "won"))
    ok &= good
    print("PASS" if good else "FAIL", "events", {k: len(v) for k, v in kinds.items()})
    req_logs = sorted(p.name for p in root.rglob("*requests.jsonl"))
    print("request logs:", req_logs)
(HERE / "evidence").mkdir(exist_ok=True)
(HERE / "evidence/test_e2e.json").write_text(json.dumps({"ok": bool(ok), "calls": len(CALLS)}, indent=1))
sys.exit(0 if ok else 1)
