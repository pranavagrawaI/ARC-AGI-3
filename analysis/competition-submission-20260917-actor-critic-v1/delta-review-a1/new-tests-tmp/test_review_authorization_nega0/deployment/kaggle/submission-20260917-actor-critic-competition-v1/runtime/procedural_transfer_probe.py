"""Run one bounded synthetic two-decision Qwen transfer screen."""
import argparse, hashlib, importlib.util, json, os, sys, threading, time, urllib.request, signal, ast
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "analysis/deployment-probes/20260908-procedural-transfer-screen-rerun2"
RAW = OUT / "raw"
CARD = OUT / "EXPERIMENT-CARD.md"
MODEL = "qwen/qwen3.8-27b"

def append(name, row):
    with (RAW / name).open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")

def load_key():
    for path in (ROOT / ".env", Path("/home/pranav/projects/arc-agi-3/.env")):
        if path.exists():
            for line in path.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    return os.environ["OPENROUTER_API_KEY"]

def main():
    global RAW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument('--preflight', choices=['success', 'call_limit', 'deadline', 'rejection'])
    args = parser.parse_args()
    from freeze_candidate import verify
    manifest = verify()
    if not args.execute and not args.preflight:
        print(json.dumps({"source_verified": manifest["contract_sha256"], "paid_calls": 0}))
        return
    if args.preflight:
        RAW = OUT / ('preflight-' + args.preflight)
    RAW.mkdir(parents=True, exist_ok=False)
    if (OUT / "STARTED.json").exists():
        raise RuntimeError("started run is immutable; no retry")
    provider_key = load_key() if args.execute else 'mock-only'
    if args.execute:
        for mode in ('success', 'call_limit', 'deadline', 'rejection'):
            record = json.loads((OUT / ('preflight-' + mode) / 'RESULTS.json').read_text())
            assert record['manifest'] == manifest['contract_sha256']
            assert record['preflight_pass'], mode
    for name in list(os.environ):
        if "KEY" in name or "TOKEN" in name:
            del os.environ[name]
    state = {"calls": 0, "reserved": 0.0, "fatal": None, 'decision': 0, 'per_decision': [0, 0]}
    deadline = time.monotonic() + (0 if args.preflight == 'deadline' else 600)
    def abort(reason):
        state['fatal'] = state['fatal'] or reason
        for ipc in (RAW / 'worker').rglob('ipc'):
            (ipc / 'ABORT').write_text(reason)
    lock = threading.Lock()

    class Relay(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            names = {item.get("function", {}).get("name") for item in payload.get("tools", [])}
            append("received_requests.jsonl", {"time": time.time(), "request": payload})
            if state['fatal'] or time.monotonic() >= deadline:
                abort(state['fatal'] or 'deadline'); self.send_error(400); return
            if args.preflight == 'rejection':
                abort('injected rejection'); self.send_error(400); return
            if "execute_python" not in names:
                state["fatal"] = "summarizer request"
                abort('summarizer request'); self.send_error(400); return
            with lock:
                reserve = length * 0.00000042 + 16384 * 0.000003 + 0.01
                if state['per_decision'][state['decision']] >= 2 or state["calls"] >= 4 or state["reserved"] + reserve > 1:
                    state["fatal"] = "call/spend cap"
                    abort('call/spend cap'); self.send_error(400); return
                if payload.get("reasoning", {}).get("enabled") is not False or payload.get('chat_template_kwargs', {}).get('enable_thinking') is not False or payload.get('max_tokens') != 16384:
                    state["fatal"] = "reasoning switch drift"
                    self.send_error(500); return
                rendered = json.dumps(payload, ensure_ascii=False)
                content = '\n'.join(str(m.get('content', '')) for m in payload['messages'])
                demo = (HERE / 'chassis/discovery-demo.md').read_text().strip()
                if demo not in content or "PROCEDURAL TRANSFER DIAGNOSTIC" not in rendered or 'Your PLAN lives in code' in content or '<arc_skill>' in content:
                    state["fatal"] = "diagnostic prompt drift"
                    abort('diagnostic prompt drift'); self.send_error(400); return
                state["calls"] += 1; state["reserved"] += reserve
                state['per_decision'][state['decision']] += 1
                identity = {"call": state["calls"], "time": time.time()}
            payload.pop("parallel_tool_calls", None)
            payload.update(model=MODEL, stream=False, temperature=0.2, top_p=0.95, top_k=20,
                provider={"only": ["coreweave/fp8"], "allow_fallbacks": False,
                          'max_price': {'prompt': .42, 'completion': 3, 'request': 0},
                          "require_parameters": True})
            append("requests.jsonl", {**identity, "request": payload, "reservation": reserve})
            try:
                request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + provider_key,
                    "Content-Type": "application/json"})
                if args.preflight:
                    code = "print(self.grid_summary())" if args.preflight == 'call_limit' else (
                        "g = self.model_grid()\nprint(self.grid_summary())\nself.set_experiment('Does UP move the marker?', ['upward movement', 'no effect'], 'UP', {'upward movement': 'support here', 'unchanged': 'reject here', 'inconclusive': 'unresolved'})\nself.submit_actions(['UP'], rationale='predict: upward movement or no effect')" if state['decision'] == 0 else
                        "print(self.changed_region())\nt = self.last_transition(True)\nassert t['changed_cells'] == 0\nself.set_experiment('Does DOWN move the marker?', ['movement', 'no effect'], 'DOWN', {'changed': 'support here', 'unchanged': 'reject here', 'inconclusive': 'unresolved'})\nself.submit_actions(['DOWN'], rationale='predict: UP was unchanged; test DOWN for movement')")
                    raw = json.dumps({'id': 'mock', 'object': 'chat.completion', 'created': 1, 'model': MODEL, 'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c' + str(state['calls']), 'type': 'function', 'function': {'name': 'execute_python', 'arguments': json.dumps({'code': code})}}]}}], 'usage': {'prompt_tokens': 100, 'completion_tokens': 100, 'completion_tokens_details': {'reasoning_tokens': 0}}}).encode()
                else:
                    with urllib.request.urlopen(request, timeout=max(1, min(240, deadline - time.monotonic()))) as response:
                        raw = response.read()
                parsed = json.loads(raw)
                append("responses.jsonl", {**identity, "response": parsed})
                reasoning = (parsed.get('usage', {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
                if reasoning != 0:
                    abort('reasoning usage nonzero or missing'); self.send_error(400); return
                for call in parsed['choices'][0]['message'].get('tool_calls', []):
                    fn = call['function']
                    if fn['name'] != 'execute_python':
                        abort('non-CodeAct response'); self.send_error(400); return
                    tree = ast.parse(json.loads(fn['arguments'])['code'])
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                            if node.func.attr in ('write_helper', 'load_helpers', 'save_procedure', 'deep_investigate'):
                                abort('helper/procedure detour'); self.send_error(400); return
                            if node.func.attr == 'set_experiment':
                                supplied = len(node.args) + len(node.keywords)
                                if supplied != 4 or any(k.arg not in ('question', 'hypotheses', 'action', 'outcomes') for k in node.keywords):
                                    abort('malformed experiment signature'); self.send_error(400); return
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
            except Exception as error:
                body = error.read().decode(errors="replace")[:8000] if hasattr(error, "read") else None
                append("errors.jsonl", {**identity, "type": type(error).__name__, "message": str(error), "body": body})
                abort('upstream failure')
                try: self.send_error(400)
                except Exception: pass

    server = HTTPServer(("127.0.0.1", 0), Relay)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    os.environ.update(LITELLM_LOCAL_MODEL_COST_MAP="True", ARC_DISABLE_DOTENV="1",
        ARC_LLM_MODEL=MODEL, ARC_LLM_API_KEY="local-relay",
        ARC_LLM_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
        ARC_LLM_EXTRA_BODY='{"reasoning":{"enabled":false},"chat_template_kwargs":{"enable_thinking":false}}',
        ARC_LLM_MAX_TOKENS="16384", ARC_LLM_TIMEOUT_SECONDS="240", ARC_STALL_REVIEW="0",
        ARC_PROCEDURAL_DIAGNOSTIC="1", TYCHO_KAGGLE_MAX_ACTIONS="2",
        TYCHO_NOOA_WORKER_TIMEOUT_SECONDS="600", NOOA_AGENT_READY_TIMEOUT_SECONDS="30",
        NOOA_JOIN_ROOT=str(HERE), NOOA_AGENT_COMMAND=json.dumps([sys.executable, str(HERE / "headless.py"), "--run-dir", "."]))
    ((OUT if args.execute else RAW) / "STARTED.json").write_text(json.dumps({"manifest": manifest,
        "card_sha256": hashlib.sha256(CARD.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "started": time.time()}, indent=2) + "\n")
    sys.path[:0] = [str(ROOT / "deployment/kaggle/tests"), str(ROOT / "deployment/kaggle")]
    spec = importlib.util.spec_from_file_location("runtime.local_solver", HERE / "shared_local_solver.py")
    runtime = importlib.util.module_from_spec(spec); sys.modules["runtime.local_solver"] = runtime; spec.loader.exec_module(runtime)
    import pytest
    import test_my_agent
    import test_nooa_join as join
    import test_join_qualification as qual
    from arcengine import GameAction
    test_my_agent.AGENT_PATH = HERE / "adapter.py"; join.CANDIDATE = HERE
    patch = pytest.MonkeyPatch()
    actions = (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4)
    initial = test_my_agent.frame([[0,0,0],[0,3,0],[0,0,0]], actions=actions)
    agent = qual.build_agent(patch, RAW / "worker", initial)
    error = None; chosen = []
    def expired(*_):
        abort('deadline')
        raise TimeoutError('600s admission deadline')
    signal.signal(signal.SIGALRM, expired); signal.alarm(600)
    try:
        first = agent.choose_action(agent.frames, initial); chosen.append(first.name); agent.action_counter += 1
        successors = {
            "ACTION1": [[0,0,0],[0,3,0],[0,0,0]],
            "ACTION2": [[0,3,0],[0,0,0],[0,0,0]],
            "ACTION3": [[0,0,0],[0,0,0],[0,3,0]],
            "ACTION4": [[0,0,0],[0,0,3],[0,0,0]],
        }
        second_frame = test_my_agent.frame(successors[first.name], actions=actions)
        agent.frames.append(second_frame)
        state['decision'] = 1
        second = agent.choose_action(agent.frames, second_frame); chosen.append(second.name); agent.action_counter += 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        state['closed_before_cleanup'] = True
        closed_reason = state['fatal']
        abort('cleanup')
        try: agent.cleanup(scorecard=None)
        except Exception as exc: error = error or f"cleanup: {exc}"
        patch.undo(); server.shutdown(); thread.join(timeout=3)
    ipc = next((RAW / "worker").rglob("ipc"), None)
    def lines(name):
        path = ipc / name if ipc else None
        return [json.loads(line) for line in path.read_text().splitlines()] if path and path.exists() else []
    preflight_pass = (chosen == ['ACTION1', 'ACTION2'] and not error) if args.preflight == 'success' else (bool(error) and state['calls'] == (2 if args.preflight == 'call_limit' else 0))
    result = {'manifest': manifest['contract_sha256'], 'preflight_pass': preflight_pass if args.preflight else None, 'per_decision': state['per_decision'], "calls": state["calls"], "reserved": state["reserved"], "fatal": closed_reason,
              "error": error, "actions": chosen, "experiments": lines("decision_experiments.jsonl"),
              "model_calls": lines("model_calls.jsonl"), "transitions": lines("transitions.jsonl"),
              "helpers": [p.name for p in (ipc.parent / "team_nemo/shared/helpers").glob("*")] if ipc else [],
              "procedures": lines("procedures.jsonl")}
    ((OUT if args.execute else RAW) / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ('model_calls', 'transitions', 'experiments')}, indent=2))

if __name__ == "__main__": main()
