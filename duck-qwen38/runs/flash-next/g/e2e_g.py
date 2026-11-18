"""End-to-end proof of arm G on KF's real code path: offline bp35 (every action animates), KF's
_HarnessGameSession + ToolAgent with the grafted cell installed, and a scripted stub model server.

  cd runs/flash-next && PYTHONPATH=kf/source-dataset/src/ARC3-Inference:kf/source-dataset/src/tufa-arc-agi-framework/src \
    ~/projects/arc-agi-3-reference/duck-critic-adapter/ARC3-Inference/.venv/bin/python -B g/e2e_g.py [--control]

--control skips the graft (KF as shipped) so the two request streams can be compared.
Checks: step-2 request carries a 'Frame sheet:' line + sheet image + a 768x768 board; the sandbox sees
last_transition.frames; history images are kept only for the newest 2 image-bearing messages.
"""
import base64, io, json, os, sys, tempfile, threading, time, types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL = "--control" in sys.argv
ENV_DIR = str(Path.home() / "projects/arc-agi-3-reference/competition-env/environment_files")
# the env KF's serving setup persisted on Kaggle (model id, yield, context, upscale 4)
os.environ.update({k: str(v) for k, v in json.loads((HERE.parent / "out-kf2/taaf_setup_env.json").read_text()).items()})

if not CONTROL:
    nb = json.loads((HERE / "g1/duck-qwen3-8-flash-next-nvfp4-mtp.ipynb").read_text())
    exec(compile("".join(nb["cells"][10]["source"]), "graft-cell", "exec"), {"__name__": "__graft__"})

import arc_agi, taaf.game_api
import inference.agent.tool_agent as ta
from inference.framework import solver as solv

REQUESTS = []
SCRIPT = [
    ("tool", "act = valid_actions[0]\nprint('ACT', act)\naction([act])"),
    ("text", "World model: probe.\nPlan: look at frames."),
    ("tool", "lt = last_transition\nprint('FRAMES', len(getattr(lt, 'frames', []) or []), getattr(lt, 'frame_count', None))\n"
             "print('HIST', [len(getattr(h, 'frames', []) or []) for h in history[-3:]])\naction([valid_actions[0]])"),
    ("text", "World model: probe 2."),
    ("tool", "print('FRAMES3', len(getattr(last_transition, 'frames', []) or []))\naction([valid_actions[0]])"),
    ("text", "World model: probe 3."),
    ("text", "World model: probe 4."),
]


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append(body)
        kind, text = SCRIPT[min(len(REQUESTS) - 1, len(SCRIPT) - 1)]
        msg = {"role": "assistant", "content": text if kind == "text" else ""}
        if kind == "tool":
            msg["tool_calls"] = [{"id": f"c{len(REQUESTS)}", "type": "function",
                                  "function": {"name": "python", "arguments": json.dumps({"code": text})}}]
        out = json.dumps({"id": "x", "object": "chat.completion", "model": body.get("model"),
                          "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if kind == "tool" else "stop"}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


def images(message):
    c = message.get("content")
    return [p for p in c if isinstance(p, dict) and p.get("type") == "image_url"] if isinstance(c, list) else []


def size(part):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))).size


def main():
    server = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV_DIR)
    game = taaf.game_api.GameAPI(env_name="bp35", arcade_spec=spec)
    game.start_game()
    agent = ta.ToolAgent(timeout=60, base_url=f"http://127.0.0.1:{server.server_port}/v1", provider="vllm", api_key="x")
    tmp = Path(tempfile.mkdtemp())
    stop = threading.Event()
    fake_solver = types.SimpleNamespace(max_runtime_s_per_game=120.0, max_actions_per_game=None, job_dir=None,
                                        soft_time_remaining_seconds=lambda: None)
    session = solv._HarnessGameSession(
        solver=fake_solver, game=game, analyzer=agent, game_index=0, pass_index=0,
        state_path=tmp / "state.json", transcript_path=tmp / "t.txt", analysis_html_relpath="a.html",
        stop_event=stop, viewer_data_path=tmp / "viewer.json")
    def _stopper():
        while len(REQUESTS) < len(SCRIPT):
            time.sleep(0.1)
        stop.set()
    threading.Thread(target=_stopper, daemon=True).start()
    session.play()

    report = {"arm": "KF control" if CONTROL else "G", "requests": len(REQUESTS), "actions": session.action_count}
    for i, req in enumerate(REQUESTS):
        users = [m for m in req["messages"] if m.get("role") == "user"]
        last = users[-1]
        text = " ".join(p.get("text", "") for p in last["content"] if isinstance(p, dict)) if isinstance(last.get("content"), list) else str(last.get("content"))
        report[f"req{i+1}"] = {
            "user_msgs": len(users),
            "images_per_user_msg": "".join(str(len(images(m))) for m in users),
            "newest_image_sizes": [size(p) for p in images(last)],
            "sheet_line": next((ln[:120] for ln in text.split("\n") if ln.startswith("Frame sheet:")), None),
        }
    tool_out = [m.get("content") for r in REQUESTS for m in r["messages"] if m.get("role") == "tool"]
    report["sandbox_frames"] = sorted({ln for c in tool_out for ln in str(c).replace("\\n", "\n").split("\n") if "FRAMES" in ln or "HIST" in ln})
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
