"""Mock SGLang server for rehearsing replay.py / sweep.py without a GPU.

python3 -B mock_server.py PORT [--speed 20] [--early-stop 0.1] [--error-rate 0.01] [--fail-start]

POST /v1/chat/completions decodes max_tokens under processor sharing with the production
per-stream rate table (sim.GEN_TOK_S), times --speed. GET /metrics serves the Prometheus
names replay.py reads. GET /health is 200 once "loaded". --fail-start exits during
"loading", to rehearse a launch failure.
"""
import argparse
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sim

ap = argparse.ArgumentParser()
ap.add_argument("port", type=int)
ap.add_argument("--speed", type=float, default=20.0)
ap.add_argument("--early-stop", type=float, default=0.1)
ap.add_argument("--error-rate", type=float, default=0.0)
ap.add_argument("--load-seconds", type=float, default=2.0)
ap.add_argument("--fail-start", action="store_true")
ap.add_argument("--accept-len", type=float, default=2.7)
args = ap.parse_args()

lock = threading.Lock()
state = dict(running=0, gen=0.0, prompt=0.0, cached=0.0, loaded=False)
rng = random.Random(0)


def rate(r):
    return sim.per_stream_rate(r) * args.speed


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200 if state["loaded"] else 503, {"ok": state["loaded"]})
        if self.path == "/metrics":
            with lock:
                s = dict(state)
            text = (f'sglang:num_running_reqs{{model_name="m"}} {s["running"]}\n'
                    f'sglang:num_queue_reqs{{model_name="m"}} 0\n'
                    f'sglang:generation_tokens_total{{model_name="m"}} {s["gen"]}\n'
                    f'sglang:prompt_tokens_total{{model_name="m"}} {s["prompt"]}\n'
                    f'sglang:cached_tokens_total{{model_name="m"}} {s["cached"]}\n'
                    f'sglang:spec_accept_length{{model_name="m"}} {args.accept_len}\n')
            return self._send(200, text.encode(), "text/plain")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        for key in ("model", "messages", "max_tokens", "temperature", "top_p", "top_k",
                    "chat_template_kwargs"):
            if key not in body:
                return self._send(400, {"error": f"missing {key}"})
        if any("_arc3_control" in m for m in body["messages"]):
            return self._send(400, {"error": "control key leaked to the wire"})
        if rng.random() < args.error_rate:
            return self._send(500, {"error": "injected"})
        target = body["max_tokens"]
        if rng.random() < args.early_stop:
            target = max(1, int(target * rng.uniform(0.2, 0.9)))
        prompt = sum(len(json.dumps(m)) for m in body["messages"]) // 4
        with lock:
            state["running"] += 1
            state["prompt"] += prompt
            state["cached"] += int(prompt * 0.93)
        done = 0.0
        try:
            while done < target:
                time.sleep(0.02)
                with lock:
                    step = rate(state["running"]) * 0.02
                    step = min(step, target - done)
                    state["gen"] += step
                done += step
        finally:
            with lock:
                state["running"] -= 1
        finish = "length" if target == body["max_tokens"] else "stop"
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": ""},
                                      "finish_reason": finish}],
                         "usage": {"prompt_tokens": prompt, "completion_tokens": int(target),
                                   "prompt_tokens_details": {"cached_tokens": int(prompt * 0.93)}}})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(args.load_seconds)
    if args.fail_start:
        print("mock: simulated load failure", flush=True)
        sys.exit(3)
    state["loaded"] = True
    print(f"mock: ready on {args.port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
