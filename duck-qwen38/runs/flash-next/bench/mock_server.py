"""Minimal stand-in for the vLLM OpenAI server: /tokenize, /v1/chat/completions, /metrics. Tests only."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class State:
    def __init__(self, slots: int = 4, latency: float = 0.05, prefix: bool = False, undercount: int = 510,
                 corrupt_prefix: bool = False):
        self.lock = threading.Lock()
        self.c = {"gen": 0, "prompt": 0, "done": 0, "drafts": 0, "acc": 0, "p0": 0, "p1": 0, "p2": 0,
                  "q": 0, "h": 0, "pre": 0}
        self.running = 0
        self.sem = threading.Semaphore(slots)
        self.latency, self.prefix, self.undercount, self.corrupt_prefix = latency, prefix, undercount, corrupt_prefix
        self.ok_requests = 0
        self.requests: list[dict] = []
        self.fail_context_once = set()


def make_handler(st: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/metrics":
                return self._json(404, {})
            c = st.c
            text = "\n".join([
                "# HELP x", f'vllm:generation_tokens_total{{engine="0"}} {c["gen"]}',
                f'vllm:prompt_tokens_total{{engine="0"}} {c["prompt"]}',
                f'vllm:request_success_total{{finished_reason="stop"}} {c["done"]}',
                f'vllm:request_success_total{{finished_reason="length"}} 0.0',
                f'vllm:num_preemptions_total{{engine="0"}} {c["pre"]}',
                f'vllm:prefix_cache_queries_total{{engine="0"}} {c["q"]}',
                f'vllm:prefix_cache_hits_total{{engine="0"}} {c["h"]}',
                f'vllm:spec_decode_num_drafts_total{{engine="0"}} {c["drafts"]}',
                f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {3 * c["drafts"]}',
                f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {c["acc"]}',
                *[f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="{i}"}} {c[f"p{i}"]}'
                  for i in range(3)],
                f'vllm:spec_decode_num_accepted_tokens_per_pos_created{{engine="0",position="0"}} 1.7e9',
                f'vllm:num_requests_running{{engine="0"}} {st.running}',
                f'vllm:num_requests_waiting{{engine="0"}} 0.0',
                f'vllm:kv_cache_usage_perc{{engine="0"}} 0.5',
            ]) + "\n"
            data = text.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            tokens = int(sum(len(json.dumps(m)) for m in body.get("messages", [])) / 3.37)
            if self.path == "/tokenize":
                # the real /tokenize undercounts the chat prompt by up to ~510 tokens (speed-screen v1)
                count = max(1, tokens - st.undercount)
                out = {"count": count}
                if body.get("tools") is None:      # only the logprob check reads ids; keep the mock fast otherwise
                    out["tokens"] = [hash(json.dumps(body["messages"])) % 1000 + i for i in range(count)]
                return self._json(200, out)
            if self.path == "/v1/completions":
                ids = body["prompt"]
                with st.lock:
                    c = st.c
                    c["q"] += len(ids); c["h"] += len(ids) // 2 if st.prefix else 0
                base = {f"tok{(sum(ids[-5:]) + j) % 97}": -0.1 * (j + 1) for j in range(20)}
                if st.prefix and st.corrupt_prefix:
                    base = {k: v - (0.5 if i == 0 else 0) for i, (k, v) in enumerate(base.items())}
                return self._json(200, {"choices": [{"text": "x", "logprobs": {"top_logprobs": [base]}}]})
            if self.path != "/v1/chat/completions":
                return self._json(404, {})
            msgs = body["messages"]
            # structural contract: tool results follow a tool-calling assistant with the same ids
            for i, m in enumerate(msgs):
                if m["role"] == "tool":
                    prev = next(msgs[j] for j in range(i - 1, -1, -1) if msgs[j]["role"] != "tool")
                    if prev["role"] != "assistant" or m["tool_call_id"] not in {c["id"] for c in prev.get("tool_calls", [])}:
                        return self._json(400, {"error": f"orphan tool message at {i}"})
            if tokens + (body.get("max_tokens") or 0) > 32_768:
                return self._json(400, {"error": {"message": "This model's maximum context length is 32768 tokens. "
                    f"However, you requested {body.get('max_tokens')} output tokens and your prompt contains at least "
                    f"{tokens} input tokens, for a total of at least {tokens + (body.get('max_tokens') or 0)} tokens."}})
            with st.lock:
                st.ok_requests += 1
            st.requests.append({"tokens": tokens, "temperature": body["temperature"], "n": len(msgs),
                                "max_tokens": body.get("max_tokens"), "seed": body.get("seed")})
            with st.sem:
                with st.lock:
                    st.running += 1
                time.sleep(st.latency)
                h = hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()
                call = int(h[0], 16) < 11 and body["temperature"] > 0
                msg = {"role": "assistant", "content": "" if call else f"final {h[:8]}",
                       "reasoning_content": f"thinking {h[:16]} " * 20}
                if call:
                    msg["tool_calls"] = [{"id": f"chatcmpl-tool-{h[:12]}", "type": "function",
                                          "function": {"name": "python", "arguments": json.dumps({"code": "print(1)"})}}]
                with st.lock:
                    st.running -= 1
                    c = st.c
                    c["gen"] += 120; c["prompt"] += tokens; c["done"] += 1; c["drafts"] += 40
                    c["acc"] += 72; c["p0"] += 30; c["p1"] += 24; c["p2"] += 18
                    c["q"] += tokens; c["h"] += tokens // 2 if st.prefix else 0
            return self._json(200, {"choices": [{"message": msg, "finish_reason": "tool_calls" if call else "stop"}],
                                    "usage": {"prompt_tokens": tokens, "completion_tokens": 120}})
    return H


def serve(port: int, **kw) -> tuple[ThreadingHTTPServer, State]:
    st = State(**kw)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(st))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, st
