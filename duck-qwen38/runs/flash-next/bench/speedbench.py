"""Serving benchmark: replay P's 25 Duck sessions against the local vLLM server at sustained demand.

Runs inside the Kaggle notebook after the server is up (stdlib + requests only). Tested locally against
bench/mock_server.py by bench/test_speedbench.py.

Workload model (mirrors Duck's tool_agent as far as serving is concerned):
  - one session per game, all 25 at once (P: 25 games, 12 server slots, ~16 waiting);
  - identical system prompt + python tool in every session (P's real prompt, parsed from its snapshots);
  - a turn starts by trimming the oldest history block until the prompt is <= TURN_BUDGET tokens
    (Duck's sliding window), then appends; within a turn requests are append-only (assistant + tool result);
  - a turn ends when the model stops calling tools or after MAX_TURN_REQUESTS requests
    (P1: 1,981 requests / ~1,057 turns = 1.9 per turn);
  - Duck's sampling: temperature 0.6, top_p 0.95, top_k 20, thinking on, tool_choice auto, no max_tokens cap
    beyond the context (P's max_tokens mean was 11.5K = context - prompt).
"""
from __future__ import annotations

import itertools
import json
import zlib
import re
import subprocess
import threading
import time
from typing import Any

import requests

BASE = "http://127.0.0.1:1234"
MODEL = "Qwen/Qwen3.8-Flash-Next-NVFP4"
CTX = 32_768
TURN_BUDGET = 20_500
MAX_TURN_REQUESTS = 2
CHARS_PER_TOKEN = 3.37          # P: 71.8K chars of rendered input <-> 21.3K prompt tokens
REQUEST_TIMEOUT = 900
LOGPROB_GAMES = 4
LOGPROB_CUTS = 4
LOGPROB_BUDGET = 12_000

COUNTERS = (
    "vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
    "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc")
OVERFLOW = re.compile(r"prompt contains at least (\d+) input tokens")
LINE = re.compile(r'^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+0-9.eEinfINFnaN]+)$')


def python_tool(description: str) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {
        "name": "python", "description": description,
        "parameters": {"type": "object", "properties": {"code": {
            "type": "string",
            "description": "Python code to run. The snippet is ephemeral and is not saved across tool calls."}},
            "required": ["code"]}}}]


def payload(messages, tools, *, temperature=0.6, max_tokens=None, seed=None) -> dict[str, Any]:
    p = {"model": MODEL, "messages": messages, "stream": False, "temperature": temperature, "top_p": 0.95,
         "tools": tools, "tool_choice": "auto", "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
    if max_tokens is not None:
        p["max_tokens"] = max_tokens
    if seed is not None:
        p["seed"] = seed
    return p


def estimate_tokens(messages) -> int:
    return int(sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) / CHARS_PER_TOKEN)


def count_tokens(messages, tools, stats: dict) -> int:
    try:
        r = requests.post(f"{BASE}/tokenize", timeout=60, json={
            "model": MODEL, "messages": messages, "tools": tools, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": True}})
        r.raise_for_status()
        stats["tokenize_ok"] = stats.get("tokenize_ok", 0) + 1
        return int(r.json()["count"])
    except Exception:
        stats["tokenize_fallback"] = stats.get("tokenize_fallback", 0) + 1
        return estimate_tokens(messages)


def drop_oldest_block(hist: list[dict]) -> bool:
    """Duck's _drop_oldest_history_block: remove from the front up to the next user message."""
    nxt = next((i for i in range(1, len(hist)) if hist[i]["role"] == "user"), None)
    if nxt is None:
        return False
    del hist[:nxt]
    return True


def assistant_message(msg: dict) -> tuple[dict, list[dict]]:
    out = {"role": "assistant", "content": msg.get("content") or ""}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        out["reasoning_content"] = reasoning
    calls = [{"id": c.get("id") or f"call-{i}", "type": "function",
              "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or "{}"}}
             for i, c in enumerate(msg.get("tool_calls") or [])]
    if calls:
        out["tool_calls"] = calls
    return out, calls


class Session(threading.Thread):
    def __init__(self, game: str, data: dict, tools, stop: threading.Event, records: list, lock: threading.Lock,
                 stats: dict, start_delay: float):
        super().__init__(daemon=True, name=f"session-{game}")
        self.game, self.tools, self.stop, self.records, self.lock, self.stats = game, tools, stop, records, lock, stats
        self.system = data["messages"][0]
        self.hist = [dict(m) for m in data["messages"][1:]]
        self.users = itertools.cycle(data["user_pool"])
        self.tool_results = itertools.cycle(data["tool_pool"])
        self.start_delay = start_delay

    def grow(self) -> None:
        """Pre-grow short snapshots to Duck's steady state by prepending copies of their own earlier turns."""
        blocks, cur = [], []
        for m in self.hist:
            if m["role"] == "user" and cur:
                blocks.append(cur)
                cur = []
            cur.append(m)
        blocks.append(cur)
        earlier = [m for b in (blocks[:-1] or blocks) for m in b]
        for _ in range(12):
            if count_tokens([self.system, *self.hist], self.tools, self.stats) >= TURN_BUDGET:
                break
            self.hist = [dict(m) for m in earlier] + self.hist

    def post(self, messages, prompt_est: int, n: int):
        """One request; an output-length overflow is retried with the server's own prompt count."""
        max_tokens = max(256, CTX - int(prompt_est * 1.03) - 1024)
        for attempt in range(3):
            r = requests.post(f"{BASE}/v1/chat/completions", timeout=REQUEST_TIMEOUT,
                              json=payload(messages, self.tools, max_tokens=max_tokens,
                                           seed=zlib.crc32(f"{self.game}:{n}".encode()) & 0x7FFFFFFF))
            m = OVERFLOW.search(r.text) if r.status_code == 400 else None
            if not m or int(m[1]) > CTX - 1024:
                return r
            with self.lock:
                self.stats["max_tokens_retries"] = self.stats.get("max_tokens_retries", 0) + 1
            max_tokens = CTX - int(m[1]) - 256
        return r

    def run(self) -> None:
        if self.stop.wait(self.start_delay):
            return
        self.grow()
        n, in_turn, turn_start, prompt_est = 0, 0, True, 0
        while not self.stop.is_set():
            if turn_start:
                while True:
                    prompt_est = count_tokens([self.system, *self.hist], self.tools, self.stats)
                    if prompt_est <= TURN_BUDGET or not drop_oldest_block(self.hist):
                        break
            messages = [self.system, *self.hist]
            rec = {"game": self.game, "n": n, "turn_start": turn_start, "prompt_est": prompt_est,
                   "t0": time.time()}
            try:
                r = self.post(messages, prompt_est, n)
                rec["t1"], rec["status"] = time.time(), r.status_code
                if r.status_code != 200:
                    rec["error"] = r.text[:300]
                    if r.status_code == 400 and OVERFLOW.search(r.text):
                        drop_oldest_block(self.hist)            # true prompt overflow: Duck's _force_reduce_messages
                        turn_start = True
                    with self.lock:
                        self.records.append(rec)
                    if self.stop.wait(1.0):
                        return
                    continue
                body = r.json()
            except Exception as exc:                            # server restarts end the config: not an error then
                rec["t1"], rec["status"], rec["error"] = time.time(), -1, f"{type(exc).__name__}: {exc}"[:300]
                with self.lock:
                    self.records.append(rec)
                if self.stop.wait(2.0):
                    return
                continue
            choice = body["choices"][0]
            usage = body.get("usage") or {}
            msg, calls = assistant_message(choice["message"])
            rec.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                       finish=choice.get("finish_reason"), tool_calls=len(calls))
            with self.lock:
                self.records.append(rec)
            self.hist.append(msg)
            for c in calls:
                self.hist.append({"role": "tool", "tool_call_id": c["id"], "content": next(self.tool_results)})
            n += 1
            in_turn += 1
            if not calls or in_turn >= MAX_TURN_REQUESTS:
                self.hist.append({"role": "user", "content": next(self.users)})
                in_turn, turn_start = 0, True
            else:
                turn_start = False
                prompt_est = (usage.get("prompt_tokens") or prompt_est) + (usage.get("completion_tokens") or 0) \
                    + estimate_tokens(self.hist[-len(calls):])


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        name, labels, value = m[1], m[2] or "", float(m[3])
        if name in COUNTERS or name in GAUGES:
            out[name] = out.get(name, 0.0) + value
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            pos = re.search(r'position="(\d+)"', labels)
            key = f"accepted_pos{pos[1] if pos else '?'}"
            out[key] = out.get(key, 0.0) + value
        elif name == "vllm:request_success_total":
            out["request_success_total"] = out.get("request_success_total", 0.0) + value
    return out


class Poller(threading.Thread):
    def __init__(self, stop: threading.Event, interval: float = 5.0):
        super().__init__(daemon=True, name="metrics-poller")
        self.stop, self.interval, self.samples = stop, interval, []

    def run(self) -> None:
        while not self.stop.is_set():
            self.samples.append(sample())
            self.stop.wait(self.interval)


def gpu_memory_mib() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return float(out.split()[0])
    except Exception:
        return None


def sample() -> dict:
    s = {"t": time.time(), "gpu_mib": gpu_memory_mib()}
    try:
        s.update(parse_metrics(requests.get(f"{BASE}/metrics", timeout=20).text))
    except Exception as exc:
        s["metrics_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return s


def window(samples: list[dict], records: list[dict], t0: float, t1: float) -> dict:
    inside = [s for s in samples if t0 <= s["t"] <= t1 and "vllm:generation_tokens_total" in s]
    if len(inside) < 2:
        return {"t0": t0, "t1": t1, "error": "too few metric samples"}
    a, b = inside[0], inside[-1]
    dt = b["t"] - a["t"]
    d = {k: b[k] - a.get(k, 0.0) for k in b if k not in {"t", "gpu_mib", "metrics_error", *GAUGES}}
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0.0)
    done = [r for r in records if r.get("status") == 200 and t0 <= r["t1"] <= t1]
    lat = sorted(r["t1"] - r["t0"] for r in done)
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {
        "t0": a["t"], "t1": b["t"], "seconds": dt,
        "gen_tok_s": d.get("vllm:generation_tokens_total", 0.0) / dt,
        "prompt_tok_s": d.get("vllm:prompt_tokens_total", 0.0) / dt,
        "requests_s": d.get("request_success_total", 0.0) / dt,
        "client_completed": len(done),
        "client_errors": sum(1 for r in records if r.get("status") != 200 and t0 <= r["t1"] <= t1),
        "latency_mean_s": mean(lat), "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "prompt_tokens_mean": mean([r["prompt_tokens"] for r in done if r.get("prompt_tokens")]),
        "completion_tokens_mean": mean([r["completion_tokens"] for r in done if r.get("completion_tokens")]),
        "preemptions": d.get("vllm:num_preemptions_total", 0.0),
        "prefix_hit_rate": (d.get("vllm:prefix_cache_hits_total", 0.0) / d["vllm:prefix_cache_queries_total"]
                            if d.get("vllm:prefix_cache_queries_total") else None),
        "accept_rate_by_pos": {k: v / drafts for k, v in sorted(d.items()) if k.startswith("accepted_pos")} if drafts else {},
        "tokens_per_draft": (1 + d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) / drafts) if drafts else None,
        "running_mean": mean([s.get("vllm:num_requests_running", 0.0) for s in inside]),
        "waiting_mean": mean([s.get("vllm:num_requests_waiting", 0.0) for s in inside]),
        "kv_usage_mean": mean([s.get("vllm:kv_cache_usage_perc", 0.0) for s in inside]),
        "gpu_mib_max": max((s["gpu_mib"] for s in inside if s.get("gpu_mib") is not None), default=None),
    }


def prompt_ids(messages, tools) -> list[int]:
    r = requests.post(f"{BASE}/tokenize", timeout=60, json={
        "model": MODEL, "messages": messages, "tools": tools, "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True}})
    r.raise_for_status()
    return r.json()["tokens"]


def logprob_check(sessions: dict) -> list[dict]:
    """Next-token top-20 logprobs for fixed token-id prompts, each sent twice (the second can hit the prefix cache).

    Sampled text is not reproducible on this stack even at T=0, so prefix-cache correctness is judged on the
    first-token distribution of identical token ids, against the server's own repeat as the noise floor.
    """
    out = []
    tools = None
    for game in sorted(sessions)[:LOGPROB_GAMES]:
        system, hist = sessions[game]["messages"][0], [dict(m) for m in sessions[game]["messages"][1:]]
        stats: dict = {}
        while count_tokens([system, *hist], tools, stats) > LOGPROB_BUDGET and drop_oldest_block(hist):
            pass
        cuts = [i for i, m in enumerate(hist) if m["role"] in ("user", "tool")][-LOGPROB_CUTS:]
        prompts = [(i, prompt_ids([system, *hist[:i + 1]], tools)) for i in cuts]
        for attempt in (0, 1):
            for i, ids in prompts:
                row = {"game": game, "cut": i, "attempt": attempt, "n_tokens": len(ids)}
                try:
                    r = requests.post(f"{BASE}/v1/completions", timeout=REQUEST_TIMEOUT, json={
                        "model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "logprobs": 20})
                    r.raise_for_status()
                    row["top"] = r.json()["choices"][0]["logprobs"]["top_logprobs"][0]
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                out.append(row)
    return out


def compare_logprobs(a: list[dict], b: list[dict], attempt_a: int, attempt_b: int) -> dict | None:
    """Top-1 disagreements and |logprob| differences between two logprob_check runs."""
    ta = {(r["game"], r["cut"]): r.get("top") for r in a if r["attempt"] == attempt_a}
    tb = {(r["game"], r["cut"]): r.get("top") for r in b if r["attempt"] == attempt_b}
    if not ta or ta.keys() != tb.keys() or any(v is None for v in (*ta.values(), *tb.values())):
        return None
    mism, diffs = 0, []
    for k in ta:
        x, y = ta[k], tb[k]
        top_x, top_y = max(x, key=x.get), max(y, key=y.get)
        mism += top_x != top_y
        floor = min(y.values()) - 1.0
        diffs.append(abs(x[top_x] - y.get(top_x, floor)))
    return {"n": len(ta), "top1_mismatch": mism, "mean_abs": sum(diffs) / len(diffs), "max_abs": max(diffs)}


def run_load(sessions: dict, tools, *, warmup_s: float, measure_s: float, windows: int,
             poll_s: float = 5.0, stagger_s: float = 20.0) -> dict:
    stop, lock, records, stats = threading.Event(), threading.Lock(), [], {}
    poller = Poller(stop, poll_s)
    poller.start()
    threads = [Session(g, sessions[g], tools, stop, records, lock, stats, i * stagger_s / len(sessions))
               for i, g in enumerate(sorted(sessions))]
    started = time.time()
    for t in threads:
        t.start()
    time.sleep(warmup_s + measure_s)
    ended = time.time()
    time.sleep(poll_s + 1)                     # one sample past the end so the last window closes on data
    stop.set()
    for t in threads:
        t.join(timeout=5)
    poller.join(timeout=poll_s + 30)
    m0 = started + warmup_s
    step = measure_s / windows
    wins = [window(poller.samples, records, m0 + i * step, m0 + (i + 1) * step) for i in range(windows)]
    total = window(poller.samples, records, m0, ended)
    with lock:
        recs = list(records)
    turn_starts = sum(1 for r in recs if r.get("turn_start") and r.get("status") == 200)
    ok = sum(1 for r in recs if r.get("status") == 200)
    return {"started": started, "measure_start": m0, "ended": ended, "windows": wins, "total": total,
            "requests_per_turn": ok / turn_starts if turn_starts else None, "stats": stats,
            "records": recs, "samples": poller.samples,
            "errors_after_warmup": sum(1 for r in recs if r.get("status") != 200 and r["t1"] >= m0 and r["t1"] <= ended)}
