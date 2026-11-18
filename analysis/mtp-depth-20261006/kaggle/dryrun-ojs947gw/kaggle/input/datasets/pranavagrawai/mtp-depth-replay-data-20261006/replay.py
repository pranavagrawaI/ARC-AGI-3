"""Closed-loop replay driver for one server launch of the MTP-depth sweep (CARD.md).

One thread per game replays the control run's requests from the minute-40 checkpoint:
send the logged prefix (payload.build), wait for the response, sleep alpha * gap, send
the next. A FIFO gate with S slots mirrors dfranzen's _PriorityGate: a game holds its slot
across gaps and hands it over at a trimmed request. A poller samples /metrics every second.
Everything goes to one JSONL event stream; summarize() reduces it for the decision rule.

Stdlib only, so the same file runs locally against mock_server.py and inside Kaggle.
"""
import http.client
import json
import re
import threading
import time
import urllib.parse
from pathlib import Path

import payload
import sim

METRIC = re.compile(r"^(sglang:[a-z_]+)(\{[^}]*\})? ([0-9.eE+-]+|NaN)$")
WANTED = {"sglang:num_running_reqs": "running", "sglang:num_queue_reqs": "queue",
          "sglang:generation_tokens_total": "gen_tokens", "sglang:prompt_tokens_total": "prompt_tokens",
          "sglang:cached_tokens_total": "cached_tokens", "sglang:spec_accept_length": "accept_len",
          "sglang:cache_hit_rate": "cache_hit_rate"}


class Events:
    def __init__(self, path):
        self.path = Path(path)
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        self.f = self.path.open("a", buffering=1)
        self.closed = False

    def emit(self, kind, **kw):
        """Events after close (responses still in flight at stop) are dropped on purpose:
        the decision uses only the block and the sample taken right after it."""
        rec = {"t": round(time.monotonic() - self.t0, 3), "wall": time.time(), "kind": kind, **kw}
        with self.lock:
            if not self.closed:
                self.f.write(json.dumps(rec) + "\n")
        return rec

    def close(self):
        with self.lock:
            self.closed = True
            self.f.close()


class Gate:
    """FIFO slot gate; holders and queue start in production's order (sim.initial_admission)."""

    def __init__(self, slots, holders, queue):
        self.cond = threading.Condition()
        self.slots = slots
        self.admitted = set(holders)
        self.queue = list(queue)
        self.free = slots - len(self.admitted)
        with self.cond:
            self._pump()

    def _pump(self):
        while self.free > 0 and self.queue:
            self.admitted.add(self.queue.pop(0))
            self.free -= 1
        self.cond.notify_all()

    def wait(self, game, stop):
        with self.cond:
            while game not in self.admitted and not stop.is_set():
                self.cond.wait(0.5)
            return game in self.admitted

    def handover(self, game):
        with self.cond:
            self.admitted.discard(game)
            self.free += 1
            self.queue.append(game)
            self._pump()

    def release(self, game):
        with self.cond:
            if game in self.admitted:
                self.admitted.discard(game)
                self.free += 1
            elif game in self.queue:
                self.queue.remove(game)
            self._pump()

    def holding(self):
        with self.cond:
            return len(self.admitted)


class Bodies:
    """Lazy access to logged request bodies by (game, pair index) via byte offsets."""

    def __init__(self, logs_dir, games):
        self.files, self.offsets = {}, {}
        for g in games:
            path = next(Path(logs_dir).glob(f"{g}*_p0_requests.jsonl"))
            offs, pos = [], 0
            with path.open("rb") as f:
                for line in f:
                    if line.startswith(b'{') and b'"event": "request"' in line:
                        offs.append(pos)
                    pos += len(line)
            self.files[g], self.offsets[g] = path, offs

    def get(self, game, index):
        with self.files[game].open("rb") as f:
            f.seek(self.offsets[game][index])
            rec = json.loads(f.readline())
        if rec.get("event") != "request":
            raise ValueError(f"{game}[{index}] is not a request record")
        return rec


class Replay:
    def __init__(self, *, workload, logs_dir, alpha, base_url, model, events_path,
                 slots=sim.SLOTS, request_timeout=1800.0, time_scale=1.0, header=None):
        wl = json.loads(Path(workload).read_text())
        self.games = sim.build(wl)
        self.alpha, self.slots, self.model = alpha, slots, model
        self.time_scale = time_scale  # mock rehearsal only: gaps are divided by this
        u = urllib.parse.urlparse(base_url)
        self.host, self.port = u.hostname, u.port or 80
        self.request_timeout = request_timeout
        self.bodies = Bodies(logs_dir, self.games)
        for g in self.games:  # pair i must be request line i; extra requests may only trail
            n_pairs = len(wl["games"][g]["requests"])
            if not n_pairs <= len(self.bodies.offsets[g]) <= n_pairs + 1:
                raise ValueError(f"{g}: {len(self.bodies.offsets[g])} request lines vs {n_pairs} pairs")
        self.ev = Events(events_path)
        self.ev.emit("header", **(header or {}))  # first line: what executed (commit, launch)
        holders, queue = sim.initial_admission(self.games, slots)
        self.initial_holders = holders
        self.gate = Gate(slots, holders, queue)
        self.stop = threading.Event()
        self.completed = {g: 0 for g in self.games}
        self.lock = threading.Lock()
        self.ev.emit("replay_init", games=sorted(self.games), holders=holders, queue=queue,
                     alpha=alpha, slots=slots, time_scale=time_scale,
                     requests_available={g: len(v["reqs"]) for g, v in self.games.items()})

    # ---- HTTP ----
    def _post(self, body):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.request_timeout)
        try:
            data = json.dumps(body).encode()
            conn.request("POST", "/v1/chat/completions", body=data,
                         headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            return r.status, r.read()
        finally:
            conn.close()

    def _get(self, path, timeout=5.0):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            conn.request("GET", path)
            r = conn.getresponse()
            return r.status, r.read()
        finally:
            conn.close()

    # ---- workers ----
    def _sleep(self, seconds):
        self.stop.wait(seconds / self.time_scale)

    def _game(self, g):
        reqs = self.games[g]["reqs"]
        try:
            for j, p in enumerate(reqs):
                if self.stop.is_set():
                    break
                if j > 0 and not p["extends_previous"]:
                    self.gate.handover(g)
                    self.ev.emit("handover", game=g, idx=j)
                if not self.gate.wait(g, self.stop):
                    break
                body = payload.build(self.bodies.get(g, p["line_index"]), self.model,
                                     p["completion_tokens"])
                t_send = self.ev.emit("send", game=g, idx=j, line_index=p["line_index"],
                                      recorded_prompt=p["prompt_tokens"],
                                      recorded_completion=p["completion_tokens"])["t"]
                try:
                    status, raw = self._post(body)
                    resp = json.loads(raw) if status == 200 else {"error": raw[:500].decode(errors="replace")}
                except Exception as e:  # server killed at stop, or a transport fault
                    status, resp = -1, {"error": repr(e)[:500]}
                usage = resp.get("usage") or {}
                choice = (resp.get("choices") or [{}])[0]
                rec = self.ev.emit(
                    "response", game=g, idx=j, status=status, t_send=t_send,
                    prompt_tokens=usage.get("prompt_tokens"),
                    cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    recorded_completion=p["completion_tokens"],
                    finish_reason=choice.get("finish_reason"),
                    error=resp.get("error") if status != 200 else None)
                if status == 200:
                    with self.lock:
                        self.completed[g] += 1
                elif self.stop.is_set():
                    break
                self._sleep(self.alpha * p["gap"])
            else:
                self.ev.emit("game_exhausted", game=g)
        finally:
            self.gate.release(g)

    def _sample(self):
        vals = {}
        try:
            status, raw = self._get("/metrics")
            for line in raw.decode(errors="replace").splitlines():
                m = METRIC.match(line)
                if m and m.group(1) in WANTED:
                    key = WANTED[m.group(1)]
                    vals[key] = vals.get(key, 0.0) + float(m.group(3))
        except Exception as e:
            vals["error"] = repr(e)[:200]
        self.ev.emit("metrics", holding=self.gate.holding(), **vals)

    def _poll(self):
        while not self.stop.is_set():
            self._sample()
            self.stop.wait(max(0.05, 1.0 / self.time_scale))

    # ---- phases ----
    def run(self, *, measure_s=240.0, warmup_max_s=600.0, settle_s=30.0, profile=None):
        """Warm-up (state criterion) -> optional profile window -> measured block -> stop.

        profile: optional (start_fn, stop_fn, seconds); its window is excluded from U and is
        followed by a settle period before measurement."""
        threads = [threading.Thread(target=self._poll, daemon=True)]
        threads += [threading.Thread(target=self._game, args=(g,), daemon=True) for g in self.games]
        self.ev.emit("phase", name="start")
        for th in threads:
            th.start()
        t0 = time.monotonic()
        warm_at = None
        while time.monotonic() - t0 < warmup_max_s / self.time_scale:
            with self.lock:
                ready = all(self.completed[g] >= 1 for g in self.initial_holders)
            if ready and warm_at is None:
                warm_at = time.monotonic()
                self.ev.emit("phase", name="holders_served")
            if warm_at is not None and time.monotonic() - warm_at >= settle_s / self.time_scale:
                break
            time.sleep(0.2)
        self.ev.emit("phase", name="warmup_end", criterion_met=warm_at is not None)
        if profile is not None:
            start_fn, stop_fn, seconds = profile
            self.ev.emit("phase", name="profile_start")
            ok = start_fn()
            time.sleep(seconds / self.time_scale)
            ok = stop_fn() and ok
            self.ev.emit("phase", name="profile_end", ok=bool(ok))
            time.sleep(settle_s / self.time_scale)
        self.ev.emit("phase", name="measure_start", seconds=measure_s)
        time.sleep(measure_s / self.time_scale)
        self.ev.emit("phase", name="measure_end")
        self._sample()  # guarantees a counter reading after the block for interpolation
        self.stop.set()
        self.ev.emit("phase", name="stopped")
        self.ev.close()


def _phase(events, name):
    return next((e["t"] for e in events if e["kind"] == "phase" and e["name"] == name), None)


def summarize(events_path, subwindows=4):
    """Per-launch numbers for the decision rule (CARD.md). U = generated-token counter rate."""
    events = [json.loads(line) for line in Path(events_path).open()]
    m0, m1 = _phase(events, "measure_start"), _phase(events, "measure_end")
    if m0 is None or m1 is None:
        return {"valid": False, "reason": "no measured block"}
    mets = [e for e in events if e["kind"] == "metrics" and "gen_tokens" in e]

    def counter_at(t):
        """gen_tokens counter linearly interpolated at time t."""
        before = [e for e in mets if e["t"] <= t]
        after = [e for e in mets if e["t"] >= t]
        if not before or not after:
            return None
        x, y = before[-1], after[0]
        if y["t"] == x["t"]:
            return x["gen_tokens"]
        return x["gen_tokens"] + (y["gen_tokens"] - x["gen_tokens"]) * (t - x["t"]) / (y["t"] - x["t"])

    def counter_rate(a, b):
        ca, cb = counter_at(a), counter_at(b)
        return None if ca is None or cb is None else (cb - ca) / (b - a)

    block = [e for e in mets if m0 <= e["t"] <= m1]
    width = (m1 - m0) / subwindows
    subs = [counter_rate(m0 + i * width, m0 + (i + 1) * width) for i in range(subwindows)]
    subs_ok = [s for s in subs if s is not None]
    mean = sum(subs_ok) / len(subs_ok) if subs_ok else None
    cv = (sum((s - mean) ** 2 for s in subs_ok) / len(subs_ok)) ** 0.5 / mean if mean else None
    resp = [e for e in events if e["kind"] == "response" and e["status"] == 200]
    in_block = [e for e in resp if m0 <= e["t"] <= m1]
    early = [e for e in resp if e.get("finish_reason") == "stop"
             and (e.get("completion_tokens") or 0) < e["recorded_completion"]]
    prompt = sum(e.get("prompt_tokens") or 0 for e in in_block)
    cached = sum(e.get("cached_tokens") or 0 for e in in_block)
    running = [e["running"] for e in block if "running" in e]
    return {
        "valid": True,
        "U": counter_rate(m0, m1),
        "U_subwindows": subs,
        "U_cv": cv,
        "mean_R": sum(running) / len(running) if running else None,
        "p_R_ge_9": sum(r >= 9 for r in running) / len(running) if running else None,
        "accept_len": (sum(e["accept_len"] for e in block if "accept_len" in e) /
                       max(1, sum("accept_len" in e for e in block))),
        "cached_fraction_block": cached / prompt if prompt else None,
        "responses_block": len(in_block),
        "responses_total": len(resp),
        "early_stop_rate": len(early) / len(resp) if resp else None,
        "errors": sum(1 for e in events if e["kind"] == "response" and e["status"] != 200),
        "warmup_criterion_met": next((e.get("criterion_met") for e in events
                                      if e["kind"] == "phase" and e["name"] == "warmup_end"), None),
        "measure_s": m1 - m0,
    }
