#!/usr/bin/env python3
"""Offline-qualified endpoint screen. Run with python -B; see FINDINGS.md."""
from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import re
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ORACLE = HERE.parent / "oracle-hint-20261010"
SOURCE = HERE.parent / "m2-winners-20261001/audit/dfranzen-repo"
NOTEBOOK = HERE.parent / "m2-winners-20261001/runs/control/dfranzen-m2-control.ipynb"
DEFAULT_ENV = Path.home() / "projects/arc-agi-3-reference/competition-env/environment_files"
PYTHON = REPO / "solvers/tycho_nooa/.venv/bin/python"
TERMINAL = {"solved", "token_budget", "action_budget", "wall_budget", "gave_up", "game_over"}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load(path, default=None):
    if not path.exists():
        return default
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def compressed_receipt(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as zipped:
            zipped.write(json.dumps(value).encode())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def context_hash(payload, cfg, root):
    """Compare visible context on replay; normalize only documented clock fields."""
    p = wire_payload(payload, cfg, 1)
    visible = {k: p.get(k) for k in ("messages", "tools", "tool_choice")}
    text = json.dumps(visible, sort_keys=True)
    for field in ("run_elapsed_seconds", "time_remaining_seconds"):
        text = re.sub(r"('" + field + r"':\s*)(?:None|[0-9.eE+-]+)", r"\1<CLOCK>", text)
    # Root is fixed on resume; this replacement also allows test artifacts to
    # be inspected without making machine-specific paths part of the hash.
    return sha(text.replace(str(root), "<START_DIR>").encode())


def append(path, value):
    with path.open("a") as f:
        f.write(json.dumps(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def inside(path):
    path = path.resolve()
    if not path.is_relative_to(HERE):
        raise ValueError(f"output must be inside {HERE}")
    return path


@contextlib.contextmanager
def locked(path):
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def starts():
    rows = load(ORACLE / "starts.json")
    keys = [f"{r['game']}-L{r['level']}" for r in rows]
    if len(rows) != 40 or len(set(keys)) != 40:
        raise ValueError("expected exactly 40 unique starts")
    if not all(r["lands_ok"] and r["same_entry"] for r in rows):
        raise ValueError("unverified start manifest")
    return dict(zip(keys, rows))


def notebook_env():
    """Read literal setup_env without executing notebook or its shell cells."""
    nb = load(NOTEBOOK)
    cell = next("".join(c["source"]) for c in nb["cells"]
                if "USE_PRIORITY_SCHEDULING = True" in "".join(c["source"]))
    text = cell[cell.index("setup_env = {"):cell.index("if USE_PRIORITY_SCHEDULING:")]
    tree = ast.parse(text)
    values = tree.body[0].value
    def literal(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return {"SERVER_BASE_URL": "http://endpoint-screen.invalid/v1", "SERVED_MODEL_NAME": "screen"}[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            a, b = literal(node.left), literal(node.right)
            return a + b if isinstance(node.op, ast.Add) else a * b
        raise ValueError(f"unsupported notebook literal: {ast.dump(node)}")
    return {literal(k): str(literal(v)) for k, v in zip(values.keys, values.values)}


def child_env(root, cfg):
    env = os.environ.copy()
    # Ambient experiment flags must not silently retune the incumbent.
    for key in list(env):
        if key.startswith(("ARC3_", "LOCAL_ANALYZER_", "MULTIMODAL_", "OPENROUTER_")):
            env.pop(key)
    env.update(notebook_env())
    env.update({"ONLY_RESET_LEVELS": "true", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join([str(SOURCE / "tufa-arc-agi-framework/src"),
                                              str(SOURCE / "ARC3-Inference"), str(HERE), str(ORACLE)]),
                "LOCAL_ANALYZER_MODEL_ID": cfg["model"], "INFERENCE_ANALYZER_MODEL": cfg["model"],
                "LOCAL_ANALYZER_API_KEY": "local-transport", "OPENAI_API_KEY": "local-transport",
                "ARC3_MAX_ACTIVE_STREAMS": "0", "ARC3_HTTP_RETRIES": "0",
                "ARC3_HTTP_RETRY_INITIAL_SECONDS": "0", "ARC3_WARMUP_ACTION_GAMES": "0",
                "TAAF_MINIMAL_DIAGNOSTICS": "1", "TAAF_RUN_AS_SUBMISSION": "0", "MPLBACKEND": "Agg",
                "LOCAL_ANALYZER_CONTEXT_WINDOW": str(cfg["context_window"]),
                "LOCAL_ANALYZER_MAX_OUTPUT": str(cfg["max_output_tokens"]),
                "MPLCONFIGDIR": str(root / "cache/matplotlib"), "XDG_CACHE_HOME": str(root / "cache"),
                "TMPDIR": str(root / "tmp"), "RECORDINGS_DIR": str(root / "recordings"),
                "ORACLE_EVENTS": str(root / "oracle-events.jsonl")})
    for key in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "RECORDINGS_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    # A caller may use OPENAI_API_KEY or LOCAL_ANALYZER_API_KEY as the real
    # endpoint variable; do not destroy it while wiring the dummy transport.
    selected_key = os.environ.get(cfg["api_key_env"])
    if selected_key is not None:
        env[cfg["api_key_env"]] = selected_key
    return env


class Halt(BaseException):
    """Bypass the incumbent's Exception retry/rollback paths; preserve receipts."""
    def __init__(self, status, detail=""):
        self.status, self.detail = status, detail
        super().__init__(detail)


def retry_seconds(headers, attempt, now):
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, parsedate_to_datetime(value).timestamp() - now)
            except (ValueError, TypeError):
                pass
    return min(60.0, 2.0 ** attempt)


class Quota:
    """Persist every attempted request (including 429) before sending it."""
    def __init__(self, path, rpm, daily, clock=time.time, sleep=time.sleep):
        self.path, self.rpm, self.daily = path, rpm, daily
        self.clock, self.sleep = clock, sleep

    def reserve(self):
        now = self.clock()
        state = load(self.path, {"sent": []})
        sent = state["sent"]
        day = dt.datetime.fromtimestamp(now, dt.timezone.utc).date()
        used = sum(dt.datetime.fromtimestamp(t, dt.timezone.utc).date() == day for t in sent)
        if self.daily and used >= self.daily:
            raise Halt("daily_cap", "local UTC-day request allowance exhausted")
        recent = [t for t in sent if now - t < 60]
        if self.rpm and len(recent) >= self.rpm:
            self.sleep(max(0.0, 60 - (now - recent[-self.rpm])) + 0.01)
            return self.reserve()
        sent.append(now)
        atomic(self.path, {"sent": [t for t in sent if now - t < 172800]})


def wire_payload(payload, cfg, remaining):
    p = json.loads(json.dumps(payload))
    p["model"] = cfg["model"]
    p["stream"] = False
    limit = min(int(p.get("max_tokens") or cfg["max_output_tokens"]), remaining)
    if cfg["wire_style"] == "standard":
        for key in ("top_k", "chat_template_kwargs", "priority", "stream_options"):
            p.pop(key, None)
        for msg in p["messages"]:
            for key in ("reasoning_content", "reasoning", "reasoning_details"):
                msg.pop(key, None)
    if cfg["modality"] == "text":
        for msg in p["messages"]:
            if isinstance(msg.get("content"), list):
                msg["content"] = "\n".join(part.get("text", "") for part in msg["content"] if part.get("type") == "text")
    p.update(cfg["extra_body"])
    # Budget and identity override extras; no streaming, routing plugins, or model fallback.
    p["model"], p["stream"] = cfg["model"], False
    p.pop("max_tokens", None)
    p.pop("max_completion_tokens", None)
    p[cfg["max_tokens_field"]] = limit
    return p


def identity(reply, headers, configured_provider):
    metadata = {k: reply[k] for k in ("provider", "model", "system_fingerprint", "quantization", "quantisation", "metadata") if k in reply}
    reported_headers = {k: v for k, v in headers.items() if any(s in k.lower() for s in ("provider", "model", "quant", "request-id", "ratelimit"))}
    return {"requested_provider": configured_provider, "returned_model": reply.get("model"),
            "reported_provider": reply.get("provider"), "reported_metadata": metadata,
            "reported_headers": reported_headers,
            "quantisation": reply.get("quantization", reply.get("quantisation")),
            "quantisation_status": "reported" if ("quantization" in reply or "quantisation" in reply) else "not reported"}


def usage(reply):
    u = reply.get("usage") or {}
    out = u.get("completion_tokens")
    if isinstance(out, bool) or not isinstance(out, int) or out < 0:
        return {"known": False, "completion_tokens": None, "prompt_tokens": u.get("prompt_tokens"), "raw": u}
    return {"known": True, "completion_tokens": out, "prompt_tokens": u.get("prompt_tokens"), "raw": u}


def check_receipt(row, max_tokens_field):
    if not row["usage"]["known"]:
        raise Halt("usage_unknown", "raw response preserved without executing its tools")
    if row["usage"]["completion_tokens"] > row["request"][max_tokens_field]:
        raise Halt("budget_violation", "endpoint reported more tokens than requested; response tools not executed")


def receipt_totals(directory):
    rows = [load(p) for p in sorted(directory.glob("*.json.gz"))]
    return {"requests": len(rows), "generated_tokens": sum(r["usage"]["completion_tokens"] or 0 for r in rows),
            "prompt_tokens": sum(r["usage"].get("prompt_tokens") or 0 for r in rows),
            "usage_known": all(r["usage"]["known"] for r in rows),
            "identities": [r["identity"] for r in rows]}


def send_http(url, payload, key, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"error": raw}
        return exc.code, dict(exc.headers), body


def model_action(a):
    names = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT",
             "ACTION5": "SPACE", "ACTION7": "UNDO", "RESET": "RESET"}
    return {"action": "MOUSE", "row": a["data"]["y"], "col": a["data"]["x"]} if a["id"] == "ACTION6" else {"action": names[a["id"]]}


def dry_script(start):
    """Test fixture only: sibling actions never enter real endpoint requests."""
    if not start["solved_runs"]:
        return None
    run = {"dfz-control": "out-control", "dfz-r2": "out-control-r2", "dfz-s1": "out-s1"}[start["solved_runs"][0]]
    bm = load(HERE.parent / "m2-winners-20261001/runs" / run / "benchmark.json")
    row = next(r for r in bm["game_runs"] if r["game_id"] == start["gid"])
    parse = lambda v: v if isinstance(v, list) else ast.literal_eval(v)
    hist, apl = parse(row["history"]), parse(row["actions_per_level"])
    lo, hi = sum(apl[:start["level"]]), sum(apl[:start["level"] + 1])
    return [model_action(h["action"]) for h in hist[lo:hi]]


def mock_reply(index, script, allowance, action_index=0):
    # A stopped/blocked action does not advance the fixture. The next actor turn
    # re-grounds through the normal harness and issues the same next action.
    code = f"action([{script[action_index]!r}])" if script is not None and action_index < len(script) else "print('offline mock: no solving fixture')"
    msg = {"role": "assistant", "content": "", "reasoning_content": "offline fixture, no model judgment"}
    msg["tool_calls"] = [{"id": f"dry-{index}", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}]
    tokens = min(64, allowance) if script is not None else allowance
    return {"id": f"dry-{index}", "object": "chat.completion", "created": 0, "model": "offline-scripted-mock",
            "provider": "offline", "quantization": "none/mock", "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": tokens, "total_tokens": 100 + tokens}}


class Transport:
    """Replay received responses before issuing new requests; fail closed on ambiguity."""
    def __init__(self, root, cfg, snapshot, script=None, sender=send_http, clock=time.time, sleep=time.sleep):
        self.root, self.cfg, self.snapshot, self.script = root, cfg, snapshot, script
        self.sender, self.clock, self.sleep = sender, clock, sleep
        self.cursor = 0
        self.receipts = root / "receipts"
        self.receipts.mkdir(exist_ok=True)
        self.quota = Quota(root.parent / "quota.json", cfg["max_rpm"], cfg["daily_cap"], clock, sleep)
        self.started = clock()
        self.previous_elapsed = load(root / "progress.json", {}).get("elapsed_seconds", 0)

    def elapsed(self):
        return self.previous_elapsed + self.clock() - self.started

    def totals(self):
        return receipt_totals(self.receipts)

    def progress(self):
        atomic(self.root / "progress.json", {**self.totals(), "elapsed_seconds": self.elapsed()})

    def receive(self, payload):
        i = self.cursor
        receipt_path = self.receipts / f"{i:06d}.json.gz"
        pending = self.root / "pending.json"
        snap = self.snapshot()
        visible_hash = context_hash(payload, self.cfg, self.root)
        cached = load(receipt_path)
        if cached:
            if cached["snapshot"] != snap:
                raise Halt("replay_mismatch", f"state/actions differ before cached response {i}")
            if cached["context_hash"] != visible_hash:
                raise Halt("replay_mismatch", f"model-visible context differs before cached response {i}")
            check_receipt(cached, self.cfg["max_tokens_field"])
            self.cursor += 1
            return cached["response"]
        if pending.exists():
            unresolved = load(pending)
            completed = self.receipts / f"{unresolved['index']:06d}.json.gz"
            if not completed.exists():
                raise Halt("pending_unknown", "request may have executed; inspect pending.json before any retry")
            pending.unlink()
        # Check the post-response replay boundary, including tool side effects.
        checkpoint = load(self.root / "checkpoint.json")
        if checkpoint and checkpoint["next_index"] == i and checkpoint["snapshot"] != snap:
            raise Halt("replay_mismatch", "state/actions differ at resume boundary")
        if checkpoint and checkpoint["next_index"] == i and checkpoint.get("context_hash", visible_hash) != visible_hash:
            raise Halt("replay_mismatch", "model-visible context differs at resume boundary")
        totals = self.totals()
        if not totals["usage_known"]:
            raise Halt("usage_unknown", "endpoint omitted completion_tokens; efficiency and remaining budget unknown")
        remaining = self.cfg["token_budget"] - totals["generated_tokens"]
        if remaining <= 0:
            raise Halt("token_budget")
        if self.cfg["action_budget"] and snap["actions"] >= self.cfg["action_budget"]:
            raise Halt("action_budget")
        if self.cfg["wall_seconds"] and self.elapsed() >= self.cfg["wall_seconds"]:
            raise Halt("wall_budget")
        request = wire_payload(payload, self.cfg, remaining)
        atomic(self.root / "checkpoint.json", {"next_index": i, "snapshot": snap, "context_hash": visible_hash})
        try:
            reply, headers = self.new_request(i, request, snap, pending)
        finally:
            self.progress()
        row = {"index": i, "snapshot": snap, "context_hash": visible_hash, "request": request, "response": reply,
               "identity": identity(reply, headers, self.cfg["provider"]), "usage": usage(reply),
               "received_at": self.clock()}
        compressed_receipt(receipt_path, row)
        pending.unlink(missing_ok=True)
        self.cursor += 1
        self.progress()
        check_receipt(row, self.cfg["max_tokens_field"])
        return reply

    def new_request(self, i, request, snap, pending):
        if self.cfg["dry"]:
            return mock_reply(i, self.script, request[self.cfg["max_tokens_field"]], snap["actions"]), {}
        key = os.environ.get(self.cfg["api_key_env"])
        if not key:
            raise Halt("missing_key", f"{self.cfg['api_key_env']} is unset")
        for attempt in range(self.cfg["retries"] + 1):
            self.quota.reserve()
            if self.cfg["wall_seconds"] and self.elapsed() >= self.cfg["wall_seconds"]:
                raise Halt("wall_budget")
            atomic(pending, {"index": i, "snapshot": snap, "request": request, "sent_at": self.clock()})
            try:
                timeout = self.cfg["request_timeout"]
                if self.cfg["wall_seconds"]:
                    timeout = min(timeout, max(.001, self.cfg["wall_seconds"] - self.elapsed()))
                status, headers, reply = self.sender(self.cfg["base_url"].rstrip("/") + "/chat/completions", request, key, timeout)
            except Exception as exc:
                raise Halt("pending_unknown", f"transport failed ({type(exc).__name__}); request outcome unknown") from exc
            append(self.root / "http.jsonl", {"index": i, "attempt": attempt, "status": status, "headers": headers, "body": reply if status != 200 else None})
            if status == 200:
                if not reply.get("choices"):
                    atomic(self.root / "invalid-response.json", reply)
                    raise Halt("invalid_response", "no choices; pending receipt preserved")
                return reply, headers
            pending.unlink()
            if status != 429:
                raise Halt("endpoint_error", f"HTTP {status}; see http.jsonl")
            if attempt == self.cfg["retries"]:
                raise Halt("rate_limited", "429 retry allowance exhausted")
            delay = retry_seconds(headers, attempt, self.clock())
            if self.cfg["wall_seconds"] and self.elapsed() + delay >= self.cfg["wall_seconds"]:
                raise Halt("rate_limited", "Retry-After exceeds remaining wall allowance")
            append(self.root / "http.jsonl", {"index": i, "wait_seconds": delay, "kind": "backoff"})
            self.sleep(delay)
        raise AssertionError("unreachable")


def summarize(root, selected):
    rows = []
    for key in selected:
        attempt = root / key
        row = load(attempt / "result.json")
        if not row and (attempt / "progress.json").exists():
            checkpoint = load(attempt / "checkpoint.json", {}).get("snapshot", {})
            row = {"key": key, "status": "interrupted", "solved": False, "actions": checkpoint.get("actions", 0),
                   "actions_status": "checkpoint lower bound", **receipt_totals(attempt / "receipts")}
        rows.append(row)
    present = [r for r in rows if r]
    terminal = [r for r in present if r["status"] in TERMINAL]
    complete_usage = all(r["usage_known"] for r in present)
    solves = sum(r["solved"] for r in present)
    tokens = sum(r["generated_tokens"] for r in present)
    report = {"selected": len(selected), "recorded": len(present), "completed": len(terminal),
              "interrupted_or_error": len(present) - len(terminal), "unstarted": len(selected) - len(present),
              "solves": solves, "generated_tokens": tokens, "prompt_tokens": sum(r["prompt_tokens"] for r in present),
              "actions": sum(r["actions"] for r in present), "usage_known": complete_usage,
              "solves_per_100k_generated_tokens": solves * 100000 / tokens if tokens > 0 and complete_usage else None,
              "solve_rate_completed": solves / len(terminal) if terminal else None,
              "solve_rate_selected": solves / len(selected) if selected else None,
              "metric_valid": len(terminal) == len(selected) and complete_usage and tokens > 0,
              "dry": load(root / "config.json")["dry"], "per_start": present,
              "token_semantics": "provider completion_tokens, including reasoning when included by provider; prompt_tokens separate"}
    atomic(root / "summary.json", report)
    return report


def run(cfg):
    root = inside(Path(cfg.pop("output")))
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / "run.lock"):
        cfg["starts_sha256"] = sha((ORACLE / "starts.json").read_bytes())
        cfg["source_hashes"] = source_hashes(Path(cfg["env_dir"]))
        old = load(root / "config.json")
        if old and old != cfg:
            raise ValueError("configuration/source changed; use a new output directory")
        atomic(root / "config.json", cfg)
        for key in cfg["selected"]:
            attempt = root / key
            attempt.mkdir(exist_ok=True)
            prior = load(attempt / "result.json", {})
            if prior.get("status") in TERMINAL:
                continue
            env = child_env(attempt, cfg)
            cmd = [str(PYTHON), "-B", str(HERE / "worker.py"), str(root / "config.json"), key]
            with (attempt / "worker.log").open("a") as out:
                p = subprocess.run(cmd, cwd=attempt, env=env, stdout=out, stderr=subprocess.STDOUT)
            summary = summarize(root, cfg["selected"])
            print(json.dumps({"key": key, "exit": p.returncode, "solves": summary["solves"], "tokens": summary["generated_tokens"]}), flush=True)
            status = load(attempt / "result.json", {}).get("status")
            if p.returncode or status not in TERMINAL:
                break
        return summarize(root, cfg["selected"])


def source_hashes(env_dir):
    paths = [NOTEBOOK, ORACLE / "overlay.py", ORACLE / "starts.json", HERE / "screen.py", HERE / "worker.py"]
    paths += sorted((SOURCE / "ARC3-Inference/inference").rglob("*.py"))
    paths += sorted((SOURCE / "tufa-arc-agi-framework/src/taaf").rglob("*.py"))
    for s in starts().values():
        paths += sorted((env_dir / s["game"] / s["gid"].split("-", 1)[1]).glob("*"))
    return {str(p): sha(p.read_bytes()) for p in sorted(set(paths)) if p.is_file()}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(HERE / "dry-run"))
    p.add_argument("--dry", action="store_true", help="offline scripted mock through the actual harness; no external network")
    p.add_argument("--base-url", default=os.environ.get("ENDPOINT_BASE_URL", ""))
    p.add_argument("--model", default=os.environ.get("ENDPOINT_MODEL", "offline-scripted-mock"))
    p.add_argument("--provider", default="unspecified")
    p.add_argument("--api-key-env", default="ENDPOINT_API_KEY")
    p.add_argument("--zero-cost-confirmed", action="store_true", help="operator verified model/key route cannot spend money")
    p.add_argument("--token-budget", type=int, default=80000, help="generated tokens per start")
    p.add_argument("--max-output-tokens", type=int, default=12288)
    p.add_argument("--max-tokens-field", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")
    p.add_argument("--context-window", type=int, default=131072)
    p.add_argument("--action-budget", type=int, default=0, help="0 disables; counts RESET and all executed actions")
    p.add_argument("--wall-seconds", type=float, default=0, help="0 disables; accumulated active wall time, including waits")
    p.add_argument("--max-rpm", type=int, default=5)
    p.add_argument("--daily-cap", type=int, default=50, help="persistent local requests/day across all starts and retries; 0 disables")
    p.add_argument("--request-timeout", type=float, default=120)
    p.add_argument("--retries", type=int, default=3, help="429 retries only; ambiguous transport errors never auto-retry")
    p.add_argument("--wire-style", choices=["standard", "native"], default="standard")
    p.add_argument("--modality", choices=["native", "text"], default="native")
    p.add_argument("--extra-body", type=json.loads, default={})
    p.add_argument("--env-dir", default=str(DEFAULT_ENV))
    p.add_argument("--keys", nargs="+", help="default all 40; e.g. tn36-L1 lp85-L7")
    return p


def config(args):
    cfg = vars(args).copy()
    cfg["selected"] = cfg.pop("keys") or list(starts())
    if not all(k in starts() for k in cfg["selected"]) or len(set(cfg["selected"])) != len(cfg["selected"]):
        raise ValueError("keys must be unique verified starts")
    for key in ("token_budget", "max_output_tokens", "context_window", "request_timeout"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("action_budget", "wall_seconds", "max_rpm", "daily_cap", "retries"):
        if cfg[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if not isinstance(cfg["extra_body"], dict):
        raise ValueError("extra-body must be a JSON object")
    if any(k in cfg["extra_body"] for k in ("model", "messages", "tools", "stream", "plugins", "models", "route")):
        raise ValueError("extra-body cannot replace identity, harness, or add routing/plugins")
    if not cfg["dry"]:
        if not cfg["zero_cost_confirmed"] or not cfg["base_url"]:
            raise ValueError("live mode needs --base-url and --zero-cost-confirmed; do not run real screens under this brief")
        from urllib.parse import urlparse
        parsed = urlparse(cfg["base_url"])
        if parsed.scheme not in ("https", "http") or parsed.username or parsed.password or parsed.query:
            raise ValueError("base-url must be an HTTP(S) endpoint without embedded credentials/query")
        if parsed.hostname == "openrouter.ai" and not cfg["model"].endswith(":free"):
            raise ValueError("OpenRouter requires an explicit :free model, never the random router")
    return cfg


if __name__ == "__main__":
    try:
        result = run(config(parser().parse_args()))
        print(json.dumps({k: v for k, v in result.items() if k != "per_start"}, indent=2))
    except (ValueError, BlockingIOError) as exc:
        raise SystemExit(str(exc))
