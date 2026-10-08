"""One-load diagnostic. Embedded verbatim in the notebook; stdlib only."""
import concurrent.futures
import hashlib
import gzip
from http.client import HTTPConnection
import json
import math
import re
import statistics
import time
from pathlib import Path

import payload
import replay
import sim

RUN_ID = "serving-headroom-v2-20261008"
PLAN = [("PROFILE1", 1), ("PROFILE10", 10), ("N10A", 10), ("N10B", 10)]
PORT = 8001


def http(path, body=None, timeout=30):
    conn = HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    try:
        conn.request("GET" if body is None else "POST", path,
                     body=None if body is None else json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read()
        if response.status != 200:
            raise RuntimeError(f"{path}: HTTP {response.status}: {raw[:500]!r}")
        return raw
    finally:
        conn.close()


def sample(directory=None):
    raw = http("/metrics")
    vals = parse_metrics(raw)
    if directory is not None:
        path = directory / "metrics" / f"{time.monotonic_ns()}.prom"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        vals.update(raw_metrics=str(path.relative_to(directory)), raw_sha256=hashlib.sha256(raw).hexdigest())
    return vals


def parse_metrics(raw):
    vals = {}
    wanted = {**replay.WANTED, "sglang:num_retracted_requests_total": "retractions",
              "sglang:token_usage": "kv_fraction", "sglang:mamba_usage": "mamba_fraction"}
    for line in raw.decode().splitlines():
        match = replay.METRIC.match(line)
        if match and match.group(1) in wanted:
            key = wanted[match.group(1)]
            vals[key] = vals.get(key, 0.) + float(match.group(3))
        if match and match.group(1) == "sglang:realtime_tokens_total":
            labels = match.group(2) or ""
            for mode, key in (("decode", "decode_tokens"), ("prefill_compute", "prefill_compute"),
                              ("prefill_cache", "prefill_cache")):
                if re.search(r'(?:\{|,)mode="' + mode + r'"(?:,|\})', labels):
                    if key in vals:
                        raise RuntimeError("multiple scheduler series for realtime mode: " + mode)
                    vals[key] = float(match.group(3))
    if any(key in vals for key in ("decode_tokens", "prefill_compute", "prefill_cache")):
        # The pinned collector only instantiates a mode after its first nonzero increment.
        vals.setdefault("prefill_compute", 0.)
        vals.setdefault("prefill_cache", 0.)
    return vals


def steady(a, b, n):
    """Require actual occupancy and no realtime prefill, including cache hits."""
    required = ("running", "queue", "pending", "prefill_compute", "prefill_cache")
    if any(key not in x or not math.isfinite(x[key]) for x in (a, b) for key in required):
        return False
    return (a["t"] >= 5 and b["t"] > a["t"] and
            all(x["running"] == n and x["queue"] == 0 and x["pending"] == n for x in (a, b)) and
            all(a[key] == b[key] for key in ("prefill_compute", "prefill_cache")))


def profile_ready(samples, n):
    # Independent of token throughput: a broken counter must not suppress diagnosis.
    return (len(samples) >= 4 and samples[-1]["t"] - samples[-4]["t"] >= 3 and
            all(steady(a, b, n) for a, b in zip(samples[-4:], samples[-3:])))


def plateau(samples, n):
    """Exclude prefill, changing occupancy, the first 5s, and counter resets."""
    intervals = []
    counters = [x["decode_tokens"] for x in samples if "decode_tokens" in x]
    first_seen = next((i for i, x in enumerate(samples) if "decode_tokens" in x), len(samples))
    if (any("decode_tokens" not in x for x in samples[first_seen:]) or
        any(not math.isfinite(x) for x in counters) or any(b < a for a, b in zip(counters, counters[1:]))):
        return {"valid": False, "seconds": 0, "U": None, "CV": None,
                "intervals": [], "all_qualifying_intervals": [], "reason": "missing, nonfinite or reset decode counter"}
    for a, b in zip(samples, samples[1:]):
        if not steady(a, b, n):
            continue
        dt = b["t"] - a["t"]
        delta = b.get("decode_tokens", math.nan) - a.get("decode_tokens", math.nan)
        if dt > 0 and math.isfinite(delta) and delta >= 0:
            intervals.append({"start": a["t"], "end": b["t"], "dt": dt,
                              "tokens": delta, "rate": delta / dt})
    # Preserve disjoint segments and select the longest contiguous segment.
    segments = []
    for row in intervals:
        if not segments or segments[-1][-1]["end"] != row["start"]:
            segments.append([])
        segments[-1].append(row)
    segment = max(segments, key=lambda xs: sum(x["dt"] for x in xs), default=[])
    seconds = sum(x["dt"] for x in segment)
    # Counters update in bursts. Compare >=5s windows rather than scrape ticks.
    rates, pending = [], []
    for row in segment:
        pending.append(row)
        elapsed = sum(x["dt"] for x in pending)
        if elapsed >= 5:
            rates.append(sum(x["tokens"] for x in pending) / elapsed)
            pending = []
    mean = statistics.mean(rates) if rates else None
    return {"valid": len(segment) >= 5 and seconds >= 8 and mean is not None and mean > 0, "seconds": seconds,
            "U": sum(x["tokens"] for x in segment) / seconds if seconds else None,
            "CV": statistics.pstdev(rates) / mean if mean and len(rates) >= 2 else None,
            "intervals": segment, "all_qualifying_intervals": intervals}


def cohort(data):
    games = sim.build(json.loads((data / "workload.json").read_text()))
    chosen = sorted((g["reqs"][0]["prompt_tokens"], name, g["reqs"][0]["line_index"])
                    for name, g in games.items())[:10]
    bodies = replay.Bodies(data, [x[1] for x in chosen])
    result = []
    for prompt, game, index in chosen:
        body = payload.build(bodies.get(game, index), "flashnext", 4096)
        body.update(ignore_eos=True, seed=20261008)
        result.append({"game": game, "index": index, "recorded_prompt": prompt, "body": body})
    return result


def generate(job, directory, ev, max_tokens):
    game = job["game"]
    body = dict(job["body"], max_tokens=max_tokens)
    path = directory / (game + "-request.json")
    path.write_text(json.dumps(body))
    ev.emit("send", game=game, request_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    t0 = time.monotonic()
    try:
        raw = http("/v1/chat/completions", body, timeout=600)
        (directory / (game + "-response.json")).write_bytes(raw)
        value = json.loads(raw)
        usage = value["usage"]
        length = usage["completion_tokens"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        rec = {"game": game, "usage": usage, "elapsed": time.monotonic() - t0,
               "cache_fraction": cached / usage["prompt_tokens"], "length_ok": length == max_tokens}
        ev.emit("response", **rec)
        if length != max_tokens:
            raise RuntimeError(f"{game}: expected {max_tokens}, got {length}")
        return rec
    except Exception as error:
        ev.emit("request_error", game=game, error=repr(error))
        raise


def profile_block(label, out, ev):
    req = {"output_dir": str(out / "traces"), "num_steps": 64, "activities": ["CPU", "GPU"],
           "with_stack": False, "record_shapes": False, "profile_id": label,
           "profile_by_stage": False, "detailed_annotations": True}
    ev.emit("profile_request", request=req)
    raw = http("/start_profile", req, timeout=120)
    ev.emit("profile_response", raw=raw.decode())
    if raw.strip() != b"Start profiling.":
        raise RuntimeError(f"{label}: unexpected profiler response: {raw!r}")


def verify_trace(label, out):
    paths = sorted((out / "traces").glob(label + "-*.trace.json.gz"))
    if not paths:
        raise RuntimeError(f"{label}: profiler produced no trace")
    receipts = []
    for path in paths:
        with gzip.open(path, "rt") as stream:
            events = json.load(stream)["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X" and e.get("dur", 0) > 0]
        if not kernels:
            raise RuntimeError(f"{label}: trace contains no CUDA kernels")
        receipts.append({"path": str(path.relative_to(out)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "kernel_events": len(kernels)})
    return receipts


def capacity(info, jobs):
    if len(jobs) != 10:
        raise RuntimeError("expected ten fixed workload prefixes")
    states = info.get("internal_states", [])
    if len(states) != 1 or states[0].get("effective_max_running_requests_per_dp") != 10:
        raise RuntimeError("expected one scheduler with effective max-running=10")
    state = states[0]
    # Conservative sum: do not rely on unknown inter-request prefix sharing.
    required = sum(((j["recorded_prompt"] + 4096 + 4 + 63) // 64) * 64 for j in jobs)
    available = state.get("memory_usage", {}).get("token_capacity", 0)
    if available < required:
        raise RuntimeError(f"insufficient KV capacity: {available} < {required}")
    return {"effective_max_running": 10, "token_capacity": available, "required_tokens": required,
            "max_mamba_cache_size": state.get("max_mamba_cache_size")}


def accounting(before, samples, after, total, n):
    counts = [before, *samples, after]
    for key in ("decode_tokens", "gen_tokens", "retractions"):
        first = next((i for i, x in enumerate(counts) if key in x), len(counts))
        values = [x[key] for x in counts[first:] if key in x]
        if (any(key not in x for x in counts[first:]) or
            any(not math.isfinite(x) or x < 0 for x in values) or
            any(b < a for a, b in zip(values, values[1:]))):
            raise RuntimeError("invalid counter lifetime: " + key)
    completion = after.get("gen_tokens", 0) - before.get("gen_tokens", 0)
    decode = after.get("decode_tokens", 0) - before.get("decode_tokens", 0)
    retractions = after.get("retractions", 0) - before.get("retractions", 0)
    # Overlap can count delayed work on finished requests, as well as first-token
    # and final-chunk differences. This is a predeclared error tolerance, not an
    # exact physical bound or a correction fitted after the run. Warmup is diagnostic.
    tolerance = .02*total if total >= 4096*n else None
    if completion != total or (tolerance is not None and abs(decode - total) > tolerance) or retractions != 0:
        raise RuntimeError(f"wire/counter reconciliation failed: completion={completion}, decode={decode}, wire={total}, retractions={retractions}")
    return {"completion_delta": completion, "decode_delta": decode,
            "wire_tokens": total, "decode_boundary_tolerance": tolerance,
            "decode_relative_difference": decode/total-1, "retractions": retractions}


def batch(label, jobs, out, max_tokens=4096):
    directory = out / label
    directory.mkdir()
    n = len(jobs)
    ev = replay.Events(directory / "events.jsonl")
    ev.emit("header", label=label, n=n, max_tokens=max_tokens)
    samples, profile_attempted = [], False
    result = {"label": label, "n": n, "profile": label.startswith("PROFILE")}
    start = time.monotonic()
    try:
        before = sample(directory)
        ev.emit("metrics_before", **before)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(generate, job, directory, ev, max_tokens) for job in jobs]
            while not all(f.done() for f in futures):
                if time.monotonic() - start > 600:
                    raise TimeoutError("batch deadline")
                vals = ev.emit("metrics", pending=sum(not f.done() for f in futures), **sample(directory))
                samples.append(vals)
                if result["profile"] and not profile_attempted and profile_ready(samples, n):
                    profile_attempted = True
                    profile_block(label, out, ev)
                time.sleep(1)
            responses = [f.result() for f in futures]
        result.update(status="complete", responses=responses, plateau=plateau(samples, n),
                      cache_ok=all(x["cache_fraction"] >= .9 for x in responses),
                      wall_seconds=time.monotonic() - start, profile_attempted=profile_attempted)
        after = sample(directory)
        ev.emit("metrics_after", **after)
        result["accounting"] = accounting(before, samples, after, sum(x["usage"]["completion_tokens"] for x in responses), n)
        result["retractions"] = result["accounting"]["retractions"]
        if result["profile"]:
            result["traces"] = verify_trace(label, out)
        if label != "WARMUP" and (not result["plateau"]["valid"] or not result["cache_ok"] or result["retractions"] != 0):
            raise RuntimeError(f"{label}: decode/profile qualification failed")
        ev.emit("finished", **result)
        return result
    except Exception as error:
        result.update(status="failed", error=repr(error), plateau=plateau(samples, n))
        ev.emit("failed", **result)
        raise
    finally:
        (directory / "summary.json").write_text(json.dumps(result, indent=1))
        ev.close()


def run_probe(data, out, deadline):
    jobs = cohort(data)
    (out / "cohort.json").write_text(json.dumps([{k: v for k, v in j.items() if k != "body"} for j in jobs], indent=1))
    raw_info = http("/server_info")
    (out / "server-info-before.json").write_bytes(raw_info)
    state = {"run_id": RUN_ID, "blocks": []}
    try:
        state["capacity"] = capacity(json.loads(raw_info), jobs)
        # Fixed prefix warmup is excluded from throughput. No tokenization refit.
        state["warmup"] = batch("WARMUP", jobs, out, max_tokens=32)
        for label, n in PLAN:
            if deadline - time.monotonic() < 660:
                state["stop"] = "insufficient deadline for bounded next batch"
                break
            print("headroom block:", label, flush=True)
            state["blocks"].append(batch(label, jobs[:n], out))
            (out / "state.json").write_text(json.dumps(state, indent=1))
        (out / "server-info-after.json").write_bytes(http("/server_info"))
    except Exception as error:
        state["stop"] = repr(error)
        raise
    finally:
        (out / "state.json").write_text(json.dumps(state, indent=1))
