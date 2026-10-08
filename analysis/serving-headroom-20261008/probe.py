"""One-load diagnostic. Embedded verbatim in the notebook; stdlib only."""
import concurrent.futures
import hashlib
from http.client import HTTPConnection
import json
import math
import statistics
import time
from pathlib import Path

import payload
import replay
import sim

PLAN = [("N1", 1), ("N4", 4), ("N7", 7), ("N10A", 10),
        ("N16A", 16), ("N10B", 10), ("N16B", 16), ("N10C", 10),
        ("PROFILE1", 1), ("PROFILE10", 10), ("PROFILE16", 16), ("N10POST", 10)]
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


def sample():
    vals = {}
    wanted = {**replay.WANTED, "sglang:num_retracted_requests_total": "retractions",
              "sglang:token_usage": "kv_fraction", "sglang:mamba_usage": "mamba_fraction"}
    for line in http("/metrics").decode().splitlines():
        match = replay.METRIC.match(line)
        if match and match.group(1) in wanted:
            key = wanted[match.group(1)]
            vals[key] = vals.get(key, 0.) + float(match.group(3))
    return vals


def plateau(samples, n):
    """Exclude prefill, changing occupancy, the first 5s, and counter resets."""
    intervals = []
    for a, b in zip(samples, samples[1:]):
        if a["t"] < 5 or any(x.get("running") != n or x.get("queue") != 0 or x.get("pending", n) != n for x in (a, b)):
            continue
        if a.get("prompt_tokens") != b.get("prompt_tokens"):
            continue
        dt = b["t"] - a["t"]
        delta = b.get("gen_tokens", math.nan) - a.get("gen_tokens", math.nan)
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
                    for name, g in games.items())[:16]
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
    raw = http("/start_profile", req)
    ev.emit("profile_response", raw=raw.decode())


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
        before = sample()
        ev.emit("metrics_before", **before)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(generate, job, directory, ev, max_tokens) for job in jobs]
            while not all(f.done() for f in futures):
                if time.monotonic() - start > 600:
                    raise TimeoutError("batch deadline")
                vals = ev.emit("metrics", pending=sum(not f.done() for f in futures), **sample())
                samples.append(vals)
                if result["profile"] and not profile_attempted and plateau(samples, n)["valid"]:
                    profile_attempted = True
                    try:
                        profile_block(label, out, ev)
                    except Exception as error:
                        ev.emit("profile_error", error=repr(error))
                        result["profile_error"] = repr(error)
                time.sleep(1)
            responses = [f.result() for f in futures]
        result.update(status="complete", responses=responses, plateau=plateau(samples, n),
                      cache_ok=all(x["cache_fraction"] >= .9 for x in responses),
                      wall_seconds=time.monotonic() - start, profile_attempted=profile_attempted)
        after = sample()
        ev.emit("metrics_after", **after)
        result["retractions"] = after.get("retractions", 0) - before.get("retractions", 0)
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
    (out / "server-info-before.json").write_bytes(http("/server_info"))
    state = {"run_id": "serving-headroom-20261008", "blocks": []}
    try:
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
