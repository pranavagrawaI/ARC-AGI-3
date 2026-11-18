"""Replay captured development requests, not game actions, against a local server.

Use TRACE_FULL_REQUESTS=1 when collecting logs. Run each point against a freshly
started server (with the same warmup) to avoid cross-experiment cache contamination.
This is a serving test, not a closed-loop ARC score comparison.
"""
import argparse
import glob
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from one_tool_runtime import decode_reply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True, help="Glob for development events.jsonl files")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="benchmark_result.json")
    args = parser.parse_args()
    requests = []
    for path in sorted(glob.glob(args.logs)):
        for line in open(path):
            event = json.loads(line)
            if event["kind"] == "request" and "messages" in event["body"]:
                requests.append((event["t_monotonic"], event["body"]))
    requests = [body for _, body in sorted(requests, key=lambda x: x[0])][:args.limit]
    if not requests:
        parser.error("No full requests found; collect development traces with TRACE_FULL_REQUESTS=1")
    local = threading.local()
    clients = []
    lock = threading.Lock()

    def one(item):
        index, original = item
        if not hasattr(local, "client"):
            local.client = httpx.Client(base_url=args.url, trust_env=False,
                                       transport=httpx.HTTPTransport(retries=0), timeout=1800)
            with lock:
                clients.append(local.client)
        body = dict(original)
        body["seed"] = index  # same request seed at every concurrency point
        start = time.monotonic()
        try:
            response = local.client.post("/v1/chat/completions", json=body)
            response.raise_for_status()
            value = response.json()
            valid = True
            try:
                decode_reply(value, 32)
            except Exception:
                valid = False
            return {"seconds": time.monotonic()-start, "valid_tool_reply": valid,
                    "usage": value.get("usage", {})}
        except Exception as exc:
            return {"seconds": time.monotonic()-start, "valid_tool_reply": False, "error": repr(exc)}

    start = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(one, enumerate(requests)))
    finally:
        for client in clients:
            client.close()
    elapsed = time.monotonic()-start
    output_tokens = sum(x.get("usage", {}).get("completion_tokens", 0) for x in results)
    valid = sum(x["valid_tool_reply"] for x in results)
    summary = {"concurrency": args.concurrency, "requests": len(results), "wall_seconds": elapsed,
               "valid_tool_replies": valid, "valid_tool_replies_per_minute": valid*60/elapsed,
               "output_tokens": output_tokens, "aggregate_output_tokens_per_second": output_tokens/elapsed,
               "results": results,
               "caveat": "Actual generations can vary with batching. No live games were executed."}
    with open(args.output, "w") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
