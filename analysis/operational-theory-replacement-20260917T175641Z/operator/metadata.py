"""C7R1 station 2: fresh, unauthenticated OpenRouter public metadata. Exact model, singleton pinned endpoint,
frozen metadata_check; raw bytes + receipts preserved. Drift/unavailable/ambiguous => BLOCKED. Zero generation."""
import datetime, hashlib, json, signal, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path("/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus")
CAND = ROOT / "analysis/operational-theory-t1r7-20260917"
STAGE = Path(sys.argv[1])
sys.path.insert(0, str(CAND))
from identity import verify  # noqa: E402
from transport import MODEL, ENDPOINT, NoRedirect  # noqa: E402
from run_pair import metadata_check  # noqa: E402

pre = json.loads((STAGE / "PREFLIGHT.json").read_text())
assert pre["status"] == "GO" and verify() == pre["expected"]["manifest_sha256"]


def fetch(url, name):
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(); clock = time.monotonic()
    raw = b""; status = None; error = None

    def timeout(*_): raise TimeoutError("metadata absolute deadline")
    prior = signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        opener = urllib.request.build_opener(NoRedirect())
        try:
            response = opener.open(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=25)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            status = response.status; raw = response.read(10_000_001)
            if len(raw) > 10_000_000: raise ValueError("metadata response exceeds bound")
    except Exception as exc:
        error = type(exc).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, prior)
    (STAGE / (name + ".raw")).write_bytes(raw)
    receipt = {"url": url, "method": "GET", "started_at": started, "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "elapsed_seconds": time.monotonic() - clock, "http_status": status, "error_type": error,
               "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "authenticated": False, "credential_loaded": False,
               "retries": 0, "generation_calls": 0}
    (STAGE / (name + ".receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt, raw


receipt, raw = fetch("https://openrouter.ai/api/v1/models", "models")
result = {"task": "C7R1", "requested_exact_model": MODEL, "pinned_endpoint": ENDPOINT, "generation_calls": 0, "metadata_requests": 1,
          "models_sha256": receipt["sha256"], "resolution_policy": "exact id equality only; no fuzzy match, fallback or substitution"}
if receipt["http_status"] != 200 or receipt["error_type"]:
    result.update(status="BLOCKED", reason="fresh model metadata lookup failed", receipt=receipt)
else:
    try:
        rows = json.loads(raw)["data"]; matches = [m for m in rows if m.get("id") == MODEL]
        result["exact_match_count"] = len(matches)
        if len(matches) != 1:
            result.update(status="BLOCKED", reason="exact frozen model ID unavailable or ambiguous in fresh advertised catalog")
        else:
            result["resolved_model"] = matches[0]
            endpoint_receipt, endpoint_raw = fetch("https://openrouter.ai/api/v1/models/" + MODEL + "/endpoints", "endpoints")
            result["metadata_requests"] += 1
            try:
                if endpoint_receipt["http_status"] != 200 or endpoint_receipt["error_type"]:
                    raise ValueError("endpoint metadata fetch failed")
                data = json.loads(endpoint_raw)["data"]
                tags = [e["tag"] for e in data["endpoints"]]
                result["fresh_endpoint_tags"] = tags
                result["pinned_endpoint_count"] = tags.count(ENDPOINT)
                result["metadata_sha256"] = metadata_check(STAGE / "endpoints.raw")   # frozen gate: exact quantization/context/max completion/provider/model/prices/params/status
                fresh = [e for e in data["endpoints"] if e["tag"] == ENDPOINT][0]
                pinned = json.loads((CAND / "metadata/selected.json").read_text())["endpoint"]
                result["fresh_vs_pinned"] = {k: {"pinned": pinned[k], "fresh": fresh[k]} for k in ("provider_name", "model_id", "quantization", "context_length", "max_completion_tokens", "status")}
                result["fresh_vs_pinned"]["pricing"] = {"pinned": {k: pinned["pricing"][k] for k in ("prompt", "completion")}, "fresh": {k: fresh["pricing"][k] for k in ("prompt", "completion")}}
                result["fresh_uptime"] = {k: fresh.get(k) for k in ("uptime_last_5m", "uptime_last_30m", "uptime_last_1d")}
                result["status"] = "RESOLVED"
            except Exception as exc:
                result.update(status="BLOCKED", reason=str(exc) or type(exc).__name__)
    except Exception as exc:
        result.update(status="BLOCKED", reason="invalid model catalog: " + type(exc).__name__)
result["at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(STAGE / "MODEL-RESOLUTION.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "resolved_model"}, indent=2))
