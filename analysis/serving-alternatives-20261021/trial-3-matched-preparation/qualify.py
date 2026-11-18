"""Local paired-screen reducer. Native traces are inputs, never inferred from HTTP.

This is an analysis contract, not a launcher or a qualified engine adapter.
Times use the same host CLOCK_MONOTONIC seconds as the fixed measurement window.
"""
import collections
import math
import statistics


CONTRACT = "native-first-committed-target-output-to-finish-v1"


def finite_number(value):
    return type(value) in (float, int) and math.isfinite(value)


def occupancy(events, start, end):
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("invalid window")
    if not events or events[0]["event"] != "trace_start" or events[-1]["event"] != "trace_stop":
        raise ValueError("missing trace boundaries")
    if events[0]["t"] > start or events[-1]["t"] < end:
        raise ValueError("trace does not cover window")
    active, seen, hist = set(), set(), collections.defaultdict(float)
    previous = events[0]["t"]
    pid = events[0]["pid"]
    boot = events[0].get("boot_id")
    if type(pid) is not int or pid <= 0 or not isinstance(boot, str) or not boot:
        raise ValueError("missing native process/boot identity")
    for seq, row in enumerate(events):
        now = row["t"]
        if (type(row["seq"]) is not int or row["seq"] != seq or row["pid"] != pid
                or row.get("boot_id") != boot or not finite_number(now) or now < previous):
            raise ValueError("sequence, process, or clock discontinuity")
        if now - previous > 2.0:
            raise ValueError("native heartbeat gap exceeds two seconds")
        width = max(0., min(now, end) - max(previous, start))
        hist[len(active)] += width
        event = row["event"]
        if event in {"trace_start", "trace_stop"} and seq not in (0, len(events)-1):
            raise ValueError("interior native trace boundary")
        if event == "decode_start":
            rid = row["request_id"]
            if not isinstance(rid, str) or not rid or rid in seen:
                raise ValueError("duplicate native first-token event")
            active.add(rid); seen.add(rid)
            if len(active) > 10:
                raise ValueError("native residence exceeds S10")
        elif event == "decode_end":
            rid = row["request_id"]
            if rid not in active:
                raise ValueError("native finish without first-token event")
            active.remove(rid)
        elif event not in {"trace_start", "trace_stop", "heartbeat"}:
            raise ValueError("unknown native event")
        previous = now
    elapsed = end - start
    if not math.isclose(sum(hist.values()), elapsed, abs_tol=1e-6):
        raise ValueError("incomplete native occupancy integral")
    integral = sum(n * dt for n, dt in hist.items())
    return {"mean": integral / elapsed, "p9": sum(dt for n, dt in hist.items() if n >= 9) / elapsed,
            "stream_seconds": integral, "seconds_by_count": dict(hist)}


def band(value):
    return (isinstance(value, dict) and finite_number(value.get("mean")) and finite_number(value.get("p9"))
            and 8.57 <= value["mean"] <= 9.37 and .62 <= value["p9"] <= .82)


def summarize(arm):
    issues = []
    if not isinstance(arm, dict):
        return {"valid": False, "issues": ["arm must be an object"]}
    window = arm.get("window")
    if (not isinstance(window, (list, tuple)) or len(window) != 2
            or not all(finite_number(x) for x in window) or window[1] <= window[0]):
        return {"valid": False, "issues": ["missing or malformed measurement window"]}
    for key in ("responses", "attempts", "native_events"):
        rows = arm.get(key)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            return {"valid": False, "issues": ["missing or malformed " + key]}
    required = ("native_adapter_qualified", "same_host_monotonic_verified", "full_image_fidelity",
                "all_corpus_preprocess_fits", "raw_responses_complete", "identity_verified", "warmup_complete")
    issues.extend(key for key in required if arm.get(key) is not True)
    if arm.get("contract") != CONTRACT:
        issues.append("native contract")
    if any(type(arm.get(key)) is not int or arm[key] != 0 for key in
           ("preemptions", "retractions", "restarts", "trace_drops", "exhausted_games", "request_errors")):
        issues.append("preemption/retraction/restart/drop/exhaustion/request-error counter")
    start, end = arm["window"]
    if not math.isclose(end - start, 900., abs_tol=1e-6):
        issues.append("fixed 900s measurement")
    try:
        occ = occupancy(arm["native_events"], start, end)
    except (ValueError, KeyError, TypeError) as exc:
        return {"valid": False, "issues": issues + [str(exc)]}
    if not band(occ):
        issues.append("native occupancy band")
    attempts, responses, native_finished, attempt_native_ids = {}, set(), {}, set()
    for row in arm["native_events"]:
        if row["event"] == "decode_end":
            native_finished[row["request_id"]] = row["t"]
    for row in arm.get("attempts", []):
        rid = row.get("request_id")
        native_id = row.get("native_request_id")
        if (not isinstance(rid, str) or not rid or rid in attempts or not finite_number(row.get("sent_t"))
                or not isinstance(native_id, str) or not native_id or native_id in attempt_native_ids):
            issues.append("invalid or duplicate request attempt")
            continue
        attempts[rid] = row
        attempt_native_ids.add(native_id)
    native_starts = {row["request_id"]: row["t"] for row in arm["native_events"] if row["event"] == "decode_start"}
    native_ids = set(native_starts)
    if native_ids - attempt_native_ids:
        issues.append("native stream without matching HTTP attempt")
    for row in attempts.values():
        native_id = row["native_request_id"]
        if native_id in native_starts and row["sent_t"] > native_starts[native_id]:
            issues.append("native first output precedes matching HTTP send")
    measured, rates, native_response_ids = [], [0., 0., 0.], set()
    for row in arm["responses"]:
        rid = row.get("request_id")
        native_id = row.get("native_request_id")
        done = row.get("done_t")
        if (not isinstance(rid, str) or not rid or rid in responses or rid not in attempts
                or not isinstance(native_id, str) or native_id in native_response_ids
                or not finite_number(done)):
            issues.append("invalid, duplicate or unmatched response identity/timestamp")
            continue
        responses.add(rid);native_response_ids.add(native_id)
        attempt = attempts[rid]
        if (attempt.get("native_request_id") != native_id or attempt["sent_t"] > done
                or native_id not in native_finished or native_finished[native_id] > done):
            issues.append("response/native lifecycle mismatch")
            continue
        if row.get("status") != 200:
            issues.append("HTTP error")
            continue
        usage = row.get("usage")
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if type(tokens) is not int or tokens <= 0:
            issues.append("missing or invalid completion usage")
            continue
        if start <= done < end:
            measured.append(row)
            rates[min(2, int((done - start) / 300))] += tokens / 300.
    unresolved = 0
    for rid, row in attempts.items():
        if rid not in responses:
            cutoff = row.get("cutoff_t")
            if row.get("terminal") != "cutoff_pending" or not finite_number(cutoff) or cutoff < end or cutoff < row["sent_t"]:
                issues.append("attempt missing terminal response or explicit end-cutoff pending receipt")
            else:
                unresolved += 1
                native_id = row["native_request_id"]
                if native_id in native_finished and native_finished[native_id] < end:
                    issues.append("unreconciled native finish before window cutoff")
    if len(measured) < 30:
        issues.append("fewer than 30 completed responses")
    total = sum(row["usage"]["completion_tokens"] for row in measured)
    mean = statistics.mean(rates)
    cv = statistics.pstdev(rates) / mean if mean else math.inf
    if cv > .15:
        issues.append("300s block CV above 0.15")
    return {"valid": not issues, "issues": sorted(set(issues)), "occupancy": occ,
            "completed": len(measured), "completion_tokens": total, "U_resp": total / (end-start),
            "unresolved_at_cutoff": unresolved,
            "tokens_per_running_stream_second": total / occ["stream_seconds"] if occ["stream_seconds"] else None,
            "block_rates": rates, "block_cv": cv}


def compare(control, candidate, pair):
    a, b = summarize(control), summarize(candidate)
    issues = []
    if not isinstance(pair, dict):
        return {"verdict": "FEASIBILITY_ONLY", "issues": ["pair evidence must be an object"], "control": a, "candidate": b}
    if not a["valid"] or not b["valid"]:
        issues.append("invalid arm")
    if not band(pair.get("control_legacy_gauge", {"mean": -1, "p9": -1})):
        issues.append("control legacy production band")
    for key in ("same_corpus_manifest", "same_payload_policy", "same_image_bytes", "same_processor_geometry"):
        if pair.get(key) is not True:
            issues.append(key)
    for key, limit in (("prompt_bin_tvd", .10), ("image_bin_tvd", .10), ("cap_bin_tvd", .10),
                       ("early_stop_rate_difference", .05)):
        value = pair.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= limit:
            issues.append(key)
    length_ratio = pair.get("mean_completion_fraction_ratio")
    if type(length_ratio) not in (float, int) or not math.isfinite(length_ratio) or not .90 <= length_ratio <= 1.10:
        issues.append("completion length mix")
    if "occupancy" in a and "occupancy" in b:
        if abs(a["occupancy"]["mean"] - b["occupancy"]["mean"]) > .25:
            issues.append("paired mean occupancy")
        if abs(a["occupancy"]["p9"] - b["occupancy"]["p9"]) > .05:
            issues.append("paired occupancy tail")
    if issues:
        return {"verdict": "FEASIBILITY_ONLY", "issues": sorted(set(issues)), "control": a, "candidate": b}
    ratio = b["U_resp"] / a["U_resp"]
    return {"verdict": "CLOSE_TESTED_CONFIG" if ratio < 1.20 else "QUALITY_INVESTIGATION_ONLY",
            "ratio": ratio, "control": a, "candidate": b}
