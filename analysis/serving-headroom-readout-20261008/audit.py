"""Read-only reduction of frozen probe outputs; no launch or candidate mutation."""
import argparse
import collections
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent / "serving-headroom-20261008"
sys.path.insert(0, str(PROBE))
sys.path.insert(0, str(HERE.parent / "mtp-depth-20261006"))
from probe import plateau


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def trace_summary(path):
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X" and e.get("dur", 0) > 0]
    grouped = collections.defaultdict(lambda: {"calls": 0, "microseconds": 0.})
    for event in kernels:
        row = grouped[event["name"]]
        row["calls"] += 1
        row["microseconds"] += event["dur"]
    intervals = [(e["ts"], e["ts"] + e["dur"]) for e in kernels]
    summed = sum(e["dur"] for e in kernels)
    span = max((b for a, b in intervals), default=0) - min((a for a, b in intervals), default=0)
    # Conservative known namespace from pinned marlin_moe source. Leave other GEMMs unassigned.
    experts = [e for e in kernels if "marlin_moe" in e["name"].lower() or "moe_wna16_marlin" in e["name"].lower()]
    expert_sum = sum(e["dur"] for e in experts)
    expert_intervals = [(e["ts"], e["ts"]+e["dur"]) for e in experts]
    busy = union(intervals)
    return {"kernel_events": len(kernels), "summed_kernel_us": summed, "kernel_union_us": busy,
            "kernel_span_us": span, "overlap_us": summed - busy, "no_kernel_us_inside_span": span-busy,
            "identified_expert_sum_us": expert_sum, "identified_expert_union_us": union(expert_intervals),
            "identified_expert_share_of_sum": expert_sum/summed if summed else None,
            "attribution": "Known expert namespace only. Unassigned kernels remain in denominator. No bandwidth saturation, routing locality or unprofiled speedup claim.",
            "top_kernels": sorted(({"name": name, **row} for name, row in grouped.items()), key=lambda x:x["microseconds"], reverse=True)[:40]}


def audit(raw):
    root = raw / "headroom"
    result = {"run_id": "serving-headroom-20261008", "raw": str(raw.resolve())}
    hashes = json.loads((root / "output-hashes.json").read_text()) if (root / "output-hashes.json").exists() else {}
    bad = [name for name, expected in hashes.items() if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest() != expected]
    result["output_hashes"] = {"count": len(hashes), "mismatches": bad, "manifest_present": bool(hashes)}
    result["hardware"] = json.loads((root / "hardware.json").read_text()) if (root / "hardware.json").exists() else None
    result["startup"] = json.loads((root / "startup.json").read_text()) if (root / "startup.json").exists() else None
    state = json.loads((root / "state.json").read_text()) if (root / "state.json").exists() else {}
    result["stop"] = state.get("stop")
    rows = {}
    for path in sorted(root.glob("*/summary.json")):
        row = json.loads(path.read_text())
        events = [json.loads(line) for line in (path.parent / "events.jsonl").read_text().splitlines()]
        computed = plateau([e for e in events if e["kind"] == "metrics"], row["n"])
        if row["status"] == "complete":
            assert computed == row["plateau"], "raw plateau summary mismatch: " + row["label"]
        rows[row["label"]] = {k:v for k,v in row.items() if k != "responses"}
        rows[row["label"]]["response_count"] = len(row.get("responses", []))
        wire = []
        for reqpath in sorted(path.parent.glob("*-request.json")):
            request = json.loads(reqpath.read_text())
            responsepath = reqpath.with_name(reqpath.name.replace("-request", "-response"))
            response = json.loads(responsepath.read_text()) if responsepath.exists() else {}
            usage = response.get("usage") or {}
            prompt = usage.get("prompt_tokens", 0)
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            wire.append({"game":reqpath.stem.removesuffix('-request'),
                         "fixed_length_ok":usage.get("completion_tokens") == request.get("max_tokens"),
                         "params_ok":request.get("ignore_eos") is True and request.get("seed") == 20261008
                         and request.get("temperature") == .7 and request.get("top_p") == .95 and request.get("top_k") == 20,
                         "cache_fraction":cached/prompt if prompt else 0,
                         "choice_sha256":hashlib.sha256(json.dumps(response.get("choices"),sort_keys=True).encode()).hexdigest()})
        rows[row["label"]]["wire"] = wire
        rows[row["label"]]["wire_ok"] = len(wire) == row["n"] and all(x["fixed_length_ok"] and x["params_ok"] and x["cache_fraction"] >= .9 for x in wire)
    result["blocks"] = rows
    labels10, labels16 = ["N10A", "N10B", "N10C"], ["N16A", "N16B"]
    required = labels10 + labels16
    qualified = bool(hashes) and not bad and bool(result["hardware"] and result["hardware"].get("valid"))
    qualified &= all(label in rows and rows[label].get("status") == "complete" and rows[label].get("cache_ok")
                     and rows[label].get("wire_ok")
                     and rows[label].get("retractions") == 0 and rows[label]["plateau"]["valid"]
                     and rows[label]["plateau"].get("CV") is not None for label in required)
    if qualified:
        ten = [rows[x]["plateau"]["U"] for x in labels10]
        sixteen = [rows[x]["plateau"]["U"] for x in labels16]
        drift = max(ten)/min(ten)-1
        variation = max(rows[x]["plateau"]["CV"] for x in required)
        advantage = min(sixteen)/max(ten)-1
        result["comparison"] = {"qualified": True, "minimum_relative_gain": advantage,
                                "baseline_drift": drift, "maximum_window_CV": variation,
                                "strong_headroom": advantage > max(.15, drift+variation),
                                "limit": "Favorable fixed-prefix forced-decode cohort, common max16 graph envelope; no production scheduling or ARC efficacy claim."}
    else:
        result["comparison"] = {"qualified": False, "reason": "Missing/invalid hashes, hardware, blocks, occupancy, cache, retraction or variation evidence; no superiority decision."}
    result["profiles"] = {path.name: trace_summary(path) for path in sorted((root / "traces").glob("*.gz"))}
    return result


def report(result):
    lines = ["# Serving headroom observation", "", "Raw output audit: " + json.dumps(result["output_hashes"]), "",
             "| Block | Status | U tok/s | Plateau seconds | Cache gate | Retracted |", "|---|---|---:|---:|---|---:|"]
    for label, row in result["blocks"].items():
        p = row.get("plateau") or {}
        lines.append(f"| {label} | {row['status']} | {p.get('U')} | {p.get('seconds')} | {row.get('cache_ok')} | {row.get('retractions')} |")
    lines += ["", "Comparison: " + json.dumps(result["comparison"]), "", "Profiles: " + str(len(result["profiles"])), "",
              "Native profiles are excluded from throughput comparisons. Kernel names and durations require manual inspection against installed/pinned source before choosing an expert intervention. No model promotion or scored submission.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("raw", type=Path)
    args = parser.parse_args(); value = audit(args.raw)
    (HERE / "audit.json").write_text(json.dumps(value, indent=1) + "\n")
    (HERE / "RESULT.md").write_text(report(value))
    print(json.dumps(value["comparison"]))
