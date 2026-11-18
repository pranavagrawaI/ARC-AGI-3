"""Post-run q6 audit; preserves original host and automatic-audit artifacts.

Run with the Kaggle CLI's Python. Fetches only missing completed-response
receipts, checks their frozen evidence-index hashes, and writes AUDIT.json.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from collect_outputs import KaggleOutput


def load(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    root = HERE / "output"
    indexed = {r["path"]: r for r in load(root / "tycho-dfranzen/evidence-index.json")["files"]}
    report = {"kernel": "pranavagrawai/tycho-dfranzen-gpu-q6", "version": 1,
              "issues": [], "games": [], "additional_response_receipts": []}
    remote = None
    for name, record in indexed.items():
        path = root / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            report["issues"].append("hash mismatch: " + name)
    benchmark = load(root / "tycho-dfranzen/benchmark.json")
    automatic = load(HERE / "AUTOMATIC-AUDIT.json")
    report["original_automatic_issues"] = automatic["issues"]
    report["issues"].extend(i for i in automatic["issues"]
                            if not i.endswith(("trace tokens mismatch", "call count mismatch")))
    for run in benchmark["game_runs"]:
        path, = (root / "tycho-dfranzen/hybrid-games").glob(run["game_id"] + "-*")
        trace = lines(path / "llm-trace.jsonl")
        compute = load(path / "compute.json")
        prefix = trace[:compute["calls"]]
        assert sum(r["completion_tokens"] for r in prefix) == compute["completion_tokens"]
        prefix_roles = {}
        for record in prefix:
            role = prefix_roles.setdefault(record["source"], {"calls": 0, "completion_tokens": 0})
            role["calls"] += 1
            role["completion_tokens"] += record["completion_tokens"]
        assert prefix_roles == compute["roles"]
        tail = trace[compute["calls"]:]
        for event in tail:
            response_path = path / "decision-checkpoints/calls" / f'{event["call_index"]:06d}' / "response.json"
            name = response_path.relative_to(root).as_posix()
            expected = indexed[name]["sha256"]
            if not response_path.exists():
                remote = remote or KaggleOutput(report["kernel"], 1)
                data = remote.fetch(name)
                assert hashlib.sha256(data).hexdigest() == expected
                response_path.parent.mkdir(parents=True, exist_ok=True)
                response_path.write_bytes(data)
            assert hashlib.sha256(response_path.read_bytes()).hexdigest() == expected
            response = load(response_path)
            assert response["usage"]["completion_tokens"] == event["completion_tokens"]
            assert event["sequence"] == len(run["history"])
            report["additional_response_receipts"].append({"path": name, "sha256": expected,
                                                          "completion_tokens": event["completion_tokens"]})
        triggers = lines(path / "trigger-builder-trace.jsonl") if (path / "trigger-builder-trace.jsonl").exists() else []
        builder_tools = lines(path / "builder-tool-trace.jsonl") if (path / "builder-tool-trace.jsonl").exists() else []
        roles = {}
        for event in trace:
            role = roles.setdefault(event["source"], {"calls": 0, "completion_tokens": 0})
            role["calls"] += 1
            role["completion_tokens"] += event["completion_tokens"]
        row = {"game": run["game_id"], "levels": run["levels_completed"],
               "score": run["final_score"], "actions": len(run["history"]),
               "host_reported_calls": compute["calls"], "completed_calls": len(trace),
               "host_reported_completion_tokens": compute["completion_tokens"],
               "completed_completion_tokens": sum(r["completion_tokens"] for r in trace),
               "uncredited_completed_calls": len(tail), "roles": roles,
               "builder_requests": sum(t["event"] == "actor_request" for t in triggers),
               "builder_deadline_exceeded": [t for t in triggers if t["event"] == "builder_deadline_exceeded"],
               "builder_tools": dict(Counter(t["name"] for t in builder_tools)),
               "execution_receipt": load(path / "execution-receipt.json")}
        report["games"].append(row)
    report["mean_score"] = sum(g["score"] for g in report["games"]) / len(report["games"])
    report["completed_completion_tokens"] = sum(g["completed_completion_tokens"] for g in report["games"])
    report["host_reported_completion_tokens"] = sum(g["host_reported_completion_tokens"] for g in report["games"])
    report["accounting_diagnosis"] = (
        "AR25 and SB26 host stats equal exact prefixes of durable completed-response traces, "
        "including per-role counts. Eight later completed responses from the final uncommitted "
        "decision were omitted from compute.json and benchmark token totals. Raw provider "
        "receipts independently match all eight trace token counts. Unresolved in-flight "
        "provider usage remains unknown and is separate from these known completed responses.")
    report["interpretation"] = (
        "Gameplay/action receipts reconcile and all games reached their configured deadlines. "
        "Host accounting is defective; original automatic qualification remains failed. "
        "Two Builder invocations were admitted, but both timed out and rolled back; zero "
        "accepted Builder reports. Mechanism correctness and recovery are unmeasured. "
        "No matched efficacy verdict or authorization for a new launch follows from this audit.")
    (HERE / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("mean_score", "completed_completion_tokens",
                                            "host_reported_completion_tokens", "issues")}))


if __name__ == "__main__":
    main()
