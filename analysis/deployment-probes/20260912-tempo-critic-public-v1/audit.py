"""Read-only validity audit and descriptive metrics for the frozen public run."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
Q = RAW / "qualification"


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dist(values):
    values = list(values)
    if not values:
        return {"n": 0}
    return {"n": len(values), "total": sum(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "maximum": max(values)}


def audit():
    build = load(HERE / "source/BUILD.json")
    manifest = load(RAW / "nooa-join/CHASSIS.json")
    assert build["contract_sha256"] == manifest["contract_sha256"]
    assert build["contract_sha256"] == "b294519f23a9d9d33d9ee8e773a8db14fd3ddc84e21b862da8a1bcc9201da9b9"
    assert all(digest(HERE / "source" / name) == value for name, value in build["files"].items())
    assert all(digest(RAW / "nooa-join" / name) == value for name, value in manifest["files"].items())
    local_nb = load(HERE / "source/submission.ipynb")
    remote_nb = load(HERE / "remote-source/arc3-tempo-critic-public-20260912.ipynb")
    assert ["".join(c["source"]) for c in local_nb["cells"]] == [
        "".join(c["source"]) for c in remote_nb["cells"]
    ]

    result = load(Q / "result.json")
    assert result["status"] == "passed" and result["model_verification"] == "matched"
    lifecycle = result["lifecycle"]
    assert lifecycle["status"] == "passed" and lifecycle["containment_passed"]
    assert lifecycle["root_returncode"] == 0 and lifecycle["term_only"]
    progress = rows(Q / "game-progress.jsonl")
    allocations = rows(Q / "budget-allocations.jsonl")
    assert [r["game_id"].split("-")[0] for r in progress] == ["ls20", "s5i5", "cd82"]
    assert [r["game_id"] for r in allocations] == [r["game_id"] for r in progress]
    assert all(r["admission_seconds"] == 600 for r in allocations)
    sessions = [p for p in (Q / "solver-workspaces").iterdir() if p.is_dir()]
    assert len(sessions) == 3
    reports = []
    all_calls = []
    for game, allocation in zip(progress, allocations):
        session, = [s for s in sessions if s.name.startswith(game["game_id"] + "-")]
        ipc = session / "ipc"
        calls = rows(ipc / "model_calls.jsonl")
        events = rows(ipc / "request_lifecycle.jsonl")
        timings = rows(ipc / "execution_timing.jsonl")
        reviews = rows(ipc / "critic_reviews.jsonl")
        assert len(timings) == game["solver_actions"]
        admitted = [e for e in events if e["event"] == "admitted"]
        completed = [e for e in events if e["event"] == "completed"]
        assert not [e for e in events if e["event"] == "failed"]
        assert sorted(e["request_id"] for e in admitted) == sorted(e["request_id"] for e in completed)
        assert sorted(e["request_id"] for e in completed) == sorted(c["request_id"] for c in calls)
        assert len({c["request_id"] for c in calls}) == len(calls)
        start = allocation["admission_deadline"] - 600
        end = start + game["game_elapsed_seconds"]
        assert all(start <= e["at_monotonic"] < allocation["admission_deadline"] for e in admitted)
        assert all(e["at_monotonic"] <= end for e in completed)
        assert not (ipc / "ABORT").exists() and not (ipc / "request_failure.json").exists()
        assert load(session / "worker-metadata.json")["contract_sha256"] == build["contract_sha256"]
        purposes = collections.Counter(c["request_purpose"] for c in calls)
        assert purposes["critic"] == len(reviews) == 20
        assert purposes["actor"] + purposes["critic"] == len(calls)
        assert all(not c.get("reasoning_present") and not c.get("reasoning_tokens") for c in calls)
        assert all((c.get("completion_tokens") or 0) <= (256 if c["request_purpose"] == "critic" else 16384) for c in calls)
        review_statuses = collections.Counter(r["status"] for r in reviews)
        valid = [r for r in reviews if r["status"] == "completed"]
        verdicts = collections.Counter(r["critique"]["tempo_verdict"] for r in valid)
        assessments = collections.Counter(r["critique"]["assessment"] for r in valid)
        completed_at = {e["request_id"]: e["at_monotonic"] for e in completed}
        delays = []
        for call in calls:
            if call["request_purpose"] != "critic":
                continue
            later = [t["observed_at_monotonic"] for t in timings
                     if t["observed_at_monotonic"] >= completed_at[call["request_id"]]]
            if later:
                delays.append(min(later) - completed_at[call["request_id"]])
        reports.append({
            **game, "session": session.name,
            "actions_per_minute": game["solver_actions"] / game["game_elapsed_seconds"] * 60,
            "seconds_to_first_action": min(t["observed_at_monotonic"] for t in timings) - start,
            "request_purposes": dict(purposes), "review_statuses": dict(review_statuses),
            "valid_assessments": dict(assessments), "valid_verdicts": dict(verdicts),
            "actor_call_wall_seconds": dist(c["wall_seconds"] for c in calls if c["request_purpose"] == "actor"),
            "critic_call_wall_seconds": dist(c["wall_seconds"] for c in calls if c["request_purpose"] == "critic"),
            "review_to_next_action_seconds": dist(delays),
            "tokens": {purpose: {
                "prompt": sum(c.get("prompt_tokens") or 0 for c in calls if c["request_purpose"] == purpose),
                "cached_prompt": sum(c.get("cached_prompt_tokens") or 0 for c in calls if c["request_purpose"] == purpose),
                "completion": sum(c.get("completion_tokens") or 0 for c in calls if c["request_purpose"] == purpose),
            } for purpose in ("actor", "critic")},
        })
        all_calls.extend(calls)

    facts = result["request_count_facts"]
    assert len(all_calls) == facts["observed_successful_launcher_solver_requests"] == 219
    assert facts["observed_total_chat_completion_accesses"] == facts["observed_total_successful_chat_completion_accesses"] == 220
    assert facts["whole_model_request_count_verified"]
    server_log = (Q / "server.log").read_text()
    markers = re.findall(
        r"out of memory|OutOfMemoryError|\bOOM\b|Request state was deleted|Traceback \(most recent call last\)|HTTP/1\.1\" [45]\d\d",
        server_log, re.I,
    )
    assert not markers, markers
    assert "Remaining number of requests 0" in server_log
    terminal = load(RAW / "arc3-tempo-critic-public-20260912.log")
    assert max(row["time"] for row in terminal) < 3600
    totals = collections.Counter(c["request_purpose"] for c in all_calls)
    return {
        "status": "passed", "scope": "public descriptive run; no causal attribution",
        "contract_sha256": build["contract_sha256"], "games": reports,
        "totals": {"executed_actions": sum(g["solver_actions"] for g in progress),
                   "levels_completed": sum(g["levels_completed"] for g in progress),
                   "actor_calls": totals["actor"], "critic_calls": totals["critic"],
                   "valid_reviews": sum(g["review_statuses"].get("completed", 0) for g in reports),
                   "invalid_reviews": sum(g["review_statuses"].get("invalid_output", 0) for g in reports)},
        "outer_elapsed_seconds": result["elapsed_s"],
        "notebook_last_log_seconds": max(row["time"] for row in terminal),
    }


if __name__ == "__main__":
    result = audit()
    (HERE / "AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
