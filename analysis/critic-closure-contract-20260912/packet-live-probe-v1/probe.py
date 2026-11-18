#!/usr/bin/env python3
"""Live packet probe. See EXPERIMENT-CARD.md; invoke with --execute to spend."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "raw"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "deployment/kaggle/submission-runtime-20260912-v9/chassis"))

import s5i5_fixture  # noqa: E402
import epistemic_critic as critic  # noqa: E402
from objection_ledger import ObjectionLedger  # noqa: E402

MODEL = "qwen/qwen3.8-27b"
CAP_CALLS = 20
CAP_WALL = 600.0
TIMEOUT = 60
ABORT_FAILURES = 4

OBJECTIVE = re.compile(r"level|complet|win\b|winning|objective|goal|64|finish", re.I)
PROXY = re.compile(r"\bbar\b|fill|click|counter|row 63|bottom", re.I)
MECHANICS = re.compile(r"which cell|why (?:did|do|does) .* (?:differ|change)|\(\d+,\s*\d+\)|"
                       r"coordinate|color \d|colou?r|pixel|delta of|\+2|\+1", re.I)


def classify(question: str) -> str:
    if not question.strip():
        return "empty"
    if OBJECTIVE.search(question) and PROXY.search(question):
        return "objective_gap"
    if MECHANICS.search(question):
        return "mechanics"
    return "other"


def conditions(built: list[dict]) -> list[dict]:
    by_id = {row["review_id"]: row for row in built}
    calls = []
    for row in built:                      # A: empty ledger, two samples each
        for sample in (1, 2):
            calls.append({"condition": "A", "boundary": row["review_id"],
                          "sample": sample, "packet": row["packet"]})
    seven = by_id["critic:7"]["packet"]
    probe = ObjectionLedger()
    probe.open(review_id="critic:3", step=6, wall_remaining=300.0, question={
        "claim_ref": "actor-message-119-tool-0",
        "question": "Is filling the bar the objective?", "reason": "It decides the next clicks."})
    probe.close({"review_id": "critic:3", "disposition": "probe",
                 "evidence": "fill the bar to 64 and watch for a level change",
                 "actions_budgeted": 46}, step=14)
    withdraw = ObjectionLedger()
    withdraw.open(review_id="critic:3", step=6, wall_remaining=300.0, question={
        "claim_ref": "actor-message-119-tool-0",
        "question": "Is filling the bar the objective?", "reason": "It decides the next clicks."})
    withdraw.close({"review_id": "critic:3", "disposition": "withdraw",
                    "evidence": "the bar is a counter, not the goal",
                    "retire_actions": ["CLICK 32 32"]}, step=14)
    for label, ledger in (("B", probe), ("C", withdraw)):
        packet = dict(seven, objections=ledger.view())
        for sample in (1, 2, 3):
            calls.append({"condition": label, "boundary": "critic:7",
                          "sample": sample, "packet": packet})
    assert len(calls) == CAP_CALLS
    return calls


def call(key: str, packet: dict) -> tuple[dict, float]:
    body = {"model": MODEL, "messages": critic.critic_messages(packet), "max_tokens": 192,
            "reasoning": {"enabled": False}}
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        payload = json.loads(response.read())
    return payload, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    built = s5i5_fixture.build()
    plan = conditions(built)
    if not args.execute:
        print(json.dumps({"planned_calls": len(plan), "paid_calls": 0,
                          "conditions": sorted({c["condition"] for c in plan})}))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "packets").mkdir(exist_ok=True)
    if (OUT / "STARTED.json").exists():
        raise RuntimeError("Never reuse a started run; a retry needs a new card and directory")
    key = None
    for line in Path("/home/pranav/projects/arc-agi-3/.env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            key = line.split("=", 1)[1].strip().strip("\"'")
    assert key, "no OpenRouter key"
    (OUT / "STARTED.json").write_text(json.dumps({"at": time.time(), "model": MODEL,
                                                  "planned": len(plan)}))
    run_started = time.monotonic()
    failures = 0
    for index, item in enumerate(plan):
        if time.monotonic() - run_started > CAP_WALL:
            print("wall cap reached; stopping"); break
        rendered = json.dumps(item["packet"], ensure_ascii=False, default=str)
        sha = hashlib.sha256(rendered.encode()).hexdigest()
        (OUT / "packets" / f"{item['condition']}-{item['boundary'].replace(':', '')}.json"
         ).write_text(rendered)
        row = {"index": index, "condition": item["condition"], "boundary": item["boundary"],
               "sample": item["sample"], "packet_sha256": sha, "packet_chars": len(rendered),
               "estimated_prompt_tokens": round(len(rendered) / 3.6)}
        try:
            payload, wall = call(key, item["packet"])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
            failures += 1
            row.update(status="transport_failure", error=repr(error))
            with (OUT / "calls.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{index:2d} {item['condition']} {item['boundary']:9} FAIL {error!r}")
            if failures >= ABORT_FAILURES:
                print("abort: failure cap"); break
            continue
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        usage = payload.get("usage") or {}
        reasoning_tokens = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                            or 0)
        row.update(status="completed", wall_seconds=round(wall, 2),
                   reported_model=payload.get("model"),
                   prompt_tokens=usage.get("prompt_tokens"),
                   completion_tokens=usage.get("completion_tokens"),
                   reasoning_tokens=reasoning_tokens,
                   reasoning_present=bool((message.get("reasoning") or "").strip()),
                   finish_reason=choice.get("finish_reason"), raw_response=content,
                   tool_calls=bool(message.get("tool_calls")))
        try:
            question = critic.validate(content)
            evidence = item["packet"]["actor_claims"]
            binding = (critic.bind_claim_ref(question, evidence)
                       if question["claim_ref"] else {"claim_ref": "", "match": None})
            row.update(valid=True, question=question, bound=bool(binding["match"]),
                       klass=classify(question["question"]))
        except ValueError as error:
            row.update(valid=False, validation_error=str(error), klass="invalid")
        with (OUT / "calls.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        q = row.get("question", {}).get("question", "") if row.get("valid") else row.get("validation_error")
        print(f"{index:2d} {item['condition']} {item['boundary']:9} s{item['sample']} "
              f"{row['wall_seconds']:5.1f}s ptok={row['prompt_tokens']} "
              f"[{row['klass']}] {q}")
    (OUT / "FINISHED.json").write_text(json.dumps({"at": time.time(), "failures": failures}))


if __name__ == "__main__":
    main()
