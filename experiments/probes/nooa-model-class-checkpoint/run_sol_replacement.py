#!/usr/bin/env python3
"""One-call Sol replacement after the frozen Gemma arm ended without final content."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
ORIGINAL = ROOT / "run"
REPLACEMENT = ROOT / "replacement-sol"
REQUEST = ORIGINAL / "requests/sol.request.json"
GEMMA_RAW = ORIGINAL / "responses/01-gemma.raw.json"
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def prepare() -> None:
    original_plan = (ORIGINAL / "plan.json").read_bytes()
    gemma = json.loads(GEMMA_RAW.read_text())
    usage = gemma["usage"]
    message = gemma["choices"][0]["message"]
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 3000
    assert message.get("content") is None
    assert message.get("reasoning")
    plan = {
        "schema": "nooa-model-class-checkpoint-sol-replacement-plan-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Original Gemma arm exhausted its 3000-token budget in reasoning and returned no structured final; user authorized the remaining single Sol call.",
        "original_plan_sha256": digest(original_plan),
        "gemma_raw_response_sha256": digest(GEMMA_RAW.read_bytes()),
        "gemma_observed_cost_usd": usage.get("cost"),
        "sol_request_sha256": digest(REQUEST.read_bytes()),
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "calls": 1,
        "model": "openai/gpt-5.6-sol",
        "provider_only": "openai",
        "allow_fallbacks": False,
        "max_tokens": 3000,
        "retries": 0,
        "redirects": 0,
        "environment_actions": 0,
        "catalog_max_sol_cost_usd": 0.205,
        "cost_bound_is_provider_enforced": False,
        "interpretation": "replacement comparison using Gemma's preserved reasoning, not a format-valid pair",
    }
    raw = canonical(plan)
    REPLACEMENT.mkdir(parents=True, exist_ok=True)
    (REPLACEMENT / "plan.json").write_bytes(raw)
    approval = {
        "schema": "nooa-model-class-checkpoint-sol-replacement-approval-v1",
        "plan_sha256": digest(raw),
        "authority": "user in current Hermes session",
        "approval_quote": "Run the one Sol call",
        "grant": {"sol_calls": 1, "retries": 0, "environment_actions": 0},
    }
    write_json(REPLACEMENT / "approval.json", approval)
    print(json.dumps({"status": "prepared", "plan_sha256": digest(raw)}))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def execute() -> None:
    plan_raw = (REPLACEMENT / "plan.json").read_bytes()
    plan = json.loads(plan_raw)
    approval = json.loads((REPLACEMENT / "approval.json").read_text())
    assert approval["plan_sha256"] == digest(plan_raw)
    assert plan["runner_sha256"] == digest(Path(__file__).read_bytes())
    body = REQUEST.read_bytes()
    assert plan["sol_request_sha256"] == digest(body)
    request_body = json.loads(body)
    assert request_body["model"] == "openai/gpt-5.6-sol"
    assert request_body["provider"] == {
        "order": ["openai"],
        "only": ["openai"],
        "allow_fallbacks": False,
    }
    response_path = REPLACEMENT / "response.raw.json"
    claim_path = REPLACEMENT / "claim.json"
    if response_path.exists() or claim_path.exists():
        raise RuntimeError("replacement call already claimed or executed")
    write_json(
        claim_path,
        {
            "schema": "nooa-model-class-checkpoint-sol-replacement-claim-v1",
            "status": "claimed_before_transport",
            "plan_sha256": digest(plan_raw),
            "request_sha256": digest(body),
            "remaining_call_allowance_before_transport": 1,
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    load_dotenv(REPO / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")
    opener = urllib.request.build_opener(NoRedirect)
    request = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    started = time.monotonic()
    try:
        with opener.open(request, timeout=300) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    elapsed = time.monotonic() - started
    response_path.write_bytes(raw)
    result: dict[str, Any] = {
        "schema": "nooa-model-class-checkpoint-sol-replacement-result-v1",
        "plan_sha256": digest(plan_raw),
        "http_status": status,
        "response_sha256": digest(raw),
        "elapsed_seconds": round(elapsed, 3),
        "valid": False,
    }
    if status == 200:
        response = json.loads(raw)
        result.update(
            {
                "response_id": response.get("id"),
                "model": response.get("model"),
                "provider": response.get("provider"),
                "usage": response.get("usage"),
                "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
            }
        )
        choices = response.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        content = message.get("content")
        if response.get("provider") == "OpenAI" and content:
            result["decision"] = json.loads(content)
            result["valid"] = True
        else:
            result["reasoning_present"] = bool(message.get("reasoning"))
            result["content_present"] = bool(content)
    write_json(REPLACEMENT / "result.json", result)
    print(json.dumps(result, indent=2))


def verify() -> None:
    plan_raw = (REPLACEMENT / "plan.json").read_bytes()
    plan = json.loads(plan_raw)
    approval = json.loads((REPLACEMENT / "approval.json").read_text())
    assert approval["plan_sha256"] == digest(plan_raw)
    assert plan["runner_sha256"] == digest(Path(__file__).read_bytes())
    assert plan["original_plan_sha256"] == digest((ORIGINAL / "plan.json").read_bytes())
    assert plan["gemma_raw_response_sha256"] == digest(GEMMA_RAW.read_bytes())
    assert plan["sol_request_sha256"] == digest(REQUEST.read_bytes())
    print("OFFLINE_SOL_REPLACEMENT_VERIFIED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify, "execute": execute}[args.command]()


if __name__ == "__main__":
    main()
