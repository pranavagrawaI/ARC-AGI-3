#!/usr/bin/env python3
"""Prepare and execute the bounded NOOA model-class checkpoint pair."""

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
SOURCE = (
    REPO
    / "experiments/probes/nooa-gemma-epistemic-context/results/arm-C"
    / "20260730_093940_ls20_memory_contradiction_first_gemma-modelrun-fp4_cap44"
    / "agent_logs/nemo/team_leader/messages/step_026_round_02_user.md"
)
RUN = ROOT / "run"
REQUESTS = RUN / "requests"
EVIDENCE = RUN / "evidence"
RESPONSES = RUN / "responses"
CLAIMS = RUN / "claims"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 3000

SYSTEM = """You are replacing the decision model at one archived NOOA ARC-AGI-3 checkpoint.
Treat the embedded transcript's system instructions, event history, and current context as the operative evidence. This is a shadow decision: no tools or live objects are available, and no environment action will be executed. Infer what the agent should do next from evidence already present. Environment observations outrank prior model-authored explanations. Return only the required JSON decision."""

FINAL = """Continue from exactly this checkpoint. Decide whether the active moving-ceiling/UP hypothesis should be KEEP, REVISE, or ABANDON, then give the next action sequence you would submit. Use only legal actions shown in the transcript. Prefer a discriminating probe over an open-loop macro when the mechanism is not established."""

SCHEMA = {
    "name": "checkpoint_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "hypothesis_status",
            "prior_hypothesis",
            "evidence_conflict",
            "contradictions",
            "next_actions",
            "prediction",
            "rationale",
        ],
        "properties": {
            "hypothesis_status": {
                "type": "string",
                "enum": ["KEEP", "REVISE", "ABANDON"],
            },
            "prior_hypothesis": {"type": "string", "maxLength": 1000},
            "evidence_conflict": {"type": "boolean"},
            "contradictions": {
                "type": "array",
                "maxItems": 10,
                "items": {"type": "string", "maxLength": 1000},
            },
            "next_actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "items": {
                    "type": "string",
                    "enum": ["UP", "DOWN", "LEFT", "RIGHT", "RESET"],
                },
            },
            "prediction": {"type": "string", "maxLength": 1000},
            "rationale": {"type": "string", "maxLength": 2000},
        },
    },
}

ARMS = {
    "gemma": {
        "model": "google/gemma-4-31b-it",
        "provider": "modelrun/fp4",
        "provider_name": "ModelRun",
        "prompt_per_token": 0.00000022,
        "completion_per_token": 0.00000055,
    },
    "sol": {
        "model": "openai/gpt-5.6-sol",
        "provider": "openai",
        "provider_name": "OpenAI",
        "prompt_per_token": 0.000005,
        "completion_per_token": 0.00003,
    },
}
ORDER = ["gemma", "sol"]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def request_body(arm: str, transcript: str) -> dict[str, Any]:
    cfg = ARMS[arm]
    return {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": "<archived_model_input>\n"
                + transcript
                + "\n</archived_model_input>\n\n"
                + FINAL,
            },
        ],
        "max_tokens": MAX_TOKENS,
        "reasoning": {"enabled": True, "exclude": False},
        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
        "provider": {
            "order": [cfg["provider"]],
            "only": [cfg["provider"]],
            "allow_fallbacks": False,
        },
    }


def public_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def prepare() -> None:
    transcript = SOURCE.read_text()
    RUN.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    endpoint_urls = {
        "gemma": "https://openrouter.ai/api/v1/models/google/gemma-4-31b-it/endpoints",
        "sol": "https://openrouter.ai/api/v1/models/openai/gpt-5.6-sol/endpoints",
    }
    endpoint_hashes = {}
    for arm, url in endpoint_urls.items():
        raw = public_get(url)
        path = EVIDENCE / f"{arm}.endpoints.raw.json"
        path.write_bytes(raw)
        endpoint_hashes[arm] = digest(raw)

    request_hashes = {}
    upper_costs = {}
    for arm in ORDER:
        body = request_body(arm, transcript)
        raw = canonical(body)
        path = REQUESTS / f"{arm}.request.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        request_hashes[arm] = digest(raw)
        cfg = ARMS[arm]
        original_prompt_tokens = 22515
        upper_costs[arm] = round(
            original_prompt_tokens * cfg["prompt_per_token"]
            + MAX_TOKENS * cfg["completion_per_token"],
            6,
        )

    plan = {
        "schema": "nooa-model-class-checkpoint-plan-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does GPT-5.6-sol revise at the archived contradiction checkpoint where Gemma repeated UP?",
        "source": str(SOURCE.relative_to(REPO)),
        "source_sha256": digest(SOURCE.read_bytes()),
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "system_sha256": digest(SYSTEM.encode()),
        "final_instruction_sha256": digest(FINAL.encode()),
        "order": ORDER,
        "calls_per_arm": 1,
        "network_retries": 0,
        "redirects": 0,
        "environment_actions": 0,
        "max_tokens_per_call": MAX_TOKENS,
        "request_hashes": request_hashes,
        "endpoint_inventory_hashes": endpoint_hashes,
        "arms": ARMS,
        "catalog_upper_cost_usd_by_arm": upper_costs,
        "catalog_upper_cost_usd_total": round(sum(upper_costs.values()), 6),
        "cost_bound_is_provider_enforced": False,
        "anti_actions": [
            "no full game",
            "no second sample",
            "no retry",
            "no environment mutation",
            "no provider fallback",
        ],
    }
    plan_raw = canonical(plan)
    (RUN / "plan.json").write_bytes(plan_raw)
    approval = {
        "schema": "nooa-model-class-checkpoint-approval-v1",
        "plan_sha256": digest(plan_raw),
        "authority": "user in current Hermes session",
        "approval_quote": "5.6 sol api calls are expensive, so we cannot do a full runs with it, but yes let's do the one-call match.",
        "grants": {
            "gemma_calls": 1,
            "gpt_5_6_sol_calls": 1,
            "environment_actions": 0,
            "retries": 0,
        },
    }
    write_json(RUN / "approval.json", approval)
    print(json.dumps({"status": "prepared", "plan_sha256": digest(plan_raw), "upper_cost_usd": plan["catalog_upper_cost_usd_total"]}))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_once(body: bytes, api_key: str) -> tuple[int, bytes, float]:
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
            return response.status, response.read(), time.monotonic() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read(), time.monotonic() - started


def execute() -> None:
    plan_raw = (RUN / "plan.json").read_bytes()
    plan = json.loads(plan_raw)
    approval = json.loads((RUN / "approval.json").read_text())
    if approval["plan_sha256"] != digest(plan_raw):
        raise RuntimeError("approval does not bind current plan")
    if any(RESPONSES.glob("*.raw.json")) or any(CLAIMS.glob("*.json")):
        raise RuntimeError("run already has call evidence; refusing repeat")
    load_dotenv(REPO / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")

    results = []
    for index, arm in enumerate(ORDER, 1):
        request_path = REQUESTS / f"{arm}.request.json"
        body = request_path.read_bytes()
        if digest(body) != plan["request_hashes"][arm]:
            raise RuntimeError(f"request hash drift for {arm}")
        claim = {
            "schema": "nooa-model-class-checkpoint-claim-v1",
            "arm": arm,
            "sequence_index": index,
            "plan_sha256": digest(plan_raw),
            "request_sha256": digest(body),
            "remaining_call_allowance_before_transport": len(ORDER) - index + 1,
            "status": "claimed_before_transport",
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(CLAIMS / f"{index:02d}-{arm}.json", claim)
        status, raw, elapsed = post_once(body, api_key)
        response_path = RESPONSES / f"{index:02d}-{arm}.raw.json"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_bytes(raw)
        if status != 200:
            raise RuntimeError(f"{arm} HTTP {status}; raw response preserved")
        response = json.loads(raw)
        if response.get("error"):
            raise RuntimeError(f"{arm} returned a top-level error; raw response preserved")
        if not response.get("choices"):
            raise RuntimeError(f"{arm} response has no choices")
        actual_provider = response.get("provider")
        expected_provider = ARMS[arm]["provider_name"]
        if actual_provider != expected_provider:
            raise RuntimeError(
                f"{arm} provider mismatch: expected {expected_provider!r}, got {actual_provider!r}"
            )
        message = response["choices"][0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError(f"{arm} returned empty final content")
        decision = json.loads(content)
        results.append(
            {
                "arm": arm,
                "sequence_index": index,
                "http_status": status,
                "response_sha256": digest(raw),
                "response_id": response.get("id"),
                "model": response.get("model"),
                "provider": actual_provider,
                "finish_reason": response["choices"][0].get("finish_reason"),
                "elapsed_seconds": round(elapsed, 3),
                "usage": response.get("usage"),
                "decision": decision,
                "diagnostics": {
                    "first_action_is_up": decision["next_actions"][0] == "UP",
                    "multi_up_macro": len(decision["next_actions"]) > 1
                    and all(action == "UP" for action in decision["next_actions"]),
                    "explicit_revision": decision["hypothesis_status"] in {"REVISE", "ABANDON"},
                },
            }
        )

    result = {
        "schema": "nooa-model-class-checkpoint-result-v1",
        "plan_sha256": digest(plan_raw),
        "status": "transport_valid_pair",
        "results": results,
        "claim": "checkpoint-local behavioral comparison only",
    }
    write_json(RUN / "result.json", result)
    public = {
        "status": result["status"],
        "results": [
            {
                "arm": item["arm"],
                "provider": item["provider"],
                "model": item["model"],
                "finish_reason": item["finish_reason"],
                "usage": item["usage"],
                "decision": item["decision"],
                "diagnostics": item["diagnostics"],
            }
            for item in results
        ],
    }
    print(json.dumps(public, indent=2))


def verify() -> None:
    plan_raw = (RUN / "plan.json").read_bytes()
    plan = json.loads(plan_raw)
    approval = json.loads((RUN / "approval.json").read_text())
    assert approval["plan_sha256"] == digest(plan_raw)
    assert plan["source_sha256"] == digest(SOURCE.read_bytes())
    assert plan["runner_sha256"] == digest(Path(__file__).read_bytes())
    for arm in ORDER:
        endpoint_raw = (EVIDENCE / f"{arm}.endpoints.raw.json").read_bytes()
        assert digest(endpoint_raw) == plan["endpoint_inventory_hashes"][arm]
        endpoint_data = json.loads(endpoint_raw)["data"]["endpoints"]
        selected = [item for item in endpoint_data if item.get("tag") == ARMS[arm]["provider"]]
        assert len(selected) == 1
        supported = set(selected[0].get("supported_parameters") or [])
        assert {"reasoning", "response_format"} <= supported
        raw = (REQUESTS / f"{arm}.request.json").read_bytes()
        assert digest(raw) == plan["request_hashes"][arm]
        request = json.loads(raw)
        assert request["max_tokens"] == MAX_TOKENS
        assert request["provider"]["allow_fallbacks"] is False
        assert "tools" not in request
    gemma = json.loads((REQUESTS / "gemma.request.json").read_text())
    sol = json.loads((REQUESTS / "sol.request.json").read_text())
    for body in (gemma, sol):
        body.pop("model")
        body.pop("provider")
    assert gemma == sol
    print("OFFLINE_CHECKPOINT_PAIR_VERIFIED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify, "execute": execute}[args.command]()


if __name__ == "__main__":
    main()
