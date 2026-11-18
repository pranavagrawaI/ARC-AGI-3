#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import urllib.error
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from constructor import ConstructionLimits, ConstructionResult, run_construction
from accounting import receipts
from real_trace import load_case
from sandbox_eval import evaluate_candidate
from request_contract import validate_request

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
MODEL = "qwen/qwen3.8-27b"
ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
TIMEOUT_SECONDS = 90
WALL_SECONDS = 900
AUTHORIZED_KEY = None


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(event, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


class OpenRouterTransport:
    def __init__(self, key: str, started: float) -> None:
        self.key = key
        self.started = started
        self.index = 0

    def __call__(self, body: dict) -> dict:
        validate_request(body)
        if time.monotonic() - self.started >= WALL_SECONDS:
            raise TimeoutError("construction episode wall limit reached")
        index = self.index
        request_path = RAW / "requests" / f"{index:02d}.json"
        response_path = RAW / "responses" / f"{index:02d}.json"
        write_json(request_path, body)
        wire_body = json.dumps(body).encode()
        request_path.with_suffix(".body").write_bytes(wire_body)
        request = urllib.request.Request(
            ENDPOINT,
            data=wire_body,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
        )
        called = time.monotonic()
        append_event(RAW / "request_attempts.jsonl", {"index": index, "status": "started",
                     "request_path": str(request_path), "requested_tokens": body["max_tokens"]})
        self.index += 1
        try:
            remaining = WALL_SECONDS - (time.monotonic() - self.started)
            with urllib.request.urlopen(request, timeout=min(TIMEOUT_SECONDS, remaining)) as response:
                raw = response.read()
            (RAW / "responses" ).mkdir(parents=True, exist_ok=True)
            (RAW / "responses" / f"{index:02d}.body").write_bytes(raw)
            payload = json.loads(raw)
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError):
                response_path.parent.mkdir(parents=True, exist_ok=True)
                (RAW / "responses" / f"{index:02d}.error.body").write_bytes(error.read())
            append_event(RAW / "request_attempts.jsonl", {"index": index,
                         "status": "failed", "error_type": type(error).__name__,
                         "http_status": getattr(error, 'code', None),
                         "wall_seconds": round(time.monotonic() - called, 6)})
            raise
        append_event(RAW / "request_attempts.jsonl", {"index": index, "status": "completed", "wall_seconds": round(time.monotonic() - called, 6), "provider": payload.get("provider")})
        write_json(response_path, payload)
        with (RAW / "model_calls.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "index": index,
                        "request_path": request_path.relative_to(HERE).as_posix(),
                        "response_path": response_path.relative_to(HERE).as_posix(),
                        "request_sha256": hashlib.sha256(
                            request_path.read_bytes()
                        ).hexdigest(),
                        "response_sha256": hashlib.sha256(
                            response_path.read_bytes()
                        ).hexdigest(),
                        "wall_seconds": round(time.monotonic() - called, 3),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return payload


def reasoning_was_used(transcript: list[dict]) -> bool:
    for item in transcript:
        response = item["response"]
        details = (response.get("usage") or {}).get("completion_tokens_details") or {}
        if int(details.get("reasoning_tokens") or 0) > 0:
            return True
        for choice in response.get("choices") or []:
            message = choice.get("message") or {}
            if any(message.get(k) for k in ('reasoning','reasoning_content','reasoning_details')):
                return True
    return False


def persist_progress(construction):
    write_json(RAW / 'progress.json', asdict(construction))
    write_json(RAW / 'transcript.json', construction.transcript)


def failure_result(error, started):
    def lines(name):
        path = RAW / name
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    attempts = lines('request_attempts.jsonl')
    admitted = [a for a in attempts if a['status'] == 'started']
    completed = [json.loads(p.read_text()) for p in sorted((RAW/'responses').glob('*.json'))]
    # Transport persistence can fail after receipt acquisition but before return
    # to the constructor. Reconstruct lossless completed pairs from durable raw.
    reconstructed = []
    for response_path in sorted((RAW/'responses').glob('*.json')):
        request_path = RAW/'requests'/response_path.name
        if request_path.exists():
            reconstructed.append({'request':json.loads(request_path.read_text()),
                                  'response':json.loads(response_path.read_text())})
    write_json(RAW/'transcript.json', reconstructed)
    account = receipts(completed, len(admitted))
    progress_path = RAW/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    unknown_attempt = any(a['index'] not in {x['index'] for x in attempts if x['status']=='completed'} for a in admitted)
    result = {
        'classification': 'INTEGRATION_INVALID',
        'failure_boundary': 'attempt_without_receipt' if unknown_attempt else 'local_after_receipts' if admitted else 'local_pre_send',
        'error': f'{type(error).__name__}: {error}',
        'model_requests': len(admitted),
        'requested_tokens': sum(a['requested_tokens'] for a in admitted),
        'usage': {k:account[k] for k in ('prompt_tokens','completion_tokens')} if account['complete'] else None,
        'known_successful_usage': account['known_successful_usage'],
        'provider_reported_cost_usd': str(account['cost']) if account['cost'] is not None else None,
        'usage_status': 'complete_receipts' if account['complete'] else 'unknown_total',
        'inspections': progress.get('inspections',0), 'submissions':progress.get('submissions',0),
        'elapsed_seconds': round(time.monotonic()-started,3),
    }
    write_json(HERE/'RESULT.json', result)
    return result


def execute() -> dict:
    started = time.monotonic()
    try:
        return _execute()
    except Exception as error:
        if not (HERE/'STARTED.json').exists() or (HERE/'RESULT.json').exists():
            raise
        return failure_result(error, started)


def _execute() -> dict:
    if RAW.exists() or (HERE / "STARTED.json").exists():
        raise RuntimeError("experiment identity already started; do not retry in place")
    key = AUTHORIZED_KEY
    if not key:
        raise RuntimeError(
            "Use the authorized single-launch key loader; "
            "this runner does not read credential files"
        )
    prelaunch = json.loads((HERE / 'PRELAUNCH.json').read_text())
    if prelaunch.get('passed') is not True:
        raise RuntimeError('prelaunch verification has not passed')
    for name, digest in prelaunch['subject_sha256'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('source changed after verification')
    case = load_case()
    started = time.monotonic()
    write_json(
        HERE / "STARTED.json",
        {
            "model": MODEL,
            "provider_routing": json.loads((HERE / "PROVIDER-SELECTION.json").read_text())["routing"],
            "artifact_sha256": {**prelaunch['subject_sha256'],
                                'PRELAUNCH.json': hashlib.sha256((HERE/'PRELAUNCH.json').read_bytes()).hexdigest()},
            "frontier": case.frontier,
            "incumbent_sha256": case.incumbent_sha256,
            "sealed_suffix_sha256": case.sealed_suffix_sha256,
            "wall_limit_seconds": WALL_SECONDS,
            "started_unix": time.time(),
        },
    )
    transport = OpenRouterTransport(key, started)
    construction = ConstructionResult(status='WORKING')
    try:
        construction = run_construction(
            case,
            transport=transport,
            limits=ConstructionLimits(),
            event_sink=lambda event: append_event(RAW / "tool_events.jsonl", event),
            progress_sink=persist_progress,
            result=construction,
        )
    except Exception as error:
        persist_progress(construction)
        return failure_result(error, started)

    summary = {
        key: value
        for key, value in asdict(construction).items()
        if key not in {"candidate_source", "transcript", "tool_events"}
    }
    summary["reasoning_used"] = reasoning_was_used(construction.transcript)
    write_json(RAW / "construction.json", summary)
    write_json(RAW / "transcript.json", construction.transcript)

    if time.monotonic() - started >= WALL_SECONDS:
        classification = "INTEGRATION_INVALID"
        suffix = None
    elif summary["reasoning_used"]:
        classification = "INTEGRATION_INVALID"
        suffix = None
    elif construction.status == "CANDIDATE_READY":
        candidate = construction.candidate_source
        assert candidate is not None
        candidate_path = RAW / "frozen_candidate.py"
        candidate_path.write_text(candidate)
        frozen_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        write_json(
            RAW / "FROZEN.json",
            {
                "model_sha256": frozen_sha,
                "source_path": candidate_path.relative_to(HERE).as_posix(),
                "work_item": construction.work_item,
                "visible_evaluation": construction.evaluation,
                "sealed_suffix_sha256": case.sealed_suffix_sha256,
            },
        )
        append_event(RAW / 'suffix_attempts.jsonl', {'status': 'started', 'model_sha256': frozen_sha})
        suffix = evaluate_candidate(candidate, case, include_suffix=True)["suffix"]
        append_event(RAW / 'suffix_attempts.jsonl', {'status': 'completed', 'model_sha256': frozen_sha})
        write_json(RAW / "sealed_suffix_evaluation.json", suffix)
        incumbent = prelaunch['incumbent_suffix']
        write_json(RAW / "incumbent_suffix_evaluation.json", incumbent)
        improved = suffix.get("compatible_depth", 0) > incumbent.get(
            "compatible_depth", 0
        )
        classification = (
            "INTEGRATION_VALID_CANDIDATE_SUPPORTED"
            if improved
            else "INTEGRATION_VALID_CANDIDATE_NOT_SUPPORTED"
        )
    elif construction.status == "NEEDS_EVIDENCE":
        classification = "INTEGRATION_VALID_NEEDS_EVIDENCE"
        suffix = None
    elif construction.status == "BUDGET_EXHAUSTED":
        classification = "INTEGRATION_VALID_BUDGET_EXHAUSTED"
        suffix = None
    else:
        classification = "INTEGRATION_INVALID"
        suffix = None

    result = {
        "classification": classification,
        "construction_status": construction.status,
        "model_requests": construction.model_requests,
        "inspections": construction.inspections,
        "submissions": construction.submissions,
        "requested_tokens": construction.requested_tokens,
        "usage": None if construction.status == 'INVALID_USAGE' else construction.usage,
        "reasoning_used": summary["reasoning_used"],
        "sealed_suffix_evaluation": suffix,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "scope": (
            "one LS20 integration episode; machinery and local construction evidence "
            "only, not game-general competence"
        ),
    }
    write_json(HERE / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        case = load_case()
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "paid_calls": 0,
                    "model": MODEL,
                    "frontier": case.frontier,
                    "planned_actions": case.planned_actions,
                    "sealed_steps": len(case.sealed_suffix),
                    "credential_from_environment": bool(
                        os.environ.get("OPENROUTER_API_KEY")
                    ),
                },
                sort_keys=True,
            )
        )
        return
    raise RuntimeError('Kaggle isolation is unqualified: this is an offline regression candidate, not a launchable package')


if __name__ == "__main__":
    main()
