"""Run repeated local E4B acquisition decisions against a hidden terminal label."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "experiments/hypothesis/0003-temporal-game-judgment"))

from acquisition import assemble_request, parse_response, score  # noqa: E402
from game_judgment.experiment import HTTPTransport, RunError, audit_openai_compatible_endpoint  # noqa: E402
from game_judgment.serialization import canonical_bytes, digest  # noqa: E402


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, default=ROOT / "fixtures/case.json")
    parser.add_argument("--authorize-local-model-calls", action="store_true")
    args = parser.parse_args()
    if not args.authorize_local_model_calls:
        raise RunError("execution requires --authorize-local-model-calls")
    if args.output.exists():
        raise FileExistsError(args.output)
    card = args.experiment_card.read_bytes()
    if len(card) > 1024:
        raise RunError("experiment card exceeds 1 KiB")
    endpoint = os.environ.get("GEMMA_E4B_ENDPOINT")
    if not endpoint:
        raise RunError("GEMMA_E4B_ENDPOINT is required")
    args.output.mkdir(parents=True)

    case = json.loads(args.case_file.read_text())
    label_path = ROOT / "scorer/label.json"
    config = json.loads((REPO / "experiments/hypothesis/0003-temporal-game-judgment/runtime-config.json").read_text())
    transfer = deepcopy(config)
    transfer["model"] = config["transfer"]["model"]
    transfer.pop("provider", None)
    request = assemble_request(case, transfer)
    write(args.output / "manifest.json", {
        "card_sha256": sha256(card).hexdigest(), "case_sha256": digest(case),
        "case_path": str(args.case_file),
        "label_sha256": sha256(label_path.read_bytes()).hexdigest(), "request_sha256": digest(request),
        "repetitions": 3, "environment_actions": 0,
    })
    write(args.output / "request.json", request)
    write(args.output / "provider-audit.json", audit_openai_compatible_endpoint(
        endpoint, config["transfer"]["model_audit_path"], config["transfer"]["model"]
    ))
    transport = HTTPTransport(
        endpoint.rstrip("/") + "/v1/chat/completions", "",
        int(config["request_timeout_seconds"]), 0.0, 1, 0.0,
        tuple(int(value) for value in config["transport"]["retry_http_statuses"]),
    )
    rows = []
    raw_rows = []
    try:
        for repeat in range(3):
            payload, audit = transport.call(
                {"phase": "acquisition", "case_id": case["case_id"], "repeat": repeat}, request
            )
            raw_rows.append({"repeat": repeat, "response": payload, "transport_audit": audit})
            write(args.output / "raw-responses.json", raw_rows)
            decision = parse_response(payload)
            # The scorer-only terminal label is consulted only after the response is immutable.
            label = json.loads(label_path.read_text())
            rows.append({
                "repeat": repeat, "response": payload, "transport_audit": audit,
                "decision": {"tool": decision.tool, "arguments": decision.arguments},
                "signature": list(decision.signature), "score": score(case, label, decision),
            })
            write(args.output / "responses.json", rows)
        signatures = [row["signature"] for row in rows]
        acquired = [row["score"]["acquired"] for row in rows]
        result = {
            "schema": "0004-acquisition-result-v1", "status": "PASS" if all(acquired) and len({tuple(row) for row in signatures}) == 1 else "FAIL",
            "stable": len({tuple(row) for row in signatures}) == 1,
            "acquired": acquired, "signatures": signatures,
            "model_calls": len(rows), "environment_actions": 0, "paid_cost_usd": 0.0,
        }
    except Exception as exc:
        result = {
            "schema": "0004-acquisition-result-v1", "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}", "model_calls": len(raw_rows),
            "environment_actions": 0, "paid_cost_usd": 0.0,
        }
    write(args.output / "result.json", result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
