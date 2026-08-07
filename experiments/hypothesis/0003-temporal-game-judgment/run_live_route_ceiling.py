"""Run the bounded verified-route ceiling against one online LS20 level."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from game_judgment import (  # noqa: E402
    assemble_provider_request, canonical_bytes, digest, normalize_response,
    parse_decision, score,
)
from game_judgment.experiment import HTTPTransport, RunError, audit_openai_compatible_endpoint  # noqa: E402
from game_judgment.live import frame_record, validate_current, validate_successor  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-live-actions", action="store_true")
    args = parser.parse_args()
    if not args.authorize_live_actions:
        raise RunError("online execution requires --authorize-live-actions")
    if args.output.exists():
        raise FileExistsError(args.output)
    card = args.experiment_card.read_bytes()
    if len(card) > 1024:
        raise RunError("experiment card exceeds 1 KiB")
    if not os.environ.get("ARC_API_KEY"):
        raise RunError("ARC_API_KEY is not configured")
    endpoint = os.environ.get("GEMMA_E4B_ENDPOINT")
    if not endpoint:
        raise RunError("GEMMA_E4B_ENDPOINT is not configured")

    args.output.mkdir(parents=True)
    cases = json.loads((ROOT / "fixtures/live-route-ceiling.json").read_text())["cases"]
    config = json.loads((ROOT / "runtime-config.json").read_text())
    transfer = deepcopy(config)
    transfer["model"] = config["transfer"]["model"]
    transfer.pop("provider", None)
    result: dict[str, Any] = {
        "schema": "0003-live-route-ceiling-result-v1",
        "status": "RUNNING",
        "environment_actions": 0,
        "probe_actions": 0,
        "provider_calls": 0,
        "paid_cost_usd": 0.0,
        "levels_completed": 0,
        "experiment_card_sha256": sha256(card).hexdigest(),
    }
    write_json(args.output / "manifest.json", {
        "cases_sha256": digest(cases),
        "labels_sha256": sha256((ROOT / "scorer/live-route-ceiling-labels.json").read_bytes()).hexdigest(),
        "config_sha256": digest(transfer),
        "card_sha256": result["experiment_card_sha256"],
        "max_environment_actions": 13,
        "claim_boundary": "verified-route ceiling; not autonomous reconstruction",
    })
    write_json(args.output / "provider-audit.json", audit_openai_compatible_endpoint(
        endpoint, config["transfer"]["model_audit_path"], config["transfer"]["model"]
    ))
    transport = HTTPTransport(
        endpoint.rstrip("/") + "/v1/chat/completions",
        os.environ.get(config["transfer"]["api_key_environment"], ""),
        int(config["request_timeout_seconds"]), 0.0, 1, 0.0,
        tuple(int(value) for value in config["transport"]["retry_http_statuses"]),
    )

    arcade = None
    trace = []
    try:
        arcade = arc_agi.Arcade(
            arc_api_key=os.environ["ARC_API_KEY"], operation_mode=OperationMode.ONLINE,
            recordings_dir=str(args.output / "raw-recordings"),
        )
        environment = arcade.make("ls20", seed=0, save_recording=True)
        if environment is None:
            raise RunError("online environment was unavailable")
        current = environment.reset()
        labels = json.loads((ROOT / "scorer/live-route-ceiling-labels.json").read_text())["labels"]
        for index, (case, label) in enumerate(zip(cases, labels), 1):
            step_dir = args.output / "steps" / f"{index:02d}"
            write_json(step_dir / "current-frame.json", frame_record(current))
            current_check = validate_current(current, case, label)
            if not current_check["pass"]:
                raise RunError(f"step {index} current state mismatched certificate")
            request, audit = assemble_provider_request(case, "G", ROOT, transfer)
            write_json(step_dir / "request.json", request)
            write_json(step_dir / "request-audit.json", audit)
            started = perf_counter()
            response, transport_audit = transport.call(
                {"phase": "live-route-ceiling", "case_id": case["case_id"], "arm": "G"}, request
            )
            result["provider_calls"] += 1
            write_json(step_dir / "response.json", response)
            write_json(step_dir / "transport-audit.json", transport_audit)
            normalized = normalize_response(response)
            write_json(step_dir / "normalized.json", normalized)
            decision = parse_decision(normalized["decision"])
            decision_score = score(case, label, decision)
            if not decision_score["pass"]:
                raise RunError(f"step {index} decision failed the frozen contract")
            action = decision.arguments["first_action"]
            if result["environment_actions"] >= 13:
                raise RunError("action cap reached")
            successor = environment.step(GameAction.from_id(action))
            result["environment_actions"] += 1
            write_json(step_dir / "successor-frame.json", frame_record(successor))
            successor_check = validate_successor(
                successor, decision.arguments["predicted_public_fields"], label
            )
            trace.append({
                "step": index, "action": action, "decision": normalized["decision"],
                "decision_score": decision_score, "current_validation": current_check,
                "successor_validation": successor_check,
                "latency_seconds": perf_counter() - started,
            })
            write_json(args.output / "trace.json", trace)
            if not successor_check["pass"]:
                raise RunError(f"step {index} successor mismatched certificate")
            current = successor
        result["levels_completed"] = current.levels_completed
        if result["environment_actions"] != 13 or current.levels_completed != 1:
            raise RunError("route ceiling did not complete exactly one level in 13 actions")
        result["status"] = "PASS"
        write_json(args.output / "result.json", result)
        return 0
    except Exception as exc:
        result["levels_completed"] = getattr(current, "levels_completed", 0) if "current" in locals() else 0
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        write_json(args.output / "result.json", result)
        return 1
    finally:
        if arcade is not None:
            try:
                arcade.close_scorecard()
            except Exception as exc:
                write_json(args.output / "scorecard-close-error.json", {"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    raise SystemExit(main())
