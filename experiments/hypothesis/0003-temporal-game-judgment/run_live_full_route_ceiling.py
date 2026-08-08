"""Run the full-game verified-route ceiling until WIN or first mismatch."""

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
from game_judgment.live import (  # noqa: E402
    configure_remote_step_timeout, frame_sha256, public_state, validate_current,
    validate_successor,
)


class InvalidRun(RunError):
    pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as output:
        output.write(canonical_bytes(value) + b"\n")
        output.flush()


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
    endpoint = os.environ.get("GEMMA_E4B_ENDPOINT")
    if not os.environ.get("ARC_API_KEY") or not endpoint:
        raise RunError("ARC_API_KEY and GEMMA_E4B_ENDPOINT are required")

    args.output.mkdir(parents=True)
    fixture = json.loads((ROOT / "fixtures/live-full-route-ceiling.json").read_text())
    cases = fixture["cases"]
    labels_path = ROOT / "scorer/live-full-route-ceiling-labels.json"
    labels = json.loads(labels_path.read_text())["labels"]
    config = json.loads((ROOT / "runtime-config.json").read_text())
    transfer = deepcopy(config)
    transfer["model"] = config["transfer"]["model"]
    transfer.pop("provider", None)
    result: dict[str, Any] = {
        "schema": "0003-live-full-route-ceiling-result-v1",
        "status": "RUNNING", "environment_actions": 0, "probe_actions": 0,
        "provider_calls": 0, "paid_cost_usd": 0.0, "levels_completed": 0,
        "submitted_actions": 0,
        "completion_action_counts": [], "experiment_card_sha256": sha256(card).hexdigest(),
        "natural_stop": "WIN",
    }
    write_json(args.output / "manifest.json", {
        "fixture_sha256": digest(fixture), "labels_sha256": sha256(labels_path.read_bytes()).hexdigest(),
        "config_sha256": digest(transfer), "card_sha256": result["experiment_card_sha256"],
        "certificate_actions": len(cases), "action_cap": None,
        "arc_read_timeout_seconds": 60,
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
    current = None
    try:
        arcade = arc_agi.Arcade(
            arc_api_key=os.environ["ARC_API_KEY"], operation_mode=OperationMode.ONLINE,
            recordings_dir=str(args.output / "raw-recordings"),
        )
        environment = arcade.make("ls20", seed=0, save_recording=True)
        if environment is None:
            raise RunError("online environment was unavailable")
        current = environment.reset()
        configure_remote_step_timeout(environment, 60.0)
        for index, (case, label) in enumerate(zip(cases, labels), 1):
            if current.state.value == "WIN":
                break
            current_check = validate_current(current, case, label)
            if not current_check["pass"]:
                raise RunError(f"action {index} current state mismatched certificate")
            request, request_audit = assemble_provider_request(case, "G", ROOT, transfer)
            started = perf_counter()
            response, transport_audit = transport.call(
                {"phase": "live-full-route-ceiling", "case_id": case["case_id"], "arm": "G"}, request
            )
            result["provider_calls"] += 1
            normalized = normalize_response(response)
            decision = parse_decision(normalized["decision"])
            decision_score = score(case, label, decision)
            if not decision_score["pass"]:
                raise RunError(f"action {index} decision failed the frozen contract")
            previous_levels = current.levels_completed
            result["submitted_actions"] += 1
            successor = environment.step(GameAction.from_id(decision.arguments["first_action"]))
            if successor is None:
                raise InvalidRun(f"action {index} returned no successor")
            result["environment_actions"] += 1
            successor_check = validate_successor(
                successor, decision.arguments["predicted_public_fields"], label
            )
            record = {
                "action_number": index, "case_id": case["case_id"],
                "request": request, "request_audit": request_audit,
                "response": response, "transport_audit": transport_audit,
                "normalized": normalized, "decision_score": decision_score,
                "current_validation": current_check, "successor_validation": successor_check,
                "successor_frame_sha256": frame_sha256(successor),
                "successor_public": public_state(successor),
                "latency_seconds": perf_counter() - started,
            }
            append_jsonl(args.output / "trace.jsonl", record)
            if not successor_check["pass"]:
                raise RunError(f"action {index} successor mismatched certificate")
            current = successor
            if current.levels_completed > previous_levels:
                result["completion_action_counts"].append(result["environment_actions"])
                result["levels_completed"] = current.levels_completed
                write_json(args.output / "checkpoint.json", result)
        result["levels_completed"] = current.levels_completed
        if current.state.value != "WIN":
            raise RunError("certificate exhausted before WIN")
        result["status"] = "PASS"
        write_json(args.output / "result.json", result)
        return 0
    except Exception as exc:
        if current is not None:
            result["levels_completed"] = current.levels_completed
        result["status"] = "INVALID" if isinstance(exc, InvalidRun) else "FAIL"
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
