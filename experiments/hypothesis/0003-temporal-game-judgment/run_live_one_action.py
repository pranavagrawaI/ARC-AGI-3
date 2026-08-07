"""Execute the fail-closed, precommitted one-action online gate."""

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
    assemble_provider_request,
    canonical_bytes,
    digest,
    normalize_response,
    parse_decision,
    score,
)
from game_judgment.experiment import (  # noqa: E402
    HTTPTransport,
    RunError,
    audit_openai_compatible_endpoint,
)
from game_judgment.live import (  # noqa: E402
    frame_record,
    validate_current,
    validate_successor,
)


CASE_PATH = ROOT / "fixtures/live-initial-case.json"
LABEL_PATH = ROOT / "scorer/live-initial-label.json"
CONFIG_PATH = ROOT / "runtime-config.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def finish(output: Path, result: dict[str, Any], status: str, error: str | None = None) -> int:
    result["status"] = status
    if error is not None:
        result["error"] = error
    write_json(output / "result.json", result)
    return 0 if status == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-live-action", action="store_true")
    args = parser.parse_args()

    if not args.authorize_live_action:
        raise RunError("online execution requires --authorize-live-action")
    if args.output.exists():
        raise FileExistsError(args.output)
    card_bytes = args.experiment_card.read_bytes()
    if len(card_bytes) > 1024:
        raise RunError("experiment card exceeds 1 KiB")
    if not os.environ.get("ARC_API_KEY"):
        raise RunError("ARC_API_KEY is not configured")
    endpoint = os.environ.get("GEMMA_E4B_ENDPOINT")
    if not endpoint:
        raise RunError("GEMMA_E4B_ENDPOINT is not configured")

    args.output.mkdir(parents=True)
    result: dict[str, Any] = {
        "schema": "0003-live-one-action-result-v1",
        "environment_actions": 0,
        "provider_calls": 0,
        "paid_cost_usd": 0.0,
        "experiment_card": {
            "path": str(args.experiment_card),
            "sha256": sha256(card_bytes).hexdigest(),
            "bytes": len(card_bytes),
        },
    }
    case = json.loads(CASE_PATH.read_text())
    config = json.loads(CONFIG_PATH.read_text())
    transfer_config = deepcopy(config)
    transfer_config["model"] = config["transfer"]["model"]
    transfer_config.pop("provider", None)
    write_json(args.output / "manifest.json", {
        "case_sha256": digest(case),
        "config_sha256": digest(transfer_config),
        "label_sha256": sha256(LABEL_PATH.read_bytes()).hexdigest(),
        "experiment_card_sha256": result["experiment_card"]["sha256"],
        "max_environment_actions": 1,
    })

    audit = audit_openai_compatible_endpoint(
        endpoint,
        config["transfer"]["model_audit_path"],
        config["transfer"]["model"],
    )
    write_json(args.output / "provider-audit.json", audit)
    transport_config = config["transport"]
    transport = HTTPTransport(
        endpoint.rstrip("/") + "/v1/chat/completions",
        os.environ.get(config["transfer"]["api_key_environment"], ""),
        int(config["request_timeout_seconds"]),
        0.0,
        1,
        0.0,
        tuple(int(value) for value in transport_config["retry_http_statuses"]),
    )

    arcade = None
    try:
        arcade = arc_agi.Arcade(
            arc_api_key=os.environ["ARC_API_KEY"],
            operation_mode=OperationMode.ONLINE,
            recordings_dir=str(args.output / "raw-recordings"),
        )
        environment = arcade.make("ls20", seed=0, save_recording=True)
        if environment is None:
            return finish(args.output, result, "INVALID", "online environment was unavailable")
        current = environment.reset()
        write_json(args.output / "current-frame.json", frame_record(current))

        request, request_audit = assemble_provider_request(case, "G", ROOT, transfer_config)
        write_json(args.output / "provider-request.json", request)
        write_json(args.output / "request-audit.json", request_audit)

        # Freeze the request before opening the scorer-only expected observations.
        label = json.loads(LABEL_PATH.read_text())
        current_validation = validate_current(current, case, label)
        result["current_validation"] = current_validation
        if not current_validation["pass"]:
            return finish(args.output, result, "INVALID", "online reset did not match the frozen current state")

        started = perf_counter()
        response, transport_audit = transport.call(
            {"phase": "live-one-action", "case_id": case["case_id"], "arm": "G"},
            request,
        )
        result["provider_calls"] = 1
        write_json(args.output / "provider-response.json", response)
        write_json(args.output / "transport-audit.json", transport_audit)
        result["decision_latency_seconds"] = perf_counter() - started
        normalized = normalize_response(response)
        write_json(args.output / "normalized-response.json", normalized)
        decision = parse_decision(normalized["decision"])
        decision_score = score(case, label, decision)
        result["decision"] = normalized["decision"]
        result["decision_score"] = decision_score
        if not decision_score["pass"]:
            return finish(args.output, result, "FAIL", "E4B decision did not pass the frozen contract")

        action_id = decision.arguments["first_action"]
        if action_id != 3 or result["environment_actions"] != 0:
            return finish(args.output, result, "FAIL", "decision did not authorize the sole precommitted action")
        successor = environment.step(GameAction.from_id(action_id))
        result["environment_actions"] = 1
        write_json(args.output / "successor-frame.json", frame_record(successor))
        successor_validation = validate_successor(
            successor, decision.arguments["predicted_public_fields"], label
        )
        result["successor_validation"] = successor_validation
        if not successor_validation["pass"]:
            return finish(args.output, result, "FAIL", "live successor did not match the replay certificate")
        return finish(args.output, result, "PASS")
    except Exception as exc:
        return finish(args.output, result, "FAIL", f"{type(exc).__name__}: {exc}")
    finally:
        if arcade is not None:
            try:
                arcade.close_scorecard()
            except Exception as exc:
                # Preserve cleanup failure without replacing the already recorded gate result.
                write_json(args.output / "scorecard-close-error.json", {
                    "error": f"{type(exc).__name__}: {exc}"
                })


if __name__ == "__main__":
    raise SystemExit(main())
