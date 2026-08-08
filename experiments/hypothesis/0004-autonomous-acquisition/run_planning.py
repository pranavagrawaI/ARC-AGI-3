"""Ask E4B for routes and execute them in the real offline LS20 engine."""

from __future__ import annotations

import argparse
from base64 import b64encode
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "experiments/hypothesis/0003-temporal-game-judgment"))
from planning import SYSTEM_PROMPT, parse_response, tool_definitions  # noqa: E402
from game_judgment.experiment import HTTPTransport, RunError, audit_openai_compatible_endpoint  # noqa: E402
from game_judgment.serialization import canonical_bytes, digest  # noqa: E402


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def assemble_request(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    visible = deepcopy(case)
    images = visible.pop("images")
    content = []
    for image in images:
        path = ROOT / image["path"]
        data = path.read_bytes()
        if sha256(data).hexdigest() != image["sha256"]:
            raise RunError("planning image drift")
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64encode(data).decode()}})
    content.append({"type": "text", "text": json.dumps(visible, sort_keys=True, indent=2)})
    return {
        "model": config["model"], "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
        "tools": tool_definitions(), "tool_choice": "required", "temperature": config["temperature"],
        "top_p": config["top_p"], "top_k": config["top_k"], "max_tokens": config["max_completion_tokens"],
        "reasoning": deepcopy(config["reasoning"]), "seed": config["seed"],
    }


def checkpoint(label: dict[str, Any]):
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    environment = arcade.make("ls20")
    environment.reset()
    environment._game.set_level(label["offline_level_index"])
    frame = None
    for action in label["checkpoint_prefix_actions"]:
        frame = environment.step(GameAction.from_id(action))
    actual = sha256(canonical_bytes(frame.frame[-1].tolist())).hexdigest()
    if actual != label["expected_current_grid_sha256"]:
        raise RunError("offline checkpoint drift")
    return arcade, environment, frame


def execute_plan(label: dict[str, Any], actions: list[int]) -> dict[str, Any]:
    arcade, environment, frame = checkpoint(label)
    start = frame.levels_completed
    executed = []
    try:
        for action in actions:
            frame = environment.step(GameAction.from_id(action))
            executed.append(action)
            if frame.levels_completed > start or frame.state.value in {"WIN", "GAME_OVER"}:
                break
        return {
            "actions_submitted": len(actions), "actions_executed": len(executed),
            "checkpoint_actions": len(label["checkpoint_prefix_actions"]),
            "executed_actions": executed, "start_levels_completed": start,
            "end_levels_completed": frame.levels_completed, "end_state": frame.state.value,
            "completed_level": frame.levels_completed > start or frame.state.value == "WIN",
        }
    finally:
        arcade.close_scorecard()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, default=ROOT / "fixtures/planning-case.json")
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
    label_path = ROOT / "scorer/planning-label.json"
    label = json.loads(label_path.read_text())
    runtime = json.loads((REPO / "experiments/hypothesis/0003-temporal-game-judgment/runtime-config.json").read_text())
    config = deepcopy(runtime); config["model"] = runtime["transfer"]["model"]
    request = assemble_request(case, config)
    write(args.output / "manifest.json", {
        "card_sha256": sha256(card).hexdigest(), "case_sha256": digest(case),
        "case_path": str(args.case_file),
        "label_sha256": sha256(label_path.read_bytes()).hexdigest(), "request_sha256": digest(request),
        "model_calls_cap": 3, "offline_actions_cap": 261, "environment_actions": 0,
        "claim_boundary": "one-shot planning from a supplied observation prefix and acquired objective",
    })
    write(args.output / "request.json", request)
    write(args.output / "provider-audit.json", audit_openai_compatible_endpoint(endpoint, runtime["transfer"]["model_audit_path"], config["model"]))
    transport = HTTPTransport(endpoint.rstrip("/") + "/v1/chat/completions", "", int(runtime["request_timeout_seconds"]), 0.0, 1, 0.0, tuple(runtime["transport"]["retry_http_statuses"]))
    raw_rows, rows = [], []
    try:
        for repeat in range(3):
            payload, audit = transport.call({"phase": "planning", "repeat": repeat}, request)
            raw_rows.append({"repeat": repeat, "response": payload, "transport_audit": audit})
            write(args.output / "raw-responses.json", raw_rows)
            decision = parse_response(payload)
            replay = None if decision.tool == "ABSTAIN" else execute_plan(label, decision.arguments["actions"])
            rows.append({"repeat": repeat, "decision": {"tool": decision.tool, "arguments": decision.arguments}, "signature": list(decision.signature), "offline_replay": replay})
            write(args.output / "responses.json", rows)
        completed = [bool(row["offline_replay"] and row["offline_replay"]["completed_level"]) for row in rows]
        result = {
            "schema": "0004-planning-result-v1", "status": "PASS" if all(completed) else "FAIL",
            "completed": completed, "model_calls": len(rows),
            "offline_actions": sum(
                (row["offline_replay"] or {}).get("checkpoint_actions", 0)
                + (row["offline_replay"] or {}).get("actions_executed", 0)
                for row in rows
            ),
            "environment_actions": 0, "paid_cost_usd": 0.0,
        }
    except Exception as exc:
        result = {"schema": "0004-planning-result-v1", "status": "INVALID", "error": f"{type(exc).__name__}: {exc}", "model_calls": len(raw_rows), "environment_actions": 0, "paid_cost_usd": 0.0}
    write(args.output / "result.json", result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
