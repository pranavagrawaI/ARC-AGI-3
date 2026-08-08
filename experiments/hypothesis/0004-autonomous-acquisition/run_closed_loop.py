"""Run E4B one action at a time from the frozen LS20 planning checkpoint."""

from __future__ import annotations

import argparse
from base64 import b64encode
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from typing import Any

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "experiments/hypothesis/0003-temporal-game-judgment"))
from closed_loop import SYSTEM_PROMPT, parse_response  # noqa: E402
from freeze_planning import PALETTE  # noqa: E402
from game_judgment.experiment import HTTPTransport, RunError, audit_openai_compatible_endpoint  # noqa: E402
from game_judgment.serialization import canonical_bytes, digest  # noqa: E402


EPISODES = 3
ACTION_CAP = 20


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_bytes(value) + b"\n")


def png_bytes(grid: Any) -> bytes:
    values = np.asarray(grid)
    image = Image.new("RGB", (values.shape[1] * 5, values.shape[0] * 5))
    pixels = image.load()
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            rgb = PALETTE[int(values[row, column]) % len(PALETTE)]
            for y in range(row * 5, row * 5 + 5):
                for x in range(column * 5, column * 5 + 5):
                    pixels[x, y] = rgb
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    visible = deepcopy(case)
    visible.pop("images")
    visible["checkpoint_state"] = visible.pop("current_state")
    visible["instruction"] = "Choose one action, observe its public successor, and complete the level within 20 actions."
    visible["feedback_contract"] = "Every next request contains the immediately previous and current public frames plus the complete action/outcome log."
    return visible


def frame_content(label: str, grid: Any) -> tuple[dict[str, Any], str, bytes]:
    data = png_bytes(grid)
    grid_hash = sha256(canonical_bytes(np.asarray(grid).tolist())).hexdigest()
    content = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + b64encode(data).decode()},
    }
    return content, grid_hash, data


def assemble_request(
    case: dict[str, Any], config: dict[str, Any], *, step: int,
    before_grid: Any | None, current_grid: Any, history: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[tuple[str, str, bytes]]]:
    content: list[dict[str, Any]] = []
    frame_rows: list[tuple[str, str, bytes]] = []
    if before_grid is not None:
        image, grid_hash, data = frame_content("previous", before_grid)
        content.append({"type": "text", "text": f"previous public frame; sha256={grid_hash}"})
        content.append(image)
        frame_rows.append(("previous", grid_hash, data))
    image, grid_hash, data = frame_content("current", current_grid)
    content.append({"type": "text", "text": f"current public frame; sha256={grid_hash}"})
    content.append(image)
    frame_rows.append(("current", grid_hash, data))
    visible = {
        "case": public_case(case),
        "closed_loop": {
            "step": step,
            "actions_remaining": ACTION_CAP - step,
            "history": history,
        },
    }
    content.append({"type": "text", "text": json.dumps(visible, sort_keys=True, indent=2)})
    request = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": config["top_k"],
        "max_tokens": 8,
        "reasoning": deepcopy(config["reasoning"]),
        "seed": config["seed"],
    }
    return request, frame_rows


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
        arcade.close_scorecard()
        raise RunError("offline checkpoint drift")
    return arcade, environment, frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, default=ROOT / "fixtures/planning-affordance-compact-case.json")
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
    forbidden = {"candidate_plans", "verified_route", "protected_successor", "checkpoint_prefix_actions"}
    if forbidden.intersection(case):
        raise RunError("model-visible case contains protected planning content")
    label_path = ROOT / "scorer/planning-label.json"
    label = json.loads(label_path.read_text())
    runtime_path = REPO / "experiments/hypothesis/0003-temporal-game-judgment/runtime-config.json"
    runtime = json.loads(runtime_path.read_text())
    config = deepcopy(runtime)
    config["model"] = runtime["transfer"]["model"]
    write(args.output / "manifest.json", {
        "card_sha256": sha256(card).hexdigest(),
        "case_sha256": digest(case),
        "case_path": str(args.case_file),
        "label_sha256": sha256(label_path.read_bytes()).hexdigest(),
        "runtime_sha256": sha256(runtime_path.read_bytes()).hexdigest(),
        "model_calls_cap": EPISODES * ACTION_CAP,
        "offline_actions_cap": EPISODES * (len(label["checkpoint_prefix_actions"]) + ACTION_CAP),
        "environment_actions": 0,
        "claim_boundary": "one-action feedback planning from a historical checkpoint with a machinery-supplied objective, action model, and causal affordance",
        "protected_label_use": "offline checkpoint reconstruction and external scoring only",
    })
    write(args.output / "provider-audit.json", audit_openai_compatible_endpoint(
        endpoint, runtime["transfer"]["model_audit_path"], config["model"]
    ))
    transport = HTTPTransport(
        endpoint.rstrip("/") + "/v1/chat/completions", "",
        int(runtime["request_timeout_seconds"]), 0.0, 1, 0.0,
        tuple(runtime["transport"]["retry_http_statuses"]),
    )
    results: list[dict[str, Any]] = []
    calls = 0
    try:
        for repeat in range(EPISODES):
            arcade, environment, frame = checkpoint(label)
            start_levels = frame.levels_completed
            before_grid = None
            current_grid = np.asarray(frame.frame[-1]).copy()
            history: list[dict[str, Any]] = []
            completed = False
            try:
                for step in range(ACTION_CAP):
                    request, frame_rows = assemble_request(
                        case, config, step=step, before_grid=before_grid,
                        current_grid=current_grid, history=history,
                    )
                    for frame_label, grid_hash, data in frame_rows:
                        path = args.output / "frames" / f"repeat-{repeat}-step-{step:02d}-{frame_label}-{grid_hash[:12]}.png"
                        if not path.exists():
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(data)
                    request_hash = digest(request)
                    append(args.output / "requests.jsonl", {
                        "repeat": repeat, "step": step, "request_sha256": request_hash,
                        "visible_case": public_case(case), "history": history,
                        "frame_sha256": {name: value for name, value, _ in frame_rows},
                    })
                    payload, audit = transport.call({"phase": "closed_loop", "repeat": repeat, "step": step}, request)
                    calls += 1
                    append(args.output / "raw-responses.jsonl", {
                        "repeat": repeat, "step": step, "request_sha256": request_hash,
                        "response": payload, "transport_audit": audit,
                    })
                    decision = parse_response(payload)
                    successor = environment.step(GameAction.from_id(decision.action))
                    successor_grid = np.asarray(successor.frame[-1]).copy()
                    outcome = {
                        "step": step,
                        "action": decision.action,
                        "changed_cells": int(np.count_nonzero(current_grid != successor_grid)),
                        "levels_completed": successor.levels_completed,
                        "state": successor.state.value,
                        "grid_sha256": sha256(canonical_bytes(successor_grid.tolist())).hexdigest(),
                    }
                    history.append(outcome)
                    append(args.output / "trace.jsonl", {"repeat": repeat, **outcome})
                    completed = successor.levels_completed > start_levels or successor.state.value == "WIN"
                    before_grid, current_grid, frame = current_grid, successor_grid, successor
                    if completed or successor.state.value == "GAME_OVER":
                        break
                results.append({
                    "repeat": repeat,
                    "completed_level": completed,
                    "actions": [row["action"] for row in history],
                    "actions_executed": len(history),
                    "start_levels_completed": start_levels,
                    "end_levels_completed": frame.levels_completed,
                    "end_state": frame.state.value,
                })
                write(args.output / "episodes.json", results)
            finally:
                arcade.close_scorecard()
        completed_rows = [row["completed_level"] for row in results]
        result = {
            "schema": "0004-closed-loop-result-v1",
            "status": "PASS" if all(completed_rows) else "MIXED" if any(completed_rows) else "FAIL",
            "completed": completed_rows,
            "model_calls": calls,
            "offline_actions": EPISODES * len(label["checkpoint_prefix_actions"]) + sum(row["actions_executed"] for row in results),
            "environment_actions": 0,
            "paid_cost_usd": 0.0,
        }
    except Exception as exc:
        result = {
            "schema": "0004-closed-loop-result-v1",
            "status": "INVALID",
            "error": f"{type(exc).__name__}: {exc}",
            "model_calls": calls,
            "episodes_completed": len(results),
            "environment_actions": 0,
            "paid_cost_usd": 0.0,
        }
    write(args.output / "result.json", result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
