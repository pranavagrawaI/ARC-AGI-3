#!/usr/bin/env python3
"""Replay the selected typed-goal checkpoint through the current lifecycle gate."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tycho_nooa import ActionDecision, GameTurn, TychoNooaAgent

ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "runs/tycho-nooa-typed-replacement-gemma4-31b-ar25-20260821T170433Z"
    / "game"
)
OUTPUT = Path(__file__).resolve().parent
WORKSPACE_ROOT = OUTPUT / "workspace"
WORKSPACE = WORKSPACE_ROOT / "ar25"
CHECKPOINT_CALL = "000006"
GOAL_ID = "goal-0001"
PREDICATE = (
    "Match the shape at (15,36)-(23,44) with the shape at (45,51)-(53,59) "
    "using a target region or specific alignment."
)
FALSIFIER = (
    "The game ends or is solved without the player moving the shape at "
    "(15,36)-(23,44) to the position of the shape at (45,51)-(53,59), or "
    "vice versa."
)
DISCRIMINATOR = (
    "Retain only until a movement sequence places the two shapes in the proposed "
    "alignment; then compare host levels_completed and terminal evidence against "
    "the live alternative that alignment is merely visual similarity."
)


def main() -> None:
    if WORKSPACE_ROOT.exists():
        raise FileExistsError(
            f"refusing to overwrite preserved replay output: {WORKSPACE_ROOT}"
        )
    manifest = _read_json(_call_path() / "workspace.json")
    _materialize(manifest, WORKSPACE)
    source_goal = _source_goal()
    if source_goal != {
        "id": GOAL_ID,
        "predicate": PREDICATE,
        "evidence_refs": ["level_0/turn_000.txt"],
        "falsifier": FALSIFIER,
        "status": "proposed",
        "note": "",
        "created_level": 0,
        "created_turn": 0,
        "updated_level": 0,
        "updated_turn": 0,
        "revision": 1,
    }:
        raise RuntimeError("selected source goal drifted")

    turn = _selected_turn()
    script = _scripted_lifecycle()
    llm = FakeLLMClient(scripted_responses=script)
    canonical_before = _sha256(WORKSPACE / "world_model.py")
    with _typed_environment():
        agent = TychoNooaAgent(
            llm=llm,
            game_id="ar25",
            workspace_root=WORKSPACE_ROOT,
            available_actions=turn.available_actions,
            tool_trace_path=OUTPUT / "tool-trace.jsonl",
            resume_workspace=True,
        )
        decision = _run(agent, turn)

    canonical_after = _sha256(WORKSPACE / "world_model.py")
    calls = _read_jsonl(OUTPUT / "tool-trace.jsonl")
    goal = _goal(agent)
    review = agent._goal_ledger.latest_review(GOAL_ID)
    if review is None:
        raise RuntimeError("current lifecycle produced no objective review")
    review_data = review.model_dump(mode="json")
    take_actions = [row for row in calls if row.get("name") == "take_action"]
    request = next(row for row in calls if row.get("name") == "request_builder")
    update = next(row for row in calls if row.get("name") == "update_goal")
    acceptance = {
        "source_goal_loaded_exact": source_goal["revision"] == 1,
        "action_blocked_until_objective_review": (
            take_actions[0]["status"] == "error"
            and "request_builder" in str(take_actions[0]["result"])
        ),
        "objective_review_completed": request["status"] == "complete",
        "action_blocked_until_explicit_resolution": (
            take_actions[1]["status"] == "error"
            and "update_goal" in str(take_actions[1]["result"])
        ),
        "goal_revised_before_action": (
            update["status"] == "complete"
            and goal["revision"] == 2
            and goal["status"] == "proposed"
        ),
        "resolution_cites_discriminator": (
            goal["evidence_refs"] == ["level_0/turn_000.txt"]
            and goal["note"] == DISCRIMINATOR
        ),
        "preserved_action_permitted_after_resolution": (
            take_actions[-1]["status"] == "complete"
            and decision == ActionDecision(action="UP")
        ),
        "canonical_model_unchanged": canonical_before == canonical_after,
    }
    result = {
        "schema": "tycho-nooa-selected-goal-checkpoint-replay-v1",
        "source_run": str(SOURCE.parent.relative_to(ROOT)),
        "source_checkpoint_call": CHECKPOINT_CALL,
        "claim_ceiling": (
            "host-contract replay from the exact post-proposal workspace and public "
            "turn; scripted responses test lifecycle enforcement, not model uptake or "
            "gameplay benefit"
        ),
        "source_goal": source_goal,
        "review": review_data,
        "goal_after": goal,
        "decision": decision.model_dump(),
        "tool_sequence": [
            {"name": row["name"], "status": row["status"]} for row in calls
        ],
        "canonical_before_sha256": canonical_before,
        "canonical_after_sha256": canonical_after,
        "acceptance": acceptance,
        "status": "pass" if all(acceptance.values()) else "fail",
        "execution": {
            "provider_model_calls": 0,
            "scripted_llm_responses": len(script),
            "arc_environment_calls": 0,
        },
    }
    _write_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


def _scripted_lifecycle() -> list[LLMResponse]:
    return [
        _tool("take_action", {"action": "UP"}, "preserved-blocked-action"),
        _tool(
            "request_builder",
            {
                "focus": "objective",
                "evidence_refs": ["level_0/turn_000.txt"],
                "candidate_goal_id": GOAL_ID,
            },
            "objective-review",
        ),
        _tool(
            "write_file",
            {
                "path": "notes/world_model.md",
                "content": (
                    "The selected alignment hypothesis remains provisional. Initial "
                    "evidence shows shapes but no transition or terminal consequence."
                ),
            },
            "builder-notes",
        ),
        _tool(
            "write_file",
            {
                "path": "notes/world_model_report.md",
                "content": (
                    "candidate_goal_assessment: inconclusive\n"
                    "candidate_goal_reason: level_0/turn_000.txt shows the cited shapes "
                    "but no action-result or terminal evidence distinguishes alignment "
                    "as the objective from visual similarity\n"
                    "candidate_goal_route: none\n"
                    "recommended_action: UP\n"
                ),
            },
            "builder-report",
        ),
        _prose("objective review complete"),
        _tool("take_action", {"action": "UP"}, "blocked-before-update"),
        _tool(
            "update_goal",
            {
                "goal_id": GOAL_ID,
                "status": "proposed",
                "evidence_refs": ["level_0/turn_000.txt"],
                "note": DISCRIMINATOR,
            },
            "retain-with-discriminator",
        ),
        _tool("take_action", {"action": "UP"}, "preserved-action"),
    ]


def _selected_turn() -> GameTurn:
    first_action = next(
        row
        for row in _read_jsonl(SOURCE / "transcript.jsonl")
        if row.get("type") == "action_result" and row.get("sequence") == 0
    )
    before = first_action["before"]
    return GameTurn(
        grid=before["decision_frame"],
        available_actions=before["available_actions"],
        sequence=0,
        level=0,
        turn_in_level=0,
        levels_completed=before["levels_completed"],
        state=before["state"],
    )


def _source_goal() -> dict:
    goals = json.loads(
        (WORKSPACE / "epistemic/candidate-goals.json").read_text(encoding="utf-8")
    )
    if not isinstance(goals, list) or len(goals) != 1:
        raise RuntimeError("expected exactly one selected source goal")
    return goals[0]


def _goal(agent: TychoNooaAgent) -> dict:
    goal = agent._goal_ledger.get(GOAL_ID)
    if goal is None:
        raise RuntimeError(GOAL_ID)
    return goal.model_dump(mode="json")


def _run(agent: TychoNooaAgent, turn: GameTurn) -> ActionDecision:
    import asyncio

    return asyncio.run(agent.act(turn))


@contextmanager
def _typed_environment():
    names = {"TYCHO_MODE": "orchestrator", "TYCHO_NOOA_TREATMENT": "typed_candidate_goals"}
    prior = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    try:
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _tool(name: str, arguments: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _prose(content: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _materialize(manifest: dict, destination: Path) -> None:
    destination.mkdir(parents=True)
    for directory in manifest["directories"]:
        path = destination / directory["path"]
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(directory["mode"])
    for entry in manifest["files"]:
        path = destination / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_blob(entry["sha256"]).read_bytes())
        path.chmod(entry["mode"])
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"blob commitment mismatch: {entry['path']}")


def _call_path() -> Path:
    return SOURCE / "decision-checkpoints/calls" / CHECKPOINT_CALL


def _blob(sha256: str) -> Path:
    path = SOURCE / "decision-checkpoints/blobs/sha256" / sha256
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
