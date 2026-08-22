"""Matched, one-decision warm restarts from a preserved ARC run.

This is intentionally not called an exact continuation: legacy runs do not
contain NOOA's model-call checkpoint or actor event history. The packet freezes
the public transition and durable workspace that all arms receive, and records
that lifetime limitation in its manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from nooa.unifiedllm import UnifiedLLM

from tycho_nooa import builder
from tycho_nooa.agent import TychoNooaAgent
from tycho_nooa.models import GameTurn

PACKET_SCHEMA = "tycho-nooa-matched-warm-restart-v1"
REPORT_SCHEMA = "tycho-nooa-matched-warm-restart-report-v1"


@dataclass(frozen=True)
class ReplayArm:
    name: str
    mode: str
    treatment: str = "control"


ARMS = (
    ReplayArm("trigger", "trigger"),
    ReplayArm("exact_orchestrator", "orchestrator"),
    ReplayArm(
        "orchestrator_typed_candidate_goals",
        "orchestrator",
        "typed_candidate_goals",
    ),
)
THEORY_TRANSACTION_ARMS = (
    ReplayArm("builder_control", "trigger"),
    ReplayArm("builder_transactional", "trigger", "transactional_builder"),
)


def prepare_legacy_run_packet(
    run_dir: Path,
    destination: Path,
    *,
    arms: tuple[ReplayArm, ...] = ARMS,
) -> dict[str, Any]:
    """Freeze the final observed transition and workspace of a legacy run."""
    if destination.exists():
        raise FileExistsError(destination)
    game_id = _read_object(run_dir / "summary.json")["game_id"]
    workspace = run_dir / "workspace" / game_id
    if not workspace.is_dir():
        raise FileNotFoundError(workspace)
    last, finish = _final_run_records(run_dir / "transcript.jsonl")
    if finish["final_observation"] != last["after"]:
        raise ValueError("finish observation differs from the last action result")
    previous_turn, replay_turn = _replay_turns(workspace, last)
    _write_legacy_packet(
        run_dir,
        destination,
        game_id,
        workspace,
        last,
        finish,
        previous_turn,
        replay_turn,
        arms,
    )
    return validate_packet(destination, arms=arms)


def _final_run_records(transcript: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    events = [json.loads(line) for line in transcript.read_text().splitlines()]
    action_results = [event for event in events if event.get("type") == "action_result"]
    finishes = [event for event in events if event.get("type") == "finish"]
    if not action_results or not finishes:
        raise ValueError("run lacks a final action_result and finish record")
    return action_results[-1], finishes[-1]


def _replay_turns(
    workspace: Path, last: dict[str, Any]
) -> tuple[GameTurn, GameTurn]:
    current_frame = _read_object(workspace / ".current_frame.json")
    previous_turn = _turn_from_observation(
        last["before"],
        sequence=last["sequence"],
        level=last["level"],
        turn_in_level=last["turn_in_level"],
    )
    replay_turn = _turn_from_observation(
        last["after"],
        sequence=last["sequence"] + 1,
        level=last["level"],
        turn_in_level=int(current_frame["turn"]) + 1,
        previous_decision=last["decision"],
    )
    return previous_turn, replay_turn


def _write_legacy_packet(
    run_dir: Path,
    destination: Path,
    game_id: str,
    workspace: Path,
    last: dict[str, Any],
    finish: dict[str, Any],
    previous_turn: GameTurn,
    replay_turn: GameTurn,
    arms: tuple[ReplayArm, ...],
) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        source = temporary / "source"
        shutil.copytree(workspace, source / "workspace")
        _write_json(source / "last-action-result.json", last)
        _write_json(source / "finish.json", finish)
        _write_json(source / "previous-turn.json", previous_turn.model_dump())
        _write_json(source / "replay-turn.json", replay_turn.model_dump())
        manifest = {
            "schema": PACKET_SCHEMA,
            "source_run": str(run_dir.resolve()),
            "game_id": game_id,
            "arms": [arm.__dict__ for arm in arms],
            "workspace_commitment": _tree_commitment(source / "workspace"),
            "transition_commitment": _commitment(last),
            "replay_turn_commitment": _commitment(replay_turn.model_dump()),
            "claim_ceiling": (
                "matched fresh-actor warm restart from durable workspace and public "
                "transition; not exact continuation because the legacy run predates "
                "model-call checkpoints and preserved NOOA actor history"
            ),
        }
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_packet(
    path: Path,
    *,
    arms: tuple[ReplayArm, ...] = ARMS,
) -> dict[str, Any]:
    manifest = _read_object(path / "manifest.json")
    if manifest.get("schema") != PACKET_SCHEMA:
        raise ValueError("unsupported matched replay packet schema")
    source = path / "source"
    checks = {
        "workspace_commitment": _tree_commitment(source / "workspace"),
        "transition_commitment": _commitment(
            _read_object(source / "last-action-result.json")
        ),
        "replay_turn_commitment": _commitment(
            _read_object(source / "replay-turn.json")
        ),
    }
    for field, actual in checks.items():
        if manifest.get(field) != actual:
            raise ValueError(f"stale matched replay packet: {field}")
    if manifest.get("arms") != [arm.__dict__ for arm in arms]:
        raise ValueError("matched replay arms drifted")
    return manifest


async def execute_matched_replay(
    packet: Path,
    destination: Path,
    *,
    llms: Mapping[str, UnifiedLLM],
    arms: tuple[ReplayArm, ...] = ARMS,
) -> dict[str, Any]:
    """Run each frozen arm once; callers own model/cost authorization."""
    manifest = validate_packet(packet, arms=arms)
    if destination.exists():
        raise FileExistsError(destination)
    if set(llms) != {arm.name for arm in arms}:
        raise ValueError("one LLM client is required for every precommitted arm")
    destination.mkdir(parents=True)
    observations = []
    for arm in arms:
        observations.append(
            await _execute_arm(
                packet,
                destination / arm.name,
                arm,
                llms[arm.name],
                arms,
            )
        )
    report = {
        "schema": REPORT_SCHEMA,
        "packet": str(packet.resolve()),
        "packet_commitment": _commitment(manifest),
        "claim_ceiling": manifest["claim_ceiling"],
        "observations": observations,
    }
    _write_json(destination / "report.json", report)
    return report


async def _execute_arm(
    packet: Path,
    arm_dir: Path,
    arm: ReplayArm,
    llm: UnifiedLLM,
    arms: tuple[ReplayArm, ...],
) -> dict[str, Any]:
    manifest = validate_packet(packet, arms=arms)
    workspace_root = arm_dir / "workspace"
    workspace = workspace_root / manifest["game_id"]
    workspace.parent.mkdir(parents=True)
    shutil.copytree(packet / "source" / "workspace", workspace)
    previous_turn = GameTurn.model_validate(
        _read_object(packet / "source" / "previous-turn.json")
    )
    replay_turn = GameTurn.model_validate(
        _read_object(packet / "source" / "replay-turn.json")
    )
    world_model = workspace / "world_model.py"
    before_hash = _file_hash(world_model)
    trace_path = arm_dir / "tool-trace.jsonl"
    with _arm_environment(arm):
        agent = TychoNooaAgent(
            llm=llm,
            game_id=manifest["game_id"],
            workspace_root=workspace_root,
            available_actions=replay_turn.available_actions,
            tool_trace_path=trace_path,
            resume_workspace=True,
            previous_turn=previous_turn,
        )
        decision = await agent.act(replay_turn)
        verifier_text, _ = builder.verify_state(agent._tool_executor)
    goal_snapshot = _read_array(
        workspace / "epistemic" / "candidate-goals.json"
    )
    planner_records = _jsonl_records(arm_dir / "planner-trace.jsonl")
    builder_records = _jsonl_records(arm_dir / "trigger-builder-trace.jsonl")
    result = _arm_result(
        arm,
        decision,
        before_hash,
        world_model,
        goal_snapshot,
        workspace,
        trace_path,
        verifier_text,
        planner_records,
        builder_records,
    )
    _write_json(arm_dir / "result.json", result)
    return result


def _arm_result(
    arm: ReplayArm,
    decision: Any,
    before_hash: str,
    world_model: Path,
    goal_snapshot: list[Any],
    workspace: Path,
    trace_path: Path,
    verifier_text: str,
    planner_records: list[dict[str, Any]],
    builder_records: list[dict[str, Any]],
) -> dict[str, Any]:
    theory_store = workspace.parent / ".theory-transactions" / workspace.name
    return {
        "arm": arm.__dict__,
        "decision": decision.model_dump(),
        "world_model_changed": before_hash != _file_hash(world_model),
        "candidate_goals": goal_snapshot,
        "goal_events": _event_names(
            workspace / "epistemic" / "candidate-goal-events.jsonl"
        ),
        "verifier_result": verifier_text,
        "planner_invocations": len(planner_records),
        "planner_events": [record.get("event", "") for record in planner_records],
        "builder_requests": _builder_requests(builder_records),
        "builder_results": _event_records(builder_records, "builder_result"),
        "actor_uptake": _event_records(builder_records, "actor_uptake"),
        "builder_events": [record.get("event", "") for record in builder_records],
        "theory_publication": _read_optional_object(theory_store / "canonical.json"),
        "theory_ledger_events": _event_names(theory_store / "ledger.jsonl"),
        "tool_names": _tool_names(trace_path),
        "workspace_commitment_after": _tree_commitment(workspace),
    }


def _builder_requests(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if _is_builder_request(record)]


def _is_builder_request(record: dict[str, Any]) -> bool:
    if record.get("event") == "actor_request":
        return True
    return record.get("event") == "mechanical_trigger" and record.get("fired") is True


def _event_records(
    records: list[dict[str, Any]], event: str
) -> list[dict[str, Any]]:
    return [record for record in records if record.get("event") == event]


def _turn_from_observation(
    observation: Mapping[str, Any],
    *,
    sequence: int,
    level: int,
    turn_in_level: int,
    previous_decision: Mapping[str, Any] | None = None,
) -> GameTurn:
    previous = previous_decision or {}
    grid = observation.get("decision_frame", observation.get("grid"))
    return GameTurn(
        grid=grid,
        available_actions=observation["available_actions"],
        sequence=sequence,
        level=level,
        turn_in_level=turn_in_level,
        levels_completed=observation["levels_completed"],
        state=observation["state"],
        previous_action=str(previous.get("action", "")),
        previous_row=previous.get("row"),
        previous_col=previous.get("col"),
    )


@contextmanager
def _arm_environment(arm: ReplayArm):
    prior = {name: os.environ.get(name) for name in ("TYCHO_MODE", "TYCHO_NOOA_TREATMENT")}
    os.environ["TYCHO_MODE"] = arm.mode
    os.environ["TYCHO_NOOA_TREATMENT"] = arm.treatment
    try:
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _tool_names(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [json.loads(line)["name"] for line in path.read_text().splitlines()]


def _event_names(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [json.loads(line).get("event", "") for line in path.read_text().splitlines()]


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _read_array(path: Path) -> list[Any]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise TypeError(f"expected JSON array: {path}")
    return value


def _tree_commitment(root: Path) -> str:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _file_hash(path),
                }
            )
    return _commitment(records)


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commitment(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_optional_object(path: Path) -> dict[str, Any] | None:
    return _read_object(path) if path.is_file() else None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
