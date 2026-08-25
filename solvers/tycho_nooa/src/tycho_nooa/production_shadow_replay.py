"""Off-policy long-trajectory shadows through one persistent production Actor."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from typing import Any

from nooa.unifiedllm import UnifiedLLM

from tycho_nooa.agent import TychoNooaAgent
from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.modes import MODE_ENV, served_mode
from tycho_nooa.treatments import TREATMENT_ENV, treatment_from_env

REPORT_SCHEMA = "tycho-nooa-production-shadow-replay-v1"


async def execute_production_shadow_replay(
    *,
    transcript: Path,
    destination: Path,
    llm: UnifiedLLM,
    game_id: str,
    mode: str,
    treatment: str,
) -> dict[str, Any]:
    """Replay recorded observations while never executing proposed decisions.

    The recorded trajectory remains authoritative after a shadow decision
    diverges.  Later decisions are therefore off-policy observations, not
    claims about the unobserved counterfactual world the proposed action would
    have created.
    """

    if destination.exists():
        raise FileExistsError(destination)
    events = _action_results(transcript)
    _validate_causal_chain(events)
    mode_policy = served_mode(mode)
    treatment_policy = treatment_from_env(
        mode_policy,
        {TREATMENT_ENV: treatment},
    )
    destination.mkdir(parents=True)
    trace_path = destination / "tool-trace.jsonl"
    workspace_root = destination / "workspace"
    first = events[0]
    first_turn = _game_turn(first, previous_decision=None)
    trajectory: list[dict[str, Any]] = []
    first_divergence: int | None = None

    with _runtime_environment(mode_policy.name, treatment_policy.name):
        agent = TychoNooaAgent(
            llm=llm,
            game_id=game_id,
            workspace_root=workspace_root,
            available_actions=first_turn.available_actions,
            tool_trace_path=trace_path,
            trigger_builder=False,
        )
        previous_decision: Mapping[str, Any] | None = None
        for event in events:
            turn = _game_turn(event, previous_decision=previous_decision)
            calls_before = _checkpoint_count(destination)
            llm_before = len(_jsonl_records(destination / "llm-trace.jsonl"))
            decision = await agent.act(turn)
            calls_after = _checkpoint_count(destination)
            llm_records = _jsonl_records(destination / "llm-trace.jsonl")
            historical = ActionDecision.model_validate(event["decision"])
            agrees = decision == historical
            if not agrees and first_divergence is None:
                first_divergence = int(event["sequence"])
            call_indices = range(calls_before + 1, calls_after + 1)
            request_sizes = [
                _request_bytes(destination, call_index) for call_index in call_indices
            ]
            usage = llm_records[llm_before:]
            trajectory.append(
                {
                    "sequence": int(event["sequence"]),
                    "level": int(event["level"]),
                    "turn_in_level": int(event["turn_in_level"]),
                    "decision": decision.model_dump(mode="json"),
                    "historical_decision": historical.model_dump(mode="json"),
                    "historical_action_agreement": agrees,
                    "after_action_divergence": first_divergence is not None,
                    "provider_calls": len(usage),
                    "input_tokens": sum(
                        int(row.get("prompt_tokens") or 0) for row in usage
                    ),
                    "output_tokens": sum(
                        int(row.get("completion_tokens") or 0) for row in usage
                    ),
                    "provider_request_bytes": request_sizes[-1] if request_sizes else 0,
                    "provider_request_bytes_total": sum(request_sizes),
                    "available_exact_evidence_bytes": _evidence_bytes(
                        agent.workspace_dir
                    ),
                }
            )
            previous_decision = event["decision"]

    report = {
        "schema": REPORT_SCHEMA,
        "controller": "tycho_nooa.agent.TychoNooaAgent",
        "actor_instances": 1,
        "environment_actions_executed": 0,
        "turns": len(trajectory),
        "historical_action_agreement": sum(
            row["historical_action_agreement"] for row in trajectory
        ),
        "first_action_divergence_sequence": first_divergence,
        "claim_ceiling": (
            "off-policy shadow over recorded public observations; the production "
            "Actor is persistent and its decisions are validated but never executed; "
            "after the first disagreement, later observations belong to the historical "
            "trajectory rather than the Actor's unobserved counterfactual trajectory"
        ),
        "trajectory": trajectory,
    }
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _action_results(transcript: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in transcript.read_text().splitlines()]
    actions = [record for record in records if record.get("type") == "action_result"]
    if not actions:
        raise ValueError("transcript has no action_result records")
    return actions


def _validate_causal_chain(events: Sequence[Mapping[str, Any]]) -> None:
    for previous, current in pairwise(events):
        if int(current["sequence"]) != int(previous["sequence"]) + 1:
            raise ValueError("action_result sequence is not contiguous")
        if _decision_grid(previous["after"]) != _decision_grid(current["before"]):
            raise ValueError("recorded after observation does not match next before")


def _game_turn(
    event: Mapping[str, Any],
    *,
    previous_decision: Mapping[str, Any] | None,
) -> GameTurn:
    observation = event["before"]
    previous = previous_decision or {}
    return GameTurn(
        grid=_decision_grid(observation),
        available_actions=observation["available_actions"],
        sequence=int(event["sequence"]),
        level=int(event["level"]),
        turn_in_level=int(event["turn_in_level"]),
        levels_completed=int(observation.get("levels_completed", 0)),
        state=str(observation.get("state", "NOT_FINISHED")),
        previous_action=str(previous.get("action", "")),
        previous_row=previous.get("row"),
        previous_col=previous.get("col"),
    )


def _decision_grid(observation: Mapping[str, Any]) -> list[list[int]]:
    value = observation.get("decision_frame", observation.get("grid"))
    if not isinstance(value, list):
        raise TypeError("observation lacks a public decision grid")
    return value


def _checkpoint_count(destination: Path) -> int:
    root = destination / "decision-checkpoints" / "calls"
    return sum(path.is_dir() for path in root.iterdir()) if root.is_dir() else 0


def _request_bytes(destination: Path, call_index: int) -> int:
    path = (
        destination
        / "decision-checkpoints"
        / "calls"
        / f"{call_index:06d}"
        / "request.json"
    )
    request = json.loads(path.read_text())
    return len(
        json.dumps(
            request["request_body"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def _evidence_bytes(workspace: Path) -> int:
    total = 0
    for path in workspace.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".txt"}:
            continue
        if path.name == ".current_frame.json" or any(
            part.startswith("level_") for part in path.relative_to(workspace).parts
        ):
            total += path.stat().st_size
    return total


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


@contextmanager
def _runtime_environment(mode: str, treatment: str):
    prior = {name: os.environ.get(name) for name in (MODE_ENV, TREATMENT_ENV)}
    os.environ[MODE_ENV] = mode
    os.environ[TREATMENT_ENV] = treatment
    try:
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
