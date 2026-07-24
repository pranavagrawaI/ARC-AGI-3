"""Execute an offline replay through a prediction policy and persist evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol

from .compliance import assess_reconstruction_compliance
from .replay import SequentialReplay


_REQUIRED_ENTRYPOINTS = {"parse_observation", "predict", "update"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PolicyPrediction:
    """A policy's prospective result before the next observation is revealed."""

    prediction: Mapping[str, Any]
    candidate_predictions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    artifact_sha256: str | None = None
    artifact_entrypoints: tuple[str, ...] = ()
    artifact_loaded: bool = False


@dataclass(frozen=True)
class PolicyUpdate:
    """Evidence that the policy processed the newly revealed transition."""

    ok: bool
    detail: Mapping[str, Any] = field(default_factory=dict)


class ReplayPolicy(Protocol):
    def predict(
        self,
        *,
        step: int,
        observation: dict[str, Any],
        recorded_action: int,
        legal_actions: tuple[int, ...],
    ) -> PolicyPrediction: ...

    def update(
        self,
        *,
        step: int,
        recorded_action: int,
        observation: dict[str, Any],
    ) -> PolicyUpdate: ...


def execute_replay(
    *,
    replay: SequentialReplay,
    policy: ReplayPolicy,
    condition: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one optional or required replay with fail-closed reveal ordering."""

    if condition not in {"optional", "required"}:
        raise ValueError("condition must be 'optional' or 'required'")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir must not contain existing files")
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    event_index = 0
    events: list[dict[str, Any]] = []
    revealed_steps: list[int] = []

    def write_event(event_type: str, *, step: int, **payload: Any) -> None:
        nonlocal event_index
        event_index += 1
        event = {
            "event_index": event_index,
            "type": event_type,
            "step": step,
            **payload,
        }
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":"), sort_keys=True))
            stream.write("\n")
        events.append(event)

    try:
        for step in replay.eligible_steps:
            observation = replay.current_observation
            available = observation.get("available_actions")
            if not isinstance(available, list) or any(
                isinstance(action, bool) or not isinstance(action, int)
                for action in available
            ):
                raise ValueError(f"observation at step {step - 1} has invalid actions")
            legal_actions = tuple(available)
            recorded_action = replay.recorded_action
            prospective = policy.predict(
                step=step,
                observation=observation,
                recorded_action=recorded_action,
                legal_actions=legal_actions,
            )
            if not isinstance(prospective, PolicyPrediction):
                raise TypeError("policy.predict must return PolicyPrediction")

            if prospective.artifact_sha256 is not None or prospective.artifact_loaded:
                write_event(
                    "artifact_loaded",
                    step=step,
                    artifact_sha256=prospective.artifact_sha256,
                    entrypoints=list(prospective.artifact_entrypoints),
                    ok=prospective.artifact_loaded,
                )
            write_event(
                "candidate_predictions",
                step=step,
                predictions=dict(prospective.candidate_predictions),
            )

            artifact_valid = (
                prospective.artifact_loaded
                and isinstance(prospective.artifact_sha256, str)
                and _SHA256.fullmatch(prospective.artifact_sha256) is not None
                and _REQUIRED_ENTRYPOINTS.issubset(prospective.artifact_entrypoints)
            )
            if condition == "required" and not artifact_valid:
                write_event(
                    "contract_violation",
                    step=step,
                    reason="missing_or_invalid_required_reconstruction_artifact",
                )
                raise RuntimeError(
                    "required reconstruction artifact was not loaded with the frozen entrypoints"
                )

            write_event(
                "prediction_committed",
                step=step,
                artifact_sha256=prospective.artifact_sha256,
                prediction=dict(prospective.prediction),
            )
            revealed = replay.commit_and_reveal(prospective.prediction)
            write_event(
                "observation_revealed",
                step=step,
                observation_sha256=_canonical_sha256(revealed.observation),
            )
            revealed_steps.append(step)
            update = policy.update(
                step=step,
                recorded_action=recorded_action,
                observation=dict(revealed.observation),
            )
            if not isinstance(update, PolicyUpdate):
                raise TypeError("policy.update must return PolicyUpdate")
            write_event(
                "model_updated",
                step=step,
                ok=update.ok,
                detail=dict(update.detail),
            )
    except BaseException as error:
        _write_json(
            output_dir / "manifest.json",
            {
                "schema": "reconstruction-uptake-run-manifest-v1",
                "status": "failed",
                "condition": condition,
                "eligible_steps": list(replay.eligible_steps),
                "revealed_steps": revealed_steps,
                "source_trace_sha256": replay.source_sha256,
                "model_calls": getattr(policy, "model_calls", None),
                "python_calls": getattr(policy, "python_calls", None),
                "error": repr(error),
            },
        )
        raise

    compliance = assess_reconstruction_compliance(
        events,
        eligible_steps=replay.eligible_steps,
    )
    result = {
        "schema": "reconstruction-uptake-run-result-v1",
        "status": "complete",
        "condition": condition,
        "eligible_steps": list(replay.eligible_steps),
        "revealed_steps": revealed_steps,
        "source_trace_sha256": replay.source_sha256,
        "model_calls": getattr(policy, "model_calls", None),
        "python_calls": getattr(policy, "python_calls", None),
        "provider_usage": getattr(policy, "provider_usage", None),
        "compliance": compliance,
    }
    _write_json(output_dir / "compliance.json", compliance)
    _write_json(output_dir / "manifest.json", result)
    return result


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
