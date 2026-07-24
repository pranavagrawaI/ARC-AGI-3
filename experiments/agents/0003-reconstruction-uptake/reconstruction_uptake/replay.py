"""Sequential, commitment-before-reveal access to an admissible trace."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class _ReplayTransition:
    step: int
    action: int
    action_data: Mapping[str, Any] | None
    before: Mapping[str, Any]
    after: Mapping[str, Any]


@dataclass(frozen=True)
class RevealedTransition:
    """The observation revealed after a prospective prediction commitment."""

    step: int
    recorded_action: int
    prediction: Mapping[str, Any]
    observation: Mapping[str, Any]


class SequentialReplay:
    """Expose one recorded action at a time and conceal its result until commitment."""

    def __init__(
        self,
        *,
        source_path: Path,
        source_sha256: str,
        transitions: tuple[_ReplayTransition, ...],
    ) -> None:
        if not transitions:
            raise ValueError("replay window must contain at least one transition")
        self.source_path = source_path
        self.source_sha256 = source_sha256
        self._transitions = transitions
        self._index = 0

    @classmethod
    def from_trace(
        cls,
        path: Path,
        *,
        first_step: int,
        last_step: int,
    ) -> "SequentialReplay":
        if first_step < 1 or last_step < first_step:
            raise ValueError("replay window must satisfy 1 <= first_step <= last_step")
        payload = path.read_bytes()
        records = [
            json.loads(line)
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        ]
        if not records or records[0].get("type") != "observation" or records[0].get("step") != 0:
            raise ValueError("trace must begin with the step-0 observation")

        initial = _observation(records[0], step=0)
        transitions: list[_ReplayTransition] = []
        previous = initial
        expected_step = 1
        for record in records[1:]:
            record_type = record.get("type")
            if record_type == "end":
                break
            if record_type != "transition":
                raise ValueError(f"unsupported trace record type before end: {record_type!r}")
            step = record.get("step")
            if step != expected_step:
                raise ValueError(
                    "trace transitions must be contiguous; "
                    f"expected step {expected_step}, found {step!r}"
                )
            action = record.get("action")
            if isinstance(action, bool) or not isinstance(action, int):
                raise ValueError(f"trace step {step} has no integer action")
            raw_data = record.get("action_data")
            if raw_data is not None and not isinstance(raw_data, Mapping):
                raise ValueError(f"trace step {step} action_data must be an object or null")
            raw_after = record.get("observation")
            if not isinstance(raw_after, Mapping):
                raise ValueError(f"trace step {step} has no observation object")
            after = _observation(raw_after, step=step)
            transitions.append(
                _ReplayTransition(
                    step=step,
                    action=action,
                    action_data=dict(raw_data) if isinstance(raw_data, Mapping) else None,
                    before=previous,
                    after=after,
                )
            )
            previous = after
            expected_step += 1

        available_steps = {transition.step for transition in transitions}
        requested = set(range(first_step, last_step + 1))
        missing = sorted(requested - available_steps)
        if missing:
            raise ValueError(f"trace does not cover requested replay steps: {missing}")
        selected = tuple(
            transition
            for transition in transitions
            if first_step <= transition.step <= last_step
        )
        return cls(
            source_path=path.resolve(),
            source_sha256=sha256(payload).hexdigest(),
            transitions=selected,
        )

    @property
    def eligible_steps(self) -> tuple[int, ...]:
        return tuple(transition.step for transition in self._transitions)

    @property
    def finished(self) -> bool:
        return self._index >= len(self._transitions)

    @property
    def current_observation(self) -> dict[str, Any]:
        transition = self._current()
        return deepcopy(dict(transition.before))

    @property
    def recorded_action(self) -> int:
        return self._current().action

    @property
    def recorded_action_data(self) -> dict[str, Any] | None:
        data = self._current().action_data
        return deepcopy(dict(data)) if data is not None else None

    def reveal(self) -> None:
        raise RuntimeError("next observation requires a prediction commitment")

    def commit_and_reveal(self, prediction: Mapping[str, Any]) -> RevealedTransition:
        if not isinstance(prediction, Mapping):
            raise TypeError("prediction commitment must be an object")
        transition = self._current()
        self._index += 1
        return RevealedTransition(
            step=transition.step,
            recorded_action=transition.action,
            prediction=deepcopy(dict(prediction)),
            observation=deepcopy(dict(transition.after)),
        )

    def plan_records(self) -> tuple[dict[str, Any], ...]:
        """Return the public replay schedule without any post-action observation."""

        return tuple(
            {
                "step": transition.step,
                "recorded_action": transition.action,
                "recorded_action_data": deepcopy(
                    dict(transition.action_data)
                    if transition.action_data is not None
                    else None
                ),
                "observation_sha256": _canonical_sha256(transition.before),
                "available_actions": list(transition.before["available_actions"]),
            }
            for transition in self._transitions
        )

    def _current(self) -> _ReplayTransition:
        if self.finished:
            raise RuntimeError("replay is complete")
        return self._transitions[self._index]


def _observation(raw: Mapping[str, Any], *, step: int) -> dict[str, Any]:
    required = {
        "pixels",
        "available_actions",
        "levels_completed",
        "state",
        "full_reset",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"observation at step {step} is missing fields: {missing}")
    return {
        "step": step,
        **{key: deepcopy(raw[key]) for key in sorted(required)},
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()
