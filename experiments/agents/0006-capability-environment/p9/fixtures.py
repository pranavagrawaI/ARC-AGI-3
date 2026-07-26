"""Deterministic stand-ins for the provider and the environment.

``ScriptedChatClient`` returns a fixed sequence of assistant messages while
producing the same transport evidence a real response carries, so ``ProcessLog``
commitment checks exercise their real paths. ``RecordedEnvironment`` replays
public frames from a ``recordings/*.jsonl`` capture.

Neither needs a provider, an API key, or the ARC endpoint. The recorded frames
are public observation data only; the environment ignores which action was
chosen, so a fixture episode measures the harness, never gameplay.
"""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from llm_repl_agent.client import ChatResponse, ModelRequestError, serialize_request_payload


# -- model ---------------------------------------------------------------------


def tool_turn(code: str, *, call_id: str | None = None) -> dict[str, Any]:
    """A scripted assistant turn that calls the python tool."""

    return {
        "kind": "tool",
        "code": code,
        "call_id": call_id,
        "raw_arguments": None,
    }


def malformed_tool_turn(raw_arguments: str, *, call_id: str | None = None) -> dict[str, Any]:
    """A scripted tool call whose arguments the agent cannot parse."""

    return {
        "kind": "tool",
        "code": None,
        "call_id": call_id,
        "raw_arguments": raw_arguments,
    }


def action_turn(action: int, *, data: Mapping[str, Any] | None = None, reason: str = "") -> dict[str, Any]:
    return {
        "kind": "content",
        "content": json.dumps({"action": action, "data": data, "reason": reason}),
    }


def transport_failure(message: str, *, retryable: bool, http_status: int | None = None) -> dict[str, Any]:
    return {
        "kind": "transport_failure",
        "message": message,
        "retryable": retryable,
        "http_status": http_status,
    }


@dataclass
class Attempt:
    index: int
    outcome: str
    payload: dict[str, Any]
    detail: str | None = None


class ScriptedFixtureExhausted(RuntimeError):
    pass


class ScriptedChatClient:
    """Replay scripted assistant turns with real request/response commitments."""

    def __init__(self, script: Sequence[Mapping[str, Any]], *, model: str = "fixture-model") -> None:
        self.model = model
        self._script = [dict(entry) for entry in script]
        self._cursor = 0
        self.attempts: list[Attempt] = []
        self.requests: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return len(self._script) - self._cursor

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "tools": [dict(tool) for tool in tools],
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 2048,
            "seed": 0,
        }
        if self._cursor >= len(self._script):
            raise ScriptedFixtureExhausted(
                f"the fixture script ran out after {self._cursor} model calls"
            )
        entry = self._script[self._cursor]
        self._cursor += 1
        index = len(self.attempts) + 1
        if entry["kind"] == "transport_failure":
            self.attempts.append(
                Attempt(index=index, outcome="transport_failure", payload=payload, detail=str(entry["message"]))
            )
            payload_bytes = serialize_request_payload(payload)
            raise ModelRequestError(
                str(entry["message"]),
                request_payload=payload,
                request_payload_sha256=sha256(payload_bytes).hexdigest(),
                http_status=entry.get("http_status"),
                retryable=bool(entry["retryable"]),
            )
        self.attempts.append(Attempt(index=index, outcome="ok", payload=payload))
        self.requests.append(payload)
        return _scripted_response(payload, entry, call_index=index)

    def request_payloads(self) -> list[dict[str, Any]]:
        return [json.loads(json.dumps(payload)) for payload in self.requests]


def _scripted_response(
    payload: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    call_index: int,
) -> ChatResponse:
    if entry["kind"] == "tool":
        arguments = (
            entry["raw_arguments"]
            if entry.get("raw_arguments") is not None
            else json.dumps({"code": entry["code"]})
        )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": entry.get("call_id") or f"fixture-call-{call_index}",
                    "type": "function",
                    "function": {"name": "python", "arguments": arguments},
                }
            ],
        }
    else:
        message = {"role": "assistant", "content": entry["content"]}

    body = {
        "id": f"fixture-{call_index:04d}",
        "object": "chat.completion",
        "model": payload["model"],
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    body_bytes = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_bytes = serialize_request_payload(payload)
    return ChatResponse(
        message=message,
        usage=dict(body["usage"]),
        response_id=str(body["id"]),
        request_payload=json.loads(payload_bytes.decode("utf-8")),
        request_payload_sha256=sha256(payload_bytes).hexdigest(),
        request_payload_encoding="json-utf8-sort-keys-compact-v1",
        response_metadata={"object": body["object"], "model": body["model"]},
        response_body=body,
        response_body_sha256=sha256(body_bytes).hexdigest(),
        response_body_raw=body_bytes.decode("utf-8"),
        response_body_base64=b64encode(body_bytes).decode("ascii"),
    )


# -- environment ---------------------------------------------------------------


@dataclass(frozen=True)
class RecordedFrame:
    frame: list[list[list[int]]]
    state: str
    levels_completed: int
    available_actions: tuple[int, ...]
    full_reset: bool


@dataclass
class RecordedEnvironment:
    """Replay captured public frames in order, ignoring which action was chosen."""

    frames: tuple[RecordedFrame, ...]
    _cursor: int = field(default=0, repr=False)
    executed: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def reset(self) -> RecordedFrame:
        self._cursor = 0
        self.executed.clear()
        return self.frames[0]

    def step(
        self,
        action: Any,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> RecordedFrame:
        self.executed.append({"action": getattr(action, "name", str(action)), "data": data})
        self._cursor = min(self._cursor + 1, len(self.frames) - 1)
        return self.frames[self._cursor]


def load_recorded_frames(
    path: Path,
    *,
    count: int,
    crop: int | None = None,
    terminal_last: bool = True,
) -> tuple[RecordedFrame, ...]:
    """Load the first ``count`` public frames from a recording capture.

    ``crop`` keeps a top-left square of each grid so a fixture episode stays
    small; it changes the observation size, never its structure or semantics.
    ``terminal_last`` marks the final frame ``WIN`` so ``run_episode`` stops on
    the recording rather than replaying its last frame forever.
    """

    if count < 1:
        raise ValueError("count must be positive")
    frames: list[RecordedFrame] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            data = record.get("data", record)
            if "frame" not in data:
                continue
            grid = np.asarray(data["frame"], dtype=int)
            if grid.ndim != 3:
                raise ValueError(f"unexpected recorded frame shape: {grid.shape}")
            if crop is not None:
                grid = grid[:, :crop, :crop]
            frames.append(
                RecordedFrame(
                    frame=grid.tolist(),
                    state=str(data.get("state", "NOT_FINISHED")),
                    levels_completed=int(data.get("levels_completed", 0)),
                    available_actions=tuple(int(item) for item in data.get("available_actions", ())),
                    full_reset=bool(data.get("full_reset", False)),
                )
            )
            if len(frames) == count:
                break
    if len(frames) < count:
        raise ValueError(f"recording {path} holds {len(frames)} frames, needed {count}")
    if terminal_last:
        frames[-1] = RecordedFrame(
            frame=frames[-1].frame,
            state="WIN",
            levels_completed=frames[-1].levels_completed,
            available_actions=(),
            full_reset=frames[-1].full_reset,
        )
    return tuple(frames)


def default_recording(repo: Path, *, minimum_frames: int = 1) -> Path:
    """Pick the first recorded ls20 capture long enough for the fixture."""

    candidates = sorted((repo / "recordings").glob("*/ls20-*.jsonl"))
    for candidate in candidates:
        with candidate.open("r", encoding="utf-8") as stream:
            frames = sum(1 for line in stream if '"frame"' in line)
        if frames >= minimum_frames:
            return candidate
    raise FileNotFoundError(
        f"no ls20 recording holds {minimum_frames} frames (checked {len(candidates)})"
    )
