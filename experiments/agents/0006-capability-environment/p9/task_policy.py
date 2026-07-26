"""A reactive fixture policy with a real task, and the host's ground truth for it.

Every earlier prototype drives a *script*: a fixed sequence that ignores what the
harness returns, so the harness can only fail an assertion, never a task. This
policy instead reads the conversation it is given, decides from the last tool
result, and pursues one objective across the whole episode:

    maintain ``index.json`` mapping every observation step seen so far to the
    sha256 of that observation's bytes.

Getting that right at the end requires the workspace to persist, every past
observation to remain reachable, tool results to be paired with their calls, and
failures to be recoverable. The host checks the finished index against digests it
computed itself. One boolean, dependent on nearly every harness property, and it
restates none of the arm contract.

Ground truth is re-derived by the host from the recorded frames using the public
observation transforms — not read back from any component under test, and not
from the workspace the model can write to.
"""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from llm_repl_agent.client import ChatResponse, ModelRequestError, serialize_request_payload


INDEX_CODE = """\
import hashlib, json
from pathlib import Path
index_path = Path('index.json')
index = json.loads(index_path.read_text()) if index_path.exists() else {}
found = {}
root = Path('observations')
for path in sorted(root.glob('step-*.json')):
    found.setdefault(path.stem.split('-')[1], path)
for step, path in sorted(found.items()):
    if step not in index:
        index[step] = hashlib.sha256(path.read_bytes()).hexdigest()
index_path.write_text(json.dumps(index, sort_keys=True))
print('INDEXED', len(index), 'of', len(found))
"""

TIMEOUT_PROBE = "while True:\n    pass\n"
OVERSIZED_PROBE = "print('x' * 200000)\n"
MALFORMED_ARGUMENTS = "{not json"


@dataclass
class TaskPolicy:
    """Decide from the last tool result, not from a fixed script."""

    probe_at_step: dict[int, str] = field(default_factory=dict)
    malformed_at_step: set[int] = field(default_factory=set)
    max_repairs_per_step: int = 2
    model: str = "task-policy"

    transport_failure_at_call: int | None = None
    transport_failure_retryable: bool = True

    calls: int = field(default=0, init=False)
    requests: list[dict[str, Any]] = field(default_factory=list, init=False)
    tool_failures_seen: list[str] = field(default_factory=list, init=False)
    probes_issued: list[str] = field(default_factory=list, init=False)

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        self.calls += 1
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "tools": [dict(tool) for tool in tools],
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 2048,
            "seed": 0,
        }
        if self.calls == self.transport_failure_at_call:
            self.transport_failure_at_call = None
            raise ModelRequestError(
                "simulated provider failure",
                request_payload=payload,
                request_payload_sha256=sha256(serialize_request_payload(payload)).hexdigest(),
                http_status=503 if self.transport_failure_retryable else 400,
                retryable=self.transport_failure_retryable,
            )
        self.requests.append(payload)

        step, results, tool_available = _read_turn(messages)
        decision = self._decide(step, results, tool_available)
        return _respond(payload, decision, self.calls)

    def _decide(
        self,
        step: int,
        results: list[dict[str, Any]],
        tool_available: bool,
    ) -> dict[str, Any]:
        if not tool_available:
            return {"action": 1, "reason": "tool unavailable on the final round"}

        probe = self.probe_at_step.get(step)
        if probe is not None and probe not in self.probes_issued:
            self.probes_issued.append(probe)
            return {"code": probe}
        if step in self.malformed_at_step:
            self.malformed_at_step.discard(step)
            return {"raw_arguments": MALFORMED_ARGUMENTS}

        indexed = [item for item in results if "INDEXED" in item.get("stdout", "")]
        if indexed:
            return {"action": 1, "reason": "index current"}

        failures = [item for item in results if not item.get("ok")]
        for failure in failures:
            marker = f"step-{step}:{failure.get('traceback') or failure.get('stderr')}"
            if marker not in self.tool_failures_seen:
                self.tool_failures_seen.append(marker)
        if len(failures) > self.max_repairs_per_step:
            return {"action": 1, "reason": "giving up on the index this step"}
        return {"code": INDEX_CODE}


def _read_turn(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[int, list[dict[str, Any]], bool]:
    """Recover the current step, this turn's tool results, and tool availability."""

    step = 0
    boundary = 0
    for index, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "text":
                    try:
                        step = int(json.loads(part["text"])["step"])
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
            boundary = index
    results: list[dict[str, Any]] = []
    tool_available = True
    for message in list(messages)[boundary:]:
        if message.get("role") == "tool":
            try:
                results.append(json.loads(str(message.get("content"))))
            except json.JSONDecodeError:
                results.append({"ok": False, "stdout": "", "stderr": "unparsable"})
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            if "final deliberation round" in message["content"]:
                tool_available = False
    return step, results, tool_available


def _respond(payload: Mapping[str, Any], decision: Mapping[str, Any], call: int) -> ChatResponse:
    if "action" in decision:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": json.dumps(
                {"action": decision["action"], "data": None, "reason": decision["reason"]}
            ),
        }
    else:
        arguments = (
            decision["raw_arguments"]
            if "raw_arguments" in decision
            else json.dumps({"code": decision["code"]})
        )
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"task-call-{call}",
                    "type": "function",
                    "function": {"name": "python", "arguments": arguments},
                }
            ],
        }
    body = {
        "id": f"task-{call:04d}",
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


# -- the host's ground truth ---------------------------------------------------


def expected_index(frames, *, colors, through_step: int) -> dict[str, str]:
    """Digests the host re-derives from the frames, independent of the harness."""

    import numpy as np
    from experiments.harness import Observation
    from llm_repl_agent.observations import observation_metadata

    expected: dict[str, str] = {}
    for step, frame in enumerate(frames):
        if step > through_step:
            break
        pixels = np.array(frame.frame, copy=True)
        pixels.setflags(write=False)
        observation = Observation(
            pixels=pixels,
            available_actions=tuple(frame.available_actions),
            levels_completed=frame.levels_completed,
            state=frame.state,
            full_reset=frame.full_reset,
        )
        record = {
            **observation_metadata(observation),
            "pixels": colors.apply(observation.pixels).tolist(),
            "step": step,
        }
        encoded = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        expected[f"{step:05d}"] = sha256(encoded.encode("utf-8")).hexdigest()
    return expected


def grade_index(index_text: str, expected: Mapping[str, str]) -> dict[str, Any]:
    """Compare the model-authored index against the host's digests."""

    try:
        index = json.loads(index_text) if index_text else {}
    except json.JSONDecodeError:
        return {"complete": False, "reason": "index.json is not valid JSON"}
    missing = sorted(set(expected) - set(index))
    wrong = sorted(step for step in set(expected) & set(index) if index[step] != expected[step])
    invented = sorted(set(index) - set(expected))
    return {
        "complete": not missing and not wrong,
        "entries": len(index),
        "expected_entries": len(expected),
        "missing": missing,
        "wrong": wrong,
        "invented_beyond_expected": invented,
    }
