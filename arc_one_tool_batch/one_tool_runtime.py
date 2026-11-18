"""One reasoning model, one do_actions tool, concurrent independent ARC games.

The model lives in a separately launched vLLM process. This module owns no CUDA
objects. The official Swarm can call play_game from its existing game threads.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import httpx


class StopRun(RuntimeError):
    """Stop without manufacturing or retrying an environment action."""


class Deadline(StopRun):
    pass


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class Config:
    base_url: str = "http://127.0.0.1:8000"
    model: str = "arc-model"
    concurrency: int = 16
    max_new_tokens: int = 16384
    max_tool_actions: int = 32
    max_game_actions: int = 10000
    max_decisions: int = 10000
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    chat_template_kwargs: str = '{"enable_thinking":true}'
    deadline: float = float("inf")  # time.monotonic clock, shared by all games
    log_dir: str = "one_tool_logs"
    trace_full_requests: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        # The launcher requires this timestamp, set in the notebook's first cell.
        start = float(os.environ["ARC_RUN_STARTED_MONOTONIC"])
        budget = float(os.getenv("ARC_BUDGET_SECONDS", str(9 * 3600)))
        reserve = float(os.getenv("ARC_SHUTDOWN_RESERVE_SECONDS", "300"))
        if not 0 < budget <= 9 * 3600 or not 0 < reserve < budget:
            raise ValueError("Require 0 < reserve < budget <= 9 hours")
        config = cls(
            base_url=os.getenv("LLM_BASE_URL", cls.base_url).rstrip("/"),
            model=os.getenv("LLM_MODEL", cls.model),
            concurrency=int(os.getenv("LLM_CONCURRENCY", "16")),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "16384")),
            max_tool_actions=int(os.getenv("MAX_TOOL_ACTIONS", "32")),
            max_game_actions=int(os.getenv("MAX_GAME_ACTIONS", "10000")),
            max_decisions=int(os.getenv("MAX_DECISIONS", "10000")),
            temperature=float(os.getenv("TEMPERATURE", "0.6")),
            top_p=float(os.getenv("TOP_P", "0.95")),
            top_k=int(os.getenv("TOP_K", "20")),
            chat_template_kwargs=os.getenv("CHAT_TEMPLATE_KWARGS", cls.chat_template_kwargs),
            deadline=start + budget - reserve,
            log_dir=os.getenv("ONE_TOOL_LOG_DIR", cls.log_dir),
            trace_full_requests=os.getenv("TRACE_FULL_REQUESTS", "0") == "1",
        )
        for name in ("concurrency", "max_new_tokens", "max_tool_actions",
                     "max_game_actions", "max_decisions"):
            if getattr(config, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not isinstance(json.loads(config.chat_template_kwargs), dict):
            raise ValueError("CHAT_TEMPLATE_KWARGS must be a JSON object")
        return config

    def remaining(self) -> float:
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            raise Deadline("Shared play deadline reached")
        return seconds


def tool_schema(limit: int) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "do_actions",
            "description": (
                f"Execute an ordered sequence of at most {limit} primitive actions "
                "after expanding repeat. Returns every action result and every frame. "
                "Stops on a level change, GAME_OVER, WIN, RESET, an unavailable "
                "action, or the run deadline. Unexecuted actions are discarded."
            ),
            "parameters": {
                "type": "object", "additionalProperties": False,
                "required": ["actions"],
                "properties": {"actions": {
                    "type": "array", "minItems": 1, "maxItems": limit,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string", "enum": ["RESET"] + [f"ACTION{i}" for i in range(1, 8)]},
                            "repeat": {"type": "integer", "minimum": 1, "maximum": limit, "default": 1},
                            "x": {"type": "integer", "minimum": 0, "maximum": 63},
                            "y": {"type": "integer", "minimum": 0, "maximum": 63},
                        },
                    },
                }},
            },
        },
    }


SYSTEM = """Play this unfamiliar turn-based grid game and complete its levels.
Infer the action meanings, mechanics and goals from the observations. You have
exactly one tool: do_actions. Reason before calling it. Choose a short or long
sequence according to your uncertainty; a sequence is executed without model
feedback between its primitive actions. You may call it with just one action.
Use only the currently available action names, or RESET. On GAME_OVER, RESET
restarts the current attempt. WIN is the only signal that the whole game is won.
Do not output an ordinary final answer: use exactly one do_actions call per reply.

Grids are lossless rows of hexadecimal color labels: 0-9,A-F mean colors 0-15.
Coordinates are (x,y), with (0,0) at the top left. Color labels do not imply meaning.
Each action result includes all returned frames in chronological order; the last
is the latest observation. Earlier frames can be animation evidence. Each result
also includes state, levels_completed, and available_actions. ACTION6 requires x,y.
The tool stops a sequence on level changes, GAME_OVER, WIN or RESET and reports
how many actions were executed. Do not assume that discarded actions occurred.
"""


@dataclass(frozen=True)
class Action:
    name: str
    x: int | None = None
    y: int | None = None

    def data(self) -> dict[str, int]:
        return {} if self.x is None else {"x": self.x, "y": self.y}  # type: ignore[dict-item]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, **self.data()}


def validate_actions(arguments: str, limit: int) -> list[Action]:
    """Validate the entire syntax before executing even the first action."""
    try:
        value = json.loads(arguments)
    except (ValueError, TypeError) as exc:
        raise StopRun("Malformed tool JSON; no actions executed") from exc
    if not isinstance(value, dict) or set(value) != {"actions"}:
        raise StopRun("Expected only an actions field")
    items = value["actions"]
    if not isinstance(items, list) or not 1 <= len(items) <= limit:
        raise StopRun("Invalid action-list length")
    actions: list[Action] = []
    legal_names = {"RESET"} | {f"ACTION{i}" for i in range(1, 8)}
    for item in items:
        if not isinstance(item, dict) or set(item) - {"name", "repeat", "x", "y"}:
            raise StopRun("Invalid action object")
        name = item.get("name")
        if not isinstance(name, str) or name not in legal_names:
            raise StopRun("Unknown action name")
        repeat = item.get("repeat", 1)
        if type(repeat) is not int or not 1 <= repeat <= limit:
            raise StopRun("Invalid repeat count")
        if name == "ACTION6":
            x, y = item.get("x"), item.get("y")
            if any(type(v) is not int or not 0 <= v <= 63 for v in (x, y)):
                raise StopRun("ACTION6 requires integer x,y in [0,63]")
        else:
            if "x" in item or "y" in item:
                raise StopRun("Only ACTION6 accepts coordinates")
            x = y = None
        if len(actions) + repeat > limit:
            raise StopRun("Expanded action sequence exceeds tool limit")
        actions.extend(Action(name, x, y) for _ in range(repeat))
    return actions


def observation(raw: Any) -> dict[str, Any]:
    """Keep every returned grid, without screenshots, resizing or segmentation."""
    if raw is None:
        raise StopRun("Missing environment response; action outcome unknown, not retried")
    state = getattr(raw.state, "name", str(raw.state))
    available = []
    for value in raw.available_actions:
        number = int(getattr(value, "value", value))
        available.append("RESET" if number == 0 else f"ACTION{number}")
    frames = []
    for grid in raw.frame:
        rows = grid.tolist() if hasattr(grid, "tolist") else grid
        encoded = []
        width = len(rows[0]) if len(rows) else 0
        if len(rows) > 64 or width > 64:
            raise StopRun("Unexpected grid dimensions")
        for row in rows:
            if len(row) != width or any(not 0 <= int(v) <= 15 for v in row):
                raise StopRun("Invalid grid")
            encoded.append("".join("0123456789ABCDEF"[int(v)] for v in row))
        frames.append(encoded)
    return {
        "state": state,
        "levels_completed": int(raw.levels_completed),
        "win_levels": int(getattr(raw, "win_levels", 0)),
        "available_actions": available,
        "full_reset": bool(getattr(raw, "full_reset", False)),
        "frames": frames,
    }


class Game(Protocol):
    def observe(self) -> Any: ...
    def step(self, action: Action) -> Any: ...


class Journal:
    def __init__(self, root: str, game_id: str):
        stem = hashlib.sha256(game_id.encode()).hexdigest()[:24]
        folder = Path(root) / stem
        folder.mkdir(parents=True, exist_ok=True)
        # Refuse silent overwriting / accidental same-run duplicate game ownership.
        self.file = (folder / "events.jsonl").open("x", encoding="utf-8")
        self.write("game", game_id=game_id)

    def write(self, kind: str, **fields: Any) -> None:
        self.file.write(compact({"kind": kind, "t_monotonic": time.monotonic(), **fields}) + "\n")
        self.file.flush()  # one write per event, never one write per token

    def close(self) -> None:
        self.file.close()


class Admission:
    """FIFO request admission; the lock is NEVER held during model execution."""
    def __init__(self, config: Config):
        self.config = config
        self.condition = threading.Condition()
        self.waiting: deque[object] = deque()
        self.active = 0

    def __enter__(self) -> None:
        ticket = object()
        with self.condition:
            self.waiting.append(ticket)
            try:
                while self.waiting[0] is not ticket or self.active >= self.config.concurrency:
                    self.condition.wait(timeout=self.config.remaining())
                self.config.remaining()
                self.waiting.popleft()
                self.active += 1
                self.condition.notify_all()
            except BaseException:
                self.waiting.remove(ticket)
                self.condition.notify_all()
                raise

    def __exit__(self, *_: Any) -> None:
        with self.condition:
            self.active -= 1
            self.condition.notify_all()


class VLLM:
    """One shared admission gate; persistent, thread-local HTTP connections."""
    def __init__(self, config: Config):
        self.config = config
        self.admission = Admission(config)
        self.local = threading.local()

    def client(self) -> httpx.Client:
        if not hasattr(self.local, "client"):
            self.local.client = httpx.Client(
                base_url=self.config.base_url,
                transport=httpx.HTTPTransport(retries=0),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                trust_env=False,
            )
        return self.local.client

    def close_thread(self) -> None:
        if hasattr(self.local, "client"):
            self.local.client.close()
            del self.local.client

    def complete(self, messages: list[dict[str, Any]], key: str,
                 journal: Journal) -> dict[str, Any]:
        queued = time.monotonic()
        with self.admission:
            remaining = self.config.remaining()
            body = {
                "model": self.config.model, "messages": messages,
                "tools": [tool_schema(self.config.max_tool_actions)],
                "tool_choice": "auto", "parallel_tool_calls": False,
                "n": 1, "stream": False,
                "max_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature, "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "chat_template_kwargs": json.loads(self.config.chat_template_kwargs),
            }
            started = time.monotonic()
            journal.write("request", key=key, queued_at=queued, admitted_at=started,
                          body=body if self.config.trace_full_requests else
                          {k: v for k, v in body.items() if k != "messages"},
                          message_count=len(messages))
            # No SDK retries. A timeout stops this game instead of duplicating work.
            response = self.client().post(
                "/v1/chat/completions", json=body,
                headers={"X-Request-Id": key},
                timeout=httpx.Timeout(remaining, connect=min(5.0, remaining)),
            )
            if response.status_code != 200:
                journal.write("http_error", key=key, status=response.status_code, body=response.text)
                raise StopRun(f"Model server HTTP {response.status_code}; see journal")
            result = response.json()
            journal.write("response", key=key, seconds=time.monotonic()-started, body=result)
            self.config.remaining()  # never execute a result after the play deadline
            return result


def decode_reply(reply: dict[str, Any], limit: int) -> tuple[dict[str, Any], str, list[Action]]:
    choices = reply.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") not in {"stop", "tool_calls"}:
        raise StopRun("Missing/ambiguous/truncated generation; no actions executed")
    message = choices[0]["message"]
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "do_actions":
        raise StopRun("Expected exactly one do_actions call; no actions executed")
    call = calls[0]
    if not isinstance(call.get("id"), str) or not call["id"]:
        raise StopRun("Tool call has no ID")
    actions = validate_actions(call["function"]["arguments"], limit)
    # Preserve native reasoning fields instead of splicing them into user content.
    # Whether historical thinking is rendered is the checkpoint's chat-template policy.
    assistant = {k: v for k, v in message.items() if k in {
        "role", "content", "tool_calls", "reasoning", "reasoning_content"
    } and v is not None}
    assistant["role"] = "assistant"
    return assistant, call["id"], actions


def play_game(game: Game, game_id: str, backend: Any, config: Config) -> dict[str, Any]:
    """A game owns one sequential conversation and at most one model request."""
    journal = Journal(config.log_dir, game_id)
    steps = decisions = 0
    latest: dict[str, Any] = {}
    status = "ERROR"
    try:
        latest = observation(game.observe())
        journal.write("initial_observation", observation=latest)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": compact(latest)}]
        while latest["state"] != "WIN":
            config.remaining()
            if steps >= config.max_game_actions:
                status = "ACTION_LIMIT"
                break
            if decisions >= config.max_decisions:
                status = "DECISION_LIMIT"
                break
            key = hashlib.sha256(game_id.encode()).hexdigest()[:16] + f"-d{decisions}"
            reply = backend.complete(messages, key, journal)
            config.remaining()
            assistant, call_id, actions = decode_reply(reply, config.max_tool_actions)
            decisions += 1
            messages.append(assistant)
            results = []
            stopped = "sequence_complete"
            for action in actions:
                config.remaining()
                if steps >= config.max_game_actions:
                    stopped = "action_limit"
                    break
                if latest["state"] in {"GAME_OVER", "NOT_PLAYED"}:
                    available = {"RESET"}
                else:
                    available = set(latest["available_actions"]) | {"RESET"}
                if action.name not in available:
                    stopped = "action_unavailable"
                    break
                old_level = latest["levels_completed"]
                journal.write("action_submitted", key=key, index=steps, action=action.as_dict())
                # Exactly one call. Even None/timeout must never reissue this action.
                steps += 1  # Submitted calls, not a substitute for official action accounting.
                raw = game.step(action)
                latest = observation(raw)
                journal.write("action_observed", key=key, index=steps-1, observation=latest)
                results.append({"action": action.as_dict(), "observation": latest})
                if latest["state"] in {"WIN", "GAME_OVER"}:
                    stopped = latest["state"]
                    break
                if latest["levels_completed"] != old_level:
                    stopped = "level_change"
                    break
                if action.name == "RESET":
                    stopped = "reset"
                    break
            payload = {"executed": len(results), "requested": len(actions),
                       "stopped": stopped, "results": results}
            journal.write("tool_result", key=key, tool_call_id=call_id, result=payload)
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": compact(payload)})
        else:
            status = "WIN"
    except Deadline as exc:
        status = "DEADLINE"
        journal.write("stopped", error=repr(exc))
    except Exception as exc:
        # Fail this game, not the other games. No random fallback actions or retries.
        journal.write("stopped", error=repr(exc))
    finally:
        result = {"status": status, "actions": steps, "decisions": decisions,
                  "levels_completed": latest.get("levels_completed", 0)}
        journal.write("finished", **result)
        journal.close()
    return result
