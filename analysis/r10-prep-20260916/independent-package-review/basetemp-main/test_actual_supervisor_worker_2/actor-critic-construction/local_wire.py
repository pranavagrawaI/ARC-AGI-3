"""Explicit logical-to-SGLang projection; raw local receipts remain local."""
from __future__ import annotations
import copy
from typing import Any

LOCAL_MODEL = "/kaggle/input/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot"


def _require(condition, message):
    if not condition:
        raise ValueError("local wire: " + message)


def validate_identity(local_model, mounted_weights, startup_argv):
    _require(local_model == mounted_weights == LOCAL_MODEL, "model/weights identity")
    _require(isinstance(startup_argv, list) and all(isinstance(x, str) for x in startup_argv), "startup argv")
    _require("--model-path" in startup_argv, "startup argv lacks model-path")
    i = startup_argv.index("--model-path")
    _require(i + 1 < len(startup_argv) and startup_argv[i + 1] == LOCAL_MODEL, "startup weights mismatch")
    if "--served-model-name" in startup_argv:
        i = startup_argv.index("--served-model-name")
        _require(i + 1 < len(startup_argv) and startup_argv[i + 1] == LOCAL_MODEL, "served identity mismatch")


def project_messages(messages):
    _require(isinstance(messages, list) and bool(messages), "messages")
    result, pending, seen = [], set(), set()
    for message in messages:
        _require(isinstance(message, dict), "message object")
        role = message.get("role")
        _require(role in {"system", "user", "assistant", "tool"}, "role")
        allowed = {"role", "content", "name"}
        if role == "assistant":
            allowed.add("tool_calls")
        if role == "tool":
            allowed.add("tool_call_id")
        projected = {k: copy.deepcopy(v) for k, v in message.items() if k in allowed}
        if role == "tool":
            identity = message.get("tool_call_id")
            _require(identity in pending, "unmatched tool result")
            pending.remove(identity)
        else:
            _require(not pending, "missing tool results")
        if role == "assistant":
            calls = projected.get("tool_calls")
            if calls is not None:
                _require(isinstance(calls, list), "tool_calls")
                projected["tool_calls"] = []
                for call in calls:
                    _require(isinstance(call, dict), "tool call")
                    identity, function = call.get("id"), call.get("function")
                    _require(isinstance(identity, str) and bool(identity) and identity not in seen, "duplicate/invalid tool id")
                    _require(call.get("type") == "function" and isinstance(function, dict), "tool function")
                    _require(isinstance(function.get("name"), str) and isinstance(function.get("arguments"), str), "function fields")
                    seen.add(identity)
                    pending.add(identity)
                    projected["tool_calls"].append({"id": identity, "type": "function", "function": {k: function[k] for k in ("name", "arguments")}})
        result.append(projected)
    _require(not pending, "unanswered tool call")
    return result


def project_local(payload: dict[str, Any], *, local_model: str, mounted_weights: str, startup_argv: list[str]) -> dict[str, Any]:
    validate_identity(local_model, mounted_weights, startup_argv)
    _require(isinstance(payload, dict), "request object")
    allowed = {"model", "messages", "tools", "tool_choice", "max_tokens", "stream", "reasoning", "chat_template_kwargs", "provider", "parallel_tool_calls", "extra_body"}
    _require(not set(payload) - allowed, "unsupported request fields")
    body = {key: copy.deepcopy(payload[key]) for key in ("tools", "tool_choice", "max_tokens") if key in payload}
    body.update(model=LOCAL_MODEL, messages=project_messages(payload.get("messages")), stream=False,
                chat_template_kwargs={"enable_thinking": False})
    # Gateway reasoning/routing are not SGLang API parameters. No extra_body on wire.
    tokens = body.setdefault("max_tokens", 16384)
    _require(type(tokens) is int and 1 <= tokens <= 16384, "max_tokens")
    extra = payload.get("extra_body") or {}
    _require(isinstance(extra, dict) and not set(extra) - {"reasoning", "chat_template_kwargs", "provider", "parallel_tool_calls"}, "unsupported extra body")
    return body


def request_purpose(payload):
    names = {(t.get("function") or {}).get("name") for t in payload.get("tools") or [] if isinstance(t, dict)}
    if names & {"submit_candidate", "request_evidence"}:
        return "constructor"
    if payload.get("tools") is None and payload.get("max_tokens") == 192:
        return "critic"
    return "actor" if "execute_python" in names else "summarizer"


def usage_receipt(parsed):
    if not isinstance(parsed, dict) or parsed.get("error") or not parsed.get("choices"):
        return None
    usage = parsed.get("usage")
    if not isinstance(usage, dict):
        return None
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
        return None
    details = usage.get("completion_tokens_details") or {}
    if not isinstance(details, dict) or details.get("reasoning_tokens", 0) != 0:
        return None
    return {k: usage[k] for k in ("prompt_tokens", "completion_tokens")}


def valid_response(parsed):
    if not isinstance(parsed, dict) or parsed.get("model") != LOCAL_MODEL or parsed.get("provider") is not None:
        return False
    choices = parsed.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return False
    message = choices[0].get("message")
    return isinstance(message, dict) and message.get("role") == "assistant" and not message.get("reasoning_content") and not message.get("reasoning") and usage_receipt(parsed) is not None
