"""Inherited exact message projection; unrelated local serving code omitted."""

import copy


def _require(condition, message):
    if not condition:
        raise ValueError("local wire: " + message)


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
                    _require(
                        isinstance(identity, str)
                        and bool(identity)
                        and identity not in seen,
                        "duplicate/invalid tool id",
                    )
                    _require(
                        call.get("type") == "function" and isinstance(function, dict),
                        "tool function",
                    )
                    _require(
                        isinstance(function.get("name"), str)
                        and isinstance(function.get("arguments"), str),
                        "function fields",
                    )
                    seen.add(identity)
                    pending.add(identity)
                    projected["tool_calls"].append(
                        {
                            "id": identity,
                            "type": "function",
                            "function": {k: function[k] for k in ("name", "arguments")},
                        }
                    )
        result.append(projected)
    _require(not pending, "unanswered tool call")
    return result
