"""The exact wire payload for a replayed request (shared by replay.py and tokcheck.py).

Mirrors dfranzen's ToolAgent._chat_completion for provider 'vllm' with the control
notebook's environment: non-streaming, temperature 0.7, top_p 0.95, top_k 20,
chat_template_kwargs {enable_thinking, preserve_thinking}, private control keys stripped.
max_tokens is the recorded completion length (CARD.md), so a natural EOS can end early.
"""
CONTROL_KEY = "_arc3_control"  # dfranzen tool_agent._CONTROL_MESSAGE_KEY
TEMPERATURE, TOP_P, TOP_K = 0.7, 0.95, 20


def wire_messages(messages):
    return [{k: v for k, v in m.items() if k != CONTROL_KEY} for m in messages]


def build(logged_request, model, max_tokens):
    payload = {
        "model": model,
        "messages": wire_messages(logged_request["messages"]),
        "stream": False,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": int(max_tokens),
        "top_k": TOP_K,
        "chat_template_kwargs": {"enable_thinking": True,
                                 **(logged_request.get("chat_template_kwargs") or {})},
    }
    if logged_request.get("tools"):
        payload["tools"] = logged_request["tools"]
        if logged_request.get("tool_choice"):
            payload["tool_choice"] = logged_request["tool_choice"]
    return payload
