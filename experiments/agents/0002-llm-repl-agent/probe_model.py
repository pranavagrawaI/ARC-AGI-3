"""Make one sanitized multimodal/tool compatibility probe against the Qwen API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from dotenv import load_dotenv

from llm_repl_agent import (
    OpenAICompatibleChatClient,
    model_request_options,
    resolve_model_environment,
)
from llm_repl_agent.agent import PYTHON_TOOL


# Valid one-pixel PNG. The probe tests image transport, not visual competence.
PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


def _safe_usage(usage: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(key): value
        for key, value in usage.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def valid_python_tool_call(tool_calls: Any) -> bool:
    """Require the response shape that the real agent can execute."""

    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        return False
    call = tool_calls[0]
    if not isinstance(call, Mapping):
        return False
    function = call.get("function")
    if not isinstance(function, Mapping) or function.get("name") != "python":
        return False
    arguments = function.get("arguments")
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, Mapping) and isinstance(parsed.get("code"), str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    load_dotenv(REPO / ".env")
    try:
        environment = resolve_model_environment(os.environ)
        if environment.profile != "qwen3.6":
            raise ValueError("probe requires QWEN_MODEL so the qwen3.6 profile is explicit")
        request_options = model_request_options(
            environment.profile,
            max_completion_tokens=args.max_completion_tokens,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    client = OpenAICompatibleChatClient(
        base_url=environment.endpoint,
        model=environment.model,
        api_key=environment.api_key,
        timeout_seconds=args.timeout,
        **request_options,
    )
    response = client.complete(
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Compatibility test: inspect the attached image, then call the python "
                        "tool exactly once with code that prints 6 * 7. Do not answer directly."
                    ),
                },
                {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
            ],
        }],
        tools=(PYTHON_TOOL,),
    )
    tool_calls = response.message.get("tool_calls")
    calls = tool_calls if isinstance(tool_calls, list) else []
    tool_names = []
    for call in calls:
        function = call.get("function") if isinstance(call, Mapping) else None
        if isinstance(function, Mapping):
            tool_names.append(str(function.get("name")))
    arguments_valid = valid_python_tool_call(tool_calls)
    passed = tool_names == ["python"] and arguments_valid
    print(json.dumps({
        "api_key_source": environment.api_key_source,
        "arguments_valid": arguments_valid,
        "endpoint": environment.endpoint,
        "has_reasoning_content": isinstance(
            response.message.get("reasoning_content"), str
        ),
        "model": environment.model,
        "passed": passed,
        "profile": environment.profile,
        "response_id": response.response_id,
        "tool_names": tool_names,
        "usage": _safe_usage(response.usage),
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("probe response did not contain one executable python tool call")


if __name__ == "__main__":
    main()
