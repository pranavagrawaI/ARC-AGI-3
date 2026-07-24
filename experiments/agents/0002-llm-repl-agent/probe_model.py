"""Make one sanitized multimodal/tool compatibility probe against the Qwen API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
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
    resolve_model_environment,
)
from llm_repl_agent.agent import PYTHON_TOOL
from llm_repl_agent.client import serialize_request_payload
from llm_repl_agent.experiment_protocol import (
    canonical_endpoint_metadata,
    endpoint_metadata_sha256,
    fetch_endpoint_metadata,
    frozen_request_options,
    load_frozen_runtime,
    participating_source_hashes,
    validate_endpoint_metadata,
)
from llm_repl_agent.workspace import PythonWorkspace


# Valid 16x16 PNG. The probe tests image transport, not visual competence.
PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAGUlEQVR4nGP8//8/AymA"
    "iSTVoxpGNQwpDQBVbQMdPVIhQwAAAABJRU5ErkJggg=="
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
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    load_dotenv(REPO / ".env")
    try:
        config = load_frozen_runtime()
        environment = resolve_model_environment(
            os.environ,
            model=config.model,
            endpoint=config.endpoint,
            profile=config.model_profile,
        )
        request_options = frozen_request_options(config)
        raw_endpoint_metadata = fetch_endpoint_metadata(
            config,
            timeout_seconds=min(args.timeout, 30.0),
        )
        endpoint_metadata = canonical_endpoint_metadata(
            raw_endpoint_metadata,
            model=config.model,
            tag=config.provider.only[0],
        )
        validate_endpoint_metadata(endpoint_metadata, config)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or AGENT_ROOT / "artifacts" / "compatibility" / timestamp
    if output_dir.exists():
        raise SystemExit(f"compatibility output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "endpoint-metadata.raw.json").write_text(
        json.dumps(raw_endpoint_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "endpoint-metadata.canonical.json").write_text(
        json.dumps(endpoint_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    client = OpenAICompatibleChatClient(
        base_url=environment.endpoint,
        model=environment.model,
        api_key=environment.api_key,
        timeout_seconds=args.timeout,
        **request_options,
    )
    initial_messages = [{
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
        }]
    response = client.complete(
        messages=initial_messages,
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
    reasoning = response.message.get("reasoning_content")
    payload = response.request_payload if isinstance(response.request_payload, Mapping) else {}
    provider = payload.get("provider") if isinstance(payload, Mapping) else None
    request_contract_valid = (
        payload.get("seed") == 0
        and payload.get("top_k") == config.request.top_k
        and payload.get("max_tokens") == config.request.max_completion_tokens
        and "max_completion_tokens" not in payload
        and provider == config.provider.request_value()
    )
    reasoning_returned = isinstance(reasoning, str) and bool(reasoning.strip())
    tool_execution_valid = False
    tool_result_payload: dict[str, Any] = {}
    continuation_valid = False
    continuation = None
    if arguments_valid:
        call = calls[0]
        function = call["function"]
        arguments = function["arguments"]
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        with PythonWorkspace(timeout_seconds=20) as workspace:
            execution = workspace.execute(str(parsed["code"]))
        tool_result_payload = json.loads(execution.model_text())
        tool_execution_valid = execution.ok and execution.stdout.strip() == "42"
        assistant = {
            key: response.message.get(key)
            for key in ("role", "content", "tool_calls", "reasoning_content", "reasoning_details")
            if key in response.message
        }
        assistant["role"] = "assistant"
        continuation_messages = [
            *initial_messages,
            assistant,
            {
                "role": "tool",
                "tool_call_id": str(call.get("id")),
                "content": execution.model_text(),
            },
            {"role": "user", "content": "Confirm the Python result in one short sentence."},
        ]
        continuation = client.complete(messages=continuation_messages, tools=(PYTHON_TOOL,))
        continuation_content = continuation.message.get("content")
        continuation_valid = (
            isinstance(continuation_content, str)
            and bool(continuation_content.strip())
            and not continuation.message.get("tool_calls")
        )
    archived: dict[str, str] = {}
    for name, response_value in (("1", response), ("2", continuation)):
        if response_value is None:
            continue
        request_bytes = serialize_request_payload(response_value.request_payload or {})
        request_path = output_dir / f"request-{name}.json"
        request_path.write_bytes(request_bytes)
        raw = response_value.response_body_raw
        if not isinstance(raw, str):
            raise SystemExit("client did not retain the exact raw compatibility response")
        response_path = output_dir / f"response-{name}.raw.json"
        response_path.write_text(raw, encoding="utf-8")
        archived[f"request_{name}_sha256"] = sha256(request_bytes).hexdigest()
        archived[f"response_{name}_sha256"] = sha256(raw.encode()).hexdigest()
    tool_result_path = output_dir / "tool-result.json"
    tool_result_path.write_text(
        json.dumps(tool_result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archived["tool_result_sha256"] = sha256(tool_result_path.read_bytes()).hexdigest()
    passed = (
        tool_names == ["python"]
        and arguments_valid
        and reasoning_returned
        and request_contract_valid
        and tool_execution_valid
        and continuation_valid
        and continuation is not None
        and all(
            digest == expected
            for digest, expected in (
                (archived["request_1_sha256"], response.request_payload_sha256),
                (archived["response_1_sha256"], response.response_body_sha256),
                (archived["request_2_sha256"], continuation.request_payload_sha256),
                (archived["response_2_sha256"], continuation.response_body_sha256),
            )
        )
    )
    result = {
        "schema": "scientist-game-designer-player-compatibility-v1",
        "api_key_source": environment.api_key_source,
        "arguments_valid": arguments_valid,
        "endpoint": environment.endpoint,
        "endpoint_metadata_sha256": endpoint_metadata_sha256(endpoint_metadata),
        "has_reasoning_content": reasoning_returned,
        "model": environment.model,
        "passed": passed,
        "profile": environment.profile,
        "request_contract_valid": request_contract_valid,
        "tool_execution_valid": tool_execution_valid,
        "continuation_valid": continuation_valid,
        "artifact_hashes": archived,
        "request_payload_sha256": response.request_payload_sha256,
        "continuation_request_payload_sha256": (
            continuation.request_payload_sha256 if continuation is not None else None
        ),
        "response_id": response.response_id,
        "continuation_response_id": continuation.response_id if continuation is not None else None,
        "tool_names": tool_names,
        "usage": _safe_usage(response.usage),
        "source_hashes": participating_source_hashes(),
        "output_dir": str(output_dir),
    }
    (output_dir / "probe-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(
            "probe failed image/reasoning/tool/top_k/requested-seed compatibility"
        )


if __name__ == "__main__":
    main()
