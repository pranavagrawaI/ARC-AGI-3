"""One-request append-only JSONL bridge for the ephemeral Python 3.12 runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_MODE"] = "PRODUCTION"

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from agent import GroundingAgent
from provider import AuditedProviderClient, build_llm

PROTOCOL = "0158-grounding-ipc-v1"
LIVE_PROVIDER_ROUTE = {
    "order": ["coreweave/fp8"],
    "only": ["coreweave/fp8"],
    "allow_fallbacks": False,
    "require_parameters": False,
    "quantizations": ["fp8"],
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


async def _tail_from_start(path: str, poll_interval: float = 0.5):
    """Like runtime.producers.tail() but starts at position 0, not EOF.

    The host may write request 0 between process creation and the tail task's
    first run. Reading from the start makes attach timing irrelevant and never
    treats a partial final line as a record.
    """
    fh = open(path)  # noqa: ASYNC230, SIM115 - exact NOOA tail pattern; closed in finally
    try:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if line.endswith("\n"):
                if line.strip():
                    yield line.rstrip("\n")
            else:
                fh.seek(pos)
                await asyncio.sleep(poll_interval)
    finally:
        fh.close()


def _validate_request_lines(lines: list[str], path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("protocol") != PROTOCOL:
            raise ValueError(f"{path}:{line_number}: invalid protocol record")
        claimed_hash = value.get("request_sha256")
        unhashed = dict(value)
        unhashed.pop("request_sha256", None)
        if claimed_hash != sha256(_canonical(unhashed).encode("utf-8")).hexdigest():
            raise ValueError(f"{path}:{line_number}: request content hash mismatch")
        artifact_input = value.get("artifact_input")
        if not isinstance(artifact_input, dict):
            raise TypeError(f"{path}:{line_number}: artifact_input must be an object")
        if artifact_input.get("phase") != value.get("phase"):
            raise ValueError(f"{path}:{line_number}: phase binding mismatch")
        if artifact_input.get("current_observation_id") != value.get("observation_id"):
            raise ValueError(f"{path}:{line_number}: observation binding mismatch")
        if sha256(_canonical(artifact_input.get("evidence")).encode("utf-8")).hexdigest() != value.get("evidence_sha256"):
            raise ValueError(f"{path}:{line_number}: evidence binding mismatch")
        if sha256(_canonical(artifact_input.get("model_contract")).encode("utf-8")).hexdigest() != value.get("model_contract_sha256"):
            raise ValueError(f"{path}:{line_number}: model contract binding mismatch")
        records.append(value)
    identities = [(record.get("run_id"), record.get("artifact_id"), record.get("sequence")) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate request identity")
    return records


async def _read_requests_through(path: Path, request_index: int) -> list[dict[str, Any]]:
    lines: list[str] = []
    async for line in _tail_from_start(str(path), poll_interval=0.01):
        lines.append(line)
        if len(lines) > request_index:
            break
    return _validate_request_lines(lines, path)


class AuditedFixtureLLM(FakeLLMClient):
    def __init__(self, cells: list[str]) -> None:
        super().__init__()
        self.audit: list[dict[str, Any]] = []
        self._response_queue.extend(
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"fixture-cell-{index}",
                        name="execute_python",
                        arguments=json.dumps({"code": cell}),
                    )
                ],
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": ""},
            )
            for index, cell in enumerate(cells)
        )

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        visible_tools = [tool for tool in tools or () if tool.name != "return_result"]
        if len(visible_tools) != 1 or visible_tools[0].name != "execute_python":
            raise RuntimeError("inline-result fixture must expose only execute_python")
        kwargs["tool_choice"] = "auto"
        request = {
            "messages": messages,
            "tools": [
                {"name": tool.name, "description": tool.description, "parameters": tool.get_parameter_schema(strict=True)}
                for tool in visible_tools
            ],
        }
        self.audit.append(
            {
                "canonical_request_sha256": sha256(_canonical(request).encode("utf-8")).hexdigest(),
                "request": request,
            }
        )
        return await super().acall(
            messages,
            tools=visible_tools,
            output_model=output_model,
            **kwargs,
        )


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _client_for(request: dict[str, Any]) -> tuple[Any, bool]:
    mode = request.get("worker_mode", "fixture")
    if mode == "fixture":
        cells = request.get("fixture_python_cells")
        if not isinstance(cells, list) or not cells or not all(isinstance(cell, str) for cell in cells):
            raise ValueError("offline request requires non-empty fixture_python_cells")
        return AuditedFixtureLLM(cells), False
    if mode != "provider":
        raise ValueError("worker_mode must be fixture or provider")
    profile = request.get("provider_request_profile")
    if not isinstance(profile, dict):
        raise TypeError("provider request profile is required")
    if request.get("provider_route") != LIVE_PROVIDER_ROUTE:
        raise ValueError("live request does not carry the frozen provider route")
    key = os.environ.get("ARC_LLM_API_KEY")
    if not key:
        raise RuntimeError("ARC_LLM_API_KEY is unavailable to the provider worker")
    provider = build_llm(profile, api_key=key)
    if "extra_body" in provider.config:
        raise RuntimeError("provider client unexpectedly supplied extra_body")
    provider.config["extra_body"] = {"provider": LIVE_PROVIDER_ROUTE}
    retries = request.get("transport_max_retries")
    if retries != 5:
        raise ValueError("live HTTP 429 retry cap must be exactly five")
    return (
        AuditedProviderClient(
            provider,
            profile,
            transport_max_retries=retries,
            require_actual_crossings=True,
            known_credentials=(key,),
        ),
        True,
    )


def _audit_calls(client: Any) -> list[dict[str, Any]]:
    calls = getattr(client, "audit_calls", None)
    if isinstance(calls, list):
        return calls
    calls = getattr(client, "audit", None)
    return calls if isinstance(calls, list) else []


async def run_once(requests_path: Path, responses_path: Path, request_index: int, workspace: Path) -> None:
    requests = await _read_requests_through(requests_path, request_index)
    if not 0 <= request_index < len(requests):
        raise IndexError("request_index outside append-only request log")
    request = requests[request_index]
    if request.get("sequence") != request_index:
        raise ValueError("request sequence does not match append position")
    client, live = _client_for(request)
    workspace.mkdir(parents=True, exist_ok=False)
    workspace = workspace.resolve()
    limits = request.get("runtime_limits", {})
    if not isinstance(limits, dict):
        raise TypeError("runtime_limits must be an object")
    agent = GroundingAgent(
        llm=client,
        workspace=str(workspace),
        require_usage=live,
        max_iterations=int(limits.get("max_iterations", 10)),
        max_retries=int(limits.get("max_retries", 1)),
        timeout_seconds=float(limits.get("sandbox_timeout_seconds", 20.0)),
    )
    identity = {
        key: request[key]
        for key in (
            "protocol", "run_id", "artifact_id", "sequence", "phase", "observation_id",
            "evidence_sha256", "model_contract_sha256", "request_sha256",
        )
    }
    original_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        artifact = await agent.produce(request["artifact_input"])
        _append(
            responses_path,
            {
                **identity,
                "artifact": artifact,
                "artifact_sha256": sha256(_canonical(artifact).encode("utf-8")).hexdigest(),
                "audit": {"model_calls": _audit_calls(client), "usage": agent.usage},
                "complete": True,
            },
        )
    except BaseException as error:
        encoded = str(error).encode("utf-8", errors="replace")
        _append(
            responses_path,
            {
                **identity,
                "error": {
                    "type": type(error).__name__,
                    "message": (
                        f"redacted provider failure; chars={len(str(error))}; "
                        f"sha256={sha256(encoded).hexdigest()}"
                        if live
                        else str(error)
                    ),
                },
                "audit": {"model_calls": _audit_calls(client), "usage": agent.usage},
                "complete": True,
            },
        )
        if not live:
            raise
    finally:
        close = getattr(client, "aclose", None)
        if close is not None:
            await close()
        os.chdir(original_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--request-index", type=int, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run_once(args.requests, args.responses, args.request_index, args.workspace))


if __name__ == "__main__":
    main()
