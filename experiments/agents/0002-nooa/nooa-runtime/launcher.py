# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned append-only file bridge for the nested Python 3.12 control.

Derived by copy-and-prune from NVIDIA NOOA's pinned
``examples/arc_agi_3/launcher.py``; provenance is recorded in
``../SOURCE_MANIFEST.json``.
"""

from __future__ import annotations

import argparse
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

# NOOA imports LiteLLM. Force its bundled registry and suppress its implicit
# dotenv loader before the dependency can execute either import-time path.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_MODE"] = "PRODUCTION"

from nooa.errors import GenerationError
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from agent import ModelBudgetExceeded, NooaControlAgent, PublicActionPoint
from llm import AuditedProviderClient, ProviderTransportAbort, build_llm

PROTOCOL = "0002-nooa-ipc-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: record must be an object")
        if value.get("protocol") != PROTOCOL:
            raise ValueError(f"{path}:{line_number}: unsupported protocol")
        records.append(value)
    return records


def _validate_request_identities(records: list[dict[str, Any]]) -> None:
    identities: set[tuple[str, int]] = set()
    paired_run_spec: object = None
    for record in records:
        run_id = record.get("run_id")
        action_point_id = record.get("action_point_id")
        if not isinstance(run_id, str) or not isinstance(action_point_id, int):
            raise ValueError("run_id must be a string and action_point_id must be an integer")
        identity = (run_id, action_point_id)
        if identity in identities:
            raise ValueError(f"duplicate action point: {identity!r}")
        identities.add(identity)
        current_spec = record.get("paired_run_spec")
        if paired_run_spec is None:
            paired_run_spec = current_spec
        elif current_spec != paired_run_spec:
            raise ValueError("paired_run_spec changed inside one worker run")


def _fixture_responses(cells: list[str]) -> list[LLMResponse]:
    return [
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
    ]


def _fixture_cells(record: dict[str, Any]) -> list[str]:
    cells = record.get("fixture_python_cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("offline fixture requests require non-empty fixture_python_cells")
    if not all(isinstance(cell, str) for cell in cells):
        raise ValueError("fixture_python_cells must contain only strings")
    return cells


def _runtime_options(record: dict[str, Any] | None) -> dict[str, Any]:
    spec = record.get("paired_run_spec") if record is not None else None
    if spec is None:
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError("paired_run_spec must be an object")
    options = {
        "max_context_turns": spec.get("max_context_turns", 6),
        "max_model_rounds_per_action": spec.get("max_model_rounds_per_action", 10),
        "max_decision_repairs": spec.get("max_decision_repairs", 2),
        "workspace_timeout_seconds": spec.get("workspace_timeout_seconds", 20.0),
        "max_model_calls": spec.get("max_model_calls", 100),
        "max_reported_tokens": spec.get("max_reported_tokens", 250_000),
    }
    if any(
        value is not None and type(value) not in {int, float}
        for value in options.values()
    ):
        raise ValueError("paired runtime settings must be numeric")
    return options


class _FixtureLLM(FakeLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.audit_calls: list[dict[str, Any]] = []

    def enqueue(self, cells: list[str]) -> None:
        self._response_queue.extend(_fixture_responses(cells))

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        output_model: type[Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        canonical_request = {
            "messages": messages,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_parameter_schema(strict=True),
                }
                for tool in tools or ()
            ],
        }
        canonical_bytes = json.dumps(
            canonical_request,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        audit = json.loads(
            json.dumps(
                {
                    **canonical_request,
                    "canonical_request": canonical_request,
                    "canonical_request_sha256": sha256(canonical_bytes).hexdigest(),
                    "canonicalization": "json-utf8-sort-keys-compact-v1",
                    "actual_provider_crossings": [],
                }
            )
        )
        self.audit_calls.append(audit)
        response = await super().acall(
            messages,
            tools=tools,
            output_model=output_model,
            **kwargs,
        )
        audit["outcome"] = {
            "kind": "success",
            "response": {
                "assistant_message": response.assistant_message,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                    for tool_call in response.tool_calls
                ],
            },
        }
        return response


def _fixture_client(records: list[dict[str, Any]]) -> _FixtureLLM:
    client = _FixtureLLM()
    for record in records:
        client.enqueue(_fixture_cells(record))
    return client


def _provider_client(
    request: dict[str, Any],
    *,
    api_key: str | None,
) -> AuditedProviderClient:
    spec = request.get("paired_run_spec")
    if not isinstance(spec, dict):
        raise ValueError("paired_run_spec must be an object")
    profile = spec.get("provider_request_profile")
    if not isinstance(profile, dict) or not profile:
        raise ValueError("provider_request_profile must be a non-empty object")
    retries = spec.get("transport_max_retries")
    if type(retries) is not int or retries < 0:
        raise ValueError("transport_max_retries must be a non-negative integer")
    provider = build_llm(profile, api_key=api_key)
    return AuditedProviderClient(
        provider,
        profile,
        transport_max_retries=retries,
        require_actual_crossings=True,
        known_credentials=(api_key,) if api_key else (),
    )


def _append_response(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _append_failure_response(
    path: Path,
    *,
    run_id: str,
    action_point_id: int,
    error: dict[str, Any],
    model_calls: list[dict[str, Any]],
) -> None:
    _append_response(
        path,
        {
            "protocol": PROTOCOL,
            "run_id": run_id,
            "action_point_id": action_point_id,
            "error": error,
            "audit": {"model_calls": model_calls},
            "complete": True,
        },
    )


def _find_model_budget(error: BaseException) -> ModelBudgetExceeded | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModelBudgetExceeded):
            return current
        current = current.__cause__ or current.__context__
    return None


async def run_offline_fixture(
    requests_path: Path,
    responses_path: Path,
    workspace: Path,
    model_frame_path: Path | None = None,
) -> int:
    """Execute deterministic scripted model cells over the real control path."""

    requests = _read_jsonl(requests_path)
    _validate_request_identities(requests)
    if responses_path.exists() and responses_path.stat().st_size:
        existing = _read_jsonl(responses_path)
        existing_ids = {(item["run_id"], item["action_point_id"]) for item in existing}
        requested_ids = {(item["run_id"], item["action_point_id"]) for item in requests}
        overlap = existing_ids & requested_ids
        if overlap:
            raise ValueError(f"response already exists for action point: {sorted(overlap)!r}")

    client = _fixture_client(requests)
    workspace.mkdir(parents=True, exist_ok=True)
    control = NooaControlAgent(
        llm=client,
        workspace=str(workspace.resolve()),
        model_frame_path=(
            str(model_frame_path.resolve()) if model_frame_path is not None else None
        ),
        **_runtime_options(requests[0] if requests else None),
    )
    original_cwd = Path.cwd()
    completed = 0
    try:
        os.chdir(workspace)
        for request in requests:
            call_offset = len(client.audit_calls)
            action_point = PublicActionPoint(
                observation=request.get("observation", {}),
                legal_actions=tuple(request.get("legal_actions", ())),
                coordinate_actions=tuple(request.get("coordinate_actions", ())),
                frame_width=request.get("frame_width"),
                frame_height=request.get("frame_height"),
            )
            decision = await control.choose_action(action_point)
            _append_response(
                responses_path,
                {
                    "protocol": PROTOCOL,
                    "run_id": request["run_id"],
                    "action_point_id": request["action_point_id"],
                    "decision": decision.model_dump(mode="json"),
                    "audit": {"model_calls": client.audit_calls[call_offset:]},
                    "complete": True,
                },
            )
            completed += 1
    finally:
        try:
            await control.aclose()
        finally:
            os.chdir(original_cwd)
    return completed


async def serve_offline_fixture(
    requests_path: Path,
    responses_path: Path,
    workspace: Path,
    model_frame_path: Path | None = None,
    *,
    poll_interval: float = 0.01,
    _provider_mode: bool = False,
    _api_key: str | None = None,
) -> int:
    """Serve appended requests while preserving one control process and client."""

    requests_path.parent.mkdir(parents=True, exist_ok=True)
    requests_path.touch(exist_ok=True)
    client: Any = None if _provider_mode else _FixtureLLM()
    workspace.mkdir(parents=True, exist_ok=True)
    control: NooaControlAgent | None = None
    original_cwd = Path.cwd()
    seen: set[tuple[str, int]] = set()
    completed = 0
    offset = 0
    pending = ""
    active_spec: object = None
    try:
        os.chdir(workspace)
        while True:
            with requests_path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            if not chunk:
                await asyncio.sleep(poll_interval)
                continue
            pending += chunk
            lines = pending.splitlines(keepends=True)
            pending = ""
            if lines and not lines[-1].endswith("\n"):
                pending = lines.pop()
            for raw_line in lines:
                if not raw_line.strip():
                    continue
                request = json.loads(raw_line)
                if not isinstance(request, dict) or request.get("protocol") != PROTOCOL:
                    raise ValueError("unsupported request record")
                if request.get("kind") == "stop":
                    return completed
                run_id = request.get("run_id")
                action_point_id = request.get("action_point_id")
                if not isinstance(run_id, str) or not isinstance(action_point_id, int):
                    raise ValueError("invalid action point identity")
                identity = (run_id, action_point_id)
                if identity in seen:
                    raise ValueError(f"duplicate action point: {identity!r}")
                seen.add(identity)
                if _provider_mode:
                    if client is None:
                        client = _provider_client(request, api_key=_api_key)
                else:
                    client.enqueue(_fixture_cells(request))
                if control is None:
                    active_spec = request.get("paired_run_spec")
                    control = NooaControlAgent(
                        llm=client,
                        workspace=str(workspace.resolve()),
                        model_frame_path=(
                            str(model_frame_path.resolve())
                            if model_frame_path is not None
                            else None
                        ),
                        **_runtime_options(request),
                    )
                elif request.get("paired_run_spec") != active_spec:
                    raise ValueError("paired_run_spec changed inside one worker run")
                call_offset = len(client.audit_calls)
                try:
                    decision = await control.choose_action(
                        PublicActionPoint(
                            observation=request.get("observation", {}),
                            legal_actions=tuple(request.get("legal_actions", ())),
                            coordinate_actions=tuple(request.get("coordinate_actions", ())),
                            frame_width=request.get("frame_width"),
                            frame_height=request.get("frame_height"),
                        )
                    )
                except ProviderTransportAbort as error:
                    _append_failure_response(
                        responses_path,
                        run_id=run_id,
                        action_point_id=action_point_id,
                        error={
                            "kind": "provider_transport",
                            "disposition": error.disposition.value,
                            "attempts": error.attempts,
                        },
                        model_calls=client.audit_calls[call_offset:],
                    )
                    continue
                except GenerationError as error:
                    budget = _find_model_budget(error)
                    failure = (
                        {
                            "kind": "budget_exhausted",
                            "resource": budget.resource,
                            "used": budget.used,
                            "limit": budget.limit,
                        }
                        if budget is not None
                        else {
                            "kind": "decision_failure",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    _append_failure_response(
                        responses_path,
                        run_id=run_id,
                        action_point_id=action_point_id,
                        error=failure,
                        model_calls=client.audit_calls[call_offset:],
                    )
                    continue
                _append_response(
                    responses_path,
                    {
                        "protocol": PROTOCOL,
                        "run_id": run_id,
                        "action_point_id": action_point_id,
                        "decision": decision.model_dump(mode="json"),
                        "audit": {"model_calls": client.audit_calls[call_offset:]},
                        "complete": True,
                    },
                )
                completed += 1
    finally:
        try:
            if control is not None:
                await control.aclose()
        finally:
            try:
                if _provider_mode and client is not None:
                    await client.aclose()
            finally:
                os.chdir(original_cwd)


async def serve_provider(
    requests_path: Path,
    responses_path: Path,
    workspace: Path,
    model_frame_path: Path | None = None,
    *,
    api_key: str | None,
    poll_interval: float = 0.01,
) -> int:
    """Serve the explicit provider profile; credentials are never read by llm.py."""

    return await serve_offline_fixture(
        requests_path,
        responses_path,
        workspace,
        model_frame_path,
        poll_interval=poll_interval,
        _provider_mode=True,
        _api_key=api_key,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model-frame", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline-fixture", action="store_true")
    mode.add_argument("--provider", action="store_true")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if args.provider:
        if not args.serve:
            parser.error("provider mode requires --serve")
        completed = asyncio.run(
            serve_provider(
                args.requests,
                args.responses,
                args.workspace,
                args.model_frame,
                api_key=os.environ.get("ARC_LLM_API_KEY"),
            )
        )
    else:
        operation = serve_offline_fixture if args.serve else run_offline_fixture
        completed = asyncio.run(
            operation(
                args.requests,
                args.responses,
                args.workspace,
                args.model_frame,
            )
        )
    print(json.dumps({"protocol": PROTOCOL, "completed": completed}))


if __name__ == "__main__":
    main()
