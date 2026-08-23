"""Authorized, globally budgeted execution of the frozen parity screen."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nooa.unifiedllm import LLMResponse, Tool, UnifiedLLM
from pydantic import BaseModel

from tycho_nooa.cognitive_corpus import load_corpus
from tycho_nooa.parity_experiment import execute_matched_packet
from tycho_nooa.parity_manifest import (
    ExperimentCaps,
    ModelRoute,
    load_experiment_manifest,
)


class BudgetState:
    """Reserve before every provider call and stop on missing cost evidence."""

    def __init__(self, caps: ExperimentCaps, *, path: Path):
        if caps.max_cost_usd is None or caps.reserve_per_call_usd is None:
            raise ValueError("live budget requires cost cap and per-call reserve")
        self.caps = caps
        self.path = path
        self.calls = 0
        self.observed_cost_usd = 0.0
        self.stop_reason: str | None = None

    def reserve(self) -> int:
        if self.stop_reason is not None:
            raise RuntimeError(f"experiment budget is stopped: {self.stop_reason}")
        if self.calls >= self.caps.calls_total:
            self.stop_reason = "call_cap"
            self.persist()
            raise RuntimeError("global provider-call cap reached")
        assert self.caps.max_cost_usd is not None
        assert self.caps.reserve_per_call_usd is not None
        if (
            self.observed_cost_usd + self.caps.reserve_per_call_usd
            > self.caps.max_cost_usd
        ):
            self.stop_reason = "cost_reservation"
            self.persist()
            raise RuntimeError("global cost reservation refused")
        self.calls += 1
        self.persist()
        return self.calls

    def observe(self, response: LLMResponse) -> float:
        usage = response.usage if isinstance(response.usage, dict) else {}
        raw_cost = usage.get("cost")
        if (
            isinstance(raw_cost, bool)
            or not isinstance(raw_cost, (int, float))
            or raw_cost < 0
        ):
            self.stop_reason = "missing_cost_evidence"
            self.persist()
            raise RuntimeError("provider response lacks valid observed cost")
        self.observed_cost_usd += float(raw_cost)
        assert self.caps.max_cost_usd is not None
        if self.observed_cost_usd > self.caps.max_cost_usd:
            self.stop_reason = "observed_cost_exceeded"
            self.persist()
            raise RuntimeError("observed provider cost exceeded global cap")
        self.persist()
        return float(raw_cost)

    def persist(self) -> None:
        _write_json(
            self.path,
            {
                "calls": self.calls,
                "calls_total_cap": self.caps.calls_total,
                "max_cost_usd": self.caps.max_cost_usd,
                "observed_cost_usd": round(self.observed_cost_usd, 12),
                "reserve_per_call_usd": self.caps.reserve_per_call_usd,
                "stop_reason": self.stop_reason,
            },
        )


class CappedParityLLM(UnifiedLLM):
    """Trace and enforce one arm while sharing a global budget across arms."""

    def __init__(
        self,
        delegate: UnifiedLLM,
        *,
        arm: str,
        expected_route: ModelRoute,
        budget: BudgetState,
        trace_path: Path,
    ):
        super().__init__(delegate.model, **delegate.config)
        self._delegate = delegate
        self._arm = arm
        self._expected_route = expected_route
        self._budget = budget
        self._trace_path = trace_path
        self.cache_control_injection_points = delegate.cache_control_injection_points

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del messages, tools, output_model, kwargs
        raise RuntimeError("parity screen uses async provider calls only")

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        call_index = self._budget.reserve()
        request = {
            "messages": messages,
            "tools": [tool.name for tool in tools or ()],
            "output_model": getattr(output_model, "__name__", None),
            "parameters": kwargs,
        }
        started = time.monotonic()
        base = {
            "arm": self._arm,
            "call_index": call_index,
            "request_sha256": _commitment(request),
        }
        try:
            response = await self._delegate.acall(
                messages, tools=tools, output_model=output_model, **kwargs
            )
            route = _provider_route(response.raw_response)
            _validate_observed_route(route, self._expected_route)
            cost = self._budget.observe(response)
        except BaseException as error:
            if self._budget.stop_reason is None:
                self._budget.stop_reason = f"provider_error:{type(error).__name__}"
                self._budget.persist()
            _append_jsonl(
                self._trace_path,
                {
                    **base,
                    "request": request,
                    "status": "error",
                    "elapsed_seconds": time.monotonic() - started,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            raise
        _append_jsonl(
            self._trace_path,
            {
                **base,
                "request": request,
                "status": "complete",
                "elapsed_seconds": time.monotonic() - started,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
                "observed_cost_usd": cost,
                "provider_route": route,
                "response": {
                    "content": response.content,
                    "reasoning": response.reasoning,
                    "tool_calls": [
                        {"name": call.name, "arguments": call.arguments}
                        for call in response.tool_calls
                    ],
                },
            },
        )
        return response


async def execute_authorized_screen(
    manifest_path: Path,
    *,
    repository_root: Path,
    destination: Path,
    native_llm: UnifiedLLM,
    codeact_llm: UnifiedLLM,
) -> None:
    """Validate authorization before creating output or contacting a provider."""
    manifest = load_experiment_manifest(manifest_path, repository_root=repository_root)
    manifest.require_execution_authorized()
    corpus, packets = load_corpus(
        repository_root / manifest.corpus_manifest.path,
        repository_root=repository_root,
    )
    del corpus
    if manifest.caps.sessions != len(packets) * 2:
        raise ValueError("session cap does not match the frozen corpus and arms")
    _validate_configured_model(native_llm, manifest.model_route)
    _validate_configured_model(codeact_llm, manifest.model_route)
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "experiment.json", manifest.model_dump(mode="json"))
    budget = BudgetState(manifest.caps, path=destination / "budget.json")
    trace = destination / "provider-calls.jsonl"
    native = CappedParityLLM(
        native_llm,
        arm="native",
        expected_route=manifest.model_route,
        budget=budget,
        trace_path=trace,
    )
    codeact = CappedParityLLM(
        codeact_llm,
        arm="codeact",
        expected_route=manifest.model_route,
        budget=budget,
        trace_path=trace,
    )
    for packet in packets:
        await execute_matched_packet(
            packet,
            repository_root=repository_root,
            destination=destination / packet.task.packet_id,
            native_llm=native,
            codeact_llm=codeact,
            max_llm_calls=manifest.caps.calls_per_session,
            max_output_tokens=manifest.caps.output_tokens_per_call,
            session_timeout_seconds=manifest.caps.wall_seconds_per_session,
            cell_timeout_seconds=min(5.0, manifest.caps.wall_seconds_per_session),
        )


def _validate_configured_model(llm: UnifiedLLM, route: ModelRoute) -> None:
    if llm.model != route.model:
        raise ValueError("configured model does not match frozen route")
    effort = llm.config.get("reasoning_effort")
    reasoning = llm.config.get("reasoning")
    if effort is None and isinstance(reasoning, Mapping):
        effort = reasoning.get("effort")
    if effort != route.reasoning_effort:
        raise ValueError("configured reasoning effort does not match frozen route")
    if llm.config.get("temperature") != route.temperature:
        raise ValueError("configured temperature does not match frozen route")


def _validate_observed_route(observed: dict[str, str], expected: ModelRoute) -> None:
    if observed.get("provider") != expected.provider:
        raise RuntimeError("observed provider route drift")
    if observed.get("model") != expected.model:
        raise RuntimeError("observed model route drift")


def _provider_route(raw: Any) -> dict[str, str]:
    fields = {
        "provider": _raw_field(raw, "provider"),
        "model": _raw_field(raw, "model"),
    }
    return {key: value for key, value in fields.items() if isinstance(value, str)}


def _raw_field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _commitment(value: object) -> str:
    encoded = json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, default=str, separators=(",", ":")) + "\n")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
