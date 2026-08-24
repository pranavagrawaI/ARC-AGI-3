"""Resource-safe provider adapter for V2 experiments."""

from __future__ import annotations

from typing import Any, Literal

from nooa.unifiedllm import (
    CompletionClient,
    FakeLLMClient,
    LLMResponse,
    RetryConfig,
    Tool,
    UnifiedLLM,
)
from nooa.unifiedllm.unifiedllm import (
    _DUMMY_TOOL_SCHEMA,
    _maybe_sanitize_response_format,
    _messages_have_tool_calls,
    _needs_dummy_tool,
)
from pydantic import BaseModel

from tycho_nooa.compatibility_authorization_v2 import DurableAuthorizationLedger
from tycho_nooa.compatibility_launch import (
    _exception_response,
    _json_safe,
    _provider_failure,
    _provider_route,
    _response_value,
    _validate_output_ceiling,
)

CompatibilityArm = Literal["native", "codeact"]


class AuthorizedCompatibilityLLM(UnifiedLLM):
    """Admit the exact assembled request before invoking the provider."""

    def __init__(
        self,
        delegate: UnifiedLLM,
        *,
        gate_id: str,
        arm: CompatibilityArm,
        ledger: DurableAuthorizationLedger,
    ):
        super().__init__(delegate.model, **delegate.config)
        self._delegate = delegate
        self._gate_id = gate_id
        self._arm = arm
        self._ledger = ledger
        self.cache_control_injection_points = delegate.cache_control_injection_points
        _disable_endpoint_retries(delegate)

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del messages, tools, output_model, kwargs
        raise RuntimeError("V2 provider adapter uses async calls only")

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        _validate_output_ceiling(kwargs, self._ledger.authorization)
        request = provider_request_value(
            self._delegate,
            messages,
            tools,
            output_model,
            kwargs,
        )
        call = self._ledger.plan_call(
            gate_id=self._gate_id,
            arm=self._arm,
            request=request,
        )
        self._ledger.dispatch(call.call_id)
        try:
            response = await self._delegate.acall(
                messages,
                tools=tools,
                output_model=output_model,
                **kwargs,
            )
        except BaseException as error:
            self._ledger.fail(
                call.call_id,
                reason=f"{type(error).__name__}: {error}",
                response=_exception_response(error),
            )
            raise
        route = _provider_route(response.raw_response)
        self._ledger.receive(
            call.call_id,
            response=_response_value(response),
            usage=response.usage if isinstance(response.usage, dict) else {},
            observed_provider=route.get("provider"),
            observed_model=route.get("model"),
            observed_reasoning_effort=route.get("reasoning_effort"),
        )
        provider_failure = _provider_failure(response.raw_response)
        if provider_failure is None:
            self._ledger.reconcile(call.call_id)
        else:
            self._ledger.reconcile_provider_failure(
                call.call_id,
                reason=provider_failure,
            )
        if self._ledger.state.run_state == "stopped":
            raise RuntimeError(
                f"authorized provider call stopped: {self._ledger.state.stop_reason}"
            )
        return response


def provider_request_value(
    delegate: UnifiedLLM,
    messages: list[dict[str, Any]],
    tools: list[Tool] | None,
    output_model: type[BaseModel] | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build the pinned client's exact token-bearing request before dispatch."""
    if isinstance(delegate, CompletionClient):
        return _completion_request_value(
            delegate,
            messages,
            tools,
            output_model,
            kwargs,
        )
    return _generic_request_value(delegate, messages, tools, output_model, kwargs)


def _completion_request_value(
    delegate: CompletionClient,
    messages: list[dict[str, Any]],
    tools: list[Tool] | None,
    output_model: type[BaseModel] | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Mirror pinned CompletionClient preparation, excluding non-token secrets."""
    call_kwargs = dict(kwargs)
    explicit_cache_points = call_kwargs.pop("cache_control_injection_points", None)
    cache_points = (
        delegate.cache_control_injection_points
        if explicit_cache_points is None
        else explicit_cache_points
    )
    prepared_messages = delegate._inject_cache_control(messages, cache_points)
    request = {
        "model": delegate.model,
        "messages": prepared_messages,
        **_token_bearing_config(delegate.config),
        **call_kwargs,
    }
    if tools:
        request["tools"] = [delegate._convert_tool_to_schema(tool) for tool in tools]
        request["parallel_tool_calls"] = False
    if output_model is not None:
        request["response_format"] = _maybe_sanitize_response_format(
            delegate.model,
            output_model,
        )
    if (
        "tools" not in request
        and _needs_dummy_tool(delegate.model)
        and _messages_have_tool_calls(prepared_messages)
    ):
        request["tools"] = [_DUMMY_TOOL_SCHEMA]
    if "tools" not in request:
        request.pop("tool_choice", None)
        request.pop("parallel_tool_calls", None)
    return _json_safe(request)


def _generic_request_value(
    delegate: UnifiedLLM,
    messages: list[dict[str, Any]],
    tools: list[Tool] | None,
    output_model: type[BaseModel] | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic FakeLLM/test fallback; paid preflight requires CompletionClient."""
    return {
        "model": delegate.model,
        "messages": _json_safe(messages),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_parameter_schema(),
                },
            }
            for tool in tools or ()
        ],
        "output_schema": (
            None if output_model is None else output_model.model_json_schema()
        ),
        "parameters": _json_safe(kwargs),
    }


def require_resource_safe_delegate(delegate: UnifiedLLM) -> None:
    """Paid execution accepts only the pinned, exactly prepared client path."""
    if isinstance(delegate, FakeLLMClient):
        return
    if not isinstance(delegate, CompletionClient):
        raise TypeError("paid V2 execution requires a CompletionClient delegate")


def _disable_endpoint_retries(delegate: UnifiedLLM) -> None:
    if isinstance(delegate, CompletionClient):
        delegate.retry_config = RetryConfig(
            max_retries=0,
            rate_limit_extra_retries=0,
        )


def _token_bearing_config(config: dict[str, Any]) -> dict[str, Any]:
    """Exclude transport credentials/clients that cannot affect model tokens."""
    excluded = {
        "api_key",
        "api_base",
        "base_url",
        "client",
        "timeout",
    }
    return {key: value for key, value in config.items() if key not in excluded}
