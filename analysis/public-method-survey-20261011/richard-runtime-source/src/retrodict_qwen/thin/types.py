"""Verbatim from thinharness v0.5.4 ``thinharness/types.py``: Json, RunUsage, HarnessError."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

Json = dict[str, Any]


@dataclass
class RunUsage:
    """Provider and tool usage for one harness run."""

    model_requests: int = 0
    tool_calls: int = 0
    cancelled_tool_calls: int = 0
    output_retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    tool_retries: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> Json:
        """Serialize run usage for a JSON envelope."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: Any, *, label: str) -> RunUsage:
        """Validate and decode run usage from a JSON envelope.

        The four counter fields are required; token totals default to 0 so
        envelopes written before token accounting existed still resume; unknown
        keys are ignored.
        """
        if not isinstance(data, dict):
            raise HarnessError(f"{label} field 'usage' has wrong type")
        usage = dict(data)
        retries = usage.get("tool_retries", {})
        if not isinstance(retries, dict) or not all(isinstance(key, str) and isinstance(val, int) for key, val in retries.items()):
            raise HarnessError(f"{label} field 'usage' has wrong type")
        model_requests = usage.get("model_requests")
        tool_calls = usage.get("tool_calls")
        cancelled_tool_calls = usage.get("cancelled_tool_calls")
        output_retries = usage.get("output_retries")
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)
        counters = [model_requests, tool_calls, cancelled_tool_calls, output_retries, input_tokens, output_tokens, cached_tokens]
        if not all(isinstance(value, int) for value in counters):
            raise HarnessError(f"{label} field 'usage' has wrong type")
        assert isinstance(model_requests, int)
        assert isinstance(tool_calls, int)
        assert isinstance(cancelled_tool_calls, int)
        assert isinstance(output_retries, int)
        assert isinstance(input_tokens, int)
        assert isinstance(output_tokens, int)
        assert isinstance(cached_tokens, int)
        return cls(
            model_requests=model_requests,
            tool_calls=tool_calls,
            cancelled_tool_calls=cancelled_tool_calls,
            output_retries=output_retries,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            tool_retries=dict(retries),
        )


class HarnessError(RuntimeError):
    """Raised when the harness cannot complete a run."""
