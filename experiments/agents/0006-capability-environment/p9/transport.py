"""Transport wrapper that makes retries host-visible and model-invisible.

``OpenAICompatibleChatClient`` already retries transient transport failures, but
it does so with no record: the run evidence cannot show that two requests were
sent for one deliberation round. This wrapper moves the retry loop out of the
client (constructed with ``rate_limit_retries=0``) and logs every attempt.

Failures the client marked ``retryable`` are retried, and so are malformed
response bodies the client marks fatal — a body that will not parse as JSON means
no completion was produced, and a chat completion is stateless, so re-sending is
idempotent. That case cost a live run 12 actions of work before it was widened.
Anything else propagates to the agent, which records ``model_request_failed`` and stops. No
transport failure of either kind reaches the model: a transport failure is not
model-repairable, and surfacing it would put host-authored infrastructure text
into the model's context.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from llm_repl_agent.client import ChatClient, ChatResponse, ModelRequestError
from llm_repl_agent.process_log import ProcessLog

from .failures import RETRIED_TRANSPORT_KINDS, classify_transport_error


class RetryingChatClient:
    """Retry only demonstrably retryable transport failures, and log each attempt."""

    def __init__(
        self,
        inner: ChatClient,
        *,
        process_log: ProcessLog,
        max_retries: int = 2,
        backoff_seconds: float = 0.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")
        self.inner = inner
        self.process_log = process_log
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._attempts = 0
        self._retries = 0

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def retries(self) -> int:
        return self._retries

    def policy(self) -> dict[str, Any]:
        return {
            "retried_kinds": sorted(RETRIED_TRANSPORT_KINDS),
            "max_retries": self.max_retries,
            "backoff_seconds": self.backoff_seconds,
            "model_visible": False,
            "attempt_event_types": [
                "model_transport_attempt_failed",
                "model_transport_retry",
            ],
        }

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        attempt = 0
        while True:
            self._attempts += 1
            try:
                return self.inner.complete(messages=messages, tools=tools)
            except ModelRequestError as error:
                kind = classify_transport_error(
                    retryable=bool(error.retryable),
                    http_status=error.http_status,
                    response_body=error.response_body_raw,
                )
                self.process_log.write(
                    "model_transport_attempt_failed",
                    attempt=attempt + 1,
                    failure_kind=kind,
                    http_status=error.http_status,
                    error=str(error),
                    request_payload_sha256=error.request_payload_sha256,
                    response_body_sha256=error.response_body_sha256,
                )
                if kind not in RETRIED_TRANSPORT_KINDS or attempt >= self.max_retries:
                    raise
                attempt += 1
                self._retries += 1
                delay = self.backoff_seconds * (2 ** (attempt - 1))
                self.process_log.write(
                    "model_transport_retry",
                    attempt=attempt,
                    delay_seconds=delay,
                    failure_kind=kind,
                    model_visible=False,
                )
                if delay:
                    time.sleep(delay)
