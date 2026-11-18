#!/usr/bin/env python3
"""Exercise the frozen NOOA client through LiteLLM against a loopback endpoint.

No model, GPU or outer Actor/worker is run. Timeouts are shortened experimental
controls. The endpoint is deliberately not cancellation-aware like a real engine.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from reproduce import NOOA, digest

LITELLM = Path("/home/pranav/.cache/uv/archive-v0/_HySxVjE4sG4S8gi")
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["DO_NOT_TRACK"] = "1"
sys.path[:0] = [str(NOOA), str(LITELLM)]


def local_only(event, args):
    if event == "socket.connect":
        address = args[1]
        if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1"}:
            raise RuntimeError("Probe forbids non-loopback connections")


sys.addaudithook(local_only)
from nooa.unifiedllm import CompletionClient, RetryConfig  # noqa: E402
import litellm  # noqa: E402

litellm.suppress_debug_info = True  # Keep stdout machine-readable on expected timeouts.


async def run_case(
    name, nooa_retries, sdk_retries, *, delay=0.35, cancel=False, explicit_retry=False
):
    events, handlers = [], set()
    admitted = asyncio.Event()
    provider_retired = asyncio.Event()
    start = time.monotonic()

    def record(event, **fields):
        events.append(
            {
                "event": event,
                "monotonic_offset_s": round(time.monotonic() - start, 6),
                **fields,
            }
        )

    unified = importlib.import_module("nooa.unifiedllm.unifiedllm")
    original_consume = unified._consume_litellm_acompletion_result

    def observed_consume(task):
        # Observe the existing discard callback without changing its behavior.
        record(
            "provider_result_discarded",
            cancelled=task.cancelled(),
            failed=False if task.cancelled() else task.exception() is not None,
        )
        original_consume(task)
        provider_retired.set()

    if cancel:
        unified._consume_litellm_acompletion_result = observed_consume

    async def handle(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            lines = header.decode().split("\r\n")
            headers = dict(line.split(": ", 1) for line in lines[1:] if ": " in line)
            headers = {k.lower(): v for k, v in headers.items()}
            body = await reader.readexactly(int(headers["content-length"]))
            payload = json.loads(body)
            attempt = sum(e["event"] == "http_attempt_admitted" for e in events) + 1
            record(
                "http_attempt_admitted",
                attempt=attempt,
                stream=payload.get("stream", False),
                prompt_sha256=hashlib.sha256(
                    json.dumps(payload["messages"], sort_keys=True).encode()
                ).hexdigest(),
                sdk_retry_header=headers.get("x-stainless-retry-count"),
            )
            admitted.set()
            await asyncio.sleep(delay)
            record("fake_work_completed", attempt=attempt)
            response = json.dumps(
                {
                    "id": f"probe-{name}-{attempt}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "ownership-probe",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                + str(len(response)).encode()
                + b"\r\n\r\n"
                + response
            )
            try:
                await writer.drain()
                # A successful write does NOT establish client receipt.
                record("response_write_completed_delivery_unknown", attempt=attempt)
            except (ConnectionError, OSError):
                record("response_write_failed", attempt=attempt)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            handlers.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    params = (
        {} if sdk_retries is None else {"max_retries": sdk_retries, "num_retries": 0}
    )
    client = CompletionClient(
        model="openai/ownership-probe",
        api_base=f"http://127.0.0.1:{port}/v1",
        api_key="local-probe-only",
        timeout=2.0 if cancel else 0.08,
        retry_config=RetryConfig(
            max_retries=nooa_retries,
            rate_limit_extra_retries=0,
            base_delay=0,
            max_delay=0,
            jitter_factor=0,
        ),
        **params,
    )
    try:
        record("logical_generation_started", logical_generation_id=name)
        task = asyncio.create_task(
            client.acall([{"role": "user", "content": "Return ok."}])
        )
        if cancel:
            await asyncio.wait_for(admitted.wait(), 5)
            record("caller_cancel_requested")
            task.cancel()
        try:
            await asyncio.wait_for(task, 30)
            record("result_delivered")
        except asyncio.CancelledError:
            record("caller_cancel_observed")
        except Exception as exc:
            record(
                "caller_failed",
                exception_type=type(exc).__name__,
                status_code=getattr(exc, "status_code", None),
            )
        await asyncio.sleep(delay + 0.1)
        if handlers:
            await asyncio.wait_for(asyncio.gather(*list(handlers)), 5)
        if explicit_retry:
            # The endpoint's completion event is the retirement witness in this
            # fake-only control. A real scheduler needs its own terminal witness.
            assert not handlers
            record("owned_retry_after_fake_retirement")
            try:
                await client.acall([{"role": "user", "content": "Return ok."}])
            except Exception as exc:
                record(
                    "caller_failed",
                    exception_type=type(exc).__name__,
                    status_code=getattr(exc, "status_code", None),
                )
            await asyncio.sleep(delay + 0.1)
            if handlers:
                await asyncio.wait_for(asyncio.gather(*list(handlers)), 5)
        if cancel:
            await asyncio.wait_for(provider_retired.wait(), 5)
        record("probe_cleanup_started")
    finally:
        unified._consume_litellm_acompletion_result = original_consume
        server.close()
        await server.wait_closed()
        await client.aclose()
    return {
        "case": name,
        "nooa_retries": nooa_retries,
        "sdk_retries": sdk_retries,
        "attempts": sum(e["event"] == "http_attempt_admitted" for e in events),
        "events": events,
    }


async def main():
    assert (
        digest(NOOA)
        == "f2217a8d95d2239ca830a0d47ec7973123f618253a6c219fa64e1e43f278cf0c"
    )
    versions = {
        p: importlib.metadata.version(p) for p in ["nooa", "litellm", "openai", "httpx"]
    }
    assert versions == {
        "nooa": "0.0.1.dev95",
        "litellm": "1.97.0",
        "openai": "2.54.0",
        "httpx": "0.28.1",
    }
    results = []
    for spec in [
        ("normal", 0, 0, {"delay": 0}),
        ("no_retries", 0, 0, {}),
        ("sdk_default", 0, None, {}),
        ("nooa_retry_eligibility", 1, 0, {}),
        ("one_owned_retry", 0, 0, {"explicit_retry": True}),
        ("caller_cancellation", 0, 0, {"cancel": True}),
    ]:
        name, outer, inner, options = spec
        results.append(await run_case(name, outer, inner, **options))
    # Observed LiteLLM Timeout carries status 408; this pinned NOOA retry
    # classifier excludes it before checking timeout names. Do not assume retry.
    assert [r["attempts"] for r in results] == [1, 1, 3, 1, 2, 1], results
    assert any(e["event"] == "result_delivered" for e in results[0]["events"])
    cancel_events = [e["event"] for e in results[-1]["events"]]
    assert cancel_events.index("caller_cancel_observed") < cancel_events.index(
        "fake_work_completed"
    )
    assert cancel_events.index("caller_cancel_observed") < cancel_events.index(
        "provider_result_discarded"
    )
    print(
        json.dumps(
            {
                "scope": "real NOOA CompletionClient and LiteLLM/OpenAI HTTP; fake endpoint; no outer Actor or GPU",
                "versions": versions,
                "nooa_wheel_sha256": digest(NOOA),
                "litellm_source": litellm.__file__,
                "cases": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
