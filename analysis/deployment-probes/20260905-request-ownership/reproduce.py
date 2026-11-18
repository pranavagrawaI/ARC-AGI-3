#!/usr/bin/env python3
"""CPU control-flow probes using unchanged methods from manifest-verified wheels.

This is not an HTTP/SGLang/GPU reproduction. Tokenization, IPC and scheduler
execution are fakes; generate_request, response wait, cleanup and abort are real
wheel methods. Assertions characterize the current behavior, including defects.
"""

from __future__ import annotations

import ast
import asyncio
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace as NS
import zipfile

ROOT = Path(__file__).resolve().parents[3]
RUN = (
    ROOT
    / "analysis/deployment-probes/20260905T-nooa-join-public3-reasoning-v2/kaggle-output"
)
WHEEL = (
    RUN
    / "qwen38-fp8-r8l3b-bootstrap/sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl"
)
NOOA = Path(
    "/home/pranav/.cache/uv/sdists-v9/git/d261e076395c2c92/55a4b57872b49bfe/nooa-0.0.1.dev95-py3-none-any.whl"
)
ENTRY = "sglang/srt/managers/tokenizer_manager.py"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_methods(wheel, entry, names, namespace, class_name=None):
    with zipfile.ZipFile(wheel) as archive:
        source = archive.read(entry)
    tree = ast.parse(source)
    nodes = tree.body
    if class_name:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    selected = [
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    assert {n.name for n in selected} == set(names)
    # Future annotations allow loading methods without GPU-related import graphs.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), f"{wheel}!{entry}", "exec"),
        namespace,
    )
    return {n.name: namespace[n.name] for n in selected}, {
        "entry": entry,
        "sha256": hashlib.sha256(source).hexdigest(),
        "methods": {n.name: [n.lineno, n.end_lineno] for n in selected},
    }


class Request:
    is_single = True
    max_thinking_tokens = None
    routed_dp_rank = None
    return_prompt_token_ids = False
    background = False

    def __init__(self, rid, stream):
        self.rid, self.stream = rid, stream

    def normalize_batch_and_arguments(self):
        pass


class BackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, fn):
        self.tasks.append(fn)

    async def __call__(self):
        for task in self.tasks:
            await task()


@asynccontextmanager
async def reader_lock():
    yield


async def disconnected():
    return True


def make_manager(methods, case):
    manager = type("WheelManager", (), methods)()
    events = []
    states = {}

    class TrackedStates(dict):
        def pop(self, rid, default=None):
            events.append(["tokenizer_state_removed", rid])
            return super().pop(rid, default)

    manager.rid_to_state = TrackedStates()
    manager.enable_metrics = False
    manager.incremental_streaming_output = False
    manager.is_pause = False
    manager.is_pause_cond = asyncio.Condition()
    manager.model_update_lock = NS(reader_lock=reader_lock())
    manager.request_logger = NS(
        log_received_request=lambda *a: None, log_finished_request=lambda *a, **k: None
    )
    manager.request_metrics_exporter_manager = NS(exporter_enabled=lambda: False)
    manager.tokenizer = None
    manager.admitted = asyncio.Event()
    manager.auto_create_handle_loop = lambda: None
    manager._set_default_priority = lambda obj: None

    def init(obj, request):
        state = NS(
            event=asyncio.Event(),
            finished=False,
            out_list=[],
            time_stats=NS(response_sent_to_client_time=1),
        )
        states[obj.rid] = state
        manager.rid_to_state[obj.rid] = state
        events.append(["tokenizer_state_created", obj.rid])

    async def validate(obj):
        if case == "pre_dispatch_failure":
            raise ValueError("injected input validation failure")

    async def tokenize(obj):
        return obj

    def send(obj):
        events.append(["fake_scheduler_admitted", obj.rid])
        manager.admitted.set()

    def dispatch(req):
        events.append(["abort_dispatched", req.rid, req.rid in manager.rid_to_state])

    manager._init_req_state = init
    manager._validate_and_resolve_lora = validate
    manager._tokenize_one_request = tokenize
    manager._send_one_request = send
    manager._dispatch_to_scheduler = dispatch
    return manager, states, events


async def manager_case(methods, case):
    manager, states, events = make_manager(methods, case)
    obj = Request(case, stream=case in {"cancel_during_wait", "generator_close"})
    raw = (
        NS(is_disconnected=disconnected)
        if case.startswith("nonstream_disconnect")
        else None
    )
    gen = manager.generate_request(obj, raw)
    task = asyncio.create_task(anext(gen))
    if case == "pre_dispatch_failure":
        try:
            await task
        except ValueError:
            pass
    else:
        await asyncio.wait_for(manager.admitted.wait(), 1)
        state = states[obj.rid]
        if case == "cancel_during_wait":
            events.append(["caller_cancel_requested", obj.rid])
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                events.append(["caller_cancel_observed", obj.rid])
        elif case == "nonstream_disconnect_wait":
            try:
                await asyncio.wait_for(task, 1)
            except ValueError:
                events.append(["disconnect_detected", obj.rid])
        else:
            state.out_list.append(
                {"text": "token", "meta_info": {"finish_reason": None}}
            )
            if case == "normal_completion":
                state.finished = True
                # Normal state removal belongs to _handle_batch_output (not executed).
                manager.rid_to_state.pop(obj.rid)
            state.event.set()
            try:
                await asyncio.wait_for(task, 1)
            except ValueError:
                events.append(["disconnect_detected", obj.rid])
            else:
                if case == "generator_close":
                    events.append(["generator_close_requested", obj.rid])
                    await gen.aclose()
                else:
                    try:
                        await anext(gen)
                    except StopAsyncIteration:
                        pass
        # Execute wheel's real two-second delayed callback, after cleanup.
        events.append(["delayed_abort_started", obj.rid])
        await manager.create_abort_task(obj)()
    aborts = [event for event in events if event[0] == "abort_dispatched"]
    expected = 1 if case.startswith("nonstream_disconnect") else 0
    assert len(aborts) == expected, (case, events)
    assert not manager.rid_to_state, (case, events)
    return {"case": case, "abort_dispatch_count": len(aborts), "events": events}


async def shield_case():
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []

    async def fake_provider(**kwargs):
        events.append("fake_provider_started")
        started.set()
        await release.wait()
        events.append("fake_provider_completed")
        finished.set()
        return "late result"

    methods, identity = load_methods(
        NOOA,
        "nooa/unifiedllm/unifiedllm.py",
        ["_litellm_acompletion", "_consume_litellm_acompletion_result"],
        {"asyncio": asyncio, "litellm": NS(acompletion=fake_provider)},
    )
    task = asyncio.create_task(methods["_litellm_acompletion"]({}))
    await started.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        events.append("caller_cancelled")
    assert not finished.is_set()
    release.set()
    await asyncio.wait_for(finished.wait(), 1)
    await asyncio.sleep(0)
    return {
        "identity": identity,
        "events": events,
        "provider_survived_caller_cancellation": True,
    }


def log_summary():
    import re

    path = RUN / "nooa-join-public3/server.log"
    by_rid = defaultdict(list)
    disconnects = Counter()
    for number, line in enumerate(path.open(), 1):
        match = re.search(
            r"Received output for rid='([^']+)' but the state was deleted", line
        )
        if match:
            by_rid[match[1]].append({"line": number, "timestamp": line[1:20]})
        if "Request is disconnected from the client side" in line:
            disconnects[line.strip()] += 1
    counts = Counter(len(items) for items in by_rid.values())
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "deleted_state_messages": sum(k * v for k, v in counts.items()),
        "distinct_rids": len(by_rid),
        "messages_per_rid_histogram": dict(counts),
        "per_rid": dict(by_rid),
        "disconnect_log_lines": sum(disconnects.values()),
    }


async def main():
    manifest = json.loads(
        (ROOT / "deployment/kaggle/runtime-patches/r8l3b/manifest.json").read_text()
    )
    assert digest(WHEEL) == manifest["participants"]["wheel"]["sha256"]
    nooa_manifest = json.loads(
        (
            ROOT
            / "deployment/kaggle/tycho-nooa-full-wheelhouse-v11/wheelhouse-manifest.json"
        ).read_text()
    )
    # Pin independently of container installation; its sources may have changed.
    assert (
        digest(NOOA)
        == "f2217a8d95d2239ca830a0d47ec7973123f618253a6c219fa64e1e43f278cf0c"
    )
    assert digest(NOOA) in json.dumps(nooa_manifest)
    namespace = {
        "asyncio": asyncio,
        "GenerateReqInput": Request,
        "get_serving": lambda: NS(tokenizer_worker_num=1),
        "get_disagg": lambda: NS(language_only=False),
        "AbortReq": NS,
        "BackgroundTasks": BackgroundTasks,
        "logger": logging.getLogger("probe"),
        "_REQUEST_STATE_WAIT_TIMEOUT": 0.05,
    }
    methods, identity = load_methods(
        WHEEL,
        ENTRY,
        [
            "generate_request",
            "_wait_one_response",
            "_discard_pending_req_states",
            "abort_request",
            "create_abort_task",
        ],
        namespace,
        "TokenizerManager",
    )
    assert identity["sha256"] == manifest["source_projection"][ENTRY]["patched_sha256"]
    cases = [
        "cancel_during_wait",
        "generator_close",
        "nonstream_disconnect_wait",
        "nonstream_disconnect_running",
        "normal_completion",
        "pre_dispatch_failure",
    ]
    results = await asyncio.gather(*(manager_case(methods, case) for case in cases))
    print(
        json.dumps(
            {
                "scope": "CPU control flow; fake tokenizer/IPC/scheduler; no GPU or HTTP",
                "sglang_wheel_sha256": digest(WHEEL),
                "nooa_wheel_sha256": digest(NOOA),
                "tokenizer_manager": identity,
                "cases": results,
                "nooa_shield": await shield_case(),
                "saved_log": log_summary(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
