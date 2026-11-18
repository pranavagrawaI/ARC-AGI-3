"""Probe-only observation, inherited by the real worker and headless child.

No cancellation, timeouts, queues, results or control flow are replaced.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import CodeType

directory = os.environ.get("OWNERSHIP_TRACE_DIR")
if directory:
    trace_fd = os.open(
        str(Path(directory) / f"trace-{os.getpid()}.jsonl"),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )

    def emit(event, **fields):
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        os.write(
            trace_fd,
            (
                json.dumps(
                    {
                        "event": event,
                        "pid": os.getpid(),
                        "monotonic_ns": time.monotonic_ns(),
                        "task_id": id(task) if task else None,
                        **fields,
                    }
                )
                + "\n"
            ).encode(),
        )

    def local_only(event, args):
        if event == "socket.connect":
            address = args[1]
            if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1"}:
                raise RuntimeError("Worker probe forbids non-loopback connections")

    sys.addaudithook(local_only)
    emit("process_started", argv=sys.argv)
    sources = {}
    witnessed = set()

    def matching_code(code, target):
        if code.co_name == target.co_name and code.co_firstlineno == target.co_firstlineno:
            return code == target
        return any(matching_code(child, target) for child in code.co_consts if isinstance(child, CodeType))

    def observe(frame, event, arg):
        path = frame.f_code.co_filename
        base = os.path.basename(path)
        function = frame.f_code.co_name
        relevant = (
            (
                base == "nooa_worker.py"
                and function in {"_wait_batch", "serve", "finish", "close"}
            )
            or (base == "headless.py" and function == "_headless_dispatch")
            or (base == "solver_agent.py" and function == "submit_actions")
        )
        if not relevant:
            return None
        if path not in witnessed:
            data = Path(path).read_bytes()
            emit("loaded_source", path=path, basename=base, source_sha256=hashlib.sha256(data).hexdigest(),
                 function=function, code_matches_compiled_source=matching_code(compile(data,path,"exec"),frame.f_code))
            witnessed.add(path)
        if event == "line":
            if path not in sources:
                sources[path] = Path(path).read_text().splitlines()
            line = sources[path][frame.f_lineno - 1].strip()
            if function == "_wait_batch" and line.startswith("time.sleep("):
                emit(
                    "host_poll",
                    deadline=frame.f_locals.get("deadline"),
                    operation_id=id(frame),
                )
            elif function == "_headless_dispatch":
                if "await agent.handle(notification)" in line:
                    emit(
                        "handle_started",
                        notification_channels=list(
                            frame.f_locals.get("notification", {})
                        ),
                    )
                elif line == "consecutive_errors = 0" and "result" in frame.f_locals:
                    result = frame.f_locals["result"]
                    emit(
                        "handle_returned",
                        result_kind=str(getattr(result, "kind", None)),
                    )
                elif "await agent.queue_manager.race()" in line:
                    emit("dispatcher_waiting")
                elif (
                    line == "continue"
                    and 'if state.get("turn")' in sources[path][frame.f_lineno - 2]
                ):
                    emit(
                        "stale_state_notification_skipped",
                        turn=frame.f_locals["state"].get("turn"),
                    )
            elif (
                function == "submit_actions" and "with self._actions_path.open" in line
            ):
                emit("action_commit_entered", turn=frame.f_locals.get("turn"))
        return observe

    sys.settrace(observe)
