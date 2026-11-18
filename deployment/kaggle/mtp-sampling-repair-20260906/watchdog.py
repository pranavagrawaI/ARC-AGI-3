"""Fail promptly on a dead engine while preserving bounded HTTP waits."""

import queue
import threading
import time


def guarded_call(call, server, timeout=900):
    results = queue.Queue(maxsize=1)

    def execute():
        try:
            results.put((True, call()))
        except BaseException as error:
            results.put((False, error))

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while True:
        code = server.poll()
        if code is not None:
            raise RuntimeError(
                f"Engine exited with code {code}; stop this trial and inspect server.log"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Generation deadline exhausted")
        try:
            good, value = results.get(timeout=min(0.1, remaining))
        except queue.Empty:
            continue
        if good:
            return value
        raise value
