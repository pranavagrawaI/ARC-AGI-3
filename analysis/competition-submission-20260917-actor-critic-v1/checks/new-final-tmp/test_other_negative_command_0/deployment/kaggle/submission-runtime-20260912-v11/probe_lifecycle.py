"""Admission ownership for one bounded probe arm, independent of HTTP I/O."""
from contextlib import contextmanager
import threading
import time


class AdmissionWindow:
    def __init__(self, seconds, *, clock=time.monotonic):
        self.clock = clock
        self.deadline = clock() + seconds
        self.reason = None
        self.active = 0
        self.condition = threading.Condition(threading.RLock())

    def close(self, reason):
        # Never acquire the relay's I/O lock here: its request may be blocked
        # at the provider when the main thread receives its deadline signal.
        with self.condition:
            self.reason = self.reason or reason

    @contextmanager
    def admit(self):
        with self.condition:
            if self.clock() >= self.deadline:
                self.reason = self.reason or 'arm deadline'
            if self.reason is not None:
                raise PermissionError(self.reason)
            self.active += 1
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def wait_idle(self, timeout):
        # Completion is transport evidence, not a claim about remote scheduler
        # retirement after a transport error. Never transfer an active arm.
        with self.condition:
            return self.condition.wait_for(lambda: self.active == 0, timeout)


def shutdown_bounded(server, timeout):
    """Do not let a blocked HTTP handler undo an earlier cleanup deadline."""
    completed = threading.Event()

    def stop():
        server.shutdown()
        server.server_close()
        completed.set()

    thread = threading.Thread(target=stop, daemon=True)
    thread.start()
    return completed.wait(timeout)
