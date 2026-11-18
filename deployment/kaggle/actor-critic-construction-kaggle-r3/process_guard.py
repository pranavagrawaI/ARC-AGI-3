"""Dedicated Linux subreaper. Owns all descendants, including setsid workers."""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def children():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            result[int(path.parent.name)] = (int(fields[1]), fields[0], fields[19])
        except (OSError, ValueError, IndexError):
            pass
    owned = {os.getpid()}
    while True:
        new = {pid for pid, value in result.items() if value[0] in owned}
        if new <= owned:
            break
        owned |= new
    return {pid: result[pid] for pid in owned if pid != os.getpid() and pid in result}


def stop_children():
    killed = set()
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        live = children()
        if not live:
            return sorted(killed)
        for pid in live:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.add(pid)
            except ProcessLookupError:
                pass
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                break
        time.sleep(.02)
    raise RuntimeError('owned descendants did not retire: ' + repr(children()))


def main():
    parent, seconds, receipt, *command = sys.argv[1:]
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) or libc.prctl(1, signal.SIGTERM, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'subreaper/PDEATHSIG')
    def abort(*_):
        raise TimeoutError('outer parent died or wall expired')
    signal.signal(signal.SIGTERM, abort)
    signal.signal(signal.SIGALRM, abort)
    record = {'status': 'failed', 'command': command, 'parent': int(parent)}
    process = None
    try:
        if os.getppid() != int(parent):
            raise RuntimeError('parent death race')
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True)
        record['child'] = process.pid
        rc = process.wait()
        record['returncode'] = rc
        record['status'] = 'passed' if rc == 0 else 'failed'
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            record['killed_remaining_descendants'] = stop_children()
            record['cleanup'] = 'retired'
            if record['killed_remaining_descendants'] and record['status'] == 'passed':
                record['status'] = 'failed'
                record['error'] = 'successful child left live descendants'
        except BaseException as error:
            record['status'] = 'failed'
            record['cleanup'] = repr(error)
        Path(receipt).write_text(json.dumps(record, indent=2) + '\n')
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
