"""CPU source-path fixture: extract native methods unchanged, replace GPU collaborators.

The first native poll is parked until TERM, then delayed 50ms, so both arms
exercise the observed sleeping-watchdog ordering deterministically. Remaining
asynchronous five-second poll waits are time-compressed to 50ms. No native
engine, tokenizer, model, GPU or socket is loaded. The watchdog and signal methods
are compiled from the pinned wheel source copies without editing their ASTs.
"""
import ast
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import List, Optional

HERE = Path(__file__).resolve().parent
PIN = HERE.parent / 'trial10-release-local-20261023/sources'


def emit(path, event, **fields):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(dict(t=time.monotonic(), event=event, pid=os.getpid(), **fields))+'\n')


def extract(path, names, namespace):
    nodes = []
    for item in ast.parse(path.read_text()).body:
        if isinstance(item, ast.ClassDef):
            if item.name in names:
                nodes.append(item)
            else:
                nodes.extend(n for n in item.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names)
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def frontend(path, stubborn=False):
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(20)'])
    class Proc:
        pid = child.pid
        def is_alive(self): return child.poll() is None
        @property
        def exitcode(self): return child.poll()
    class Log:
        def error(self, message, *args, **kwargs): emit(path, 'native_error', message=message % args if args else message)
        def warning(self, message, *args, **kwargs): emit(path, 'native_term', message=message)
        def info(self, message, *args, **kwargs): emit(path, 'native_info', message=message)
    ns = dict(List=List, Optional=Optional, Process=Proc, threading=threading,
              os=os, signal=signal, logger=Log(), time=time, sys=sys,
              asyncio=None,
              ServerStatus=SimpleNamespace(UnHealthy='unhealthy'),
              get_bool_env_var=lambda name: False, ShutdownReq=lambda: 'shutdown',
              collect_scheduler_processes=lambda: [child] if child.poll() is None else [])
    def kill_process_tree(pid, include_parent=True):
        emit(path, 'native_final_cleanup')
        if child.poll() is None: child.kill()
        child.wait(timeout=1)
        if include_parent: raise SystemExit(0)
    ns['kill_process_tree'] = kill_process_tree
    extract(PIN/'pinned-sglang-srt-utils-watchdog.py', ['SubprocessWatchdog'], ns)
    extract(PIN/'pinned-sglang-srt-managers-tokenizer_manager.py', ['SignalHandler', 'sigterm_watchdog'], ns)
    watchdog = ns['SubprocessWatchdog']([Proc()], ['detokenizer'], interval=.01)
    original_stop = watchdog.stop
    def stop():
        emit(path, 'watchdog_stopped')
        original_stop()
    watchdog.stop = stop
    def dispatch(request):
        emit(path, 'scheduler_shutdown_dispatched')
        child.terminate()
    manager = SimpleNamespace(gracefully_exit=False, rid_to_state={}, server_status='up',
                              _subprocess_watchdog=watchdog, _dispatch_to_scheduler=dispatch,
                              dump_requests_before_crash=lambda: emit(path, 'crash_path'),
                              force_exit_handler=lambda: None)
    first_poll=True
    async def native_sleep(seconds):
        nonlocal first_poll
        if first_poll:
            first_poll=False
            emit(path,'native_poll_parked')
            while not manager.gracefully_exit:
                await asyncio.sleep(.005)
        await asyncio.sleep(seconds/100)
    ns['asyncio']=SimpleNamespace(sleep=native_sleep)
    handler = ns['SignalHandler'](manager)
    signal.signal(signal.SIGTERM, signal.SIG_IGN if stubborn else handler.sigterm_handler)
    signal.signal(signal.SIGQUIT, handler.running_phase_sigquit_handler)
    watchdog.start()
    def ready():
        emit(path, 'frontend_ready', child=child.pid, stubborn=stubborn,
             guard_active='sitecustomize' in sys.modules)
    async def run():
        task=asyncio.create_task(ns['sigterm_watchdog'](manager))
        await asyncio.sleep(0) # source method reaches its first await before ready
        ready()
        await task
    try:
        if stubborn:
            ready()
            time.sleep(20)
        else:
            asyncio.run(run())
    finally:
        if child.poll() is None: child.kill()
        child.wait(timeout=1)


if __name__ == '__main__':
    path = sys.argv[2]
    if sys.argv[1] == 'wrapper':
        child = subprocess.Popen([sys.executable, __file__, 'frontend', path, *sys.argv[3:]])
        emit(path, 'wrapper_ready', frontend=child.pid)
        raise SystemExit(child.wait())
    frontend(path, '--stubborn' in sys.argv)
