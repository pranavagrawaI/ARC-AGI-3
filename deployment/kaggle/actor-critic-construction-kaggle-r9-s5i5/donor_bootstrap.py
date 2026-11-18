"""Verified prerequisite composition. No fixture branch or donor game entry."""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PR_SET_CHILD_SUBREAPER = 36


def verify():
    expected = json.loads((HERE / 'ASSET-HASHES.json').read_text())
    for name, digest in expected.items():
        path = HERE / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('runtime asset drift: ' + name)


def become_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')


def owned_descendants():
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


def descendant_identity(pid):
    record = {'pid': pid}
    try:
        record['comm'] = Path(f'/proc/{pid}/comm').read_text().strip()
    except OSError:
        record['comm'] = None
    try:
        record['cmdline'] = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode('utf-8', 'replace').strip()
    except OSError:
        record['cmdline'] = None
    return record


def reap_owned_descendants(receipt=None, timeout=30):
    """Reap leftover supervisor/SGLang/worker orphans before bootstrap success."""
    killed = []
    seen = {pid: descendant_identity(pid) for pid in owned_descendants()}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        live = owned_descendants()
        if not live:
            record = {'status': 'retired', 'seen': list(seen.values()), 'reaped': killed, 'remaining': []}
            if receipt is not None:
                Path(receipt).write_text(json.dumps(record, indent=2) + '\n')
            return record
        for pid in live:
            if pid not in seen:
                seen[pid] = descendant_identity(pid)
            try:
                os.kill(pid, signal.SIGKILL)
                if pid not in killed:
                    killed.append(pid)
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
    remaining = [descendant_identity(pid) for pid in owned_descendants()]
    record = {'status': 'failed', 'seen': list(seen.values()), 'reaped': killed, 'remaining': remaining}
    if receipt is not None:
        Path(receipt).write_text(json.dumps(record, indent=2) + '\n')
    raise RuntimeError('live descendants after trusted shutdown: ' + repr(remaining))


def main():
    verify()
    become_subreaper()
    os.environ['ARC_DISABLE_DOTENV'] = '1'
    os.environ['ARC_NOTEBOOK_WORK_DEADLINE'] = str(time.monotonic() + 3100)
    # Actual trusted GPU discovery and sealed runner precede installation/server warmup.
    from gameplay_play import inspect_gpu, require_seal
    inspect_gpu(HERE)
    require_seal(HERE)
    from game_preflight import qualify
    qualify(HERE / 'GAME-PREFLIGHT.json')
    runpy.run_path(str(HERE / 'runtime_install.py'), run_name='__main__')
    command = [sys.executable, str(HERE / 'official_launcher_runtime.py'),
        '--output-dir', '/kaggle/working/model-runtime',
        '--manifest', str(HERE / 'model-metadata.json'), '--input-root', '/kaggle/input',
        '--model-source', 'jakobbrggen/qwen3-8-27b-fp8-hf-snapshot/versions/1',
        '--runtime-provenance', '/tmp/sglang-runtime-provenance.json',
        '--startup-timeout-s', '1200', '--request-timeout-s', '180',
        '--launcher-timeout-s', '1800', '--attention-backend', 'triton',
        '--sampling-backend', 'pytorch', '--profile-imports', '--r8-l3b',
        '--official-launcher-probe', str(HERE / 'gameplay_play.py'),
        '--launcher-python', '/tmp/solver-venv/bin/python']
    # Only the trusted supervisor/server tree receives validated driver paths.
    # Never restore ambient notebook loader state or change candidate defaults.
    from accelerator import loader_environment
    trusted_loader, _ = loader_environment()
    supervisor_environment = os.environ.copy()
    supervisor_environment.pop('LD_LIBRARY_PATH', None)
    if 'LD_LIBRARY_PATH' in trusted_loader:
        supervisor_environment['LD_LIBRARY_PATH'] = trusted_loader['LD_LIBRARY_PATH']
    try:
        subprocess.run(command, env=supervisor_environment, check=True, timeout=max(1, float(os.environ['ARC_NOTEBOOK_WORK_DEADLINE']) - time.monotonic()))
    finally:
        reap_owned_descendants(HERE / 'SHUTDOWN-REAP.json')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        (HERE / 'BOOTSTRAP-FAILURE.json').write_text(json.dumps({'status': 'failed', 'error': f'{type(error).__name__}: {error}'}) + '\n')
        raise
