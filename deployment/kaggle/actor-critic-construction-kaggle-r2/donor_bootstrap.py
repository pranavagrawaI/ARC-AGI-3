"""Verified prerequisite composition. No fixture branch or donor game entry."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def verify():
    expected = json.loads((HERE / 'ASSET-HASHES.json').read_text())
    for name, digest in expected.items():
        path = HERE / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('runtime asset drift: ' + name)


def main():
    verify()
    os.environ['ARC_DISABLE_DOTENV'] = '1'
    os.environ['ARC_NOTEBOOK_WORK_DEADLINE'] = str(time.monotonic() + 3100)
    # Actual trusted GPU discovery and sealed runner precede installation/server warmup.
    from kaggle_probe import inspect_gpu, require_seal
    inspect_gpu(HERE)
    require_seal(HERE)
    runpy.run_path(str(HERE / 'runtime_install.py'), run_name='__main__')
    command = [sys.executable, str(HERE / 'official_launcher_runtime.py'),
        '--output-dir', '/kaggle/working/model-runtime',
        '--manifest', str(HERE / 'model-metadata.json'), '--input-root', '/kaggle/input',
        '--model-source', 'jakobbrggen/qwen3-8-27b-fp8-hf-snapshot/versions/1',
        '--runtime-provenance', '/tmp/sglang-runtime-provenance.json',
        '--startup-timeout-s', '1200', '--request-timeout-s', '180',
        '--launcher-timeout-s', '1020', '--attention-backend', 'triton',
        '--sampling-backend', 'pytorch', '--profile-imports', '--r8-l3b',
        '--official-launcher-probe', str(HERE / 'kaggle_probe.py'),
        '--launcher-python', '/tmp/solver-venv/bin/python']
    subprocess.run(command, check=True, timeout=max(1, float(os.environ['ARC_NOTEBOOK_WORK_DEADLINE']) - time.monotonic()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        (HERE / 'BOOTSTRAP-FAILURE.json').write_text(json.dumps({'status': 'failed', 'error': f'{type(error).__name__}: {error}'}) + '\n')
        raise
