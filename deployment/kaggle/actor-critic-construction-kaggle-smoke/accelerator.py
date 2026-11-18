"""Trusted-only runtime diagnostics. Never used as a candidate environment."""
import csv
import os
from pathlib import Path
import re
import sys

BINARY_PATHS = ('/opt/bin/nvidia-smi', '/usr/bin/nvidia-smi', '/usr/local/bin/nvidia-smi', '/bin/nvidia-smi')
LIBRARY_PATHS = ('/usr/local/nvidia/lib64', '/usr/local/nvidia/lib', '/usr/lib/x86_64-linux-gnu', '/usr/lib64')
BASE_ENV = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C'}


def loader_environment(paths=LIBRARY_PATHS):
    evidence, accepted = [], []
    for name in paths:
        path = Path(name)
        row = {'path': name, 'accepted': False, 'libraries': []}
        evidence.append(row)
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_dir() or any(p.stat().st_uid != 0 or p.stat().st_mode & 0o022 for p in (resolved, *resolved.parents)):
                row['error'] = 'UNTRUSTED_LIBRARY_DIRECTORY'
                continue
            for pattern in ('libnvidia-ml.so*', 'libcuda.so*'):
                for lib in sorted(path.glob(pattern)):
                    target = lib.resolve(strict=True)
                    row['libraries'].append({'name': lib.name, 'resolved': str(target), 'size': target.stat().st_size,
                                             'root_owned': target.stat().st_uid == 0, 'writable': bool(target.stat().st_mode & 0o022)})
                    if not target.is_file() or target.stat().st_uid != 0 or target.stat().st_mode & 0o022:
                        raise RuntimeError('UNTRUSTED_LIBRARY_FILE')
            accepted.append(name)
            row['accepted'] = True
        except (OSError, RuntimeError) as error:
            row['error'] = type(error).__name__ + ':' + str(error)
    env = dict(BASE_ENV)
    if accepted:
        env['LD_LIBRARY_PATH'] = ':'.join(accepted)
    return env, evidence


def parse_device(receipt):
    if not receipt or receipt.get('status', 'completed') != 'completed' or receipt.get('returncode') != 0:
        raise ValueError('ACCELERATOR_QUERY_FAILED')
    rows = list(csv.reader(receipt['stdout'].splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError('ACCELERATOR_QUERY_SHAPE')
    name, uuid, driver, memory = (s.strip() for s in rows[0])
    if not re.fullmatch(r'(?:NVIDIA )?RTX PRO 6000(?: Blackwell(?: Server| Workstation)? Edition)?', name, re.I):
        raise ValueError('ACCELERATOR_NAME_MISMATCH')
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid):
        raise ValueError('ACCELERATOR_UUID_MALFORMED')
    if not re.fullmatch(r'\d+(?:\.\d+)+', driver) or not re.fullmatch(r'[1-9]\d* MiB', memory):
        raise ValueError('ACCELERATOR_FIELDS_MALFORMED')
    return {'name': name, 'uuid': uuid, 'driver_version': driver, 'memory_mib': int(memory.split()[0]), 'device_count': 1}


def discover(execute, paths=BINARY_PATHS):
    env, libraries = loader_environment()
    attempts = []
    result = {'returncode': None, 'stdout': '', 'stderr': '', 'attempts': attempts,
              'loader_directories': libraries, 'trusted_environment': env, 'dependency_checks': {}}
    # Independent diagnostics: a query failure must not suppress driver loading facts.
    for library in ('libnvidia-ml.so.1', 'libcuda.so.1'):
        try:
            result['dependency_checks'][library] = execute([sys.executable, '-I', '-S', '-c',
                'import ctypes; ctypes.CDLL(' + repr(library) + '); print("LOAD_OK")'], timeout=5, output_cap=4096, env=env)
        except OSError as error:
            result['dependency_checks'][library] = {'error': type(error).__name__ + ':' + str(error)}
    for name in paths:
        attempt = {'path': name}
        attempts.append(attempt)
        try:
            if not Path(name).is_file() or not os.access(name, os.X_OK):
                attempt['error'] = 'NOT_USABLE_FILE'
                continue
            attempt.update(execute([name, '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'],
                                   timeout=10, output_cap=8192, env=env))
        except OSError as error:
            attempt['error'] = type(error).__name__ + ':' + str(error)
        result.update(attempt)
        try:
            result['observed_device'] = parse_device(result)
        except (ValueError, KeyError) as error:
            result['validation_error'] = str(error)
        return result
    result['error'] = 'NVIDIA_SMI_UNAVAILABLE'
    return result
