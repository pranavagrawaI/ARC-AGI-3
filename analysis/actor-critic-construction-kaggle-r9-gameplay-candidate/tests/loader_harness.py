"""Test-only chroot fixture; never included in the generated capsule.

Uses real generated bootstrap/CLI/query and exact server launch statements.
No NVIDIA, installer, model mount, or SGLang qualification is claimed.
"""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def supervisor(command):
    output = Path('/cli-output')
    command[command.index('--output-dir') + 1] = str(output)
    real = subprocess.run(command, capture_output=True, text=True, timeout=20)
    result = json.loads((output / 'result.json').read_text())
    snapshot = json.loads((output / 'system-before.json').read_text())
    evidence = {'scope': 'synthetic dynamic-loader fixture, NOT hardware or model evidence',
                'cli_returncode': real.returncode, 'failure_stage': result['failure_stage'],
                'gpu': snapshot['gpu'], 'loader': os.environ.get('LD_LIBRARY_PATH')}
    Path('/loader-evidence.json').write_text(json.dumps(evidence))
    # Real main proceeds beyond its GPU gate. Missing /proc/mounts in this fixture
    # remain real unavailable prerequisites, not fabricated passing receipts.
    assert snapshot['gpu']['returncode'] == 0, evidence
    assert result['failure_stage'] != 'accelerator_preflight', evidence
    # The SAME process environment that entered the real CLI now executes the
    # exact generated environment construction + Popen, not a copied dict test.
    tree = ast.parse(Path(command[1]).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_probe')
    statements = None
    # Locate a contiguous production block in any statement list, without
    # altering its assignments, conditionals, arguments or environment handling.
    for node in ast.walk(fn):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            start = next((i for i, s in enumerate(value) if isinstance(s, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'process_environment' for t in s.targets)), None)
            if start is None:
                continue
            stop = next(i for i in range(start, len(value)) if isinstance(value[i], ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'process' for t in value[i].targets))
            statements = value[start:stop + 1]
    assert statements is not None
    with Path('/server.log').open('w') as log:
        scope = {'os': os, 'subprocess': subprocess, 'args': SimpleNamespace(r8_l3b=True, profile_imports=True),
                 'R8_L3B_OWNER': 'synthetic-loader-fixture', 'output_dir': Path('/'),
                 'write_json': lambda path, data: path.write_text(json.dumps(data)),
                 'command': ['/opt/bin/nvidia-smi', '--synthetic-server'], 'server_log': log}
        exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), 'generated-server-launch', 'exec'), scope)
        assert scope['process'].wait(timeout=5) == 0
    evidence['server_stdout'] = Path('/server.log').read_text().strip()
    assert evidence['server_stdout'] == 'SYNTHETIC_SERVER_LOADER_OK owner=synthetic-loader-fixture profile=1'
    # Lifecycle compute-app query is a sibling inherited-env consumer.
    query = subprocess.run(['/opt/bin/nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
    assert query.returncode == 0 and query.stdout == ''
    evidence['compute_query_returncode'] = query.returncode
    Path('/loader-evidence.json').write_text(json.dumps(evidence))


def bootstrap():
    sys.path.insert(0, '/capsule')
    import accelerator
    import donor_bootstrap
    try:
        import gameplay_play  # r8 driver
    except ModuleNotFoundError:
        import kaggle_probe as gameplay_play  # frozen r2-r7 probe driver
    # Default production paths and actual ownership checks, not patched validators.
    os.environ['LD_LIBRARY_PATH'] = '/ambient-canary'
    os.environ['LIBRARY_PATH'] = '/ambient-canary'
    env, rows = accelerator.loader_environment()
    os.environ.pop('LD_LIBRARY_PATH')
    os.environ.pop('LIBRARY_PATH')
    assert '/usr/local/nvidia/lib64' in env.get('LD_LIBRARY_PATH', '').split(':')
    assert all(r['path'] in accelerator.LIBRARY_PATHS for r in rows)
    assert '/ambient-canary' not in env.get('LD_LIBRARY_PATH', '')
    for name in ('directory', 'file', 'directory-owner', 'file-owner'):
        env, rows = accelerator.loader_environment(['/untrusted-' + name])
        assert 'LD_LIBRARY_PATH' not in env and not rows[0]['accepted']
    # chroot deliberately lacks /proc and full nested sealing runtime. Real seal
    # execution is covered independently by the LD-bearing candidate/inspection test.
    gameplay_play.require_seal = lambda out: (out / 'SEAL-FIXTURE-NOT-QUALIFICATION.json').write_text('{"scope":"unavailable nested fixture prerequisite"}')
    donor_bootstrap.runpy.run_path = lambda path, **kwargs: Path('/INSTALL-FIXTURE-NOT-QUALIFICATION.json').write_text(json.dumps({'scope': 'offline installer not executed', 'path': str(path)}))
    original_run = subprocess.run

    def dispatch(command, **kwargs):
        if len(command) > 1 and command[1] == '/capsule/official_launcher_runtime.py':
            Path('/bootstrap-command.json').write_text(json.dumps(command))
            # Only a test wrapper around the actual argv; it runs the unchanged
            # CLI and then the generated server environment block in its exec env.
            return original_run([sys.executable, '-I', '-S', '-B', '/loader_harness.py', 'supervisor', *command], **kwargs)
        return original_run(command, **kwargs)

    donor_bootstrap.subprocess.run = dispatch
    donor_bootstrap.main()
    assert 'LD_LIBRARY_PATH' not in os.environ, 'trusted child env must not mutate bootstrap globals'


if __name__ == '__main__':
    if sys.argv[1] == 'supervisor':
        supervisor(sys.argv[2:])
    else:
        bootstrap()
