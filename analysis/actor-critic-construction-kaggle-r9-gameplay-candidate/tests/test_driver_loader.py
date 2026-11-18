"""Compiled dlopen regression at the generated trusted exec boundary.

A private chroot makes every default allowlisted parent genuinely mapped-root
owned, rather than monkeypatching ownership checks on a world-writable /tmp path.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
import pytest
from test_runtime import HERE, ROOT, materialize

LIBRARY = 'int synthetic_driver_fixture(void) { return 73; }\n'
EXECUTABLE = r'''
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    void *h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!h) { fprintf(stderr, "synthetic fixture: %s\n", dlerror()); return 12; }
    int (*f)(void) = (int (*)(void))dlsym(h, "synthetic_driver_fixture");
    if (!f || f() != 73) return 13;
    if (argc > 1 && !strcmp(argv[1], "--synthetic-server")) {
        printf("SYNTHETIC_SERVER_LOADER_OK owner=%s profile=%s\n",
               getenv("R8_L3B_OWNER"), getenv("PYTHONPROFILEIMPORTTIME"));
    } else if (argc > 1 && strstr(argv[1], "query-compute-apps")) {
        /* Synthetic empty process census, not NVIDIA process evidence. */
    } else if (argc > 1 && strstr(argv[1], "uuid")) {
        puts("NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU-01234567-0123-0123-0123-0123456789ab, 580.1, 98304 MiB");
    } else {
        puts("NVIDIA RTX PRO 6000 Blackwell Server Edition, 98304, 98300, 580.1");
    }
    dlclose(h);
    return 0;
}
'''


def compile_fixture(tmp_path):
    library = tmp_path / 'libnvidia-ml.so.1'
    executable = tmp_path / 'nvidia-smi'
    for source, target, flags in [(LIBRARY, library, ['-shared', '-fPIC']), (EXECUTABLE, executable, ['-ldl'])]:
        result = subprocess.run(['cc', '-x', 'c', '-', '-o', str(target), *flags], input=source, text=True, capture_output=True, timeout=20)
        assert result.returncode == 0, result.stderr
    return library, executable


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    target.chmod(0o755)


def make_jail(tmp_path, capsule, library, executable):
    jail = tmp_path / 'synthetic-root'
    jail.mkdir(mode=0o755)
    stdlib = Path(sysconfig.get_path('stdlib'))
    runtime = jail / 'runtime'
    # Standalone test interpreter, never part of package or candidate rootfs.
    copy_file(Path(sys.executable).resolve(), runtime / 'bin/python')
    extensions = []
    for source in stdlib.rglob('*'):
        if {'site-packages', '__pycache__'} & set(source.parts) or not source.is_file():
            continue
        if source.suffix not in {'.py', '.so'}:
            continue
        copy_file(source, runtime / 'lib' / stdlib.name / source.relative_to(stdlib))
        if source.suffix == '.so':
            extensions.append(source)
    # Copy loader dependencies at their actual absolute paths; no host mounts,
    # no ldconfig cache, no synthetic NVIDIA library on default system paths.
    for source in [Path(sys.executable).resolve(), executable, *extensions]:
        output = subprocess.run(['ldd', str(source)], capture_output=True, text=True, timeout=5)
        assert output.returncode == 0, output.stderr
        for name in re.findall(r'(?:=>\s*)?(/[^\s()]+)', output.stdout):
            dependency = Path(name)
            destination = jail / dependency.relative_to('/')
            if not destination.exists():
                copy_file(dependency, destination)
            if dependency.name.startswith('libpython'):
                copy_file(dependency, runtime / 'lib' / dependency.name)
    shutil.copytree(capsule, jail / 'capsule')
    from game_mount import copy_game
    copy_game(jail)
    copy_file(library, jail / 'usr/local/nvidia/lib64/libnvidia-ml.so.1')
    copy_file(library, jail / 'usr/local/nvidia/lib64/libcuda.so.1')
    copy_file(executable, jail / 'opt/bin/nvidia-smi')
    copy_file(HERE / 'tests/loader_harness.py', jail / 'loader_harness.py')
    for kind in ('directory', 'file', 'directory-owner', 'file-owner'):
        target = jail / ('untrusted-' + kind)
        copy_file(library, target / 'libnvidia-ml.so.1')
        untrusted = target if kind.startswith('directory') else target / 'libnvidia-ml.so.1'
        if kind.endswith('-owner'):
            os.chown(untrusted, 1, 1)
        else:
            untrusted.chmod(0o777)
    (jail / 'dev').mkdir()
    (jail / 'dev/null').touch()
    return jail


@pytest.mark.parametrize('revision', ['r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8-gameplay', 'r9'])
def test_generated_bootstrap_cli_gpu_and_server_dynamic_loader(tmp_path, revision):
    assert os.getuid() == 0, 'run suite under mapped-root unshare; do not weaken trust checks'
    if revision == 'r9':
        ns, _ = materialize(tmp_path)
    else:
        nb = json.loads((ROOT / f'deployment/kaggle/actor-critic-construction-kaggle-{revision}/construction.ipynb').read_text())
        ns = {}
        exec(compile(''.join(nb['cells'][1]['source']).replace("Path('/kaggle/working')", f'Path({str(tmp_path)!r})'), f'preserved-{revision}-notebook-materialization', 'exec'), ns)
    library, executable = compile_fixture(tmp_path)
    jail = make_jail(tmp_path, ns['root'], library, executable)
    command = [sys.executable, '-I', '-S', '-B', '-c',
               'import os,sys; os.chroot(sys.argv[1]); os.chdir("/"); os.execve("/runtime/bin/python", ["/runtime/bin/python","-I","-S","-B","/loader_harness.py","bootstrap"], {"PATH":"/opt/bin:/usr/local/bin:/usr/bin:/bin","HOME":"/capsule","LANG":"C.UTF-8","ARC_DISABLE_DOTENV":"1"})', str(jail)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    evidence_path = jail / 'loader-evidence.json'
    assert evidence_path.exists(), result.stderr
    evidence = json.loads(evidence_path.read_text())
    initial = json.loads((jail / 'capsule/GPU.json').read_text())
    assert initial['status'] == 'passed', initial
    assert initial['discovery']['dependency_checks']['libnvidia-ml.so.1']['returncode'] == 0
    if revision == 'r2':
        assert result.returncode != 0
        assert evidence['failure_stage'] == 'accelerator_preflight'
        assert evidence['gpu']['returncode'] == 12
        assert 'libnvidia-ml.so.1' in evidence['gpu']['stderr']
        assert evidence['loader'] is None
        assert not (jail / 'server.log').exists()
    else:
        assert result.returncode == 0, result.stderr
        assert evidence['gpu']['returncode'] == 0
        assert '/usr/local/nvidia/lib64' in evidence['loader'].split(':')
        assert evidence['compute_query_returncode'] == 0
        assert evidence['server_stdout'].startswith('SYNTHETIC_SERVER_LOADER_OK')
    print(json.dumps({'revision': revision, **evidence}, sort_keys=True))


def test_driver_loader_never_reaches_generated_candidate_or_inspection(tmp_path):
    ns, _ = materialize(tmp_path)
    library, _ = compile_fixture(tmp_path)
    # Driver is genuinely loadable in the trusted parent process at exec time.
    # Candidate and inspection must lose both the loader setting and the file.
    source = f'''
import os
def probe():
    try:
        open({str(library)!r}, 'rb').read(1)
        file_denied = False
    except OSError:
        file_denied = True
    try:
        import ctypes
        ctypes.CDLL('libnvidia-ml.so.1')
        load_denied = False
    except (ImportError, OSError):
        load_denied = True
    return {{'loader': os.environ.get('LD_LIBRARY_PATH'), 'file_denied': file_denied, 'load_denied': load_denied}}
def initialize(obs): return probe()
def predict(state, action): return []
def assimilate(state, obs, verdict): return state
'''
    code = f'''import sys,json,ctypes
sys.path.insert(0,{str(ns['root'])!r})
assert ctypes.CDLL('libnvidia-ml.so.1').synthetic_driver_fixture()==73
from integration_boundary import CandidateRPC,SealedBoundary
source={source!r}
rpc=CandidateRPC(SealedBoundary(),source).initialize({{}})
inspection=SealedBoundary().inspect(source+'\\nimport json\\nprint(json.dumps(probe()))', {{'visible':'prefix'}})
print(json.dumps({{'rpc':rpc,'inspection':json.loads(inspection['stdout'])}}))
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code], env={**os.environ, 'LD_LIBRARY_PATH': str(library.parent)}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    expected = {'loader': None, 'file_denied': True, 'load_denied': True}
    assert json.loads(result.stdout) == {'rpc': expected, 'inspection': expected}
