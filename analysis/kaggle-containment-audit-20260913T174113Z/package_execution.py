"""Execute the exact generated notebook in a local mapped-root fixture.
No security primitives are mocked. NVIDIA CSV is explicitly synthetic.
"""
import ast
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = REPO/'deployment/kaggle'/HERE.name
MAPPING = ['unshare','--user','--map-users','0:1000:1','--map-users','65534:100000:1',
           '--map-groups','0:1000:1','--map-groups','65534:100000:1','--setgroups','allow']
CSV = 'NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU-12345678-1234-1234-1234-123456789abc, 580.1, 97280 MiB'

LIB_SOURCE = 'const char *fixture(void) { return "' + CSV + '"; }\n'
EXE_SOURCE = '''#include <dlfcn.h>
#include <stdio.h>
int main(void) {
 void *p = dlopen("libcontainment_fixture.so", RTLD_NOW);
 if (!p) { fprintf(stderr,"SYNTHETIC_LOADER_FAILURE:%s\\n",dlerror()); return 12; }
 const char *(*f)(void) = dlsym(p,"fixture");
 if (!f) return 13;
 puts(f()); return 0;
}
'''


def compile_fixture(base):
    binary = base/'bin'; libraries = base/'lib64'
    binary.mkdir(); libraries.mkdir()
    (base/'fixture.c').write_text(LIB_SOURCE)
    (base/'main.c').write_text(EXE_SOURCE)
    for command in (['gcc','-shared','-fPIC',str(base/'fixture.c'),'-o',str(libraries/'libcontainment_fixture.so')],
                    ['gcc',str(base/'main.c'),'-ldl','-o',str(binary/'nvidia-smi')]):
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return binary, libraries


def fixture_command(binary, libraries, output, script):
    return MAPPING + ['bwrap','--cap-add','ALL','--tmpfs','/usr','--ro-bind','/usr/bin','/usr/bin',
        '--ro-bind','/usr/lib','/usr/lib','--symlink','lib','/usr/lib64',
        '--symlink','usr/bin','/bin','--symlink','usr/lib','/lib','--symlink','usr/lib','/lib64',
        '--ro-bind','/etc','/etc','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--tmpfs','/usr/local',
        '--ro-bind',str(REPO),'/repo','--bind',str(output),'/kaggle/working',
        '--ro-bind',str(binary),'/opt/bin','--ro-bind',str(libraries),'/usr/local/nvidia/lib64',
        '--chdir','/kaggle/working','/usr/bin/python3','-B','-c',script]


def run(output, variant='success'):
    output.mkdir(parents=True, exist_ok=False)
    (output/'FIXTURE-ONLY.json').write_text(json.dumps({'scope':'LOCAL_SYNTHETIC_NVIDIA_FIXTURE_NOT_GPU_QUALIFICATION',
        'security':'real mapped-root chroot/setgroups/setresuid/capabilities/seccomp/rlimits/exec',
        'variant':variant}, indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='containment-target-fixture-') as temp:
        base = Path(temp)
        binary, libraries = compile_fixture(base)
        if variant == 'loader_missing':
            (libraries/'libcontainment_fixture.so').unlink()
        elif variant == 'malformed':
            (binary/'nvidia-smi').write_text('#!/bin/sh\nprintf "RTX PRO 6000\\n"\n')
            (binary/'nvidia-smi').chmod(0o755)
        elif variant not in ('success','cached_modules'):
            raise ValueError(variant)
        source_dir = '/repo/analysis/'+HERE.name
        notebook_path = '/repo/deployment/kaggle/'+HERE.name+'/qualification.ipynb'
        baseline_path = '/repo/deployment/kaggle/kaggle-containment-qualification-20260913T160618Z/qualification.ipynb'
        script = f'''import ast,json,sys,subprocess
from pathlib import Path
sys.path.insert(0,{source_dir!r})
import probe
# Read the actual frozen bootstrap environment from its AST, not a retyped proxy.
old=json.loads(Path({baseline_path!r}).read_text())['cells'][0]['source']
tree=ast.parse(old)
env=next(ast.literal_eval(k.value) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run' for k in n.keywords if k.arg=='env')
p=subprocess.run(['/opt/bin/nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],stdin=subprocess.DEVNULL,close_fds=True,capture_output=True,text=True,timeout=5,env=env)
Path('baseline-loader.json').write_text(json.dumps({{'scope':'SYNTHETIC_DLOPEN_REPRODUCTION','environment':env,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}},indent=2))
Path('candidate-loader.json').write_text(json.dumps(probe.discover_accelerator(),indent=2))
# Unload host source modules: bootstrap must import its actual unpacked bytes.
for name in ('probe','accelerator','audit_remote'): sys.modules.pop(name,None)
sys.path.remove({source_dir!r})
code=json.loads(Path({notebook_path!r}).read_text())['cells'][0]['source']
if {variant!r}=='cached_modules':
 import types
 for name in ('probe','audit_remote','accelerator','runner'):
  fake=types.ModuleType(name)
  def forbidden(*a,**k):
   Path('CACHED_MODULE_USED').write_text('false provenance')
   raise RuntimeError('CACHED_MODULE_EXECUTION')
  fake.bounded=forbidden; fake.audit_directory=forbidden; fake.arm_parent_death=forbidden
  sys.modules[name]=fake
exec(compile(code,{notebook_path!r},'exec'),{{'__name__':'__main__'}})
'''
        command = fixture_command(binary, libraries, output, script)
        process = subprocess.run(command, stdin=subprocess.DEVNULL, close_fds=True,
            env={**os.environ,'SYNTHETIC_SECRET':'LOCAL_FIXTURE_SECRET_MUST_NOT_REACH_CANDIDATE'},
            capture_output=True, text=True, timeout=210)
        (output/'notebook.stdout').write_text(process.stdout)
        (output/'notebook.stderr').write_text(process.stderr)
        report = {'variant':variant,'notebook_returncode':process.returncode,'fixture_scope':'LOCAL_ONLY_NO_GPU_FACTS'}
        lifecycle = json.loads((output/'package-lifecycle.json').read_text())
        baseline = json.loads((output/'baseline-loader.json').read_text())
        fixed = json.loads((output/'candidate-loader.json').read_text())
        if variant in ('success','cached_modules'):
            assert baseline['returncode']==12 and 'SYNTHETIC_LOADER_FAILURE' in baseline['stderr'], baseline
            assert fixed['returncode']==0 and fixed['observed_device']['device_count']==1, fixed
            assert process.returncode==0 and lifecycle['qualified'] is True, process.stderr + process.stdout
            assert lifecycle['audit']['passed']==16 and lifecycle['audit']['unexercised']==0
            assert (output/'package-worker.json').exists() and (output/'qualification/result.json').exists()
            assert not (output/'CACHED_MODULE_USED').exists()
        else:
            assert process.returncode!=0 and lifecycle['qualified'] is False, lifecycle
            result=json.loads((output/'qualification/result.json').read_text())
            assert result['completed_cases']==0 and result['unexercised_cases']==16, result
        report['verified']=True
        (output/'VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n')
        return report

if __name__=='__main__':
    print(json.dumps(run(Path(sys.argv[1]).resolve(), sys.argv[2] if len(sys.argv)>2 else 'success'),indent=2))
