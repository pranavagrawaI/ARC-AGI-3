"""Prepared argv/interpreter/source -> observed frontend -> controller binding."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from core import shutdown
import artifacts


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(prepared_path,source_files,run_id):
    path=Path(prepared_path).resolve();raw=path.read_bytes()
    if len(raw)>65536:raise ValueError('prepared receipt size')
    prepared=json.loads(raw);argv=prepared['argv'];python=Path(prepared['python'])
    if not argv or not all(type(v) is str and v and '\0' not in v for v in argv):raise ValueError('exact argv required')
    entry=Path(argv[0])
    if not python.is_absolute() or not entry.is_absolute():raise ValueError('absolute interpreter/entrypoint')
    first=entry.read_bytes().split(b'\n',1)[0]
    if len(first)>4096 or not first.startswith(b'#!'):raise ValueError('explicit script interpreter required')
    shebang=first[2:].strip().decode()
    if len(shebang.split())!=1 or not Path(shebang).is_absolute() or Path(shebang).resolve()!=python.resolve():
        raise ValueError('prepared/shebang interpreter mismatch')
    sources={str(Path(k).resolve()):v for k,v in source_files.items()}
    shutdown.check_sources(sources)
    sources.update({str(path):hashlib.sha256(raw).hexdigest(),str(entry):sha(entry)})
    return dict(run_id=run_id,prepared_path=str(path),prepared_sha256=sha(path),launch_argv=argv,
                observed_argv=[shebang,*argv],python=str(python),interpreter_realpath=str(python.resolve()),
                interpreter_sha256=sha(python),source_files=sources)


def emit(child_pid,prepared_path,source_files,run_id,output,end):
    expected=contract(prepared_path,source_files,run_id)
    wrapper=shutdown.read_process(os.getpid())
    wanted=hashlib.sha256(b''.join(os.fsencode(s)+b'\0' for s in expected['observed_argv'])).hexdigest()
    while time.monotonic()<end:
        child=shutdown.read_process(child_pid)
        if child is None:raise ValueError('frontend exited before registration')
        if (child['ppid']==os.getpid() and child['identity']['cmdline_sha256']==wanted
                and child['identity']['exe']==expected['interpreter_realpath']):break
        time.sleep(min(.01,max(0,end-time.monotonic())))
    else:raise TimeoutError('registration exec/identity deadline')
    receipt=dict(contract=expected,wrapper=wrapper['identity'],frontend=child['identity'],registered_at=time.monotonic())
    output=Path(output)
    pending=output.with_name(output.name+'.pending-'+uuid.uuid4().hex)
    with pending.open('x') as stream:json.dump(receipt,stream,indent=2)
    os.link(pending,output) # publish only complete bytes; never overwrite authority/evidence
    return receipt


def bind(root,registration_path,expected):
    if Path(registration_path).stat().st_size>65536:raise ValueError('registration receipt size')
    receipt=json.loads(Path(registration_path).read_text())
    if receipt['contract']!=expected:raise ValueError('registration/prepared contract mismatch')
    if sha(expected['python'])!=expected['interpreter_sha256']:raise ValueError('interpreter drift')
    wrapper=shutdown.read_process(root.pid)
    child=shutdown.read_process(receipt['frontend']['pid'])
    if wrapper is None or child is None or wrapper['identity']!=receipt['wrapper'] or child['identity']!=receipt['frontend']:
        raise ValueError('registered process identity drift')
    sources=dict(expected['source_files']);sources[str(Path(registration_path).resolve())]=sha(registration_path)
    binding=shutdown.bind(root,child['pid'],expected['observed_argv'],sources)
    binding['run_id']=expected['run_id']
    return binding


def emit_control(child_pid,data,plan,prefix,output,end):
    """Shipping wrapper hook: retain argv/env/prefetch; no native-engine edits."""
    output=Path(output);prepared=output/'control-prepared.json'
    saved=json.loads(prepared.read_text())
    if saved['argv']!=data['argv'] or saved['python']!=data['python']:raise ValueError('prepared serve identity drift')
    sites=list(Path(prefix).glob('venv/lib/python*/site-packages'))
    if len(sites)!=1:raise ValueError('one pinned installed source root required')
    sources={str(sites[0]/name):expected for name,expected in plan['installed_source_sha256'].items()}
    return emit(child_pid,prepared,sources,plan['run_id'],output/'control-frontend-registration.json',end)
