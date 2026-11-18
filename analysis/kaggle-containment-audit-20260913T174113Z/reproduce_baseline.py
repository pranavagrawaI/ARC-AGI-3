"""Bounded reproductions against untouched frozen source; no external calls."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'kaggle-containment-qualification-20260913T160618Z'
spec=importlib.util.spec_from_file_location('frozen_probe',OLD/'probe.py')
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)


def reproduce():
    findings={}
    with tempfile.TemporaryDirectory() as temp:
        base=Path(temp)
        root=base/'root'
        manifest=old.rootfs(root,'def run(): return 1','candidate',[],require_root=False)
        root.chmod(0o755); (root/'extra-directory').mkdir(mode=0o555)
        old.validate_tree(root,manifest,require_root=False)
        findings['extra_empty_directory_accepted']=True
        for p in (root,*root.rglob('*')):
            if p.is_dir(): p.chmod(0o755)
        # Reproduce the actual parent setup fault and close anything leaked by
        # the old implementation, so the regression itself is not a leak.
        actual_open=os.open; actual_socket=socket.socket; fds=[]; listeners=[]
        def tracked(path,flags,*a,**kw):
            fd=actual_open(path,flags,*a,**kw)
            if str(path).startswith('/tmp/containment-q-') and flags & os.O_DIRECTORY: fds.append(fd)
            return fd
        class Listener:
            def __init__(self): self.socket=actual_socket(); self.closed=False; listeners.append(self)
            def bind(self,*a): raise OSError('SYNTHETIC_BIND_FAILURE')
            def close(self): self.closed=True; self.socket.close()
        try:
            with patch.object(old,'snapshot',return_value={}), patch.object(old.os,'open',side_effect=tracked), patch.object(socket,'socket',side_effect=Listener):
                rc=old.main(base/'bind',local=True)
            result=json.loads((base/'bind/result.json').read_text())
            leaked=[]
            for fd in fds:
                try: os.fstat(fd); leaked.append(fd)
                except OSError: pass
            findings['bind_failure']={'returncode':rc,'reported_cleanup_complete':result['cleanup_complete'],
                'directory_fd_leaked':bool(leaked),'listener_close_omitted':any(not s.closed for s in listeners)}
            assert findings['bind_failure']['directory_fd_leaked'] and findings['bind_failure']['listener_close_omitted']
        finally:
            for fd in fds:
                try: os.close(fd)
                except OSError: pass
            for listener in listeners: listener.close()
        # Caller-chain proof: changing a hashed source after rootfs() reaches
        # the execution function before any admission hash check in old main().
        actual_rootfs=old.rootfs; called=[]
        def changed(*a,**kw):
            manifest=actual_rootfs(*a,**kw)
            file=a[0]/'work/candidate.py'; file.chmod(0o644); file.write_text(file.read_text()+'\n# synthetic drift'); file.chmod(0o444)
            return manifest
        def execution(*a,**kw):
            called.append(True)
            raise RuntimeError('STOP_BEFORE_UNTRUSTED_EXEC_IN_BASELINE_REPRODUCTION')
        with patch.object(old,'snapshot',return_value={}), patch.object(old,'rootfs',side_effect=changed), patch.object(old,'bounded',side_effect=execution):
            old.main(base/'admission',local=True)
        findings['mutated_rootfs_reaches_exec_before_validation']=bool(called)
        assert called
    return findings

if __name__=='__main__':
    result=reproduce()
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
