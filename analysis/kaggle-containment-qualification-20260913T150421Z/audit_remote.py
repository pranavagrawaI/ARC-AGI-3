"""Read-only independent assertion audit of downloaded allowlisted receipts."""
import json
from pathlib import Path
import re
import sys
from verify_package import verify, HERE

NAMES=('benign','filesystem','inherited','syscalls','readonly','wall','output','forged')
DENIED='socket connect ptrace process_vm_readv pidfd_getfd mount chroot setns unshare setuid setgid capset prctl clone fork vfork clone3 execve execveat io_uring_setup open_by_handle_at prlimit64 fcntl sched_getaffinity x32_getpid'.split()

def audit():
    frozen=json.loads((HERE/'PRELAUNCH.json').read_text())
    assert verify()==frozen
    result=json.loads((HERE/'remote/qualification/result.json').read_text())
    platform=json.loads((HERE/'remote/qualification/platform.json').read_text())
    status=json.loads((HERE/'status-terminal.json').read_text())
    server=json.loads((HERE/'server-metadata/kernel-metadata.json').read_text())
    assert status['status'] in ('COMPLETE','ERROR','CANCELLED','CANCELED')
    assert server['id']==frozen['kernel']
    assert server['is_private'] is True and server['enable_internet'] is False
    assert server['machine_shape']=='NvidiaRtxPro6000'
    assert server['docker_image']=='gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
    assert all(server[k]==[] for k in ('dataset_sources','kernel_sources','competition_sources','model_sources'))
    rows=result['cases']
    checks=[]
    if result['classification']=='QUALIFIED_SYNTHETIC_PROBE_ONLY':
        assert status['status']=='COMPLETE'
        assert len(rows)==16 and result['completed_cases']==16 and result['passed_cases']==16
        assert {(r['mode'],r['name']) for r in rows}=={(m,n) for m in ('candidate','inspection') for n in NAMES}
        assert result['cleanup_complete'] is True and result['model_loaded'] is False
        assert platform['uid']==[0,0,0] and platform['uname'][0]=='Linux' and platform['uname'][4]=='x86_64'
        accelerator=platform['accelerator']
        assert accelerator['returncode']==0 and len(accelerator['stdout'].strip().splitlines())==1
        assert 'RTX PRO 6000' in accelerator['stdout'].upper()
        for row in rows:
            assert row['check'] is True and row['reaped'] is True
            assert row['parent_canary_unchanged'] is True and row['parent_no_loopback_connection'] is True
            assert row['rootfs_unchanged'] is True
            seal=json.loads(row['stderr'])['sealed']
            assert seal['uid']==[65534]*3 and seal['gid']==[65534]*3 and seal['groups']==[]
            assert seal['capabilities_zero'] is True and seal['ambient_cleared'] is True
            assert seal['bounding_dropped']>=38 and seal['no_new_privs'] is True
            assert seal['cwd']=='/' and seal['seccomp']=='native-allowlist' and seal['parent_death_signal']==9
            name=row['name']; mode=row['mode']
            if name in ('wall','output'):
                assert row['returncode']==-9
                assert row['status']==('timeout' if name=='wall' else 'output_limit')
                assert row['elapsed_s']<5
            else:
                assert row['status']=='completed' and row['returncode']==0
                if name=='forged':
                    assert row['stdout']=='PASS\n{"qualified": true}\n'
                else:
                    value=json.loads(row['stdout'])
                    if name=='benign': assert value=={'answer':5,'roundtrip':{'a':2,'b':3}}
                    elif name=='readonly': assert value=='denied'
                    elif name=='inherited': assert value=={'stdin':'','env':{'LANG':'C','PYTHONDONTWRITEBYTECODE':'1'},'extra_fds':[]}
                    elif name=='syscalls': assert value=={**{n:[-1,1] for n in DENIED},'subprocess':'denied','loopback':'denied'}
                    elif name=='filesystem':
                        expected={k:'denied' for k in ('absolute','traversal','symlink','proc','proc_environ','corpus','future','verifier','device','openat_fd','fchdir')}
                        expected.update(cwd='/',prefix={'visible':['SYNTHETIC_VISIBLE_PREFIX']} if mode=='inspection' else None)
                        assert value==expected
            manifest=json.loads((HERE/'remote/qualification'/f'{mode}-{name}-rootfs.json').read_text())
            assert ('work/prefix.json' in manifest)==(mode=='inspection')
            assert not any(k.startswith(('proc/','sys/','dev/')) or 'site-packages' in k or '..' in Path(k).parts or Path(k).is_absolute() for k in manifest)
            for path,h in manifest.items():
                assert path.startswith('runtime/') or path in ('work/candidate.py','work/args.json','work/prefix.json','work/escape-link')
                assert re.fullmatch('[0-9a-f]{64}',h) or (path=='work/escape-link' and name=='filesystem' and h.startswith('synthetic-test-symlink:'))
            checks.append({'mode':mode,'property':name,'independent_check':'passed'})
    before=json.loads(json.loads((HERE/'quota-before.json').read_text())['stdout'])
    after=json.loads(json.loads((HERE/'quota-after.json').read_text())['stdout'])
    used=lambda data:float(next(r['used'] for r in data if r['resource']=='GPU').removesuffix('h'))
    delta=round(used(after)-used(before),6)
    assert 0<=delta<=.5, ('quota delta outside cap',delta)
    output={'classification':result['classification'],'terminal_status':status['status'],
            'kernel':frozen['kernel'],'version':1,'provider_image_matches_requested':True,
            'source_hashes_unchanged':True,'checked_cases':len(checks),'cases':checks,
            'quota_display_delta_gpu_hours':delta,'quota_precision_hours':.01,
            'remote_elapsed_seconds':result['elapsed_s'],'since_launch_terminal_seconds':status['since_launch_s'],
            'limitations':['quota display rounded; not exact per-job bill','timeout300 requested, no independent server timeout attestation','container digest matches server metadata, not full kernel/runtime equivalence','supervisor-kill behavior exercised locally; '+('remote arming receipts present' if checks else 'no remote runner started, so remote arming not exercised')]}
    assert status['since_launch_s']<=600, 'LIFECYCLE_CAP_EXCEEDED'
    (HERE/'AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__': audit()
