#!/usr/bin/env python3
"""Supplemental read-only collector after preserved transient DNS failure."""
import json,time
from pathlib import Path
import collect_readonly as base

HERE=Path(__file__).resolve().parent
E=HERE/'evidence'

def main():
    launch=json.loads((E/'LAUNCH-RESULT.json').read_text())
    ref=f"{launch['returned_identity']}/{launch['version']}"
    observations=[]
    deadline=time.monotonic()+28800
    while True:
        receipt=base.call(['kernels','status',ref])
        observations.append(receipt)
        base.save(E/'status-observations-recovery-2.json',observations)
        if receipt['returncode']!=0:
            if time.monotonic()>=deadline:
                raise RuntimeError('RECOVERY_STATUS_DEADLINE')
            time.sleep(30)
            continue
        status=base.status_name(receipt)
        if status in base.TERMINAL:
            break
        if status not in {'QUEUED','RUNNING'} or time.monotonic()>=deadline:
            raise RuntimeError('RECOVERY_MONITOR_STOPPED_WITHOUT_TERMINAL_STATUS')
        time.sleep(30)
    base.save(E/'status-exact-final.json',{'ref':ref,'status':status,'receipt':receipt,'prior_monitor_failure':'preserved in status-observations.json'})
    readback=E/'server-readback'
    readback.mkdir(exist_ok=False)
    receipt=base.call(['kernels','pull',ref,'--metadata','-p',str(readback)])
    base.save(E/'server-metadata-receipt.json',receipt)
    if receipt['returncode']!=0: raise RuntimeError('SERVER_METADATA_READBACK_FAILED')
    receipt=base.call(['kernels','pull',ref,'-p',str(readback)])
    base.save(E/'server-source-receipt.json',receipt)
    if receipt['returncode']!=0: raise RuntimeError('SERVER_SOURCE_READBACK_FAILED')
    import fnmatch,hashlib,shutil,tempfile
    with tempfile.TemporaryDirectory(prefix='construction-live-output-') as tmp:
        receipt=base.call(['kernels','output',ref,'-p',tmp,'--force'],timeout=900)
        base.save(E/'output-retrieval-receipt.json',receipt)
        if receipt['returncode']!=0: raise RuntimeError('OUTPUT_RETRIEVAL_FAILED')
        policy=json.loads((HERE/'EVIDENCE-ALLOWLIST.json').read_text())
        retained=[]; rejected=[]; remote=E/'remote'; remote.mkdir(exist_ok=False)
        for src in sorted(Path(tmp).rglob('*')):
            if not src.is_file(): continue
            rel=src.relative_to(tmp).as_posix(); lower=rel.lower()
            secret=any(fnmatch.fnmatch(lower,p.lower()) for p in policy['always_reject_without_reading'])
            if secret or not base.allowed(rel,policy['retained_paths']): rejected.append(rel); continue
            dst=remote/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
            retained.append({'path':rel,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
        base.save(E/'RETRIEVAL.json',{'exact_ref':ref,'terminal_status':status,'retained':retained,'rejected_without_reading':rejected,'collector':'supplemental_after_transient_dns'})
    quota=base.call(['quota','--format','json'])
    base.save(E/'quota-after.json',quota)
    if quota['returncode']!=0: raise RuntimeError('FINAL_QUOTA_FAILED')
    print(json.dumps({'exact_ref':ref,'terminal_status':status,'retained':len(retained)},indent=2))

if __name__=='__main__': main()
