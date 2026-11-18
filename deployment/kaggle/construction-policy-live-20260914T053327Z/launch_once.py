#!/usr/bin/env python3
"""Sole launch path: consume once before one exact Kaggle push; never retry."""
from datetime import datetime,timezone
import hashlib,json,os,re,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]; ANALYSIS=REPO/'analysis'/HERE.name
CLI='/home/pranav/.local/bin/kaggle'; TERMINAL={'COMPLETE','ERROR','CANCELLED'}
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def invoke(args,timeout=120):
    try:
        p=subprocess.run([CLI,*args],stdin=subprocess.DEVNULL,close_fds=True,capture_output=True,text=True,timeout=timeout)
        return {'arguments':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'utc':now(),'timed_out':False}
    except subprocess.TimeoutExpired as e:
        return {'arguments':args,'returncode':None,'stdout':e.stdout or '','stderr':e.stderr or '','utc':now(),'timed_out':True}
def quota_gpu(receipt):
    if receipt['returncode']!=0: raise RuntimeError('QUOTA_QUERY_FAILED')
    rows=json.loads(receipt['stdout']); gpu=next((x for x in rows if x.get('resource')=='GPU'),None)
    if not gpu or not str(gpu.get('remaining','')).endswith('h'): raise RuntimeError('GPU_QUOTA_UNPARSEABLE')
    if float(gpu['remaining'][:-1])<1.0: raise RuntimeError('GPU_QUOTA_BELOW_1H')
    return gpu
def target_absent(receipt,expected_id):
    if receipt['returncode']!=0: return False
    text=receipt['stdout'].strip()
    if text in {'Not found','No kernels found','[]'}: return True
    try: rows=json.loads(text)
    except Exception: return False
    if not isinstance(rows,list): return False
    return all((x.get('ref') or x.get('id'))!=expected_id for x in rows if isinstance(x,dict)) and not rows
def local_gate():
    import sys
    sys.path.insert(0,str(ANALYSIS)); import prelaunch_check
    out=prelaunch_check.check(write=True)
    if out['status']!='APPROVED_NOT_LAUNCHED': raise RuntimeError('FINAL_REVIEW_NOT_APPROVED:'+out['status'])
    return out
def consume(command,gate):
    path=ANALYSIS/'LAUNCH-CONSUMED.json'
    value={'identity':HERE.name,'authorization_sha256':sha(ANALYSIS/'AUTHORIZATION.json'),'review_sha256':sha(ANALYSIS/'REVIEW-RESOLVED.json'),'manifest_sha256':gate['manifest_sha256'],'command':command,'consumed_utc':now(),'epoch':time.time(),'no_retry':True}
    raw=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode(); fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    try: os.write(fd,raw); os.fsync(fd)
    finally: os.close(fd)
    return value
def main():
    gate=local_gate(); meta=json.loads((HERE/'kernel-metadata.json').read_text()); reference=json.loads((ANALYSIS/'SERVER-REFERENCE.json').read_text())
    if meta!=reference: raise RuntimeError('EXACT_SERVER_REFERENCE_MISMATCH')
    slug=meta['id'].split('/',1)[1]; evidence=ANALYSIS/'evidence'; evidence.mkdir(exist_ok=True)
    q=invoke(['quota','--format','json']); save(evidence/'quota-before.json',q); quota_gpu(q)
    absent=invoke(['kernels','list','--mine','--search',slug,'--page-size','100','--format','json']); save(evidence/'target-absence.json',absent)
    if not target_absent(absent,meta['id']): raise RuntimeError('FRESH_TARGET_NOT_ABSENT')
    gate=local_gate()
    command=['kernels','push','--path',str(HERE),'--timeout','3600','--accelerator','NvidiaRtxPro6000']
    consume(command,gate)
    receipt=invoke(command,timeout=3660); save(evidence/'launch-receipt.json',receipt)
    match=re.search(r'Kernel version (\d+) successfully pushed\.\s+Please check progress at (https://www\.kaggle\.com/code/(\S+))',receipt.get('stdout','')) if receipt['returncode']==0 else None
    if not match:
        save(evidence/'LAUNCH-RESULT.json',{'status':'UNCERTAIN_ADMISSION_NO_RETRY','pushes':1,'retries':0,'receipt':'launch-receipt.json'})
        raise RuntimeError('UNCERTAIN_ADMISSION_NO_RETRY')
    version=int(match.group(1)); url=match.group(2); returned=match.group(3)
    if returned!=meta['id'] or version!=1:
        save(evidence/'LAUNCH-RESULT.json',{'status':'IDENTITY_MISMATCH_NO_RETRY','returned_identity':returned,'version':version,'url':url,'pushes':1,'retries':0})
        raise RuntimeError('RETURNED_IDENTITY_OR_VERSION_MISMATCH_NO_RETRY')
    result={'status':'ADMITTED_MONITOR_READ_ONLY','returned_identity':returned,'version':version,'url':url,'pushes':1,'retries':0,'command':command}
    save(evidence/'LAUNCH-RESULT.json',result); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
