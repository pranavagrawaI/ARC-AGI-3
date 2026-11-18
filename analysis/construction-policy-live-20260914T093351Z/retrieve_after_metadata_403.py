#!/usr/bin/env python3
"""Read-only recovery after Kaggle rejects exact-version source pull."""
import fnmatch,hashlib,json,shutil,tempfile
from pathlib import Path
import collect_readonly as base
HERE=Path(__file__).resolve().parent; E=HERE/'evidence'
def main():
    launch=json.loads((E/'LAUNCH-RESULT.json').read_text()); identity=launch['returned_identity']; ref=f"{identity}/{launch['version']}"
    current=E/'server-readback-current'; current.mkdir(exist_ok=False)
    r=base.call(['kernels','pull',identity,'--metadata','-p',str(current)]); base.save(E/'server-metadata-current-receipt.json',r)
    if r['returncode']!=0: raise RuntimeError('CURRENT_SERVER_METADATA_READBACK_FAILED')
    r=base.call(['kernels','pull',identity,'-p',str(current)]); base.save(E/'server-source-current-receipt.json',r)
    if r['returncode']!=0: raise RuntimeError('CURRENT_SERVER_SOURCE_READBACK_FAILED')
    with tempfile.TemporaryDirectory(prefix='construction-live-output-') as tmp:
        r=base.call(['kernels','output',ref,'-p',tmp,'--force'],timeout=900); base.save(E/'output-retrieval-receipt.json',r)
        if r['returncode']!=0: raise RuntimeError('OUTPUT_RETRIEVAL_FAILED')
        policy=json.loads((HERE/'EVIDENCE-ALLOWLIST.json').read_text()); retained=[]; rejected=[]; remote=E/'remote'; remote.mkdir(exist_ok=False)
        for src in sorted(Path(tmp).rglob('*')):
            if not src.is_file(): continue
            rel=src.relative_to(tmp).as_posix(); lower=rel.lower()
            secret=any(fnmatch.fnmatch(lower,p.lower()) for p in policy['always_reject_without_reading'])
            if secret or not base.allowed(rel,policy['retained_paths']): rejected.append(rel); continue
            dst=remote/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
            retained.append({'path':rel,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
        base.save(E/'RETRIEVAL.json',{'exact_ref':ref,'terminal_status':'COMPLETE','retained':retained,'rejected_without_reading':rejected,'source_readback':'current identity after exact-version pull 403'})
    q=base.call(['quota','--format','json']); base.save(E/'quota-after.json',q)
    if q['returncode']!=0: raise RuntimeError('FINAL_QUOTA_FAILED')
    print(json.dumps({'exact_ref':ref,'terminal_status':'COMPLETE','retained':len(retained),'rejected_without_reading':len(rejected)},indent=2))
if __name__=='__main__': main()
