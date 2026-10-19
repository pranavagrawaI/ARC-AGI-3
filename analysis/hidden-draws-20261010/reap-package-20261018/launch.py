"""One approved bounded launch. Any existing attempt receipt forbids another push."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'analysis/marlin-fixed-replay-r3-20261011'))
from kaggle_backend import KaggleBackend


def main():
    if (HERE/'launch-attempt.json').exists():
        raise RuntimeError('An attempt already exists; inspect it, never relaunch')
    authority=json.loads((HERE/'AUTHORITY.json').read_text())
    assert authority['qualification_launches']==1 and authority['qualification_gpu_cap_s']==2700
    verification=json.loads((HERE/'VERIFICATION.json').read_text())
    metadata=json.loads((HERE/'kernel/kernel-metadata.json').read_text())
    notebook=HERE/'kernel'/metadata['code_file']
    assert hashlib.sha256(notebook.read_bytes()).hexdigest()==verification['candidate_sha256']
    assert hashlib.sha256((HERE/'kernel/kernel-metadata.json').read_bytes()).hexdigest()==verification['metadata_sha256']
    assert metadata['machine_shape']=='NvidiaRtxPro6000' and metadata['is_private'] is True
    backend=KaggleBackend()
    quota=backend.quota().gpu_quota
    active=[]
    for session in backend.active_sessions('pranavagrawai'):
        meta=backend.get_kernel(session['ref']).metadata
        active.append({**session,'enable_gpu':meta.enable_gpu,
                       'enable_tpu':meta.enable_tpu,'machine_shape':meta.machine_shape})
    gpu_active=[s for s in active if s['enable_gpu'] or s['machine_shape'].startswith('Nvidia')]
    admission=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   used_s=quota.time_used.total_seconds(),reserved_s=quota.time_reserved.total_seconds(),
                   allowed_s=quota.total_time_allowed.total_seconds(),active=active,
                   gpu_active=gpu_active,new_cap_s=2700)
    (HERE/'admission-launch.json').write_text(json.dumps(admission,indent=2)+'\n')
    assert admission['used_s']+admission['reserved_s']+2700 <= min(108000,admission['allowed_s'])
    assert len(gpu_active)<2 and all(a.get('ref') for a in active),active
    backend.require_new_slug(metadata['id'])
    freeze_files=['AUTHORITY.json','EXPERIMENT-CARD.md','HIDDEN-DRAW-CARD.md',
                  'VERIFICATION.json','packaging.patch','launch.py','collect.py',
                  'kernel/kernel-metadata.json','kernel/'+metadata['code_file']]
    frozen={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in freeze_files}
    (HERE/'launch-freeze.json').write_text(json.dumps(frozen,indent=2)+'\n')
    attempt=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),state='attempting',
                 slug=metadata['id'],cap_s=2700,notebook_sha256=verification['candidate_sha256'],
                 admission=admission)
    with (HERE/'launch-attempt.json').open('x') as f:
        json.dump(attempt,f,indent=2)
    response=backend.push(HERE/'kernel',2700)
    (HERE/'launch-response.json').write_text(json.dumps(response,indent=2)+'\n')
    assert not response.get('error') and response.get('versionNumber')==1 and response.get('kernelId'),response
    assert response.get('url')=='https://www.kaggle.com/code/'+metadata['id'],response
    assert not any(response.get(k) for k in ['invalidTags','invalidDatasetSources','invalidCompetitionSources','invalidKernelSources','invalidModelSources']),response
    attempt.update(state='accepted',response=response)
    (HERE/'launch-attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
    checked=backend.verify_remote(metadata['id'],response,json.loads(notebook.read_text()),metadata)
    checked.pop('source')
    (HERE/'launch-readback.json').write_text(json.dumps(checked,indent=2)+'\n')
    print(json.dumps(response),flush=True)


if __name__=='__main__':
    main()
