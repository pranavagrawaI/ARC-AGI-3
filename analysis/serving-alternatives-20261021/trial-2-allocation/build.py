"""Build and test one hardware-only notebook. Does not contact Kaggle."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SLUG='serving-rtx-envelope-probe-20261021'

PROBE='''import csv, hashlib, json, pathlib, subprocess, sys, time
CONFIG = __CONFIG__
OUT=pathlib.Path('/kaggle/working/probe.json')
def classify(query):
    if query['returncode'] != 0: return False
    try:
        rows=list(csv.reader(query['stdout'].strip().splitlines()))
        return len(rows)==1 and 'rtx pro 6000' in rows[0][0].lower() and int(rows[0][1].strip())>=90000
    except (ValueError,IndexError): return False
def command(args):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=15)
        return {'command':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    except Exception as e:
        return {'command':args,'returncode':None,'stdout':'','stderr':repr(e)}
record={'probe_id':CONFIG['id'],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'requested_metadata':CONFIG,'requested_metadata_sha256':hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest(),
        'python':sys.version,'no_runtime_install':True,'no_model_load':True}
record['nvidia_smi']=command(['nvidia-smi'])
record['gpu_query']=command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'])
record['pass']=bool(record['nvidia_smi']['returncode']==0 and classify(record['gpu_query']))
for key,path in [('cgroup_memory_max','/sys/fs/cgroup/memory.max'),('os_release','/etc/os-release')]:
    p=pathlib.Path(path)
    record[key]=p.read_text() if p.exists() else None
OUT.write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record,indent=2),flush=True)
assert record['pass'], 'allocation probe failed; stop GPU work; no retry'
'''

def main():
    donor_path=ROOT/'analysis/flare-desk-20261018/smoke3e/kernel/kernel-metadata.json'
    donor=json.loads(donor_path.read_text())
    meta={**donor,'id':'pranavagrawai/'+SLUG,'title':SLUG,'code_file':SLUG+'.ipynb',
          'dataset_sources':[],'kernel_sources':[],'model_sources':[]}
    assert meta['competition_sources']==['arc-prize-2026-arc-agi-3']
    assert meta['machine_shape']=='NvidiaRtxPro6000'
    assert 'docker_image_pinning_type' not in meta  # preserves successful envelope
    code=PROBE.replace('__CONFIG__',repr(meta));compile(code,'probe.py','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
        'cells':[{'id':'hardware-envelope','cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(keepends=True)}]}
    import nbformat
    nbformat.validate(nbformat.from_dict(nb))
    out=HERE/'kernel';out.mkdir(exist_ok=True)
    (out/meta['code_file']).write_text(json.dumps(nb,indent=1)+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (HERE/'probe.py').write_text(code)
    old=json.loads((HERE.parent/'trial-1/kernel/kernel-metadata.json').read_text())
    delta={k:{'trial1':old.get(k),'probe':meta.get(k)} for k in sorted(set(old)|set(meta)) if old.get(k)!=meta.get(k)}
    (HERE/'config-delta.json').write_text(json.dumps(delta,indent=2)+'\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    freeze={'hard_timeout_s':180,'coordinator':'d74e17d0-679e-4189-a52b-bfcc9e6895be',
            'successful_envelope_source':str(donor_path.relative_to(ROOT)), 'successful_envelope_sha256':sha(donor_path),
            'files':{str(p.relative_to(HERE)):sha(p) for p in [HERE/'build.py',HERE/'probe.py',HERE/'verify.py',HERE/'launch.py',HERE/'EXPERIMENT-CARD.md',HERE/'config-delta.json',*out.iterdir()]}}
    (HERE/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps({'kernel':meta['id'],'hard_timeout_s':180,'launch_status':'held_for_coordinator'}))

if __name__=='__main__':main()
