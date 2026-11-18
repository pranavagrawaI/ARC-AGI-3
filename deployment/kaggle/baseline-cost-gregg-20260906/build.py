"""Deterministic offline notebook assembly. Does not upload or launch."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DONOR=HERE.parent/'actor-mtp-20260906'


def sha(data): return hashlib.sha256(data).hexdigest()


def build():
    manifest=json.loads((DONOR/'manifest.json').read_text())
    raw=(DONOR/'actor_mtp.ipynb').read_bytes()
    assert sha(raw)==manifest['notebook_sha256'], 'Frozen donor drift'
    notebook=json.loads(raw)
    assets=''.join(notebook['cells'][2]['source'])
    hashes={}
    sources={'actor-replay.py':HERE/'runner.py','actor-reference.py':DONOR/'runner.py',
             'watchdog.py':HERE.parent/'mtp-sampling-repair-20260906'/'watchdog.py'}
    for name,path in sources.items():
        data=path.read_bytes();hashes[name]=sha(data)
        assets+=f"\n_data=base64.b64decode({base64.b64encode(data).decode()!r})\nassert hashlib.sha256(_data).hexdigest()=={sha(data)!r}\nPath('/tmp/{name}').write_bytes(_data)\n"
    notebook['cells'][2]['source']=assets
    notebook['cells'][0]['source']='# Baseline context cost and finite-workload batching\nOffline RTX PRO 6000 diagnostic; no gameplay, tools, submission, automatic optimization or promotion. Read the experiment card.\n'
    notebook['cells'][4]['source']="""import subprocess, sys
from pathlib import Path
out=Path('/kaggle/working/baseline-cost-gregg-20260906');out.mkdir(exist_ok=True)
with (out/'runner.log').open('w') as log:
    completed=subprocess.run([sys.executable,'/tmp/actor-replay.py'],stdout=log,stderr=subprocess.STDOUT,timeout=5700)
print((out/'runner.log').read_text()[-12000:])
assert completed.returncode==0, 'Diagnostic failed; retain artifacts, do not relaunch automatically'
"""
    for cell in notebook['cells']:
        if cell['cell_type']=='code': compile(''.join(cell['source']),'notebook','exec')
    data=(json.dumps(notebook,indent=1)+'\n').encode()
    meta=json.loads((DONOR/'kernel-metadata.json').read_text())
    meta.update(id='pranavagrawai/baseline-cost-gregg-20260906',title='Baseline Cost Gregg 20260906',code_file='baseline_cost.ipynb')
    assert meta['competition_sources']==['arc-prize-2026-arc-agi-3']
    return {'baseline_cost.ipynb':data,
        'kernel-metadata.json':(json.dumps(meta,indent=2)+'\n').encode(),
        'manifest.json':(json.dumps({'schema':'baseline-cost-v1','donor_notebook_sha256':sha(raw),
            'corpus_sha256':manifest['corpus_sha256'],'notebook_sha256':sha(data),'assets':hashes,
            'card_sha256':sha((HERE/'EXPERIMENT-CARD.md').read_bytes()),
            'runner_cap_s':5400,'outer_runner_cap_s':5700,'explicit_request_cap':72,
            'source_pin_sha256':sha((HERE/'SOURCE-PIN.json').read_bytes()),'upload_ready':True,
            'promotion':False},indent=2)+'\n').encode()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    for name,data in build().items():
        if args.check: assert (HERE/name).read_bytes()==data,name
        else: (HERE/name).write_bytes(data)
    print('Package verified; no upload or launch')
