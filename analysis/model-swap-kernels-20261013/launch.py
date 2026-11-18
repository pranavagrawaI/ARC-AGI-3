"""One-attempt Tinfield launch. Run with the existing Kaggle SDK Python and -B."""
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'analysis/mtp-depth-20261006'))
import launch_watch as lw
lw.EVID=ROOT
from kaggle.api.kaggle_api_extended import KaggleApi

def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def main():
    d=ROOT/'tinfield';kd=d/'kernel';assert not (d/'launch-attempt.json').exists()
    q=json.loads((d/'LOCAL-QUALIFICATION.json').read_text())
    meta=json.loads((kd/'kernel-metadata.json').read_text());nb=kd/meta['code_file']
    assert hashlib.sha256(nb.read_bytes()).hexdigest()==q['notebook_sha256']
    assert hashlib.sha256((kd/'kernel-metadata.json').read_bytes()).hexdigest()==q['metadata_sha256']
    assert not list(kd.rglob('__pycache__'))
    assert meta['title'].replace(' ','-')==meta['id'].split('/')[1]
    assert q['submission_if_bodies_identical']
    lw.refresh_token();api=KaggleApi();api.authenticate()
    used,quota=lw.quota_used_s(api);active=lw.active_sessions(api)
    save(d/'admission.json',{'utc':lw.now(),'used_and_reserved_s':used,'quota':quota,'active':active,'task_cap_s':12600,'run_cap_s':6300})
    assert len(active)<2 and all(not s.startswith('unknown') for _,s in active),active
    assert used+6300<=30*3600
    kernels=api.kernels_list(mine=True,user='pranavagrawai',page_size=100,sort_by='dateRun')
    assert meta['id'] not in {getattr(k,'ref',None) for k in kernels}
    datasets=api.dataset_list(mine=True,search=meta['id'].split('/')[1])
    assert meta['id'] not in {getattr(k,'ref',None) for k in datasets}
    # The claim immediately precedes the only push, after live admission.
    body=f'GPU slot claim: codex-model-swap Tinfield smoke {meta["id"]} v1, one slot, hard cap 6300 s (1.75 h), 40 min public play. Live used/reserved {used/3600:.3f} h; active {active}. Task <=3.5 GPU-h; no AZ launch (quantized routed expert changes). Named c14 base, unchanged INT4 MTP and BF16 PLE, ModelOpt NVFP4 target; notebook SHA256 {q["notebook_sha256"]}. Card tinfield/EXPERIMENT-CARD.md. No automatic retry or competition-submit call.'
    subprocess.run([sys.executable,'-B',str(Path.home()/'.agent-board/board.py'),'reply','682','--as','codex-model-swap','--body',body],check=True)
    identity={**q,'metadata':meta,'run_cap_s':6300,'card_sha256':hashlib.sha256((d/'EXPERIMENT-CARD.md').read_bytes()).hexdigest()}
    save(d/'launch-identity.json',identity)
    with (d/'launch-attempt.json').open('x') as f:json.dump({'utc':lw.now(),'state':'attempting','identity':identity},f,indent=2)
    try:
        resp=json.loads(str(api.kernels_push(str(kd),timeout='6300')))
        save(d/'launch-response.json',resp)
        assert resp.get('versionNumber') and not resp.get('error'),resp
        save(d/'launch-attempt.json',{'utc':lw.now(),'state':'accepted','response':resp,'identity':identity})
        print(json.dumps(resp,indent=2),flush=True)
    except Exception as e:
        save(d/'launch-error.json',{'utc':lw.now(),'error':repr(e),'acceptance':'unresolved; reconcile read-only; do not push again'})
        raise

if __name__=='__main__':main()
