"""Exactly one private launch after qualification, admission and a slot claim."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'analysis/mtp-depth-20261006'))
import launch_watch as lw


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def verify():
    identity=json.loads((HERE/'identity.json').read_text())
    assert identity['hard_cap_seconds']==9000 and identity['generated_token_cap']==524288
    assert hashlib.sha256((HERE/'EXPERIMENT-CARD.md').read_bytes()).hexdigest()==identity['card_sha256']
    metadata=json.loads((HERE/'kernel/kernel-metadata.json').read_text())
    assert metadata==identity['metadata'] and metadata['is_private'] and metadata['machine_shape']=='NvidiaRtxPro6000'
    assert hashlib.sha256((HERE/'kernel'/metadata['code_file']).read_bytes()).hexdigest()==identity['notebook_sha256']
    for name,expected in identity['manifest'].items():
        assert hashlib.sha256((HERE/'package'/name).read_bytes()).hexdigest()==expected,name
    qualification=json.loads((HERE/'evidence/qualification.json').read_text())
    assert qualification['cpu_fixtures_only'] and qualification['package_source_verified']
    assert qualification['notebook_sha256']==identity['notebook_sha256']
    assert [row['status'] for row in qualification['http']]==['solved','http_error']
    assert len(qualification['fixtures'])==4 and len(qualification['trickles'])==2
    return identity


def main():
    evidence=HERE/'evidence'
    if (evidence/'launch-attempt.json').exists():raise RuntimeError('Already attempted; reconcile only')
    identity=verify()
    lw.EVID=evidence;lw.NOTEBOOK_DIR=HERE/'kernel';lw.refresh_token()
    api=lw.KaggleApi();api.authenticate()
    used,quota=lw.quota_used_s(api);active=lw.active_sessions(api)
    admission={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'used_and_reserved_s':used,
               'working_cap_s':108000,'hard_cap_s':9000,'active':active,'quota':quota}
    save(evidence/'admission.json',admission)
    if len(active)>=2 or any(str(status).startswith('unknown') for _,status in active) or used+9000>108000:
        raise RuntimeError('Admission failed; no launch attempted')
    existing=api.kernels_list(mine=True,user='pranavagrawai',page_size=100,sort_by='dateRun')
    if identity['metadata']['id'] in {getattr(kernel,'ref',None) for kernel in existing}:
        raise RuntimeError('Remote identity exists; no launch attempted')
    if '--dry' in sys.argv:
        print(json.dumps(admission,indent=2));return
    claim=f"GPU slot claim: planning-adoption-20261014 v1, one RTX PRO6000, 9000 GPU-seconds TOTAL including startup and evidence, one SaveKernel attempt and no retry. Pranav said give it a try $implement. Sixteen fresh-process continuations: control/planner/probing/both x two original checkpoints x seeds1701/2903; <=32768 generated tokens,32 calls,150 new native actions,900s each; <=524288 tokens total. Used/reserved {used/3600:.3f}h; active {active}. Notebook SHA {identity['notebook_sha256']}; card analysis/planning-adoption-20261014/EXPERIMENT-CARD.md. No submission or promotion."
    subprocess.run([sys.executable,'-B',str(Path.home()/'.agent-board/board.py'),'reply','682',
                    '--as','codex-planning-adoption','--body',claim],check=True)
    attempt={**admission,'state':'attempting','identity_sha256':hashlib.sha256((HERE/'identity.json').read_bytes()).hexdigest()}
    with (evidence/'launch-attempt.json').open('x') as file:json.dump(attempt,file,indent=2)
    try:
        response=json.loads(str(api.kernels_push(str(HERE/'kernel'),timeout='9000')))
        save(evidence/'launch-response.json',response)
        if response.get('error') or not response.get('versionNumber'):raise RuntimeError(str(response))
        attempt.update(state='accepted',response=response)
    except Exception as error:
        attempt.update(state='acceptance_unresolved',error=repr(error))
        response=getattr(error,'response',None)
        if response is not None:attempt.update(http_status=response.status_code,response_body=response.text)
    save(evidence/'launch-attempt.json',attempt)
    print(json.dumps(attempt,indent=2),flush=True)


if __name__=='__main__':main()
