"""Read-only exact unassigned packet/binding verification; fresh receipt only."""
import argparse,base64,hashlib,json
from pathlib import Path
import transport,reconciliation
HERE=Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def prove():
    freeze=json.loads((HERE/'freeze.json').read_text());spec=json.loads((HERE/'packet-spec.json').read_text());manifest=json.loads((HERE/'review-packet.json').read_text())
    for name,expected in freeze['files'].items():assert sha(HERE/name)==expected,name
    for name,expected in freeze['references'].items():assert sha(name)==expected,name
    oldroot=HERE.parent/'trial-15-wire-capacity-local';old=json.loads((oldroot/'freeze.json').read_text())
    for name,expected in old['files'].items():assert sha(oldroot/name)==expected,name
    for name,expected in old['references'].items():assert sha(name)==expected,name
    for name,expected in manifest['packet_files'].items():assert sha(HERE/'review'/name)==expected,name
    outer=(HERE/'review'/spec['metadata']['code_file']).read_bytes();inner=transport.decode(outer);values=transport.literals(transport.cell_source(json.loads(inner)))
    assert values['ASSIGNMENT'] is None and values['CONFIG']==spec['config']
    assert {name:item['sha256'] for name,item in values['PAYLOAD'].items()}==manifest['files']
    for name,item in values['PAYLOAD'].items():assert base64.b64decode(item['base64'],validate=True)==Path(spec['files'].get(name,HERE/name)).read_bytes(),name
    proof=json.loads((HERE/'qualification-handoff/PROOF.json').read_text());sdk=json.loads((HERE/'qualification-handoff/SDK.json').read_text());bodies=json.loads((HERE/'qualification-handoff/BODIES.json').read_text())
    assert proof['pass_'] and proof['source_stable'] and not proof['baseline65_rerun'] and all(step['returncode']==0 for step in proof['steps'])
    assembly_delta=[]
    for name,expected in proof['source_hashes'].items():
        if sha(HERE/name)!=expected:
            assert name=='freeze_packet.py',name
            assembly_delta.append(name)
    for name in manifest['files']:
        assert sha(HERE/name)==proof['source_hashes'][name],name
    assert sdk['sources_sha256']==manifest['sources_sha256'] and sdk['config_sha256']==manifest['config_sha256'] and sdk['network_calls']==sdk['network_attempts']==0
    assert bodies['records']==1369 and bodies['catalog_sha256']==sha(HERE/'catalog.json')
    reconciliation.load_policy()
    c=spec['config'];phase_sum=c['hardware']['seconds']+c['corpus_prepare']['seconds']+sum(c['candidate'][name]['seconds'] for name in ('prepare','identity','server','replay','qualify'))+c['final_resource_seconds']+c['verify_seconds']+c['primary_seconds']+60
    assert phase_sum==4600 and c['total_seconds']-c['cleanup_seconds']==4800 and c['platform_seconds']==5400
    assert not any((HERE/name).exists() for name in ('slot-confirmation.json','launch-attempt.json','launch-response.json','kernel'))
    return dict(status='UNASSIGNED_LOCAL_PACKET_VERIFIED',assembly_only_after_qualification=assembly_delta+['prove_review.py'],runtime_payload_qualification_exact=True,freeze_sha256=sha(HERE/'freeze.json'),files=len(freeze['files']),references=len(freeze['references']),bindings=freeze['bindings'],prior15_bindings_verified=len(old['files'])+len(old['references']),payloads=len(manifest['files']),sources_sha256=manifest['sources_sha256'],config_sha256=manifest['config_sha256'],policy_sha256=sha(HERE/'prior-reconciliation-policy.json'),qualification_proof_sha256=sha(HERE/'qualification-handoff/PROOF.json'),qualification_seconds=proof['seconds'],phase_sum_seconds=phase_sum,work_slack_seconds=200,platform_seconds=5400,cooperative_seconds=5100,work_seconds=4800,cleanup_seconds=300,original140_report_only=True,API_calls=0,GPU_work=False,launch_authorized=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();value=prove()
    with a.output.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(value))
