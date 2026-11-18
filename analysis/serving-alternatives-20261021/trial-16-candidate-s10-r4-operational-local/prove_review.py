"""Read-only exact operational rebind proof; no API or runtime requalification."""
import argparse,base64,hashlib,json
from pathlib import Path
import reconciliation,transport
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'trial-16-candidate-s10-r3-operational-local'
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def prove():
    freeze=json.loads((HERE/'freeze.json').read_text());old=json.loads((PREVIOUS/'freeze.json').read_text())
    for root,value in ((HERE,freeze),(PREVIOUS,old)):
        for name,expected in value['files'].items():assert sha(root/name)==expected,name
        for name,expected in value['references'].items():assert sha(name)==expected,name
    manifest=json.loads((HERE/'review-packet.json').read_text());prior=json.loads((PREVIOUS/'review-packet.json').read_text())
    spec=json.loads((HERE/'packet-spec.json').read_text());previous_spec=json.loads((PREVIOUS/'packet-spec.json').read_text())
    assert manifest['files']==prior['files'] and manifest['sources_sha256']==prior['sources_sha256'] and manifest['config_sha256']==prior['config_sha256']
    assert spec['config']==previous_spec['config'] and spec['metadata']==previous_spec['metadata']
    for name,expected in manifest['packet_files'].items():assert sha(HERE/'review'/name)==expected,name
    outer=(HERE/'review'/spec['metadata']['code_file']).read_bytes();assert outer==(PREVIOUS/'review'/spec['metadata']['code_file']).read_bytes()
    inner=transport.decode(outer);values=transport.literals(transport.cell_source(json.loads(inner)))
    assert values['ASSIGNMENT'] is None and values['CONFIG']==spec['config']
    assert {k:v['sha256'] for k,v in values['PAYLOAD'].items()}==manifest['files']
    for name,item in values['PAYLOAD'].items():assert base64.b64decode(item['base64'],validate=True)==Path(spec['files'].get(name,HERE/name)).read_bytes(),name
    qualification=json.loads((HERE/'qualification-operational/PROOF.json').read_text());sdk=json.loads((HERE/'qualification-operational/SDK.json').read_text())
    assert qualification['pass_'] and qualification['source_stable'] and not qualification['runtime61_rerun'] and not qualification['baseline65_rerun']
    for name,expected in qualification['source_hashes'].items():assert sha(HERE/name)==expected,name
    assert sdk['sources_sha256']==prior['sources_sha256'] and sdk['config_sha256']==prior['config_sha256'] and sdk['network_calls']==sdk['network_attempts']==0
    assert sha(HERE/'operational-evidence/coordinator-wire-diagnosis/response-body.json')=='34d4ca81483ff40c6341d6f20f6464e82a70a458d66622aa5d43aa6879c83355'
    policy=reconciliation.load_policy();assert policy['historical_predecessor_used_s']==77433.978 and policy['current_quota']==reconciliation.REVIEWED_QUOTA
    assert not any((HERE/name).exists() for name in ('slot-confirmation.json','launch-attempt.json','launch-response.json','kernel','collection-space-prelaunch.json','quota-prelaunch.json','quota-before-prelaunch.json','quota-after-prelaunch.json','quota-pair-prelaunch.json','sessions-prelaunch.json','predecessor-status-prelaunch.json','failed-save-reconciliation-prelaunch.json'))
    return dict(status='UNASSIGNED_OPERATIONAL_WIRE_REBIND_VERIFIED',freeze_sha256=sha(HERE/'freeze.json'),bindings=freeze['bindings'],files=len(freeze['files']),references=len(freeze['references']),
                preserved_r3_bindings_verified=old['bindings'],payloads=42,payloads_config_metadata_exact=True,unassigned_outer_byte_exact=True,sources_sha256=manifest['sources_sha256'],config_sha256=manifest['config_sha256'],
                quota_wire_helper_sha256=sha(HERE/'quota_wire.py'),genuine_wire_body_sha256=sha(HERE/'operational-evidence/coordinator-wire-diagnosis/response-body.json'),policy_sha256=sha(HERE/'prior-reconciliation-policy.json'),qualification_proof_sha256=sha(HERE/'qualification-operational/PROOF.json'),qualification_seconds=qualification['seconds'],current_quota=policy['current_quota'],
                historical_predecessor_used_s=policy['historical_predecessor_used_s'],platform_seconds=5400,API_calls=0,GPU_work=False,runtime61_rerun=False,baseline65_rerun=False,launch_authorized=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();value=prove()
    with a.output.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(value))
