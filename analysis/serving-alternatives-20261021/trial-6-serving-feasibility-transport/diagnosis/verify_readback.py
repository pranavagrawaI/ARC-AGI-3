import base64,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(P));import transport
r=P/'remote-readback';attempt=json.loads((P/'launch-attempt.json').read_text())
source=(r/'source-latest.ipynb').read_bytes();local=(P/'kernel/serving-candidate-feas6-20261023.ipynb').read_bytes()
meta=json.loads((r/'metadata-latest.json').read_text());raw=transport.decode(source);values=transport.literals(transport.cell_source(json.loads(raw)))
assert meta['id']==140341756 and meta['currentVersionNumber']==1 and meta['ref']=='pranavagrawai/serving-candidate-feas6-20261023'
assert transport.cell_source(json.loads(source))==transport.cell_source(json.loads(local))
assert transport.sha(raw)==attempt['inner_notebook_sha']
root=r/'output/serving-candidate-feas6'
assert json.loads((root/'assignment.json').read_text())==values['ASSIGNMENT']
assert json.loads((root/'config.json').read_text())==values['CONFIG']
for name,record in values['PAYLOAD'].items():
    data=(root/name).read_bytes();assert data==base64.b64decode(record['base64'])
    assert transport.sha(data)==record['sha256']
f=json.loads((P/'freeze.json').read_text())
assert all(transport.sha((P/name).read_bytes())==sha for name,sha in f['files'].items())
status=json.loads((r/'terminal-status-quota.json').read_text());result=json.loads((root/'output/result.json').read_text())
assert status['status']['status']=='ERROR' and status['quota']['reserved_s']==0 and result['surviving_groups']==[]
proof=dict(classification='FEASIBILITY_FAILED_HOST_IDENTITY_GUARD_FALSE_REJECTION',kernel_ref=meta['ref'],kernel_id=meta['id'],version=meta['currentVersionNumber'],
 remote_latest_source_sha256=transport.sha(source),remote_latest_metadata_sha256=transport.sha((r/'metadata-latest.json').read_bytes()),
 remote_wrapper_cell_matches_armed=True,decoded_inner_matches_armed_sha256=True,decoded_inner_sha256=transport.sha(raw),
 all15_remote_extracted_sources_match=True,remote_assignment_config_match=True,
 explicit_version_readback='version_label1 GetKernel HTTP404 retained; latest metadata binds currentVersionNumber1/id140341756',
 failure='Formatted nvidia-smi table truncates exact GPU name; prepare.host_guard line72 compares full name against truncated text.',
 observed_hardware=(root/'output/hardware.stdout').read_text().strip(),
 model_loads=0,HTTP_cases=0,runtime_install_started=False,supervisor_result=result,
 terminal_status='ERROR',quota=status['quota'],frozen53_unchanged=True,retry_performed=False,
 artifacts={str(x.relative_to(P)):transport.sha(x.read_bytes()) for x in sorted(r.rglob('*')) if x.is_file()})
(P/'diagnosis/RESULT.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items() if k!='artifacts'},indent=2))
