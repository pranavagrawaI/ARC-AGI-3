"""Independent frozen reducer on joined CRC-prefix + missing-file raw evidence.

No original members modified; no full-archive verification claim.
"""
import datetime as dt,hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;HERE=P.parent;sys.path.insert(0,str(HERE))
import catalog,reduce_api,transport,verify

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 recovery=json.loads((P/'partial-members/RECOVERY.json').read_text());members={r['name']:r for r in recovery['entries']}
 missing=json.loads((P/'missing-raw-collection.json').read_text());assert not missing.get('error'),missing.get('error')
 raw=P/'partial-members/members/serving-wire-capacity10/output/control';additional=P/'missing-raw/serving-wire-capacity10/output/control'
 view=P/'raw-view';view.mkdir(exist_ok=False);dest=view/'output/control';dest.mkdir(parents=True)
 origins={}
 for origin,directory in [('CRC_VERIFIED_PARTIAL_ZIP',raw),('SCOPED_MISSING_FILE_FETCH',additional)]:
  for file in directory.rglob('*'):
   if not file.is_file():continue
   name=file.relative_to(directory);target=dest/name
   assert not target.exists(),str(name)
   if origin=='CRC_VERIFIED_PARTIAL_ZIP':
    item=members['serving-wire-capacity10/output/control/'+str(name)];assert sha(file)==item['sha256'] and file.stat().st_size==item['bytes']
   target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(file.resolve())
   origins[str(name)]={'origin':origin,'sha256':sha(file),'bytes':file.stat().st_size}
 events=list(map(json.loads,(dest/'events.jsonl').read_text().splitlines()));sends=[r for r in events if r['kind']=='send'];responses={r['rid'] for r in events if r['kind']=='response'}
 for send in sends:
  directory=dest/send['rid'];http=json.loads((directory/'http.json').read_text())
  for name in ('wire.http','wire.json'):assert (directory/name).is_file(),str(directory/name)
  if http['status']==200:assert (directory/'body.bin').is_file(),str(directory/'body.bin')
 report=json.loads((HERE/'review-packet.json').read_text());spec=json.loads((HERE/'packet-spec.json').read_text())
 crc_source_files=[];missing_archive_source_files=[]
 for name,digest in report['files'].items():
  member=members.get('serving-wire-capacity10/'+name)
  if member is None:missing_archive_source_files.append(name)
  else:
   assert member['sha256']==digest,name
   crc_source_files.append(name)
 assert members['serving-wire-capacity10/config.json']['sha256']==report['config_sha256']
 inner=transport.decode((HERE/'kernel'/spec['metadata']['code_file']).read_bytes());assignment=transport.literals(transport.cell_source(json.loads(inner)))['ASSIGNMENT']
 assert members['serving-wire-capacity10/assignment.json']['sha256']==hashlib.sha256(json.dumps(assignment).encode()).hexdigest()
 remote=(P/'source-terminal.ipynb').read_bytes();verify.verify_inner(transport.decode(remote),report,assignment)
 assert transport.sha(transport.decode(remote))=='f4c82c2d321951caf84d906bfcedc9f5be498d80a98321d161189fb4bfc1cb6c'
 frozen=json.loads((HERE/'freeze.json').read_text());assert all(sha(HERE/name)==digest for name,digest in frozen['files'].items())
 value=reduce_api.compare(view/'output')
 primary=P/'priority-output/serving-wire-capacity10/output';original=json.loads((primary/'control.qualify.stdout').read_text())
 assert value['control']==original,'independent control reduction differs from original'
 assert value['verdict']=='FEASIBILITY_ONLY' and value['ratio'] is None
 for path,name in ((P/'RAW-REDUCER-RESULT.json',value),(P/'RAW-ORIGINS.json',origins)):
  with path.open('x') as stream:json.dump(name,stream,indent=2);stream.write('\n')
 command="python3 -c 'import json,reduce_api; from pathlib import Path; print(json.dumps(reduce_api.compare(Path(\"remote-readback/raw-view/output\")),indent=2))'"
 proof=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='COMPLETE_CONTROL_RAW_JOIN_VERIFIED_PARTIAL_ARCHIVE',kernel_ref=spec['metadata']['id'],kernel_id=140391414,version=1,
            archive_complete=False,archive_sha256=recovery['archive_sha256'],archive_bytes=recovery['archive_bytes'],crc_verified_local_members=recovery['verified_members'],central_directory_verified=False,
            raw_control_files=len(origins),request_attempts=len(sends),complete_response_events=len(responses),pending=len(sends)-len(responses),all_required_control_raw_files_present=True,
            raw_origins_sha256=sha(P/'RAW-ORIGINS.json'),exact_remote_embedded_payloads=len(report['files']),crc_verified_archive_source_files=crc_source_files,archive_source_files_unavailable=missing_archive_source_files,exact_config_assignment=True,original_frozen_bindings_exact=len(frozen['files']),
            reducer_source_sha256=sha(HERE/'reduce_api.py'),reducer_command=command,derived_result_sha256=sha(P/'RAW-REDUCER-RESULT.json'),remote_control_report_equals_independent_raw=True,
            control_valid=value['control']['valid'],control_measured_completed=value['control'].get('completed'),candidate_spawned=False,verdict=value['verdict'],ratio=None,
            resource_release_qualified=False,limitation='Full ZIP/central directory unavailable; full control raw data assembled from CRC-verified prefix and precisely scoped direct output fetch. No candidate or paired contrast.')
 with (P/'RAW-REDUCER-RECEIPT.json').open('x') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
 print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
