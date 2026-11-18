"""Read-only raw/archive verification; derived files live outside archive members."""
import datetime as dt,hashlib,json,sys,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent;HERE=P.parent;sys.path.insert(0,str(HERE))
import catalog,reduce_api,transport,verify

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 receipt=json.loads((P/'archive-collection.json').read_text())
 if receipt.get('error'):raise ValueError('single archive download failed; no retry')
 archive=P/'output-v1-kernel140391414.zip';assert sha(archive)==receipt['archive_sha256'] and archive.stat().st_size==receipt['archive_bytes']
 extract=P/'archive-extracted';inventory={};seen=set()
 with zipfile.ZipFile(archive) as z:
  for info in z.infolist():
   assert info.filename not in seen;seen.add(info.filename)
   if info.is_dir():continue
   name=Path(info.filename);assert not name.is_absolute() and '..' not in name.parts
   data=z.read(info);path=extract/name;assert data==path.read_bytes()
   inventory[info.filename]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),crc32=f'{info.CRC:08x}')
 raw=extract/'serving-wire-capacity10';report=json.loads((HERE/'review-packet.json').read_text());spec=json.loads((HERE/'packet-spec.json').read_text())
 for name,digest in report['files'].items():assert sha(raw/name)==digest,name
 assert json.loads((raw/'config.json').read_text())==spec['config']
 local_inner=transport.decode((HERE/'kernel'/spec['metadata']['code_file']).read_bytes());assignment=transport.literals(transport.cell_source(json.loads(local_inner)))['ASSIGNMENT']
 assert json.loads((raw/'assignment.json').read_text())==assignment
 remote=(P/'source-terminal.ipynb').read_bytes();values=verify.verify_inner(transport.decode(remote),report,assignment)
 assert transport.sha(transport.decode(remote))=='f4c82c2d321951caf84d906bfcedc9f5be498d80a98321d161189fb4bfc1cb6c'
 frozen=json.loads((HERE/'freeze.json').read_text());assert all(sha(HERE/name)==digest for name,digest in frozen['files'].items())
 value=reduce_api.compare(raw/'output')
 original=json.loads((raw/'output/control.qualify.stdout').read_text());assert value['control']==original,'raw-derived control differs from remote original'
 trace=list(map(json.loads,(raw/'output/supervisor.jsonl').read_text().splitlines()));result=json.loads((raw/'output/result.json').read_text())
 assert sum(r['event']=='spawn' and r['phase'].startswith('candidate.') for r in trace)==0
 assert value['verdict']=='FEASIBILITY_ONLY' and value['ratio'] is None
 for file in (P/'priority-output').rglob('*'):
  if file.is_file():assert file.read_bytes()==(extract/file.relative_to(P/'priority-output')).read_bytes()
 command="python3 -c 'import json,reduce_api; from pathlib import Path; print(json.dumps(reduce_api.compare(Path(\"remote-readback/archive-extracted/serving-wire-capacity10/output\")),indent=2))'"
 (P/'RAW-REDUCER-RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
 (P/'ARCHIVE-INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
 proof=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='ARCHIVE_SOURCE_RAW_REDUCTION_VERIFIED_FEASIBILITY_ONLY',kernel_ref=spec['metadata']['id'],kernel_id=140391414,version=1,
            archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),archive_complete=True,archive_members=len(inventory),all_member_crc_and_extracted_bytes_exact=True,
            inventory_sha256=sha(P/'ARCHIVE-INVENTORY.json'),exact_payload_files=len(report['files']),exact_config_assignment=True,original_frozen_bindings_exact=len(frozen['files']),
            reducer_command=command,reducer_source_sha256=sha(HERE/'reduce_api.py'),derived_result_sha256=sha(P/'RAW-REDUCER-RESULT.json'),remote_control_report_equals_independent_raw=True,
            control_valid=value['control']['valid'],control_measured_completed=value['control'].get('completed'),candidate_spawned=False,
            verdict=value['verdict'],ratio=value['ratio'],runtime_result=result,derived_outputs_separate_from_original_members=True,
            original_archive_and_extracted_members_unmodified=True,resource_release_qualified=False)
 (P/'RAW-REDUCER-RECEIPT.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
