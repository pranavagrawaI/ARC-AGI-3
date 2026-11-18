"""CPU-only primary artifact verification. No API calls or source/receipt rewrites."""
import base64,hashlib,json,sys,zipfile,zlib
from pathlib import Path
P=Path(__file__).resolve().parent;HERE=P.parent;sys.path.insert(0,str(HERE));import transport
sha=lambda data:hashlib.sha256(data).hexdigest()
freeze=json.loads((HERE/'freeze.json').read_text())
for n,h in freeze['files'].items():assert sha((HERE/n).read_bytes())==h,n
for n,h in freeze['references'].items():assert sha(Path(n).read_bytes())==h,n
marker=json.loads((HERE/'launch-attempt.json').read_text());spec=json.loads((HERE/'packet-spec.json').read_text())
local=(HERE/'kernel'/spec['metadata']['code_file']).read_bytes();remote=(P/'source-terminal.ipynb').read_bytes()
assert transport.cell_source(json.loads(remote))==transport.cell_source(json.loads(local))
inner=transport.decode(remote);assert sha(inner)==marker['inner_notebook_sha'];values=transport.literals(transport.cell_source(json.loads(inner)))
archive=P/'output-v1-kernel140425916.zip';receipt=json.loads((P/'archive-collection.json').read_text());assert sha(archive.read_bytes())==receipt['archive_sha256']
assert archive.stat().st_size==receipt['archive_bytes']
inventory={}
with zipfile.ZipFile(archive) as zipped:
 assert zipped.testzip() is None
 assert len(zipped.namelist())==len(set(zipped.namelist()))==25
 for entry in zipped.infolist():
  data=zipped.read(entry);assert zlib.crc32(data)&0xffffffff==entry.CRC
  assert (P/'archive-extracted'/entry.filename).read_bytes()==data
  inventory[entry.filename]=dict(bytes=len(data),sha256=sha(data),crc=entry.CRC)
root=P/'archive-extracted/serving-release11'
for n,record in values['PAYLOAD'].items():assert (root/n).read_bytes()==base64.b64decode(record['base64']),n
assert json.loads((root/'config.json').read_text())==values['CONFIG']
assert json.loads((root/'assignment.json').read_text())==values['ASSIGNMENT']
assert (root/'.one-attempt').read_text()==values['ASSIGNMENT']['serialized_assignment']
terminal=json.loads((P/'terminal-source-status-quota.json').read_text());assert terminal['status']['status']=='ERROR'
assert terminal['metadata_version_identity_verified'] and terminal['decoded_inner_matches_armed']
assert terminal['quota']['reserved_s']==0
log=P/'polls/20261023T170057Z.log';text=''.join(row['data'] for row in json.loads(log.read_text()))
assert "ModuleNotFoundError: No module named 'release_runner'" in text
assert '/kaggle/working/serving-release11/supervisor.py' in text
assert not (root/'output').exists()
monitor=json.loads((P/'monitor-stopped.json').read_text());assert monitor['reason']=='TERMINAL'
report=dict(verdict='BOOTSTRAP_IMPORT_FAILURE_NO_RELEASE_RESULT',kernel_ref=spec['metadata']['id'],version=1,kernel_id=140425916,
 freeze_sha256=sha((HERE/'freeze.json').read_bytes()),frozen_files=len(freeze['files']),frozen_refs=len(freeze['references']),
 exact_payloads=len(values['PAYLOAD']),config_exact=True,assignment_exact=True,source_inner_sha256=sha(inner),
 archive=dict(path=archive.name,bytes=archive.stat().st_size,sha256=receipt['archive_sha256'],members=len(inventory),crc_and_member_bytes_verified=True,inventory=inventory),
 primary_failure="ModuleNotFoundError: No module named 'release_runner' at supervisor.py line2 before Runner construction",
 primary_log_sha256=sha(log.read_bytes()),phase_evidence=dict(hardware='NOT_REACHED',queries='NOT_REACHED',model='NOT_REACHED',release='NOT_REACHED',cleanup='NOT_INVOKED; no fabricated empty-groups receipt'),
 typed_terminal_quota=terminal['quota'],baseline_used_s=67694.232,delta_used_s=round(terminal['quota']['used_s']-67694.232,3),
 monitor_stopped=monitor,schedule_created=False,slot_closure_owned_by_coordinator=True,no_rate_or_release_qualification=True)
with (P/'COLLECTION-PROOF.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in report.items() if k!='archive'},indent=2))
