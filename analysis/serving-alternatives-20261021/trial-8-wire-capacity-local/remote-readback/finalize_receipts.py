"""Local read-only evidence verification and derived handoff; no SDK or GPU."""
import datetime as dt,hashlib,json,zlib,zipfile
from decimal import Decimal
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent;E=P/'archive-extracted';R=E/'serving-wire-capacity8';O=R/'output'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=P/'output-v1-kernel140368352.zip';meta=json.loads((P/'archive-collection.json').read_text())
assert sha(archive)==meta['archive_sha256'] and archive.stat().st_size==meta['archive_bytes']
rows=[]
with zipfile.ZipFile(archive) as z:
 for info in z.infolist():
  if info.is_dir():continue
  f=E/info.filename;b=f.read_bytes()
  assert len(b)==info.file_size and zlib.crc32(b)&0xffffffff==info.CRC,info.filename
  rows.append(dict(path=info.filename,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),CRC=info.CRC))
assert len(rows)==2654
inventory={'archive_sha256':sha(archive),'files':rows,'count':len(rows),'derived_files_not_in_original_archive':['serving-wire-capacity8/output/paired-result.json']}
(P/'ARCHIVE-INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
packet=json.loads((ROOT/'review-packet.json').read_text())
for n,h in packet['files'].items():assert sha(R/n)==h,n
config=json.loads((R/'config.json').read_text());assert hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==packet['config_sha256']
assignment=json.loads((R/'assignment.json').read_text());attempt=json.loads((ROOT/'launch-attempt.json').read_text())
assert assignment['approved_freeze_sha256']==attempt['slot']['approved_freeze_sha256']
assert assignment['serialized_assignment']==attempt['slot']['serialized_assignment']
events=[json.loads(line) for line in (O/'supervisor.jsonl').read_text().splitlines()]
spawns=[x['phase'] for x in events if x['event']=='spawn']
assert 'control.replay' not in spawns and not any(x.startswith('candidate.') for x in spawns)
result=json.loads((O/'result.json').read_text());reduced=json.loads((O/'paired-result.json').read_text());terminal=json.loads((P/'terminal-source-status-quota.json').read_text())
assert reduced['verdict']=='FEASIBILITY_ONLY' and reduced['ratio'] is None
assert result['surviving_groups']==[] and terminal['status']['status']=='ERROR'
frozen=json.loads((ROOT/'freeze-r4.json').read_text());assert all(sha(ROOT/n)==h for n,h in frozen['files'].items())
overlap=[]
for f in (P/'priority-output').rglob('*'):
 if f.is_file() and (E/f.relative_to(P/'priority-output')).is_file():
  assert sha(f)==sha(E/f.relative_to(P/'priority-output'));overlap.append(str(f.relative_to(P/'priority-output')))
receipt=dict(status='TERMINAL_ERROR_FEASIBILITY_ONLY_NO_RATE',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
 kernel_ref=meta['kernel_ref'],kernel_id=meta['kernel_id'],version=meta['version'],archive_bytes=meta['archive_bytes'],archive_sha256=meta['archive_sha256'],
 inventory_file='remote-readback/ARCHIVE-INVENTORY.json',inventory_sha256=sha(P/'ARCHIVE-INVENTORY.json'),original_archive_files=len(rows),all_original_sizes_crc_verified=True,
 payload_files_verified=len(packet['files']),sources_sha256=packet['sources_sha256'],config_sha256=packet['config_sha256'],assignment_matches=True,
 final_remote_source_metadata_identity_verified=terminal['metadata_version_identity_verified'],decoded_inner_sha256=terminal['decoded_inner_sha256'],
 freeze_sha256=sha(ROOT/'freeze-r4.json'),frozen_bindings_verified=len(frozen['files']),priority_original_overlap_verified=overlap,
 runtime_result=result,observed_health503='output/control.health-615.stdout; HTTP503, Content-Length0; previous health614 exit75',
 replay_spawns=0,candidate_spawns=0,initial_health_poll_processes=sum(x.startswith('control.health-') for x in spawns),
 cleanup={'seq1867_remaining':[566],'seq1868_remaining':[],'terminal_surviving_groups':result['surviving_groups'],'inter_arm_release_unqualified':True,'final_gpu_idle_receipt_available':False},
 terminal_quota=terminal['quota'],terminal_delta_used_s=str(Decimal(str(terminal['quota']['used_s']))-Decimal('60473.273')),
 reducer_command='python3 reduce_api.py remote-readback/archive-extracted/serving-wire-capacity8/output',reducer_file='remote-readback/reducer.stdout',reducer_sha256=sha(P/'reducer.stdout'),reducer_verdict=reduced['verdict'],ratio=None,
 limits=['No replay started; S10/full-reply fit, package capacity, processor accounting and quality unproved.','No final inter-arm GPU/port/host release receipt because arm aborted.','503 alone cannot distinguish Starting, graceful shutdown or health-generation timeout; pinned source diagnosis separate.'],
 scheduled_monitor_disabled=True,slot_closure='Coordinator-owned; no launch authority remains',source_edits=False,gpu_retries=0)
(P/'TERMINAL-HANDOFF.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('priority_original_overlap_verified','limits')},indent=2))
