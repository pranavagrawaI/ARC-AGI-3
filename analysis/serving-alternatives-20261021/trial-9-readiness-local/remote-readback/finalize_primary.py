"""Finalize collected PRIMARY evidence only. The interrupted archive stays explicitly incomplete."""
import base64,collections,datetime as dt,hashlib,json,sys
from decimal import Decimal
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.path.insert(0,str(ROOT))
import catalog,control_prepare,reduce_api,transport
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
S=P/'priority-output/serving-wire-capacity9/output';C=P/'priority-events/serving-wire-capacity9/output/control';H=P/'priority-pending/serving-wire-capacity9/output/control'
trace=[json.loads(x) for x in (S/'supervisor.jsonl').read_text().splitlines()];events=[json.loads(x) for x in (C/'events.jsonl').read_text().splitlines()]
assert [x['seq'] for x in events]==list(range(len(events))) and [x['seq'] for x in trace]==list(range(len(trace)))
end=next(x['boundary'] for x in events if x['kind']=='measure_end');errors=[x for x in events if x['kind']=='error'];assert len(errors)==1
rid=errors[0]['rid'];failed=read(H/rid/'http.json');assert rid.endswith('-550')
validations=[]
for path in sorted(H.glob('*/http.json')):
 v=read(path)
 try:reduce_api.pending_http(v,end,lambda ref:(H/ref).read_bytes(),v['rid']);valid=True;error=None
 except (ValueError,KeyError,OSError) as e:valid=False;error=str(e)
 validations.append(dict(rid=v['rid'],raw_receipt=str(path.relative_to(ROOT)),sha256=sha(path),valid_under_frozen_pending_gate=valid,rejection=error,termination=v['termination'],cause=v['cause'],error_observed_at=v['error_observed_at'],cutoff_shutdown_at=v['cutoff_shutdown_at']))
assert len(validations)==10 and sum(x['valid_under_frozen_pending_gate'] for x in validations)==9
body=reduce_api.transport((H/rid/'wire.http').read_bytes(),rid);assert body==read(H/rid/'wire.json')['body']
assert catalog.wire_hash(body)==catalog.load()['records'][failed['key']]['wire_sha']
assert sum(x['bytes'] for x in failed['writes'])==(H/rid/'wire.http').stat().st_size
start=events[0]['t'];counts=dict(collections.Counter(x['kind'] for x in events))
health_a=read(S/'control-package/control-health-identity.json');health_b=read(S/'control-package/control-health-serve.json')
assert control_prepare.verify_health_record(health_a,'identity')==control_prepare.verify_health_record(health_b,'serve')
terminal=read(P/'terminal-source-status-quota.json');assert terminal['status']['status']=='ERROR' and terminal['metadata_version_identity_verified']
outer=(P/'source-terminal.ipynb').read_bytes();values=transport.literals(transport.cell_source(json.loads(transport.decode(outer))))
report=read(ROOT/'review-packet.json');embedded={n:hashlib.sha256(base64.b64decode(r['base64'],validate=True)).hexdigest() for n,r in values['PAYLOAD'].items()};assert embedded==report['files']
assert hashlib.sha256(json.dumps(values['CONFIG'],sort_keys=True).encode()).hexdigest()==report['config_sha256']
assert values['ASSIGNMENT']['serialized_assignment']==read(ROOT/'launch-attempt.json')['slot']['serialized_assignment']
frozen=read(ROOT/'freeze.json');assert all(sha(ROOT/n)==h for n,h in frozen['files'].items())
result=read(S/'result.json');assert result['surviving_groups']==[]
assert not any(x['event']=='spawn' and x['phase'].startswith('candidate') for x in trace)
reduced=read(C.parent/'paired-result.json');assert reduced['verdict']=='FEASIBILITY_ONLY' and reduced['ratio'] is None
files=[]
for directory in ('priority-output','priority-events','priority-pending'):
 for path in sorted((P/directory).rglob('*')):
  if path.is_file():files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path),derived=path.name=='paired-result.json'))
for n in ('source-latest.ipynb','source-terminal.ipynb','metadata-latest.json','metadata-terminal.json','initial-source-status-quota.json','terminal-source-status-quota.json','archive-collection.json','PARTIAL-ARCHIVE-RECEIPT.json','priority-collection.json','events-collection.json','pending-collection.json','reducer.stdout','reducer.stderr','scheduled-monitor-disabled.json'):
 path=P/n;files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path),derived=False))
(P/'PRIMARY-INVENTORY.json').write_text(json.dumps(dict(scope='PRIMARY COLLECTION ONLY; not complete response/API/archive audit',files=files,count=len(files)),indent=2)+'\n')
ordering=dict(rid=rid,key=failed['key'],fixed_cutoff=end,error_observed_at=failed['error_observed_at'],owned_shutdown_at=failed['cutoff_shutdown_at'],error_event_published_at=errors[0]['t'],last_write_end=failed['writes'][-1]['ended'],observed_minus_cutoff_s=failed['error_observed_at']-end,shutdown_minus_observed_s=failed['cutoff_shutdown_at']-failed['error_observed_at'],no_headers_or_body=failed['status'] is None and failed['received_bytes']==0,body_file_exists=(H/rid/'body.bin').exists(),timer_arm_receipt_available=False,read_phase_receipt_available=False,socket_state_receipt_available=False,interpretation='Recorded timeout precedes owned-shutdown marker; both follow cutoff. Exact timer arming/provenance is not observed. Original frozen classifier and pending reducer reject it; no retroactive relabeling.')
(P/'PENDING-CLOCK-RECEIPT.json').write_text(json.dumps(dict(ordering=ordering,pending_validation=validations,wire_550_transport_plan_corpus_equal=True,wire_550_write_bytes_exact=True),indent=2)+'\n')
receipt=dict(status='CLOSED_ERROR_FEASIBILITY_ONLY_PARTIAL_ARCHIVE',utc=dt.datetime.now(dt.timezone.utc).isoformat(),kernel_ref='pranavagrawai/serving-wire-capacity9-20261023',kernel_id=140375737,version=1,primary_inventory='remote-readback/PRIMARY-INVENTORY.json',primary_inventory_sha256=sha(P/'PRIMARY-INVENTORY.json'),primary_inventory_files=len(files),complete_archive_verified=False,partial_archive=read(P/'PARTIAL-ARCHIVE-RECEIPT.json'),original_frozen_bindings_verified=len(frozen['files']),sources_sha256=report['sources_sha256'],config_sha256=report['config_sha256'],exact_remote_24_payload_config_assignment=True,runtime_result=result,event_counts=counts,first_error=errors[0],pending_clocks=ordering,candidate_spawns=0,interarm_release_qualified=False,actual_health_identity_serve_match=True,terminal_quota=terminal['quota'],terminal_delta_s=str(Decimal(str(terminal['quota']['used_s']))-Decimal('61844.901')),reducer_verdict=reduced['verdict'],ratio=None,reducer_command='python3 reduce_api.py remote-readback/priority-events/serving-wire-capacity9/output',reducer_scope='Complete failure event ledger, incomplete per-response raw collection; raw/API/occupancy/rate gates remain unaudited.',scheduled_monitor_disabled=True,archive_retry_attempted=False,slot_status='Coordinator CLOSED/RELEASED; no launch authority',remaining_evidence_gaps=['Full version-bound archive incomplete after stream failure; no full member CRC audit.','541 completed response raw/API/image/usage gates not audited.','No candidate phase or paired contrast.','Timer arm/read-stage/socket-state receipt absent.'])
(P/'TERMINAL-HANDOFF.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('status','primary_inventory_files','primary_inventory_sha256','event_counts','terminal_delta_s','reducer_verdict','ratio','complete_archive_verified')},indent=2))
