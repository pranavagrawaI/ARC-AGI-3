"""Codec size witness only; synthetic maximum-count shapes never scientific evidence."""
import base64,copy,hashlib,json,time
from pathlib import Path
import primary_bundle as p
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'trial-16-candidate-s10-r4-operational-local/remote-readback/current-primary-fallback-local/primary-current/PRIMARY.json'
def main(destination):
 destination=Path(destination);destination.mkdir(exist_ok=False)
 original=json.loads(SOURCE.read_bytes());value=copy.deepcopy(original);value['schema']=p.SCHEMA
 real=json.loads(base64.b64decode(original['files']['result.json']['base64']));arm=real['within_arm']
 catalog=json.loads((HERE/'catalog.json').read_bytes());records=catalog['records'];items=list(records.values()) if isinstance(records,dict) else records
 accounting=list(arm['accounting'].values());measured=arm['measured_rows']
 pair={}
 for label in ('candidate','control'):
  bounded=copy.deepcopy(arm);bounded['accounting']={};bounded.pop('measured_rows',None)
  for index in range(1369):
   row=copy.deepcopy(accounting[index%len(accounting)])
   # Distinct nonrepeating hashes from real immutable wire catalog, maximum 58 PNGs.
   record=items[index%len(items)];hashes=record['images'] or row['source_png_sha256']
   row['source_png_sha256']=(hashes*58)[:58]
   bounded['accounting'][f'pair17-{label}-{index}']=row
   # Actual paired reducer removes measured_rows from both returned arms.
   pass
  pair[label]=bounded
 result=dict(schema='SIZE_WITNESS_NOT_SCIENTIFIC_PASS',arms=pair,real_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
 result_bytes=json.dumps(result,sort_keys=True).encode();value['files']['paired-result.json']=p.record(result_bytes)
 # Actual top-level result/terminal contain verdict/ratio and cleanup, not duplicated arm accounting.
 top={key:entry for key,entry in real.items() if key!='within_arm'}
 value['files']['result.json']=p.record(json.dumps(top,sort_keys=True).encode())
 trace=base64.b64decode(original['files']['supervisor.jsonl']['base64']);rows=[json.loads(x) for x in trace.splitlines()];base=[x for x in rows if x['event']!='terminal']
 # Preserve real trace row shapes and exceed observed two-arm health count expansion.
 pressure=base*5+[dict(seq=len(base)*5,event='terminal',t=10200,**top)]
 value['files']['supervisor.jsonl']=p.record(b'\n'.join(json.dumps(x,sort_keys=True).encode() for x in pressure)+b'\n')
 value['oom_projection']=[dict(original['oom_projection'][i%5384],start=i*.5,end=i*.5+.01) for i in range(20401)]
 value['health_inventory']=[dict(original['health_inventory'][i%946],path=f'{"control" if i>=1802 else "candidate"}.health-{i}.stdout') for i in range(4504)]
 # Add both complete raw event indexes scaled beyond real observed request counts.
 raw_events=(HERE.parent/'trial16-three-pass-audit-result-20261024/evidence/serving-candidate-s10-16/output/candidate/events.jsonl').read_bytes()
 for label in ('candidate','control'):
  value['files'][label+'/events.jsonl']=p.record(raw_events*3)
 # Full retained resource receipt shapes at both shared180 duration maxima.
 native_root=HERE.parent/'trial-14-phase-handles-local/remote-readback/archive-v1-kernel140449504/members/serving-post-control-load14/output'
 candidate_release=json.loads(base64.b64decode(original['files']['candidate-release.json']['base64']))
 native_release=json.loads((native_root/'release.json').read_bytes())
 def varied(sample,index):
  row=copy.deepcopy(sample)
  def walk(value):
   if isinstance(value,dict):
    for key,child in value.items():
     if key=='sha256' and isinstance(child,str):value[key]=hashlib.sha256((str(index)+child).encode()).hexdigest()
     elif key=='path' and isinstance(child,str):value[key]=child+str(index)
     else:walk(child)
   elif isinstance(value,list):
    for child in value:walk(child)
  walk(row);return row
 candidate_release['samples']=[varied(candidate_release['samples'][-1],i) for i in range(91)]
 native_release['samples']=[varied(native_release['samples'][-1],i) for i in range(92)]
 value['files']['candidate-release.json']=p.record(json.dumps(candidate_release,sort_keys=True).encode())
 value['files']['control-admission.json']=p.record(json.dumps(dict(sample=varied(candidate_release['samples'][-1],92)),sort_keys=True).encode())
 value['files']['release.json']=p.record(json.dumps(native_release,sort_keys=True).encode())
 value['files']['pre-release.json']=p.record(json.dumps(varied(native_release['samples'][-1],93),sort_keys=True).encode())
 for key in ('native-lease.json','release-scope.json','control-package/control-prepared.json','control-package/control-source-identity.json',
             'control-package/control-model-identity.json','control-package/control-frontend-registration.json','control-package/control-health-identity.json',
             'control-package/control-health-serve.json','control-package/control-package-identity.json'):
  value['files'][key]=p.record((native_root/key).read_bytes())
 # Include every actual mandatory primary slot; binding-size surrogates are explicitly not semantic receipts.
 actual_candidate=HERE.parent/'trial16-three-pass-audit-result-20261024/evidence/serving-candidate-s10-16/output'
 for label,folder in (('candidate',actual_candidate),('control',native_root)):
  for key in ('actual-gpu-query.json','host-headroom.json'):
   value['files'][label+'-package/'+key]=p.record((folder/(label+'-package')/key).read_bytes())
 registration=json.loads((native_root/'control-package/control-frontend-registration.json').read_bytes())
 for label in ('candidate','control'):
  value['files'][label+'-server-binding.json']=p.record(json.dumps(dict(SIZE_SURROGATE_NOT_A_RECEIPT=[registration]*4),sort_keys=True).encode())
 if not set(p.REQUIRED)<=set(value['files']):raise ValueError('size witness omitted mandatory primary slots')
 spec=json.loads((HERE/'packet-spec.json').read_bytes())
 value['source_sha256']={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in spec['files'].items() if name.endswith('.py')}
 # Keep all original bytes, add another copy under the second package/arm path.
 for key,record in original['files'].items():
  if key.startswith('candidate') and key not in ('candidate.server.stdout','candidate.server.stderr'):
   value['files']['control-shape/'+key]=record
 for key in ('candidate.server.stdout','candidate.server.stderr','control.server.stdout','native/stderr.log'):
  data=bytes(range(256))*256;value['files'][key]=p.record(data)
 begin=time.monotonic();encoded=p.encode(value);elapsed=time.monotonic()-begin
 assert p.verify(encoded)==value
 report=dict(scope='SIZE_ONLY_NOT_SCIENCE',source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
   real_primary_bytes=SOURCE.stat().st_size,requests_per_arm=1369,arms=2,observer_samples=20401,health_inventory=4504,candidate_release_samples=91,fresh_samples=1,native_release_samples=92,pre_release_samples=1,
   decoded_bytes=json.loads(encoded)['decoded_bytes'],encoded_bytes=len(encoded),limit=p.MAX_BYTES,encode_seconds=elapsed,
   roundtrip_lossless=True,scientific_dependencies_omitted=False,shape_note='Every mandatory primary slot represented; paired-result contains two-arm accounting; top result/terminal use actual producer layout; indexes tripled from retained633-send capture; binding size surrogates are four retained registrations, not fabricated semantic receipts',
   limitation='Real retained shape/count pressure is not a universal entropy bound; runtime rejects any encoded or decoded excess rather than truncating scientific files.')
 (destination/'PRIMARY-SIZE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
 import sys
 main(sys.argv[1])
