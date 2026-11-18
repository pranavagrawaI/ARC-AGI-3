"""Deterministic upfront dependency recipe; no outcome-based file selection."""
import base64,hashlib,json
from pathlib import Path
from collector import require,safe_name

def build(bundle,payload):
 files=bundle['files'];decoded={key:base64.b64decode(row['base64'],validate=True) for key,row in files.items()}
 cfg=json.loads(decoded['config.json']);prefix=Path(cfg['cwd']).name;output=prefix+'/output/'
 rows={};embedded={};tail={row['path'] for row in bundle['truncated']}
 def add(name,conditional=False,origin='RAW_REQUIRED',expected_bytes=None,expected_sha256=None):
  safe_name(name);record=dict(name=name,conditional=conditional,dependency=origin,expected_bytes=expected_bytes,expected_sha256=expected_sha256)
  if name in rows:
   require(rows[name]['conditional']==conditional,'contradictory required-file condition')
   if expected_sha256 is None:return
   require(rows[name]['expected_sha256'] in (None,expected_sha256),'contradictory required-file hash')
  rows[name]=record
 for key,data in decoded.items():
  name=(prefix+'/'+key) if key in ('config.json','assignment.json') else output+key
  if key!='result.json' and key not in tail:embedded[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),primary_key=key)
  else:add(name,origin='INDEPENDENT_ACTUAL_RESULT' if key=='result.json' else 'FULL_DIAGNOSTIC_LOG',expected_bytes=len(data) if key=='result.json' else None,expected_sha256=hashlib.sha256(data).hexdigest() if key=='result.json' else None)
 for name,record in payload.items():
  data=base64.b64decode(record['base64'],validate=True)
  add(prefix+'/'+name,origin='ACTUAL_RUNTIME_PAYLOAD',expected_bytes=len(data),expected_sha256=record['sha256'])
 for key in ('candidate-package/actual-nvidia-smi.json','control-package/actual-nvidia-smi.json','primary-bundle-receipt.json','primary-publication-completion.json','host-series.jsonl','candidate-package/installed-package-identity.json'):
  add(output+key)
 candidate_plan=json.loads(base64.b64decode(payload['candidate-plan.json']['base64'],validate=True))
 control_plan=json.loads(base64.b64decode(payload['control-plan.json']['base64'],validate=True))
 for label,names in (('candidate',candidate_plan['candidate_frontend_source_sha']),('control',control_plan['installed_source_sha256'])):
  for name in names:add(output+label+'-package/installed-source/'+name)
 for label in ('candidate','control'):
  events=[json.loads(line) for line in decoded[label+'/events.jsonl'].splitlines()]
  sent=[row for row in events if row['kind']=='send'];done={row['rid'] for row in events if row['kind']=='response'}
  require(len(sent)<=1369 and len({row['rid'] for row in sent})==len(sent),'per-arm immutable request bound')
  for row in sent:
   require(row['rid'].startswith(cfg['run_id']+'-'+label+'-'),'cross-arm required request identity')
   for ref in (row['wire_ref'],row['transport_ref'],row['rid']+'/http.json'):add(output+label+'/'+safe_name(ref))
   add(output+label+'/'+row['rid']+'/body.bin',conditional=row['rid'] not in done)
 trace=[json.loads(line) for line in decoded['supervisor.jsonl'].splitlines()]
 for row in trace:
  if row['event']=='spawn':
   for suffix in ('.stdout','.stderr'):
    name=output+safe_name(row['phase']+suffix)
    if name not in embedded:add(name)
 for row in bundle['health_inventory']:add(output+safe_name(row['path']),expected_bytes=row['bytes'],expected_sha256=row['sha256'])
 def refs(value):
  if isinstance(value,dict):
   if {'path','bytes','sha256','complete'}<=value.keys():add(output+'query-artifacts/'+safe_name(value['path']),expected_bytes=value['bytes'],expected_sha256=value['sha256'])
   else:
    for child in value.values():refs(child)
  elif isinstance(value,list):
   for child in value:refs(child)
 for key in ('candidate-release.json','control-admission.json','pre-release.json','release.json','query-compatibility.json'):
  refs(json.loads(decoded[key]))
 lease=json.loads(decoded['native-lease.json'])
 for key in ('marker_path','launch_path'):
  add(output+str(Path(lease[key]).relative_to(Path(cfg['cwd'])/'output')))
 add(output+str(Path(lease['created']['log_path']).relative_to(Path(cfg['cwd'])/'output')))
 # Embedded files and actual fetched files have explicit separate origins.
 for name in list(rows):
  if name in embedded:rows.pop(name)
 require(output+'result.json' in rows and rows[output+'result.json']['dependency']=='INDEPENDENT_ACTUAL_RESULT','actual result must never use primary alias')
 require(len(bundle['health_inventory'])<=4504,'health inventory envelope expanded')
 require(len(json.loads(decoded['candidate-release.json'])['samples'])<=91 and len(json.loads(decoded['release.json'])['samples'])<=92,'resource sample envelope expanded')
 require(len(rows)<=18403 and len(rows)*2<=40000,'required dependency envelope expanded')
 return dict(schema='PAIR17_UPFRONT_REQUIRED_FILES_V1',rows=[rows[key] for key in sorted(rows)],embedded=embedded,
   required_names=len(rows),maximum_GETs=len(rows)*2,current_session=True,numeric_archive_or_CD_claim=False)
