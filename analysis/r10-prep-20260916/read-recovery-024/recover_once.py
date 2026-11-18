"""One authorized exact numeric-version read, no retry or latest fallback."""
import hashlib,json,sys,time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[1];OLD=BASE/'exact-collection-20260916';LIMIT=4*1024*1024
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
protected=json.loads((OLD/'EVIDENCE-HASHES.json').read_text());protected[str((OLD/'EVIDENCE-HASHES.json').relative_to(ROOT))]=sha(OLD/'EVIDENCE-HASHES.json')
for n,h in protected.items():assert sha(ROOT/n)==h,n
write('PRIOR-ARCHIVE-HASHES.json',protected)
old=json.loads((OLD/'receipts/155-response.json').read_text());elapsed=time.time()-old['finished_unix']
# Original receipt retained no Retry-After header. Do not invent a value.
retry_after=old.get('retry_after')
if retry_after is not None:
 assert elapsed>=float(retry_after),'Retry-After still active; no attempt'
write('PREFLIGHT.json',{'elapsed_since_original_429_seconds':elapsed,'original_retry_after':retry_after,'retry_after_limit':'Original header was not retained; no known outstanding Retry-After is available. Elapsed time is recorded, not treated as a recovered header.','original_429_sha256':sha(OLD/'receipts/155-response.json'),'authorized_read_attempt_cap':1,'launch_retries':0})
api=KaggleApi();api.authenticate();r=ApiDownloadKernelOutputRequest();r.owner_slug='pranavagrawai';r.kernel_slug='r10a-cd82-constructor-repair';r.version_number=1;r.file_path='model-runtime/solver-workspaces/raw/raw-requests/024.wire.json'
write('REQUEST.json',{'method':'DownloadKernelOutput','type':'ApiDownloadKernelOutputRequest','request':r.to_dict(),'started_unix':time.time(),'max_bytes':LIMIT,'attempt':1})
response=None;record={};raw=bytearray()
try:
 with api.build_kaggle_client() as client:
  response=client.kernels.kernels_api_client.download_kernel_output(r)
  record['http_status']=response.status_code;record['retry_after']=response.headers.get('Retry-After');response.raise_for_status()
  for chunk in response.iter_content(65536):
   if len(raw)+len(chunk)>LIMIT:raise RuntimeError('file exceeds 4 MiB bound')
   raw.extend(chunk)
 with (OUT/'024.wire.json').open('xb') as f:f.write(raw)
 record.update(status='RETRIEVED',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),response_file='024.wire.json')
except Exception as exc:
 record.update(status='FAILED',error_type=type(exc).__name__)
 resp=getattr(exc,'response',None)
 if resp is not None:
  record['http_status']=resp.status_code;record['retry_after']=resp.headers.get('Retry-After');body=resp.content[:65536]
  record['error_body_sha256']=hashlib.sha256(resp.content).hexdigest();record['error_body_bytes']=len(resp.content)
  # API error JSON is nonsecret; exclude arbitrary HTML and signed URLs.
  try:
   error=json.loads(body)
   if isinstance(error,dict) and 'http' not in body.decode().lower():write('ERROR-BODY.json',error)
  except (ValueError,UnicodeError):pass
finally:
 if response is not None:response.close()
 record['finished_unix']=time.time();write('RESPONSE.json',record)
if record['status']=='RETRIEVED':
 wire=json.loads(raw);ledger=[json.loads(s) for s in (OLD/'raw/model-runtime/solver-workspaces/raw/requests.jsonl').read_text().splitlines() if s][23];assert ledger['call']==24
 ipc=next((OLD/'raw/model-runtime/solver-workspaces/arm-A').glob('*/ipc/model_calls.jsonl'));ipc_row=[json.loads(s) for s in ipc.read_text().splitlines() if s][23]
 sys.path.insert(0,str(BASE/'r10a-package'));from local_wire import project_local
 identity=json.loads((OLD/'raw/model-runtime/solver-workspaces/raw/MODEL-IDENTITY.json').read_text());slim=ledger['request'];payload={k:v for k,v in slim.items() if k not in ['messages_sha256','messages_bytes','messages_count']};payload['messages']=ipc_row['request_messages']
 expected=project_local(payload,local_model=identity['model'],mounted_weights=identity['mounted_weights'],startup_argv=identity['startup_argv']);expected_raw=json.dumps(expected).encode();messages=json.dumps(wire['messages'],sort_keys=True).encode()
 check={'scope':'New exact numeric-version bytes, compared with old relay ledger and independently retained IPC messages using frozen wire projection','call':24,'raw_sha256':hashlib.sha256(raw).hexdigest(),'reconstructed_wire_sha256':hashlib.sha256(expected_raw).hexdigest(),'raw_bytes_equal_reconstructed_wire':bytes(raw)==expected_raw,'message_sha256':hashlib.sha256(messages).hexdigest(),'ledger_message_sha256':slim['messages_sha256'],'message_digest_matches':hashlib.sha256(messages).hexdigest()==slim['messages_sha256'],'message_bytes_match':len(messages)==slim['messages_bytes'],'message_count_matches':len(wire['messages'])==slim['messages_count'],'wire_metadata_matches_ledger':{k:v for k,v in wire.items() if k!='messages'}=={k:v for k,v in slim.items() if k not in ['messages_sha256','messages_bytes','messages_count']},'parsed_wire_matches_reconstruction':wire==expected,'raw_file_sha256_rechecked':sha(OUT/'024.wire.json')==record['sha256'],'projection_source_sha256':sha(BASE/'r10a-package/local_wire.py'),'ledger_sha256':sha(OLD/'raw/model-runtime/solver-workspaces/raw/requests.jsonl'),'ipc_sha256':sha(ipc)}
 write('VERIFICATION.json',check)
print(json.dumps(record,indent=2))
if record['status']=='RETRIEVED':print(json.dumps(check,indent=2))
drift=[n for n,h in protected.items() if sha(ROOT/n)!=h];write('PRIOR-ARCHIVE-UNCHANGED.json',{'files_checked':len(protected),'drift':drift});assert not drift
