"""Read-only exact-version known-path collection. No lists, bundles, fallback or retry."""
import hashlib,json,time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
OUT=Path(__file__).resolve().parent
plan=json.loads((OUT/'ALLOWLIST.json').read_text());assert plan['target']=='pranavagrawai/r10a-cd82-constructor-repair' and type(plan['version_number']) is int and plan['version_number']==1
receipts=OUT/'receipts';receipts.mkdir(exist_ok=False);rawdir=OUT/'raw';rawdir.mkdir(exist_ok=False)
api=KaggleApi();api.authenticate();total=0;start=time.monotonic();results=[]
with api.build_kaggle_client() as client:
 for i,spec in enumerate(plan['files'],1):
  if i>plan['max_requests'] or total>=plan['max_total_bytes'] or time.monotonic()-start>900:break
  name=spec['path'];r=ApiDownloadKernelOutputRequest();r.owner_slug='pranavagrawai';r.kernel_slug='r10a-cd82-constructor-repair';r.version_number=1;r.file_path=name
  request={'request_type':'ApiDownloadKernelOutputRequest','method':'DownloadKernelOutput','fields':{'owner_slug':r.owner_slug,'kernel_slug':r.kernel_slug,'version_number':1,'file_path':name},'wire_fields':r.to_dict(),'started_unix':time.time(),'max_bytes':spec['max_bytes'],'basis':spec['basis']}
  with (receipts/f'{i:03d}-request.json').open('x') as f:json.dump(request,f,indent=2)
  row={'index':i,'file_path':name};response=None;dest=rawdir/name;count=0;digest=hashlib.sha256()
  try:
   response=client.kernels.kernels_api_client.download_kernel_output(r);row['http_status']=response.status_code;response.raise_for_status()
   dest.parent.mkdir(parents=True,exist_ok=True)
   with dest.open('xb') as f:
    for chunk in response.iter_content(65536):
     if count+len(chunk)>spec['max_bytes'] or total+len(chunk)>plan['max_total_bytes']:raise RuntimeError('BOUNDED_SIZE_STOP_PARTIAL_PRESERVED')
     f.write(chunk);digest.update(chunk);count+=len(chunk);total+=len(chunk)
   row.update(status='RETRIEVED',bytes=count,sha256=digest.hexdigest(),response_file=str(dest.relative_to(OUT)))
  except Exception as exc:
   row.update(status='FAILED',error_type=type(exc).__name__)
   resp=getattr(exc,'response',None)
   if resp is not None:
    row['http_status']=resp.status_code;row['error_body_bytes']=len(resp.content);row['error_body_sha256']=hashlib.sha256(resp.content).hexdigest()
    # Error status/hash retained; authentication headers and redirect URLs never logged.
   if dest.exists():row.update(partial_file=str(dest.relative_to(OUT)),partial_bytes=count,partial_sha256=digest.hexdigest())
  finally:
   if response is not None:response.close()
   row['finished_unix']=time.time()
   with (receipts/f'{i:03d}-response.json').open('x') as f:json.dump(row,f,indent=2)
  results.append(row)
  if row['status']!='RETRIEVED' or i%25==0:print(json.dumps(row),flush=True)
  if row.get('http_status')==429 or row.get('partial_file'):break
with (OUT/'COLLECTION.json').open('x') as f:json.dump({'target':plan['target'],'version_number':1,'files_planned':len(plan['files']),'requests':len(results),'bytes':total,'results':results,'elapsed_seconds':time.monotonic()-start},f,indent=2)
print(json.dumps({'requests':len(results),'retrieved':sum(r['status']=='RETRIEVED' for r in results),'failed':sum(r['status']=='FAILED' for r in results),'bytes':total}),flush=True)
