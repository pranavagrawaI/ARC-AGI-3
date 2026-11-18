import datetime,hashlib,json,signal,time,urllib.request,urllib.error,sys
from pathlib import Path
root=Path.cwd();candidate=root/'analysis/operational-theory-t1r3-20260917';out=Path(Path('/tmp/arc-t3-operator/output-path').read_text());sys.path.insert(0,str(candidate))
from identity import verify
from transport import MODEL,NoRedirect
assert verify()=='992cc66ae2e8577f69f852f00a5330f8f0fb2269155c435f4d1d35ba2931c53c'
assert '2 passed' in Path('/tmp/arc-t3-operator/preflight-tests.txt').read_text()
(out/'preflight-tests.txt').write_bytes(Path('/tmp/arc-t3-operator/preflight-tests.txt').read_bytes())
def fetch(url,name):
 started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic();raw=b'';status=None;error=None
 def timeout(*args):raise TimeoutError('metadata absolute deadline')
 prior=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,30)
 try:
  opener=urllib.request.build_opener(NoRedirect())
  try:
   response=opener.open(urllib.request.Request(url,headers={'Accept':'application/json'}),timeout=25)
  except urllib.error.HTTPError as exc:response=exc
  with response:
   status=response.status;raw=response.read(10_000_001)
   if len(raw)>10_000_000:raise ValueError('metadata response exceeds bound')
 except Exception as exc:error=type(exc).__name__
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)
 (out/(name+'.raw')).write_bytes(raw)
 receipt={'url':url,'method':'GET','started_at':started,'ended_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-clock,'http_status':status,'error_type':error,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'authenticated':False,'retries':0,'generation_calls':0}
 (out/(name+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
 return receipt,raw
receipt,raw=fetch('https://openrouter.ai/api/v1/models','models')
result={'requested_exact_model':MODEL,'generation_calls':0,'metadata_requests':1,'models_sha256':receipt['sha256']}
if receipt['http_status']!=200 or receipt['error_type']:
 result.update(status='BLOCKED',reason='fresh model metadata lookup failed',receipt=receipt)
else:
 try:
  rows=json.loads(raw)['data'];matches=[m for m in rows if m.get('id')==MODEL]
  result['exact_match_count']=len(matches)
  if len(matches)!=1:result.update(status='BLOCKED',reason='exact frozen model ID unavailable or ambiguous in fresh advertised catalog')
  else:
   result['resolved_model']=matches[0]
   endpoint_receipt,endpoint_raw=fetch('https://openrouter.ai/api/v1/models/'+MODEL+'/endpoints','endpoints')
   result['metadata_requests']+=1
   from run_pair import metadata_check
   try:
    if endpoint_receipt['http_status']!=200 or endpoint_receipt['error_type']:raise ValueError('endpoint metadata fetch failed')
    result['metadata_sha256']=metadata_check(out/'endpoints.raw');result['status']='RESOLVED'
   except Exception as exc:result.update(status='BLOCKED',reason=str(exc))
 except Exception as exc:result.update(status='BLOCKED',reason='invalid model catalog: '+type(exc).__name__)
(out/'MODEL-RESOLUTION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
