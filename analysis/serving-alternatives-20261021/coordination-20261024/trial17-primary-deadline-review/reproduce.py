"""Offline primary retrieval timing witness. Real codec+primary.verify; synthetic terminal identity only."""
import base64,contextlib,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
BASE=Path('/home/pranav/projects/arc-agi-3/analysis/serving-alternatives-20261021')
P=BASE/'trial-17-candidate-first-pair-local'; OUT=Path(__file__).parent
sys.path[:0]=[str(P/'collection_tools'),str(P)]
import primary,primary_bundle,transport
spec=json.loads((P/'packet-spec.json').read_bytes()); outer=(P/'review-kernel'/spec['metadata']['code_file']).read_bytes(); inner=transport.decode(outer)
values=transport.literals(transport.cell_source(json.loads(inner)))
ref='CPU_ONLY/primary-deadline'; kid=1;version=1
result={'pass_':False,'qualification':'CPU_TIMING_FIXTURE_ONLY','ratio':None}
event=dict(seq=0,t=1,event='terminal',**result)
bundle=dict(schema=primary_bundle.SCHEMA,capacity_or_rate_qualification=False,run_id=spec['config']['run_id'],missing=[],truncated=[],files={
 'config.json':primary_bundle.record(json.dumps(spec['config']).encode()),'assignment.json':primary_bundle.record(json.dumps(values['ASSIGNMENT']).encode()),
 'result.json':primary_bundle.record(json.dumps(result).encode()),'supervisor.jsonl':primary_bundle.record((json.dumps(event)+'\n').encode())},
 source_sha256={n:r['sha256'] for n,r in values['PAYLOAD'].items() if n.endswith('.py')})
raw=primary_bundle.encode(bundle); original_verify=primary.verify; original_write=Path.write_text
proof=[]
for mode in ('verification_completion','receipt_persistence'):
 root=OUT/mode;root.mkdir(exist_ok=False);rb=root/'readback';rb.mkdir()
 (root/'packet-spec.json').write_bytes((P/'packet-spec.json').read_bytes());(root/'launch-attempt.json').write_text(json.dumps({'inner_notebook_sha':transport.sha(inner)}))
 (rb/'READBACK.json').write_text(json.dumps(dict(error=None,source_exact=True,kernel_ref=ref,kernel_id=kid,version=version,status={'status':'COMPLETE'})))
 (rb/'metadata.json').write_text(json.dumps(dict(id=kid,currentVersionNumber=version,ref=ref)));(rb/'source.ipynb').write_bytes(outer)
 clock=NS(now=0.); actual_verification=[]
 class Stream:
  status_code=200;headers={'content-length':str(len(raw))}
  def __enter__(self):return self
  def __exit__(self,*a):pass
  def raise_for_status(self):pass
  def iter_content(self,chunk_size):yield raw
 class Client:
  def __enter__(self):return self
  def __exit__(self,*a):pass
  def __init__(self):self.kernels=NS(kernels_api_client=NS(list_kernel_session_output=lambda r:NS(files=[NS(file_name=primary_bundle.NAME,url='https://cpu.invalid/never-network')])) )
 class Api:
  def build_kaggle_client(self):return Client()
 def verified(*args):
  v=original_verify(*args);actual_verification.append(v)
  if mode=='verification_completion':clock.now=121.
  return v
 def delayed_write(path,*args,**kwargs):
  v=original_write(path,*args,**kwargs)
  if mode=='receipt_persistence' and path.name=='PRIMARY-RECEIPT.json':clock.now=121.
  return v
 with patch.object(primary,'HERE',root),patch.object(primary,'accepted',return_value=(ref,kid,version)),patch.object(primary.space,'check',return_value={'eligible':True}),patch.object(primary.time,'monotonic',side_effect=lambda:clock.now),patch.object(primary,'verify',side_effect=verified),patch.object(Path,'write_text',delayed_write):
  returned=primary.fetch(Api(),lambda *a,**k:Stream(),root/'download',rb)
 saved=json.loads((root/'download/PRIMARY-RECEIPT.json').read_bytes())
 proof.append(dict(mode=mode,elapsed=clock.now,cap=120,returned_complete=returned['complete'],saved_complete=saved['complete'],actual_verify_executed=len(actual_verification),scientific_pass_claim=False))
print(json.dumps(proof,indent=2));(OUT/'RESULT.json').write_text(json.dumps(proof,indent=2)+'\n')
