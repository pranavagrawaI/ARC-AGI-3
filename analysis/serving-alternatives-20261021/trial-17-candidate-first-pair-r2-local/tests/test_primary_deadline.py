"""Real codec/identity verifier, synthetic terminal identity, zero network/GPU."""
import json,sys,tempfile,unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'collection_tools'))
import primary,primary_bundle,transport

class PrimaryDeadline(unittest.TestCase):
 def fixture(self,root):
  spec=json.loads((HERE/'packet-spec.json').read_bytes());outer=(HERE/'review-kernel'/spec['metadata']['code_file']).read_bytes()
  inner=transport.decode(outer);values=transport.literals(transport.cell_source(json.loads(inner)))
  ref='CPU_ONLY/primary-deadline';kid=1;version=1
  result=dict(pass_=False,qualification='CPU_TIMING_FIXTURE_ONLY',ratio=None)
  event=dict(seq=0,t=1,event='terminal',**result)
  bundle=dict(schema=primary_bundle.SCHEMA,capacity_or_rate_qualification=False,run_id=spec['config']['run_id'],missing=[],truncated=[],files={
   'config.json':primary_bundle.record(json.dumps(spec['config']).encode()),'assignment.json':primary_bundle.record(json.dumps(values['ASSIGNMENT']).encode()),
   'result.json':primary_bundle.record(json.dumps(result).encode()),'supervisor.jsonl':primary_bundle.record((json.dumps(event)+'\n').encode())},
   source_sha256={n:r['sha256'] for n,r in values['PAYLOAD'].items() if n.endswith('.py')})
  raw=primary_bundle.encode(bundle);rb=root/'readback';rb.mkdir()
  (root/'packet-spec.json').write_bytes((HERE/'packet-spec.json').read_bytes())
  (root/'launch-attempt.json').write_text(json.dumps(dict(inner_notebook_sha=transport.sha(inner))))
  (rb/'READBACK.json').write_text(json.dumps(dict(error=None,source_exact=True,kernel_ref=ref,kernel_id=kid,version=version,status={'status':'COMPLETE'})))
  (rb/'metadata.json').write_text(json.dumps(dict(id=kid,currentVersionNumber=version,ref=ref)));(rb/'source.ipynb').write_bytes(outer)
  return ref,kid,version,raw,rb
 def exercise(self,mode):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);ref,kid,version,raw,rb=self.fixture(root);clock=NS(now=0.);gets=[];verified=[]
   actual_verify=primary.verify;actual_write=Path.write_text
   class Stream:
    status_code=200;headers={'content-length':str(len(raw))}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):pass
    def iter_content(self,chunk_size):yield raw
   def listing(request):
    if mode=='listing_expired':clock.now=121.
    return NS(files=[NS(file_name=primary_bundle.NAME,url='https://cpu.invalid/never-network')])
   class Client:
    def __init__(self):self.kernels=NS(kernels_api_client=NS(list_kernel_session_output=listing))
    def __enter__(self):return self
    def __exit__(self,*args):pass
   api=NS(build_kaggle_client=Client)
   def get(*args,**kwargs):gets.append(clock.now);return Stream()
   def verify(*args):
    value=actual_verify(*args);verified.append(value)
    if mode=='verification_completion':clock.now=121.
    return value
   def write(path,*args,**kwargs):
    result=actual_write(path,*args,**kwargs)
    if path.name=='PRIMARY-RECEIPT.json':
     if mode=='receipt_persistence':clock.now=121.
     if mode=='receipt_error':raise OSError('CPU_RECEIPT_PERSISTENCE_FAILURE')
    return result
   with ExitStack() as stack:
    for obj,name,value in [(primary,'HERE',root),(primary,'accepted',lambda:(ref,kid,version)),(primary.space,'check',lambda p:dict(eligible=True)),
     (primary.time,'monotonic',lambda:clock.now),(primary,'verify',verify),(Path,'write_text',write)]:stack.enter_context(patch.object(obj,name,value))
    if mode=='timely':
     returned=primary.fetch(api,get,root/'download',rb)
     self.assertTrue(returned['complete']);self.assertTrue(returned['current_exact'])
     saved=json.loads((root/'download/PRIMARY-RECEIPT.json').read_bytes());self.assertTrue(saved['complete'])
    else:
     expected=OSError if mode=='receipt_error' else TimeoutError
     with self.assertRaises(expected):primary.fetch(api,get,root/'download',rb)
     path=root/'download/PRIMARY-RECEIPT.json'
     if path.exists():self.assertFalse(json.loads(path.read_bytes())['complete'])
     failed=json.loads(path.read_bytes())
     self.assertIn('failed_at',failed);self.assertNotIn('cpu.invalid',json.dumps(failed))
     if mode in ('receipt_error','receipt_persistence'):
      self.assertTrue((root/'download/PRIMARY-RECEIPT.unaccepted.json').exists())
   if mode=='listing_expired':self.assertEqual(gets,[])
   else:self.assertEqual(len(gets),1);self.assertEqual(len(verified),1)
 def test_verification_completion_witness(self):self.exercise('verification_completion')
 def test_receipt_persistence_witness(self):self.exercise('receipt_persistence')
 def test_expired_listing_cannot_start_get(self):self.exercise('listing_expired')
 def test_observed_receipt_write_failure_cannot_leave_accepted_completion(self):self.exercise('receipt_error')
 def test_timely_real_codec_and_identity_path(self):self.exercise('timely')
