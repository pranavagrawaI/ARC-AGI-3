"""Actual new outer bytes and genuine SDK quota parse; synthetic current metadata only."""
import copy,datetime as dt,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE/'collection_tools'))
import packet,transport,remote,decimal_quota
from quota_http_fixture import HttpApi
RAW=json.loads(Path(__file__).with_name('quota-captured-body.json').read_bytes())
class Current(unittest.TestCase):
 def test_current_identity_source_status_and_decimal_reserved_fail_closed(self):
  spec=json.loads((HERE/'packet-spec.json').read_bytes());ref=spec['metadata']['id'];kid=999999
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);manifest=packet.prepare(root/'kernel');rb=root/'readback';rb.mkdir()
   outer=(root/'kernel'/spec['metadata']['code_file']).read_bytes();(rb/'source.ipynb').write_bytes(outer)
   (root/'packet-spec.json').write_text(json.dumps(spec));(root/'launch-attempt.json').write_text(json.dumps(dict(inner_notebook_sha=manifest['inner_notebook_sha256'])))
   meta=dict(ref=ref,id=kid,currentVersionNumber=1);(rb/'metadata.json').write_text(json.dumps(meta))
   wire=copy.deepcopy(RAW);wire['gpuQuota']['timeUsed']='2814.079s';quota=decimal_quota.read(HttpApi([wire]))
   receipt=dict(kernel_ref=ref,kernel_id=kid,version=1,status={'status':'COMPLETE'},source_exact=True,inner_sha256=manifest['inner_notebook_sha256'],typed_quota=quota)
   def validate(row):
    (rb/'READBACK.json').write_text(json.dumps(row))
    with patch.object(remote,'ROOT',root),patch.object(remote,'accepted',return_value=(ref,kid,1)):return remote.validate_readback(rb)
   self.assertEqual(validate(receipt)['kernel_id'],kid)
   for key,value in [('version',2),('status',{'status':'RUNNING'}),('source_exact',False),('inner_sha256','0'*64)]:
    row=copy.deepcopy(receipt);row[key]=value
    with self.subTest(key=key),self.assertRaises(ValueError):validate(row)
   wire['gpuQuota']['timeReserved']='0.000000001s';row=copy.deepcopy(receipt);row['typed_quota']=decimal_quota.read(HttpApi([wire]))
   with self.assertRaises(ValueError):validate(row)
 def test_installed_current_list_request_omits_version_label(self):
  from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
  captured=[]
  class Client:
   def __enter__(self):return self
   def __exit__(self,*args):pass
   def list_kernel_session_output(self,request):captured.append(ApiListKernelSessionOutputRequest.to_dict(request));return types.SimpleNamespace(files=[],next_page_token='')
   @property
   def kernels(self):return types.SimpleNamespace(kernels_api_client=self)
  client=Client();api=remote.Remote();api.api=types.SimpleNamespace(build_kaggle_client=lambda:client)
  ref='pranavagrawai/serving-candidate-first-capacity17-20261024'
  with patch.object(remote,'accepted',return_value=(ref,999999,1)):
   self.assertEqual(api.list_page('',100),([],''));self.assertEqual(api.list_page('next',100),([],''))
  self.assertEqual(captured[0],dict(userName='pranavagrawai',kernelSlug=ref.split('/')[1],pageSize=100));self.assertNotIn('versionLabel',captured[1]);self.assertEqual(captured[1]['pageToken'],'next')
