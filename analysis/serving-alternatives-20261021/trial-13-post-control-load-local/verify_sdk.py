"""Actual installed SDK serialization, local-only authority fixture, network blocked."""
import base64,contextlib,datetime as dt,json,socket,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import packet,transport
HERE=Path(__file__).resolve().parent

def check(ok,message):
 if not ok:raise ValueError(message)

def verify_inner(raw,manifest,assignment):
 values=transport.literals(transport.cell_source(json.loads(raw)));spec=json.loads((HERE/'packet-spec.json').read_text())
 check(values['ASSIGNMENT']==assignment and values['CONFIG']==spec['config'],'inner authority/config')
 for name,record in values['PAYLOAD'].items():
  path=Path(spec['files'][name]) if name in spec['files'] else HERE/name
  check(base64.b64decode(record['base64'],validate=True)==path.read_bytes(),'source payload '+name)
 check({k:v['sha256'] for k,v in values['PAYLOAD'].items()}==manifest['files'],'payload set')
 return values

def qualify():
 import requests
 spec=json.loads((HERE/'packet-spec.json').read_text());network=[];captured=[]
 with tempfile.TemporaryDirectory() as td:
  directory=Path(td);unassigned=packet.prepare(directory/'unassigned')
  outer=(directory/'unassigned'/spec['metadata']['code_file']).read_bytes();verify_inner(transport.decode(outer),unassigned,None)
  with patch.object(Path,'mkdir',side_effect=AssertionError('unassigned extraction')),patch('subprocess.Popen',side_effect=AssertionError('unassigned process')):
   try:exec(transport.cell_source(json.loads(outer)),{})
   except RuntimeError as error:check(str(error).startswith('PREPARATION ONLY'),'wrong refusal')
   else:raise AssertionError('unassigned packet executed')
  fixture=dict(launch_authorized=True,serialized_assignment='LOCAL_ONLY_SERIALIZATION_FIXTURE_NOT_AUTHORITY',local_only=True,
   confirmed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),total_seconds=6000,platform_hard_timeout_s=6300,
   kernel_ref=spec['metadata']['id'],sources_sha256=unassigned['sources_sha256'],config_sha256=unassigned['config_sha256'])
  armed=packet.prepare(directory/'armed-local-fixture',fixture);outer=(directory/'armed-local-fixture'/spec['metadata']['code_file']).read_bytes()
  raw=transport.decode(outer);verify_inner(raw,armed,fixture)
  with patch.object(Path,'mkdir',side_effect=AssertionError('local fixture extracted')):
   try:exec(transport.cell_source(json.loads(outer)),{})
   except RuntimeError as error:check('LOCAL ONLY fixture' in str(error),'fixture refusal')
   else:raise AssertionError('fixture executed')
  class Endpoint:
   def save_kernel(self,request):captured.append(json.loads(str(request)));return SimpleNamespace()
  @contextlib.contextmanager
  def client(_self):yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=Endpoint()))
  def forbidden(*args,**kwargs):network.append('blocked');raise AssertionError('NETWORK FORBIDDEN')
  with patch.object(socket.socket,'connect',forbidden),patch.object(socket,'create_connection',forbidden),patch.object(requests.Session,'send',forbidden):
   from kaggle.api.kaggle_api_extended import KaggleApi
   with patch.object(KaggleApi,'build_kaggle_client',client):KaggleApi().kernels_push(str(directory/'armed-local-fixture'),timeout='6300')
  check(len(captured)==1 and not network,'SDK interception/network')
  request=captured[0];text=request.pop('text');check(transport.decode(text.encode())==raw,'SDK exact inner roundtrip')
  expected=dict(slug=spec['metadata']['id'],newTitle=spec['metadata']['title'],sessionTimeoutSeconds=6300,machineShape='NvidiaRtxPro6000',enableGpu=True,enableTpu=False,enableInternet=False,isPrivate=True,
   competitionDataSources=['arc-prize-2026-arc-agi-3'],datasetDataSources=spec['metadata']['dataset_sources'],modelDataSources=spec['metadata']['model_sources'],kernelDataSources=[],dockerImage=spec['metadata']['docker_image'],dockerImagePinningType=None,language='python',kernelType='notebook')
  for name,value in expected.items():check(request.get(name)==value,'SDK field '+name)
  check(len(text.encode())<1000000,'transport size')
  return dict(status='LOCAL_SERIALIZATION_ONLY_NOT_LAUNCHED',network_calls=0,network_attempts=0,intercepted_SaveKernel=1,fixture_assignment=fixture,
   inner_notebook_sha256=transport.sha(raw),inner_source_sha256=transport.sha(transport.cell_source(json.loads(raw)).encode()),outer_bytes=len(outer),sdk_text_bytes=len(text.encode()),sdk_text_sha256=transport.sha(text.encode()),
   decoded_payloads=len(armed['files']),sources_sha256=armed['sources_sha256'],config_sha256=armed['config_sha256'],request=request,local_fixture_refused_before_extraction=True,unassigned_refused_before_extraction=True)

if __name__=='__main__':
 result=qualify()
 if len(sys.argv)==2:
  with Path(sys.argv[1]).open('x') as stream:json.dump(result,stream,indent=2)
 else:print(json.dumps(result,indent=2))
