"""Offline execution of installed SDK request-construction source.
No KaggleApi initialization, authentication or real client. Socket audit forbids network.
"""
import ast,contextlib,hashlib,io,json,os,sys,types
from pathlib import Path
from typing import cast
from slugify import slugify
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

def no_network(event,args):
 if event in ('socket.connect','socket.getaddrinfo','socket.bind'):raise AssertionError('network forbidden')
sys.addaudithook(no_network)
HERE=Path(__file__).resolve().parent;PREP=HERE.parent;P=PREP/'r10a-package'
source=Path('/home/pranav/.local/share/uv/tools/kaggle/lib/python3.13/site-packages/kaggle/api/kaggle_api_extended.py')
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='KaggleApi')
names={'kernels_push','get_or_default','get_bool','parse_kernel_string','validate_kernel_string','validate_dataset_string'}
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
# Preserve executable statements exactly. Postponed annotations avoid unrelated SDK imports.
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='RequestBuilderOnly',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
ns={'os':os,'json':json,'cast':cast,'slugify':slugify,'List':list,'ApiSaveKernelRequest':ApiSaveKernelRequest}
exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
captured=[]
class StopAtRequest(Exception):pass
class NoRemoteClient:
 def __enter__(self):return types.SimpleNamespace(kernels=types.SimpleNamespace(kernels_api_client=self))
 def __exit__(self,*args):return False
 def save_kernel(self,request):
  captured.append(request)
  raise StopAtRequest('request captured; never sent')
obj=ns['RequestBuilderOnly']();obj.KERNEL_METADATA_FILE='kernel-metadata.json';obj.valid_push_language_types=['python','r','rmarkdown'];obj.valid_push_kernel_types=['script','notebook'];obj.valid_push_pinning_types=['original','latest'];obj.build_kaggle_client=lambda:NoRemoteClient()
stdout=io.StringIO()
with contextlib.redirect_stdout(stdout):
 try:obj.kernels_push(str(P),timeout='3600',acc='NvidiaRtxPro6000')
 except StopAtRequest:pass
assert len(captured)==1
request=captured[0];meta=json.loads((P/'kernel-metadata.json').read_text())
fields=['slug','new_title','language','kernel_type','is_private','enable_gpu','enable_tpu','enable_internet','dataset_data_sources','competition_data_sources','kernel_data_sources','model_data_sources','docker_image','session_timeout_seconds','machine_shape']
out={k:getattr(request,k) for k in fields}
assert out['slug']==meta['id'] and out['is_private'] is True and out['enable_internet'] is False
assert out['dataset_data_sources']==meta['dataset_sources'] and out['competition_data_sources']==meta['competition_sources']
assert out['machine_shape']=='NvidiaRtxPro6000' and out['session_timeout_seconds']==3600
original=json.loads((P/'construction.ipynb').read_text());wire=json.loads(request.text)
for cell in original['cells']:
 if 'outputs' in cell and cell['cell_type']=='code':cell['outputs']=[]
 if isinstance(cell.get('source'),list):cell['source']=''.join(cell['source'])
assert original==wire
result={'scope':'Offline installed-source request construction; real typed SDK request, replacement client prevents send','sdk_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'metadata_sha256':hashlib.sha256((P/'kernel-metadata.json').read_bytes()).hexdigest(),'requested':out,'transformed_notebook_sha256':hashlib.sha256(request.text.encode()).hexdigest(),'notebook_transform':'only standard SDK source-list joining/output clearing/JSON serialization; parsed structure matched','stdout_advisory':stdout.getvalue(),'title_slug':slugify(meta['title']),'remote_calls':0}
(HERE/'sdk-mapping.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
