"""Read-only official SDK get_kernel; retain only source and contract metadata."""
import datetime,hashlib,json,re
from pathlib import Path
from kaggle import api
from kaggle.api.kaggle_api_extended import ApiGetKernelRequest
HERE=Path(__file__).resolve().parent
launch=json.loads((HERE/'launch.json').read_text())
m=re.search(r'Kernel version (\d+) successfully pushed\.  Please check progress at https://www.kaggle.com/code/(\S+)',launch['stdout']);assert m and launch['returncode']==0
version=int(m[1]);kernel=m[2];owner,slug=kernel.split('/')
request=ApiGetKernelRequest();request.user_name=owner;request.kernel_slug=slug
with api.build_kaggle_client() as client:r=client.kernels.kernels_api_client.get_kernel(request)
meta=r.metadata
server={k:getattr(meta,k) for k in ('ref','current_version_number','is_private','enable_gpu','enable_tpu','enable_internet','machine_shape','docker_image')}
server.update(id=meta.ref,dataset_sources=list(meta.dataset_data_sources),kernel_sources=list(meta.kernel_data_sources),competition_sources=list(meta.competition_data_sources),model_sources=list(meta.model_data_sources))
(HERE/'server-notebook.ipynb').write_text(r.blob.source)
(HERE/'SERVER.json').write_text(json.dumps(server,indent=2)+'\n')
receipt={'method':'official SDK get_kernel','request':{'user_name':owner,'kernel_slug':slug},'returncode':0,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returned_ref':meta.ref,'returned_version':meta.current_version_number,'source_sha256':hashlib.sha256(r.blob.source.encode()).hexdigest(),'limitation':'unversioned metadata read reconciled using returned current_version_number and launch version; not assumed version'}
(HERE/'SERVER-RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert meta.ref==kernel and meta.current_version_number==version
assert server['competition_sources']==['arc-prize-2026-arc-agi-3'], 'INVALID_SERVER_COMPETITION_ASSOCIATION'
assert all(server[k]==[] for k in ('dataset_sources','kernel_sources','model_sources')), 'INVALID_SERVER_UNINTENDED_SOURCES'
print(json.dumps({'receipt':receipt,'server':server},indent=2))
