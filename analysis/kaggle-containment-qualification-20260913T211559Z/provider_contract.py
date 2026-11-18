"""Offline exercise of installed official Kaggle request construction.
Uses the existing CLI environment; replaces ONLY the outbound service client.
No provider mutation, auth value logging, network submission or credentials.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from kaggle.api.kaggle_api_extended import KaggleApi
import importlib
implementation = importlib.import_module('kaggle.api.kaggle_api_extended')

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]/'deployment/kaggle'/HERE.name
captured=[]

def save_kernel(request):
    fields=('slug','is_private','enable_gpu','enable_tpu','enable_internet',
            'machine_shape','docker_image','docker_image_pinning_type','session_timeout_seconds',
            'dataset_data_sources','competition_data_sources','kernel_data_sources','model_data_sources')
    captured.append({k:getattr(request,k) for k in fields})
    return SimpleNamespace(offline=True)

@contextmanager
def fake_client():
    yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(save_kernel=save_kernel)))

api=KaggleApi()
api.build_kaggle_client=fake_client
api.kernels_push(str(PACKAGE),timeout=300,acc='NvidiaRtxPro6000')
assert len(captured)==1
request=captured[0]
assert request['session_timeout_seconds']==300
assert request['machine_shape']=='NvidiaRtxPro6000'
assert request['is_private'] is True and request['enable_internet'] is False
assert request['enable_gpu'] is True and request['enable_tpu'] is False
assert request['docker_image']==json.loads((PACKAGE/'kernel-metadata.json').read_text())['docker_image']
assert request['competition_data_sources']==['arc-prize-2026-arc-agi-3']
assert all(not request[k] for k in ('dataset_data_sources','kernel_data_sources','model_data_sources'))
source=Path(implementation.__file__)
result={'network_calls':0,'official_cli_request_construction_passed':True,'request':request,
        'installed_implementation':str(source),'implementation_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'limitation':'Outbound request construction only; not a test of provider timeout enforcement or billing guarantees'}
(HERE/'PROVIDER-CONTRACT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
