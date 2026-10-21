"""Exercise installed Kaggle SDK request construction with the network replaced.

Captures real ApiSaveKernelRequest serialization. Never authenticates, sends or starts a kernel.
"""
import contextlib
import hashlib
import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi

HERE = Path(__file__).resolve().parent


def capture(meta, acc=None):
    requests = []
    class Endpoint:
        def save_kernel(self, request):
            requests.append(json.loads(str(request)))
            return SimpleNamespace()
    @contextlib.contextmanager
    def client(_self):
        yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=Endpoint()))
    with tempfile.TemporaryDirectory(prefix='kaggle-request-proof-') as tmp:
        root=Path(tmp)
        (root/'kernel-metadata.json').write_text(json.dumps(meta))
        source=HERE/'kernel'/meta['code_file']
        (root/meta['code_file']).write_bytes(source.read_bytes())
        with patch.object(KaggleApi,'build_kaggle_client',client):
            KaggleApi().kernels_push(str(root),timeout='5400',acc=acc)
    assert len(requests)==1
    wire=requests[0]
    # Avoid retaining the large notebook again; its exact wire identity remains available.
    wire['textSha256']=hashlib.sha256(wire.pop('text').encode()).hexdigest()
    return wire


def main():
    meta=json.loads((HERE/'kernel/kernel-metadata.json').read_text())
    cases={
        'trial1_explicit_acc':capture(meta,'NvidiaRtxPro6000'),
        'trial1_metadata_only':capture(meta),
        'generic_gpu_default':capture({k:v for k,v in meta.items() if k!='machine_shape'}),
    }
    assert cases['trial1_explicit_acc']['machineShape']=='NvidiaRtxPro6000'
    assert cases['trial1_metadata_only']['machineShape']=='NvidiaRtxPro6000'
    assert not cases['generic_gpu_default'].get('machineShape')
    assert cases['trial1_explicit_acc']['enableGpu'] is True
    assert cases['trial1_explicit_acc']['sessionTimeoutSeconds']==5400
    assert cases['trial1_explicit_acc']==cases['trial1_metadata_only']
    src=inspect.getsource(KaggleApi.kernels_push)
    (HERE/'request-construction-source.py.txt').write_text(src)
    report={'network_calls':0,'passes':['explicit-acc-preserved','metadata-acc-preserved','explicit-and-metadata-wire-identical','default-negative-control','timeout-preserved'],
            'sdk_source_sha256':hashlib.sha256(src.encode()).hexdigest(),'requests':cases,
            'conclusion':'Trial-1 requested NvidiaRtxPro6000 on the SDK wire. Lost SDK default is excluded. Backend allocation of 2x T4 is observed; a corrected allocation has not been established.'}
    (HERE/'request-capture.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='requests'}))


if __name__=='__main__': main()
