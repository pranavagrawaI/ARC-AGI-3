"""Local behavioral checks and actual SDK request serialization, without network."""
import ast
import contextlib
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi

HERE=Path(__file__).resolve().parent


def main():
    ns={'csv':__import__('csv')}
    tree=ast.parse((HERE/'probe.py').read_text())
    classify=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='classify')
    exec(compile(ast.Module(body=[classify],type_ignores=[]),'classification','exec'),ns)
    check=ns['classify']
    assert check({'returncode':0,'stdout':'NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 580.178.04\n'})
    assert not check({'returncode':0,'stdout':'Tesla T4, 15360, 580.178.04\nTesla T4, 15360, 580.178.04\n'})
    assert not check({'returncode':0,'stdout':'NVIDIA RTX PRO 6000, 16000, 580\n'})
    assert not check({'returncode':1,'stdout':'NVIDIA RTX PRO 6000, 97887, 580\n'})
    assert not check({'returncode':0,'stdout':''})
    assert not check({'returncode':0,'stdout':'malformed'})
    captured=[]
    class Endpoint:
        def save_kernel(self,request):
            captured.append(json.loads(str(request)))
            return SimpleNamespace()
    @contextlib.contextmanager
    def client(_self):
        yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=Endpoint()))
    with patch.object(KaggleApi,'build_kaggle_client',client):
        KaggleApi().kernels_push(str(HERE/'kernel'),timeout='180')
    assert len(captured)==1
    req=captured[0]
    assert req['sessionTimeoutSeconds']==180 and req['machineShape']=='NvidiaRtxPro6000'
    assert req['enableGpu'] is True and req['enableInternet'] is False
    assert req['competitionDataSources']==['arc-prize-2026-arc-agi-3']
    assert req['datasetDataSources']==req['kernelDataSources']==req['modelDataSources']==[]
    assert req['dockerImage'].endswith('57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c')
    assert req.get('dockerImagePinningType') is None
    source=req.pop('text');req['textSha256']=hashlib.sha256(source.encode()).hexdigest()
    # The probe has only stdlib imports and nvidia-smi subprocesses; model/runtime imports are absent.
    imports={n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)}
    assert imports <= {'csv','hashlib','json','pathlib','subprocess','sys','time'}
    freeze=json.loads((HERE/'freeze.json').read_text())
    assert all(hashlib.sha256((HERE/k).read_bytes()).hexdigest()==v for k,v in freeze['files'].items())
    report={'network_calls':0,'checks_passed':15,'actual_allocation_proven':False,'request':req,
            'classification_tests':['RTX-pass','2xT4-reject','small-VRAM-reject','failed-query-reject','empty-reject','malformed-reject'],
            'note':'This proves local request construction and fail-closed classification, not backend allocation.'}
    (HERE/'request-capture.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='request'}))


if __name__=='__main__':main()
