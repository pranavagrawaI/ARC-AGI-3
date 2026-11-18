"""Capture the installed Kaggle CLI request without network or upload."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi

HERE=Path(__file__).resolve().parent
def main():
    captured=[]
    @contextmanager
    def client(_):
        yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(save_kernel=lambda r:captured.append(r))))
    with patch.object(KaggleApi,'build_kaggle_client',client):
        KaggleApi().kernels_push(str(HERE),timeout=10800,acc='NvidiaRtxPro6000')
    assert len(captured)==1
    request=captured[0]
    assert request.slug=='pranavagrawai/designer-player-20260907'
    assert request.session_timeout_seconds==10800 and request.machine_shape=='NvidiaRtxPro6000'
    assert request.is_private and request.enable_gpu and not request.enable_internet
    nb=json.loads(request.text)
    code='\n'.join(''.join(c['source']) for c in nb['cells'])
    assert "os.environ['ARC_SOLVER_ARM']='loop'" in code and "os.environ['ARC_SOLVER_ARM']='mindset'" in code
    assert code.count("'--launcher-timeout-s', '2700'")==2
    assert code.count("TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = '1800'")==2
    assert 'kaggle competitions submit' not in code
    metadata=json.loads((HERE/'kernel-metadata.json').read_text())
    assert metadata['competition_sources']==['arc-prize-2026-arc-agi-3']
    print(json.dumps(dict(status='PASS',network_calls=0,competition=metadata['competition_sources'],kernel=request.slug,arms=['loop','mindset'])))
if __name__=='__main__':main()
