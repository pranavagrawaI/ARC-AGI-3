"""Local packet/actual SDK constructor verification. save_kernel is intercepted.

Runs no notebook GPU work, runtime installation, model load, upload or quota call.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi

HERE=Path(__file__).resolve().parent


def main(packet):
    packet=Path(packet);spec=json.loads((HERE/'packet-spec.json').read_text())
    report=json.loads((HERE/'review-packet.json').read_text());config=spec['config']
    actual_sources={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in spec['files'].items()}
    for name in ('supervisor.py','http_once.py'):
        actual_sources[name]=hashlib.sha256((HERE.parent/'runner'/name).read_bytes()).hexdigest()
    if actual_sources!=report['files']:raise ValueError('embedded sources differ from current review files')
    if hashlib.sha256(json.dumps(actual_sources,sort_keys=True).encode()).hexdigest()!=report['sources_sha256']:
        raise ValueError('source digest differs')
    code=''.join(json.loads((packet/spec['metadata']['code_file']).read_text())['cells'][0]['source'])
    try:exec(compile(code,'prepared-notebook','exec'),{})
    except RuntimeError as error:
        if 'PREPARATION ONLY' not in str(error):raise
    else:raise ValueError('unassigned notebook did not stop before extraction/GPU')
    captured=[]
    class Endpoint:
        def save_kernel(self,request):
            captured.append(json.loads(str(request)));return SimpleNamespace()
    @contextlib.contextmanager
    def fake_client(_self):yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=Endpoint()))
    with patch.object(KaggleApi,'build_kaggle_client',fake_client):
        KaggleApi().kernels_push(str(packet),timeout='5400')
    if len(captured)!=1:raise ValueError('unexpected request count')
    request=captured[0]
    expected=dict(sessionTimeoutSeconds=5400,machineShape='NvidiaRtxPro6000',enableGpu=True,enableInternet=False,
                  competitionDataSources=['arc-prize-2026-arc-agi-3'],
                  datasetDataSources=spec['metadata']['dataset_sources'],modelDataSources=spec['metadata']['model_sources'])
    for name,value in expected.items():
        if request.get(name)!=value:raise ValueError('serialized launch request mismatch '+name)
    if request.get('dockerImagePinningType') is not None:raise ValueError('unexpected image pinning field')
    if request['dockerImage']!=spec['metadata']['docker_image']:raise ValueError('image changed in serialization')
    request['textSha256']=hashlib.sha256(request.pop('text').encode()).hexdigest()
    if config['purpose']!='CANDIDATE_FEASIBILITY' or config['arm_order']!=['candidate'] or config['total_seconds']!=5100:
        raise ValueError('scientific purpose/cap changed')
    cases=json.loads((HERE/'cases.json').read_text())['cases']
    audit=json.loads((HERE.parent/'prompt-audit/functional-receipt.json').read_text())
    for case in cases:
        body=(HERE/case['request_path']).read_bytes()
        proof=next(row for row in audit['cases'] if row['request_sha256']==case['request_sha256'])
        if hashlib.sha256(body).hexdigest()!=case['request_sha256']:raise ValueError('actual case payload mismatch')
        if proof['expected_prompt_tokens']['candidate']!=case['local_prompt_tokens']:raise ValueError('expected processor count mismatch')
    output=dict(status='LOCAL_PACKET_CONSTRUCTION_CHECKED_ONLY',network_calls=0,GPU_calls=0,weights_loaded=False,
                unassigned_notebook_blocked_before_execution=True,request=request,
                sources_sha256=report['sources_sha256'],config_sha256=report['config_sha256'],cases=len(cases),
                actual_runtime_fit_qualified=False,launch_authorized=False)
    (HERE/'sdk-request-capture.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='request'}))


if __name__=='__main__':main(sys.argv[1])
