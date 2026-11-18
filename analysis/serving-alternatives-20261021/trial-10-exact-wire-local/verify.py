"""Read-only local qualification; actual SDK serialization is intercepted, sockets blocked."""
import contextlib
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import packet
import transport
import launch
import audit_bindings

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'trial-6-serving-feasibility-transport'


def check(condition,message):
    if not condition:raise ValueError(message)


def verify_inner(raw,manifest,expected_assignment):
    values=transport.literals(transport.cell_source(json.loads(raw)))
    spec=json.loads((HERE/'packet-spec.json').read_text())
    check(values['ASSIGNMENT']==expected_assignment,'inner authority mismatch')
    check(values['CONFIG']==spec['config'],'decoded config mismatch')
    actual={}
    import base64
    for name,record in values['PAYLOAD'].items():
        data=base64.b64decode(record['base64'],validate=True)
        expected_path=Path(spec['files'][name]) if name in spec['files'] else packet.RUNNER/name
        check(data==expected_path.read_bytes(),'decoded source bytes mismatch '+name)
        check(transport.sha(data)==record['sha256'],'decoded source hash mismatch '+name)
        actual[name]=record['sha256']
    check(actual==manifest['files'],'decoded source file set mismatch')
    check(transport.sha(json.dumps(actual,sort_keys=True).encode())==manifest['sources_sha256'],'aggregate source mismatch')
    check(transport.sha(json.dumps(values['CONFIG'],sort_keys=True).encode())==manifest['config_sha256'],'config digest mismatch')
    return values


def verify_unassigned_refusal(outer):
    source=transport.cell_source(json.loads(outer))
    harness='''import pathlib,subprocess,sys
from unittest.mock import patch
with patch.object(pathlib.Path,'mkdir',side_effect=AssertionError('extraction attempted')), patch.object(pathlib.Path,'open',side_effect=AssertionError('extraction attempted')), patch.object(pathlib.Path,'write_text',side_effect=AssertionError('extraction attempted')), patch.object(subprocess,'Popen',side_effect=AssertionError('process attempted')):
    try:exec(compile(sys.stdin.read(),'<successor-wrapper>','exec'))
    except RuntimeError as e:
        if str(e)!='PREPARATION ONLY: no reviewed serialized assignment; do not execute GPU work':raise
        print(str(e))
    else:raise AssertionError('unassigned packet did not refuse')
'''
    with tempfile.TemporaryDirectory() as directory:
        result=subprocess.run([sys.executable,'-c',harness],input=source,text=True,capture_output=True,cwd=directory,timeout=10)
    check(result.returncode==0,'unassigned pre-extraction refusal failed: '+result.stderr)
    return result.stdout.strip()


def serialize_fixture(directory,manifest):
    import requests
    fixture=dict(launch_authorized=True,serialized_assignment='LOCAL_ONLY_SERIALIZATION_FIXTURE_NOT_LAUNCH_AUTHORITY',
                 total_seconds=9900,sources_sha256=manifest['sources_sha256'],config_sha256=manifest['config_sha256'],local_only=True,kernel_ref=launch.KERNEL_REF,platform_hard_timeout_s=10200,confirmed_utc='2026-10-23T00:00:00+00:00')
    armed=packet.prepare(Path(directory)/'armed-fixture',fixture)
    spec=json.loads((HERE/'packet-spec.json').read_text())
    outer=(Path(directory)/'armed-fixture'/spec['metadata']['code_file']).read_bytes()
    raw=transport.decode(outer);verify_inner(raw,armed,fixture)
    captured=[];network_attempts=[]
    class Endpoint:
        def save_kernel(self,request):
            captured.append(json.loads(str(request)));return SimpleNamespace()
    @contextlib.contextmanager
    def client(_self):yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=Endpoint()))
    def forbidden(*args,**kwargs):
        network_attempts.append('blocked');raise AssertionError('NETWORK FORBIDDEN')
    with patch.object(socket.socket,'connect',forbidden),patch.object(socket,'create_connection',forbidden),patch.object(requests.Session,'send',forbidden):
        from kaggle.api.kaggle_api_extended import KaggleApi
        with patch.object(KaggleApi,'build_kaggle_client',client):
            KaggleApi().kernels_push(str(Path(directory)/'armed-fixture'),timeout='10200')
    check(len(captured)==1 and not network_attempts,'SDK interception/network check failed')
    request=captured[0];text=request.pop('text')
    check(transport.decode(text.encode())==raw,'SDK wire decode differs from armed inner')
    check(transport.cell_source(json.loads(text))==transport.cell_source(json.loads(outer)),'SDK changed wrapper source')
    expected=dict(slug=launch.KERNEL_REF,newTitle=spec['metadata']['title'],sessionTimeoutSeconds=10200,machineShape='NvidiaRtxPro6000',enableGpu=True,enableTpu=False,
                  enableInternet=False,isPrivate=True,competitionDataSources=['arc-prize-2026-arc-agi-3'],
                  datasetDataSources=spec['metadata']['dataset_sources'],modelDataSources=spec['metadata']['model_sources'],
                  kernelDataSources=[],dockerImage=spec['metadata']['docker_image'],dockerImagePinningType=None,language='python',kernelType='notebook')
    for key,value in expected.items():check(request.get(key)==value,'SDK field mismatch '+key)
    check(len(text.encode())<1_000_000,'SDK text exceeds size target')
    return dict(assignment=fixture,fixture_only=True,executable_fixture_retained=False,network_calls=0,network_attempts=0,save_kernel_calls_intercepted=1,
                outer_notebook_bytes=len(outer),outer_notebook_sha256=transport.sha(outer),
                sdk_text_bytes=len(text.encode()),sdk_text_sha256=transport.sha(text.encode()),
                inner_notebook_sha256=transport.sha(raw),inner_source_sha256=transport.sha(transport.cell_source(json.loads(raw)).encode()),
                decoded_files=len(armed['files']),request=request)


def qualify():
    bindings=audit_bindings.verify()
    report=json.loads((HERE/'review-packet.json').read_text())
    spec=json.loads((HERE/'packet-spec.json').read_text())
    with tempfile.TemporaryDirectory() as directory:
        rebuilt=packet.prepare(Path(directory)/'unassigned')
        check(rebuilt==report,'fresh packet differs from review')
        outer=(HERE/'review-packet'/spec['metadata']['code_file']).read_bytes()
        verify_inner(transport.decode(outer),report,None)
        refusal=verify_unassigned_refusal(outer)
        fixture=serialize_fixture(directory,report)
    preserved={}
    for name in ('trial-9-readiness-local/freeze.json','trial-8-wire-capacity-local/freeze-r4.json','trial-7-serving-feasibility-guard/freeze.json','trial-6-serving-feasibility-transport/freeze.json','trial-5-serving-screen/candidate-feasibility/freeze-terminal-r3.json'):
        freeze=HERE.parent/name;value=json.loads(freeze.read_text())
        for file,expected in value['files'].items():check(transport.sha((freeze.parent/file).read_bytes())==expected,'prior binding changed '+file)
        preserved[name]=transport.sha(freeze.read_bytes())
    check(not (HERE/'slot-confirmation.json').exists() and not (HERE/'launch-attempt.json').exists(),'unexpected real authority/attempt')
    return dict(audit_dependencies=bindings,status='LOCAL_PACKET_QUALIFIED_NO_GPU_AUTHORITY',kernel_ref=launch.KERNEL_REF,
                unassigned_pre_extraction_refusal=refusal,sources_sha256=report['sources_sha256'],config_sha256=report['config_sha256'],
                unassigned_outer_bytes=len(outer),fixture_serialization=fixture,preserved=preserved,remote_gpu_or_model_work=False,
                estimand='WIRE_EQUIVALENT_TOTAL_PACKAGE_API_CAPACITY',strict_trial4='PRESERVED_BLOCKED',
                reconciliation='NEW scoped proposal; final coordinator policy review and assignment required; no predicate override')

if __name__=='__main__':
    result=qualify()
    if len(sys.argv)==3 and sys.argv[1]=='--receipt':Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
    else:print(json.dumps(result,indent=2))
