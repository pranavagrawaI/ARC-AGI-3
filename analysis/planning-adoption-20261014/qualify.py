"""Exercise the exact materialized package and HTTP boundary before launch."""
import argparse
import hashlib
import http.server
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ENV=ROOT.parent/'arc-agi-3-reference/competition-env/environment_files'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def package_env(package):
    return {**os.environ,'PLANNING_ENV_DIR':str(ENV),'PLANNING_ENGINE_SOURCE':str(ENV/'s5i5/18d95033/s5i5.py'),
            'PLANNING_CODE_B1':str(package/'sources/b1'),'PLANNING_CODE_CONTROL':str(package/'sources/control'),
            'PLANNING_TAAF_DIR':str(package/'sources/taaf'),'PYTHONDONTWRITEBYTECODE':'1'}


def process_case(case,variant,code=None,checkpoint='b1'):
    package=HERE/'package';out=HERE/'work'/case;log=HERE/'evidence'/(case+'.log')
    command=[sys.executable,'-B',str(package/'analysis/planning-transfer-20261014/continuation.py'),
             '--checkpoint',checkpoint,'--variant',variant,'--mode','fixture' if code is not None else 'restore','--output',str(out)]
    if code is not None:
        fixture=HERE/'work'/(case+'.py');fixture.write_text(code);command+=['--fixture',str(fixture)]
    with log.open('x') as output:
        subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=package_env(package),check=True,timeout=120)
    return json.loads((out/'result.json').read_text())


def http_case(case,error=False):
    requests=[]
    receipt=json.loads((ROOT/'analysis/score-conversion-20261013/native-receipts.json').read_text())
    route=next(row for row in receipt if row['run']=='b1-r1')['actions']
    clicks={'Og':12,'Os':6,'Sg':27,'Ss':21,'Yg':42,'Ys':36,'Ng':57,'Ns':51}
    actions=[{'action':'MOUSE','row':57,'col':clicks[row['token']]} for row in route]
    route_code='action('+repr(actions)+')'
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request)
            for message in request['messages']:
                assert '_arc3_control' not in message
                for tool_call in message.get('tool_calls') or []:
                    assert 'index' not in tool_call
            if error:
                body={'error':{'message':'qualification HTTP400'}};status=400
            else:
                code="print('FIRST_WIRE_OK')" if len(requests)==1 else route_code
                body={'id':'fixture','object':'chat.completion','choices':[{'index':0,'finish_reason':'tool_calls',
                    'message':{'role':'assistant','content':None,'reasoning_content':'fixture reasoning',
                               'tool_calls':[{'index':0,'id':'fixture-'+str(len(requests)),'type':'function',
                                  'function':{'name':'python','arguments':json.dumps({'code':code})}}]}}],
                      'usage':{'completion_tokens':1,'prompt_tokens':1,'total_tokens':2}}
                status=200
            encoded=json.dumps(body).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(encoded)))
            self.end_headers();self.wfile.write(encoded)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    out=HERE/'work'/case;log=HERE/'evidence'/(case+'.log')
    try:
        command=[sys.executable,'-B',str(HERE/'package/analysis/planning-transfer-20261014/continuation.py'),
                 '--checkpoint','b1','--variant','both','--mode','live','--output',str(out),
                 '--url',f'http://127.0.0.1:{server.server_port}/v1']
        with log.open('x') as output:
            subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=package_env(HERE/'package'),check=True,timeout=120)
    finally:
        server.shutdown();server.server_close()
    result=json.loads((out/'result.json').read_text())
    first=json.loads((out/'first-request.json').read_text())
    if not requests or requests[0]['messages']!=first['messages'] or requests[0]['tools']!=first['tools']:
        raise RuntimeError('First actual wire differs from declared intervention')
    if error:
        assert result['status']=='http_error' and result['new_model_calls']==1 and result['unresolved']
    else:
        assert result['status']=='solved' and result['new_model_calls']==2 and result['tokens']==2
        assert not result['unresolved'] and result['levels_completed']==2
    return {'case':case,'status':result['status'],'calls':result['new_model_calls'],
            'native_actions':result['new_native_actions'],'wire_messages_exact':True,
            'known_tokens':result['tokens'],'unresolved':result['unresolved']}


def trickle_cases():
    sys.path.insert(0,str(HERE/'package/analysis/planning-transfer-20261014'))
    continuation=load('transport_qualification',HERE/'package/analysis/planning-transfer-20261014/continuation.py')
    results=[]
    for status in [200,400]:
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(status);self.send_header('Content-Length','200');self.end_headers()
                try:
                    for _ in range(200):self.wfile.write(b' ');self.wfile.flush();time.sleep(.04)
                except (BrokenPipeError,ConnectionResetError):pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        out=HERE/'work'/('trickle-'+str(status));out.mkdir()
        transport=continuation.Transport(out,f'http://127.0.0.1:{server.server_port}/v1',1701,request_seconds=1,episode_seconds=4)
        started=time.monotonic()
        try:
            transport.post(transport.url+'/chat/completions',json={'max_tokens':16})
            raise AssertionError('Trickling request escaped deadline')
        except continuation.Stop as stop:
            elapsed=time.monotonic()-started
            assert stop.status=='request_deadline' and elapsed<2 and transport.unresolved and transport.calls==1
            results.append({'status_code':status,'halt':stop.status,'seconds':elapsed,'calls':1,'unresolved':transport.unresolved})
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);server.shutdown();server.server_close()
    return results


def main():
    identity=json.loads((HERE/'identity.json').read_text())
    for name,expected in identity['manifest'].items():
        assert hashlib.sha256((HERE/'package'/name).read_bytes()).hexdigest()==expected,name
    spec=importlib.util.spec_from_file_location('parent_fixture_builder',ROOT/'solvers/m2_planning_tools/qualify.py')
    sys.path.insert(0,str(ROOT/'solvers/m2_planning_tools'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fixtures=[]
    for variant in ['control','planner','probing','both']:
        result=process_case('package-'+variant,variant,module.fixture_code(variant))
        marker=result['saved_data'].get('fixture_result',{}).get('value',{})
        assert result['qualified_restoration'] and result['protected_inputs_unchanged'] and marker.get('complete')
        fixtures.append({'variant':variant,'operations':marker,'restored':True,'model_calls':0})
    raw_control=process_case('package-raw-control','control',checkpoint='control')
    assert raw_control['status']=='restored' and raw_control['qualified_restoration'] and raw_control['protected_inputs_unchanged']
    http=[http_case('http-success'),http_case('http-error',True)]
    trickles=trickle_cases()
    report={'fixtures':fixtures,'http':http,'trickles':trickles,'cpu_fixtures_only':True,
            'raw_control_restored':True,
            'package_source_verified':True,'notebook_sha256':identity['notebook_sha256']}
    (HERE/'evidence/qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
