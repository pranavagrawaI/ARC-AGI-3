import importlib.util
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def client(monkeypatch,extra):
    monkeypatch.setenv('ARC_DISABLE_DOTENV','1')
    monkeypatch.setenv('ARC_LLM_MODEL','qwen/qwen3.8-27b')
    monkeypatch.setenv('ARC_LLM_BASE_URL','http://127.0.0.1:9/v1')
    monkeypatch.setenv('ARC_LLM_API_KEY','test-only')
    monkeypatch.setenv('ARC_LLM_MAX_TOKENS','16384')
    monkeypatch.setenv('ARC_LLM_EXTRA_BODY',extra)
    spec=importlib.util.spec_from_file_location('reasoning_arc_llm',HERE/'chassis/arc_llm.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.build_llm()

@pytest.mark.asyncio
@pytest.mark.parametrize('field',['reasoning','reasoning_content'])
async def test_real_client_parses_reasoning_and_null_content_tool_call_once(monkeypatch,field):
    import nooa.unifiedllm.unifiedllm as impl
    from litellm import ModelResponse
    seen=[]
    async def completion(params):
        seen.append(params)
        return ModelResponse(model='qwen/qwen3.8-27b',choices=[{'index':0,'finish_reason':'tool_calls','message':{
            'role':'assistant','content':None,field:'Synthetic reasoning fixture.',
            'tool_calls':[{'id':'call1','type':'function','function':{'name':'execute_python','arguments':'{"code":"print(1)"}'}}]}}],
            usage={'prompt_tokens':20,'completion_tokens':30,'total_tokens':50,'completion_tokens_details':{'reasoning_tokens':25}})
    monkeypatch.setattr(impl,'_litellm_acompletion',completion)
    c=client(monkeypatch,'{"reasoning":{"enabled":true}}')
    response=await c.acall([{'role':'user','content':'test'}])
    assert len(seen)==1
    assert seen[0]['extra_body']['reasoning']['enabled'] is True
    assert seen[0]['max_tokens']==16384
    assert response.reasoning=='Synthetic reasoning fixture.'
    assert response.tool_calls[0].name=='execute_python'
    assert response.usage['completion_tokens_details']['reasoning_tokens']==25

@pytest.mark.parametrize('extra',['{}','{"reasoning":{"enabled":false}}','{"reasoning":{"enabled":true},"chat_template_kwargs":{"enable_thinking":false}}'])
def test_disabled_or_conflicting_reasoning_refused(monkeypatch,extra):
    with pytest.raises(ValueError,match='enabled reasoning'):client(monkeypatch,extra)

def test_kaggle_switch_and_output_allowance(monkeypatch):
    c=client(monkeypatch,'{"chat_template_kwargs":{"enable_thinking":true},"reasoning":{"enabled":true}}')
    assert c.config['extra_body']['chat_template_kwargs']['enable_thinking'] is True
    assert c.config['max_tokens']==16384
    assert c.retry_config.retry_on_empty_content is False

def test_actual_worker_records_reasoning_and_commits_tool_action(monkeypatch,tmp_path):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler,HTTPServer
    root=ROOT
    sys.path[:0]=[str(root/'deployment/kaggle/tests'),str(root/'deployment/kaggle')]
    import test_my_agent, test_nooa_join, test_join_qualification as qual
    spec=importlib.util.spec_from_file_location('runtime.local_solver',HERE/'shared_local_solver.py')
    runtime=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,'runtime.local_solver',runtime);spec.loader.exec_module(runtime)
    monkeypatch.setattr(test_my_agent,'AGENT_PATH',HERE/'adapter.py')
    monkeypatch.setattr(test_nooa_join,'CANDIDATE',HERE)
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            raw=json.dumps({'id':'reason-fixture','object':'chat.completion','created':1,'model':'qwen/qwen3.8-27b',
                'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                'reasoning_content':'Synthetic reasoning fixture, not a model quality test.',
                'tool_calls':[{'id':'call1','type':'function','function':{'name':'execute_python','arguments':json.dumps({'code':
                    'assert self.last_transition() is None\n'
                    'assert self.preflight_model(["UP"])["status"] == "unavailable"\n'
                    'src = \'def encode(g): return {"cells": [(0,0)]}\\ndef predict(z,a): return {"n":len(set(z["cells"]))}\\n\'\n'
                    'self.write_helper("world_model.py", src)\n'
                    'assert "unhashable" in self.preflight_model(["UP"])["reason"]\n'
                    'self.write_helper("world_model.py", src.replace(\'set(z["cells"])\', \'set(map(tuple,z["cells"]))\'))\n'
                    'assert self.preflight_model(["UP"])["status"] == "ready"\n'
                    'self.submit_actions(["UP"], rationale="predict: a one-action exploratory move")'})}}]}}],
                'usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150,'completion_tokens_details':{'reasoning_tokens':30}}}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    settings={'ARC_DISABLE_DOTENV':'1','ARC_LLM_MODEL':'qwen/qwen3.8-27b','ARC_LLM_API_KEY':'fixture',
        'ARC_LLM_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1','ARC_LLM_EXTRA_BODY':'{"reasoning":{"enabled":true}}',
        'ARC_LLM_MAX_TOKENS':'16384','ARC_SOLVER_ARM':'mindset','NOOA_JOIN_ROOT':str(HERE),
        'NOOA_AGENT_COMMAND':json.dumps([sys.executable,str(HERE/'headless.py'),'--run-dir','.']),
        'TYCHO_NOOA_WORKER_TIMEOUT_SECONDS':'30','LITELLM_LOCAL_MODEL_COST_MAP':'True','TYCHO_KAGGLE_MAX_ACTIONS':'1'}
    for k,v in settings.items():monkeypatch.setenv(k,v)
    first=test_my_agent.frame([[0,1],[0,0]])
    agent=qual.build_agent(monkeypatch,tmp_path,first)
    try:
        action=agent.choose_action(agent.frames,first)
        assert action.name=='ACTION1'
        agent.frames.append(test_my_agent.frame([[1,0],[0,0]]))
    finally:
        agent.cleanup(scorecard=None);server.shutdown();thread.join(timeout=2)
    assert seen and all(r.get('reasoning',{}).get('enabled') is True and r.get('max_tokens')==16384 for r in seen)
    records=[json.loads(s) for p in tmp_path.rglob('model_calls.jsonl') for s in p.read_text().splitlines()]
    assert records and records[0]['reasoning_tokens']==30
    assert records[0]['reasoning_present']
    assert not list(tmp_path.rglob('ABORT'))
    assert any(p.read_text().strip() for p in tmp_path.rglob('actions.jsonl'))
    commits=[s for p in tmp_path.rglob('model_predictions.jsonl') for s in p.read_text().splitlines()]
    assert len(commits)==1, 'preflight must not publish predictions'
