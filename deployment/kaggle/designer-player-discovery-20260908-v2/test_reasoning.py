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
    module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,module);spec.loader.exec_module(module)
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
    c=client(monkeypatch,'{}')
    response=await c.acall([{'role':'user','content':'test'}],
        _reasoning_scope=sys.modules['reasoning_arc_llm']._DEEP_REASONING_SCOPE)
    assert len(seen)==1
    assert seen[0]['extra_body']['reasoning']['enabled'] is True
    assert seen[0]['max_tokens']==16384
    assert response.reasoning=='Synthetic reasoning fixture.'
    assert response.tool_calls[0].name=='execute_python'
    assert response.usage['completion_tokens_details']['reasoning_tokens']==25

@pytest.mark.parametrize('extra',['{}','{"reasoning":{"enabled":false}}'])
def test_actor_reasoning_defaults_off(monkeypatch,extra):
    c = client(monkeypatch,extra)
    assert c.config['extra_body']['reasoning']['enabled'] is False
    assert c.config['extra_body']['chat_template_kwargs']['enable_thinking'] is False

def test_conflicting_reasoning_refused(monkeypatch):
    with pytest.raises(ValueError,match='Conflicting reasoning'):
        client(monkeypatch,'{"reasoning":{"enabled":true},"chat_template_kwargs":{"enable_thinking":false}}')

@pytest.mark.asyncio
async def test_default_off_reaches_completion_transport(monkeypatch):
    import nooa.unifiedllm.unifiedllm as impl
    from litellm import ModelResponse
    seen = []
    async def completion(params):
        seen.append(params)
        return ModelResponse(choices=[{'index':0,'finish_reason':'tool_calls','message':{
            'role':'assistant','content':None,'tool_calls':[{'id':'c1','type':'function',
            'function':{'name':'execute_python','arguments':'{"code":"print(1)"}'}}]}}])
    monkeypatch.setattr(impl,'_litellm_acompletion',completion)
    response = await client(monkeypatch,'{}').acall([{'role':'user','content':'inspect'}])
    assert len(seen) == 1
    assert seen[0]['extra_body']['reasoning']['enabled'] is False
    assert seen[0]['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    assert response.tool_calls[0].name == 'execute_python'

def test_kaggle_switch_and_output_allowance(monkeypatch):
    with pytest.raises(ValueError, match='Routine reasoning'):
        client(monkeypatch,'{"chat_template_kwargs":{"enable_thinking":true},"reasoning":{"enabled":true}}')
    c=client(monkeypatch,'{}')
    assert c.config['max_tokens']==16384
    assert c.retry_config.retry_on_empty_content is False


@pytest.mark.asyncio
async def test_gate_rejects_bypasses_before_transport(monkeypatch):
    c = client(monkeypatch, '{}')
    for override in (
        {'extra_body': {'reasoning': {'enabled': True}}},
        {'chat_template_kwargs': {'enable_thinking': True}},
        {'reasoning_effort': 'high'},
        {'_reasoning_scope': 'deep'},
    ):
        with pytest.raises(ValueError):
            await c.acall([{'role': 'user', 'content': 'inspect'}], **override)
    c.config['extra_body']['reasoning']['enabled'] = True
    with pytest.raises(ValueError, match='Routine reasoning'):
        await c.acall([{'role': 'user', 'content': 'inspect'}])


@pytest.mark.asyncio
async def test_deep_permission_does_not_leak_to_concurrent_summary(monkeypatch):
    import asyncio
    import nooa.unifiedllm.unifiedllm as impl
    from litellm import ModelResponse
    from nooa import Agent
    from nooa.agents.summarization import SummarizationAgent
    seen, entered, release = [], asyncio.Event(), asyncio.Event()
    async def completion(params):
        seen.append(params)
        if params['extra_body']['reasoning']['enabled']:
            entered.set()
            await release.wait()
        return ModelResponse(choices=[{'index':0,'finish_reason':'stop',
            'message':{'role':'assistant','content':'{"value":"Observed movement."}'}}])
    monkeypatch.setattr(impl, '_litellm_acompletion', completion)
    c = client(monkeypatch, '{}')
    parent = Agent(llm=c)
    summary = SummarizationAgent(parent)
    deep = asyncio.create_task(c.acall([{'role':'user','content':'hard question'}],
        _reasoning_scope=sys.modules['reasoning_arc_llm']._DEEP_REASONING_SCOPE))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await summary.summarize('A movement observation.', 100)
        release.set()
        await deep
        await c.acall([{'role':'user','content':'next ordinary turn'}])
        assert [p['extra_body']['reasoning']['enabled'] for p in seen] == [True, False, False]
        assert all('_reasoning_scope' not in p for p in seen)
    finally:
        release.set()
        await deep
        summary._uninstall()


def test_notebook_refuses_global_reasoning():
    sys.path.insert(0, str(HERE))
    import build_notebook
    with pytest.raises(ValueError, match='Routine reasoning'):
        build_notebook.build(extra_body='{"chat_template_kwargs":{"enable_thinking":true}}')

@pytest.mark.parametrize('enabled', [False, True])
def test_actual_worker_records_reasoning_and_commits_tool_action(monkeypatch,tmp_path,enabled):
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
            if body.get('chat_template_kwargs', {}).get('enable_thinking'):
                raw=json.dumps({'id':'deep-fixture','object':'chat.completion','created':1,'model':'qwen/qwen3.8-27b',
                    'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'Test the two alternatives.', 'reasoning_content':'Scoped fixture reasoning.'}}],
                    'usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20,'completion_tokens_details':{'reasoning_tokens':5}}}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
                return
            raw=json.dumps({'id':'reason-fixture','object':'chat.completion','created':1,'model':'qwen/qwen3.8-27b',
                'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                'reasoning_content':None,
                'tool_calls':[{'id':'call1','type':'function','function':{'name':'execute_python','arguments':json.dumps({'code':
                    ('await self.deep_investigate("Which movement rule fits?", ["ipc/states.jsonl"])\n'
                     'try:\n'
                     '    await self.deep_investigate("Another question", ["ipc/states.jsonl"])\n'
                     'except RuntimeError as exc:\n'
                     '    assert "budget exhausted" in str(exc)\n'
                     'else:\n'
                     '    raise AssertionError("deep budget bypassed")\n' if enabled else '') +
                    'assert self.last_transition() is None\n'
                    'assert self.preflight_model(["UP"])["status"] == "unavailable"\n'
                    'src = \'def encode(g): return {"cells": [(0,0)]}\\ndef predict(z,a): return {"n":len(set(z["cells"]))}\\n\'\n'
                    'self.write_helper("world_model.py", src)\n'
                    'assert "unhashable" in self.preflight_model(["UP"])["reason"]\n'
                    'self.write_helper("world_model.py", src.replace(\'set(z["cells"])\', \'set(map(tuple,z["cells"]))\'))\n'
                    'assert self.preflight_model(["UP"])["status"] == "ready"\n'
                    'self.submit_actions(["UP"], rationale="predict: a one-action exploratory move")'})}}]}}],
                'usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150,'completion_tokens_details':{'reasoning_tokens':0}}}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    settings={'ARC_DISABLE_DOTENV':'1','ARC_LLM_MODEL':'qwen/qwen3.8-27b','ARC_LLM_API_KEY':'fixture',
        'ARC_LLM_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1','ARC_LLM_EXTRA_BODY':json.dumps({'reasoning':{'enabled':False}}),
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
    assert [r['chat_template_kwargs']['enable_thinking'] for r in seen] == ([False, True] if enabled else [False])
    assert all(r['reasoning']['enabled'] == r['chat_template_kwargs']['enable_thinking'] for r in seen)
    assert all('_reasoning_scope' not in r for r in seen)
    records=[json.loads(s) for p in tmp_path.rglob('model_calls.jsonl') for s in p.read_text().splitlines()]
    assert records and records[0]['reasoning_tokens']==0
    assert records[0]['reasoning_present'] is False
    assert not list(tmp_path.rglob('ABORT'))
    assert any(p.read_text().strip() for p in tmp_path.rglob('actions.jsonl'))
    commits=[s for p in tmp_path.rglob('model_predictions.jsonl') for s in p.read_text().splitlines()]
    assert len(commits)==1, 'preflight must not publish predictions'
